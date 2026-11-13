"""Rigid rooms and the connectors between them, found in a map's own space.

Seen rooms keep their shape; a stretch of space that only leads from one room to another (a hallway, a tram
tunnel, a lift shaft) may run longer than the map draws it. The space is taken as a graph of voxels (bmab.solve:
a floor cell with the room above it) joined where the player could really pass: walls thinner than a voxel part
two voxels side by side, so each join is tested against the map's solid space, and a voxel joins the one above
it only inside one unbroken column.

Walking outward from one end of the space, the cells at each distance form cross-sections. Where consecutive
cross-sections stay one piece and stay small, the space is a tube; the tube is a connector if cutting it really
parts the map in two, with something on both sides. Each connector says how the player gets through it (its way, claims.WAYS), from what
runs along it: a train's path or a lift (ride), a ladder (climb), a mostly vertical tube with neither (fall), else a hallway
(walk), or a crawl (crouch) where the map's own faces leave room to crouch but not to stand along most of it. The voxels give
every floor at least two of them of room, so they can't tell a duct from a hall; ducts are also looked for on their own
(_ducts), as the walking misses short ones and ones that join rooms joined another way too. Side-by-side tubes are cut together
and run longer together. The cut goes where the tube is straightest
near its middle, square to it; the far side slides along that line as one block and the gap is more of the same passage.
A drop that nothing travels through (no lift, no ladder) is no way through: people mark one by hand where it is.
"""
import hashlib
import inspect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from . import brushes, bsp, plan, sources
from .graph import INCH
from .solve import CELL, CELL_M, MIN_HEAD, SKY_HEAD, _encode, columns

MAX_SECTION = 40       # cells in a cross-section (about 60 m2: a tram tunnel fits, a hall does not)
MAX_TRACK = 160        # along a train's path: a tram tunnel can be 16 cells wide and 10 high
MIN_LENGTH_M = 12.0    # shorter tubes are door frames and vestibules
MIN_DUCT_M = 3.0       # a crawl is a way through even short: a duct between two rooms
STAND = 72             # units: how tall a player stands; lower than this, a level passage is a crawl
CROUCH = 36            # units: the least room a crouching player needs
MAX_DUCT_W = 128       # units: a low space wider than this is room under something, not a duct
FINE = 16              # units: the grid low spaces are looked for on
MIN_SHAFT_M = 8.0
SIDE_M3 = 80.0         # what must lie on each side of a cut
TRACK_M, LIFT_M, LADDER_M = 8.0, 4.0, 4.0  # how near a train's path, a lift or a ladder must run to the tube's middle line
MIN_RIDE_M = 30.0      # a func_train that travels less than this is a moving part, not something to ride
MIN_LIFT_M = 2.0       # a vertical mover travelling less is a hatch or a gate
OVERLAP = 0.4          # share of a shaft's height lifts or ladders must span
TANDEM_M = 12.0        # tubes running the same way this close, side by side, stretch together
TANDEM_COS = 0.94      # within about 20 degrees
CURVED = 0.97          # straightness around a cut below this: the cut is on a bend
END_M = 3.0            # metres from a tube's ends where it isn't cut
LIMIT_CELLS, TOUCH = 80, 20  # how far to look for what a stretch would run into; voxels that count as touching
FRAME_SLACK = 96       # how far past the tube's own cells a cut's frame may reach to find the real wall
VERTICAL = 0.7         # share of the tube's length that is rise, for a shaft
BOXY = 0.75            # share of a corridor's cross-section box that is open: a sloped ledge or terrain fills about half
ROOFED = 0.75          # share of a corridor's cells that must be under a ceiling
SNAP = 0.92            # a tube this close to running along an axis gets a slice square to that axis
WINDOW_M = 6.0         # how far either side of a cut its slice is made straight
SLICE = CELL           # a claimed cut is this far either side of its plane: no join between cells steps over it
SLICE_REACH = 384      # how far past a claimed cut's outline its slice may spread (an outline is measured at the middle: bays and ledges are wider)
NOT_SHAPING = ("end_m", "whole_m")  # what a claim can change without changing where the map is cut
WHOLE_STEP, WHOLE_MOST = 0.5, 60.0  # metres: how finely and how far a passage is followed to see that it keeps its shape
SHAPE_FIT = 16  # units a passage's walls may differ from the cut's and still be the same passage
MIN_SHARED = 16  # cells two levels' rooms must both hold to count as the same place
ON_TRACK_COS = 0.866  # a tram line within 30 degrees of a passage is the line it follows
SLOPE_REACH, LEVEL_SLOPE = 120.0, 0.02  # units along a passage to compare floors, and the rise per unit that counts as level
_CHUNK = 2_000_000


def _open(b, pts):
    out = np.zeros(len(pts), bool)
    for i in range(0, len(pts), _CHUNK):
        out[i:i + _CHUNK] = b.vis.open(pts[i:i + _CHUNK])
    return out


def _passable(b, a, step):
    """For cells a (n x 3) and their neighbours a + step (one horizontal axis), whether a straight line can be drawn
    between them through open space, at any of a few heights and sideways offsets within the cell."""
    side = np.array([step[1], step[0], 0]) * 16
    run = np.arange(0, 49, 12)[:, None] * step[None]
    ok = np.zeros(len(a), bool)
    for z in (24, 12, 36):
        for s in (0, 1, -1):
            todo = np.flatnonzero(~ok)
            if not len(todo):
                return ok
            start = (a[todo] + [0.5, 0.5, 0]) * CELL + [0, 0, z] + side * s
            pts = (start[:, None] + run[None]).reshape(-1, 3)
            ok[todo] = _open(b, pts).reshape(len(todo), -1).all(1)
    return ok


def space_graph(b):
    """The map's voxels and how they join: cells (n x 3, sorted by key), a sparse symmetric adjacency matrix, and
    whether each cell's column has a ceiling (no open sky)."""
    runs = columns(b)[0]
    if not len(runs):
        return np.zeros((0, 3), np.int64), None, None
    n = runs[:, 3]
    xy = np.repeat(runs[:, :2], n, axis=0)
    z = np.concatenate([np.arange(f, f + h) for f, h in zip(runs[:, 2], n)])
    cells = np.unique(np.column_stack([xy, z]), axis=0)
    keys = _encode(cells)
    order = np.argsort(keys)
    cells, keys = cells[order], keys[order]
    frm, to = [], []
    n_cells = len(cells)
    open_x, open_y = np.zeros(n_cells, bool), np.zeros(n_cells, bool)
    for step, opened in ((np.array([1, 0, 0]), open_x), (np.array([0, 1, 0]), open_y)):
        q = _encode(cells + step)
        pos = np.minimum(np.searchsorted(keys, q), n_cells - 1)
        a = np.flatnonzero(keys[pos] == q)
        ok = _passable(b, cells[a], step)
        opened[a[ok]] = True
        frm.append(a[ok])
        to.append(pos[a[ok]])
    # a cell counts only as part of a 2 x 2 square of joined cells: a row one cell wide is the edge of a wall, not a way through
    def at(c):
        q = _encode(c)
        pos = np.minimum(np.searchsorted(keys, q), n_cells - 1)
        return pos, keys[pos] == q
    roomy = np.zeros(n_cells, bool)
    px, hx = at(cells + [1, 0, 0])
    py, hy = at(cells + [0, 1, 0])
    a = np.flatnonzero(open_x & open_y & hx & hy & open_y[px] & open_x[py])
    pxy = at(cells[a] + [1, 1, 0])[0]
    roomy[np.r_[a, px[a], py[a], pxy]] = True
    lows = np.concatenate([np.arange(f, f + h - 1) for f, h in zip(runs[:, 2], n)]) if (n > 1).any() else np.zeros(0, np.int64)
    if len(lows):
        low = np.column_stack([np.repeat(runs[:, :2], n - 1, axis=0), lows])
        a = np.unique(np.searchsorted(keys, _encode(low)))
        frm.append(a)
        to.append(np.searchsorted(keys, _encode(cells[a] + [0, 0, 1])))
    frm, to = np.concatenate(frm), np.concatenate(to)
    both = roomy[frm] & roomy[to]
    frm, to = frm[both], to[both]
    g = sparse.csr_matrix((np.ones(2 * len(frm), bool), (np.r_[frm, to], np.r_[to, frm])), shape=(n_cells, n_cells))
    sky = np.isin(_encode(np.column_stack([cells[:, :2], np.zeros(n_cells, np.int64)])),
                  _encode(np.column_stack([runs[:, :2], np.zeros(len(runs), np.int64)])[runs[:, 3] == SKY_HEAD]))
    return cells, g, ~sky


def _tubes(g, members, centre, segs):
    """Tubes of one connected part: the chains of narrow cross-sections (lists of section ids), {section id: indices
    (into members) of its cells at its distance}, and each section's size. A section one cell thick across the way
    is a sliver beside a wall or a pillar, not something a person goes through."""
    sub = g[members][:, members]
    d = csgraph.shortest_path(sub, unweighted=True, indices=0)
    d = csgraph.shortest_path(sub, unweighted=True, indices=int(np.argmax(np.where(np.isfinite(d), d, -1)))).astype(np.int64)
    n = len(members)
    # a band is two consecutive distances; a cell has a node in the band it starts (c) and in the one it ends (c + n)
    coo = sub.tocoo()
    up, same = d[coo.col] == d[coo.row] + 1, d[coo.col] == d[coo.row]
    frm = np.r_[coo.row[up], coo.row[same], coo.row[same] + n]
    to = np.r_[coo.col[up] + n, coo.col[same], coo.col[same] + n]
    labels = csgraph.connected_components(sparse.csr_matrix((np.ones(len(frm), bool), (frm, to)), shape=(2 * n, 2 * n)), directed=False)[1]
    size = np.bincount(labels) // 2
    band = np.zeros(len(size), np.int64)
    band[labels] = np.r_[d, d - 1]
    valid = d > 0
    down, ups = defaultdict(list), defaultdict(list)
    for a, b in np.unique(np.column_stack([labels[n:][valid], labels[:n][valid]]), axis=0).tolist():
        ups[a].append(b)
        down[b].append(a)

    lower = labels[:n]
    lo = np.full((len(size), 3), np.iinfo(np.int64).max)
    hi = np.full((len(size), 3), np.iinfo(np.int64).min)
    at = np.round(centre[members] / CELL).astype(np.int64)
    np.minimum.at(lo, lower, at)
    np.maximum.at(hi, lower, at)
    thick = np.sort(hi - lo + 1, axis=1)[:, 1] >= 2
    count = np.maximum(np.bincount(lower, minlength=len(size)), 1)
    middle = np.stack([np.bincount(lower, centre[members, a], len(size)) / count for a in range(3)], 1)
    wide = np.flatnonzero((size > MAX_SECTION) & (size <= MAX_TRACK) & (band >= 0) & thick)
    on_track = np.zeros(len(size), bool)
    if len(wide):
        on_track[wide] = _distance(middle[wide], segs) <= TRACK_M / INCH

    def narrow(c):
        return band[c] >= 0 and thick[c] and (size[c] <= MAX_SECTION or on_track[c])
    chains, used = [], set()
    for c in np.argsort(band, kind="stable").tolist():
        if c in used or not narrow(c):
            continue
        p = down.get(c, [])
        if len(p) == 1 and len(ups.get(p[0], [])) == 1 and narrow(p[0]):
            continue
        chain = [c]
        used.add(c)
        while len(ups.get(chain[-1], [])) == 1:
            nxt = ups[chain[-1]][0]
            if nxt in used or not narrow(nxt) or len(down.get(nxt, [])) != 1:
                break
            chain.append(nxt)
            used.add(nxt)
        if len(chain) >= 3:
            chains.append(chain)
    shell = defaultdict(list)
    for c, lab in enumerate(labels[:n].tolist()):
        shell[lab].append(c)
    return chains, shell, size


def _origin(e):
    return np.array([float(x) for x in e["origin"].split()])


def _chain(b, first):
    """Points of the path that starts at the named path_track or path_corner, following target."""
    nodes = {e["targetname"]: e for e in b.entities if e.get("classname") in ("path_track", "path_corner") and e.get("targetname") and "origin" in e}
    out, seen, name = [], set(), first
    while name in nodes and name not in seen:
        seen.add(name)
        out.append(_origin(nodes[name]))
        name = nodes[name].get("target")
    return np.array(out).reshape(-1, 3)


def _trains(b):
    """(entity, its path's points) of every func_train and func_tracktrain with a path."""
    return [(e, c) for e in b.entities if e.get("classname") in ("func_train", "func_tracktrain") and e.get("target")
            and len(c := _chain(b, e["target"])) >= 2]


def _paths(b):
    """Segments (n x 2 x 3, units) of the paths of trains the player rides: every track of a map with a func_tracktrain
    (a train can reach all of it, through switches and backwards), and the route of a func_train that travels far. Helicopters,
    fish and patrolling monsters walk paths too, and a func_train is often a small moving part: neither is a track."""
    seg = []
    if any(e.get("classname") == "func_tracktrain" for e in b.entities):
        nodes = {e["targetname"]: e for e in b.entities if e.get("classname") == "path_track" and e.get("targetname") and "origin" in e}
        for e in nodes.values():
            for next_ in (e.get("target"), e.get("altpath")):
                if next_ in nodes:
                    seg.append((_origin(e), _origin(nodes[next_])))
    for e, c in _trains(b):
        if e["classname"] == "func_train" and np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1).sum() * INCH >= MIN_RIDE_M:
            seg += list(zip(c[:-1], c[1:]))
    return np.array(seg).reshape(-1, 2, 3)


def _box(b, e):
    """A brush entity's box in the map's coordinates: one with an origin has its model around that origin."""
    lo, hi = (np.asarray(v, float) for v in b.models[int(e["model"][1:])][:2])
    at = np.array([float(x) for x in e.get("origin", "0 0 0").split()])
    return lo + at, hi + at


def _upright(e):
    """Whether a door or sliding platform moves straight up or down: GoldSrc says so by angle or angles, Source by movedir."""
    return e.get("angle", "") in ("-1", "-2") or any(e.get(k, "0 0 0").split()[0] in ("-90", "90") for k in ("angles", "movedir"))


def _movers(b):
    """Things that really travel up and down: (xy, z low, z high, label, box lo, box hi), units."""
    out = []
    box = lambda e: _box(b, e)  # noqa: E731
    for e, c in _trains(b):
        if np.ptp(c[:, 2]) * INCH >= MIN_LIFT_M and np.ptp(c[:, :2], axis=0).max() * INCH <= LIFT_M and e.get("model", "").startswith("*"):
            lo, hi = box(e)
            out.append((c[:, :2].mean(0), c[:, 2].min(), c[:, 2].max() + hi[2] - lo[2], e, lo, hi))
    for e in b.entities:
        cls = e.get("classname", "")
        if cls not in ("func_plat", "func_door", "momentary_door", "func_movelinear") or not e.get("model", "").startswith("*"):
            continue
        lo, hi = box(e)
        if cls == "func_plat":
            travel = abs(float(e.get("height") or 0)) or float(hi[2] - lo[2]) - 8
            z = (lo[2] - travel, hi[2] + travel)
        else:
            if not _upright(e):
                continue
            travel = float(e.get("movedistance") or 0) if cls == "func_movelinear" else float(hi[2] - lo[2]) - float(e.get("lip") or 0)
            z = (lo[2] - travel, hi[2] + travel)
        if travel * INCH >= MIN_LIFT_M:
            out.append(((lo[:2] + hi[:2]) / 2, z[0], z[1], e, lo, hi))
    return out


def _covered(ranges, path):
    """Share of a tube's height that these (low, high) ranges cover together."""
    a, b = float(path[:, 2].min()), float(path[:, 2].max())
    clipped = sorted((max(lo, a), min(hi, b)) for lo, hi in ranges if min(hi, b) > max(lo, a))
    total, end = 0.0, a
    for lo, hi in clipped:
        lo = max(lo, end)
        if hi > lo:
            total += hi - lo
            end = hi
    return total / max(b - a, 1e-6)


def _carrier(b, path, kind):
    """The lift or ladder that really runs through a tube, or None: (entity, box lo, box hi). Ladders often come in flights one
    above the other, so together they must cover enough of the shaft."""
    if kind == "lift":
        near = [(float(np.linalg.norm(xy - path[:, :2], axis=1).min()), z0, z1, e, lo, hi) for xy, z0, z1, e, lo, hi in _movers(b)]
        reach = LIFT_M / INCH
    else:
        near = []
        for e in b.entities:
            if e.get("classname") == "func_ladder" and e.get("model", "").startswith("*"):
                lo, hi = _box(b, e)
            elif e.get("classname") == "func_useableladder":  # Source: its two ends, no brush
                ends = np.array([[float(x) for x in e[k].split()] for k in ("point0", "point1")])
                lo, hi = ends.min(0) - 16, ends.max(0) + 16
            else:
                continue
            near.append((float(np.linalg.norm((lo[:2] + hi[:2]) / 2 - path[:, :2], axis=1).min()), lo[2], hi[2], e, lo, hi))
        reach = LADDER_M / INCH
    near = [n for n in near if n[0] <= reach]
    if not near or _covered([(n[1], n[2]) for n in near], path) < OVERLAP:
        return None
    return min(near, key=lambda n: n[0])[3:]


def _distance(pts, segs):
    """Each point's distance (units) to the nearest segment; inf with no segments."""
    if not len(segs):
        return np.full(len(pts), np.inf)
    a, ab = segs[:, 0], segs[:, 1] - segs[:, 0]
    t = np.clip(np.einsum("ijk,jk->ij", pts[:, None] - a[None], ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(pts[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)


def _marks(b, classes):
    """Centres (n x 3, units) of the entities of these classes, brush entities by their box."""
    out = []
    for e in b.entities:
        if e.get("classname") not in classes:
            continue
        if e.get("model", "").startswith("*"):
            lo, hi = _box(b, e)
            out.append((lo + hi) / 2)
        elif "origin" in e:
            out.append(np.array([float(x) for x in e["origin"].split()]))
    return np.array(out).reshape(-1, 3)


def _built(cells, roofed, ends):
    """Whether a tube is a built passage: most of it under a ceiling (rock ledges lie open to the sky) and its
    cross-section filling its box (walls and a ceiling, not a sloping ledge)."""
    if roofed.mean() < ROOFED:
        return False
    axis = int(np.argmax(np.abs((ends[1] - ends[0])[:2])))
    at = cells[cells[:, axis] == int(np.round(np.median(cells[:, axis])))]
    return len(at) > 0 and len(at) >= (np.ptp(at[:, 1 - axis]) + 1) * (np.ptp(at[:, 2]) + 1) * BOXY


def _kind(b, path, rise, length, segs, low=False):
    """How the player gets through a tube (claims.WAYS), from what runs along its middle line (n x 3, units). low: its sections are no
    taller than the least room the space search gives a floor, the only tubes that can be a crawl; one is where there is room to
    crouch but not to stand along most of it."""
    if rise >= VERTICAL * length:
        return "ride" if _carrier(b, path, "lift") else "climb" if _carrier(b, path, "ladder") else "fall"
    if (_distance(path, segs) <= TRACK_M / INCH).mean() >= 0.5:
        return "ride"
    if not low:
        return "walk"
    room = [headroom(b, p) for p in path[np.linspace(0, len(path) - 1, min(len(path), 9)).round().astype(int)]]
    crawl = [CROUCH <= r < STAND for r in room if r is not None]  # lower than a crouch is the top of something, not the way
    return "crouch" if crawl and np.mean(crawl) > 0.5 else "walk"


def planes(b):
    """The map's surfaces as arrays for _first_face: (normals, plane offsets, lowest and highest corners, the faces)."""
    if getattr(b, "surfaces", None) is None:  # kept on the map, not as a Bsp field: bsp.py's text keys the connector caches
        fs = [f for f in b.faces if not f.skybox]
        placed = [brushes.place(b, f) for f in fs]  # brush entities where they stand
        n = np.array([m for _v, m in placed])
        b.surfaces = (n, np.array([float(np.mean(v @ m)) for v, m in placed]), np.array([v.min(0) for v, _m in placed]),
                      np.array([v.max(0) for v, _m in placed]), fs, [v for v, _m in placed])
    return b.surfaces


def _first_face(surfaces, p, d, reach):
    """The first drawn face a ray meets from the front, within reach: (how far along, its index); None if there is none. Source's
    terrain isn't solid to the map's tree, so the faces are asked too."""
    n, offset, lo, hi, fs, corners = surfaces
    along = n @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset - n @ p) / along
        hit = p[None] + t[:, None] * d[None]
    near = np.flatnonzero((along < -1e-3) & (t > 0) & (t <= reach) & ((hit >= lo - 1) & (hit <= hi + 1)).all(1))
    for k in near[np.argsort(t[near])].tolist():
        v, q = corners[k], hit[k]
        side = np.cross(np.roll(v, -1, axis=0) - v, q - v) @ n[k]
        slack = 1e-3 * np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1) * np.linalg.norm(q - v, axis=1)
        if ((side >= -slack).all() or (side <= slack).all()) and not bsp.is_tool(fs[k].texture):  # inside, whichever way the face is wound
            return float(t[k]), k
    return None


def headroom(b, p):
    """Units from the floor under p (a point, units) to the ceiling over it, from the map's faces; None where either is missing.
    The space search gives every floor at least two cells of room, so its cells can't tell a duct from a hall."""
    s = planes(b)
    down = _first_face(s, np.asarray(p, float) + [0, 0, 8], np.array([0.0, 0.0, -1.0]), 8 * STAND)
    if down is None:
        return None
    floor = np.asarray(p, float) + [0, 0, 8 - down[0] + 1]
    up = _first_face(s, floor, np.array([0.0, 0.0, 1.0]), 8 * STAND)
    return None if up is None else up[0] + 1


def _place(path, along, sizes):
    """Which section of a tube to cut: one where it runs straight, as near the middle as that allows, where its cross-section holds
    steady. Only when it is bent everywhere does the cut go on a bend."""
    length = along[-1]
    margin = min(END_M, 0.1 * length)
    best, at = None, int(np.argmin(np.abs(along - length / 2)))
    for i in np.flatnonzero((along >= margin) & (along <= length - margin)).tolist():
        w = np.flatnonzero(np.abs(along - along[i]) <= 3.0)
        if len(w) < 2:
            continue
        run = along[w[-1]] - along[w[0]]
        straight = float(np.linalg.norm(path[w[-1]] - path[w[0]])) * INCH / max(run, 1e-6)
        steady = float(np.std(sizes[w]) / max(np.mean(sizes[w]), 1.0))
        score = (straight >= CURVED, 2.0 * straight + 0.7 * (1 - abs(along[i] - length / 2) / max(length / 2, 1e-6)) - 0.5 * steady)
        if best is None or score > best:
            best, at = score, i
    return at


def _direction(path, along, mid):
    """The line a tube runs along at a section: (unit vector, straightness). Upright unless the tube is a shaft: a cut across
    a ramp is a vertical slice. Square to an axis where the tube nearly is."""
    w = np.flatnonzero(np.abs(along - along[mid]) <= 3.0)
    tan = path[w[-1]] - path[w[0]]
    straight = float(np.linalg.norm(tan)) * INCH / max(float(along[w[-1]] - along[w[0]]), 1e-6)
    tan = tan / max(float(np.linalg.norm(tan)), 1e-9)
    if abs(tan[2]) < VERTICAL:
        tan = np.array([tan[0], tan[1], 0.0])
        tan /= max(float(np.linalg.norm(tan)), 1e-9)
    a = int(np.argmax(np.abs(tan)))
    if abs(tan[a]) >= SNAP:
        tan = np.eye(3)[a] * np.sign(tan[a])
    return tan, straight


def find(b):
    """Connectors of a map: (list of connectors, cells n x 3, the room each cell belongs to). A map without any has
    every cell in room 0. Each connector: way, length_m, rise_m, section_m2, along (unit vector, the way it stretches),
    path (units), middle (units), rooms (behind and ahead of the cut), outline (the cut's), limit_m (how far it can
    stretch before the far side hits this map's own space, None: no limit found), curved, stretches_with (connectors
    stretching with it)."""
    if b.vis is None:
        return [], None, None
    cells, g, roofed = space_graph(b)
    if not len(cells):
        return [], cells, np.zeros(0, np.int64)
    pieces = csgraph.connected_components(g, directed=False)[1]
    segs = _paths(b)
    found = []
    centre = (cells + [0.5, 0.5, 0.5]) * CELL
    for k in np.flatnonzero(np.bincount(pieces) >= 50).tolist():
        members = np.flatnonzero(pieces == k)
        chains, shell, size = _tubes(g, members, centre, segs)
        for chain in chains:
            sect = [members[shell[c]] for c in chain]
            path = np.array([centre[s].mean(0) for s in sect])
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
            length = float(steps.sum()) * INCH
            rise = float(np.abs(np.diff(path[:, 2])).sum()) * INCH
            if length < min(MIN_SHAFT_M, MIN_DUCT_M):
                continue
            kind = _kind(b, path, rise, length, segs, np.median([np.ptp(cells[s, 2]) + 1 for s in sect]) <= MIN_HEAD)
            if length < (MIN_SHAFT_M if rise >= VERTICAL * length else MIN_DUCT_M if kind == "crouch" else MIN_LENGTH_M):
                continue
            if kind == "fall":  # a drop nothing travels through is no way through
                continue
            tube = np.concatenate(sect)
            if kind in ("walk", "crouch") and not _built(cells[tube], roofed[tube], path[[0, -1]]):
                continue
            along = np.r_[0, np.cumsum(steps)] * INCH
            sizes = size[chain].astype(float)
            mid = _place(path, along, sizes)
            tan, straight = _direction(path, along, mid)
            if abs(tan[2]) < VERTICAL:
                tan = _along_track(path[mid], tan, segs) if kind == "ride" else _up_the_floor(b, path[mid], tan)
            found.append({"sections": sect, "mid": mid, "way": kind, "length_m": round(length, 1), "rise_m": round(rise, 1),
                          "section_m2": round(float(np.median(size[chain])) * CELL_M ** 2, 1), "tan": tan, "curved": bool(straight < CURVED),
                          "dists": along, "path": path.round().tolist(), "middle": path[mid].round().tolist(), "cut": sect[mid]})
    found += _ducts(b, cells, found)
    return _separate(b, cells, g, found)


def _heights(polys):
    """{(x, y): [z]} on the FINE grid: each polygon covering a cell, at its mean height (units)."""
    from PIL import Image, ImageDraw
    out = defaultdict(list)
    for p in polys:
        z = float(p[:, 2].mean())
        x0, y0 = np.floor(p[:, :2].min(0) / FINE).astype(int)
        x1, y1 = np.ceil(p[:, :2].max(0) / FINE).astype(int)
        m = Image.new("1", (x1 - x0 + 1, y1 - y0 + 1))
        ImageDraw.Draw(m).polygon([(x / FINE - x0, y / FINE - y0) for x, y, _ in p], fill=1)
        yy, xx = np.nonzero(np.array(m))
        for x, y in zip((xx + x0).tolist(), (yy + y0).tolist()):
            out[(x, y)].append(z)
    return out


def _low(b):
    """Floor points under a ceiling too low to stand under but high enough to crawl: (FINE cells n x 2, floor heights, headroom),
    units for the heights. The space search's cells are too coarse for this: every floor gets at least two of them."""
    floors = plan.floors(b, np.zeros(3), facility=True)
    ceilings = [f.verts for f in b.faces if f.model == 0 and not f.skybox and not f.outside and f.normal[2] < -0.7 and not bsp.is_tool(f.texture.lower())]
    fl, ce = _heights(floors), _heights(ceilings)
    xy, zs, room = [], [], []
    for k, here in fl.items():
        cz = np.sort(ce.get(k, []))
        for zf in set(here):
            i = int(np.searchsorted(cz, zf + 1))
            if i < len(cz) and CROUCH <= cz[i] - zf < STAND:
                xy.append(k), zs.append(zf), room.append(cz[i] - zf)
    return np.array(xy, np.int64).reshape(-1, 2), np.array(zs), np.array(room)


def _run(xy, axis):
    """How many cells long the unbroken row along axis is that each cell (n x 2) is in."""
    order = np.lexsort((xy[:, axis], xy[:, 1 - axis]))
    a, o = xy[order, axis], xy[order, 1 - axis]
    run = np.cumsum(np.r_[True, (o[1:] != o[:-1]) | (a[1:] != a[:-1] + 1)]) - 1
    out = np.empty(len(xy), np.int64)
    out[order] = np.bincount(run)[run]
    return out


def _ducts(b, cells, found):
    """Ducts: long, narrow, walled-in floor too low to stand on, as tubes in the space search's cells. The tube search misses them
    when they are short, or join rooms that are joined another way too (its sections are bands of distance from one end, which
    meet in the middle of such a duct)."""
    xy, zs, room = _low(b)
    if not len(xy):
        return []
    index = {k: [] for k in map(tuple, xy.tolist())}
    for i, k in enumerate(map(tuple, xy.tolist())):
        index[k].append(i)
    frm, to = [], []
    for i, (x, y) in enumerate(xy.tolist()):
        for k in ((x + 1, y), (x, y + 1)):
            for j in index.get(k, ()):
                if abs(zs[j] - zs[i]) <= 18:  # a step the player crouch-walks up
                    frm.append(i), to.append(j)
    n = len(xy)
    joins = sparse.csr_matrix((np.ones(len(frm), bool), (frm, to)), shape=(n, n))
    lab = csgraph.connected_components(joins, directed=False)[1]
    legs = []
    for k in np.flatnonzero(np.bincount(lab) * FINE >= MIN_DUCT_M / INCH).tolist():
        m = np.flatnonzero(lab == k)
        rx, ry = _run(xy[m], 0), _run(xy[m], 1)
        narrow = np.minimum(rx, ry) * FINE <= MAX_DUCT_W  # wide low space (under a stage, a crawl space) is no duct
        along_x = rx >= ry  # a bent duct is a leg along x and a leg along y: each point goes with its longer run
        for part in (m[narrow & along_x], m[narrow & ~along_x]):
            if len(part):
                sub = csgraph.connected_components(joins[part][:, part], directed=False)[1]
                legs += [part[sub == j] for j in range(int(sub.max()) + 1)]
    centre = (cells + [0.5, 0.5, 0.5]) * CELL
    taken = np.zeros(len(cells), bool)
    for c in found:
        for s in c["sections"]:
            taken[s] = True
    out = []
    for m in legs:
        if len(m) * FINE < MIN_DUCT_M / INCH:
            continue
        q = (xy[m] + 0.5) * FINE
        mid_xy = q.mean(0)
        w, v = np.linalg.eigh((q - mid_xy).T @ (q - mid_xy))
        u = v[:, 1]
        side = np.array([-u[1], u[0]])
        s, r = (q - mid_xy) @ u, (q - mid_xy) @ side
        length, width = float(np.ptp(s)) + FINE, float(np.ptp(r)) + FINE
        if length * INCH < MIN_DUCT_M or width > MAX_DUCT_W or length < 2 * width:
            continue
        floor, head = float(np.median(zs[m])), float(np.median(room[m]))
        line = mid_xy + side * (r.min() + r.max()) / 2
        ts = np.linspace(s.min(), s.max(), max(2, int(round(length / CELL)) + 1))
        along_line = line[None] + ts[:, None] * u[None]
        reach = np.arange(4.0, width / 2 + 32, 4.0)
        probes = (along_line[:, None, None] + np.array([1, -1])[None, :, None, None] * reach[None, None, :, None] * side).reshape(-1, 2)
        walled = ~_open(b, np.column_stack([probes, np.full(len(probes), floor + head / 2)])).reshape(-1, len(reach)).all(1)
        if walled.mean() < 0.7:
            continue  # no wall beside it: room under a pipe or a shelf, not a duct
        path = np.column_stack([along_line, np.full(len(ts), floor + head / 2)])
        z0 = int(np.floor(floor / CELL))
        near = np.flatnonzero((np.abs(centre[:, :2] - mid_xy) <= length / 2 + width + CELL).all(1) & (cells[:, 2] >= z0) & (cells[:, 2] <= z0 + 1))
        sect = []
        for p_ in path:
            d = centre[near, :2] - p_[:2]
            sect.append(near[(np.abs(d @ u) <= CELL / 2) & (np.abs(d @ side) <= width / 2 + CELL / 2)])
        keep = [i for i, s_ in enumerate(sect) if len(s_)]
        if len(keep) < 2 or any(taken[s_].any() for s_ in sect):
            continue
        path, sect = path[keep], [sect[i] for i in keep]
        mid = len(path) // 2
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        out.append({"sections": sect, "mid": mid, "way": "crouch", "length_m": round(length * INCH, 1), "rise_m": round(float(np.ptp(zs[m])) * INCH, 1),
                    "section_m2": round(width * head * INCH ** 2, 1), "tan": np.array([u[0], u[1], 0.0]), "curved": False,
                    "dists": np.r_[0, np.cumsum(steps)] * INCH, "path": path.round().tolist(), "middle": path[mid].round().tolist(), "cut": sect[mid]})
    return out


def _along_track(p, tan, segs):
    """A track connector's direction: exactly the tram line's, at the nearest piece of it that runs its way (a line crossing it is another line)."""
    if not len(segs):
        return tan
    a, ab = segs[:, 0], segs[:, 1] - segs[:, 0]
    lengths = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    near = np.linalg.norm(a + np.clip(np.einsum("ij,ij->i", p - a, ab) / lengths, 0, 1)[:, None] * ab - p, axis=1)
    for k in np.argsort(near):
        if near[k] > TRACK_M / INCH:
            break
        u = ab[k] / np.sqrt(lengths[k])
        if abs(float(u @ tan)) >= ON_TRACK_COS:
            return u if u @ tan > 0 else -u
    return tan


def _up_the_floor(b, p, tan):
    """A passage's line with its slope: how the floor rises either side of the middle. Within a degree it is level."""
    down, t = np.array([0.0, 0.0, 1.0]), np.arange(0.0, 640.0, 2.0)

    def floor(q):
        ok = _open(b, q[None] - t[:, None] * down)
        return None if not ok[0] or ok.all() else float(q[2] - t[np.argmin(ok)])
    a, c = floor(p - tan * SLOPE_REACH), floor(p + tan * SLOPE_REACH)
    if a is None or c is None or abs(c - a) / (2 * SLOPE_REACH) < LEVEL_SLOPE:
        return tan
    out = np.array([tan[0], tan[1], (c - a) / (2 * SLOPE_REACH)])
    return out / np.linalg.norm(out)


def _stretching_together(found):
    """Groups of tubes (lists of indices into found) that run the same way side by side and so must stretch together."""
    parent = list(range(len(found)))

    def top(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(found):
        for j, c in enumerate(found[:i]):
            if abs(float(a["tan"] @ c["tan"])) < TANDEM_COS:
                continue
            gap = (np.array(a["middle"], float) - c["middle"]) * INCH
            along = abs(float(gap @ a["tan"]))
            if along <= TANDEM_M and float(np.linalg.norm(gap - (gap @ a["tan"]) * a["tan"])) <= TANDEM_M:
                parent[top(i)] = top(j)
    groups = defaultdict(list)
    for i in range(len(found)):
        groups[top(i)].append(i)
    return [sorted(g) for g in groups.values()]


def _orient(group, cells, g, room):
    """Each connector's two rooms as [behind, ahead] of its cut along the group's shared direction, or None."""
    ref = group[0]["tan"]
    out = []
    for c in group:
        if float(c["tan"] @ ref) < 0:
            c["tan"] = -c["tan"]
            if "sides" in c:
                c["sides"] = c["sides"][::-1]
        if "sides" in c:  # a claimed slice can be wide: its ends are the passage straight on either way, not a side room it brushes
            found = [room[s][room[s] >= 0] for s in c["sides"]]
            ends = [int(np.bincount(f).argmax()) if len(f) else None for f in found]
            out.append(ends if None not in ends and ends[0] != ends[1] else None)
            continue
        nb = np.unique(g[c["cut"]].indices)
        nb = nb[room[nb] >= 0]
        if not len(nb):
            out.append(None)
            continue
        ids, count = np.unique(room[nb], return_counts=True)
        if len(ids) < 2:
            out.append(None)
            continue
        two = ids[np.argsort(-count)][:2]
        side = {int(r): float((((cells[nb[room[nb] == r]] + 0.5) * CELL - c["middle"]) @ c["tan"]).mean()) for r in two}
        out.append(sorted(side, key=side.get))
    return out


def _separate(b, cells, g, found, tied=None):
    """Keep the connectors whose cut really parts the map, with enough on both sides, and label the rooms. Tubes that stretch
    together are cut together: the rooms on each side of the group count as one. tied: the groups, if they are already known."""
    groups = [[found[i] for i in g_] for g_ in (tied or _stretching_together(found))]
    while True:
        cut_of = np.zeros(len(cells), bool)
        for group in groups:
            for c in group:
                cut_of[c["cut"]] = True
        keep = np.flatnonzero(~cut_of)
        _, lab = csgraph.connected_components(g[keep][:, keep], directed=False)
        room = np.full(len(cells), -1)
        room[keep] = lab
        n = int(lab.max()) + 1 if len(lab) else 0
        fuse = list(range(n))

        def top(i):
            while fuse[i] != i:
                fuse[i] = fuse[fuse[i]]
                i = fuse[i]
            return i
        ends = [_orient(group, cells, g, room) for group in groups]
        for pairs in ends:
            whole = [p for p in pairs if p is not None]
            for p in whole[1:]:
                fuse[top(p[0])] = top(whole[0][0])
                fuse[top(p[1])] = top(whole[0][1])
        edges = [[None if p is None else (top(p[0]), top(p[1])) for p in pairs] for pairs in ends]
        m3 = np.zeros(n)
        np.add.at(m3, [top(r) for r in lab.tolist()], CELL_M ** 3)
        marked = {top(r) for r in _route_rooms(b, cells, room)}
        drop = _weakest(edges, m3, marked, n, 0.0 if tied else SIDE_M3)
        if drop is None:
            break
        if len(groups[drop]) > 1:  # they don't part the map together: each may still do so alone
            groups[drop:drop + 1] = [[c] for c in groups[drop]]
        else:
            groups.pop(drop)
    cuts = [c for group in groups for c in group]
    if not cuts:
        return [], cells, np.zeros(len(cells), np.int64)
    fused = np.array([top(i) for i in range(n)])
    used = sorted({r for es in edges for e in es for r in e})
    number = {r: i for i, r in enumerate(used)}
    first = 0
    for group, es in zip(groups, edges):
        members = list(range(first, first + len(group)))
        for k, (c, e) in enumerate(zip(group, es)):
            c["rooms"] = [number[e[0]], number[e[1]]]
            c["stretches_with"] = [m for m in members if m != first + k]
            c.pop("cut")
            c.pop("sides", None)
        first += len(group)
    # the cut cells and whatever no connector joins (loose bits, separate pieces) go with the nearest room
    from scipy.spatial import cKDTree
    room = np.where(room >= 0, fused[np.maximum(room, 0)], -1)
    has = np.isin(room, used)
    out = np.full(len(cells), -1, np.int64)
    out[has] = np.searchsorted(used, room[has])
    out[~has] = out[has][cKDTree(cells[has]).query(cells[~has])[1]]
    for c in cuts:
        _straighten(b, c, cells, out)
    keys, joined = _encode(cells), g.getnnz(axis=1) > 0
    for c in cuts:
        c["limit_m"] = _limit(c, cuts, keys, out, joined)
    for c in cuts:
        del c["sections"], c["mid"], c["dists"]
        c["along"] = [round(float(x), 3) for x in c.pop("tan")]
    return cuts, cells, out


def _far_rooms(c, cuts):
    """The rooms on the far side of a cut: reached from its ahead room without crossing it or the cuts stretching with it."""
    tied = {id(c)} | {id(cuts[i]) for i in c["stretches_with"]}
    nbrs = defaultdict(list)
    for o in cuts:
        if id(o) not in tied:
            nbrs[o["rooms"][0]].append(o["rooms"][1])
            nbrs[o["rooms"][1]].append(o["rooms"][0])
    seen, todo = {c["rooms"][1]}, [c["rooms"][1]]
    while todo:
        for y in nbrs[todo.pop()]:
            if y not in seen:
                seen.add(y)
                todo.append(y)
    return seen


def _limit(c, cuts, keys, room, joined):
    """Metres the far side of a cut can slide along it before running into this map's near side; None: no limit found. Only
    cells of real rooms count (joined): a loose bit of wall-edge voxels isn't something a stretch runs into."""
    far = np.isin(room, list(_far_rooms(c, cuts)))
    if not (far & joined).any() or not (joined & ~far).any():
        return None
    near = np.sort(keys[joined & ~far])
    moving = keys[joined & far]
    zero = _encode(np.zeros(3, np.int64))
    for d in range(2, LIMIT_CELLS + 1, 2):
        moved = moving + _encode(np.round(c["tan"] * d).astype(np.int64)) - zero
        pos = np.minimum(np.searchsorted(near, moved), len(near) - 1)
        if int((near[pos] == moved).sum()) >= TOUCH:
            return round((d - 2) * CELL_M, 1)
    return None


def _frame(b, c, cells, middle, tan):
    """The cut's outline: the real walls either side of the tube's middle, by looking for solid space from it. Returns
    the frame's centre and its four corners, units."""
    up = np.array([0.0, 0.0, 1.0]) if abs(tan[2]) < VERTICAL else np.array([1.0, 0.0, 0.0])
    u = np.cross(tan, up)
    u /= np.linalg.norm(u)
    v = np.cross(u, tan)
    at = (cells[c["cut"] if "cut" in c else c["sections"][c["mid"]]] + 0.5) * CELL
    rel = at - middle
    bounds = [float((rel @ d).min()) - CELL / 2 - FRAME_SLACK for d in (u, v)] + [float((rel @ d).max()) + CELL / 2 + FRAME_SLACK for d in (u, v)]
    at = at[np.argsort(np.linalg.norm(at - middle, axis=1))]
    p = next((q for q in [middle, *at] if b.vis.open(np.array([q]))[0]), middle)
    t = np.arange(4.0, 3200.0, 4.0)

    def reach(q, d):
        ok = _open(b, q + t[:, None] * d)
        return float(t[np.argmin(ok)] - 4.0) if not ok.all() else float(t[-1])
    for _ in range(2):  # the walls, but never past the tube's own space: a side opening or a thin partition isn't its edge
        off_u, off_v = float((p - middle) @ u), float((p - middle) @ v)
        lo_u, hi_u = max(-reach(p, -u), bounds[0] - off_u), min(reach(p, u), bounds[2] - off_u)
        lo_v, hi_v = max(-reach(p, -v), bounds[1] - off_v), min(reach(p, v), bounds[3] - off_v)
        p = p + u * (lo_u + hi_u) / 2 + v * (lo_v + hi_v) / 2
    half_u, half_v = (hi_u - lo_u) / 2, (hi_v - lo_v) / 2
    return p, [(p + x * half_u * u + y * half_v * v).round().tolist() for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _straighten(b, c, cells, room):
    """The walk outward cuts on a slant (cells at one walking distance); make the cut a straight slice across the tube, square
    to it, and record its outline and centre."""
    path, mid, sect, along, tan = np.array(c["path"], float), c["mid"], c["sections"], c["dists"] / INCH, c["tan"]
    middle, c["outline"] = _frame(b, c, cells, path[mid], tan)
    c["middle"] = middle.round().tolist()
    reach = max(WINDOW_M / INCH, float(np.ptp(cells[sect[mid]], axis=0).max() + 1) * CELL)
    near = np.concatenate([sect[i] for i in np.flatnonzero(np.abs(along - along[mid]) <= reach)])
    ahead = ((cells[near] + 0.5) * CELL - middle) @ tan > 0
    behind_room, ahead_room = c["rooms"]
    room[near[ahead]], room[near[~ahead]] = ahead_room, behind_room


def _route_rooms(b, cells, room):
    """Rooms holding a level change, a landmark or a start point: they count however small."""
    marks = _marks(b, ("info_landmark", "info_player_start", "trigger_changelevel"))
    if not len(marks) or not (room >= 0).any():
        return set()
    from scipy.spatial import cKDTree
    has = np.flatnonzero(room >= 0)
    near = cKDTree((cells[has] + 0.5) * CELL).query(marks)[1]
    return set(room[has[near]].tolist())


def _weakest(groups, m3, marked, n, least):
    """Index of a group of cuts to undo (the smallest side first): one that doesn't part anything, or has too little beyond it (under least, m3)."""
    worst, worst_m3 = None, np.inf
    rooms = np.arange(n)
    for i, es in enumerate(groups):
        if any(e is None for e in es):
            return i
        side = _side(groups, i, es[0][0], n)
        if side is None:
            return i
        a, b = float(m3[side].sum()), float(m3[~side].sum())
        here = set(rooms[side].tolist())
        enough = (a >= least or marked & here) and (b >= least or marked - here)
        if not enough and min(a, b) < worst_m3:
            worst, worst_m3 = i, min(a, b)
    return worst


def _side(groups, skip, start, n):
    """Rooms reached from start without crossing group skip; None if any of its cuts' far ends is reached too."""
    nbrs = defaultdict(list)
    for i, es in enumerate(groups):
        if i != skip:
            for e in es:
                if e is not None:
                    nbrs[e[0]].append(e[1])
                    nbrs[e[1]].append(e[0])
    seen, todo = {start}, [start]
    while todo:
        for y in nbrs[todo.pop()]:
            if y not in seen:
                seen.add(y)
                todo.append(y)
    if any(e[0] in seen and e[1] in seen for e in groups[skip]):
        return None
    mask = np.zeros(n, bool)
    mask[list(seen)] = True
    return mask


class Part:
    """One room of a cut map, usable wherever a map is. Its coordinates are the map's."""

    def __init__(self, b, n, faces, volume):
        self.name, self.map, self.faces, self._volume = f"{b.name}#{n}", b.name, faces, volume
        self.entities, self.models, self.vis, self.overlays = b.entities, b.models, b.vis, b.overlays
        self.pieces, self.face_piece, self.piece_grid = [], None, None

    def volume(self):
        return self._volume


def proof(b, c):
    """What says a connector is what it is: {text, box: (lo, hi) of the lift or ladder, or None}."""
    up = abs(c["along"][2]) >= VERTICAL * float(np.linalg.norm(c["along"]))
    if c["way"] == "ride" and not up:
        return {"text": "Confirmed by the game: the path of a train the player rides runs through it (magenta line).", "box": None}
    if c["way"] == "walk":
        return {"text": "Found by shape alone: long, narrow, walled and roofed.", "box": None}
    if c["way"] == "crouch":
        return {"text": "Too low to stand in: the player crouches through it.", "box": None}
    if c["way"] == "fall":
        return {"text": "A drop with nothing found running through it, marked by hand.", "box": None}
    found = _carrier(b, np.array(c["path"], float), "lift" if c["way"] == "ride" else "ladder")
    if found is None:
        return {"text": "No entity found running through it.", "box": None}
    e, lo, hi = found
    what = "a lift or moving platform travels through it" if c["way"] == "ride" else "a ladder runs up it"
    return {"text": f"Confirmed by the game: {what} ({e['classname']}, {e.get('targetname') or 'unnamed'}), boxed in cyan.", "box": (lo, hi)}


def _as_read(game, b):
    """Names a map as read: its file, and which of its faces are left out of the facility (bmab.pieces, after what people say), as
    its space is the rest."""
    from .game import Game
    out = np.array([i for i, f in enumerate(b.faces) if f.outside], np.int64)
    return f"{Game(sources.DATA / game).version(b.name)['sha256'][:16]}.{hashlib.sha256(out.tobytes()).hexdigest()[:8]}"


def _cached(game, b):
    """find(b), kept on disk per map as read and per version of this code and of the map reader: finding is slow and the same every time."""
    folder = sources.CACHE / "connectors" / game
    version = hashlib.sha256(Path(__file__).read_bytes() + Path(bsp.__file__).read_bytes() + inspect.getsource(columns).encode()).hexdigest()[:10]
    path = folder / f"{b.name}.{_as_read(game, b)}.{version}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return json.loads(str(z["conn"])), z["cells"], z["room"]
    conn, cells, room = find(b)
    if cells is not None:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(path, conn=json.dumps(conn), cells=cells, room=room)
    return conn, cells, room


def measured(r):
    """What a claimed connector's shape says, in metres: its length and rise along its middle line, its outline's area, and whether
    it is cut on a bend (straightness within 3 m of the cut). The redlines keep only the shape."""
    path, ring = np.array(r["path"], float), np.array(r["outline"], float)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] * INCH
    mid = int(np.argmin(np.linalg.norm(path - np.array(r["at"], float), axis=1)))
    w = np.flatnonzero(np.abs(along - along[mid]) <= 3.0)
    span = float(along[w[-1]] - along[w[0]])
    straight = float(np.linalg.norm(path[w[-1]] - path[w[0]])) * INCH / span if span > 0 else 1.0
    return {"length_m": round(float(along[-1]), 1), "rise_m": round(float(np.abs(np.diff(path[:, 2])).sum()) * INCH, 1),
            "section_m2": round(float(np.linalg.norm(ring[1] - ring[0]) * np.linalg.norm(ring[3] - ring[0])) * INCH ** 2, 1), "curved": straight < CURVED}


def way_of(b, r):
    """How the player gets through a connector placed by hand (claims.WAYS), worked out as for one the detectors find."""
    path = np.array(r["path"], float)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    t = np.linspace(0, along[-1], 9)
    line = np.column_stack([np.interp(t, along, path[:, i]) for i in range(3)])
    m = measured(r)
    return _kind(b, line, m["rise_m"], m["length_m"], _paths(b), low=True)


def _claimed_tube(r, cells, g):
    """A connector the redlines claim, in the form _separate takes; None if no cell lies in its outline. Its slice is the part of the
    slab around the plane that is joined to the outline: all of the passage there, and nothing the passage only runs past."""
    tan = np.array(r["along"], float)
    tan /= np.linalg.norm(tan)
    shift = tan * r.get("start_m", 0.0) / INCH
    ring = np.array(r["outline"], float) + shift
    u, v = ring[1] - ring[0], ring[3] - ring[0]
    half_u, half_v = float(np.linalg.norm(u)) / 2, float(np.linalg.norm(v)) / 2
    rel = (cells + 0.5) * CELL - ring.mean(0)
    across = np.abs(rel @ (u / (2 * half_u))), np.abs(rel @ (v / (2 * half_v)))
    along, near = rel @ tan, (across[0] <= half_u + SLICE_REACH) & (across[1] <= half_v + SLICE_REACH)
    outline = (across[0] <= half_u + CELL / 2) & (across[1] <= half_v + CELL / 2)

    def joined(half):
        at = np.flatnonzero(near & (np.abs(along) <= half))
        lab = csgraph.connected_components(g[at][:, at], directed=False)[1] if len(at) else np.zeros(0, np.int64)
        return at[np.isin(lab, lab[outline[at]])]
    cut = joined(SLICE)
    if not len(cut):
        return None
    middle = np.array(r["at"], float) + shift
    window = joined(WINDOW_M / INCH)

    def straight_on(sign):
        band = (sign * along > SLICE) & (sign * along <= SLICE + 3 * CELL)
        ahead = np.flatnonzero(outline & band)
        return ahead if len(ahead) else window[band[window]]
    return {"id": r["id"], "sections": [window], "mid": 0, "dists": np.zeros(1), "cut": cut, "sides": [straight_on(-1), straight_on(1)], "tan": tan,
            "way": r["way"], **measured(r),
            "path": [middle.tolist()], "middle": middle.round().tolist(), "claimed": {"outline": ring.round().tolist(), "path": r["path"]}}


def cut_as_claimed(b, claims):
    """Cut a map where the claims say: (connectors as find gives them, each with its claim's id, cells, the room of each cell)."""
    cells, g, _roofed = space_graph(b)
    if not len(cells):
        return [], cells, np.zeros(0, np.int64)
    tubes = [t for t in (_claimed_tube(r, cells, g) for r in claims) if t is not None]
    index = {t["id"]: k for k, t in enumerate(tubes)}
    parent = list(range(len(tubes)))

    def top(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for r in claims:
        for other in r.get("stretches_with", []) if r["id"] in index else ():
            if other in index:
                parent[top(index[r["id"]])] = top(index[other])
    tied = defaultdict(list)
    for k in range(len(tubes)):
        tied[top(k)].append(k)
    conn, cells, room = _separate(b, cells, g, tubes, list(tied.values()))
    for c in conn:
        c.update(c.pop("claimed"))
        c["middle"] = [float(x) for x in c["middle"]]
    return conn, cells, room


CUT_CODE = (columns, space_graph, _passable, measured, _claimed_tube, cut_as_claimed, _separate, _orient, _weakest, _side, _straighten, _frame, _limit, _far_rooms, _route_rooms)


def claim_key(claims):
    """Names one way of cutting a map: its claims, the code that turns them into rooms, and the map reader."""
    shaping = json.dumps([{k: v for k, v in r.items() if k not in NOT_SHAPING} for r in claims])
    return hashlib.sha256((shaping + "".join(inspect.getsource(f) for f in CUT_CODE) + Path(bsp.__file__).read_text(encoding="utf-8")).encode()).hexdigest()[:10]


def claimed(b, claims, game):
    """The map (of game) cut at its connector claims: (connectors, cells, the room of each cell); ([], None, None) without any.
    Kept on disk per map as read and way of cutting (claim_key)."""
    if not claims or getattr(b, "vis", None) is None:
        return [], None, None
    folder = sources.CACHE / "cuts" / game
    path = folder / f"{b.name}.{_as_read(game, b)}.{claim_key(claims)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return json.loads(str(z["conn"])), z["cells"], z["room"]
    conn, cells, room = cut_as_claimed(b, claims)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{b.name}.*.npz"):
        if old.name.split(".", 1)[0] == b.name:
            old.unlink()
    np.savez(path, conn=json.dumps(conn), cells=cells, room=room)
    return conn, cells, room


def _sky_share(b, cells, room):
    """{room: (lowest sky z, units; m2 of sky)} of the sky columns over the cut map's rooms."""
    from .solve import CELL as C
    sk = columns(b)[1]
    out = defaultdict(list)
    columns_at = {}
    for (x, y, _z), r in zip(cells.tolist(), room.tolist()):
        columns_at.setdefault((x, y), r)
    for at, (zl, floors) in sk.items():
        r = columns_at.get(at)
        out[r].extend(z for z in zl if any(zf < z for zf in floors))
    return {r: (min(z) * C, len(z) * CELL_M ** 2) for r, z in out.items() if r is not None and z}


def _section(c):
    """The cells (n x 3, the map's own frame) a cut's cross-section fills: what a stretched connector sweeps along its gap."""
    ring = np.array(c["outline"], float)
    u, v = ring[1] - ring[0], ring[3] - ring[0]
    nu, nv = max(1, int(round(float(np.linalg.norm(u)) / CELL))), max(1, int(round(float(np.linalg.norm(v)) / CELL)))
    a, b = np.meshgrid((np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv, indexing="ij")
    return np.unique(np.floor((ring[0] + a.ravel()[:, None] * u + b.ravel()[:, None] * v) / CELL).astype(np.int64), axis=0)


def cut_frame(at, along, outline):
    """A connector's cut: (middle, along, across, across, half width, half height), units."""
    tan, ring = np.array(along, float), np.array(outline, float)
    u, v = ring[1] - ring[0], ring[3] - ring[0]
    hu, hv = float(np.linalg.norm(u)) / 2, float(np.linalg.norm(v)) / 2
    return np.array(at, float), tan / np.linalg.norm(tan), u / (2 * hu), v / (2 * hv), hu, hv


def keeps_shape(b, o, metres):
    """Whether the passage still has its cut's walls this far along it."""
    middle, tan, u, v, hu, hv = o
    p, t = middle + tan * metres / INCH, np.arange(4.0, hu + hv + 64.0, 4.0)

    def reach(d):
        ok = _open(b, p + t[:, None] * d)
        return float(t[np.argmin(ok)] - 4.0) if not ok.all() else float(t[-1])
    return all(abs(reach(d) - h) <= SHAPE_FIT for d, h in ((u, hu), (-u, hu), (v, hv), (-v, hv)))


def whole(b, at, along, outline):
    """How far the passage keeps its cut's shape each way: [behind, ahead], metres along it from the cut. All of that could be taken out."""
    o = cut_frame(at, along, outline)

    def run(sign):
        s = 0.0
        while s < WHOLE_MOST and keeps_shape(b, o, sign * (s + WHOLE_STEP)):
            s += WHOLE_STEP
        return s
    return [-run(-1), run(1)]


SLAB_M = (3.0, 2.0, 1.2)  # lengths tried for the piece of passage that is repeated, longest first


def slab(b, at, along, outline):
    """The piece of passage just ahead of a cut that keeps the cut's shape, which copies of fill the gap when the passage runs
    longer: {m: its length, clean: whether the walls keep their shape that far, back_m: how far back along the passage there is
    room to stand}, metres."""
    o = cut_frame(at, along, outline)
    middle, tan = o[0], o[1]
    depth, clean = SLAB_M[-1], False
    for metres in SLAB_M:
        if keeps_shape(b, o, metres):
            depth, clean = metres, True
            break
    behind = _open(b, middle - np.arange(4.0, 640.0, 4.0)[:, None] * tan)
    back = float(np.arange(4.0, 640.0, 4.0)[np.argmin(behind)] - 4.0) if not behind.all() else 636.0
    return {"m": depth, "clean": clean, "back_m": round(back * INCH, 1)}


def split(maps, placed, links, anchor, locations=None, surface=None, tiers=None, games=None, log=print, shorter=False, said=None):
    """Cut every map with connectors into its rooms, joined by links of kind inner-<kind>. Other maps stay as they are;
    links, the anchor, relations and the surface reference move to the room holding them. Rooms of two levels that hold the same
    place are tied to each other, so that a corridor both levels have can't come apart. shorter: a connector may lose its whole
    length (bmab.connectors.whole, or what the claim says). said: {map: its claims}. Returns the same arguments."""
    from .solve import _agreeing, _landmark
    from scipy.spatial import cKDTree
    tiers, games, said = tiers or {}, games or {}, said or {}
    out_maps, out_placed, out_tiers, out_games, inner, cut = {}, {}, {}, {}, [], {}
    free_of_ground = []  # a map's ground rule stays with one of its rooms, as it was one rule for the whole map
    for n, b in ((n, maps[n]) for n in placed):
        mine = [r for r in said.get(n, ()) if r["kind"] == "connector"]
        conn, cells, room = claimed(b, mine, games.get(n, ""))
        left_out = {r["id"] for r in mine} - {c["id"] for c in conn}
        if left_out and cells is not None:
            log(f"{n}: not cut at {', '.join(sorted(left_out))}: that doesn't part the map")
        if not conn:
            out_maps[n], out_placed[n] = b, placed[n]
            out_tiers[n], out_games[n] = tiers.get(n, 1), games.get(n, "")
            continue
        centre = (cells + 0.5) * CELL
        tree = cKDTree(centre)
        face_room = room[tree.query(np.array([f.verts.mean(0) for f in b.faces]))[1]] if b.faces else np.zeros(0, np.int64)
        keys_of = {}
        sky = _sky_share(b, cells, room)
        carrier = _ground_carrier(sky, cells, room, surface)
        names = []
        for r in range(int(room.max()) + 1):
            part_cells = cells[room == r]
            keys = np.sort(_encode(part_cells))
            z, m2 = sky.get(r, (None, 0.0)) if r == carrier else (None, 0.0)
            if r == carrier and sky:
                z, m2 = min(z_ for z_, _m in sky.values()), sum(m for _z, m in sky.values())
            part = Part(b, r, [f for f, fr in zip(b.faces, face_room.tolist()) if fr == r], (keys, np.array([part_cells.min(0), part_cells.max(0)]), (z, m2)))
            keys_of[r] = keys
            out_maps[part.name], out_placed[part.name] = part, placed[n]
            out_tiers[part.name], out_games[part.name] = tiers.get(n, 1), games.get(n, "")
            names.append(part.name)
            if r != carrier:
                free_of_ground.append(part.name)
        cut[n] = {"tree": tree, "room": room, "names": names, "keys": keys_of}
        by_id = {r["id"]: r for r in mine}
        for c in conn:
            can_lose = 0.0
            if shorter:
                r = by_id[c["id"]]
                first, last = r.get("whole_m") or whole(b, r["at"], r["along"], r["outline"])
                can_lose = (last - first) / INCH
            inner.append({"maps": [names[c["rooms"][0]], names[c["rooms"][1]]], "landmark": f"connector, {c['length_m']} m",
                          "kind": f"inner-{c['way']}", "gives": False, "delta": np.zeros(3), "game": games.get(n, ""), "tier": tiers.get(n, 1),
                          "axis": np.array(c["along"], float), "length_m": c["length_m"], "path": c["path"], "middle": c["middle"],
                          "section": _section(c), "inner": len(inner), "shorter": can_lose})

    def part_at(n, p):
        if n not in cut:
            return n
        return cut[n]["names"][int(cut[n]["room"][cut[n]["tree"].query(p)[1]])]

    def largest(n):
        if n not in cut:
            return n
        return cut[n]["names"][int(np.argmax(np.bincount(cut[n]["room"])))]

    out_links = list(inner)
    for l in links:
        a, b = l["maps"]
        if a not in placed or b not in placed:
            continue
        if l["kind"] == "same" or (a not in cut and b not in cut):
            pa, pb = _same_ends(a, b, l, cut, _agreeing, maps) if l["kind"] == "same" else (a, b)
        else:
            mark = l["landmark"]
            pa, pb = part_at(a, _landmark(maps[a], mark)), part_at(b, _landmark(maps[b], mark))
            if l["kind"] != "teleport" and not l.get("gives"):
                out_links += [{**l, "maps": [x, y], "kind": "same", "landmark": f"same place, {n} cells", "fixed": True}
                              for n, x, y in _sharing(a, b, l, cut, _agreeing, maps) if n >= MIN_SHARED and (x, y) != (pa, pb)]
        out_links.append({**l, "maps": [pa, pb]})
    locs = locations
    if locations:
        locs = dict(locations)
        locs["here"] = [{**e, "maps": [x for m in e["maps"] for x in (cut[m]["names"] if m in cut else [m])]} for e in locations.get("here", [])]
        locs["adjacent"] = [{**a, "from": f"{g}:{largest(m)}"} for a in locations.get("adjacent", []) for g, m in [a["from"].split(":")]]
    surf = surface
    if surface:
        surf = dict(surface)
        if surface.get("reference") in cut:
            names = cut[surface["reference"]]["names"]
            surf["reference"] = max(names, key=lambda x: out_maps[x].volume()[2][1])
        surf["exclude"] = [{**e, "map": p} for e in surface.get("exclude", []) for p in (cut[e["map"]]["names"] if e["map"] in cut else [e["map"]])]
        surf["exclude"] += [{"map": p, "why": "room of a cut map: the map's rule stays with one room"} for p in free_of_ground]
    log(f"{len(cut)} of {len(placed)} maps cut into {sum(len(c['names']) for c in cut.values())} rooms by {len(inner)} connectors")
    return out_maps, out_placed, out_links, largest(anchor), locs, surf, out_tiers, out_games


def _ground_carrier(sky, cells, room, surface):
    """The room that keeps the map's ground rule: the one with the lowest open sky if it is a sky map, else the highest."""
    whole = sum(m2 for _z, m2 in sky.values())
    if surface and whole >= surface.get("min_sky_m2", 500):
        return min(sky, key=lambda r: sky[r][0])
    top = np.zeros(int(room.max()) + 1, np.int64) - 1 << 40
    np.maximum.at(top, room, cells[:, 2])
    return int(np.argmax(top))


def _sharing(a, b, l, cut, agreeing, maps):
    """[(voxels both hold, room of a, room of b)] of the rooms of a and b at a link's alignment."""
    from . import solve
    rel = np.round(np.asarray(l["delta"], float) / CELL).astype(np.int64)

    def rooms(n):
        if n in cut:
            return [(name, cut[n]["keys"][r]) for r, name in enumerate(cut[n]["names"])]
        return [(n, solve.volume(maps[n])[0])]
    return [(len(agreeing(ka, kb, rel)), na, nb) for na, ka in rooms(a) for nb, kb in rooms(b)]


def _same_ends(a, b, l, cut, agreeing, maps):
    """The rooms of a and b that a link saying they are the same room joins: the pair sharing the most voxels."""
    return max(_sharing(a, b, l, cut, agreeing, maps), key=lambda x: x[0])[1:]
