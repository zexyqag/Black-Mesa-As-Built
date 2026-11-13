"""What a part claim bounds (bmab.claims): the open space inside its boxes that is joined to the first box's middle, and the
faces around that space, cut at the boxes. A box can hold more than its part (a room beside it, the end of a corridor): only what
is joined to the middle counts, so a part needn't be a box, and extra boxes (turned as the first) make the shapes one can't hold
alone, such as an L-shaped room with a hallway in its corner.

The space is the map's open space sampled every STEP units across the boxes, joined face to face; it starts from the open point
nearest the middle. A face is the part's where it faces into that space: from the middle of what lies inside a box, a few steps
out along its normal reach the space (Source's open space pads the brushes, so the steps go that far).

A part's openings are where something can join it (opening claims, bmab.claims). Where its space runs out through a side of
its boxes and the map goes on past it, there is an opening the size of the hole; level changes, fake and ought doors and fake
outs that lead on inside the space are openings too, though the space doesn't leave the boxes there. What stands in an opening
says what it is (a level change, a connector, a fake or ought door, a fake out that leads on, a door, else the map just goes on
through it, or nothing is past it: one placed by hand on a wall), worked out again whenever it is asked (named), as the claims
around it may have changed.
"""
import numpy as np
from scipy import ndimage, spatial

from . import brushes, bsp, connectors, doors, fake_outs, levels

STEP = 16           # units between samples of the boxes' space
MOST = 4_000_000    # samples: bigger boxes are sampled more coarsely
DEPTH = 48          # units the space must go on past a box's side for an opening: a box stopping short of a wall leaves a sliver
FINE = 4            # units between samples when an opening is measured
SECTION = 256       # units either way of a fake out within which its passage is measured
NEAR = 64           # units from an opening within which a sign is on it or beside it
LEVEL_REACH = 768   # units around a level change within which the next level's space is set against this one's
FALL = 64           # units the floor past an opening in a side may lie below its bottom before the way through is a drop
KINDS = ("level_change", "connector", "fake_door", "ought_door", "fake_out", "door", "goes_on", "nothing")  # what an opening is: the first that applies


def frame(line):
    """A part's boxes: (the first one's middle, the three axes as unit vectors: length, width, height, and each box as its middle
    in those axes from the first one's and half its size)."""
    f = np.array(line["facing"], float)
    f /= np.linalg.norm(f)
    middle = np.array(line["at"], float)
    axes = np.array([f, [-f[1], f[0], 0.0], [0.0, 0.0, 1.0]])
    boxes = [(np.zeros(3), np.array(line["size"], float) / 2)]
    boxes += [((np.array(m["at"], float) - middle) @ axes.T, np.array(m["size"], float) / 2) for m in line.get("extra_boxes", [])]
    return middle, axes, boxes


def space(b, line):
    """The open space of a part: (a boolean grid over its boxes, the step, the grid's first sample, in the part's axes)."""
    middle, axes, boxes = frame(line)
    lo = np.min([c - h for c, h in boxes], 0)
    hi = np.max([c + h for c, h in boxes], 0)
    step = float(STEP)
    while np.prod(np.ceil((hi - lo) / step)) > MOST:
        step *= 2
    n = np.maximum(np.ceil((hi - lo) / step).astype(int), 1)
    first = lo + step / 2
    grid = np.stack(np.meshgrid(*[first[k] + step * np.arange(n[k]) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
    within = np.zeros(len(grid), bool)
    for c, h in boxes:
        within |= (np.abs(grid - c) <= h).all(1)
    open_ = np.zeros(len(grid), bool)
    open_[within] = b.vis.open(middle + grid[within] @ axes) if b.vis is not None else True
    open_ = open_.reshape(n)
    if not open_.any():
        return open_, step, first
    labels, _ = ndimage.label(open_)
    at = np.argwhere(open_)
    start = at[np.argmin(np.linalg.norm(first + at * step, axis=1))]
    return labels == labels[tuple(start)], step, first


def _clip(poly, centre, half):
    """A polygon (the part's axes) cut to one box; maybe empty."""
    for axis in range(3):
        for sign in (1, -1):
            if not len(poly):
                return poly
            d = half[axis] - sign * (poly[:, axis] - centre[axis])
            out = []
            for k in range(len(poly)):
                a, c, da, dc = poly[k], poly[(k + 1) % len(poly)], d[k], d[(k + 1) % len(poly)]
                if da >= 0:
                    out.append(a)
                if (da >= 0) != (dc >= 0):
                    out.append(a + (c - a) * (da / (da - dc)))
            poly = np.array(out).reshape(-1, 3)
    return poly


def _bounds(b):
    """Each face's lowest and highest corner where the map has it (n x 3 each), kept on the map: bsp.py's text keys the connector
    caches, so not a field."""
    if getattr(b, "face_bounds", None) is None:
        corners = [brushes.place(b, f)[0] for f in b.faces]
        b.face_bounds = np.array([v.min(0) for v in corners]).reshape(-1, 3), np.array([v.max(0) for v in corners]).reshape(-1, 3)
    return b.face_bounds


def faces(b, line):
    """Indices into b.faces of the part's faces, and its space in cubic metres."""
    middle, axes, boxes = frame(line)
    inside, step, first = space(b, line)
    depths = (step / 2, step * 1.5, step * 2.5) if isinstance(b.vis, bsp.SourceVis) else (step / 2, step * 1.5)
    lo, hi = _bounds(b)
    near = np.zeros(len(lo), bool)
    for c, h in boxes:
        mid, reach = middle + c @ axes, np.abs(axes).T @ h  # the box's middle and half extent along x, y and z
        near |= ((hi >= mid - reach - 1) & (lo <= mid + reach + 1)).all(1)
    out = []
    for i in np.flatnonzero(near).tolist():
        f = b.faces[i]
        if f.skybox or bsp.is_tool(f.texture.lower()):
            continue
        v, n = brushes.place(b, f)
        local, n = (v - middle) @ axes.T, n @ axes.T
        if any(_faces_into(_clip(local, c, h), n, depths, inside, step, first) for c, h in boxes):
            out.append(i)
    return out, float(inside.sum()) * (step * 0.0254) ** 3


def _faces_into(poly, n, depths, inside, step, first):
    """Whether a face's piece inside a box faces into the space: a few steps out along its normal from its middle land in it."""
    if len(poly) < 3:
        return False
    for d in depths:
        k = np.floor((poly.mean(0) + n * d - first) / step + 0.5).astype(int)
        if (k >= 0).all() and (k < inside.shape).all() and inside[tuple(k)]:
            return True
    return False


def _within(boxes, p):
    """Which points (the part's axes) lie in any of the boxes."""
    w = np.zeros(len(p), bool)
    for c, h in boxes:
        w |= (np.abs(p - c) <= h + 1e-6).all(1)
    return w


def _exit(boxes, p, axis, sign):
    """How far points (the part's axes) go along an axis, one way, before they leave the boxes."""
    t = np.zeros(len(p))
    for _ in boxes:  # each round leaves one box, perhaps into another
        q = p.copy()
        q[:, axis] += sign * t
        more = np.zeros(len(p))
        for c, h in boxes:
            held = (np.abs(q - c) <= h + 1e-6).all(1)
            more = np.where(held, np.maximum(more, h[axis] - sign * (q[:, axis] - c[axis])), more)
        if not (more > 1e-6).any():
            break
        t += more
    return t


def _goes_on(b, middle, axes, boxes, p, d):
    """Which points (the part's axes) on a box's side have the map's open space going on DEPTH units past them, out of the boxes."""
    ok = np.ones(len(p), bool)
    for s in np.arange(8.0, DEPTH + 1, 8.0):
        q = p + d * s
        ok &= ~_within(boxes, q)
        ok[ok] = b.vis.open(middle + q[ok] @ axes)
    return ok


def _extent(keep, gu, gv, pad):
    """lo and hi (two axes) of the kept samples of a FINE grid, grown by half a sample and pad."""
    iu, iv = np.nonzero(keep)
    return np.array([gu[iu].min(), gv[iv].min()]) - FINE / 2 - pad, np.array([gu[iu].max(), gv[iv].max()]) + FINE / 2 + pad


def _pad(b):
    """Units Source's open space stops short of a wall (bsp.SourceVis pads the brushes); GoldSrc's reaches it."""
    return bsp.SOLID_PAD + bsp.SOLID_CELL / 2 if isinstance(b.vis, bsp.SourceVis) else 0.0


def _holes(b, middle, axes, boxes, inside, step, first):
    """Where the part's space runs out through its boxes' sides and the map goes on: [((axis, sign, the side's plane, lo, hi of the
    hole on it in the two other axes), its open samples across it (n x 2, those axes))], all in the part's axes. A hole is
    measured on a FINE grid within a step of the samples that found it: the wall is no further than that."""
    at = np.argwhere(inside)
    local = first + at * step
    lo_all, hi_all = np.min([c - h for c, h in boxes], 0), np.max([c + h for c, h in boxes], 0)
    out = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (1, -1):
            d = np.zeros(3)
            d[axis] = sign
            t = _exit(boxes, local, axis, sign)
            edge = np.flatnonzero(t <= step)
            side = local[edge] + d * t[edge, None]
            ok = _goes_on(b, middle, axes, boxes, side, d)
            if not ok.any():
                continue
            mask, plane = np.zeros(inside.shape, bool), np.zeros(inside.shape)
            mask[tuple(at[edge[ok]].T)] = True
            plane[tuple(at[edge[ok]].T)] = side[ok, axis]
            labels, n = ndimage.label(mask, np.ones((3, 3, 3)))
            for g in range(1, n + 1):
                cells = np.argwhere(labels == g)
                p = float(np.median(plane[tuple(cells.T)]))
                across = first[[u, v]] + cells[:, [u, v]] * step
                lo, hi = np.maximum(across.min(0) - step, lo_all[[u, v]]), np.minimum(across.max(0) + step, hi_all[[u, v]])
                gu, gv = np.arange(lo[0] + FINE / 2, hi[0], FINE), np.arange(lo[1] + FINE / 2, hi[1], FINE)
                q = np.zeros((len(gu), len(gv), 3))
                q[..., axis], q[..., u], q[..., v] = p, gu[:, None], gv[None, :]
                open_ = _goes_on(b, middle, axes, boxes, q.reshape(-1, 3), d).reshape(len(gu), len(gv))
                mine = np.zeros((inside.shape[u], inside.shape[v]), bool)
                mine[cells[:, u], cells[:, v]] = True
                ku = np.clip(np.floor((gu - first[u]) / step + 0.5).astype(int), 0, inside.shape[u] - 1)
                kv = np.clip(np.floor((gv - first[v]) / step + 0.5).astype(int), 0, inside.shape[v] - 1)
                pieces, _ = ndimage.label(open_, np.ones((3, 3)))
                found = np.unique(pieces[open_ & mine[ku][:, kv]])
                keep = np.isin(pieces, found[found > 0])
                if keep.any():
                    lo, hi = _extent(keep, gu, gv, _pad(b))
                    lo, hi = np.maximum(lo, lo_all[[u, v]]), np.minimum(hi, hi_all[[u, v]])
                    iu, iv = np.nonzero(keep)
                    across = np.unique(np.round(np.stack([gu[iu], gv[iv]], 1) / 8) * 8, axis=0)
                else:
                    lo, hi = across.min(0) - step / 2, across.max(0) + step / 2
                out.append(((axis, sign, p, lo, hi), across))
    return out


def _section(b, middle, axes, boxes, at, axis, sign):
    """lo and hi (the two other axes) of the passage across a point (the part's axes) just before it, square to an axis: the open
    space within the boxes and SECTION units of the point, joined to the point."""
    u, v = [a for a in range(3) if a != axis]
    lo_all, hi_all = np.min([c - h for c, h in boxes], 0), np.max([c + h for c, h in boxes], 0)
    lo, hi = np.maximum(at[[u, v]] - SECTION, lo_all[[u, v]]), np.minimum(at[[u, v]] + SECTION, hi_all[[u, v]])
    gu, gv = np.arange(lo[0] + FINE / 2, hi[0], FINE), np.arange(lo[1] + FINE / 2, hi[1], FINE)
    if not len(gu) or not len(gv):
        return None
    q = np.zeros((len(gu), len(gv), 3))
    q[..., axis], q[..., u], q[..., v] = at[axis] - sign * STEP, gu[:, None], gv[None, :]
    q = q.reshape(-1, 3)
    open_ = _within(boxes, q)
    open_[open_] = b.vis.open(middle + q[open_] @ axes)
    open_ = open_.reshape(len(gu), len(gv))
    if not open_.any():
        return None
    pieces, _ = ndimage.label(open_, np.ones((3, 3)))
    iu, iv = np.nonzero(open_)
    k = np.argmin((gu[iu] - at[u]) ** 2 + (gv[iv] - at[v]) ** 2)
    return _extent(pieces == pieces[iu[k], iv[k]], gu, gv, _pad(b))


def _samples(lo, hi, most=16):
    """Points across a box (world), at most most along each axis, its corners among them."""
    n = [min(max(int(np.ceil((hi[k] - lo[k]) / STEP)) + 1, 2), most) for k in range(3)]
    g = np.meshgrid(*[np.linspace(lo[k], hi[k], n[k]) for k in range(3)], indexing="ij")
    return np.stack(g, -1).reshape(-1, 3)


def _along(path):
    """Points every 8 units along a path (world)."""
    path = np.asarray(path, float)
    out = [path[:1]]
    for a, c in zip(path[:-1], path[1:]):
        n = max(int(np.linalg.norm(c - a) / 8), 1)
        out.append(a + (c - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def _things(b, claims, fired):
    """What can stand in an opening: [(what it says of the opening, points it covers (world), whether it only stands in one it
    passes through: a connector or a train's path, which may end at a door without going through it)], in KINDS order."""
    out = [({"kind": "level_change", "map": m}, _samples(lo, hi), False) for m, _landmark, lo, hi in levels.triggers(b)]
    out += [({"kind": "connector", "claim": r["id"], "way": r.get("way")}, _along(r["path"]), True) for r in claims if r["kind"] == "connector" and r.get("path")]
    out += [({"kind": r["kind"], "claim": r["id"], "behind": r.get("behind")}, np.array([r["at"]], float), False)
            for r in claims if r["kind"] in ("fake_door", "ought_door")]
    out += [({"kind": "fake_out", "claim": r["id"], "way": r.get("way")}, np.array([r["at"]], float), False)
            for r in claims if r["kind"] == "fake_out" and r.get("type") == "leads_on"]
    opens = doors.brush_doors(b, fired)
    for e in b.entities:
        m = e.get("model", "")
        if e.get("classname") in levels.DOORS and m[1:].isdigit() and int(m[1:]) in opens:
            lo, hi = b.models[int(m[1:])][:2]
            at = np.array([float(x) for x in e.get("origin", "0 0 0").split()])
            if max((hi - lo)[:2]) >= doors.MIN_WIDTH and (hi - lo)[2] >= doors.MIN_HEIGHT:  # not a lever or a hatch
                out.append(({"kind": "door", "door": "opens" if opens[int(m[1:])] else "shut"}, _samples(lo + at, hi + at), False))
    w, h = doors.PROP_DOOR
    leaf = np.array([[0, s, z] for s in np.linspace(-w / 2, w / 2, 8) for z in np.linspace(-h / 2, h / 2, 8)])
    out += [({"kind": "door", "door": "opens" if o else "shut"}, c + leaf @ np.array([[0, 0, 0], way, [0, 0, 1]]), False) for c, way, o in doors.model_doors(b)]
    out += [({"kind": "goes_on", "way": "ride"}, _along(s), True) for s in connectors._paths(b)]  # a train runs through it
    return sorted(out, key=lambda t: KINDS.index(t[0]["kind"]))


def openings(b, line, claims=(), fired=(), next_level=None):
    """Where something can join a part (line, with its id), as opening claims without ids: [{kind "opening", part, at, facing (a
    unit vector out of the part), size (units: across and up on a side; along the part's length and width on its floor or
    ceiling), way (claims.WAYS)}]. claims: the map's claims; fired: names fired from other maps (doors.fired_names);
    next_level: map -> (its bsp, offset) or None, for the level a level change leads to (a point there plus offset is a point
    here); asked only for a level change in the part.

    The way through is a connector's or a fake out's that stands in the opening, a ride where a train runs through it, else
    guessed (_way).

    A hole a player can't fit through is none. Holes on one side that one door or claim stands in are one opening (a door with a
    post in its middle). A level change inside the space faces the way along its passage where the next level goes on and this one
    doesn't (_toward), or failing that, the shorter way, where this one ends."""
    if b.vis is None:
        return []
    middle, axes, boxes = frame(line)
    inside, step, first = space(b, line)
    if not inside.any():
        return []
    local = lambda p: (np.asarray(p, float) - middle) @ axes.T  # noqa: E731
    things = _things(b, claims, fired)
    found = []
    for side, across in _holes(b, middle, axes, boxes, inside, step, first):
        o = {"side": side, "open": across, "things": _standing(things, side, across, local, step)}
        o["joins"] = {k for k in o["things"] if things[k][0].get("claim") or things[k][0]["kind"] == "door"}
        twin = next((m for m in found if m["side"][:2] == side[:2] and abs(m["side"][2] - side[2]) <= step and m["joins"] & o["joins"]), None)
        if twin is None:
            found.append(o)
            continue
        axis, sign, plane, lo, hi = twin["side"]
        twin["side"] = (axis, sign, plane, np.minimum(lo, side[3]), np.maximum(hi, side[4]))
        twin["open"] = np.concatenate([twin["open"], across])
        twin["things"] |= o["things"]
        twin["joins"] |= o["joins"]
    found = [o for o in found if _passable(o["side"])]
    used = set().union(*[o["things"] for o in found])
    lo_all, hi_all = np.min([c - h for c, h in boxes], 0), np.max([c + h for c, h in boxes], 0)
    for k, (says, pts, _through) in enumerate(things):
        if k in used or says["kind"] not in ("level_change", "fake_door", "ought_door", "fake_out") or not _held(inside, step, first, local(pts)).any():
            continue
        if says["kind"] == "level_change":
            q = local(pts)
            lo, hi = q.min(0), q.max(0)
            axis = int(np.argmin((hi - lo)[:2]))
            c = middle + (lo + hi) / 2 @ axes
            there = next_level(says["map"]) if next_level else None
            sign = _toward(b, *there, c, axes[axis]) if there else None
            if sign is None:
                sign = 1 if _reach(b, c, axes[axis]) <= _reach(b, c, -axes[axis]) else -1
            u, v = [a for a in range(3) if a != axis]
            side = (axis, sign, float((lo + hi)[axis] / 2), np.maximum(lo, lo_all)[[u, v]], np.minimum(hi, hi_all)[[u, v]])
        else:
            r = next(r for r in claims if r["id"] == says["claim"])
            towards = np.array(r["back"] if "back" in r else np.subtract(r["reaches"], r["at"]), float) @ axes.T
            axis = int(np.argmax(np.abs(towards)))
            sign = 1 if towards[axis] >= 0 else -1
            at = local(r["at"])
            u, v = [a for a in range(3) if a != axis]
            if says["kind"] == "fake_out":
                got = _section(b, middle, axes, boxes, at, axis, sign)
                if got is None:
                    continue
                lo, hi = got
            else:
                half = np.array([r["width"] / 2, r["height"] / 2]) if axis < 2 else np.array([r["width"] / 2, r["width"] / 2])
                lo, hi = at[[u, v]] - half, at[[u, v]] + half
            side = (axis, sign, float(at[axis]), lo, hi)
        found.append({"side": side, "open": None, "things": _standing(things, side, None, local, step) | {k}})
        used |= found[-1]["things"]
    out = []
    for o in found:
        axis, sign, plane, lo, hi = o["side"]
        at, facing = _middle(middle, axes, o["side"])
        out.append({"kind": "opening", "part": line["id"], "at": [round(float(x)) for x in at], "facing": [round(float(x), 3) for x in facing],
                    "size": [round(float(x)) for x in hi - lo], "way": _guessed(b, middle, axes, o["side"], _says(things, o["things"]))})
    return out


def way(b, line, opening, claims=(), fired=()):
    """How the player goes through one of a part's openings (line: the part), as openings finds it: for one placed by hand."""
    middle, axes, boxes = frame(line)
    side = _side(middle, axes, opening)
    things = _things(b, claims, fired) if b.vis is not None else []
    across = _across(b, middle, axes, boxes, side) if b.vis is not None else None
    return _guessed(b, middle, axes, side, _says(things, _standing(things, side, across, lambda p: (np.asarray(p, float) - middle) @ axes.T, STEP)))


def _guessed(b, middle, axes, side, says):
    """An opening's way through: what stands in it says (a connector's, a fake out's, a train's), else _way guesses."""
    axis, _sign, _plane, lo, _hi = side
    at, facing = _middle(middle, axes, side)
    return says.get("way") or _way(b, at, facing, middle[2] + lo[1] if axis < 2 else None, says.get("kind") in ("fake_door", "ought_door", "door"))


def named(b, line, found, claims=(), signs=(), fired=()):
    """What stands in each of a part's openings (opening claims), worked out from the map and the claims as they are now: [{type
    (KINDS), ring (its four corners), signs (indices into signs, dicts with an "at", of those on it or beside it), and as they
    apply door ("opens" or "shut"), map (where a level change leads), claim (the connector's, fake or ought door's or fake out's
    id), behind (a fake or ought door's)}], in found's order."""
    middle, axes, boxes = frame(line)
    local = lambda p: (np.asarray(p, float) - middle) @ axes.T  # noqa: E731
    things = _things(b, claims, fired) if b.vis is not None else []
    out = []
    for o in found:
        side = _side(middle, axes, o)
        across = _across(b, middle, axes, boxes, side) if b.vis is not None else None
        says = _says(things, _standing(things, side, across, local, STEP))
        says.pop("way", None)
        axis, _sign, _plane, lo, hi = side
        u, v = [a for a in range(3) if a != axis]
        mid = local(_middle(middle, axes, side)[0])
        corners = []
        for a, c in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])):
            q = mid.copy()
            q[u], q[v] = a, c
            corners.append([round(float(x)) for x in middle + q @ axes])
        out.append({"type": says.pop("kind", "goes_on" if across is not None else "nothing"), "ring": corners,
                    "signs": [k for k, s in enumerate(signs) if _at(side, across, local([s["at"]]), NEAR)], **says})
    return out


def _standing(things, side, across, local, margin):
    """Which things stand in an opening (_at), those that must pass through it only if they do."""
    return {k for k, (_says, pts, through) in enumerate(things) if _at(side, across, local(pts), margin, through)}


def _says(things, picked):
    """What the things standing in an opening say of it: of each field, what the first of them in KINDS order says."""
    out = {}
    for k in sorted(picked):
        for key, v in things[k][0].items():
            if v is not None:
                out.setdefault(key, v)
    return out


def _middle(middle, axes, side):
    """An opening's middle and facing (world) from its side (the part's axes)."""
    axis, sign, plane, lo, hi = side
    u, v = [a for a in range(3) if a != axis]
    mid, d = np.zeros(3), np.zeros(3)
    mid[axis], mid[[u, v]], d[axis] = plane, (lo + hi) / 2, sign
    return middle + mid @ axes, d @ axes


def _side(middle, axes, o):
    """An opening claim's side: (axis, sign, plane, lo, hi), the part's axes, as _holes gives them."""
    f = np.asarray(o["facing"], float) @ axes.T
    axis = int(np.argmax(np.abs(f)))
    at = (np.asarray(o["at"], float) - middle) @ axes.T
    u, v = [a for a in range(3) if a != axis]
    half = np.asarray(o["size"], float) / 2
    return axis, 1 if f[axis] > 0 else -1, float(at[axis]), at[[u, v]] - half, at[[u, v]] + half


def _across(b, middle, axes, boxes, side):
    """The open samples across an opening on its side, as _holes keeps them; None where the map doesn't go on through it (a fake
    door, a fake out's end)."""
    axis, sign, plane, lo, hi = side
    u, v = [a for a in range(3) if a != axis]
    gu, gv = np.arange(lo[0] + FINE / 2, hi[0], FINE), np.arange(lo[1] + FINE / 2, hi[1], FINE)
    q = np.zeros((len(gu), len(gv), 3))
    q[..., axis], q[..., u], q[..., v] = plane, gu[:, None], gv[None, :]
    d = np.zeros(3)
    d[axis] = sign
    open_ = _goes_on(b, middle, axes, boxes, q.reshape(-1, 3), d).reshape(len(gu), len(gv))
    if not open_.any():
        return None
    iu, iv = np.nonzero(open_)
    return np.unique(np.round(np.stack([gu[iu], gv[iv]], 1) / 8) * 8, axis=0)


def _passable(side):
    """Whether a player fits through a hole: 32 units wide, and on a side as high as a crouch."""
    axis, _sign, _plane, lo, hi = side
    return bool((hi - lo)[0] >= 32 and (hi - lo)[1] >= (connectors.CROUCH if axis < 2 else 32))


def _toward(b, there, offset, c, d):
    """Which way along d (1 or -1) from a level change's middle c the next level (there, offset) goes on: the space joined to c
    within LEVEL_REACH that it has and this level hasn't lies mostly that way. None if it shows no way."""
    ax = np.arange(-LEVEL_REACH, LEVEL_REACH + 1, STEP, dtype=float)
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    n, m = len(ax), len(ax) // 2
    joined = []
    for level, at in ((b, c), (there, c - offset)):
        labels, _ = ndimage.label(level.vis.open(grid + at).reshape(n, n, n))
        mid = labels[m - 2:m + 3, m - 2:m + 3, m - 2:m + 3]  # the middle can be in a wall's padding: the samples around it
        joined.append(np.isin(labels, mid[mid > 0]).ravel())
    only = joined[1] & ~joined[0]
    lean = float(grid[only].mean(0) @ d) if only.any() else 0.0
    return None if abs(lean) < STEP else 1 if lean > 0 else -1


def _reach(b, p, d, most=4096):
    """Units from a point along a direction to the first closed space (most if none)."""
    t = np.arange(8.0, most + 1, 8.0)
    shut = np.flatnonzero(~b.vis.open(p[None] + t[:, None] * d))
    return float(t[shut[0]]) if len(shut) else float(most)


def _way(b, at, facing, bottom, door):
    """How the player goes through an opening that nothing in it says: out of a side (bottom: its lowest height) other than a door's,
    into open space over a floor more than FALL below or none, a drop; else as a fake out's passage is guessed, tracks aside (a
    train running through an opening is a ride)."""
    p = at + facing * 48  # from the hole's middle height: Source's padded space can put its bottom a little under the floor
    if bottom is not None and not door and b.vis is not None and b.vis.open(p[None])[0]:
        down = connectors._first_face(connectors.planes(b), p, np.array([0.0, 0.0, -1.0]), at[2] - bottom + 8 * connectors.STAND)
        if down is None or p[2] - down[0] < bottom - FALL:
            return "fall"
    return fake_outs.way(b, {"at": at + facing * 32, "reaches": at + facing * 192}, tracks=False)


def _held(inside, step, first, p):
    """Which points (the part's axes) are in the part's space: the sample each is in, or one beside it."""
    k = np.floor((np.atleast_2d(p) - first) / step + 0.5).astype(int)
    out = np.zeros(len(k), bool)
    for off in np.argwhere(np.ones((3, 3, 3))) - 1:
        j = k + off
        ok = ((j >= 0) & (j < inside.shape)).all(1)
        out[ok] |= inside[tuple(j[ok].T)]
    return out


def _at(side, across, p, margin, through=False):
    """Whether any point (the part's axes) stands in an opening: from margin inside the part to DEPTH and margin past its side, and
    within margin of its open samples across it (across), or of its rectangle where it has none (an opening inside the space).
    through: only if such points lie on both sides of it (a path passing through it)."""
    axis, sign, plane, lo, hi = side
    u, v = [a for a in range(3) if a != axis]
    p = np.atleast_2d(p)
    depth = sign * (p[:, axis] - plane)
    keep = ((depth >= -margin) & (depth <= DEPTH + margin) & (p[:, u] >= lo[0] - margin) & (p[:, u] <= hi[0] + margin)
            & (p[:, v] >= lo[1] - margin) & (p[:, v] <= hi[1] + margin))
    p, depth = p[keep], depth[keep]
    if len(p) and across is not None:
        d, _ = spatial.cKDTree(across).query(p[:, [u, v]], p=np.inf, distance_upper_bound=margin + 4)
        p, depth = p[np.isfinite(d)], depth[np.isfinite(d)]
    return bool(len(p)) and (not through or (depth.min() < 0 < depth.max()))
