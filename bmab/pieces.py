"""Pieces of a map's space that are not the facility.

A piece is a connected part of the map's space (floor cells with the room above them, up to the ceiling).
Cells on Xen floors and on facility floors never join, nor do cells the route can and can't see.
The route is every piece holding a level change, a landmark or a start point. Off the route:
- a Xen pocket: most of the floor and nearly all the walls in Xen textures (glimpses during the resonance
  cascade, Xen islands; Xen growth in a facility hall has facility walls and stays), or holding Opposing
  Force's displacer target or return portal (a start point beside the target doesn't make it the route): dropped
- no spot on the route can see it (the map's visibility table): a sealed room the player never sees, kept
  for the game's workings (holding boxes for cut scenes and level changes, monster and model stores, test
  rooms): dropped
- mostly black and holding a start point, a teleport or a level change: a holding box (where a fresh chapter
  start spawns the player before teleporting them in, or where a cut scene parks them for a level change):
  dropped, even on the route
- holds a teleport destination and is big enough to be a place: reached by teleport, kept (its position in
  its own map was the designer's choice)
- otherwise the route can see it: scenery (a tram's canyon, a hall behind glass), kept
PS2 maps have no visibility table read here, so for them only the Xen and teleport rules apply.
Black floors are voids (the drop under a Xen island), not floor, so they don't count in the Xen share.
What people say (area claims, passed to analyse) overrides a verdict: a point inside the piece, keep or drop.
Each face belongs to the piece in front of it, or failing that the nearest; a face of the latter kind that the
route can see stays when its piece is one the route can't. Faces of dropped pieces are marked outside
(bsp.Face.outside); plan.floors(facility=True) leaves them out.
"""
import bisect

import numpy as np

from . import bsp

CELL = 48            # as solve.CELL
MIN_HEAD, SKY_HEAD, MAX_HEAD = 2, 8, 100
XEN_SHARE = 0.25     # share of a piece's floor area in Xen textures that makes it a Xen pocket...
XEN_ALL = 0.9        # ...if its walls and ceilings are Xen too (Xen growth in a facility hall scores 0.5 or less)
XEN_MARKS = ("info_displacer_xen_target", "trigger_xen_return")  # Opposing Force's displacer pocket
PLACE_M3 = 300       # a teleport destination in a piece this big is a place, not a holding box
ROUTE = ("info_landmark", "info_player_start", "trigger_changelevel")
CHANGE = ("info_landmark", "trigger_changelevel")
HOLD = ("info_player_start", "trigger_teleport", "info_teleport_destination", "trigger_changelevel", "info_landmark")
HOLD_BLACK = 0.75    # share of a piece's face area in black that makes a box holding one of HOLD a holding box


def _raster(polys):
    from .solve import _raster as r  # one rasteriser for the solver and this
    return r(polys)


def _origin(b, e):
    if "origin" in e:
        return np.array([float(x) for x in e["origin"].split()])
    if e.get("model", "").startswith("*"):
        lo, hi = b.models[int(e["model"][1:])][:2]
        return (lo + hi) / 2
    return None


def _label(pts, runs, lo, status=None):
    """Connected pieces of the cells: sideways at the same height, up and down only within one column of open
    space (two columns stacked at one spot are parted by a floor slab, so a room above another never joins it
    through the slab); with status (a category per cell) only cells of the same status join. Heights are doubled
    so the gap between stacked columns stays empty. Returns a label grid over the cells' box and the count."""
    from scipy import ndimage
    size = pts.max(0) - lo + 1
    lab = np.zeros(size, np.int64)
    n = 0
    for want in ((None,) if status is None else np.unique(status).tolist()):
        grid2 = np.zeros((size[0], size[1], 2 * size[2]), bool)
        for x, y, zf, head in runs:
            grid2[x - lo[0], y - lo[1], 2 * (zf - lo[2]):2 * (zf + head - 1 - lo[2]) + 1] = True
        if want is not None:  # keep only cells of this status (and the gaps between them in a column)
            keep = np.zeros(size, bool)
            keep[tuple((pts[status == want] - lo).T)] = True
            mask = np.repeat(keep, 2, axis=2)
            mask[:, :, 1:-1:2] &= keep[:, :, 1:]  # a gap joins two cells of the column only if both are kept
            grid2 &= mask
        lab2, k = ndimage.label(grid2)
        part = lab2[:, :, ::2]
        lab[part > 0] = part[part > 0] + n
        n += k
    return lab, n


def _seen(b, pts, cells_route, marks):
    """Which cells the route can see, from the map's visibility table. The walked leaves grow from the ones
    holding the route's marks: a leaf of the route's cells counts once a walked leaf can see it (so a sealed
    room that the route's cells leak into through a thin wall never does); water, slime and lava joined to the
    route count at once (the table may not see across a water surface, but the player swims in). A cell whose
    centre is inside a wall or in the sky's volume takes the answer of the nearest cell that isn't. Returns
    the cells' answers and the leaves seen; None, None without a visibility table."""
    if b.vis is None or not len(marks):
        return None, None
    from scipy.spatial import cKDTree
    leaves = b.vis.leaf_of((pts + 0.5) * CELL)
    solid = b.vis.solid_leaves(leaves)
    liquid = b.vis.liquid_leaves(leaves)
    route_leaves = set(leaves[cells_route & ~solid].tolist())
    walked = set(b.vis.leaf_of(np.asarray(marks, float)).tolist()) | set(leaves[cells_route & liquid].tolist())
    seen = np.zeros(len(b.vis.leaves), bool)
    done = set()
    while True:
        for leaf in walked - done:
            seen |= b.vis.visible(int(leaf))
            seen[int(leaf)] = True
        done |= walked
        grow = {l for l in route_leaves if seen[l]} - walked
        if not grow:
            break
        walked |= grow
    seen[0] = False
    out = seen[leaves]
    if solid.any() and (~solid).any():
        near = cKDTree(pts[~solid]).query(pts[solid])[1]
        out[solid] = out[~solid][near]
    return out, seen


def analyse(b, areas=None):
    """Classify the map's pieces and mark the faces of dropped ones; cached on the map, so whoever reads a map analyses it first.
    areas: people's whole answer (the map's area claims, once a person has checked all its areas: bmab.redlines.said), which then
    decides which pieces are dropped; None: the rules decide. Each piece's "found" says what the rules alone make of it, whatever people say, so the detectors' claims
    don't depend on what read the map first."""
    if b.pieces is not None:
        return b.pieces
    from . import plan
    floor_faces = [i for i, f in enumerate(b.faces) if f.model == 0 and not f.skybox
                   and f.normal[2] > plan.FLOOR_NZ and not bsp.is_tool(f.texture)]
    if not floor_faces:
        b.pieces = []
        return b.pieces
    ceilings = [f.verts for f in b.faces if f.model == 0 and not f.skybox and f.normal[2] < -0.7
                and not bsp.is_tool(f.texture.lower())]
    ce = _raster(ceilings)
    for k in ce:
        ce[k].sort()
    cols = {}  # (x, y, z floor) -> floor face
    for i in floor_faces:
        for (x, y), zs in _raster([b.faces[i].verts]).items():
            for z in zs:
                cols.setdefault((x, y, z), i)
    pts, owner, runs = [], [], []
    for (x, y, zf), i in cols.items():
        cz = ce.get((x, y), [])
        j = bisect.bisect_right(cz, zf)
        head = min(MAX_HEAD, max(MIN_HEAD, cz[j] - zf)) if j < len(cz) else SKY_HEAD
        pts += [(x, y, z) for z in range(zf, zf + head)]
        owner += [i] * head
        runs.append((x, y, zf, head))
    pts, first = np.unique(np.array(pts), axis=0, return_index=True)  # overlapping columns share cells
    owner = np.array(owner)[first]
    lo = pts.min(0)
    def label_at(p, reach=3, grid_=None):
        g = lab if grid_ is None else grid_
        c = np.floor(p / CELL).astype(int) - lo
        sl = tuple(slice(max(0, v - reach), max(0, v + reach + 1)) for v in c)
        sub = g[sl]
        return set(np.unique(sub[sub > 0]).tolist()) if sub.size else set()

    marks = [p for e in b.entities if e.get("classname") in ROUTE and (p := _origin(b, e)) is not None]
    lab, n = _label(pts, runs, lo)
    route = set().union(*[label_at(p) for p in marks]) if marks else set()
    seen_cell, seen_leaf = _seen(b, pts, np.isin(lab[tuple((pts - lo).T)], list(route)), marks)
    # Xen floor and facility floor never join (a displacer pocket can share a sky, and so a floor height, with
    # the route); black floors (voids, holding boxes) go with the nearest floor that isn't black
    ftex = np.array([f.texture.lower() for f in b.faces], object)
    xen_cell = np.array([bool(plan.XEN.search(t)) for t in ftex[owner]])
    black = ftex[owner] == "black"
    if black.any() and (~black).any():
        from scipy.spatial import cKDTree
        xen_cell[black] = xen_cell[~black][cKDTree(pts[~black]).query(pts[black])[1]]
    status = xen_cell.astype(int) + (0 if seen_cell is None else 2 * seen_cell)
    lab, n = _label(pts, runs, lo, status)
    route = set().union(*[label_at(p) for p in marks]) if marks else set()
    if seen_cell is not None:  # the route is what it can see
        route = {k for k in route if seen_cell[lab[tuple((pts - lo).T)] == k].any()}
    grid = lab > 0
    vlab = lab[tuple((pts - lo).T)]
    tele = set()
    holders = set()  # pieces holding a start, a teleport or a level change
    pocket, change = set(), set()  # displacer pockets (they hold a start point too); pieces with a level change
    for e in b.entities:
        if e.get("classname") == "info_teleport_destination" and (p := _origin(b, e)) is not None:
            tele |= label_at(p, 2)
        if e.get("classname") in HOLD and (p := _origin(b, e)) is not None:
            holders |= label_at(p, 1)
        if e.get("classname") in XEN_MARKS and (p := _origin(b, e)) is not None:
            pocket |= label_at(p, 1)
        if e.get("classname") in CHANGE and (p := _origin(b, e)) is not None:
            change |= label_at(p)

    # faces: floors belong to the piece their columns are mostly in, other faces to the piece in front
    b.face_piece = np.zeros(len(b.faces), np.int32)  # piece of each face, 0: none
    for i in floor_faces:
        ks = vlab[owner == i]
        if len(ks):
            b.face_piece[i] = np.bincount(ks).argmax()
    from scipy import ndimage
    near = ndimage.distance_transform_edt(~grid, return_distances=False, return_indices=True)
    loose = []  # faces with no cell in front (tall walls, high ceilings, the undersides of islands)
    for i, f in enumerate(b.faces):
        if f.model != 0 or b.face_piece[i]:
            continue
        c = np.floor((f.verts.mean(0) + f.normal * CELL / 2) / CELL).astype(int) - lo
        if not ((c >= 0).all() and (c < grid.shape).all() and lab[tuple(c)]):
            loose.append(i)
            c = np.clip(c, 0, np.array(grid.shape) - 1)
            c = near[:, c[0], c[1], c[2]]  # the nearest piece
        b.face_piece[i] = lab[tuple(c)]
    # a loose face the route can see stays even when its nearest piece is one the route can't see
    seen_face = set()
    if seen_leaf is not None and loose:
        front = np.array([b.faces[i].verts.mean(0) + b.faces[i].normal * 4 for i in loose])
        seen_face = {i for i, l in zip(loose, b.vis.leaf_of(front)) if seen_leaf[l]}
    area = np.array([0.5 * np.linalg.norm(np.cross(f.verts[1:-1] - f.verts[0], f.verts[2:] - f.verts[0]).sum(0))
                     for f in b.faces])
    xen = np.array([bool(plan.XEN.search(t)) for t in ftex])
    void = ftex == "black"
    plain = np.array([f.model == 0 and not f.skybox and not bsp.is_tool(t) and t not in ("sky", "black")
                      and not t.startswith("!") for f, t in zip(b.faces, ftex)])  # walls that say what a place is
    m3 = (CELL * 0.0254) ** 3
    pieces, drop = [], np.zeros(n + 1, bool)
    for k in range(1, n + 1):
        sel = vlab == k
        faces_k = np.unique(owner[sel])
        solid = faces_k[~void[faces_k]]
        share = float(area[solid][xen[solid]].sum() / max(area[solid].sum(), 1e-9))
        size = float(sel.sum() * m3)
        faces_all = np.flatnonzero(b.face_piece == k)
        black = float(area[faces_all][void[faces_all]].sum() / max(area[faces_all].sum(), 1e-9))
        shown = faces_all[plain[faces_all]]
        share_all = float(area[shown][xen[shown]].sum() / max(area[shown].sum(), 1e-9))
        if k in holders and black >= HOLD_BLACK:
            kind = "holding"
        elif k in pocket and k not in change:
            kind = "xen"
        elif k in route:
            kind = "route"
        elif share >= XEN_SHARE and share_all >= XEN_ALL:
            kind = "xen"
        elif k in tele and size >= PLACE_M3:
            kind = "teleport"
        elif seen_cell is None:
            kind = "unknown"
        elif seen_cell[sel].any():
            kind = "seen"
        else:
            kind = "unseen"
        drop[k] = kind in ("xen", "unseen", "holding")
        c = pts[sel]
        at = c[np.argmin(np.abs(c - np.median(c, 0)).sum(1))]  # a cell of the piece near its middle
        tex = {}
        for i in faces_all:
            tex[b.faces[i].texture] = tex.get(b.faces[i].texture, 0) + area[i] * 0.0254 ** 2
        pieces.append({"id": k, "kind": kind, "dropped": bool(drop[k]), "m3": round(size),
                       "textures": [[t, round(a)] for t, a in sorted(tex.items(), key=lambda kv: -kv[1])[:5]],
                       "xen_share": round(share, 2), "xen_share_all": round(share_all, 2), "lo": (c.min(0) * CELL).tolist(),
                       "hi": ((c.max(0) + 1) * CELL).tolist(), "at": ((at + 0.5) * CELL).tolist(),
                       "floors": len(faces_k)})
    for p in pieces:
        p["found"] = {"dropped": p["dropped"], "kind": p["kind"]}
    if areas is not None:
        drop[:] = False
        for p in pieces:
            p.update(dropped=False)
        for r in areas:
            for k in label_at(np.array(r["at"], float), 0):
                drop[k] = True
                pieces[k - 1].update(dropped=True, kind=r["type"])
    for i, f in enumerate(b.faces):
        if b.face_piece[i] and drop[b.face_piece[i]]:
            if f.normal[2] > plan.FLOOR_NZ and not drop[vlab[owner == i]].all():
                continue  # a floor reaching into a kept piece stays
            if i in seen_face and pieces[b.face_piece[i] - 1]["kind"] == "unseen":
                continue
            f.outside = True
    b.pieces, b.piece_grid = pieces, (lab, lo)
    return pieces


def piece_at(b, p):
    """The piece at a point (analyse first), or the most common one in the cells around it; 0: none."""
    lab, lo = b.piece_grid
    c = np.floor(np.asarray(p, float) / CELL).astype(int) - lo
    sub = lab[tuple(slice(max(0, v - 1), max(0, v + 2)) for v in c)]
    if (c >= 0).all() and (c < lab.shape).all() and lab[tuple(c)]:
        return int(lab[tuple(c)])
    sub = sub[sub > 0]
    return int(np.bincount(sub).argmax()) if sub.size else 0
