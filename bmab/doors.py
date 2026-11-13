"""Fake doors: doors the player can't go through, and what lies behind them (it needs the BSP tree).

- locked: a func_door or func_door_rotating in door textures that never opens. GoldSrc: it has a name, nothing in
  the game fires that name, and it neither opens on use nor starts open. Source: it doesn't start open, the
  player can't open it (neither use nor touch opens it, or it starts locked), and no output sends it Open,
  Toggle, Unlock or SetPosition. Source's model doors (prop_door_rotating) count when they start locked or ignore
  the player's use and nothing opens or unlocks them; their size is taken as PROP_DOOR (the model isn't read), and
  of the two sides of the hinge, the one in open space.
- painted: door textures on a wall (world, func_wall, func_illusionary) that don't touch a moving or locked door
  (its frame, or the half of a tall door that doesn't move)
Behind each, the map's own BSP is searched straight back from the door for open space: within THROUGH units it's
the next room (the door would lead somewhere the map already has), further but within REACH more of the map,
none: nothing, the map ends there and the door leads into the blank. Doors in dropped pieces (bmab.pieces) are
left out: they aren't the facility.
"""
import re

import numpy as np

from . import bsp, pieces

PATTERN = re.compile(r"door|_dr|dr\d")  # Half-Life's are mostly *_dr*, the add-ons' *door*
NOT_DOORS = ("indoor", "drain", "drk", "dr_cntrl", "dooredge")  # tram insides, drains, dark metal, keypads, trim
MOVING = ("func_door", "func_door_rotating")
STATIC = ("worldspawn", "func_wall", "func_illusionary")
NOT_TARGETS = {"targetname", "classname", "origin", "model", "angles", "angle"}
USE_ONLY, STARTS_OPEN = 256, 1  # GoldSrc spawnflags
SRC_STARTS_OPEN, SRC_USE_OPENS, SRC_TOUCH_OPENS, SRC_LOCKED = 1, 256, 1024, 2048  # Source spawnflags
OPENERS = {"open", "toggle", "unlock", "setposition"}  # Source inputs that open a door or let the player
PROP_IGNORE_USE = 32768
PROP_DOOR = (54, 108)  # width and height of HL2's door01 models, the commonest in Black Mesa
THROUGH, REACH = 64, 512
MIN_HEIGHT, MIN_WIDTH, MIN_FRONT = 48, 24, 32  # a player is 32 units wide


def is_door_texture(t):
    return bool(PATTERN.search(t)) and not any(x in t for x in NOT_DOORS)


def fired_names(entity_lists):
    """Names a level change fires in the next map (trigger_changelevel's changetarget), over the entities of a game's maps."""
    return {e["changetarget"] for ents in entity_lists for e in ents if e.get("changetarget")}


def _open(b, pts):
    return b.vis.open(pts)


def _first_open(b, p, d, start=0.0):
    """Distance from p along d to the first open leaf, from start + 8 up to REACH; None: solid all the way."""
    ts = np.arange(start + 8, start + REACH, 8)
    hit = np.flatnonzero(_open(b, p[None] + ts[:, None] * d[None]))
    return None if not len(hit) else float(ts[hit[0]])


def _free(b, p, d, most=400):
    """Open distance from p along d (where a camera can stand in front of the door)."""
    ts = np.arange(8, most + 8, 8)
    shut = np.flatnonzero(~_open(b, p[None] + ts[:, None] * d[None]))
    return float(ts[shut[0]] - 8) if len(shut) else float(most)


def _area(f):
    v = f.verts
    return float(np.linalg.norm(sum(np.cross(v[k] - v[0], v[k + 1] - v[0]) for k in range(1, len(v) - 1)))) / 2


def _groups(fs):
    """Faces in one plane whose boxes touch: one door (the BSP splits a door into several faces)."""
    parent = list(range(len(fs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    lo = [f.verts.min(0) - 4 for f in fs]
    hi = [f.verts.max(0) + 4 for f in fs]
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            a, c = fs[i], fs[j]
            if a.normal @ c.normal > 0.99 and abs(a.normal @ (a.verts[0] - c.verts[0])) < 2 \
                    and (lo[i] <= hi[j]).all() and (lo[j] <= hi[i]).all():
                parent[root(i)] = root(j)
    out = {}
    for i in range(len(fs)):
        out.setdefault(root(i), []).append(i)
    return list(out.values())


def _overlap(a, b):
    """Two painted doors in parallel planes up to THROUGH apart that cover the same spot of the wall."""
    if abs(a["back"] @ b["back"]) < 0.99:
        return False
    d = b["c"] - a["c"]
    depth = d @ a["back"]
    side = np.array([-a["back"][1], a["back"][0], 0.0])
    return abs(depth) <= THROUGH and abs(d @ side) < min(a["width"], b["width"]) / 2 \
        and abs(d[2]) < min(a["height"], b["height"]) / 2


def _openers(b, fired=()):
    """(names something in the map fires, names an output opens) of a map: what may open a named door."""
    named, opened = set(fired), set()
    for e in b.entities:
        named.update(v for k, v in bsp.items(e) if k not in NOT_TARGETS)
        if e.get("classname") == "multi_manager":
            named.update(e)  # its keys are the names it fires
        opened.update(t.lower() for _o, t, i in bsp.outputs(e) if i.lower() in OPENERS)
    return named, opened


def _opens(e, source, named, opened):
    """Whether a func_door or func_door_rotating ever opens: it starts open, the player opens it, or something in the game does."""
    flags, name = int(e.get("spawnflags", 0) or 0), e.get("targetname")
    if source:
        player = flags & (SRC_USE_OPENS | SRC_TOUCH_OPENS) and not flags & SRC_LOCKED
        return bool(flags & SRC_STARTS_OPEN or e.get("spawnpos") == "1" or player or (name or "").lower() in opened)
    return bool(not name or flags & (USE_ONLY | STARTS_OPEN) or name in named)


def _prop_opens(e, opened):
    """Whether a Source model door (prop_door_rotating) ever opens: it starts open, the player can use it, or something opens or
    unlocks it."""
    flags, name = int(e.get("spawnflags", 0) or 0), e.get("targetname")
    return bool(flags & SRC_STARTS_OPEN or not flags & (SRC_LOCKED | PROP_IGNORE_USE) or (name or "").lower() in opened)


def _leaf(b, e):
    """Where a model door's leaf stands shut: (a point in its middle, its normal, the way across the doorway along it); None if
    neither side of the hinge has room for it. Of the hinge's two sides, the one with open space on both faces (a doorway), else
    any open one."""
    o = np.array([float(x) for x in e["origin"].split()])
    yaw = np.radians(float((e.get("angles") or "0 0 0").split()[1]))
    n, w = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    cands = [o + sgn * w * PROP_DOOR[0] / 2 + [0, 0, dz] for sgn in (1, -1) for dz in (0, PROP_DOOR[1] / 2)]
    c = next((c for c in cands if _open(b, np.array([c + n * 16, c - n * 16])).all()), None)
    if c is None:
        c = next((c for c in cands if _open(b, c)[0]), None)
    return None if c is None else (c, n, w)


def model_doors(b):
    """The map's model doors where they stand shut: [(the middle of the leaf, the way across the doorway, whether it opens)]. Only
    Source has them."""
    if not isinstance(b.vis, bsp.SourceVis):
        return []
    opened, out = _openers(b)[1], []
    for e in b.entities:
        if e.get("classname") == "prop_door_rotating" and "origin" in e and (leaf := _leaf(b, e)) is not None:
            out.append((leaf[0], leaf[2], _prop_opens(e, opened)))
    return out


def brush_doors(b, fired=()):
    """{model index: whether it opens} of the map's moving brush doors, lifts and platforms aside (their thinnest side is up).
    fired: names fired from other maps (fired_names)."""
    if not any(e.get("classname") in MOVING for e in b.entities):
        return {}
    named, opened = _openers(b, fired)
    source, out = isinstance(b.vis, bsp.SourceVis), {}
    for e in b.entities:
        m = e.get("model", "")
        if e.get("classname") not in MOVING or not m.startswith("*") or not m[1:].isdigit():
            continue
        lo, hi = b.models[int(m[1:])][:2]
        if np.argmin(hi - lo) != 2:
            out[int(m[1:])] = _opens(e, source, named, opened)
    return out


def find(b, fired=()):
    """The map's doors the player can't use: [{id, kind, at, back, width, height, behind, front, piece, textures,
    name, sides}]. fired: names fired from other maps (fired_names). behind: units to open space behind the door
    (0: open on both sides of a locked door; None: none within REACH); sides 2: seen from both sides."""
    if b.vis is None:
        return []
    ps = pieces.analyse(b)
    if not ps:
        return []
    named, opened = _openers(b, fired)
    source = isinstance(b.vis, bsp.SourceVis)
    owner = {0: "worldspawn"}
    for e in b.entities:
        if e.get("model", "").startswith("*"):
            owner[int(e["model"][1:])] = e.get("classname")
    by_model = {}
    for i, f in enumerate(b.faces):
        by_model.setdefault(f.model, []).append(i)

    def piece_in_front(c, front):
        k = pieces.piece_at(b, c + front * pieces.CELL / 2)
        return k, (k == 0 or ps[k - 1]["dropped"])

    out, frames = [], []
    for e in b.entities:
        if e.get("classname") not in MOVING or not e.get("model", "").startswith("*"):
            continue
        mi = int(e["model"][1:])
        origin = np.array([float(x) for x in e.get("origin", "0 0 0").split()])
        lo, hi = b.models[mi][0] + origin, b.models[mi][1] + origin
        frames.append((lo - 8, hi + 8))
        faces = by_model.get(mi, [])
        tex = sorted({b.faces[i].texture for i in faces if is_door_texture(b.faces[i].texture)})
        if not tex or _opens(e, source, named, opened):
            continue
        name = e.get("targetname")
        c, size = (lo + hi) / 2, hi - lo
        ax = int(np.argmin(size[:2]))
        d = np.zeros(3)
        d[ax] = 1
        sides = _open(b, np.array([c + d * (size[ax] / 2 + 12), c - d * (size[ax] / 2 + 12)]))
        if sides.all():
            back, behind = (d if _free(b, c, d) < _free(b, c, -d) else -d), 0.0
        else:
            back = -d if sides[0] else d
            behind = _first_open(b, c + back * size[ax] / 2, back)
        k, dropped = piece_in_front(c - back * size[ax] / 2, -back)
        if dropped:
            continue
        out.append({"kind": "locked", "c": c, "back": back, "width": float(size[1 - ax]), "height": float(size[2]),
                    "behind": behind, "front": _free(b, c - back * (size[ax] / 2 + 4), -back), "piece": k,
                    "textures": tex, "name": name, "sides": 2 if behind == 0 else 1})

    for e in b.entities if source else ():
        if e.get("classname") != "prop_door_rotating" or "origin" not in e:
            continue
        if _prop_opens(e, opened):
            continue
        name, leaf = e.get("targetname"), _leaf(b, e)
        if leaf is None:
            continue
        c, n, _w = leaf
        sides = _open(b, np.array([c + n * 12, c - n * 12]))
        back = n if not sides[0] else -n if not sides[1] else (n if _free(b, c, n) < _free(b, c, -n) else -n)
        behind = 0.0 if sides.all() else _first_open(b, c, back)
        k, dropped = piece_in_front(c, -back)
        if dropped:
            continue
        out.append({"kind": "locked", "c": c, "back": back, "width": float(PROP_DOOR[0]), "height": float(PROP_DOOR[1]),
                    "behind": behind, "front": _free(b, c - back * 4, -back), "piece": k,
                    "textures": [e.get("model", "")], "name": name, "sides": 2 if behind == 0 else 1})

    painted = [i for i, f in enumerate(b.faces) if owner.get(f.model) in STATIC and is_door_texture(f.texture)]
    flat = []
    for g in _groups([b.faces[i] for i in painted]):
        faces = [painted[j] for j in g]
        f = b.faces[max(faces, key=lambda i: _area(b.faces[i]))]
        n = f.normal / np.linalg.norm(f.normal)
        if abs(n[2]) > 0.5:
            continue  # floors and ceilings
        v = np.concatenate([b.faces[i].verts for i in faces])
        lo, hi = v.min(0), v.max(0)
        c = (lo + hi) / 2
        side = np.array([-n[1], n[0], 0.0]) / max(1e-9, np.hypot(n[0], n[1]))
        width, height = float(np.ptp(v @ side)), float(hi[2] - lo[2])
        if height < MIN_HEIGHT or width < MIN_WIDTH or not _open(b, c + n * 8)[0]:
            continue
        if any((lo <= z).all() and (hi >= a).all() for a, z in frames):
            continue  # part of a moving or locked door, or its frame
        k, dropped = piece_in_front(c, n)
        if dropped:
            continue
        start = 0.0
        if f.model:  # brush entities aren't in the world's tree: search from the far side of the entity
            mlo, mhi = b.models[f.model][0] - 1, b.models[f.model][1] + 1
            start = next((t for t in range(0, 132, 4) if not ((c - n * t >= mlo).all() and (c - n * t <= mhi).all())), 128)
        front = _free(b, c, n)
        if front < MIN_FRONT:
            continue
        flat.append({"kind": "painted", "c": c, "back": -n, "width": width, "height": height, "front": front,
                     "behind": _first_open(b, c, -n, float(start)), "piece": k, "name": None, "sides": 1,
                     "textures": sorted({b.faces[i].texture for i in faces})})
    # one door in several planes: a frame and the panel set back in it (kept: the panel), or both sides of a wall
    flat.sort(key=lambda x: -(x["c"] @ x["back"]))
    for x in flat:
        same = next((y for y in out if y["kind"] == "painted" and _overlap(x, y)), None)
        if same is None:
            out.append(x)
        elif same["back"] @ x["back"] < 0:
            same["sides"] = 2
    out.sort(key=lambda x: (x["behind"] is not None, x["kind"], tuple(np.round(x["c"]))))
    for k, x in enumerate(out, 1):
        x["id"] = k
        x["at"] = [round(float(t)) for t in x.pop("c")]
        x["back"] = [round(float(t), 3) for t in x["back"]]
        for key in ("width", "height", "front"):
            x[key] = round(x[key])
    return out


def measure(b, at, back):
    """A door placed by hand, as the search would describe it: units to open space behind it, or None."""
    c, back = np.array(at, float), np.array(back, float)
    for e in b.entities:
        if e.get("classname") in MOVING + STATIC[1:] and e.get("model", "").startswith("*"):
            origin = np.array([float(x) for x in e.get("origin", "0 0 0").split()])
            lo, hi = [b.models[int(e["model"][1:])][k] + origin + slack for k, slack in ((0, -2), (1, 2))]
            if (c >= lo).all() and (c <= hi).all():  # brush entities aren't in the world's tree: search from the far side of it
                start = next((t for t in range(0, 132, 4) if not ((c + back * t >= lo).all() and (c + back * t <= hi).all())), 128)
                return _first_open(b, c, back, float(start))
    return _first_open(b, c, back)


def status(door):
    """nothing behind / the next room / more of the map, as page and summary labels use it."""
    if door["behind"] is None:
        return "nothing"
    return "next" if door["behind"] <= THROUGH else "further"
