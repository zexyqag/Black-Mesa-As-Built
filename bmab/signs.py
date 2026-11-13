"""Signs in a map: where each sign the register knows (bmab.register: evidence of kind sign or wall map, and the facts citing it) is
drawn, which way it faces, and which way its arrows point there. The register says once per texture what a sign says and where its
arrows point on the picture (facts "points": the angle in degrees on the picture, 0 to its right, 90 up); where the sign is and so
which way that is in the map is worked out here. The register is passed in (catalogue), never read here.

A sign is a patch of faces in its texture (a brush on a wall), a decal (GoldSrc's infodecal, on the wall nearest it) or an
overlay (Source). An arrow pointing up on a wall sign means straight on, past the sign; down, below it (a hatch, a lower level).
A sign on a door is where the door stands shut: it moves away when the door opens, so it is read while the door is shut.
"""
import re
import struct
from collections import defaultdict

import numpy as np

from . import brushes, bsp, connectors, doors, families

MIN_SIDE = 8  # units: a narrower patch is the edge of a sign board


def _name(texture):
    """A texture or material name as the register gives it: Source's copies of a material for a place's reflections go back to it."""
    t = re.sub(r"(_-?\d+){3}$", "", re.sub(r"^maps/[^/]+/", "", texture.lower()))
    head, _, last = t.rpartition("/")
    return (head + "/" if head else "") + families.base(last)


def catalogue(register):
    """{texture name: [sign]} of the register's signs (register: bmab.redlines.register), for find and described: each {id (of its
    evidence), game, transcription, claims: [[place, relation, place or value]], points: [[place, angle]]}."""
    citing = defaultdict(list)
    for f in register["facts"]:
        for e in f["evidence"]:
            citing[e].append(f)
    out = defaultdict(list)
    for e in register["evidence"].values():
        if e["kind"] in ("sign", "wall map"):
            facts = citing[e["id"]]
            out[_name(e["ref"])].append({"id": e["id"], "game": e["source"], "transcription": e.get("transcription", ""),
                                         "claims": [[f["subject"], f["relation"], *([f["object"]] if "object" in f else [f["value"]] if "value" in f else [])]
                                                    for f in facts if f["relation"] != "points"],
                                         "points": [[f["subject"], f["value"]] for f in facts if f["relation"] == "points"]})
    return dict(out)


def _sign_of(name, game, known):
    signs = known.get(_name(name))
    if not signs:
        return None
    return next((f for f in signs if f["game"] == game), signs[0])


def _flat(v, n):
    v = v - (v @ n) * n
    return v / max(np.linalg.norm(v), 1e-9)


def _on_door(b, models, fired):
    """What a sign on these brush models is on: "opens" (a door that opens), "shut" (a door that never does) or None."""
    if getattr(b, "door_models", None) is None:  # kept on the map, not as a Bsp field: bsp.py's text keys the connector caches
        b.door_models = doors.brush_doors(b, fired)
    on = max(set(models), key=models.count)
    return None if on not in b.door_models else "opens" if b.door_models[on] else "shut"


def _faces(b, game, fired, known):
    """Signs drawn as faces: a patch of joined faces in one plane and texture is one sign."""
    groups = defaultdict(list)
    for i, f in enumerate(b.faces):
        if f.skybox or f.axes is None:
            continue
        fact = _sign_of(f.texture, game, known)
        if fact is not None:
            v, n = brushes.place(b, f)
            groups[(fact["id"], tuple(np.round(n, 2)), round(float(v.mean(0) @ n)))].append(i)
    out = []
    for (fid, _n, _d), members in groups.items():
        corners = {i: brushes.place(b, b.faces[i])[0] for i in members}
        boxes = [(corners[i].min(0) - 2, corners[i].max(0) + 2) for i in members]
        parent = list(range(len(members)))

        def top(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k
        for a in range(len(members)):
            for c in range(a):
                if (boxes[a][0] <= boxes[c][1]).all() and (boxes[c][0] <= boxes[a][1]).all():
                    parent[top(a)] = top(c)
        patches = defaultdict(list)
        for k, i in enumerate(members):
            patches[top(k)].append(i)
        for patch in patches.values():
            f = b.faces[patch[0]]
            n = brushes.place(b, f)[1]
            right, up = _flat(brushes.turned(b, f, f.axes[0, :3]), n), _flat(brushes.turned(b, f, -f.axes[1, :3]), n)
            s = {**_sign(fid, np.concatenate([corners[i] for i in patch]), n, right, up), "door": _on_door(b, [b.faces[i].model for i in patch], fired)}
            if min(s["size"]) >= MIN_SIDE and _seen(b, s):
                out.append(s)
    return _both_sides([s for s in out if not _edge(s, out)], known)


def _both_sides(signs, known):
    """A board with the same sign on its front and back (a door's both faces) is one sign seen from both sides, unless it has
    arrows: those point opposite ways for someone on each side."""
    arrows = {f["id"] for fs in known.values() for f in fs if f["points"]}
    out = []
    for s in signs:
        twin = next((o for o in out if o["evidence"] == s["evidence"] and s["evidence"] not in arrows and o["normal"] @ s["normal"] < -0.9
                     and abs((s["at"] - o["at"]) @ o["normal"]) <= 40 and np.linalg.norm((s["at"] - o["at"]) - ((s["at"] - o["at"]) @ o["normal"]) * o["normal"]) <= 16), None)
        if twin is None:
            out.append({**s, "sides": 1})
        else:
            twin["sides"] = 2
    return out


def _edge(s, signs):
    """Whether a patch is the edge of a thicker sign board: square to a much bigger patch of the same sign beside it."""
    area = s["size"][0] * s["size"][1]
    return any(o["evidence"] == s["evidence"] and abs(o["normal"] @ s["normal"]) < 0.3 and o["size"][0] * o["size"][1] > 3 * area
               and np.linalg.norm(o["at"] - s["at"]) <= max(o["size"]) for o in signs)


def _seen(b, s):
    """Whether open space is in front of a sign: a sign board carries its picture on its edges and back too."""
    if b.vis is None:
        return True
    return bool(b.vis.open((s["at"] + s["normal"] * (8 + (bsp.SOLID_CELL + bsp.SOLID_PAD if isinstance(b.vis, bsp.SourceVis) else 0)))[None])[0])


def _sign(fid, verts, n, right, up):
    r, u = verts @ right, verts @ up
    at = verts.mean(0)
    at = at + right * ((r.min() + r.max()) / 2 - at @ right) + up * ((u.min() + u.max()) / 2 - at @ up)
    return {"evidence": fid, "at": at, "normal": n, "right": right, "up": up, "size": [float(np.ptp(r)), float(np.ptp(u))]}


def _decals(b, game, fired, known):
    """GoldSrc's infodecals: on the wall nearest each, its picture laid as that wall's texture is."""
    out, surfaces = [], None
    for e in b.entities:
        if e.get("classname") != "infodecal" or "origin" not in e:
            continue
        fact = _sign_of(e.get("texture", ""), game, known)
        if fact is None:
            continue
        surfaces = surfaces or connectors.planes(b)
        p = np.array([float(x) for x in e["origin"].split()])
        hits = [(h, d) for d in np.vstack([np.eye(3), -np.eye(3)]) if (h := connectors._first_face(surfaces, p, d, 24.0)) is not None]
        if not hits:
            continue
        (t, k), d = min(hits, key=lambda h: h[0][0])
        f, n = surfaces[4][k], surfaces[0][k]
        right, up = ((_flat(brushes.turned(b, f, f.axes[0, :3]), n), _flat(brushes.turned(b, f, -f.axes[1, :3]), n)) if f.axes is not None
                     else (_flat(np.cross([0, 0, 1.0], n), n), np.array([0, 0, 1.0])))
        out.append({"evidence": fact["id"], "at": p + d * t, "normal": n, "right": right, "up": up, "size": [64.0, 64.0], "door": _on_door(b, [f.model], fired)})
    return out


def _overlays(raw, game, known):
    """Source's overlays: each its own picture on the wall, its basis U kept in the z of its corner points."""
    if raw[:4] != b"VBSP":
        return []

    def lump(key, i):
        off, ln, _ver, size = struct.unpack_from("<iiii", raw, 8 + i * 16)
        return bsp._lzma(raw[off:off + ln], size, "", key) if size else raw[off:off + ln]
    texinfo = np.frombuffer(lump("texinfo", bsp.VBSP_LUMPS["texinfo"]), dtype=np.dtype([("v", "<f4", 16), ("flags", "<i4"), ("texdata", "<i4")]))
    texdata = np.frombuffer(lump("texdata", bsp.VBSP_LUMPS["texdata"]), dtype=np.dtype([("refl", "<f4", 3), ("name", "<i4"), ("size", "<i4", 4)]))
    table = np.frombuffer(lump("texdata_table", bsp.VBSP_LUMPS["texdata_table"]), dtype="<i4")
    strings = lump("texdata_strings", bsp.VBSP_LUMPS["texdata_strings"])
    data = lump("overlays", bsp.VBSP_LUMPS["overlays"])
    out = []
    for o in np.frombuffer(data[:len(data) // bsp.OVERLAY.itemsize * bsp.OVERLAY.itemsize], dtype=bsp.OVERLAY):
        if o["texinfo"] < 0:
            continue
        s = table[texdata[texinfo[o["texinfo"]]["texdata"]]["name"]]
        fact = _sign_of(strings[s:strings.index(b"\0", s)].decode("latin-1"), game, known)
        if fact is None:
            continue
        n = o["normal"].astype(float) / max(np.linalg.norm(o["normal"]), 1e-9)
        u = o["uvpoints"][:3, 2].astype(float)
        u = _flat(u, n)
        v = np.cross(n, u)
        pts = o["origin"] + o["uvpoints"][:, :1] * u + o["uvpoints"][:, 1:2] * v
        right = u * (1 if o["u"][1] >= o["u"][0] else -1)
        up = -v * (1 if o["v"][1] >= o["v"][0] else -1)
        out.append({**_sign(fact["id"], pts.astype(float), n, right, up), "door": None})
    return out


def find(b, game, known, raw=None, fired=()):
    """The map's signs: [{evidence, at, normal, right, up (unit vectors across and up the picture), size [across, up] in units, door
    (on a door: "opens" or "shut"; None)}]. game: the source the map is from (its own signs are preferred where a texture name is
    in several games); known: catalogue; raw: the map file's bytes, for Source's overlays; fired: names fired from other maps
    (doors.fired_names)."""
    return _faces(b, game, fired, known) + _decals(b, game, fired, known) + (_overlays(raw, game, known) if raw is not None else [])


def ways(sign, fact):
    """[{to: place, dir: unit vector in the map}] of a sign's arrows, from the register's angles on the picture."""
    out = []
    for place, angle in fact.get("points", []):
        a = np.radians(angle)
        across, along = np.cos(a), np.sin(a)
        if abs(sign["normal"][2]) < 0.7:  # on a wall: up the picture is straight on, past it, and down is below it
            d = across * sign["right"] + (-along * sign["normal"] if along > 0 else along * np.array([0.0, 0.0, 1.0]))
        else:
            d = across * sign["right"] + along * sign["up"]
        out.append({"to": place, "dir": d / max(np.linalg.norm(d), 1e-9)})
    return out


def described(b, game, known, raw=None, fired=()):
    """The map's signs with what they say (transcription), their facts' claims and their arrows' ways in the map: [{evidence, at,
    normal, right, up, size, door, transcription, claims, points [{to, dir}]}] (numpy vectors). known: catalogue."""
    by_id = {f["id"]: f for fs in known.values() for f in fs}
    return [{**s, "transcription": by_id[s["evidence"]]["transcription"], "claims": by_id[s["evidence"]]["claims"], "points": ways(s, by_id[s["evidence"]])}
            for s in find(b, game, known, raw, fired)]
