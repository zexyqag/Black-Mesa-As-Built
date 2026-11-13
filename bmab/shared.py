"""Rooms built into two games' maps: identical faces (same texture, same shape) at one translation.

Every face of one map set is keyed by texture and shape (vertices relative to the lowest one); each pair of
equal keys across the two sets votes, by face area, for the translation between its two maps. A peak is
geometry the second map shares with the first at that translation.

Classes, so a copied backdrop or a tiled panel isn't read as the same place:
- scenery: little besides one texture (canyon rock, swamp walls, black filler): under min_m2 in the others
- pattern: the same faces match again at another translation (a peak for the same pair of maps, at least a
  third the size, made of mostly the same textures): repeated panels, not one placement
- room: anything else

What bmab shared found is kept per pair of games (out/<game>/shared-<other>.json, stored reads it); places turns its rooms into
same place claims for a map.
"""
import json
from collections import defaultdict

import numpy as np

from . import bsp, sources
from .graph import INCH

PATTERN_RATIO = 1 / 3
PATTERN_OVERLAP = 0.5


def _faces(b):
    for f in b.faces:
        if f.model != 0 or bsp.is_tool(f.texture):
            continue
        v = np.round(f.verts).astype(np.int64)
        v = v[np.lexsort(v.T[::-1])]
        a = float(0.5 * np.linalg.norm(np.cross(f.verts[1:-1] - f.verts[0], f.verts[2:] - f.verts[0]).sum(0))) * INCH ** 2
        yield f.texture.lower(), tuple(map(tuple, v - v[0])), v[0], a


def signatures(maps, min_face_m2=1.0):
    """{(texture, shape): [(map, anchor vertex, m2)]} and each map's total face area."""
    sig, total = defaultdict(list), defaultdict(float)
    for n, b in maps.items():
        for tex, shape, v0, a in _faces(b):
            total[n] += a
            if a >= min_face_m2:
                sig[(tex, shape)].append((n, v0, a))
    return sig, total


def find(maps_a, maps_b, min_face_m2=1.0, max_pairs=2000, min_m2=100, top=3, log=print):
    """Peaks of maps_b against maps_a: per map of b, up to `top` (map of a, translation) with at least min_m2
    matched, each with its textures, extent and class. translation = b coordinates - a coordinates."""
    sa, _ = signatures(maps_a, min_face_m2)
    sb, total_b = signatures(maps_b, min_face_m2)
    log(f"face keys: {len(sa):,} and {len(sb):,}")
    votes = defaultdict(float)
    for key, occ_b in sb.items():
        occ_a = sa.get(key)
        if not occ_a or len(occ_a) * len(occ_b) > max_pairs:
            continue  # very common faces (tiles, trims) vote everywhere and say nothing
        for na, va, a in occ_a:
            for nb, vb, _ in occ_b:
                if na != nb:
                    votes[(nb, na, tuple(int(x) for x in vb - va))] += a
    by_pair = defaultdict(list)
    for (nb, na, t), m2 in votes.items():
        by_pair[(nb, na)].append((m2, t))
    peaks = defaultdict(list)
    for (nb, na), ts in by_pair.items():
        ts.sort(reverse=True)
        found = [_describe(nb, na, t, m2, total_b[nb], maps_a[na], maps_b[nb]) for m2, t in ts[:2] if m2 >= min_m2]
        for k, p in enumerate(found):
            other = found[1 - k] if len(found) == 2 else None
            if other and other["matched_m2"] >= PATTERN_RATIO * p["matched_m2"] and _overlap(p, other) >= PATTERN_OVERLAP:
                p["class"] = "pattern"
        peaks[nb] += found
    return [p for nb in sorted(peaks) for p in sorted(peaks[nb], key=lambda p: -p["matched_m2"])[:top]]


def stored(game, other):
    """The peaks bmab shared found for game's maps against other's (find; out/<game>/shared-<other>.json), or none."""
    f = sources.REPO / "out" / game / f"shared-{other}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def places(game, name, beside=()):
    """The same places a map is built with, from the rooms bmab shared found against any game, whichever way round it looked: same
    place claims without ids (at: the room's middle in this map, there: in the other), the biggest room for each other map. The
    levels of the same game in beside (next to it, sharing their overlap by design) and the map itself are left out."""
    best = {}
    for f in sorted((sources.REPO / "out").glob("*/shared-*.json")):
        mine, theirs = f.parent.name, f.stem.removeprefix("shared-")
        for p in stored(mine, theirs):
            centre, t = np.array(p["centre"]), np.array(p["translation"])  # centre in the "with" map; translation from it to "map"
            if p["class"] != "room":
                continue
            if mine == game and p["map"] == name:
                other, at, there = (theirs, p["with"]), centre + t, centre
            elif theirs == game and p["with"] == name:
                other, at, there = (mine, p["map"]), centre, centre + t
            else:
                continue
            if other == (game, name) or (other[0] == game and other[1] in beside):
                continue
            if other not in best or p["matched_m2"] > best[other][0]:
                best[other] = (p["matched_m2"], {"kind": "same_place", "at": [int(v) for v in at], "game": other[0], "map": other[1], "there": [int(v) for v in there]})
    return [line for _m2, line in best.values()]


def _overlap(p, q):
    """Share of texture area the two peaks have in common (0 to 1)."""
    a, b = p["_tex"], q["_tex"]
    sa, sb = sum(a.values()) or 1, sum(b.values()) or 1
    return sum(min(a[k] / sa, b.get(k, 0) / sb) for k in a)


def _describe(nb, na, t, m2, total, a, b, min_m2=100):
    t = np.array(t)
    fb = set()
    for tex, shape, v0, _ in _faces(b):
        fb.add((tex, shape, tuple(v0)))
    tex, pts = defaultdict(float), []
    for tx, shape, v0, ar in _faces(a):
        if (tx, shape, tuple(v0 + t)) in fb:
            tex[tx] += ar
            pts.append(v0)
    pts = np.array(pts) if pts else np.zeros((1, 3))
    rest = sum(tex.values()) - max(tex.values(), default=0)
    return {"map": nb, "with": na, "translation": [int(x) for x in t], "matched_m2": round(m2),
            "share_of_map": round(m2 / total, 3), "class": "scenery" if rest < min_m2 else "room",
            "textures": {k: round(v) for k, v in sorted(tex.items(), key=lambda kv: -kv[1])[:8]},
            "extent_m": [round(float(x)) for x in (pts.max(0) - pts.min(0)) * INCH],
            "centre": [round(float(x)) for x in pts.mean(0)], "_tex": dict(tex)}
