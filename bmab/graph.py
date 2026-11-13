"""Connection graph from trigger_changelevel and info_landmark, with every map placed in one frame.

Each link is classified by what surrounds its landmark: a teleport (no physical connection, so it
places maps only when nothing else connects them), a lift ride, a ladder, a track ride (the landmark
lies within TRACK_RADIUS of a path_track line: the passage is a rail line, which can run on unseen), or a
walk-through. Maps
are placed from the start map by the most trustworthy links first; where another physical link
between placed maps disagrees, the disagreement is reported as a gap on that weaker link.
"""
import heapq
from collections import defaultdict

import numpy as np

from .claims import JOINS

INCH = 0.0254  # one Hammer unit is treated as one inch

# How much a link is trusted to fix placement.
RANK = {"walk": 3, "ladder": 2, "ride": 1, "lift": 1, "teleport": 0}
TELEPORT_RADIUS = 64
NEAR_RADIUS = 320
TRACK_RADIUS = 300


def _origin(ent):
    return np.array([float(x) for x in ent["origin"].split()])


def _landmarks(b):
    return {e["targetname"]: _origin(e) for e in b.entities
            if e.get("classname") == "info_landmark" and "targetname" in e and "origin" in e}


def _positions(b):
    """(classname, position) of every entity with an origin or a brush model."""
    out = []
    for e in b.entities:
        if "origin" in e:
            out.append((e.get("classname", ""), _origin(e)))
        elif e.get("model", "").startswith("*"):
            lo, hi, *_ = b.models[int(e["model"][1:])]
            out.append((e.get("classname", ""), (lo + hi) / 2))
    return out


def _near(positions, p, classes, radius):
    return any(c in classes and np.linalg.norm(q - p) < radius for c, q in positions)


def _track_distance(b, p):
    nodes = {e["targetname"]: _origin(e) for e in b.entities
             if e.get("classname") == "path_track" and "targetname" in e and "origin" in e}
    best = np.inf
    for e in b.entities:
        if e.get("classname") == "path_track" and e.get("targetname") in nodes and e.get("target") in nodes:
            a, c = nodes[e["targetname"]], nodes[e["target"]]
            ac = c - a
            t = np.clip(np.dot(p - a, ac) / max(np.dot(ac, ac), 1e-9), 0, 1)
            best = min(best, float(np.linalg.norm(p - (a + t * ac))))
    return best


def classify(b, positions, mark_pos, fired):
    if _near(positions, mark_pos, {"info_teleport_destination"}, TELEPORT_RADIUS):
        return "teleport"
    if fired and _near(positions, mark_pos, {"func_train", "func_tracktrain"}, NEAR_RADIUS):
        return "lift"
    if _near(positions, mark_pos, {"func_ladder"}, NEAR_RADIUS):
        return "ladder"
    if _track_distance(b, mark_pos) <= TRACK_RADIUS:
        return "ride"
    return "walk"


def links(maps):
    """(from, to, landmark, fired) for every trigger_changelevel."""
    out = []
    for name, b in maps.items():
        for e in b.entities:
            if e.get("classname") == "trigger_changelevel" and e.get("map"):
                out.append((name, e["map"].lower(), e.get("landmark", ""), "targetname" in e))
    return sorted(set(out))


def key(d):
    """(map, map, landmark) of a link or a decided one (bmab.redlines.decided_links), map names sorted."""
    return (*sorted(d["maps"]), d["landmark"])


def build(maps, start=None, decided=()):
    """decided: what people say about level changes (bmab.redlines.decided_links). One that gives never places anything; one with a kind replaces the
    classified kind (a cut scene can look like a teleport while the passage is physical)."""
    banned = {key(d) for d in decided if d.get("gives")}
    kinds = {key(d): d["kind"] for d in decided if "kind" in d}
    marks = {n: _landmarks(b) for n, b in maps.items()}
    positions = {n: _positions(b) for n, b in maps.items()}
    all_links = links(maps)

    # One edge per (map pair, landmark); its kind is the least trustworthy seen from either side.
    edges, dangling, dead = {}, [], []
    order = list(JOINS)
    for a, b, mark, fired in all_links:
        if b not in maps:
            dangling.append((a, b))
            continue
        here, there = marks[a].get(mark), marks[b].get(mark)
        if here is None or there is None:
            dead.append((a, b, mark))
            continue
        kind = classify(maps[a], positions[a], here, fired)
        other = classify(maps[b], positions[b], there, fired)
        kind = min(kind, other, key=order.index)
        k = (*sorted((a, b)), mark)
        d = here - there if (a, b) == k[:2] else there - here  # b's origin in a's frame, key order
        if k in edges:
            kind = min(kind, edges[k]["kind"], key=order.index)
        edges[k] = {"maps": list(k[:2]), "landmark": mark, "kind": kinds.get(k, kind), "delta": d}

    adj = defaultdict(list)
    for k, e in edges.items():
        if k in banned:
            continue
        a, b = e["maps"]
        adj[a].append((b, e["delta"], k))
        adj[b].append((a, -e["delta"], k))

    linked = {m for k in edges for m in k[:2]}
    connected = sorted(n for n in maps if n in linked)
    roots = ([start] if start in adj else []) + connected
    offset, component, tree = {}, {}, set()
    count = 0
    for root in roots:
        if root in offset:
            continue
        offset[root], component[root] = np.zeros(3), root
        heap = []

        def push(a):
            nonlocal count
            for b, d, k in adj[a]:
                if b not in offset:
                    count += 1
                    heapq.heappush(heap, (-RANK[edges[k]["kind"]], count, a, b, k, d))

        push(root)
        while heap:
            _, _, a, b, k, d = heapq.heappop(heap)
            if b in offset:
                continue
            offset[b], component[b] = offset[a] + d, root
            tree.add(k)
            push(b)

    gaps = []
    for k, e in edges.items():
        if e["kind"] == "teleport" or k in tree:
            continue
        a, b = e["maps"]
        err = offset[a] + e["delta"] - offset[b]
        if np.linalg.norm(err) > 1:
            gaps.append({"maps": [a, b], "landmark": e["landmark"], "kind": e["kind"],
                         "error": [round(float(x), 1) for x in err],
                         "metres": round(float(np.linalg.norm(err)) * INCH, 1)})
    gaps.sort(key=lambda g: -g["metres"])

    return {
        "start": start,
        "maps": {n: {"component": component[n], "offset": [round(float(x), 1) for x in offset[n]]}
                 for n in connected},
        "unlinked": sorted(n for n in maps if n not in linked),
        "links": [{"maps": e["maps"], "landmark": e["landmark"], "kind": e["kind"], "placed_by": k in tree}
                  for k, e in sorted(edges.items())],
        "dangling": dangling,
        "dead": dead,
        "gaps": gaps,
    }
