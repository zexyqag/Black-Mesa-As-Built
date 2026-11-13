"""Levels joined by a level change are one place where they overlap, and a level change already says where a passage goes.

A claim about a spot two levels both hold is kept in both of their files, each in that level's own coordinates: same_place says
whether they both hold it, twin finds the other level's line, and carried is what that line should say.
"""
from collections import defaultdict

import numpy as np

from . import redlines, graph

KINDS = ("walk", "ladder", "lift", "ride")
DOORS = ("func_door", "func_door_rotating", "momentary_door", "prop_door_rotating")
MIRRORED = ("fake_door", "ought_door", "fake_out", "connector", "part")
TWIN = {"fake_door": 32, "ought_door": 32, "fake_out": 48, "connector": 96, "part": 96, "opening": 48}  # units apart that are still the same thing, as each level measures it itself
POINTS = ("at", "reaches", "seen_from")
PATHS = ("outline", "path")
AT_CHANGE = 128  # units from a level change's trigger: that end of the map is the level change, not a lead
FRONT = 72
PROBE = np.array([(x, y, z) for x in range(-256, 257, 32) for y in range(-256, 257, 32) for z in range(-256, 257, 32)], float)
MIN_SAME = 0.75  # of the space around a spot that either level has, the share both have: unrelated rooms never reach it


def next_to(maps, game):
    """{map: [{map, offset}]} of one game's maps ({name: bsp}): the levels a level change joins it to; a point there plus offset is a point here."""
    decided = redlines.decided_links(game)
    return joined({n: graph._landmarks(b) for n, b in maps.items()}, graph.build(maps, None, decided)["links"], decided)


def joined(marks, links, decided):
    """next_to from what it is worked out from: each map's landmarks ({map: {name: point}}), the game's level changes (bmab.graph.build's
    links) and what people say about them (bmab.redlines.decided_links)."""
    loose = {graph.key(d) for d in decided if d.get("gives")}
    out = defaultdict(list)
    for link in links:
        if link["kind"] not in KINDS or graph.key(link) in loose:
            continue
        a, b = link["maps"]
        delta = marks[a][link["landmark"]] - marks[b][link["landmark"]]
        for here, there, offset in ((a, b, delta), (b, a, -delta)):
            if not any(x["map"] == there for x in out[here]):
                out[here].append({"map": there, "offset": [round(float(v), 2) for v in offset]})
    return dict(out)


def moved(line, by):
    """A claim with every position shifted by a vector, units."""
    go = lambda p: [round(float(v + d)) for v, d in zip(p, by)]  # noqa: E731
    out = dict(line)
    for k in POINTS:
        if k in line:
            out[k] = go(line[k])
    for k in PATHS:
        if k in line:
            out[k] = [go(p) for p in line[k]]
    if "extra_boxes" in line:
        out["extra_boxes"] = [{**m, "at": go(m["at"])} for m in line["extra_boxes"]]
    return out


def _direction(line):
    if line["kind"] in ("fake_door", "ought_door"):
        return np.array(line["back"], float)
    if line["kind"] == "connector":
        return np.array(line["along"], float)
    if line["kind"] == "opening":
        return np.array(line["facing"], float)
    if line["kind"] == "fake_out":
        d = np.subtract(line["reaches"], line["at"], dtype=float)
        return d / (np.linalg.norm(d) or 1)
    return None


def twin(line, rows):
    """The line among rows (another level's) about the same thing as line, already in that level's coordinates; None if it has none."""
    a, near = _direction(line), TWIN[line["kind"]]
    best, gap = None, near
    for r in rows:
        if r["kind"] != line["kind"]:
            continue
        d = float(np.linalg.norm(np.subtract(r["at"], line["at"])))
        if d > gap:
            continue
        along = 1.0 if a is None else float(a @ _direction(r))
        if abs(along) > 0.9 if line["kind"] == "connector" else along > 0.7:
            best, gap = r, d
    return best


def carried(line, offset, here, there):
    """What another level's line should say for this one, without an id. offset: a point there plus offset is a point here; here, there: each level's lines."""
    out = moved({k: v for k, v in line.items() if k != "id"}, -np.asarray(offset, float))
    if line.get("stretches_with"):
        mine = {r["id"]: r for r in here}
        found = [twin(moved(mine[i], -np.asarray(offset, float)), there) for i in line["stretches_with"] if i in mine]
        out["stretches_with"] = [r["id"] for r in found if r is not None]
        if not out["stretches_with"]:
            del out["stretches_with"]
    return out


def _spot(line):
    if line["kind"] in ("fake_door", "ought_door", "fake_out"):
        return np.array(line["at"], float) - _direction(line) * FRONT
    return np.array(line["at"], float)


def same_place(a, b, line, offset):
    """Whether level b (bsp) holds the spot that line in level a is about: the space around it is the same in both."""
    p = _spot(line) + PROBE
    here, there = a.vis.open(p), b.vis.open(p - np.asarray(offset, float))
    return bool((here & there).sum() >= MIN_SAME * max(1, (here | there).sum()))


def _entity_boxes(b, classes):
    out = []
    for e in b.entities:
        if e.get("classname") in classes:
            m = e.get("model", "")
            if m.startswith("*"):  # a brush entity with an origin has its model around that origin
                lo, hi, *_ = b.models[int(m[1:])]
                at = np.array([float(x) for x in e.get("origin", "0 0 0").split()])
                out.append((e, np.array(lo, float) + at, np.array(hi, float) + at))
            elif "origin" in e:
                p = np.array([float(x) for x in e["origin"].split()])
                out.append((e, p, p))
    return out


def _boxes(b, classes):
    return [(lo, hi) for _e, lo, hi in _entity_boxes(b, classes)]


def _dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0)))


def changes(b):
    """The boxes of a level's level-change triggers."""
    return _boxes(b, ("trigger_changelevel",))


def triggers(b):
    """A level's level-change triggers: [(the map it leads to, its landmark, lo, hi)]."""
    return [(e.get("map", ""), e.get("landmark", ""), lo, hi) for e, lo, hi in _entity_boxes(b, ("trigger_changelevel",))]


def at_change(boxes, line):
    """Whether a fake out is at a level change (boxes: changes(b))."""
    return any(_dist(np.array(p, float), lo, hi) <= AT_CHANGE for p in (line["at"], line["reaches"]) for lo, hi in boxes)


def door_works(there, p, cant_use):
    """Whether level there (bsp) has a door at p that isn't one of its doors nobody can use (their positions). By position alone: each level often holds only one side of a door."""
    return any(_dist(p, lo, hi) <= 16 for lo, hi in _boxes(there, DOORS)) and not any(np.linalg.norm(np.subtract(q, p)) <= 96 for q in cant_use)


def answered(here, there, line, offset):
    """Whether a fake out's passage goes on in the other level: it holds the same place, and has open space past the surface."""
    return same_place(here, there, line, offset) and bool(there.vis.open((np.array(line["reaches"], float) - offset)[None])[0])
