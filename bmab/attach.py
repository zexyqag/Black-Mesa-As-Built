"""Layout, stage 3: attach other games' maps to a solved layout by the rooms they share with it, or by facts.

- Each added game is split into walkable parts: its own links without teleports (trips to Xen and other
  teleports carry no placement), joined where the game builds one room into two maps (bmab.shared against
  itself: a room revisited after a teleport). A map without links is a part of its own.
- A part is attached if one of its maps shares a room with a placed map: identical faces (bmab.shared, class
  room) or a room aligned in the place table (bmab.redlines.place_table, same_place with align). Placed by its
  largest shared room, the rest of the part as that game's own stage 2 solved it (or by its landmark chain).
  Split: every map with a shared room starts on its twin and the rest follow the nearest of them, so the part
  starts bent around all its shared rooms. Every shared room becomes a "same" link (stiff: any residual is
  the games disagreeing); the game's own links join at its tier.
- A part without a shared room starts by a fact: one of its maps next to a place (place table, adjacent)
  whose maps are placed, centred on them for the solver to push out into clear space; the relation then
  costs per metre beyond solve.NEAR_M. Parts with neither stay out, for later facts.
"""
from collections import defaultdict

import numpy as np

from . import plan, solve


def aligned(locations, game, base):
    """Rooms the place table aligns between `game` and `base` (same_place entries with align), as peaks."""
    out = []
    for e in (locations or {}).get("same_place", []):
        a = e.get("align")
        if not a or a["to"].split(":")[0] != base:
            continue
        for gm in e["maps"]:
            g, m = gm.split(":")
            if g == game:
                out.append({"map": m, "with": a["to"].split(":")[1], "class": "room", "matched_m2": 0,
                            "translation": a["translation"], "label": f"same place: {a['how']}"})
    return out


def _label(p):
    return p.get("label", f"shared room, {p['matched_m2']} m2")


def _centre(b, offset):
    v = np.concatenate(plan.floors(b, np.asarray(offset, float), facility=True))
    return (v.min(0) + v.max(0)) / 2


def _bent(own, rooms, world, pairs):
    """Each map with a shared room exactly on its twin (its largest room); every other map at its own offset
    from the anchored map fewest links (or revisits) away."""
    out = {}
    for p in sorted(rooms, key=lambda p: p["matched_m2"]):
        out[p["map"]] = np.array(world[p["with"]], float) - np.array(p["translation"], float)
    nbr = defaultdict(set)
    for a, b in pairs:
        if a in own and b in own:
            nbr[a].add(b)
            nbr[b].add(a)
    origin = {n: n for n in out}
    frontier = sorted(out)
    while frontier:
        nxt = []
        for n in frontier:
            for m in sorted(nbr[n]):
                if m not in origin:
                    origin[m] = origin[n]
                    nxt.append(m)
        frontier = nxt
    return {n: out[origin[n]] + own[n] - own[origin[n]] for n in own if n in origin}


def _by_fact(own, game, locations, world, geometry):
    """(map, place, placed maps of the place) for the first map of the part a direction sign puts next to a
    place that has placed maps."""
    here = defaultdict(set)
    for e in (locations or {}).get("here", []):
        here[e["place"]].update(e["maps"])
    for n in sorted(own):
        for a in (locations or {}).get("adjacent", []):
            targets = sorted(m for m in here.get(a["to"], ()) if m in world and m in geometry)
            if a["from"] == f"{game}:{n}" and targets:
                return n, a["to"], targets
    return None


def attach(placed, geometry, additions, locations=None, split=False):
    """placed: {map: world offset} of the base layout; geometry: {map: bsp} for its maps (added maps join it).
    additions: dicts with game, tier, maps, start, decided, rooms (peaks against placed maps), self (the
    game's peaks against itself) and solved (its own stage 2 offsets, or None: its landmark chain places each
    part). split: start each part bent around all its shared rooms instead of placed whole by the largest.
    Returns offsets and links for the added maps, and how each part was placed."""
    world = {n: np.asarray(o, float) for n, o in placed.items()}
    out_placed, out_links, info = {}, [], []
    for add in additions:
        game, tier, solved = add["game"], add["tier"], add["solved"]
        g, links = solve.game_layout(add["maps"], add["start"], add["decided"], game, tier, physical=True)
        revisit = solve.revisit_links(add["maps"], add["start"], add["decided"], game, tier, add["self"])
        part_of = {n: c for n, c in solve.parts(g).items() if not solved or n in solved}
        merged = {c: c for c in part_of.values()}

        def find(c):
            while merged[c] != c:
                c = merged[c]
            return c
        for l in revisit:
            a, b = l["maps"]
            if a in part_of and b in part_of:
                ra, rb = sorted((find(part_of[a]), find(part_of[b])))
                merged[rb] = ra
        part_of = {n: find(c) for n, c in part_of.items()}
        pairs = [l["maps"] for l in links + revisit]
        own_all = {n: np.array(solved[n] if solved else g["maps"][n]["offset"] if n in g["maps"] else (0, 0, 0),
                               float) for n in part_of}
        if not solved:  # each part was placed from its own root: put revisited rooms on themselves
            first = {}
            for n in sorted(part_of):
                first.setdefault(part_of[n], n)
            phys = solve.parts(g)
            for root in first.values():
                own_all = solve.join_revisits(own_all, revisit, phys, root)

        rooms = {}
        for p in add["rooms"]:
            key = (p["map"], p["with"])
            if p["class"] == "room" and p["with"] in world and p["map"] in part_of:
                if key not in rooms or p["matched_m2"] > rooms[key]["matched_m2"]:
                    rooms[key] = p
        by_part = defaultdict(list)
        for p in rooms.values():
            by_part[part_of[p["map"]]].append(p)
        left = {}
        for part in sorted(set(part_of.values())):
            own = {n: own_all[n] for n, c in part_of.items() if c == part}
            ps = by_part.get(part)
            if ps:
                best = max(ps, key=lambda p: p["matched_m2"])
                if split:
                    here = _bent(own, ps, world, pairs)
                else:
                    shift = world[best["with"]] - np.array(best["translation"], float) - own[best["map"]]
                    here = {n: o + shift for n, o in own.items()}
                for p in ps:
                    out_links.append({"maps": [p["with"], p["map"]], "landmark": _label(p), "kind": "same",
                                      "gives": False, "delta": -np.array(p["translation"], float),
                                      "game": game, "tier": tier})
                how = {"placed_by": f"{best['map']} = {best['with']}",
                       "rooms": [f"{p['map']} = {p['with']} ({_label(p)})" for p in ps]}
            elif fact := _by_fact(own, game, locations, world, geometry):
                n, place, targets = fact
                centre = np.mean([_centre(geometry[m], world[m]) for m in targets], axis=0)
                shift = centre - _centre(add["maps"][n], own[n])
                here = {m: o + shift for m, o in own.items()}
                how = {"placed_by": f"fact: {n} next to {place} ({', '.join(targets)})"}
            else:
                left[part] = sorted(own)
                continue
            out_placed.update(here)
            world.update(here)
            geometry.update({n: add["maps"][n] for n in here})
            out_links += [l for l in links + revisit if l["maps"][0] in here and l["maps"][1] in here]
            info.append({"game": game, "part": sorted(here), **how})
        info.append({"game": game, "not_attached": left})
    return out_placed, out_links, info
