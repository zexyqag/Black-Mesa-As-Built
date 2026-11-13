"""Layout, stages 2 and 3: move maps apart until they fit together as one place, at the least unseen cost.

Model
- Maps are rigid and only translate. Maps that contain the same room at the same height without being
  linked (revisits) are one body and move together.
- Each link is a join between two maps. Its residual (how far its two ends sit apart) is unseen passage;
  cost per metre depends on the link's kind and direction (a lift stretches cheaply up and down, a track
  ride in any direction, a door doesn't; a teleport costs nothing), times its game's tier weight. Links
  decided to give are cheaper. A "same" link joins a room built into two games' maps (bmab.shared): any
  residual is the two games disagreeing, so it is stiff.
- A connector of a cut map (bmab.connectors) can only run longer, in line with itself: more of the passage can build nothing else.
- Space is a voxel model: every facility floor cell (Xen floors left out) with the space above it up to its
  ceiling (open sky capped). Two maps in different bodies that share voxels clash, cost per cubic metre of shared
  space, except where linked maps are meant to be one space: voxels both fill at the link's ideal alignment, over
  the same floor (a transition corridor, a room built into both).
- Ground (redlines model/surface.json): a map with open sky over its floors must reach the surface, its lowest sky
  opening within a tolerance of the reference map's. A map without open sky (none, or less than min_sky_m2:
  a skylight, or a void the PS2 maps draw with the sky texture) stays underground, its highest space at
  most the tolerance above the reference sky. Cost per metre beyond, times the map's tier weight.
- Relations from the place table (bmab.redlines.place_table: a map next to a place) cost per metre beyond a set distance.

Search (deterministic for a given order and seed, every greedy step logged)
Take a violation (a clash, or a map breaking the ground rule) in the chosen order. Candidate moving sides
come from the graph of bodies joined by links: cut one link that separates the violating map from what it
clashes with (or from the surface reference), or cut the cheap links (teleports, then rides and the like,
then expansion links), alone or with one more. Try each side in each useful direction at the smallest step
that clears the violation and apply the move that lowers the total cost most. Then compaction: close each
stretched join by moving a side beyond it back, as far as the total cost keeps falling. Violations and
compaction alternate until neither changes anything. Optionally then simulated annealing (random rigid moves
of bodies or of sides beyond links, worse layouts accepted with a probability that falls over the run, the best
kept) and the greedy search again: it can cross a clash to reach free space, which greedy moves can't.
"""
import bisect
import random
from collections import defaultdict

import numpy as np

from . import bsp
from . import graph as graphmod
from . import plan
from .graph import INCH

CELL = 48          # voxel size, units (about 1.2 m)
MIN_HEAD = 2       # cells of space above a floor even under a low ceiling
SKY_HEAD = 8       # cells of space above a floor open to the sky (about 10 m)
MAX_HEAD = 100     # cap for very tall rooms (about 120 m)
_BASE = 1 << 20    # key encoding; coordinates in cells must stay within +-2^19
NUDGES = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.int64)
CELL_M = CELL * INCH

PROFILES = {
    # per-metre cost of stretching a link, by kind: (horizontal, vertical)
    "strict": {"walk": (1.0, 1.0), "ladder": (1.0, 0.4), "lift": (1.0, 0.2), "ride": (0.3, 0.3), "teleport": (0.0, 0.0),
               "same": (5.0, 5.0), "gives": 0.5, "clash_per_m3": 20.0, "relation_per_m": 1.0, "ground_per_m": 2.0},
    "lenient": {"walk": (0.3, 0.3), "ladder": (0.3, 0.12), "lift": (0.3, 0.06), "ride": (0.1, 0.1), "teleport": (0.0, 0.0),
                "same": (5.0, 5.0), "gives": 0.5, "clash_per_m3": 20.0, "relation_per_m": 1.0, "ground_per_m": 2.0},
}
# a connector of a cut map (bmab.connectors, by kind): per-metre cost of running longer along its own line, the only change more of
# it can build, (level, vertical): a longer lift ride is cheaper than a longer tram ride, and crawling, the slowest, costs most
INNER = {"inner-walk": (0.7, 0.7), "inner-crouch": (1.0, 1.0), "inner-climb": (0.4, 0.4), "inner-ride": (0.3, 0.2), "inner-fall": (0.4, 0.4)}
INNER_VERTICAL = 0.7  # share of a connector's direction that is up, for its vertical cost
UNBUILDABLE_PER_M = 10000.0  # a connector shorter than it may be, or with its two ends off its line
FIXED_PER_M = 1000.0  # with maps cut at their connectors a level change is one place: the connectors take up the distance
LEVEL_CHANGES = ("walk", "ladder", "lift", "ride")
TIER_WEIGHT = {1: 1.0, 2: 0.6, 3: 0.35, 4: 0.2, 5: 0.1}
NEAR_M = 30.0
MAX_FILLERS = 4096
CHEAP = (0.0, 0.35, 0.65)  # links up to these per-metre weights are cut together when forming moving sides
DIRS = [np.array(d, np.int64) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]


def _encode(c):
    c = np.asarray(c, dtype=np.int64) + (_BASE >> 1)
    return (c[..., 0] * _BASE + c[..., 1]) * _BASE + c[..., 2]


def _xslice(keys, x0, x1):
    """The sorted keys whose x cell lies in [x0, x1] (x is the most significant part of a key)."""
    h = _BASE >> 1
    return keys[np.searchsorted(keys, (x0 + h) * _BASE * _BASE):np.searchsorted(keys, (x1 + 1 + h) * _BASE * _BASE)]


def _floor_z(keys):
    """For each of a map's sorted voxel keys, the z part of its floor: the bottom of its unbroken column."""
    new = np.r_[True, np.diff(keys) != 1]
    return keys[np.maximum.accumulate(np.where(new, np.arange(len(keys)), 0))] % _BASE


def _agreeing(ka, kb, rel):
    """The keys of ka that kb also fills once rel (cells) is added to kb, over the same floor, give or take a cell."""
    both, ia, ib = np.intersect1d(ka, kb + (_encode(rel) - _encode(np.zeros(3, np.int64))), assume_unique=True,
                                  return_indices=True)
    return both[np.abs(_floor_z(ka)[ia] - _floor_z(kb)[ib] - rel[2]) <= 1]


def _raster(polys):
    """{(x, y): [z cells]} of the cells each polygon covers, z from the polygon's mean height."""
    from PIL import Image, ImageDraw
    cells = defaultdict(list)
    for p in polys:
        z = int(np.floor(p[:, 2].mean() / CELL))
        x0, y0 = np.floor(p[:, :2].min(0) / CELL).astype(int)
        x1, y1 = np.ceil(p[:, :2].max(0) / CELL).astype(int)
        m = Image.new("1", (x1 - x0 + 1, y1 - y0 + 1))
        ImageDraw.Draw(m).polygon([(x / CELL - x0, y / CELL - y0) for x, y, _ in p], fill=1)
        yy, xx = np.nonzero(np.array(m))
        if len(xx) == 0:  # thin strips: keep the cell under the centroid
            c = p[:, :2].mean(0) / CELL
            xx, yy = np.array([int(c[0]) - x0]), np.array([int(c[1]) - y0])
        for x, y in zip(xx + x0, yy + y0):
            cells[(int(x), int(y))].append(z)
    return cells


def columns(b):
    """A map's facility space as columns of open space (no Xen floors, no pieces outside the facility: see
    bmab.pieces): runs (n x 4: x, y, floor z, head, cells) and the sky above its floors, {(x, y): [z]}."""
    floors = plan.floors(b, np.zeros(3), facility=True)
    if not floors:
        return np.zeros((0, 4), np.int64), {}
    ceilings, skies = [], []
    for f in b.faces:
        if f.model != 0 or f.skybox or f.outside or f.normal[2] >= -0.7:
            continue
        t = f.texture.lower()
        if t == "sky":
            skies.append(f.verts)
        elif not bsp.is_tool(t) and not plan.XEN.search(t):
            ceilings.append(f.verts)
    fl, ce, sk = _raster(floors), _raster(ceilings), _raster(skies)
    for k in ce:
        ce[k].sort()
    runs = []
    for (x, y), zf_list in fl.items():
        cz = ce.get((x, y), [])
        for zf in set(zf_list):
            i = bisect.bisect_right(cz, zf)
            runs.append((x, y, zf, min(MAX_HEAD, max(MIN_HEAD, cz[i] - zf)) if i < len(cz) else SKY_HEAD))
    return np.array(runs, np.int64), {k: (v, fl.get(k, ())) for k, v in sk.items()}


def volume(b):
    """A map's facility space in its own coordinates: sorted voxel keys, their bounding box (cells), and its open
    sky as (lowest z in units, m2) of sky faces with facility floor below them."""
    if hasattr(b, "volume"):  # a room of a cut map (bmab.connectors) knows its own voxels
        return b.volume()
    runs, sk = columns(b)
    if not len(runs):
        return np.zeros(0, np.int64), np.zeros((2, 3), np.int64), (None, 0.0)
    n = runs[:, 3]
    c = np.column_stack([np.repeat(runs[:, :2], n, axis=0), np.concatenate([np.arange(f, f + h) for f, h in zip(runs[:, 2], n)])])
    open_sky = [z for zl, floors in sk.values() for z in zl if any(zf < z for zf in floors)]
    sky = (min(open_sky) * CELL, len(open_sky) * CELL_M ** 2) if open_sky else (None, 0.0)
    return np.unique(_encode(c)), np.array([c.min(0), c.max(0)]), sky


def _landmark(b, mark):
    for e in b.entities:
        if e.get("classname") == "info_landmark" and e.get("targetname") == mark:
            return np.array([float(x) for x in e["origin"].split()])
    return None


def game_layout(maps, start, decided, game, tier, physical=False):
    """One game's maps placed by its own links: the graph (every component placed from its own root) and the
    links with their landmark deltas, tagged with game and tier. physical=True: teleports place nothing and
    are left out, so components are walkable parts."""
    banned = {graphmod.key(d) for d in decided if d["gives"]}
    g = graphmod.build(maps, start, decided)
    if physical:
        cut = [{**l, "gives": True} for l in g["links"] if l["kind"] == "teleport"]
        g = graphmod.build(maps, start, [*decided, *cut])
    links = []
    for link in g["links"]:
        if physical and link["kind"] == "teleport":
            continue
        a, b = link["maps"]
        pa, pb = _landmark(maps[a], link["landmark"]), _landmark(maps[b], link["landmark"])
        links.append({"maps": [a, b], "landmark": link["landmark"], "kind": link["kind"],
                      "gives": graphmod.key(link) in banned, "delta": pa - pb,
                      "game": game, "tier": tier})
    return g, links


def parts(g):
    """{map: walkable part} of a physical game_layout graph; a map without links is a part of its own."""
    return {**{n: m["component"] for n, m in g["maps"].items()}, **{n: n for n in g["unlinked"]}}


def revisit_links(maps, start, decided, game, tier, peaks):
    """"same" links for rooms a game builds into two maps that no walk connects (a room revisited after a
    teleport), from the game's shared rooms with itself (bmab.shared): the landmark chain can't place them."""
    part = parts(game_layout(maps, start, decided, game, tier, physical=True)[0])
    best = {}
    for p in peaks:
        a, b = p["with"], p["map"]
        if p["class"] == "room" and a in part and b in part and part[a] != part[b]:
            k = tuple(sorted((a, b)))
            if k not in best or p["matched_m2"] > best[k]["matched_m2"]:
                best[k] = p
    return [{"maps": [p["with"], p["map"]], "landmark": f"same room, {p['matched_m2']} m2", "kind": "same",
             "gives": False, "delta": -np.array(p["translation"], float), "game": game, "tier": tier}
            for _k, p in sorted(best.items())]


def join_revisits(offsets, links, part, anchor):
    """Start offsets with every revisited room on itself: walkable parts reached from the anchor's part
    through revisit links shift whole by the link's delta (a teleport between them placed nothing physical)."""
    out = {n: np.asarray(o, float).copy() for n, o in offsets.items()}
    done, changed = {part[anchor]}, True
    while changed:
        changed = False
        for l in links:
            a, b = l["maps"]
            if a not in out or b not in out:
                continue
            for x, y, sign in ((a, b, 1), (b, a, -1)):
                if part[x] in done and part[y] not in done:
                    shift = out[x] + sign * l["delta"] - out[y]
                    for n in out:
                        if part.get(n) == part[y]:
                            out[n] += shift
                    done.add(part[y])
                    changed = True
    return out


def revisits(maps, offsets, linked, games=None, revisit=0.6, min_m2=20):
    """Pairs of unlinked maps of one game with the same floor at the same height where they are placed: one
    room visited twice. linked: (map, map) pairs that are linked (their shared floor is the passage)."""
    polys = {n: plan.floors(maps[n], np.asarray(o, float), facility=True) for n, o in offsets.items()}
    games = games or {}
    return [o["maps"] for o in plan.overlaps(polys, linked)
            if not o["linked"] and games.get(o["maps"][0]) == games.get(o["maps"][1])
            and o["same_height"] >= revisit and o["m2"] >= min_m2]


class Solver:
    def __init__(self, maps, placed, links, anchor, locations=None, surface=None, profile="strict", revisit=0.6,
                 tiers=None, games=None, rigid_tier=0, log=print, stretch=False, shorter=False, said=None):
        """placed: {map: start offset}; links: dicts with maps, landmark, kind, gives, delta, game, tier.
        games: {map: game}; revisits tie maps of one game only (a room two games share is a "same" link).
        rigid_tier: maps joined by links of this tier or better (teleports included: their placement was
        solved) keep their placed relative positions (one body), so lower-tier evidence can't reshape them.
        stretch: cut the maps into rooms and connectors first (bmab.connectors); level changes then stay where they are.
        shorter: a connector may also lose what its whole length says can be taken out. said: {map: its claims}, for stretch."""
        if stretch:
            from . import connectors
            maps, placed, links, anchor, locations, surface, tiers, games = connectors.split(
                maps, placed, links, anchor, locations, surface, tiers, games, log=log, shorter=shorter, said=said)
        self.maps = maps
        self.stretch = stretch
        self.p = PROFILES[profile]
        self.profile = profile
        self.log = log
        self.names = sorted(placed)
        idx = {n: i for i, n in enumerate(self.names)}
        self.idx = idx
        self.anchor = idx[anchor]
        tiers = tiers or {}
        self.tiers = [tiers.get(n, 1) for n in self.names]
        self.off0 = np.array([placed[n] for n in self.names], float)
        self.base_cells = np.round(self.off0 / CELL).astype(np.int64)
        self.shift = np.zeros((len(self.names), 3), np.int64)
        self.links = [{**l, "a": idx[l["maps"][0]], "b": idx[l["maps"][1]]} for l in links
                      if l["maps"][0] in idx and l["maps"][1] in idx]
        self._weigh_links()

        self.keys, lo, hi, skies = [], [], [], []
        for n in self.names:
            k, box, sky = volume(maps[n])
            self.keys.append(k)
            lo.append(box[0])
            hi.append(box[1])
            skies.append(sky)
        self.lo, self.hi = np.array(lo, np.int64), np.array(hi, np.int64)
        self._cache, self._fillers = {}, {}
        self.maps_of = [getattr(maps[n], "map", n) for n in self.names]
        self.rooms = defaultdict(list)
        for i, m in enumerate(self.maps_of):
            self.rooms[m].append(i)

        parent = list(range(len(self.names)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        linked = {(self.names[x], self.names[y]) for l in self.links if l["kind"] not in INNER
                  for x in self.rooms[self.maps_of[l["a"]]] for y in self.rooms[self.maps_of[l["b"]]]}
        self.ties = revisits(maps, {n: self.off0[idx[n]] for n in self.names}, linked | {tuple(l["maps"]) for l in self.links},
                             games, revisit)
        self.agreed = self._agreements()
        for a, b in self.ties:
            parent[find(idx[a])] = find(idx[b])
        for l in self.links:
            if l.get("tier", 1) <= rigid_tier and l["kind"] != "same" and l["kind"] not in INNER:
                parent[find(l["a"])] = find(l["b"])
        self.body = [find(i) for i in range(len(self.names))]
        self.tied = self._tied_links()
        self._bedges = defaultdict(list)
        for k, l in enumerate(self.links):
            ba, bb = self.body[l["a"]], self.body[l["b"]]
            if ba != bb:
                self._bedges[ba].append((bb, k))
                self._bedges[bb].append((ba, k))

        self.relations = []
        if locations:
            here = defaultdict(set)
            for e in locations.get("here", []):
                for m in e["maps"]:
                    if m in idx:
                        here[e["place"]].add(idx[m])
            for a in locations.get("adjacent", []):
                _game, m = a["from"].split(":")
                others = sorted(here.get(a["to"], set()) - {idx.get(m)})
                if m in idx and others:
                    self.relations.append((idx[m], others, self.tiers[idx[m]], f"{m} next to {a['to']}"))

        self.ground, self.ref, self.tol = {}, None, 0.0
        if surface and surface.get("reference") in idx:
            excluded = {e["map"] for e in surface.get("exclude", [])}
            for i, n in enumerate(self.names):
                z, m2 = skies[i]
                if n in excluded or not len(self.keys[i]):
                    continue
                if m2 >= surface.get("min_sky_m2", 500):
                    self.ground[i] = ("sky", z)
                else:
                    self.ground[i] = ("roof", float(self.hi[i][2] + 1) * CELL)
            ref = idx[surface["reference"]]
            if self.ground.get(ref, ("",))[0] == "sky":
                self.ref, self.tol = ref, surface.get("tolerance_m", 60)
            else:
                self.ground = {}
        self.moves = []

    def _agreements(self):
        """{(i, j): keys of map i} of what linked maps are meant to share: the voxels both fill at the link's ideal
        alignment (a cell of give either way, for rounding), over the same floor. Worked out on whole maps (a floor is
        a floor however its map was cut), then shared out to the rooms of a cut map."""
        whole = {m: np.sort(np.concatenate([self.keys[i] for i in rooms])) for m, rooms in self.rooms.items()}
        found = defaultdict(list)
        for l in self.links:
            if l["kind"] == "teleport" or l["kind"] in INNER:  # a connector's two halves meet edge to edge
                continue
            rel = np.round(np.asarray(l["delta"], float) / CELL).astype(np.int64)
            ma, mb = self.maps_of[l["a"]], self.maps_of[l["b"]]
            if not len(whole[ma]) or not len(whole[mb]):
                continue
            for nudge in NUDGES:
                found[ma, mb].append(_agreeing(whole[ma], whole[mb], rel + nudge))
                found[mb, ma].append(_agreeing(whole[mb], whole[ma], -(rel + nudge)))
        shared = {pair: np.unique(np.concatenate(parts)) for pair, parts in found.items()}
        out = {}
        for (ma, mb), keys in shared.items():
            for a in self.rooms[ma]:
                mine = np.intersect1d(keys, self.keys[a], assume_unique=True)
                if len(mine):
                    for b in self.rooms[mb]:
                        out[a, b] = mine
        out = {(a, b): v for (a, b), v in out.items() if (b, a) in out}
        for x, y in {tuple(sorted((self.maps_of[self.idx[a]], self.maps_of[self.idx[b]]))) for a, b in self.ties}:
            for a in self.rooms[x]:  # one place visited twice: the rooms of the two maps stay on each other where they are
                for b in self.rooms[y]:
                    shift = _encode(self.base_cells[b] - self.base_cells[a]) - _encode(np.zeros(3, np.int64))
                    both = np.intersect1d(self.keys[a], self.keys[b] + shift, assume_unique=True)
                    if len(both):
                        out[a, b] = np.union1d(out.get((a, b), both), both)
                        out[b, a] = np.union1d(out.get((b, a), both - shift), both - shift)
        return out

    def _weigh_links(self):
        self.inner = [l for l in self.links if l["kind"] in INNER]
        ins = self.p.get("insert")  # (fixed cost, cost per metre) of an inserted unseen area, or None
        for l in self.links:
            fixed = self.stretch and (l.get("fixed") or (l["kind"] in LEVEL_CHANGES and not l.get("gives")))
            h, v = (FIXED_PER_M, FIXED_PER_M) if fixed else self.p.get(l["kind"], self.p["walk"])
            f = (self.p["gives"] if l.get("gives") else 1.0) * TIER_WEIGHT[l.get("tier", 1)]
            l["w"] = np.array([h, h, v]) * f
            if l["kind"] in INNER:
                l["axis"] = np.asarray(l["axis"], float) / np.linalg.norm(l["axis"])
                l["w"] = INNER[l["kind"]][int(abs(l["axis"][2]) >= INNER_VERTICAL)] * f
            l["ins"] = (ins[0] * f, ins[1] * f) if ins and l["kind"] not in ("same", "teleport") and l["kind"] not in INNER else None
        self.lw = [float(np.max(l["w"])) for l in self.links]

    @staticmethod
    def link_costs(l, r):
        """Cost of a link at residuals r (..., 3 units): stretched, or, where an inserted unseen area (a corridor,
        stairwell or tunnel of any length and direction, at a fixed cost plus a lower rate) is cheaper, that. A connector
        can only run longer (or lose what may be taken out), in line with itself: its two ends are on that line to the nearest cell, or the layout can't be built."""
        if l["kind"] in INNER:
            along, off, ok = Solver._on_line(l, r)
            return np.where(ok, np.abs(along) * l["w"], (np.linalg.norm(off, axis=-1) + np.maximum(-along - l.get("shorter", 0.0), 0)) * UNBUILDABLE_PER_M) * INCH
        c = np.linalg.norm(r * l["w"], axis=-1) * INCH
        if l.get("ins"):
            c = np.minimum(c, l["ins"][0] + l["ins"][1] * np.linalg.norm(r, axis=-1) * INCH)
        return c

    @staticmethod
    def _on_line(l, r):
        """How far a connector's ends part along its line, what is left off it (units), and whether that can be built."""
        along = r @ l["axis"]
        off = r - along[..., None] * l["axis"]
        return along, off, (along >= -l.get("shorter", 0.0) - 1e-6) & (np.abs(off) <= CELL / 2 + 1e-6).all(-1)

    def _lines(self, ks):
        """The two ways along each diagonal connector among links ks: a step on the grid's axes can't part those."""
        return [w * self.links[k]["axis"] for k in ks if self.links[k]["kind"] in INNER and np.abs(self.links[k]["axis"]).max() < 0.999 for w in (1, -1)]

    def _tied_links(self):
        """{link: it and the connectors joining the same two bodies}: they can only part together."""
        pairs = defaultdict(list)
        for k, l in enumerate(self.links):
            if l["kind"] in INNER:
                pairs[frozenset((self.body[l["a"]], self.body[l["b"]]))].append(k)
        return {k: frozenset(v) for v in pairs.values() for k in v}

    def inserted(self, l, shift=None):
        """Whether the link's cheapest form is an inserted unseen area."""
        r = self._residual(l, shift)
        return bool(l.get("ins")) and l["ins"][0] + l["ins"][1] * float(np.linalg.norm(r)) * INCH < \
            float(np.linalg.norm(r * l["w"])) * INCH

    def use_profile(self, profile):
        self.p = PROFILES[profile]
        self.profile = profile
        self.links = [dict(l) for l in self.links]
        self._weigh_links()

    # geometry -----------------------------------------------------------------------------------
    def _cells(self, i, shift=None):
        return self.base_cells[i] + (self.shift[i] if shift is None else shift[i])

    def _boxes(self, shift=None):
        c = self.base_cells + (self.shift if shift is None else shift)
        return self.lo + c, self.hi + c

    def _overlap(self, i, j, si=None, sj=None, agreed=False):
        """Shared voxels of maps i and j (optionally shifted), or with agreed only those both are meant to share (the
        same count whichever map is named first); depends only on their relative offset, cached."""
        rel = self._cells(i) + (si if si is not None else 0) - self._cells(j) - (sj if sj is not None else 0)
        if np.any(self.hi[i] + rel < self.lo[j]) or np.any(self.hi[j] < self.lo[i] + rel):
            return 0
        key = (i, j, int(rel[0]), int(rel[1]), int(rel[2]), agreed)
        v = self._cache.get(key)
        if v is None:
            x0, x1 = max(self.lo[i][0] + rel[0], self.lo[j][0]), min(self.hi[i][0] + rel[0], self.hi[j][0])
            ka, kb = (self.agreed[i, j], self.agreed[j, i]) if agreed else (self.keys[i], self.keys[j])
            ki = _xslice(ka, x0 - rel[0], x1 - rel[0]) + (_encode(rel) - _encode(np.zeros(3, np.int64)))
            kj = _xslice(kb, x0, x1)
            pos = np.minimum(np.searchsorted(kj, ki), len(kj) - 1) if len(kj) else None
            v = int((kj[pos] == ki).sum()) if pos is not None and len(ki) else 0
            self._cache[key] = v
        return v

    def _may_clash(self, i, j):
        return self.body[i] != self.body[j]

    def _clash(self, i, j, si=None, sj=None):
        """Voxels maps i and j both fill and aren't meant to share."""
        v = self._overlap(i, j, si, sj)
        return v - self._overlap(i, j, si, sj, agreed=True) if v and (i, j) in self.agreed else v

    def clash_pairs(self):
        lo, hi = self._boxes()
        boxes = np.all((lo[:, None] <= hi[None]) & (lo[None] <= hi[:, None]), axis=2)
        out = []
        for i, j in np.argwhere(np.triu(boxes, 1)):
            if self._may_clash(i, j):
                v = self._clash(i, j)
                if v:
                    out.append((v * CELL_M ** 3, int(i), int(j)))
        return sorted(out, reverse=True)

    # the filler of stretched connectors ----------------------------------------------------------
    def _filler(self, l, shift=None):
        """What fills a stretched connector's gap: its cut's cross-section swept from one side's place to the other's. Keys
        and box in the frame of side a's cells, cached by the gap; (None, None, None) with no gap."""
        s = self.shift if shift is None else shift
        d = (self.base_cells[l["b"]] + s[l["b"]]) - (self.base_cells[l["a"]] + s[l["a"]])
        n = int(np.abs(d).max())
        if n == 0 or d @ l["axis"] < 0:  # taken out, nothing to fill
            return None, None, None
        key = (l["inner"], int(d[0]), int(d[1]), int(d[2]))
        got = self._fillers.get(key)
        if got is None:
            if len(self._fillers) >= MAX_FILLERS:
                self._fillers.clear()
            steps = np.rint(np.arange(n + 1)[:, None] * d / n).astype(np.int64)
            cells = (l["section"][:, None] + steps[None]).reshape(-1, 3)
            got = (np.unique(_encode(cells)), cells.min(0), cells.max(0))
            self._fillers[key] = got
        return got

    def _filler_voxels(self, l, shift=None):
        """Voxels of a connector's filler inside rooms other than its own two."""
        keys, lo, hi = self._filler(l, shift)
        if keys is None:
            return 0
        s = self.shift if shift is None else shift
        origin = self.base_cells[l["a"]] + s[l["a"]]
        box_lo, box_hi = self._boxes(shift)
        near = np.flatnonzero(np.all((box_lo <= hi + origin) & (lo + origin <= box_hi), axis=1))
        zero, total = _encode(np.zeros(3, np.int64)), 0
        for j in near.tolist():
            if j in (l["a"], l["b"]) or not len(self.keys[j]):
                continue
            q = keys + _encode(origin - (self.base_cells[j] + s[j])) - zero
            pos = np.minimum(np.searchsorted(self.keys[j], q), len(self.keys[j]) - 1)
            total += int((self.keys[j][pos] == q).sum())
        return total

    def filler_m3(self, shift=None):
        return sum(self._filler_voxels(l, shift) for l in self.inner) * CELL_M ** 3

    # costs --------------------------------------------------------------------------------------
    def _residual(self, l, shift=None):
        s = self.shift if shift is None else shift
        pa = self.off0[l["a"]] + s[l["a"]] * CELL
        pb = self.off0[l["b"]] + s[l["b"]] * CELL
        return (pb - pa) - l["delta"]

    def _link_cost(self, l, shift=None):
        r = self._residual(l, shift)
        return float(self.link_costs(l, r)), float(np.linalg.norm(r)) * INCH

    def _relation_cost(self, rel, shift=None):
        """Gap in metres between the bounding boxes of the map and the nearest map of the place."""
        i, others, tier, _ = rel
        lo, hi = self._boxes(shift)
        gaps = np.maximum(0, np.maximum(lo[others] - hi[i], lo[i] - hi[others]))
        d = float(np.linalg.norm(gaps, axis=1).min()) * CELL_M
        return max(0.0, d - NEAR_M) * self.p["relation_per_m"] * TIER_WEIGHT[tier], d

    def _ground_dev(self, i, shift=None):
        """Metres beyond the ground rule: sky maps signed (+ above the surface band, - below), roofs >= 0."""
        s = self.shift if shift is None else shift
        rule, z = self.ground[i]
        zi = z + self.off0[i][2] + s[i][2] * CELL
        zr = self.ground[self.ref][1] + self.off0[self.ref][2] + s[self.ref][2] * CELL
        d = (zi - zr) * INCH
        if rule == "sky":
            return 0.0 if abs(d) <= self.tol else d - np.sign(d) * self.tol
        return max(0.0, d - self.tol)

    def _ground_cost(self, i, shift=None):
        return abs(self._ground_dev(i, shift)) * self.p["ground_per_m"] * TIER_WEIGHT[self.tiers[i]]

    def total(self):
        """The whole layout's cost."""
        links = sum(self._link_cost(l)[0] for l in self.links)
        clashes = (sum(m for m, _, _ in self.clash_pairs()) + self.filler_m3()) * self.p["clash_per_m3"]
        rel = sum(self._relation_cost(r)[0] for r in self.relations)
        ground = sum(self._ground_cost(i) for i in self.ground)
        return links + clashes + rel + ground

    def _delta(self, moving, step):
        """Change in total cost if the maps in `moving` shift by `step` cells."""
        n = len(self.names)
        mv = np.zeros(n, bool)
        mv[list(moving)] = True
        trial = self.shift.copy()
        trial[mv] += step
        d = 0.0
        for l in self.links:
            if mv[l["a"]] != mv[l["b"]]:
                d += self._link_cost(l, trial)[0] - self._link_cost(l)[0]
        lo, hi = self._boxes()
        lo2, hi2 = self._boxes(trial)
        M, O = np.flatnonzero(mv), np.flatnonzero(~mv)
        before = np.all((lo[M][:, None] <= hi[O][None]) & (lo[O][None] <= hi[M][:, None]), axis=2)
        after = np.all((lo2[M][:, None] <= hi2[O][None]) & (lo2[O][None] <= hi2[M][:, None]), axis=2)
        for a, b in np.argwhere(before | after):
            i, j = M[a], O[b]
            if not self._may_clash(i, j):
                continue
            vb = self._clash(i, j) if before[a, b] else 0
            va = self._clash(i, j, si=step) if after[a, b] else 0
            d += (va - vb) * CELL_M ** 3 * self.p["clash_per_m3"]
        for r in self.relations:
            if mv[r[0]] or mv[r[1]].any():
                d += self._relation_cost(r, trial)[0] - self._relation_cost(r)[0]
        if self.ground:
            for i in self.ground:
                if mv[self.ref] != mv[i]:
                    d += self._ground_cost(i, trial) - self._ground_cost(i)
        if self.inner:
            d += self._filler_delta(mv, trial, lo[M], hi[M], lo2[M], hi2[M])
        return d

    def filler_delta(self, moving, step):
        """Change in the filler's cost if the maps in `moving` shift by `step` cells."""
        if not self.inner:
            return 0.0
        mv = np.zeros(len(self.names), bool)
        mv[list(moving)] = True
        trial = self.shift.copy()
        trial[mv] += step
        lo, hi = self._boxes()
        lo2, hi2 = self._boxes(trial)
        return self._filler_delta(mv, trial, lo[mv], hi[mv], lo2[mv], hi2[mv])

    def _filler_delta(self, mv, trial, lo, hi, lo2, hi2):
        """Change in the filler's cost: only connectors stretched across the moved maps, moving with them, or that they touch."""
        out = 0
        for l in self.inner:
            ma, mb = mv[l["a"]], mv[l["b"]]
            now = self._filler(l)
            if ma == mb:
                if now[0] is None:
                    continue
                if not ma:
                    origin = self.base_cells[l["a"]] + self.shift[l["a"]]
                    a, b = now[1] + origin, now[2] + origin
                    if not (np.all((lo <= b) & (a <= hi), axis=1).any() or np.all((lo2 <= b) & (a <= hi2), axis=1).any()):
                        continue
            out += self._filler_voxels(l, trial) - self._filler_voxels(l)
        return out * CELL_M ** 3 * self.p["clash_per_m3"]

    # search -------------------------------------------------------------------------------------
    def _reach(self, start, cut):
        seen, todo = {start}, [start]
        while todo:
            x = todo.pop()
            for y, k in self._bedges[x]:
                if k not in cut and y not in seen:
                    seen.add(y)
                    todo.append(y)
        return seen

    def _sides(self, keep, move, through=None):
        """Sets of whole bodies containing body `move` but not body `keep` that can shift rigidly, each with the
        links the shift stretches. through: a link that must be among the cut ones (closing that join)."""
        cheap = [frozenset(k for k, w in enumerate(self.lw) if w <= c) for c in CHEAP]
        if through is None:
            full = self._reach(move, set())
            singles = [k for k, l in enumerate(self.links)
                       if self.body[l["a"]] in full and self.body[l["a"]] != self.body[l["b"]]]
            own = [self.tied.get(k, frozenset([k])) for k in singles]
            cuts = own + [cheap[0] | c for c in own] + cheap
        else:
            cuts = [c | self.tied.get(through, frozenset([through])) for c in cheap]
        n_bodies = len(set(self.body))
        found = set()
        for cut in cuts:
            side = frozenset(self._reach(move, cut))
            if keep not in side and len(side) < n_bodies:
                found.add(side)
        if through is None:
            found.add(frozenset({move}))
        members = defaultdict(list)
        for m in range(len(self.names)):
            members[self.body[m]].append(m)
        out = []
        for side in found:
            ks = [k for k, l in enumerate(self.links) if (self.body[l["a"]] in side) != (self.body[l["b"]] in side)]
            out.append(([m for b in side for m in members[b]], ks))
        return out

    def _clear_step(self, fixed, moving_map, direction, max_cells=400):
        """Smallest multiple of direction (cells, or a diagonal connector's line) for moving_map that removes its overlap with fixed."""
        def at(n):
            return np.rint(direction * n).astype(np.int64)
        if self._clash(fixed, moving_map, sj=at(max_cells)):
            return None
        lo, hi = 1, max_cells
        while lo < hi:
            mid = (lo + hi) // 2
            if self._clash(fixed, moving_map, sj=at(mid)):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def violations(self):
        out = [("clash", m3 * self.p["clash_per_m3"], i, j, round(m3)) for m3, i, j in self.clash_pairs()]
        for i in self.ground:
            c = self._ground_cost(i)
            if c > 0 and i != self.ref:
                out.append(("ground", c, self.ref, i, round(self._ground_dev(i))))
        return out

    def _candidates(self, v):
        kind, _cost, i, j, _size = v
        out = []
        for mover, other in ((j, i), (i, j)):
            if kind == "ground" and mover != j:
                continue  # the reference stays; moving the rest relative to it is the same layout
            if self.body[mover] == self.body[other]:
                continue
            sides = self._sides(self.body[other], self.body[mover])
            if kind == "clash":
                steps = [d * n for d in DIRS if (n := self._clear_step(other, mover, d)) is not None]
            else:
                dev = self._ground_dev(mover)
                steps = [np.array([0, 0, -np.sign(dev) * int(np.ceil(abs(dev) / CELL_M))], np.int64)]
            out += [(moving, step, ks) for moving, ks in sides for step in steps]
            if kind == "clash":
                for moving, ks in sides:
                    for d in self._lines(ks):
                        n = self._clear_step(other, mover, d)
                        if n is not None:
                            out.append((moving, np.rint(d * n).astype(np.int64), ks))
        return out

    def _apply(self, what, kind, moving, step, ks, delta):
        if len(moving) > len(self.names) / 2:  # same layout, described by the smaller side moving back
            moving = [m for m in range(len(self.names)) if m not in set(moving)]
            step = -step
        self.shift[list(moving)] += step
        stretched = [f"{self.links[k]['maps'][0]}/{self.links[k]['maps'][1]} ({self.links[k]['kind']})" for k in ks]
        move = {"fixes": kind, "what": what, "moved": sorted(self.names[m] for m in moving),
                "by_m": [round(float(x) * CELL_M, 1) for x in step], "stretches": stretched,
                "cost_change": round(delta, 1)}
        self.moves.append(move)
        self.log(f"  {what}: move {len(moving)} maps {move['by_m']} m, changing {', '.join(stretched[:4])}"
                 f"{' ...' if len(stretched) > 4 else ''}; cost {delta:+.0f}")

    def _fix(self, order, max_moves, rng):
        stuck, changed = set(), False
        for _ in range(max_moves):
            vs = [v for v in self.violations() if (v[0], v[2], v[3]) not in stuck]
            if not vs:
                break
            if order == "largest":
                v = max(vs, key=lambda v: v[1])
            elif order == "smallest":
                v = min(vs, key=lambda v: v[1])
            elif order == "play":
                v = min(vs, key=lambda v: (min(v[2], v[3]), -v[1]))
            else:
                v = rng.choice(vs)
            best = None
            for moving, step, ks in self._candidates(v):
                delta = self._delta(moving, step)
                if best is None or delta < best[0]:
                    best = (delta, moving, step, ks)
            if best is None or best[0] >= 0:
                stuck.add((v[0], v[2], v[3]))
                continue
            kind, _c, i, j, size = v
            what = (f"{self.names[i]} x {self.names[j]} ({size} m3)" if kind == "clash"
                    else f"{self.names[j]} {size:+} m off the ground rule")
            self._apply(what, kind, best[1], best[2], best[3], best[0])
            changed = True
        return changed

    def solve(self, order="largest", max_moves=300, rounds=4, anneal=0, gpu=0, gpu_search=None):
        """order: largest | smallest | play | random:SEED (which violation to fix first). anneal: that many
        annealing proposals after the greedy search (seeded by the order), then the greedy search again.
        gpu: anneal on the GPU instead, `anneal` batches of this many candidate steps (gpu_search: a
        bmab.gpu.GpuAnneal of this solver to reuse)."""
        rng = random.Random(_seed(order)) if order.startswith("random") else None
        self.greedy(order, rng, max_moves, rounds)
        if anneal:
            if gpu:
                from . import gpu as gpumod
                (gpu_search or gpumod.GpuAnneal(self)).anneal(_seed(order), anneal, K=gpu, log=self.log)
            else:
                self.anneal(_seed(order), anneal)
            self.greedy(order, rng, max_moves, rounds)
        return self.report(order + (f" +{'gpu ' if gpu else ''}anneal {anneal}" if anneal else ""))

    def greedy(self, order, rng=None, max_moves=300, rounds=4):
        """Violation fixing and compaction, alternating until neither changes anything."""
        rng = rng or (random.Random(_seed(order)) if order.startswith("random") else None)
        for _ in range(rounds):
            changed = self._fix(order, max_moves, rng)
            changed = self.compact() or changed
            if not changed:
                break

    def set_offsets(self, offsets):
        """Take the layout of a report (its offsets relative to the anchor)."""
        for i, n in enumerate(self.names):
            self.shift[i] = np.round((np.asarray(offsets[n], float) - self.off0[i]) / CELL).astype(np.int64)

    def _groups(self):
        """Every set of bodies that can shift rigidly without the anchor's body: single bodies, and the side
        beyond a link cut alone or with the cheap links. Each as (map indices, links it stretches)."""
        keep = self.body[self.anchor]
        cheap = [frozenset(k for k, w in enumerate(self.lw) if w <= c) for c in CHEAP]
        found = {frozenset({b}) for b in set(self.body) if b != keep}
        for k, l in enumerate(self.links):
            ba, bb = self.body[l["a"]], self.body[l["b"]]
            if ba == bb:
                continue
            own = self.tied.get(k, frozenset([k]))
            for cut in (own, *(c | own for c in cheap)):
                for start in (ba, bb):
                    side = frozenset(self._reach(start, cut))
                    if keep not in side:
                        found.add(side)
        members = defaultdict(list)
        for m in range(len(self.names)):
            members[self.body[m]].append(m)
        out = []
        for side in sorted(found, key=sorted):
            ks = [k for k, l in enumerate(self.links) if (self.body[l["a"]] in side) != (self.body[l["b"]] in side)]
            out.append(([m for b in sorted(side) for m in members[b]], ks))
        return out

    def _propose(self, rng, groups, crossing):
        """A random rigid move (map indices, step in cells): a group shifted by a log-uniform amount, a group
        closing a stretched join partway, or a fix for a violation (sometimes with a sideways offset)."""
        def amount(top):
            return int(round(np.exp(rng.uniform(0, np.log(top)))))
        r = rng.random()
        if r < 0.5:
            moving, ks = rng.choice(groups)
            lines = self._lines(ks)
            if lines and rng.random() < 0.5:
                return moving, np.rint(rng.choice(lines) * amount(120)).astype(np.int64)
            if rng.random() < 0.6:
                step = DIRS[rng.randrange(6)] * amount(120 if rng.random() < 0.7 else 30)
            else:
                step = np.array([rng.choice((-1, 1)) * amount(80), rng.choice((-1, 1)) * amount(80),
                                 rng.choice((-1, 0, 1)) * amount(30)], np.int64)
            return moving, step
        if r < 0.8:
            costs = np.array([self._link_cost(l)[0] if crossing[k] else 0.0 for k, l in enumerate(self.links)])
            if costs.sum() <= 0:
                return None, None
            k = rng.choices(range(len(self.links)), weights=costs)[0]
            moving = rng.choice(crossing[k])
            sign = 1 if self.links[k]["b"] in set(moving) else -1
            res = np.round(self._residual(self.links[k]) / CELL).astype(np.int64)
            step = -sign * np.round(res * rng.choice((1.0, 0.75, 0.5, 0.25))).astype(np.int64)
            return (moving, step) if step.any() else (None, None)
        vs = self.violations()
        if not vs:
            return None, None
        cands = self._candidates(rng.choice(vs))
        if not cands:
            return None, None
        moving, step, _ks = rng.choice(cands)
        if rng.random() < 0.5:
            step = step + np.array([rng.choice((-1, 1)) * amount(60), rng.choice((-1, 1)) * amount(60), 0], np.int64)
        return moving, step

    def anneal(self, seed, iters, t0=30.0, t1=0.3, max_cache=1_000_000):
        """Simulated annealing from the current layout: random rigid moves, a worse one accepted with
        probability exp(-cost increase / temperature) as the temperature falls from t0 to t1; ends on the best
        layout seen. Lets the search climb out of what the greedy one can't (a group that must pass through a
        clash to reach free space)."""
        import math
        rng = random.Random(seed)
        groups = self._groups()
        crossing = defaultdict(list)
        for moving, ks in groups:
            for k in ks:
                crossing[k].append(moving)
        start = cur = self.total()
        best, best_shift, accepted = cur, self.shift.copy(), 0
        for it in range(iters):
            temp = t0 * (t1 / t0) ** (it / iters)
            moving, step = self._propose(rng, groups, crossing)
            if moving is None:
                continue
            d = self._delta(moving, step)
            if d < 0 or rng.random() < math.exp(-d / temp):
                self.shift[moving] += step
                cur += d
                accepted += 1
                if cur < best - 1e-6:
                    best, best_shift = cur, self.shift.copy()
            if len(self._cache) > max_cache:
                self._cache.clear()
        self.shift = best_shift
        self.moves.append({"fixes": "anneal", "what": f"annealing, seed {seed}: {accepted} of {iters} proposals "
                           f"accepted", "moved": [], "by_m": [], "stretches": [],
                           "cost_change": round(best - start, 1)})
        self.log(f"  annealing: cost {start:.0f} -> {best:.0f} ({accepted} of {iters} accepted)")

    def compact(self, passes=4):
        """Close stretched joins: shift a side beyond a join back toward closing it (whole or in part), as far
        as the total cost keeps falling (no clash, ground or relation penalty outweighing the saving)."""
        any_change = False
        for _ in range(passes):
            improved = False
            for k, l in sorted(enumerate(self.links), key=lambda kl: -self._link_cost(kl[1])[0]):
                w, length = self._link_cost(l)
                ba, bb = self.body[l["a"]], self.body[l["b"]]
                if l["kind"] == "teleport" or length < CELL_M or ba == bb:
                    continue
                r = np.round(self._residual(l) / CELL).astype(np.int64)
                best = None
                for keep, move, sign in ((ba, bb, -1), (bb, ba, 1)):
                    for moving, ks in self._sides(keep, move, through=k):
                        for frac in (1.0, 0.75, 0.5, 0.25):
                            step = sign * np.round(r * frac).astype(np.int64)
                            if step.any():
                                delta = self._delta(moving, step)
                                if best is None or delta < best[0]:
                                    best = (delta, moving, step, ks)
                if best and best[0] < -0.5:
                    self._apply(f"shorten {l['maps'][0]}/{l['maps'][1]}", "compact", best[1], best[2], best[3], best[0])
                    improved = any_change = True
            if not improved:
                break
        return any_change

    def report(self, order=""):
        joins, teleports, shared = [], [], []
        for l in self.links:
            weighted, length = self._link_cost(l)
            base = {"maps": l["maps"], "landmark": l["landmark"], "game": l.get("game", ""), "tier": l.get("tier", 1)}
            if l["kind"] == "same":
                shared.append({**base, "disagreement_m": round(length, 1), "cost": round(weighted, 1)})
            elif length > 1.0 and l["kind"] == "teleport":
                teleports.append({**base, "offset_m": round(length, 1)})
            elif length > 1.0:
                joins.append({**base, "kind": l["kind"], "gives": bool(l.get("gives")), "inserted": self.inserted(l),
                              "unseen_m": round(length, 1), "cost": round(weighted, 1), **({"at": l["middle"]} if "middle" in l else {})})
        clashes = [{"maps": [self.names[i], self.names[j]], "m3": round(m)} for m, i, j in self.clash_pairs()]
        rels = []
        for r in self.relations:
            _c, d = self._relation_cost(r)
            rels.append({"relation": r[3], "distance_m": round(d), "satisfied": bool(d <= NEAR_M)})
        ground = [{"map": self.names[i], "rule": self.ground[i][0], "beyond_m": round(float(self._ground_dev(i)), 1)}
                  for i in sorted(self.ground)]
        anchor = self.shift[self.anchor].copy()
        offsets = {n: [round(float(x)) for x in self.off0[i] + (self.shift[i] - anchor) * CELL]
                   for i, n in enumerate(self.names)}
        return {"profile": self.profile, "order": order, "maps": self.names, "offsets": offsets, "ties": self.ties,
                "moves": self.moves, "joins": sorted(joins, key=lambda j: -j["unseen_m"]), "teleports": teleports,
                "shared_rooms": sorted(shared, key=lambda s: -s["disagreement_m"]),
                "clashes": clashes, "relations": rels, "ground": ground,
                "unseen_total_m": round(sum(j["unseen_m"] for j in joins), 1),
                "disagreement_total_m": round(sum(s["disagreement_m"] for s in shared), 1),
                "clash_total_m3": round(sum(c["m3"] for c in clashes)), "filler_clash_m3": round(self.filler_m3()),
                "off_ground_maps": sum(1 for g in ground if g["beyond_m"] != 0),
                "unbuildable_connectors": sum(1 for l in self.inner if not self._on_line(l, self._residual(l))[2]),
                "total_cost": round(self.total(), 1)}


def draw(maps, report, path, size=4000):
    """Floor plan of the solved layout with every stretched join drawn between its two ends."""
    from PIL import Image, ImageDraw, ImageFont
    offs = {n: np.array(o, float) for n, o in report["offsets"].items()}
    polys = {n: plan.floors(maps[n], offs[n], facility=True) for n in report["maps"]}
    plan.render(polys, path, size)
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    v = np.concatenate([p for ps in polys.values() for p in ps])
    lo, hi = v.min(0), v.max(0)
    scale = size / max(hi[0] - lo[0], hi[1] - lo[1])
    h = img.height
    font = ImageFont.load_default(size=22)

    def px(p):
        return (20 + (p[0] - lo[0]) * scale, h - 20 - (p[1] - lo[1]) * scale)
    for j in report["joins"]:
        if j["kind"] in INNER:
            continue
        a, b = j["maps"]
        pa = _landmark(maps[a], j["landmark"]) + offs[a]
        pb = _landmark(maps[b], j["landmark"]) + offs[b]
        colour = (255, 220, 40) if j["gives"] or j["kind"] != "walk" else (255, 120, 40)
        d.line([px(pa), px(pb)], fill=colour, width=5)
        for q in (px(pa), px(pb)):
            d.ellipse([q[0] - 8, q[1] - 8, q[0] + 8, q[1] + 8], outline=colour, width=3)
        m = ((px(pa)[0] + px(pb)[0]) / 2, (px(pa)[1] + px(pb)[1]) / 2)
        d.text((m[0] + 10, m[1]), f"{a}/{b} {j['unseen_m']} m", fill=colour, font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(path)


def _seed(order):
    return int(order.split(":")[1]) if order.startswith("random") else 0


_BASE_SOLVER = None


def _copy(profile, shift=None, moves=()):
    import copy
    s = copy.copy(_BASE_SOLVER)
    s.shift = (_BASE_SOLVER.shift if shift is None else shift).copy()
    s.moves = list(moves)
    s.use_profile(profile)
    s.log = lambda *_: None
    return s


def _run_variant(args):
    profile, order, anneal = args
    return _copy(profile).solve(order, anneal=anneal)


def _run_greedy(args):
    profile, order, shift, moves, label = args
    s = _copy(profile, shift, moves)
    s.greedy(order)
    return s.report(label) if label else (s.shift, s.moves)


def warm(solver, profile, seeds, batches, gpu, t0, log=print):
    """Go on from the solver's current layout, one report per seed: a cool GPU search that keeps what is there, then the greedy
    search, then a last one that clears connector filler out of other rooms. No batches: only that last search."""
    global _BASE_SOLVER
    _BASE_SOLVER = solver
    search, reports = None, []
    for seed in range(1, seeds + 1):
        c = _copy(profile, solver.shift)
        if batches:
            c.log = log
            if search is None:
                from . import gpu as gpumod
                search = gpumod.GpuAnneal(c)
            else:
                search.adopt(c)
            found, exact = search.anneal(seed, batches, K=gpu, t0=t0, log=log), c.total()
            if abs(found - exact) > 1:
                raise RuntimeError(f"the search's cost {found:.1f} and the exact cost {exact:.1f} disagree")
        g = _copy(profile, c.shift, c.moves)  # on its own copy: the GPU's grid follows c
        if batches:
            g.greedy(f"random:{seed}")
        g.greedy("largest")
        reports.append(g.report(f"random:{seed} +warm gpu anneal {batches}" if batches else "largest"))
    return reports


def compare(solver, variants, processes=10, gpu=0):
    """Run (profile, order, anneal) variants in parallel from one prepared solver (fork shares its voxels).
    gpu: the greedy searches run in parallel, the annealing one variant at a time on the GPU (batches of this
    many candidate steps, one occupancy grid kept and updated), then the final greedy searches in parallel."""
    import multiprocessing as mp
    global _BASE_SOLVER
    _BASE_SOLVER = solver
    ctx = mp.get_context("fork")
    if not gpu:
        with ctx.Pool(processes) as pool:
            return pool.map(_run_variant, variants)
    from . import gpu as gpumod
    with ctx.Pool(processes) as pool:
        first = pool.map(_run_greedy, [(p, o, None, (), None) for p, o, _a in variants])
    search, annealed = None, []
    for (p, o, a), (shift, moves) in zip(variants, first):
        s = _copy(p, shift, moves)
        if search is None:
            search = gpumod.GpuAnneal(s)
        else:
            search.adopt(s)
        if a:
            search.anneal(_seed(o), a, K=gpu, log=solver.log)
        annealed.append((p, o, s.shift.copy(), s.moves, o + (f" +gpu anneal {a}" if a else "")))
    with ctx.Pool(processes) as pool:
        return pool.map(_run_greedy, annealed)
