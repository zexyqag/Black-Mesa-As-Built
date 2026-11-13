"""Real paths for the passages a layout implies.

A stretched link, an inserted area and a connector that runs longer are drawn and costed as straight lines, which pass
through whatever lies between their ends. Here each is routed through the space no map fills: a shortest path over a
grid of coarse cells (4 voxels, about 5 m), longest passages first, each taking its cells so that no two share one. A
route may start and end inside the two maps it joins (a landmark is inside them), but not cross any other map.
"""
import heapq

import numpy as np

from .graph import INCH
from .solve import CELL, INNER, _landmark

COARSE = 4
OWN_COST = 4        # a coarse cell of the route's own two maps costs this much: prefer leaving them soon
NEAR_ENDS = 8       # coarse cells from an end within which a route may run through its own maps
MARGIN = (40, 120, 240)  # coarse cells around the two ends to search in, widened until a path is found
MAX_NODES = 300_000      # cells one search may look at: a walled-in end must not fill the memory


def _decode(k):
    base = 1 << 20
    z = k % base
    y = (k // base) % base
    return np.stack([k // (base * base), y, z], -1) - (base >> 1)


def _occupied(s):
    """{coarse cell: set of maps in it} at the solver's current layout."""
    occ = {}
    for i, k in enumerate(s.keys):
        if len(k):
            c = np.unique((_decode(k) + s._cells(i)) // COARSE, axis=0)
            for t in map(tuple, c.tolist()):
                occ.setdefault(t, set()).add(i)
    return occ


def _ends(s, l):
    """The two ends of a link (world units), or None for a link without a place to start from."""
    if l["kind"] in INNER:
        m = np.array(l["middle"], float)
        return m + s.off0[l["a"]] + s.shift[l["a"]] * CELL, m + s.off0[l["b"]] + s.shift[l["b"]] * CELL
    if l["kind"] in ("same", "teleport"):
        return None
    pa, pb = _landmark(s.maps[l["maps"][0]], l["landmark"]), _landmark(s.maps[l["maps"][1]], l["landmark"])
    if pa is None or pb is None:
        return None
    return pa + s.off0[l["a"]] + s.shift[l["a"]] * CELL, pb + s.off0[l["b"]] + s.shift[l["b"]] * CELL


def _coarse(p):
    return tuple((np.floor(p / CELL).astype(int) // COARSE).tolist())


def _shortest(start, goal, occ, own, margin):
    """Cells of a shortest path from start to goal avoiding occupied cells of maps other than `own`, or None."""
    lo = [min(a, b) - margin for a, b in zip(start, goal)]
    hi = [max(a, b) + margin for a, b in zip(start, goal)]

    def h(c):
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1]) + abs(c[2] - goal[2])

    def cost(c):
        there = occ.get(c)
        if there is None:
            return 1
        if there <= own and min(h(c), abs(c[0] - start[0]) + abs(c[1] - start[1]) + abs(c[2] - start[2])) <= NEAR_ENDS:
            return OWN_COST
        return None
    best = {start: 0}
    came = {}
    todo = [(h(start), 0, start)]
    while todo:
        _f, g, c = heapq.heappop(todo)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return path[::-1]
        if g > best[c]:
            continue
        if len(best) > MAX_NODES:
            return None
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if not all(a <= v <= b for a, v, b in zip(lo, n, hi)):
                continue
            step = 1 if n == goal else cost(n)
            if step is None:
                continue
            g2 = g + step
            if g2 < best.get(n, 1 << 60):
                best[n], came[n] = g2, c
                heapq.heappush(todo, (g2 + h(n), g2, n))
    return None


def _pull(path, occ, own):
    """The path with corners cut wherever a straight line stays clear: grid steps overstate a diagonal by up to 1.7 times."""
    start, goal = path[0], path[-1]

    def clear(a, b):
        n = 2 * max(abs(b[i] - a[i]) for i in range(3)) + 1
        for t in np.linspace(0, 1, n):
            c = tuple(np.rint(np.array(a) + (np.array(b) - a) * t).astype(int).tolist())
            there = occ.get(c)
            if there is None or c in (start, goal):
                continue
            near = min(sum(abs(c[i] - e[i]) for i in range(3)) for e in (start, goal))
            if not (there <= own and near <= NEAR_ENDS):
                return False
        return True
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = next(k for k in range(len(path) - 1, i, -1) if k == i + 1 or clear(path[i], path[k]))
        out.append(path[j])
        i = j
    return out


def through_others(s, l, ends, occ):
    """Metres of the straight line between a link's ends that lie inside maps other than its own two."""
    a, b = ends
    n = max(2, int(np.linalg.norm(b - a) / (CELL * COARSE)) + 1)
    own = {l["a"], l["b"]}
    hits = sum(bool(occ.get(_coarse(a + (b - a) * t), own) - own) for t in np.linspace(0, 1, n))
    return hits * float(np.linalg.norm(b - a)) * INCH / n


def route_all(s, min_m=3.0, log=print):
    """A route for every link whose ends sit apart: [{maps, kind, straight_m, route_m, through_other_m, cells}]; cells is
    None where no path could be found. Longest first, each taking its cells."""
    occ = _occupied(s)
    jobs = []
    for l in s.links:
        ends = _ends(s, l)
        if ends is None:
            continue
        if l["kind"] in INNER:  # a stretched connector fills its own gap: routes keep out of it
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(ends[1] - ends[0]) / (CELL * COARSE)) + 1)):
                occ.setdefault(_coarse(ends[0] + (ends[1] - ends[0]) * t), set()).add(-1)
        elif float(np.linalg.norm(ends[1] - ends[0])) * INCH >= min_m:
            jobs.append((float(np.linalg.norm(ends[1] - ends[0])) * INCH, l, ends))
    out = []
    for straight, l, ends in sorted(jobs, key=lambda j: -j[0]):
        own = {l["a"], l["b"]}
        start, goal = _coarse(ends[0]), _coarse(ends[1])
        through = through_others(s, l, ends, occ)
        path = None
        for margin in MARGIN:
            path = _shortest(start, goal, occ, own, margin)
            if path:
                break
        rec = {"maps": l["maps"], "kind": l["kind"], "straight_m": round(straight, 1), "through_other_m": round(through, 1)}
        if path:
            line = _pull(path, occ, own)
            for a, b in zip(line, line[1:]):  # the corridor takes the cells its straight runs pass through
                n = 2 * max(abs(b[i] - a[i]) for i in range(3)) + 1
                for t in np.linspace(0, 1, n):
                    occ.setdefault(tuple(np.rint(np.array(a) + (np.array(b) - a) * t).astype(int).tolist()), set()).add(-1)
            rec["route_m"] = round(float(np.linalg.norm(np.diff(np.array(line, float), axis=0), axis=1).sum()) * COARSE * CELL * INCH, 1)
            rec["cells"] = line
        else:
            rec["route_m"] = None
            rec["cells"] = None
        out.append(rec)
    done = [r for r in out if r["cells"]]
    log(f"{len(out)} passages: {len(done)} routed, {len(out) - len(done)} without a path; straight {sum(r['straight_m'] for r in out):.0f} m "
        f"({sum(r['through_other_m'] for r in out):.0f} m of it through other areas), routed {sum(r['route_m'] for r in done):.0f} m")
    return out
