"""Simulated annealing on the GPU (PyTorch with CUDA): solve.Solver's cost, with many candidate moves of one
group scored at once.

The layout lives on the GPU as one occupancy grid (how many maps fill each voxel). Scoring a group's candidate
steps: take the group out of the grid, gather the grid at its voxels shifted by each step (overlap with every
other map at once), subtract what it overlaps of maps it is linked to where the two are meant to coincide, put it back.
Links, ground and relations are cheap and scored on the CPU, vectorised over the steps. One step is then drawn
with probability proportional to exp(-cost change / temperature), staying put included, as the temperature
falls; the best layout seen is kept. What fills stretched connectors is checked on the CPU, for the drawn step only: it is
taken with the probability its change in cost allows.
"""
import math
import random
import time

import numpy as np

from . import solve
from .graph import INCH

CELL, CELL_M, TIER_WEIGHT = solve.CELL, solve.CELL_M, solve.TIER_WEIGHT


def _decode(keys):
    h, b = solve._BASE >> 1, solve._BASE
    return np.stack([keys // (b * b) - h, (keys // b) % b - h, keys % b - h], 1)


class GpuAnneal:
    def __init__(self, s, margin=(160, 160, 60), device="cuda"):
        import torch
        self.torch, self.s, self.dev, self.margin = torch, s, device, np.array(margin)
        # Windows drivers spill into system memory instead of failing an allocation, so PyTorch's cache never
        # hears "out of memory" and grows past the card, and the search crawls. Capped at what is free now, the
        # cache is released when full, or the run fails with a real out-of-memory error.
        free, total = torch.cuda.mem_get_info()
        self.cap = max(0.05 * total, free - (256 << 20))
        torch.cuda.set_per_process_memory_fraction(self.cap / total)
        self.n = len(s.names)
        self.cells = [_decode(k) for k in s.keys]
        self._build()
        self.local = {}  # (map, map) -> dense grid of what the first is meant to share with the second
        self.agreed = {}  # (map, map) -> cells of the first the pair are meant to share
        self.pairs = {}  # map pairs with voxels they are meant to share
        for a, b in s.agreed:
            self.pairs.setdefault(a, set()).add(b)

    def _build(self):
        """The occupancy grid around the current layout, with the margin on every side."""
        lo, hi = self.s._boxes()
        self.org = lo.min(0) - self.margin
        self.dims = hi.max(0) - lo.min(0) + 1 + 2 * self.margin
        self.stride = np.array([self.dims[1] * self.dims[2], self.dims[2], 1], np.int64)
        self.grid = None  # the old one goes first: both may not fit on the card
        self.grid = self.torch.zeros(int(np.prod(self.dims)), dtype=self.torch.uint8, device=self.dev)
        self.lin = [self.torch.as_tensor(self._world(i) @ self.stride, device=self.dev) for i in range(self.n)]
        for lin in self.lin:
            self._add(lin, 1)
        # a batch's temporary arrays get what the grid leaves: a widely spread layout works in smaller chunks
        left = max(0, int(self.cap - self.torch.cuda.memory_reserved()))
        self.chunk, self.pair_chunk = max(1 << 20, min(1 << 27, left // 32)), max(1 << 19, min(1 << 26, left // 64))

    def adopt(self, s):
        """Switch to another copy of the same solver (another layout or profile): move the grid to its layout,
        or build a new one if the layout has too little room around it in this one."""
        old, self.s = self.s, s
        lo, hi = s._boxes()
        if (lo.min(0) - self.org < self.margin // 2).any() or (self.org + self.dims - 1 - hi.max(0) < self.margin // 2).any():
            self._build()
            return
        for i in range(self.n):
            if (old.shift[i] != s.shift[i]).any():
                self._add(self.lin[i], -1)
                self.lin[i] = self.lin[i] + int((s.shift[i] - old.shift[i]) @ self.stride)
                self._add(self.lin[i], 1)

    def _world(self, i, step=None):
        return self.cells[i] + s_off(self.s, i) - self.org + (0 if step is None else step)

    def _add(self, lin, v):
        self.grid.index_put_((lin,), self.torch.tensor(v, dtype=self.torch.uint8, device=self.dev).expand(lin.shape),
                             accumulate=True)

    def _shared(self, m, o):
        """Cells of map m (its own frame) it is meant to share with map o, as an n x 3 array."""
        if (m, o) not in self.agreed:
            self.agreed[m, o] = _decode(self.s.agreed[m, o]) if (m, o) in self.s.agreed else np.zeros((0, 3), np.int64)
        return self.agreed[m, o]

    def _dense(self, o, m):
        """Dense box (grid, size, its first cell in o's frame) of the voxels of map o it is meant to share with map m."""
        if (o, m) not in self.local:
            c = self._shared(o, m)
            lo = c.min(0)
            size = c.max(0) - lo + 1
            g = self.torch.zeros(int(np.prod(size)), dtype=self.torch.uint8, device=self.dev)
            g[self.torch.as_tensor((c - lo) @ np.array([size[1] * size[2], size[2], 1]), device=self.dev)] = 1
            self.local[o, m] = (g, size, lo)
        return self.local[o, m]

    def clash_counts(self, moving, steps):
        """Voxels shared with maps outside `moving` that may clash, for each step (K x 3 cells)."""
        t = self.torch
        chunk = self.chunk
        K = len(steps)
        dl = t.as_tensor(steps @ self.stride, device=self.dev)
        idx = t.cat([self.lin[m] for m in moving])
        self._add(idx, -1)  # uint8 wraps but the group's own voxels are in the grid, so no count goes negative
        out = t.zeros(K, dtype=t.int64, device=self.dev)
        size = self.grid.numel()
        per = max(1, chunk // max(1, idx.numel()))
        for k0 in range(0, K, per):
            g = (idx[None, :] + dl[k0:k0 + per, None]).clamp_(0, size - 1)
            out[k0:k0 + per] = self.grid[g].sum(1, dtype=t.int64)
        self._add(idx, 1)
        inside = set(moving)
        for m in moving:
            for o in self.pairs.get(m, ()):
                if o not in inside:
                    out -= self._pair(m, o, steps)
        return out.cpu().numpy()

    def _pair(self, m, o, steps):
        """Overlap with map o of the voxels of map m that the two are meant to share, m shifted by each step."""
        t = self.torch
        chunk = self.pair_chunk
        shared = self._shared(m, o)
        if not len(shared) or not len(self._shared(o, m)):
            return t.zeros(len(steps), dtype=t.int64, device=self.dev)
        g, size, lo = self._dense(o, m)
        rel = t.as_tensor(shared + s_off(self.s, m) - s_off(self.s, o) - lo, device=self.dev)
        st = t.as_tensor(steps, device=self.dev)
        sz = t.as_tensor(size, device=self.dev)
        mul = t.as_tensor([size[1] * size[2], size[2], 1], device=self.dev)
        out = t.zeros(len(steps), dtype=t.int64, device=self.dev)
        per = max(1, chunk // max(1, rel.shape[0] * 3))
        for k0 in range(0, len(steps), per):
            r = rel[None] + st[k0:k0 + per, None, :]
            ok = ((r >= 0) & (r < sz)).all(2)
            lin = (r * mul).sum(2).clamp_(0, g.numel() - 1)
            out[k0:k0 + per] = (g[lin] * ok).sum(1, dtype=t.int64)
        return out

    def move(self, moving, step):
        dl = int(step @ self.stride)
        for m in moving:
            self._add(self.lin[m], -1)
            self.lin[m] = self.lin[m] + dl
            self._add(self.lin[m], 1)
        self.s.shift[moving] += step

    # ---------- CPU parts of the cost, vectorised over steps ----------
    def other_costs(self, moving, ks, steps):
        s = self.s
        mv = np.zeros(self.n, bool)
        mv[moving] = True
        d = np.zeros(len(steps))
        for k in ks:
            l = s.links[k]
            r = s._residual(l)
            sign = 1.0 if mv[l["b"]] else -1.0
            r2 = r[None] + sign * steps * CELL
            d += s.link_costs(l, r2) - s.link_costs(l, r)
        if s.ground:
            zr = s.ground[s.ref][1] + s.off0[s.ref][2] + s.shift[s.ref][2] * CELL
            for i, (rule, z) in s.ground.items():
                if mv[i] == mv[s.ref]:
                    continue
                zi = z + s.off0[i][2] + s.shift[i][2] * CELL
                dz = steps[:, 2] * CELL * (1 if mv[i] else -1)
                d += (self._ground(rule, (zi - zr) * INCH + dz * INCH) - self._ground(rule, np.array([(zi - zr) * INCH]))) \
                    * s.p["ground_per_m"] * TIER_WEIGHT[s.tiers[i]]
        if s.relations:
            lo, hi = s._boxes()
            for i, others, tier, _ in s.relations:
                if not (mv[i] or mv[others].any()):
                    continue
                before = self._gap(lo, hi, i, others, mv, np.zeros((1, 3), np.int64))
                after = self._gap(lo, hi, i, others, mv, steps)
                w = s.p["relation_per_m"] * TIER_WEIGHT[tier]
                d += (np.maximum(0, after - solve.NEAR_M) - np.maximum(0, before - solve.NEAR_M)) * w
        return d

    def _ground(self, rule, dm):
        tol = self.s.tol
        if rule == "sky":
            return np.where(np.abs(dm) <= tol, 0.0, np.abs(dm) - tol)
        return np.maximum(0.0, dm - tol)

    @staticmethod
    def _gap(lo, hi, i, others, mv, steps):
        """Box gap in metres from map i to the nearest of others, per step (moving maps shifted)."""
        sh_i = steps if mv[i] else np.zeros_like(steps)
        loi, hii = lo[i][None] + sh_i, hi[i][None] + sh_i
        lo_o = lo[others][None] + steps[:, None] * mv[others][None, :, None]
        hi_o = hi[others][None] + steps[:, None] * mv[others][None, :, None]
        gaps = np.maximum(0, np.maximum(lo_o - hii[:, None], loi[:, None] - hi_o))
        return np.linalg.norm(gaps, axis=2).min(1) * CELL_M

    def deltas(self, moving, ks, steps):
        """Total cost change of each step; inf where a map would leave the grid."""
        s = self.s
        lo, hi = s._boxes()
        glo, ghi = lo[moving].min(0) - self.org, hi[moving].max(0) - self.org
        ok = ((glo[None] + steps) >= 0).all(1) & ((ghi[None] + steps) < self.dims).all(1)
        c = self.clash_counts(moving, np.where(ok[:, None], steps, 0))
        d = (c - c[0]) * CELL_M ** 3 * s.p["clash_per_m3"] + self.other_costs(moving, ks, steps)
        d[~ok] = np.inf
        return d

    # ---------- search ----------
    def _steps(self, rng, ks, K):
        s = self.s
        out = [np.zeros(3, np.int64)]

        def amount(top):
            return int(round(math.exp(rng.uniform(0, math.log(top)))))
        for k in ks:  # close each stretched join through this cut, whole or in part
            res = np.round(s._residual(s.links[k]) / CELL).astype(np.int64)
            if res.any():
                for f in (1.0, 0.75, 0.5, 0.25):
                    out += [np.round(res * f).astype(np.int64), -np.round(res * f).astype(np.int64)]
        for d in s._lines(ks):
            out += [np.rint(d * amount(150)).astype(np.int64) for _ in range(8)]
        while len(out) < K:
            if rng.random() < 0.5:
                out.append(solve.DIRS[rng.randrange(6)] * amount(150))
            else:
                out.append(np.array([rng.choice((-1, 1)) * amount(100), rng.choice((-1, 1)) * amount(100),
                                     rng.choice((-1, 0, 1)) * amount(40)], np.int64))
        return np.array(out[:K])

    def anneal(self, seed, iters, K=256, t0=30.0, t1=0.3, log=print):
        s = self.s
        rng = random.Random(seed)
        groups = s._groups()
        start = cur = s.total()
        best, best_shift, accepted = cur, s.shift.copy(), 0
        t_start = time.time()
        for it in range(iters):
            temp = t0 * (t1 / t0) ** (it / iters)
            moving, ks = rng.choice(groups)
            steps = self._steps(rng, ks, K)
            d = self.deltas(moving, ks, steps)
            x = -(d - d.min()) / temp
            p = np.exp(np.where(np.isfinite(x), x, -np.inf))
            k = rng.choices(range(len(d)), weights=p)[0]
            fill = s.filler_delta(moving, steps[k]) if k and np.isfinite(d[k]) else math.inf
            if fill <= 0 or (math.isfinite(fill) and rng.random() < math.exp(-fill / temp)):
                self.move(moving, steps[k])
                cur += d[k] + fill
                accepted += 1
                if cur < 0:  # every part of the cost is at least zero: the steps' costs have gone wrong, so stop
                    raise RuntimeError(f"cost below zero ({cur:.0f}) at batch {it}: the GPU's and the CPU's costs disagree")
                if cur < best - 1e-6:
                    best, best_shift = cur, s.shift.copy()
            if log and it % max(1, iters // 10) == 0:
                log(f"  gpu anneal {it}/{iters}: cost {cur:.0f}, best {best:.0f}, temperature {temp:.2f}, "
                    f"{time.time() - t_start:.0f} s, GPU memory {self.torch.cuda.max_memory_reserved() / 2**30:.1f} GB")
        # back to the best layout (grid rebuilt from it)
        for i in range(self.n):
            if (s.shift[i] != best_shift[i]).any():
                self._add(self.lin[i], -1)
                self.lin[i] = self.lin[i] + int((best_shift[i] - s.shift[i]) @ self.stride)
                self._add(self.lin[i], 1)
        s.shift = best_shift
        s.moves.append({"fixes": "anneal", "what": f"GPU annealing, seed {seed}: {accepted} of {iters} batches of "
                        f"{K} steps moved", "moved": [], "by_m": [], "stretches": [], "cost_change": round(best - start, 1)})
        return best


def s_off(s, i):
    return s.base_cells[i] + s.shift[i]
