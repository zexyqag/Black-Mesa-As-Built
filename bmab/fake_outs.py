"""Fake outs: where a map looks like it goes on. Open space joined to what the player sees, which runs on out of sight (a
second rail curving round a corner, a channel past a bend, a tunnel mouth). It needs the BSP tree.

Sight lines are cast from where the player can be: eye height over the route's floors (bmab.pieces), and on a
ride along the train's path, from inside the car (its own faces block the view, so it is seen through its windows
and doors). Every open cell a sight line crosses before it meets a wall is seen. Open space
that no sight line reaches, joined to seen space, and at least MIN_M3 big, is a fake out: where the player's view
of it ends, and how far it goes on. The game's own visibility table is too generous for this (it counts space
round a corner as visible), hence the sight lines.
Besides, a dark end: a black surface where seen space ends (a tunnel or a pit closed with a black wall), the
designers' way of showing that it goes on into darkness.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import bsp, connectors, memory, pieces
from .graph import INCH

G = 32             # cell size of the seen grid, units
STEP, RANGE = 16, 4000
RAYS = 4000        # sight lines from each eye point
EYE = 64           # eye height over the floor
EVERY = 4          # one eye point every EVERY route floor cells (pieces.CELL) each way
MIN_M3 = 20
NO_CONTROL = 2   # func_tracktrain spawnflag
RIDE_NEAR = 400  # a landmark this close to a train's path: the player rides it through the level change
RIDE_CLEAR = 1536  # route floors this close to a ride's track are what the ride shows, not walked
DARK = ("black", "toolsblack")  # texture names (Source: the last part of the material path)
MIN_DARK_M2 = 2


def _open(b, p):
    return b.vis.open(p)


def _origin(e):
    return np.array([float(x) for x in e["origin"].split()])


def _path(b, first):
    """The segments of the path_track chain from first, following target and altpath."""
    tracks = {e["targetname"]: e for e in b.entities if e.get("classname") == "path_track" and e.get("targetname")}
    segs, todo, done = [], [first], set()
    while todo:
        t = todo.pop()
        if t in done or t not in tracks:
            continue
        done.add(t)
        for k in ("target", "altpath"):
            if tracks[t].get(k) in tracks:
                segs.append((_origin(tracks[t]), _origin(tracks[tracks[t][k]])))
                todo.append(tracks[t][k])
    return segs


def _near(p, segs):
    """Distance from each point to the nearest segment."""
    out = np.full(len(p), np.inf)
    for a, c in segs:
        d = c - a
        t = np.clip(((p - a) @ d) / max(float(d @ d), 1e-9), 0, 1)
        out = np.minimum(out, np.linalg.norm(p - (a + t[:, None] * d), axis=1))
    return out


def eyes(b):
    """Where the player's eyes can be: along the path of a train the player drives (func_tracktrain), and over
    the route's floor cells, every EVERY-th. On a map with a train that carries the player (no user control,
    its path passing a level change's landmark), route floors within RIDE_CLEAR of its track don't count: they
    are the track bed and the channels beside it, seen from the train, not walked (where a ride ends at a
    station, a corner of the platform the train can't see may come up as a fake out).
    Returns the points, and for each the car it is in (an index into cars, -1: none) and the car's heading; cars
    holds each train's faces as triangles around its origin, and the eye's height over that origin (a train built
    as a prop model, like Black Mesa's trams, has no faces: nothing of it blocks the view)."""
    marks = np.array([_origin(e) for e in b.entities if e.get("classname") == "info_landmark" and "origin" in e])
    out, car, yaw, cars, rides = [], [], [], [], []
    for e in b.entities:
        if e.get("classname") != "func_tracktrain" or not e.get("model", "").startswith("*"):
            continue
        segs = _path(b, e.get("target"))
        if not segs:
            continue
        carries = len(marks) and _near(marks, segs).min() <= RIDE_NEAR
        if int(e.get("spawnflags", 0) or 0) & NO_CONTROL and not carries:
            continue  # moves on its own and nobody rides it: scenery
        mi = int(e["model"][1:])
        floor_z = b.models[mi][0][2]
        tris = [f.verts[[0, j, j + 1]] for f in b.faces if f.model == mi and not bsp.is_tool(f.texture)
                for j in range(1, len(f.verts) - 1)]
        cars.append((np.array(tris) if tris else None, floor_z + EYE))
        for a, c in segs:
            k = max(2, int(np.linalg.norm(c - a) / (EVERY * pieces.CELL)))
            out.append(a + (c - a) * np.linspace(0, 1, k)[:, None] + [0, 0, floor_z + EYE])
            car.append(np.full(k, len(cars) - 1))
            yaw.append(np.full(k, np.arctan2(c[1] - a[1], c[0] - a[0])))
        if carries:
            rides += segs
    ps = pieces.analyse(b)
    if ps:
        lab, lo = b.piece_grid
        route = np.isin(lab, [p["id"] for p in ps if p["kind"] == "route"])
        floor = route & ~np.concatenate([np.zeros_like(route[:, :, :1]), route[:, :, :-1]], 2)  # lowest cell
        c = np.argwhere(floor)
        c = c[(c[:, 0] % EVERY == 0) & (c[:, 1] % EVERY == 0)]
        p = np.column_stack([(c[:, :2] + lo[:2] + 0.5) * pieces.CELL, (c[:, 2] + lo[2]) * pieces.CELL + EYE])
        p = p[_near(p, rides) > RIDE_CLEAR] if rides else p
        out.append(p)
        car.append(np.full(len(p), -1))
        yaw.append(np.zeros(len(p)))
    if not out:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0), cars
    p, car, yaw = np.concatenate(out), np.concatenate(car), np.concatenate(yaw)
    ok = _open(b, p)
    return p[ok], car[ok], yaw[ok], cars


def _car_reach(dirs, height, tris, yaw):
    """How far each sight line from an eye in a car gets before a face of the car (RANGE if none): the line
    turned into the car's frame (origin height below the eye, heading yaw), against its triangles."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = dirs @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # rows rotated by -yaw about z
    o = np.array([0.0, 0.0, height])
    a, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    reach = np.full(len(d), float(RANGE))
    for k in range(0, len(d), 512):
        dd = d[k:k + 512, None, :]
        h = np.cross(dd, e2[None])
        det = np.einsum("rtk,tk->rt", h, e1)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        sv = (o - a)[None]
        u = np.einsum("rtk,rtk->rt", np.broadcast_to(sv, h.shape), h) * inv
        q = np.cross(np.broadcast_to(sv, h.shape), e1[None])
        v = np.einsum("rtk,rtk->rt", np.broadcast_to(dd, q.shape), q) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-3)
        reach[k:k + 512] = np.where(hit, t, np.inf).min(1).clip(max=RANGE)
    return reach


def _viewpoint(b, eyes, at):
    """The nearest eye point with a clear line to at (stopping short of it, as at is just out of sight); failing
    that the nearest."""
    order = np.argsort(np.linalg.norm(eyes - at, axis=1))
    for i in order[:200]:
        d = at - eyes[i]
        n = np.linalg.norm(d)
        t = np.arange(0, max(0.0, n - 2 * G), 8)
        if not len(t) or _open(b, eyes[i] + t[:, None] * d / n).all():
            return eyes[i]
    return eyes[order[0]]


def find(b):
    """[{id, type "out_of_sight", at, reaches, seen_from, m3, size_m, length_m}]: at is where the view of the space ends, reaches
    its farthest cell, seen_from the nearest eye point that sees up to at (to look from); map units."""
    if b.vis is None:
        return []
    e, car, yaw, cars = eyes(b)
    if not len(e):
        return []
    k = np.arange(RAYS) + 0.5
    phi, th = np.arccos(1 - 2 * k / RAYS), np.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    lo, hi = getattr(b.vis, "bounds", (b.models[0][0], b.models[0][1]))  # Source: the 3D skybox left out
    shape = np.ceil((hi - lo) / G).astype(int) + 1
    seen = np.zeros(shape, bool)
    for i, p in enumerate(e):
        d = dirs
        reach = _car_reach(dirs, cars[car[i]][1], cars[car[i]][0], yaw[i]) \
            if car[i] >= 0 and cars[car[i]][0] is not None else np.full(len(dirs), float(RANGE))
        for s in range(1, RANGE // STEP):
            q = p + d * (s * STEP)
            ok = _open(b, q) & (reach > s * STEP)
            if not ok.any():
                break
            q, d, reach = q[ok], d[ok], reach[ok]
            c = np.clip(((q - lo) // G).astype(int), 0, shape - 1)
            seen[c[:, 0], c[:, 1], c[:, 2]] = True
    axes = [lo[i] + G / 2 + np.arange(shape[i]) * G for i in range(3)]
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), -1).reshape(-1, 2)
    open_ = np.zeros(shape, bool)
    for i, x in enumerate(axes[0]):  # a slab at a time: the coordinates of a big outdoor map won't fit in memory at once
        open_[i] = _open(b, np.column_stack([np.full(len(yz), x), yz])).reshape(shape[1:])
    seen &= open_
    hidden = open_ & ~ndimage.binary_dilation(seen)  # a cell next to a seen one is as good as seen
    lab, _n = ndimage.label(hidden)
    touch = ndimage.binary_dilation(seen, iterations=2) & hidden
    ids = np.unique(lab[touch])
    ids = ids[ids > 0]
    m3 = (G * 0.0254) ** 3
    sizes = np.bincount(lab[lab > 0])[ids] * m3 if len(ids) else []
    out = []
    for i, size in sorted(zip(ids, sizes), key=lambda x: -x[1]):
        if size < MIN_M3:
            break
        w = np.argwhere(lab == i)
        con = np.argwhere((lab == i) & touch).mean(0)
        at = lo + G / 2 + con * G
        far = lo + G / 2 + w[np.argmax(np.linalg.norm(w - con, axis=1))] * G
        eye = _viewpoint(b, e, at)
        out.append({"id": len(out) + 1, "type": "out_of_sight", "at": [round(float(x)) for x in at], "reaches": [round(float(x)) for x in far],
                    "seen_from": [round(float(x)) for x in eye], "m3": round(float(size)),
                    "size_m": [round(float(x), 1) for x in (w.max(0) - w.min(0) + 1) * G * 0.0254],
                    "length_m": round(float(np.linalg.norm(far - at)) * 0.0254, 1)})
    return out


def need_gb(b):
    """About how much memory find needs for a map: per grid cell the seen, open and hidden grids, the labels and the
    dilations (Source also its solid grid, 8 finer cells a cell), and the map itself."""
    lo, hi = getattr(b.vis, "bounds", (b.models[0][0], b.models[0][1]))
    cells = float(np.prod(np.ceil((np.asarray(hi) - lo) / G) + 1))
    return 0.5 + cells * (18 if isinstance(b.vis, bsp.SourceVis) else 10) / 2 ** 30


def _one(job):
    """A map's fake outs, or None if it needed more memory than it may have (a failure here beats exhausting the machine)."""
    from . import sources
    from .game import Game
    src, name, areas, limit_gb = job
    memory.limit(limit_gb)
    try:
        b = Game(sources.DATA / src).read(name)
        pieces.analyse(b, areas)
        return name, search(b)
    except MemoryError:
        return name, None


def search(b):
    """A map's fake outs: spaces that run on out of sight, then its dark ends."""
    found = find(b)
    return found + [{**d, "id": len(found) + k} for k, d in enumerate(dark_ends(b), 1)]


def dark_ends(b, e=None):
    """Black surfaces in pieces the player sees: [{type "dark_end", at (middle), reaches (into the black), seen_from, m2}]."""
    from .doors import _groups, _area
    ps = pieces.analyse(b)
    if not ps:
        return []
    if e is None:
        e = eyes(b)[0]
    faces = [i for i, f in enumerate(b.faces) if f.model == 0 and not f.skybox and f.texture.split("/")[-1] in DARK]
    out = []
    for g in _groups([b.faces[i] for i in faces]):
        fs = [faces[j] for j in g]
        k = int(b.face_piece[fs[0]])
        if not k or ps[k - 1]["dropped"]:
            continue
        m2 = sum(_area(b.faces[i]) for i in fs) * 0.0254 ** 2
        if m2 < MIN_DARK_M2:
            continue
        f = b.faces[max(fs, key=lambda i: _area(b.faces[i]))]
        n = f.normal / np.linalg.norm(f.normal)
        v = np.concatenate([b.faces[i].verts for i in fs])
        at = (v.min(0) + v.max(0)) / 2 + n * 16
        eye = _viewpoint(b, e, at) if len(e) else at + n * 128
        out.append({"type": "dark_end", "at": [round(float(x)) for x in at], "reaches": [round(float(x)) for x in at - n * 160],
                    "seen_from": [round(float(x)) for x in eye], "m2": round(m2, 1), "m3": 0,
                    "size_m": [round(float(x), 1) for x in (v.max(0) - v.min(0)) * 0.0254], "length_m": 0})
    return out


def way(b, line, tracks=True):
    """How the player would go on along the passage a fake out ends in (claims.WAYS): straight down or up it is a drop, unless a
    ladder or a lift runs there; level, a ride where a train's path runs (unless tracks is False: the caller knows), a crawl
    where there is room to crouch but not to stand, else a walk."""
    at, far = np.array(line["at"], float), np.array(line["reaches"], float)
    into = (far - at) / max(np.linalg.norm(far - at), 1e-9)
    p = at - into * 16
    if abs(into[2]) >= connectors.VERTICAL:
        path = np.array([p, at + into * 160])
        return "ride" if connectors._carrier(b, path, "lift") else "climb" if connectors._carrier(b, path, "ladder") else "fall"
    if tracks and (connectors._distance(np.array([p, at]), connectors._paths(b)) <= connectors.TRACK_M / INCH).any():
        return "ride"
    room = connectors.headroom(b, p)
    return "crouch" if room is not None and connectors.CROUCH <= room < connectors.STAND else "walk"


def end_surface(b, surfaces, fake_out):
    """Where a fake out's passage stops: (a point on the surface it ends at, a point past it, straight into the surface). None if it never stops."""
    at, far = np.array(fake_out["at"], float), np.array(fake_out["reaches"], float)
    d = far - at
    if np.linalg.norm(d) < 1:
        d = far - np.array(fake_out["seen_from"], float)
    d /= np.linalg.norm(d)
    start = far if _open(b, far[None])[0] else at
    for reach in (RANGE, 32768):
        t = np.arange(0, reach, 4.0)
        shut = np.flatnonzero(~_open(b, start[None] + t[:, None] * d))
        if len(shut):
            break
    else:
        return None
    # Source's open space is a grid that pads the brushes, so the space can shut that much before the face
    past = 8 + (bsp.SOLID_CELL + bsp.SOLID_PAD if isinstance(b.vis, bsp.SourceVis) else 0)
    face = connectors._first_face(surfaces, start, d, t[shut[0]] + past)
    n, offset, lo, hi, _fs, _corners = surfaces
    if face is not None:
        q, into = start + face[0] * d, -n[face[1]]
    else:
        q = start + t[shut[0]] * d
        on = np.flatnonzero((np.abs(n @ q - offset) <= 8) & (q >= lo - 8).all(1) & (q <= hi + 8).all(1))
        into = n[on[np.argmax(np.abs(n[on] @ d))]] if len(on) else d
        into = into if into @ d > 0 else -into
    return [round(float(x)) for x in q], [round(float(x)) for x in q + into * 160]


def find_all(jobs, workers, out, log=print):
    """Fake outs of [(source, map, what people say about its areas or None)] jobs over worker processes: ({map: fake outs}, [maps that ran out of memory]). The largest
    go first, and no more run at once than their estimates fit in the memory free at the start, less memory.RESERVE_GB; each
    map may use a little over its estimate. Results so far stay in out + ".partial", so a crash loses nothing: running
    again carries on from there. Each map gets a fresh process, so its memory cap counts only that map, not what the caller
    or an earlier map holds."""
    import multiprocessing

    from . import sources
    from .game import Game
    out = Path(out)
    partial = out.with_name(out.name + ".partial")
    done = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    budget = max(2.0, memory.available_gb() - memory.RESERVE_GB)
    areas = {n: a for _s, n, a in jobs}
    todo = sorted(((need_gb(Game(sources.DATA / s).read(n)), s, n) for s, n, _a in jobs if n not in done), reverse=True)
    log(f"{len(todo)} maps to search ({len(done)} done before), {budget:.1f} GB to use, largest needs {todo[0][0] if todo else 0:.1f} GB")
    failed, running = [], {}
    with multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        while todo or running:
            used = sum(gb for _r, gb in running.values())
            for job in list(todo):
                if len(running) < workers and (not running or used + job[0] <= budget):
                    gb, s, n = job
                    running[n] = pool.apply_async(_one, [(s, n, areas[n], gb * 1.5 + 2)]), gb
                    used += gb
                    todo.remove(job)
            for n in [n for n, (r, _gb) in running.items() if r.ready()]:
                name, found = running.pop(n)[0].get()
                if found is None:
                    failed.append(name)
                    log(f"{name}: out of memory, left out")
                else:
                    done[name] = found
                    partial.write_text(json.dumps(done), encoding="utf-8", newline="\n")
                    log(f"{len(done)}/{len(done) + len(todo) + len(running)} {name}: {len(found)}")
            time.sleep(0.2)
    out.write_text(json.dumps(dict(sorted(done.items()))), encoding="utf-8", newline="\n")
    if partial.exists():
        partial.unlink()
    return done, failed
