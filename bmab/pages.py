"""The review pages (inspector.html), one for each edition of the game (bmab.register.EDITIONS) that has games here: the maps of
the games the register counts as that edition's, the levels a level change joins, the order they are played in and how each map
meets the maps next to it; and each map's view (bmab.view.map_view), built from the game's files the first time it is asked for, in
a process of its own, and kept in the cache folder (bmab.sources.CACHE) until its file or bmab changes. It knows nothing of HTTP.

A map is (game, name); the pages name it "game/name", as a page holds games whose maps share names.
"""
import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import shutil
import threading
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import bsp, graph, jobs, levels, pieces, redlines, register, shared, solve, sources, view
from .game import Game

AREA_KINDS = {"route": ("the route (where the player walks)", "#8a8f98", False),
              "seen": ("seen from the route, not walked (scenery)", "#4aa3ff", False),
              "teleport": ("reached by teleport", "#ffd04a", False),
              "unknown": ("off the route, no visibility data (PS2)", "#e8e8e8", False),
              "xen": ("Xen pocket", "#c070ff", True),
              "unseen": ("never seen from the route", "#ff5a5a", True),
              "holding": ("holding box (start or level-change box)", "#ff9a3c", True)}
CUBE_LIMIT = 1500  # cells shown of what two neighbouring maps share
LEFT_OUT = ("rejected", "duplicate")  # what the register says of games no page shows
CODE = hashlib.sha256(b"".join(p.read_bytes() for p in [*sorted(Path(__file__).parent.glob("*.py")), sources.MANIFEST])).hexdigest()[:16]  # views change with bmab and its sources
VIEWS = sources.CACHE / "views"  # VIEWS/CODE/<game>/: each map's view, its models, its frame and its volume, by its file's sha256
BUILDS_A_PROCESS = 8  # then a new build process, so what one keeps of a game's textures doesn't pile up

_pages = {}  # {edition: {games, maps, levels, play, links, landmarks}}
_lock = threading.Lock()
_running = {}  # {what is being built: (its future, the executor it runs in)}
_pool = []  # the executor of the build process, made when first needed


def key(game, name):
    return f"{game}/{name}"


def _games():
    """{edition: [(game, its title)]} of the games here the register counts, the main games first."""
    out = defaultdict(list)
    for s in sorted(redlines.register()["sources"].values(), key=lambda s: (s.get("tier", 9), s["title"])):
        if s["kind"] == "game" and s.get("verdict") not in LEFT_OUT and (sources.DATA / s["id"] / "maps").is_dir() and Game(sources.DATA / s["id"]).map_names:
            out[s.get("edition", register.EDITIONS[0])].append((s["id"], s["title"]))
    return {e: out[e] for e in register.EDITIONS if out[e]}


def _play_order(maps, links, start):
    """{map: level changes from the game's first map}, by its level changes (bmab.graph.build's links); a part no level change reaches
    starts again after the rest."""
    next_up = defaultdict(set)
    for link in links:
        a, b = link["maps"]
        next_up[a].add(b)
        next_up[b].add(a)
    depth, first = {}, 0
    for root in [start, *sorted(maps)]:
        if root not in maps or root in depth:
            continue
        depth[root], todo = first, [root]
        while todo:
            m = todo.pop(0)
            for o in sorted(next_up[m]):
                if o in maps and o not in depth:
                    depth[o] = depth[m] + 1
                    todo.append(o)
        first = max(depth.values()) + 1
    return depth


def _game(g):
    """One game's part of its page, read from its maps' entities alone: {maps, levels (the levels next to each), play (the order they
    are played in), links (its level changes), landmarks (each map's)}. Kept in the cache per version of its map files and of what
    people say about its level changes, as reading them all takes a while."""
    game, decided = Game(sources.DATA / g), redlines.decided_links(g)
    files = [[n, game.path(n).stat().st_mtime_ns, game.path(n).stat().st_size] for n in game.map_names]
    path = VIEWS / CODE / "pages" / f"{g}.{hashlib.sha256(json.dumps([files, decided], sort_keys=True).encode()).hexdigest()[:16]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    outlines = {n: bsp.outline(game.path(n), n) for n in game.map_names}
    marks = {n: graph._landmarks(b) for n, b in outlines.items()}
    links = graph.build(outlines, None, decided)["links"]
    out = {"maps": game.map_names, "levels": levels.joined(marks, links, decided), "play": _play_order(outlines, links, game.start),
           "links": [{"maps": link["maps"], "kind": link["kind"], "landmark": link["landmark"],
                     "delta": [float(x) for x in marks[link["maps"][0]][link["landmark"]] - marks[link["maps"][1]][link["landmark"]]]} for link in links],
           "landmarks": {n: {m: [float(x) for x in at] for m, at in ms.items()} for n, ms in marks.items()}}
    for old in path.parent.glob(f"{g}.*.json"):
        old.unlink()
    _write(path, json.dumps(out).encode())
    return out


def _shared_links(games):
    """"same" links between the maps of these games: rooms bmab shared found built into two of them (bmab.shared.stored), the
    biggest for each pair of maps."""
    best = {}
    for g in games:
        for other in games:
            for p in shared.stored(g, other):
                pair = ((other, p["with"]), (g, p["map"]))
                if p["class"] == "room" and pair[0] != pair[1] and (pair not in best or p["matched_m2"] > best[pair][1]):
                    best[pair] = ({"maps": [list(pair[0]), list(pair[1])], "kind": "same", "delta": [-float(x) for x in p["translation"]],
                                   "landmark": f"shared room, {p['matched_m2']} m2"}, p["matched_m2"])
    return [link for link, _m2 in best.values()]


def _page(games):
    """One edition's page from its games ([(game, its title)])."""
    page = {"games": [g for g, _t in games], "maps": [], "levels": {}, "play": {}, "links": [], "landmarks": {}}
    for g, title in games:
        part = _game(g)
        page["maps"] += [{"name": n, "source": g, "game": title} for n in part["maps"]]
        page["levels"].update({(g, n): others for n, others in part["levels"].items()})
        page["play"].update({key(g, n): depth for n, depth in part["play"].items()})
        page["links"] += [{**link, "maps": [[g, m] for m in link["maps"]]} for link in part["links"]]
        page["landmarks"].update({(g, n): marks for n, marks in part["landmarks"].items()})
    page["links"] += _shared_links(page["games"])
    return page


def setup():
    """The pages, from the games here and what the register says of them; drops the views of other versions of bmab. Returns what
    bmab.editing is set up with: the levels next to each map ({(game, map): [{map, offset}]}) and every (game, map) a claim may name."""
    for old in VIEWS.iterdir() if VIEWS.is_dir() else ():
        if old.name != CODE:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    now = {e: _page(games) for e, games in _games().items()}
    _pages.clear()
    _pages.update(now)
    return ({k: v for q in _pages.values() for k, v in q["levels"].items()},
            {(m["source"], m["name"]) for q in _pages.values() for m in q["maps"]})


def editions():
    return list(_pages)


def meta(edition):
    """What the edition's page is shown with: its maps (each with how many of some kinds of claim it has now), the levels next to
    each, the play order and the area kinds."""
    page = _pages[edition]
    counts = {k: Counter(r["kind"] for r in rows) for k, rows in redlines.every_map().items()}
    return {"area_kinds": {k: {"label": v[0], "colour": v[1], "dropped": v[2]} for k, v in AREA_KINDS.items()}, "cube_limit": CUBE_LIMIT,
            "maps": [{**m, **{kind + "s": counts.get((m["source"], m["name"]), Counter())[kind] for kind in ("connector", "fake_door", "fake_out")}} for m in page["maps"]],
            "levels": {key(*k): v for k, v in page["levels"].items()}, "play": page["play"]}


def maps(edition):
    """The maps the edition's page lists: [{name, source, game (its title)}]."""
    return _pages[edition]["maps"]


def elsewhere(edition):
    """The maps of the other editions' pages: [{source, name, game}]."""
    return [m for e, q in _pages.items() if e != edition for m in q["maps"]]


def listed(game, name, edition=None):
    """Whether a page (that edition's, or any) lists the map."""
    return any(m["source"] == game and m["name"] == name for e, q in _pages.items() if edition in (None, e) for m in q["maps"])


def _sha(game, name):
    return Game(sources.DATA / game).version(name)["sha256"][:16]


def _files(game, name):
    """Where a map's view, models and frame are kept, for its file as it is here."""
    return {k: VIEWS / CODE / game / f"{name}.{_sha(game, name)}.{k}" for k in ("js", "models.js", "frame.json")}


def _volume_file(game, name):
    """Where a map's volume is kept, for its file as it is here and what people say about its areas (their whole answer drops parts of it)."""
    said = hashlib.sha256(json.dumps(redlines.said(game, name, "area"), sort_keys=True).encode()).hexdigest()[:16]
    return VIEWS / CODE / game / f"{name}.{_sha(game, name)}.{said}.volume.npy"


def _drop_others(path, pattern, keep):
    """Remove what is kept of the same map as path that matches pattern (a glob after its name) and whose name doesn't start with keep."""
    name = path.name.split(".", 1)[0]
    for old in path.parent.glob(f"{name}.{pattern}"):
        if old.name.split(".", 1)[0] == name and not old.name.startswith(keep) and not old.name.endswith(".part"):
            old.unlink(missing_ok=True)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)


def _read(game, name, said):
    b = Game(sources.DATA / game).read(name)
    pieces.analyse(b, said)
    if isinstance(b.vis, bsp.SourceVis):
        b.vis.release()
    return b


def _save_volume(b, path):
    out = io.BytesIO()
    np.save(out, solve.volume(b)[0])
    _write(path, out.getvalue())


def _build(job):
    """In the build process: a map's view, models and frame, and its volume while it is read."""
    game, name, said, files, volume = job
    b = _read(game, name, said)
    _save_volume(b, volume)
    data, models, frame = view.map_view(game, name, b)
    if models is not None:
        _write(files["models.js"], f"modelsLoaded({json.dumps(models)});\n".encode())
    _write(files["frame.json"], json.dumps(frame).encode())
    _write(files["js"], f"mapLoaded({json.dumps(data)});\n".encode())


def _measure(job):
    """In the build process: a map's volume."""
    game, name, said, volume = job
    _save_volume(_read(game, name, said), volume)


def _run(work, job, what):
    """work(job) in the build process, one at a time, once for the same what however many ask; waits for it. A map that can't be
    built raises ValueError."""
    with _lock:
        if what not in _running:
            if not _pool:
                _pool.append(concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=jobs.with_the_app,
                                                                    max_tasks_per_child=BUILDS_A_PROCESS))
            _running[what] = (_pool[0].submit(work, job), _pool[0])
        running, executor = _running[what]
    try:
        running.result()
    except concurrent.futures.process.BrokenProcessPool:
        with _lock:
            if _pool and _pool[0] is executor:
                _pool.clear()  # it ran out of memory, say: the next build starts a new process
        raise ValueError(f"building {what} stopped its process") from None
    finally:
        with _lock:
            if _running.get(what, (None,))[0] is running:
                del _running[what]


def view_file(game, name, what="js"):
    """The file of a map's view (js) or its models (models.js), built first if it isn't kept yet. None: it places no models."""
    files = _files(game, name)
    if not files["js"].exists():
        _run(_build, (game, name, redlines.said(game, name, "area"), files, _volume_file(game, name)), key(game, name))
        _drop_others(files["js"], "*", f"{name}.{_sha(game, name)}.")  # what was kept for the map's file before it changed
    return files[what] if files[what].exists() else None


def _volume(game, name):
    path = _volume_file(game, name)
    if not path.exists():
        _run(_measure, (game, name, redlines.said(game, name, "area"), path), f"the volume of {key(game, name)}")
        _drop_others(path, "*.volume.npy", path.name)  # for what people said of its areas before, or for its file before it changed
    return np.load(path)


def _cells_of(keys):
    base = 1 << 20
    return np.stack([keys // (base * base), (keys // base) % base, keys % base], -1) - (base >> 1)


def neighbours(edition, game, name):
    """How a map meets each map its page links to it, placed by the link alone: [{map, source, game (its title), kind, landmark,
    offset (units, added to its coordinates), at, at_local, agreed and conflict cells}], one per other map."""
    page = _pages[edition]
    view_file(game, name)
    frame = json.loads(_files(game, name)["frame.json"].read_text(encoding="utf-8"))
    centre, quant = np.asarray(frame["centre"]), frame["quant"]
    titles = {m["source"]: m["game"] for m in page["maps"]}

    def local(p):
        return [round(float(x), 1) for x in (np.asarray(p, float) - centre) / quant]

    def sample(keys):
        return [local((c + 0.5) * solve.CELL) for c in _cells_of(keys[:: max(1, len(keys) // CUBE_LIMIT)])]
    here, out, seen, mine = [game, name], [], set(), _volume(game, name)
    for link in page["links"]:
        a, b = link["maps"]
        if here not in (a, b) or a == b:
            continue
        other = b if here == a else a
        if tuple(other) in seen:
            continue
        seen.add(tuple(other))
        offset = np.asarray(link["delta"], float) * (1 if here == a else -1)
        ka, kb = mine, _volume(*other)
        rel = np.round(offset / solve.CELL).astype(np.int64)
        both = np.intersect1d(ka, kb + solve._encode(rel) - solve._encode(np.zeros(3, np.int64)), assume_unique=True)
        agreed = solve._agreeing(ka, kb, rel) if len(both) else both
        at = page["landmarks"][game, name].get(link["landmark"])
        if at is None:
            at = (_cells_of(both).mean(0) + 0.5) * solve.CELL if len(both) else frame["middle"]
        out.append({"map": other[1], "source": other[0], "game": titles[other[0]], "kind": link["kind"], "landmark": str(link["landmark"]),
                    "offset": [round(float(x), 2) for x in offset], "at": [round(float(x)) for x in at], "at_local": local(at),
                    "agreed": sample(agreed), "conflict": sample(np.setdiff1d(both, agreed, assume_unique=True))})
    return out
