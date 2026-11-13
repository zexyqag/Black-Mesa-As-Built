"""Editing the redlines with the games at hand: a claim placed by hand is completed from its map as detection would (what is behind a
door, how the player gets through), checked against it (a landmark that exists, a map that is known), and kept in step everywhere
it belongs (the levels holding the same place, the other map of a claim about two, a part's openings). Also what the editor
works out from a map to show (a part alone, the doors that open, the signs, the level-change triggers, the cuts), running the
detectors over a map again and working out register entries from a game (bmab.generate), as the app's one job (bmab.jobs).

A map is (game, name). It reads maps from the imported games and is set up with what it can't work out alone (setup); it knows
nothing of pages or HTTP.
"""
import base64
import collections

import numpy as np

from . import bsp, claims, connectors, doors, evaluate, fake_outs, generate, graph, jobs, levels, memory, parts, pieces, redlines, register, signs, solve, sources, styles, view
from .game import Game

LAST_MAPS = 3
BOXES = ("at", "facing", "size", "extra_boxes")  # what says where a part's boxes are: changing any finds its openings again
_levels = {}  # {(game, map): [{map, offset}]}: the levels of the same game next to each one (bmab.levels)
_known = set()  # every (game, map) a claim may name
_read = collections.OrderedDict()  # the last maps read
_fired_by, _signs = {}, {}


def setup(next_levels, known_maps):
    """What it works with: the levels next to each map ({(game, map): [{map, offset}]}) and every (game, map) a claim may name.
    Forgets what it read before."""
    _levels.clear()
    _levels.update(next_levels)
    _known.clear()
    _known.update(known_maps)
    _read.clear()
    _fired_by.clear()
    _signs.clear()


def _game(game):
    if not (sources.DATA / game).is_dir():
        raise ValueError(f"{game} isn't imported here")
    return Game(sources.DATA / game)


def _bsp(game, name):
    """A map as read, its pieces analysed with what people say about its areas (so every detector sees them)."""
    key = (game, name)
    if key not in _read:
        g = _game(game)
        if name not in g.map_names:
            raise ValueError(f"{game} has no map {name}")
        b = g.read(name)
        if b.vis is None:
            raise ValueError(f"can't measure {name}: it has no visibility data")
        pieces.analyse(b, redlines.said(game, name, "area"))
        _read[key] = b
        while len(_read) > LAST_MAPS:
            _read.popitem(last=False)
    _read.move_to_end(key)
    return _read[key]


def _version(game, name):
    """The version claim of a map as this computer has it."""
    return {"id": "version", "kind": "version", **_game(game).version(name)}


def _write(game, name, rows):
    """Write a map's claims, saying which file of the map they are about when nothing says so yet and the game is here; a version
    alone says nothing, so a map left with only that has no file."""
    if all(r["kind"] == "version" for r in rows):
        rows = []
    elif not any(r["kind"] == "version" for r in rows) and (sources.DATA / game).is_dir():
        rows = rows + [_version(game, name)]
    redlines.write_map(game, name, rows)


def _held(maps):
    """Refuse an edit while one of the maps it may write has claims made on another file of the map than the one here: they may
    not fit this one. Checking them (checked) takes the hold away."""
    for game, name in dict.fromkeys(maps):
        said = next((r for r in redlines.claims(game, name) if r["kind"] == "version"), None)
        if said is not None and (sources.DATA / game).is_dir() and said["sha256"] != _game(game).version(name)["sha256"]:
            raise ValueError(f"{name}'s claims were made on another version of the map than yours: check them first, then say they fit")


def _touched(game, name, line):
    """The maps an edit of this claim may write: its own, the levels holding the same place, the other map of a claim about two."""
    return [(game, name), *(((game, x["map"]) for x in _levels.get((game, name), ())) if line["kind"] in levels.MIRRORED else ()),
            *([(line["game"], line["map"])] if line["kind"] in claims.PAIRED else [])]


def version(game, name):
    """Which file of a map its claims are about, against the one here: {here, claims (None when nothing says), same}."""
    here = _game(game).version(name)
    said = next((r for r in redlines.claims(game, name) if r["kind"] == "version"), None)
    return {"here": here, "claims": None if said is None else {"file": said["file"], "sha256": said["sha256"]},
            "same": said is None or said["sha256"] == here["sha256"]}


def checked(game, name):
    """Someone has checked a map's claims against the file here: they are about this file now."""
    with redlines.lock:
        rows = redlines.claims(game, name)
        if rows:
            _write(game, name, [r for r in rows if r["kind"] != "version"])
    return version(game, name)


def _measured(game, name, line):
    """A claim placed by hand comes without what the tool works out for it: what lies behind a fake or ought door, how the player
    gets through a connector or an opening or goes on along a fake out's passage (its way), what it looks like. Guess those as
    detection does; people correct them."""
    if line["kind"] in claims.MAP_KINDS:
        return line
    b = _bsp(game, name)
    if line["kind"] in ("fake_door", "ought_door") and "behind" not in line:
        line = {**line, "behind": doors.status({"behind": doors.measure(b, line["at"], line["back"])})}
    if line["kind"] in ("connector", "fake_out") and "way" not in line:
        line = {**line, "way": (connectors.way_of if line["kind"] == "connector" else fake_outs.way)(b, line)}
    if line["kind"] == "opening" and "way" not in line:
        rows = redlines.claims(game, name)
        part = next((r for r in rows if r["kind"] == "part" and r["id"] == line["part"]), None)
        if part is None:
            raise ValueError("an opening belongs to a part of this map")
        line = {**line, "way": parts.way(b, part, line, rows, _fired(game, b))}
    if styles.needs(line) and "style" not in line and (style := styles.guess(b, line, map_signs(game, name))):
        line = {**line, "style": style}
    return line


def _in_step(game, name, old, new):
    """Do in every level holding the same place what was just done here (new None: deleted)."""
    if (old or new)["kind"] not in levels.MIRRORED:
        return
    here = redlines.claims(game, name)
    for x in _levels.get((game, name), ()):
        other, offset = x["map"], np.array(x["offset"], float)
        rows = redlines.claims(game, other)
        was = levels.twin(levels.moved(old, -offset), rows) if old else None
        rest = [r for r in rows if was is None or r["id"] != was["id"]]
        if new is None:
            if was is not None:
                _write(game, other, [r for r in rest if r.get("part") != was["id"]])  # a part's openings go with it
            continue
        if was is None and not levels.same_place(_bsp(game, name), _bsp(game, other), new, offset):
            continue
        line = levels.carried(new, offset, here, rows)
        line = {"id": was["id"] if was else claims.claim_id(line["kind"], line["at"], {r["id"] for r in rows}), **line}
        _write(game, other, _with_openings(game, other, line, rest + [line]) if _reshaped(was, line) else rest + [line])


def save(game, name, line):
    """Add a claim, or replace the one with its id, in this map and wherever else it belongs. Returns it."""
    line = _measured(game, name, claims.check(line))
    if line["kind"] == "version":
        raise ValueError("which file a map's claims are about is said by checking them")
    if line["kind"] == "fake_out" and levels.at_change(levels.changes(_bsp(game, name)), line):
        raise ValueError("a level change is there: it already says where this leads")
    with redlines.lock:
        _held(_touched(game, name, line))
        rows = redlines.claims(game, name)
        old = next((r for r in rows if r["id"] == line.get("id")), None)
        if old is None and line["kind"] in claims.PAIRED:  # one per level change, and one per other map for a same place: whichever way it was asked for
            old = next((r for r in rows if claims.same_pair(r, line)), None)
            line = {"id": old["id"], **{k: v for k, v in line.items() if k != "id"}} if old else line
        if old is None and line["kind"] in claims.SINGLE:  # one per map: a new one replaces it
            old = next((r for r in rows if r["kind"] == line["kind"]), None)
            line = {"id": old["id"], **{k: v for k, v in line.items() if k != "id"}} if old else line
        if old == line:
            return line
        if old is None:
            key = line["place"] if line["kind"] == "place" else line.get("at")
            line = {"id": claims.claim_id(line["kind"], key, {r["id"] for r in rows}), **{k: v for k, v in line.items() if k != "id"}}
        if line["kind"] == "opening" and not any(r["kind"] == "part" and r["id"] == line["part"] for r in rows):
            raise ValueError("an opening belongs to a part of this map")
        if line["kind"] == "level_change" and (line["game"] != game or line["landmark"] not in graph._landmarks(_bsp(game, name))):
            raise ValueError(f"{name} has no level change to {line['game']} {line['map']} at landmark {line['landmark']}")
        if line["kind"] == "same_place" and (line["game"], line["map"]) not in _known:
            raise ValueError(f"{line['game']} {line['map']} is a map nobody here knows")
        rows = [r for r in rows if r["id"] != line["id"]] + [line]
        _write(game, name, _with_openings(game, name, line, rows) if _reshaped(old, line) else rows)
        _in_step(game, name, old, line)
        if line["kind"] in claims.PAIRED:
            _paired(game, name, line)
    return line


def delete(game, name, claim):
    with redlines.lock:
        rows = redlines.claims(game, name)
        old = next((r for r in rows if r["id"] == claim), None)
        if old is None:
            raise ValueError("no such claim")
        if old["kind"] == "version":
            raise ValueError("which file a map's claims are about goes with its last claim")
        _held(_touched(game, name, old))
        _write(game, name, [r for r in rows if r["id"] != claim and r.get("part") != claim])  # a part's openings go with it
        _in_step(game, name, old, None)
        if old["kind"] in claims.PAIRED:
            _paired(game, name, old, gone=True)
    return {"id": claim}


def discard(game, name):
    """Put a map back as it was last committed, with the levels next to it that changed too (their lines keep in step). Its claims."""
    changed = set(redlines.changed())
    redlines.discard([(game, name), *((game, x["map"]) for x in _levels.get((game, name), ()) if (game, x["map"]) in changed)])
    return redlines.claims(game, name)


def _counterpart(game, name, line):
    """A claim about two maps as the other one (line["game"], line["map"]) holds it, in its own coordinates; None if it can't be
    said there (a level change into a level of a game not imported here)."""
    if line["kind"] == "same_place":
        return {"kind": "same_place", "at": line["there"], "game": game, "map": name, "there": line["at"], **({"changed": True} if line.get("changed") else {})}
    try:
        at = graph._landmarks(_bsp(line["game"], line["map"])).get(line["landmark"])
    except ValueError:
        return None
    return None if at is None else {"kind": "level_change", "at": [round(float(v)) for v in at], "game": game, "map": name, "landmark": line["landmark"],
                                    **{k: line[k] for k in ("loose", "by") if k in line}}


def _paired(game, name, line, gone=False):
    """Have the other map of a claim about two (claims.PAIRED) say the same (line deleted: gone), in its own file."""
    other_game, other, rows = line["game"], line["map"], redlines.claims(line["game"], line["map"])
    mine = {**line, "game": game, "map": name}
    was, rest = next((r for r in rows if claims.same_pair(r, mine)), None), [r for r in rows if not claims.same_pair(r, mine)]
    if not gone:
        twin = _counterpart(game, name, line)
        if twin is None:
            return
        rest.append({"id": was["id"] if was else claims.claim_id(twin["kind"], twin["at"], {r["id"] for r in rest}), **twin})
    if was is not None or not gone:
        _write(other_game, other, rest)


def _reshaped(old, line):
    """Whether a claim is a part whose boxes are new or have changed."""
    return line["kind"] == "part" and (old is None or any(old.get(k) != line.get(k) for k in BOXES))


def _with_openings(game, name, part, rows):
    """A map's claims (rows, the part among them) with the part's openings found again (bmab.parts.openings). One found where one
    of its openings was (levels.twin) keeps that one's id and way, so a way someone set stays; its other openings go."""
    b = _bsp(game, name)
    mine = [r for r in rows if r["kind"] == "opening" and r["part"] == part["id"]]
    rest = [r for r in rows if not (r["kind"] == "opening" and r["part"] == part["id"])]
    taken, out = {r["id"] for r in rest}, []
    for o in parts.openings(b, part, rest, _fired(game, b), lambda m: _next_level(game, b, m)):
        was = levels.twin(o, mine)
        if was is not None:
            mine.remove(was)
        i = was["id"] if was else claims.claim_id("opening", o["at"], taken)
        taken.add(i)
        out.append({"id": i, **o, **({"way": was["way"]} if was else {})})
    return rest + out


def find_openings(game, name, claim):
    """Find a part's openings again, as when its boxes change (a part marked before openings were, or the claims around it changed)."""
    with redlines.lock:
        rows = redlines.claims(game, name)
        part = next((r for r in rows if r["id"] == claim and r["kind"] == "part"), None)
        if part is None:
            raise ValueError("no such part")
        _held([(game, name)])
        rows = _with_openings(game, name, part, rows)
        _write(game, name, rows)
    return {"openings": sum(r.get("part") == claim for r in rows)}


def _runs(b, keep):
    """Runs [first, count] of the vertices of the faces keep(index, face) picks in the page's mesh (the map's drawn faces in order,
    as bmab.view packs them), and the mesh's vertex count to check them against."""
    runs, at = [], 0
    for i, f, _v, _n in view.drawn(b):
        if keep(i, f):
            if runs and runs[-1][0] + runs[-1][1] == at:
                runs[-1][1] += len(f.verts)
            else:
                runs.append([at, len(f.verts)])
        at += len(f.verts)
    return runs, at


def alone(game, name, claim):
    """A part as the page shows it alone (bmab.parts): runs of its faces' vertices in the page's mesh (_runs), the mesh's vertex
    count, its faces, its space in m3, the signs it holds and its openings (the claims, each with what stands in it: parts.named)."""
    rows = redlines.claims(game, name)
    line = next((r for r in rows if r["id"] == claim and r["kind"] == "part"), None)
    if line is None:
        raise ValueError("no such part")
    b = _bsp(game, name)
    keep, m3 = parts.faces(b, line)
    kept = set(keep)
    runs, at = _runs(b, lambda i, _f: i in kept)
    middle, axes, boxes = parts.frame(line)
    every = map_signs(game, name)
    said = lambda s: {"evidence": s["evidence"], "transcription": s["transcription"], "places": sorted({p for c in s["claims"] for p in (c[0], c[2] if len(c) > 2 and isinstance(c[2], str) else None) if p})}  # noqa: E731
    held = [s for s in every if any((np.abs((np.array(s["at"], float) - middle) @ axes.T - c) <= h + 1).all() for c, h in boxes)]
    mine = [r for r in rows if r["kind"] == "opening" and r["part"] == claim]
    found = sorted(zip(mine, parts.named(b, line, mine, rows, every, _fired(game, b))), key=lambda f: (parts.KINDS.index(f[1]["type"]), f[0]["id"]))
    return {"runs": runs, "vertices": at, "faces": len(keep), "m3": round(m3), "signs": [said(s) for s in held],
            "openings": [{**o, **n, "signs": [said(every[k]) for k in n["signs"]]} for o, n in found]}


def _next_level(game, b, m):
    """(the bsp, offset) of the level m of the game a level change of level b leads to, if it is there: a point there plus offset is
    a point here, by the landmark they share. None otherwise."""
    landmark = next((lm for to, lm, _lo, _hi in levels.triggers(b) if to == m), None)
    try:
        there = _bsp(game, m)
    except ValueError:
        return None
    here, theirs = graph._landmarks(b), graph._landmarks(there)
    return (there, here[landmark] - theirs[landmark]) if landmark in here and landmark in theirs else None


def opening(game, name):
    """The map's doors that open (bmab.doors.brush_doors), for the page to leave out to see through them: runs of their faces'
    vertices in the page's mesh (_runs) and the mesh's vertex count."""
    b = _bsp(game, name)
    opens = {m for m, o in doors.brush_doors(b, _fired(game, b)).items() if o}
    runs, at = _runs(b, lambda _i, f: f.model in opens)
    return {"runs": runs, "vertices": at}


def _fired(game, b):
    """Names a level change fires in the game's maps (doors.fired_names), read once; only GoldSrc's doors heed them."""
    if isinstance(b.vis, bsp.SourceVis):
        return ()
    if game not in _fired_by:
        g = _game(game)
        _fired_by[game] = doors.fired_names([bsp.entities(g.path(n)) for n in g.map_names])
    return _fired_by[game]


def map_signs(game, name):
    """A map's signs (bmab.signs, as the register knows them) for the page: where each is, its picture's axes, what it says (its
    transcription), its facts' claims and its arrows' ways in the map."""
    if (game, name) not in _signs:
        b = _bsp(game, name)
        raw = _game(game).raw(name) if isinstance(b.vis, bsp.SourceVis) else None
        tidy = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
        _signs[game, name] = [{"evidence": s["evidence"], "at": [round(float(x)) for x in s["at"]], "normal": tidy(s["normal"]), "right": tidy(s["right"]),
                               "up": tidy(s["up"]), "size": [round(x) for x in s["size"]], "sides": s.get("sides", 1), "door": s["door"],
                               "transcription": s["transcription"], "claims": s["claims"],
                               "points": [{"to": w["to"], "dir": tidy(w["dir"])} for w in s["points"]]}
                              for s in signs.described(b, game, signs.catalogue(redlines.register()), raw, _fired(game, b))]
    return _signs[game, name]


def known_places(game, name):
    """The places a place claim can name: {known: every place of the register a map can be in (register.MAPPABLE) or a place claim
    names, named: the ones of those the game's evidence seen or heard in this map names, the most named first, parents: what holds
    each}."""
    from . import places
    reg = redlines.register()
    tree = places.build(reg, {})
    claimed = {r["place"] for rows in redlines.every_map().values() for r in rows if r["kind"] == "place"}
    return {"known": sorted({p for p, n in tree.items() if n["type"] in register.MAPPABLE} | claimed), "named": places.named_in_map(reg, tree, game, name),
            "parents": {p: sorted(n["parents"]) for p, n in tree.items() if n["parents"]}}


def cut(game, name):
    """How the map is cut by its claims now: {cuts: {id: rooms and limit}, shapes: {id: what each connector's passage is like
    (connectors.whole, connectors.slab and connectors.proof)}, left_out: ids that part nothing, cell, cells, room}."""
    b = _bsp(game, name)
    said = {r["id"]: r for r in redlines.claims(game, name) if r["kind"] == "connector"}
    conn, cells, room = connectors.claimed(b, list(said.values()), game)

    def shape(r):
        proof = connectors.proof(b, r)
        return {"whole": connectors.whole(b, r["at"], r["along"], r["outline"]), "slab": connectors.slab(b, r["at"], r["along"], r["outline"]),
                "proof": {"text": proof["text"], "box": proof["box"] and [np.asarray(x, float).tolist() for x in proof["box"]]}}
    pack = lambda a, kind: base64.b64encode(np.asarray(a).astype(kind).tobytes()).decode()  # noqa: E731
    return {"cuts": {c["id"]: {"rooms": c["rooms"], "limit_m": c["limit_m"]} for c in conn}, "shapes": {i: shape(r) for i, r in said.items()},
            "left_out": sorted(set(said) - {c["id"] for c in conn}),
            "cell": solve.CELL, "cells": None if cells is None else pack(cells, np.int16), "room": None if room is None else pack(room, np.uint16)}


def level_changes(game, name):
    """The level's level-change triggers, for the page: [{map, landmark, lo, hi}] (units)."""
    return [{"map": m, "landmark": mark, "lo": [round(float(v)) for v in lo], "hi": [round(float(v)) for v in hi]} for m, mark, lo, hi in levels.triggers(_bsp(game, name))]


def again(game, name):
    """Start running the detectors over a map again; evaluated says how it goes."""
    if name not in _game(game).map_names:
        raise ValueError(f"{game} has no map {name}")
    limit = max(2.0, memory.available_gb() - memory.RESERVE_GB)
    jobs.start(("evaluate", game, name), f"evaluating {name} again", evaluate.again, (game, name, _levels.get((game, name), []), limit))
    return evaluated(game, name)


def evaluated(game, name):
    """How the map's evaluation again goes (_state), and when done its changes: bmab.evaluate.proposals, against the claims as they
    are now; a same place with a map nobody here knows can't be saved, so isn't offered."""
    state, rows = jobs.state(("evaluate", game, name))
    if rows is not None:
        changes = evaluate.proposals(rows, redlines.claims(game, name), redlines.history(f"maps/{game}/{name}.jsonl"))
        state["changes"] = [c for c in changes if (c["then"] or c["line"])["kind"] != "same_place" or ((c["then"] or c["line"])["game"], (c["then"] or c["line"])["map"]) in _known]
    return state


def forget(game, name):
    """Drop what the map's evaluation again said, once it is done."""
    jobs.forget(("evaluate", game, name))
    return evaluated(game, name)


_work = {}  # the generator and game last asked for, so the Register page can ask how it goes without saying


def work_out(what, game):
    """Start working out register entries from a game's files (bmab.generate.GENERATORS); worked_out says how it goes."""
    if what not in generate.GENERATORS:
        raise ValueError(f"no generator {what}")
    _game(game)
    jobs.start(("generate", what, game), f"working out {generate.GENERATORS[what]['title']} in {game}", generate.run, (what, game, redlines.tier(game), redlines.evidence(game)))
    _work.update(what=what, game=game)
    return worked_out()


def worked_out():
    """How working out register entries goes (_state, with what and game), and when done its changes against the register as it
    is now (bmab.generate.differences): table, then [{change, line, then, keep, why, says (a new line's words, if any)}]."""
    if not _work:
        return jobs.state(("generate",))[0]
    task = ("generate", _work["what"], _work["game"])
    state, result = jobs.state(task)
    state.update(what=_work["what"], game=_work["game"])
    if result is not None:
        stored = {e["id"]: e for e in redlines.evidence(_work["game"])}
        state.update(table=generate.GENERATORS[_work["what"]]["table"],
                     changes=generate.proposals(_work["what"], result, stored, redlines.history(f"evidence/{_work['game']}.jsonl")))
    return state


def forget_work():
    """Drop what working out said, once it is done."""
    if _work:
        jobs.forget(("generate", _work["what"], _work["game"]))
        _work.clear()
    return worked_out()
