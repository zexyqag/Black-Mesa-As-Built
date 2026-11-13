"""A map's baseline for the redlines: every claim the detectors make about it, one line each, positions and verdicts only.

Areas that aren't the facility, fake doors (doors the player can't use), fake outs (where a map looks like it goes on), and the
connectors it can be cut at; and the openings of the parts people have marked. A claim that isn't there isn't made: nothing says what a map doesn't have. An id comes from what the
claim is and where, so evaluating a map again changes only the lines that really differ. A door that works in the next level and a
passage that level already shows are no claims at all, and neither is a fake out at a level change.
"""
from collections import defaultdict

import numpy as np

from . import bsp, connectors, doors, fake_outs, levels, memory, parts, pieces, redlines, shared, signs, sources, styles
from .claims import MAP_KINDS, PAIRED, claim_id, same_pair
from .game import Game

FAKE_OUT_FIELDS = ("seen_from", "m3", "m2")
CONNECTOR_FIELDS = ("along", "path", "outline")
BY_PEOPLE = {("fake_out", "leads_on")}  # claims only people make: the detectors never find them, so never offer to take them away
PEOPLE_KINDS = {"ought_door", "part", "level_change", "same_place", *MAP_KINDS}  # kinds the detectors find none or only some of (same places: rooms built alike), for the same reason
WHOLE = {"connector"}  # claims whose fields hang together (a connector's outline, path and length follow from where it is): never merged field by field


def _tidy(v):
    """Numbers as a file should keep them: whole where they are, three decimals where they aren't, never minus zero."""
    if isinstance(v, (list, tuple)):
        return [_tidy(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    v = round(float(v), 3)
    return int(v) if v == int(v) else v


def _identified(rows):
    """Each row with its id first."""
    seen, out = set(), []
    for r in rows:
        i = claim_id(r["kind"], r["at"], seen)
        seen.add(i)
        out.append({"id": i, **r})
    return out


def claims(game, name, b, fired, found, known, around=None, now=None, areas=None):
    """What the detectors say about one map (b, the game's map name as read), as lines for the redlines. around: {map: (bsp, offset,
    its doors nobody can use)} of the levels next to it; known: the register's signs (bmab.signs.catalogue); now: the map's claims as
    they are, whose parts get their openings found (none without); areas: people's whole answer about its areas
    (bmab.redlines.said), None if nobody has checked them all."""
    rows, around = [], around or {}
    for p in pieces.analyse(b, areas):
        if p["found"]["dropped"]:
            rows.append({"kind": "area", "type": p["found"]["kind"], "at": _tidy(p["at"])})
    found_signs = signs.described(b, None, known, fired=fired)
    for d in doors.find(b, fired):
        row = {"kind": "fake_door", "at": _tidy(d["at"]), "back": _tidy(d["back"]), "width": _tidy(d["width"]), "height": _tidy(d["height"]), "behind": doors.status(d),
               **({"sides": 2} if d["sides"] == 2 else {})}
        if not any(levels.door_works(o, np.subtract(row["at"], off), cant) for o, off, cant in around.values()):
            rows.append(_styled(b, row, found_signs))
    if isinstance(b.vis, bsp.SourceVis):
        b.vis.release()
    surfaces, changes = connectors.planes(b), levels.changes(b)
    for h in found.get(name, []):
        end = fake_outs.end_surface(b, surfaces, h)
        if end is None:
            continue
        row = {"kind": "fake_out", "type": h["type"], "at": end[0], "reaches": end[1], **{k: _tidy(h[k]) for k in FAKE_OUT_FIELDS if k in h}}
        if not levels.at_change(changes, row) and not any(levels.answered(b, o, row, np.asarray(off, float)) for o, off, _ in around.values()):
            rows.append(_styled(b, {**row, "way": fake_outs.way(b, row)}))
    beside = lambda m: (around[m][0], np.asarray(around[m][1], float)) if m in around else None  # noqa: E731
    for part in [r for r in now or () if r["kind"] == "part"]:
        rows += [{k: _tidy(v) for k, v in o.items()} for o in parts.openings(b, part, now, fired, beside)]
    return _identified(rows) + connector_rows(connectors._cached(game, b)[0])


def _styled(b, row, near=()):
    style = styles.guess(b, row, near) if styles.needs(row) else None
    return {**row, "style": style} if style else row


def connector_rows(found):
    """A map's connectors as bmab.connectors finds them, as lines for the redlines."""
    cuts = _identified([{"kind": "connector", "way": c["way"], "at": _tidy(c["middle"]), **{k: _tidy(c[k]) for k in CONNECTOR_FIELDS}} for c in found])
    for row, c in zip(cuts, found):
        if c["stretches_with"]:
            row["stretches_with"] = [cuts[k]["id"] for k in c["stretches_with"]]
    return cuts


def detected(games, todo, found):
    """What the detectors claim about each map in todo, one at a time: (map, lines). games: {map: (source, bsp)}, every map, because a door
    may be opened from another map and the next level may hold the same place."""
    fired = {g: doors.fired_names([b.entities for h, b in games.values() if h == g]) for g, _b in games.values()}
    known = signs.catalogue(redlines.register())
    groups, nearby, cant = defaultdict(dict), {}, {}
    for n, (g, b) in games.items():
        groups[g][n] = b
    for g, maps in groups.items():
        nearby.update(levels.next_to(maps, g))

    def cant_use(n):
        if n not in cant:
            g, b = games[n]
            pieces.analyse(b, redlines.said(g, n, "area"))
            cant[n] = [d["at"] for d in doors.find(b, fired[g])]
        return cant[n]
    for n in todo:
        g, b = games[n]
        yield n, claims(g, n, b, fired[g], found, known, {x["map"]: (games[x["map"]][1], x["offset"], cant_use(x["map"])) for x in nearby.get(n, ())},
                        areas=redlines.said(g, n, "area"))


def run(games, todo, found, log=print):
    """Write the lines of the maps in todo, with which file of each map they are about."""
    total = 0
    for k, (n, rows) in enumerate(detected(games, todo, found), 1):
        g = games[n][0]
        redlines.write_map(g, n, rows + [{"id": "version", "kind": "version", **Game(sources.DATA / g).version(n)}] if rows else [])
        total += len(rows)
        log(f"{k}/{len(todo)} {n}: {len(rows)}")
    return total


def again(job):
    """What the detectors say about one map now, in a process of its own: its lines, or None if it needed more memory than limit_gb.
    job: (game, the map, [{map, offset}] the levels of the game next to it, limit_gb). With them the same places it is built with
    (bmab.shared.places, from what bmab shared found on this computer)."""
    game, name, around, limit_gb = job
    memory.limit(limit_gb)
    try:
        g = Game(sources.DATA / game)
        fired = doors.fired_names([bsp.entities(g.path(n)) for n in g.map_names])

        def beside(x):
            b = g.read(x["map"])
            pieces.analyse(b, redlines.said(game, x["map"], "area"))
            return b, x["offset"], [d["at"] for d in doors.find(b, fired)]
        b = g.read(name)
        areas = redlines.said(game, name, "area")
        pieces.analyse(b, areas)  # first: the fake out search reads the pieces too
        rows = claims(game, name, b, fired, {name: fake_outs.search(b)}, signs.catalogue(redlines.register()), {x["map"]: beside(x) for x in around},
                      redlines.claims(game, name), areas)
        taken = {r["id"] for r in rows}
        for line in shared.places(game, name, {x["map"] for x in around}):
            i = claim_id("same_place", line["at"], taken)
            taken.add(i)
            rows.append({"id": i, **line})
        return rows
    except MemoryError:
        return None


def proposals(rows, now, history):
    """How the detectors' lines differ from a map's claims now, as changes to pick from, new ones first, then different, then gone:
    [{change, line (as it is, None if new), then (as it would be, None if gone), keep (picked to begin with), why}]. history:
    redlines.history. A claim the detectors find again moved a little (bmab.levels.TWIN) is the same claim, different.
    What people changed since the detectors first wrote a claim stays: only the fields nobody touched take the detectors' new
    values, and a claim whose differences are all such changes isn't offered. Of one added later nobody can say which fields
    people set, so all of it is offered, not picked; so is all of a claim in WHOLE that people changed. Picked to begin with: what is new and was never deleted, and what is different
    in fields nobody touched; never what is gone, which may be something someone checked and left as it was."""
    base, versions = history
    by_id = {r["id"]: r for r in now}
    left = [r for r in now if r["id"] not in {x["id"] for x in rows} and (r["kind"], r.get("type")) not in BY_PEOPLE and r["kind"] not in PEOPLE_KINDS]
    new, different = [], []
    for r in rows:
        line = by_id.get(r["id"])
        if line is None and r["kind"] in levels.TWIN:
            line = levels.twin(r, [x for x in left if x["kind"] == r["kind"]])
            if line is not None:
                left.remove(line)
        if line is None and r["kind"] in PAIRED:  # one per other map: a person's, wherever they put it, is the same claim
            line = next((x for x in now if same_pair(x, r)), None)
        if line is None:
            known = r["id"] in versions
            new.append({"change": "new", "line": None, "then": r, "keep": not known, "why": "deleted before" if known else ""})
            continue
        was = base.get(line["id"])
        touched = was is not None and any(line.get(k) != v for k, v in was.items())
        whole = was is None or (touched and r["kind"] in WHOLE)
        theirs = {k: v for k, v in r.items() if k != "id" and line.get(k) != v and (whole or line.get(k) == was.get(k))}
        if theirs:
            different.append({"change": "different", "line": line, "then": {**line, **theirs}, "keep": not whole,
                              "why": "added later" if was is None else "changed by hand" if whole else "keeps what was changed by hand" if touched else ""})
    gone = [{"change": "gone", "line": r, "then": None, "keep": False, "why": "added later" if r["id"] not in base else ""} for r in left]
    return new + different + gone
