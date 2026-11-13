"""What bmab works out from the games on this computer that belongs in the register (bmab.register), offered to a person rather
than written: a generator says what it finds, differences says how that differs from what the redlines hold, and a person picks
what to keep (the app saves the picked ones like any edit, and Commit commits them). Nothing here reads or writes the redlines:
the stored entries and their history are passed in.

GENERATORS: usage, which of a game's maps show or play each piece of its evidence in files (seen_in for a sign's or wall map's
texture and for text shown on screen, heard_in for a voice line, an announcement or a caption); lines, the lines bmab can read
in a game that name a place (announcements, messages shown on screen, captions) and the register lacks, as new evidence.
"""
import hashlib
import re
import types

from . import bsp, signs, sources, text
from .game import Game

SEEN = ("sign", "wall map", "screen text")  # the rest a map plays: heard_in
GENERATORS = {"usage": {"title": "which maps show or play each piece of evidence", "table": "evidence", "fields": ("seen_in", "heard_in")},
              "lines": {"title": "lines that name places, as new evidence", "table": "evidence", "fields": ()}}
CAPTIONS = "resource/closecaption_english.txt"  # a Source game's captions, UTF-16


def games_here():
    """The games whose maps are on this computer."""
    return sorted(d.name for d in sources.DATA.iterdir() if (d / "maps").is_dir()) if sources.DATA.is_dir() else []


def _used(folder):
    """Per map of the game in folder: (texture names as the register gives them, entities), read without the maps' geometry."""
    g = Game(folder)
    out = {}
    for m in g.map_names:
        ents = bsp.entities(g.path(m))
        decals = {v for e in ents if e.get("classname", "") in ("infodecal", "info_overlay") for k, v in bsp.items(e) if k.lower() in ("texture", "material")}
        out[m] = ({signs._name(t) for t in bsp.textures(g.path(m)) | decals}, ents)
    return out


def usage(folder, evidence):
    """{evidence id: (field, maps)} for the game in folder's evidence in files that a map can show or play (evidence: that game's
    entries); maps is empty where no map does."""
    used = _used(folder)
    maps = {m: types.SimpleNamespace(entities=ents) for m, (_t, ents) in used.items()}
    said, shown, _messages, _names, groups = text.references(maps)
    sentences = folder / "sound" / "sentences.txt"
    voices = text.voice_usage(sentences.read_text("latin-1"), maps) if sentences.exists() else {}
    lines = text.source_line_usage(folder, maps) if (folder / "scripts").is_dir() else {}
    out = {}
    for e in evidence:
        if "file" not in e:
            continue
        kind, ref = e["kind"], e["ref"]
        if kind in ("sign", "wall map"):
            name = signs._name(ref)
            found = {m for m, (tex, _e) in used.items() if name in tex}
        elif kind == "screen text":
            found = shown.get(ref.upper(), set())
        elif kind == "announcement":
            found = set(said.get(ref.upper(), set())) | {m for g, ms in groups.items() if ref.upper().startswith(g) for m in ms}
        elif kind == "voice line" and voices:
            found = set(voices.get(re.sub(r"^[^/]+/sound/", "", e["file"].lower()).removesuffix(".wav"), {}).get("maps", []))
        elif kind == "caption":  # through talker rules or scenes, or a map's entity playing its sound by name
            found = set(lines.get(ref.lower(), [])) | {m for m, (_t, ents) in used.items() if any(v.lower() == ref.lower() for x in ents for v in bsp.values(x))}
        else:
            continue
        out[e["id"]] = ("seen_in" if kind in SEEN else "heard_in", sorted(found & used.keys()))
    return out


def lines(folder, game, tier, evidence):
    """New evidence for the lines bmab can read in the game in folder that name a place (bmab.text.PLACE) and that evidence (the
    game's entries) lacks, by file and position: the announcer's sentences (sound/sentences.txt), messages shown on screen
    (titles.txt) and captions (CAPTIONS). Each comes with the maps that play or show it and the game's tier, one lower when no
    map does (shipped but unused); a line no map uses is offered only for a game at tiers 1 to 3, as a mod's unused lines are
    mostly copies of the game it builds on. Returns (entries, {id: the line}); the line is for a person to read, never stored."""
    have = {(e.get("file"), e.get("ref")) for e in evidence}
    out, says = [], {}

    def add(eid, kind, path, ref, line):
        rel = f"{game}/{path.relative_to(folder).as_posix()}"
        if (rel, ref) not in have and text.PLACE.search(line.replace("_", " ")):
            out.append({"id": eid, "source": game, "kind": kind, "file": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ref": ref, "tier": tier})
            says[eid] = line
    path = folder / "sound" / "sentences.txt"
    if path.exists():
        for name, s in text.sentences(path.read_text("latin-1"))[0].items():
            if s["dir"] in ("vox", "fvox"):
                add(f"{game}/{name}", "announcement", path, name, s["text"])
    path = folder / "titles.txt"
    if path.exists():
        for name, line in text.titles(path.read_text("latin-1")).items():
            add(f"{game}/text/{name}", "screen text", path, name, line)
    path = folder / CAPTIONS
    if path.exists():
        for key, line in re.findall(r'"([^"]+)"\s+"([^"]*)"', path.read_text(encoding="utf-16", errors="replace")):
            add(f"{game}/caption/{key}", "caption", path, key, re.sub(r"<[^>]*>", "", line).strip())
    used = usage(folder, out)
    kept = []
    for e in out:
        field, maps = used.get(e["id"], (None, []))
        if maps:
            kept.append({**e, field: maps})
        elif tier <= 3:
            kept.append({**e, "tier": tier + 1})
    return kept, {e["id"]: says[e["id"]] for e in kept}


def run(job):
    """A generator over one game, in a process of its own: job = (generator, game, its tier, its evidence). Returns (the entries
    it proposes, {id: what a new one says, for a person to read}); an entry it changes is the stored one with what the generator
    works out in place."""
    what, game, tier, evidence = job
    folder = sources.DATA / game
    if what == "lines":
        return lines(folder, game, tier, evidence)
    if what != "usage":
        raise ValueError(f"no generator {what}")
    found = usage(folder, evidence)
    out = []
    for e in evidence:
        if e["id"] in found:
            field, maps = found[e["id"]]
            out.append({**{k: v for k, v in e.items() if k != field}, **({field: maps} if maps else {})})
    return out, {}


def proposals(what, result, stored, history):
    """The changes a generator's result (run) offers against the stored entries of its table (differences), each with what a new
    line says; a line no map plays or shows isn't picked to begin with."""
    rows, says = result
    changes = differences(rows, stored, history, GENERATORS[what]["fields"])
    for c in changes:
        i = c["then"]["id"]
        if i in says:
            c["says"] = says[i]
            if c["change"] == "new" and not (c["then"].get("seen_in") or c["then"].get("heard_in")):
                c.update(keep=False, why=c["why"] or "in no map")
    return changes


def differences(proposed, stored, history, fields):
    """How the proposed entries differ from the stored ones, as changes to pick from, new ones first, then different: [{change,
    line (as it is, None if new), then (as it would be), keep (picked to begin with), why}]. history: (base, versions) of the
    file they are kept in (bmab.redlines.history); fields: the ones the generator works out. As with the detectors' claims
    (bmab.evaluate.proposals), what people changed since the file was first written stays: only fields nobody touched take the
    generator's values, and an entry whose differences are all in touched fields isn't offered. An entry added later is offered
    whole but not picked, and so is one that was deleted before."""
    base, versions = history
    new, different = [], []
    for r in proposed:
        line = stored.get(r["id"])
        if line is None:
            known = r["id"] in versions
            new.append({"change": "new", "line": None, "then": r, "keep": not known, "why": "deleted before" if known else ""})
            continue
        was = base.get(r["id"])
        then = dict(line)
        for k in fields:
            if was is not None and line.get(k) != was.get(k):
                continue  # a person changed it
            if k in r:
                then[k] = r[k]
            else:
                then.pop(k, None)
        if then != line:
            different.append({"change": "different", "line": line, "then": then, "keep": was is not None, "why": "" if was is not None else "added later"})
    return new + different
