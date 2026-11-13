"""The register: what is known about the facility as a whole, beside the claims about each map (bmab.claims). It says what each
kind of entry holds and checks one; nothing here reads or writes anything (bmab.redlines keeps them). Each entry is one JSON
object with an id; entries refer to each other by id, like the rows of tables in a database.

sources: a game, expansion, mod, map source, outside work or website that evidence comes from: id (bmab's manifest id for a game
  on this computer, its own otherwise), title, kind (SOURCE_KINDS), edition (EDITIONS; games only), tier (1 to 5: how much it
  counts against the others, the lowest first) and, for a mod once it is judged, verdict (VERDICTS), rank (its place among the
  admitted or the reserve, 1 first), reason (REASONS; rejected only) and grounds (the evidence the verdict rests on).
evidence: something a source says, never interpreted: id, source, kind, tier (how much it counts: its source's, or one lower when
  it isn't seen or heard in the campaign: shipped but unused, only in training or multiplayer maps, a manual, a developer's
  comment), then either where it is in the source's files or the web page it is. In files: kind (EVIDENCE_KINDS), file (its path
  in the source), sha256, ref (where in the file: a texture's name, a line's id, a page, a timecode), seen_in and heard_in (the
  source's maps that show or play it) and transcription, only for text bmab can't read itself (text in a picture: a sign, a wall
  map). Text bmab can read (a line, a caption, a page) is kept by reference only, so no game's text is copied into the redlines.
  On the web: kind (PAGE_KINDS), url, read (the day it was read, YYYY-MM-DD) and how (HOW: read first-hand, or only through a
  search engine's summary of it).
facts: one statement about places: id, subject (a place), relation (RELATIONS), then object (a place) for most relations, value
  for the measured ones (VALUED: metres, or degrees for the way a sign's arrow points on its picture) and nothing for exists;
  evidence: the ids of what says it.
places: id, type (PLACE_TYPES) and implied_by: the places that imply one nothing names (a series' missing member).
events: something that happens at the facility on a day: id, date (MM-DD: no source gives the year), time (HH:MM), place, and
  evidence: the ids of what says it.
"""
import re

SOURCE_KINDS = ("game", "work", "site")
EDITIONS = ("original", "remake")
VERDICTS = ("admitted", "reserve", "rejected", "pending", "duplicate", "reference")
RANKED = ("admitted", "reserve")
REASONS = ("not in Black Mesa", "contradicts canon", "unreliable geography", "copies canon", "restores cut designs", "too thin",
           "never released")
EVIDENCE_KINDS = ("sign", "wall map", "announcement", "caption", "screen text", "voice line", "developer comment", "designer label",
                  "navigation graph", "manual", "book", "documentary")
PAGE_KINDS = ("store page", "wiki page", "review", "article", "community post")
HOW = ("first-hand", "search summary")
RELATIONS = ("exists", "in", "near", "above", "below", "across from", "leads to", "connects", "tram to", "tram stop", "serves",
             "controls", "is", "modelled on", "older than", "length", "elevation", "points")
VALUED = {"length": "m", "elevation": "m", "points": "degrees"}
ALONE = ("exists",)
# what a place is, from outside in: its rank (a place holds only places of a later rank, or of its own for complexes and rooms)
PLACE_TYPES = {"region": 0, "facility": 1, "sector": 2, "transit": 2, "area": 3, "complex": 3, "level": 4, "room": 5, "thing": 6, "elsewhere": 0}
MAPPABLE = ("sector", "area", "complex", "level", "room", "transit")  # what a map can be in
ID = re.compile(r"[a-z0-9][a-z0-9-]*(/[^/\s][^/]*)*")
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
SHA256 = re.compile(r"[0-9a-f]{64}")
DAY = re.compile(r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])")
DATE = re.compile(r"(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])")
TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


def _fields(entry, allowed, what):
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not ID.fullmatch(entry["id"]):
        raise ValueError(f"a {what} needs an id")
    if extra := set(entry) - allowed - {"id"}:
        raise ValueError(f"not part of a {what}: {', '.join(sorted(extra))}")


def _ids(x, key, known, message):
    if key in x and not (isinstance(x[key], list) and x[key] and all(i in known for i in x[key])):
        raise ValueError(message)


def check_source(s, evidence):
    _fields(s, {"title", "kind", "edition", "tier", "verdict", "rank", "reason", "grounds"}, "source")
    if not (isinstance(s.get("title"), str) and s["title"] and s.get("kind") in SOURCE_KINDS and s.get("tier") in (1, 2, 3, 4, 5)):
        raise ValueError(f"a source has a title, a kind ({', '.join(SOURCE_KINDS)}) and a tier from 1 to 5")
    if (s["kind"] == "game") != ("edition" in s) or s.get("edition", EDITIONS[0]) not in EDITIONS:
        raise ValueError(f"a game has an edition ({', '.join(EDITIONS)}) and a work none")
    if s.get("verdict", VERDICTS[0]) not in VERDICTS:
        raise ValueError(f"a verdict is one of {', '.join(VERDICTS)}")
    if "rank" in s and not (s.get("verdict") in RANKED and type(s["rank"]) is int and s["rank"] >= 1):
        raise ValueError(f"a rank is a whole number from 1, among the {' or the '.join(RANKED)}")
    if "reason" in s and not (s.get("verdict") == "rejected" and s["reason"] in REASONS):
        raise ValueError(f"a rejection's reason is one of {', '.join(REASONS)}")
    if "grounds" in s and "verdict" not in s:
        raise ValueError("grounds are what a verdict rests on")
    _ids(s, "grounds", evidence, "grounds name evidence the register has")
    return s


def check_evidence(e, sources):
    if isinstance(e, dict) and "url" in e:
        return _check_page(e, sources)
    _fields(e, {"source", "kind", "file", "sha256", "ref", "tier", "seen_in", "heard_in", "transcription"}, "piece of evidence")
    _source_and_tier(e, sources)
    if e.get("kind") not in EVIDENCE_KINDS:
        raise ValueError(f"evidence in files is one of {', '.join(EVIDENCE_KINDS)}")
    if not (isinstance(e.get("file"), str) and isinstance(e.get("ref"), str) and isinstance(e.get("sha256"), str) and SHA256.fullmatch(e["sha256"])):
        raise ValueError("evidence names its file, that file's sha256 and where in it (ref), or the web page it is (url)")
    for k in ("seen_in", "heard_in"):
        if k in e and not (isinstance(e[k], list) and all(isinstance(m, str) for m in e[k])):
            raise ValueError(f"{k} lists maps of the source")
    if "transcription" in e and (e["kind"] not in ("sign", "wall map") or not isinstance(e["transcription"], str)):
        raise ValueError("only text in a picture (a sign, a wall map) is transcribed; bmab reads the rest itself")
    return e


def _source_and_tier(e, sources):
    if e.get("source") not in sources:
        raise ValueError(f"no source {e.get('source')}")
    if e.get("tier") not in (1, 2, 3, 4, 5):
        raise ValueError("evidence has a tier from 1 to 5")


def _check_page(e, sources):
    _fields(e, {"source", "kind", "url", "read", "how", "tier"}, "piece of evidence")
    _source_and_tier(e, sources)
    if e.get("kind") not in PAGE_KINDS:
        raise ValueError(f"a web page is one of {', '.join(PAGE_KINDS)}")
    if not (isinstance(e["url"], str) and e["url"].startswith(("https://", "http://"))):
        raise ValueError("a web page's url starts with https://")
    if not (isinstance(e.get("read"), str) and DAY.fullmatch(e["read"])):
        raise ValueError("a web page says the day it was read (YYYY-MM-DD)")
    if e.get("how") not in HOW:
        raise ValueError(f"a web page was read {' or through a '.join(HOW)}")
    return e


def check_fact(f, places, evidence):
    _fields(f, {"subject", "relation", "object", "value", "evidence"}, "fact")
    if f.get("relation") not in RELATIONS:
        raise ValueError(f"a relation is one of {', '.join(RELATIONS)}")
    if f.get("subject") not in places:
        raise ValueError(f"no place {f.get('subject')}")
    rel = f["relation"]
    if rel in VALUED:
        if not isinstance(f.get("value"), (int, float)) or "object" in f:
            raise ValueError(f"{rel} takes a value in {VALUED[rel]}, not a place")
    elif rel in ALONE:
        if "object" in f or "value" in f:
            raise ValueError(f"{rel} takes neither a place nor a value")
    elif f.get("object") not in places or "value" in f:
        raise ValueError(f"{rel} takes a place")
    _ids({"evidence": f.get("evidence")}, "evidence", evidence, "a fact names the evidence that says it")
    return f


def check_place(p, places):
    _fields(p, {"type", "implied_by"}, "place")
    if not NAME.fullmatch(p["id"]):
        raise ValueError("a place is named in small letters and digits joined by hyphens, as sector-c")
    if p.get("type") not in PLACE_TYPES:
        raise ValueError(f"a place's type is one of {', '.join(PLACE_TYPES)}")
    if "implied_by" in p and not (isinstance(p["implied_by"], list) and p["implied_by"] and all(q in places for q in p["implied_by"])):
        raise ValueError("an implied place names the places that imply it")
    return p


def check_event(v, places, evidence):
    _fields(v, {"date", "time", "place", "evidence"}, "event")
    if not NAME.fullmatch(v["id"]):
        raise ValueError("an event is named in small letters and digits joined by hyphens, as resonance-cascade")
    if not (isinstance(v.get("date"), str) and DATE.fullmatch(v["date"])):
        raise ValueError("an event has a date, as MM-DD (no source gives the year)")
    if "time" in v and not (isinstance(v["time"], str) and TIME.fullmatch(v["time"])):
        raise ValueError("a time is HH:MM, on a 24 hour clock")
    if "place" in v and v["place"] not in places:
        raise ValueError(f"no place {v['place']}")
    _ids({"evidence": v.get("evidence")}, "evidence", evidence, "an event names the evidence that says it")
    return v


def refers(table, entry):
    """What an entry of a table (sources, places, facts, events, evidence) names, as (table, id): the entries it needs there. A fact
    or an event its places and evidence, a verdict its grounds, evidence its source, an implied place the places implying it."""
    if table in ("facts", "events"):
        named = (entry.get("subject"), entry.get("object")) if table == "facts" else (entry.get("place"),)
        return [("places", p) for p in named if p] + [("evidence", e) for e in entry.get("evidence", [])]
    if table == "places":
        return [("places", p) for p in entry.get("implied_by", [])]
    if table == "sources":
        return [("evidence", e) for e in entry.get("grounds", [])]
    return [("sources", entry["source"])] if table == "evidence" else []
