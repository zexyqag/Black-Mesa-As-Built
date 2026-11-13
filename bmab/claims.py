"""What a claim is: the kinds people and the detectors say about a map, what each kind holds, the words some fields take, and the
check a claim passes before it is kept. Nothing here reads or writes anything (bmab.redlines keeps them).

A claim is one JSON object about one map of one game: id, kind (SHAPE), at (a point in the map's own coordinates, units; only
the kinds about a spot have one) and what else its kind holds: what people review, nothing the tool works out from the map and
the rest (a connector's length, rise and outline area come from its shape; a fake door's being painted or locked from the map).
Claims about a spot are matched to what the tool finds by position, because what it finds moves between versions.

Claims about the whole map (MAP_KINDS) have no position: a place says which place of the facility the map is in (the most
specific one known: a place of the facts register, whose hierarchy says what holds it, or one it doesn't know yet; a map that
spans two places neither of which holds the other has one for each); era says whether it shows the facility before the incident
or after (ERAS; the main game's c1a0e is the last before); version says which file the claims are about (its path in the game and
sha256), so a copy that differs is noticed; review says which kinds of claim a person has checked all of in this map.

A fake door is a door the player can't go through (it never opens, or it is painted on a wall), and behind says what is behind it;
an ought door is where a door ought to be and isn't. A fake out is where a map looks like it goes on (type: space out of sight, a
dark end, or leads on, which people mark). An area is a piece of a map that isn't the facility (type: never seen, a Xen pocket, a
holding box). A connector's way, and a fake out's, is how the player gets through (WAYS).
For reusing maps, a style (STYLES) says what a place looks like, so that what fills it matches: a fake door that leads nowhere has
the style of what should be behind it, a fake out that of where it leads, and a part its own. The tool guesses them (bmab.styles)
and people correct them. A part is an upright box (at its middle, facing its length, size: length, width, height in units) around
something that may be used elsewhere, and extra boxes if one can't hold it alone (extra_boxes: [{at, size}], turned as the first).
The boxes only bound it: its shape is the space inside them joined to the first one's middle, so a room needn't be a box.
An opening is where something can join a part (part: the part's id): its middle, its facing (out of the part), its size (across and
up on a side; along the part's length and width on its floor or ceiling) and its way through. The tool finds them (bmab.parts) and
people correct them; what stands in one (a door, a level change, a connector...) is worked out from the map and the claims.
A level change claim (game and map: the level it leads to, landmark: which of their level changes) says what the map alone gets wrong about
it: loose, it doesn't line up (the two levels disagree about where they meet, so the layout may pull them apart there); by, what
really joins them (JOINS: a cut scene can look like a teleport). A same place claim says the map around it is the same place as a
spot in another map (game and map; there: that spot in the other map's own coordinates), of this game or another, edition or not: the two
maps then sit so that the spots meet. changed: the other is another version of the place (damaged, rebuilt), so only where it is
counts, not its walls. Both are claims about two maps (PAIRED), kept in both maps' files. People mark these.
"""
import hashlib
import json
import re

SHAPE = {"connector": {"way", "along", "outline", "path", "whole_m", "start_m", "end_m", "stretches_with"},
         "fake_door": {"back", "width", "height", "behind", "sides", "style"},
         "ought_door": {"back", "width", "height", "behind", "sides", "style"},
         "fake_out": {"type", "reaches", "seen_from", "m3", "m2", "way", "style"},
         "area": {"type"},
         "part": {"style", "facing", "size", "extra_boxes"},
         "opening": {"part", "facing", "size", "way"},
         "level_change": {"game", "map", "landmark", "loose", "by"},
         "same_place": {"game", "map", "there", "changed"},
         "place": {"place"},
         "era": {"era"},
         "version": {"file", "sha256"},
         "review": {"kinds"}}
MAP_KINDS = ("place", "era", "version", "review")  # about the whole map: no position
STYLES = ("office", "lab", "industrial", "maintenance", "storage", "living", "transit", "outdoors", "xen")
WAYS = ("walk", "crouch", "climb", "ride", "fall")  # how the player gets through; level or vertical is the connector's along
JOINS = ("teleport", "lift", "ladder", "ride", "walk")  # what joins two levels, the least trustworthy first
PAIRED = ("level_change", "same_place")
ERAS = ("before", "after")
SINGLE = ("era", "version", "review")  # one per map: its id is its kind
MAP_NAME = re.compile(r"[\w.-]+")
PLACE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # a place's name as the facts register writes them
CONNECTOR_FIELDS = {"along", "outline", "path"}


def claim_id(kind, at, taken=()):
    """The same kind of claim at the same place gets the same id every time, so evaluating a map again only changes what differs.
    at: its position, or for a claim about the whole map what tells it from others of its kind (a place's name)."""
    if kind in SINGLE:
        return kind
    key, n = json.dumps([kind, at]), 0
    while (i := hashlib.sha1(f"{key}{n or ''}".encode()).hexdigest()[:8]) in taken:
        n += 1
    return i


def _triple(v):
    return isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)


def check(line):
    """The claim, if it is one (ValueError saying what is wrong otherwise)."""
    if not isinstance(line, dict) or line.get("kind") not in SHAPE:
        raise ValueError("a claim needs a known kind")
    if line["kind"] in MAP_KINDS:
        if "at" in line:
            raise ValueError(f"a {line['kind']} claim is about the whole map: it has no position")
    elif not _triple(line.get("at")):
        raise ValueError("a claim needs a position")
    if extra := set(line) - {"id", "kind", "at"} - SHAPE[line["kind"]]:
        raise ValueError(f"not part of a {line['kind']} claim: {', '.join(sorted(extra))} (reload the page if it is older than the tool)")
    if "style" in line and line["style"] not in STYLES:
        raise ValueError(f"a style is one of {', '.join(STYLES)}")
    if "way" in line and line["way"] not in WAYS:
        raise ValueError(f"a way through is one of {', '.join(WAYS)}")
    if line["kind"] == "part" and not (_triple(line.get("size")) and isinstance(line.get("facing"), list)):
        raise ValueError("a part needs a facing and a size")
    if "extra_boxes" in line and not (isinstance(line["extra_boxes"], list) and all(
            isinstance(m, dict) and set(m) == {"at", "size"} and _triple(m["at"]) and _triple(m["size"]) for m in line["extra_boxes"])):
        raise ValueError("a part's extra boxes each need a middle and a size")
    if line["kind"] == "connector" and CONNECTOR_FIELDS - set(line):
        raise ValueError(f"a connector needs {', '.join(sorted(CONNECTOR_FIELDS - set(line)))}")
    if line["kind"] == "opening" and not (isinstance(line.get("part"), str) and _triple(line.get("facing")) and isinstance(line.get("size"), list)
                                          and len(line["size"]) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in line["size"])):
        raise ValueError("an opening needs its part, a facing and a size")
    if line["kind"] == "place" and not (isinstance(line.get("place"), str) and PLACE.fullmatch(line["place"])):
        raise ValueError("a place is named in small letters and digits joined by hyphens, as sector-c")
    if line["kind"] in PAIRED and not all(isinstance(line.get(k), str) and MAP_NAME.fullmatch(line[k]) for k in ("game", "map")):
        raise ValueError(f"a {line['kind']} claim names the other map and its game")
    if line["kind"] == "same_place" and not (_triple(line.get("there")) and line.get("changed") in (None, True)):
        raise ValueError("a same place claim names the spot in the other map that is the same place")
    if line["kind"] == "level_change" and not (isinstance(line.get("landmark"), str) and line.get("loose") in (None, True) and line.get("by") in (None, *JOINS)
                                               and (line.get("loose") or line.get("by"))):
        raise ValueError(f"a level change claim names the landmark, and says it doesn't line up (loose) or what joins them (by: {', '.join(JOINS)})")
    if line["kind"] == "era" and line.get("era") not in ERAS:
        raise ValueError(f"an era is one of {', '.join(ERAS)}")
    if line["kind"] == "version" and not (isinstance(line.get("file"), str) and isinstance(line.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", line["sha256"])):
        raise ValueError("a version names the map's file and its sha256")
    if line["kind"] == "review" and not (isinstance(line.get("kinds"), list) and set(line["kinds"]) <= set(SHAPE) - set(MAP_KINDS)):
        raise ValueError("a review lists the kinds of claim about a spot that a person has checked all of")
    return line


def same_pair(r, line):
    """Whether claim r is about the same two maps as line (PAIRED; for a level change, also the same one of their level changes)."""
    return (r["kind"] == line["kind"] and (r["game"], r["map"]) == (line.get("game"), line.get("map"))
            and (line["kind"] != "level_change" or r["landmark"] == line.get("landmark")))
