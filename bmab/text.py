"""Text a GoldSrc game carries about places: announcements, titles, map messages and names.

Everything is returned with where it came from, so a fact can be traced to its game file and map.
"""
import re
from collections import defaultdict

from . import bsp

# Sound effects and pauses in announcements, not words.
VOX_NOISE = {"deeoo", "buzwarn", "bizwarn", "dadeda", "doop", "bloop", "beep", "buzz", "blip", "dut"}
VOX_PAUSE = {"_comma": ",", "_period": "."}

NUMBERS = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
           "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred").split()
NATO = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
        "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
KINDS = (r"sector|area|level|sub[- ]?level|silo|lab|laboratory|complex|facility|dock|bay|wing|zone|station|hub|"
         r"line|elevator|lift|yard|reactor|plant|depot|storage|unit|dormitory|dormitories|checkpoint|tram|"
         r"track|entrance|chamber|core|tower|dam|canal|hangar|bunker|gate|door|room|hall|corridor|shaft|tunnel")
# A place kind followed by a series identifier: one letter, a number, a number word or a NATO word.
PLACE = re.compile(rf"\b({KINDS})[\s_-]+([a-z]|\d+[a-z]?|(?:{'|'.join(NUMBERS + NATO)})(?:[\s-](?:{'|'.join(NUMBERS)}))?)\b", re.I)


def sentences(text):
    """name -> {"dir", "text"}; also the developer comments, in order."""
    out, comments = {}, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("//"):
            if line[2:].strip():
                comments.append(line[2:].strip())
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        name, body = parts
        words = re.sub(r"\([^)]*\)", " ", body).split()
        folder = "vox"
        if words and "/" in words[0]:
            folder, words[0] = words[0].rsplit("/", 1)
        clean = [VOX_PAUSE.get(w, w) for w in words if w and w.strip(",.!?") not in VOX_NOISE]
        out[name.upper()] = {"dir": folder, "text": re.sub(r"\s+([,.!?])", r"\1", " ".join(clean)).strip()}
    return out, comments


def titles(text):
    out = {}
    for name, body in re.findall(r"^\s*([A-Za-z0-9_]+)\s*\{(.*?)\}", text, re.S | re.M):
        out[name.upper()] = " ".join(body.split())
    return out


# speaker entities pick an announcement group by preset number (Half-Life SDK).
SPEAKER_PRESETS = ["", "C1A0_", "C1A1_", "C1A2_", "C1A3_", "C1A4_", "C2A1_", "C2A2_", "C2A3_", "C2A4_",
                   "C2A5_", "C3A1_", "C3A2_"]


def references(maps):
    """Which maps use which announcement (!NAME) or title (env_message), and their own text."""
    said, shown, messages, names = defaultdict(set), defaultdict(set), [], defaultdict(set)
    groups = defaultdict(set)  # announcement group prefix -> maps with a speaker playing it
    for m, b in maps.items():
        for e in b.entities:
            cls = e.get("classname", "")
            if cls == "speaker":
                preset = int(e.get("preset") or 0)
                group = SPEAKER_PRESETS[preset] if 0 < preset < len(SPEAKER_PRESETS) else e.get("message", "")
                if group:
                    groups[group.upper()].add(m)
            for k, v in bsp.items(e):
                if v.startswith("!") and len(v) > 1:
                    said[v[1:].upper()].add(m)
            if cls == "env_message" and e.get("message"):
                shown[e["message"].upper()].add(m)
            if cls == "game_text" and e.get("message"):
                messages.append({"map": m, "text": e["message"]})
            for k in ("targetname", "landmark"):
                if e.get(k):
                    names[e[k]].add(m)
    return said, shown, messages, names, groups


def places(items):
    """(source label, text) pairs -> phrase -> sources."""
    found = defaultdict(set)
    for label, text in items:
        for kind, ident in PLACE.findall(text.replace("_", " ")):
            found[f"{kind.lower()} {ident.lower()}"].add(label)
    return found


def voice_usage(sentences_text, maps):
    """Voice file (folder/name, no extension) -> {"sentences": [...], "maps": [...]} for files a sentence uses
    and the maps that play those sentences or the file directly."""
    said, _shown, _msgs, _names, groups = references(maps)
    out = defaultdict(lambda: {"sentences": set(), "maps": set()})
    for line in sentences_text.splitlines():
        parts = line.split()
        if len(parts) < 2 or line.lstrip().startswith("//") or "/" not in parts[1]:
            continue
        name = parts[0].upper()
        folder, first = parts[1].split("/", 1)
        played = set(said.get(name, set()))
        for group, ms in groups.items():
            if name.startswith(group):
                played |= ms
        for word in [first] + parts[2:]:
            word = re.sub(r"\([^)]*\)", "", word).strip(",.")
            if word:
                out[f"{folder}/{word}"]["sentences"].add(name)
                out[f"{folder}/{word}"]["maps"] |= played
    for m, b in maps.items():
        for e in b.entities:
            for v in bsp.values(e):
                if v.lower().endswith(".wav"):
                    out[v.lower()[:-4]]["maps"].add(m)
    return {k: {"sentences": sorted(v["sentences"]), "maps": sorted(v["maps"])} for k, v in out.items()}


def source_line_usage(folder, maps):
    """Black Mesa (Source): caption/sound key -> maps that play it, through talker rules, scene files or map
    entities. Keys spoken by generic NPC response sets get "(any map with that character)"."""
    from pathlib import Path
    folder = Path(folder)
    map_values = {m: {v.lower() for e in b.entities for v in bsp.values(e)} for m, b in maps.items()}
    usage = defaultdict(set)
    for f in sorted((folder / "scripts" / "talker").glob("*.txt")):
        t = f.read_text("latin-1")
        crit_maps = set(re.findall(r'criterion\s+\S+\s+"map"\s+"(\w+)"', t, re.I))
        for key in re.findall(r'(?:speak|sentence)\s+"!?([^"]+)"', t, re.I):
            usage[key.lower()] |= {m for m in crit_maps if m in maps} or {"(any map, talker rules)"}
    scene_maps = defaultdict(set)
    for m, vals in map_values.items():
        for v in vals:
            if v.endswith(".vcd"):
                scene_maps[v.replace("\\", "/")].add(m)
    for vcd in (folder / "scenes").rglob("*.vcd"):
        rel = vcd.relative_to(folder).as_posix().lower()
        for key in re.findall(r'"([A-Za-z_][\w.]+\.[\w.]+)"', vcd.read_text("latin-1")):
            usage[key.lower()] |= scene_maps.get(rel, set()) or ({"(any map with that character)"} if rel.startswith("scenes/npc/") else set())
    for m, vals in map_values.items():
        for v in vals:
            if v and v in usage:
                usage[v].add(m)
    return {k: sorted(v) for k, v in usage.items()}
