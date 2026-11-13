"""What a place looks like (bmab.claims.STYLES), guessed from the textures around it, for the redlines to start from: what fills
a fake or ought door that leads nowhere, a fake out (a passage past the map's end) or a part used elsewhere should match it. People correct the guesses,
and what they say is what these rules are measured against.

A texture's style comes from its name: the first of RULES that matches (Source's material folder counts as part of the name).
Many names say nothing (concrete, trims, generic panels) and don't count; the sky counts as outdoors. Around a point: the area
of the view that the faces within REACH fill, seen from the point. A door: its own door textures first, since a door's look says
where it leads, which needn't be what the hall it is in looks like; then what is around its front. A part: the faces inside its box.
Where nothing within REACH says anything, four times as far does, and failing that the map as a whole, its sky left out; a
map with no telling textures at all gives no guess.
"""
import functools
import re

import numpy as np

from . import brushes, bsp, doors, families
from .claims import STYLES

RULES = [(style, re.compile(pattern)) for style, pattern in (
    (None, r"^(dev|tools|decals|effects|lights?|signs?|glass|liquids)/|black|water|slime|fluid|toxic|lava|trim|generic|^light|^sign"),
    ("xen", r"xen|c4a\d|gon_|gonarch|nihil|qzone|hostile_island|swmp|alien|organic"),
    ("storage", r"crate|freezer|frost|barrel|pallet|shel[fv]|warehouse|storage|cargo|freight|flatbed"),
    ("transit", r"tram|train|trn_|subway|(^|[/_])rail|railway|track|trk|elev|^oar_|/oar_|c2a2|trrm|station"),
    ("living", r"(^|[/_])bed|bath|toilet|shower|kitchen|cafe|dorm|locker|lounge|wallpaper|ceramic"),
    ("office", r"office|offflr|fift|carpet|wood|c1a2|cubicle|desk|plaster"),
    ("lab", r"(^|[/_\d])lab|babtech|c1a0|c1a1|reac|lambda|c3a2|c2a4x|xray|portal|teleport"),
    ("maintenance", r"tnnl|tunnel|duct|vent|swr|sewer|pipe|grate|grt|steam|maint|boiler|cinderblock"),
    ("industrial", r"silo|metal|mtl|steel|(^|_)iron|rust|catwalk|turbine|(^|_)vat|gear|machin|factory|plant|power|tank|rocket|gasworks"
                   r"|stripe|caution|hazard|c1a4|c2a1|c2a3|c2a4|c3a1|nuk|corrugat|truss|beam"),
    ("outdoors", r"(^|[/_])\d?out_|nature/|rock|(^|_)rk\d|cliff|grass|sand|(^|_)snd|mud|dirt(?!y)|grnd|grvl|gravel|asphalt|pave|road|snow"
                 r"|canyon|desert|cave|terrain|dam_"),
)]
SKY = ("sky", "tools/toolsskybox", "tools/toolsskybox2d")
SIGN_RULES = [(style, re.compile(pattern)) for style, pattern in (  # over the places a sign names (bmab.signs)
    ("storage", r"storage|warehouse|freight|supply"),
    ("living", r"cafeteria|canteen|dorm|lounge|locker|restroom|infirmary|refreshment"),
    ("office", r"office|admin|conference|management"),
    ("lab", r"lab|research|test|specimen|observation|chamber|laser"),
    ("industrial", r"generator|power|reactor|pump|turbine|coolant|fuel|silo|waste|processing|repair|engine|launch"),
    ("maintenance", r"maintenance|ventilation|shaft|sewer|drain|hatch|utility"),
    ("transit", r"elevator|lift|transit|tram|train|rail|transport|parking|platform|hangar|dock"),
)]
SIGN_NEAR = 160  # units: a sign this close beside a door, on its side of the wall, says what is behind it
REACH = 768      # units: how far around a point its look is taken; where nothing that far says anything, four times as far
NEAR = 64        # units: closer than this a face counts as if this far, so the floor under the point doesn't outweigh the rest
DOOR_SHARE = 1.5  # a door's own faces against all around it: where they say something, they decide
SAMPLES = 16


@functools.lru_cache(maxsize=None)
def of_texture(texture):
    """A texture's style, or None if its name doesn't say."""
    t = texture.lower()
    if t in SKY:
        return "outdoors"
    if bsp.is_tool(t):
        return None
    t = re.sub(r"(_-?\d+){3}$", "", re.sub(r"^maps/[^/]+/", "", t))  # Source's copies of a material for a place's reflections
    head, _, last = t.rpartition("/")
    t = (head + "/" if head else "") + families.base(last)
    return next((style for style, rule in RULES if rule.search(t)), None)


def _faces(b):
    """(middles, areas, unit normals, style index or -1, in door textures, sky) of the map's faces, the 3D skybox left out."""
    if getattr(b, "looks", None) is None:  # kept on the map, not as a Bsp field: bsp.py's text keys the connector caches
        fs = [f for f in b.faces if not f.skybox]
        placed = [brushes.place(b, f) for f in fs]
        area = np.array([0.5 * np.linalg.norm(np.cross(f.verts[1:-1] - f.verts[0], f.verts[2:] - f.verts[0]).sum(axis=0)) for f in fs])
        n = np.array([nrm for _v, nrm in placed], float).reshape(-1, 3)
        style = np.array([-1 if (s := of_texture(f.texture)) is None else STYLES.index(s) for f in fs], int)
        door = np.array([doors.is_door_texture(f.texture) for f in fs], bool)
        sky = np.array([f.texture.lower() in SKY for f in fs], bool)
        b.looks = (np.array([v.mean(0) for v, _n in placed]).reshape(-1, 3), area, n, style, door, sky)
    return b.looks


def _open_from(b, p, away):
    """The first open point from p going away (up to 128 units); None if there is none."""
    if b.vis is None:  # PS2 maps: no tree read, so every point counts as open
        return p
    for t in range(0, 136, 8):
        q = p + away * t
        if b.vis.open(q[None])[0]:
            return q
    return None


def _seen(b, p, reach=REACH):
    """Votes (one per style): how much of what p sees within reach is in each style's textures (solid angle)."""
    mid, area, n, style, is_door, sky = _faces(b)
    d = np.linalg.norm(mid - p, axis=1)
    k = np.flatnonzero((style >= 0) & (d <= reach) & (np.einsum("ij,ij->i", p - mid, n) > 0))
    votes = np.zeros(len(STYLES))
    if not len(k):
        return votes if reach > REACH else _seen(b, p, 4 * REACH)
    # Source's open space pads the brushes, so a sight line is tested only up to that far before the face
    margin = 8 + (bsp.SOLID_CELL + bsp.SOLID_PAD if isinstance(b.vis, bsp.SourceVis) else 0)
    upto = np.maximum(d[k] - margin, 0)
    t = np.linspace(0, 1, SAMPLES)[None, :, None] * upto[:, None, None]
    u = (mid[k] - p) / np.maximum(d[k], 1e-9)[:, None]
    if b.vis is not None:
        k = k[b.vis.open((p + t * u[:, None, :]).reshape(-1, 3)).reshape(len(k), SAMPLES).all(1)]
    np.add.at(votes, style[k], area[k] * np.einsum("ij,ij->i", p - mid[k], n[k]) / np.maximum(d[k], NEAR) ** 3)  # how much of the view it fills
    return votes if votes.sum() > 0 or reach > REACH else _seen(b, p, 4 * REACH)


def _whole_map(b):
    """Votes by the area of each style's textures over the map, the sky left out (its brushes are huge)."""
    mid, area, n, style, is_door, sky = _faces(b)
    k = (style >= 0) & ~sky
    votes = np.zeros(len(STYLES))
    np.add.at(votes, style[k], area[k])
    return votes


def _pick(b, votes):
    if votes.sum() <= 0:
        votes = _whole_map(b)
    return STYLES[int(np.argmax(votes))] if votes.sum() > 0 else None


def _share(votes):
    return votes / votes.sum() if votes.sum() > 0 else votes


def around(b, p, away):
    """The style seen from p, or from the first open point from it going away (a unit vector)."""
    q = _open_from(b, np.asarray(p, float), np.asarray(away, float))
    return _pick(b, np.zeros(len(STYLES)) if q is None else _seen(b, q))


def _signed(line, signs):
    """The style a sign beside a door names (a plaque that says what is behind it; a sign with arrows points elsewhere), or None."""
    at, front = np.array(line["at"], float), -np.array(line["back"], float)
    for s in sorted(signs, key=lambda s: np.linalg.norm(np.asarray(s["at"], float) - at)):
        off = np.asarray(s["at"], float) - at
        if s["points"] or np.asarray(s["normal"], float) @ front < 0.7 or abs(off @ front) > 32 or np.linalg.norm(off) > SIGN_NEAR:
            continue
        named = " ".join(p for c in s["claims"] for p in (c[0], c[2] if len(c) > 2 else "") if p)
        style = next((style for style, rule in SIGN_RULES if rule.search(named)), None)
        if style:
            return style
    return None


def door(b, line, signs=()):
    """What a door's far side should look like: what a sign beside it names, or its own door textures' style, or else what is
    around its front. signs: bmab.signs.described."""
    if (style := _signed(line, signs)) is not None:
        return style
    at, back = np.array(line["at"], float), np.array(line["back"], float)
    side = np.array([-back[1], back[0], 0.0]) / max(np.hypot(back[0], back[1]), 1e-9)
    mid, area, n, style, is_door, sky = _faces(b)
    off = mid - at
    own = (style >= 0) & is_door & (n @ -back > 0.7) & (np.abs(off @ back) <= 16) & (np.abs(off @ side) <= line["width"] / 2) & (np.abs(off[:, 2]) <= line["height"] / 2)
    votes = np.zeros(len(STYLES))
    np.add.at(votes, style[own], area[own])
    q = _open_from(b, at - back * 32, -back)
    return _pick(b, DOOR_SHARE * _share(votes) + (_share(_seen(b, q)) if q is not None else 0))


def fake_out(b, line):
    """What the passage a fake out ends in looks like: a tram line or a lift shaft where it is a ride (tram tunnels are in the same
    textures as service tunnels), else what is around the point in front of the surface it stops at."""
    if line.get("way") == "ride":
        return "transit"
    at = np.array(line["at"], float)
    back = at - np.array(line["reaches"], float)
    back /= max(np.linalg.norm(back), 1e-9)
    return around(b, at + back * 16, back)


def part(b, line):
    """What a part looks like: the faces inside its box."""
    at, facing = np.array(line["at"], float), np.array(line["facing"], float)
    across = np.array([-facing[1], facing[0], 0.0])
    mid, area, n, style, is_door, sky = _faces(b)
    off = mid - at
    inside = (style >= 0) & (np.abs(off @ facing) <= line["size"][0] / 2) & (np.abs(off @ across) <= line["size"][1] / 2) & (np.abs(off[:, 2]) <= line["size"][2] / 2)
    votes = np.zeros(len(STYLES))
    np.add.at(votes, style[inside], area[inside])
    return _pick(b, votes)


def needs(line):
    """Whether a claim says what it looks like: a fake or ought door that leads nowhere, a fake out, a part."""
    return line["kind"] in ("fake_out", "part") or (line["kind"] in ("fake_door", "ought_door") and line.get("behind") == "nothing")


def guess(b, line, signs=()):
    """What a claim looks like (needs says which do). signs: bmab.signs.described, for doors."""
    return door(b, line, signs) if line["kind"] in ("fake_door", "ought_door") else {"fake_out": fake_out, "part": part}[line["kind"]](b, line)
