"""How well the detectors do against maps reviewed by hand. The answer key is the maps' files at a commit of the redlines; the
detectors' claims are made again now (bmab.evaluate). For each kind of claim: right (the review kept it, perhaps adjusted),
wrong (the review deleted it, or moved it out of reach), missed (the review added it), and how the kept ones were changed.

Ought doors (where a door ought to be) aren't scored: only people place them.
"""
from collections import Counter

import numpy as np

from . import redlines, pieces

# how far apart (units) and how alike in direction (cosine) a found claim and a reviewed one may be and still be the same thing
REACH = {"fake_door": 48, "fake_out": 96, "connector": 128}
ALIKE = {"fake_door": 0.7, "fake_out": 0.7, "connector": 0.966}
SCORED = ("connector", "fake_door", "fake_out", "area")


def answers(game, name, rev):
    """A map's claims in the redlines at a commit."""
    rows = redlines.committed(f"maps/{game}/{name}.jsonl", rev)
    if rows is None:
        raise SystemExit(f"{game}/{name} isn't in the redlines at {rev}")
    return rows


def _direction(r):
    if r["kind"] == "fake_door":
        return np.array(r["back"], float)
    if r["kind"] == "connector":
        return np.array(r["along"], float)
    d = np.subtract(r["reaches"], r["at"], dtype=float)
    return d / (np.linalg.norm(d) or 1)


def _area_of(ps, at):
    """The index of the area (bmab.pieces) a claim's point names: the one it is the point of, or else the smallest whose box holds it."""
    for k, p in enumerate(ps):
        if [round(x) for x in p["at"]] == [round(x) for x in at]:
            return k
    inside = [k for k, p in enumerate(ps) if all(p["lo"][a] <= at[a] <= p["hi"][a] for a in range(3))]
    return min(inside, key=lambda k: ps[k]["m3"]) if inside else None


def _pairs(kind, found, key):
    """Found and reviewed claims of one kind paired up, nearest first: (pairs, found left over, reviewed left over)."""
    options = []
    for i, f in enumerate(found):
        for j, k in enumerate(key):
            d = float(np.linalg.norm(np.subtract(f["at"], k["at"])))
            along = float(_direction(f) @ _direction(k))
            if d <= REACH[kind] and (abs(along) if kind == "connector" else along) >= ALIKE[kind]:
                options.append((d, i, j))
    pairs, used_f, used_k = [], set(), set()
    for _d, i, j in sorted(options):
        if i not in used_f and j not in used_k:
            pairs.append((found[i], key[j]))
            used_f.add(i)
            used_k.add(j)
    return pairs, [f for i, f in enumerate(found) if i not in used_f], [k for j, k in enumerate(key) if j not in used_k]


def _changes(kind, f, k):
    """How the review changed a claim it kept."""
    out = []
    for field in ("type", "way"):
        if f.get(field) != k.get(field):
            out.append(f"{field} {f.get(field)} -> {k.get(field)}")
    if kind == "fake_door" and f.get("behind") != k.get("behind"):
        out.append(f"behind {f.get('behind')} -> {k.get('behind')}")
    if kind == "fake_door" and f.get("sides") != k.get("sides"):
        out.append("both sides set" if k.get("sides") == 2 else "both sides taken off")
    if kind == "connector":
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(_direction(f) @ _direction(k)))))))
        if angle > 1:
            out.append(f"turned {angle:.0f} degrees")
        if float(np.linalg.norm(np.subtract(f["at"], k["at"]))) > 24:
            out.append("moved")
        for field in ("start_m", "end_m", "whole_m"):
            if field in k and k.get(field) != f.get(field):
                out.append(f"{field} set")
    return out


def compare(name, b, found, key):
    """{kind: {right, wrong, missed, changes: Counter}, with the claims themselves under "items"} for one map."""
    out = {}
    ps = pieces.analyse(b) if any(r["kind"] == "area" for r in found + key) else []
    for kind in SCORED:
        f, k = [r for r in found if r["kind"] == kind], [r for r in key if r["kind"] == kind]
        if kind == "area":
            fi, ki = {_area_of(ps, r["at"]) for r in f}, {_area_of(ps, r["at"]) for r in k}
            out[kind] = {"right": len(fi & ki), "wrong": len(fi - ki), "missed": len(ki - fi), "changes": Counter(),
                         "items": {"wrong": sorted(x for x in fi - ki if x is not None), "missed": sorted(x for x in ki - fi if x is not None)}}
            continue
        pairs, extra, left = _pairs(kind, f, k)
        changes = Counter(c for a, b_ in pairs for c in _changes(kind, a, b_))
        out[kind] = {"right": len(pairs), "wrong": len(extra), "missed": len(left), "changes": changes,
                     "items": {"wrong": [r["at"] for r in extra], "missed": [r["at"] for r in left]}}
    return out


def table(results):
    """The scores as lines of text: per map and over all of them."""
    lines, total = [], {kind: Counter() for kind in SCORED}
    for name, kinds in results.items():
        lines.append(name)
        for kind, s in kinds.items():
            total[kind].update({x: s[x] for x in ("right", "wrong", "missed")})
            total[kind].update(s["changes"])
            lines.append(_line(kind, s, s["changes"]))
    lines.append("all of them")
    for kind, s in total.items():
        lines.append(_line(kind, s, Counter({c: n for c, n in s.items() if c not in ("right", "wrong", "missed")})))
    return lines


def _line(kind, s, changes):
    found, kept = s["right"] + s["wrong"], s["right"] + s["missed"]
    share = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"  # noqa: E731
    said = ", ".join(f"{c} {n}" for c, n in changes.most_common(4))
    return f"  {kind:10} right {s['right']:4}  wrong {s['wrong']:4}  missed {s['missed']:4}  of what it found, right {share(s['right'], found):>4}  of what is there, found {share(s['right'], kept):>4}" + (f"  changed: {said}" if said else "")
