"""Layout, stage 1: place one game's maps by their links and log what doesn't fit.

- Maps are placed by graph.build, with the level changes people say don't line up (level change claims) never placing anything.
- Every link that disagrees with the placement becomes an elastic join: the two ends of the passage as
  placed, and the unseen length between them. Seen geometry is literal; joins may be longer than shown,
  never shorter (the compressed-space rule), so each join records how much hidden passage it implies.
- Overlapping floor between maps that aren't linked is listed as a clash for stage 2, unless the two maps
  share the floor at the same height (one room built into both maps, a revisit).
"""
import numpy as np

from . import graph, plan
from .graph import INCH


def _landmark(b, mark):
    for e in b.entities:
        if e.get("classname") == "info_landmark" and e.get("targetname") == mark:
            return np.array([float(x) for x in e["origin"].split()])
    return None


def stage1(maps, start, decided, component=None, min_clash_m2=20, revisit=0.6):
    banned = {graph.key(d) for d in decided if d["gives"]}
    g = graph.build(maps, start, decided)
    offsets = {n: np.array(m["offset"]) for n, m in g["maps"].items()}
    root = component or g["start"]
    names = [n for n, m in g["maps"].items() if m["component"] == root]

    joins = []
    for gap in g["gaps"]:
        a, b = gap["maps"]
        if a not in names:
            continue
        pa = _landmark(maps[a], gap["landmark"]) + offsets[a]
        pb = _landmark(maps[b], gap["landmark"]) + offsets[b]
        key = graph.key(gap)
        reason = next((d.get("reason", "") for d in decided if graph.key(d) == key), "")
        joins.append({"maps": [a, b], "landmark": gap["landmark"], "kind": gap["kind"],
                      "status": "decided" if key in banned else "open", "reason": reason,
                      "from": [round(float(x)) for x in pa], "to": [round(float(x)) for x in pb],
                      "length_m": round(float(np.linalg.norm(pb - pa)) * INCH, 1),
                      "rise_m": round(float(pb[2] - pa[2]) * INCH, 1)})
    joins.sort(key=lambda j: (j["status"] != "open", -j["length_m"]))

    polys = {n: plan.floors(maps[n], offsets[n], facility=True) for n in names}
    linked = {tuple(link["maps"]) for link in g["links"]}
    overlaps = plan.overlaps(polys, linked)
    clashes = [o for o in overlaps if not o["linked"] and o["same_height"] < revisit and o["m2"] >= min_clash_m2]
    revisits = [o for o in overlaps if not o["linked"] and o["same_height"] >= revisit]
    return {"graph": g, "component": root, "maps": sorted(names), "joins": joins, "clashes": clashes,
            "revisits": revisits, "polys": polys, "offsets": {n: offsets[n].tolist() for n in names}}


def draw(result, path, size=4000):
    """The stage-1 plan, with elastic joins as yellow lines (open ones dashed) and clashing maps outlined."""
    from PIL import ImageDraw, ImageFont, Image
    info = plan.render(result["polys"], path, size)
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    v = np.concatenate([p for ps in result["polys"].values() for p in ps])
    lo, hi = v.min(0), v.max(0)
    scale = size / max(hi[0] - lo[0], hi[1] - lo[1])
    h = img.height

    def px(p):
        return (20 + (p[0] - lo[0]) * scale, h - 20 - (p[1] - lo[1]) * scale)

    font = ImageFont.load_default(size=22)
    for j in result["joins"]:
        a, b = px(j["from"]), px(j["to"])
        colour = (255, 220, 40) if j["status"] == "decided" else (255, 120, 40)
        d.line([a, b], fill=colour, width=6)
        for p in (a, b):
            d.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], outline=colour, width=4)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        d.text((mid[0] + 12, mid[1]), f"{j['maps'][0]}/{j['maps'][1]} {j['length_m']} m ({j['status']})",
               fill=colour, font=font)
    for c in result["clashes"][:12]:
        for n in c["maps"]:
            ps = result["polys"][n]
            q = np.concatenate(ps)
            x0, y0 = px((q[:, 0].min(), q[:, 1].max()))
            x1, y1 = px((q[:, 0].max(), q[:, 1].min()))
            d.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=3)
    img.save(path)
    return info
