"""Measurements a map's geometry and entities give directly: ride paths and open sky.

Rides: func_train (path_corner chains), func_tracktrain (path_track chains), func_plat, and
func_door that moves up or down. Lengths are in map units; INCH converts to metres. Speeds are
gameplay pacing, so durations are soft evidence; lengths are geometry.
"""
import numpy as np

from .graph import INCH


def _origin(e):
    return np.array([float(x) for x in e["origin"].split()]) if "origin" in e else None


def _model_box(b, e):
    m = e.get("model", "")
    if m.startswith("*") and int(m[1:]) < len(b.models):
        lo, hi, *_ = b.models[int(m[1:])]
        return lo, hi
    return None


def _follow(by_name, start, max_steps=2000):
    """Corners from start along `target`, stopping at the end or when the path loops."""
    path, seen, name = [], set(), start
    while name and name in by_name and name not in seen and len(path) < max_steps:
        seen.add(name)
        path.append(by_name[name])
        name = by_name[name].get("target")
    return path, bool(name and name in seen)


def _path_stats(corners, base_speed):
    pts = [p for p in (_origin(c) for c in corners) if p is not None]
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    speed, seconds, speeds = base_speed, 0.0, set()
    for c, d in zip(corners[:-1], seg):
        s = float(c.get("speed") or 0)
        if s > 0:
            speed = s
        if speed > 0:
            seconds += d / speed
            speeds.add(speed)
        seconds += max(0.0, float(c.get("wait") or 0))
    return {
        "corners": len(pts),
        "length_m": round(float(seg.sum()) * INCH, 1),
        "rise_m": round(float(pts[-1, 2] - pts[0, 2]) * INCH, 1),
        "z_span_m": round(float(np.ptp(pts[:, 2])) * INCH, 1),
        "xy_span_m": round(float(np.linalg.norm(np.ptp(pts[:, :2], axis=0))) * INCH, 1),
        "speeds": sorted(speeds),
        "seconds": round(seconds, 1) if speeds else None,
        "start": [round(float(x)) for x in pts[0]],
        "end": [round(float(x)) for x in pts[-1]],
    }


def rides(b):
    by_name = {}
    for e in b.entities:
        if e.get("classname") in ("path_corner", "path_track") and e.get("targetname"):
            by_name[e["targetname"]] = e
    out = []
    for e in b.entities:
        cls = e.get("classname", "")
        rec = None
        if cls in ("func_train", "func_tracktrain") and e.get("target"):
            corners, loops = _follow(by_name, e["target"])
            stats = _path_stats(corners, float(e.get("speed") or 0))
            if stats:
                rec = {**stats, "loops": loops}
        elif cls == "func_plat":
            box = _model_box(b, e)
            height = float(e.get("height") or 0) or (box and float(box[1][2] - box[0][2]) - 8)
            if height:
                speed = float(e.get("speed") or 150)
                rec = {"length_m": round(height * INCH, 1), "rise_m": round(height * INCH, 1),
                       "z_span_m": round(height * INCH, 1), "speeds": [speed], "seconds": round(height / speed, 1)}
        elif cls == "func_door":
            angle = e.get("angle", "")
            pitch = e.get("angles", "0 0 0").split()[0]
            vertical = angle in ("-1", "-2") or pitch in ("-90", "90")
            box = _model_box(b, e)
            if vertical and box is not None:
                travel = float(box[1][2] - box[0][2]) - float(e.get("lip") or 0)
                speed = float(e.get("speed") or 100)
                up = angle == "-1" or pitch == "-90"
                if travel * INCH >= 2:  # smaller vertical doors are hatches and gates, not lifts
                    rec = {"length_m": round(travel * INCH, 1), "rise_m": round((travel if up else -travel) * INCH, 1),
                           "z_span_m": round(travel * INCH, 1), "speeds": [speed], "seconds": round(travel / speed, 1)}
        if rec:
            out.append({"map": b.name, "class": cls, "name": e.get("targetname", ""), **rec})
    return out


def tracks(b):
    """The whole path_track network of a map (every segment, not just ahead of a train) and its track lifts.

    Track lifts (func_trackchange, func_trackautochange) move a train between two tracks by `height`.
    """
    pts = {e["targetname"]: e for e in b.entities if e.get("classname") == "path_track" and e.get("targetname")}
    segments = []
    for e in pts.values():
        nxt = pts.get(e.get("target"))
        if nxt is not None and "origin" in e and "origin" in nxt:
            segments.append([[round(float(x)) for x in _origin(e)], [round(float(x)) for x in _origin(nxt)]])
    lifts = []
    for e in b.entities:
        if e.get("classname") in ("func_trackchange", "func_trackautochange") and e.get("height"):
            h = float(e["height"])
            lifts.append({"name": e.get("targetname", ""), "top": e.get("toptrack"), "bottom": e.get("bottomtrack"),
                          "height_m": round(h * INCH, 1), "seconds": round(h / float(e.get("speed") or 100), 1)})
    length = sum(float(np.linalg.norm(np.subtract(b2, a))) for a, b2 in segments)
    return {"map": b.name, "length_m": round(length * INCH, 1), "segments": segments, "lifts": lifts}


def sky(b, is_tool):
    """Area of sky-textured faces in the world, and how high the openings sit in the map."""
    sky_area, sky_z, floor_z = 0.0, [], []
    for f in b.faces:
        if f.model != 0:
            continue
        if f.texture == "sky" or f.texture.startswith("tools/toolsskybox"):
            v = f.verts
            area = 0.5 * np.linalg.norm(np.cross(v[1:-1] - v[0], v[2:] - v[0]).sum(axis=0))
            sky_area += area
            sky_z.append(v[:, 2].min())
        elif not is_tool(f.texture) and f.normal[2] > 0.7 and not getattr(f, "skybox", False):
            floor_z.append(f.verts[:, 2].mean())
    rec = {"map": b.name, "sky_m2": round(sky_area * INCH * INCH)}
    if floor_z:
        rec["floor_z_m"] = [round(float(min(floor_z)) * INCH, 1), round(float(max(floor_z)) * INCH, 1)]
    if sky_z:
        rec["lowest_sky_z_m"] = round(float(min(sky_z)) * INCH, 1)
    return rec
