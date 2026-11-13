"""Top-down floor plan of placed maps, and the 3D overlaps between them."""
import colorsys
import re
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import bsp
from .graph import INCH

FLOOR_NZ = 0.7  # faces steeper than this are walls
CELL = 32       # overlap grid, units
BAND = 64       # vertical tolerance for overlap, units
EXACT = 4       # floor heights within this many units count as the same floor
# Xen textures (named after the Xen chapter, c4a*, or xeno*), after GoldSrc's animation, random-tile,
# water and decal prefixes: floors in them are Xen pockets or alien growth, not facility space.
XEN = re.compile(r"^[-+!~{]?[0-9a-j]?(xeno|c4a)")


def floors(b, offset, facility=False):
    """World floor polygons; facility=True leaves out Xen floors and pieces of the map that are not the
    facility (bmab.pieces: Xen pockets, sealed rooms the player never sees)."""
    if facility:
        from . import pieces
        pieces.analyse(b)
    return [f.verts + offset for f in b.faces
            if f.model == 0 and not f.skybox and f.normal[2] > FLOOR_NZ and not bsp.is_tool(f.texture)
            and not (facility and (f.outside or XEN.search(f.texture.lower())))]


def placed_floors(maps, graph, component):
    names = [n for n, m in graph["maps"].items() if m["component"] == component]
    return {n: floors(maps[n], np.array(graph["maps"][n]["offset"])) for n in names}


def _bounds(polys):
    v = np.concatenate([p for ps in polys.values() for p in ps])
    return v.min(0), v.max(0)


def chapter(name):
    m = re.search(r"c\d+a\d+", name)
    return m.group(0) if m else name[:4]


def render(polys, path, size=4000, group=chapter):
    lo, hi = _bounds(polys)
    span = hi - lo
    scale = size / max(span[0], span[1])
    w, h = int(span[0] * scale) + 40, int(span[1] * scale) + 40
    img = Image.new("RGB", (w, h), (12, 12, 16))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    groups = sorted({group(n) for n in polys})
    hue = {g: i / len(groups) for i, g in enumerate(groups)}

    def px(v):
        return [(20 + (x - lo[0]) * scale, h - 20 - (y - lo[1]) * scale) for x, y, *_ in v]

    def rgb(g, light):
        return tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue[g], light, 0.65))

    for z, n, p in sorted(((p[:, 2].mean(), n, p) for n, ps in polys.items() for p in ps), key=lambda t: t[0]):
        draw.polygon(px(p), fill=rgb(group(n), 0.25 + 0.6 * (z - lo[2]) / max(span[2], 1)))
    for n, ps in polys.items():
        if ps:
            draw.text(px([np.concatenate(ps).mean(0)])[0], n, fill=(255, 255, 255), font=font)
    for i, g in enumerate(groups):
        draw.rectangle([20, 20 + i * 14, 30, 30 + i * 14], fill=rgb(g, 0.55))
        draw.text((36, 19 + i * 14), g, fill=(230, 230, 230), font=font)
    draw.text((20, h - 16), f"{max(span[0], span[1]) * INCH:.0f} m across; brighter is higher",
              fill=(200, 200, 200), font=font)
    img.save(path)
    return {"width_m": round(span[0] * INCH), "depth_m": round(span[1] * INCH), "height_m": round(span[2] * INCH)}


def overlaps(polys, linked):
    """Pairs of maps with floor in the same place; same_height near 1 suggests one room built twice."""
    lo, hi = _bounds(polys)
    gw, gh = int((hi[0] - lo[0]) / CELL) + 2, int((hi[1] - lo[1]) / CELL) + 2
    near, exact = {}, {}
    for n, ps in polys.items():
        by_height = defaultdict(list)
        for p in ps:
            by_height[int(round((p[:, 2].mean() - lo[2]) / EXACT))].append(p)
        nk, ek = [], []
        for hz, group in by_height.items():
            cells = _raster(group, lo, gw)
            ek.append(hz * gw * gh + cells)
            band = hz * EXACT // BAND
            nk += [band * gw * gh + cells, (band + 1) * gw * gh + cells]
        near[n] = np.unique(np.concatenate(nk)) if nk else np.zeros(0, np.int64)
        exact[n] = np.unique(np.concatenate(ek)) if ek else np.zeros(0, np.int64)

    cell_m2 = (CELL * INCH) ** 2
    out = []
    names = sorted(polys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = np.intersect1d(near[a], near[b], assume_unique=True).size / 2
            if shared:
                same = np.intersect1d(exact[a], exact[b], assume_unique=True).size
                out.append({"maps": [a, b], "linked": (a, b) in linked or (b, a) in linked,
                            "m2": round(shared * cell_m2), "same_height": round(min(1.0, same / shared), 2)})
    return sorted(out, key=lambda r: -r["m2"])


def _raster(group, lo, gw):
    pmin = np.min([p.min(0) for p in group], 0)
    pmax = np.max([p.max(0) for p in group], 0)
    x0, y0 = int((pmin[0] - lo[0]) // CELL), int((pmin[1] - lo[1]) // CELL)
    x1, y1 = int((pmax[0] - lo[0]) // CELL) + 1, int((pmax[1] - lo[1]) // CELL) + 1
    m = Image.new("1", (x1 - x0 + 1, y1 - y0 + 1))
    d = ImageDraw.Draw(m)
    for p in group:
        d.polygon([((x - lo[0]) / CELL - x0, (y - lo[1]) / CELL - y0) for x, y, _ in p], fill=1)
    yy, xx = np.nonzero(np.array(m))
    return (yy + y0) * gw + (xx + x0)


def elevation(polys, path, axis=0, size=4000, group=chapter, marks=(), bands=()):
    """Side view: each floor drawn as a bar at its height along one horizontal axis (0 = x: looking north,
    1 = y: looking east). Colour by group, as in render.
    marks: (x0, x1, z, label) drawn as white bars (e.g. sky openings); bands: (z0, z1, label) shaded across."""
    v = np.concatenate([p for ps in polys.values() for p in ps])
    lo, hi = v.min(0), v.max(0)
    for x0, x1, z, _ in marks:
        lo[axis], hi[axis] = min(lo[axis], x0), max(hi[axis], x1)
        lo[2], hi[2] = min(lo[2], z), max(hi[2], z)
    for z0, z1, _ in bands:
        lo[2], hi[2] = min(lo[2], z0), max(hi[2], z1)
    span = hi - lo
    scale = size / max(span[axis], span[2])
    w, h = int(span[axis] * scale) + 40, int(span[2] * scale) + 80
    img = Image.new("RGB", (w, h), (12, 12, 16))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)
    groups = sorted({group(n) for n in polys})
    hue = {g: i / len(groups) for i, g in enumerate(groups)}
    for z0, z1, label in bands:
        y0, y1 = h - 40 - (z1 - lo[2]) * scale, h - 40 - (z0 - lo[2]) * scale
        draw.rectangle([0, y0, w, y1], fill=(28, 40, 28))
        draw.text((w - 400, y0 + 4), label, fill=(150, 210, 150), font=font)
    for n, ps in polys.items():
        r, g, b = (int(c * 255) for c in colorsys.hls_to_rgb(hue[group(n)], 0.55, 0.65))
        for p in ps:
            z = p[:, 2].mean()
            x0, x1 = p[:, axis].min(), p[:, axis].max()
            y = h - 40 - (z - lo[2]) * scale
            draw.rectangle([20 + (x0 - lo[axis]) * scale, y - 1, 20 + (x1 - lo[axis]) * scale, y + 1], fill=(r, g, b))
    for x0, x1, z, label in marks:
        y = h - 40 - (z - lo[2]) * scale
        xa, xb = 20 + (x0 - lo[axis]) * scale, 20 + (x1 - lo[axis]) * scale
        draw.rectangle([xa, y - 2, xb, y + 2], fill=(255, 255, 255))
        draw.text((xa, y - 22), label, fill=(255, 255, 255), font=font)
    for i, gname in enumerate(groups):
        r, g, b = (int(c * 255) for c in colorsys.hls_to_rgb(hue[gname], 0.55, 0.65))
        draw.rectangle([20, 20 + i * 22, 34, 34 + i * 22], fill=(r, g, b))
        draw.text((40, 18 + i * 22), gname, fill=(230, 230, 230), font=font)
    label = "looking north (x across)" if axis == 0 else "looking east (y across)"
    draw.text((20, h - 30), f"side view, {label}; {span[axis] * INCH:.0f} m across, {span[2] * INCH:.0f} m high",
              fill=(220, 220, 220), font=font)
    img.save(path)
