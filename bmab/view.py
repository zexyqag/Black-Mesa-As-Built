"""bmab's HTML pages (plain WebGL from webgl.js, no outside code):
- export: a solved layout in 3D, one self-contained file: every map at its solved offset, coloured by game,
  floors, walls and ceilings switchable, a height slice, the ground band and the stretched joins.
- map_view: one map as the review page (inspector.html, served by bmab.serve) draws it, from the game's files alone: its faces
  with their real textures (one WebP sheet and texture coordinates), each face's area (bmab.pieces, as its rules find them), its
  tracks and the models it places. What people say about it (claims) the page asks for as it shows the map.
The pages hold Valve's and the modders' geometry: they are for viewing on this machine, never for publishing.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np

from . import redlines, brushes, bsp, pieces, solve, sources
from .graph import INCH

HERE = Path(__file__).parent
GAMES = {"half-life": ("Half-Life", "#3ec9d6"), "opposing-force": ("Opposing Force", "#9b6cff"),
         "blue-shift": ("Blue Shift", "#ff6b5b"), "ps2-decay": ("Decay", "#b6e05a")}
OTHER = ["#ffb84d", "#ff7bd0", "#7bd389", "#c9c9c9"]  # sources without a set colour, in order

def drawn(b):
    """The faces a page draws, in the order it packs them: [(index into b.faces, face, corners where the map has them, unit normal)]."""
    return [(i, f, *brushes.place(b, f)) for i, f in enumerate(b.faces) if brushes.drawn(b, f)]


_images = {}


def texture_images(source):
    """Every texture a GoldSrc source's WADs hold (its base game's included): {name: RGBA image}."""
    if source not in _images:
        from . import wad
        manifest, _ = sources.load()
        out = {}
        for sid in [manifest[source].get("base"), source]:
            for w in sorted((sources.DATA / sid).glob("*.wad")) if sid else []:
                out.update(wad.wad(w))
        _images[source] = out
    return _images[source]


def _source_images(source, names):
    """A Source game's textures for these material names: {name: (image scaled for the sheet, its real width and height)}."""
    from . import vtf
    manifest, _ = sources.load()
    folders = [sources.DATA / s for s in (source, manifest[source].get("base")) if s]
    out = {}
    for n in names:
        if (source, n) not in _source_cache:
            tex, see_through = vtf.material(folders, n)
            if tex is not None and (tex, see_through) not in _source_cache:  # a map's own copies of a material share its texture
                data = tex.read_bytes()
                img = vtf.image(data, SHEET_MAX)
                if img is not None and not see_through:  # otherwise a texture's alpha is its shine
                    img.putalpha(255)
                _source_cache[tex, see_through] = (img, vtf.size(data)) if img is not None else None
            _source_cache[source, n] = _source_cache[tex, see_through] if tex is not None else None
        if _source_cache[source, n] is not None:
            out[n] = _source_cache[source, n]
    return out


def _mean(img):
    return list(img.convert("RGB").resize((1, 1), 4).getpixel((0, 0)))


SHEET_MAX = 256  # textures bigger than this on their longer side are scaled down in the sheet
WEBP_MAX = 16383
SHEET_PIXELS = 4096 * 4096  # what one sheet may hold: encoding a bigger one runs out of memory
_source_cache = {}


def _sheet(images):
    """Pack images (name -> RGBA image) on shelves into one WebP. Returns its bytes and {name: [x, y, w, h]}. Many textures get smaller
    places, as the page draws them smaller too, and a sheet that would pass the format's limit has them halved again."""
    from PIL import Image
    most = SHEET_MAX if len(images) <= 300 else SHEET_MAX // 2
    while most > 8 and len(images) * most * most > SHEET_PIXELS:
        most //= 2
    while True:
        small = {}
        for n, img in images.items():
            f = min(1.0, most / max(img.size))
            small[n] = img if f == 1 else img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), 1)
        width = int(min(WEBP_MAX, max(2048, 1.15 * sum(i.width * i.height for i in small.values()) ** 0.5)))
        x = y = row = 0
        rects = {}
        for n in sorted(small, key=lambda n: -small[n].height):
            w, h = small[n].size
            if x + w > width:
                x, y, row = 0, y + row, 0
            rects[n] = [x, y, w, h]
            x, row = x + w, max(row, h)
        if y + row <= WEBP_MAX or most <= 8:
            break
        most //= 2
    sheet = Image.new("RGBA", (width, max(1, y + row)))
    for n, (x, y, _w, _h) in rects.items():
        sheet.paste(small[n], (x, y))
    buf = io.BytesIO()
    sheet.save(buf, "WEBP", lossless=True, quality=60, method=4)
    return buf.getvalue(), rects


def _pack(faces, colours=None, uvs=None, layers=None):
    """faces: (vertices, normal, item); colours: an RGB triple per face, optional; uvs and layers (with
    colours): texture coordinates per vertex in repeats of the texture and the texture's index per face
    (65535: none). Returns the binary blob, its layout, and the frame (centre, quant)."""
    pos, nrm, item, kind, idx, col, uv, lay = [], [], [], [], [], [], [], []
    count = 0
    for fi, (v, normal, it) in enumerate(faces):
        k = len(v)
        if colours is not None:
            col.append(np.tile(np.asarray(colours[fi], np.uint8), (k, 1)))
        if uvs is not None:
            uv.append(np.asarray(uvs[fi], np.float32))
            lay.append(np.full(k, layers[fi], np.uint16))
        pos.append(v)
        nrm.append(np.repeat(normal[None], k, 0))
        item.append(np.full(k, it, np.uint16))
        kind.append(np.full(k, 0 if normal[2] > 0.7 else 2 if normal[2] < -0.7 else 1, np.uint8))
        j = np.arange(1, k - 1)
        idx.append(np.stack([np.full(k - 2, count), count + j, count + j + 1], 1).ravel())
        count += k
    pos = np.concatenate(pos)
    lo, hi = pos.min(0), pos.max(0)
    centre = (lo + hi) / 2
    quant = max(1.0, float(np.ceil((hi - lo).max() / 64000)))  # units per int16 step
    parts = [np.round((pos - centre) / quant).astype(np.int16).ravel(),
             np.round(np.concatenate(nrm) * 127).astype(np.int8).ravel(),
             np.concatenate(item), np.concatenate(kind), np.concatenate(idx).astype(np.uint32)]
    if colours is not None:
        parts.append(np.concatenate(col).ravel())
    if uvs is not None:
        parts += [np.concatenate(uv).ravel(), np.concatenate(lay)]
    blob, layout = b"", []
    for a in parts:
        blob += b"\0" * (-len(blob) % 4)
        layout.append([len(blob), int(a.size)])
        blob += a.tobytes()
    return blob, layout, centre, quant, int(parts[4].size // 3)


def _page(template, meta, **parts):
    """A page from its template, with the scripts the pages share (webgl.js, signin.js) and its meta in it."""
    page = (HERE / template).read_text(encoding="utf-8")
    for mark, script in (("/*WEBGL*/", "webgl.js"), ("/*SIGNIN*/", "signin.js")):
        page = page.replace(mark, (HERE / script).read_text(encoding="utf-8"))
    page = page.replace("/*META*/null", json.dumps(meta))
    for k, v in parts.items():
        page = page.replace(k, v)
    return page


def export(report, games, path, band=None, title="", routes=None, fillers=()):
    """report: a solve/attach report; games: {map or room of a cut map: (game name, bsp or room)}; band: (z low, z high, label)
    in units. routes: bmab.routes results, drawn in place of the straight joins they replace. fillers: (outline at one side, outline at
    the other), units: what a stretched connector adds, drawn as its four walls."""
    names = [n for n in report["maps"] if n in games]
    order = list(dict.fromkeys(games[n][0] for n in names))
    faces = []
    for i, n in enumerate(names):
        b = games[n][1]
        pieces.analyse(b, redlines.said(games[n][0], n, "area"))
        off = np.array(report["offsets"][n], float)
        faces += [(v + off, nrm, i) for _k, f, v, nrm in drawn(b) if not f.outside]
    for a, z in fillers:
        for k in range(4):
            quad = np.array([a[k], a[(k + 1) % 4], z[(k + 1) % 4], z[k]], float)
            normal = np.cross(quad[1] - quad[0], quad[3] - quad[0])
            faces.append((quad, normal / max(float(np.linalg.norm(normal)), 1e-9), len(names)))
    blob, layout, centre, quant, tris = _pack(faces)

    def local(p):
        return [round(float(x), 1) for x in (np.asarray(p, float) - centre) / quant]
    allv = np.concatenate([v for v, _n, _i in faces])
    joins = []
    routed = {tuple(r["maps"]) for r in routes or [] if r["cells"]}
    for j in report["joins"]:
        a, b = j["maps"]
        if a in games and b in games and j["kind"] not in solve.INNER and tuple(j["maps"]) not in routed:
            pa = solve._landmark(games[a][1], j["landmark"])
            pb = solve._landmark(games[b][1], j["landmark"])
            if pa is not None and pb is not None:
                joins.append({"a": local(pa + report["offsets"][a]), "b": local(pb + report["offsets"][b]),
                              "label": f"{a}/{b} {j['kind']} {j['unseen_m']:.0f} m", "inserted": bool(j.get("inserted"))})
    inserted = {tuple(j["maps"]) for j in report["joins"] if j.get("inserted")}
    for r in routes or []:
        if r["cells"] and r["kind"] not in solve.INNER:
            pts = [local((np.array(c, float) + 0.5) * ROUTE_CELL) for c in r["cells"]]
            for k, (a, b) in enumerate(zip(pts, pts[1:])):
                joins.append({"a": a, "b": b, "inserted": tuple(r["maps"]) in inserted,
                              "label": f"{r['maps'][0]}/{r['maps'][1]} {r['kind']} {r['route_m']:.0f} m" if k == 0 else ""})
    meta = {"title": title, "quant": quant, "metres": quant * INCH, "lo": local(allv.min(0)), "hi": local(allv.max(0)),
            "maps": [{"name": n, "game": order.index(games[n][0])} for n in names],
            "games": [{"name": GAMES.get(g, (g,))[0], "colour": GAMES[g][1] if g in GAMES else OTHER[k % len(OTHER)]}
                      for k, g in enumerate(order)],
            "layout": layout, "joins": joins,
            "band": None if band is None else {"lo": (band[0] - centre[2]) / quant, "hi": (band[1] - centre[2]) / quant,
                                               "label": band[2]}}
    if fillers:
        meta["maps"].append({"name": "filler of a stretched connector", "game": len(order)})
        meta["games"].append({"name": "Filler (stretched connectors)", "colour": "#6fe08a"})
    page = _page("view.html", meta, GEOMETRY=base64.b64encode(blob).decode())
    Path(path).write_text(page, encoding="utf-8", newline="\n")
    return {"maps": len(names), "triangles": tris, "mb": round(len(page) / 1e6, 1)}


def _textured(g, n, b, shown, faces):
    """Pack a map's faces with its real textures: the blob, its layout, centre, quant, the texture sheet and its rects."""
    from . import wad
    images = dict(texture_images(g))
    path = next((sources.DATA / g / "maps").glob(f"{n}.bsp"), None)
    if path is not None and not isinstance(b.vis, bsp.SourceVis):  # textures stored in a GoldSrc map itself
        images.update(wad.embedded(path.read_bytes())[1])
    sizes = {}
    if isinstance(b.vis, bsp.SourceVis):
        found = _source_images(g, {f.texture for _i, f in shown if f.texture})
        images, sizes = {n: i for n, (i, _s) in found.items()}, {n: s for n, (_i, s) in found.items()}
    used = sorted({f.texture for _i, f in shown if f.texture in images and f.axes is not None})
    means = {t: _mean(images[t]) for t in used}
    pictures = {}
    layer = {t: pictures.setdefault(id(images[t]), len(pictures)) for t in used}  # names sharing a picture share a layer
    by_layer = {layer[t]: images[t] for t in used}
    uvs, layers = [], []
    for _i, f in shown:
        if f.texture in layer:
            w, h = sizes.get(f.texture) or images[f.texture].size
            uvs.append((f.verts @ f.axes[:, :3].T + f.axes[:, 3]) / (w, h))
            layers.append(layer[f.texture])
        else:
            uvs.append(np.zeros((len(f.verts), 2)))
            layers.append(65535)
    blob, layout, centre, quant, _t = _pack(faces, [means.get(f.texture, [128, 128, 128]) for _i, f in shown], uvs, layers)
    sheet, rects = _sheet({str(k): img for k, img in by_layer.items()}) if used else (b"", {})
    return blob, layout, centre, quant, sheet, [rects[str(k)] for k in range(len(by_layer))]


MODEL_BUDGET = 1_500_000  # triangles of a map's model shapes together: within it every model is drawn at full detail


def _cap(counts, budget):
    """The most triangles any one shape may keep for shapes of these counts to stay within budget together (the biggest are cut
    down alike); None if they fit as they are."""
    left, counts = budget, sorted(counts)
    for k, c in enumerate(counts):
        if c * (len(counts) - k) > left:
            return max(left // (len(counts) - k), 1)
        left -= c
    return None


def _models(g, b, centre, quant, n):
    """A map's placed models for the page: (markers [[x, y, z, name index]], the names, and the real shapes or None if it has none:
    each shape's model, pose, skin, body and size (units), and which marker each of its copies is). Source maps place props; GoldSrc
    maps place monsters, items and the like, whose models carry their own pictures."""
    from . import models
    from .game import Game
    manifest, _ = sources.load()
    folders = [sources.DATA / s for s in (g, manifest[g].get("base")) if s]
    gold = not isinstance(b.vis, bsp.SourceVis)
    props = models.gs_placed(b) if gold else models.placed(b, Game(sources.DATA / g).raw(n))
    names = sorted({p["model"] for p in props})
    index = {m: k for k, m in enumerate(names)}
    spot = lambda at: (np.asarray(at, float) - centre) / quant  # noqa: E731
    markers = [[*(round(float(x), 1) for x in spot(p["at"])), index[p["model"]]] for p in props]
    pos, nrm, uv, layer, idx, parts, matrices, wanted, grouped, pictures_of, shapes, copies = [], [], [], [], [], [], [], {}, {}, {}, [], []
    for k, p in enumerate(props):
        grouped.setdefault((p["model"], p["skin"], p.get("body", 0), p.get("pose")), []).append((k, p))
    used = 0
    full = {k: models.gs_read(folders, k[0], k[2]) if gold else models.read(folders, k[0], k[3]) for k in grouped}
    cap = None if gold else _cap([sum(len(t) for _m, t in s["parts"]) for s in full.values() if s is not None], MODEL_BUDGET)
    for (model, skin, body, pose), group in sorted(grouped.items(), key=lambda kv: (*kv[0][:3], kv[0][3] or "")):
        shape = full[(model, skin, body, pose)]
        if shape is None:
            continue
        if cap is not None and sum(len(t) for _m, t in shape["parts"]) > cap:
            shape = models.read(folders, model, pose, cap)
        skins = shape["skins"][skin if skin < len(shape["skins"]) else 0]
        first_idx = sum(len(i) for i in idx)
        faces, layers = [], np.full(len(shape["points"]), 65535, np.uint16)
        for texture, tris in shape["parts"]:
            if gold:  # a GoldSrc part is already in its picture's index, the first skin's: another skin swaps it by the skin table
                t = skins[shape["skins"][0].index(texture)] if texture in shape["skins"][0] else texture
                name = f"{model}#{t}" if t < len(shape["images"]) else None
                if name:
                    pictures_of[name] = (shape["images"][t],)
            else:
                name = models.material(folders, shape, shape["textures"][skins[texture]]) if texture < len(skins) and skins[texture] < len(shape["textures"]) else None
            if name is not None:
                layers[tris.ravel()] = wanted.setdefault(name, len(wanted))
            faces.append(tris)
        lo, step = shape["points"].min(0), max(float(np.ptp(shape["points"], axis=0).max()), 1e-6) / 65534  # each shape's points as whole numbers, its scale in its copies' matrices
        pos.append(np.round((shape["points"] - lo) / step - 32767).astype(np.int16)); nrm.append(np.clip(shape["normals"] * 127, -127, 127).astype(np.int8))
        # a few models' files put a corner millions of repeats out: as far as half floats reach looks the same (a blur of repeats)
        uv.append(np.clip(shape["uvs"], -65504, 65504).astype(np.float16)); layer.append(layers)
        idx.append(np.concatenate(faces).astype(np.uint32).ravel() + used)
        used += len(shape["points"])
        parts.append([first_idx, len(idx[-1]), sum(len(m) for m in matrices), len(group)])
        shapes.append({"model": model, "pose": pose, "skin": skin, "body": body, "size": [round(float(x), 1) for x in np.ptp(shape["points"], axis=0)]})
        copies += [k for k, _p in group]
        rows = []
        for _k, p in group:
            r = models.rotation(p["angles"]) / quant
            rows.append(np.column_stack([r * step, r @ (lo + 32767 * step) + spot(p["at"])]).ravel())
        matrices.append(np.array(rows, np.float32))
    if not parts:
        return markers, names, None
    images = pictures_of if gold else _source_images(g, set(wanted))
    pictures, remap = {}, np.full(len(wanted) + 1, 65535, np.uint16)
    for name, k in wanted.items():
        if name in images:
            remap[k] = pictures.setdefault(id(images[name][0]), len(pictures))
    by_layer = {remap[k]: images[name][0] for name, k in wanted.items() if name in images}
    flat = np.concatenate(layer)
    flat = np.where(flat == 65535, 65535, remap[np.minimum(flat, len(wanted))]).astype(np.uint16)
    arrays = [np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv).view(np.uint16), flat, np.concatenate(idx), np.concatenate(matrices)]
    blob, layout = bytearray(), []
    for a in arrays:
        pad = -len(blob) % 8
        blob += bytes(pad)
        layout.append([len(blob), int(a.size)])
        blob += a.tobytes()
    sheet, rects = _sheet({str(k): img for k, img in by_layer.items()}) if by_layer else (b"", {})
    data = {"name": n, "source": g, "b64": base64.b64encode(bytes(blob)).decode(), "layout": layout, "parts": parts, "shapes": shapes, "copies": copies,
            "sheet": "data:image/webp;base64," + base64.b64encode(sheet).decode() if sheet else None, "tex": [rects[str(k)] for k in range(len(by_layer))]}
    return markers, names, data


ROUTE_CELL = 4 * solve.CELL  # as bmab.routes


def map_view(game, name, b):
    """One map as the review page draws it: b, the game's map name as read, its pieces analysed (bmab.pieces: the areas shown are what
    its rules find, whatever people say). Returns the view (for the page's mapLoaded), its models' real shapes (for modelsLoaded, None
    if it places none) and its frame: {centre, quant, middle} (units), which the page's local coordinates are counted from."""
    from . import connectors as C
    placed = drawn(b)
    if not placed:
        raise ValueError(f"{name} has nothing to draw")
    shown = [(i, f) for i, f, _v, _n in placed]
    area_of = [int(b.face_piece[i]) if b.face_piece is not None else 0 for i, _f in shown]
    item_area = sorted(set(area_of))
    item = {a: k for k, a in enumerate(item_area)}
    faces = [(v, nrm, item[a]) for (_i, _f, v, nrm), a in zip(placed, area_of)]
    blob, layout, centre, quant, sheet, tex = _textured(game, name, b, shown, faces)
    allv = np.concatenate([v for v, _n, _i in faces])

    def local(p):
        return [round(float(x), 1) for x in (np.asarray(p, float) - centre) / quant]

    def area(p):
        return {**p, **p["found"], "at": [round(x) for x in p["at"]], "plo": local(p["lo"]), "phi": local(p["hi"])}
    markers, names, models = _models(game, b, centre, quant, name)
    view = {"name": name, "source": game, "layout": layout, "b64": base64.b64encode(blob).decode(),
            "sheet": "data:image/webp;base64," + base64.b64encode(sheet).decode() if sheet else None, "tex": tex,
            "lo": local(allv.min(0)), "hi": local(allv.max(0)), "quant": quant, "centre": [round(float(x), 2) for x in centre],
            "per_metre": 1 / (quant * INCH), "cell_half": solve.CELL / 2 / quant, "item_area": item_area,
            "areas": [area(p) for p in b.pieces or []], "tracks": [[*local(a), *local(z)] for a, z in C._paths(b)], "props": markers, "prop_names": names}
    return view, models, {"centre": [float(x) for x in centre], "quant": quant, "middle": [float(x) for x in allv.mean(0)]}
