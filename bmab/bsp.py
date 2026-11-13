"""Minimal BSP reader for GoldSrc (v30), its PlayStation 2 variant (v40) and Source (VBSP v20, LZMA lumps too).

Reads entities, faces, texture names and models, and the BSP tree with its visibility (GoldSrc: per leaf; Source:
per cluster, plus the solid brushes, as Source's detail brushes don't fill the tree's leaves).
"""
import lzma
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LUMPS = ["entities", "planes", "textures", "vertices", "visibility", "nodes", "texinfo",
         "faces", "lighting", "clipnodes", "leaves", "marksurfaces", "edges", "surfedges", "models"]

# Brush textures that are not visible geometry.
TOOL_TEXTURES = {"aaatrigger", "sky", "clip", "origin", "null", "hint", "skip", "bevel"}

_ENT_TOKEN = re.compile(r'"([^"]*)"|([{}])')


@dataclass
class Face:
    model: int
    normal: np.ndarray
    verts: np.ndarray  # (n, 3)
    texture: str
    skybox: bool = False  # part of a Source 3D skybox, not the playable map
    outside: bool = False  # in a piece of the map that is not the facility (bmab.pieces)
    axes: "np.ndarray | None" = None  # texture axes (2, 4): texel s, t = verts @ axes[:, :3].T + axes[:, 3]


@dataclass
class Bsp:
    name: str
    entities: list
    faces: list = field(default_factory=list)
    models: list = field(default_factory=list)  # (mins, maxs, firstface, numfaces)
    overlays: list = field(default_factory=list)  # Source only: (material, origin, normal)
    vis: "Vis | None" = None  # GoldSrc only
    pieces: "list | None" = None  # bmab.pieces.analyse, once run
    face_piece: "np.ndarray | None" = None  # with it: the piece each face belongs to (0: none)
    piece_grid: "tuple | None" = None  # with it: the piece of each cell (0: none) and the first cell's index


GS_NODE = np.dtype([("plane", "<i4"), ("children", "<i2", 2), ("mins", "<i2", 3), ("maxs", "<i2", 3),
                    ("firstface", "<u2"), ("numfaces", "<u2")])
GS_LEAF = np.dtype([("contents", "<i4"), ("visofs", "<i4"), ("mins", "<i2", 3), ("maxs", "<i2", 3),
                    ("firstmark", "<u2"), ("nummarks", "<u2"), ("ambient", "u1", 4)])


class _Tree:
    def leaf_of(self, points):
        """Leaf index for each point (n x 3), walking the tree for all points at once."""
        points = np.atleast_2d(points)
        node = np.full(len(points), self.head, np.int64)
        todo = node >= 0
        while todo.any():
            nd = self.nodes[node[todo]]
            pl = self.planes[nd["plane"]]
            side = (np.einsum("ij,ij->i", points[todo], pl["n"]) - pl["d"]) < 0
            node[todo] = nd["children"][np.arange(len(nd)), side.astype(int)]
            todo = node >= 0
        return -node - 1


def _rle(data, ofs, need):
    """A visibility row: zero bytes are run-length coded (0, count)."""
    row, i = bytearray(), ofs
    while len(row) < need:
        if data[i]:
            row.append(data[i])
            i += 1
        else:
            row.extend(b"\0" * data[i + 1])
            i += 2
    return np.unpackbits(np.frombuffer(bytes(row[:need]), np.uint8), bitorder="little")


@dataclass
class Vis(_Tree):
    """GoldSrc BSP tree and potentially visible sets: which leaves can be seen from which."""
    planes: np.ndarray
    nodes: np.ndarray
    leaves: np.ndarray
    data: bytes
    visleafs: int  # leaves 1..visleafs have visibility rows
    head: int

    def visible(self, leaf):
        """Leaves potentially visible from a leaf, as a boolean array over all leaves."""
        out = np.zeros(len(self.leaves), bool)
        ofs = int(self.leaves[leaf]["visofs"])
        if leaf == 0 or ofs < 0:
            out[1:self.visleafs + 1] = ofs < 0 and leaf != 0
            return out
        out[1:self.visleafs + 1] = _rle(self.data, ofs, (self.visleafs + 7) // 8)[:self.visleafs].astype(bool)
        return out

    def open(self, points):
        """Whether each point is in open space (empty or liquid): sight passes, the player could be there."""
        return np.isin(self.leaves["contents"][self.leaf_of(points)], (-1, -3, -4, -5))

    def solid_leaves(self, leaves):
        return np.isin(self.leaves["contents"][leaves], (-2, -6))  # solid, sky

    def liquid_leaves(self, leaves):
        return np.isin(self.leaves["contents"][leaves], (-3, -4, -5))  # water, slime, lava


SOURCE_SOLID, SOURCE_LIQUID = 0x1, 0x10 | 0x20  # contents bits: solid; slime, water
SOLID_CELL, SOLID_PAD = 16, 8


@dataclass
class SourceVis(_Tree):
    """Source BSP tree, visibility by cluster, and the solid brushes. Detail brushes (most of a Source map's walls)
    don't fill the tree's leaves, so open() also asks a grid of all solid brushes, built on first use: a cell is
    solid when its centre is within SOLID_PAD of a brush, so sight lines can't slip through thin walls."""
    planes: np.ndarray
    nodes: np.ndarray
    leaves: np.ndarray
    data: bytes  # the visibility lump
    head: int
    brushes: np.ndarray
    sides: np.ndarray
    leafbrushes: np.ndarray
    bounds: tuple  # (lo, hi) of the playable world (the 3D skybox left out)
    _grid: "tuple | None" = None

    def visible(self, leaf):
        out = np.zeros(len(self.leaves), bool)
        c = int(self.leaves[leaf]["cluster"])
        n = struct.unpack_from("<i", self.data, 0)[0] if len(self.data) >= 4 else 0
        if c < 0 or c >= n:
            return out
        bits = _rle(self.data, struct.unpack_from("<i", self.data, 4 + 8 * c)[0], (n + 7) // 8)[:n].astype(bool)
        lc = self.leaves["cluster"].astype(np.int64)
        out[lc >= 0] = bits[lc[lc >= 0]]
        return out

    def grid(self):
        if self._grid is None:
            lo = np.floor(np.asarray(self.bounds[0]) / SOLID_CELL) * SOLID_CELL
            shape = np.ceil((np.asarray(self.bounds[1]) - lo) / SOLID_CELL).astype(int) + 1
            g = np.zeros(shape, bool)
            lv = self.leaves
            owner = np.repeat(np.arange(len(lv)), lv["numleafbrushes"])
            entry = np.concatenate([np.arange(f, f + k) for f, k in zip(lv["firstleafbrush"], lv["numleafbrushes"])] or [[]]).astype(int)
            bmin = np.full((len(self.brushes), 3), np.inf)
            bmax = np.full((len(self.brushes), 3), -np.inf)
            ids = self.leafbrushes[entry].astype(int)
            np.minimum.at(bmin, ids, lv["mins"][owner].astype(float))
            np.maximum.at(bmax, ids, lv["maxs"][owner].astype(float))
            for k in np.flatnonzero(self.brushes["contents"] & SOURCE_SOLID):
                if not np.isfinite(bmin[k]).all():
                    continue
                sd = self.sides[self.brushes[k]["firstside"]:self.brushes[k]["firstside"] + self.brushes[k]["numsides"]]
                pl = self.planes[sd["plane"]]
                box_lo, box_hi = bmin[k].copy(), bmax[k].copy()  # the leaves' box can be far bigger than the brush
                for axis in range(3):  # the compiler adds axial bevel planes: they give the brush's own box
                    up, down = pl["n"][:, axis] > 0.999, pl["n"][:, axis] < -0.999
                    if up.any():
                        box_hi[axis] = min(box_hi[axis], pl["d"][up].min())
                    if down.any():
                        box_lo[axis] = max(box_lo[axis], (-pl["d"][down]).max())
                a = np.clip(np.floor((box_lo - SOLID_PAD - lo) / SOLID_CELL).astype(int), 0, shape - 1)
                z = np.clip(np.ceil((box_hi + SOLID_PAD - lo) / SOLID_CELL).astype(int), 0, shape - 1)
                yz = np.stack(np.meshgrid(np.arange(a[1], z[1] + 1), np.arange(a[2], z[2] + 1), indexing="ij"), -1).reshape(-1, 2)
                for x in range(a[0], z[0] + 1):  # a slab at a time, for the memory
                    c = np.column_stack([np.full(len(yz), x), yz])
                    inside = (((lo + (c + 0.5) * SOLID_CELL) @ pl["n"].T - pl["d"]) <= SOLID_PAD).all(1)
                    g[tuple(c[inside].T)] = True
            self._grid = (lo, g)
        return self._grid

    def release(self):
        self._grid = None

    def open(self, points):
        points = np.atleast_2d(points)
        ok = (self.leaves["contents"][self.leaf_of(points)] & SOURCE_SOLID) == 0
        lo, g = self.grid()
        c = np.floor((points - lo) / SOLID_CELL).astype(int)
        inside = ((c >= 0) & (c < g.shape)).all(1)
        ok[inside] &= ~g[tuple(c[inside].T)]
        return ok

    def solid_leaves(self, leaves):
        return (self.leaves["contents"][leaves] & SOURCE_SOLID) != 0

    def liquid_leaves(self, leaves):
        return (self.leaves["contents"][leaves] & SOURCE_LIQUID) != 0


def parse_entities(text):
    """Braces inside quoted values (decal names like {c2a2_a) are part of the value. A key given again (Source's
    outputs) keeps its first value; the others are listed under "_more" as (key, value)."""
    ents, cur, key = [], None, None
    for quoted, brace in _ENT_TOKEN.findall(text):
        if brace == "{":
            cur, key = {}, None
        elif brace == "}":
            if cur is not None:
                ents.append(cur)
            cur = None
        elif cur is not None:
            if key is None:
                key = quoted
            else:
                if key in cur:  # Source repeats output keys (OnTrigger ...): the first stays, the rest in "_more"
                    cur.setdefault("_more", []).append((key, quoted))
                else:
                    cur[key] = quoted
                key = None
    return ents


def items(e):
    """Every (key, value) of an entity, repeated keys (parse_entities' "_more") included."""
    for k, v in e.items():
        if k == "_more":
            yield from v
        else:
            yield k, v


def values(e):
    return (v for _k, v in items(e))


def outputs(e):
    """Source outputs: (output, target, input) from values like "target\x1bInput\x1bparam\x1bdelay\x1btimes" (older
    maps use commas)."""
    for k, v in items(e):
        sep = "\x1b" if "\x1b" in v else "," if k.startswith("On") and v.count(",") >= 4 else None
        if sep:
            f = v.split(sep)
            yield k, f[0], f[1] if len(f) > 1 else ""


PLANE = np.dtype([("n", "<f4", 3), ("d", "<f4"), ("t", "<i4")])

# Source lump indices used here.
VBSP_LUMPS = {"entities": 0, "planes": 1, "texdata": 2, "vertices": 3, "texinfo": 6, "faces": 7,
              "edges": 12, "surfedges": 13, "models": 14, "texdata_strings": 43, "texdata_table": 44,
              "nodes": 5, "leaves": 10, "leaffaces": 16, "overlays": 45, "visibility": 4, "leafbrushes": 17,
              "brushes": 18, "brushsides": 19, "dispinfo": 26, "disp_verts": 33}
BRUSH = np.dtype([("firstside", "<i4"), ("numsides", "<i4"), ("contents", "<u4")])
BRUSHSIDE = np.dtype([("plane", "<u2"), ("texinfo", "<i2"), ("dispinfo", "<i2"), ("bevel", "<i1"), ("thin", "<i1")])
OVERLAY = np.dtype([("id", "<i4"), ("texinfo", "<i2"), ("count_order", "<u2"), ("faces", "<i4", 64),
                    ("u", "<f4", 2), ("v", "<f4", 2), ("uvpoints", "<f4", (4, 3)), ("origin", "<f4", 3),
                    ("normal", "<f4", 3)])
NODE = np.dtype([("plane", "<i4"), ("children", "<i4", 2), ("mins", "<i2", 3), ("maxs", "<i2", 3),
                 ("firstface", "<u2"), ("numfaces", "<u2"), ("area", "<i2"), ("pad", "<i2")])
LEAF = np.dtype([("contents", "<i4"), ("cluster", "<i2"), ("areaflags", "<i2"), ("mins", "<i2", 3),
                 ("maxs", "<i2", 3), ("firstleafface", "<u2"), ("numleaffaces", "<u2"),
                 ("firstleafbrush", "<u2"), ("numleafbrushes", "<u2"), ("water", "<i2"), ("pad", "<i2")])


# PlayStation 2 (Gearbox) maps, as documented by BS2PC (github.com/Triang3l/bs2pc): 16 lumps, stored as
# offsets, then lengths, then counts. Faces and nodes point at planes and textures by byte offset.
GBX_LUMPS = ["planes", "nodes", "leaves", "edges", "surfedges", "vertices", "hull0", "clipnodes", "models",
             "faces", "marksurfaces", "visibility", "lighting", "textures", "entities", "polygons"]
GBX_FACE = np.dtype([("tv", "<f4", 8), ("side", "<u2"), ("flags", "<u2"), ("texture", "<u4"), ("light", "<u4"),
                     ("plane", "<u4"), ("u0", "<u4"), ("firstedge", "<u4"), ("numedges", "<u4"), ("area", "<f4"),
                     ("tmins", "<i2", 2), ("extents", "<i2", 2), ("u1", "<u4", 7), ("styles", "u1", 4),
                     ("u2", "<u4", 6), ("polygons", "<u4"), ("u3", "<u4", 3)])
FORMATS = ("source", "gs", "ps2")
GS_MODEL = np.dtype([("mins", "<f4", 3), ("maxs", "<f4", 3), ("origin", "<f4", 3), ("head", "<i4", 4), ("visleafs", "<i4"), ("firstface", "<i4"), ("numfaces", "<i4")])
SOURCE_MODEL = np.dtype([("mins", "<f4", 3), ("maxs", "<f4", 3), ("origin", "<f4", 3), ("head", "<i4"), ("firstface", "<i4"), ("numfaces", "<i4")])
GBX_MODEL = np.dtype([("mins4", "<f4", 4), ("maxs4", "<f4", 4), ("origin", "<f4", 4), ("head", "<i4", 4),
                      ("visleafs", "<u4"), ("firstface", "<u4"), ("numfaces", "<u4"), ("u0", "<u4")])


def _unpacked(data):
    """A map file's bytes as the reader wants them: PS2 .bs2 files are stored as their uint32 size, then zlib."""
    if data[4:6] in (b"\x78\x9c", b"\x78\xda", b"\x78\x01") and struct.unpack_from("<I", data, 0)[0] > len(data):
        return zlib.decompress(data[4:])
    return data


def _lumps(path, wanted):
    """Some lumps of a map file, read without the rest of it (a PS2 map is stored packed, so it is unpacked whole). wanted: {format
    ("source", "gs" or "ps2"): the lumps wanted of a map of that format}. Returns its format, {lump: bytes}, and for PS2 its lumps'
    offsets, which its faces point by."""
    with open(path, "rb") as f:
        head = f.read(8 + 64 * 16)

        def at(off, ln):
            f.seek(off)
            return f.read(ln)
        if head[:4] == b"VBSP":
            out = {}
            for k in wanted["source"]:
                off, ln, _ver, size = struct.unpack_from("<iiii", head, 8 + VBSP_LUMPS[k] * 16)
                out[k] = _lzma(at(off, ln), size, Path(path).stem, k) if size else at(off, ln)
            return "source", out, None
        if struct.unpack_from("<i", head, 0)[0] == 30:
            spot = {k: struct.unpack_from("<ii", head, 4 + i * 8) for i, k in enumerate(LUMPS)}
            if at(spot["entities"][0], 1) != b"{" and at(spot["planes"][0], 1) == b"{":  # Blue Shift swapped them
                spot["entities"], spot["planes"] = spot["planes"], spot["entities"]
            return "gs", {k: at(*spot[k]) for k in wanted["gs"]}, None
        f.seek(0)
        data = _unpacked(f.read())
    if struct.unpack_from("<i", data, 0)[0] != 40:
        raise ValueError(f"{Path(path).name}: BSP version {struct.unpack_from('<i', data, 0)[0]}, expected 30, 40 or VBSP")
    n = len(GBX_LUMPS)
    offsets, lengths = struct.unpack_from(f"<{n}I", data, 4), struct.unpack_from(f"<{n}I", data, 4 + 4 * n)
    return "ps2", {k: data[o:o + ln] for k, o, ln in zip(GBX_LUMPS, offsets, lengths) if k in wanted["ps2"]}, dict(zip(GBX_LUMPS, offsets))


def _parsed(lump):
    return parse_entities(lump.split(b"\0", 1)[0].decode("latin-1"))


def entities(path):
    """A map's entities alone, read from its file without the rest of it."""
    return _parsed(_lumps(path, dict.fromkeys(FORMATS, ("entities",)))[1]["entities"])


def outline(path, name):
    """A map with only its entities and its models' bounds, read from its file without the rest of it: enough to follow its
    level changes (bmab.graph)."""
    kind, lump, _offsets = _lumps(path, dict.fromkeys(FORMATS, ("entities", "models")))
    models = np.frombuffer(lump["models"], dtype={"source": SOURCE_MODEL, "gs": GS_MODEL, "ps2": GBX_MODEL}[kind])
    lows, highs = (models["mins"], models["maxs"]) if kind != "ps2" else (models["mins4"][:, :3], models["maxs4"][:, :3])
    b = Bsp(name, _parsed(lump["entities"]))
    b.models = [(lo.copy(), hi.copy(), int(m["firstface"]), int(m["numfaces"])) for lo, hi, m in zip(lows, highs, models)]
    return b


def _gs_names(tex):
    """The texture names a GoldSrc map's texture lump lists, in its order."""
    n = struct.unpack_from("<i", tex, 0)[0]
    return ["" if off < 0 else tex[off:off + 16].split(b"\0", 1)[0].decode("latin-1").lower() for off in struct.unpack_from(f"<{n}i", tex, 4)]


def _ps2_names(tex, pointers, start):
    """The texture names a PS2 map's faces point at (its texture lump holds pixels between the names), one per pointer."""
    return [tex[i + 16:i + 32].split(b"\0", 1)[0].decode("latin-1").lower() for i in ((np.asarray(pointers, dtype=np.int64) - start) // 64 * 64)]


def _source_names(table, strings):
    """The material names a Source map's texdata string table lists, in its order."""
    return [strings[o:strings.index(b"\0", o)].decode("latin-1").lower() for o in np.frombuffer(table, dtype="<i4")]


def textures(path):
    """The texture (or material) names a map uses, read from its file without the rest of it."""
    kind, lump, offsets = _lumps(path, {"source": ("texdata_table", "texdata_strings"), "gs": ("textures",), "ps2": ("faces", "textures")})
    if kind == "source":
        return set(_source_names(lump["texdata_table"], lump["texdata_strings"]))
    if kind == "ps2":
        pointers = np.unique(np.frombuffer(lump["faces"], dtype=GBX_FACE)["texture"])
        return set(_ps2_names(lump["textures"], pointers, offsets["textures"])) - {""}
    return set(_gs_names(lump["textures"])) - {""}


def read(data, name):
    """data: a map file's bytes; PS2 .bs2 files (uint32 size, then zlib) are accepted as stored."""
    if data[:4] == b"VBSP":
        return _read_source(data, name)
    data = _unpacked(data)
    version = struct.unpack_from("<i", data, 0)[0]
    if version == 40:
        return _read_ps2(data, name)
    if version != 30:
        raise ValueError(f"{name}: BSP version {version}, expected 30 or VBSP")
    lumps = {}
    for i, lump in enumerate(LUMPS):
        off, ln = struct.unpack_from("<ii", data, 4 + i * 8)
        lumps[lump] = data[off:off + ln]
    # Blue Shift ships its BSPs with the entity and plane lumps swapped.
    if lumps["entities"][:1] != b"{" and lumps["planes"][:1] == b"{":
        lumps["entities"], lumps["planes"] = lumps["planes"], lumps["entities"]

    texinfo = np.frombuffer(lumps["texinfo"], dtype=np.dtype([("v", "<f4", 8), ("miptex", "<i4"), ("flags", "<i4")]))
    faces = np.frombuffer(lumps["faces"], dtype=np.dtype([
        ("plane", "<u2"), ("side", "<u2"), ("firstedge", "<i4"), ("numedges", "<u2"),
        ("texinfo", "<u2"), ("styles", "u1", 4), ("lightofs", "<i4")]))
    models = np.frombuffer(lumps["models"], dtype=GS_MODEL)

    names = _gs_names(lumps["textures"])
    face_tex = [names[texinfo[t]["miptex"]] for t in faces["texinfo"]]
    bsp = _assemble(name, lumps, faces, models, face_tex, texinfo["v"][faces["texinfo"]].reshape(-1, 2, 4))
    bsp.vis = Vis(np.frombuffer(lumps["planes"], dtype=PLANE), np.frombuffer(lumps["nodes"], dtype=GS_NODE),
                  np.frombuffer(lumps["leaves"], dtype=GS_LEAF), lumps["visibility"], int(models[0]["visleafs"]),
                  int(models[0]["head"][0]))
    return bsp


def _read_ps2(data, name):
    n = len(GBX_LUMPS)
    offsets = struct.unpack_from(f"<{n}I", data, 4)
    lengths = struct.unpack_from(f"<{n}I", data, 4 + 4 * n)
    lumps = {k: data[o:o + ln] for k, o, ln in zip(GBX_LUMPS, offsets, lengths)}
    raw = np.frombuffer(lumps["faces"], dtype=GBX_FACE)
    faces = np.zeros(len(raw), dtype=[("plane", "<u4"), ("side", "<u2"), ("firstedge", "<u4"), ("numedges", "<u4")])
    faces["plane"] = (raw["plane"] - offsets[0]) // PLANE.itemsize
    faces["side"] = raw["side"]
    faces["firstedge"], faces["numedges"] = raw["firstedge"], raw["numedges"]
    face_tex = _ps2_names(lumps["textures"], raw["texture"], offsets[GBX_LUMPS.index("textures")])
    m = np.frombuffer(lumps["models"], dtype=GBX_MODEL)
    models = np.zeros(len(m), dtype=[("mins", "<f4", 3), ("maxs", "<f4", 3), ("firstface", "<u4"), ("numfaces", "<u4")])
    models["mins"], models["maxs"] = m["mins4"][:, :3], m["maxs4"][:, :3]
    models["firstface"], models["numfaces"] = m["firstface"], m["numfaces"]
    lumps["vertices"] = np.frombuffer(lumps["vertices"], dtype="<f4").reshape(-1, 4)[:, :3].tobytes()
    return _assemble(name, lumps, faces, models, face_tex)


def _lzma(raw, size, name, key):
    """A Source lump stored compressed: "LZMA", its size, the packed size, 5 property bytes, then the raw stream."""
    if raw[:4] != b"LZMA":
        raise ValueError(f"{name}: lump {key} has an uncompressed size but no LZMA header")
    actual, packed = struct.unpack_from("<II", raw, 4)
    props, dict_size = raw[12], struct.unpack_from("<I", raw, 13)[0]
    lc, lp, pb = props % 9, props // 9 % 5, props // 45
    out = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA1, "dict_size": dict_size,
                                                          "lc": lc, "lp": lp, "pb": pb}]).decompress(raw[17:17 + packed], actual)
    if len(out) != actual or actual != size:
        raise ValueError(f"{name}: lump {key} unpacked to {len(out)} bytes, expected {size}")
    return out


def _read_source(data, name):
    version = struct.unpack_from("<i", data, 4)[0]
    if version != 20:
        raise ValueError(f"{name}: VBSP version {version}, expected 20")
    lumps = {}
    for key, i in VBSP_LUMPS.items():
        off, ln, ver, size = struct.unpack_from("<iiii", data, 8 + i * 16)
        if key == "leaves" and ver != 1:
            raise ValueError(f"{name}: leaf lump version {ver}, expected 1")
        lumps[key] = _lzma(data[off:off + ln], size, name, key) if size else data[off:off + ln]
    texinfo = np.frombuffer(lumps["texinfo"], dtype=np.dtype([("v", "<f4", 16), ("flags", "<i4"), ("texdata", "<i4")]))
    texdata = np.frombuffer(lumps["texdata"], dtype=np.dtype([("refl", "<f4", 3), ("name", "<i4"), ("size", "<i4", 4)]))
    names = _source_names(lumps["texdata_table"], lumps["texdata_strings"])
    faces = np.frombuffer(lumps["faces"], dtype=np.dtype([
        ("plane", "<u2"), ("side", "u1"), ("onnode", "u1"), ("firstedge", "<i4"), ("numedges", "<i2"),
        ("texinfo", "<i2"), ("dispinfo", "<i2"), ("fogvolume", "<i2"), ("styles", "u1", 4),
        ("lightofs", "<i4"), ("area", "<f4"), ("lmmins", "<i4", 2), ("lmsize", "<i4", 2),
        ("origface", "<i4"), ("numprims", "<u2"), ("firstprim", "<u2"), ("smoothing", "<u4")]))
    models = np.frombuffer(lumps["models"], dtype=SOURCE_MODEL)
    face_tex = ["" if t < 0 else names[texdata[texinfo[t]["texdata"]]["name"]] for t in faces["texinfo"]]
    bsp = _assemble(name, lumps, faces, models, face_tex, texinfo["v"][faces["texinfo"]][:, :8].reshape(-1, 2, 4), sided=False)
    _displace(bsp, faces, lumps)
    for o in np.frombuffer(lumps["overlays"][:len(lumps["overlays"]) // OVERLAY.itemsize * OVERLAY.itemsize], dtype=OVERLAY):
        if o["texinfo"] >= 0:
            bsp.overlays.append((names[texdata[texinfo[o["texinfo"]]["texdata"]]["name"]], o["origin"].copy(), o["normal"].copy()))
    sky = _skybox_faces(bsp, lumps)
    if sky:
        # The skybox is a sealed room: anything inside its box belongs to it, including faces no leaf lists.
        v = np.concatenate([bsp.faces[i].verts for i in sky])
        lo, hi = v.min(0) - 1, v.max(0) + 1
        for f in bsp.faces:
            f.skybox = bool(((f.verts >= lo) & (f.verts <= hi)).all())
    world = [f.verts for f in bsp.faces if f.model == 0 and not f.skybox]
    if world:
        v = np.concatenate(world)
        leaves = np.frombuffer(lumps["leaves"], dtype=LEAF)
        bsp.vis = SourceVis(np.frombuffer(lumps["planes"], dtype=PLANE), np.frombuffer(lumps["nodes"], dtype=NODE),
                            leaves, lumps["visibility"], int(models[0]["head"]),
                            np.frombuffer(lumps["brushes"], dtype=BRUSH), np.frombuffer(lumps["brushsides"], dtype=BRUSHSIDE),
                            np.frombuffer(lumps["leafbrushes"], dtype="<u2"), (v.min(0) - 64, v.max(0) + 64))
    return bsp


def _skybox_faces(bsp, lumps):
    """Faces in the area holding the sky_camera, which is the 3D skybox."""
    cam = next((e["origin"] for e in bsp.entities if e.get("classname") == "sky_camera" and "origin" in e), None)
    if cam is None:
        return set()
    p = np.array([float(x) for x in cam.split()])
    planes = np.frombuffer(lumps["planes"], dtype=PLANE)
    nodes = np.frombuffer(lumps["nodes"], dtype=NODE)
    leaves = np.frombuffer(lumps["leaves"], dtype=LEAF)
    leaffaces = np.frombuffer(lumps["leaffaces"], dtype="<u2")
    i = 0
    while i >= 0:
        node = nodes[i]
        plane = planes[node["plane"]]
        i = node["children"][0 if p @ plane["n"] - plane["d"] >= 0 else 1]
    area = leaves[-i - 1]["areaflags"] & 0x1FF
    out = set()
    for leaf in leaves[(leaves["areaflags"] & 0x1FF) == area]:
        out.update(leaffaces[leaf["firstleafface"]:leaf["firstleafface"] + leaf["numleaffaces"]].tolist())
    return out


DISPINFO = 176  # bytes of one displacement's record
DISP_VERT = np.dtype([("v", "<f4", 3), ("d", "<f4"), ("alpha", "<f4")])


def _displace(bsp, faces, lumps):
    """Source's sculpted terrain: each displacement's face becomes the triangles of its displaced grid, the first in the face's place (so
    that the map's own face numbers still name it) and the rest after all the faces."""
    info, moves = lumps.get("dispinfo", b""), np.frombuffer(lumps.get("disp_verts", b""), DISP_VERT)
    extra = []
    for fi in np.flatnonzero(faces["dispinfo"] >= 0).tolist():
        base, at = bsp.faces[fi], int(faces["dispinfo"][fi]) * DISPINFO
        if len(base.verts) != 4 or at + DISPINFO > len(info):
            continue
        start = np.frombuffer(info, "<f4", 3, at)
        first, _tris, power = struct.unpack_from("<3i", info, at + 12)
        corners = np.roll(base.verts, -int(np.argmin(np.linalg.norm(base.verts - start, axis=1))), axis=0)
        n = (1 << power) + 1
        if first + n * n > len(moves):
            continue
        t = np.linspace(0.0, 1.0, n)[:, None]
        left, right = corners[0] + (corners[1] - corners[0]) * t, corners[3] + (corners[2] - corners[3]) * t
        grid = left[:, None] + (right - left)[:, None] * t[None]
        m = moves[first:first + n * n]
        grid = (grid + (m["v"] * m["d"][:, None]).reshape(n, n, 3)).reshape(-1, 3)
        i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
        a, b, c, d = (i * n + j).ravel(), ((i + 1) * n + j).ravel(), ((i + 1) * n + j + 1).ravel(), (i * n + j + 1).ravel()
        tris = []
        for tri in np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]):
            v = grid[tri]
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            length = np.linalg.norm(normal)
            if length < 1e-6:
                continue
            normal /= length
            if normal @ base.normal < 0:
                v, normal = v[::-1], -normal
            tris.append(Face(base.model, normal, v, base.texture, axes=base.axes))
        if tris:
            bsp.faces[fi] = tris[0]
            extra += tris[1:]
    bsp.faces += extra


def _assemble(name, lumps, faces, models, face_tex, axes=None, sided=True):
    """sided: a face's side says it faces away from its plane (GoldSrc's); Source's planes face as their faces do."""
    bsp = Bsp(name, _parsed(lumps["entities"]))
    planes = np.frombuffer(lumps["planes"], dtype=PLANE)
    verts = np.frombuffer(lumps["vertices"], dtype="<f4").reshape(-1, 3)
    edges = np.frombuffer(lumps["edges"], dtype="<u2").reshape(-1, 2)
    surfedges = np.frombuffer(lumps["surfedges"], dtype="<i4")

    face_model = np.full(len(faces), -1)
    for mi, m in enumerate(models):
        face_model[m["firstface"]:m["firstface"] + m["numfaces"]] = mi
        bsp.models.append((m["mins"].copy(), m["maxs"].copy(), int(m["firstface"]), int(m["numfaces"])))

    for fi, f in enumerate(faces):
        se = surfedges[f["firstedge"]:f["firstedge"] + f["numedges"]]
        idx = np.where(se >= 0, edges[np.abs(se), 0], edges[np.abs(se), 1])
        normal = planes[f["plane"]]["n"] * (-1 if sided and f["side"] else 1)
        bsp.faces.append(Face(int(face_model[fi]), normal, verts[idx], face_tex[fi],
                              axes=None if axes is None else axes[fi].astype(np.float64)))
    return bsp


def is_tool(texture):
    return texture in TOOL_TEXTURES or texture.startswith(("sky", "tools/"))
