"""Source engine textures (.vtf) and materials (.vmt): the first frame's full-size image as Pillow.

Formats decoded: RGBA8888, BGRA8888, BGR888, RGB888, DXT1, DXT3, DXT5. Others return None.
"""
import re
import struct

import numpy as np
from PIL import Image

RGBA8888, RGB888, BGR888, DXT1, DXT3, DXT5, BGRA8888 = 0, 2, 3, 13, 14, 15, 12
_BYTES = {RGBA8888: 4, BGRA8888: 4, RGB888: 3, BGR888: 3}
_BLOCK = {DXT1: 8, DXT3: 16, DXT5: 16}


def _size(fmt, w, h):
    if fmt in _BLOCK:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * _BLOCK[fmt]
    return w * h * _BYTES.get(fmt, 0)


def _colors565(c):
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    return np.stack([r, g, b], -1).astype(np.int32)


def _color_blocks(blocks, four_always):
    """blocks: (n, 8) uint8 colour blocks -> (n, 16, 3) uint8 rgb and (n, 16) bool punched-through alpha."""
    c0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    c1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    p0, p1 = _colors565(c0), _colors565(c1)
    four = four_always | (c0 > c1)[:, None]
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    pal = np.stack([p0, p1, p2, p3], 1)
    bits = blocks[:, 4:8].astype(np.uint32)
    idx = (bits[:, 0] | bits[:, 1] << 8 | bits[:, 2] << 16 | bits[:, 3] << 24)[:, None] >> (2 * np.arange(16)) & 3
    rgb = np.take_along_axis(pal, idx[:, :, None].repeat(3, 2), 1)
    transparent = (~four[:, 0])[:, None] & (idx == 3)
    return rgb.astype(np.uint8), transparent


def _dxt(data, w, h, fmt):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    raw = np.frombuffer(data[:bw * bh * _BLOCK[fmt]], np.uint8).reshape(bw * bh, _BLOCK[fmt])
    if fmt == DXT1:
        rgb, holes = _color_blocks(raw, np.zeros((1, 1), bool))
        alpha = np.where(holes, 0, 255).astype(np.uint8)
    else:
        rgb, _ = _color_blocks(raw[:, 8:], np.ones((1, 1), bool))
        if fmt == DXT3:
            nib = raw[:, :8, None] >> np.array([0, 4], np.uint8) & 15
            alpha = (nib.reshape(-1, 16) * 17).astype(np.uint8)
        else:
            a0, a1 = raw[:, 0].astype(np.int32), raw[:, 1].astype(np.int32)
            bits = np.zeros(len(raw), np.uint64)
            for i in range(6):
                bits |= raw[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
            idx = (bits[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64)) & np.uint64(7)).astype(np.int32)
            k = np.arange(8)
            eight = ((7 - k[1:7]) * a0[:, None] + k[1:7] * a1[:, None]) // 7
            six = ((5 - k[1:5]) * a0[:, None] + k[1:5] * a1[:, None]) // 5
            six = np.concatenate([six, np.zeros((len(raw), 1), int), np.full((len(raw), 1), 255)], 1)
            mids = np.where((a0 > a1)[:, None], eight, six)
            pal = np.concatenate([a0[:, None], a1[:, None], mids], 1)
            alpha = np.take_along_axis(pal, idx, 1).astype(np.uint8)
    px = np.concatenate([rgb, alpha[:, :, None]], 2).reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4)
    return px.reshape(bh * 4, bw * 4, 4)[:h, :w]


def size(data):
    """A texture's width and height."""
    return struct.unpack_from("<HH", data, 16)


def image(data, fit=None):
    """A .vtf's picture, RGBA; fit: the largest of its mip levels no wider or taller than this."""
    if data[:4] != b"VTF\0":
        return None
    major, minor, header = struct.unpack_from("<3I", data, 4)
    w, h, flags, frames, _first = struct.unpack_from("<HHIHH", data, 16)
    fmt = struct.unpack_from("<i", data, 52)[0]
    mips = data[56]
    low_fmt, low_w, low_h = struct.unpack_from("<iBB", data, 57)
    faces = 6 if flags & 0x4000 else 1
    depth = struct.unpack_from("<H", data, 63)[0] if (major, minor) >= (7, 2) else 1
    if (major, minor) >= (7, 3):
        count = struct.unpack_from("<I", data, 68)[0]
        start = None
        for i in range(count):
            tag, rflags, off = struct.unpack_from("<3sBI", data, 80 + 8 * i)
            if tag == b"\x30\x00\x00":
                start = off
        if start is None:
            return None
    else:
        start = header + (_size(low_fmt, low_w, low_h) if low_fmt >= 0 else 0)
    if fmt not in _BYTES and fmt not in _BLOCK:
        return None
    top = 0
    while fit and top < mips - 1 and max(w >> top, h >> top) > fit:
        top += 1
    pos = start
    for m in range(mips - 1, top, -1):  # smallest mips come first
        pos += _size(fmt, max(1, w >> m), max(1, h >> m)) * frames * faces * max(1, depth >> m)
    w, h = max(1, w >> top), max(1, h >> top)
    body = data[pos:pos + _size(fmt, w, h)]
    if fmt in _BLOCK:
        px = _dxt(body, w, h, fmt)
    else:
        a = np.frombuffer(body, np.uint8).reshape(h, w, _BYTES[fmt])
        if fmt in (BGRA8888, BGR888):
            a = a[:, :, [2, 1, 0] + ([3] if a.shape[2] == 4 else [])]
        px = a if a.shape[2] == 4 else np.concatenate([a, np.full((h, w, 1), 255, np.uint8)], 2)
    return Image.fromarray(np.ascontiguousarray(px), "RGBA")


PATCHED = re.compile(r"^maps/[^/]+/(.*?)(_-?\d+_-?\d+_-?\d+)?(_wvt_patch)?$")  # a map's own copy of a material, with its light probe's position
_BASE = re.compile(r'"?\$basetexture"?\s+"?([^"\s}]+)', re.I)
_SEE_THROUGH = re.compile(r'"?\$(alphatest|translucent)"?\s+"?1', re.I)
_INCLUDE = re.compile(r'"?include"?\s+"?([^"\s}]+)', re.I)


def base_texture(vmt_text):
    m = _BASE.search(vmt_text)
    return m.group(1).lower().replace("\\", "/").removesuffix(".vtf") if m else None


def include(vmt_text):
    """Patch materials point at the material they modify."""
    m = _INCLUDE.search(vmt_text)
    return m.group(1).lower().replace("\\", "/").removeprefix("materials/").removesuffix(".vmt") if m else None


def material(folders, name):
    """A Source material by name, searching these game folders in turn: (its .vtf file, whether its alpha means see-through), following patch includes; (None, False) if not found."""
    m = PATCHED.match(name)
    name = (m.group(1) if m else name).replace("\\", "/")
    for _ in range(4):
        vmt = next((v for v in (f / "materials" / f"{name}.vmt" for f in folders) if v.exists()), None)
        if vmt is None:
            return None, False
        text = vmt.read_text("latin-1")
        base = base_texture(text)
        if base:
            return next((t for t in (f / "materials" / f"{base}.vtf" for f in folders) if t.exists()), None), bool(_SEE_THROUGH.search(text))
        name = include(text)
        if not name:
            return None, False
    return None, False
