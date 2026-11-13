"""GoldSrc textures: WAD3 archives and textures embedded in BSPs, as Pillow images."""
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MIPTEX = 0x43


def _miptex(buf, base=0):
    """Decode one miptex at buf[base:]; None if its pixels live in a WAD instead."""
    name = buf[base:base + 16].split(b"\0", 1)[0].decode("latin-1").lower()
    w, h, *offs = struct.unpack_from("<II4I", buf, base + 16)
    if offs[0] == 0 or w == 0 or h == 0:
        return name, None
    pixels = buf[base + offs[0]:base + offs[0] + w * h]
    pal_at = base + offs[3] + (w // 8) * (h // 8) + 2
    palette = buf[pal_at:pal_at + 768]
    img = Image.frombytes("P", (w, h), pixels)
    img.putpalette(palette)
    img = img.convert("RGBA")
    if name.startswith("{"):  # palette index 255 is transparent
        mask = Image.frombytes("L", (w, h), bytes(0 if p == 255 else 255 for p in pixels))
        img.putalpha(mask)
    return name, img


def wad(path):
    data = Path(path).read_bytes()
    magic, n, diroff = struct.unpack_from("<4sii", data, 0)
    if magic not in (b"WAD3", b"WAD2"):
        raise ValueError(f"{path}: not a WAD file")
    out = {}
    for i in range(n):
        pos, disk, size, typ, comp = struct.unpack_from("<iiiBB", data, diroff + 32 * i)
        if typ == MIPTEX and comp == 0:
            name, img = _miptex(data, pos)
            if img is not None:
                out[name] = img
    return out


def embedded(bsp_data):
    """Textures stored inside a GoldSrc BSP (lump 2), plus the names of all textures it uses."""
    off, ln = struct.unpack_from("<ii", bsp_data, 4 + 2 * 8)
    lump = bsp_data[off:off + ln]
    n = struct.unpack_from("<i", lump, 0)[0]
    names, images = [], {}
    for o in struct.unpack_from(f"<{n}i", lump, 4):
        if o < 0:
            continue
        name, img = _miptex(lump, o)
        names.append(name)
        if img is not None:
            images[name] = img
    return names, images


def sheet(items, path, cell=96, cols=12, label=True):
    """Contact sheet of (caption, image) pairs, each scaled to fit a cell."""
    font = ImageFont.load_default(size=11)
    rows = (len(items) + cols - 1) // cols
    lh = 14 if label else 0
    out = Image.new("RGB", (cols * cell, rows * (cell + lh)), (40, 40, 48))
    d = ImageDraw.Draw(out)
    for i, (caption, img) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + lh)
        im = img.copy()
        s = min((cell - 4) / im.width, (cell - 4) / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.NEAREST if s >= 1 else Image.LANCZOS)
        out.paste(im, (x + 2, y + 2), im)
        if label:
            d.text((x + 2, y + cell), caption[:cell // 6], fill=(230, 230, 230), font=font)
    out.save(path)


def model_textures(data):
    """Textures embedded in a GoldSrc studio model (.mdl version 10): {name: image}.

    Models whose textures live in a companion *t.mdl file carry none here; read that file instead.
    """
    if data[:4] != b"IDST" or struct.unpack_from("<i", data, 4)[0] != 10:
        return {}
    count, index = struct.unpack_from("<ii", data, 180)
    out = {}
    for i in range(count):
        name, _flags, w, h, pix = struct.unpack_from("<64siiii", data, index + 80 * i)
        if w <= 0 or h <= 0 or pix + w * h + 768 > len(data):
            continue
        img = Image.frombytes("P", (w, h), data[pix:pix + w * h])
        img.putpalette(data[pix + w * h:pix + w * h + 768])
        out[name.split(b"\0", 1)[0].decode("latin-1").lower().removesuffix(".bmp")] = img.convert("RGBA")
    return out
