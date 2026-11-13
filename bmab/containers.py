"""Read-only views of a folder or archive as {relative path: Entry}.

Paths are lowercase posix. Supported containers: folders, zip, Quake PAK, Valve VPK (v1 and
v2, single file or _dir with numbered parts), Garry's Mod GMA, and CD images (ISO 9660, raw
2352-byte or 2048-byte sectors). PAKs and CD images also open from inside another container.
"""
import fnmatch
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Entry:
    size: int
    read: Callable[[], bytes]
    path: Path | None = None  # set for loose files


def _slice(path, off, n):
    def read():
        with open(path, "rb") as f:
            f.seek(off)
            return f.read(n)
    return read


def folder(root, patterns=("*",)):
    root = Path(root)
    out = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root).as_posix().lower()
        if any(fnmatch.fnmatch(rel, pat) for pat in patterns) and p.is_file():
            out[rel] = Entry(p.stat().st_size, p.read_bytes, p)
    return out


def zip_members(path, prefix=""):
    z = zipfile.ZipFile(path)
    prefix = prefix.lower()
    out = {}
    for info in z.infolist():
        name = info.filename.lower()
        if not info.is_dir() and name.startswith(prefix):
            out[name[len(prefix):]] = Entry(info.file_size, lambda i=info: z.read(i))
    return out


def _ranges(entry):
    """(header reader, slice factory) for an entry on disk or held in memory."""
    if entry.path is not None:
        def head(off, n):
            return _slice(entry.path, off, n)()
        return head, lambda off, n: _slice(entry.path, off, n)
    data = entry.read()
    return (lambda off, n: data[off:off + n]), (lambda off, n: lambda: data[off:off + n])


def pak(entry):
    head, piece = _ranges(entry)
    magic, off, size = struct.unpack("<4sii", head(0, 12))
    if magic != b"PACK":
        raise ValueError("not a PAK file")
    table = head(off, size)
    out = {}
    for i in range(0, size, 64):
        name, o, n = struct.unpack_from("<56sii", table, i)
        out[name.split(b"\0", 1)[0].decode("latin-1").lower()] = Entry(n, piece(o, n))
    return out


def cd_image(entry):
    data = entry.read() if entry.path is None else entry.path.read_bytes()
    for size, skip in ((2352, 24), (2352, 16), (2048, 0)):
        if data[16 * size + skip + 1:16 * size + skip + 6] == b"CD001":
            break
    else:
        raise ValueError("no ISO 9660 volume found")

    def read(lba, n):
        out = bytearray()
        for i in range((n + 2047) // 2048):
            p = (lba + i) * size + skip
            out += data[p:p + 2048]
        return bytes(out[:n])

    out = {}

    def walk(lba, length, prefix):
        d, pos = read(lba, length), 0
        while pos < len(d):
            rec = d[pos]
            if rec == 0:
                pos = (pos // 2048 + 1) * 2048
                continue
            e_lba, e_len = struct.unpack_from("<I", d, pos + 2)[0], struct.unpack_from("<I", d, pos + 10)[0]
            flags, name = d[pos + 25], d[pos + 33:pos + 33 + d[pos + 32]]
            if name not in (b"\x00", b"\x01"):
                n = name.decode("latin-1").split(";")[0].lower()
                if flags & 2:
                    walk(e_lba, e_len, prefix + n + "/")
                else:
                    out[prefix + n] = Entry(e_len, lambda a=e_lba, b=e_len: read(a, b))
            pos += rec

    pvd = read(16, 2048)
    walk(struct.unpack_from("<I", pvd, 158)[0], struct.unpack_from("<I", pvd, 166)[0], "")
    return out


def _cstr(buf, pos):
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def vpk(path):
    path = Path(path)
    with open(path, "rb") as f:
        sig, version, tree = struct.unpack("<III", f.read(12))
        if sig != 0x55AA1234:
            raise ValueError(f"{path}: not a VPK file")
        header = {1: 12, 2: 28}[version]
        f.seek(0)
        data = f.read(header + tree)
    parts = path.name.removesuffix("_dir.vpk") if path.name.endswith("_dir.vpk") else None
    out = {}
    pos = header
    while True:
        ext, pos = _cstr(data, pos)
        if not ext:
            break
        while True:
            d, pos = _cstr(data, pos)
            if not d:
                break
            while True:
                name, pos = _cstr(data, pos)
                if not name:
                    break
                _crc, preload, index, off, n, _end = struct.unpack_from("<IHHIIH", data, pos)
                pos += 18
                pre = data[pos:pos + preload]
                pos += preload
                full = "/".join(x for x in (d.strip(), name) if x) + ("" if ext == " " else "." + ext)
                if index == 0x7FFF:
                    src, base = path, header + tree
                else:
                    src, base = path.with_name(f"{parts}_{index:03d}.vpk"), 0
                body = _slice(src, base + off, n)
                read = (lambda p=pre, b=body: p + b()) if preload else body
                out[full.lower()] = Entry(preload + n, read)
    return out


def gma(path):
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    if head[:4] != b"GMAD":
        raise ValueError(f"{path}: not a GMA file")
    version = head[4]
    pos = 5 + 8 + 8
    if version > 1:
        while True:
            s, pos = _cstr(head, pos)
            if not s:
                break
    for _ in range(3):  # name, description, author
        _, pos = _cstr(head, pos)
    pos += 4
    files = []
    while True:
        (num,) = struct.unpack_from("<I", head, pos)
        pos += 4
        if num == 0:
            break
        name, pos = _cstr(head, pos)
        size, _crc = struct.unpack_from("<qI", head, pos)
        pos += 12
        files.append((name, size))
    out, off = {}, pos
    for name, size in files:
        out[name.lower()] = Entry(size, _slice(path, off, size))
        off += size
    return out


VPK_PART = re.compile(r"_\d{3}\.vpk$")

READERS = {".zip": lambda e: zip_members(e.path), ".pak": pak, ".vpk": lambda e: vpk(e.path),
           ".gma": lambda e: gma(e.path), ".bin": cd_image, ".iso": cd_image}


def expand(tree, patterns):
    """Open the containers matching each pattern in turn, so later patterns can reach into earlier ones.

    A container's members are placed under the container's own folder; loose files take precedence.
    """
    for pattern in patterns:
        merged = {}
        for rel in sorted(tree):
            if VPK_PART.search(rel):  # data parts of a _dir.vpk, not archives themselves
                continue
            if fnmatch.fnmatch(rel, pattern):
                folder_prefix = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
                for k, v in READERS[Path(rel).suffix](tree[rel]).items():
                    merged.setdefault(folder_prefix + k, v)
        tree = {**merged, **tree}
    return tree
