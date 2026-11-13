"""What a piece of evidence (bmab.register) looks, sounds or reads like, read from this computer's own copies when asked and never
stored: a sign's or wall map's texture as a PNG, a voice line as WAV, and for an announcement, caption or documentary timecode the
text there. Games are under the data folder (bmab.sources.DATA), outside works under the downloads folder.
"""
import hashlib
import io
import re
from pathlib import Path

from . import sources, vtf, wad

TEXT_KINDS = ("announcement", "caption", "screen text", "documentary")


def _file(e):
    """The evidence's file on this computer, or None."""
    p = sources.DATA / e["file"]
    if p.is_file():
        return p
    hits = sorted(sources.DOWNLOADS.rglob(Path(e["file"]).name)) if sources.DOWNLOADS.is_dir() else []
    return hits[0] if hits else None


def _picture(path, ref):
    data = path.read_bytes()
    if path.suffix == ".vtf":
        return vtf.image(data)
    if path.suffix == ".wad":
        return wad.wad(path).get(ref.lower())
    if path.suffix == ".bsp":
        return wad.embedded(data)[1].get(ref.lower())
    return None


def _text(path, e):
    """The line an announcement, caption, screen text (a titles.txt message) or documentary timecode refers to, as plain text."""
    if e["kind"] == "caption":
        m = re.search(rf'"{re.escape(e["ref"])}"\s+"([^"]*)"', path.read_text(encoding="utf-16", errors="replace"))
        return re.sub(r"<[^>]*>", "", m.group(1)).strip() if m else None
    lines = path.read_text("latin-1").splitlines()  # as bmab.generate reads them
    if e["kind"] == "announcement":
        return next((line.split(None, 1)[1].strip() for line in lines if line.split(None, 1)[:1] == [e["ref"]] and " " in line.strip()), None)
    if e["kind"] == "screen text":
        m = re.search(rf"^{re.escape(e['ref'])}\s*\n\{{\n(.*?)\n\}}", "\n".join(lines), re.M | re.S)
        return " ".join(line.strip() for line in m.group(1).splitlines() if line.strip()) if m else None
    start = next((k for k, line in enumerate(lines) if line.startswith(e["ref"] + " ")), None)  # a documentary: up to the next timecode
    if start is None:
        return None
    out = [lines[start][len(e["ref"]):].strip()]
    for line in lines[start + 1:]:
        if re.match(r"\d+:\d\d ", line):
            break
        out.append(line.strip())
    return " ".join(out)


def media(e):
    """(content type, bytes, whether this computer's file is the version the evidence was recorded from) of what the evidence
    is; ValueError when there is nothing to show here."""
    if "url" in e:
        raise ValueError("a web page is shown by its address")
    path = _file(e)
    if path is None:
        raise ValueError(f"{e['file']} isn't on this computer")
    same = hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"]
    if e["kind"] in ("sign", "wall map"):
        img = _picture(path, e["ref"])
        if img is None:
            raise ValueError(f"no picture {e['ref']} in {e['file']}")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "image/png", buf.getvalue(), same
    if e["kind"] == "voice line" and path.suffix == ".wav":
        return "audio/wav", path.read_bytes(), same
    if e["kind"] in TEXT_KINDS:
        text = _text(path, e)
        if text is None:
            raise ValueError(f"no {e['ref']} in {e['file']}")
        return "text/plain; charset=utf-8", text.encode(), same
    raise ValueError(f"nothing to show for a {e['kind']}")
