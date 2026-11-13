"""An imported game folder: its start map (GoldSrc's liblist.gam, or the first chapter of a Source game's chapter manifest) and its
maps (.bsp, or PS2 .bs2)."""
import hashlib
import re
from pathlib import Path

from . import bsp

_hashed = {}  # {map file: (its modification time, size, sha256)}, so a big map is hashed once


class Game:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.start = None
        lib = self.folder / "liblist.gam"
        if lib.exists():
            m = re.search(r'^\s*startmap\s+"([^"]+)"', lib.read_text("latin-1"), re.I | re.M)
            self.start = m.group(1).lower().removesuffix(".bsp") if m else None
        maps = (self.folder / "maps")
        self._maps = {p.stem.lower(): p for p in sorted([*maps.glob("*.bsp"), *maps.glob("*.bs2")])}
        chapters = self.folder / "scripts" / "chapter_manifest.txt"
        if self.start is None and chapters.exists():  # "map name prefix" chapter, per line
            first = min(((int(n), p.lower()) for p, n in re.findall(r"^\s*(\w+)\s+(\d+)\s*$", chapters.read_text("latin-1"), re.M)), default=None)
            self.start = next((n for n in self._maps if first and n.startswith(first[1])), None)

    @property
    def map_names(self):
        return sorted(self._maps)

    def read(self, name):
        return bsp.read(self._maps[name].read_bytes(), name)

    def raw(self, name):
        return self._maps[name].read_bytes()

    def path(self, name):
        return self._maps[name]

    def version(self, name):
        """Which file a map is: {file (its path among the imported games: the game's folder, then inside it), sha256}."""
        path = self._maps[name]
        st = path.stat()
        if _hashed.get(path, (None, None))[:2] != (st.st_mtime_ns, st.st_size):
            _hashed[path] = (st.st_mtime_ns, st.st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        return {"file": f"{self.folder.name}/{path.relative_to(self.folder).as_posix()}", "sha256": _hashed[path][2]}

    def load_all(self):
        return {n: self.read(n) for n in self.map_names}


