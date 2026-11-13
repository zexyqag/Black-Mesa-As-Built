"""Texture families: how alike maps look, as evidence (soft) for which area unsigned maps belong to.

Each map gets a fingerprint: visible wall and floor area per texture (animation-frame and random-tile
prefixes folded into the base name). Similarity is cosine over those fingerprints. Half-Life names many
textures after the chapter that introduced them (c1a0_..., c2a4...), so a map's area by chapter prefix
shows which chapter's art it reuses.
"""
import re
from collections import Counter

import numpy as np

from . import bsp

_FRAME = re.compile(r"^[+-][0-9a-j]")
_CHAPTER = re.compile(r"^(c\da\d)")


def base(texture):
    t = _FRAME.sub("", texture.lower())
    return t.lstrip("{!~")


def fingerprint(b):
    area = Counter()
    for f in b.faces:
        if f.model != 0 or bsp.is_tool(f.texture) or getattr(f, "skybox", False):
            continue
        v = f.verts
        a = 0.5 * np.linalg.norm(np.cross(v[1:-1] - v[0], v[2:] - v[0]).sum(axis=0))
        area[base(f.texture)] += float(a)
    return area


def chapters(fp):
    """Share of a fingerprint's area in textures named after a Half-Life chapter."""
    total = sum(fp.values()) or 1
    out = Counter()
    for t, a in fp.items():
        m = _CHAPTER.match(t)
        if m:
            out[m.group(1)] += a / total
    return out


def similarity(fps):
    names = sorted(fps)
    vocab = sorted({t for fp in fps.values() for t in fp})
    col = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(names), len(vocab)))
    for i, n in enumerate(names):
        for t, a in fps[n].items():
            m[i, col[t]] = a
    m = np.sqrt(m)  # damp a few huge surfaces (floors, sky) dominating
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return names, m @ m.T
