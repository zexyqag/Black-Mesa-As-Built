"""Match an in-game wall map (a top-down wireframe drawn on a sign) against the real map geometry.

The sign's wireframe pixels are scored against distance transforms of rendered wall outlines over
rotations (0, 90, 180, 270), mirroring, a range of scales and every translation (by FFT correlation).
Score: symmetric chamfer distance in sign pixels: the mean distance from each wireframe pixel to the
nearest rendered wall, plus the mean distance from each rendered wall pixel inside the sign's footprint
to the nearest wireframe pixel (both capped at 10). Lower is better; a dense map can't fake a match.
"""
import cv2
import numpy as np
from scipy import ndimage, signal

from . import bsp


def wireframe(img, hue=(80, 100), min_sat=80, min_val=90, faint=False):
    """Mask of the drawn lines: by default the cyan of Half-Life's wall maps.

    faint also takes the dim red-brown lines some signs use for the rest of the map around a highlighted route.
    """
    hsv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= hue[0]) & (h <= hue[1]) & (s >= min_sat) & (v >= min_val)
    if faint:
        mask |= ((h <= 15) | (h >= 165)) & (s >= 30) & (s <= 170) & (v >= 45) & (v <= 150)
    return mask


def wall_segments(b, offset=(0, 0, 0)):
    segs = []
    for f in b.faces:
        if f.model == 0 and abs(f.normal[2]) < 0.3 and not bsp.is_tool(f.texture) and not f.skybox:
            v = f.verts[:, :2] + np.asarray(offset)[:2]
            segs += list(zip(v, np.roll(v, -1, axis=0)))
    return segs


def render(segs, scale, pad=4):
    """scale: one factor, or (x, y) factors for signs squashed to fit their texture."""
    scale = np.broadcast_to(np.asarray(scale, float), (2,))
    p = np.array([q for s in segs for q in s])
    lo = p.min(0)
    size = np.ceil((p.max(0) - lo) * scale).astype(int) + 2 * pad + 1
    img = np.zeros((size[1], size[0]), np.uint8)
    for a, b in segs:
        a = (a - lo) * scale + pad
        b = (b - lo) * scale + pad
        cv2.line(img, (int(a[0]), int(size[1] - 1 - a[1])), (int(b[0]), int(size[1] - 1 - b[1])), 1, 1)
    return img


def _orient(mask, rot, mirror):
    m = np.fliplr(mask) if mirror else mask
    return np.rot90(m, rot)


def match(sign_mask, segs, scales, aspects=(1.0,)):
    """aspects: width/height stretch factors to try, for signs squashed to fit their texture."""
    ys, xs = np.nonzero(sign_mask)
    tmpl = sign_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    n = tmpl.sum()
    best = None
    for base_scale in scales:
        for aspect in aspects:
            walls = render(segs, (base_scale * aspect ** 0.5, base_scale / aspect ** 0.5))
            if walls.shape[0] < 8 or walls.shape[1] < 8:
                continue
            walls = np.pad(walls, max(tmpl.shape))
            dist = np.minimum(ndimage.distance_transform_edt(walls == 0).astype(np.float32), 10)
            for mirror in (False, True):
                for rot in range(4):
                    t = _orient(tmpl, rot, mirror)
                    if t.shape[0] > dist.shape[0] or t.shape[1] > dist.shape[1]:
                        continue
                    cost = signal.fftconvolve(dist, t[::-1, ::-1], mode="valid") / n
                    back = np.minimum(ndimage.distance_transform_edt(t == 0), 10)
                    for flat in np.argsort(cost, axis=None)[:5]:
                        iy, ix = np.unravel_index(flat, cost.shape)
                        window = walls[iy:iy + t.shape[0], ix:ix + t.shape[1]] > 0
                        reverse = float(back[window].mean()) if window.any() else 10.0
                        total = float(cost[iy, ix]) + reverse
                        if best is None or total < best["score"]:
                            best = {"score": round(total, 2), "forward": round(float(cost[iy, ix]), 2),
                                    "reverse": round(reverse, 2), "scale": float(base_scale), "aspect": float(aspect),
                                    "rot": rot * 90, "mirror": bool(mirror), "at": [int(ix), int(iy)]}
    return best


def overlay(sign_mask, segs, result, margin=30):
    """RGB image: rendered walls blue, the sign's wireframe red at the matched pose, both white."""
    ys, xs = np.nonzero(sign_mask)
    t = _orient(sign_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                result["rot"] // 90, result["mirror"])
    a = result.get("aspect", 1.0)
    walls = np.pad(render(segs, (result["scale"] * a ** 0.5, result["scale"] / a ** 0.5)), max(sign_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].shape))
    x, y = result["at"]
    hit = np.zeros(walls.shape, bool)
    hit[y:y + t.shape[0], x:x + t.shape[1]] = t
    rgb = np.zeros(walls.shape + (3,), np.uint8)
    rgb[walls > 0] = (0, 160, 255)
    rgb[hit] = (255, 80, 80)
    rgb[(walls > 0) & hit] = (255, 255, 255)
    y0, x0 = max(0, y - margin), max(0, x - margin)
    return rgb[y0:y + t.shape[0] + margin, x0:x + t.shape[1] + margin]
