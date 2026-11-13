"""Brush entities (doors, walls, breakables, trains and the like) as they stand in the map. The map file keeps some of their models
around their own origin: in Source every one, in GoldSrc one built with an origin brush (it has an origin key). Their faces are
moved into place here, turned by their angles too in Source (GoldSrc's angles say which way a door moves, not how it is turned).

What a page draws: the world and every brush entity a player could see, so not triggers, ladders, clip or other invisible volumes,
nor one rendered fully see-through.
"""
import re

import numpy as np

from . import bsp

HIDDEN = re.compile(r"^(trigger_|func_ladder|func_useableladder|func_clip|func_buyzone|func_areaportal|func_occluder|func_precipitation"
                    r"|func_dust|func_smokevolume|func_vehicleclip|func_nav|func_bomb|func_hostage|func_monitor|func_friction"
                    r"|func_traincontrols|func_mortar_field|env_|game_)")


def _rotation(angles):
    """Source's pitch, yaw and roll (degrees) as a matrix: yaw about z, pitch down about y, roll about x."""
    p, y, r = np.radians(angles)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return rz @ ry @ rx


def _seen(e):
    if HIDDEN.match(e.get("classname", "")):
        return False
    mode, amount = e.get("rendermode", "0") or "0", e.get("renderamt", "255") or "255"
    return not (mode not in ("0", "") and float(amount) == 0) and mode != "10"  # Source's rendermode 10 draws nothing


def placement(b):
    """{model index: (turn 3 x 3, shift, drawn)} of the map's brush entities, kept on the map (bsp.py's text keys the connector
    caches, so not a field)."""
    if getattr(b, "brush_places", None) is None:
        source, out = isinstance(b.vis, bsp.SourceVis), {}
        for e in b.entities:
            m = e.get("model", "")
            if not m.startswith("*") or not m[1:].isdigit():
                continue
            turn, shift = np.eye(3), np.zeros(3)
            if source or "origin" in e:
                shift = np.array([float(x) for x in (e.get("origin") or "0 0 0").split()])
            if source and e.get("angles"):
                turn = _rotation([float(x) for x in e["angles"].split()])
            out[int(m[1:])] = (turn, shift, _seen(e))
        b.brush_places = out
    return b.brush_places


def place(b, f):
    """A face's corners and unit normal where the map has it."""
    n = f.normal / np.linalg.norm(f.normal)
    if f.model == 0:
        return f.verts, n
    turn, shift, _ = placement(b).get(f.model, (np.eye(3), np.zeros(3), True))
    return f.verts @ turn.T + shift, turn @ n


def turned(b, f, v):
    """A vector of a face's own (its texture axes, say) as it is turned in the map."""
    return v if f.model == 0 else placement(b).get(f.model, (np.eye(3),))[0] @ v


def drawn(b, f):
    """Whether a page draws a face: the world's and those of brush entities a player could see, never tools, sky or the 3D skybox."""
    t = f.texture.lower()
    if f.skybox or bsp.is_tool(t) or t == "sky":
        return False
    return f.model == 0 or placement(b).get(f.model, (None, None, False))[2]
