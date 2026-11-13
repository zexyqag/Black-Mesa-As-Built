"""Source maps' 3D models: where a map places them (its static props and prop entities), and their shapes from the game's
.mdl, .vvd and .dx90.vtx files. The models are the games' content: they only ever go in the local inspector pages.

A .vvd holds a model in its reference pose, which can lie on its side or upside down (a chair whose idle stands it up): a prop
entity is posed as the game first shows it, its default animation's first frame (its first sequence's if it names none). A static
prop is drawn as the file holds it, as the game draws it.
"""
import struct

import numpy as np

from . import bsp as bspmod, vtf

GAME_LUMP = 35
PROP_CLASSES = ("prop_dynamic", "prop_dynamic_override", "prop_physics", "prop_physics_override", "prop_physics_multiplayer", "prop_door_rotating")
GOLDSRC_MOST = 1500  # triangles a GoldSrc model is simplified to
VERTEX = np.dtype([("weights", "<f4", 3), ("bones", "u1", 4), ("p", "<f4", 3), ("n", "<f4", 3), ("uv", "<f4", 2)])
STRIP_LIST, STRIP_STRIP = 1, 2
STRIP_GROUP_SIZES = (25, 33)  # Source 2013, and the longer form some games write
_read = {}


def _game_lump(raw, tag):
    """A game lump's bytes by its four letters, or None."""
    ofs = struct.unpack_from("<i", raw, 8 + GAME_LUMP * 16)[0]
    want = struct.unpack(">i", tag)[0]
    for k in range(struct.unpack_from("<i", raw, ofs)[0]):
        gid, _flags, _version, at, length = struct.unpack_from("<iHHii", raw, ofs + 4 + 16 * k)
        if gid != want:
            continue
        if raw[at:at + 4] != b"LZMA":
            return raw[at:at + length]
        actual, packed = struct.unpack_from("<II", raw, at + 4)
        return bspmod._lzma(raw[at:at + 17 + packed], actual, tag.decode(), "game lump")
    return None


def placed(b, raw):
    """What a map places: [{model, at, angles, skin}], its static props and then its prop entities, which also say which pose
    they stand in (pose: their default animation, "" for the first)."""
    out = []
    d = _game_lump(raw, b"sprp")
    if d is not None:
        n = struct.unpack_from("<i", d, 0)[0]
        names = [d[4 + k * 128:4 + (k + 1) * 128].split(b"\0")[0].decode("latin-1").lower().replace("\\", "/") for k in range(n)]
        pos = 4 + n * 128
        pos += 4 + 2 * struct.unpack_from("<i", d, pos)[0]
        count = struct.unpack_from("<i", d, pos)[0]
        pos += 4
        size = (len(d) - pos) // count if count else 0
        for i in range(count):
            at = pos + i * size
            values = struct.unpack_from("<6f", d, at)
            kind = struct.unpack_from("<H", d, at + 24)[0]
            if kind < n:
                out.append({"model": names[kind], "at": list(values[:3]), "angles": list(values[3:]), "skin": struct.unpack_from("<i", d, at + 32)[0]})
    for e in b.entities:
        if e.get("classname") in PROP_CLASSES and e.get("model", "").endswith(".mdl") and "origin" in e:
            at, angles, skin = _numbers(e["origin"]), _numbers(e.get("angles", "0 0 0")), _numbers(e.get("skin") or "0")
            if len(at) == 3 and len(angles) == 3:
                out.append({"model": e["model"].lower().replace("\\", "/"), "at": at, "angles": angles, "skin": int(skin[0]) if skin else 0,
                            "pose": (e.get("DefaultAnim") or "").lower()})
    return out


def _numbers(text):
    """The numbers in an entity's value; mappers' files hold the odd stray character, which counts for nothing."""
    out = []
    for x in text.split():
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def rotation(angles):
    """A Source entity's angles (pitch, yaw, roll, degrees) as a 3 x 3 rotation matrix."""
    p, y, r = np.radians(angles)
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])


def _text(data, at):
    return data[at:data.index(b"\0", at)].decode("latin-1").lower().replace("\\", "/")


def _vertices(vvd, lod):
    n_lods = struct.unpack_from("<i", vvd, 12)[0]
    counts = struct.unpack_from("<8i", vvd, 16)
    n_fix, fix_at, data_at = struct.unpack_from("<3i", vvd, 48)
    every = np.frombuffer(vvd, VERTEX, counts[0], data_at)
    if not n_fix:
        return every
    pieces = [every[src:src + n] for level, src, n in (struct.unpack_from("<3i", vvd, fix_at + 12 * k) for k in range(n_fix)) if level >= min(lod, n_lods - 1)]
    return np.concatenate(pieces)


def _strip_group_indices(vtx, at, step, verts_needed):
    """The triangles of a strip group as positions in the mesh's vertices, or None if this reading of the file doesn't fit it."""
    n_verts, vert_at, n_idx, idx_at, n_strips, strip_at = struct.unpack_from("<6i", vtx, at)
    if not (0 <= n_verts <= 1 << 16 and 0 <= n_idx <= 1 << 22 and 0 <= n_strips <= 1 << 12) or at + idx_at + 2 * n_idx > len(vtx) or at + vert_at + 9 * n_verts > len(vtx):
        return None
    original = np.frombuffer(vtx, np.dtype([("skip", "V4"), ("id", "<u2"), ("tail", "V3")]), n_verts, at + vert_at)["id"]
    indices = np.frombuffer(vtx, "<u2", n_idx, at + idx_at)
    if len(indices) and (indices.max() >= n_verts or (len(original) and original.max() >= verts_needed)):
        return None
    tris = []
    strips = at + strip_at
    for s in range(n_strips):
        here = strips + s * step
        count, first = struct.unpack_from("<2i", vtx, here)
        flags = vtx[here + 18]
        part = indices[first:first + count]
        if flags & STRIP_STRIP:
            t = np.column_stack([part[:-2], part[1:-1], part[2:]])
            t[1::2] = t[1::2][:, [1, 0, 2]]
            tris.append(t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])])
        else:
            tris.append(part.reshape(-1, 3))
    return original[np.concatenate(tris)] if tris else np.zeros((0, 3), np.int64)


def _mesh_triangles(vtx, mesh_at, verts_needed):
    """All triangles of one .vtx mesh, as positions in the .mdl mesh's vertices. Which strip group size a file uses shows in whether it reads sensibly."""
    n_groups, groups_at = struct.unpack_from("<2i", vtx, mesh_at)
    for size in STRIP_GROUP_SIZES:
        out = [_strip_group_indices(vtx, mesh_at + groups_at + g * size, 27, verts_needed) for g in range(n_groups)]
        if all(x is not None for x in out):
            return np.concatenate(out) if out else np.zeros((0, 3), np.int64)
    return None


def _shape(mdl, vvd, vtx, lod):
    """(vertices, [(texture index, triangles)]) of a model's default body groups, at a level of detail; None if the files can't be read."""
    n_body, body_at = struct.unpack_from("<2i", mdl, 232)
    vtx_body = struct.unpack_from("<i", vtx, 32)[0]
    vertices = _vertices(vvd, lod)
    parts = []
    for k in range(n_body):
        here = body_at + k * 16
        n_models, _base, model_rel = struct.unpack_from("<3i", mdl, here + 4)
        if not n_models:
            continue
        model = here + model_rel
        n_mesh, mesh_rel, _nv, vert_at = struct.unpack_from("<4i", mdl, model + 72)
        first = vert_at // 48
        vb = vtx_body + k * 8
        _vn, vm_rel = struct.unpack_from("<2i", vtx, vb)
        vm = vb + vm_rel
        n_lods, lod_rel = struct.unpack_from("<2i", vtx, vm)
        vl = vm + lod_rel + min(lod, n_lods - 1) * 12
        vn_mesh, vmesh_rel = struct.unpack_from("<2i", vtx, vl)
        for m in range(min(n_mesh, vn_mesh)):
            mesh = model + mesh_rel + m * 116
            material, _a, _n, offset = struct.unpack_from("<4i", mdl, mesh)
            tris = _mesh_triangles(vtx, vl + vmesh_rel + m * 9, len(vertices))
            if tris is None:
                return None
            parts.append((material, tris + first + offset))
    if parts and max(int(t.max()) for _m, t in parts if t.size) >= len(vertices):
        return None
    return vertices, parts


def _file(folders, path):
    for f in folders:
        p = f / path
        if p.exists():
            return p.read_bytes()
    return None


def read(folders, path, pose=None, most=None):
    """A model: {points, normals, uvs, parts [(texture index, triangles)], textures [names by skin, per texture], folders of its
    materials}, in a pose (_posed) or, with None, as its file holds it; at full detail or, given most, with at most that many
    triangles (its finest level of detail with so few, else snapped: _simplified); None if it can't be found or read."""
    key = (tuple(folders), path, pose, most)
    if key in _read:
        return _read[key]
    out = None
    mdl, vvd, vtx = _file(folders, path), _file(folders, path[:-4] + ".vvd"), _file(folders, path[:-4] + ".dx90.vtx")
    if mdl and vvd and vtx and mdl[:4] == b"IDST" and vvd[:4] == b"IDSV":
        try:
            got = _shape(mdl, vvd, vtx, 0)
            if got is not None and most is not None and sum(len(t) for _m, t in got[1]) > most:
                got = _coarser(mdl, vvd, vtx, struct.unpack_from("<i", vvd, 12)[0], most) or got
        except (struct.error, ValueError, IndexError):
            got = None
        if got is not None and got[1]:
            n_tex, tex_at = struct.unpack_from("<2i", mdl, 204)
            n_cd, cd_at = struct.unpack_from("<2i", mdl, 212)
            n_ref, n_fam, skin_at = struct.unpack_from("<3i", mdl, 220)
            names = [_text(mdl, tex_at + t * 64 + struct.unpack_from("<i", mdl, tex_at + t * 64)[0]) for t in range(n_tex)]
            cds = [_text(mdl, struct.unpack_from("<i", mdl, cd_at + c * 4)[0]) for c in range(n_cd)]
            skins = [list(struct.unpack_from(f"<{n_ref}h", mdl, skin_at + f * n_ref * 2)) for f in range(n_fam)] if n_ref and n_fam else [list(range(n_tex))]
            v = got[0]
            points, normals = np.asarray(v["p"], np.float32), np.asarray(v["n"], np.float32)
            if pose is not None:
                points, normals = _posed(mdl, _file(folders, path[:-4] + ".ani"), v, pose)
            uvs = np.array(v["uv"], np.float32)
            uvs[np.abs(uvs) > 1e38] = 0  # a vertex the file leaves without a texture mapping has FLT_MAX there, as rounded in text (3.40282e+38)
            points, normals, uvs, parts = (points, normals, uvs, got[1]) if most is None else _simplified(points, normals, uvs, got[1], most)
            out = {"points": points, "normals": normals, "uvs": uvs, "parts": parts, "textures": names, "skins": skins, "cds": cds}
    if most is not None:  # a full shape is read once per map; only the few cut down are kept
        _read[key] = out
    return out


ANIM_RAWPOS, ANIM_RAWROT, ANIM_ANIMPOS, ANIM_ANIMROT, ANIM_DELTA, ANIM_RAWROT2 = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
BONE_SIZE, SEQ_SIZE, ANIM_DESC_SIZE = 216, 212, 100


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _euler_quat(a):
    """Source's RadianEuler (about x, y, z) as a quaternion (x, y, z, w)."""
    (sr, cr), (sp, cp), (sy, cy) = ((np.sin(t / 2), np.cos(t / 2)) for t in a)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def _quat48(data, at):
    x, y, zw = struct.unpack_from("<3H", data, at)
    q = [(x - 32768) / 32768, (y - 32768) / 32768, ((zw & 0x7FFF) - 16384) / 16384]
    w = np.sqrt(max(0.0, 1 - sum(c * c for c in q)))
    return np.array([*q, -w if zw >> 15 else w])


def _quat64(data, at):
    bits = struct.unpack_from("<Q", data, at)[0]
    q = [(((bits >> s) & 0x1FFFFF) - 1048576) / 1048576.5 for s in (0, 21, 42)]
    w = np.sqrt(max(0.0, 1 - sum(c * c for c in q)))
    return np.array([*q, -w if bits >> 63 else w])


def _first_value(data, at, scale):
    """Frame 0 of one channel of an animation (its run-length values start at at), times its scale: the run's first value, or as
    the game reads a run with none, the run's own count."""
    return struct.unpack_from("<h", data, at + 2 * bool(data[at]))[0] * scale


def _frame_zero(mdl, ani, sequence):
    """{bone: (quaternion or None, position or None, delta)} of a sequence's first frame, from its first animation; None if the file
    doesn't hold it (an animation in another model)."""
    n_seq, seq_at = struct.unpack_from("<2i", mdl, 188)
    if not n_seq:
        return None
    labels = [_text(mdl, seq_at + s * SEQ_SIZE + struct.unpack_from("<i", mdl, seq_at + s * SEQ_SIZE + 4)[0]) for s in range(n_seq)]
    seq = seq_at + (labels.index(sequence) if sequence in labels else 0) * SEQ_SIZE
    anim = struct.unpack_from("<h", mdl, seq + struct.unpack_from("<i", mdl, seq + 60)[0])[0]
    n_anim, anim_at = struct.unpack_from("<2i", mdl, 180)
    if not 0 <= anim < n_anim:
        return None
    desc = anim_at + anim * ANIM_DESC_SIZE
    block, index = struct.unpack_from("<2i", mdl, desc + 52)
    section_at, section_frames = struct.unpack_from("<2i", mdl, desc + 80)
    if section_frames:
        block, index = struct.unpack_from("<2i", mdl, desc + section_at)
    data, at = mdl, desc + index
    if block:
        n_blocks, blocks_at = struct.unpack_from("<2i", mdl, 352)
        if ani is None or not 0 < block < n_blocks:
            return None
        data, at = ani, struct.unpack_from("<i", mdl, blocks_at + block * 8)[0] + index
    if not index:
        return {}
    out = {}
    while True:
        bone, flags, step = struct.unpack_from("<BBh", data, at)
        body = at + 4
        q = p = None
        if flags & ANIM_RAWROT:
            q = _quat48(data, body)
        elif flags & ANIM_RAWROT2:
            q = _quat64(data, body)
        elif flags & ANIM_ANIMROT:
            q = ("euler", [struct.unpack_from("<h", data, body + 2 * j)[0] for j in range(3)], body)
        if flags & ANIM_RAWPOS:
            p = np.frombuffer(data, "<f2", 3, body + 6 * bool(flags & ANIM_RAWROT) + 8 * bool(flags & ANIM_RAWROT2)).astype(float)
        elif flags & ANIM_ANIMPOS:
            v = body + 6 * bool(flags & ANIM_ANIMROT)
            p = ("values", [struct.unpack_from("<h", data, v + 2 * j)[0] for j in range(3)], v)
        out[bone] = (q, p, bool(flags & ANIM_DELTA), data)
        if not step:
            return out
        at += step


def _bone_poses(mdl, ani, sequence):
    """Each bone's matrix (3 x 4) from where the .vvd holds a vertex to where the sequence's first frame puts it; None if the frame
    can't be read."""
    frame = _frame_zero(mdl, ani, sequence)
    if frame is None:
        return None
    n_bones, bones_at = struct.unpack_from("<2i", mdl, 156)
    to_model, out = [], []
    for k in range(n_bones):
        b = bones_at + k * BONE_SIZE
        parent = struct.unpack_from("<i", mdl, b + 4)[0]
        pos, quat, rot, pos_scale, rot_scale = (np.array(struct.unpack_from(f"<{n}f", mdl, b + o)) for n, o in ((3, 32), (4, 44), (3, 60), (3, 72), (3, 84)))
        q, p, delta, data = frame.get(k, (None, None, False, mdl))
        if isinstance(q, tuple):
            _kind, offsets, ptr = q
            euler = np.array([_first_value(data, ptr + o, s) if o else 0.0 for o, s in zip(offsets, rot_scale)])
            q = _euler_quat(euler if delta else euler + rot)
        elif q is None:
            q = np.array([0.0, 0.0, 0.0, 1.0]) if delta else quat
        if isinstance(p, tuple):
            _kind, offsets, ptr = p
            p = np.array([_first_value(data, ptr + o, s) if o else 0.0 for o, s in zip(offsets, pos_scale)])
            p = p if delta else p + pos
        elif p is None:
            p = np.zeros(3) if delta else pos
        m = np.column_stack([_quat_matrix(q / (np.linalg.norm(q) or 1)), p])
        if 0 <= parent < k:
            pm = to_model[parent]
            m = np.column_stack([pm[:, :3] @ m[:, :3], pm[:, :3] @ m[:, 3] + pm[:, 3]])
        to_model.append(m)
        pose_to_bone = np.array(struct.unpack_from("<12f", mdl, b + 96)).reshape(3, 4)
        out.append(np.column_stack([m[:, :3] @ pose_to_bone[:, :3], m[:, :3] @ pose_to_bone[:, 3] + m[:, 3]]))
    return np.array(out)


def _posed(mdl, ani, vertices, sequence):
    """A model's points and normals in the first frame of a sequence (by its name, the first if it has no such one), each vertex
    moved by its bones as weighted; as the file holds them if that frame can't be read."""
    points, normals = np.asarray(vertices["p"], np.float32), np.asarray(vertices["n"], np.float32)
    try:
        bones = _bone_poses(mdl, ani, sequence)
    except (struct.error, ValueError, IndexError):
        bones = None
    if bones is None or not len(bones):
        return points, normals
    ids = np.minimum(vertices["bones"][:, :3].astype(np.int64), len(bones) - 1)
    weights = np.asarray(vertices["weights"], np.float64)
    weights[np.arange(3)[None, :] >= np.maximum(vertices["bones"][:, 3:4].astype(np.int64), 1)] = 0
    weights /= weights.sum(1, keepdims=True).clip(1e-9)
    p, n = np.zeros((len(points), 3)), np.zeros((len(points), 3))
    for j in range(3):
        m = bones[ids[:, j]]
        w = weights[:, j:j + 1]
        p += w * (np.einsum("nij,nj->ni", m[:, :, :3], points) + m[:, :, 3])
        n += w * np.einsum("nij,nj->ni", m[:, :, :3], normals)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)
    return p.astype(np.float32), n.astype(np.float32)


def _coarser(mdl, vvd, vtx, lods, most):
    """The finest level of detail with few enough triangles, or the coarsest there is."""
    best = None
    for level in range(1, lods):
        best = _shape(mdl, vvd, vtx, level)
        if best is None or sum(len(t) for _m, t in best[1]) <= most:
            break
    return best


def _simplified(points, normals, uvs, parts, most):
    """A model with at most most triangles: its points snapped to an ever coarser grid until few enough triangles are left
    (each texture's part on its own, so they don't share points). Enough to see its shape and size. A corner keeps its own place in
    its texture: points close together can be far apart in a texture sheet, so a snapped point keeps its own, not its neighbour's."""
    if sum(len(t) for _m, t in parts) <= most:
        return points, normals, uvs, parts
    lo, size = points.min(0), max(float(np.ptp(points, axis=0).max()), 1e-6)
    owner = np.zeros(len(points), np.int64)
    for k, (_m, t) in enumerate(parts):
        owner[t.ravel()] = k
    steps = 64
    while True:
        cell = np.floor((points - lo) / size * steps).astype(np.int64)
        keys = ((owner * (steps + 1) + cell[:, 0]) * (steps + 1) + cell[:, 1]) * (steps + 1) + cell[:, 2]
        _keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        kept, count = [], 0
        for m, t in parts:
            r = inverse[t]
            t = t[(r[:, 0] != r[:, 1]) & (r[:, 1] != r[:, 2]) & (r[:, 0] != r[:, 2])]
            kept.append((m, t))
            count += len(t)
        if count <= most or steps <= 4:
            used, renumber = np.unique(np.concatenate([t.ravel() for _m, t in kept]), return_inverse=True)
            out, at = [], 0
            for m, t in kept:
                out.append((m, renumber[at:at + t.size].reshape(-1, 3)))
                at += t.size
            return points[first][inverse][used], normals[used], uvs[used], out
        steps = max(4, int(steps * 0.7))


def material(folders, model, texture):
    """The material name of one of a model's textures: the first of its folders that has it; None if none does."""
    for cd in model["cds"]:
        name = (cd + texture).replace("//", "/")
        if vtf.material(folders, name)[0] is not None:
            return name
    return None


# GoldSrc's models (Half-Life and its add-ons): a map places them as entities, mostly by class alone; one .mdl holds the shape, its
# bones and its textures (or a <name>t.mdl beside it does).
GS_CLASSES = {
    "item_healthkit": "w_medkit", "item_battery": "w_battery", "item_suit": "w_suit", "item_longjump": "w_longjump", "item_antidote": "w_antidote",
    "item_security": "w_security", "item_helmet": "barney_helmet", "item_armorvest": "barney_vest",
    "weapon_crowbar": "w_crowbar", "weapon_9mmhandgun": "w_9mmhandgun", "weapon_glock": "w_9mmhandgun", "weapon_357": "w_357", "weapon_python": "w_357",
    "weapon_9mmAR": "w_9mmar", "weapon_mp5": "w_9mmar", "weapon_shotgun": "w_shotgun", "weapon_crossbow": "w_crossbow", "weapon_rpg": "w_rpg",
    "weapon_gauss": "w_gauss", "weapon_egon": "w_egon", "weapon_hornetgun": "w_hgun", "weapon_handgrenade": "w_grenade", "weapon_satchel": "w_satchel",
    "weapon_tripmine": "v_tripmine", "weapon_snark": "w_sqknest", "ammo_9mmclip": "w_9mmclip", "ammo_glockclip": "w_9mmclip", "ammo_9mmAR": "w_9mmarclip",
    "ammo_mp5clip": "w_9mmarclip", "ammo_9mmARclip": "w_9mmarclip", "ammo_9mmbox": "w_chainammo", "ammo_ARgrenades": "w_argrenade", "ammo_mp5grenades": "w_argrenade",
    "ammo_buckshot": "w_shotbox", "ammo_357": "w_357ammobox", "ammo_crossbow": "w_crossbow_clip", "ammo_rpgclip": "w_rpgammo", "ammo_gaussclip": "w_gaussammo",
    "ammo_egonclip": "w_gaussammo", "monster_scientist": "scientist", "monster_sitting_scientist": "scientist", "monster_scientist_dead": "scientist",
    "monster_rosenberg": "scientist", "monster_barney": "barney", "monster_barney_dead": "barney", "monster_human_grunt": "hgrunt", "monster_hgrunt_dead": "hgrunt",
    "monster_grunt_repel": "hgrunt", "monster_headcrab": "headcrab", "monster_babycrab": "baby_headcrab", "monster_zombie": "zombie", "monster_houndeye": "houndeye",
    "monster_bullchicken": "bullsquid", "monster_alien_slave": "islave", "monster_alien_slave_dead": "islave", "monster_alien_grunt": "agrunt",
    "monster_alien_controller": "controller", "monster_barnacle": "barnacle", "monster_cockroach": "roach", "monster_leech": "leech", "monster_gman": "gman",
    "monster_hevsuit_dead": "player", "monster_tentacle": "tentacle2", "monster_rat": "bigrat", "monster_gargantua": "garg", "monster_ichthyosaur": "icky",
    "monster_human_assassin": "hassassin", "monster_sentry": "sentry", "monster_turret": "turret", "monster_miniturret": "miniturret", "monster_apache": "apache",
    "monster_osprey": "osprey", "monster_bigmomma": "big_mom", "monster_nihilanth": "nihilanth", "monster_flyer_flock": "aflock", "monster_tripmine": "v_tripmine",
    "xen_plantlight": "light", "xen_hair": "hair", "xen_tree": "tree", "xen_spore_small": "fungus(small)", "xen_spore_medium": "fungus", "xen_spore_large": "fungus(large)",
    "ammo_spore": "spore", "ammo_556": "w_saw_clip", "ammo_762": "w_m40a1clip", "weapon_m249": "w_saw", "weapon_sniperrifle": "w_m40a1", "weapon_displacer": "w_displacer",
    "weapon_sporelauncher": "w_spore_launcher", "weapon_shockrifle": "w_shock_rifle", "weapon_eagle": "w_desert_eagle", "weapon_pipewrench": "w_pipe_wrench",
    "weapon_knife": "w_knife", "weapon_grapple": "w_bgrap", "weapon_penguin": "w_penguinnest", "monster_male_assassin": "massn", "monster_assassin_repel": "massn",
    "monster_pitdrone": "pit_drone", "monster_human_grunt_ally": "hgrunt_opfor", "monster_human_grunt_ally_dead": "hgrunt_opfor", "monster_grunt_ally_repel": "hgrunt_opfor",
    "monster_human_medic_ally": "hgrunt_medic", "monster_human_torch_ally": "hgrunt_torch", "monster_shocktrooper": "strooper", "monster_alien_voltigore": "voltigore",
    "monster_alien_babyvoltigore": "baby_voltigore", "monster_otis": "otis", "monster_otis_dead": "otis", "monster_drillsergeant": "drill", "monster_recruit": "recruit",
    "monster_cleansuit_scientist": "cleansuit_scientist", "monster_cleansuit_scientist_dead": "cleansuit_scientist", "monster_gonome": "gonome",
    "monster_zombie_soldier": "zombie_soldier", "monster_zombie_soldier_dead": "zombie_soldier", "monster_zombie_barney": "zombie_barney", "monster_skeleton_dead": "skeleton",
    "monster_houndeye_dead": "houndeye_dead", "monster_pitworm_up": "pit_worm_up", "monster_geneworm": "geneworm", "monster_blkop_apache": "blkop_apache", "monster_op4loader": "loader",
}
GS_MASKED = 0x40  # texture flag: palette entry 255 is see-through
_gs_read = {}


def gs_placed(b):
    """What a GoldSrc map places as models: [{model, at, angles, skin, body}], monsters, items and the like by their class, and entities that name their own .mdl."""
    out = []
    for e in b.entities:
        named = e.get("model", "").lower().replace("\\", "/")
        model = named if named.endswith(".mdl") else f"models/{GS_CLASSES[e.get('classname')]}.mdl" if e.get("classname") in GS_CLASSES else None
        at = _numbers(e.get("origin", ""))
        if model is None or len(at) != 3:
            continue
        angles = _numbers(e.get("angles", "")) or [0.0, (_numbers(e.get("angle", "0")) or [0.0])[0], 0.0]
        skin, body = (_numbers(e.get(k) or "0") or [0.0] for k in ("skin", "body"))
        out.append({"model": model, "at": at, "angles": angles if len(angles) == 3 else [0.0, 0.0, 0.0], "skin": int(skin[0]), "body": int(body[0])})
    return out


def _gs_rotation(a):
    """Half-Life's bone angles (radians about x, y, z) as a rotation matrix, by way of its quaternion."""
    (sr, cr), (sp, cp), (sy, cy) = ((np.sin(t / 2), np.cos(t / 2)) for t in a)
    x, y, z, w = sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _gs_bones(mdl):
    """Each bone's (rotation, position) in the model's first pose: its first sequence's first frame, or the bones' defaults when that
    sequence's frames are in another file."""
    n_bones, bone_at, _nc, _ci, _nh, _hi, n_seq, seq_at = struct.unpack_from("<8i", mdl, 140)
    anim_at = struct.unpack_from("<i", mdl, seq_at + 124)[0] if n_seq and struct.unpack_from("<i", mdl, seq_at + 156)[0] == 0 else None
    out = []
    for k in range(n_bones):
        at = bone_at + k * 112
        parent = struct.unpack_from("<i", mdl, at + 32)[0]
        value, scale = np.array(struct.unpack_from("<6f", mdl, at + 64)), np.array(struct.unpack_from("<6f", mdl, at + 88))
        if anim_at is not None:
            for j, rel in enumerate(struct.unpack_from("<6H", mdl, anim_at + k * 12)):
                if rel:
                    value[j] += struct.unpack_from("<h", mdl, anim_at + k * 12 + rel + 2)[0] * scale[j]
        r, t = _gs_rotation(value[3:]), value[:3]
        if 0 <= parent < len(out):
            pr, pt = out[parent]
            r, t = pr @ r, pr @ t + pt
        out.append((r, t))
    return out


def _gs_textures(mdl):
    """A model file's textures: ([names], [(width, height, picture)], skin families [[texture index by skin reference]])."""
    from PIL import Image
    n_tex, tex_at, _data_at, n_ref, n_fam, skin_at = struct.unpack_from("<6i", mdl, 180)
    names, pictures = [], []
    for k in range(n_tex):
        at = tex_at + k * 80
        names.append(mdl[at:at + 64].split(b"\0")[0].decode("latin-1").lower())
        flags, w, h, index = struct.unpack_from("<4i", mdl, at + 64)
        img = Image.frombytes("P", (w, h), mdl[index:index + w * h])
        img.putpalette(mdl[index + w * h:index + w * h + 768])
        if flags & GS_MASKED:
            img.info["transparency"] = 255
        pictures.append((w, h, img.convert("RGBA")))
    skins = [list(struct.unpack_from(f"<{n_ref}h", mdl, skin_at + f * n_ref * 2)) for f in range(n_fam)] if n_ref and n_fam else [list(range(n_tex))]
    return names, pictures, skins


def gs_read(folders, path, body=0):
    """A GoldSrc model in its first pose, as read gives a Source one, its pictures under images; None if it can't be found or read."""
    key = (tuple(folders), path, body)
    if key not in _gs_read:
        _gs_read[key] = None
        mdl = _file(folders, path)
        try:
            if mdl and mdl[:4] == b"IDST" and struct.unpack_from("<i", mdl, 4)[0] == 10:
                _gs_read[key] = _gs_shape(mdl, _file(folders, path[:-4] + "t.mdl"), body)
        except (struct.error, ValueError, IndexError):
            pass
    return _gs_read[key]


def _gs_shape(mdl, textures_file, body):
    names, pictures, skins = _gs_textures(textures_file if struct.unpack_from("<i", mdl, 180)[0] == 0 and textures_file else mdl)
    bones = _gs_bones(mdl)
    n_parts, part_at = struct.unpack_from("<2i", mdl, 204)
    corners, points, normals, uvs, parts = {}, [], [], [], []
    for k in range(n_parts):
        at = part_at + k * 76
        n_models, base, model_at = struct.unpack_from("<3i", mdl, at + 64)
        if not n_models:
            continue
        m = model_at + (body // max(base, 1)) % n_models * 112
        n_mesh, mesh_at, n_verts, vbone_at, v_at, n_norms, nbone_at, nrm_at = struct.unpack_from("<8i", mdl, m + 72)
        verts = np.frombuffer(mdl, "<f4", n_verts * 3, v_at).reshape(-1, 3)
        norms = np.frombuffer(mdl, "<f4", n_norms * 3, nrm_at).reshape(-1, 3)
        vbones, nbones = mdl[vbone_at:vbone_at + n_verts], mdl[nbone_at:nbone_at + n_norms]
        for j in range(n_mesh):
            _numtris, tri_at, skin_ref = struct.unpack_from("<3i", mdl, mesh_at + j * 20)
            texture = skins[0][skin_ref] if skin_ref < len(skins[0]) else 0
            w, h = pictures[texture][:2] if texture < len(pictures) else (1, 1)
            tris, at = [], tri_at
            while True:
                count = struct.unpack_from("<h", mdl, at)[0]
                at += 2
                if count == 0:
                    break
                run = []
                for v, n, s, t in struct.iter_unpack("<4h", mdl[at:at + 8 * abs(count)]):
                    c = (k, v, n, s, t, texture)
                    if c not in corners:
                        corners[c] = len(points)
                        br, bt = bones[vbones[v]]
                        points.append(br @ verts[v] + bt)
                        normals.append(bones[nbones[n]][0] @ norms[n])
                        uvs.append((s / w, t / h))
                    run.append(corners[c])
                at += 8 * abs(count)
                if count < 0:  # a fan
                    tris += [(run[0], run[i - 1], run[i]) for i in range(2, len(run))]
                else:  # a strip, every other triangle turned back the right way round
                    tris += [(run[i - 1], run[i - 2], run[i]) if i % 2 else (run[i - 2], run[i - 1], run[i]) for i in range(2, len(run))]
            if tris:
                parts.append((texture, np.array(tris, np.int64)))
    if not parts:
        return None
    p, n, uv, parts = _simplified(np.array(points, np.float32), np.array(normals, np.float32), np.array(uvs, np.float32), parts, GOLDSRC_MOST)
    return {"points": p, "normals": n, "uvs": uv, "parts": parts, "textures": names, "skins": skins, "images": [x[2] for x in pictures]}
