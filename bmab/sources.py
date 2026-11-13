"""Import the source material listed in sources.json into data/, pinned by sources.lock.json.

Each source is found on this machine (Steam app, Workshop item, or a file or unpacked folder the
user put into downloads/), the files matching its include patterns are copied to data/<id>/, and their
SHA-256 is checked against the lock. A source that differs from the lock is not imported unless
re-pinned.
"""
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import containers, steam

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "sources.json"
LOCK = REPO / "sources.lock.json"
HOME = Path(os.environ.get("BMAB_HOME") or REPO)  # the folder bmab keeps its own things in (setting: BMAB_HOME)
DATA = HOME / "data"
DOWNLOADS = HOME / "downloads"
CACHE = HOME / "cache"  # what bmab works out from the games and keeps to be quick next time: deleting it only costs time


@dataclass
class Result:
    id: str
    status: str  # imported, pinned, missing, differs
    detail: str = ""


def load():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    return manifest, lock


def where_to_get(src):
    f = src["from"]
    if "steam_app" in f:
        how = f"install Steam app {f['steam_app']}"
    elif "workshop" in f:
        app, item = f["workshop"]
        how = f"subscribe to https://steamcommunity.com/sharedfiles/filedetails/?id={item} (needs Steam app {app})"
    else:
        how = f"download {f['download']} from {f['page']} into {DOWNLOADS}"
    return f"{how}; then {src['get']}" if "get" in src else how


def _found(src):
    """Where the source is on this computer (its Steam folder, or what the person put into downloads/), or None."""
    f = src["from"]
    if "download" in f:
        hits = sorted(DOWNLOADS.rglob(f["download"])) if DOWNLOADS.is_dir() else []
        return hits[0] if hits else None
    base = steam.app_dir(f["steam_app"]) if "steam_app" in f else steam.workshop_dir(*f["workshop"])
    return base / f.get("path", "") if base is not None and (base / f.get("path", "")).is_dir() else None


def here():
    """Where each source stands on this computer: {id: (state, detail)}, state one of copied (every file the lock pins is in data/),
    found (on this computer, ready to copy: detail where) or missing (detail: how to get it)."""
    manifest, lock = load()
    out = {}
    for sid, src in manifest.items():
        pinned = lock.get(sid, {})
        if pinned and all((DATA / sid / k).is_file() for k in pinned):
            out[sid] = ("copied", f"{len(pinned)} files")
        else:
            where = _found(src)
            out[sid] = ("found", str(where)) if where is not None else ("missing", where_to_get(src))
    return out


def locate(src):
    """Return the source's file tree, or None when it isn't on this machine."""
    f = src["from"]
    patterns = src["include"] + src.get("expand", [])
    found = _found(src)
    if found is None:
        return None
    if "download" in f:
        if found.is_dir():  # an archive the user unpacked (formats bmab doesn't read, like 7z)
            tree = containers.folder(found / f.get("path", ""), patterns)
        elif found.suffix == ".zip":
            tree = containers.zip_members(found, f.get("path", ""))
        else:
            tree = {found.name.lower(): containers.Entry(found.stat().st_size, found.read_bytes, found)}
    else:
        tree = containers.folder(found, patterns)
    tree = containers.expand(tree, src.get("expand", []))
    root = f.get("root", "")
    return {k[len(root):]: v for k, v in tree.items() if k.startswith(root)}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def run(ids=None, pin=False, log=print, step=None):
    """Copy the sources (ids, or all) into data/, checked against the lock (pin: taking what is found as the lock). step(id, files
    done, files) as it goes. Returns a Result for each."""
    manifest, lock = load()
    was = json.dumps(lock, sort_keys=True)
    results = []
    for sid, src in manifest.items():
        if ids and sid not in ids:
            continue
        tree = locate(src)
        chosen = sorted(k for k in tree or {} if any(fnmatch.fnmatch(k, p) for p in src["include"]))
        if not chosen:
            detail = where_to_get(src) if tree is None else f"found, but nothing matches {src['include']}"
            results.append(Result(sid, "missing", detail))
            log(f"{sid}: missing")
            continue
        pinned = {} if pin else lock.get(sid, {})
        out = DATA / sid
        found, differ = {}, []
        for done, k in enumerate(chosen, 1):
            if step:
                step(sid, done, len(chosen))
            data = tree[k].read()
            found[k] = _sha(data)
            if pinned and pinned.get(k) != found[k]:
                differ.append(k)  # never write a file the lock disagrees with
                continue
            dest = out / k
            if not (dest.exists() and dest.stat().st_size == len(data) and _sha(dest.read_bytes()) == found[k]):
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")  # a copy stopped halfway (the app closed) leaves no file that looks copied
                part.write_bytes(data)
                part.replace(dest)
        gone = sorted(set(pinned) - set(found))
        if differ or gone:
            results.append(Result(sid, "differs", f"{len(differ)} changed or new (e.g. {(differ + gone)[0]}), "
                                  f"{len(gone)} missing; re-pin with --pin if intended"))
            log(f"{sid}: differs from lock")
            continue
        for p in list(out.rglob("*")):
            if p.is_file() and p.relative_to(out).as_posix() not in found:
                p.unlink()
        status = "imported" if pinned else "pinned"
        lock[sid] = found
        size = sum(tree[k].size for k in found)
        results.append(Result(sid, status, f"{len(found)} files, {size / 1e6:.0f} MB"))
        log(f"{sid}: {status}, {len(found)} files")
    if json.dumps(lock, sort_keys=True) != was:
        LOCK.write_text(json.dumps({k: lock[k] for k in sorted(lock)}, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return results


PROGRESS = CACHE / "copying.json"  # how far copying games in has got: {source, done, files}


def copy_in(ids):
    """run(ids) as the app's job (bmab.jobs), saying how far it has got in PROGRESS as it goes: [{id, status, detail}]."""
    def step(sid, done, files):
        if done == 1 or done == files or done % 100 == 0:
            PROGRESS.parent.mkdir(parents=True, exist_ok=True)
            part = PROGRESS.with_name(PROGRESS.name + ".part")
            part.write_text(json.dumps({"source": sid, "done": done, "files": files}), encoding="utf-8", newline="\n")
            part.replace(PROGRESS)
    PROGRESS.unlink(missing_ok=True)
    return [vars(r) for r in run(ids, log=lambda _: None, step=step)]
