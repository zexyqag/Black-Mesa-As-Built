"""The redlines: what people have said about the maps (bmab.claims), as opposed to what can be worked out from them, in a git
repository of their own. This module reads and writes its files, commits them and syncs them with the shared repository.

Settings (environment): BMAB_REDLINES, the folder (by default next to this repository); BMAB_REDLINES_REPO, the git address to clone
into it when it isn't a clone yet. maps/<game>/<map>.jsonl holds a map's claims (bmab.claims); the register (bmab.register) is
sources.jsonl, places.jsonl, facts.jsonl, events.jsonl and evidence/<source>.jsonl; rules/<name>.json holds a decided modelling rule (surface:
the ground rule). One entry per line, sorted, so that edits to different things merge cleanly; git says who changed what and when.
A map is named by its game (a source) and its name in that game. What others push is taken in entry by entry (sync), and where
someone else changed an entry this computer changed too, a person decides which to keep (conflicts, decide).
"""
import base64
import collections
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import github, register as schema, sources

REDLINES = Path(os.environ.get("BMAB_REDLINES") or sources.REPO.parent / "Black-Mesa-Redlines")
GIT = sources.HOME / "git"  # where bmab's download for Windows carries git (MinGit), which few people there have
NO_GIT = "git isn't installed here: install it (https://git-scm.com/downloads) and start the app again"
lock = threading.Lock()  # held while the files change: by edits (bmab.editing) and by commits
_offered = {}  # {commit: pull request address} of the last push that had to go through a fork


TABLES = ("sources", "places", "facts", "events")


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _by_id(row):
    return row["id"]


def _by_kind(claim):
    return claim["kind"], claim["id"]


def _lines(rows, order=_by_id):
    """Entries as their file holds them: one per line, sorted, so that an edited line stays where it is."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=order))


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_lines(rows), encoding="utf-8", newline="\n")


def table(name):
    """A table of the register (TABLES), one dict per entry."""
    return _rows(REDLINES / f"{name}.jsonl")


def write_table(name, rows):
    _write(REDLINES / f"{name}.jsonl", rows)


def evidence(source=None):
    """The register's evidence: a source's, or all of it."""
    paths = [REDLINES / "evidence" / f"{source}.jsonl"] if source else sorted((REDLINES / "evidence").glob("*.jsonl"))
    return [e for p in paths for e in _rows(p)]


def write_evidence(source, rows):
    _write(REDLINES / "evidence" / f"{source}.jsonl", rows)


def register():
    """The whole register, keyed for looking up: {sources: {id: entry}, places: {id: entry}, evidence: {id: entry}, facts: [entry],
    events: [entry]}."""
    return {"sources": {s["id"]: s for s in table("sources")}, "places": {p["id"]: p for p in table("places")},
            "evidence": {e["id"]: e for e in evidence()}, "facts": table("facts"), "events": table("events")}


def save_entry(table, entry):
    """Add a register entry or replace the one with its id, once it passes the register's checks against everything else (a
    fact without an id gets one from what it says). Returns it."""
    return save_entries(table, [entry])[0]


def save_entries(table, entries):
    """save_entry for several entries of one table at once: all of them pass the checks or none is saved."""
    with lock:
        reg = register()
        checked = []
        for entry in entries:
            if table == "sources":
                schema.check_source(entry, reg["evidence"])
            elif table == "places":
                schema.check_place(entry, {**reg["places"], entry.get("id"): entry})
            elif table == "evidence":
                schema.check_evidence(entry, reg["sources"])
            elif table == "facts":
                if "id" not in entry:
                    body = {k: entry[k] for k in ("subject", "relation", "object", "value") if k in entry}
                    entry = {"id": hashlib.sha1(json.dumps([body, sorted(entry.get("evidence", []))], sort_keys=True).encode()).hexdigest()[:8], **entry}
                schema.check_fact(entry, reg["places"], reg["evidence"])
            elif table == "events":
                schema.check_event(entry, reg["places"], reg["evidence"])
            else:
                raise ValueError(f"no table {table}")
            checked.append(entry)
        ids = {e["id"] for e in checked}
        if table == "evidence":
            moved = {reg["evidence"][e["id"]]["source"] for e in checked if e["id"] in reg["evidence"]}  # a source's file it may leave
            for source in moved | {e["source"] for e in checked}:
                write_evidence(source, [e for e in evidence(source) if e["id"] not in ids] + [e for e in checked if e["source"] == source])
        else:
            write_table(table, [r for r in table_rows(table) if r["id"] not in ids] + checked)
    return checked


def table_rows(table):
    return evidence() if table == "evidence" else _rows(REDLINES / f"{table}.jsonl")


def delete_entry(table, entry_id):
    """Take a register entry away, unless something still refers to it (a fact or an event to a place or a piece of evidence, a
    verdict to its grounds, evidence to a source, an implied place to the places implying it)."""
    with lock:
        reg = register()
        users = {"places": [f["id"] for f in reg["facts"] if entry_id in (f["subject"], f.get("object"))]
                 + [p["id"] for p in reg["places"].values() if entry_id in p.get("implied_by", [])]
                 + [v["id"] for v in reg["events"] if v.get("place") == entry_id],
                 "evidence": [x["id"] for x in reg["facts"] + reg["events"] if entry_id in x["evidence"]]
                 + [s["id"] for s in reg["sources"].values() if entry_id in s.get("grounds", [])],
                 "sources": [e["id"] for e in reg["evidence"].values() if e["source"] == entry_id],
                 "facts": [], "events": []}.get(table)
        if users is None:
            raise ValueError(f"no table {table}")
        if users:
            raise ValueError(f"{len(users)} entries still refer to {entry_id}, such as {users[0]}")
        if table == "evidence":
            e = reg["evidence"].get(entry_id)
            if e is None:
                raise ValueError(f"no evidence {entry_id}")
            write_evidence(e["source"], [x for x in evidence(e["source"]) if x["id"] != entry_id])
        else:
            rows = table_rows(table)
            if not any(r["id"] == entry_id for r in rows):
                raise ValueError(f"no {table} entry {entry_id}")
            write_table(table, [r for r in rows if r["id"] != entry_id])
    return {"id": entry_id}


def source(source_id):
    """A source's entry in the register."""
    s = next((s for s in table("sources") if s["id"] == source_id), None)
    if s is None:
        raise ValueError(f"the register has no source {source_id}")
    return s


def tier(source_id):
    """How much a source counts against the others (1 to 5, the lowest first)."""
    return source(source_id)["tier"]


def rule(name):
    """A decided modelling rule (rules/<name>.json), or None."""
    path = REDLINES / "rules" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _path(game, name):
    return REDLINES / "maps" / game / f"{name}.jsonl"


def claims(game, name):
    """What the redlines claim about a map of a game, one dict per claim."""
    return _rows(_path(game, name))


def every_map():
    """{(game, map): its claims} of every map the redlines hold."""
    return {(p.parent.name, p.stem): claims(p.parent.name, p.stem) for p in sorted((REDLINES / "maps").glob("*/*.jsonl"))}


def has_map(game, name):
    return _path(game, name).exists()


def said(game, name, kind):
    """People's whole answer about one kind of claim in a map: its claims of that kind, once a person has checked all of that kind
    there (the map's review claim lists it); None until then."""
    rows = claims(game, name)
    reviewed = next((r["kinds"] for r in rows if r["kind"] == "review"), [])
    return [r for r in rows if r["kind"] == kind] if kind in reviewed else None


def write_map(game, name, rows):
    """A map's claims, one per line, sorted so that an edited line stays where it is and edits to different things merge. A map
    nobody has said anything about has no file."""
    path = _path(game, name)
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_lines(rows, _by_kind), encoding="utf-8", newline="\n")


def place_table():
    """Where the maps are, as the layout reads it, from the claims: {here: [{place, game, maps}], adjacent, same_place: [{maps,
    align: {to, translation, how}}]}. A place claim puts its map in its place; a same place claim aligns its map with the other
    (translation: this map's coordinates less the other's, as bmab.shared's peaks have it). Which maps lie next to which places
    will come from the direction signs' arrows (the facts register), and adjacent is empty until then."""
    here, same = {}, []
    for (game, name), rows in every_map().items():
        for r in rows:
            if r["kind"] == "place":
                here.setdefault((r["place"], game), set()).add(name)
            if r["kind"] == "same_place":
                same.append({"maps": [f"{game}:{name}"], "align": {"to": f"{r['game']}:{r['map']}", "translation": [a - b for a, b in zip(r["at"], r["there"])],
                                                                  "how": "marked by hand" + (", another version" if r.get("changed") else "")}})
    return {"here": [{"place": p, "game": g, "maps": sorted(ms)} for (p, g), ms in sorted(here.items())], "adjacent": [], "same_place": same}


def decided_links(game):
    """What people say about the level changes of a game's maps, as the layout reads it (bmab.graph.build's decided): [{maps,
    landmark, gives, kind}], from their level change claims."""
    out = {}
    for (g, name), rows in every_map().items():
        for r in rows:
            if g == game and r["kind"] == "level_change" and r["game"] == game:
                out[(*sorted([name, r["map"]]), r["landmark"])] = {"maps": sorted([name, r["map"]]), "landmark": r["landmark"], "gives": bool(r.get("loose")),
                                                                    **({"kind": r["by"]} if r.get("by") else {})}
    return list(out.values())


def _signed(token):
    """git's options to sign in with a GitHub token for one command: never written to the clone, and the machine's own login isn't
    tried instead."""
    if not token:
        return []
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", "credential.helper=", "-c", f"http.extraHeader=Authorization: Basic {auth}"]


def git():
    """The git to run: the one bmab's download carries (GIT), else the one installed; None if there is neither."""
    carried = GIT / "cmd" / ("git.exe" if os.name == "nt" else "git")
    return str(carried) if carried.is_file() else shutil.which("git")


def _run(*args, env=None, timeout=None):
    if not git():
        raise ValueError(NO_GIT)
    try:
        return subprocess.run([git(), *args], capture_output=True, encoding="utf-8", timeout=timeout, env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})})
    except subprocess.TimeoutExpired:
        raise ValueError(f"git didn't answer within {timeout} s") from None


def _git(*args, author=None, token=None, env=None, timeout=None):
    """git in the redlines, never asking anything. author: {name, email} to commit as; token: _signed; env: more of git's
    environment."""
    who = [a for k in ("name", "email") if author for a in ("-c", f"user.{k}={author[k]}")]
    return _run("-C", str(REDLINES), *who, *_signed(token), *args, env=env, timeout=timeout)


def committed(path, rev="HEAD"):
    """A file's entries as the redlines' commit rev has them (path inside the redlines), or None if it hasn't the file."""
    shown = _git("show", f"{rev}:{path}")
    return [json.loads(line) for line in shown.stdout.splitlines() if line.strip()] if shown.returncode == 0 else None


def history(path):
    """A file's entries through the redlines' commits since it was last made (path: inside the redlines, as maps/<game>/<map>.jsonl
    or evidence/<source>.jsonl): (base, versions). base: {id: entry} as the commit that made the file wrote them (what a detector
    or the seeding said); versions: {id: [entry, ...]} every version each entry has had since, oldest first. Edits not committed
    yet are not in it."""
    log = _git("log", "--reverse", "-p", "--no-color", "--format=%x00", "--", path).stdout
    base, versions = {}, {}
    for commit in log.split("\0")[1:]:
        made = "\nnew file mode" in commit
        if made:
            base, versions = {}, {}
        for line in commit.splitlines():
            if line.startswith("+{"):
                r = json.loads(line[1:])
                versions.setdefault(r["id"], []).append(r)
                if made:
                    base[r["id"]] = r
    return base, versions


def origin():
    """The git address the redlines were cloned from."""
    return _git("remote", "get-url", "origin").stdout.strip()


def waiting():
    """Whether the redlines are still to be cloned (BMAB_REDLINES_REPO set, the folder not a clone yet)."""
    return bool(os.environ.get("BMAB_REDLINES_REPO")) and not (REDLINES / ".git").exists()


def clone(token=None):
    """Clone BMAB_REDLINES_REPO into the folder if it isn't a clone yet: with token (someone signing in with GitHub), else with
    git's own credentials. Whether they are there now."""
    if waiting():
        done = _run(*_signed(token), "clone", os.environ["BMAB_REDLINES_REPO"], str(REDLINES))
        if done.returncode and token:
            raise ValueError(f"couldn't clone the redlines: {done.stderr.strip()}")
    return not waiting()


REGISTER_PATHS = ("sources.jsonl", "places.jsonl", "facts.jsonl", "events.jsonl", "evidence", "rules")


def changed():
    """The maps whose file differs from what is committed: [(game, map)]."""
    paths = [Path(line[3:].strip('"')) for line in _git("status", "--porcelain", "--untracked-files=all", "--", "maps").stdout.splitlines()]
    return sorted((p.parent.name, p.stem) for p in paths if len(p.parts) == 3 and p.suffix == ".jsonl")  # maps/<game>/<map>.jsonl


def changed_register():
    """The register's files that differ from what is committed (sources.jsonl, evidence/<source>.jsonl...)."""
    return sorted(line[3:].strip('"') for line in _git("status", "--porcelain", "--untracked-files=all", "--", *REGISTER_PATHS).stdout.splitlines())


def _register_summary(paths):
    """What changed in the register's files since the last commit, in words: added, edited and deleted entries per table."""
    counts = collections.Counter()
    for path in paths:
        if not path.endswith(".jsonl"):
            counts["edit", path] += 1
            continue
        was = {r["id"]: r for r in committed(path) or []}
        now = {r["id"]: r for r in _rows(REDLINES / path)}
        table = path.split("/")[0].removesuffix(".jsonl")
        for i in was.keys() | now.keys():
            verb = "add" if i not in was else "delete" if i not in now else "edit" if was[i] != now[i] else None
            if verb:
                counts[verb, table] += 1
    return ", ".join(f"{verb} {n} {table}" for (verb, table), n in sorted(counts.items()))


def summary(game, name):
    """What changed in a map since its last commit, in words."""
    was = {r["id"]: r for r in committed(f"maps/{game}/{name}.jsonl") or []}
    now = {r["id"]: r for r in claims(game, name)}
    counts = collections.Counter()
    for i in was.keys() | now.keys():
        if i not in was:
            counts["add", now[i]["kind"]] += 1
        elif i not in now:
            counts["delete", was[i]["kind"]] += 1
        elif was[i] != now[i]:
            counts["edit", now[i]["kind"]] += 1
    return ", ".join(f"{verb} {n} {kind}{'s' if n > 1 else ''}" for (verb, kind), n in sorted(counts.items()))


def commit(author=None):
    """One commit per changed map, as author ({name, email}; else as git is set up). Not while a conflict waits: committing would
    decide it."""
    with lock:
        _undecided()
        for game, name in changed():
            _git("add", f"maps/{game}/{name}.jsonl")
            done = _git("commit", "-m", f"{game}/{name}: {summary(game, name)}", author=author)
            if done.returncode:
                raise ValueError(done.stderr.strip() or done.stdout.strip())
        if paths := changed_register():
            _git("add", "--", *paths)
            done = _git("commit", "-m", f"register: {_register_summary(paths)}", author=author)
            if done.returncode:
                raise ValueError(done.stderr.strip() or done.stdout.strip())


def discard_register():
    """Put the register back as it was last committed (files never committed go)."""
    with lock:
        for path in changed_register():
            if _git("cat-file", "-e", f"HEAD:{path}").returncode:
                (REDLINES / path).unlink(missing_ok=True)
            elif (done := _git("checkout", "HEAD", "--", path)).returncode:
                raise ValueError(done.stderr.strip())


def discard(maps):
    """Put maps ([(game, map)]) back as they were last committed; a map never committed goes."""
    with lock:
        for game, name in maps:
            if _git("cat-file", "-e", f"HEAD:maps/{game}/{name}.jsonl").returncode:
                _path(game, name).unlink(missing_ok=True)
                continue
            done = _git("checkout", "HEAD", "--", f"maps/{game}/{name}.jsonl")
            if done.returncode:
                raise ValueError(done.stderr.strip())


def push(account=None):
    """Push the commits, once what others pushed is taken in (sync) and no conflict waits. Signed in with GitHub (account: {token,
    login}) and the redlines on GitHub: with their token, or, if they may not push there, to their fork's bmab-<login> branch with a
    pull request (pushing again updates it). Otherwise with git's own credentials. A GitHub token only ever goes to GitHub. The
    number of commits taken in."""
    taken = sync(account)
    repo = github.repository(origin()) if account else None
    with lock:
        _undecided()
        if not int(_git("rev-list", "--count", "@{upstream}..HEAD").stdout.strip() or 0):
            return taken
        if repo and not (allowed := github.access(repo, account["token"]))[0]:
            fork, branch = github.fork(repo, account["token"]), f"bmab-{account['login']}"
            for _ in range(10):  # a new fork takes GitHub a moment
                done = _git("push", "--force", f"https://github.com/{fork}.git", f"HEAD:refs/heads/{branch}", token=account["token"])
                if done.returncode == 0:
                    break
                time.sleep(3)
            if done.returncode:
                raise ValueError(done.stderr.strip())
            n = _git("rev-list", "--count", "@{upstream}..HEAD").stdout.strip()
            _offered.clear()
            _offered[_head()] = github.pull_request(repo, f"{fork.split('/')[0]}:{branch}", allowed[1], f"Claims from bmab by {account['login']}",
                                                    f"{n} commits of claims changed in bmab.", account["token"])
            return taken
        done = _git("push", token=account["token"] if repo else None)
    if done.returncode:
        raise ValueError(done.stderr.strip())
    return taken


def _head():
    return _git("rev-parse", "HEAD").stdout.strip()


def status():
    """{git: whether git is here, here: whether the redlines are cloned, changed: [(game, map)], register: the register's changed
    files, ahead: commits to push, pull_request: the one the last push opened, while it holds every commit here, conflicts: {unit:
    how many of its entries wait to be decided}}."""
    if not git():
        return {"git": False, "here": not waiting(), "changed": [], "register": [], "ahead": 0, "pull_request": None, "conflicts": {}}
    ahead = _git("rev-list", "--count", "@{upstream}..HEAD")
    return {"git": True, "here": not waiting(), "changed": changed(), "register": changed_register(), "ahead": int(ahead.stdout.strip() or 0) if ahead.returncode == 0 else 0,
            "pull_request": _offered.get(_head()), "conflicts": dict(collections.Counter(c["unit"] for c in conflicts()))}


# ---------- taking in others' changes ----------
# A unit is what is merged as one: a map's claims, a table of the register, every evidence file together (an entry moves between
# them with its source), a rule. Its entries are {id: entry}; a rule's, {its path: its text}.

TABLE_FILES = tuple(f"{t}.jsonl" for t in TABLES)
CONFLICTS = "bmab-conflicts.json"  # in the clone's .git, so never committed: {unit: {id: {base, theirs}}}


def _unit(path):
    """The unit a file of the redlines (its path inside them) belongs to; None for a file bmab doesn't write (the README)."""
    parts = path.split("/")
    if (len(parts) == 3 and parts[0] == "maps" and path.endswith(".jsonl")) or path in TABLE_FILES or (len(parts) == 2 and parts[0] == "rules" and path.endswith(".json")):
        return path
    return "evidence" if len(parts) == 2 and parts[0] == "evidence" and path.endswith(".jsonl") else None


def _rev(name):
    """The commit name names, or None."""
    return _git("rev-parse", "--verify", "--quiet", f"{name}^{{commit}}").stdout.strip() or None


def _changed_between(a, b):
    """The paths whose files differ between commits a and b."""
    return {p for p in _git("diff", "--name-only", "-z", a, b).stdout.split("\0") if p}


def _edited():
    """The paths whose files here differ from the last commit (edits not committed, new files among them)."""
    out = _git("status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split("\0")
    return {e[3:] for e in out if len(e) > 3}


def _unit_paths(unit, rev):
    """The files a unit is in at commit rev."""
    if unit == "evidence":
        return [p for p in _git("ls-tree", "--name-only", "-z", rev, "evidence/").stdout.split("\0") if p.endswith(".jsonl")]
    return [unit] if _git("cat-file", "-e", f"{rev}:{unit}").returncode == 0 else []


def _at(unit, rev):
    """A unit's entries as commit rev has them."""
    if unit.startswith("rules/"):
        shown = _git("show", f"{rev}:{unit}")
        return {unit: shown.stdout} if shown.returncode == 0 else {}
    return {e["id"]: e for path in _unit_paths(unit, rev) for e in committed(path, rev) or []}


def _here(unit):
    """A unit's entries as the files here have them."""
    if unit.startswith("rules/"):
        path = REDLINES / unit
        return {unit: path.read_text(encoding="utf-8")} if path.exists() else {}
    return {e["id"]: e for e in (evidence() if unit == "evidence" else _rows(REDLINES / unit))}


def _texts(unit, entries):
    """{path: text} of the files that hold a unit's entries, as bmab writes them (a map nobody says anything about has none)."""
    if unit.startswith("rules/"):
        return dict(entries)
    if unit == "evidence":
        by_source = collections.defaultdict(list)
        for e in entries.values():
            by_source[f"evidence/{e['source']}.jsonl"].append(e)
        return {path: _lines(rows) for path, rows in by_source.items()}
    if unit.startswith("maps/"):
        return {unit: _lines(entries.values(), _by_kind)} if entries else {}
    return {unit: _lines(entries.values())}


def _put(unit, entries):
    """Write a unit's entries into the files here; its files left without entries go."""
    texts = _texts(unit, entries)
    present = [f"evidence/{p.name}" for p in (REDLINES / "evidence").glob("*.jsonl")] if unit == "evidence" else [unit]
    for path in set(present) - texts.keys():
        (REDLINES / path).unlink(missing_ok=True)
    for path, text in texts.items():
        (REDLINES / path).parent.mkdir(parents=True, exist_ok=True)
        (REDLINES / path).write_text(text, encoding="utf-8", newline="\n")


def _take(unit, rev):
    """A unit's files here as commit rev has them, byte for byte."""
    paths = _unit_paths(unit, rev)
    present = [f"evidence/{p.name}" for p in (REDLINES / "evidence").glob("*.jsonl")] if unit == "evidence" else [unit]
    for path in set(present) - set(paths):
        (REDLINES / path).unlink(missing_ok=True)
    if paths:
        _git("checkout", rev, "--", *paths)


def _merge(base, ours, theirs):
    """Three-way merge of a unit's entries by id: (merged {id: entry} leaving out the conflicting ids, the conflicting ids). An
    entry added, changed or deleted on one side only is that side's; changed alike on both, either; changed differently on both, a
    conflict."""
    merged, clash = {}, []
    for i in base.keys() | ours.keys() | theirs.keys():
        b, o, t = base.get(i), ours.get(i), theirs.get(i)
        if o == t or t == b:
            kept = o
        elif o == b:
            kept = t
        else:
            clash.append(i)
            continue
        if kept is not None:
            merged[i] = kept
    return merged, clash


def _scratch(name, text):
    """A file in the clone's .git with text, written as is: git reads what is handed to it this way, as a text pipe would turn
    every newline into the system's (Windows: CRLF)."""
    path = REDLINES / ".git" / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return str(path)


def _commit_on(parent, changes, like):
    """A commit on parent with its units' entries as changes has them ({unit: entries}) and commit like's message and author; None
    when that changes nothing."""
    index = {"GIT_INDEX_FILE": str(REDLINES / ".git" / "bmab-index")}
    _git("read-tree", parent, env=index)
    for unit, entries in changes.items():
        texts = _texts(unit, entries)
        for path in set(_unit_paths(unit, parent)) - texts.keys():
            _git("update-index", "--force-remove", "--", path, env=index)
        for path, text in texts.items():
            blob = _git("hash-object", "-w", "--no-filters", _scratch("bmab-blob", text)).stdout.strip()
            _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=index)
    tree = _git("write-tree", env=index).stdout.strip()
    if tree == _git("rev-parse", f"{parent}^{{tree}}").stdout.strip():
        return None
    name, email, date, message = _git("log", "-1", "--format=%an%x00%ae%x00%aI%x00%B", like).stdout.split("\0", 3)
    who = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
    made = _git("commit-tree", tree, "-p", parent, "-F", _scratch("bmab-message", message), env=who)
    if made.returncode:
        raise ValueError(made.stderr.strip())
    return made.stdout.strip()


def _keep_a_copy(head):
    """Everything here as it is, edits included, kept as refs/bmab/before-sync: what to go back to by hand if taking in went wrong."""
    index = {"GIT_INDEX_FILE": str(REDLINES / ".git" / "bmab-index"), "GIT_AUTHOR_NAME": "bmab", "GIT_AUTHOR_EMAIL": "bmab@localhost",
             "GIT_COMMITTER_NAME": "bmab", "GIT_COMMITTER_EMAIL": "bmab@localhost"}
    _git("read-tree", head, env=index)
    _git("add", "--all", env=index)
    tree = _git("write-tree", env=index).stdout.strip()
    copy = _git("commit-tree", tree, "-p", head, "-m", "bmab: the redlines here before taking in others' changes", env=index).stdout.strip()
    _git("update-ref", "refs/bmab/before-sync", copy)


def sync(account=None):
    """Take in what others have pushed to the redlines, keeping everything of this computer's: commits not pushed yet are made again
    on top of theirs (same message and author), edits not committed stay edits. Entry by entry (_merge): where this computer and
    the shared redlines changed an entry differently, the commits keep theirs and the files here keep yours, and the conflict waits
    (conflicts) until someone decides it. With account, signed in with GitHub ({token}). The number of commits taken in."""
    with lock:
        if not _rev("@{upstream}"):
            return 0
        repo = github.repository(origin()) if account else None
        fetched = _git("fetch", "--quiet", "origin", token=account["token"] if repo else None, timeout=120)
        if fetched.returncode:
            raise ValueError(f"couldn't get what others pushed: {fetched.stderr.strip()}")
        old, up = _rev("HEAD"), _rev("@{upstream}")
        base = _git("merge-base", old, up).stdout.strip()
        if base == up:
            return 0
        _keep_a_copy(old)
        cur, clashed = up, collections.defaultdict(set)
        for c in _git("rev-list", "--reverse", f"{base}..{old}").stdout.split():  # this computer's commits again, on top
            changes = {}
            for unit in {u for u in map(_unit, _changed_between(f"{c}^", c)) if u}:
                now = _at(unit, cur)
                merged, clash = _merge(_at(unit, f"{c}^"), now, _at(unit, c))
                changes[unit] = merged | {i: now[i] for i in clash if i in now}
                clashed[unit].update(clash)
            cur = _commit_on(cur, changes, c) or cur
        edited = _edited()
        work, undecided = {}, {}
        for unit in {u for u in map(_unit, _changed_between(old, cur) | edited) if u}:  # the edits here, on top of that
            mine, now = _here(unit), _at(unit, cur)
            merged, clash = _merge(_at(unit, old), mine, now)
            started = _at(unit, base) if clash or clashed[unit] else {}
            for i in {*clash, *clashed[unit]}:
                if mine.get(i) is not None:
                    merged[i] = mine[i]
                else:
                    merged.pop(i, None)
                if mine.get(i) != now.get(i):
                    undecided.setdefault(unit, {})[i] = {"base": started.get(i), "theirs": now.get(i)}
            work[unit] = merged, mine, now
        _git("update-ref", "-m", "bmab: took in others' changes", "HEAD", cur, old)
        _git("read-tree", cur)
        for unit, (merged, mine, now) in work.items():
            if merged == now:
                _take(unit, cur)
            elif merged != mine:
                _put(unit, merged)
        for path in _changed_between(old, cur) - edited:  # bmab's files are units; others (the README) as the new commit has them
            if _unit(path) is None:
                if _git("cat-file", "-e", f"{cur}:{path}").returncode == 0:
                    _git("checkout", cur, "--", path)
                else:
                    (REDLINES / path).unlink(missing_ok=True)
        kept = _conflicts_kept()
        for unit, entries in undecided.items():
            kept.setdefault(unit, {}).update(entries)
        _keep_conflicts(kept)
        return int(_git("rev-list", "--count", f"{old}..{up}").stdout.strip())


def _conflicts_kept():
    path = REDLINES / ".git" / CONFLICTS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _keep_conflicts(kept):
    path = REDLINES / ".git" / CONFLICTS
    kept = {u: e for u, e in kept.items() if e}
    if kept:
        path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def conflicts():
    """The entries this computer and the shared redlines changed differently, still to decide: [{unit, id, base, theirs, yours}]
    (None: no such entry, deleted). One the files here now have as theirs (taken, discarded, edited alike) is decided."""
    found = []
    for unit, entries in _conflicts_kept().items():
        here = _here(unit)
        found += [{"unit": unit, "id": i, **c, "yours": here.get(i)} for i, c in entries.items() if here.get(i) != c["theirs"]]
    return found


def decide(unit, entry_id, keep):
    """Settle a conflict: keep yours (the files here as they are) or take theirs (the entry as the shared redlines have it)."""
    if keep not in ("yours", "theirs"):
        raise ValueError("keep yours or theirs")
    with lock:
        kept = _conflicts_kept()
        c = kept.get(unit, {}).pop(entry_id, None)
        if c is None:
            raise ValueError("no such conflict")
        if keep == "theirs":
            entries = _here(unit)
            if c["theirs"] is None:
                entries.pop(entry_id, None)
            else:
                entries[entry_id] = c["theirs"]
            _put(unit, entries)
        _keep_conflicts(kept)


def _undecided():
    """Refuse while a conflict waits (ValueError naming where)."""
    if left := conflicts():
        where = sorted({"the register" if not c["unit"].startswith("maps/") else c["unit"][5:-6] for c in left})
        raise ValueError(f"first decide what to keep where someone else changed what you changed ({len(left)}: {', '.join(where)})")
