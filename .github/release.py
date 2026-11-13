"""The release workflow's steps (.github/workflows/release.yml), standard library only.

  python .github/release.py next                 version= and release= for $GITHUB_OUTPUT: a push to main, the next version (empty
                                                 when nothing since the last release is released); a run by hand, 0.0.0-<branch>.g<sha>
  python .github/release.py notes VERSION        the release's notes: its commits' subjects since the release before
  python .github/release.py versioninfo VERSION  bmab.exe's version details, for PyInstaller's --version-file
  python .github/release.py check ARCHIVE VERSION  the download, unpacked, starts, says its version and ends on Quit

Versions follow the commits' Conventional Commits prefixes: fix: and perf: release the next patch, feat: the next minor, a
breaking change (feat!:, fix!:, or a BREAKING CHANGE: line) the next major; before 1.0, the next minor for a breaking change and
the next patch for anything else. Other prefixes (docs:, ci:, refactor:, test:, chore:) and none release nothing. The first
release is 0.1.0.
"""
import http.client
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

NAME = "Black Mesa As Built"
PREFIX = re.compile(r"(\w+)(?:\([^)]*\))?(!)?: ")
KINDS = {"breaking": "Breaking changes", "feat": "Features", "fix": "Fixes"}


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _last(before="HEAD"):
    """The last release's tag reachable from before, or None."""
    got = subprocess.run(["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*.[0-9]*.[0-9]*", "--exclude", "v*-*", before],
                         capture_output=True, text=True)
    return got.stdout.strip() or None


def _commits(since, until="HEAD"):
    """(sha, kind, subject) of each commit after since up to until that releases something; kind is breaking, feat or fix."""
    found = []
    for entry in _git("log", "--format=%h%x1f%s%x1f%b%x1e", f"{since}..{until}" if since else until).split("\x1e"):
        if not entry.strip():
            continue
        sha, subject, body = entry.strip("\n").split("\x1f")
        m = PREFIX.match(subject)
        if not m:
            continue
        kind = "breaking" if m.group(2) or re.search(r"^BREAKING[ -]CHANGE: ", body, re.M) else {"feat": "feat", "fix": "fix", "perf": "fix"}.get(m.group(1))
        if kind:
            found.append((sha, kind, subject[m.end():]))
    return found


def next_version():
    last = _last()
    kinds = {kind for _, kind, _ in _commits(last)}
    if not kinds:
        return None
    if last is None:
        return "0.1.0"
    major, minor, patch = map(int, last[1:].split("."))
    if "breaking" in kinds:
        return f"{major + 1}.0.0" if major else f"0.{minor + 1}.0"
    if "feat" in kinds and major:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def notes(version):
    last = _last()
    commits = _commits(last, "HEAD")
    parts = [f"## {title}\n\n" + "".join(f"- {subject} ({sha})\n" for sha, k, subject in commits if k == kind)
             for kind, title in KINDS.items() if any(k == kind for _, k, _ in commits)]
    return "\n".join(([] if last else ["The first release of bmab.\n"]) + parts) or f"bmab {version}\n"


def versioninfo(version):
    numbers = tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()) + (0,)
    strings = {"CompanyName": "", "FileDescription": NAME, "FileVersion": version, "InternalName": "bmab", "LegalCopyright": "AGPL-3.0-or-later",
               "OriginalFilename": "bmab.exe", "ProductName": NAME, "ProductVersion": version}
    table = ", ".join(f"StringStruct({k!r}, {v!r})" for k, v in strings.items())
    return (f"VSVersionInfo(ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),\n"
            f"  kids=[StringFileInfo([StringTable('040904B0', [{table}])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])\n")


def _ask(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, r.read()


def check(archive, version):
    """Unpacked into a folder of its own, the download's bmab serve answers with the front page, says its version, and ends with
    exit code 0 when the front page's Quit asks it to. Exits 1 if not."""
    folder = Path(tempfile.mkdtemp())
    if archive.endswith(".zip"):
        zipfile.ZipFile(archive).extractall(folder)
    else:
        with tarfile.open(archive) as t:
            t.extractall(folder, filter="tar")
    exe = folder / "bmab" / ("bmab.exe" if sys.platform == "win32" else "bmab")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BMAB_")}
    proc = subprocess.Popen([str(exe), "serve", "--port", "8664"], env=env)
    try:
        end = time.time() + 180
        while True:
            try:
                status, page = _ask(8664, "GET", "/")
                break
            except OSError:
                if time.time() > end or proc.poll() is not None:
                    raise SystemExit(f"the download didn't answer (exit {proc.poll()})")
                time.sleep(1)
        said = json.loads(_ask(8664, "GET", "/status")[1])
        print(f"front page {status}, status {said}", flush=True)
        if status != 200 or NAME.encode() not in page or said.get("version") != version:
            raise SystemExit(f"expected the front page and version {version}")
        print("Quit:", _ask(8664, "POST", "/quit", {}), flush=True)
        if proc.wait(timeout=30) != 0:
            raise SystemExit(f"it ended with exit code {proc.returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()
        log = folder / "bmab" / "bmab.log"
        if log.exists():
            print(log.read_text(encoding="utf-8"), flush=True)
    print(f"{archive}: version {version} starts and quits", flush=True)


if __name__ == "__main__":
    what, *rest = sys.argv[1:]
    if what == "next":
        if os.environ["GITHUB_EVENT_NAME"] == "push":
            print(f"version={next_version() or ''}\nrelease=true")
        else:
            branch = re.sub(r"[^A-Za-z0-9]+", "-", os.environ["GITHUB_REF_NAME"])
            print(f"version=0.0.0-{branch}.g{os.environ['GITHUB_SHA'][:7]}\nrelease=false")
    elif what == "notes":
        sys.stdout.write(notes(*rest))
    elif what == "versioninfo":
        sys.stdout.write(versioninfo(*rest))
    elif what == "check":
        check(*rest)
    else:
        raise SystemExit(__doc__)
