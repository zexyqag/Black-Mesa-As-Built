"""bmab's local app over HTTP: serves the front page (front.html: the redlines and signing in, the games to copy in from Steam, the
pages), the review pages of the games here (bmab.pages, each edition's at /<edition>/) and each map's view as it is asked for, and
routes what the pages ask to bmab.editing (changing claims with the games at hand), bmab.sources (copying games in, as bmab.jobs'
one job), bmab.redlines (status, commit, discard, push) and bmab.github (signing in). It answers changes only from its own pages,
and ends when the front page's Quit asks it to.
"""
import http.server
import json
import os
import socket
import sys
import urllib.parse
from pathlib import Path

from . import claims, editing, generate, github, jobs, media, pages, redlines, register, sources, steam, view

MAX_BODY = 256 * 1024
_copying = []  # a copy of games in started since the pages were last set up: they are set up again when it ends


def setup():
    """The pages to serve (bmab.pages), and bmab.editing set up with the levels next to each of their maps and every map they hold."""
    editing.setup(*pages.setup())
    return pages.editions()


def status():
    """For the page: the redlines' status (bmab.redlines.status), who is signed in with GitHub, if signing in is on, and the
    version of bmab running (a download's; None from the source)."""
    s, account = redlines.status(), github.account()
    return {"git": s["git"], "redlines": s["here"], "changed": [{"game": g, "map": m} for g, m in s["changed"]], "register": s["register"], "ahead": s["ahead"], "pull_request": s["pull_request"],
            "github": bool(github.client_id()), "login": account["login"] if account else None, "version": os.environ.get("BMAB_VERSION")}


def _signing_in(device_code):
    """Ask GitHub whether the code has been entered; once it has, keep the account and clone the redlines if they aren't here yet."""
    got = github.poll(device_code)
    if "access_token" not in got:
        return {"state": got.get("error", "failed"), "interval": got.get("interval")}
    address = os.environ.get("BMAB_REDLINES_REPO", "") if redlines.waiting() else redlines.origin()
    if repo := github.repository(address):
        try:
            github.access(repo, got["access_token"])
        except ValueError:
            return {"state": "refused", "message": f"this GitHub account can't see {repo}"}
    account = github.remember(got["access_token"])
    with redlines.lock:
        redlines.clone(account["token"])
    setup()  # the register now here says which games the pages show
    return {"state": "done", "login": account["login"]}


def _register_data():
    """The register for its page, and the words its fields take."""
    reg = redlines.register()
    return {"sources": list(reg["sources"].values()), "places": list(reg["places"].values()), "evidence": list(reg["evidence"].values()), "facts": reg["facts"],
            "events": reg["events"],
            "words": {"relations": register.RELATIONS, "valued": register.VALUED, "alone": register.ALONE, "evidence_kinds": register.EVIDENCE_KINDS,
                      "page_kinds": register.PAGE_KINDS, "how": register.HOW, "place_types": list(register.PLACE_TYPES), "editions": register.EDITIONS,
                      "verdicts": register.VERDICTS, "ranked": register.RANKED, "reasons": register.REASONS, "source_kinds": register.SOURCE_KINDS,
                      "generators": {k: g["title"] for k, g in generate.GENERATORS.items()}, "games_here": [g for g in generate.games_here() if g in reg["sources"]]}}


def games():
    """For the front page: the Steam installs found, and every source bmab can copy in, where it stands here (bmab.sources.here) and
    whether it is suggested (found, and a game the register counts), in the register's order: [{id, title, state, detail, suggested}]."""
    reg = redlines.register()["sources"]
    order = lambda sid: (0, register.EDITIONS.index(reg[sid].get("edition", register.EDITIONS[0])), reg[sid].get("tier", 9), reg[sid]["title"]) if sid in reg else (1, 0, 0, sid)  # noqa: E731
    return {"steam": [str(r) for r in steam.roots()], "job": copying(),
            "games": [{"id": sid, "title": reg[sid]["title"] if sid in reg else sid, "state": state, "detail": detail,
                       "suggested": state == "found" and reg.get(sid, {}).get("verdict") not in pages.LEFT_OUT}
                      for sid, (state, detail) in sorted(sources.here().items(), key=lambda kv: order(kv[0]))]}


def copy(ids):
    """Start copying the sources ids in, as the app's one job (bmab.jobs), each after the game it builds on (its base), which comes
    along if it isn't here yet."""
    known, here = sources.load()[0], sources.here()
    if not ids or any(i not in known for i in ids):
        raise ValueError("no such source")
    todo = []

    def add(sid):
        base = known[sid].get("base")
        if base and here[base][0] != "copied":
            add(base)
        if sid not in todo:
            todo.append(sid)
    for sid in ids:
        add(sid)
    jobs.start(("copy",), "copying games in", sources.copy_in, todo)
    _copying.append(True)
    return copying()


def copying():
    """How copying games in goes (bmab.jobs.state), with how far it has got while it runs and what became of each source once done.
    Once it has ended, the pages are set up again with the games now here."""
    state, results = jobs.state(("copy",))
    if state["state"] == "running" and sources.PROGRESS.exists():
        try:
            state["progress"] = json.loads(sources.PROGRESS.read_text(encoding="utf-8"))
        except ValueError:  # being written
            pass
    if results is not None:
        state["results"] = results
    if state["state"] in ("done", "failed"):
        try:
            _copying.pop()
        except IndexError:
            pass
        else:
            setup()
    return state


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # a map's view keeps its address when it is built again: ask before using a kept copy
        super().end_headers()

    def _send(self, code, data, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _reply(self, code, body):
        if code == 200:
            self._send(code, json.dumps(body).encode(), "application/json")
        else:
            self._send(code, str(body).encode(), "text/plain")

    def _ours(self):
        """Only the app's own pages may change anything: any site open in the browser can reach this computer's ports, or name
        itself as this one."""
        host, origin = self.headers.get("Host", ""), self.headers.get("Origin")
        return host.split(":")[0] in ("localhost", "127.0.0.1") and origin in (None, f"http://{host}") and self.headers.get("Content-Type") == "application/json"

    def _route(self):
        """The page a request is about (the first part of its path names its edition) and the rest of its path; None for the app's own."""
        p = [x for x in self.path.split("?")[0].split("/") if x]
        return (p[0], p[1:]) if p and p[0] in pages.editions() else (None, p)

    def _act(self, fn):
        if not self._ours():
            return self._reply(403, "not from this app's pages")
        try:
            self._reply(200, fn())
        except ValueError as e:
            self._reply(400, e)

    def _get(self, fn):
        try:
            self._reply(200, fn())
        except ValueError as e:
            self._reply(400, e)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("too long")
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        page, p = self._route()
        game, name = (p[1], p[2]) if len(p) > 2 and all(claims.MAP_NAME.fullmatch(x) for x in p[1:3]) else (None, None)
        if p == ["status"]:
            return self._reply(200, status())
        if self.path.split("?")[0].strip("/") in pages.editions() and not self.path.split("?")[0].endswith("/"):  # /page: what it asks for is relative to /page/
            self.send_response(301)
            self.send_header("Location", self.path.split("?")[0] + "/")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if len(p) == 3 and p[0] == "claims" and name:
            return self._reply(200, redlines.claims(game, name))
        if len(p) == 3 and p[0] == "changes" and name:
            return self._get(lambda: editing.level_changes(game, name))
        if len(p) == 3 and p[0] == "evaluate" and name:
            return self._reply(200, editing.evaluated(game, name))
        if len(p) == 4 and p[0] == "part" and name:
            return self._get(lambda: editing.alone(game, name, p[3]))
        if len(p) == 3 and p[0] == "opening" and name:
            return self._get(lambda: editing.opening(game, name))
        if len(p) == 3 and p[0] == "signs" and name:
            return self._get(lambda: editing.map_signs(game, name))
        if len(p) == 3 and p[0] == "places" and name:
            return self._get(lambda: editing.known_places(game, name))
        if len(p) == 3 and p[0] == "version" and name:
            return self._get(lambda: editing.version(game, name))
        if p in (["register"], ["register", "index.html"]) and not page:
            return self._send(200, (Path(__file__).parent / "register.html").read_bytes(), "text/html; charset=utf-8")
        if p == ["register", "data"] and not page:
            return self._reply(200, _register_data())
        if p == ["register", "workout"] and not page:
            return self._get(editing.worked_out)
        if p == ["register", "media"] and not page:
            q = urllib.parse.parse_qs(self.path.partition("?")[2])
            e = redlines.register()["evidence"].get(q.get("id", [""])[0])
            if e is None:
                return self._reply(404, "no such evidence")
            try:
                kind, data, same = media.media(e)
            except ValueError as x:
                return self._reply(404, x)
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("X-Same-Version", "yes" if same else "no")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if p == ["placetree"] and not page:
            from . import places
            reg = redlines.register()
            return self._send(200, places.page(places.build(reg, redlines.place_table()), reg, home="./").encode(), "text/html; charset=utf-8")
        if p == ["elsewhere"] and page:
            return self._reply(200, pages.elsewhere(page))
        if len(p) == 3 and p[0] == "maps" and page and claims.MAP_NAME.fullmatch(p[1]) and p[2].endswith(".js"):  # a map's view, or its models
            name, what = p[2].split(".", 1)
            if not claims.MAP_NAME.fullmatch(name) or what not in ("js", "models.js") or not pages.listed(p[1], name):
                return self._reply(404, "no such map")
            try:
                found = pages.view_file(p[1], name, what)
            except ValueError as e:
                return self._reply(404, e)
            return self._send(200, found.read_bytes(), "text/javascript; charset=utf-8") if found else self._reply(404, "it places no models")
        if len(p) == 3 and p[0] == "neighbours" and page and name and pages.listed(game, name, page):
            return self._get(lambda: pages.neighbours(page, game, name))
        if p == ["parts"] and page:
            return self._reply(200, [{"game": m["source"], "map": m["name"], **r} for m in pages.maps(page)
                                     for r in redlines.claims(m["source"], m["name"]) if r["kind"] == "part"])
        if p in ([], ["index.html"]) and page:
            return self._send(200, view._page("inspector.html", pages.meta(page)).encode(), "text/html; charset=utf-8")
        if p in ([], ["index.html"]) and not page:
            return self._send(200, view._page("front.html", None).encode(), "text/html; charset=utf-8")
        if p == ["favicon.ico"] and not page:  # the pages' tab icon (bmab.exe's too), asked for by the browser
            return self._send(200, (Path(__file__).parent / "icon.ico").read_bytes(), "image/x-icon")
        if p == ["games"] and not page:
            return self._get(games)
        if p == ["games", "copy"] and not page:
            return self._get(copying)
        if p == ["pages"] and not page:
            return self._reply(200, [{"edition": e, "maps": len(pages.maps(e)), "games": list(dict.fromkeys(m["game"] for m in pages.maps(e)))} for e in pages.editions()])
        self._reply(404, "no such page")

    def _changing(self, p):
        """Whether a change can go on now: not before the redlines are here (signing in clones them)."""
        if redlines.waiting() and p[:1] not in (["login"], ["logout"]):
            self._reply(409, "the redlines aren't here yet: sign in with GitHub to clone them")
            return False
        return True

    def do_PUT(self):
        _page, p = self._route()
        if not self._changing(p):
            return
        if len(p) == 3 and p[0] == "claims" and all(claims.MAP_NAME.fullmatch(x) for x in p[1:]):
            return self._act(lambda: editing.save(p[1], p[2], self._body()))
        if len(p) == 2 and p[0] == "register" and not _page:
            return self._act(lambda: redlines.save_entry(p[1], self._body()))
        self._reply(404, "no such thing")

    def do_DELETE(self):
        _page, p = self._route()
        if not self._changing(p):
            return
        if len(p) == 4 and p[0] == "claims" and all(claims.MAP_NAME.fullmatch(x) for x in p[1:3]):
            return self._act(lambda: editing.delete(p[1], p[2], p[3]))
        if len(p) == 3 and p[0] == "evaluate" and all(claims.MAP_NAME.fullmatch(x) for x in p[1:]):
            return self._act(lambda: editing.forget(p[1], p[2]))
        if p == ["register", "workout"] and not _page:
            return self._act(editing.forget_work)
        self._reply(404, "no such thing")

    def do_POST(self):
        _page, p = self._route()
        if p == ["quit"] and not _page:
            return self._quit()
        if not self._changing(p):
            return
        game, name = (p[1], p[2]) if len(p) > 2 and all(claims.MAP_NAME.fullmatch(x) for x in p[1:3]) else (None, None)
        if p == ["commit"]:
            return self._act(lambda: (redlines.commit(github.account()), status())[1])
        if p == ["push"]:
            return self._act(lambda: (redlines.push(github.account()), status())[1])
        if p == ["login", "start"] and github.client_id():
            return self._act(github.start)
        if p == ["login", "poll"] and github.client_id():
            return self._act(lambda: _signing_in(str(self._body().get("device_code", ""))))
        if p == ["logout"]:
            return self._act(lambda: (github.forget(), status())[1])
        if p == ["register", "delete"] and not _page:
            return self._act(lambda: (lambda b: redlines.delete_entry(b.get("table"), b.get("id")))(self._body()))
        if p == ["register", "discard"] and not _page:
            return self._act(lambda: (redlines.discard_register(), status())[1])
        if p == ["register", "workout"] and not _page:
            return self._act(lambda: (lambda b: editing.work_out(str(b.get("what", "")), str(b.get("game", ""))))(self._body()))
        if p == ["games", "copy"] and not _page:
            return self._act(lambda: copy(self._body().get("ids") or []))
        if p == ["register", "apply"] and not _page:
            return self._act(lambda: (lambda b: redlines.save_entries(b.get("table"), b.get("entries") or []))(self._body()))
        if len(p) == 3 and p[0] == "cut" and name:
            return self._act(lambda: editing.cut(game, name))
        if len(p) == 3 and p[0] == "evaluate" and name:
            return self._act(lambda: editing.again(game, name))
        if len(p) == 3 and p[0] == "version" and name:
            return self._act(lambda: editing.checked(game, name))
        if len(p) == 3 and p[0] == "discard" and name:
            return self._act(lambda: editing.discard(game, name))
        if len(p) == 4 and p[0] == "openings" and name:
            return self._act(lambda: editing.find_openings(game, name, p[3]))
        self._reply(404, "no such thing")

    def _quit(self):
        """The front page's Quit: the app stops answering (serving it ends: bmab.app then ends), asking the page first while a job
        runs (409, what the job is doing), as closing the app's window does."""
        if not self._ours():
            return self._reply(403, "not from this app's pages")
        anyway = self._body().get("anyway")  # read before the reply: what is left unread when the process ends resets the connection
        if (busy := jobs.running()) and not anyway:
            return self._reply(409, busy)
        self._reply(200, {})
        self.server.shutdown()


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = sys.platform != "win32"  # on Windows that lets a second program listen on the same port

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def server(port):
    """The app listening on port (this computer only), set up but not answering yet: the redlines cloned if they can be, the pages set up. OSError
    when the port is taken."""
    listening = Server(("127.0.0.1", port), Handler)
    if not redlines.git():
        print(redlines.NO_GIT, flush=True)
    elif not redlines.clone():
        print(f"the redlines aren't cloned into {redlines.REDLINES} yet: signing in with GitHub on the front page clones them", flush=True)
    names = setup()
    print(f"http://localhost:{port}: {', '.join(names) or 'no games here yet'} (saving to {redlines.REDLINES})", flush=True)
    return listening


def run(port):
    server(port).serve_forever()
    end()


def end():
    """bmab ends at once, and the processes it started with it (bmab.jobs.with_the_app): a job stops, as quitting said it would,
    and a view being built isn't waited for; an edit or a commit under way is (it holds bmab.redlines.lock)."""
    redlines.lock.acquire()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
