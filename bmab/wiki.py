"""Local copies of MediaWiki wikis, searchable offline, for cross-checking (never as evidence on their own).

fetch: every article (namespace 0, no redirects) as wikitext with its revision id, one JSON object per line.
import_dump: the same lines from a MediaWiki XML export (.xml or .xml.zst), for wikis that refuse automated
  clients but publish dumps; only pages whose titles match a pattern, in any namespace.
index: SQLite FTS5 over a plain-text rendering of the wikitext.
search: matching passages with site, title and revision id, so each hit can be cited and re-checked.
"""
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request

USER_AGENT = "bmab/0.1 (offline lore index for cross-checking)"


def _get(api, params):
    url = api + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch(api, out_path, pause=1.0, log=print):
    """Download all articles of the wiki at `api` (its api.php URL) into out_path as JSON lines."""
    params = {"action": "query", "generator": "allpages", "gapnamespace": 0, "gapfilterredir": "nonredirects",
              "gaplimit": 50, "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main"}
    count = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        cont = {}
        while True:
            data = _get(api, {**params, **cont})
            for page in data.get("query", {}).get("pages", []):
                rev = (page.get("revisions") or [{}])[0]
                content = rev.get("slots", {}).get("main", {}).get("content", rev.get("content", ""))
                out.write(json.dumps({"title": page["title"], "pageid": page["pageid"], "revid": rev.get("revid"),
                                      "timestamp": rev.get("timestamp"), "text": content}) + "\n")
                count += 1
            log(f"{count} pages")
            if "continue" not in data:
                return count
            cont = data["continue"]
            time.sleep(pause)


def import_dump(dump_path, out_path, title_pattern, log=print):
    """Write the pages of a MediaWiki XML export whose title matches title_pattern (regex) as JSON lines."""
    import xml.etree.ElementTree as ET
    rx = re.compile(title_pattern)
    if str(dump_path).endswith(".zst"):
        import zstandard
        stream = zstandard.ZstdDecompressor().stream_reader(open(dump_path, "rb"))
    else:
        stream = open(dump_path, "rb")
    count = seen = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for _event, el in ET.iterparse(stream, events=("end",)):
            if el.tag.rsplit("}", 1)[-1] != "page":
                continue
            seen += 1
            ns = el.tag[:-4]  # "{namespace}" prefix of the export schema
            title = el.findtext(f"{ns}title") or ""
            if rx.search(title) and el.find(f"{ns}redirect") is None:
                rev = el.find(f"{ns}revision")
                out.write(json.dumps({"title": title, "pageid": int(el.findtext(f"{ns}id") or 0),
                                      "revid": int(rev.findtext(f"{ns}id") or 0) if rev is not None else None,
                                      "timestamp": rev.findtext(f"{ns}timestamp") if rev is not None else None,
                                      "text": (rev.findtext(f"{ns}text") or "") if rev is not None else ""}) + "\n")
                count += 1
            el.clear()
            if seen % 100000 == 0:
                log(f"{seen} pages read, {count} kept")
    return count


_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")


def plain(wikitext):
    t = re.sub(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", "", wikitext, flags=re.S)
    for _ in range(4):
        t = _TEMPLATE.sub(" ", t)
    t = _LINK.sub(r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}|<[^>]+>", "", t)
    return re.sub(r"[ \t]+", " ", t)


def index(db_path, sources):
    """sources: {site name: jsonl path}. Rebuilds the index."""
    db = sqlite3.connect(db_path)
    db.execute("drop table if exists pages")
    db.execute("create virtual table pages using fts5(site, title, revid unindexed, text, tokenize='porter')")
    n = 0
    for site, path in sources.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                db.execute("insert into pages values (?, ?, ?, ?)", (site, p["title"], p["revid"], plain(p["text"])))
                n += 1
    db.commit()
    return n


def search(db_path, query, limit=10):
    db = sqlite3.connect(db_path)
    rows = db.execute("select site, title, revid, snippet(pages, 3, '[', ']', ' ... ', 40) from pages "
                      "where pages match ? order by rank limit ?", (query, limit)).fetchall()
    return [{"site": s, "title": t, "revid": r, "passage": p} for s, t, r, p in rows]
