"""The facility's place hierarchy, a view of the register (bmab.register: its places and facts) and the place table
(bmab.redlines.place_table, from the place claims): containment from "in" facts, other names from "is" facts, implied places (a
series' missing members) under the parent their siblings share, and the maps that are each place. Each place keeps its type and
the evidence behind every fact that names it or says what holds it, so a view of the hierarchy loses nothing.
"""
import html
from collections import Counter, defaultdict

from .register import MAPPABLE, PLACE_TYPES

TIER_NAMES = {1: "Half-Life", 2: "expansions", 3: "Black Mesa", 4: "mods", 5: "reasoning"}


def build(register, locations):
    """{place: {"parents", "children", "aliases", "tier", "implied" (the places implying it), "maps", "type", "named_in" (evidence ids),
    "said" ({parent: evidence ids saying it holds the place})}}. register: bmab.redlines.register."""
    evidence = register["evidence"]
    tier_of = lambda e: evidence[e]["tier"] if e in evidence else 5  # noqa: E731
    node = defaultdict(lambda: {"parents": set(), "children": set(), "aliases": set(), "tier": None,
                                "implied": None, "maps": set(), "evidence": set(), "type": None, "named_in": set(), "said": defaultdict(set)})
    for p, info in register["places"].items():
        node[p]["type"], node[p]["implied"] = info["type"], info.get("implied_by")
    for f in register["facts"]:
        said = set(f["evidence"])
        for q in (f["subject"], f.get("object")):
            if q:
                node[q]["named_in"] |= said
        if f["relation"] == "in":
            a, b = f["subject"], f["object"]
            node[a]["parents"].add(b)
            node[a]["evidence"] |= said
            node[a]["said"][b] |= said
        elif f["relation"] == "is":
            node[f["subject"]]["aliases"].add(f["object"])
            node[f["object"]]["aliases"].add(f["subject"])
    # alternative names become one place, named after its sector if it has one
    group = {}
    for p in list(node):
        if p in group:
            continue
        todo, members = [p], set()
        while todo:
            q = todo.pop()
            if q not in members:
                members.add(q)
                todo += list(node[q]["aliases"])
        main = min(members, key=lambda q: (not q.startswith("sector-"), len(q), q))
        for q in members:
            group[q] = main
    for q, main in group.items():
        if q != main:
            n, m = node[q], node[main]
            m["parents"] |= n["parents"]
            m["maps"] |= n["maps"]
            m["evidence"] |= n["evidence"]
            m["named_in"] |= n["named_in"]
            m["type"] = m["type"] or n["type"]
            m["implied"] = m["implied"] or n["implied"]
            for k, ids in n["said"].items():
                m["said"][k] |= ids
            del node[q]
    for n in node.values():
        n["parents"] = {group.get(q, q) for q in n["parents"]}
        said = defaultdict(set)
        for k, ids in n["said"].items():
            said[group.get(k, k)] |= ids
        n["said"] = dict(said)
        n["tier"] = 5 if n["implied"] else min((tier_of(e) for e in n["named_in"]), default=5)
    for q, main in group.items():
        node[main]["parents"].discard(main)
        if q != main:
            node[main]["aliases"].add(q)
        node[main]["aliases"].discard(main)
    for p, info in register["places"].items():  # implied places sit where their siblings do
        if "implied_by" in info and p in node and not node[p]["parents"]:
            sib = [node[group.get(q, q)]["parents"] for q in info["implied_by"]]
            node[p]["parents"] |= set.intersection(*sib) if sib else set()
    for e in (locations or {}).get("here", []):
        node[group.get(e["place"], e["place"])]["maps"].update(f"{e['game']}:{m}" for m in e["maps"])
    for p, n in list(node.items()):
        for q in n["parents"]:
            node[q]["children"].add(p)
    return dict(node)


def names(tree):
    """{name: the place of the tree it names} for every place and every other name of one."""
    return {**{a: p for p, n in tree.items() for a in n["aliases"]}, **{p: p for p in tree}}


def ancestors(tree, place):
    """Every place holding a place of the tree, however far up."""
    seen, todo = set(), [place]
    while todo:
        for q in tree[todo.pop()]["parents"]:
            if q not in seen:
                seen.add(q)
                todo.append(q)
    return seen


def named_in_map(register, tree, game, name):
    """The places a map's own evidence names (signs seen and lines heard in it), of the kinds a map can be in (MAPPABLE), the most
    named first: the guess at where the map is. register: bmab.redlines.register; tree: build."""
    here = {e["id"] for e in register["evidence"].values() if e["source"] == game and name in e.get("seen_in", []) + e.get("heard_in", [])}
    called, named = names(tree), Counter()
    for f in register["facts"]:
        if here.intersection(f["evidence"]):
            for p in dict.fromkeys(called[q] for q in (f["subject"], f.get("object")) if q in called):
                if tree[p]["type"] in MAPPABLE:
                    named[p] += 1
    return sorted(named, key=lambda p: -named[p])


def render(tree):
    """Indented text: each top place (one with no parent) and everything under it, then counts."""
    lines, seen = [], set()

    def label(p):
        n = tree[p]
        bits = [p]
        if n["aliases"]:
            bits.append("= " + ", ".join(sorted(n["aliases"])))
        if n["implied"]:
            bits.append(f"[implied by {', '.join(n['implied'])}]")
        elif n["tier"]:
            bits.append(f"[tier {n['tier']}]")
        if n["maps"]:
            bits.append("maps: " + " ".join(sorted(n["maps"])))
        return "  ".join(bits)

    def walk(p, depth):
        if p in seen and depth:
            lines.append("    " * depth + p + "  (see above)")
            return
        seen.add(p)
        lines.append("    " * depth + label(p))
        for c in sorted(tree[p]["children"]):
            walk(c, depth + 1)

    tops = [p for p, n in tree.items() if not n["parents"] and n["children"]]
    for p in sorted(tops, key=lambda p: (-len(tree[p]["children"]), p)):
        walk(p, 0)
        lines.append("")
    loose = sorted(p for p, n in tree.items() if not n["parents"] and not n["children"])
    lines.append(f"Places not yet placed in the hierarchy ({len(loose)}): " + ", ".join(loose))
    return "\n".join(lines) + "\n"


TIER_COLOURS = {1: "#ff9d3c", 2: "#ffd23c", 3: "#5acbff", 4: "#b48cff", 5: "#9a9aa8"}
TYPE_NAMES = {"region": "region", "facility": "facility", "sector": "sector", "area": "area", "complex": "complex", "level": "level",
              "room": "room", "transit": "transit", "thing": "thing", "elsewhere": "elsewhere"}
UNPLACED = (("sector", "Sectors"), ("area", "Areas"), ("complex", "Complexes"), ("level", "Levels"), ("room", "Rooms"), ("transit", "Transit"),
            ("thing", "Things"))


def odd(tree):
    """Nestings against the types: [(place, what holds it)] where the place is of an earlier rank (PLACE_TYPES) than its holder, or a
    complex in a room, or anything in a thing."""
    out = []
    for p, n in tree.items():
        for q in n["parents"]:
            a, b = n["type"], tree.get(q, {}).get("type")
            if a and b and (PLACE_TYPES[a] < PLACE_TYPES[b] or b == "thing" or (a, b) == ("complex", "room")):
                out.append((p, q))
    return sorted(out)


def page(tree, register, home=None):
    """The hierarchy as a page of its own, a view of the facts register and the maps' places (nothing is stored from it): each
    region and what it holds, as the facts say; where they are silent, by type (a sector in the facility, the rest in the facility
    under "where not known yet", things beside their place, elsewhere apart). Each place shows its type, what else it is called,
    which games first name it (or why it is implied), the maps in it, and on hover every fact that names it or says what holds it;
    then the nestings against the types. register: bmab.redlines.register, to name each piece of evidence; home: a link to the app's
    front page, when the app serves it."""
    esc = html.escape
    words = lambda p: p.replace("-", " ")  # noqa: E731
    by_id = register["evidence"]
    source = lambda i: f'{by_id[i]["source"]} {by_id[i]["kind"]}: {by_id[i].get("ref", by_id[i].get("url"))}' if i in by_id else i  # noqa: E731
    kind = lambda p: tree[p]["type"]  # noqa: E731
    seen, more = set(), {}  # more: list items a place's branch gets besides its own (the facility: what it holds by type)

    def sources(p):
        n = tree[p]
        lines = [f"named by {len(n['named_in'])}:"] + [f"  {source(i)}" for i in sorted(n["named_in"])]
        for q, ids in sorted(n["said"].items()):
            lines += [f"in {words(q)}, said by:"] + [f"  {source(i)}" for i in sorted(ids)]
        return esc("\n".join(lines))

    def chips(p):
        n, out = tree[p], []
        out.append(f'<span class="type t-{esc(n["type"] or "none")}">{esc(TYPE_NAMES.get(n["type"], "no type"))}</span>')
        if n["implied"]:
            out.append(f'<span class="chip implied">implied</span> <span class="why">by {esc(", ".join(words(q) for q in n["implied"]))}</span>')
        elif n["tier"]:
            out.append(f'<span class="chip" style="background:{TIER_COLOURS.get(n["tier"], "#777")}">{esc(TIER_NAMES.get(n["tier"], str(n["tier"])))}</span>')
        if n["aliases"]:
            out.append('<span class="also">also ' + ", ".join(esc(words(a)) for a in sorted(n["aliases"])) + "</span>")
        if n["maps"]:
            out.append('<span class="maps">' + " ".join(f'<span class="map">{esc(m.split(":")[1])}</span>' for m in sorted(n["maps"])) + "</span>")
        if n["named_in"] or n["said"]:
            k = len(n["named_in"].union(*n["said"].values()))
            out.append(f'<span class="src" title="{sources(p)}">{k} source{"" if k == 1 else "s"}</span>')
        return " ".join(out)

    def head(p):
        return f'<span class="name" data-name="{esc(p)} {esc(" ".join(tree[p]["aliases"]))}">{esc(words(p))}</span> {chips(p)}'

    def node(p, under=None):
        n = tree[p]
        if p in seen:  # a place in two others: drawn once, pointed at from the other
            return f'<li><span class="name" data-name="{esc(p)}">{esc(words(p))}</span> <span class="why">(also in {esc(words(under))}; drawn above)</span></li>'
        seen.add(p)
        kids = sorted((c for c in n["children"] if kind(c) != "thing"), key=lambda c: (PLACE_TYPES.get(kind(c), 9), not tree[c]["children"], c))
        things = sorted(c for c in n["children"] if kind(c) == "thing")
        seen.update(things)
        line = (f'<div class="things">things: {", ".join(f"<span class=name data-name={esc(t)}>{esc(words(t))}</span> {chips(t)}" for t in things)}</div>'
                if things else "")
        extra = more.get(p, "")
        if not kids and not line and not extra:
            return f"<li>{head(p)}</li>"
        return f'<li><details open><summary>{head(p)} <span class="n">{len(kids)}</span></summary>{line}<ul>{"".join(node(c, p) for c in kids)}{extra}</ul></details></li>'

    def group(title, members, note=""):
        if not members:
            return ""
        return (f'<li><details open><summary><span class="grp">{esc(title)}</span> <span class="why">{esc(note)}</span> <span class="n">{len(members)}</span>'
                f'</summary><ul>{"".join(node(m) for m in members)}</ul></details></li>')

    loose = lambda t: sorted(p for p, n in tree.items() if not n["parents"] and n["type"] == t)  # noqa: E731
    facility = next((p for p, n in tree.items() if n["type"] == "facility"), None)
    if facility:  # what the facts don't place anywhere is in the facility, by type: sectors as its own, the rest where not known yet
        more[facility] = "".join(node(p) for p in loose("sector")) + "".join(group(f"{title} where not known yet", loose(t)) for t, title in UNPLACED if t != "sector")
    body = "".join(node(r) for r in loose("region"))  # the facility sits in its region by the facts
    if facility and facility not in seen:
        body += node(facility)
    body += group("Elsewhere", loose("elsewhere"), "not the facility")
    body += group("Not typed yet", sorted(p for p, n in tree.items() if not n["parents"] and not n["type"]))
    oddities = odd(tree)
    counts = {t: sum(1 for n in tree.values() if n["type"] == t) for t in PLACE_TYPES}
    legend = " ".join(f'<span class="chip" style="background:{c}">{esc(TIER_NAMES[t])}</span>' for t, c in TIER_COLOURS.items() if t != 5)
    back = f'<p class="small"><a href="{esc(home)}" style="color:#8cc8ff">Front page</a></p>' if home else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Places of the facility</title><style>
body {{ background:#0e0e14; color:#ddd; font:14px/1.5 system-ui, sans-serif; margin:20px 28px; }}
h1 {{ font-size:18px; margin:0 0 4px; }} h2 {{ font-size:15px; margin:22px 0 6px; }}
.small, .why {{ color:#9a9aa8; font-size:12px; }}
ul {{ list-style:none; margin:0; padding-left:20px; border-left:1px solid #2a2a36; }} #tree {{ padding-left:0; border:0; }}
li {{ margin:2px 0; }} summary {{ cursor:pointer; }} .name {{ font-weight:600; }} .grp {{ font-style:italic; color:#bbb; }}
.chip {{ display:inline-block; font-size:11px; color:#111; border-radius:8px; padding:0 7px; margin-left:4px; }}
.chip.implied {{ background:#9a9aa8; }} .also {{ color:#b8c8a0; font-size:12px; margin-left:6px; }}
.type {{ display:inline-block; font-size:10px; border:1px solid #556; color:#aab; border-radius:3px; padding:0 4px; margin-left:4px; text-transform:uppercase; letter-spacing:.04em; }}
.t-sector {{ border-color:#c9a; color:#e7c; }} .t-area {{ border-color:#9c9; color:#9e9; }} .t-complex {{ border-color:#99c; color:#aaf; }}
.t-level {{ border-color:#7aa; color:#8cc; }} .t-thing {{ border-color:#666; color:#888; }} .t-transit {{ border-color:#c96; color:#fb8; }}
.map {{ display:inline-block; font-size:11px; background:#263040; color:#a8d0ff; border-radius:4px; padding:0 5px; margin-left:4px; }}
.src {{ font-size:11px; color:#8cc8ff; margin-left:6px; cursor:help; border-bottom:1px dotted #8cc8ff; }}
.things {{ color:#9a9aa8; font-size:12px; margin:2px 0 2px 20px; }} .things .name {{ font-weight:400; color:#bbb; }}
.n {{ color:#777; font-size:11px; margin-left:4px; }} input {{ background:#1b1b24; color:#ddd; border:1px solid #333; padding:4px 8px; width:260px; }}
.hide {{ display:none; }} .hit > .name, .hit > summary > .name {{ background:#5a4a10; }}
</style></head><body>
{back}
<h1>Places of the facility</h1>
<div class="small">A view of the register (what the games' signs, announcements and the like say about places, each piece of evidence kept) and of the
maps' places. {len(tree)} places: {", ".join(f"{v} {k}" + ("" if v == 1 or k in ("transit", "elsewhere") else "es" if k.endswith("x") else "s") for k, v in counts.items() if v)}. Nested as the facts say;
where they say nothing, by type: sectors in the facility, other places in it under "where not known yet", things listed beside the place
that holds them. Hover over "sources" for every piece of evidence behind a place and what holds it.</div>
<div class="small">First named by: {legend} <span class="chip implied">implied</span> (a series' missing members)</div>
<p><input id="find" placeholder="Find a place or a map"> <button id="fold">Fold all</button> <button id="unfold">Unfold all</button></p>
<ul id="tree">{body}</ul>
<h2>Nestings against the types ({len(oddities)})</h2>
<div class="small">A place said to be in something it should hold, by their types: one of the two types, or the fact, needs a look.</div>
<ul>{"".join(f"<li>{head(p)} <span class=why>in</span> {head(q)}</li>" for p, q in oddities) or "<li class=small>none</li>"}</ul>
<script>
const items = [...document.querySelectorAll("li")];
document.getElementById("find").oninput = e => {{
  const q = e.target.value.trim().toLowerCase().replace(/ /g, "-");
  for (const li of items) li.classList.remove("hide", "hit");
  if (!q) return;
  const own = li => [...li.querySelectorAll(":scope > .name, :scope > details > summary > .name, :scope .map")].some(x => (x.dataset.name || x.textContent).toLowerCase().replace(/ /g, "-").includes(q));
  for (const li of items) {{
    const mine = own(li), below = [...li.querySelectorAll("li")].some(own);
    if (!mine && !below) li.classList.add("hide");
    if (mine) li.classList.add("hit");
    const d = li.querySelector(":scope > details"); if (d && below) d.open = true;
  }}
}};
document.getElementById("fold").onclick = () => document.querySelectorAll("details").forEach(d => d.open = false);
document.getElementById("unfold").onclick = () => document.querySelectorAll("details").forEach(d => d.open = true);
</script></body></html>
"""
