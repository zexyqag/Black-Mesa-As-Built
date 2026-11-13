"""usage:
  python -m bmab app [--port PORT]
  python -m bmab import [SOURCE ...] [--pin]
  python -m bmab plan SOURCE OUT [--exclude REGEX] [--start MAP]
  python -m bmab mine SOURCE OUT [--top N]
  python -m bmab measure SOURCE OUT [--exclude REGEX] [--top N]
  python -m bmab labels SOURCE OUT [--exclude REGEX]
  python -m bmab wiki fetch | index | search QUERY...
  python -m bmab solve SOURCE OUT [--profile strict|lenient] [--order ORDER | --compare N] [--exclude REGEX]
  python -m bmab layout SOURCE OUT [--exclude REGEX] [--start MAP]
  python -m bmab attach SOURCE BASE.json OUT --add SOURCE... [--order ORDER | --compare N | --seeds N] [--from REPORT]
  python -m bmab routed SOURCE BASE.json REPORT OUT.html --add SOURCE...
  python -m bmab recipe OUT [--quick]
  python -m bmab evaluate REPORT --sources SOURCE... [--maps REGEX] [--redo]
  python -m bmab shared SOURCE OUT [--with SOURCE] [--maps REGEX] [--with-maps REGEX] [--min-m2 M2]
  python -m bmab match IMAGE SOURCE OUT [--maps REGEX] [--graph JSON [--group]]
  python -m bmab transcribe SOURCE OUT [--include GLOB ...] [--model NAME]
  python -m bmab textures SOURCE OUT [--names A,B | --unused] [--known JSON ...] [--exclude REGEX] [--skip REGEX] [--cell PX --cols N --rows N]

import copies the material listed in sources.json into data/ and checks it against
sources.lock.json; plan writes graph.json, plan.png and overlaps.json for an imported
GoldSrc source.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import redlines, bsp, graph, measure, pieces, plan, shared, sources, text, vmf, vtf, wad
from .game import Game


def cmd_import(args):
    results = sources.run(args.sources or None, pin=args.pin)
    print()
    for r in results:
        print(f"{r.status:8} {r.id}: {r.detail}")
    return 1 if any(r.status in ("missing", "differs") for r in results) else 0


def cmd_plan(args):
    folder = sources.DATA / args.source
    if not (folder / "maps").is_dir():
        sys.exit(f"{folder} has no maps; run: python -m bmab import {args.source}")
    game = Game(folder)
    skip = re.compile(args.exclude) if args.exclude else None
    maps = {n: _read(args.source, game, n) for n in game.map_names if not (skip and skip.search(n))}
    decided = redlines.decided_links(args.source)
    banned = {graph.key(d) for d in decided if d["gives"]}
    g = graph.build(maps, args.start or game.start, decided)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(g, indent=1), encoding="utf-8", newline="\n")

    comps = {}
    for n, m in g["maps"].items():
        comps.setdefault(m["component"], []).append(n)
    print(f"{len(maps)} maps, {len(g['links'])} links, {len(g['unlinked'])} unlinked")
    for root, ns in comps.items():
        print(f"  component from {root}: {len(ns)} maps")
    for a, b in g["dangling"]:
        print(f"  dangling: {a} > {b} (map not present)")
    for a, b, mark in g["dead"]:
        print(f"  dead: {a} > {b} via {mark!r} (landmark missing)")
    kinds = {}
    for link in g["links"]:
        kinds[link["kind"]] = kinds.get(link["kind"], 0) + 1
    print("  link kinds: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    for gap in g["gaps"]:
        state = "decided" if (*gap["maps"], gap["landmark"]) in banned else "open"
        print(f"  gap {gap['metres']} m on {gap['kind']} link {gap['maps'][0]} / {gap['maps'][1]} via {gap['landmark']}"
              f" {gap['error']} ({state})")

    if not comps:
        print("no map links another: nothing to place")
        return 0
    root = g["start"] if g["start"] in g["maps"] else max(comps, key=lambda r: len(comps[r]))
    polys = plan.placed_floors(maps, g, root)
    size = plan.render(polys, out / "plan.png")
    print(f"plan of {len(polys)} maps from {root}: {size['width_m']} x {size['depth_m']} m, {size['height_m']} m high")
    ov = plan.overlaps(polys, {tuple(x["maps"]) for x in g["links"]})
    (out / "overlaps.json").write_text(json.dumps(ov, indent=1), encoding="utf-8", newline="\n")
    unlinked = [r for r in ov if not r["linked"]]
    print(f"overlapping pairs: {len(ov)}, {len(unlinked)} not directly linked; largest:")
    for r in unlinked[:12]:
        print(f"  {r['maps'][0]} x {r['maps'][1]}: {r['m2']} m2, same height {r['same_height']:.0%}")
    return 0


def cmd_mine(args):
    folder = sources.DATA / args.source
    game = Game(folder)
    maps = {n: game.read(n) for n in game.map_names}
    sent_file, title_file = folder / "sound" / "sentences.txt", folder / "titles.txt"
    sents, comments = text.sentences(sent_file.read_text("latin-1")) if sent_file.exists() else ({}, [])
    tits = text.titles(title_file.read_text("latin-1")) if title_file.exists() else {}
    said, shown, messages, names, groups = text.references(maps)
    for n in sents:
        for group, ms in groups.items():
            if n.startswith(group):
                said[n] |= ms
    items = [(f"sentence {n} ({', '.join(sorted(said.get(n, []))) or 'no map'})", s["text"]) for n, s in sents.items()]
    items += [("comment in sentences.txt", c) for c in comments]
    items += [(f"title {n} ({', '.join(sorted(shown.get(n, []))) or 'no map'})", t) for n, t in tits.items()]
    items += [(f"game_text in {m['map']}", m["text"]) for m in messages]
    # Entity names are level-design labels (door_1, room_2): kept in names, not searched for places.
    found = text.places(items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "text.json").write_text(json.dumps({
        "sentences": {n: {**s, "maps": sorted(said.get(n, []))} for n, s in sents.items()},
        "comments": comments,
        "titles": {n: {"text": t, "maps": sorted(shown.get(n, []))} for n, t in tits.items()},
        "messages": messages,
        "names": {n: sorted(ms) for n, ms in sorted(names.items())},
        "places": {k: sorted(v) for k, v in sorted(found.items())},
    }, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(sents)} sentences ({sum(1 for n in sents if n in said)} used by a map), {len(comments)} comments, "
          f"{len(tits)} titles, {len(messages)} map messages, {len(found)} place phrases")
    for k in sorted(found, key=lambda k: (-len(found[k]), k))[:args.top]:
        print(f"  {k}: {len(found[k])}")
    return 0


def _source_material(folder, name):
    """Source material name -> (image, file)."""
    tex, _see_through = vtf.material([folder], name)
    return (vtf.image(tex.read_bytes()), tex.relative_to(sources.DATA).as_posix()) if tex else (None, None)


def cmd_textures(args):
    manifest, _ = sources.load()
    folder = sources.DATA / args.source
    skip_maps = re.compile(args.exclude) if args.exclude else None
    skip_tex = re.compile(args.skip) if args.skip else None
    pool, own, origin = {}, set(), {}
    for sid in [manifest[args.source].get("base"), args.source]:
        for w in sorted((sources.DATA / sid).glob("*.wad")) if sid else []:
            textures = wad.wad(w)
            pool.update(textures)
            origin.update({n: f"{sid}/{w.name}" for n in textures})
            if sid == args.source:
                own |= set(textures)
    source_engine = (folder / "materials").is_dir()
    used, walls = {}, {}

    def count(n, mapname, wall):
        if source_engine:
            m = vtf.PATCHED.match(n)
            n = m.group(1) if m else n
        used.setdefault(n, set()).add(mapname)
        if wall:
            walls[n] = walls.get(n, 0) + 1

    for m in sorted([*(folder / "maps").glob("*.bsp"), *(folder / "maps").glob("*.bs2")]):
        if skip_maps and skip_maps.search(m.stem):
            continue
        data = m.read_bytes()
        if not source_engine and m.suffix == ".bsp":
            _, images = wad.embedded(data)
            pool.update(images)
            origin.update({n: f"{args.source}/maps/{m.name}" for n in images})
        b = bsp.read(data, m.stem)
        for e in b.entities:
            if e.get("classname") == "infodecal" and e.get("texture"):
                count(e["texture"].lower(), m.stem, True)
        for material, _origin, _normal in b.overlays:
            count(material, m.stem, True)
        for f in b.faces:
            count(f.texture, m.stem, abs(f.normal[2]) < 0.3)
    if source_engine:
        for n in list(used):
            if walls.get(n) and not bsp.is_tool(n) and not (skip_tex and skip_tex.search(n)):
                img, file = _source_material(folder, n)
                if img is not None:
                    pool[n], origin[n] = img, file
    known = set()
    for k in args.known or []:
        known |= {v["sha"] for v in json.loads(Path(k).read_text(encoding="utf-8")).values()}
    sha = {n: hashlib.sha256(pool[n].tobytes()).hexdigest()[:16] for n in (own if args.unused else used) if n in pool}
    if args.names:
        names = args.names.split(",")
    elif args.unused:  # shipped in the source's own WADs but on no map: cut or spare art
        names = sorted(n for n in own - set(used) if not bsp.is_tool(n) and not n.startswith("!") and sha[n] not in known)
    else:
        names = sorted(n for n in used if walls.get(n) and n in pool and not bsp.is_tool(n) and not n.startswith("!")
                       and not (skip_tex and skip_tex.search(n)) and sha[n] not in known)
    out = Path(args.out)
    (out / "textures").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
    index = {}
    for n in names:
        target = out / "textures" / f"{n}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        pool[n].save(target)
        index[n] = {"size": list(pool[n].size), "walls": walls.get(n, 0), "maps": sorted(used.get(n, [])), "sha": sha[n],
                    "file": origin.get(n)}
    (out / ("unused.json" if args.unused else "textures.json")).write_text(json.dumps(index, indent=1), encoding="utf-8", newline="\n")
    per = args.cols * args.rows
    tag = "pick" if args.names else "unused" if args.unused else "all"
    for i in range(0, len(names), per):
        chunk = names[i:i + per]
        wad.sheet([(f"{i + j} {n}", pool[n]) for j, n in enumerate(chunk)],
                  out / "sheets" / f"{tag}-{i // per:02d}.png", cell=args.cell, cols=args.cols)
    print(f"{len(used)} textures used, {len(names)} on sheets ({(len(names) + per - 1) // per} sheets)")
    return 0


def cmd_measure(args):
    game = Game(sources.DATA / args.source)
    skip = re.compile(args.exclude) if args.exclude else None
    rides, skies, nets = [], [], []
    for n in game.map_names:
        if skip and skip.search(n):
            continue
        b = game.read(n)
        rides += measure.rides(b)
        skies.append(measure.sky(b, bsp.is_tool))
        net = measure.tracks(b)
        if net["segments"]:
            nets.append(net)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "measures.json").write_text(json.dumps({"rides": rides, "tracks": nets, "sky": skies}, indent=1), encoding="utf-8", newline="\n")
    open_maps = [s for s in skies if s["sky_m2"] > 0]
    print(f"{len(skies)} maps, {len(open_maps)} with open sky; {len(rides)} rides; "
          f"track in {len(nets)} maps, {sum(n['length_m'] for n in nets):.0f} m before merging overlaps")
    for n in nets:
        for lift in n["lifts"]:
            print(f"  track lift {n['map']} {lift['name']}: {lift['height_m']} m, ~{lift['seconds']} s")
    for r in sorted(rides, key=lambda r: -r["length_m"])[:args.top]:
        print(f"  {r['map']:10} {r['class']:15} {r['name'][:22]:22} {r['length_m']:7.1f} m"
              f"  rise {r['rise_m']:7.1f} m  speeds {r['speeds']}  ~{r['seconds']} s")
    return 0


def cmd_labels(args):
    folder = sources.DATA / args.source
    skip = re.compile(args.exclude) if args.exclude else None
    out, files = {}, sorted(folder.rglob("*.vmf"))
    for p in files:
        rel = p.relative_to(folder).as_posix()
        if skip and skip.search(rel):
            continue
        out[rel] = vmf.labels(p.read_text("latin-1"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "vmf-labels.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    totals = {k: sum(len(v[k]) for v in out.values()) for k in ("visgroups", "instances", "names", "comments")}
    print(f"{len(out)} source files: " + ", ".join(f"{v} {k}" for k, v in totals.items()))
    return 0


def cmd_transcribe(args):
    from . import speech
    folder = sources.DATA / args.source
    files = sorted(p for pat in args.include for p in folder.glob(pat))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "transcripts.json"
    done = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    tr = speech.Transcriber(args.model)
    for i, p in enumerate(files):
        rel = p.relative_to(folder).as_posix()
        if rel in done and done[rel].get("model") == args.model:
            continue
        data = p.read_bytes()
        r = tr(data, args.prompt)
        done[rel] = {**r, "model": args.model, "sha256": hashlib.sha256(data).hexdigest()}
        if i % 25 == 0:
            target.write_text(json.dumps(done, indent=1), encoding="utf-8", newline="\n")
            print(f"{i + 1}/{len(files)} {rel}: {r['text'][:70]}")
    target.write_text(json.dumps(done, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(done)} transcripts in {target}")
    return 0


def cmd_match(args):
    from PIL import Image
    from . import match
    folder = sources.DATA / args.source
    game = Game(folder)
    pat = re.compile(args.maps)
    names = [n for n in game.map_names if pat.search(n)]
    graph_file = Path(args.graph) if args.graph else None
    offsets = json.loads(graph_file.read_text(encoding="utf-8"))["maps"] if graph_file else {}
    segs = {n: match.wall_segments(game.read(n), offsets.get(n, {}).get("offset", (0, 0, 0))) for n in names}
    cands = {n: [n] for n in names}
    if offsets:
        groups = {}
        for n in names:
            if n in offsets:
                groups.setdefault(re.sub(r"[a-z]?\d?$", "", n) if args.group else "all", []).append(n)
        cands.update({"+".join(v) if len(v) < 4 else f"{k}* ({len(v)} maps)": v for k, v in groups.items() if len(v) > 1})
    sign = match.wireframe(Image.open(args.image), hue=tuple(args.hue), faint=args.faint)
    scales = np.geomspace(args.min_scale, args.max_scale, args.steps)
    results = []
    for label, ms in cands.items():
        r = match.match(sign, [s for m in ms for s in segs[m]], scales, tuple(args.aspects))
        if r:
            results.append({"candidate": label, "maps": ms, **r})
    results.sort(key=lambda r: r["score"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    (out / f"match-{stem}.json").write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    best = results[0]
    img = Image.fromarray(match.overlay(sign, [s for m in best["maps"] for s in segs[m]], best))
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(out / f"match-{stem}.png")
    for r in results[:args.top]:
        print(f"{r['score']:5.2f} {r['candidate'][:40]:40} forward {r['forward']} reverse {r['reverse']} "
              f"scale {r['scale']:.4f} rot {r['rot']} mirror {r['mirror']}")
    return 0


def cmd_layout(args):
    from . import layout
    game = Game(sources.DATA / args.source)
    skip = re.compile(args.exclude) if args.exclude else None
    maps = {n: _read(args.source, game, n) for n in game.map_names if not (skip and skip.search(n))}
    decided = redlines.decided_links(args.source)
    r = layout.stage1(maps, args.start or game.start, decided)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = layout.draw(r, out / "layout-stage1.png")
    (out / "layout-stage1.json").write_text(json.dumps(
        {k: r[k] for k in ("component", "maps", "joins", "clashes", "revisits", "offsets")}, indent=1), encoding="utf-8", newline="\n")
    print(f"stage 1: {len(r['maps'])} maps from {r['component']}, {info['width_m']} x {info['depth_m']} m, "
          f"{info['height_m']} m high")
    print(f"elastic joins: {len(r['joins'])} ({sum(j['status'] == 'open' for j in r['joins'])} open)")
    for j in r["joins"]:
        print(f"  {j['status']:7} {j['maps'][0]}/{j['maps'][1]} via {j['landmark']} ({j['kind']}): "
              f"{j['length_m']} m unseen, rise {j['rise_m']} m")
    print(f"revisits (same room in two maps): {len(r['revisits'])}; clashes for stage 2: {len(r['clashes'])}")
    for c in r["clashes"][:args.top]:
        print(f"  {c['maps'][0]} x {c['maps'][1]}: {c['m2']} m2, same height {c['same_height']:.0%}")
    return 0


WIKIS = {"combine-overwiki": "https://combineoverwiki.net/api.php", "half-life-fandom": "https://half-life.fandom.com/api.php"}


def cmd_wiki(args):
    from . import wiki
    folder = sources.DOWNLOADS / "wikis"
    db = sources.REPO / "out" / "wikis.sqlite"
    if args.action == "fetch":
        folder.mkdir(parents=True, exist_ok=True)
        for name, api in WIKIS.items():
            print(f"{name}: {wiki.fetch(api, folder / f'{name}.jsonl', log=lambda s: None)} articles")
    elif args.action == "import":
        name, dump, pattern = args.query
        print(f"{name}: {wiki.import_dump(dump, folder / f'{name}.jsonl', pattern)} pages kept")
    elif args.action == "index":
        db.parent.mkdir(exist_ok=True)
        print(f"{wiki.index(db, {p.stem: p for p in sorted(folder.glob('*.jsonl'))})} pages indexed in {db}")
    else:
        for hit in wiki.search(db, " ".join(args.query), args.limit):
            print(f"{hit['site']} / {hit['title']} (rev {hit['revid']}):\n  {' '.join(hit['passage'].split())}")
    return 0


def cmd_shared(args):
    game, other = Game(sources.DATA / args.source), Game(sources.DATA / args.with_source)

    def pick(g, pattern):
        rx = re.compile(pattern) if pattern else None
        return {n: g.read(n) for n in g.map_names if not rx or rx.search(n)}
    found = shared.find(pick(other, args.with_maps), pick(game, args.maps), min_m2=args.min_m2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in found:
        del p["_tex"]
    (out / f"shared-{args.with_source}.json").write_text(json.dumps(found, indent=1), encoding="utf-8", newline="\n")
    for p in found:
        if p["class"] == "room":
            tex = ", ".join(list(p["textures"])[:4])
            print(f"{p['map']:14} = {p['with']:10} at {p['translation']}: {p['matched_m2']} m2 "
                  f"({p['share_of_map']:.0%}), {p['extent_m']} m; {tex}")
    counts = {c: sum(1 for p in found if p["class"] == c) for c in ("room", "scenery", "pattern")}
    print(f"{len(found)} peaks: {counts}")
    return 0


def _run_solver(s, maps, out, args, name):
    """Solve once (--order), compare variants (--compare N, --seeds N) or go on from a finished layout (--from); write <name>-<profile>.json/png."""
    from . import solve
    rules = [r for r, _ in s.ground.values()]
    print(f"{len(s.names)} maps in {len(set(s.body))} bodies; {len(s.clash_pairs())} clashing pairs; "
          f"{rules.count('sky')} maps under open sky, {rules.count('roof')} indoors; cost {s.total():.0f} at the start")
    out.mkdir(parents=True, exist_ok=True)
    keys = ("clash_total_m3", "off_ground_maps", "unseen_total_m", "disagreement_total_m", "total_cost")
    if args.layout:
        if args.anneal and not args.gpu:
            sys.exit("--from with --anneal needs --gpu")
        old = json.loads(Path(args.layout).read_text(encoding="utf-8"))
        s.set_offsets({n: old["offsets"][n if n in old["offsets"] else n.split("#")[0]] for n in s.names})
        print(f"going on from {args.layout}: cost {s.total():.1f} here, {old['total_cost']} in that report", flush=True)
        results = solve.warm(s, args.profile, args.seeds or 1, args.anneal, args.gpu, args.t0)
    elif args.compare or args.seeds:
        if args.seeds:
            variants, processes = [(args.profile, f"random:{k}", args.anneal) for k in range(1, args.seeds + 1)], min(4, args.seeds)
        else:
            variants, processes = [(p, o, args.anneal) for p in ("strict", "lenient")
                                   for o in ["largest", "smallest", "play"] + [f"random:{k}" for k in range(1, args.compare + 1)]], 10
        results = solve.compare(s, variants, processes, gpu=args.gpu)
        (out / "variants").mkdir(exist_ok=True)
        for r in results:
            tag = r["order"].split()[0].replace(":", "")
            (out / "variants" / f"{r['profile']}-{tag}.json").write_text(json.dumps(r, indent=1), encoding="utf-8", newline="\n")
        (out / f"{name}-compare.json").write_text(json.dumps(
            [{k: r[k] for k in ("profile", "order") + keys} | {"moves": len(r["moves"]), "joins": len(r["joins"])}
             for r in results], indent=1), encoding="utf-8", newline="\n")
    else:
        results = [s.solve(args.order, anneal=args.anneal, gpu=args.gpu)]
    print(f"{'profile':8} {'order':10} {'moves':>5} {'clash m3':>8} {'off-ground':>10} {'unseen m':>8} {'joins':>5} "
          f"{'disagree m':>10} {'cost':>8}")
    for r in sorted(results, key=lambda r: (r["profile"], r["total_cost"])):
        print(f"{r['profile']:8} {r['order']:10} {len(r['moves']):5} {r['clash_total_m3']:8} {r['off_ground_maps']:10} "
              f"{r['unseen_total_m']:8} {len(r['joins']):5} {r['disagreement_total_m']:10} {r['total_cost']:8}")
    best = {}
    for r in results:
        if r["profile"] not in best or r["total_cost"] < best[r["profile"]]["total_cost"]:
            best[r["profile"]] = r
    for prof, r in best.items():
        (out / f"{name}-{prof}.json").write_text(json.dumps(r, indent=1), encoding="utf-8", newline="\n")
        solve.draw(maps, r, out / f"{name}-{prof}.png")
        print(f"best {prof}: order {r['order']}, cost {r['total_cost']}")
    return 0


def _surface(source, args):
    if args.no_surface:
        return None
    rules = (redlines.rule("surface") or {}).get(source)
    return {**rules, "tolerance_m": args.ground_m} if rules and args.ground_m else rules


def cmd_solve(args):
    from . import attach, solve
    game = Game(sources.DATA / args.source)
    skip = re.compile(args.exclude) if args.exclude else None
    maps = {n: _read(args.source, game, n) for n in game.map_names if not (skip and skip.search(n))}
    tier = redlines.tier(args.source)
    start, decided = args.start or game.start, redlines.decided_links(args.source)
    if start not in maps:
        sys.exit(f"no start map for {args.source} (Source games don't name one): pass --start")
    g, links = solve.game_layout(maps, start, decided, args.source, tier)
    same = attach.aligned(redlines.place_table(), args.source, args.source)  # two maps of the game people say are one place
    revisit = solve.revisit_links(maps, start, decided, args.source, tier, shared.stored(args.source, args.source) + same)
    links += revisit
    root = g["start"]
    placed = {n: np.array(m["offset"], float) for n, m in g["maps"].items() if m["component"] == root}
    part = solve.parts(solve.game_layout(maps, start, decided, args.source, tier, physical=True)[0])
    placed = solve.join_revisits(placed, revisit, part, root)
    s = solve.Solver(maps, placed, links, root, redlines.place_table(), _surface(args.source, args), args.profile,
                     tiers={n: tier for n in placed})
    return _run_solver(s, maps, Path(args.out), args, "solve")


def _attach_solver(args):
    """The stage 3 solver for attach's arguments: (solver, what attach found, {map: its game}, the whole maps)."""
    from . import attach, solve
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    game = Game(sources.DATA / args.source)
    maps = {n: _read(args.source, game, n) for n in base["maps"]}
    tier = redlines.tier(args.source)
    _g, links = solve.game_layout(maps, args.start or game.start, redlines.decided_links(args.source), args.source, tier)
    placed = {n: np.array(o, float) for n, o in base["offsets"].items()}
    locations = redlines.place_table()
    additions = []
    for item in args.add:
        src, _, solved = item.partition("=")
        g = Game(sources.DATA / src)
        additions.append({"game": src, "tier": redlines.tier(src), "maps": {n: _read(src, g, n) for n in g.map_names},
                          "start": g.start, "decided": redlines.decided_links(src),
                          "rooms": shared.stored(src, args.source) + attach.aligned(locations, src, args.source),
                          "self": shared.stored(src, src) + attach.aligned(locations, src, src), "solved": json.loads(Path(solved).read_text(encoding="utf-8"))["offsets"] if solved else None})
    geometry = dict(maps)
    added, added_links, info = attach.attach(placed, geometry, additions, locations, split=args.rigid_tier < 2)
    for a in info:
        if "part" in a:
            print(f"{a['game']}: {len(a['part'])} maps placed by {a['placed_by']}"
                  + (f"; rooms: {'; '.join(a['rooms'])}" if "rooms" in a else ""))
        elif a["not_attached"]:
            print(f"{a['game']}: not attached: {', '.join(f'{c} ({len(m)})' for c, m in a['not_attached'].items())}")
    tiers = {n: tier for n in maps}
    games = {n: args.source for n in maps}
    for add in additions:
        for n in added:
            if n in add["maps"]:
                tiers[n], games[n] = add["tier"], add["game"]
    s = solve.Solver(geometry, {**placed, **added}, links + added_links, args.start or game.start, locations,
                     _surface(args.source, args), args.profile, tiers=tiers, games=games,
                     rigid_tier=args.rigid_tier, stretch=args.stretch, shorter=args.shorter, said={n: redlines.claims(g, n) for n, g in games.items()})
    return s, info, games, geometry


def cmd_attach(args):
    s, info, _games, _whole = _attach_solver(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attach-parts.json").write_text(json.dumps(info, indent=1), encoding="utf-8", newline="\n")
    (out / "attach-links.json").write_text(json.dumps([{"maps": l["maps"], "kind": l["kind"], "delta": [float(x) for x in l["delta"]], "landmark": str(l["landmark"]),
                                                        "game": l.get("game", "")} for l in s.links], indent=1), encoding="utf-8", newline="\n")
    return _run_solver(s, s.maps, out, args, "attach")


def cmd_routed(args):
    from . import connectors, routes, solve, view
    from .graph import INCH
    s, _info, games, whole = _attach_solver(args)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    s.set_offsets(report["offsets"])
    print(f"{len(s.names)} pieces, cost {s.total():.1f} (report {report['total_cost']}), filler {s.filler_m3():.0f} m3", flush=True)
    paths = routes.route_all(s)
    cuts = {str(c["middle"]): c for n, b in whole.items()
            for c in connectors.claimed(b, [r for r in redlines.claims(games[n], n) if r["kind"] == "connector"], games[n])[0]}
    fillers = []
    for l in s.inner:
        gap_a, gap_b = s.off0[l["a"]] + s.shift[l["a"]] * solve.CELL, s.off0[l["b"]] + s.shift[l["b"]] * solve.CELL
        if np.abs(gap_b - gap_a).max() >= solve.CELL:
            outline = np.array(cuts[str(l["middle"])]["outline"], float)
            fillers.append((outline + gap_a, outline + gap_b))
    z, tol = s.ground[s.ref][1] + s.off0[s.ref][2] + s.shift[s.ref][2] * solve.CELL, s.tol / INCH
    pieces = {n: (games[n.split("#")[0]], s.maps[n]) for n in s.names}
    info = view.export({**report, "maps": s.names}, pieces, args.out, (z - tol, z + tol, "surface band"), args.title or Path(args.report).parent.name, paths, fillers)
    done = [r for r in paths if r["cells"]]
    print(f"{args.out}: {info['maps']} maps, {info['mb']} MB, {len(fillers)} fillers; routed {sum(r['route_m'] for r in done):.0f} m, "
          f"straight {sum(r['straight_m'] for r in paths):.0f} m, no path {len(paths) - len(done)}")
    return 0


def cmd_app(args):
    from . import app
    return app.run(args.port)


def cmd_serve(args):
    from . import serve
    serve.run(args.port)
    return 0


def cmd_recipe(args):
    from . import recipe
    return recipe.run(args.out, args.quick, args.gpu)


def _read(source, game, name):
    """A map read for detection or a layout: its pieces analysed with what people say about its areas (bmab.pieces.analyse), so
    everything after sees them."""
    b = game.read(name)
    pieces.analyse(b, redlines.said(source, name, "area"))
    return b


def _report_games(report, srcs):
    """{map: (source, bsp)} for a report's maps, found in the listed sources, in source order."""
    found = [(src, Game(sources.DATA / src)) for src in srcs]
    games = {}
    for n in report["maps"]:
        src, g = next(((src, g) for src, g in found if n in g.map_names), (None, None))
        if g is None:
            sys.exit(f"{n} is in none of {', '.join(srcs)}")
        games[n] = (src, _read(src, g, n))
    return dict(sorted(games.items(), key=lambda kv: (srcs.index(kv[1][0]), kv[0])))


def cmd_view(args):
    from . import solve, view
    from .graph import INCH
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    games = _report_games(report, args.sources)
    band = None
    surf = (redlines.rule("surface") or {}).get(args.sources[0])
    if surf and surf["reference"] in games:
        _k, _b, (z, _m2) = solve.volume(games[surf["reference"]][1])
        z += report["offsets"][surf["reference"]][2]
        tol_m = args.band_m or surf["tolerance_m"]
        band = (z - tol_m / INCH, z + tol_m / INCH, f"{surf['reference']} sky +-{tol_m:g} m")
    title = args.title or f"{Path(args.report).parent.name}: {report['profile']} {report['order']}"
    info = view.export(report, games, args.out, band, title)
    print(f"{args.out}: {info['maps']} maps, {info['triangles']:,} triangles, {info['mb']} MB")
    return 0


def cmd_places(args):
    from . import places
    reg = redlines.register()
    tree = places.build(reg, redlines.place_table())
    Path(args.out).write_text(places.page(tree, reg) if args.out.endswith(".html") else places.render(tree), encoding="utf-8", newline="\n")
    placed = sum(1 for n in tree.values() if n["parents"] or n["children"])
    print(f"{args.out}: {len(tree)} places, {placed} in the hierarchy, "
          f"{sum(1 for n in tree.values() if n['implied'])} implied")
    return 0


def cmd_fake_outs(args):
    from . import fake_outs
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    found = [(src, Game(sources.DATA / src)) for src in args.sources]
    jobs = []
    for n in report["maps"]:
        src = next((src for src, g in found if n in g.map_names), None)
        if src is None:
            sys.exit(f"{n} is in none of {', '.join(args.sources)}")
        jobs.append((src, n, redlines.said(src, n, "area")))
    out, failed = fake_outs.find_all(jobs, args.jobs, args.out)
    print(f"{args.out}: {sum(map(len, out.values()))} fake outs in {sum(1 for v in out.values() if v)} of {len(out)} maps")
    if failed:
        print(f"out of memory, left out: {', '.join(failed)}")
    return 0


def cmd_evaluate(args):
    from . import evaluate, fake_outs
    games = _report_games(json.loads(Path(args.report).read_text(encoding="utf-8")), args.sources)
    pick = re.compile(args.maps) if args.maps else None
    todo = [n for n in games if (not pick or pick.search(n)) and (args.redo or not redlines.has_map(games[n][0], n))]
    if args.fake_outs:
        found = json.loads(Path(args.fake_outs).read_text(encoding="utf-8"))
    else:
        temp = sources.REPO / "out" / "evaluate-fake-outs.json"
        found, failed = fake_outs.find_all([(games[n][0], n, redlines.said(games[n][0], n, "area")) for n in todo], args.jobs, temp)
        temp.unlink()
        if failed:
            print(f"out of memory, no fake outs for: {', '.join(failed)}")
    total = evaluate.run(games, todo, found)
    print(f"{total} claims about {len(todo)} maps in {redlines.REDLINES / 'maps'}; nothing is committed, look at the diff first")
    return 0


def cmd_score(args):
    from . import evaluate, fake_outs, score
    games = _report_games(json.loads(Path(args.report).read_text(encoding="utf-8")), args.sources)
    pick = re.compile(args.maps)
    todo = [n for n in games if pick.search(n)]
    if args.fake_outs and Path(args.fake_outs).exists():
        found = json.loads(Path(args.fake_outs).read_text(encoding="utf-8"))
    else:
        found, failed = fake_outs.find_all([(games[n][0], n, redlines.said(games[n][0], n, "area")) for n in todo], args.jobs, Path(args.fake_outs or sources.REPO / "out" / "score-fake-outs.json"))
        if failed:
            print(f"out of memory, no fake outs for: {', '.join(failed)}")
    results = {n: score.compare(n, games[n][1], rows, score.answers(games[n][0], n, args.against)) for n, rows in evaluate.detected(games, todo, found)}
    print("\n".join(score.table(results)))
    out = sources.REPO / "out" / "score.json"
    out.write_text(json.dumps({n: {k: {**v, "changes": dict(v["changes"])} for k, v in kinds.items()} for n, kinds in results.items()}, indent=1), encoding="utf-8", newline="\n")
    print(f"what was wrong and missed, by position: {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="bmab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="copy sources into data/ and verify them")
    imp.add_argument("sources", nargs="*", help="source ids from sources.json (default: all)")
    imp.add_argument("--pin", action="store_true", help="accept the files found and rewrite their lock entries")
    imp.set_defaults(fn=cmd_import)
    pl = sub.add_parser("plan", help="connection graph, floor plan and overlaps of an imported source")
    pl.add_argument("source", help="source id, e.g. half-life")
    pl.add_argument("out")
    pl.add_argument("--exclude", help="regex of map names to leave out")
    pl.add_argument("--start", help="map to place first (default: the game's start map)")
    pl.set_defaults(fn=cmd_plan)
    mi = sub.add_parser("mine", help="announcements, titles, messages and names that mention places")
    mi.add_argument("source")
    mi.add_argument("out")
    mi.add_argument("--top", type=int, default=40, help="how many place phrases to list")
    mi.set_defaults(fn=cmd_mine)
    me = sub.add_parser("measure", help="ride paths (trains, trams, lifts) and open sky per map")
    me.add_argument("source")
    me.add_argument("out")
    me.add_argument("--exclude", help="regex of map names to leave out")
    me.add_argument("--top", type=int, default=30, help="how many of the longest rides to list")
    me.set_defaults(fn=cmd_measure)
    lb = sub.add_parser("labels", help="designer labels in Hammer sources (visgroups, instances, names, comments)")
    lb.add_argument("source")
    lb.add_argument("out")
    lb.add_argument("--exclude", help="regex of source paths to leave out")
    lb.set_defaults(fn=cmd_labels)
    sp = sub.add_parser("transcribe", help="voice files to text with Whisper (needs openai-whisper)")
    sp.add_argument("source")
    sp.add_argument("out")
    sp.add_argument("--include", nargs="+", default=["sound/**/*.wav"], help="globs inside the imported source")
    sp.add_argument("--model", default="large-v3")
    sp.add_argument("--prompt", default="Black Mesa Research Facility. Sector C, Lambda Complex, Freeman, Rosenberg, "
                    "Keller, Colette Green, Gina Cross, Barney, HEV, Xen, anti-mass spectrometer, tram.",
                    help="vocabulary hint for the model")
    sp.set_defaults(fn=cmd_transcribe)
    wk = sub.add_parser("wiki", help="local wiki copies for cross-checking: fetch, index, search")
    wk.add_argument("action", choices=["fetch", "import", "index", "search"])
    wk.add_argument("query", nargs="*", help="search: an FTS5 query, e.g. '\"freight yard\" NEAR prototype'; "
                    "import: NAME DUMP TITLE_REGEX (a MediaWiki XML export, .xml or .xml.zst)")
    wk.add_argument("--limit", type=int, default=8)
    wk.set_defaults(fn=cmd_wiki)
    sh = sub.add_parser("shared", help="rooms built into two sources' maps: identical faces at one translation")
    sh.add_argument("source")
    sh.add_argument("out")
    sh.add_argument("--with", dest="with_source", default="half-life", help="source to compare against")
    sh.add_argument("--maps", help="regex: only these maps of SOURCE")
    sh.add_argument("--with-maps", help="regex: only these maps of the --with source")
    sh.add_argument("--min-m2", type=float, default=100)
    sh.set_defaults(fn=cmd_shared)
    rules = argparse.ArgumentParser(add_help=False)
    rules.add_argument("--profile", choices=["strict", "lenient"], default="strict")
    rules.add_argument("--no-surface", action="store_true", help="ignore the redlines' surface rules (to measure what they cost)")
    rules.add_argument("--ground-m", type=float, metavar="M", help="how far open-sky maps may sit from the surface (default: the redlines' surface rules)")
    search = argparse.ArgumentParser(add_help=False)
    search.add_argument("--order", default="largest", help="largest | smallest | play | random:SEED")
    search.add_argument("--anneal", type=int, default=0, metavar="N",
                        help="after the greedy search, N simulated-annealing proposals, then greedy again")
    search.add_argument("--gpu", type=int, default=0, metavar="K",
                        help="anneal on the GPU (needs PyTorch with CUDA): --anneal batches of K candidate steps each")
    search.add_argument("--compare", type=int, metavar="N",
                        help="run both profiles with every order and N random orders in parallel; keep the best of each")
    search.add_argument("--seeds", type=int, metavar="N", help="run N random orders of this profile and keep the best")
    search.add_argument("--from", dest="layout", metavar="REPORT",
                        help="go on from a finished layout: with --anneal a cool GPU search per --seeds, then the greedy search; without, only the greedy search")
    search.add_argument("--t0", type=float, default=0.5, help="with --from: the search's starting temperature")
    placed = argparse.ArgumentParser(add_help=False)
    placed.add_argument("source", help="the solved game")
    placed.add_argument("base", help="its solve report (solve-<profile>.json)")
    placed.add_argument("--add", nargs="+", required=True, metavar="SOURCE[=REPORT]",
                        help="sources to attach (by out/<source>/shared-<game>.json, place-table alignments and "
                             "facts), each optionally with its own stage 2 report to place its parts by")
    placed.add_argument("--start", help="the solved game's anchor map (default: its start map)")
    placed.add_argument("--stretch", action="store_true", help="cut maps into rooms and connectors that may run longer")
    placed.add_argument("--shorter", action="store_true", help="with --stretch: connectors may also lose what their whole length says can be taken out")
    placed.add_argument("--rigid-tier", type=int, default=1,
                        help="maps linked at this tier or better move only as whole groups (0: everything may bend)")
    so = sub.add_parser("solve", parents=[rules, search], help="stage 2 layout: move maps apart at the least unseen cost")
    so.add_argument("source")
    so.add_argument("out")
    so.add_argument("--exclude", help="regex of map names to leave out")
    so.add_argument("--start", help="map to place first (default: the game's start map)")
    so.set_defaults(fn=cmd_solve)
    at = sub.add_parser("attach", parents=[placed, rules, search], help="stage 3 layout: add other games' maps to a solved layout by shared rooms")
    at.add_argument("out")
    at.set_defaults(fn=cmd_attach)
    ro = sub.add_parser("routed", parents=[placed, rules], help="a finished layout in 3D with passages routed through free space and what fills stretched "
                                                                "connectors; holds the games' geometry: never publish it")
    ro.add_argument("report", help="the finished attach report; pass the arguments that made it")
    ro.add_argument("out", help="the .html file to write")
    ro.add_argument("--title")
    ro.set_defaults(fn=cmd_routed)
    rc = sub.add_parser("recipe", help="every step from the imported games to the finished pages, with the settings of the best layout so far; "
                                       "hours of work, needs PyTorch with CUDA")
    rc.add_argument("out", help="folder for every step's results")
    rc.add_argument("--quick", action="store_true", help="a few minutes of search instead of hours: to check that everything runs")
    rc.add_argument("--gpu", type=int, default=256, metavar="K", help="candidate steps per GPU batch")
    rc.set_defaults(fn=cmd_recipe)
    vw = sub.add_parser("view", help="a solved layout in 3D: one HTML file for this machine (it holds the games' "
                                     "geometry: never publish it)")
    vw.add_argument("report", help="a solve or attach report")
    vw.add_argument("out", help="the .html file to write")
    vw.add_argument("--sources", nargs="+", required=True, help="sources the report's maps come from, base game first")
    vw.add_argument("--title")
    vw.add_argument("--band-m", type=float, help="ground band half-height in metres (default: the redlines' surface rules)")
    vw.set_defaults(fn=cmd_view)
    ps = sub.add_parser("places", help="the place hierarchy from the facts register and the place table, as text")
    ps.add_argument("out", help="text file to write")
    ps.set_defaults(fn=cmd_places)
    ev = sub.add_parser("evaluate", help="run the detectors over maps and write what they claim to the redlines (maps/<map>.jsonl), to review as a git diff")
    ev.add_argument("report", help="a solve or attach report: its maps are in scope")
    ev.add_argument("--sources", nargs="+", required=True, help="sources the report's maps come from")
    ev.add_argument("--maps", help="regex: only these maps")
    ev.add_argument("--redo", action="store_true", help="also maps that already have a file: replaces what is in them")
    ev.add_argument("--fake-outs", help="fake outs already found, from bmab fake-outs (default: search these maps now)")
    ev.add_argument("--jobs", type=int, default=6, help="worker processes for the fake out search (default 6)")
    ev.set_defaults(fn=cmd_evaluate)
    sc = sub.add_parser("score", help="how well the detectors do against maps reviewed by hand: right, wrong and missed, for each kind of claim")
    sc.add_argument("report", help="a solve or attach report: its maps are in scope")
    sc.add_argument("--sources", nargs="+", required=True, help="sources the report's maps come from")
    sc.add_argument("--maps", required=True, help="regex: the reviewed maps")
    sc.add_argument("--against", default="HEAD", help="the redlines' commit holding the reviewed maps (default HEAD)")
    sc.add_argument("--fake-outs", help="a file of fake outs already found for these maps, written by the first run (default: search them now)")
    sc.add_argument("--jobs", type=int, default=4, help="worker processes for the fake out search (default 4)")
    sc.set_defaults(fn=cmd_score)
    ap_ = sub.add_parser("app", help="the app at its front page, in a window of its own on Windows, else in the browser (bmab.app)")
    ap_.add_argument("--port", type=int, default=8650)
    ap_.set_defaults(fn=cmd_app)
    sv = sub.add_parser("serve", help="the app on this computer: the review page of each edition's games here at /<edition>/, edits to the "
                                     "redlines' files, committed only by the Commit button (bmab.serve); its pages hold the games' geometry: never publish them")
    sv.add_argument("--port", type=int, default=8650)
    sv.set_defaults(fn=cmd_serve)
    hn = sub.add_parser("fake-outs", help="fake outs, where a map looks like it goes on: space joined to what the player sees that "
                                          "runs on out of sight (sight lines from the route and rides), and dark ends")
    hn.add_argument("report", help="a solve or attach report: its maps are searched")
    hn.add_argument("out", help="JSON file to write: {map: [fake out]}")
    hn.add_argument("--sources", nargs="+", required=True, help="sources the report's maps come from")
    hn.add_argument("--jobs", type=int, default=6, help="worker processes (default 6)")
    hn.set_defaults(fn=cmd_fake_outs)
    ly = sub.add_parser("layout", help="stage 1 layout: placement, elastic joins, revisits and clashes")
    ly.add_argument("source")
    ly.add_argument("out")
    ly.add_argument("--exclude", help="regex of map names to leave out")
    ly.add_argument("--start", help="map to place first (default: the game's start map)")
    ly.add_argument("--top", type=int, default=15)
    ly.set_defaults(fn=cmd_layout)
    mt = sub.add_parser("match", help="match a wall-map image (top-down wireframe) against map geometry")
    mt.add_argument("image")
    mt.add_argument("source")
    mt.add_argument("out")
    mt.add_argument("--maps", default=".", help="regex of candidate maps")
    mt.add_argument("--graph", help="graph.json to place maps in one frame; adds placed groups as candidates")
    mt.add_argument("--group", action="store_true", help="group placed maps by chapter instead of all together")
    mt.add_argument("--hue", type=int, nargs=2, default=[80, 100], help="OpenCV hue range of the drawn lines")
    mt.add_argument("--faint", action="store_true", help="also use dim red-brown lines (context drawn around a route)")
    mt.add_argument("--aspects", type=float, nargs="+", default=[0.75, 0.87, 1.0, 1.15, 1.33],
                    help="width/height stretch factors to try (signs are often squashed to fit their texture)")
    mt.add_argument("--min-scale", type=float, default=0.006)
    mt.add_argument("--max-scale", type=float, default=0.12)
    mt.add_argument("--steps", type=int, default=30)
    mt.add_argument("--top", type=int, default=8)
    mt.set_defaults(fn=cmd_match)
    tx = sub.add_parser("textures", help="wall textures as images and contact sheets, with the maps that use them")
    tx.add_argument("source")
    tx.add_argument("out")
    tx.add_argument("--names", help="comma-separated texture names to render instead of all wall textures")
    tx.add_argument("--exclude", help="regex of map names to leave out")
    tx.add_argument("--skip", help="regex of texture names to leave out (blood, stains...)")
    tx.add_argument("--unused", action="store_true", help="textures in the source's own WADs that no map uses")
    tx.add_argument("--known", nargs="*", help="textures.json files of textures already read; identical images are skipped")
    tx.add_argument("--cell", type=int, default=96)
    tx.add_argument("--cols", type=int, default=12)
    tx.add_argument("--rows", type=int, default=10)
    tx.set_defaults(fn=cmd_textures)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
