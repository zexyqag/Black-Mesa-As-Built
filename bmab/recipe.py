"""The recipe: from the imported games to the solved layout and its 3D page, with the settings of the best layout so far.

Each step is an ordinary bmab command. A step whose result exists is skipped, so a stopped run picks up where it was.
"""
import subprocess
import sys
from pathlib import Path

from . import sources

GAMES = ("half-life", "opposing-force", "blue-shift", "ps2-decay")  # order matters: Decay's dorms need Opposing Force placed first
SOLO = {"half-life": ["--exclude", "^(c[45]|t0|hldemo|[a-bd-z])"],
        "opposing-force": ["--start", "of1a1", "--exclude", "^(op4|ofboot|of0a0|of7a0)"],
        "blue-shift": ["--exclude", "^ba_hazard"]}
GROUND_M = 100
T0 = 0.5
FULL = {"solo_random": 4, "solo_anneal": 40000, "seeds": 3, "batches": 24000, "warm_seeds": 2, "warm_batches": 12000}
QUICK = {"solo_random": 1, "solo_anneal": 400, "seeds": 1, "batches": 400, "warm_seeds": 1, "warm_batches": 400}


def steps(out, budget, gpu):
    """[(what, the file that shows it is done, bmab arguments)]."""
    solo = {g: out / g / "solve-strict.json" for g in SOLO}
    whole, cut = out / "whole" / "attach-strict.json", out / "cut" / "attach-strict.json"
    joined = ["--add", *(f"{g}={solo[g]}" if g in SOLO else g for g in GAMES[1:]), "--rigid-tier", "0", "--ground-m", str(GROUND_M)]
    hardware = ["--gpu", str(gpu)]
    todo = [(f"import {g}", sources.DATA / g / "maps", ["import", g]) for g in GAMES]
    for g in GAMES:
        folder = sources.REPO / "out" / g
        todo += [(f"rooms {g} shares with {w}", folder / f"shared-{w}.json", ["shared", g, str(folder), "--with", w]) for w in dict.fromkeys((g, GAMES[0]))]
    todo += [(f"{g} alone", solo[g], ["solve", g, str(out / g), *SOLO[g], "--compare", str(budget["solo_random"]), "--anneal", str(budget["solo_anneal"])]) for g in SOLO]
    base = ["half-life", str(solo["half-life"])]
    return todo + [
        ("every game at once, whole maps", whole, ["attach", *base, str(whole.parent), *joined, "--seeds", str(budget["seeds"]), "--anneal", str(budget["batches"]), *hardware]),
        ("maps cut at their connectors", cut, ["attach", *base, str(cut.parent), *joined, "--stretch", "--from", str(whole), "--seeds", str(budget["warm_seeds"]),
                                               "--anneal", str(budget["warm_batches"]), "--t0", str(T0), *hardware]),
        ("3D page, passages routed", out / "layout.html", ["routed", *base, str(cut), str(out / "layout.html"), *joined, "--stretch", "--title", "The facility, passages routed"]),
    ]


def run(out, quick, gpu):
    todo = steps(Path(out).resolve(), QUICK if quick else FULL, gpu)
    for k, (what, result, args) in enumerate(todo, 1):
        if result.exists():
            print(f"[{k}/{len(todo)}] {what}: done already", flush=True)
            continue
        print(f"[{k}/{len(todo)}] {what}: bmab {' '.join(args)}", flush=True)
        code = subprocess.run([sys.executable, "-m", "bmab", *args], cwd=sources.REPO).returncode
        if code:
            print(f"stopped at step {k}: {what}")
            return code
    print(f"done: {result.parent}")
    return 0
