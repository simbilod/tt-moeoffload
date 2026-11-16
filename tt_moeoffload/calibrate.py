"""The engine's prefill cost model measured on this host, into $TT_MOEOFFLOAD_DATA/tune.json (tt/tune.py reads it when the
engine starts). One engine build on the card (tt/serve/check_card.py spec, step fit; ~10 min, Flash-Next's served build:
131K context, 4 spec rows) times each prefill expert path (the CPU lane alone, 1 or 2 quarters of the experts on the
card beside it, the card alone) at prompt sizes 256 .. 8,192, the 4-row prompt loop and the reset. Each path's line is
fit where it is the cheapest; the bounds are where the lines cross: how many CPU cores the host has and how fast
its PCIe link is decides which path a prompt takes. GLM's costs and the card-side terms (a second chunk, the
attention's quadratic) keep their defaults.

    python3 tt_moeoffload/calibrate.py [--cpus all] [--dry-run] [--from RESULT.json]

--cpus: the engine container's CPUs (as served); --from: fit an earlier run's measurements (no card); --dry-run: print,
do not write.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt_moeoffload.launch import data_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULT = "results/card_check_spec_ctx131072_fit.json"   # under the data root: check_card.py writes it there
ORDER = ("cpu", "split1", "split2", "card")      # engine.use_experts' paths, in the order prompt sizes take them
# the card alone is not linear below this (1,536-2,048 tokens cost more than 3,072, 2026-11-16, every run); its line
# is the long prompts' (prefill_s past every bound), so it is fit from here up
CARD_FIT_FROM = 3072


def line(pts: list[tuple[int, float]]) -> tuple[float, float]:
    """least squares wall = fixed + per * k"""
    n = len(pts)
    mk, mw = sum(k for k, _ in pts) / n, sum(w for _, w in pts) / n
    sk = sum((k - mk) ** 2 for k, _ in pts)
    per = sum((k - mk) * (w - mw) for k, w in pts) / sk if sk else 0.0
    return mw - per * mk, per


def fit(m: dict, quad: float) -> dict:
    """check_card.py's fit step -> tune.json's values. The walls less the attention's quadratic term (kept)."""
    walls = {p: {int(k): w - quad * int(k) ** 2 for k, w in m["prefill_s"][p].items()} for p in ORDER}
    sizes = sorted(walls["card"])
    best, r = [], 0
    for n in sizes:                                # the cheapest path at each size, never back down ORDER (noise)
        r = max(r, min((i for i, p in enumerate(ORDER) if n in walls[p]), key=lambda i: walls[ORDER[i]][n]))
        best.append(r)
    used = sorted(set(best))
    lines = {}
    for i in range(len(ORDER)):                    # each line over where it wins and the sizes beside it
        at = [j for j, b in enumerate(best) if b == i]
        lo, hi = (max(at[0] - 1, 0), min(at[-1] + 1, len(sizes) - 1)) if at else (0, len(sizes) - 1)
        pts = [(n, walls[ORDER[i]][n]) for n in sizes[lo:hi + 1] if n in walls[ORDER[i]]]
        if ORDER[i] == "card":
            pts = [(n, w) for n, w in walls["card"].items() if n >= CARD_FIT_FROM]
        lines[i] = line(pts if len(pts) > 1 else sorted(walls[ORDER[i]].items()))
    bounds = {}
    for i, j in zip(used, used[1:]):               # where path i hands over to j: the lines' crossing between the sizes
        a = sizes[max(k for k, b in enumerate(best) if b == i)]
        b = sizes[min(k for k, b in enumerate(best) if b == j)]
        (fi, pi), (fj, pj) = lines[i], lines[j]
        x = (fj - fi) / (pi - pj) if pi != pj else None
        bounds[i] = int(x) if x is not None and a <= x <= b else (a + b) // 2
    fixed, per = lines[ORDER.index("card")]
    if per <= 0:
        raise SystemExit(f"calibrate: the card's prefill line has slope {per:.2e} s a token: re-run (a noisy run?)")
    cf, cp = lines[0]
    return {"PREFILL_S": round(fixed, 3), "PREFILL_TOK_S": round(1 / per, 1),
            "CPU_PREFILL_MAX": bounds.get(0, 0), "CPU_PREFILL_S": round(cf, 3), "CPU_PREFILL_TOK_S": round(cp, 6),
            "SPLIT": [[bounds[i], int(ORDER[i][5:]), round(lines[i][0], 3), round(lines[i][1], 6)]
                      for i in used if ORDER[i].startswith("split") and i in bounds],
            "SpecModel.step_s": m["step_s"], "SpecModel.reset_s": m["reset_s"]}


def engine_defaults() -> dict:
    """the engine's values now (its defaults, or the tune.json in place), for the report and the quadratic"""
    spec = importlib.util.spec_from_file_location("ttcard_engine", REPO / "tt" / "serve" / "engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    return {"PREFILL_QUAD_S": e.PREFILL_QUAD_S, "PREFILL_S": e.PREFILL_S, "PREFILL_TOK_S": e.PREFILL_TOK_S,
            "CPU_PREFILL_MAX": e.CPU_PREFILL_MAX, "CPU_PREFILL_S": e.CPU_PREFILL_S,
            "CPU_PREFILL_TOK_S": e.CPU_PREFILL_TOK_S, "SPLIT": [list(r) for r in e.SPLIT],
            "SpecModel.step_s": e.SpecModel.step_s, "SpecModel.reset_s": e.SpecModel.reset_s}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cpus", default="all", help="the engine container's CPUs, as it is served (docker --cpuset-cpus)")
    ap.add_argument("--from", dest="src", type=Path, help="an earlier run's result: fit it, no card")
    ap.add_argument("--dry-run", action="store_true", help="print the values, do not write tune.json")
    a = ap.parse_args(argv)
    src = a.src
    if src is None:
        subprocess.run([sys.executable, str(REPO / "tt_moeoffload" / "launch.py"), "--engine",
                        str(REPO / "tt" / "serve" / "check_card.py"), "--name", "tt-moeoffload-calibrate", "--cpus", a.cpus,
                        "--memory-max", str(220 << 30), "--env", "TT_CTX=131072", "--env", "TTC_STEPS=fit", "--",
                        "spec"], check=True)
        src = data_root() / RESULT
    m = json.loads(src.read_text())["fit"]
    now = engine_defaults()
    new = fit(m, now["PREFILL_QUAD_S"])
    print(f"{'':20} {'now':>36} {'measured':>36}   ({m['lane_threads']} lane threads)")
    for k, v in new.items():
        print(f"{k:20} {json.dumps(now[k]):>36} {json.dumps(v):>36}")
    if a.dry_run:
        return 0
    path = data_root() / "tune.json"
    tune = json.loads(path.read_text()) if path.exists() else {}
    tune.update(new, _calibrated={"date": time.strftime("%Y-%m-%d %H:%M"), "cpus": a.cpus, "from": str(src),
                                  "lane_threads": m["lane_threads"]})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(tune, indent=1) + "\n")
    os.replace(tmp, path)
    print(f"wrote {path} (the engine reads it when it starts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
