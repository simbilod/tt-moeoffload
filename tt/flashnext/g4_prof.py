"""Where a decode token's time goes, the whole token at once (g4_model.Model, 48 layers + the head).

The host's clocks partition the token's wall time: Link.trace records every exchange (waiting for the card's page,
the CPU lane or the host's selection, writing the reply) and Model.step every launch; what lies between them is the
host's own Python. A wait is the card's time from the previous reply (or launch) to the page, when the host was there
first: the rest of the previous layer (R2 + combine), the next layer up to its router, a program's launch on the card,
PCIe. The tail after the last exchange is the last layer's rest, the head and the logits' read-back. Sums over a
token's events add up to its wall time by construction; inside a card wait, g3_profile.py splits a layer by phase.

Timing only: the token runs at positions P0.. without the context before them (the attention caches hold whatever
they hold), which costs what a real token costs there (program 1 scores every pool; program 2 attends over the
selected 512 once pos >= 2,048).

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((120 << 30)) --engine tt/flashnext/g4_prof.py

Options (environment): G4_P0 (first position, default 7,900), G4_TOKENS (timed tokens, default 200), G4_TRACE=1
(the token replayed from one captured trace, Model.step_traced), G4_CACHE=1 (the card computes up to G4_CARD_MAX,
default 4, of a DeltaNet layer's routes from its expert cache: g4_model.Cache, fused/dexp_*.h), G4_HEAD (ids, the
default: the head's top 64 read back, 2 KB; logits: the logits, 1 MB; Model.set_head).
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import statistics
import sys
import time
from collections import defaultdict

import numpy as np

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from cpulane import Lane  # noqa: E402
from g2_layers import Gguf  # noqa: E402
from g3_layer import WORKER_L1  # noqa: E402
from g4_host import Host  # noqa: E402
from g4_model import SSD_MODEL, Cache, Model, is_attention  # noqa: E402

OUT = os.path.join(HERE, "results")
REF = os.path.join(DATA, "g4_ref")
P0 = int(os.environ.get("G4_P0", 7900))
N = int(os.environ.get("G4_TOKENS", 200))
WARM = 3
TRACED = os.environ.get("G4_TRACE") == "1"
CACHE = os.environ.get("G4_CACHE") == "1"          # the card computes its cached share of the experts (g5_cache.py)
CARD_MAX = int(os.environ.get("G4_CARD_MAX", 4))
HEAD = os.environ.get("G4_HEAD", "ids")


def group(kind: str, layer: int) -> str:
    """What the card ran before an exchange's page: a DeltaNet layer, attention program 1 or program 2."""
    if kind == "s":
        return "attention p1"
    return "attention p2" if is_attention(layer) else "deltanet"


def partition(trace: list, t_begin: float, t_end: float) -> dict[str, float]:
    """One token's wall seconds by category; the categories sum to t_end - t_begin."""
    out = defaultdict(float)
    cursor = t_begin
    for ev in sorted(trace, key=lambda e: e[2]):
        if ev[0] == "launch":
            _, stage, a, b = ev
            out["host python"] += max(0.0, a - cursor)
            out["host launch"] += b - a
            cursor = b
            continue
        kind, layer, t0, t1, t2, t3 = ev
        g = group(kind, layer)
        out["host python"] += max(0.0, t0 - cursor)
        out[f"card {g}"] += t1 - t0
        out["cpu lane" if kind == "x" else "host selection"] += t2 - t1
        out["link reply write"] += t3 - t2
        cursor = t3
    out["card tail: last layer, head, logits read-back"] += t_end - cursor
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32).tolist()
    lane = Lane()
    t0 = time.time()
    lane.load(SSD_MODEL)
    print(f"lane: experts loaded in {time.time() - t0:.0f} s", flush=True)
    host = Host(SSD_MODEL)
    g = Gguf(SSD_MODEL)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "p0": P0, "tokens": N, "traced": TRACED, "card": CACHE,
           "card_max": CARD_MAX, "head": HEAD}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    try:
        cache = Cache(dev, card_max=CARD_MAX) if CACHE else None
        if cache:
            print(f"the card's expert cache: {len(cache.tensors)} layers in {cache.seconds:.0f} s", flush=True)
        M = Model(dev, g, lane, log=lambda s: print(s, flush=True), cache=cache, head_mode=HEAD)
        print(f"model built ({len(M.segs)} programs per token)", flush=True)
        if TRACED:   # one plain token first: the programs' first launches load their binaries (not allowed in a capture)
            M.step(host.embed(ids[0]), host.ple(ids[0], []), P0 - 1)
            M.capture()
            print("token captured as one trace", flush=True)
        walls, parts, per_layer = [], [], defaultdict(list)
        lane_steps = None
        for i in range(WARM + N):
            if i == WARM:
                lane.stats()                       # resets the lane's per-step clocks
                if CACHE:
                    M.card_share()                 # and the card's counts
            tok = ids[i % len(ids)]
            M.link.trace = []
            t_begin = time.perf_counter()
            x0, emb = host.embed(tok), host.ple(tok, ids[:i % len(ids)])
            (M.step_traced if TRACED else M.step)(x0, emb, P0 + i)
            t_end = time.perf_counter()
            if i < WARM:
                continue
            walls.append(t_end - t_begin)
            parts.append(partition(M.link.trace, t_begin, t_end))
            for ev in M.link.trace:
                if ev[0] in ("x", "s"):
                    per_layer[f"{ev[0]} {ev[1]}"].append((ev[3] - ev[2], ev[4] - ev[3], ev[5] - ev[4]))
        lane_steps = (lane.stats() / N / 1e6).round(3).tolist()
        if CACHE:
            res["card_share"], res["card_share_per_layer"] = M.card_share()
    finally:
        ttnn.close_mesh_device(dev)

    ms = lambda v: round(1e3 * v, 3)  # noqa: E731
    wall = statistics.median(walls)
    cats = sorted({k for p in parts for k in p})
    table = {k: ms(statistics.median([p.get(k, 0.0) for p in parts])) for k in cats}
    res["token_ms"] = ms(wall)
    res["tok_s"] = round(1 / wall, 2)
    res["categories_ms"] = dict(sorted(table.items(), key=lambda kv: -kv[1]))
    res["attributed"] = round(sum(table.values()) / (1e3 * wall), 4)
    res["lane_steps_ms_per_token"] = dict(zip(("quantize hn", "gate|up", "quantize h", "down"), lane_steps))
    res["per_exchange_us"] = {k: [round(1e6 * statistics.median(c[j] for c in v), 1) for j in range(3)]
                              for k, v in sorted(per_layer.items(), key=lambda kv: int(kv[0].split()[1]))}
    print(f"\n{res['token_ms']:.2f} ms per token ({res['tok_s']:.1f} tok/s) at positions {P0}-{P0 + WARM + N - 1}, "
          f"medians over {N} tokens; the categories' medians sum to {100 * res['attributed']:.1f}% of it")
    for k, v in res["categories_ms"].items():
        print(f"  {v:8.3f} ms  {k}")
    print(f"  the lane's steps per token (ms): {res['lane_steps_ms_per_token']}")
    print("  per exchange (us: card wait, lane or selection, reply write), first and last few:")
    keys = list(res["per_exchange_us"])
    for k in keys[:6] + ["..."] + keys[-6:]:
        print(f"    {k}: {res['per_exchange_us'][k]}" if k != "..." else "    ...")
    if CACHE:
        print(f"  the card's share: {res['card_share']:.2f} of 10 routes per DeltaNet exchange on average (G4_CARD_MAX "
              f"{CARD_MAX}); per layer: {res['card_share_per_layer']}")
    name = ("g4_prof_traced" if TRACED else "g4_prof") + (f"_card{CARD_MAX}" if CACHE else "") + ".json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
