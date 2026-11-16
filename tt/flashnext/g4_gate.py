"""The one-command gate. One process builds the model once (g4_model.Model, the CPU lane) and runs
what accepts a card task: the traced timing (g4_prof.py's G4_TRACE=1 run: 200 tokens at positions 7,900.., the token
replayed from one captured trace, the host's waterfall), the short parity (g4_token.py: the 92 tokens of
$TT_MOEOFFLOAD_DATA/g4_ref, teacher-forced from position 0, every position's logits against llama.cpp's: top-1, KL) and the
long one (G4_REF=$TT_MOEOFFLOAD_DATA/g4_ref_long G4_FROM=4775: 5,031 tokens from position 0, the logits compared from 4,775).
Then it prints PASS or FAIL per check and overall, against the thresholds (llama.cpp's own path-to-path distance)
and the host's stored baseline ($TT_MOEOFFLOAD_DATA/gate_baseline.json: ms per token and its run-to-run noise; a host's
first run that passes writes it), and writes
$TT_MOEOFFLOAD_DATA/results/gate_<branch>.json (the branch and commit of the tree this script lives in).

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((120 << 30)) --engine tt/flashnext/g4_gate.py

Order and state. The timing runs first, on the fresh model exactly as g4_prof.py's (one plain token at P0 - 1, which
also loads every program's binaries, then the capture and WARM + N traced tokens), and reads nothing from the card
inside its loop beyond the token's own top 64 (g4_phases.py: reading device tensors back between traced tokens adds
~6 ms of host time to the next one). Each parity then starts from a sequence's start: reset() writes zeros, in place
(the programs hold their addresses), into every state a token leaves behind: the DeltaNet layers' conv states and
recurrent states S, the PLE block's 9-token history ring, the attention layers' K and V caches, raw indexer keys and
pooled keys; the X tensors, the step tensor (position) and the PLE input are rewritten by every token, the CPU lane
holds no state. The parities replay the same captured trace (step_traced: the path being timed). Checked 2026-10-24
during development: the 92 short tokens' logits identical to the bit untraced from a fresh build,
untraced after reset(), traced after reset() and untraced after the capture; both parities' per-position KL and
top-1 lists identical to g4_token.py's.

The head (Model.set_head): the timing runs it in ids mode (the token's top 64 found on the card, 2 KB read back:
the serving path), the parities in both mode (the logits for the KL, and the top 64 beside them): at every position of
both references the card's top 64 must be the logits' own top 64 (ids and value bits, a tie to the lowest id; [0]
the argmax). --check-ids replays the references in ids mode too (the head without logits) and compares every
position's top 64 with the parity's. Measured 2026-10-24 (6fe75fc): the tail 1.786 -> 1.497 ms (the 2 KB read
adds ~15 us after the head against the logits' ~290), 22.41 ms per token against 22.82; the top 64 right at all 92 +
5,031 positions in both modes.

Pass: short top-1 >= 93% and KL mean <= 0.045, long top-1 >= 98% and KL mean <= 0.015; the top 64 as above; ms per
token not slower than the baseline beyond its noise band (max(the baseline runs' spread, --noise-floor-pct of it)). A
speed number without its parity is not reported as passing: --skip-long marks the long check SKIPPED and the overall
result PARTIAL.

Options: --skip-long; --check-ids; --tag T (writes gate_<branch>_T.json); --baseline PATH (default $TT_MOEOFFLOAD_DATA/gate_baseline.json; a
run's output works too, e.g. the baseline branch's gate run of the same sitting); --write-baseline (writes
gate_baseline.json from this run; with --runs A.json B.json ..., from those earlier runs' outputs instead, no card);
--noise-floor-pct (default 1.0). Environment as g4_prof.py's (passed with the launcher's --env K=V: the container gets none of the
caller's environment but STRATA_, TT_MOEOFFLOAD_ and TT_METAL_ variables): G4_P0 (7,900), G4_TOKENS (200), G4_CACHE=1 with G4_CARD_MAX (4): the card's share of
the routed experts (g4_model.Cache), recorded in the output with the card's routes per DeltaNet exchange. Exit status
0 on PASS or PARTIAL, 1 on FAIL.

Measured 2026-10-24, ~4 min per gate including the container's start (in the
process: the lane's experts 23 s, the build 8 s, the timing 6 s, the short parity 13 s, the long one 161-179 s):
25.84, 26.33 and 26.19 ms per token (g4_prof.py G4_TRACE=1 in the same sitting: 25.88); the card's waits before the
exchanges (DeltaNet 10.42, attention 1 2.48, attention 2 2.27 ms) the same within 0.008 ms in every run and in
g4_prof.py's; the run-to-run noise (~0.5 ms) is the host's: the CPU lane (7.53-7.88), the logits' read-back in the
tail (1.78-2.09), the host's Python (0.75-0.87). Short parity 95.7% / 0.0389, long 99.6% / 0.0119 in every run.
"""
from __future__ import annotations

import argparse
import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "results")   # this run's output; the references stay in the repo's results/
BASELINE = os.path.join(DATA, "gate_baseline.json")   # this host's: timings are the host's, not the code's alone
IOMMU = "/sys/class/tenstorrent/*/device/iommu_group/type"   # the card's group (preflight.py finds it so)
REF_SHORT = os.path.join(DATA, "g4_ref")
REF_LONG, LONG_FROM = os.path.join(DATA, "g4_ref_long"), 4775
P0 = int(os.environ.get("G4_P0", 7900))
N = int(os.environ.get("G4_TOKENS", 200))
WARM = 3
VOCAB = 248320
LIMITS = {"short": (0.93, 0.045), "long": (0.98, 0.015)}     # top-1 at least, KL mean at most
CACHE = os.environ.get("G4_CACHE") == "1"                     # the card's share of the experts (g4_model.Cache)
CARD_MAX = int(os.environ.get("G4_CARD_MAX", 4))


def git_info(path: str) -> dict:
    """Branch and commit of the tree holding `path` (git, or the .git files themselves inside the container)."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", path, *a], capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout.strip()
        return dict(branch=run("rev-parse", "--abbrev-ref", "HEAD"), commit=run("rev-parse", "HEAD"),
                    dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except (OSError, subprocess.CalledProcessError):
        pass
    top = path
    while not os.path.exists(os.path.join(top, ".git")):
        top = os.path.dirname(top)
        if top == "/":
            return dict(branch="unknown", commit="unknown", dirty=None)
    gd = os.path.join(top, ".git")
    if os.path.isfile(gd):                                   # a worktree: "gitdir: <path>"
        gd = open(gd).read().split(":", 1)[1].strip()
    common = gd
    if os.path.exists(os.path.join(gd, "commondir")):
        common = os.path.normpath(os.path.join(gd, open(os.path.join(gd, "commondir")).read().strip()))
    head = open(os.path.join(gd, "HEAD")).read().strip()
    if not head.startswith("ref: "):
        return dict(branch="HEAD", commit=head, dirty=None)
    ref = head[5:]
    commit = "unknown"
    for d in (gd, common):
        if os.path.exists(os.path.join(d, ref)):
            commit = open(os.path.join(d, ref)).read().strip()
            break
    else:
        packed = os.path.join(common, "packed-refs")
        for line in open(packed) if os.path.exists(packed) else []:
            if line.strip().endswith(" " + ref):
                commit = line.split()[0]
    return dict(branch=ref.removeprefix("refs/heads/"), commit=commit, dirty=None)


def require_dma_fq() -> None:
    import glob
    paths = sorted(glob.glob(IOMMU))
    if not paths:
        sys.exit(f"gate: no {IOMMU}: the token loop needs the host link (sockets), which needs the card's IOMMU group "
                 f"in DMA-FQ (the IOMMU on; tools/iommu)")
    for path in paths:
        mode = open(path).read().strip()
        if mode not in ("DMA", "DMA-FQ"):        # translated, as preflight.py accepts
            sys.exit(f"gate: {path} is {mode!r}, not 'DMA-FQ': the token loop needs the host link (sockets). After a "
                     f"reset the udev hook restores it within ~2 s; if it stays {mode!r}, stop (setting it is root-level)")


def reset(M) -> None:
    """Every state a token leaves behind back to a sequence's start: zeros written into the tensors in place."""
    import torch
    import ttnn
    from g4_model import Attention, DeltaRun

    def zero(t):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype,
                                                        layout=t.layout), t)
    for p in M.progs:
        if isinstance(p, DeltaRun):
            for L in p.keep:
                zero(L.cstate)
                zero(L.S)
        elif isinstance(p, Attention):   # not the caches: g4_model.Model.reset's note
            continue
        else:                                                      # the PLE block
            zero(p.ring_t)
    ttnn.synchronize_device(M.dev)


def card_waits(categories: dict) -> float:
    """The card's waits before the exchanges (DeltaNet runs, attention programs 1 and 2): the card categories without
    the tail, whose logits' read-back is the host's."""
    return round(sum(v for k, v in categories.items() if k.startswith("card") and "tail" not in k), 3)


def timing(M, host, lane, ids) -> dict:
    """g4_prof.py's traced run: one plain token at P0 - 1, the capture, WARM + N tokens; the host's waterfall."""
    from collections import defaultdict
    import numpy as np
    from g4_prof import partition
    t_start = time.time()
    M.step(host.embed(ids[0]), host.ple(ids[0], []), P0 - 1)
    M.capture()
    walls, parts = [], []
    for i in range(WARM + N):
        if i == WARM:
            lane.stats()                       # resets the lane's per-step clocks
        tok = ids[i % len(ids)]
        M.link.trace = []
        t_begin = time.perf_counter()
        x0, emb = host.embed(tok), host.ple(tok, ids[:i % len(ids)])
        M.step_traced(x0, emb, P0 + i)
        t_end = time.perf_counter()
        if i < WARM:
            continue
        walls.append(t_end - t_begin)
        parts.append(partition(M.link.trace, t_begin, t_end))
    lane_steps = (lane.stats() / N / 1e6).round(3).tolist()
    M.link.trace = None
    ms = lambda v: round(1e3 * v, 3)  # noqa: E731
    wall = statistics.median(walls)
    cats = sorted({k for p in parts for k in p})
    table = defaultdict(float, {k: ms(statistics.median([p.get(k, 0.0) for p in parts])) for k in cats})
    q = np.percentile(walls, [10, 90])
    return dict(p0=P0, tokens=N, token_ms=ms(wall), tok_s=round(1 / wall, 2), p10_ms=ms(q[0]), p90_ms=ms(q[1]),
                categories_ms=dict(sorted(table.items(), key=lambda kv: -kv[1])),
                card_ms=round(sum(v for k, v in table.items() if k.startswith("card")), 3),
                card_waits_ms=card_waits(table),
                attributed=round(sum(table.values()) / (1e3 * wall), 4),
                lane_steps_ms_per_token=dict(zip(("quantize hn", "gate|up", "quantize h", "down"), lane_steps)),
                seconds=round(time.time() - t_start, 1))


def top_ref(lg):
    """The top 64 of logits lg (torch, 1-D) best first, a tie to the lowest id (the card's rule): ids, values (numpy).
    g3_head.topk_ref's order, its sort over the values at least the 64th only (numpy: torch's threads here contend
    with the lane's spinning workers, ~70 ms per position)."""
    import numpy as np
    v = lg.numpy()
    cand = np.flatnonzero(v >= v[np.argpartition(v, -64)[-64:]].min())
    order = cand[np.lexsort((cand, -v[cand]))][:64]
    return order, v[order]


def parity(M, host, name: str, ref: str, start: int) -> dict:
    """Teacher-forced from position 0 through the reference's ids (the traced token, head mode both), the logits from
    `start` on against llama.cpp's: top-1 agreement and KL(llama.cpp || card), as g4_token.py; at every position the
    card's top 64 against the logits' (top_ref). The top 64 per position kept in "tops" (for ids_replay)."""
    import numpy as np
    import torch
    t_start = time.time()
    M.set_head("both")
    reset(M)
    ids = np.fromfile(os.path.join(ref, "ids.i32"), np.int32).tolist()
    ref_logits = np.fromfile(os.path.join(ref, "logits.f32"), np.float32).reshape(-1, VOCAB)
    assert len(ref_logits) == len(ids) - start
    top1, kls, tops, top_bad = [], [], [], []
    for pos, tok in enumerate(ids):
        lg = M.step_traced(host.embed(tok), host.ple(tok, ids[:pos]), pos)
        c_ids, c_vals = (a.copy() for a in M.top64())
        r_ids, r_vals = top_ref(lg)
        if not (np.array_equal(c_ids, r_ids) and np.array_equal(c_vals.view(np.int32), r_vals.view(np.int32))):
            top_bad.append(pos)
        tops.append((c_ids, c_vals))
        if pos >= start:
            r = torch.from_numpy(ref_logits[pos - start]).double()
            lp, lq = torch.log_softmax(r, -1), torch.log_softmax(lg.double(), -1)
            kls.append(float((lp.exp() * (lp - lq)).sum()))
            top1.append(int(r.argmax()) == int(lg.argmax()))
        if pos % 1000 == 999:
            print(f"  {name} parity: position {pos + 1} of {len(ids)}", flush=True)
    return dict(ref=os.path.relpath(ref, DATA), tokens=len(ids), start=start, compared=len(kls), top1=sum(top1) / len(top1),
                kl_mean=statistics.mean(kls), kl_max=max(kls), kl_p99=float(np.percentile(kls, 99)),
                kl_over_0_1_at=[start + i for i, k in enumerate(kls) if k > 0.1],
                top64_checked=len(tops), top64_wrong_at=top_bad,
                seconds=round(time.time() - t_start, 1), kl=kls, top1_each=top1), tops


def ids_replay(M, host, name: str, ref: str, tops: list) -> dict:
    """The reference again in head mode ids (no logits written): every position's top 64 (ids, value bits) and its
    greedy id against the parity's (tops)."""
    import numpy as np
    t_start = time.time()
    M.set_head("ids")
    reset(M)
    ids = np.fromfile(os.path.join(ref, "ids.i32"), np.int32).tolist()
    bad, greedy_bad = [], []
    for pos, tok in enumerate(ids):
        c_ids, c_vals = M.step_traced(host.embed(tok), host.ple(tok, ids[:pos]), pos, top=True)
        r_ids, r_vals = tops[pos]
        if not (np.array_equal(c_ids, r_ids) and np.array_equal(c_vals.view(np.int32), r_vals.view(np.int32))):
            bad.append(pos)
        if int(c_ids[0]) != int(r_ids[0]):
            greedy_bad.append(pos)
        if pos % 1000 == 999:
            print(f"  {name} ids replay: position {pos + 1} of {len(ids)}", flush=True)
    return dict(ref=os.path.relpath(ref, DATA), positions=len(ids), top64_differ_at=bad, greedy_differ_at=greedy_bad,
                seconds=round(time.time() - t_start, 1))


def run_card(skip_long: bool, check_ids: bool = False) -> dict:
    """The model built once; the timing (head mode ids), then the short and the long parity (both), then with
    check_ids their ids replays."""
    import numpy as np
    import ttnn
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
    sys.path.insert(0, HERE)
    from cpulane import Lane
    from g2_layers import Gguf
    from g3_layer import WORKER_L1
    from g4_host import Host
    from g4_model import SSD_MODEL, Cache, Model
    secs = {}
    t0 = time.time()
    lane = Lane()
    lane.load(SSD_MODEL)
    secs["lane load"] = round(time.time() - t0, 1)
    print(f"lane: experts loaded in {secs['lane load']:.0f} s", flush=True)
    host = Host(SSD_MODEL)
    g = Gguf(SSD_MODEL)
    ids = np.fromfile(os.path.join(REF_SHORT, "ids.i32"), np.int32).tolist()
    res = {}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    try:
        t0 = time.time()
        cache = Cache(dev, card_max=CARD_MAX) if CACHE else None
        if cache:
            secs["card cache"] = round(cache.seconds, 1)
        M = Model(dev, g, lane, log=lambda s: None, cache=cache, head_mode="ids")
        secs["model build"] = round(time.time() - t0, 1)
        print(f"model built in {secs['model build']:.0f} s ({len(M.segs)} programs per token)", flush=True)
        res["timing"] = timing(M, host, lane, ids)
        if CACHE:   # the card's routes per DeltaNet exchange over the timing (its warm-up and first token included)
            res["timing"]["card_share"], res["timing"]["card_share_per_layer"] = M.card_share()
        secs["timing"] = res["timing"]["seconds"]
        print(f"timing: {res['timing']['token_ms']:.2f} ms per token ({secs['timing']:.0f} s)", flush=True)
        res["head"] = dict(timing="ids", parity="both", replay="ids" if check_ids else None)
        tops = {}
        for part, ref, start in (("short", REF_SHORT, 0), ("long", REF_LONG, LONG_FROM))[:1 if skip_long else 2]:
            res[part], tops[part] = parity(M, host, part, ref, start)
            r = res[part]
            secs[f"{part} parity"] = r["seconds"]
            print(f"{part} parity: top-1 {100 * r['top1']:.1f}%, KL mean {r['kl_mean']:.4f}; the card's top 64 "
                  f"wrong at {len(r['top64_wrong_at'])} of {r['top64_checked']} positions ({r['seconds']:.0f} s)",
                  flush=True)
        for part, ref in (("short", REF_SHORT), ("long", REF_LONG))[:len(tops) if check_ids else 0]:
            r = res[f"{part}_ids"] = ids_replay(M, host, part, ref, tops[part])
            secs[f"{part} ids replay"] = r["seconds"]
            print(f"{part} ids replay: top 64 differ at {len(r['top64_differ_at'])}, greedy at "
                  f"{len(r['greedy_differ_at'])} of {r['positions']} positions ({r['seconds']:.0f} s)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    res["seconds"] = secs
    return res


def judge(res: dict, base: dict | None, floor_pct: float) -> dict:
    """PASS / FAIL per check and overall."""
    checks = {}
    for part, (t1, kl) in LIMITS.items():
        r = res.get(part)
        if r is None:
            checks[part] = dict(result="SKIPPED")
            continue
        ok = r["top1"] >= t1 and r["kl_mean"] <= kl
        checks[part] = dict(result="PASS" if ok else "FAIL", top1=r["top1"], kl_mean=r["kl_mean"],
                            top1_min=t1, kl_mean_max=kl)
        if base and part in base:
            checks[part].update(base_top1=base[part]["top1"], base_kl_mean=base[part]["kl_mean"])
    wrong = {p: res[p]["top64_wrong_at"] for p in LIMITS if p in res}
    wrong.update({f"{p}_ids": res[f"{p}_ids"]["top64_differ_at"] for p in LIMITS if f"{p}_ids" in res})
    checks["top64"] = dict(result="FAIL" if any(wrong.values()) else "PASS",
                           wrong_positions={k: len(v) for k, v in wrong.items()})
    ms = res["timing"]["token_ms"]
    if base is None:
        checks["speed"] = dict(result="NO BASELINE", token_ms=ms)
    else:
        b = base["token_ms"]
        band = max(base["spread_ms"], floor_pct / 100 * b)
        d = ms - b
        verdict = ("faster beyond noise" if d < -band else "slower beyond noise" if d > band else "within noise")
        checks["speed"] = dict(result="FAIL" if d > band else "PASS", token_ms=ms, base_ms=b, delta_ms=round(d, 3),
                               delta_pct=round(100 * d / b, 2), noise_band_ms=round(band, 3), verdict=verdict,
                               base_commit=base["commit"])
        if "card_waits_ms" in base:   # the card's waits before the exchanges alone: no host part (information)
            w = card_waits(res["timing"]["categories_ms"])
            checks["speed"].update(card_waits_ms=w, base_card_waits_ms=base["card_waits_ms"],
                                   card_waits_delta_ms=round(w - base["card_waits_ms"], 3),
                                   base_card_waits_spread_ms=base["card_waits_spread_ms"])
    results = [c["result"] for c in checks.values()]
    overall = ("FAIL" if "FAIL" in results else "PARTIAL" if any(r in ("SKIPPED", "NO BASELINE") for r in results)
               else "PASS")
    return dict(checks=checks, overall=overall)


def report(res: dict, verdict: dict, base: dict | None, base_path: str) -> None:
    t = res["timing"]
    print(f"\n== gate: {res['branch']} @ {res['commit'][:10]}{' (dirty)' if res.get('dirty') else ''}, "
          f"{res['date']} ==")
    print(f"timing: {t['token_ms']:.2f} ms per token ({t['tok_s']:.1f} tok/s; p10 {t['p10_ms']:.2f}, p90 "
          f"{t['p90_ms']:.2f}) at positions {t['p0']}-{t['p0'] + WARM + t['tokens'] - 1}, medians over {t['tokens']} "
          f"traced tokens; the categories' medians sum to {100 * t['attributed']:.1f}% of it")
    for k, v in t["categories_ms"].items():
        b = base["categories_ms"].get(k) if base else None
        print(f"  {v:8.3f} ms  {k}" + (f"   (baseline {b:.3f}, {v - b:+.3f})" if b is not None else ""))
    print(f"  card total {t['card_ms']:.3f} ms; the lane's steps per token (ms): {t['lane_steps_ms_per_token']}")
    c = verdict["checks"]
    for part in ("short", "long"):
        r = c[part]
        if r["result"] == "SKIPPED":
            print(f"{part} parity: SKIPPED")
            continue
        p = res[part]
        print(f"{part} parity: {r['result']}  top-1 {100 * r['top1']:.1f}% (>= {100 * r['top1_min']:.0f}%), KL mean "
              f"{r['kl_mean']:.4f} (<= {r['kl_mean_max']}), max {p['kl_max']:.3f}; {p['compared']} positions of "
              f"{p['tokens']}" + (f"; baseline {100 * r['base_top1']:.1f}% / {r['base_kl_mean']:.4f}"
                                  if "base_top1" in r else ""))
    print(f"top 64 (the card's against the logits'; ids replays against the parities'): {c['top64']['result']}  "
          f"positions wrong {c['top64']['wrong_positions']}")
    s = c["speed"]
    if s["result"] == "NO BASELINE":
        print(f"speed: NO BASELINE ({base_path} missing): {s['token_ms']:.2f} ms per token")
    else:
        print(f"speed: {s['result']}  {s['token_ms']:.2f} ms vs baseline {s['base_ms']:.2f} ({s['base_commit'][:10]}, "
              f"{os.path.basename(base_path)}): "
              f"{s['delta_ms']:+.3f} ms ({s['delta_pct']:+.2f}%), noise band +-{s['noise_band_ms']:.3f} ms: "
              f"{s['verdict']}")
        if "card_waits_ms" in s:
            print(f"  (information) the card's waits before the exchanges (DeltaNet, attention 1 and 2; no host part): "
                  f"{s['card_waits_ms']:.3f} ms vs {s['base_card_waits_ms']:.3f}, {s['card_waits_delta_ms']:+.3f} ms; "
                  f"their spread over the baseline's runs {s['base_card_waits_spread_ms']:.3f}")
    print("run time: " + ", ".join(f"{k} {v:.0f} s" for k, v in res["seconds"].items())
          + f"; total {res['total_s']:.0f} s")
    print(f"GATE {verdict['overall']}", flush=True)


def baseline_from(runs: list[dict], files: list[str]) -> dict:
    """The stored baseline from one or more gate runs: the mean ms per token and the runs' spread."""
    ms = [r["timing"]["token_ms"] for r in runs]
    waits = [card_waits(r["timing"]["categories_ms"]) for r in runs]
    cats = sorted({k for r in runs for k in r["timing"]["categories_ms"]})
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), branch=runs[0]["branch"], commit=runs[0]["commit"],
               runs=[dict(file=f, date=r["date"], commit=r["commit"], token_ms=r["timing"]["token_ms"])
                     for f, r in zip(files, runs)],
               token_ms=round(statistics.mean(ms), 3), spread_ms=round(max(ms) - min(ms), 3),
               card_waits_ms=round(statistics.mean(waits), 3), card_waits_spread_ms=round(max(waits) - min(waits), 3),
               categories_ms={k: round(statistics.mean(r["timing"]["categories_ms"].get(k, 0.0) for r in runs), 3)
                              for k in cats})
    for part in ("short", "long"):
        got = [r[part] for r in runs if r.get(part)]
        if got:
            out[part] = dict(top1=statistics.mean(p["top1"] for p in got),
                             kl_mean=statistics.mean(p["kl_mean"] for p in got),
                             identical_across_runs=all(p["kl"] == got[0]["kl"] for p in got))
    if len({r["commit"] for r in runs}) > 1:
        print(f"warning: the runs come from different commits: {sorted({r['commit'] for r in runs})}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--skip-long", action="store_true", help="skip the long parity (the result is then PARTIAL)")
    ap.add_argument("--check-ids", action="store_true",
                    help="replay the references in head mode ids, every position's top 64 against the parity's")
    ap.add_argument("--tag", help="write results/gate_<branch>_<tag>.json")
    ap.add_argument("--write-baseline", action="store_true", help="write $TT_MOEOFFLOAD_DATA/gate_baseline.json")
    ap.add_argument("--runs", nargs="+", help="with --write-baseline: earlier runs' outputs (no card run)")
    ap.add_argument("--baseline", default=BASELINE,
                    help="the baseline to compare with: gate_baseline.json (default) or a run's output, e.g. the\n"
                         "baseline branch's gate run of the same sitting (its spread then 0: the band is the floor)")
    ap.add_argument("--noise-floor-pct", type=float, default=1.0,
                    help="the noise band's floor, %% of the baseline's ms per token (default 1.0)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.runs:
        if not a.write_baseline:
            sys.exit("gate: --runs goes with --write-baseline")
        runs = [json.load(open(f)) for f in a.runs]
        b = baseline_from(runs, [os.path.basename(f) for f in a.runs])
        with open(BASELINE, "w") as f:
            json.dump(b, f, indent=1)
        print(f"baseline {BASELINE}: {b['token_ms']:.3f} ms per token, spread {b['spread_ms']:.3f} over "
              f"{len(runs)} runs ({', '.join(str(r['token_ms']) for r in b['runs'])})")
        return
    require_dma_fq()
    t0 = time.time()
    res = dict(date=time.strftime("%Y-%m-%d %H:%M"), **git_info(HERE), card=CACHE, card_max=CARD_MAX if CACHE else 0)
    res.update(run_card(a.skip_long, a.check_ids))
    res["total_s"] = round(time.time() - t0, 1)
    base = None
    if os.path.exists(a.baseline) and not a.write_baseline:
        base = json.load(open(a.baseline))
        if "timing" in base:                               # a run's output
            base = baseline_from([base], [os.path.basename(a.baseline)])
    verdict = judge(res, base, a.noise_floor_pct)
    res.update(verdict)
    report(res, verdict, base, a.baseline)
    branch = res["branch"].replace("/", "_")
    name = f"gate_{branch}" + (f"_{a.tag}" if a.tag else "") + ".json"
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")
    first = base is None and a.baseline == BASELINE and verdict["overall"] != "FAIL"   # a host's first passing run
    if a.write_baseline or first:
        b = baseline_from([res], [name])
        with open(BASELINE, "w") as f:
            json.dump(b, f, indent=1)
        print(f"wrote {BASELINE}" + (" (this host's first: later runs compare their speed with it)" if first else "")
              + " (one run: spread 0, the noise band is the floor)")
    sys.exit(0 if verdict["overall"] in ("PASS", "PARTIAL") else 1)


if __name__ == "__main__":
    main()
