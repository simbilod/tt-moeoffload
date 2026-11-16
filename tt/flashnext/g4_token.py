"""G4 step 2b: the decode token loop on the card (g4_model.Model), teacher-forced through llama.cpp's own token ids
for a short prompt ($TT_MOEOFFLOAD_DATA/g4_ref, made by tools/llama's dump-logits on llama.cpp: every position's logits and
per-layer tensors), the routed experts from the CPU lane over the host link.

Checked per position: X after every segment (a DeltaNet run, an attention layer) against llama.cpp's l_last of that
segment's last layer (the residual streams, [4 x 2560]); with all 48 layers, the logits: top-1 agreement and
KL(llama.cpp || card). Two correct engines of a near-tie-routed MoE differ by KL ~0.02, so the bar is
relative. G4_LAYERS=n stops after n layers (no head): the bisect mode.

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((120 << 30)) --engine tt/flashnext/g4_token.py
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import statistics
import sys
import time

import numpy as np
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from cpulane import Lane  # noqa: E402
from g2_layers import Gguf  # noqa: E402
from g3_layer import WORKER_L1  # noqa: E402
from g4_host import Host  # noqa: E402
from g4_model import D, HC, SSD_MODEL, Cache, Model  # noqa: E402

OUT = os.path.join(HERE, "results")
REF = os.path.expanduser(os.environ.get("G4_REF", os.path.join(DATA, "g4_ref")))
FROM = int(os.environ.get("G4_FROM", 0))   # compare the logits from this position on (the reference holds only
                                           # those: --dump-from); no per-segment X checks (each costs a sync)
N_LAYERS = int(os.environ.get("G4_LAYERS", 48))
N_TOKENS = int(os.environ.get("G4_TOKENS", 0)) or None
CHECK = os.environ.get("G4_CHECK", "1") == "1"     # G4_CHECK=0: the timing alone
CACHE = os.environ.get("G4_CACHE") == "1"          # the card computes its cached share of the experts (g5_cache.py)
CARD_MAX = int(os.environ.get("G4_CARD_MAX", 4))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32).tolist()[:N_TOKENS]
    n = len(ids)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layers": N_LAYERS, "tokens": n, "segments": {}, "card": CACHE,
           "card_max": CARD_MAX}
    lane = Lane()
    t0 = time.time()
    lane.load(SSD_MODEL, layers=range(N_LAYERS))
    print(f"lane: experts of {N_LAYERS} layers in {time.time() - t0:.0f} s", flush=True)
    host = Host(SSD_MODEL)
    l_last = {} if FROM else {
        il: np.fromfile(os.path.join(REF, "tensors", f"l_last-{il}.f32"), np.float32).reshape(-1, HC * D)
        for il in range(N_LAYERS)}
    g = Gguf(SSD_MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    try:
        t0 = time.time()
        cache = Cache(dev, layers=range(N_LAYERS), card_max=CARD_MAX) if CACHE else None
        if cache:
            print(f"the card's expert cache: {len(cache.tensors)} layers in {cache.seconds:.0f} s", flush=True)
        M = Model(dev, g, lane, N_LAYERS, log=lambda s: print(s, flush=True), cache=cache)
        res["build_s"] = time.time() - t0
        print(f"model built in {res['build_s']:.0f} s ({len(M.segs)} programs per token)", flush=True)
        errs = {f"{k} {w}": [] for k, w in M.segs if k != "ple"}
        pos_now = [0]

        def after(kind, what):
            if kind == "ple":
                return
            last = what[-1] if kind == "run" else what
            x = M.x().flatten()
            ref = torch.from_numpy(l_last[last][pos_now[0]]).double()
            errs[f"{kind} {what}"].append(float((x - ref).abs().max() / ref.abs().max()))

        ref_logits = (np.fromfile(os.path.join(REF, "logits.f32"), np.float32).reshape(-1, 248320)
                      if M.head is not None else None)
        top1, kls, times, card_logits, compared, t_cmp = [], [], [], [], [], []
        for pos, tok in enumerate(ids if CHECK else ids[:1]):
            pos_now[0] = pos
            t0 = time.perf_counter()
            x0, emb = host.embed(tok), host.ple(tok, ids[:pos])
            logits = M.step(x0, emb, pos, after=None if FROM else after)
            if FROM and pos >= FROM:
                t_cmp.append((time.perf_counter() - t0) * 1e3)
            if logits is not None and pos >= FROM:
                compared.append(pos)
                card_logits.append(logits.float().numpy())
                ref = torch.from_numpy(ref_logits[pos - FROM]).double()
                lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(logits.double(), -1)
                kls.append(float((lp.exp() * (lp - lq)).sum()))
                top1.append(int(ref.argmax()) == int(logits.argmax()))
            if FROM and (pos % 500 == 0 or (pos >= FROM and pos % 32 == 0) or pos == n - 1):
                print(f"pos {pos}" + (f": top-1 {sum(top1)}/{len(top1)}, KL {kls[-1]:.4f}" if kls and pos >= FROM
                                      else ""), flush=True)
            elif not FROM and (pos % 10 == 0 or pos == n - 1):
                worst = max(errs.items(), key=lambda kv: kv[1][-1])
                print(f"pos {pos}: X after the last segment within {list(errs.values())[-1][-1]:.1e} of llama.cpp "
                      f"(worst segment {worst[0]}: {worst[1][-1]:.1e})"
                      + (f"; top-1 {sum(top1)}/{len(top1)}, KL {kls[-1]:.4f}" if kls else ""), flush=True)
        if CACHE:   # the card's share of the routes per DeltaNet exchange (the checked positions)
            res["card_share"], res["card_share_per_layer"] = M.card_share()
            print(f"the card's share: {res['card_share']:.2f} of 10 routes per DeltaNet exchange", flush=True)
        for k, v in errs.items():
            if v:
                res["segments"][k] = dict(max=max(v), median=statistics.median(v), first=v[0], last=v[-1])
        res["kl"], res["top1_each"], res["from"] = kls, top1, FROM
        if t_cmp:
            res["token_ms_compared"] = statistics.median(t_cmp)   # embedding + step + the logits' read-back
            res["kl_over_0.1_at"] = [p for p, k in zip(compared, kls) if k > 0.1]
        if card_logits and CHECK:   # for comparisons against llama.cpp's other paths (logits_decode.f32 ...)
            np.stack(card_logits).astype(np.float32).tofile(os.path.join(REF, "logits_card.f32"))
        if kls and CHECK:
            res.update(top1=sum(top1) / len(top1), kl_mean=statistics.mean(kls), kl_max=max(kls),
                       kl_p99=float(np.percentile(kls, 99)))
            print(f"logits vs llama.cpp over {len(top1)} positions: top-1 {100 * res['top1']:.1f}%, KL mean "
                  f"{res['kl_mean']:.4f}, max {res['kl_max']:.4f}"
                  + (f"; KL > 0.1 at {res['kl_over_0.1_at']}; {res['token_ms_compared']:.2f} ms per token there "
                     f"(with the logits' read-back)" if FROM else ""), flush=True)
        # the steady state without the checks' syncs
        if M.head is not None:
            prof, reps = {}, 10
            for pos in range(n, n + reps + 2):
                if pos == n + 2:
                    prof.clear()
                    M.link.t[:] = 0
                    lane.stats()
                t0 = time.perf_counter()
                x0, emb = host.embed(ids[-1]), host.ple(ids[-1], ids)
                if pos >= n + 2:
                    prof["host embedding + PLE gather"] = prof.get("host embedding + PLE gather", 0.0) + \
                        time.perf_counter() - t0
                M.step(x0, emb, pos, prof=prof if pos >= n + 2 else None)
                times.append((time.perf_counter() - t0) * 1e3)
            res["token_ms"] = statistics.median(times[2:])
            res["stages_ms"] = {k: v / reps * 1e3 for k, v in prof.items()}
            res["link_ms"] = dict(zip(("wait for pages", "lane", "write replies"), (M.link.t / reps * 1e3).tolist()))
            res["lane_steps_ms"] = (lane.stats() / reps / 1e6).round(3).tolist()
            print(f"decode step without the checks: {res['token_ms']:.2f} ms per token "
                  f"({1e3 / res['token_ms']:.1f} tok/s)", flush=True)
            print("  stages (ms per token): " + ", ".join(f"{k} {v:.2f}" for k, v in res["stages_ms"].items()),
                  flush=True)
            print("  inside them, the link: " + ", ".join(f"{k} {v:.2f}" for k, v in res["link_ms"].items()) +
                  f"; the lane's steps (quantize hn, gate|up, quantize h, down) {res['lane_steps_ms']}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    name = ("g4_token.json" if CHECK else "g4_token_timing.json") if N_LAYERS == 48 else f"g4_token_{N_LAYERS}l.json"
    if FROM:
        name = "g4_token_long.json"
    if CACHE:
        name = name.replace(".json", "_card.json")
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
