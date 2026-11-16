"""The draft side of speculation on the host (no card). Strata's MTP draft layer in float64 (tt/spec/mtp.py),
checked against llama.cpp's tensors, its f32/bf16 error, and an acceptance replay of Strata's drafters (tt/spec/
strata_spec.py: Strata's own suffix drafter and draft policy through ctypes) with the target's greedy tokens as the
verifier: E = tokens committed per verify pass.

Data ($TT_MOEOFFLOAD_DATA/w4_draft/census/<id>/, tools/llama/gen_greedy.cpp: llama.cpp's own greedy continuations of the census
prompts, 256 tokens each, with the target's final multi-stream residual "l_last-47" at every position and its top-8):

  python tt/flashnext/w4_draft.py check            # float64 checks (g4_ref) + f32/bf16 vs float64 + CPU step costs
  python tt/flashnext/w4_draft.py drafts HEAD      # MTP chains (4 steps, f32) at every decode position; HEAD = full,
                                                   #   en (40,525), fr (46,211), cyr (58,963): Strata's draft vocabs;
                                                   #   full_bf16x: the whole head, the BF16 experts (a variant)
  python tt/flashnext/w4_draft.py replay           # E per pass for the policies -> results/w4_draft.json

Run in the tt-metal container without the card (tt/env/ttenv.sh; numpy + torch), 8 threads.

The replay. For a sequence x (prompt + greedy output), a pass whose row 0 is x[r] verifies T rows [x[r], d_0..d_{T-2}];
draft d_j counts if d_0..d_j equal x[r+1..r+1+j]; the pass commits a + 1 tokens (a = the drafts accepted, + the bonus)
and the next pass starts at r + a + 1. The MTP chain of a pass at r starts at cell r - 1 (R_{r-1} with x[r]; mtp.cpp's
catch-up row a), drafts at cells r, r + 1, ...; the verified cells < r are teacher-forced (x's own tokens and residuals),
so a chain depends only on r and is computed once for every r. The first pass of a request is one row (Strata's
first_window).
"""
from __future__ import annotations

import json
import os
DATA_ROOT = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tt", "spec"))
import mtp as M  # noqa: E402
import strata_spec as S  # noqa: E402

RES = os.path.join(HERE, "results")
CACHE = os.path.join(DATA_ROOT, "w4_draft")
DATA = os.path.join(CACHE, "census")
G4REF = os.path.join(DATA_ROOT, "g4_ref")
XDATA = os.path.join(REPO, "third_party", "strata", "data")
STEPS = 4                                      # MTP draft steps computed per chain (windows up to 5 rows)
THREADS = int(os.environ.get("W4_THREADS", 8))
W4_COST = os.path.join(RES, "w4_cost.json")


def heads() -> dict:
    """The draft heads: the whole vocabulary (Strata here: rt/ has no draft_vocab.bin) and Strata's shipped subsets
    (tools/draft_vocab.py): en = the base the others extend (the first 40,525 ids of draft_vocab_fr.bin, which both
    files share; mtp.hpp: data/draft_vocab_en.bin, '40K tokens'), fr (46,211, subset), cyrillic (58,963)."""
    fr = np.fromfile(os.path.join(XDATA, "draft_vocab_fr.bin"), dtype=np.int32)
    cyr = np.fromfile(os.path.join(XDATA, "draft_vocab_cyrillic.bin"), dtype=np.int32)
    n = int(np.argmax(fr[:len(cyr)] != cyr[:len(fr)]))
    return {"full": None, "en": fr[:n].copy(), "fr": fr, "cyr": cyr}


def sequences():
    """The census continuations: [(id, meta, ids, R memmap [rows, 10240])]."""
    out = []
    for p in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, p)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        m = json.load(open(os.path.join(d, "meta.json")))
        ids = np.fromfile(os.path.join(d, "ids.i32"), dtype=np.int32)
        R = np.memmap(os.path.join(d, "R.f32"), dtype=np.float32, mode="r").reshape(-1, 10240)
        out.append((p, m, ids, R))
    return out


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


# ---------------------------------------------------------------- check

def chain(m, R, ids, r0, starts, steps=STEPS, hnorm_stream=False):
    """MTP chains from `starts` (absolute cells): -> drafts [n, steps], probs [n, steps], step-0 logits."""
    hi = int(max(starts)) + 1
    Rt = torch.from_numpy(np.ascontiguousarray(R[:hi - r0])).to(m.dt)
    cache, (R0, lg0) = M.teacher_forced(m, Rt, ids[r0 + 1:hi + 1], r0, rest_rows=np.asarray(starts) - r0,
                                        hnorm_stream=hnorm_stream)
    d0, p0 = m.pick(lg0)
    ds, ps = [d0], [p0]
    for d, p in M.chain_steps(m, cache, np.asarray(starts), R0, d0, steps - 1, hnorm_stream=hnorm_stream):
        ds.append(d)
        ps.append(p)
    return torch.stack(ds, 1).numpy(), torch.stack(ps, 1).numpy(), lg0


def check() -> dict:
    torch.set_num_threads(THREADS)
    res = {"date": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    # -- the weights against their sources
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(os.path.join(os.path.dirname(M.RT), "mtp-q2_0.gguf"))
    tt = {t.name: t for t in r.tensors}
    gu_t, dn_t = tt["mtp.layers.0.mlp.experts.gate_up_proj"], tt["mtp.layers.0.mlp.experts.down_proj"]
    sample = [0, 1, 255, 511]
    g, u, d = M.decode_experts(which=sample)
    dif = 0.0
    for k, e in enumerate(sample):
        gu = dequantize(np.asarray(gu_t.data[e]), gu_t.tensor_type).reshape(1280, 2560)
        dn = dequantize(np.asarray(dn_t.data[e]), dn_t.tensor_type).reshape(2560, 640)
        dif = max(dif, float(np.abs(gu[:640] - g[k]).max()), float(np.abs(gu[640:] - u[k]).max()),
                  float(np.abs(dn - d[k]).max()))
    res["experts_bin_vs_gguf_q2_0"] = {"experts": sample, "max_abs_diff": dif,
                                       "what": "rt/experts.bin decoded (tt/spec/mtp.py) vs gguf-py's Q2_0 dequantize "
                                               "of mtp-q2_0.gguf (the file mtp_rt.py relaid)"}
    ids = np.fromfile(os.path.join(G4REF, "ids.i32"), dtype=np.int32)
    n = len(ids)
    E = M.gguf_tensor("token_embd.weight")
    ie = np.fromfile(os.path.join(G4REF, "tensors", "model.input_embed.f32"), dtype=np.float32).reshape(n, -1)
    res["embedding_vs_llama"] = {"max_abs_diff": float(np.abs(E[ids] - ie).max()),
                                 "what": "token_embd.weight rows (gguf-py) vs g4_ref tensors/model.input_embed.f32"}
    # -- float64: the target's own final mixer through Mtp.hc_read (the same gr_read the draft layer runs three
    #    times) and the LM head, against llama.cpp's result_norm and logits
    m = M.Mtp("float64", experts="lazy", threads=THREADS)
    for k, nm in (("hc_norm.weight", "output_hc_norm.weight"), ("input_mix_weight_down.weight", "output_hc_down.weight"),
                  ("input_mix_weight_up.weight", "output_hc_up.weight")):
        a = torch.from_numpy(np.array(M.gguf_tensor(nm))).double()
        (m.wn if a.ndim == 1 else m.w)["T." + k] = a
    L47 = np.fromfile(os.path.join(G4REF, "tensors", "l_last-47.f32"), dtype=np.float32).reshape(n, 10240)
    rn = np.fromfile(os.path.join(G4REF, "tensors", "result_norm.f32"), dtype=np.float32).reshape(n, 2560)
    logits = np.memmap(os.path.join(G4REF, "logits.f32"), dtype=np.float32, mode="r").reshape(n, -1)
    _, mixed, _ = m.hc_read(torch.from_numpy(L47).double().view(n, 4, 2560), "T.", False)
    lg = m.mm(mixed, m.head).numpy()
    la = np.asarray(logits)
    e = mixed.numpy() - rn
    res["final_mixer_vs_llama"] = {
        "positions": n, "result_norm_rel_rms": float(np.sqrt((e ** 2).mean() / (rn ** 2).mean())),
        "result_norm_max_abs": float(np.abs(e).max()),
        "logits_max_abs": float(np.abs(lg - la).max()), "logits_rel_max": rel(lg, la),
        "argmax_agree": float((lg.argmax(1) == la.argmax(1)).mean()),
        "what": "float64 Mtp.hc_read + output.weight on g4_ref's l_last-47 (llama.cpp, GPU bf16 path) vs its "
                "result_norm and logits: the hyper-connection read and head as the draft layer uses them"}
    # -- float64 MTP step 0, teacher-forced over g4_ref's prompt (cell i: R_i, ids[i+1]; predicts ids[i+2])
    d64, p64, _ = chain(m, L47, ids, 0, np.arange(n - 2), steps=1)
    tgt = la.argmax(1)
    res["mtp_step0_g4_ref"] = {
        "positions": n - 2, "top1_eq_target_argmax": float((d64[:, 0] == tgt[1:n - 1]).mean()),
        "top1_eq_text": float((d64[:, 0] == ids[2:n]).mean()),
        "target_argmax_eq_text": float((tgt[1:n - 1] == ids[2:n]).mean()),
        "what": "teacher-forced over llama.cpp's prompt text (92 tokens, prose + code): the draft layer's top-1 at "
                "cell i vs llama.cpp's argmax at position i + 1 (logits.f32), the token both predict"}
    dps, _, _ = chain(m, L47, ids, 0, np.arange(n - 2), steps=1, hnorm_stream=True)
    res["mtp_step0_g4_ref"]["hnorm_per_stream_top1_eq_target_argmax"] = float((dps[:, 0] == tgt[1:n - 1]).mean())
    print(json.dumps(res, indent=1), flush=True)
    # -- float64 chains on the census continuations (the replay's data) vs the greedy tokens; f32 and bf16 against it
    seqs = sequences()
    pick = [s for s in seqs if s[0] in ("A.code.0", "A.chat.18", "A.math.12", "A.zh.40")] or seqs[:2]
    per = {"positions": 0, "f32": {}, "bf16": {}}
    m32 = M.Mtp("float32", experts="lazy", threads=THREADS)
    mbf = M.Mtp("bf16", experts="lazy", threads=THREADS)
    stats = {k: {"logit_rel_max": 0.0, "logit_rel_rms": [], "draft_eq": np.zeros(STEPS), "prob_abs_max": 0.0}
             for k in ("f32", "bf16")}
    acc = np.zeros(STEPS)
    nn = 0
    for name, meta, sids, R in pick:
        q = np.arange(meta["n_prompt"] - 1, meta["n_prompt"] - 1 + 24)          # the first 24 decode positions
        d64, p64, l64 = chain(m, R, sids, meta["r0"], q)
        tg = np.stack([sids[q + 2 + j] for j in range(STEPS)], 1)
        ok = np.cumprod(d64 == tg, axis=1)
        acc += ok.sum(0)
        nn += len(q)
        for k, mm in (("f32", m32), ("bf16", mbf)):
            dk, pk, lk = chain(mm, R, sids, meta["r0"], q)
            a, b = lk.double().numpy(), l64.numpy()
            st = stats[k]
            st["logit_rel_max"] = max(st["logit_rel_max"], rel(a, b))
            st["logit_rel_rms"].append(float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())))
            st["draft_eq"] += (dk == d64).sum(0)
            st["prob_abs_max"] = max(st["prob_abs_max"], float(np.abs(pk - p64).max()))
    for k in stats:
        st = stats[k]
        per[k] = {"step0_logits_rel_max": st["logit_rel_max"], "step0_logits_rel_rms": float(np.mean(st["logit_rel_rms"])),
                  "drafts_equal_float64_by_depth": (st["draft_eq"] / nn).round(4).tolist(),
                  "draft_prob_abs_max": st["prob_abs_max"]}
    per["positions"] = nn
    per["prompts"] = [s[0] for s in pick]
    per["float64_cumulative_accept_by_depth"] = (acc / nn).round(4).tolist()
    per["what"] = ("float64 chains of 4 steps at the first 24 decode positions of 4 census continuations vs the greedy "
                   "tokens; f32 = float32 activations, bf16 = weights and every GEMV input rounded to bfloat16 "
                   "(f32 accumulation), each vs float64 on the same positions; logits = step 0's (whole vocabulary)")
    res["precision"] = per
    # -- CPU cost of one draft step (f32, one row, 8 threads; context only: the card is priced elsewhere)
    name, meta, sids, R = pick[0]
    q = np.array([meta["n_prompt"] + 100])
    Rt = torch.from_numpy(np.ascontiguousarray(R[:q[0] + 1 - meta["r0"]]))
    cpu = {}
    for hname, vocab in (("full", None), ("fr", heads()["fr"])):
        mm = M.Mtp("float32", experts="all", vocab=vocab, threads=THREADS)
        cache, (R0, lg0) = M.teacher_forced(mm, Rt, sids[meta["r0"] + 1:q[0] + 2], meta["r0"], rest_rows=q - meta["r0"])
        d0, _ = mm.pick(lg0)
        M.chain_steps(mm, cache, q, R0, d0, 1)
        tm = []
        for _ in range(5):
            t1 = time.time()
            M.chain_steps(mm, cache, q, R0, d0, 3)
            tm.append((time.time() - t1) / 3)
        cpu[hname] = round(min(tm) * 1e3, 2)
        del mm
    res["cpu_ms_per_draft_step_f32_8_threads"] = cpu
    res["cpu_note"] = ("one row, K/V of %d verified cells cached; torch f32 on 8 cores, experts and head "
                       "resident as f32; Strata's CUDA step is not measured here; context only" % (
                           q[0] - meta["r0"] + 1))
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- drafts

def drafts(head: str):
    """MTP chains at every pass start of every census continuation -> CACHE/drafts_<head>/<id>.npz (q0, d, p);
    sequences already written are skipped."""
    torch.set_num_threads(THREADS)
    out = os.path.join(CACHE, f"drafts_{head}")
    os.makedirs(out, exist_ok=True)
    todo = [s for s in sequences() if not os.path.exists(os.path.join(out, s[0] + ".npz"))]
    if not todo:
        return
    src = "bf16" if head.endswith("_bf16x") else "q2_0"         # full_bf16x: the checkpoint's BF16 experts (variant)
    m = M.Mtp("float32", experts="all", vocab=heads()[head.replace("_bf16x", "")], threads=THREADS, expert_src=src)
    t0 = time.time()
    for name, meta, ids, R in todo:
        q = np.arange(meta["n_prompt"] - 1, len(ids) - 1)       # every pass start r = q + 1 with x[r] known
        d, p, _ = chain(m, R, ids, meta["r0"], q)
        np.savez(os.path.join(out, name + ".npz"), q0=np.array(q[0]), d=d.astype(np.int32), p=p.astype(np.float32))
        print(f"{head} {name}: {len(q)} chains, {time.time() - t0:.0f} s", flush=True)


def load_drafts(head: str) -> dict:
    out = {}
    d = os.path.join(CACHE, f"drafts_{head}")
    for f in sorted(os.listdir(d)):
        z = np.load(os.path.join(d, f))
        out[f[:-4]] = (int(z["q0"]), z["d"], z["p"])
    return out


# ---------------------------------------------------------------- replay

def costs() -> dict:
    """Verify-pass times T(M) (results/w4_cost.json scenarios), ms by rows M = 1..8."""
    sc = json.load(open(W4_COST))["scenarios"]
    sub = sc["design + card experts balanced, W1 floors (dense x0.65)"]["T_ms"]
    sfx = sc["design + card experts balanced, W1 floors, suffix drafts"]["T_ms"]
    d_full = [a - b for a, b in zip(sc["design, MTP full head"]["T_ms"],
                                    sc["design (owner arena, chunked delta rule, C++ selection, MTP subset head), "
                                       "no card experts"]["T_ms"])]
    return {"mtp_subset": sub, "suffix": sfx, "mtp_full": [a + b for a, b in zip(sub, d_full)],
            "source": "results/w4_cost.json scenarios: 'design + card experts balanced, W1 floors (dense x0.65)' "
                      "(MTP subset head, M - 1 steps per pass), its 'suffix drafts' twin (no MTP cost), and the "
                      "former + ('design, MTP full head' - 'design ... no card experts') for the full head (estimate)"}


def run(policy, seqs, dr, cost, spec=4, min_p=0.30):
    """One policy over every sequence -> totals. policy(state) picks the rows of a pass."""
    tot = dict(passes=0, tokens=0, rows=0, ms=0.0, drafted=0, accepted=0, lookup_passes=0)
    for name, meta, ids, _ in seqs:
        n, L = meta["n_prompt"], len(ids)
        q0, D, P = dr[name] if dr is not None else (n - 1, None, None)
        st = policy["init"](ids[:n], spec, min_p)
        r, first = n - 1, True
        while r + 8 + 1 < L:                      # every row of the widest window has a known target
            drafts = D[r - 1 - q0] if D is not None and r - 1 >= q0 else None
            probs = P[r - 1 - q0] if P is not None and r - 1 >= q0 else None
            rows, info = policy["window"](st, ids[r], drafts, probs, first)
            a = 0
            while a < len(rows) - 1 and rows[a + 1] == ids[r + 1 + a]:
                a += 1
            T = len(rows)
            ms = cost[T - 1]
            policy["commit"](st, list(ids[r + 1:r + 2 + a]), info, a, ms)
            tot["passes"] += 1
            tot["tokens"] += a + 1
            tot["rows"] += T
            tot["ms"] += ms
            tot["drafted"] += T - 1
            tot["accepted"] += a
            tot["lookup_passes"] += int(bool(info.get("from_sfx")))
            r += a + 1
            first = False
    p = tot["passes"]
    return dict(E=round(tot["tokens"] / p, 4), passes=p, tokens=tot["tokens"], mean_rows=round(tot["rows"] / p, 3),
                accept_per_draft=round(tot["accepted"] / max(1, tot["drafted"]), 4),
                tok_s_card_model=round(1e3 * tot["tokens"] / tot["ms"], 1), lookup_passes=tot["lookup_passes"])


def fixed_mtp(M_rows, use_min_p=False):
    """MTP only: every pass M_rows rows (M_rows - 1 chain drafts), or Strata's --spec-min-p gate (T = 1 + the leading
    drafts with probability >= min_p, at most M_rows)."""
    def init(prompt, spec, min_p):
        return {"min_p": min_p}

    def window(st, x, drafts, probs, first):
        if first or drafts is None:
            return [x], {}
        T = M_rows
        if use_min_p:
            T = 1
            while T < M_rows and probs[T - 1] >= st["min_p"]:
                T += 1
        return [x] + [int(v) for v in drafts[:T - 1]], {}
    return dict(init=init, window=window, commit=lambda st, out, info, a, ms: None)


def suffix_greedy(M_rows):
    """The suffix drafter alone, its proposal always taken (up to M_rows - 1 tokens): Strata's SuffixDrafter(3, 64)."""
    def init(prompt, spec, min_p):
        d = S.SuffixDrafter(3, 64, 1 << 20)
        d.append(list(map(int, prompt)))
        return {"sfx": d}

    def window(st, x, drafts, probs, first):
        return [x] + st["sfx"].propose(M_rows - 1), {}

    def commit(st, out, info, a, ms):
        st["sfx"].append(list(map(int, out)))
    return dict(init=init, window=window, commit=commit)


def strata(use_mtp=True, use_sfx=True, mtp_rows=None):
    """Strata's serving loop (tt/spec/strata_spec.StrataWindow): --spec 4 (MTP windows <= 4 rows, lookup <= 6),
    --suffix-draft 3, --spec-min-p 0.30, DraftPolicy fed each pass's time from the card cost model. One policy for
    the whole run (Strata's is per process: "learned over the whole process")."""
    spec0 = mtp_rows or 4
    shared = S.DraftPolicy(min(spec0 + 2, 8))

    def init(prompt, spec, min_p):
        return S.StrataWindow(spec=mtp_rows or spec, prompt=list(map(int, prompt)), spec_min_p=min_p, use_mtp=use_mtp,
                              use_sfx=use_sfx, policy=shared)

    def window(w, x, drafts, probs, first):
        w.first = first
        dr = [int(v) for v in drafts] if drafts is not None else [-1] * STEPS
        pr = [float(v) for v in probs] if probs is not None else [0.0] * STEPS
        return w.window(int(x), dr, pr)

    def commit(w, out, info, a, ms):
        w.commit(list(map(int, out)), info, a, ms)
    return dict(init=init, window=window, commit=commit)


def depth_accept(seqs, dr):
    """MTP chains at every decode position: P(d_0..d_j all right) and the conditional rates."""
    hit, n = np.zeros(STEPS), 0
    for name, meta, ids, _ in seqs:
        q0, D, _ = dr[name]
        L = len(ids)
        for i in range(len(D)):
            q = q0 + i
            if q + 2 + STEPS > L:
                break
            hit += np.cumprod(D[i] == ids[q + 2:q + 2 + STEPS])
            n += 1
    cum = hit / n
    cond = [cum[0]] + [cum[j] / cum[j - 1] for j in range(1, STEPS)]
    return dict(positions=n, cumulative=np.round(cum, 4).tolist(), conditional=np.round(cond, 4).tolist())


STRATA_LOG = os.environ.get("TT_MOEOFFLOAD_STRATA_LOG", "")   # a Strata engine log to read its own E from (none: skipped)


def strata_log_E(path: str = STRATA_LOG) -> dict | None:
    """Strata's own E, from its serve log (TT_MOEOFFLOAD_STRATA_LOG; runs with --spec 4, MTP + suffix): per request
    'N generated ... drafts accepted A of O'; a pass commits accepted drafts + 1, so passes = N - A and E = N / (N - A).
    Requests of >= 64 tokens with speculation on (O > N / 5; the diag-strata-spec0 runs offer ~nothing)."""
    import re
    if not os.path.exists(path):
        return None
    pat = re.compile(r"prompt (\d+) tokens .*?, (\d+) generated in .*?drafts accepted (\d+) of (\d+)")
    rows = [tuple(map(int, m.groups())) for m in map(pat.search, open(path, errors="replace")) if m]
    rows = [r for r in rows if r[1] >= 64 and r[3] > r[1] / 5]
    s1 = [r for r in rows if 6000 <= r[0] <= 7000 and r[1] == 512]

    def e(rs):
        g, a = sum(r[1] for r in rs), sum(r[2] for r in rs)
        return round(g / (g - a), 3) if rs else None
    return {"requests": len(rows), "E": e(rows), "S1_8k_requests": len(s1), "S1_8k_E": e(s1),
            "accept_per_draft": round(sum(r[2] for r in rows) / sum(r[3] for r in rows), 3) if rows else None,
            "source": path + " (Strata's serve lines; E = generated / (generated - accepted))"}


def replay() -> dict:
    seqs = sequences()
    c = costs()
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "cost_model": c,
           "sequences": len(seqs), "decode_tokens": int(sum(m["n_gen"] for _, m, _, _ in seqs)),
           "verifier": "llama.cpp's own greedy continuations (tools/llama/gen_greedy.cpp on third_party/llama.cpp, "
                       "UD-IQ4_XS, -ngl 99 -ot exps=CPU: GPU dense, CPU experts; decode path one token at a time) of "
                       "the census prompts ($TT_MOEOFFLOAD_DATA/census/qwen3.8-flash-next/prompts.jsonl), 256 tokens each or to "
                       "EOG; $TT_MOEOFFLOAD_DATA/w4_draft/census"}
    dr = {h: load_drafts(h) for h in ("full", "en", "fr", "cyr", "full_bf16x") if os.path.isdir(os.path.join(CACHE, f"drafts_{h}"))}
    missing = sorted({s[0] for s in seqs for d in dr.values() if s[0] not in d})
    if missing:
        print("no drafts yet for", missing, "- replaying the rest")
        seqs = [s for s in seqs if s[0] not in missing]
        res["sequences"], res["missing_drafts"] = len(seqs), missing
    res["mtp_depth_accept"] = {h: depth_accept(seqs, d) for h, d in dr.items()}
    pol = {}
    for Mr in range(2, 9):
        pol[f"a_suffix_only_greedy_M{Mr}"] = run(suffix_greedy(Mr), seqs, None, c["suffix"])
    pol["a_suffix_only_strata_policy_spec4"] = run(strata(use_mtp=False), seqs, None, c["suffix"])
    for h, d in dr.items():
        cc = c["mtp_full"] if h.startswith("full") else c["mtp_subset"]
        tag = {"full": "b_mtp_full", "full_bf16x": "b_mtp_full_bf16_experts"}.get(h, f"c_mtp_subset_{h}")
        for Mr in range(2, STEPS + 2):
            pol[f"{tag}_M{Mr}"] = run(fixed_mtp(Mr), seqs, d, cc)
        pol[f"{tag}_min_p0.30_M4"] = run(fixed_mtp(4, True), seqs, d, cc)
        pol[f"d_strata_spec4_{h}"] = run(strata(), seqs, d, cc)
    if "full" in dr:
        for sp in (3, 5):
            pol[f"d_strata_spec{sp}_full"] = run(strata(mtp_rows=sp), seqs, dr["full"], c["mtp_full"])
        # a verify pass whose cost does not grow with its rows: the policy then takes every lookup window that pays in E
        pol["d_strata_spec4_full_flat_cost"] = run(strata(), seqs, dr["full"], [25.0] * 8)
        pol["d_strata_spec4_full_flat_cost"]["note"] = "T(M) = 25 ms for every M: tok_s is not a card estimate"
    res["policies"] = pol
    res["strata_own_log"] = strata_log_E()
    if "full" in dr and "fr" in dr:
        P = pol
        res["summary"] = {
            "E_per_pass_spec4": {
                "a_suffix_only (proposal always taken, M = 4 rows)": P["a_suffix_only_greedy_M4"]["E"],
                "a_suffix_only (Strata's policy, no MTP)": P["a_suffix_only_strata_policy_spec4"]["E"],
                "b_mtp_full (M = 4 rows every pass)": P["b_mtp_full_M4"]["E"],
                "b_mtp_full (--spec-min-p 0.30, <= 4 rows)": P["b_mtp_full_min_p0.30_M4"]["E"],
                "c_mtp_subset_fr_46211 (M = 4 rows)": P["c_mtp_subset_fr_M4"]["E"],
                "c_mtp_subset_fr_46211 (--spec-min-p 0.30)": P["c_mtp_subset_fr_min_p0.30_M4"]["E"],
                "d_strata_combined_full_head (as Strata runs here)": P["d_strata_spec4_full"]["E"],
                "d_strata_combined_subset_fr": P["d_strata_spec4_fr"]["E"],
            },
            "strata_reference": {"plan": 2.77, "its_own_log_E": (res["strata_own_log"] or {}).get("E")},
            "subset_head_loss_E_M4": round(P["b_mtp_full_M4"]["E"] - P["c_mtp_subset_fr_M4"]["E"], 4),
            "tok_s_card_model_strata_policy": {"full head": P["d_strata_spec4_full"]["tok_s_card_model"],
                                               "fr subset head": P["d_strata_spec4_fr"]["tok_s_card_model"]},
        }
    res["mtp_inputs"] = (
        "per draft chain: the target's final multi-stream residual R (llama.cpp 'l_last-47' = 'h_nextn', 4 x 2560 f32, "
        "the residual after the last layer, before the final mixer) at the last accepted position and the token after "
        "it (the pass's bonus token); and the draft layer's own K/V over every verified cell, so each pass also runs "
        "the layer's front (embedding, fc, attention HC read, k/v) on its a + 1 accepted rows (mtp.cpp's catch-up). "
        "Source here: tools/llama/gen_greedy.cpp's dump of l_last-47 at every position (prompt rows: the last 8,192 of "
        "the 6 long B prompts; the draft layer's attention there sees only those cells, an approximation)")
    res["by_prompt_class"] = by_class(seqs, dr)
    return res


def by_class(seqs, dr):
    """E per prompt class (code, math, chat, ..., the B long contexts) for the main policies."""
    classes = {}
    for s in seqs:
        k = s[0].split(".")[1] if s[0].startswith("A.") else "B"
        classes.setdefault(k, []).append(s)
    out = {}
    c = costs()
    for k, ss in sorted(classes.items()):
        row = {"sequences": len(ss), "suffix_greedy_M4": run(suffix_greedy(4), ss, None, c["suffix"])["E"]}
        for h in ("full", "fr"):
            if h in dr:
                row[f"mtp_{h}_M4"] = run(fixed_mtp(4), ss, dr[h], c["mtp_full"])["E"]
                row[f"strata_{h}"] = run(strata(), ss, dr[h], c["mtp_full"])["E"]
        out[k] = row
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if cmd == "check":
        res = check()
        path = os.path.join(RES, "w4_draft_check.json")
    elif cmd == "drafts":
        drafts(sys.argv[2])
        return
    else:
        res = replay()
        path = os.path.join(RES, "w4_draft.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
