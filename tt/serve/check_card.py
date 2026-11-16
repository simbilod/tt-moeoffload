"""On the card: the serving engine (tt/serve/engine.py's CardModel and Engine, built once as the server
builds them: head mode ids) against the G4 references, in one process:

  1. dump: g4_ref's 92 ids teacher-forced (engine.py --dump-logits' path: each traced step, then the logits head
     launched untraced on the same X) -> the logits against g4_token.py's ($TT_MOEOFFLOAD_DATA/g4_ref/logits_card.f32) and
     llama.cpp's (logits.f32: top-1 and KL as the gate's short parity); at every position the traced step's top 64
     against the top 64 of those logits (g4_gate.top_ref: ids and value bits);
  2. generate: greedy from the same prompt through Engine.generate (the server's path: GEN in, T and DONE out). Its
     first token is the argmax of the card's last-position logits; a second request with the same prompt (the state
     reset in place) gives the same tokens; a follow-up prompt that extends the state (RESUME) gives the same tokens
     as the same prompt read from position 0 after a reset; a sampled request (S3's settings) from the top 64 gives
     the same tokens as from the full logits at every step;
  3. the server path's ms per prompt token and per generated token (information; the gate times the decode);
  4. the prefill against the decode loop over the same positions (the decode-only logits measured in this run):
     g4_ref's prefill 46 + decode 46 and g4_ref_long's prefill 4,775 + decode 256, teacher-forced, each position's
     full logits against the decode-only path's (KL, max |diff|, top-1) and both against llama.cpp's; the server
     path with the prefill (Engine.generate, the prefill forced): the short prompt's greedy tokens and a follow-up's
     against the decode loop's, the long prompt's (4,776) first token against the decode-only argmax; the short
     prompt's greedy answer teacher-forced both ways (where the top-1 differs: each side's logit gap); the long
     prefill with the engine's padding (PREFILL_PAD) against a 128-row padding and without a reset after a 16,000-token
     prefill (stale cells past the prompt), both bit for bit; the 16K limit (a prompt + max_new past it refused, the
     state kept);
  5. the prefill's wall time per prompt size (ids to the decode-ready state; the first of each size may compile) and
     the reset's: engine.py's PREFILL_S / PREFILL_TOK_S / RESET_S.

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((220 << 30)) --engine tt/serve/check_card.py
                                                         (writes $TT_MOEOFFLOAD_DATA/results/card_check.json)
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import queue
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import PREFILL_PAD, CardModel, Engine  # noqa: E402

REF = os.path.join(DATA, "g4_ref")
REF_LONG = os.path.join(DATA, "g4_ref_long")
LONG_FROM = 4775                                       # g4_ref_long's llama.cpp logits: positions 4,775..5,030
CTX = int(os.environ.get("TT_CTX", 16384))                    # g3_attn1.CTX: the build's context, for the file names
OUT = os.path.join(DATA, "results", "card_check" + ("" if CTX == 16384 else f"_ctx{CTX}") + ".json")
VOCAB = 248320
S3 = {"temperature": "1.0", "top_p": "0.95", "seed": "1234"}


class Lines:
    """The engine's stdout, kept."""

    def __init__(self):
        self.lines: list[str] = []

    def write(self, s: str) -> None:
        self.lines += [x for x in s.split("\n") if x]

    def flush(self) -> None:
        pass


def kl(ref, lg) -> float:
    """KL(ref || lg) of two logit rows (numpy or torch)."""
    lp, lq = (torch.log_softmax(torch.as_tensor(x).double(), -1) for x in (ref, lg))
    return float((lp.exp() * (lp - lq)).sum())


def compare(rows: np.ndarray, decode: np.ndarray, llama: np.ndarray | None, start: int) -> dict:
    """Prefill + decode rows against the decode-only rows, and both against llama.cpp's, over the same positions
    (start ..); the prefill path's worst position against llama.cpp left out as well (one sensitive position carries
    most of a short check's KL). Where the top-1 differs: each path's logit gap between its own top-1 and the other's
    (a near-tie: both small)."""
    k_dec = [kl(d, r) for d, r in zip(decode, rows)]
    a_pf, a_do = rows.argmax(1), decode.argmax(1)
    differ = np.flatnonzero(a_pf != a_do)
    out = {"positions": [start, start + len(rows) - 1],
           "vs_decode_only": {"kl_mean": float(np.mean(k_dec)), "kl_max": max(k_dec),
                              "kl_max_at": start + int(np.argmax(k_dec)),
                              "max_abs_diff": float(np.abs(rows - decode).max()),
                              "top1_same": float(np.mean(a_pf == a_do)),
                              "top1_differ": [{"pos": start + int(i), "prefill": int(a_pf[i]), "decode": int(a_do[i]),
                                               "gap_prefill": float(rows[i, a_pf[i]] - rows[i, a_do[i]]),
                                               "gap_decode": float(decode[i, a_do[i]] - decode[i, a_pf[i]])}
                                              for i in differ]}}
    if llama is None:
        return out
    k_pf, k_do = [kl(x, r) for x, r in zip(llama, rows)], [kl(x, d) for x, d in zip(llama, decode)]
    worst = int(np.argmax(k_pf))
    return {**out,
            "prefill_vs_llama": {"top1": float(np.mean(rows.argmax(1) == llama.argmax(1))),
                                 "kl_mean": float(np.mean(k_pf)), "kl_max": max(k_pf), "kl_max_at": start + worst,
                                 "kl_mean_without_worst": float(np.mean(np.delete(k_pf, worst)))},
            "decode_only_vs_llama": {"top1": float(np.mean(decode.argmax(1) == llama.argmax(1))),
                                     "kl_mean": float(np.mean(k_do)), "kl_max": max(k_do),
                                     "kl_max_at": start + int(np.argmax(k_do)), "kl_at_prefill_worst": k_do[worst],
                                     "kl_mean_without_prefill_worst": float(np.mean(np.delete(k_do, worst)))}}


def teacher_forced(M, ids: list[int], start: int, end: int) -> np.ndarray:
    """The full logits at positions start .. end - 1, ids fed from the state's position start."""
    return np.stack([M.step(ids[pos], pos, ids[:pos]).full().t.numpy().copy() for pos in range(start, end)])


def request(eng: Engine, out: Lines, max_new: int, ids: list[int], keys: dict | None = None) -> dict:
    start = len(out.lines)
    eng.generate(max_new, keys or {}, ids)
    got = out.lines[start:]
    done = got[-1].split()
    assert done[0] == "DONE", got[-1]
    gen = int(done[1])
    return {"tokens": [int(x[2:]) for x in got if x.startswith("T ")], "resumed": int(done[8]),
            "prompt_read": int(done[14]), "prompt_ms": float(done[3]), "decode_ms": float(done[4]),
            "finish": done[5], "ms_per_prompt_token": float(done[3]) / max(1, int(done[14])),
            "ms_per_generated_token": float(done[4]) / max(1, gen - 1)}


def main() -> None:
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32).tolist()
    ref = np.fromfile(os.path.join(REF, "logits.f32"), np.float32).reshape(-1, VOCAB)
    card_ref = np.fromfile(os.path.join(REF, "logits_card.f32"), np.float32).reshape(-1, VOCAB)
    t0 = time.time()
    M = CardModel()
    from g4_gate import top_ref                        # CardModel put tt/flashnext on the path
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ref": REF, "tokens": len(ids), "build_s": round(time.time() - t0)}
    try:
        rows, top_bad = [], []
        for pos, tok in enumerate(ids):
            top = M.step(tok, pos, ids[:pos])
            lg = top.full().t
            r_ids, r_vals = top_ref(lg)
            if not (r_ids.tolist() == top.ids and
                    np.array_equal(np.array(top.vals, np.float32).view(np.int32), r_vals.view(np.int32))):
                top_bad.append(pos)
            rows.append(lg.numpy().copy())
        dump = np.stack(rows)
        kls, top1 = [], []
        for d, r in zip(dump, ref):
            lp = torch.log_softmax(torch.from_numpy(r).double(), -1)
            lq = torch.log_softmax(torch.from_numpy(d).double(), -1)
            kls.append(float((lp.exp() * (lp - lq)).sum()))
            top1.append(int(r.argmax()) == int(d.argmax()))
        res["dump"] = {"identical_to_g4_token": bool(np.array_equal(dump, card_ref)),
                       "max_abs_diff_vs_g4_token": float(np.abs(dump - card_ref).max()),
                       "top64_differ_at": top_bad,
                       "top1_vs_llama": sum(top1) / len(top1), "kl_mean_vs_llama": float(np.mean(kls)),
                       "kl_max_vs_llama": float(np.max(kls))}
        print(f"dump: {res['dump']}", flush=True)
        M.reset()
        out = Lines()
        eng = Engine(M, out, queue.Queue())
        eng.prefill = None                             # the decode loop (section 4's reference)
        a = request(eng, out, 32, ids)
        b = request(eng, out, 32, ids)
        follow = ids + a["tokens"]
        c = request(eng, out, 16, follow)
        d = request(eng, out, 16, follow)
        s_top = request(eng, out, 32, ids, S3)
        eng.full_logits = True
        s_full = request(eng, out, 32, ids, S3)
        eng.full_logits = False
        res["generate"] = {
            "first": a["tokens"][0], "argmax_last_card": int(card_ref[-1].argmax()),
            "argmax_last_llama": int(ref[-1].argmax()),
            "first_is_the_card_argmax": a["tokens"][0] == int(card_ref[-1].argmax()),
            "same_after_reset": a["tokens"] == b["tokens"] and b["resumed"] == 0,
            "resumed": c["resumed"], "resume_equals_fresh_read": c["tokens"] == d["tokens"] and d["resumed"] == 0,
            "sampled_top64_equals_full": s_top["tokens"] == s_full["tokens"],
            "requests": {"greedy": a, "greedy_again": b, "resumed": c, "fresh": d, "sampled_top64": s_top,
                         "sampled_full": s_full}}
        g = res["generate"]
        print(f"generate: first {g['first']} (card argmax {g['argmax_last_card']}, llama.cpp "
              f"{g['argmax_last_llama']}); same after reset {g['same_after_reset']}; resumed {g['resumed']} tokens, "
              f"equal to a fresh read {g['resume_equals_fresh_read']}; sampled from the top 64 = from the full logits "
              f"{g['sampled_top64_equals_full']}; {a['ms_per_prompt_token']:.2f} ms per prompt token, "
              f"{a['ms_per_generated_token']:.2f} ms per generated token (full logits: "
              f"{s_full['ms_per_generated_token']:.2f})", flush=True)
        prefill_checks(M, eng, out, ids, dump, ref, a, c, res)
        prefill_timing(M, res)
    finally:
        M.close()
    res["host_peak_rss_gb"] = round(next(int(x.split()[1]) for x in open("/proc/self/status")
                                         if x.startswith("VmHWM")) / 1e6, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT}")


def prefill_checks(M, eng: Engine, out: Lines, ids, dump, ref, a, c, res) -> None:
    """Section 4: the prefill against the decode loop, teacher-forced and through the server's path."""
    half = 46
    M.reset()
    M.prefill(ids[:half])
    r = {"short_prefill_46_decode_46": compare(teacher_forced(M, ids, half, len(ids)), dump[half:], ref[half:], half)}
    ids_l = np.fromfile(os.path.join(REF_LONG, "ids.i32"), np.int32).tolist()
    ref_l = np.fromfile(os.path.join(REF_LONG, "logits.f32"), np.float32).reshape(-1, VOCAB)
    M.reset()
    t0 = time.time()
    for pos in range(LONG_FROM):                          # the decode-only path up to the reference's positions
        M.step(ids_l[pos], pos, ids_l[:pos])
    dec_l = teacher_forced(M, ids_l, LONG_FROM, len(ids_l))
    r["decode_only_long_s"] = round(time.time() - t0)
    tf = lambda: teacher_forced(M, ids_l, LONG_FROM, len(ids_l))   # noqa: E731
    M.reset()
    M.P.pad = 128                                         # the first prefill's 128-row padding, then the engine's
    M.prefill(ids_l[:LONG_FROM])
    rows128 = tf()
    M.P.pad = PREFILL_PAD
    M.reset()
    M.prefill(ids_l[:LONG_FROM])
    rows = tf()
    r["long_prefill_4775_decode_256"] = compare(rows, dec_l, ref_l, LONG_FROM)
    r["long_pad"] = {"pad": PREFILL_PAD, "bitwise_equal_to_pad_128": bool(np.array_equal(rows, rows128)),
                     "max_abs_diff": float(np.abs(rows - rows128).max())}
    M.prefill(list(range(2000, 18000)))                   # every cache cell written ...
    M.prefill(ids_l[:LONG_FROM])                          # ... and no reset before this one
    rows_nr = tf()
    r["long_no_reset"] = {"bitwise_equal": bool(np.array_equal(rows, rows_nr)),
                          "max_abs_diff": float(np.abs(rows - rows_nr).max())}
    seq = ids + a["tokens"]                               # the decode loop's greedy answer, teacher-forced both ways
    M.reset()
    for pos in range(len(ids) - 1):
        M.step(seq[pos], pos, seq[:pos])
    dec_s = teacher_forced(M, seq, len(ids) - 1, len(seq))
    M.reset()
    M.prefill(ids[:-1])
    r["short_greedy_prefill_91"] = compare(teacher_forced(M, seq, len(ids) - 1, len(seq)), dec_s, None, len(ids) - 1)
    eng.prefill, eng.use_prefill = M.prefill, lambda n, reuse, reset=False: True    # the prefill forced
    p = request(eng, out, 32, ids)
    f = request(eng, out, 16, ids + a["tokens"])
    lg = request(eng, out, 16, ids_l[:LONG_FROM + 1])
    del eng.use_prefill
    same = next((i for i, (x, y) in enumerate(zip(p["tokens"], a["tokens"])) if x != y), len(p["tokens"]))
    r["server"] = {"short_tokens_equal_decode_loop": p["tokens"] == a["tokens"], "short_same_leading_tokens": same,
                   "short_first": [p["tokens"][0], a["tokens"][0]], "short_prompt_read": p["prompt_read"],
                   "follow_up_equals_resumed": f["tokens"] == c["tokens"] and f["resumed"] == 0,
                   "long_first": lg["tokens"][0], "long_first_decode_only": int(dec_l[0].argmax()),
                   "long_first_llama": int(ref_l[0].argmax()), "long_prompt_ms": lg["prompt_ms"],
                   "requests": {"short": p, "follow_up": f, "long": lg}}
    held, start = list(eng.held), len(out.lines)
    try:
        eng.generate(32, {}, [1] * (M.ctx - 16))   # not range(): past 151,936 cells its ids leave the vocabulary
        r["limit_refused"] = False
    except ValueError as e:                               # BadRequest: refused before anything is read
        r["limit_refused"] = "exceeds the context" in str(e) and eng.held == held and len(out.lines) == start
    res["prefill"] = r
    for k in ("short_prefill_46_decode_46", "long_prefill_4775_decode_256"):
        v = r[k]
        print(f"{k}: vs decode-only {v['vs_decode_only']}; prefill vs llama {v['prefill_vs_llama']}; decode-only vs "
              f"llama {v['decode_only_vs_llama']}", flush=True)
    print(f"short greedy answer, prefill 91 vs the decode loop: {r['short_greedy_prefill_91']['vs_decode_only']}; "
          f"long, pad {PREFILL_PAD} vs 128: {r['long_pad']}; no reset: {r['long_no_reset']}", flush=True)
    print(f"server path with the prefill: {({k: v for k, v in r['server'].items() if k != 'requests'})}; "
          f"16K limit refused {r['limit_refused']}", flush=True)


def prefill_timing(M, res: dict) -> None:
    """Section 5: the prefill's wall per prompt size (random ids, twice each: the first may compile), the reset's.
    Every one of the engine's 16 padded shapes once (so a run of this check also fills the kernel cache for them)."""
    from g7_prefill import dram_gb
    rng = np.random.default_rng(0)
    t = {}
    for n in [128] + [PREFILL_PAD * j - 100 for j in range(1, 17)]:
        walls = []
        for _ in range(2):
            M.reset()
            t0 = time.perf_counter()
            M.prefill(rng.integers(0, 200000, n).tolist())
            walls.append(round(time.perf_counter() - t0, 3))
        t[n] = walls + [round(dram_gb(M.dev), 2)]
        print(f"prefill {n}: {walls} s -> {n / walls[1]:.0f} tok/s; card memory after it {t[n][-1]} GB", flush=True)
    resets = []
    for _ in range(3):
        t0 = time.perf_counter()
        M.reset()
        resets.append(round(time.perf_counter() - t0, 3))
    res["prefill_timing"] = {"walls_s_then_card_gb": t, "reset_s": resets}
    print(f"reset: {resets} s", flush=True)


# ------------------------------------------------------------------------------------------------ speculation
LONG_PROMPT = "B.strata-drafter"                     # the census corpus's agent session that the prefill reads (4,785 tokens)
SPEC_PROMPTS = ("A.code.0", "A.chat.18", "A.math.12", "A.zh.40", "A.de.38", "A.creative.22", LONG_PROMPT)
SPEC_N = 128                                           # tokens per greedy request
TIE = 0.75                                             # a near-tie: g6_spec.py's 0.5 (the M-row numerics' ~0.3 logit), widened 2026-11-07: with the combine and gate kernels the one-row engine and the M-row passes split at a 0.60 gap (the corpus's long agent prompt before 2026-11-15)
# a split is a near-tie when either side sees its two tokens within TIE (2026-11-07, the user's call): the one-row engine
# at 0.017 against the folded-scale build's speculative run at 0.775 (that prompt) is a coin flip, not an error
SPEC_REF = os.path.join(HERE, "results", "card_check_spec_ref.json")
SPEC_OUT = os.path.join(DATA, "results", "card_check_spec" + ("" if CTX == 16384 else f"_ctx{CTX}") + ".json")


def spec_prompts() -> list[tuple[str, list[int]]]:
    """The greedy requests' prompts: g4_ref's, then the census prompts as the census tokenized them (tools/llama:
    routing-census --census-ids; LONG_PROMPT is read by the prefill)."""
    ids = os.path.join(DATA, "census", "qwen3.8-flash-next", "ids")
    out = [("g4_ref", np.fromfile(os.path.join(REF, "ids.i32"), np.int32).tolist())]
    return out + [(n, np.fromfile(os.path.join(ids, n + ".i32"), np.int32).tolist()) for n in SPEC_PROMPTS]


class Rows:
    """Every kept row's top 64 by position (the model's step or feed + keep, wrapped): the row after token pos."""

    def __init__(self, M):
        self.at: dict[int, tuple[list[int], list[float]]] = {}
        self.passes: list[int] = []                   # each pass's position
        if hasattr(M, "keep"):
            feed, keep = M.feed, M.keep

            def fed(toks, pos, prev):
                self.passes.append(pos)
                self.last = (pos, feed(toks, pos, prev))
                return self.last[1]

            def kept(n):
                pos, rows = self.last
                self.at.update({pos + m: (rows[m].ids, rows[m].vals) for m in range(n)})
                keep(n)
            M.feed, M.keep = fed, kept
        else:
            step = M.step

            def stepped(tok, pos, prev):
                top = step(tok, pos, prev)
                self.at[pos] = (top.ids, top.vals)
                return top
            M.step = stepped


def gap(row, a: int, b: int) -> float | None:
    """Token a's logit minus token b's in a row's top 64 (None: b not in it)."""
    d = dict(zip(*row))
    return round(d[a] - d[b], 4) if b in d else None


def versus(toks, ref, n_prompt: int, rows: Rows, ref_rows: dict) -> dict:
    """A run's tokens against a reference run's: the leading tokens they share, and at the first difference each
    side's gap between its own token and the other's in its row there (a near-tie: either within TIE)."""
    k = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), min(len(toks), len(ref)))
    out = {"same": k, "of": min(len(toks), len(ref))}
    if k < out["of"]:
        q = n_prompt - 1 + k
        a, b = ref[k], toks[k]
        out["first_difference"] = {"at": k, "ref": a, "run": b, "gap_ref": gap(ref_rows[q], a, b),
                                   "gap_run": gap(rows.at[q], b, a)}
        g = [x for x in out["first_difference"].values() if isinstance(x, float)]
        out["near_tie"] = len(g) == 2 and min(g) <= TIE
    return out


def main_spec_ref() -> None:
    """The one-row engine (CardModel, as tt-card without spec) on the speculation check's greedy requests: tokens and
    each generated token's row (top 64), for main_spec's comparison."""
    M = CardModel()
    rows = Rows(M)
    out, eng, res = Lines(), None, {"date": time.strftime("%Y-%m-%d %H:%M"), "requests": {}}
    try:
        eng = Engine(M, out, queue.Queue())
        for name, ids in spec_prompts():
            rows.at = {}
            r = request(eng, out, SPEC_N, ids)
            res["requests"][name] = {"n_prompt": len(ids), "tokens": r["tokens"], "prompt_ms": r["prompt_ms"],
                                     "ms_per_generated_token": r["ms_per_generated_token"],
                                     "rows": {q: rows.at[q] for q in range(len(ids) - 1, len(ids) - 1 + SPEC_N)
                                              if q in rows.at}}
            print(f"{name}: {len(ids)} + {len(r['tokens'])}, {r['ms_per_generated_token']:.2f} ms a token", flush=True)
    finally:
        M.close()
    with open(SPEC_REF, "w") as f:
        json.dump(res, f)
    print(f"wrote {SPEC_REF}")


def main_spec() -> None:
    """Speculation on the card (SpecModel, the profile's default: subset head; rows from TTC_ROWS, default 4):
      1. the hand-off into the verify build: g4_ref_long's prefill 4,775 then 256 one-row passes teacher-forced, full
         logits, against the verify build reading the same tokens in 4-row passes from 0 and against llama.cpp; the
         same prefill after a 16,000-token one (no reset), bit for bit; the MTP's K/V over LONG_PROMPT's prompt
         caught up from the prefill's rows against from the 4-row passes' (relative);
      2. greedy requests served with speculation against the same build's one-row passes (drafts off) and against
         the one-row engine (main_spec_ref's run, when there): the leading tokens shared, the first difference's gaps;
         g4_ref's prompt with the prefill forced (the hand-off and the catch-up on a short prompt); a follow-up
         (RESUME) against the same prompt read from 0; a sampled request (S3) from the top 64 against from the full
         logits; the 16K limit (refused, the state kept; a prompt of 16,352 + 32 tokens served to the last cell);
      3. the prefill's wall with the MTP's catch-up per prompt size, the reset's, tokens per pass.
    TTC_STEPS picks some of timing, hand_off, serve (default all), so each fits a short card run; TTC_SIZES the
    timing's prompt sizes; TTC_FORCE_EXPERTS card / cpu / split<g> every prefill's experts (the cost model's fit); fit:
    tt_moeoffload/calibrate.py's measurements (spec_fit). A partial run writes card_check_spec_<steps>[_<sizes>].json."""
    from engine import SpecModel
    rows_n = int(os.environ.get("TTC_ROWS", 4))
    steps = os.environ.get("TTC_STEPS", "timing,hand_off,serve").split(",")
    part = steps + ([os.environ["TTC_SIZES"].replace(",", "-")] if "TTC_SIZES" in os.environ else [])
    out = SPEC_OUT if part == ["timing", "hand_off", "serve"] else SPEC_OUT[:-5] + "_" + "_".join(part) + ".json"
    t0 = time.time()
    M = SpecModel(rows=rows_n)
    M.force_experts = os.environ.get("TTC_FORCE_EXPERTS") or None     # "card" / "cpu" / "split<g>": every prefill's experts
    from g7_prefill import dram_gb
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "rows": rows_n, "head": M.head, "build_s": round(time.time() - t0),
           "card_gb": round(dram_gb(M.dev), 2)}
    print(f"built in {res['build_s']} s, card memory {res['card_gb']} GB", flush=True)
    try:
        for step in ("timing", "hand_off", "serve", "long", "extend", "extend_time", "ckpt", "conv", "pieces", "fit"):
            if step in steps:
                {"timing": spec_timing, "hand_off": spec_hand_off, "serve": spec_serve, "long": spec_long, "fit": spec_fit,
                 "extend": spec_extend, "extend_time": spec_extend_time, "ckpt": spec_ckpt, "conv": spec_conv,
                 "pieces": spec_pieces}[step](M, res)
    finally:
        M.close()
    res["host_peak_rss_gb"] = round(next(int(x.split()[1]) for x in open("/proc/self/status")
                                         if x.startswith("VmHWM")) / 1e6, 1)
    res["pass_"] = bool(res.get("serve", {}).get("pass_", True) and res.get("long", {}).get("pass_", True) and
                        res.get("extend", {}).get("pass_", True) and
                        res.get("conv", {}).get("pass_", True) and res.get("pieces", {}).get("pass_", True) and
                        res.get("hand_off", {}).get("no_reset_bitwise_equal", True))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"spec check ({', '.join(steps)}): {'PASS' if res['pass_'] else 'FAIL'}; wrote {out}")


def tf_spec(M, ids, start: int, end: int) -> np.ndarray:
    """One-row passes teacher-forced at positions start .. end - 1 (the state at start), full logits."""
    out = []
    for pos in range(start, end):
        out.append(M.feed([ids[pos]], pos, ids[:pos])[0].full().t.numpy().copy())
        M.keep(1)
    return np.stack(out)


def read_passes(M, ids, n: int) -> None:
    """ids[:n] read from 0 in 4-row passes (the engine's prompt loop)."""
    M.reset()
    for p in range(0, n, 4):
        toks = ids[p:min(p + 4, n)]
        M.feed(toks, p, ids[:p])
        M.keep(len(toks))


def spec_hand_off(M, res: dict) -> None:
    ids_l = np.fromfile(os.path.join(REF_LONG, "ids.i32"), np.int32).tolist()
    ref_l = np.fromfile(os.path.join(REF_LONG, "logits.f32"), np.float32).reshape(-1, VOCAB)
    read_passes(M, ids_l, LONG_FROM)
    dec = tf_spec(M, ids_l, LONG_FROM, len(ids_l))
    M.reset()
    M.prefill(ids_l[:LONG_FROM])
    rows = tf_spec(M, ids_l, LONG_FROM, len(ids_l))
    M.prefill(list(range(2000, 18000)))               # every cache cell written ...
    M.prefill(ids_l[:LONG_FROM])                      # ... and no reset before this one
    rows_nr = tf_spec(M, ids_l, LONG_FROM, len(ids_l))
    r = {"long_prefill_4775_then_one_row_passes": compare(rows, dec, ref_l, LONG_FROM),
         "no_reset_bitwise_equal": bool(np.array_equal(rows, rows_nr)),
         "no_reset_max_abs_diff": float(np.abs(rows - rows_nr).max())}
    (name, ids), = [p for p in spec_prompts() if p[0] == LONG_PROMPT]
    n = len(ids) - 1
    read_passes(M, ids, n)                            # 4-row passes: the MTP's cells caught up pass by pass
    M.draft(ids, 0)
    kv_passes = M.C.kv(range(n))
    M.reset()
    M.prefill(ids[:-1])                               # the prefill's rows, caught up in 4-row K/V passes
    M.draft(ids, 0)
    kv_pf = M.C.kv(range(n))
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())    # noqa: E731
    r["mtp_kv_prefill_vs_passes"] = {"cells": n, "K_rel_max": rel(kv_pf[0], kv_passes[0]),
                                     "V_rel_max": rel(kv_pf[1], kv_passes[1])}
    res["hand_off"] = r
    v = r["long_prefill_4775_then_one_row_passes"]
    print(f"hand-off: vs the 4-row read {v['vs_decode_only']}; prefill vs llama {v['prefill_vs_llama']}; 4-row read "
          f"vs llama {v['decode_only_vs_llama']}; no reset bitwise {r['no_reset_bitwise_equal']}; MTP K/V "
          f"{r['mtp_kv_prefill_vs_passes']}", flush=True)


def spec_serve(M, res: dict) -> None:
    ref = json.load(open(SPEC_REF))["requests"] if os.path.exists(SPEC_REF) else {}
    rows, out = Rows(M), Lines()
    eng = Engine(M, out, queue.Queue())
    r, ok = {"near_tie_logits": TIE, "requests": {}}, True
    for name, ids in spec_prompts():
        M.reset()
        eng.held, eng.drafts = [], 0                  # the same build's one-row passes
        rows.at = {}
        one = request(eng, out, SPEC_N, ids)
        one_rows = dict(rows.at)
        M.reset()
        eng.held, eng.drafts = [], M.rows - 1
        rows.at, rows.passes = {}, []
        sp = request(eng, out, SPEC_N, ids)
        decode = sum(1 for q in rows.passes if q >= len(ids) - 1)
        x = {"n_prompt": len(ids), "prompt_read_by": "prefill" if eng.use_prefill(len(ids), 0, True) else "passes",
             "tokens_per_pass": round(len(sp["tokens"]) / max(1, decode), 3),
             "ms_per_token": {"spec": round(sp["ms_per_generated_token"], 2),
                              "one_row_passes": round(one["ms_per_generated_token"], 2)},
             "vs_one_row_passes": versus(sp["tokens"], one["tokens"], len(ids), rows, one_rows)}
        if name in ref:
            q = {int(k): v for k, v in ref[name]["rows"].items()}
            x["vs_one_row_engine"] = versus(sp["tokens"], ref[name]["tokens"], len(ids), rows, q)
            x["one_row_engine_ms_per_token"] = round(ref[name]["ms_per_generated_token"], 2)
        good = all(v.get("near_tie", True) for k, v in x.items() if k.startswith("vs_"))
        ok &= good
        x["tokens"] = sp["tokens"]
        r["requests"][name] = x
        print(f"{name} ({len(ids)}): " + json.dumps({k: v for k, v in x.items() if k != "tokens"}), flush=True)
    ids = spec_prompts()[0][1]
    M.reset()
    eng.held = []
    a = request(eng, out, 32, ids)
    eng.use_prefill = lambda n, reuse, reset=False: True   # the prefill forced on the short prompt
    p = request(eng, out, 32, ids)
    del eng.use_prefill
    follow = ids + p["tokens"]
    c = request(eng, out, 16, follow)                 # RESUME after the prefilled request
    M.reset()
    eng.held = []
    d = request(eng, out, 16, follow)
    s_top = request(eng, out, 32, ids, S3)
    eng.full_logits = True
    s_full = request(eng, out, 32, ids, S3)
    eng.full_logits = False
    sh = lambda u, v: next((i for i, (x, y) in enumerate(zip(u, v)) if x != y), min(len(u), len(v)))  # noqa: E731
    r["short_prefill_vs_passes_same"] = [sh(p["tokens"], a["tokens"]), len(a["tokens"])]
    r["resumed"] = c["resumed"]
    r["resume_vs_fresh_read_same"] = [sh(c["tokens"], d["tokens"]), len(c["tokens"])]
    r["sampled_top64_equals_full"] = s_top["tokens"] == s_full["tokens"]
    held, start = list(eng.held), len(out.lines)
    try:
        eng.generate(32, {}, [1] * (M.ctx - 16))   # not range(): past 151,936 cells its ids leave the vocabulary
        r["limit_refused"] = False
    except ValueError as e:                           # BadRequest: refused before anything is read
        r["limit_refused"] = "exceeds the context" in str(e) and eng.held == held and len(out.lines) == start
    edge = request(eng, out, 32, list(range(1, M.ctx - 31)))
    r["limit_edge"] = {"prompt": M.ctx - 32, "generated": len(edge["tokens"]), "finish": edge["finish"],
                       "held": len(eng.held)}
    r["pass_"] = bool(ok and r["sampled_top64_equals_full"] and r["limit_refused"] and c["resumed"] > 0
                      and (len(edge["tokens"]) == 32 or edge["finish"] == "stop"))
    res["serve"] = r
    print(f"serve: {({k: v for k, v in r.items() if k != 'requests'})}", flush=True)



def spec_ckpt(M, res: dict) -> None:
    """Prompt checkpoints (2026-11-07): LONG_PROMPT's prompt (A) saved at its last turn token q, then
    (1) A + 16 other tokens restored from it (RESUME q) against the same prompt read with no checkpoint (the prefill
        to q, the rest in passes: the restored state is the one that prefill left), tokens and kept rows bit for bit;
    (2) a follow-up as S3 sends it (A, a different reply, a new user turn and assistant opening) restored from A's
        checkpoint against the same path with no checkpoint (the prefill to q, then the rest), bit for bit, and
        against the prompt read from 0 (the prefill all the way: the read paths' distance, reported);
    (3) after another prompt read from 0 (the K/V cells overwritten) A's checkpoint is not used."""
    from engine import TURN
    ids = dict(spec_prompts())[LONG_PROMPT]
    q = max(i for i in range(len(ids) - 1) if ids[i] == TURN)
    assert len(ids) - 1 - q <= 64, (len(ids), q)
    rows, out = Rows(M), Lines()
    eng = Engine(M, out, queue.Queue())
    r = {"prompt": len(ids), "turn_at": q}

    def fresh():
        M.reset()
        eng.held, eng.dirty, eng.ckpts, rows.at = [], False, [], {}

    other = ids[100:116]
    assert TURN not in other
    fresh()
    a = request(eng, out, 32, ids)
    r["saved"] = [len(k) - 1 for k, _ in eng.ckpts]
    rows.at = {}
    b = request(eng, out, 32, ids + other)
    b_rows = dict(rows.at)
    fresh()
    c = request(eng, out, 32, ids + other)
    same_rows = all(b_rows.get(p) == v for p, v in rows.at.items()) and set(b_rows) == set(rows.at)
    r["restored"] = {"resumed": b["resumed"], "prompt_ms": round(b["prompt_ms"], 1),
                     "no_checkpoint_prompt_ms": round(c["prompt_ms"], 1),
                     "tokens_same": b["tokens"] == c["tokens"], "rows_same": same_rows}
    ok = b["resumed"] == q and b["tokens"] == c["tokens"] and same_rows
    # (2) S3's follow-up: A's prompt, a reply that is not the generated one, a user turn, the assistant opening
    tail = ids[q:]                                     # <|im_start|>assistant\n ... as A opened its turn
    f = ids + ids[200:260] + [248046, 198] + [TURN] + ids[q + 1:q + 2] + ids[300:420] + [248046, 198] + tail
    fresh()
    e = request(eng, out, 64, f)                       # from 0
    e_rows = dict(rows.at)
    fresh()
    request(eng, out, 32, ids)
    rows.at = {}
    d = request(eng, out, 64, f)                       # restored from A's checkpoint
    d_rows = dict(rows.at)
    fresh()                                            # the same path with no checkpoint: the prefill to q, then f
    M.prefill(ids[:q])
    eng.held = ids[:q]
    rows.at = {}
    g = request(eng, out, 64, f)
    same_rows = all(d_rows.get(p) == v for p, v in rows.at.items()) and set(d_rows) == set(rows.at)
    rows.at = d_rows
    r["follow_up"] = {"prompt": len(f), "resumed": d["resumed"], "prompt_ms": round(d["prompt_ms"], 1),
                      "from_0_prompt_ms": round(e["prompt_ms"], 1),
                      "vs_same_path_no_checkpoint": {"resumed": g["resumed"], "tokens_same": d["tokens"] == g["tokens"],
                                                     "rows_same": same_rows},
                      "vs_from_0": versus(d["tokens"], e["tokens"], len(f), rows, e_rows)}   # the read paths' distance
    ok &= d["resumed"] == q and g["resumed"] == q and d["tokens"] == g["tokens"] and same_rows
    # (3) another prompt read from 0 overwrites the cells: A's checkpoint must not be used after it
    fresh()
    request(eng, out, 8, ids)
    request(eng, out, 8, list(range(1000, 1000 + q + 8)))
    k = request(eng, out, 8, ids + other)
    r["invalidated"] = {"resumed_after_other_prompt": k["resumed"]}
    ok &= k["resumed"] == 0
    r["pass_"] = bool(ok)
    res["ckpt"] = r
    print("ckpt: " + json.dumps(r), flush=True)


def spec_extend(M, res: dict) -> None:
    """A follow-up continuing the held state through the prefill (2026-11-06): g4_ref_long's state at 4,775 reached
    (a) by a prefill from 0, (b) a prefill of 2,048 then the prefill from 2,048, (c) a prefill of 1,900, 4-row passes
    to 2,173, a pass of 4 keeping 3 (so the extension commits a partly kept pass), then the prefill from 2,176, (d) as
    (c) to 2,076 (2026-11-10: a start inside a tile of pooled keys, pool 519 of the tile from 512); each then 256
    one-row passes teacher-forced, full logits, against llama.cpp's and against (a)'s."""
    from g7_prefill import kl_top1
    ids = np.fromfile(os.path.join(REF_LONG, "ids.i32"), np.int32).tolist()
    ref = np.fromfile(os.path.join(REF_LONG, "logits.f32"), np.float32).reshape(-1, VOCAB)
    end = len(ids)

    def passes(a, b):
        for p in range(a, b, 4):
            toks = ids[p:min(p + 4, b)]
            M.feed(toks, p, ids[:p])
            M.keep(len(toks))
    runs = {}
    for name in ("from_0", "after_prefill", "after_passes", "mid_tile"):
        M.reset()
        t0 = time.perf_counter()
        if name == "from_0":
            M.prefill(ids[:LONG_FROM])
        elif name == "after_prefill":
            M.prefill(ids[:2048])
            t0 = time.perf_counter()
            M.prefill(ids[:LONG_FROM], start=2048)
        elif name == "after_passes":
            M.prefill(ids[:1900])
            passes(1900, 2173)
            M.feed(ids[2173:2177], 2173, ids[:2173])
            M.keep(3)                                     # n_keep 3: the extension commits 3 of the pass's 4 rows
            t0 = time.perf_counter()
            M.prefill(ids[:LONG_FROM], start=2176)
        else:
            M.prefill(ids[:1900])
            passes(1900, 2073)
            M.feed(ids[2073:2077], 2073, ids[:2073])
            M.keep(3)
            t0 = time.perf_counter()
            M.prefill(ids[:LONG_FROM], start=2076)
        wall = time.perf_counter() - t0
        runs[name] = (tf_spec(M, ids, LONG_FROM, end), wall)
    r, ok = {}, True
    base = runs["from_0"][0]
    for name, (lg, wall) in runs.items():
        vs_ref = [kl_top1(ref[q - LONG_FROM], torch.from_numpy(lg[q - LONG_FROM])) for q in range(LONG_FROM, end)]
        vs_0 = [kl_top1(base[q - LONG_FROM], torch.from_numpy(lg[q - LONG_FROM])) for q in range(LONG_FROM, end)]
        x = {"prefill_s": round(wall, 2),
             "vs_llama": {"kl_mean": float(np.mean([k for k, _ in vs_ref])), "top1": float(np.mean([t for _, t in vs_ref]))},
             "vs_from_0": {"kl_mean": float(np.mean([k for k, _ in vs_0])), "top1_same": float(np.mean([t for _, t in vs_0]))}}
        r[name] = x
        print(f"extend {name}: " + json.dumps(x), flush=True)
    # an extension is another prefill of the same positions: within the prefill's own distance to the decode's
    for name in ("after_prefill", "after_passes", "mid_tile"):
        ok &= r[name]["vs_llama"]["kl_mean"] <= r["from_0"]["vs_llama"]["kl_mean"] + 0.005
    r["pass_"] = bool(ok)
    res["extend"] = r


def spec_extend_time(M, res: dict) -> None:
    """A follow-up's prompt time on a held context (g4_ref_64k's text, TTC_HELD, default 29,952 = 234 x 128 tokens held
    from a prefill), k new tokens (TTC_FOLLOW, default 1024,4096): the prefill continuing the held state (twice: the
    first compiles), the decode loop's 4-row passes, the prefill from 0 (twice)."""
    ids = np.fromfile(os.path.join(DATA, "g4_ref_64k/ids.i32"), np.int32).tolist()
    r, modes = {}, os.environ.get("TTC_MODES", "extend,loop,from_0").split(",")
    for h, k in [(int(h), int(k)) for h in os.environ.get("TTC_HELD", "29952").split(",")
                 for k in os.environ.get("TTC_FOLLOW", "1024,4096").split(",")]:
        x = {}
        for name, reps in [(m, 1 if m == "loop" else 2) for m in modes]:
            walls = []
            for _ in range(reps):
                M.reset()
                if name != "from_0":
                    M.prefill(ids[:h])
                M.ttnn.synchronize_device(M.dev)
                t0 = time.perf_counter()
                if name == "extend":
                    M.prefill(ids[:h + k], start=h)
                elif name == "loop":
                    for p in range(h, h + k, 4):
                        M.feed(ids[p:p + 4], p, ids[:p])
                        M.keep(4)
                else:
                    M.prefill(ids[:h + k])
                M.ttnn.synchronize_device(M.dev)
                walls.append(round(time.perf_counter() - t0, 2))
            x[name] = walls
        r[f"{h}+{k}"] = x
        print(f"extend_time held {h} + {k}: " + json.dumps(x), flush=True)
    res["extend_time"] = r


def spec_long(M, res: dict) -> None:
    """Speculation past the MTP window's 8,800 cells (2026-11-06; before, passes there had one row): g4_ref_64k's real
    text, its first n tokens (TTC_LONG, default 4000,12000,30000,60000) as the prompt, 256 greedy tokens served with
    drafts and with one-row passes; tokens per decode pass, ms per token, the streams against each other (near-ties)."""
    ids_all = np.fromfile(os.path.join(DATA, "g4_ref_64k/ids.i32"), np.int32).tolist()
    rows, out = Rows(M), Lines()
    eng = Engine(M, out, queue.Queue())
    r, ok = {}, True
    for n in [int(v) for v in os.environ.get("TTC_LONG", "4000,12000,30000,60000").split(",")]:
        ids = ids_all[:n]
        got = {}
        for name, drafts in (("one_row", 0), ("spec", M.rows - 1)):
            M.reset()
            eng.held, eng.drafts = [], drafts
            rows.at, rows.passes = {}, []
            q = request(eng, out, 256, ids)
            got[name] = (q, dict(rows.at), sum(1 for p in rows.passes if p >= len(ids) - 1))
        (one, one_rows, _), (sp, _, decode) = got["one_row"], got["spec"]
        x = {"tokens_per_pass": round(len(sp["tokens"]) / max(1, decode), 3),
             "ms_per_token": {"spec": round(sp["ms_per_generated_token"], 2),
                              "one_row_passes": round(one["ms_per_generated_token"], 2)},
             "prompt_ms": {"spec": sp.get("prompt_ms"), "one_row_passes": one.get("prompt_ms")},
             "vs_one_row_passes": versus(sp["tokens"], one["tokens"], len(ids), rows, one_rows)}
        ok &= x["vs_one_row_passes"].get("near_tie", True)
        r[n] = x
        print(f"long {n}: " + json.dumps(x), flush=True)
    r["pass_"] = ok
    res["long"] = r


def spec_timing(M, res: dict) -> None:
    """The prefill's wall (ids to the verify build's state), the MTP's catch-up over its rows (K/V passes and the
    first drafts), per prompt size (random ids, twice each: the first may compile); the reset's."""
    from g7_prefill import dram_gb
    rng = np.random.default_rng(0)
    t = {}
    for n in [int(v) for v in os.environ.get("TTC_SIZES", "16000,512,2048,4096,8000,12000").split(",")]:
        walls = []
        for _ in range(2):
            ids = rng.integers(0, 200000, n + 1).tolist()
            M.reset()
            t0 = time.perf_counter()
            M.prefill(ids[:-1])
            t1 = time.perf_counter()
            M.draft(ids, M.rows - 1)
            M.ttnn.synchronize_device(M.dev)
            walls.append([round(t1 - t0, 3), round(time.perf_counter() - t1, 3)])
        t[n] = walls + [round(dram_gb(M.dev), 2)]
        print(f"prefill {n}: [prefill s, MTP catch-up s] {walls}, card memory after it {t[n][-1]} GB", flush=True)
    resets = []
    for _ in range(3):
        t0 = time.perf_counter()
        M.reset()
        resets.append(round(time.perf_counter() - t0, 3))
    res["timing"] = {"prefill_then_catchup_s": t, "reset_s": resets}
    print(f"reset: {resets} s", flush=True)


FIT_SIZES = (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
FIT_PATHS = ("cpu", "split1", "split2", "card")     # engine.use_experts: the lane, 1 / 2 quarters on the card, the card
FIT_CPU_MAX = 3072      # the lane alone loses well before here (2,048 tokens: 3.87 s against 1 quarter on the card's 2.87)


def spec_fit(M, res: dict) -> None:
    """tt_moeoffload/calibrate.py's measurements on this host: each prefill expert path's wall per prompt size (random ids;
    a warm-up, then the faster of two), the prompt loop's 4-row passes over 512 tokens, the reset after a prefill (the
    median over every size and path)."""
    rng = np.random.default_rng(0)
    walls, resets = {}, []
    for path in FIT_PATHS:
        M.force_experts = path
        for n in FIT_SIZES:
            if path == "cpu" and n > FIT_CPU_MAX:
                continue
            ids = rng.integers(0, 200000, n).tolist()
            t = []
            for rep in range(3):
                t0 = time.perf_counter()
                M.reset()
                if rep:
                    resets.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                M.prefill(ids)
                M.ttnn.synchronize_device(M.dev)
                t.append(time.perf_counter() - t0)
            walls.setdefault(path, {})[n] = round(min(t[1:]), 3)
            print(f"fit {path} {n}: {[round(x, 3) for x in t]} s", flush=True)
    M.force_experts = os.environ.get("TTC_FORCE_EXPERTS") or None
    loops = []
    ids = rng.integers(0, 200000, 512).tolist()
    for _ in range(3):
        M.reset()
        t0 = time.perf_counter()
        for p in range(0, 512, 4):
            M.feed(ids[p:p + 4], p, ids[:p])
            M.keep(4)
        M.ttnn.synchronize_device(M.dev)
        loops.append((time.perf_counter() - t0) / 512)
    res["fit"] = {"prefill_s": walls, "step_s": round(min(loops[1:]), 5), "reset_s": round(float(np.median(resets)), 3),
                  "lane_threads": M.lane.threads}
    print(f"fit: loop {res['fit']['step_s'] * 1e3:.2f} ms a token, reset {res['fit']['reset_s']} s", flush=True)


def spec_conv(M, res: dict) -> None:
    """Conversation snapshots (2026-11-08): A (LONG_PROMPT's prompt) answered, another prompt read (A snapshotted),
    then A continued (A, its answer, 16 more tokens) restored from the snapshot, against the same continuation with no
    switch between: tokens and kept rows bit for bit; the snapshot's and the restore's ms; the host's RSS over more
    switches than CONV_SLOTS (each snapshot's host tensors reused or freed: it stays bounded)."""
    from engine import CONV_SLOTS

    def rss_gb():
        return round(next(int(x.split()[1]) for x in open("/proc/self/status") if x.startswith("VmRSS")) / 1e6, 1)
    ids = dict(spec_prompts())[LONG_PROMPT]
    rng = np.random.default_rng(1)
    rows, out = Rows(M), Lines()
    eng = Engine(M, out, queue.Queue())

    def fresh():
        M.reset()
        eng.pool += [c[1] for c in eng.convs]       # the slots' host tensors kept, as the engine does
        eng.held, eng.dirty, eng.ckpts, eng.convs, rows.at = [], False, [], [], {}
    fresh()
    a = request(eng, out, 32, ids)
    cont = ids + a["tokens"] + ids[100:116]
    rows.at = {}
    b = request(eng, out, 32, cont)                    # no switch: the held state continued
    b_rows = dict(rows.at)
    fresh()
    request(eng, out, 32, ids)
    other = rng.integers(0, 200000, 3000).tolist()
    request(eng, out, 4, other)                        # the switch: A snapshotted
    snap = eng.convs[0][1] if eng.convs else {}
    rows.at = {}
    c = request(eng, out, 32, cont)                    # A back from its snapshot
    same_rows = all(b_rows.get(p) == v for p, v in rows.at.items()) and set(b_rows) == set(rows.at)
    r = {"held_a": len(ids) + len(a["tokens"]) - 1, "resumed": c["resumed"], "from": eng.req.get("from"),
         "tokens_same": b["tokens"] == c["tokens"], "rows_same": same_rows,
         "snapshot_ms": round(snap.get("copy_ms", 0), 1), "snapshot_mb": round(snap.get("mb", 0)),
         "restore_ms": round(eng.req.get("restore_ms", 0), 1), "prompt_ms": round(c["prompt_ms"], 1),
         "no_switch_prompt_ms": round(b["prompt_ms"], 1)}
    # A back with another reply than it generated (as a client re-renders it): restored to its checkpoint, against the
    # same prompt read with no switch (the checkpoint restored as it stands)
    alt = ids + ids[200:240] + ids[100:116]
    fresh()
    request(eng, out, 32, ids)
    rows.at = {}
    d = request(eng, out, 32, alt)
    d_rows, d_from = dict(rows.at), eng.req.get("from")
    fresh()
    request(eng, out, 32, ids)
    request(eng, out, 4, other)
    rows.at = {}
    e = request(eng, out, 32, alt)
    r.update({"ck_resumed": e["resumed"], "ck_from": eng.req.get("from"), "ck_no_switch_from": d_from,
              "ck_tokens_same": d["tokens"] == e["tokens"],
              "ck_rows_same": set(d_rows) == set(rows.at) and all(d_rows[p] == v for p, v in rows.at.items()),
              "ck_prompt_ms": round(e["prompt_ms"], 1), "ck_no_switch_prompt_ms": round(d["prompt_ms"], 1),
              "snapshot_alloc_ms": round(snap.get("alloc_ms", 0), 1)})
    n = int(os.environ.get("TTC_CONV_SWITCHES", CONV_SLOTS + 2))
    for key, on in (("rss_gb_per_switch", True), ("rss_gb_per_switch_no_snapshots", False)):
        eng.can_conv, rss = on, [rss_gb()]          # the same switches without snapshots: the prefill's own growth
        for _ in range(n):                          # more conversations than slots
            request(eng, out, 2, rng.integers(0, 200000, 2100).tolist())
            rss.append(rss_gb())
        r[key] = rss
    r["slots"] = len(eng.convs)
    r["pass_"] = bool(c["resumed"] == r["held_a"] and r["from"] == "conversation" and r["tokens_same"] and same_rows
                      and len(eng.convs) <= CONV_SLOTS and r["ck_from"] == "conversation's checkpoint"
                      and r["ck_resumed"] == d["resumed"] and r["ck_tokens_same"] and r["ck_rows_same"])
    res["conv"] = r
    print("conv: " + json.dumps(r), flush=True)


def spec_pieces(M, res: dict) -> None:
    """Long prompts on a large build (2026-11-08, serving 128K): reads from 0 of TTC_FIT's sizes whole (random ids;
    the first that runs out of card memory ends the list), then through the engine a prompt of the context but 64
    (TTC_PIECE: the pieces' size, default 65,536) and 8 tokens generated."""
    from g7_prefill import dram_gb
    rng = np.random.default_rng(0)
    fit, M.piece = {}, 0
    for n in [int(v) for v in os.environ.get("TTC_FIT", "65536,98304").split(",") if v]:
        M.reset()
        t0 = time.perf_counter()
        try:
            M.prefill(rng.integers(0, 200000, n).tolist())
            fit[n] = {"s": round(time.perf_counter() - t0, 1), "card_gb_after": round(dram_gb(M.dev), 2)}
        except RuntimeError as e:
            fit[n] = {"failed": str(e).splitlines()[0][:200]}
        print(f"whole read {n}: {fit[n]}", flush=True)
        if "failed" in fit[n]:
            break
    M.piece = int(os.environ.get("TTC_PIECE", 65536))
    M.reset()
    out = Lines()
    eng = Engine(M, out, queue.Queue())
    ids = rng.integers(0, 200000, M.ctx - 64).tolist()
    r = request(eng, out, 8, ids)
    long = {"prompt": len(ids), "piece": M.piece, "prompt_s": round(r["prompt_ms"] / 1e3, 1),
            "generated": len(r["tokens"]), "read": eng.req["read"], "card_gb_after": round(dram_gb(M.dev), 2)}
    print(f"pieces: {long}", flush=True)
    res["pieces"] = {"fit": fit, "long": long,
                     "pass_": len(r["tokens"]) == 8 and eng.req["read"][:1] == [f"prefill 0..{len(ids) - 1}"]}


if __name__ == "__main__":
    {"spec": main_spec, "spec-ref": main_spec_ref}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
