"""Flash-Next's card engine (tt/flashnext/g4_model.Model: the decode token's 26 programs on the p150a, the routed
experts on the CPU lane) behind the line protocol of Strata's engine (`strata --serve`), so that Strata's own API server
(third_party/strata/serve/server.py: the chat template, the tokenizer, tool calls, reasoning, streaming, /health,
/metrics, timings) serves it unchanged, as it serves Strata's own engine. tt_moeoffload/launch.py runs this inside the
tt-metal container.

The protocol (one request at a time; stdout carries only these lines, everything else goes to stderr):
  out  INFO k=v ...                         facts about the engine
  out  READY <ctx> stop                     built; the context; STOP honoured mid-request
  in   GEN <max_new> [key=value ...] <ids>  the prompt's token ids, comma-separated; keys: temperature, top_p, top_k,
                                            min_p, seed, penalty_repeat/freq/present, penalty_last_n, logit_bias
                                            (others, e.g. ckpt, pin, cvec, are accepted and ignored)
  out  RESUME <n>                           the first n prompt tokens are the state already held (not read again)
  out  PP <read> <total> <ms> <tok/s>       prompt progress (also the server's heartbeat; none during a prefill)
  out  T <id>                               a generated token (an end-of-turn token is sent, then DONE)
  out  DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> 0 0 <reused> 0 0 0 0 0 <prompt_read>
  in   STOP                                 end the request now (DONE follows)
  in   QUIT                                 close the card and exit
  out  ERR <message>                        the request failed (no DONE)

A prompt is read by the card's chunked prefill (tt/flashnext/g7_prefill.Prefill, chunks of 8,192 tokens, the
experts streamed from pinned host memory) up to its last token, its states handed off into the decode Model's tensors,
then the last token through the traced decode step (its top 64: the first pick). The prefill starts only from position
0 (its states cannot be read back from the decode's), so it is used when it is cheaper than the decode loop over the
tokens not held (use_prefill, the measured costs below): short prompts and short follow-ups stay on the decode loop.
The prefill needs no reset before it (the hand-off writes every state the decode reads below the prompt's end; the
cells past it are written by the decode before they are read: check_card.py, bit for bit). The state the last request
left (its prompt and every generated token fed back) is kept: a prompt that starts with all of it continues from there
(RESUME) unless re-reading it all through the prefill is cheaper; any other prompt read by the decode loop resets the
state in place first (g4_gate.reset: zeros written into the existing tensors, ~1.5 s) and is read from position 0. The
DeltaNet layers' recurrent state cannot be rewound, so a prompt sharing only part of the held tokens is read again in
full. decode_ms runs from the first generated token to the last (llama.cpp's convention: the first token comes out of
the prompt's last step).

Sampling: greedy without a temperature; with one, the 64 most likely tokens (top_k 1..64, else 64, as Strata's engine),
temperature, min_p, top_p, a seeded draw. Penalties (llama.cpp's rule over the last penalty_last_n tokens) and
logit_bias apply first. On the card the step reads only the head's top 64 (head mode ids); a pick that could
need a token outside them (Sampler's rule) launches the logits head once more for that step (~1.8 ms), so every pick
is the one the full logits give.

Speculative decoding (--spec-rows 2..4, the profile's ttcard.spec): SpecModel instead of CardModel (the verify
build, the MTP draft layer, the prefill handing off into the verify build). The engine runs passes: the prompt but its
last token in teacher-forced passes of up to 4 rows (or the prefill), then each pass verifies [the last token + up to
rows - 1 MTP drafts] and keeps the rows Sampler.accept takes (greedy: up to the first draft that differs from its row's
pick; sampled: rejection sampling with point-mass drafts, the same distribution as one-row picks), cut at max_new and
at an end-of-turn token. Past the MTP's dense attention (8,799 cells) the passes have one row.

  engine.py --serve [--fake] [--model-dir D] [--max-context N]     the line protocol on stdin/stdout (N: the build's
                                                                    context, TT_CTX: 8K steps up to 64K)
            [--spec-rows R --spec-head fr|full]                     speculative decoding (fake: drafts too)
  engine.py --dump-logits --ids IDS.i32 --out OUT.f32 [...]         every position's logits (parity)
  --fake: random logits, no card, no numpy (tests, the request path off the card; no speed claims from it)
"""
from __future__ import annotations

import argparse
import array
import heapq
import math
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import queue
import random
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.append(TT)                # tune (in the container tt/ is on PYTHONPATH already)
from tune import tuned  # noqa: E402  ($TT_MOEOFFLOAD_DATA/tune.json: the cost constants below for this host)
MODEL_DIR = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/qwen3.8-flash-next-ud-iq4xs/UD-IQ4_XS")   # engine.py --model-dir sets it
VOCAB = 248320
EOS_IDS = (248044, 248046)        # <|endoftext|>, <|im_end|>: Strata's engine default --eos-ids
TURN = 248045                     # <|im_start|>: a prompt's last one opens the new assistant turn (Strata's --turn-token)
# prompt checkpoints (2026-11-07, Strata's prompt cache): the recurrent states saved where a prompt's last turn opens,
# restored for a later prompt that shares the prefix through that token but not everything held (a reply sent back
# re-rendered, a regenerate, an edit of the last turn); CKPT_SLOTS kept, newest first, for prefixes of at least
# CKPT_MIN tokens whose turn opens within CKPT_TAIL of the prompt's end (the decode loop reads the tail after it)
CKPT_SLOTS, CKPT_MIN, CKPT_TAIL = int(os.environ.get("TTC_CKPT_SLOTS", 2)), 256, 64
# conversation snapshots (2026-11-08): the whole state held (the checkpoint's tensors + the K/V caches, pooled keys and
# the MTP's) copied to the host when a request switches away from a conversation of at least CONV_MIN tokens, restored
# when a later prompt continues it (an agent client's sessions alternate: each switch back read the conversation from 0, ~30 s at
# 30K); CONV_SLOTS kept, newest first
CONV_SLOTS, CONV_MIN = int(os.environ.get("TTC_CONV_SLOTS", 3)), 2048
TOP_K_MAX = 64                    # Strata's sampled path keeps at most 64 candidates
PP_EVERY_S = 1.0                  # a PP line at least this often while a prompt is read
# the prompt path's costs on the card (check_card.py's prefill timing): a prefill's fixed part (the expert
# stream, 43 layers over PCIe, and the op launches), its marginal rate, a second chunk's fixed part; the decode step;
# the state reset
PREFILL_S, PREFILL_TOK_S, CHUNK_S, STEP_S, RESET_S = (tuned(k, v) for k, v in (
    ("PREFILL_S", 4.4), ("PREFILL_TOK_S", 3000.0), ("CHUNK_S", 3.0), ("STEP_S", 0.023), ("RESET_S", 1.5)))
# Each chunk's attention covers every cell before it (selection + flash attention): 60,000 tokens took 67.3 s
# warm against 45.4 by the terms above (g7_prefill.py time, ctx 64K), 8,192 7.4 s
PREFILL_QUAD_S = tuned("PREFILL_QUAD_S", 6e-9)                 # per token squared
# a follow-up continues the held state through the prefill (2026-11-06) from a multiple of 4 cells (whole pools; 128,
# the pooled keys' tile, until 2026-11-10: up to 127 tokens through the decode loop first, ~9.5 ms each), with at
# least EXTEND_MIN tokens left (Gdn.conv_state reads 3)
EXTEND_ALIGN, EXTEND_MIN = 4, 32
# a read longer than PREFILL_PIECE tokens goes through the prefill in pieces (the first from 0, the rest continuing the
# held state, as a follow-up's extension): one read's card memory grows with its length, and past ~what fits beside the
# 128K build a prompt read whole ran out of it (2026-11-08: 65,536 fits, 98,304 not). 0: whole. Not bit for bit a
# whole read (the extension's: KL 0.005 against it, llama.cpp 0.017)
PREFILL_PIECE = int(os.environ.get("TTC_PREFILL_PIECE", 65536))
# an extension's own fixed cost over prefill_s's (the states read and committed on the host, the held prefix staged):
# 30K held + 128 / 1,024 took 5.7 / 6.0 s with the card's experts against prefill_s's 4.5 / 4.8, 2.2 / 4.45 with the
# CPU's against 0.6 / 2.9 (check_card.py spec extend_time, TTC_FORCE_EXPERTS, 2026-11-06, untraced runs); traced, on
# the 128K build, 2,304 held + 128 / 448 / 896: 0.85 / 1.39 / 2.09 s against 0.47 / ~0.98 / ~1.75 from 0 (resume 0.3 of
# it; a trace key's first two uses ~0.3 s more, 2026-11-10)
EXTEND_S = tuned("EXTEND_S", 0.4)
# a prefill of at most CPU_PREFILL_MAX new tokens takes the routed experts on the CPU lane (g7_prefill.CpuExperts) instead
# of streaming every expert layer over PCIe (~3 s a prompt), on any held prefix: at a 64K spec build (traced prefill,
# torch on one thread) 0.54 / 1.48 / 2.81 s for 128 / 512 / 1,024 tokens from 0 against the card's 4.47 at 1,024,
# 4.5 against 5.07 at 1,536, 5.7 against 5.12 at 2,048; on 30K held + 128 / 1,024 tokens 2.2 / 4.45 against 5.7 / 6.0
# (check_card.py spec timing / extend_time, TTC_FORCE_EXPERTS, 2026-11-06). The lane's batched rows (301e1f8): from 0
# 0.40 / 1.14 / 2.1 / 4.3 s at 100 / 500 / 1,000 / 2,000 tokens (g7_prefill.py traced), the card's 5.12 at 2,048
# Since the lane's buffers are kept (g7_experts.HostRows / HostOut, 2026-11-06) 1,000 / 2,000 tokens take 1.90 / 3.65 s.
# Refit 2026-11-11 (128K spec build, check_card.py spec timing, TTC_FORCE_EXPERTS): the lane
# 0.45 / 1.08 / 1.92 / 2.97 / 3.87 s at 128 / 512 / 1,024 / 1,536 / 2,048 tokens (compute-bound: about 11-06's); one
# group on the card (floor ~1.25 s) wins from ~830 tokens (768: 1.67 against ~1.5; 1,024: 1.59 against 1.92).
CPU_PREFILL_MAX, CPU_PREFILL_S, CPU_PREFILL_TOK_S = (tuned(k, v) for k, v in (
    ("CPU_PREFILL_MAX", 832), ("CPU_PREFILL_S", 0.17), ("CPU_PREFILL_TOK_S", 0.0018)))
# past it, up to SPLIT's bounds, the card's experts with `groups` quarters of the experts on the card and the
# rest on the CPU lane beside it (only the card's groups stream): from 0 at a 64K build, 1 group 1.95 / 2.44 / 2.93 s
# at 1,024 / 1,536 / 2,048 tokens, 2 groups 3.71 / 4.44 s at 3,072 / 4,096 against the card's 4.63 / 4.89, the card's
# 5.32 at 6,144 against 6.38 (g7_prefill.py time W51_CARD_GROUPS, 2026-11-06). (bound, groups, fixed s, s per token).
# The host's: its lane against the card's PCIe stream (tt_moeoffload/calibrate.py re-derives them).
# Refit 2026-11-11 (as CPU_PREFILL_MAX's): 1 group 1.59 / 2.41 / 2.87 / 4.20 s at 1,024 / 1,536 / 2,048 / 3,072, 2 groups
# 3.16 / 3.36 / 4.27 / 6.52 at 2,048 / 3,072 / 4,096 / 6,144, the card alone 4.31 / 4.67 / 4.01 / 4.09 / 4.46 / 5.25 at
# 1,024 .. 8,192; bounds where the measured lines cross (the card's prefill got faster since 11-06, the lane did not).
SPLIT = tuned("SPLIT", [(2304, 1, 0.38, 0.00124), (3840, 2, 1.04, 0.00086)])


# a cache of the experts of most prompt routes in card memory for short reads (2026-11-10): up to CPU_PREFILL_MAX new
# tokens the card computes the cached ones beside the lane (which takes the rest) instead of the lane all; TTC_PREFILL_
# CACHE slots (0: off), ranked by the census's prompt route counts, a layer's share in whole groups of 32; released
# before a read of more than CACHE_DROP tokens (its card memory), loaded again by the next short read
PREFILL_CACHE = int(os.environ.get("TTC_PREFILL_CACHE", 0))
CACHE_DROP = int(os.environ.get("TTC_CACHE_DROP", 16384))   # with 1,504 resident (4.4 GB) up to 24,576 fit (32,768 not)
CENSUS = os.path.join(DATA, "census", "qwen3.8-flash-next", "census.json")   # the model's routing census


def use_experts(m, k: int) -> None:
    """m.P.experts for a prefill of k new tokens: up to CPU_PREFILL_MAX the card's cached experts beside the lane
    (when there is a cache) or the CPU lane's (CpuExperts), then the card's with SPLIT's share on the lane, then the
    card's alone. m.force_experts ("cache", "cpu", "card", "split<quarters>", the warm-up's and check_card.py's
    choice) overrides."""
    f = m.force_experts
    E = m.experts["card"]
    E.use_cache = False
    if E.cache and (f == "cache" or (f is None and k <= CPU_PREFILL_MAX)):
        E.load_cache()
        E.lane, E.card_groups, E.use_cache = m.lane, 0, True
        m.P.experts = E
        return
    if E.cache and k > CACHE_DROP:
        E.drop_cache()
    if f == "cpu" or (f is None and k <= CPU_PREFILL_MAX):
        m.P.experts = m.experts["cpu"]
        return
    g = int(f[5:]) if f and f.startswith("split") else (
        next((g for bound, g, _, _ in SPLIT if k <= bound), None) if f is None else None)
    E = m.experts["card"]
    E.lane, E.card_groups = (m.lane, g * E.n_groups // 4) if g else (None, E.n_groups)   # g in quarters, of NG groups
    m.P.experts = E
# beside the verify build and the MTP (4.4 GB) the prefill streams every expert layer, the Q8_0 ones too (5 of
# them resident took 11.65 GB of the card's 34.1, and the prefill then fit only with 1 stream slot and 4K chunks)
SPEC_RESIDENT = []
CACHE_DIR = os.environ.get("TTC_CACHE_DIR")   # g5_cache.py's slots, the verify pass's card experts (WORKER_L1_MC)
PREFILL_CHUNK = 8192              # tokens per prefill chunk (the user's choice)
PREFILL_PAD = 1024                # the last chunk rounded up to this: 16 shapes up to 16K (a new one compiles, ~27 s
                                  # once per machine; 1024 more rows cost ~0.35 s)
# the prefill's traces (g7_prefill.Prefill.seg: ~1 MB each, 60 per row bucket from position 0, more for follow-ups'
# prefix lengths), beside the decode's; past it the rest are issued op by op
PREFILL_TRACE_MB = int(os.environ.get("TTC_PREFILL_TRACE_MB", 512))


def log(msg: str) -> None:
    print(f"[ttcard] {msg}", file=sys.stderr, flush=True)


# stacks of a hung engine (2026-11-07): py-spy (third_party/tt-pydeps/bin, tt/env/requirements.txt) may attach to any
# card process (tt/sitecustomize.py: PR_SET_PTRACER_ANY; from inside the card container only), so tools/card_stack.sh
# can dump it, and serve() starts stall_watch.py, a separate process (a hung native call holds the GIL: a watchdog
# thread in this process never ran)
PY_SPY = os.path.join(TT, "..", "third_party", "tt-pydeps", "bin", "py-spy")
STALL_DUMP_S = float(os.environ.get("TTC_STALL_DUMP_S", 60))   # a request silent this long: its stacks to the log



# ------------------------------------------------------------------------------------------------ logits
class ListLogits:
    """A plain list of floats (the fake model)."""

    def __init__(self, values: list[float]):
        self.v = values

    def add(self, i: int, delta: float) -> None:
        self.v[i] += delta

    def get(self, i: int) -> float:
        return self.v[i]

    def set(self, i: int, x: float) -> None:
        self.v[i] = x

    def argmax(self) -> int:
        return max(range(len(self.v)), key=self.v.__getitem__)

    def topk(self, k: int) -> list[tuple[int, float]]:
        return heapq.nlargest(k, enumerate(self.v), key=lambda iv: iv[1])

    def copy(self) -> "ListLogits":
        return ListLogits(list(self.v))

    def tobytes(self) -> bytes:
        return array.array("f", self.v).tobytes()


class TorchLogits:
    """A 1-D float32 torch tensor (the card's head)."""

    def __init__(self, t):
        self.t = t

    def add(self, i: int, delta: float) -> None:
        self.t[i] += delta

    def get(self, i: int) -> float:
        return float(self.t[i])

    def set(self, i: int, x: float) -> None:
        self.t[i] = x

    def argmax(self) -> int:
        return int(self.t.argmax())

    def topk(self, k: int) -> list[tuple[int, float]]:
        import torch
        v, i = torch.topk(self.t, k)
        return list(zip(i.tolist(), v.tolist()))

    def copy(self) -> "TorchLogits":
        return TorchLogits(self.t.clone())

    def tobytes(self) -> bytes:
        return self.t.numpy().astype("float32").tobytes()


# ------------------------------------------------------------------------------------------------ sampling
class Top:
    """The card's top 64 for one step (head mode ids): ids best first (a tie to the lowest id) and their logits, and
    full(): the step's full logits (the logits head launched again, untraced, on the same X; valid until the next
    step)."""

    def __init__(self, ids: list[int], vals: list[float], full):
        self.ids, self.vals, self.full = ids, vals, full

    def tobytes(self) -> bytes:
        return self.full().tobytes()


def i_of(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        raise BadRequest(f"GEN: {s!r} is not an integer") from None


def f_of(s: str, what: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise BadRequest(f"GEN: {what} is not a number") from None
    if not math.isfinite(v):
        raise BadRequest(f"GEN: {what} is not finite")
    return v


class Sampler:
    """One request's sampling settings (the GEN line's keys) and its random stream. From the card's top 64 (Top) a pick
    is exact when it needs no token outside them: greedy, top_k <= 64, min_p and top_p within the top k, penalties that
    only lower logits (repeat >= 1, freq and presence >= 0) and bans, as long as the adjusted k-th candidate stays
    above the 64th logit (any token outside can only be at or below it); positive biases only on tokens inside. Any
    other step reads the full logits (Top.full) and counts in `fallbacks`."""

    def __init__(self, keys: dict[str, str], full: bool = False, vocab: int | None = None):
        """Raises BadRequest on a value that is not a finite number or a bias outside the vocabulary (before the
        request touches the held state)."""
        f = lambda k, d: f_of(keys[k], f"{k}={keys[k]}") if k in keys else d  # noqa: E731
        i = lambda k, d: i_of(keys[k]) if k in keys else d  # noqa: E731
        self.temperature = f("temperature", 0.0)
        self.top_p = f("top_p", 1.0)
        self.min_p = f("min_p", 0.0)
        top_k = i("top_k", 0)
        self.top_k = top_k if 1 <= top_k <= TOP_K_MAX else TOP_K_MAX
        self.repeat, self.freq = f("penalty_repeat", 1.0), f("penalty_freq", 0.0)
        self.present = f("penalty_present", 0.0)
        self.last_n = i("penalty_last_n", 64)
        self.bias = {}
        for item in filter(None, keys.get("logit_bias", "").split(",")):
            t, _, b = item.partition(":")
            tok = i_of(t)
            if tok < 0 or (vocab is not None and tok >= vocab):
                raise BadRequest(f"GEN: logit_bias token {t} is outside the vocabulary")
            self.bias[tok] = f_of(b, item)
        seed = i("seed", 0)
        self.rng = random.Random(seed) if seed > 0 else random.Random()
        self.full, self.fallbacks = full, 0         # full: every step from the full logits (the checks)

    def _counts(self, recent: list[int]) -> dict[int, int]:
        counts: dict[int, int] = {}
        if self.repeat != 1.0 or self.freq or self.present:
            for t in recent[-self.last_n:] if self.last_n > 0 else []:
                counts[t] = counts.get(t, 0) + 1
        return counts

    def _adjust(self, t: int, x: float, counts: dict[int, int]) -> float:
        """Token t's logit x after the penalties (llama.cpp's rule) and logit_bias."""
        c = counts.get(t)
        if c:
            x = (x / self.repeat if x > 0 else x * self.repeat) - c * self.freq - self.present
        b = self.bias.get(t)
        if b is not None:
            x = -math.inf if b <= -100 else x + b
        return x

    def _from_top(self, top: Top, counts: dict[int, int]):
        """The pick from the top 64 (a token, or the candidates to draw from), or None when it could need a token
        outside them."""
        if counts and (self.repeat < 1.0 or self.freq < 0 or self.present < 0):
            return None
        inside = set(top.ids)
        if any(b > 0 and t not in inside for t, b in self.bias.items()):
            return None
        cand = [(t, self._adjust(t, v, counts)) for t, v in zip(top.ids, top.vals)]
        k = 1 if self.temperature <= 0.0 else self.top_k
        if any(a != v for (_, a), v in zip(cand, top.vals)):
            orig = dict(zip(top.ids, top.vals))
            cand.sort(key=lambda iv: (-iv[1], iv[0]))
            t, a = cand[k - 1]
            # at the 64th logit only a token left unchanged ranks before those outside (their ids are higher)
            if not (a > top.vals[-1] or a == top.vals[-1] == orig[t]):
                return None
        return cand[0][0] if k == 1 else cand[:k]

    def _cands(self, logits, recent: list[int]):
        """The pick from `logits` (Top, ListLogits or TorchLogits; not changed: penalties and biases apply to a copy):
        a token when no draw is needed, else the candidates to draw from (best first)."""
        counts = self._counts(recent)
        if isinstance(logits, Top):
            t = None if self.full else self._from_top(logits, counts)
            if t is not None:
                return t
            self.fallbacks += 1
            logits = logits.full()
        if counts or self.bias:
            logits = logits.copy()
            for t in set(counts) | set(self.bias):
                logits.set(t, self._adjust(t, logits.get(t), counts))
        if self.temperature <= 0.0:
            return logits.argmax()
        return sorted(logits.topk(self.top_k), key=lambda iv: (-iv[1], iv[0]))

    def pick(self, logits, recent: list[int]) -> int:
        """The next token from `logits`."""
        c = self._cands(logits, recent)
        return c if isinstance(c, int) else self._draw(c)

    def accept(self, rows: list, drafts: list[int], recent: list[int]) -> list[int]:
        """A verify pass's tokens: rows[m] the logits after the pass's row m (row m + 1 is drafts[m]). Greedy: each
        row's pick, up to the first that differs from the draft after it. Sampled: speculative rejection sampling with
        point-mass drafts (Strata's spec_prob.hpp with no q list): draft d after row m kept with probability p(d),
        else the row's token drawn from p without d, where p is the distribution pick() draws from; the last row's
        token (the bonus) a plain pick. Every token is distributed as pick()'s (the same top 64 and settings)."""
        out: list[int] = []
        for m, logits in enumerate(rows):
            r = recent + out
            if m == len(drafts) or self.temperature <= 0.0:
                out.append(self.pick(logits, r))
                if m == len(drafts) or out[-1] != drafts[m]:
                    break
                continue
            c = self._cands(logits, r)
            d = drafts[m]
            if isinstance(c, int):
                out.append(c)
                if c != d:
                    break
                continue
            toks, p = self._dist(c)
            pd = sum(x for t, x in zip(toks, p) if t == d) / sum(p)
            if self.rng.random() < pd:
                out.append(d)
                continue
            rest = [(t, x) for t, x in zip(toks, p) if t != d]
            out.append(self._walk([t for t, _ in rest], [x for _, x in rest]))
            break
        return out

    def _dist(self, cand: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
        """Temperature, min_p and top_p over the candidates (best first): the tokens kept and their weights."""
        top = cand[0][1]
        w = [math.exp((v - top) / self.temperature) for _, v in cand]
        total = sum(w)
        p = [x / total for x in w]
        keep = len(p)
        if self.min_p > 0.0:
            keep = sum(1 for x in p if x >= self.min_p * p[0]) or 1
        if self.top_p < 1.0:
            acc = 0.0
            for i in range(keep):
                acc += p[i]
                if acc >= self.top_p:
                    keep = i + 1
                    break
        return [t for t, _ in cand[:keep]], p[:keep]

    def _draw(self, cand: list[tuple[int, float]]) -> int:
        """A seeded draw over the candidates (best first)."""
        return self._walk(*self._dist(cand))

    def _walk(self, toks: list[int], p: list[float]) -> int:
        r = self.rng.random() * sum(p)
        for t, x in zip(toks, p):
            r -= x
            if r <= 0.0:
                return t
        return toks[-1]


# ------------------------------------------------------------------------------------------------ models
class FakeModel:
    """Random logits (deterministic per token and position: 64 random tokens stand out of zeros), no card: the request
    path off the card."""
    extends = True                             # prefill(ids, start): a follow-up continues the held state

    def __init__(self, vocab: int = VOCAB, ctx: int = 16384, ms: float = 0.0, seed: int = 0, top: bool = False,
                 rows: int = 1, ckpt: bool = False):
        self.vocab, self.ctx, self.ms, self.seed, self.top = vocab, ctx, ms, seed, top   # top: Top, as the card
        self.prefilled: list[int] = []
        self.resets = 0
        if ckpt:                               # prompt checkpoints (SpecModel's save / restore): recorded only
            self.saves, self.restores = [], 0
            self.save = lambda seq, spare=None, full=False: \
                self.saves.append(("conv", len(seq)) if full else len(seq) - 1) or {"at": len(seq) - 1, "full": full}
            self.restore = lambda snap: setattr(self, "restores", self.restores + 1)
        if rows > 1:                           # passes of `rows` rows, as SpecModel (feed, keep, draft)
            self.rows, self.passes = rows, []

    def reset(self) -> None:
        self.resets += 1

    def prefill(self, ids: list[int], start: int = 0) -> None:
        """The card's prefill of ids from position 0, or from `start` on the held state (the logits depend on token and
        position only: nothing kept)."""
        self.prefilled.append((start, len(ids)) if start else len(ids))

    def step(self, tok: int, pos: int, prev: list[int]) -> ListLogits | Top:
        r = random.Random(self.seed * 1_000_003 + pos * 524_287 + tok)
        if self.ms:
            time.sleep(self.ms / 1e3)
        v = [0.0] * self.vocab
        for _ in range(64):
            v[r.randrange(self.vocab)] = r.gauss(4.0, 2.0)
        if not self.top:
            return ListLogits(v)
        ids = heapq.nsmallest(TOP_K_MAX, range(self.vocab), key=lambda i: (-v[i], i))
        return Top(ids, [v[i] for i in ids], lambda: ListLogits(list(v)))

    def feed(self, toks: list[int], pos: int, prev: list[int]) -> list:
        """A pass: each row's logits."""
        self.passes.append(len(toks))
        return [self.step(t, pos + m, prev + toks[:m]) for m, t in enumerate(toks)]

    def keep(self, n: int) -> None:
        pass

    def draft(self, seq: list[int], steps: int) -> list[int]:
        """The greedy continuation of seq, one draft in three wrong (rejections)."""
        out, tok = [], seq[-1]
        for k in range(steps):
            lg = self.step(tok, len(seq) - 1 + k, [])
            tok = ((lg.argmax() if isinstance(lg, ListLogits) else lg.ids[0]) + (len(seq) % 3 == k)) % self.vocab
            out.append(tok)
        return out

    def close(self) -> None:
        pass


class CardModel:
    """g4_model.Model with the CPU lane, built once in head mode ids (the step reads the head's top 64, not its
    logits): one plain token (it loads every program's binaries), the token's capture as one trace, then the states
    reset in place (g4_gate.py's order); step() replays the trace and returns a Top. A step that needs the full
    logits launches the logits head (a second program of the same Head, untraced: ~1.8 ms) on the same X.
    Beside it the chunked prefill (g7_prefill.Prefill with the card's experts: 5 layers resident, the others streamed
    from pinned host memory on command queue 2), warmed once (its first run compiles); prefill() reads ids from
    position 0 into the Model's states."""

    def __init__(self, model_dir: str = MODEL_DIR, prefill: bool = True):
        sys.path.insert(0, TT)
        sys.path.insert(0, os.path.join(TT, "flashnext"))
        import g7_prefill                      # before ttnn: it sets the pinned-memory cache for the expert stream
        import ttnn
        from cpulane import Lane
        from g2_layers import Gguf
        from g3_attn1 import CTX
        from g3_layer import WORKER_L1
        from g4_gate import reset
        from g4_host import Host
        from g4_model import Model
        self.ttnn, self._reset = ttnn, reset
        cpus = sorted(os.sched_getaffinity(0))
        t0 = time.time()
        self.lane = Lane()
        self.lane.load(model_dir)
        log(f"lane: experts loaded in {time.time() - t0:.0f} s ({self.lane.threads} threads on {len(cpus)} CPUs)")
        self.host = Host(model_dir)
        self.history = self.host.n_gram - 1     # the PLE hash reads this many predecessors
        g = Gguf(model_dir)
        self.dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=(64 + PREFILL_TRACE_MB) << 20,
                                         worker_l1_size=WORKER_L1,
                                         num_command_queues=2)       # the prefill streams experts on the second
        self.vocab, self.ctx = VOCAB, CTX
        try:
            t0 = time.time()
            self.M = Model(self.dev, g, self.lane, log=log, head_mode="ids")
            self.logits_head = self.M.head.program()
            tok = 760
            self.M.step(self.host.embed(tok), self.host.ple(tok, []), 0)
            self.M.capture()
            self.logits_head()                 # its first launch loads its binaries
            reset(self.M)
            log(f"model built and captured in {time.time() - t0:.0f} s ({len(self.M.segs)} programs per token)")
            if prefill:
                t0 = time.time()
                from g7_experts import CardExperts
                # past 16K cells the 5 resident expert layers (11.65 GB) leave the long prefill no room
                resident = g7_prefill.resident(model_dir) if CTX <= 16384 else []
                self.P = g7_prefill.Prefill(self.dev, g, CardExperts(self.dev, resident, log=log),
                                            chunk=PREFILL_CHUNK, log=log)
                self.experts = {"card": self.P.experts, "cpu": g7_prefill.CpuExperts(self.dev, self.lane)}
                self.P.pad = PREFILL_PAD
                # each traced row bucket issued (it compiles), then captured; the first one through both experts' paths
                for b in self.P.trace_t:
                    for self.force_experts in (("card", "split1", "cpu") if b == self.P.trace_t[0] else
                                               ("cpu", "cpu")):
                        self.prefill(list(range(1000, 1000 + b - 8)))
                self.force_experts = None
                reset(self.M)
                log(f"prefill built and warmed in {time.time() - t0:.0f} s")
            else:                              # the logit dump reads every position through the step
                self.prefill = None
        except BaseException:
            ttnn.close_mesh_device(self.dev)
            raise

    def reset(self) -> None:
        self._reset(self.M)

    extends = True                             # prefill(ids, start): continues the held state from start
    cpu_prefill = True                         # up to CPU_PREFILL_MAX new tokens: the experts on the CPU lane
    force_experts = None                       # "card" / "cpu" / "split<g>": the warm-up's choice (use_experts)

    def prefill(self, ids: list[int], start: int = 0) -> None:
        """ids read by the prefill from position 0, or from `start` on the state held for ids[:start] (a follow-up),
        their states into the Model (the next step at len(ids))."""
        if len(ids) >= self.ctx:
            raise RuntimeError(f"a prefill of {len(ids)} tokens outside the card's context ({self.ctx})")
        use_experts(self, len(ids) - start)
        self.P.prefill(ids, self.host, into=self.M, start=start)   # 256K plan step 1
        self.P.hand_off(self.M)

    def step(self, tok: int, pos: int, prev: list[int]) -> Top:
        if not 0 <= pos < self.ctx:              # nothing below this stops a write past the attention caches
            raise RuntimeError(f"position {pos} outside the card's context ({self.ctx})")
        h = self.host
        ids, vals = self.M.step_traced(h.embed(tok), h.ple(tok, prev[-self.history:]), pos, top=True)
        return Top(ids.tolist(), vals.tolist(), self.full)

    def full(self) -> TorchLogits:
        """The last step's full logits."""
        self.logits_head()
        return TorchLogits(self.M.head.get_logits().float())

    def close(self) -> None:
        self.ttnn.close_mesh_device(self.dev)


class SpecModel:
    """Speculative decoding on the card. The verify build (g4_model.Model(m_max = 4): passes of up to 4 rows at
    positions pos .., one traced build, head mode ids), the MTP draft layer (g6_mtp.MtpCard with the profile's head)
    and g4_model.Spec's draft steps (traced), beside the chunked prefill (its hand-off writes the verify build's
    states: g7_prefill.Prefill.hand_off). The one-row build is not built beside them: it would not fit in card memory
    with the prefill and the MTP, and its device takes the other L1 split (WORKER_L1, not WORKER_L1_M).

    The engine drives passes: feed(toks, pos, prev) verifies the rows [the last token, drafts ..] (the DeltaNet states
    first commit the rows the previous pass kept: keep(n)); draft(seq, steps) catches the MTP's K/V up over the cells
    whose target rows are pending (cell c: X after layer 47 at c with the token at c + 1, read from seq) and chains up
    to `steps` drafts from the last of them. A cell is caught up only once its next token is known, so a follow-up's
    first new token is the one the last held cell gets. After a prefill the prompt's rows are pending (X after the
    last layer at the MTP window's positions: its dense attention reads the last g6_mtp.WINDOW cells (8,800; Strata's
    --mtp-window), so there are drafts at any position (before 2026-11-06: none past 8,799 cells, one-row passes).
    A step that needs the full logits launches the verify head's logits program once for the
    pass (every row)."""

    step_s = tuned("SpecModel.step_s", 0.0095)   # a prompt read in 4-row passes: ~38 ms a pass (T(4) + the MTP's K/V)
    reset_s = tuned("SpecModel.reset_s", 2.2)    # V.reset + the MTP K/V zeroed: 1.75-3.2 s (check_card.py spec)

    def __init__(self, model_dir: str = MODEL_DIR, rows: int = 4, head: str = "fr"):
        sys.path.insert(0, TT)
        sys.path.insert(0, os.path.join(TT, "flashnext"))
        import g7_prefill                      # before ttnn: it sets the pinned-memory cache for the expert stream
        import numpy as np
        import ttnn
        from cpulane import Lane
        from g2_layers import Gguf
        from g3_attn1 import CTX
        from g3_layer import WORKER_L1_M, WORKER_L1_MC
        from g4_host import Host
        from g4_model import M_MAX, Cache, Model, Spec
        from g6_mtp import CTX_DENSE, WINDOW, MtpCard
        assert 2 <= rows <= M_MAX
        self.ttnn, self.ctx_dense, self.window = ttnn, CTX_DENSE, WINDOW
        cpus = sorted(os.sched_getaffinity(0))
        t0 = time.time()
        self.lane = Lane()
        self.lane.load(model_dir)
        log(f"lane: experts loaded in {time.time() - t0:.0f} s ({self.lane.threads} threads on {len(cpus)} CPUs)")
        self.host = Host(model_dir)
        g = Gguf(model_dir)
        self.dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=(96 + PREFILL_TRACE_MB) << 20,
                                         worker_l1_size=WORKER_L1_MC if CACHE_DIR else WORKER_L1_M,
                                         num_command_queues=2)
        self.vocab, self.ctx, self.rows, self.head = VOCAB, CTX, rows, head
        self.n_keep, self.pending, self.at, self.rows_full = 1, None, None, None
        # Strata's --spec-min-p: drafting stops at a draft the head gives less than this (its top 64's softmax).
        # 0.3 (2026-11-07, spec_probe min_p on the served check's prompts): a pass costs 23.5 / 29.3 / 35.2 / 40.9 ms
        # at 1-4 rows, so a draft pays when it is accepted ~1 time in 4; greedy 16.6 -> 15.5 ms a token, sampled
        # (S3's) 19.6 -> 16.7 (0.5, Strata's: 16.7 / 19.0)
        self.min_p = float(os.environ.get("TTC_SPEC_MIN_P", 0.3))
        try:
            t0 = time.time()
            cache = None
            if CACHE_DIR:                      # The card's share of the verify pass's routed experts
                cache = Cache(self.dev, CACHE_DIR, card_max=int(os.environ.get("TTC_CARD_MAX", 24)))
                log(f"card cache: {cache.slots} slots ({cache.gib:.2f} GiB) in {len(cache.slot_of)} layers, card_max "
                    f"{cache.card_max}")
            V = self.V = Model(self.dev, g, self.lane, log=log, head_mode="ids", m_max=M_MAX, cache=cache)
            warm = [760, 198, 1000, 2000]
            V.warm(np.stack([self.host.embed(t) for t in warm]),
                   np.stack([self.host.ple(t, warm[:m]) for m, t in enumerate(warm)]), 0)
            V.capture()
            self.logits_head = V.head.program()
            self.logits_head()                 # its first launch loads its binaries
            t1 = time.time()
            self.C = MtpCard(self.dev, heads=(head,), log=log)
            self.S = Spec(V, self.C, self.host)
            self.S.capture()
            V.reset()
            log(f"verify model built and captured in {t1 - t0:.0f} s, the draft layer ({head} head) in "
                f"{time.time() - t1:.0f} s ({self.C.memory()['total']} MiB)")
            t0 = time.time()
            from g7_experts import CardExperts
            self.P = g7_prefill.Prefill(self.dev, g, CardExperts(self.dev, SPEC_RESIDENT, log=log),
                                        chunk=PREFILL_CHUNK, log=log)
            self.experts = {"card": self.P.experts, "cpu": g7_prefill.CpuExperts(self.dev, self.lane)}
            if PREFILL_CACHE:
                import json
                counts = {int(k): v for k, v in json.load(open(CENSUS))["prompt"].items()}
                self.P.experts.set_cache(PREFILL_CACHE, counts)
                log(f"prefill cache: {PREFILL_CACHE} experts in {len(self.P.experts.cache)} layers "
                    f"({self.P.experts.cache_bytes / 1e9:.2f} GB)")
            self.P.pad = PREFILL_PAD
            self.P.rows_card = True            # the MTP's pending rows stay on the card (draft's page copies)
            # the catch-up's K/V over a prompt's rows in blocks (0: the 4-row programs for every row, 0.44 s at 6,300)
            self.KV = g7_prefill.MtpKv(self.dev, self.C.A1, self.host) if os.environ.get("TTC_MTP_KV", "1") == "1" \
                else None
            # each traced row bucket issued (it compiles), then captured; the first one through both experts' paths
            first = ("card", "split1", "cpu") + (("cache", "cache") if PREFILL_CACHE else ())
            for b in self.P.trace_t:
                for self.force_experts in (first if b == self.P.trace_t[0] else
                                           ("cache", "cache") if PREFILL_CACHE else ("cpu", "cpu")):
                    self.prefill(list(range(1000, 1000 + b - 8)))
            self.force_experts = None
            if self.KV is not None:
                self.KV.warm()
            self.reset()
            log(f"prefill built and warmed in {time.time() - t0:.0f} s (card memory "
                f"{g7_prefill.dram_gb(self.dev):.1f} GB)")
        except BaseException:
            ttnn.close_mesh_device(self.dev)
            raise

    def reset(self) -> None:
        self.V.reset()                         # the MTP's K/V not zeroed either: its window reads only cells caught up in
        self.n_keep = 1                        # this sequence (after a prefill from its rows_from: the window's floor)
        self.set_pending(None)

    def set_pending(self, p) -> None:
        """The MTP's pending rows: (first cell, rows) with rows numpy [n, 10240] or, after a prefill, on the card
        (Prefill.rows_card's (first cell, tensor, n)); a card tensor dropped is freed here."""
        old, self.pending = self.pending, p
        if old is not None and isinstance(old[1], tuple) and (p is None or p[1] is not old[1]):
            self.ttnn.deallocate(old[1][1])

    def pending_host(self):
        """The pending rows with card rows copied to the host (a conversation snapshot keeps them)."""
        if self.pending is None or not isinstance(self.pending[1], tuple):
            return self.pending
        c0, (cf, t, n) = self.pending
        rows = self.ttnn.to_torch(t).reshape(-1, 10240)[c0 - cf:c0 - cf + n].numpy().copy()
        return (c0, rows)

    extends = True                             # prefill(ids, start): continues the held state from start
    cpu_prefill = True                         # up to CPU_PREFILL_MAX new tokens: the experts on the CPU lane
    force_experts = None                       # "card" / "cpu" / "split<g>": the warm-up's choice (use_experts)

    piece = PREFILL_PIECE

    def prefill(self, ids: list[int], start: int = 0) -> None:
        """ids read by the prefill from position 0, or from `start` on the state held for ids[:start] (a follow-up:
        the last pass's kept rows committed from n_keep, the MTP's pending cells caught up with ids' tokens), their
        states into the verify build (the next pass at len(ids), n_keep 1), their rows pending for the MTP; in pieces
        of at most `piece` tokens (PREFILL_PIECE)."""
        if len(ids) >= self.ctx:
            raise RuntimeError(f"a prefill of {len(ids)} tokens outside the card's context ({self.ctx})")
        while self.piece and len(ids) - start > self.piece:
            b = (start + self.piece) // EXTEND_ALIGN * EXTEND_ALIGN
            self._prefill(ids[:b], start)
            start = b
        self._prefill(ids, start)

    def _prefill(self, ids: list[int], start: int) -> None:
        use_experts(self, len(ids) - start)
        if start and self.pending is not None:
            self.draft(ids, 0)                 # the held cells' K/V: their next tokens are ids'
        self.set_pending(None)
        # the MTP reads only its window (g6_mtp.WINDOW cells, block-aligned): its K/V for the prompt's last cells
        self.P.rows_from, self.P.rows_out = max(start, len(ids) - self.window - 32), len(ids)
        self.P.prefill(ids, self.host, into=self.V, start=start, n_keep=self.n_keep)   # 256K plan step 1
        self.P.hand_off(self.V)
        self.n_keep = 1
        if self.P.h_rows is not None:
            self.set_pending((self.P.rows_from, self.P.h_rows))

    def _states(self) -> list:
        """The decode's recurrent state tensors (g7_prefill.state_tensors): the DeltaNet S (the verify build's whole
        tensor: both parities and the pending page) and conv states, the PLE ring, the attention layers' raw-key rings
        (their last RAWK cells, overwritten by a later sequence); not the K/V caches or pooled keys (by position)."""
        from g7_prefill import state_tensors
        st = state_tensors(self.V)
        return st["S"] + st["conv"] + st["ring"] + st["idx"][0::2]

    def _cells(self) -> list:
        """What a conversation snapshot adds to _states: the state by position, the attention layers' K / V caches and
        pooled keys, the MTP's K / V and keys."""
        from g7_prefill import state_tensors
        st, A = state_tensors(self.V), self.C.A1
        return st["kv"] + st["idx"][1::2] + [A.kc, A.vc, A.rawk, A.pooled]

    def save(self, seq: list[int], spare=None, full: bool = False) -> dict:
        """A checkpoint of the state held for seq[:-1], on the host: the MTP's pending cells caught up first (their next
        tokens from seq; the next pass would have, so the continuation is the same), then raw copies of _states into
        host tensors (spare: an evicted checkpoint's, reused). full: a conversation snapshot of the state held for seq,
        _cells too, the MTP's pending cells kept pending."""
        ttnn = self.ttnn
        if not full:
            self.draft(seq, 0)
        t0 = time.perf_counter()
        # shortcut: the whole cache tensors (~2 GB at 64K), not only seq's cells; copy cells 0..len(seq) if it shows
        ts = self._states() + (self._cells() if full else [])
        hs = spare["host"] if spare else [ttnn.allocate_tensor_on_host(t.spec, self.dev) for t in ts]
        t1 = time.perf_counter()
        for t, h in zip(ts, hs):
            ttnn.copy_device_to_host_tensor(t, h, blocking=False)
        ttnn.synchronize_device(self.dev)
        from g6_mtp import nbytes
        return {"host": hs, "parity": self.V.parity, "n_keep": self.n_keep, "copy_ms": (time.perf_counter() - t1) * 1e3,
                "alloc_ms": (t1 - t0) * 1e3, "mb": sum(nbytes(h) for h in hs) / 1e6, "full": full,
                "pending": self.pending_host() if full else None}

    def alloc(self) -> dict:
        """A conversation snapshot's host tensors, allocated and written once (the pages touched): what save() takes
        as spare. The engine makes CONV_SLOTS at start, so no request waits on a ~4 GB allocation (~4 s at 128K)."""
        return {"host": self.save([], full=True)["host"]}

    def restore(self, snap: dict) -> None:
        """The state a save() held, back into the decode's tensors in place (the traces hold their addresses)."""
        ts = self._states() + (self._cells() if snap["full"] else [])
        assert len(ts) == len(snap["host"])
        for t, h in zip(ts, snap["host"]):
            self.ttnn.copy_host_to_device_tensor(h, t)
        self.ttnn.synchronize_device(self.dev)
        self.V.parity, self.n_keep = snap["parity"], snap["n_keep"]
        self.set_pending(snap["pending"])
        self.at, self.rows_full = None, None

    def feed(self, toks: list[int], pos: int, prev: list[int]) -> list[Top]:
        """A verify pass: toks at positions pos .. (prev: every token before pos); each row's top 64."""
        if pos < 0 or pos + len(toks) > self.ctx:   # nothing below this stops a write past the attention caches
            raise RuntimeError(f"positions {pos}..{pos + len(toks) - 1} outside the card's context ({self.ctx})")
        if self.pending is not None:
            self.draft(prev + toks, 0)
        ids, vals = self.V.verify(*self.S.inputs(toks, prev, pos), pos, self.n_keep)
        self.at, self.rows_full = (pos, len(toks)), None
        return [Top(ids[m].tolist(), vals[m].tolist(), lambda m=m: self.full(m)) for m in range(len(toks))]

    def full(self, m: int) -> TorchLogits:
        """Row m's full logits of the last pass (the logits head launched once a pass; valid until the next)."""
        if self.rows_full is None:
            self.logits_head()
            self.rows_full = self.ttnn.to_torch(self.V.head.logits).float().reshape(self.V.m_max, -1)[:, :VOCAB]
        return TorchLogits(self.rows_full[m])

    def keep(self, n: int) -> None:
        """The last pass's first n rows kept: the next pass commits them; their X rows pending for the MTP."""
        from g4_model import rows_of
        (pos, M), S = self.at, self.S
        self.n_keep = n
        if pos + n <= self.ctx_dense:
            self.ttnn.copy_device_to_host_tensor(self.V.x32, S.x_host, blocking=True)
            self.set_pending((pos, rows_of(S.x_np, M)[:n].copy()))

    def draft(self, seq: list[int], steps: int) -> list[int]:
        """The MTP's K/V over the pending cells (their next tokens from seq), then up to `steps` drafts."""
        if self.pending is None:
            return []
        (c0, R), self.pending, out = self.pending, None, []
        card = isinstance(R, tuple)            # on the card: each 4 rows page-copied into the MTP's rin (320 pages a row)
        k = R[2] if card else len(R)
        nxt = seq[c0 + 1:c0 + 1 + k]
        assert len(nxt) == k, (c0, k, len(seq))
        i0 = 0
        if card:
            from g7_prefill import page_copy
            cf, t, _ = R
            if self.KV is not None:            # all but the last 1-4 rows in blocks; those through the programs (drafts)
                i0 = max(0, min((k - 1) // 4 * 4, self.ctx_dense - c0))
                if i0:
                    self.KV(t, cf, c0, i0, nxt)
        for i in range(i0, k, 4):
            m = min(4, k - i)
            if card and c0 + i + m <= self.ctx_dense:      # (past it chain() reads nothing)
                page_copy(self.dev, t, self.C.rin, [((c0 + i - cf) * 320, 0, m * 320)], 128, n_cores=8)
            out = self.S.chain(self.head, c0 + i, None if card else R[i:i + 4], nxt[i:i + 4],
                               steps if i + 4 >= k else 0, min_p=self.min_p)[0]
        if card:
            self.ttnn.deallocate(t)
        return out

    def close(self) -> None:
        self.ttnn.close_mesh_device(self.dev)


# ------------------------------------------------------------------------------------------------ the engine
class Stop(Exception):
    """STOP (or SIGTERM) during a request."""


class BadRequest(ValueError):
    """A GEN the engine refuses (ERR); the state is unchanged."""


class Engine:
    turn = TURN                                   # the token whose last occurrence a prompt checkpoint is saved at

    def __init__(self, model, out, lines: queue.Queue, ctx: int | None = None):
        self.model, self.out, self.lines = model, out, lines
        self.ctx = min(ctx or model.ctx, model.ctx)
        self.turn = getattr(model, "turn", TURN)   # the model's turn opener (GLM's, tt/glm/g7_engine.py), else Flash-Next's
        self.held: list[int] = []        # the tokens whose state the model holds, in order
        self.last = None                 # the logits after the last held token
        self.dirty = False               # the model holds state not described by `held` (an aborted step)
        self.term = False
        self.pending: list[str] = []     # lines that arrived during a request and are not STOP
        self.full_logits = False         # every pick from the full logits (check_card.py's comparison)
        self.prefill = getattr(model, "prefill", None)    # None: every prompt through the decode loop
        self.drafts = getattr(model, "rows", 1) - 1       # drafts a pass verifies (0: one-row passes)
        self.spec = self.drafts > 0                       # passes of up to model.rows rows (SpecModel's feed, keep)
        self.beat = None                                  # serve(): the stall watcher's heartbeat file
        self.req = {"held": 0, "restored": None, "read": [], "saved": None, "passes": 0, "drafted": 0, "accepted": 0}
        self.ckpts: list[tuple[list[int], dict]] = []     # (key, state): the state held for key[:-1], newest first
        self.can_ckpt = CKPT_SLOTS > 0 and hasattr(model, "save")
        self.convs: list[tuple] = []     # (held, state, checkpoint | None): conversation snapshots, newest first
        self.can_conv = CONV_SLOTS > 0 and hasattr(model, "save")
        t0 = time.perf_counter()
        self.pool = [model.alloc() for _ in range(CONV_SLOTS)] if self.can_conv and hasattr(model, "alloc") else []
        if self.pool:
            log(f"{len(self.pool)} conversation snapshot slots allocated on the host in "
                f"{time.perf_counter() - t0:.1f} s")

    @property
    def eos(self) -> tuple:
        """the ids that end a reply: the model's own (GLM's), else Flash-Next's (read at use: tests patch EOS_IDS)"""
        return tuple(getattr(self.model, "eos_ids", None) or EOS_IDS)

    def prefill_s(self, n: int, start: int = 0) -> float:
        """The prefill's estimated seconds for positions start .. n - 2 (all but the prompt's last token)."""
        if hasattr(self.model, "prefill_cost"):          # the model's own measured costs (GLM's)
            return self.model.prefill_cost(n, start)
        k, quad = n - 1 - start, PREFILL_QUAD_S * (n * n - start * start)
        card = PREFILL_S + k / PREFILL_TOK_S + CHUNK_S * ((k - 1) // PREFILL_CHUNK) + quad
        if getattr(self.model, "cpu_prefill", False):
            if k <= CPU_PREFILL_MAX:
                return min(card, CPU_PREFILL_S + k * CPU_PREFILL_TOK_S + quad)
            for bound, _, fixed, per in SPLIT:
                if k <= bound:
                    return min(card, fixed + k * per + quad)
        return card

    def loop_s(self, k: int, reset: bool = False) -> float:
        m = self.model
        return k * getattr(m, "step_s", STEP_S) + (getattr(m, "reset_s", RESET_S) if reset else 0.0)

    def use_prefill(self, n: int, reuse: int, reset: bool = False) -> bool:
        """Whether a prompt of n tokens, reuse of them held, is cheaper read from position 0 by the prefill (all but
        its last token) than by the decode loop from reuse (after a reset when `reset`)."""
        return self.prefill is not None and n > 1 and self.prefill_s(n) < self.loop_s(n - reuse, reset)

    def extend_at(self, n: int, reuse: int) -> int | None:
        """A follow-up of n tokens whose first `reuse` are held: the position the prefill continues the held state from
        (the decode loop reads up to it: a multiple of EXTEND_ALIGN, whole pools), when that beats both the
        decode loop from reuse and the prefill from 0; else None."""
        if not reuse or not getattr(self.model, "extends", False) or self.prefill is None:
            return None
        a = -(-reuse // EXTEND_ALIGN) * EXTEND_ALIGN
        if n - 1 - a < EXTEND_MIN:
            return None
        ext = self.loop_s(a - reuse) + self.prefill_s(n, a) + EXTEND_S
        return a if ext < self.loop_s(n - reuse) and ext < self.prefill_s(n) else None

    def log_request(self, ids, reuse: int, prompt_ms: float, decode_ms: float, gen: int, finish: str, sampler) -> None:
        """One line a request (Strata's 'strata serve: prompt N = R reused + K read ...', with the card's paths): the
        prompt, what it reused (a checkpoint restored), how the rest was read, the checkpoint saved, the decode's
        passes and drafts, the picks that read the full logits."""
        r = self.req
        read = r["read"] + ([f"loop {r['loop']}"] if r.get("loop") else [])
        log(f"request: prompt {len(ids)} = {reuse} reused" + (f" ({r.get('from', 'checkpoint')})" if r["restored"] else "")
 +
            f" + {len(ids) - reuse} read ({', '.join(read) or 'last token'}) in {prompt_ms:.0f} ms"
            + (f", checkpoint saved at {r['saved']}" if r["saved"] else "") +
            f"; {gen} generated in {decode_ms:.0f} ms ({gen / max(decode_ms, 1e-3) * 1e3:.1f} tok/s), "
            f"{r['passes']} passes, drafts accepted {r['accepted']} of {r['drafted']}, full logits "
            f"{sampler.fallbacks}, held {r['held']} before" + (" (snapshotted)" if r.get("stashed") else "") +
            f"; {finish}")

    def checkpoint(self, ids: list[int], reuse: int):
        """The longest checkpoint ids starts with that holds more than reuse does and is still valid: its key a prefix
        of the tokens held (the K/V cells below it this sequence's, and the MTP's last one paired with the key's last
        token), or the state just restored from it."""
        best = None
        for key, snap in self.ckpts if self.can_ckpt else ():
            if (len(key) - 1 > reuse and len(ids) >= len(key) and ids[:len(key)] == key and
                    (self.held[:len(key)] == key or self.held == key[:-1]) and (best is None or len(key) > len(best[0]))):
                best = (key, snap)
        return best

    def save_at(self, ids: list[int], start: int, ext: int | None) -> int | None:
        """Where this prompt's read stops for a checkpoint: its last turn token's position q (the state for ids[:q]),
        when the read passes it (q > start), q >= CKPT_MIN, it is within CKPT_TAIL of the end, an extension that
        starts at ext still has EXTEND_MIN tokens before it, and no checkpoint has that key yet; else None."""
        if not self.can_ckpt:
            return None
        end = len(ids) - 1
        q = next((i for i in range(end - 1, max(start, end - 1 - CKPT_TAIL), -1) if ids[i] == self.turn), None)
        if q is None or q < CKPT_MIN or (ext is not None and q - ext < EXTEND_MIN) or \
                any(key == ids[:q + 1] for key, _ in self.ckpts):
            return None
        return q

    def saved(self, ids: list[int], q: int | None) -> int | None:
        """The checkpoint saved once the read holds ids[:q] (the oldest evicted past CKPT_SLOTS, its host tensors
        reused); None from then on."""
        if not q or len(self.held) != q:
            return q
        spare = self.ckpts.pop()[1] if len(self.ckpts) >= CKPT_SLOTS else None
        t0 = time.perf_counter()
        self.ckpts.insert(0, (ids[:q + 1], self.model.save(ids[:q + 1], spare)))
        self.req["saved"] = q
        if not hasattr(self, "_ckpt_logged"):
            self._ckpt_logged = True
            snap = self.ckpts[0][1]
            log(f"checkpoint at {q} in {(time.perf_counter() - t0) * 1e3:.0f} ms, the copies "
                f"{snap.get('copy_ms', 0):.0f} ms ({snap.get('mb', 0):.0f} MB on the host)")
        return None

    def say(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()
        if self.beat:
            os.utime(self.beat)

    def set_busy(self, busy: bool) -> None:
        """The stall watcher's view: a request running or not (the beat file's first byte; its mtime the last line)."""
        if self.beat:
            with open(self.beat, "w") as f:
                f.write("1" if busy else "0")

    def stopped(self) -> bool:
        """A STOP waiting, or SIGTERM (non-blocking); other lines, and the end of the input (None), are kept for after
        the request."""
        hit = self.term
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                return hit
            if line is not None and line.strip() == "STOP":
                hit = True
            elif line is None or line.strip():
                self.pending.append(line)

    def run_pass(self, x: int, sampler: Sampler, budget: int) -> list[int]:
        """One pass from x (the last token, not yet fed): x and the model's drafts verified (one-row models: x's
        step). Returns the tokens it gives, at most budget and up to the first end-of-turn token; the rows behind them
        are kept and held (x at least)."""
        m, k = self.model, len(self.held)
        self.dirty = True
        if self.spec:
            steps = min(self.drafts, budget - 1, self.ctx - k - 1)
            drafts = m.draft(self.held + [x], steps) if steps > 0 else []
            rows = m.feed([x] + drafts, k, self.held)
        else:
            drafts, rows = [], [m.step(x, k, self.held)]
        out = sampler.accept(rows, drafts, self.held + [x])[:max(budget, 0)]
        out = out[:next((i + 1 for i, t in enumerate(out) if t in self.eos), len(out))]
        n = max(1, len(out))
        r = self.req
        r["passes"], r["drafted"], r["accepted"] = r["passes"] + 1, r["drafted"] + len(drafts), r["accepted"] + n - 1
        if self.spec:
            m.keep(n)
        self.held += ([x] + drafts)[:n]
        self.last = rows[n - 1]
        self.dirty = False
        return out

    def conversation(self, ids: list[int], than: int):
        """The conversation snapshot ids continues furthest, (entry, tokens reused, via its checkpoint), reusing more
        than `than`: its whole held sequence when ids repeats it, else the checkpoint it carries (the state at its last
        prompt's end, valid again once the snapshot's K / V cells are back) when ids starts with that key. A next
        prompt rarely repeats the generated tokens (the client re-renders the reply; a client's template may drop the
        thinking), so most returns go through the checkpoint."""
        best = None
        for c in self.convs:
            key, _, ck = c
            if len(key) < len(ids) and ids[:len(key)] == key:
                n, via = len(key), False
            elif ck is not None and len(ids) >= len(ck[0]) and ids[:len(ck[0])] == ck[0]:
                n, via = len(ck[0]) - 1, True
            else:
                continue
            if n > than and (best is None or n > best[1]):
                best = (c, n, via)
        return best

    def stash(self, ids: list[int], keep=None) -> None:
        """Before a prompt that continues nothing held (no tokens reused, no checkpoint of this sequence on its path):
        the held conversation snapshotted when it is clean and at least CONV_MIN tokens; the snapshots it extends
        dropped, the oldest evicted past CONV_SLOTS (their host tensors reused; never keep's, the one being restored)."""
        key = self.held
        if not self.can_conv or self.dirty or len(key) < CONV_MIN or self.checkpoint(ids, 0) is not None:
            return
        old = [c for c in self.convs if key[:len(c[0])] == c[0] or (c[2] and key[:len(c[2][0])] == c[2][0])]
        if any(len(c[0]) == len(key) and key == c[0] for c in old):
            return                             # restored and not extended: the snapshot holds it already
        # its checkpoint: the longest one key extends, taken out of ckpts (an evicted checkpoint's tensors are reused)
        cks = [(c, True) for c in self.ckpts] + [(o[2], False) for o in old if o[2]]
        ck = max(((c, mine) for c, mine in cks if key[:len(c[0])] == c[0]), key=lambda x: len(x[0][0]), default=None)
        if ck is not None and ck[1]:
            self.ckpts = [c for c in self.ckpts if c is not ck[0]]
        self.convs = [c for c in self.convs if c is keep or all(c is not o for o in old)]   # (identity: no tensor ==)
        free = [c[1] for c in old if c is not keep]
        rest = [i for i, c in enumerate(self.convs) if c is not keep]
        if not free and self.pool:
            free = [self.pool.pop()]
        if not free and len(self.convs) >= CONV_SLOTS and rest:
            free = [self.convs.pop(rest[-1])[1]]
        spare = free[0] if free else None
        self.pool += free[1:]                  # superseded snapshots' tensors kept for later ones (no allocation)
        t0 = time.perf_counter()
        self.convs.insert(0, (list(key), self.model.save(list(key), spare, full=True), ck and ck[0]))
        self.req["stashed"] = len(key)
        if not hasattr(self, "_conv_logged"):
            self._conv_logged = True
            snap = self.convs[0][1]
            log(f"conversation of {len(key)} snapshotted in {(time.perf_counter() - t0) * 1e3:.0f} ms "
                f"({snap.get('mb', 0):.0f} MB on the host" + (f", its checkpoint at {len(ck[0][0]) - 1}" if ck else "")
                + ")")

    def generate(self, max_new: int, keys: dict[str, str], ids: list[int]) -> None:
        t0 = time.perf_counter()
        if not ids:
            raise BadRequest("GEN: an empty prompt")
        if any(t < 0 or t >= self.model.vocab for t in ids):
            raise BadRequest("GEN: a token id outside the vocabulary")
        if len(ids) + max(max_new, 0) > self.ctx:
            raise BadRequest(f"prompt ({len(ids)} tokens) + max_new ({max_new}) exceeds the context ({self.ctx})")
        sampler = Sampler(keys, self.full_logits, self.model.vocab)   # its keys checked before the state is touched
        n = len(self.held)
        reuse = n if not self.dirty and n and len(ids) >= n and ids[:n] == self.held else 0
        self.req = r = {"held": n, "restored": None, "read": [], "saved": None, "passes": 0, "drafted": 0, "accepted": 0}
        ck = self.checkpoint(ids, reuse)
        if ck is not None and self.extend_at(len(ids), len(ck[0]) - 1) is None and \
                self.use_prefill(len(ids), len(ck[0]) - 1):
            ck = None                          # read from 0 by the prefill even with it: no restore
        cv = self.conversation(ids, max(reuse, len(ck[0]) - 1 if ck else 0))
        if cv is not None and self.extend_at(len(ids), cv[1]) is None and self.use_prefill(len(ids), cv[1]):
            cv = None
        if not reuse and ck is None and self.held:
            self.stash(ids, keep=cv and cv[0])  # a switch: the held conversation kept for its return
        if cv is not None:                     # a conversation coming back: its whole state
            (key, snap, cck), _, via = cv
            t1 = time.perf_counter()
            self.model.restore(snap)
            if via:                            # then its checkpoint: the state at its last prompt's end
                self.model.restore(cck[1])
            self.held, self.last, self.dirty = list(cck[0][:-1] if via else key), None, False
            reuse = r["restored"] = len(self.held)
            r["from"] = "conversation's checkpoint" if via else "conversation"
            r["restore_ms"] = (time.perf_counter() - t1) * 1e3
        elif ck is not None:                   # rewind to the checkpoint: the state held for its key[:-1]
            self.model.restore(ck[1])
            self.held, self.last, self.dirty = ck[0][:-1], None, False
            reuse = r["restored"] = len(self.held)
        ext = self.extend_at(len(ids), reuse)
        prefill = ext is None and self.use_prefill(len(ids), reuse, reset=not reuse and bool(self.held or self.dirty))
        if reuse and not prefill:
            self.say(f"RESUME {reuse}")
        elif self.held or self.dirty:
            if not prefill:                    # the prefill needs none (the module's note)
                self.model.reset()
            self.held, self.last, self.dirty, reuse = [], None, prefill, 0
        last_pp, finish, gen = time.perf_counter(), "length", 0
        q = self.save_at(ids, 0 if prefill else reuse, ext)   # the read stops there, the state is saved
        try:
            if prefill:
                if self.stopped():
                    raise Stop
                self.dirty = True
                try:
                    self.prefill(ids[:q] if q else ids[:-1])
                    self.held, self.dirty = ids[:q] if q else ids[:-1], False
                    r["read"].append(f"prefill 0..{len(self.held)}")
                    q = self.saved(ids, q)
                except RuntimeError as e:      # e.g. card memory (TT_FATAL): the decode loop still works
                    log(f"prefill of {len(ids) - 1} tokens failed, the decode loop reads the prompt: {e}")
                    self.model.reset()
                    self.dirty = False
            # a prompt's passes: the model's read_rows when it has them (GLM's verify build costs the same at 2 rows
            # as at 4, so it decodes at 2 and reads at 4), else its decode rows
            rows, end = getattr(self.model, "read_rows", getattr(self.model, "rows", 1)), len(ids) - 1
            while len(self.held) < end:        # all but the last token: teacher-forced passes
                if self.stopped():
                    raise Stop
                k = len(self.held)
                if ext is not None and k == ext:   # the rest through the prefill, from the held state
                    self.dirty = True
                    try:
                        self.prefill(ids[:q] if q else ids[:-1], start=k)
                        self.held, self.dirty = ids[:q] if q else ids[:-1], False
                        r["read"].append(f"extension {k}..{len(self.held)}")
                        q = self.saved(ids, q)
                        continue
                    except RuntimeError as e:  # the state at k untouched until the hand-off: the loop goes on
                        log(f"prefill of {end - k} tokens from {k} failed, the decode loop reads them: {e}")
                        self.dirty, ext = False, None
                toks = ids[k:min([k + rows, end] + [s for s in (ext, q) if s and s > k])]
                self.dirty = True
                if self.spec:
                    self.last = self.model.feed(toks, k, self.held)[-1]
                    self.model.keep(len(toks))
                else:
                    self.last = self.model.step(toks[0], k, self.held)
                self.held += toks
                self.dirty = False
                r["loop"] = r.get("loop", 0) + len(toks)
                q = self.saved(ids, q)
                now = time.perf_counter()
                if now - last_pp >= PP_EVERY_S:
                    ms = (now - t0) * 1e3
                    self.say(f"PP {len(self.held)} {len(ids)} {ms:.1f} "
                             f"{(len(self.held) - reuse) / max(ms, 1e-3) * 1e3:.1f}")
                    last_pp = now
            if self.stopped():
                raise Stop
            if len(self.held) == len(ids):     # the whole prompt held already: the logits after it
                out = [sampler.pick(self.last, self.held)][:max_new]
            else:                              # the last token's pass gives the first tokens
                out = self.run_pass(ids[-1], sampler, max_new)
            ms = (time.perf_counter() - t0) * 1e3
            self.say(f"PP {len(ids)} {len(ids)} {ms:.1f} {(len(ids) - reuse) / max(ms, 1e-3) * 1e3:.1f}")
        except Stop:
            ms = (time.perf_counter() - t0) * 1e3
            self.say(f"DONE 0 {len(ids)} {ms:.1f} 0.0 cancel 0 0 {reuse} 0 0 0 0 0 {len(self.held) - reuse}")
            self.log_request(ids, reuse, ms, 0.0, 0, "cancel", sampler)
            return
        t1 = time.perf_counter()
        while True:
            for tok in out:
                self.say(f"T {tok}")
            gen += len(out)
            if out and out[-1] in self.eos:
                finish = "stop"
                break
            if gen >= max_new or len(self.held) >= self.ctx:
                break
            if self.stopped():
                finish = "stop"
                break
            out = self.run_pass(out[-1], sampler, max_new - gen)
        t2 = time.perf_counter()
        if sampler.fallbacks:
            log(f"{sampler.fallbacks} of {gen} picks read the full logits")
        self.say(f"DONE {gen} {len(ids)} {(t1 - t0) * 1e3:.1f} {(t2 - t1) * 1e3:.1f} {finish} 0 0 {reuse} "
                 f"0 0 0 0 0 {len(ids) - reuse}")
        self.log_request(ids, reuse, (t1 - t0) * 1e3, (t2 - t1) * 1e3, gen, finish, sampler)

    def serve(self) -> int:
        if STALL_DUMP_S > 0 and os.path.exists(PY_SPY):
            import subprocess
            import tempfile
            self.beat = os.path.join(tempfile.gettempdir(), f"ttcard-beat-{os.getpid()}")
            self.set_busy(False)
            subprocess.Popen([sys.executable, os.path.join(HERE, "stall_watch.py"), str(os.getpid()), self.beat, PY_SPY,
                              str(STALL_DUMP_S)])
        self.say(f"INFO engine=ttcard ctx={self.ctx} vocab={self.model.vocab} logit_bias=1")
        self.say(f"READY {self.ctx} stop")
        while not self.term:
            if self.pending:
                line = self.pending.pop(0)
            else:
                try:
                    line = self.lines.get(timeout=0.5)
                except queue.Empty:
                    continue
            if line is None:
                break
            f = line.split()
            if not f or f[0] in ("STOP", "WRAP"):   # WRAP: Maya's server stopping (close the reasoning): nothing to do
                continue
            if f[0] == "QUIT":
                break
            if f[0] != "GEN" or len(f) < 3:
                self.say(f"ERR unsupported command: {line.strip()[:80]}")
                continue
            try:
                max_new = int(f[1])
                keys = dict(kv.split("=", 1) for kv in f[2:-1])
                ids = [int(t) for t in f[-1].split(",") if t]
            except ValueError:
                self.say(f"ERR malformed GEN line: {line.strip()[:80]}")
                continue
            self.set_busy(True)
            try:
                self.generate(max_new, keys, ids)
            except BadRequest as e:
                self.say(f"ERR {e}")
            except Exception as e:             # logged and answered, and the engine serves on (the state marked
                import traceback               # dirty: the next request resets it). Raising out of serve() made
                log(f"request failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")   # main's close() hang on
                self.say(f"ERR the engine failed: {type(e).__name__}: {str(e)[:200]}")       # the card, silently
                self.dirty = True
            finally:
                self.set_busy(False)
        return 0


def read_lines(stream, out: queue.Queue) -> None:
    for line in stream:
        out.put(line)
    out.put(None)


def dump_logits(model, ids_path: str, out_path: str, start: int = 0) -> None:
    """Every position's logits for the ids (int32), teacher-forced from position 0: f32 [n, vocab], position-major.
    start: ids[:start] read by the prefill first (as served), the rows from start on (a long reference's tail)."""
    ids = array.array("i")
    with open(ids_path, "rb") as f:
        ids.frombytes(f.read())
    if len(ids) > model.ctx:
        raise SystemExit(f"{len(ids)} ids exceed the card's context ({model.ctx})")
    t0 = time.time()
    held: list[int] = list(ids[:start])
    if start:
        model.prefill(held)
        log(f"dump: prefill of {start} in {time.time() - t0:.0f} s")
    with open(out_path, "wb") as out:
        if not hasattr(model, "step"):     # a verify-pass model (GLM's): passes of its rows, all kept
            rows = getattr(model, "read_rows", model.rows)
            for pos in range(start, len(ids), rows):
                toks = list(ids[pos:pos + rows])
                for t in model.feed(toks, pos, held):
                    out.write(t.tobytes())
                model.keep(len(toks))
                held += toks
                if pos // 256 != (pos + len(toks)) // 256:
                    log(f"dump: position {pos + len(toks)} of {len(ids)}")
        for pos, tok in enumerate(ids[start:] if hasattr(model, "step") else [], start):
            out.write(model.step(tok, pos, held).tobytes())
            held.append(tok)
            if pos % 256 == 255:
                log(f"dump: position {pos + 1} of {len(ids)}")
    log(f"dump: {len(ids)} positions in {time.time() - t0:.0f} s -> {out_path}")


def cpu_lacks(record: str, cpuinfo: str = "/proc/cpuinfo") -> list[str]:
    """the CPU flags of a library's build host (record: one per line, tt/cpulane/build.sh) that this CPU lacks"""
    with open(cpuinfo) as f:
        here = next((ln.split(":", 1)[1].split() for ln in f if ln.startswith("flags")), [])
    with open(record) as f:
        return sorted(set(f.read().split()) - set(here))


def data_problems(model: str, model_dir: str | None, spec: bool, prefill: bool) -> list[str]:
    """What must be on disk before the card opens, each missing piece with the command that makes it (the engine would
    otherwise fail minutes into the build, deep in a loader). Optional pieces log a warning: without the routing census
    the lane's warm is off; GLM's layers missing from its card expert cache prefill on the CPU lane, slower."""
    import glob
    glm = model == "glm"
    if glm:
        sys.path.insert(0, os.path.join(TT, "glm"))
        import g6_experts
        model_dir = model_dir or g6_experts.MODEL
    model_dir = model_dir or MODEL_DIR
    out = []
    if not glob.glob(os.path.join(model_dir, "*.gguf")):
        out.append(f"no GGUF in {model_dir}: run tt_moeoffload/prepare.py {model}, or point --model-dir at the folder "
                   f"of the model's shards")
    lib = "libcpulane_glm.so" if glm else "libcpulane.so"
    path = os.path.join(TT, "cpulane", "build", lib)
    if not os.path.exists(path):
        out.append(f"tt/cpulane/build/{lib} is not built: run tt/env/setup.sh")
    elif not os.path.exists(path + ".cpu"):
        log(f"warning: tt/cpulane/build/{lib} has no record of the CPU it was built for (-march=native): rebuild it "
            "with tt/env/setup.sh so the data check can tell")
    elif lacks := cpu_lacks(path + ".cpu"):
        out.append(f"tt/cpulane/build/{lib} was built (-march=native) on a CPU with flags this one lacks "
                   f"({', '.join(lacks[:8])}{', ...' if len(lacks) > 8 else ''}): rebuild it here with tt/env/setup.sh")
    if out:
        return out                                        # what follows reads the GGUF
    build = "python3 tt_moeoffload/launch.py --engine tt/{} -- build"
    if glm:
        missing = [il for il in g6_experts.CARD_LAYERS if not g6_experts.built(il)]
        if missing:
            log(f"warning: GLM's card expert cache ({g6_experts.SHAPE.out_dir()}) lacks {len(missing)} of "
                f"{len(g6_experts.CARD_LAYERS)} layers: their prefill runs on the CPU lane, slower. Build it with: "
                + build.format("glm/g6_experts.py"))
        return out
    if spec:
        rt = os.path.join(DATA, "strata", "mtp", "rt")
        if not all(os.path.exists(os.path.join(rt, f)) for f in ("dense.txt", "dense.bin", "experts.bin")):
            out.append(f"the MTP draft layer's runtime files are not in {rt} (dense.txt, dense.bin, experts.bin: "
                       f"made by Strata's tools/mtp_rt.py); without them run with --spec-rows 0")
        if not os.path.exists(os.path.join(TT, "spec", "build", "libstrata_spec.so")):
            out.append("tt/spec/build/libstrata_spec.so is not built: run tt/env/setup.sh")
    if prefill:
        sys.path.insert(0, TT)
        sys.path.insert(0, os.path.join(TT, "flashnext"))
        import g7_experts
        from expert_types import card_refusals
        out += card_refusals(model_dir)
        S = g7_experts.FLASHNEXT
        folder = S.out_dir()
        n = sum(not os.path.exists(os.path.join(folder, f"blk.{il}.p{j}.dn_s.npy"))
                for il in S.layers for j in range(S.NPIECE))
        if n:
            out.append(f"the card expert cache {folder} lacks {n} of {len(S.layers) * S.NPIECE} pieces: build it "
                       f"with " + build.format("flashnext/g7_experts.py"))
    census = os.path.join(DATA, "census", "qwen3.8-flash-next", "census.trace")
    if "G4_WARM" not in os.environ and "G4_WARM_CENSUS" not in os.environ and not os.path.exists(census):
        log(f"warning: no routing census at {census}: the lane's warm is off (G4_WARM=0)")
        os.environ["G4_WARM"] = "0"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true", help="the line protocol on stdin/stdout")
    mode.add_argument("--dump-logits", action="store_true", help="every position's logits of --ids into --out")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--dump-from", type=int, default=0, help="--dump-logits: the prefill reads ids up to it first")
    ap.add_argument("--model-dir", default=None, help="the folder of the model's GGUF shards (default: under the data root)")
    ap.add_argument("--max-context", type=int, default=None,
                    help="the attention caches' cells: the build's context (TT_CTX, 8192 steps up to 262144)")
    ap.add_argument("--fake", action="store_true", help="random logits, no card")
    ap.add_argument("--fake-vocab", type=int, default=VOCAB)
    ap.add_argument("--fake-ms", type=float, default=0.0, help="the fake model's milliseconds per step")
    ap.add_argument("--fake-ctx", type=int, default=16384)
    ap.add_argument("--spec-rows", type=int, default=0, help="speculative decoding: rows a pass verifies (2..4; 0 off)")
    ap.add_argument("--spec-head", default="fr", help="the MTP draft head: fr (the subset head) or full")
    ap.add_argument("--model", choices=("flashnext", "glm"), default="flashnext",
                    help="glm: GLM-5.3-Flash (tt/glm/g7_engine.GlmModel: speculative, --spec-rows 2..4, default 2)")
    a, _ = ap.parse_known_args(argv)        # the launcher's own options may follow
    if a.max_context and not a.fake and a.model == "flashnext":   # The caches built for it (g3_attn1.CTX)
        os.environ["TT_CTX"] = str(a.max_context)
    # the protocol owns stdout: everything else printed (tt-metal's logs, prints) goes to stderr
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    lines: queue.Queue = queue.Queue()
    state = {"engine": None}

    def term(*_):                          # PID 1 in the container ignores SIGTERM without a handler
        lines.put(None)
        if state["engine"] is not None:
            state["engine"].term = True
    signal.signal(signal.SIGTERM, term)
    spec = a.spec_rows > 1 and not a.dump_logits         # the logit dump reads every position through the step
    if a.model_dir:                                       # every module's default model folder reads it
        os.environ["TT_MOEOFFLOAD_MODEL_DIR"] = a.model_dir
    if not a.fake:
        found = data_problems(a.model, a.model_dir, spec, prefill=spec or not a.dump_logits or a.dump_from > 0)
        if found:
            log("not starting, the data is incomplete:\n" + "\n".join(f"  - {p}" for p in found))
            return 2
    if a.model == "glm" and not a.fake:
        sys.path.insert(0, os.path.join(TT, "glm"))
        from g7_engine import GlmModel
        model = GlmModel(a.max_context or 32768, a.spec_rows or 2, log=log, top=Top, logits=TorchLogits)
    else:
        model = None
    model_dir = a.model_dir or MODEL_DIR
    model = model or (FakeModel(a.fake_vocab, a.fake_ctx, a.fake_ms, rows=a.spec_rows if spec else 1) if a.fake else
             SpecModel(model_dir, a.spec_rows, a.spec_head) if spec else
             CardModel(model_dir, prefill=not a.dump_logits or a.dump_from > 0))
    try:
        if a.dump_logits:
            if not (a.ids and a.out):
                raise SystemExit("--dump-logits needs --ids and --out")
            dump_logits(model, a.ids, a.out, a.dump_from)
            return 0
        threading.Thread(target=read_lines, args=(sys.stdin, lines), daemon=True).start()
        eng = state["engine"] = Engine(model, proto, lines, a.max_context)
        return eng.serve()
    finally:
        # a close that hangs (the card wedged) ends the process anyway: no silent engine left behind (2026-11-07)
        bound = threading.Timer(30.0, lambda: (log("close: no answer from the card in 30 s, exiting"), os._exit(1)))
        bound.daemon = True
        bound.start()
        model.close()
        bound.cancel()
        log("closed")


if __name__ == "__main__":
    raise SystemExit(main())
