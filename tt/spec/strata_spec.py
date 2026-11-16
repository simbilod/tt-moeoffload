"""Strata's speculation sources for the card engine: the suffix (prompt-lookup) drafter, the draft policy and the cost-model
controller, bound with ctypes to Strata's own C++ (third_party/strata/src/spec/, read-only) through
tt/spec/spec_capi.cpp, so the drafts and picks are Strata's, not a re-implementation. `StrataWindow` (below) is the
exception: a Python restatement of generate.cpp's window rules, derived from Strata (Copyright (c) 2026 Niko1221 and
the Strata contributors, MIT License, LICENSES/Strata-MIT.txt).

    bash tt/spec/build.sh            # -> tt/spec/build/libstrata_spec.so (TT_MOEOFFLOAD_SPEC_LIB overrides)

How Strata's serving loop (third_party/strata/src/program/generate.cpp, the serial window loop) uses them, which
`StrataWindow` below reproduces for a replay:
  * --suffix-draft 3 (the default): SuffixDrafter(min_match 3, max_match 64); history = prompt + every committed
    token, including the window's last output (the next window's row 0);
  * --spec S with the suffix drafter on: the MTP keeps windows of S rows (S - 1 drafts) and a lookup window may be
    up to S + 2 rows (generate.cpp: o.mtp_max_t = S, o.spec = min(S + 2, 8)); DraftPolicy(S + 2);
  * the MTP window: T = 1 + the leading drafts whose draft-layer probability is >= --spec-min-p;
  * a lookup proposal of S + 1 tokens is considered only when its first token equals the MTP's first draft, and taken
    when DraftPolicy.choose says so; after the round, DraftPolicy.observe(lookup, T, accepted, match, round_ms).
"""
from __future__ import annotations

import ctypes as C
import os

LIB = os.environ.get("TT_MOEOFFLOAD_SPEC_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "libstrata_spec.so"))
_I32P = C.POINTER(C.c_int32)
_lib = None


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL(LIB)
        V, P, I, D, U64 = None, C.c_void_p, C.c_int, C.c_double, C.c_uint64
        IP, DP = C.POINTER(C.c_int), C.POINTER(C.c_double)
        sig = {
            "sd_new": (P, [I, I, U64]), "sd_free": (V, [P]), "sd_reset": (V, [P]), "sd_append": (V, [P, _I32P, U64]),
            "sd_propose": (I, [P, I, _I32P]), "sd_propose_after": (I, [P, _I32P, I, I, _I32P]),
            "sd_last_match": (I, [P]), "sd_size": (U64, [P]),
            "sd_propose_sources": (I, [P, _I32P, I, I, I, I, _I32P, IP, IP]),
            "dp_new": (P, [I, D]), "dp_free": (V, [P]), "dp_choose": (V, [P, I, I, I, IP, IP]),
            "dp_observe": (V, [P, I, I, I, I, D]), "dp_chain": (I, [P, I, D, I, I]),
            "dp_observe_chain": (V, [P, I, I, I, I, D]), "dp_chain_rate": (D, [P, I]),
            "dp_lookup_rate": (D, [P, I]), "dp_cost_ms": (D, [P, I]),
            "ct_new": (P, [DP, D, D]), "ct_free": (V, [P]), "ct_choose": (V, [P, I, I, I, IP, DP]),
            "ct_observe": (V, [P, I, I, D, D, I, I]), "ct_mtp_accept": (D, [P, I]), "ct_lookup_accept": (D, [P, I]),
            "ct_step_ms": (D, [P, I, I]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _lib = L
    return _lib


def _arr(xs):
    xs = list(xs)
    return (C.c_int32 * max(1, len(xs)))(*xs), len(xs)


class SuffixDrafter:
    """strata::spec::SuffixDrafter (include/strata/spec/suffix_drafter.hpp)."""

    def __init__(self, min_match: int = 3, max_match: int = 32, capacity: int = 1 << 19):
        self.h = lib().sd_new(min_match, max_match, capacity)

    def __del__(self):
        if getattr(self, "h", None) and _lib is not None:
            _lib.sd_free(self.h)
            self.h = None

    def reset(self):
        lib().sd_reset(self.h)

    def append(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        a, n = _arr(tokens)
        if n:
            lib().sd_append(self.h, a, n)

    def propose(self, max_k: int) -> list[int]:
        out = (C.c_int32 * max(1, max_k))()
        k = lib().sd_propose(self.h, max_k, out)
        return list(out[:k])

    def propose_after(self, extra, max_k: int) -> list[int]:
        a, n = _arr(extra)
        out = (C.c_int32 * max(1, max_k))()
        k = lib().sd_propose_after(self.h, a, n, max_k, out)
        return list(out[:k])

    def propose_sources(self, tail, n_pending: int, max_k: int, min_match: int):
        """propose_from_sources(PromptLookupSource(this), ...) with no extra sources: (drafts, match, source)."""
        a, n = _arr(tail)
        out = (C.c_int32 * max(1, max_k))()
        m, s = C.c_int(0), C.c_int(-1)
        k = lib().sd_propose_sources(self.h, a, n, n_pending, max_k, min_match, out, C.byref(m), C.byref(s))
        return list(out[:k]), m.value, s.value

    def last_match(self) -> int:
        return lib().sd_last_match(self.h)

    def size(self) -> int:
        return lib().sd_size(self.h)


class DraftPolicy:
    """strata::spec::DraftPolicy (include/strata/spec/draft_policy.hpp)."""

    def __init__(self, max_t: int, margin: float = 0.03):
        self.h = lib().dp_new(max_t, margin)

    def __del__(self):
        if getattr(self, "h", None) and _lib is not None:
            _lib.dp_free(self.h)
            self.h = None

    def choose(self, t_mtp: int, lookup_k: int, match: int) -> tuple[bool, int]:
        lk, t = C.c_int(0), C.c_int(0)
        lib().dp_choose(self.h, t_mtp, lookup_k, match, C.byref(lk), C.byref(t))
        return bool(lk.value), t.value

    def observe(self, lookup: bool, t: int, accepted: int, match: int, round_ms: float):
        lib().dp_observe(self.h, int(lookup), t, accepted, match, round_ms)

    def chain(self, t_mtp: int, p_mtp: float, k_avail: int, match: int) -> int:
        return lib().dp_chain(self.h, t_mtp, p_mtp, k_avail, match)

    def observe_chain(self, t_mtp: int, k: int, accepted: int, match: int, round_ms: float):
        lib().dp_observe_chain(self.h, t_mtp, k, accepted, match, round_ms)

    def chain_rate(self, match: int) -> float:
        return lib().dp_chain_rate(self.h, match)

    def lookup_rate(self, match: int) -> float:
        return lib().dp_lookup_rate(self.h, match)

    def cost_ms(self, t: int) -> float:
        return lib().dp_cost_ms(self.h, t)


class Controller:
    """strata::spec::Controller (include/strata/spec/controller.hpp). cost: None (Strata's defaults) or a dict of
    CostModel fields (dense_ratio and distinct_ratio as 9-lists)."""
    SOURCES = ("none", "lookup", "mtp")
    DEFAULT = dict(dense_ms=11.0, dense_ratio=[1.0, 1.05, 1.3, 1.45, 1.85, 2.2, 2.6, 3.0, 3.3], cpu_all_miss_ms=15.8,
                   hit_rate=0.55, distinct_ratio=[1.0, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80, 5.2],
                   extra_use_cost=0.25, sync_ms=2.4, mtp_draft_ms=1.2, lookup_draft_ms=0.01)

    def __init__(self, cost: dict | None = None, min_gain: float = 0.05, ema: float = 0.05):
        p = None
        if cost is not None:
            c = dict(self.DEFAULT, **cost)
            v = [c["dense_ms"], *c["dense_ratio"], c["cpu_all_miss_ms"], c["hit_rate"], *c["distinct_ratio"],
                 c["extra_use_cost"], c["sync_ms"], c["mtp_draft_ms"], c["lookup_draft_ms"]] + [0.0] * 9
            p = (C.c_double * len(v))(*v)
        self.h = lib().ct_new(p, min_gain, ema)

    def __del__(self):
        if getattr(self, "h", None) and _lib is not None:
            _lib.ct_free(self.h)
            self.h = None

    def choose(self, lookup_available: int, lookup_match: int, mtp_ready: bool) -> dict:
        oi, od = (C.c_int * 2)(), (C.c_double * 2)()
        lib().ct_choose(self.h, lookup_available, lookup_match, int(mtp_ready), oi, od)
        return dict(source=self.SOURCES[oi[0]], k=oi[1], expected_tokens=od[0], tokens_per_ms=od[1])

    def observe(self, choice: dict, accepted: int, lookup_match: int):
        lib().ct_observe(self.h, self.SOURCES.index(choice["source"]), choice["k"], choice["expected_tokens"],
                         choice["tokens_per_ms"], accepted, lookup_match)

    def mtp_accept(self, position: int) -> float:
        return lib().ct_mtp_accept(self.h, position)

    def lookup_accept(self, match: int) -> float:
        return lib().ct_lookup_accept(self.h, match)

    def step_ms(self, k: int, mtp: bool) -> float:
        return lib().ct_step_ms(self.h, k, int(mtp))


class StrataWindow:
    """The window choice of Strata's serial serving loop (generate.cpp, the `while (!pl_ran ...)` loop) for one
    request: which rows the next verify pass carries. MTP drafts and their probabilities come from the caller.

        w = StrataWindow(spec=4, prompt=ids)          # --spec 4, --suffix-draft 3, --spec-min-p 0.30
        rows, info = w.window(x, drafts, probs)       # x = the next token (row 0), drafts = the MTP's chain
        ...verify: a = accepted drafts, outputs = the a + 1 committed tokens...
        w.commit(outputs, info, a, round_ms)
    """

    def __init__(self, spec: int = 4, prompt=(), suffix_draft: int = 3, spec_min_p: float = 0.30, use_mtp=True,
                 use_sfx=True, policy: DraftPolicy | None = None):
        self.S_mtp = spec
        self.S = min(spec + 2, 8) if suffix_draft > 0 and spec >= 2 else spec
        self.min_p, self.use_mtp, self.use_sfx = spec_min_p, use_mtp, use_sfx and suffix_draft > 0
        self.sfx = SuffixDrafter(max(1, suffix_draft), 64, 1 << 20)
        self.sfx.append(list(prompt))
        self.policy = policy if policy is not None else DraftPolicy(self.S)
        self.first = True

    def window(self, x: int, drafts, probs):
        """-> (rows, info): rows = [x] + the drafts this pass verifies."""
        T = self.S_mtp if self.use_mtp else 1
        if self.use_mtp and self.min_p > 0.0:
            T = 1
            while T < self.S_mtp and T - 1 < len(probs) and probs[T - 1] >= self.min_p:
                T += 1
        if self.first:
            T = 1
        from_sfx, match, sbuf = False, 0, []
        if self.use_sfx and not self.first:
            sbuf = self.sfx.propose(self.S - 1)
            match = self.sfx.last_match()
            if sbuf and (not self.use_mtp or sbuf[0] == drafts[0]):
                lk, t = self.policy.choose(T, len(sbuf), match)
                if lk:
                    T, from_sfx = t, True
        rows = [x] + [(sbuf if from_sfx else drafts)[i - 1] for i in range(1, T)]
        return rows, dict(T=T, from_sfx=from_sfx, match=match, timed=not self.first)

    def commit(self, outputs, info, accepted: int, round_ms: float):
        """outputs: the a + 1 tokens the pass committed (its outputs at rows 0..a); the last is the next row 0."""
        if self.use_sfx:
            self.sfx.append(list(outputs))
        if info["timed"]:
            self.policy.observe(info["from_sfx"], info["T"], accepted, info["match"], round_ms)
        self.first = False
