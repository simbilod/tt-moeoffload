"""GLM-5.3-Flash behind tt/serve/engine.py's Engine (the line protocol of `strata --serve`, its
sampler, its pass loop), as Flash-Next's SpecModel is: the model side only. Project Maya's API server (its GLM chat
template, tokenizer, tool calls, reasoning) runs tt_moeoffload/launch.py as its
engine, which runs `engine.py --model glm` in the tt-metal container.

GlmModel is g5_spec.Spec (the trunk's verify build at M_MAX = 4 rows and the MTP block, one trace each) and
g6_prefill.Prefill (the prompt in chunks, handed off into the verify build), with SpecModel's interface:
  feed(toks, pos, prev)  a verify pass of toks at pos .. (the MTP first caught up over the rows pending for it):
                         each row's top 64 (Top), full(m) its fp32 logits
  keep(n)                the pass's first n rows kept (the MLA host states rewound past them, the KDA layers commit
                         them in the next pass); their hidden rows pending for the MTP
  draft(seq, steps)      the MTP over the pending rows (their next tokens from seq), then up to `steps` drafts
  prefill(ids, start)    ids by the prefill, its states handed off (the next pass at len(ids)); the MTP's caches by
                         g6_prefill.MtpPrefill over rows 0 .. n - 2, row n - 1 pending for it. The prefill reads up
                         to a = n rounded down to ALIGN, verify passes the rest (< 32 tokens). Whatever start the
                         engine asks, the read continues the prefill's own states where its last read ended (p, a
                         multiple of ALIGN at the previous prompt's end) when ids extend it, else reads from 0: the
                         decoded reply after p is read again (~2.6 ms a token against a verify pass's ~21), and
                         nothing is read back from the decoder's states (its KDA pending rows, the MLA host rings)
Context: the decode build's (G2_CTX, set from --max-context before g2_layers is imported); 32K at most for now (the MLA
query program's SC region holds M_MAX row pages, g3_fmq's shortcut).
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

from tune import tuned   # $TT_MOEOFFLOAD_DATA/tune.json: the cost constants for this host

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = 154880
EOS_IDS = (154820, 154827, 154829)   # <|endoftext|>, <|user|>, <|observation|>: Maya's server's stop ids for GLM
TURN = 154828                        # <|assistant|>: a prompt's last one opens the new assistant turn
CTX_MAX = 32768
CHUNK = int(os.environ.get("GLM_PREFILL_CHUNK", 2048))
STREAM_DENSE = os.environ.get("GLM_STREAM_DENSE", "1") == "1"   # g6_prefill.DenseStream; 0: resident, as before
# the prefill's routed experts on the card (g6_prefill.card_experts) beside the lane: bfp8 prefill weights for
# the room, two command queues (the experts' stream), tt-metal's pinned-memory cache for their host copies
CARD_EXPERTS = os.environ.get("GLM_CARD_EXPERTS", "1") == "1"
if CARD_EXPERTS:
    os.environ.setdefault("GP_WDT", "bfp8")
os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", str(200 << 30))
ALIGN = 32                           # ttKDA's actual_start: a chunk starts 32-aligned (else off from blk.0 on)
WARM_PROMPT = 160                    # the warm-up prompt's tokens (every program compiled before READY)


class GlmModel:
    vocab, eos_ids, turn = VOCAB, EOS_IDS, TURN
    extends = True                   # prefill(ids, start): a follow-up from the prefill's own last read (below)
    cpu_prefill = False
    # a prompt read in 4-row verify passes (read_rows): ~83 ms a pass (g5_verify.py; at 2 rows a pass, the decode's,
    # served prompts read at 32 ms a token, 2026-11-13), the reset of the states
    step_s, reset_s = tuned("GlmModel.step_s", 0.021), tuned("GlmModel.reset_s", 0.3)
    # the prefill, served (generated/g7_probe2, 2026-11-13, warm): a chunk of CHUNK rows costs ~2.3 s however few are
    # real (mHC, KDA, the projections over every row), the hand-offs and the MTP ~0.9 s, then ~2.6 ms a token (the
    # experts, the lane's and the card's): 40 tokens 4.5 s, 4,486 20.7 s
    prefill_fixed_s, prefill_chunk_s, prefill_tok_s = (tuned(f"GlmModel.prefill_{k}_s", v) for k, v in (
        ("fixed", 0.9 + 16 * 0.021), ("chunk", 2.3), ("tok", 0.0026)))  # + the unaligned tail's passes

    def __init__(self, ctx: int, rows: int = 2, log=print, top=None, logits=None):
        """top, logits: engine.py's Top and TorchLogits (the running engine's own classes, not a second import)"""
        self.Top, self.Logits = top, logits
        if not 0 < ctx <= CTX_MAX:
            raise ValueError(f"GLM's card context is at most {CTX_MAX} (got {ctx})")
        os.environ["G2_CTX"] = str(ctx)                      # before g2_layers: the decode build's caches
        sys.path.insert(0, HERE)
        import numpy as np
        import ttnn
        from g2_layers import MODEL, Gguf
        from g3_fkda import WORKER_L1_M
        from g4_token import N_DENSE, N_LAYER, _LIB, cpulane
        from g5_mtp import MTP
        from g5_spec import M_MAX, Spec
        from g6_prefill import SPLIT, Card, MtpPrefill, Prefill, card_experts
        assert 2 <= rows <= M_MAX
        self.np, self.ttnn, self.lib = np, ttnn, _LIB
        _LIB.gl_mla_set.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        self.ctx, self.rows, self.read_rows = ctx, rows, M_MAX   # decode at rows, read prompts at M_MAX a pass
        t0 = time.time()
        g = Gguf(MODEL)
        lane = cpulane.Lane()
        lane.load(MODEL, layers=set(range(N_DENSE, N_LAYER)) | {MTP})
        lane.router_load(MTP, g(f"blk.{MTP}.ffn_gate_inp.weight").numpy(), g(f"blk.{MTP}.exp_probs_b.bias").numpy())
        log(f"lane loaded in {time.time() - t0:.0f} s")
        self.dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=1 << 30, worker_l1_size=WORKER_L1_M,
                                         l1_small_size=1536, num_command_queues=2 if CARD_EXPERTS else 1)
        self.s = Spec(self.dev, g, lane)
        card = Card(self.dev)
        # the dense weights streamed (the second command queue) and their room the experts' (sized after)
        self.pf = Prefill(card, g, lane, CHUNK, stream=CARD_EXPERTS and STREAM_DENSE)
        self.mtp_pf = MtpPrefill(card, g, lane, CHUNK)
        self.experts = card_experts(self.dev) if CARD_EXPERTS else None   # in the card memory left; None: the lane
        if self.experts is not None:
            self.pf.set_experts(self.experts)
        self.split = SPLIT
        self.pending, self.pf_ids, self.pf_h = None, None, None   # the prefill's last read: its ids, last hidden row
        log(f"built in {time.time() - t0:.0f} s")
        self.warm()
        log(f"warm in {time.time() - t0:.0f} s")

    def warm(self) -> None:
        """Every program compiled and both traces captured before READY, in g5_spec's order (the trunk's verify trace
        first, then the MTP's: capturing the MTP's first left the trunk's eager pass and capture allocating over its
        freed scratch, and a later verify pass hung, 2026-11-13), then the prefill's ops and a pass after it, on a
        throwaway prompt; then the states back to position 0."""
        toks = [(37 * i + 11) % 50000 for i in range(WARM_PROMPT)]
        r = self.rows
        self.reset()
        self.feed(toks[:r], 0, [])                       # the trunk's trace
        self.keep(r)
        self.draft(toks[:r + 1], r - 1)                  # the MTP's traces (with and without the head)
        self.reset()
        if self.experts is not None:                     # the card experts' programs at each share the split takes,
            from g6_prefill import MOE_ROWS              # and the grouped calls' (2 .. MOE_ROWS / CHUNK chunks a call)
            for k in sorted({min(g, self.experts.slot_groups) for _, g in self.split} - {0}):
                old, os.environ["GP_CARD_GROUPS"] = os.environ.get("GP_CARD_GROUPS"), str(k)
                self.pf.warm_ks = tuple(range(2, MOE_ROWS // CHUNK + 1))
                try:
                    self.prefill(toks[:-1])
                finally:
                    self.pf.warm_ks = ()
                    os.environ.pop("GP_CARD_GROUPS") if old is None else os.environ.update(GP_CARD_GROUPS=old)
        self.prefill(toks[:WARM_PROMPT // 2])            # the prefill's programs (the lane's share), the hand-off,
        self.prefill(toks[:-1])                          # then an extension's (the sparse path past 0)
        d = self.draft(toks, r - 1)
        self.feed([toks[-1]] + d, len(toks) - 1, toks[:-1])
        self.keep(1)
        self.draft(toks + [toks[0]], r - 1)
        self.reset()

    def prefill_cost(self, n: int, start: int = 0) -> float:
        """Engine.prefill_s: the prefill's seconds for a prompt of n tokens (all but the last read), an extension
        (start > 0) from the prefill's last read (taken to be a prefix of the prompt, as the held state's is)"""
        p = len(self.pf_ids) if start and self.pf_ids and len(self.pf_ids) < n - 1 else 0
        k = n - 1 - p
        return self.prefill_fixed_s + -(-k // CHUNK) * self.prefill_chunk_s + k * self.prefill_tok_s

    def reset(self) -> None:
        self.s.reset()
        self.pending = None

    def prefill(self, ids: list[int], start: int = 0) -> None:
        import torch
        n = len(ids)
        if n + 1 >= self.ctx:
            raise RuntimeError(f"a prefill of {n} tokens outside the card's context ({self.ctx})")
        held, a = self.pf_ids, n // ALIGN * ALIGN
        p = len(held) if held and 1 < len(held) < a and ids[:len(held)] == held else 0
        if a:
            self.pf_ids = None                           # until both reads are in (a failed one continues nothing)
            hid = self.pf(self.s.emb, ids, a, start=p)   # rows p .. a - 1 (a failure leaves the decoder's states)
            self.reset()                                 # the decoder's states all rewritten by the hand-offs
            self.pf.hand_off(self.s.tok, self.lib)
            self.s.pos, self.s.n_keep, self.s.parity = a, 1, 0
            if p:   # the MTP's row p - 1 (pending at the last read: h_{p-1}, token p) on to a - 2
                self.mtp_pf(self.s.emb, ids, torch.cat([self.pf_h, hid[:-1]]), a - 1, r0=p - 1)
            else:   # the MTP's caches over rows 0 .. a - 2 (row j: h_j, token j + 1); row a - 1 waits for token a
                self.mtp_pf(self.s.emb, ids, hid, a - 1)
            self.mtp_pf.hand_off(self.s.mtp, self.lib)
            self.pending = (a - 1, hid[-1:].clone())
            self.pf_ids, self.pf_h = list(ids[:a]), hid[-1:].clone()
        else:
            self.reset()
        for pos in range(a, n, self.read_rows):          # the tail (< ALIGN tokens) by verify passes
            toks = ids[pos:min(n, pos + self.read_rows)]
            self.feed(toks, pos, ids[:pos])
            self.keep(len(toks))

    def feed(self, toks: list[int], pos: int, prev: list[int]):
        s, ttnn = self.s, self.ttnn
        if pos != s.pos:
            raise RuntimeError(f"a pass at {pos} with the state at {s.pos}")
        if pos + len(toks) > self.ctx:
            raise RuntimeError(f"positions {pos}..{pos + len(toks) - 1} outside the card's context ({self.ctx})")
        if self.pending is not None:
            self.draft(prev + toks, 0)
        _, self.hid = s.verify(toks)
        ids = ttnn.to_torch(s.tok.top_i).long().reshape(-1, 64)[:len(toks)]
        vals = ttnn.to_torch(s.tok.top_v).float().reshape(-1, 64)[:len(toks)]
        self.at, self.rows_full = (pos, len(toks)), None
        return [self.Top(ids[m].tolist(), vals[m].tolist(), lambda m=m: self.full(m)) for m in range(len(toks))]

    def full(self, m: int):
        if self.rows_full is None:
            self.rows_full = self.ttnn.to_torch(self.s.tok.logits).float().reshape(-1, VOCAB)
        return self.Logits(self.rows_full[m])

    def keep(self, n: int) -> None:
        pos, _ = self.at
        self.s.keep(n)
        self.pending = (pos, self.hid[:n].clone())

    def draft(self, seq: list[int], steps: int) -> list[int]:
        if self.pending is None:
            return []
        (c0, H), self.pending = self.pending, None
        nxt = seq[c0 + 1:c0 + 1 + len(H)]
        assert len(nxt) == len(H), (c0, len(H), len(seq))
        return self.s.mtp_catch_up(c0, nxt, H, steps)

    def close(self) -> None:
        self.ttnn.close_mesh_device(self.dev)
