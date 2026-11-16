"""GLM's speculative decoding on the card, greedy: the trunk's verify pass (g4_token.Token's
M-row mode) over [the pending token, the MTP's drafts], greedy acceptance, the rewind (the KDA layers' n_keep and
parity, the MLA layers' host states), then the MTP (g5_mtp.Mtp, one row a step, its own exchange list and host state)
catching up over the kept rows with the trunk's post-norm hidden rows as its h, its last row's top-1 the first draft,
then chaining the rest from its own h.

Run twice on the same programs: M = 4 (3 drafts a pass) and M = 1 (no drafts: plain greedy decoding). The two must
generate the same tokens up to near-ties of the trunk's logits (as Flash-Next's check: a different row count
changes the rounding of the attention's and the experts' sums). The prompt is the reference's first GS_PROMPT tokens
(passes of 4 rows, every row kept); then GS_GEN tokens. Reported: the accepted tokens a pass (E), tok/s.

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((320 << 30)) --engine tt/glm/g5_spec.py
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g4_token import N_DENSE, N_LAYER, REF, Token, _LIB, cpulane  # noqa: E402
from g2_layers import BF16, D, FP32, MODEL, OUT, TILE, Gguf  # noqa: E402
from g3_fkda import WORKER_L1_M  # noqa: E402
from g5_mtp import MTP, Mtp  # noqa: E402

M_MAX = 4
PROMPT, GEN = int(os.environ.get("GS_PROMPT", 16)), int(os.environ.get("GS_GEN", 64))
_LIB.gl_mla_rewind.argtypes = [ctypes.c_int, ctypes.c_uint32]
_LIB.gl_rows.argtypes = [ctypes.c_int]
_LIB.gl_seq.argtypes = [ctypes.c_int]


class Spec:
    def __init__(self, dev, g, lane):
        self.dev = dev
        self.tok = Token(dev, g, lane, m_max=M_MAX)
        self.mtp = Mtp(dev, g, self.tok.head, seq=1)
        self.emb = self.tok.emb
        self.tid = None
        self.mtid = {}   # the MTP's traces: with the head (a draft) and without (a catch-up row)
        self.reset()

    def reset(self):
        """position 0: every MLA state (the trunk's 11, the MTP's) back to it (the card's caches are rewritten by
        position); every KDA layer's states, conv history and pending pages zero (the pending G 1: commit nothing)"""
        for st in self.tok.mla_states + [self.mtp.state]:
            assert _LIB.gl_mla_rewind(st, 0) == 0
        put = lambda t, ref, lay: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(t, dtype=ttnn.float32, layout=lay), ref)
        for L in self.tok.layers:
            if "kda" in L:
                k = L["kda"]
                put(torch.zeros(tuple(k.S.shape)), k.S, TILE)
                put(torch.zeros(tuple(k.state.shape)), k.state, ttnn.ROW_MAJOR_LAYOUT)
                pend = torch.zeros(tuple(k.pend.shape)).reshape(-1, 12, 2 * M_MAX * 16)
                pend[:, 8:] = 1.0
                put(pend.reshape(tuple(k.pend.shape)), k.pend, ttnn.ROW_MAJOR_LAYOUT)
        self.pos, self.n_keep, self.parity = 0, 1, 0

    def verify(self, rows):
        """one trunk pass over the tokens rows at positions pos..: each row's top-1 and its post-norm hidden"""
        tok = self.tok
        tok.set_pass(rows, self.n_keep, self.parity)
        if self.tid is None:   # eager (compiles) with the host serving in a thread, then the trace
            th = threading.Thread(target=tok.loop.run)
            th.start()
            tok.run()
            th.join()
        else:
            ttnn.execute_trace(self.dev, self.tid, cq_id=0, blocking=False)
            tok.loop.run()
        top = ttnn.to_torch(tok.top_i).long().reshape(-1, 64)[:len(rows), 0].tolist()
        hid = ttnn.to_torch(tok.hn_rows).float().reshape(-1, D)[:len(rows)].clone()
        if self.tid is None:
            self.tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            tok.run()
            ttnn.end_trace_capture(self.dev, self.tid, cq_id=0)
        return top, hid

    def keep(self, a):
        """the pass's first a rows kept: the trunk rewinds past them"""
        self.pos += a
        for st in self.tok.mla_states:
            assert _LIB.gl_mla_rewind(st, self.pos) == 0
        self.n_keep, self.parity = a, 1 - self.parity

    def mtp_step(self, token, h=None, head=True):
        """one MTP row: token's embedding and h (a host row; None: the MTP's own last h) -> its top-1 (head) or
        None (a catch-up row: only its caches and h matter)"""
        m = self.mtp
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(self.emb[token]).reshape(1, 1, 1, D),
                                                        dtype=FP32, layout=TILE), m.e)
        if h is not None:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(h.reshape(1, 1, 1, D), dtype=FP32, layout=TILE), m.h)
        else:
            ttnn.copy(m.o["h"], m.h)
        assert _LIB.gl_rows(1) == 0
        if not self.mtid:   # the first step eager (its outputs kept), then both traces, each with its outputs
            m.first()
            eager = dict(m.o)
            self.mo = {}
            for hd, parts in ((True, 15), (False, 7)):
                self.mtid[hd] = ttnn.begin_trace_capture(self.dev, cq_id=0)
                m.run(parts)
                ttnn.end_trace_capture(self.dev, self.mtid[hd], cq_id=0)
                self.mo[hd] = dict(m.o)
            m.o = eager
        else:
            ttnn.execute_trace(self.dev, self.mtid[head], cq_id=0, blocking=False)
            m.loop.run()
            m.o = self.mo[head]   # (the next chained step copies this trace's h)
        return int(ttnn.to_torch(m.o["top_i"]).long().flatten()[0]) if head else None

    def mtp_catch_up(self, start, nxt, hid, n_drafts):
        """the MTP over the kept rows (positions start..: (the next token, the trunk's hidden) each), then n_drafts
        drafts chained; its state rewound to start first (last pass's drafts were beyond it)"""
        assert _LIB.gl_mla_rewind(self.mtp.state, start) == 0
        d = None
        for j, t in enumerate(nxt):   # the head only on the last row, and only when drafting
            d = self.mtp_step(t, hid[j], head=j == len(nxt) - 1 and n_drafts > 0)
        if n_drafts == 0:
            return []
        drafts = [d]
        for _ in range(n_drafts - 1):
            drafts.append(self.mtp_step(drafts[-1]))
        return drafts

    def generate(self, prompt, n, m):
        """greedy: the prompt (passes of 4, all kept), then n tokens with m rows a pass (m - 1 drafts)"""
        self.reset()
        toks = list(prompt)
        while self.pos < len(prompt) - 1:   # the prompt's rows but its last (that one starts the decoding)
            rows = toks[self.pos:min(self.pos + M_MAX, len(prompt) - 1)]
            _, hid = self.verify(rows)
            start = self.pos
            self.keep(len(rows))
            if m > 1:
                self.mtp_catch_up(start, toks[start + 1:start + 1 + len(rows)], hid, 0)
        drafts, out, passes, t0 = [], [], [], time.perf_counter()
        while len(out) < n:
            rows = [toks[self.pos]] + drafts
            top, hid = self.verify(rows)
            a = 1
            while a < len(rows) and rows[a] == top[a - 1]:
                a += 1
            start = self.pos
            self.keep(a)
            toks = toks[:start + 1] + rows[1:a] + [top[a - 1]]   # the accepted drafts, then the next pending token
            out += top[:a]                                        # (rows[j] == top[j - 1] for 0 < j < a)
            passes.append(a)
            if m > 1:
                drafts = self.mtp_catch_up(start, toks[start + 1:start + a + 1], hid, m - 1)
        wall = time.perf_counter() - t0
        return out[:n], {"passes": len(passes), "E": len(out) / len(passes), "tok_s": len(out) / wall}


def main() -> None:
    lane = cpulane.Lane()
    lane.load(MODEL, layers=set(range(N_DENSE, N_LAYER)) | {MTP})
    g = Gguf(MODEL)
    lane.router_load(MTP, g(f"blk.{MTP}.ffn_gate_inp.weight").numpy(), g(f"blk.{MTP}.exp_probs_b.bias").numpy())
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32)
    prompt = [int(x) for x in ids[:PROMPT]]
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "prompt": PROMPT, "gen": GEN}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=1 << 30, worker_l1_size=WORKER_L1_M)
    try:
        s = Spec(dev, g, lane)
        print(f"built (trunk {s.tok.build_s} s)", flush=True)
        greedy, rg = s.generate(prompt, GEN, 1)
        print("greedy (M = 1):", json.dumps(rg), flush=True)
        res.update({"greedy": rg, "greedy_tokens": greedy})
        for m in (2, 3, 4):
            spec, rs = s.generate(prompt, GEN, m)
            rs["differ_at"] = [i for i, (x, y) in enumerate(zip(greedy, spec)) if x != y]
            res[f"spec_m{m}"], res[f"spec_m{m}_tokens"] = rs, spec
            print(f"speculative (M = {m}):", json.dumps(rs), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, "g5_spec.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "g5_spec.json"))


if __name__ == "__main__":
    main()
