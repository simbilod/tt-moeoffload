"""G3 step 6: Flash-Next's QSA attention layer, program 2 of 2 (tt/kernels/fattn2_*.cpp), real blk.3 weights: from the
host's selection of 512 pools to the next residual streams, on all 110 cores with a barrier after each phase.

  AP      the attention partials over the selected pools (16 cores per KV head, 32 pools each) and the token's own
          block (one core per KV head, cells past the token masked): K and V tilized from the pool-major cache,
          S = q K^T, the row max, P = exp(S - m), the row sum, O = P V
  AC      the combine (16 cores: KV head x dims tile) and the output gate
  oproj   wo + the combine with w_a;  H1f, H2f, R1, R2: the FFN half (fused/hc_*.h, fused/tail_*.h)

Inputs stand in for program 1's (g3_attn1.py) outputs: Q (normed, rotated, / 16), the sigmoid gates, the caches, the
selection (512 distinct complete pools), X and w_a, random; the token at the end of a 16K context. Checked against
float64 per phase from the card's own previous outputs, on NaN-filled outputs with fresh inputs, then timed (traced,
one launch).

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/flashnext/g3_attn2.py

Measured 2026-10-23: the gated attention output within 2.0e-3 of float64 (P is bf16 for the transpose and P V, as in
the standalone kernel), everything after it within 7.1e-4; 223.0 us traced (phases: inputs 10, partials 31, combine
21, wo + combine 48, H1f 19, H2f 38, R1 27, R2 14 us). The combine's statistics as one tile each (the standalone
combine's way) took it from 66 to 21 us.
With rows 0, 1, 9 reading through NOC1 (fused/grid.py), phases on the NOC0 rows first: 212.5 us (211.1 on NOC0
alone, after the HL phase): no phase here streams on the whole grid. 206.4 with the mix's weights prefetched
before its barriers.

For the token loop (2026-10-24) the program is built once: the step tensor (uint32 [16], word 0 pos) and the
selection tensor (uint32 [528]: 512 pool ids, word 512 nsel = the valid ones, word 513 own_valid = the own block's
cells taken) carry everything per token (set_pos writes them in queue). Every AP block has a mask: core k's block b
takes 4 clamp(nsel - 32 k - 8 b, 0, 8) cells, the own block own_valid; a core with none leaves a partial weighted
exp(-10000) = 0 in the combine. The last X goes back to X's tensors in place (fp32 as X2's check, bf16 from the
writer: inplace), and -DEXCHANGE puts the host link between R1 and R2 (as the DeltaNet layer's). One program checked
at pos 0, 2, 3, 5, 9, 100, 2047, 2050, 8191 (0 to 512 pools + 1 to 4 own cells): the attention within 1.5-3.3e-3 of
float64, everything after within 9.1e-4, the bf16 X exact; 207.2 us traced. With the link: hn_f and the logits as
received equal the checks, R2 from the host's reply within 1.7e-6.
16,384 cells, no kernel change (the pools per KV head are an argument): pos 7999, 8191, 8195,
12002, 16383 with the attention within 2.0-3.1e-3 of float64, everything after within 7.4e-4; 206.3 us traced at 16383.
At 7,900-8,191 every output bit-identical to the 8K build's on the same inputs.

Attn2(m_max=4) builds the same program for a verify pass of up to 4 rows (fattn2_reader.cpp M_MAX): the rows'
selections as their union with a cell x row mask (union_words: the host's words; set_pass), AP over it on up to 32
cores per KV head (32 pools a partial, as many partials as the union needs), AC per (KV head, q^T column tile, dims
tile), oproj and the FFN half as the DeltaNet program's M rows. Checked and timed by g6_attn_m.py (errors_m per row,
chained after program 1's M rows), opened with WORKER_L1_M. m_max = 1 compiles the one-row source unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from fused.grid import NOC1_ROWS, Grid, profile_plumbing, traced_us  # noqa: E402
from g2_layers import MODEL, Gguf  # noqa: E402
from g3_attn1 import CTX, kv_caches  # noqa: E402  (the build's context, TT_CTX)
from g3_gdn import EPS, nan_tensor  # noqa: E402
from g3_hc import HC, LO, LR, model_tag, stream  # noqa: E402
from g3_layer import FF, N_ROUTER, WORKER_L1, bf, blocks, mp_of, rel, unblocks  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
LAYER = 3
D, NH, NKV, HD = 2560, 24, 2, 256
KPOOL, NSEL = 4, 512
NPOOL = CTX // KPOOL
BATCH, DEPTH = 8, 4
IDLE = 0xFFFFFFFF
NPARTS = 17
EXACT = (3, 6, 9, 11, 12, 13, 16, 22, 26, 32, 41, 42, 43, 51, 52, 53, 54, 55)
POSITIONS = tuple(p for p in (0, 2, 3, 5, 9, 100, 2047, 2050, 7999, 8191, 8195, 12002, 16387, 32770, 57346, 65538,
                              131074, 196610)
                  if p < CTX - 1) + (CTX - 1,)   # one program; nsel = min(512,
                                                                                  # complete pools)
NAP = 32          # M rows: AP cores per KV head (fattn2_reader.cpp)


def m_sizes(mm: int) -> dict:
    """The M-row program's sizes for M_MAX = mm (fattn2_reader.cpp's constants)."""
    t_max, nu = (12 * mm + 31) // 32, mm * 513
    pmax = (nu + 31) // 32
    return dict(T_MAX=t_max, NU_MAX=nu, SEL_W=(16 + nu + 15) // 16 * 16, PMAX=pmax, PSTRIDE=10 * t_max,
                NS=(pmax + 31) // 32)


def union_words(pos: int, rows, mm: int) -> np.ndarray:
    """The M-row selection tensor's words from each row's selection (rows[m] = (its pool ids, own_valid), the host's
    select_words for position pos + m): word 0 the union's size U, from word 16 the union's pools (ascending), each
    pool id | its cell x row mask << 16 (bit 4 j + m: cell j is row m's: all 4 of a selected pool, the first own_valid
    of the row's own block)."""
    mask = {}
    for m, (top, own_valid) in enumerate(rows):
        for p in np.asarray(top, dtype=np.int64).tolist():
            mask[p] = mask.get(p, 0) | (0x1111 << m)
        o = (pos + m) // KPOOL
        for j in range(own_valid):
            mask[o] = mask.get(o, 0) | (1 << (4 * j + m))
    s = m_sizes(mm)
    words = np.zeros(s["SEL_W"], np.uint32)
    pools = sorted(mask)
    assert len(pools) <= s["NU_MAX"]
    words[0] = len(pools)
    words[16:16 + len(pools)] = [p | mask[p] << 16 for p in pools]
    return words


def row_cells(words: np.ndarray, m: int) -> np.ndarray:
    """Row m's cells (cache cell ids) from the M-row selection words."""
    e = words[16:16 + int(words[0])].astype(np.int64)
    pools, bits = e & 0xFFFF, e >> 16
    return np.array([4 * p + j for p, b in zip(pools.tolist(), bits.tolist()) for j in range(4)
                     if b >> (4 * j + m) & 1], dtype=np.int64)


class Attn2:
    def __init__(self, dev, g: Gguf, rng, pos=CTX - 1, layer=LAYER, back=True, m_max=1, zero_caches=False):
        self.zero_caches = zero_caches       # g3_attn1.kv_caches: zero caches made on the card
        """back=False: no float64 host copies of the weights (the token loop; errors() then cannot run).
        m_max > 1: the M-row program (fattn2_reader.cpp M_MAX) for up to m_max (2 or 4) rows a pass at positions
        pos .. pos + M - 1, M from the step tensor; the selection is the rows' union (union_words)."""
        assert m_max in (1, 2, 4), "M_MAX: 1, 2 or 4 (M = 3 runs in the 4-row layout)"
        self.m_max = m_max
        self.dev, self.pos, self.layer = dev, pos, layer
        self.grid = Grid(dev)
        self.noc1_rows = NOC1_ROWS   # rows streaming through NOC1 (fused/grid.py)
        p = f"blk.{layer}."
        tag = model_tag(g)
        key = lambda name: f"{tag}/{p}{name}"                                   # noqa: E731  (g3_hc.stream's cache)
        norm = g(f"{p}hc_ffn_norm.weight")

        def wd():
            down, inject = g(f"{p}hc_ffn_down.weight"), g(f"{p}hc_ffn_inject.weight")
            w = torch.zeros(HC * D, LO)
            w[:, :LR + HC] = (torch.cat([down, inject]) * norm[None, :]).T
            return w
        self.wd, self.wdq = stream(dev, wd, back, key("hc_ffn_wd"))
        self.wu, self.wuq = stream(dev, lambda: g(f"{p}hc_ffn_up.weight").T, back, key("hc_ffn_wu"))
        self.hnorm = norm
        self.hnorm_t = self.rm(norm)
        self.wo, self.woq = stream(dev, lambda: g(p + "attn_output.weight").T, back, key("wo"))  # [6144, 2560]
        wr = torch.zeros(D, N_ROUTER * 32)
        wr[:, :512], wr[:, 512] = g(p + "ffn_gate_inp.weight").T, g(p + "ffn_gate_inp_shexp.weight")
        st = wr.to(torch.bfloat16).reshape(D // 32, 32, N_ROUTER, 32).permute(2, 0, 1, 3).reshape(1, 1, -1, 32)
        self.wr = ttnn.from_torch(st, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        self.wrq = bf(wr) if back else None

        def gu():
            gate, upw = g(p + "ffn_gate_shexp.weight").T, g(p + "ffn_up_shexp.weight").T
            return torch.stack([gate.reshape(D, FF // 32, 32), upw.reshape(D, FF // 32, 32)], 2).reshape(D, 2 * FF)
        self.wgu, self.wguq = stream(dev, gu, back, key("wgu"))
        self.wsd, self.wsdq = stream(dev, lambda: g(p + "ffn_down_shexp.weight").T, back, key("wsd"))
        o, a = {}, 0
        mm = m_max
        if mm == 1:
            regions = (("x32", HC * D * 4), ("x16", HC * D * 2), ("p1", HC * LO * 4), ("rs", 64),
                       ("q", NH * HD * 4), ("g", NH * HD * 4), ("sel", (NSEL + 16) * 4), ("st", 64), ("wa", 64),
                       ("o", NH * HD * 2), ("hnf", D * 2), ("lof", 640), ("wf", 64), ("l", N_ROUTER * 128),
                       ("sh", FF * 2), ("rt", D * 4))
        else:   # the owner layout (fattn2_reader.cpp M rows): x32 / x16 an H1 core's stream, rs xown, p1 then the r_s,
            # lof then w_f, l then sh; Q, G read by their consumers; sel, st, wa contiguous (the first broadcast)
            regions = (("x32", mm * D * 4), ("x16", mm * D * 2), ("p1", mm * (HC * LO * 4 + 256)),
                       ("rs", mm * HC * 128), ("q", 0), ("g", 0), ("sel", m_sizes(mm)["SEL_W"] * 4), ("st", 64),
                       ("wa", mm * 64), ("o", mm * NH * HD * 2), ("hnf", mm * D * 2), ("lof", mm * 704), ("wf", 0),
                       ("l", mm * (N_ROUTER * 128 + FF * 2)), ("sh", 0), ("rt", 0))
        for name, size in regions:
            o[name] = a
            a += size
        assert all(v % 64 == 0 for v in o.values())
        self.off, self.arena = o, a
        self.gprof, self.gprof_n = None, 32  # grid.h's per-core clocks (fused/grid.py profile_tensor; g4_phases.py)
        self.defines = []          # the host link's (EXCHANGE, the socket config addresses), as g3_link.py's
        self.inplace = False       # the last X back into X's tensors (fp32 and bf16), as in the decode loop
        self.set_inputs(rng)
        if mm == 1:
            self.set_pos(pos, rng)

    def set_pos(self, pos, rng, sel=None):
        """The token's position and the host's selection (default: min(512, complete pools before the token's
        block) of them, at random), into the step and selection tensors (in-queue writes: the program is built
        once). The own block takes its cells up to the token."""
        self.pos = pos
        complete = pos // KPOOL
        if sel is None:
            sel = np.sort(rng.choice(complete, min(NSEL, complete), replace=False))
        self.sel = np.asarray(sel, dtype=np.uint32)
        self.nsel, self.own_valid = len(self.sel), pos % KPOOL + 1
        words = np.zeros(NSEL + 16, np.uint32)
        words[:self.nsel] = self.sel
        words[NSEL], words[NSEL + 1] = self.nsel, self.own_valid
        put = lambda a, t: ttnn.copy_host_to_device_tensor(ttnn.from_torch(  # noqa: E731
            torch.from_numpy(a.view(np.int32)).reshape(tuple(t.shape)), dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT), t)
        put(words, self.sel_t)
        step = np.zeros(16, np.uint32)
        step[0] = pos
        put(step, self.step_t)

    def set_pass(self, pos, M, rows):
        """The M-row program's pass: M rows at positions pos .. pos + M - 1 and each row's selection (rows[m] = (its
        pool ids, own_valid), select_words' for pos + m) as the union's words, into the step and selection tensors
        (in queue)."""
        from g3_attn1 import step_words
        assert self.m_max > 1 and 1 <= M <= self.m_max and len(rows) == M and pos + M <= CTX
        self.pos, self.M, self.mp = pos, M, mp_of(M)
        self.words = union_words(pos, rows, self.m_max)
        put = lambda a, t: ttnn.copy_host_to_device_tensor(ttnn.from_torch(  # noqa: E731
            torch.from_numpy(a.view(np.int32)).reshape(tuple(t.shape)), dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT), t)
        put(self.words, self.sel_t)
        put(step_words(pos, M).numpy().astype(np.uint32).reshape(-1), self.step_t)

    def run(self, prog):
        """One launch; X as it was before (it changes in place) kept for the checks (M rows: the outputs reset to NaN
        before it)."""
        if self.m_max > 1:
            from g3_attn1 import x_rows
            self.x_before = x_rows(self.x32_t, self.mp)
            for t in list(self.chk.values()) + [self.parts]:
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full(tuple(t.shape), float("nan")),
                                                                dtype=t.dtype, layout=t.layout), t)
        else:
            self.x_before = ttnn.to_torch(self.x32_t).double().flatten()
        prog()
        ttnn.synchronize_device(self.dev)

    def rm(self, t):
        t = t.float().contiguous()
        return ttnn.from_torch(t.reshape(1, 1, -1, t.shape[-1]) if t.dim() > 1 else t.reshape(1, 1, 1, -1),
                               dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)

    def set_inputs_m(self, rng, M=None):
        """M rows (m_max > 1): random stand-ins for program 1's outputs at M rows (rows M..mp-1 copies of row
        M - 1): X (the M-row X tensors), Q (bf16, / 16) and the gates G (a head-row a page: row m head h at 24 m + h),
        w_a (row m's at word 16 m), the caches, the routed sums; NaN-filled outputs."""
        from g3_attn1 import x_tensors
        dev, mm = self.dev, self.m_max
        M = M or mm
        self.M, self.mp = M, mp_of(M)
        mp = self.mp
        r = lambda *s: torch.from_numpy(rng.standard_normal(s).astype(np.float32))           # noqa: E731
        dup = lambda t: torch.cat([t, t[-1:].expand(mp - M, *t.shape[1:])]) if mp > M else t  # noqa: E731
        x = dup(r(M, HC * D))
        self.x32_t, self.x16_t = x_tensors(dev, x, mm)
        q = torch.zeros(mm, NH, HD)
        q[:mp] = dup(r(M, NH, HD) / 16)
        self.q_t = ttnn.from_torch(q.reshape(1, 1, mm * NH, HD), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=dev)
        gates = torch.zeros(mm, NH, HD)
        gates[:mp] = dup(torch.sigmoid(r(M, NH, HD)))
        self.g_t = self.rm(gates.reshape(mm * NH, HD))
        wa = torch.zeros(mm, 16)
        wa[:mp, :HC] = dup(torch.from_numpy(rng.uniform(0.2, 1.8, (M, HC)).astype(np.float32)))
        self.wa_t = self.rm(wa.reshape(-1))
        u32 = lambda n: ttnn.from_torch(torch.zeros(1, 1, 1, n, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        s = m_sizes(mm)
        self.sel_t, self.step_t = u32(s["SEL_W"]), u32(16)
        self.kc0, self.vc0, self.kc, self.vc = kv_caches(dev, r, self.zero_caches)
        self.routed_t = self.rm(torch.zeros(mm * D))
        self.new_routed(rng)
        self.parts = nan_tensor(dev, (1, 1, NKV * s["PMAX"] * s["PSTRIDE"] * 32, 32), ttnn.TILE_LAYOUT)
        n = lambda *shape: nan_tensor(dev, shape)                                          # noqa: E731
        self.chk = dict(o=n(1, 1, mm * NH, HD), x1=n(1, 1, HC, mm * D), p1=n(1, 1, 1, mm * HC * LO),
                        hn=n(1, 1, 1, mm * D), w=n(1, 1, 1, mm * 16), l=n(1, 1, 1, mm * N_ROUTER * 32),
                        h=n(1, 1, 1, mm * FF), x2=n(1, 1, HC, mm * D))
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def new_routed(self, rng):
        """M rows: fresh routed sums for the pass's rows (rows M..mp-1 copies of row M - 1), in place."""
        M, mp = self.M, self.mp
        r = torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32)) * 0.1
        self.routed = torch.cat([r, r[-1:].expand(mp - M, -1)]) if mp > M else r
        flat = torch.zeros(self.m_max * D)
        flat[:mp * D] = blocks(self.routed, mp)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(flat.reshape(1, 1, 1, -1), dtype=ttnn.float32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT), self.routed_t)

    def set_inputs(self, rng):
        if self.m_max > 1:
            return self.set_inputs_m(rng)
        dev = self.dev
        r = lambda *s: torch.from_numpy(rng.standard_normal(s).astype(np.float32))           # noqa: E731
        self.x = r(HC * D).double()
        self.x32_t = self.rm(self.x)
        self.x16_t = ttnn.from_torch(self.x.float().to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.q = (r(NH, HD) / 16).double()
        self.gates = torch.sigmoid(r(NH, HD)).double()
        self.q_t, self.g_t = self.rm(self.q), self.rm(self.gates)
        u32 = lambda n: ttnn.from_torch(torch.zeros(1, 1, 1, n, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.sel_t, self.step_t = u32(NSEL + 16), u32(16)   # the selection (+ nsel, own_valid), the step (pos)
        self.wa = torch.zeros(16)
        self.wa[:HC] = torch.from_numpy(rng.uniform(0.2, 1.8, HC).astype(np.float32))
        self.wa_t = self.rm(self.wa)
        self.kc0, self.vc0, self.kc, self.vc = kv_caches(dev, r, self.zero_caches)
        self.routed = r(D) * 0.1
        self.routed_t = self.rm(self.routed)
        self.parts = nan_tensor(dev, (1, 1, NKV * NPARTS * 10 * 32, 32), ttnn.TILE_LAYOUT)
        n = lambda *shape: nan_tensor(dev, shape)                                          # noqa: E731
        self.chk = dict(o=n(1, 1, NH, HD), x1=n(1, 1, 1, HC * D), p1=n(1, 1, 1, HC * LO), hn=n(1, 1, 1, D),
                        w=n(1, 1, 1, 16), l=n(1, 1, 1, N_ROUTER * 32), h=n(1, 1, 1, FF), x2=n(1, 1, 1, HC * D))
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def program(self):
        g, o = self.grid, self.off
        A = lambda t: t.buffer_address()                                                    # noqa: E731
        order = g.order(self.noc1_rows)  # every phase on part of the grid: the NOC0 rows first
        gp, gdef, gbytes = profile_plumbing(self.gprof, self.gprof_n, 63, 64)  # in cb 63, after the 16 stamps
        mr = self.m_max > 1
        ms = m_sizes(self.m_max)
        rows = [("M_MAX", str(self.m_max))] if mr else []     # every kernel of the program takes it
        pack = lambda k: g.noc[order[k]].x | g.noc[order[k]].y << 16                # noqa: E731
        h1_cores = [pack(i) for i in range((LO // 32) * HC)] if mr else []   # M rows: X's tiles to H1's cores
        rargs, crt, wargs = {}, ttnn.RuntimeArgs(), {}
        for idx, (cx, cy) in enumerate(g.cores):
            k = order.index(idx)
            ap = k if k < NKV * (NAP if mr else NPARTS) else IDLE       # M rows: KV head k // NAP, core k % NAP
            ac = k if k < NKV * 8 * (ms["T_MAX"] if mr else 1) else IDLE  # M rows: (g, t, d), fattn2_reader.cpp
            o_tile = k if k < D // 32 else IDLE
            h_item = k if k < (LO // 32) * HC else IDLE
            h_tile = k if k <= D // 32 else IDLE
            r_item = k if k < N_ROUTER + FF // 32 else IDLE
            s_tile = k if k < D // 32 else IDLE
            rargs[(cx, cy)] = [A(self.x32_t), A(self.x16_t), A(self.q_t), A(self.g_t), A(self.sel_t), A(self.wa_t),
                               A(self.step_t)] + \
                g.args(idx) + [o[r] for r in ("x32", "x16", "p1", "rs", "q", "g", "sel", "st", "wa", "o", "hnf",
                                              "lof", "wf", "l", "sh", "rt")] + [
                ap, A(self.kc), A(self.vc), NPOOL, ac, A(self.parts), A(self.wo), o_tile,
                A(self.wd), A(self.wu), A(self.hnorm_t), h_item, h_tile, A(self.wr), A(self.wgu), A(self.wsd),
                A(self.routed_t), r_item, s_tile, A(self.prof)] + h1_cores + gp(idx)
            crt[cx][cy] = [ap, ac, o_tile, h_item, h_tile, r_item, s_tile]
            wargs[(cx, cy)] = g.args(idx) + [o[r] for r in ("x32", "x16", "p1", "rs", "o", "hnf", "lof", "wf", "l",
                                                             "sh")] + [
                A(self.parts)] + [A(self.chk[c]) for c in ("o", "x1", "p1", "hn", "w", "l", "h")] + [
                A(self.x32_t) if self.inplace else A(self.chk["x2"])] + [
                ap, ac, o_tile, h_item, h_tile, r_item, s_tile, A(self.x16_t) if self.inplace else 0] + (
                    h1_cores + [A(self.step_t), o["sel"]] if mr else []) + gp(idx)
        acc = list(ttnn.TensorAccessorArgs(self.wo).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("G4_FIDELITY", "HiFi4")), fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in EXACT:   # (M rows: cb 22 holds the bf16 mask tiles, a matmul's input)
            if not (mr and cb == 22):
                mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fattn2_compute.cpp"), core_ranges=g.ranges,
                                         compile_time_args=[BATCH], runtime_args=crt, config=cfg,
                                         **(dict(defines=rows) if mr else {}))]
        kernels += g.dm_kernels(os.path.join(KDIR, "fattn2_reader.cpp"), os.path.join(KDIR, "fattn2_writer.cpp"),
                                [BATCH, DEPTH] + acc, acc, rargs, wargs, self.defines + gdef + rows, self.noc1_rows)
        f32, bf16, u8 = ttnn.float32, ttnn.bfloat16, ttnn.uint8
        T4, T2 = 4096, 2048
        spec = [(0, ttnn.bfloat8_b, 1088, BATCH * DEPTH), (1, bf16, T2, 80), (2, bf16, T2, 1),
                (3, f32, T4, 3), (4, f32, T4, 3), (5, f32, T4, 1), (6, f32, T4, 1), (7, f32, T4, 2), (8, f32, T4, 1),
                (9, f32, T4, 4), (10, bf16, T2, 10), (11, f32, T4, 4), (12, f32, T4, 4), (13, f32, T4, 4),
                (14, bf16, T2, 1), (15, f32, T4, 1),
                (16, f32, T4, 8), (17, bf16, T2, 8), (18, bf16, T2, 8), (19, bf16, T2, 32), (20, bf16, T2, 32), (21, bf16, T2, 8), (48, bf16, T2, 4), (58, bf16, T2, 8), (59, f32, T4, 4),
                (22, f32, T4, 1), (23, f32, T4, 4), (24, f32, T4, 1), (25, bf16, T2, 4), (26, f32, T4, 1),
                (27, f32, T4, 10), (28, f32, T4, 1), (29, f32, T4, NPARTS), (30, f32, T4, 1), (31, f32, T4, 1),
                (32, f32, T4, 1), (33, f32, T4, 1), (34, f32, T4, NPARTS), (35, f32, T4, 1), (36, f32, T4, 1),
                (37, f32, T4, 1), (38, bf16, T2, 1), (39, f32, T4, 1),
                (40, bf16, T2, 2 * BATCH), (41, f32, T4, 1), (42, f32, T4, 4), (43, f32, T4, 4), (44, f32, T4, 4),
                (45, bf16, T2, 4), (46, bf16, T2, 2 * BATCH), (47, f32, T4, 1), (49, bf16, T2, 1), (50, f32, T4, 1),
                (51, f32, T4, 1), (52, f32, T4, 1), (53, f32, T4, 4), (54, f32, T4, 4), (55, f32, T4, 1),
                (56, f32, T4, 4), (57, bf16, T2, 4),
                (60, u8, T4, 1), (61, u8, self.arena, 1), (62, u8, 1024, 1), (63, u8, 64 + gbytes, 1)]
        if mr:  # M rows (fattn2_compute.cpp): T_MAX column tiles in AP's buffers, the combine's statistics in NS
            # tiles and its O tiles and D_i in groups of 16; 16 the combine's sum, 22 the mask tiles (bf16), 58 q's row
            # tiles (the reader's); the ones tile + Sel + the words, H1's raw X chunks double-buffered, a 3 KB stage
            t, ns = ms["T_MAX"], ms["NS"]
            size = {2: 3, 3: 6, 16: 1, 21: 9 * t, 48: 4 * t, 58: 9 * t, 59: 4 * t, 22: 4, 23: 4 * t, 24: t, 25: 4 * t,
                    26: t, 27: 8, 28: ns, 29: 16, 30: ns, 34: 16, 35: ns, 36: 2 * ns}  # (36: A l, then W after it)
            spec = [(i, bf16 if i == 22 else dt, T2 if i == 22 else 3072 if i == 62 else pg, size.get(i, n))
                    for i, dt, pg, n in spec]
        self.l1_bytes = sum(size * n for _, _, size, n in spec)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        ins = [self.x32_t, self.x16_t, self.q_t, self.g_t, self.sel_t, self.wa_t, self.step_t, self.kc, self.vc,
               self.parts,
               self.wo, self.wd, self.wu, self.hnorm_t, self.wr, self.wgu, self.wsd, self.routed_t, self.prof] + \
            list(self.chk.values()) + ([] if self.gprof is None else [self.gprof])
        return lambda: ttnn.generic_op(ins, prog)

    def get(self, k):
        if k == "x2" and self.inplace:
            return ttnn.to_torch(self.x32_t).double().flatten()
        return ttnn.to_torch(self.chk[k]).double().flatten()

    def routed_ref(self):
        """The experts' sum R2 added: the routed tensor, or the host's reply over the link."""
        return getattr(self, "reply", self.routed).double()

    def errors(self) -> dict:
        res = {}
        kc, vc = self.kc0.double().reshape(NKV, CTX, HD), self.vc0.double().reshape(NKV, CTX, HD)
        own = np.arange(self.pos // KPOOL * KPOOL, self.pos // KPOOL * KPOOL + self.own_valid)
        cells = np.concatenate([(self.sel[:, None].astype(np.int64) * KPOOL + np.arange(KPOOL)).ravel(), own])
        o = torch.zeros(NH, HD, dtype=torch.float64)
        for h in range(NH):
            gi = h // (NH // NKV)
            p = torch.softmax(kc[gi, cells] @ self.q[h], -1)
            o[h] = p @ vc[gi, cells]
        o = o * self.gates
        res["attention"] = rel(self.get("o").reshape(NH, HD), o)
        if os.environ.get("G3_DEBUG"):   # the partials combined on the host
            pt = ttnn.to_torch(self.parts).double().reshape(NKV, NPARTS, 10, 32, 32)
            oh = torch.zeros(NH, HD, dtype=torch.float64)
            for gi in range(NKV):
                O = pt[gi, :, :8, :12].permute(0, 2, 1, 3).reshape(NPARTS, 12, HD)
                m, l = pt[gi, :, 8, 0, :12], pt[gi, :, 9, 0, :12]
                a = torch.exp(m - m.max(0).values)
                oh[gi * 12:(gi + 1) * 12] = (a[:, :, None] * O).sum(0) / (a * l).sum(0)[:, None]
            res["partials_host"] = rel(oh * self.gates, o)
            c0 = (self.sel[:32, None].astype(np.int64) * KPOOL + np.arange(KPOOL)).ravel()
            S0 = (self.q[:12] @ kc[0, c0].T)                                   # [12 heads, 128 cells]
            print("m card", pt[0, 0, 8, 0, :12].numpy().round(3), "\nm ref ", S0.max(1).values.numpy().round(3))
            print("l card", pt[0, 0, 9, 0, :12].numpy().round(3), "\nl ref ",
                  torch.exp(S0 - S0.max(1, keepdim=True).values).sum(1).numpy().round(3))
            print("m tile row 0", pt[0, 0, 8, 0, :8].numpy().round(3))
        out = bf(self.get("o")) @ self.woq
        x0 = self.x_before if hasattr(self, "x_before") else self.x
        x1 = (x0.reshape(HC, D) + self.wa[:HC, None].double() * out[None, :]).flatten()
        res["oproj_combine"] = rel(self.get("x1"), x1)
        X = self.get("x1").reshape(HC, D)
        r = torch.rsqrt(X.pow(2).mean(-1) + EPS)
        X16 = bf(X)
        p_ref = torch.stack([r[s] * (X16[s] @ self.wdq[s * D:(s + 1) * D]) for s in range(HC)])
        p1 = self.get("p1").reshape(HC, LO)
        res["H1f"] = rel(p1[:, :LR + HC], p_ref[:, :LR + HC])
        lo = p1.sum(0)
        gate = (bf(torch.nn.functional.silu(lo[:LR] / HC)) @ self.wuq).reshape(HC, D)
        hn = (torch.sigmoid(gate) * X * self.hnorm.double().reshape(HC, D) * r[:, None]).mean(0)
        res["H2f_hn"] = rel(self.get("hn"), hn)
        res["H2f_w"] = rel(self.get("w")[:HC], 2 * torch.sigmoid(lo[LR:LR + HC] / HC))
        hnf = bf(self.get("hn"))
        logits = hnf @ self.wrq
        res["R1_logits"] = rel(self.get("l")[:513], logits[:513])
        gu = (hnf @ self.wguq).reshape(FF // 32, 2, 32)
        res["R1_h"] = rel(self.get("h"), (torch.nn.functional.silu(gu[:, 0]) * gu[:, 1]).flatten())
        sh = bf(self.get("h")) @ self.wsdq * torch.sigmoid(self.get("l")[512]) + self.routed_ref()
        x2 = (X + self.get("w")[:HC, None] * sh[None, :]).flatten()
        res["R2_combine"] = rel(self.get("x2"), x2)
        if self.inplace:     # the bf16 copy too
            x16 = ttnn.to_torch(self.x16_t).double().flatten()
            res["R2_x16"] = rel(x16, bf(self.get("x2")))
        return res

    def errors_m(self) -> dict:
        """M rows: every phase of every row m < M against float64 of the single-token step at pos + m, from the card's
        own previous outputs and the inputs as the card holds them (Q, G, w_a, the caches: program 1's when chained),
        row m attending its cells of the union (row_cells). The worst row per check and every row's ("rows")."""
        from g3_attn1 import x_rows
        M, mp, mm = self.M, self.mp, self.m_max
        T = lambda t: ttnn.to_torch(t).double()                                            # noqa: E731
        Q, G = T(self.q_t).reshape(mm, NH, HD), T(self.g_t).reshape(mm, NH, HD)
        wa = T(self.wa_t).flatten()[:mp * 16].reshape(mp, 16)
        kc, vc = T(self.kc).reshape(NKV, CTX, HD), T(self.vc).reshape(NKV, CTX, HD)
        o_card = T(self.chk["o"]).reshape(mm, NH, HD)
        un = lambda k, n: unblocks(T(self.chk[k]).flatten(), n, mp)                        # noqa: E731
        x1, p1, hn, l, h = (x_rows(self.chk["x1"], mp), un("p1", HC * LO // 32), un("hn", D // 32),
                            un("l", N_ROUTER), un("h", FF // 32))
        w = T(self.chk["w"]).flatten()[:mp * 16].reshape(mp, 16)
        x2 = x_rows(self.x32_t if self.inplace else self.chk["x2"], mp)
        rows = []
        for m in range(M):
            res = {}
            cells = row_cells(self.words, m)
            o = torch.zeros(NH, HD, dtype=torch.float64)
            for hh in range(NH):
                gi = hh // (NH // NKV)
                o[hh] = torch.softmax(kc[gi, cells] @ Q[m, hh], -1) @ vc[gi, cells]
            res["attention"] = rel(o_card[m], o * G[m])
            out = bf(o_card[m].flatten()) @ self.woq
            x1_ref = (self.x_before[m].reshape(HC, D) + wa[m, :HC, None] * out[None, :]).flatten()
            res["oproj_combine"] = rel(x1[m], x1_ref)
            X = x1[m].reshape(HC, D)
            r = torch.rsqrt(X.pow(2).mean(-1) + EPS)
            X16 = bf(X)
            p_ref = torch.stack([r[s] * (X16[s] @ self.wdq[s * D:(s + 1) * D]) for s in range(HC)])
            pm = p1[m].reshape(HC, LO)
            res["H1f"] = rel(pm[:, :LR + HC], p_ref[:, :LR + HC])
            lo = pm.sum(0)
            gate = (bf(torch.nn.functional.silu(lo[:LR] / HC)) @ self.wuq).reshape(HC, D)
            hn_ref = (torch.sigmoid(gate) * X * self.hnorm.double().reshape(HC, D) * r[:, None]).mean(0)
            res["H2f_hn"] = rel(hn[m], hn_ref)
            res["H2f_w"] = rel(w[m, :HC], 2 * torch.sigmoid(lo[LR:LR + HC] / HC))
            hnf = bf(hn[m])
            logits = hnf @ self.wrq
            res["R1_logits"] = rel(l[m, :513], logits[:513])
            gu = (hnf @ self.wguq).reshape(FF // 32, 2, 32)
            res["R1_h"] = rel(h[m], (torch.nn.functional.silu(gu[:, 0]) * gu[:, 1]).flatten())
            sh = bf(h[m]) @ self.wsdq * torch.sigmoid(l[m, 512]) + self.routed[m].double()
            res["R2_combine"] = rel(x2[m], (X + w[m, :HC, None] * sh[None, :]).flatten())
            if self.inplace:     # the bf16 copy: in bf16 steps from the fp32 X's rounding (the packer's), how many
                b, ref = x_rows(self.x16_t, mp)[m], bf(x2[m])
                ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().clamp(min=1e-30))) - 7)
                res["R2_x16_ulps"] = float(((b - ref).abs() / ulp).max())
                res["R2_x16_n"] = int((b != ref).sum())
            rows.append(res)
        out = {k: (sum if k == "R2_x16_n" else max)(r[k] for r in rows) for k in rows[0]}
        out["union"] = int(self.words[0])
        out["rows"] = rows
        return out


def link_check(dev, A2, rng) -> dict:
    """Program 2 with -DEXCHANGE: the host takes the page (hn_f, logits) and replies with a stand-in sum."""
    from u1_sockets import pinned
    core = ttnn.CoreCoord(*A2.grid.cores[0])                       # the coordinator
    where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core)
    h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 64 << 10, ttnn.H2DMode.DEVICE_PULL))
    h2d.set_page_size(D * 4)
    d2h = pinned(lambda: ttnn.D2HSocket(dev, where, 64 << 10))
    d2h.set_page_size(8192)
    A2.defines = [("EXCHANGE", "1"), ("H2D_CONFIG", str(h2d.get_config_buffer_address())),
                  ("D2H_CONFIG", str(d2h.get_config_buffer_address())), ("XCHG_OUT_BYTES", "8192")]
    A2.inplace = True
    prog = A2.program()
    page = torch.zeros(8192, dtype=torch.uint8)
    res = {}
    for i in range(2):              # the first launch compiles
        A2.set_pos(100 + i, rng)
        A2.reply = torch.from_numpy(rng.standard_normal(D).astype(np.float32)) * 0.1
        A2.x_before = ttnn.to_torch(A2.x32_t).double().flatten()
        t0 = time.perf_counter()
        prog()
        d2h.read_tensor(page)
        h2d.write_tensor(torch.from_numpy(A2.reply.numpy().view(np.uint8).copy()))
        ttnn.synchronize_device(dev)
        res["launch_us"] = (time.perf_counter() - t0) * 1e6
    hn = page[:D * 2].view(torch.bfloat16).double()
    logits = page[D * 2:D * 2 + 513 * 4].view(torch.float32).double()
    res["hn_vs_check"] = float((hn - bf(A2.get("hn"))).abs().max())
    res["logits_vs_check"] = float((logits - A2.get("l")[:513]).abs().max())
    res["max_rel_err"] = A2.errors()
    print(f"host link: received vs the checks hn_f {res['hn_vs_check']:.1e}, logits {res['logits_vs_check']:.1e}; "
          + ", ".join(f"{k} {v:.1e}" for k, v in res["max_rel_err"].items())
          + f"; one launch with the host servicing it {res['launch_us']:.0f} us", flush=True)
    A2.defines, A2.inplace = [], False
    return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": LAYER}
    try:
        A2 = Attn2(dev, g, rng)
        A2.inplace = True
        prog = A2.program()
        print(f"weights in {time.time() - t0:.0f} s; circular buffers {A2.l1_bytes / 1024:.0f} KB per core",
              flush=True)
        res["max_rel_err"] = {}
        for pos in POSITIONS:      # one program; the step and the selection change, X in place
            A2.set_pos(pos, rng)
            A2.run(prog)
            err = res["max_rel_err"][pos] = A2.errors()
            print(f"pos {pos} ({A2.nsel} pools + {A2.own_valid} own cells): "
                  + ", ".join(f"{k} {v:.1e}" for k, v in err.items()), flush=True)
        A2.inplace = False
        A2.set_pos(CTX - 1, rng)
        res["traced_us"] = traced_us(dev, A2.program())
        st = ttnn.to_torch(A2.prof).flatten().long().tolist()
        names = ["in", "AP", "AC", "oproj", "H1f", "HLf", "H2f", "R1", "R2"]
        cyc = [(st[i + 1] - st[i]) & 0xFFFFFFFF for i in range(len(names))]
        res["phase_cycles"] = dict(zip(names, cyc))
        print(f"program 2 traced {res['traced_us']:.1f} us; phases (cycles): "
              + ", ".join(f"{n} {c}" for n, c in zip(names, cyc)), flush=True)
        A2.noc1_rows = ()                                          # for comparison: every row on NOC0
        res["traced_us_noc0_only"] = traced_us(dev, A2.program())
        A2.noc1_rows = NOC1_ROWS
        print(f"traced {res['traced_us']:.1f} us with rows {NOC1_ROWS} on NOC1, {res['traced_us_noc0_only']:.1f} us "
              f"on NOC0 alone", flush=True)
        if os.environ.get("G3_LINK", "1") == "1":   # the host link (needs the IOMMU group in DMA-FQ)
            res["link"] = link_check(dev, A2, rng)
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g3_attn2.json" if CTX == 16384 else f"g3_attn2_ctx{CTX}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
