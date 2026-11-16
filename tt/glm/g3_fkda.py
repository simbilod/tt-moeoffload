"""GLM's KDA mixer as one program on all 110 cores (tt/kernels/glm_fkda_*.cpp), the port
of Flash-Next's fused DeltaNet mixer (fgdn_*.cpp, tt/flashnext/g3_gdn.py) to GLM's shapes, real blk.4 weights:

  A  hn -> q | k | v | f_lo | g_lo | beta (4096 -> 24896, bfp8_b) on all cores with the channel-local epilogues (the
     causal conv + silu with the new conv state written back, the L2 norms of the 64 q and 64 k heads, q / sqrt(128),
     sigmoid(beta))
  B  one value head per core (64 cores): the decays exp(-5 sigmoid(neg_a (f_lo f_b_h + dt))) and the gate
     sigmoid(g_lo g_b_h) from the low-rank halves (the head's 128 columns of f_b and g_b), then the KDA step of
     g3_kda.py (the state exact in fp32, the per-key decay through Dbc), the output norm (eps 1e-5) times the gate
  C  o (8192 -> 4096) on split_weighted's cores

Checked per phase against float64 from the card's own previous outputs (the card's weights dequantized, the low-rank
halves rounded to bf16 as the reader does), with every output pre-filled with NaN; and end to end against G2's
float32 mixer_ref (llama.cpp's graph on the unquantized weights). Timed traced, against the stock ttnn mixer of
g3_kda_layer.py (1,174.6 us).

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/glm/g3_fkda.py
  (FK_PHASES=1 or 3 runs phase A, or A and B, alone)
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.dirname(HERE)
KDIR = os.path.join(TT, "kernels")
sys.path.insert(0, HERE)
sys.path.insert(1, TT)
from g2_layers import KDA_LAYER, KDA_LB, MODEL, OUT, Gguf, KdaCard, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
import kda_m  # noqa: E402
from fused.mrows import blocks, mp_of, pad_rows, step_words, unblocks  # noqa: E402
from fused.grid import (NOC1_ROWS, Grid, profile_plumbing, profile_tensor, profile_words, split,  # noqa: E402
                        split_weighted, traced_us)

D, NV, HK = 4096, 64, 128
CONV = 3 * NV * HK                              # 24576 conv channels: q | k | v
LO = 128                                        # the low-rank halves' width
NT_A = CONV // 32 + 2 * LO // 32 + 2            # 778 output tiles: conv, f_lo, g_lo, beta (64)
T_FLO, T_BETA = CONV // 32, CONV // 32 + 8      # first f_lo tile, first beta tile
KA, KC, NT_C = D // 32, NV * HK // 32, D // 32
IDLE = 0xFFFFFFFF
W_PAGE = 1088
T_V, T_Q, T_K, T_P, T_B = 0, 1, 2, 3, 5
L2_EPS, NORM_EPS = 1e-6, 1e-5
BATCH, DEPTH = int(os.environ.get("FK_BATCH", 16)), int(os.environ.get("FK_DEPTH", 4))   # 16: 325 vs 349 us for A (8 x 8)
PHASES = int(os.environ.get("FK_PHASES", 7))     # bit 0 A, 1 B, 2 C
S = int(os.environ.get("FK_S", 4))               # C's K split (g3_fmo.py's)
PROF = int(os.environ.get("FK_PROF", 0))         # 1: grid.h's per-core barrier clocks (fused/grid.py profile_tensor)
M_MAX = int(os.environ.get("FK_M", 1))           # > 1: the M-row build (FK_PHASES 1, 3 or 7)
DEBUG = int(os.environ.get("FK_DEBUG", 0))       # 1: the card's f (the decay's matmuls) and its pre-sigmoid argument
WORKER_L1_M = (1536 - 39) * 1024 - 70656 - (24 << 10)   # the M-row build's device (main_m)


def units() -> list[tuple[int, int, int]]:
    """(first tile, count, type) of every work unit: q and k heads are 4 tiles (their norm needs the whole head)."""
    out = [(4 * h, 4, T_Q) for h in range(NV)] + [(256 + 4 * h, 4, T_K) for h in range(NV)]
    out += [(n, 1, T_V) for n in range(512, CONV // 32)] + [(T_FLO + j, 1, T_P) for j in range(8)]
    out += [(T_BETA + j, 1, T_B) for j in range(2)]
    return out


def assign(work, speed):
    """Greedy by tiles, the 4-tile heads first (so a core's heads lead its list and its 4-tile pushes never straddle
    the end of an 8-tile buffer); a unit goes to the core that would finish it first, the faster one on a tie."""
    n = len(speed)
    load, lists = [0] * n, [[] for _ in range(n)]
    for u in sorted(work, key=lambda u: -u[1]):
        c = min(range(n), key=lambda i: ((load[i] + u[1]) / speed[i], -speed[i]))
        lists[c].append(u)
        load[c] += u[1]
    for lst in lists:
        sizes = [u[1] for u in lst]
        assert sizes == sorted(sizes, reverse=True), "heads before single tiles"
    return lists


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Mixer:
    def __init__(self, dev, g: Gguf, rng, phases=7, states=None, out=None, in_place=False, hn=None, layer=KDA_LAYER,
                 check=True, pf=None, m_max=M_MAX, step=None):
        """states: (conv state [3, CONV], KDA state [NV, HK, HK]) instead of random ones; out: the tensor C writes
        (row-major fp32 [1, 1, 1, D]; default the check tensor); in_place: the new states over the old (the token
        loop), not into NaN-filled outputs (the check); hn: the input tensor (row-major bf16 [1, 1, 1, D]) when another
        program writes it (the mHC site's input norm); check: keep the float64 copies of the weights (the checks);
        pf: (an L1 prefetch buffer, pages): phase A's first pages come from that buffer, where the previous layer's FFN
        program put them (g3_fsh.FfnFused(pf=...), which asks a_ranges)."""
        self.dev, self.phases, self.states, self.out, self.in_place = dev, phases, states, out, in_place
        self.hn_given, self.check, self.pf, self.m_max, self.step_given = hn, check, pf, m_max, step
        self.grid = Grid(dev)
        p = f"blk.{layer}."
        r = {x: g(p + f"attn_{x}.weight") for x in "qkv"}                       # [8192, 4096]
        r.update(wq=r.pop("q"), wk=r.pop("k"), wv=r.pop("v"))
        for x in ("f_a", "f_b", "g_a", "g_b", "beta"):
            r[x] = g(p + f"ssm_{x}.weight")                                     # f_a [128, 4096], f_b [8192, 128]
        r["A"], r["dt"], r["o_norm"] = g(p + "ssm_a"), g(p + "ssm_dt.bias"), g(p + "ssm_norm.weight")
        r["wo"] = g(p + "attn_output.weight")                                   # [4096, 8192]
        r["taps"] = torch.cat([g(p + f"ssm_conv1d_{x}.weight").reshape(NV * HK, 4).T for x in "qkv"], dim=1)
        self.r = r                                                              # (the checks', dropped below)
        wa = torch.cat([r["wq"].T, r["wk"].T, r["wv"].T, r["f_a"].T, r["g_a"].T, r["beta"].T], dim=1)
        self.w, self.wq = self.stream(wa)                                       # [4096, 24896]
        self.wo, self.woq = self.stream(r["wo"].T)                              # [8192, 4096]
        # the low-rank second halves, a head's 32 tiles together: f_b_h's 16 (page 4 n + k), then g_b_h's
        blocks = []
        for h in range(NV):
            for m in ("f_b", "g_b"):
                wh = r[m][h * HK:(h + 1) * HK].T                                # [128 k, 128 n]
                blocks.append(wh.reshape(4, 32, 4, 32).permute(2, 0, 1, 3).reshape(-1, 32))
        self.lw = ttnn.from_torch(torch.cat(blocks).to(torch.bfloat16).reshape(1, 1, -1, 32), dtype=ttnn.bfloat8_b,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
        if check:
            lq = ttnn.to_torch(self.lw).double().reshape(NV, 2, 4, 4, 32, 32)   # (h, m, n, k, 32, 32)
            self.lwq = lq.permute(0, 1, 3, 4, 2, 5).reshape(NV, 2, HK, HK)      # (h, m) [k, n]
        self.na = (-r["A"]).double()                                            # ssm_a holds -exp(A_log)
        dtn = r["dt"].reshape(NV, HK).double() * self.na[:, None]
        self.dtn_host = dtn
        self.taps_t, self.dtn_t = self.rm(r["taps"].reshape(1, 1, 4, CONV)), self.rm(dtn.reshape(1, 1, NV, HK))
        self.nw_t = self.rm(r["o_norm"].reshape(1, 1, 1, HK))
        if not check:
            self.r = None
        self.set_inputs(rng)
        speeds = self.grid.speeds(NOC1_ROWS)
        self.lists = assign(units(), speeds)
        self.c_split, self.s_split = split_weighted(NT_C * S, speeds), split(NT_C, len(self.grid))
        self.max_c = max(sum(u[1] for u in lst if u[2] in (T_V, T_Q, T_K)) for lst in self.lists)
        # arena: hn (bf16), region A (fp32, 778 tiles x 32), B (bf16, 8192), P (C's partials: fp32, S x 4096)
        self.ra = KA * 64
        self.rb = self.ra + NT_A * 128
        self.rc = self.rb + NV * HK * 2
        self.arena = self.rc + S * D * 4
        if m_max > 1:  # hn mp rows; region A by owner (22 blocks), then region AC (7 rows a conv tile: the writer's)
            self.ra = m_max * KA * 64
            self.rb = self.ra + 22 * m_max * 128 + self.max_c * 7 * 128
            self.rc = self.rb + m_max * NV * HK * 2
            self.arena = self.rc + S * D * 4    # (5.2c: by owner)

    @property
    def a_src(self):
        return self.w

    def a_ranges(self, idx: int, n: int) -> list[tuple[int, int]]:
        """core idx's first n pages of phase A's weights, in its order, as (first page, count) ranges"""
        out = []
        for first, count, _ in self.lists[idx]:
            for t in range(first, first + count):
                if n <= 0:
                    return out
                out.append((t * KA, min(KA, n)))
                n -= KA
        return out

    def pf_args(self, idx: int) -> list[int]:
        if self.pf is None:
            return [0, 0]
        buf, n = self.pf
        return [buf.buffer_address(), sum(c for _, c in self.a_ranges(idx, n))]

    def rm(self, t):
        return ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.dev)

    def stream(self, w):
        """[k, n] -> bfp8_b tiles n-major (page n * kt + k), and the dequantized weights back as float64 [k, n]."""
        k, n = w.shape
        kt, nt = k // 32, n // 32
        st = w.contiguous().to(torch.bfloat16).reshape(kt, 32, nt, 32).permute(2, 0, 1, 3).reshape(1, 1, -1, 32)
        t = ttnn.from_torch(st, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.dev)
        if not self.check:
            return t, None
        return t, ttnn.to_torch(t).double().reshape(nt, kt, 32, 32).permute(1, 2, 0, 3).reshape(k, n)

    def set_inputs(self, rng):
        """Fresh hn, conv state, KDA state; every output pre-filled with NaN."""
        hn = (torch.from_numpy(rng.standard_normal(D).astype(np.float32)) if rng is not None
              else torch.zeros(D)).to(torch.bfloat16)                          # no rng: hn is written by the caller
        self.hn_host = hn.double()
        self.hn = self.hn_given if self.hn_given is not None else ttnn.from_torch(
            hn.reshape(1, 1, 1, D), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
        if self.states is not None:
            self.s0, self.S0 = (t.float().clone() for t in self.states)
        else:
            self.s0 = torch.from_numpy(rng.standard_normal((3, CONV)).astype(np.float32)) * 0.5
            self.S0 = torch.from_numpy(rng.standard_normal((NV, HK, HK)).astype(np.float32)) * 0.05
        if self.m_max > 1:   # two parities of 2 + m_max rows; the first pass reads parity 0's rows 0..2 (n_keep 1)
            st = torch.full((2 * (2 + self.m_max), CONV), float("nan"))
            st[:3] = self.s0
            self.state = self.rm(st.reshape(1, 1, -1, CONV))
            self.hn = self.hn_given if self.hn_given is not None else ttnn.from_torch(
                torch.zeros(1, 1, 1, self.m_max * D, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
            self.step_t = self.step_given if self.step_given is not None else ttnn.from_torch(   # a token's: shared
                step_words(1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
        else:
            self.state = self.rm(self.s0.reshape(1, 1, 3, CONV))
        tiles = self.S0.reshape(NV, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(1, 1, -1, 32)   # (h, j, c) tiles
        if self.m_max > 1:   # two parities of the committed states; the pending pages (Ki, U, G blocks of m_max rows)
            tiles = torch.cat([tiles, torch.full_like(tiles, float("nan"))], dim=2)
            pend = torch.full((2 * NV, 12, 2, self.m_max, 16), float("nan"))
            pend[:NV, :8] = 0.0
            pend[:NV, 8:] = 1.0                                   # the first pass commits nothing: S_a = S_c
            self.pend = self.rm(pend.reshape(1, 1, 2 * NV, -1))
        self.S = ttnn.from_torch(tiles, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.dev)
        nan = lambda shape, lay=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(  # noqa: E731
            torch.full(shape, float("nan")), dtype=ttnn.float32, layout=lay, device=self.dev)
        self.state_out = self.state if self.in_place or self.m_max > 1 else nan((1, 1, 3, CONV))
        self.S_out = self.S if self.in_place or self.m_max > 1 else nan((1, 1, NV * 16 * 32, 32), ttnn.TILE_LAYOUT)
        self.check_a = nan((1, 1, 1, NT_A * 32 * self.m_max))
        self.check_b = nan((1, 1, 1, NV * HK * self.m_max))
        self.check_c = self.out if self.out is not None else nan((1, 1, 1, D * self.m_max))
        self.dbg = nan((1, 1, NV, 2 * HK))
        self.prof = profile_tensor(self.dev, 32) if PROF else None

    def program(self):
        g = self.grid
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            lst = self.lists[idx]
            flat = [v for u in lst for v in u]
            head = idx if idx < NV else IDLE
            cf, cc = self.c_split[idx]
            s0, ns = self.s_split[idx]
            rargs[(cx, cy)] = [self.hn.buffer_address()] + g.args(idx) + [
                self.w.buffer_address(), self.state.buffer_address(), self.taps_t.buffer_address(), len(lst)] + \
                flat + [head, self.S.buffer_address(), self.nw_t.buffer_address(), self.lw.buffer_address(),
                        self.dtn_t.buffer_address(), self.wo.buffer_address(), cf, cc, s0, ns,
                        self.check_c.buffer_address(), self.ra, self.rb, self.rc] + self.pf_args(idx) + (
                [self.step_t.buffer_address(), self.pend.buffer_address()] if self.m_max > 1 else [])
            crt[cx][cy] = [len(lst)] + flat + [head, bits(float(self.na[idx])) if head != IDLE else 0, cc]
            wargs[(cx, cy)] = [self.state_out.buffer_address(), self.check_a.buffer_address()] + g.args(idx) + \
                [self.ra, len(lst)] + flat + [head, self.S_out.buffer_address(), self.check_b.buffer_address(),
                                             self.rb, cf, cc, self.rc]
            if self.m_max > 1:
                wargs[(cx, cy)] += [self.taps_t.buffer_address(), self.step_t.buffer_address()] + [
                    g.noc[h].x | g.noc[h].y << 16 for h in range(NV)] + [self.pend.buffer_address()]
                for it in range(cf, cf + cc):   # C's partial (tile n, part) to the owner of tile n (ksplit_sum_m)
                    n, part = divmod(it, S)
                    o = next(j for j, (a0, na) in enumerate(self.s_split) if a0 <= n < a0 + na)
                    a0, na = self.s_split[o]
                    wargs[(cx, cy)] += [g.noc[o].x | g.noc[o].y << 16, part * na + n - a0]
        acc = list(ttnn.TensorAccessorArgs(self.w).get_compile_time_args())
        defines = [("FK_DEBUG", str(self.dbg.buffer_address()))] if DEBUG else []
        if self.m_max > 1:
            defines.append(("M_MAX", str(self.m_max)))
        gp, gdef, gbytes = profile_plumbing(self.prof, 32, 63)
        defines += gdef
        for idx, (cx, cy) in enumerate(g.cores):
            rargs[(cx, cy)] += gp(idx)
            wargs[(cx, cy)] += gp(idx)
        kernels = g.dm_kernels(os.path.join(KDIR, "glm_fkda_reader.cpp"), os.path.join(KDIR, "glm_fkda_writer.cpp"),
                               [BATCH, DEPTH, self.phases, S] + acc, [self.phases, S] + acc, rargs, wargs,
                               defines=defines)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        exact = (3, 4, 6, 8, 10, 13, 14, 16, 24, 26, 32, 34) + ((35, 37, 38, 40, 42, 49, 51) if self.m_max > 1 else ())
        for cb in exact:    # exact: loaded straight into Dest in fp32
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_fkda_compute.cpp"),
                                             core_ranges=g.ranges, compile_time_args=[BATCH, self.phases, S],
                                             runtime_args=crt, config=cfg, defines=defines))
        f32, bf16, b8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
        spec = [(0, b8, W_PAGE, BATCH * DEPTH), (1, bf16, 2048, 2 * BATCH), (2, bf16, 2048, 1),
                (3, f32, 4096, 7 * self.max_c if self.m_max == 1 else 8), (4, f32, 4096, 4), (5, f32, 4096, 4), (6, f32, 4096, 1),
                (7, f32, 4096, 4), (8, f32, 4096, 16), (9, f32, 4096, 16), (10, f32, 4096, 4),
                (11, f32, 4096, 4), (12, f32, 4096, 4), (13, f32, 4096, 10), (14, f32, 4096, 4),
                (15, f32, 4096, 4), (16, f32, 4096, 8), (17, f32, 4096, 8), (18, f32, 4096, 12),
                (19, f32, 4096, 8), (20, f32, 4096, 16), (21, bf16, 2048, 4), (22, f32, 4096, 4),
                (23, f32, 4096, 4), (24, f32, 4096, 4), (25, f32, 4096, 4), (26, f32, 4096, 4),
                (27, b8, W_PAGE, 32), (28, bf16, 2048, 8), (29, ttnn.uint8, 4096, 1),
                (30, ttnn.uint8, self.arena, 1), (31, ttnn.uint8, 1024 if self.m_max == 1 else 4096, 1), (32, f32, 4096, 4),
                (33, f32, 4096, 4), (34, f32, 4096, 4)] + ([(63, ttnn.uint8, 512, 1)] if PROF else []) + ([(35, f32, 4096, 4), (36, f32, 4096, 4)] if DEBUG else [])
        if self.m_max > 1:   # B at M rows (glm_fkda_compute.cpp phase_b_m); cbs 9, 18, 25 unused
            spec = [x if x[0] not in (9, 18, 25, 31) else (x[0], x[1], x[2], 1) for x in spec]
            spec = [x if x[0] != 31 else (31, ttnn.uint8, 8192, 1) for x in spec]
            spec += [(35, f32, 4096, 8)] + [(i, f32, 4096, 4) for i in range(36, 54)]
        self.cb_kb = sum(s * n for _, _, s, n in spec) / 1024
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        ins = list({id(t): t for t in [self.hn, self.w, self.wo, self.lw, self.state, self.taps_t, self.dtn_t,
                                       self.nw_t, self.S, self.state_out, self.S_out, self.check_a, self.check_b,
                                       self.check_c, self.dbg] + ([self.prof] if PROF else []) +
                         ([self.step_t, self.pend] if self.m_max > 1 else []) +
                         ([self.pf[0]] if self.pf is not None else [])}.values())
        return lambda: ttnn.generic_op(ins, prog)

    def reference(self):
        """float64 on the card's inputs; each phase from the card's own previous outputs (as the card reads them)."""
        y = self.hn_host @ self.wq                                               # [24896]
        s, taps = self.s0.double(), self.r["taps"].double()
        x = torch.nn.functional.silu((torch.cat([s, y[None, :CONV]]) * taps).sum(0))
        l2 = lambda t: t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + L2_EPS)  # noqa: E731
        q, k = l2(x[:NV * HK].reshape(NV, HK)) * HK ** -0.5, l2(x[NV * HK:2 * NV * HK].reshape(NV, HK))
        A = torch.cat([q.flatten(), k.flatten(), x[2 * NV * HK:], y[CONV:CONV + 2 * LO],
                       torch.sigmoid(y[CONV + 2 * LO:])])
        # B from the card's A (the low-rank halves rounded to bf16, as the reader does)
        a = ttnn.to_torch(self.check_a).double().flatten()
        q, k, v = (a[i * NV * HK:(i + 1) * NV * HK].reshape(NV, HK) for i in range(3))
        flo, glo = (a[CONV + i * LO:CONV + (i + 1) * LO].to(torch.bfloat16).double() for i in range(2))
        b = a[CONV + 2 * LO:CONV + 2 * LO + NV]
        f = torch.einsum("k,hkn->hn", flo, self.lwq[:, 0])
        d = torch.exp(KDA_LB * torch.sigmoid(f * self.na[:, None] + self.dtn_host))
        gate = torch.sigmoid(torch.einsum("k,hkn->hn", glo, self.lwq[:, 1]))
        self._f = f
        S = self.S0.double() * d[:, :, None]
        delta = (v - torch.einsum("hk,hkv->hv", k, S)) * b[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        o = torch.einsum("hk,hkv->hv", q, S)
        B = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + NORM_EPS) * self.r["o_norm"].double() * gate).flatten()
        # C from the card's B, in bf16 as the card reads it
        C = ttnn.to_torch(self.check_b).double().flatten().to(torch.bfloat16).double() @ self.woq
        return A, torch.cat([s[1:], y[None, :CONV]]), B, S, C

    def errors(self) -> dict:
        A, sref, B, S, C = self.reference()
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()  # noqa: E731
        got = ttnn.to_torch(self.check_a).double().flatten()
        segs = {"q": (0, NV * HK), "k": (NV * HK, 2 * NV * HK), "v": (2 * NV * HK, CONV),
                "f_lo": (CONV, CONV + LO), "g_lo": (CONV + LO, CONV + 2 * LO),
                "beta": (CONV + 2 * LO, CONV + 2 * LO + NV)}
        res = {k: rel(got[a:b], A[a:b]) for k, (a, b) in segs.items()}
        res["conv_state"] = rel(ttnn.to_torch(self.state_out).double().reshape(3, CONV), sref)
        if DEBUG and self.phases & 2:
            dg = ttnn.to_torch(self.dbg).double().reshape(NV, 2, HK)
            f, na, dtn = self._f, self.na[:, None], self.dtn_host
            res["dbg_f"], res["dbg_arg"] = rel(dg[:, 0], f), rel(dg[:, 1], f * na + dtn)
        if self.phases & 2:
            res["B_out"] = rel(ttnn.to_torch(self.check_b).double().flatten(), B)
            st = ttnn.to_torch(self.S_out).double().reshape(NV, 4, 4, 32, 32).permute(0, 1, 3, 2, 4)
            res["B_state"] = rel(st.reshape(NV, HK, HK), S)
        if self.phases & 4:
            res["C_out"] = rel(ttnn.to_torch(self.check_c).double().flatten(), C)
        return res

    # ---- M rows: a pass's inputs and its checks against float64 (tt/glm/kda_m.py's conv window)
    def set_pass(self, M, n_keep, parity, rng):
        """A verify pass's M rows of hn (random), its step words; A's check refilled with NaN."""
        self.M, self.mp, self.n_keep, self.parity = M, mp_of(M), n_keep, parity
        hn = torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32)).to(torch.bfloat16)
        self.hn_rows = hn.double()
        full = torch.zeros(self.m_max * D, dtype=torch.bfloat16)
        flat = blocks(pad_rows(hn), self.mp)
        full[:flat.numel()] = flat
        rm = ttnn.ROW_MAJOR_LAYOUT
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(full.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=rm),
                                        self.hn)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(M, n_keep, parity), dtype=ttnn.uint32, layout=rm),
                                        self.step_t)
        for t in (self.check_a, self.check_b, self.check_c):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full(tuple(t.shape), float("nan")),
                                                            dtype=ttnn.float32, layout=rm), t)

    def reference_a_m(self, hist):
        """float64 phase A at the pass's rows from history hist [3, CONV]: A's rows [M, NT_A * 32], the window."""
        y = self.hn_rows @ self.wq                                              # [M, 24896]
        win = torch.cat([hist, y[:, :CONV]])
        taps = self.r["taps"].double()
        x = torch.nn.functional.silu(torch.stack([(win[t:t + 4] * taps).sum(0) for t in range(self.M)]))
        l2 = lambda t: t / torch.sqrt(t.pow(2).sum(-1, keepdim=True) + L2_EPS)  # noqa: E731
        q = l2(x[:, :NV * HK].reshape(self.M, NV, HK)) * HK ** -0.5
        k = l2(x[:, NV * HK:2 * NV * HK].reshape(self.M, NV, HK))
        A = torch.cat([q.flatten(1), k.flatten(1), x[:, 2 * NV * HK:], y[:, CONV:CONV + 2 * LO],
                       torch.sigmoid(y[:, CONV + 2 * LO:])], dim=1)
        return A, win

    def errors_m(self, hist) -> tuple[dict, torch.Tensor]:
        """A's rows and the extended history (the other parity's rows) against float64; returns them and the window."""
        A, win = self.reference_a_m(hist)
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()  # noqa: E731
        got = unblocks(ttnn.to_torch(self.check_a).double().flatten(), NT_A, self.mp)[:self.M]
        segs = {"q": (0, NV * HK), "k": (NV * HK, 2 * NV * HK), "v": (2 * NV * HK, CONV),
                "f_lo": (CONV, CONV + LO), "g_lo": (CONV + LO, CONV + 2 * LO),
                "beta": (CONV + 2 * LO, CONV + 2 * LO + NV)}
        res = {k: rel(got[:, a:b], A[:, a:b]) for k, (a, b) in segs.items()}
        o = (1 - self.parity) * (2 + self.m_max)
        st = ttnn.to_torch(self.state).double().reshape(-1, CONV)
        res["conv_history"] = rel(st[o:o + self.M + 2], win[1:])
        return res, win

    def load_states(self, conv: torch.Tensor, S: torch.Tensor) -> None:
        """A prefill's hand-off: conv (the last 3 rows of q|k|v before the conv, oldest first [3, CONV]) and S
        [NV, HK, HK] into the card's tensors, as built: the M-row build's parity 0 (the next pass n_keep 1, parity
        0), parity 1 zero, the pending pages committing nothing."""
        put = lambda t, ref, lay: ttnn.copy_host_to_device_tensor(  # noqa: E731
            ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=lay), ref)
        tiles = S.float().reshape(NV, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(1, 1, -1, 32)
        if self.m_max > 1:
            st = torch.zeros(tuple(self.state.shape)).reshape(-1, CONV)
            st[:3] = conv
            put(st.reshape(tuple(self.state.shape)), self.state, ttnn.ROW_MAJOR_LAYOUT)
            put(torch.cat([tiles, torch.zeros_like(tiles)], dim=2), self.S, ttnn.TILE_LAYOUT)
            pend = torch.zeros(tuple(self.pend.shape)).reshape(-1, 12, 2 * self.m_max * 16)
            pend[:, 8:] = 1.0
            put(pend.reshape(tuple(self.pend.shape)), self.pend, ttnn.ROW_MAJOR_LAYOUT)
        else:
            put(conv.reshape(1, 1, 3, CONV), self.state, ttnn.ROW_MAJOR_LAYOUT)
            put(tiles, self.S, ttnn.TILE_LAYOUT)

    def states_m(self):
        """The card's committed states [2, NV, HK, HK] and pending rows (Ki, U, G: [2, NV, m_max, HK] each)."""
        st = ttnn.to_torch(self.S).double().reshape(2, NV, 4, 4, 32, 32).permute(0, 1, 2, 4, 3, 5)
        pd = ttnn.to_torch(self.pend).double().reshape(2, NV, 12, 2, self.m_max, 16).permute(0, 1, 2, 4, 3, 5)
        pd = pd.reshape(2, NV, 3, 4, self.m_max, 32).permute(0, 1, 2, 4, 3, 5).reshape(2, NV, 3, self.m_max, HK)
        return st.reshape(2, NV, HK, HK), pd[:, :, 0], pd[:, :, 1], pd[:, :, 2]

    def b_inputs(self):
        """B's inputs per row from the card's A (as the reader takes them): q, k, v [M, NV, HK], beta [M, NV], the
        log-decays l and the gate [M, NV, HK]."""
        a = unblocks(ttnn.to_torch(self.check_a).double().flatten(), NT_A, self.mp)[:self.M]
        q, k, v = (a[:, i * NV * HK:(i + 1) * NV * HK].reshape(self.M, NV, HK) for i in range(3))
        flo, glo = (a[:, CONV + i * LO:CONV + (i + 1) * LO].to(torch.bfloat16).double() for i in range(2))
        b = a[:, CONV + 2 * LO:CONV + 2 * LO + NV]
        f = torch.einsum("mk,hkn->mhn", flo, self.lwq[:, 0])
        l = KDA_LB * torch.sigmoid(f * self.na[None, :, None] + self.dtn_host[None])
        gate = torch.sigmoid(torch.einsum("mk,hkn->mhn", glo, self.lwq[:, 1]))
        return q, k, v, b, l, gate

    def norm_out(self, O, gate):
        """[M, NV, HK] -> the gated norm's rows [M, NV * HK]"""
        w = self.r["o_norm"].double()
        return (O * torch.rsqrt(O.pow(2).mean(-1, keepdim=True) + NORM_EPS) * w * gate).reshape(O.shape[0], -1)

    def errors_b_m(self, before, S_seq) -> tuple[dict, torch.Tensor]:
        """B against float64 (tt/glm/kda_m.py's chunked form) from the card's inputs: its A rows, the parity's state
        and pending rows as they were before the pass (before = states_m()); and the one-row steps chained from S_seq
        (the state after the rows kept so far). Returns the errors and the chain's state after the rows kept next."""
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()  # noqa: E731
        q, k, v, b, l, gate = self.b_inputs()
        St, Ki_p, U_p, G_p = (t[self.parity] for t in before)
        a = self.n_keep
        G_last = G_p[:, a - 1]
        Kc = Ki_p[:, :a] * G_last[:, None]
        S_a = St * G_last[:, :, None] + torch.einsum("hjk,hjv->hkv", Kc, U_p[:, :a])
        tr = lambda t: t.transpose(0, 1).numpy()  # noqa: E731
        O, (K, U, L) = kda_m.rows(S_a.numpy(), tr(q), tr(k), tr(v), b.T.numpy(), tr(l))
        O = torch.from_numpy(O).transpose(0, 1)
        after = self.states_m()
        o = 1 - self.parity
        got = unblocks(ttnn.to_torch(self.check_b).double().flatten(), NV * 4, self.mp)[:self.M]
        res = {"B_out": rel(got, self.norm_out(O, gate)), "B_commit": rel(after[0][o], S_a)}
        Lt = torch.from_numpy(L)
        for name, want, have in (("B_pend_Ki", torch.from_numpy(K) * torch.exp(-Lt), after[1][o]),
                                 ("B_pend_U", torch.from_numpy(U), after[2][o]), ("B_pend_G", torch.exp(Lt), after[3][o])):
            res[name] = rel(have[:, :self.M], want)
        # the one-row steps from the kept prefix's state
        S, outs, kept = S_seq.numpy(), [], None
        for t in range(self.M):
            ot, S = kda_m.step(S, q[t].numpy(), k[t].numpy(), v[t].numpy(), b[t].numpy(), l[t].numpy())
            outs.append(torch.from_numpy(ot))
            if t == self.kept - 1:
                kept = torch.from_numpy(S)
        res["B_out_vs_steps"] = rel(got, self.norm_out(torch.stack(outs), gate))
        if self.phases & 4:   # C from the card's B rows, in bf16 as the card reads them
            want = got.to(torch.bfloat16).double() @ self.woq
            res["C_out"] = rel(unblocks(ttnn.to_torch(self.check_c).double().flatten(), D // 32, self.mp)[:self.M], want)
        return res, kept

    def end_to_end(self) -> dict:
        """C against G2's float32 mixer_ref (llama.cpp's graph, unquantized weights) on the same inputs."""
        ref = dict(self.r, conv0=self.s0, S0=self.S0)
        want = KdaCard.mixer_ref(SimpleNamespace(ref=ref), self.hn_host.float()).double()
        got = ttnn.to_torch(self.check_c).double().flatten()
        return {"pcc": round(pcc(got, want), 6), "max_rel_err": ((got - want).abs().max() / want.abs().max()).item()}


def main_m() -> None:
    """Phase A at M rows over consecutive passes keeping random rows, against float64 over the kept prefix."""
    os.makedirs(OUT, exist_ok=True)
    seed = int(time.time())
    rng = np.random.default_rng(seed)
    g = Gguf(MODEL)
    # the M-row kernels' text (~89 KB) overflows the default 70,656 B kernel config buffer: 24 KB more of it out of the
    # worker L1 (Flash-Next's WORKER_L1_M does the same)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1_M)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": KDA_LAYER, "m_max": M_MAX, "phases": PHASES,
           "seed": seed, "passes": []}
    try:
        m = Mixer(dev, g, rng, PHASES, m_max=M_MAX)
        run = m.program()
        res["cb_kb_per_core"] = round(m.cb_kb, 1)
        print(f"M_MAX {M_MAX}: circular buffers {m.cb_kb:.0f} KB a core, max conv tiles {m.max_c}", flush=True)
        hist, n_keep, parity, S_seq = m.s0.double(), 1, 0, m.S0.double()
        for M in [M for M in (4, 1, 3, 4, 2, 4, 3, 1, 2, 4) if M <= M_MAX]:
            m.set_pass(M, n_keep, parity, rng)
            before = m.states_m() if PHASES & 2 else None
            run()
            ttnn.synchronize_device(dev)
            err, win = m.errors_m(hist)
            a = int(rng.integers(1, M + 1))
            m.kept = a
            if PHASES & 2:
                eb, S_seq = m.errors_b_m(before, S_seq)
                err.update(eb)
            res["passes"].append({"M": M, "n_keep_in": n_keep, "parity": parity, "kept": a,
                                  "max_rel_err": err})
            print(f"M {M} (n_keep {n_keep}, parity {parity}): " + ", ".join(f"{k} {v:.1e}" for k, v in err.items()),
                  flush=True)
            hist, n_keep, parity = win[a:a + 3], a, 1 - parity
        worst = max(v for p in res["passes"] for v in p["max_rel_err"].values())
        res["worst"] = worst
        for M in [M for M in (1, 2, 3, 4) if M <= M_MAX]:
            m.set_pass(M, 1, 0, rng)
            res[f"traced_us_M{M}"] = round(traced_us(dev, run), 1)
        print(f"worst {worst:.1e}; traced " + ", ".join(f"{k} {v}" for k, v in res.items() if k.startswith("traced")),
              flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_fkda_m{M_MAX}_p{PHASES}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")


def main() -> None:
    if M_MAX > 1:
        return main_m()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": KDA_LAYER, "phases": PHASES, "S": S}
    try:
        m = Mixer(dev, g, rng, PHASES)
        run = m.program()
        res["cb_kb_per_core"] = round(m.cb_kb, 1)
        print(f"weights in {time.time() - t0:.0f} s; max tiles a core {max(sum(u[1] for u in lst) for lst in m.lists)}"
              f", max conv tiles {m.max_c}, circular buffers {m.cb_kb:.0f} KB a core", flush=True)
        run()
        ttnn.synchronize_device(dev)
        res["max_rel_err"] = m.errors()
        print("max rel err: " + ", ".join(f"{k} {v:.1e}" for k, v in res["max_rel_err"].items()), flush=True)
        if PHASES == 7:
            res["vs_mixer_ref"] = m.end_to_end()
            print("against G2's float32 mixer_ref:", json.dumps(res["vs_mixer_ref"]), flush=True)
        res["traced_us"] = round(traced_us(dev, run), 1)
        if PROF:
            run()
            ttnn.synchronize_device(dev)
            w = profile_words(m.prof, len(m.grid), 32).double()     # [cores, 128]: wait n, release 32 + n, arrive 64 + n, marks 96 + k
            co = 0                                                   # the coordinator: barrier releases on one clock
            start, rel = w[co, 96], w[co, 32 + 1:32 + 4]
            ghz = 1.35
            phase = torch.diff(torch.cat([start[None], rel])) / (ghz * 1e3)
            res["prof_phase_us"] = [round(float(x), 1) for x in phase]
            # each core's work in phase n: its writer's arrival n minus its reader's release n - 1 (start for n = 1)
            prev = torch.cat([w[:, 96:97], w[:, 33:35]], dim=1)
            work = (w[:, 65:68] - prev) / (ghz * 1e3)
            res["prof_slowest_core_us"] = [round(float(work[:, j].max()), 1) for j in range(3)]
            res["prof_median_core_us"] = [round(float(work[:, j].median()), 1) for j in range(3)]
            res["prof_slowest_core"] = [int(work[:, j].argmax()) for j in range(3)]
            print("profile (1.35 GHz): phase", res["prof_phase_us"], "slowest core", res["prof_slowest_core_us"],
                  "median", res["prof_median_core_us"], "which", res["prof_slowest_core"], flush=True)
        mb = (m.w.volume() + m.wo.volume() + m.lw.volume()) * 1.0625 / 1e6
        res["weights_mb"] = round(mb, 1)
        print(f"phases {PHASES}: traced {res['traced_us']} us ({mb:.1f} MB of weights; the stock mixer 1,174.6 us)",
              flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_fkda{'' if PHASES == 7 else f'_p{PHASES}'}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
