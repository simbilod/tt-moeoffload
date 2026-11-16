"""G3 step 4: one decode token through a whole Flash-Next gated-DeltaNet layer, except its routed experts, as ONE
program on all 110 cores (tt/kernels/flayer_*.cpp): 9 phases with a grid barrier after each, real blk.0 weights.

  H1a, H2a   the attention half's hyper-connection mix (g3_hc.py)
  A, B, C    the DeltaNet mixer (g3_gdn.py), C's epilogue the combine X += w_a * out
  H1f, H2f   the FFN half's hyper-connection mix on the new X
  R1         the router (512 logits + the shared expert's gate logit, bf16 weights) and the shared expert's gate|up
             with h = silu(g) * u
  R2         the shared expert's down, * sigmoid(gate logit), + the routed experts' sum (a random stand-in: the card's
             and the CPU's experts are priced separately), the combine X += w_f * (sh + routed)

Between R1 and R2 the real decode loop hands the logits to the host (top-10 and the experts), which is not part of this
program. Checked against float64 per phase from the card's own previous outputs, on NaN-filled outputs with fresh
inputs, then timed (traced, one launch) against G2's 1,112.8 us for the stock ops.

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/flashnext/g3_layer.py

Measured 2026-10-23: 389.7 us traced for the layer step (415.8 before the HL phase) against 1,112.8 us for G2's
stock ops; every phase within 1.1e-3 of float64, the router's top 10 the same as the reference's. 8 passes through
the layer in one launch take 2,981.6 us: 370.1 us per layer without a launch, the steady state of a persistent
decode loop (passes after the first were wrong until a 4-tile push into an 8-tile buffer, straddling its end from
the second pass on, became single-tile pushes; g3_layers.py checks consecutive passes). The ~84 MB of weights need
~215 us at the GEMV's 391 GB/s. Then (g3_profile.py) A's per-tile constants prefetched while card memory idles
(356.2 us per layer), rows 0, 1, 9 reading through NOC1 (fused/grid.py NOC1_ROWS; 333.9), and the hyper-connection
mixes' first weights issued before the barrier that precedes each of their GEMVs: 349.8 us traced, 320.6 us per layer.

Layer(m_max=4) builds the same program for a verify pass of up to 4 rows (flayer_reader.cpp M_MAX; M, n_keep and the
states' parity from the step tensor, words 1..4): set_inputs_m / set_step, errors_m per row; checked and timed by
g6_layer_m.py, opened with WORKER_L1_M. m_max = 1 (the default, the token loop) compiles the one-row source unchanged.
With the host link's defines (g4_model.Link) the M-row program exchanges each layer's M rows (g4_model.DeltaRun with a
step tensor, Link.serve_m; checked and timed by g6_run_m.py); set_states_m / zero_states give it a sequence's states.
With card_on the M-row program computes the card's cached share of the routed experts for its rows (
fused/dexp_*.h: the codes in cb 58, the rest borrowed; its kernels need WORKER_L1_MC's larger config buffer; checked
by g5_layer.py G5_M, with the link by g6_run_m.py).
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
from fused.grid import NOC1_ROWS, Grid, profile_defines, traced_us  # noqa: E402
from g2_layers import MODEL, Gguf  # noqa: E402
from g3_gdn import CONV, D, EPS, HK, NA, NA_T, NK, NT, NV, T_A, T_K, T_Q, T_V, ZD, assign, nan_tensor, units  # noqa
from g3_hc import HC, LO, LR, model_tag, stream  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
LAYER = 0
BATCH, DEPTH = 8, 4
IDLE = 0xFFFFFFFF
N_ROUTER, FF = 17, 640
WORKER_L1 = (1536 - 39) * 1024 - 70656 - (64 << 10)
WORKER_L1_M = WORKER_L1 - (8 << 10)   # the M-row program (m_max > 1): its kernels take ~4.5 KB more config buffer
WORKER_L1_MC = WORKER_L1_M - (8 << 10)  # with the card's experts: ~5 KB more again (its buffers are borrowed)
EXACT = (3, 6, 9, 11, 12, 13, 16, 17, 18, 20, 23, 25, 28, 30, 35, 37, 41, 42, 43, 51, 52, 53, 54, 55)


def rel(a, b) -> float:
    return ((a - b).abs().max() / b.abs().max()).item()


def bf(t):
    return t.to(torch.bfloat16).double()


# the M-row layout's host helpers, shared with GLM's programs (re-exported here for Flash-Next's scripts)
from fused.mrows import blocks, mp_of, step_words, unblocks  # noqa: E402,F401


def check_core_map(g) -> None:
    """fused/dexp_dm.h core_addr's NOC coordinates of a core index (from the grid's rectangles) are the grid's own."""
    gx = sum(x1 - x0 + 1 for x0, _, x1, _, _ in g.rects)
    for idx, n in enumerate(g.noc):
        lx, ly = idx % gx, idx // gx
        for x0, y0, x1, _, _ in g.rects:
            if lx <= x1 - x0:
                break
            lx -= x1 - x0 + 1
        assert (n.x, n.y) == (x0 + lx, y0 + ly), f"core {idx}: NOC {n.x, n.y}, core_addr {x0 + lx, y0 + ly}"


def state_tiles(S):
    """States [..., HK, HK] -> their 16 tiles each as the tile layout's rows [n x 16 x 32, 32]."""
    return S.reshape(-1, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32)


def raw_tiles(raw):
    """Pages of 1024 words in an fp32 tile's L1 order (4 faces of 16 x 16) -> the tiles' rows [n x 32, 32]."""
    return raw.reshape(-1, 2, 2, 16, 16).permute(0, 1, 3, 2, 4).reshape(-1, 32)


def tile_raw(rows):
    """raw_tiles' inverse: tile rows [n x 32, 32] -> pages [n, 1024]."""
    return rows.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1024)


def pend_pages(K, U, d):
    """B's pending rows (M rows, flayer_reader.cpp): k, U [NV, 4, HK] and the decays [NV, 4] -> pages [NV, 2, 1024]:
    page 0 tile t's face-f rows 0..3 at word t * 128 + f * 64 (U's after k's), page 1 the decays in words 0..3."""
    face = lambda x: x.reshape(NV, 4, 4, 2, 16).permute(0, 2, 3, 1, 4).reshape(NV, 512)   # noqa: E731
    p = torch.zeros(NV, 2, 1024, dtype=K.dtype)
    p[:, 0] = torch.cat([face(K), face(U)], 1)
    p[:, 1, :4] = d
    return p


def commit(S, K, U, d, n):
    """The state after a pass's first n rows from the state it started from and its pending rows (float64):
    gamma_a S + sum_j c_j k_j u_j^T, a = n - 1, c_j = d_{j+1} .. d_a, gamma_a = d_0 .. d_a."""
    c = torch.stack([d[:, j + 1:n].prod(1) for j in range(n)], 1)
    return S * d[:, :n].prod(1)[:, None, None] + torch.einsum("hj,hjk,hjv->hkv", c, K[:, :n], U[:, :n])


def conv_step(win, taps):
    """The causal conv + silu on a window of 4 inputs (oldest first) of the conv channels."""
    return torch.nn.functional.silu((win * taps).sum(0))


def delta_step(S, a, nw):
    """One token's delta rule and gated norm from A's outputs a (the card's layout): (out [NV x HK], new S)."""
    q, k = a[:2048].reshape(NK, HK).repeat(3, 1), a[2048:4096].reshape(NK, HK).repeat(3, 1)
    v, z = a[4096:CONV].reshape(NV, HK), a[CONV:CONV + ZD].reshape(NV, HK)
    d, b = a[NA_T * 32:NA_T * 32 + NV], a[NA_T * 32 + NA:NA_T * 32 + NA + NV]
    S = S * d[:, None, None]
    S = S + k[:, :, None] * ((v - torch.einsum("hk,hkv->hv", k, S)) * b[:, None])[:, None, :]
    o = torch.einsum("hk,hkv->hv", q, S)
    return (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * nw * z).flatten(), S


class Layer:
    def __init__(self, dev, g: Gguf, rng, layer=LAYER, speed=True, back=True, m_max=1):
        """back=False: no float64 host copies of the weights (the token loop; errors() then cannot run).
        m_max > 1: the M-row program (flayer_reader.cpp M_MAX) for up to m_max (<= 4) rows a pass, M from the step
        tensor (set_step), the owner arena, X tensors a stream a page, the states in two parities."""
        assert m_max in (1, 2, 4), "M_MAX: 1, 2 or 4 (M = 3 runs in the 4-row layout)"
        self.dev, self.layer, self.m_max = dev, layer, m_max
        self.grid = Grid(dev)
        p = f"blk.{layer}."
        rm = self.rm
        tag = model_tag(g)
        key = lambda name: f"{tag}/{p}{name}"                                   # noqa: E731  (g3_hc.stream's cache)
        self.hc = {}
        for site in ("attn", "ffn"):
            norm = g(f"{p}hc_{site}_norm.weight")

            def wd(site=site, norm=norm):
                down, inject = g(f"{p}hc_{site}_down.weight"), g(f"{p}hc_{site}_inject.weight")
                w = torch.zeros(HC * D, LO)
                w[:, :LR + HC] = (torch.cat([down, inject]) * norm[None, :]).T
                return w
            (wdt, wdq) = stream(dev, wd, back, key(f"hc_{site}_wd"))
            (wut, wuq) = stream(dev, lambda site=site: g(f"{p}hc_{site}_up.weight").T, back, key(f"hc_{site}_wu"))
            self.hc[site] = dict(norm=norm, wd=wdt, wdq=wdq, wu=wut, wuq=wuq, norm_t=rm(norm))
        pad = lambda m: torch.cat([m, torch.zeros(NA - NV, D)])                  # noqa: E731
        self.wa, self.waq = stream(dev, lambda: torch.cat([g(p + "attn_qkv.weight"), g(p + "attn_gate.weight"),
                                                           pad(g(p + "ssm_alpha.weight")),
                                                           pad(g(p + "ssm_beta.weight"))]).T, back, key("wa"))
        self.wo, self.woq = stream(dev, lambda: g(p + "ssm_out.weight").T, back, key("wo"))
        self.taps = g(p + "ssm_conv1d.weight").T.contiguous()
        dtna = torch.zeros(2, NA)
        dtna[0, :NV], dtna[1, :NV] = g(p + "ssm_dt.bias"), g(p + "ssm_a")
        self.dtna, self.nw = dtna, g(p + "ssm_norm.weight")
        self.taps_t, self.dtna_t, self.nw_t = rm(self.taps), rm(dtna), rm(self.nw)
        # R1: the router [2560, 544] in bf16 (512 logits, the shared gate logit, padding); gate|up interleaved by tile
        wr = torch.zeros(D, N_ROUTER * 32)
        wr[:, :512], wr[:, 512] = g(p + "ffn_gate_inp.weight").T, g(p + "ffn_gate_inp_shexp.weight")
        st = wr.to(torch.bfloat16).reshape(D // 32, 32, N_ROUTER, 32).permute(2, 0, 1, 3).reshape(1, 1, -1, 32)
        self.wr = ttnn.from_torch(st, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        self.wrq = bf(wr)

        def gu():
            gate, up = g(p + "ffn_gate_shexp.weight").T, g(p + "ffn_up_shexp.weight").T  # [2560, 640]
            return torch.stack([gate.reshape(D, FF // 32, 32), up.reshape(D, FF // 32, 32)], 2).reshape(D, 2 * FF)
        self.wgu, self.wguq = stream(dev, gu, back, key("wgu"))
        self.wsd, self.wsdq = stream(dev, lambda: g(p + "ffn_down_shexp.weight").T, back, key("wsd"))  # [640, 2560]
        self.speed = speed
        self.set_noc1(NOC1_ROWS)    # logical rows whose reader streams through NOC1 (writer NOC0)
        o, a = {}, 0
        mm = m_max
        if mm == 1:
            regions = (("x32", HC * D * 4), ("x16", HC * D * 2), ("p1", HC * LO * 4), ("rs", 64),
                       ("hna", D * 2), ("loa", 640), ("wa", 64), ("hnf", D * 2), ("lof", 640), ("wf", 64),
                       ("a", NT * 128), ("b", NV * HK * 2), ("l", N_ROUTER * 128), ("sh", FF * 2), ("rt", D * 4),
                       ("tab", 256), ("prof", 64), ("rq", 128))
        else:   # the owner layout (flayer_reader.cpp): x32 / x16 an H1 core's stream, rs the slot of xown (the owner's
            # tile of the 4 streams), p1 then rs, lo then w, l then sh by M; hn one region for both halves
            regions = (("x32", mm * D * 4), ("x16", mm * D * 2), ("rs", mm * HC * 128),
                       ("p1", mm * (HC * LO * 4 + 256)),
                       ("hna", mm * D * 2), ("loa", mm * 704), ("lof", mm * 704), ("a", mm * 18 * 128),
                       ("b", mm * NV * HK * 2), ("l", mm * (N_ROUTER * 128 + FF * 2)), ("tab", 448), ("prof", 64),
                       ("rq", 512))   # (tab + 384: B's previous decays; rq: fused/dexp_dm.h RQ_BYTES)
        for name, size in regions + (("ac", 0),):  # AC last: its size follows the assignment (program())
            o[name] = a
            a += size
        if mm > 1:
            o["hnf"] = o["hna"]
            o["wa"] = o["wf"] = o["sh"] = o["rt"] = 0   # (placed by M at runtime; no host link)
        self.off, self.arena = o, a
        self.defines = []           # g3_link.py adds the host link's (EXCHANGE, the socket config addresses)
        self.table_addr = 0         # g3_layers.py: a layer table (a row of tensor addresses per layer)
        self.gprof = None           # g3_profile.py: grid.h's per-core barrier clocks (fused/grid.py profile_tensor)
        self.gprof_n = 32           # the barriers it keeps (g4_phases.py: fused/grid.py PROFILE_N)
        self.inplace = False        # the token loop (g4_model.py): X after R2 back into x32_t / x16_t, not the check
        self.checks_inplace = False  # in place, the check tensors are written only with this (g3_layers.py compares)
        self.card = None            # g4_model.py: this layer's card expert cache (g5_cache.py), tensors by name
        self.card_on = False        # the program computes the card's routed experts between R1 and R2 (dexp_*.h)
        self.part = None            # ... into these partial rows (fp32 [800 x 32], shared by a run's layers)
        self.card_routes = None     # without the host link (g5_layer.py): the card's routes from this tensor
        self.set_inputs(rng)

    def zero_states(self):
        """The conv and DeltaNet states of a sequence's start (the decode loop's; set_inputs' are random)."""
        if self.m_max > 1:
            return self.set_states_m(torch.zeros(3, CONV), torch.zeros(NV, HK, HK))
        self.s0 = torch.zeros(3, CONV)
        self.cstate = self.rm(self.s0)
        self.S0 = torch.zeros(NV, HK, HK)
        self.S = ttnn.from_torch(torch.zeros(1, 1, NV * 16 * 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                 device=self.dev)

    def set_states_m(self, s0, S0, parity=0):
        """M rows: the states of a pass at parity `parity` that keeps n_keep = 1 row: the conv history s0 [3, CONV]
        (oldest first) at rows 0..2 of the parity's buffer, the state S0 [NV, HK, HK], a pending page that commits to
        S0 itself (d = 1, k = U = 0); the other parity zero. Written in place (a layer table keeps the addresses)."""
        mm = self.m_max
        self.s0, self.S0 = s0.float(), S0.float()
        cs = torch.zeros(2 * (2 + mm), CONV)
        cs[parity * (2 + mm):parity * (2 + mm) + 3] = self.s0
        St = torch.zeros(2, NV, HK, HK)
        St[parity] = self.S0
        pend = torch.zeros(2, NV, 2, 1024)
        pend[parity, :, 1, 0] = 1.0
        tiles = torch.cat([state_tiles(St), raw_tiles(pend)]).reshape(1, 1, -1, 32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(cs.reshape(1, 1, -1, CONV), dtype=ttnn.float32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT), self.cstate)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(tiles, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.S)

    def set_noc1(self, rows):
        """Stream through NOC1 on these logical rows (NOC0 elsewhere); A's assignment follows the bandwidth."""
        self.noc1_rows = tuple(sorted(rows))
        self.lists = assign(units(), len(self.grid), self.grid.speeds(self.noc1_rows) if self.speed else None)

    def max_const_tiles(self) -> int:
        """The most A tiles with constants (conv: v, q, k; alpha) on one core: region AC's size, in tiles."""
        return max(sum(c for _, c, t in lst if t in (T_V, T_Q, T_K, T_A)) for lst in self.lists)

    def rm(self, t):
        t = t.float().contiguous()
        return ttnn.from_torch(t.reshape(1, 1, -1, t.shape[-1]) if t.dim() > 1 else t.reshape(1, 1, 1, -1),
                               dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)

    def x_blocks(self, x):
        """X [mp, 4 x 2560] (M rows) -> the X tensors' contents (a stream a page, M rows' layout), [1, 1, 4, m_max D]."""
        mp, mm = x.shape[0], self.m_max
        st = torch.zeros(HC, mm * D)
        for s in range(HC):
            st[s, :mp * D] = blocks(x[:, s * D:(s + 1) * D].float(), mp)
        return st.reshape(1, 1, HC, -1)

    def x_tensors(self, x):
        """X [mp, 4 x 2560] (M rows) -> the program's X tensors, fp32 and bf16, a stream a page (M rows' layout)."""
        st = self.x_blocks(x)
        return self.rm(st), ttnn.from_torch(st.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                            device=self.dev)

    def x_rows(self, t):
        """An X tensor of the M-row layout (x32_t, a check) -> [mp, 4 x 2560] (float64)."""
        st = ttnn.to_torch(t).double().reshape(HC, -1)
        return torch.cat([unblocks(st[s], D // 32, self.mp) for s in range(HC)], 1)

    def set_step(self, M, n_keep=1, parity=0):
        """The M-row program's step: M rows this pass, n_keep rows kept of the previous pass, the states' parity."""
        self.M, self.mp, self.n_keep, self.parity = M, mp_of(M), n_keep, parity
        self.step_t = ttnn.from_torch(step_words(M, n_keep, parity), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=self.dev)

    def set_inputs_m(self, rng, M=None, n_keep=1, parity=0, states=True):
        """M rows (m_max > 1): fresh X and routed rows (rows M..mp-1 copies of row M - 1), NaN-filled checks; with
        states, fresh history (3 rows at [n_keep - 1, n_keep + 2) of the parity's buffer), a fresh state S0 and pending
        rows of the parity (S_in: float64 of their commit), NaN elsewhere; without, the states the last pass left."""
        dev, mm = self.dev, self.m_max
        self.set_step(M or mm, n_keep, parity)
        M, mp = self.M, self.mp
        dup = lambda t: torch.cat([t, t[-1:].expand(mp - M, -1)]) if mp > M else t        # noqa: E731
        x = dup(torch.from_numpy(rng.standard_normal((M, HC * D)).astype(np.float32)))
        self.x = x.double()
        self.x32_t, self.x16_t = self.x_tensors(x)
        if states:
            self.s0 = torch.from_numpy(rng.standard_normal((3, CONV)).astype(np.float32)) * 0.5
            cs = torch.full((2 * (2 + mm), CONV), float("nan"))
            r0 = parity * (2 + mm) + n_keep - 1
            cs[r0:r0 + 3] = self.s0
            self.cstate = self.rm(cs)
            # the state a previous pass started from and that pass's 4 rows (k unit vectors, U, decays) as its pending
            # page, of which this pass commits n_keep; NaN in the other rows and the other parity
            self.S0 = torch.from_numpy(rng.standard_normal((NV, HK, HK)).astype(np.float32)) * 0.05
            K = torch.from_numpy(rng.standard_normal((NV, 4, HK)).astype(np.float32))
            K = K / K.norm(dim=-1, keepdim=True)
            U = torch.from_numpy(rng.standard_normal((NV, 4, HK)).astype(np.float32)) * 0.3
            dec = torch.from_numpy(rng.uniform(0.2, 1.0, (NV, 4)).astype(np.float32))
            self.S_in = commit(self.S0.double(), K.double(), U.double(), dec.double(), n_keep)
            for t in (K, U, dec):
                t[:, n_keep:] = float("nan")
            St = torch.full((2, NV, HK, HK), float("nan"))
            St[parity] = self.S0
            pend = torch.full((2, NV, 2, 1024), float("nan"))
            pend[parity] = pend_pages(K, U, dec)
            pend[parity, :, 1, n_keep:] = float("nan")
            tiles = torch.cat([state_tiles(St), raw_tiles(pend)]).reshape(1, 1, -1, 32)
            self.S = ttnn.from_torch(tiles, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        self.cstate_out, self.S_out = self.cstate, self.S    # the other parity of the same tensors
        self.routed = dup(torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32)) * 0.1)
        self.routed_t = self.rm(blocks(self.routed, mp))
        n = lambda size: nan_tensor(dev, (1, 1, 1, mm * size))                       # noqa: E731
        self.chk = dict(p1a=n(HC * LO), hna=n(D), wa=n(16), a=n(NT * 32), b=n(NV * HK),
                        x1=nan_tensor(dev, (1, 1, HC, mm * D)), p1f=n(HC * LO), hnf=n(D), wf=n(16),
                        l=n(N_ROUTER * 32), h=n(FF), x2=nan_tensor(dev, (1, 1, HC, mm * D)))
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def states_m(self, parity):
        """The M-row states of a parity: the conv history buffer [2 + m_max, CONV], the committed state [NV, HK, HK],
        the pending rows k, U [NV, 4, HK] and decays [NV, 4] (pend_pages)."""
        mm = self.m_max
        cs = ttnn.to_torch(self.cstate).double().reshape(2, 2 + mm, CONV)[parity]
        lg = ttnn.to_torch(self.S).double().reshape(-1, 32)
        n = 2 * NV * 16 * 32
        st = lg[:n].reshape(2, NV, 4, 4, 32, 32).permute(0, 1, 2, 4, 3, 5).reshape(2, NV, HK, HK)[parity]
        raw = tile_raw(lg[n:]).reshape(2, NV, 2, 1024)[parity]
        unface = lambda x: x.reshape(NV, 4, 2, 4, 16).permute(0, 3, 1, 2, 4).reshape(NV, 4, HK)  # noqa: E731
        return cs, st, unface(raw[:, 0, :512]), unface(raw[:, 0, 512:]), raw[:, 1, :4]

    def set_inputs(self, rng):
        if self.m_max > 1:
            return self.set_inputs_m(rng)
        dev = self.dev
        x = torch.from_numpy(rng.standard_normal(HC * D).astype(np.float32))
        self.x = x.double()
        self.x32_t = self.rm(x)
        self.x16_t = ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.s0 = torch.from_numpy(rng.standard_normal((3, CONV)).astype(np.float32)) * 0.5
        self.cstate = self.rm(self.s0)
        self.S0 = torch.from_numpy(rng.standard_normal((NV, HK, HK)).astype(np.float32)) * 0.05
        tiles = self.S0.reshape(NV, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(1, 1, -1, 32)
        self.S = ttnn.from_torch(tiles, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        self.routed = torch.from_numpy(rng.standard_normal(D).astype(np.float32)) * 0.1
        self.routed_t = self.rm(self.routed)
        n = lambda *shape: nan_tensor(dev, shape)                                  # noqa: E731
        self.chk = dict(p1a=n(1, 1, 1, HC * LO), hna=n(1, 1, 1, D), wa=n(1, 1, 1, 16), a=n(1, 1, 1, NT * 32),
                        b=n(1, 1, 1, NV * HK), x1=n(1, 1, 1, HC * D), p1f=n(1, 1, 1, HC * LO), hnf=n(1, 1, 1, D),
                        wf=n(1, 1, 1, 16), l=n(1, 1, 1, N_ROUTER * 32), h=n(1, 1, 1, FF), x2=n(1, 1, 1, HC * D))
        self.cstate_out = n(1, 1, 3, CONV)
        self.S_out = nan_tensor(dev, (1, 1, NV * 16 * 32, 32), ttnn.TILE_LAYOUT)
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def program(self, layers=1):
        g, o = self.grid, self.off
        offs = [o[k] for k in ("x32", "x16", "p1", "rs", "hna", "loa", "wa", "hnf", "lof", "wf", "a", "b", "l", "sh",
                               "rt", "ac", "tab", "prof", "rq")]
        A = lambda t: t.buffer_address()                                           # noqa: E731
        part = A(self.part) if self.card_on else 0
        croutes = A(self.card_routes) if self.card_routes is not None else 0
        C = len(g.cores)
        ha, hf = self.hc["attn"], self.hc["ffn"]
        crt, rargs, wargs = ttnn.RuntimeArgs(), {}, {}
        gp = (lambda idx: []) if self.gprof is None else (lambda idx: [A(self.gprof), idx])  # noqa: E731
        mr = self.m_max > 1
        rows = [("M_MAX", str(self.m_max))] if mr else []     # every kernel of the program takes it
        step = [A(self.step_t)] if mr else []
        defines = self.defines + ([] if self.gprof is None else profile_defines(48, self.gprof_n)) + (
            [("CARD_EXPERTS", "1")] if self.card_on else []) + (
            [("NO_CHECKS", "1")] if self.inplace and not self.checks_inplace else []) + rows  # (the token loop: none)
        order = g.order(self.noc1_rows)  # the phases on fewer cores than the grid take the NOC0 rows; A uses all
        pack = lambda k: g.noc[order[k]].x | g.noc[order[k]].y << 16                # noqa: E731
        b_cores = [pack(h) for h in range(NV)]                      # A's outputs go to B's cores,
        h1_cores = [pack(i) for i in range((LO // 32) * HC)]        # C's and R2's X to H1's (flayer_writer.cpp)
        for idx, (cx, cy) in enumerate(g.cores):
            lst = self.lists[idx]
            flat = [v for u in lst for v in u]
            k = order.index(idx)
            h_item = k if k < (LO // 32) * HC else IDLE
            h_tile = k if k <= D // 32 else IDLE
            head = k if k < NV else IDLE
            c_tile = k if k < D // 32 else IDLE
            r_item = k if k < N_ROUTER + FF // 32 else IDLE
            s_tile = k if k < D // 32 else IDLE
            work = [head, c_tile, r_item, s_tile]
            rargs[(cx, cy)] = [A(self.x32_t), A(self.x16_t)] + g.args(idx) + offs + [
                A(ha["wd"]), A(ha["wu"]), A(ha["norm_t"]), A(hf["wd"]), A(hf["wu"]), A(hf["norm_t"]), h_item, h_tile,
                A(self.wa), A(self.cstate), A(self.taps_t), A(self.dtna_t), len(lst)] + flat + [
                head, A(self.S), A(self.nw_t), A(self.wo), c_tile, A(self.wr), A(self.wgu), r_item,
                A(self.wsd), s_tile, A(self.routed_t), A(self.prof), self.table_addr, part, croutes, idx, C] + step + (
                    h1_cores if mr else []) + gp(idx)
            crt[cx][cy] = [h_item, h_tile, len(lst)] + flat + work + [idx, C]
            chk = [A(self.chk[k]) for k in ("p1a", "hna", "wa", "a", "b", "x1", "p1f", "hnf", "wf", "l", "h", "x2")]
            if self.inplace:
                chk[-1] = A(self.x32_t)
            wargs[(cx, cy)] = g.args(idx) + offs + chk + [
                A(self.cstate_out), A(self.S_out), h_item, h_tile, len(lst)] + flat + work + [
                self.table_addr, A(self.x16_t) if self.inplace else 0, part, idx, C] + b_cores + h1_cores + step + (
                    gp(idx))
        acc = list(ttnn.TensorAccessorArgs(self.wa).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("G4_FIDELITY", "HiFi4")), fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        codes = (58 if mr else 63,) if self.card_on else ()  # the card experts' codes (M rows: cb 58)
        if self.card_on and mr:
            check_core_map(g)
        for cb in EXACT + codes + ((22,) if mr else ()):  # M rows: 22 the conv's window
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        # the compute takes only the card's define (the others are the data movement's: the host link, the profile)
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "flayer_compute.cpp"), core_ranges=g.ranges,
                                         compile_time_args=[BATCH, layers], runtime_args=crt, config=cfg,
                                         defines=([("CARD_EXPERTS", "1")] if self.card_on else []) + rows + (
                                             [("TT_MOEOFFLOAD_ZONES", "1")] if os.environ.get("TT_MOEOFFLOAD_ZONES") else []))]
        kernels += g.dm_kernels(os.path.join(KDIR, "flayer_reader.cpp"), os.path.join(KDIR, "flayer_writer.cpp"),
                                [BATCH, DEPTH, layers] + acc, [layers] + acc, rargs, wargs, defines, self.noc1_rows)
        f32, bf16, u8 = ttnn.float32, ttnn.bfloat16, ttnn.uint8
        T4, T2 = 4096, 2048
        spec = [(0, ttnn.bfloat8_b, 1088, BATCH * DEPTH), (1, bf16, T2, 80), (2, bf16, T2, 1),
                (3, f32, T4, 3), (4, f32, T4, 3), (5, f32, T4, 1), (6, f32, T4, 1), (7, f32, T4, 2), (8, f32, T4, 1),
                (9, f32, T4, 4), (10, bf16, T2, 10), (11, f32, T4, 4), (12, f32, T4, 4), (13, f32, T4, 4),
                (14, bf16, T2, 1), (15, f32, T4, 1),
                (16, f32, T4, 14), (17, f32, T4, 2), (18, f32, T4, 4), (19, f32, T4, 4), (20, f32, T4, 1),
                (21, f32, T4, 8), (22, f32, T4, 12),
                (23, f32, T4, 16), (24, f32, T4, 16), (25, f32, T4, 4), (26, f32, T4, 4), (27, f32, T4, 4),
                (28, f32, T4, 10), (29, f32, T4, 4), (30, f32, T4, 4), (31, f32, T4, 4), (32, f32, T4, 4),
                (33, f32, T4, 4), (34, f32, T4, 16), (35, f32, T4, 4), (36, f32, T4, 4), (37, f32, T4, 1),
                (38, bf16, T2, 4), (39, f32, T4, 4),
                (40, bf16, T2, 2 * BATCH), (41, f32, T4, 1), (42, f32, T4, 4), (43, f32, T4, 4), (44, f32, T4, 4),
                (45, bf16, T2, 4),
                (46, bf16, T2, 2 * BATCH), (47, f32, T4, 1), (48, u8, 16 * self.gprof_n, 1), (49, bf16, T2, 1),
                (50, f32, T4, 1),
                (51, f32, T4, 1), (52, f32, T4, 1), (53, f32, T4, 4), (54, f32, T4, 4), (55, f32, T4, 1),
                (56, f32, T4, 4), (57, bf16, T2, 4),
                (58, u8, 8192, 1), (60, u8, T4, 1),
                (61, u8, self.off["ac"] + 7 * 128 * self.max_const_tiles(), 1), (62, u8, 1024, 1)]
        # the card's experts (fused/dexp_*.h): F (bfp8_b, a 24-k-tile group) and the codes (uint32); unused otherwise
        spec += [(59, ttnn.bfloat8_b, 1088, 24), (63, ttnn.uint32, 4096, 4)] if self.card_on else [
            (59, u8, 64, 1), (63, u8, 64, 1)]
        if mr:  # M rows: the ones tile + Sel + the step's words, the stage; B (flayer_compute.cpp phase_b_m): no cb 24,
            # cb 25 the norm weight + the commit's coefficients, cb 28 + the chunk's, cb 59 the kept rows, cb 63 U
            size = {2: 3, 3: 6, 25: 5, 28: 12, 62: 3}  # (3: H1's raw X chunks double-buffered)
            # with the card's experts (fused/dexp_cmp.h compute_m) their codes in cb 58 (the M = 1 exchange's page)
            if self.card_on:
                spec = [x if x[0] != 58 else (58, ttnn.uint32, T4, 2) for x in spec]
            spec = [(i, t, pg if i != 62 else 3072, size.get(i, k) if i != 62 else 1) for i, t, pg, k in spec
                    if i not in (24, 59, 63)] + [(24, u8, 64, 1), (59, f32, T4, 8), (63, f32, T4, 4)]
        self.l1_bytes = sum(size * n for _, _, size, n in spec)
        cbs = [g.cb(*x) for x in spec]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=cbs)
        ins = [self.x32_t, self.x16_t, ha["wd"], ha["wu"], ha["norm_t"], hf["wd"], hf["wu"], hf["norm_t"], self.wa,
               self.cstate, self.taps_t, self.dtna_t, self.S, self.nw_t, self.wo, self.wr, self.wgu, self.wsd,
               self.routed_t] + list(self.chk.values()) + [self.cstate_out, self.S_out, self.prof] + (
                   [] if self.gprof is None else [self.gprof]) + ([self.step_t] if mr else [])
        return lambda: ttnn.generic_op(ins, prog)

    def get(self, k):
        return ttnn.to_torch(self.chk[k]).double().flatten()

    def hc_ref(self, site, x32, x16, tag, got=None):
        """H1 from x (fp32, bf16), H2 from the card's partials (got: the card's p1, hn, w; default its checks)."""
        h = self.hc[site]
        X, X16 = x32.reshape(HC, D), x16.reshape(HC, D)
        r = torch.rsqrt(X.pow(2).mean(-1) + EPS)
        p_ref = torch.stack([r[s] * (X16[s] @ h["wdq"][s * D:(s + 1) * D]) for s in range(HC)])
        p, hn_c, w_c = got or (self.get("p1" + tag), self.get("hn" + tag), self.get("w" + tag))
        p = p.reshape(HC, LO)
        res = {f"H1{tag}": rel(p[:, :LR + HC], p_ref[:, :LR + HC])}
        lo = p.sum(0)
        gate = (bf(torch.nn.functional.silu(lo[:LR] / HC)) @ h["wuq"]).reshape(HC, D)
        hn = (torch.sigmoid(gate) * X * h["norm"].double().reshape(HC, D) * r[:, None]).mean(0)
        res[f"H2{tag}_hn"] = rel(hn_c, hn)
        res[f"H2{tag}_w"] = rel(w_c[:HC], 2 * torch.sigmoid(lo[LR:LR + HC] / HC))
        return res

    def get_m(self, k):
        """An M-row check as [mp, ...] (float64)."""
        mp, t = self.mp, self.chk[k]
        if k in ("x1", "x2"):
            return self.x_rows(t)
        flat = ttnn.to_torch(t).double().flatten()
        if k in ("wa", "wf"):
            return flat[:mp * 16].reshape(mp, 16)
        n = {"p1a": HC * LO // 32, "p1f": HC * LO // 32, "hna": D // 32, "hnf": D // 32, "a": NT, "b": NV * HK // 32,
             "l": N_ROUTER, "h": FF // 32}[k]
        return unblocks(flat, n, mp)

    def errors_m(self, h3=None, S_in=None):
        """M rows: every phase of every row m < M against float64 from the card's own previous outputs, as errors()
        does for one token; the conv window from the history h3 (default the inputs' s0) and the earlier rows' y, the
        delta rule (the chunked form with the commit first, flayer_compute.cpp phase_b_m) row by row: B_commit the
        card's committed state (the other parity's) against S_in (default the inputs', float64 of their commit), row
        m's output from the state the card's commit and pending rows give after rows 0..m-1, B_state that state after
        row m against the float64 chain from S_in. Returns (the worst row per check, with every row's under "rows";
        aux: the conv inputs e (h3 then the float64 y rows) and the states chained in float64 from S_in after each row,
        a next pass's reference)."""
        M = self.M
        h3 = (self.s0 if h3 is None else h3).double()
        S_prev = (self.S_in if S_in is None else S_in).double()
        g = {k: self.get_m(k) for k in self.chk}
        hist, S_c, Kc, Uc, dc = self.states_m(1 - self.parity)
        card = lambda n: commit(S_c, Kc, Uc, dc, n) if n else S_c                     # noqa: E731
        taps, nw = self.taps.double(), self.nw.double()
        dt, na = self.dtna.double()
        ys = [bf(g["hna"][m]) @ self.waq for m in range(M)]
        e = torch.cat([h3, torch.stack([y[:CONV] for y in ys])])        # the conv's inputs, oldest first
        rows, chain = [], []
        for m in range(M):
            res = self.hc_ref("attn", self.x[m], bf(self.x[m]), "a", (g["p1a"][m], g["hna"][m], g["wa"][m]))
            y = ys[m]
            x = conv_step(e[m:m + 4], taps)
            q, k = x[:NK * HK].reshape(NK, HK), x[NK * HK:2 * NK * HK].reshape(NK, HK)
            q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + EPS) * HK ** -0.5
            k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
            decay = torch.exp(torch.nn.functional.softplus(y[NA_T * 32:NA_T * 32 + NA] + dt) * na)
            Aref = torch.cat([q.flatten(), k.flatten(), x[2 * NK * HK:], torch.sigmoid(y[CONV:CONV + ZD]), decay,
                              torch.sigmoid(y[NA_T * 32 + NA:])])
            a = g["a"][m]
            res["A"] = rel(a[:NA_T * 32 + NA + NV], Aref[:NA_T * 32 + NA + NV])
            res["B"] = rel(g["b"][m], delta_step(card(m), a, nw)[0])
            chain.append(delta_step(chain[-1] if chain else S_prev, a, nw)[1])
            res["B_state"] = rel(card(m + 1), chain[-1])
            x1 = (self.x[m].reshape(HC, D) + g["wa"][m][:HC, None] * (bf(g["b"][m]) @ self.woq)[None, :]).flatten()
            res["C_combine"] = rel(g["x1"][m], x1)
            res.update(self.hc_ref("ffn", g["x1"][m], bf(g["x1"][m]), "f", (g["p1f"][m], g["hnf"][m], g["wf"][m])))
            hnf = bf(g["hnf"][m])
            logits = hnf @ self.wrq
            res["R1_logits"] = rel(g["l"][m][:513], logits[:513])
            gu = (hnf @ self.wguq).reshape(FF // 32, 2, 32)
            res["R1_h"] = rel(g["h"][m], (torch.nn.functional.silu(gu[:, 0]) * gu[:, 1]).flatten())
            top = set(torch.topk(g["l"][m][:512], 10).indices.tolist())
            res["R1_top10_agree"] = len(top & set(torch.topk(logits[:512], 10).indices.tolist()))
            sh = bf(g["h"][m]) @ self.wsdq * torch.sigmoid(g["l"][m][512]) + self.routed[m].double()
            x2 = (g["x1"][m].reshape(HC, D) + g["wf"][m][:HC, None] * sh[None, :]).flatten()
            res["R2_combine"] = rel(g["x2"][m], x2)
            rows.append(res)
        res = {k: (min if k == "R1_top10_agree" else max)(r[k] for r in rows) for k in rows[0]}
        # the extended conv history: rows 0..M+1 of the other parity's buffer = h3[1], h3[2], y_0..y_{M-1}
        res["conv_state"] = rel(hist[:M + 2], e[1:M + 3])
        res["B_commit"] = rel(S_c, S_prev)
        res["rows"] = rows
        return res, dict(e=e, chain=chain)

    def errors(self) -> dict:
        res = self.hc_ref("attn", self.x, bf(self.x), "a")
        # A from the card's hn (bf16 as A reads it)
        y = bf(self.get("hna")) @ self.waq
        s, taps = self.s0.double(), self.taps.double()
        x = torch.nn.functional.silu((torch.cat([s, y[None, :CONV]]) * taps).sum(0))
        q, k = x[:NK * HK].reshape(NK, HK), x[NK * HK:2 * NK * HK].reshape(NK, HK)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + EPS) * HK ** -0.5
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + EPS)
        dt, na = self.dtna.double()
        decay = torch.exp(torch.nn.functional.softplus(y[NA_T * 32:NA_T * 32 + NA] + dt) * na)
        Aref = torch.cat([q.flatten(), k.flatten(), x[2 * NK * HK:], torch.sigmoid(y[CONV:CONV + ZD]), decay,
                          torch.sigmoid(y[NA_T * 32 + NA:])])
        a = self.get("a")
        res["A"] = rel(a[:NA_T * 32 + NA + NV], Aref[:NA_T * 32 + NA + NV])
        res["conv_state"] = rel(ttnn.to_torch(self.cstate_out).double().reshape(3, CONV),
                                torch.cat([s[1:], y[None, :CONV]]))
        # B from the card's A
        q, k = a[:2048].reshape(NK, HK).repeat(3, 1), a[2048:4096].reshape(NK, HK).repeat(3, 1)
        v, z = a[4096:CONV].reshape(NV, HK), a[CONV:CONV + ZD].reshape(NV, HK)
        d, b = a[NA_T * 32:NA_T * 32 + NV], a[NA_T * 32 + NA:NA_T * 32 + NA + NV]
        S = self.S0.double() * d[:, None, None]
        S = S + k[:, :, None] * ((v - torch.einsum("hk,hkv->hv", k, S)) * b[:, None])[:, None, :]
        o = torch.einsum("hk,hkv->hv", q, S)
        res["B"] = rel(self.get("b"), (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) *
                                       self.nw.double() * z).flatten())
        st = ttnn.to_torch(self.S_out).double().reshape(NV, 4, 4, 32, 32).permute(0, 1, 3, 2, 4)
        res["B_state"] = rel(st.reshape(NV, HK, HK), S)
        # C and the combine from the card's B and w_a
        out = bf(self.get("b")) @ self.woq
        x1 = (self.x.reshape(HC, D) + self.get("wa")[:HC, None] * out[None, :]).flatten()
        res["C_combine"] = rel(self.get("x1"), x1)
        x1c = self.get("x1")
        res.update(self.hc_ref("ffn", x1c, bf(x1c), "f"))
        # R1 from the card's hn_f
        hnf = bf(self.get("hnf"))
        logits = hnf @ self.wrq
        res["R1_logits"] = rel(self.get("l")[:513], logits[:513])
        gu = (hnf @ self.wguq).reshape(FF // 32, 2, 32)
        res["R1_h"] = rel(self.get("h"), (torch.nn.functional.silu(gu[:, 0]) * gu[:, 1]).flatten())
        top_card = set(torch.topk(self.get("l")[:512], 10).indices.tolist())
        res["R1_top10_agree"] = len(top_card & set(torch.topk(logits[:512], 10).indices.tolist()))
        # R2 from the card's h, gate logit, X1, w_f
        sh = bf(self.get("h")) @ self.wsdq * torch.sigmoid(self.get("l")[512]) + self.routed.double()
        x2 = (x1c.reshape(HC, D) + self.get("wf")[:HC, None] * sh[None, :]).flatten()
        res["R2_combine"] = rel(self.get("x2"), x2)
        return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    g = Gguf(MODEL)
    # the program's kernels (~116 KB with their arguments) exceed the default kernel config buffer (70.7 KB); it
    # grows by what the worker L1 gives up (default worker L1 = 1536 KB - 39 KB reserved - 69 KB config)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": LAYER}
    try:
        L = Layer(dev, g, rng)
        prog = L.program()
        print(f"weights in {time.time() - t0:.0f} s; circular buffers {L.l1_bytes / 1024:.0f} KB per core",
              flush=True)
        prog()
        ttnn.synchronize_device(dev)
        res["max_rel_err"] = L.errors()
        print("max rel err: " + ", ".join(f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}"
                                          for k, v in res["max_rel_err"].items()), flush=True)
        res["traced_us"] = traced_us(dev, L.program())
        print(f"layer step traced {res['traced_us']:.1f} us (G2 stock: 1112.8 us)", flush=True)
        # the phases, from the coordinator's wall clock (cycles at the AI clock) on the last replay
        st = ttnn.to_torch(L.prof).flatten().long().tolist()
        names = ["X", "H1a", "HLa", "H2a", "A", "B", "C+combine", "H1f", "HLf", "H2f", "R1", "R2+combine"]
        cyc = [(st[i + 1] - st[i]) & 0xFFFFFFFF for i in range(len(names))]
        res["phase_cycles"] = dict(zip(names, cyc))
        tot = sum(cyc)
        # the steady state: N passes through the layer in one launch (no launch between them)
        for n in (8,):
            res[f"traced_us_{n}_layers"] = traced_us(dev, L.program(n))
        res["per_layer_us_persistent"] = (res["traced_us_8_layers"] - res["traced_us"]) / 7
        print(f"8 layers in one launch {res['traced_us_8_layers']:.1f} us: {res['per_layer_us_persistent']:.1f} us "
              f"per layer without a launch", flush=True)
        print(f"phases ({tot} cycles in all; share): " + ", ".join(f"{n} {100 * c / tot:.1f}%"
                                                                      for n, c in zip(names, cyc)), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g3_layer.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
