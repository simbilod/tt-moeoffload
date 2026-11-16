"""G3 step 5: Flash-Next's QSA attention layer, program 1 of 2 (tt/kernels/fattn1_*.cpp), real blk.3 weights: one decode
token from the residual streams X to the indexer's pool scores, on all 110 cores with a barrier after each phase.

  H1a, H2a   the hyper-connection mix (fused/hc_dm.h)
  P          the projection hn -> q | gate (24 heads), k, v (2 KV heads), the indexer's q (4 heads) and raw k: 436
             tiles, <= 4 per core, sigmoid on the gate tiles
  N          the indexer scores of all pools (64 cores x 32 pools: q heads RMSNorm + RoPE, sum_h relu(q_h k_p)), the
             q heads (RMSNorm, RoPE, / 16), k (RMSNorm, RoPE) and v into the KV cache at the token's cell, the raw
             indexer key into its cache, and the pooled key of the block the token closes

The token sits at the end of a 16K context (pos 16383, so it closes block 4095); the caches hold random contents. The
host then takes the top 512 of the 4,095 complete pools' scores; program 2 (g3_attn2.py) attends. Checked against
float64 per phase from the card's own previous outputs, on NaN-filled outputs with fresh inputs, then timed (traced,
one launch).

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/flashnext/g3_attn1.py

Measured 2026-10-23: H1 6.6e-4, hn 6.7e-4, the projection 4.2e-4, q 1.2e-4, the pooled key 1.5e-4, the scores 1.2e-3
of float64, the K cache row within a bf16 rounding (3.1e-3), v and the raw key exact; the top 512 pools the same as
the reference's. 223.6 us traced (phases: X 6, H1 19, H2 38, projection 122, scores + heads + caches 25 us). With
rows 0, 1, 9 reading through NOC1 (fused/grid.py) and the projection split by bandwidth: 193.6 us (210.6 on NOC0
alone, after the HL phase); 187.8 with the mix's weights prefetched before its barriers.

For the token loop (2026-10-24) the program is built once: the position comes from the step tensor in card memory
(uint32 [16], word 0 pos; set_pos writes it in queue), the RoPE rows from a table of every position ([CTX x 4, 32]
fp32, row 4 pos + j: cos t, sin t, cos t0, sin t0 with t0 = pos - 3), the pool role is always assigned and writes
only when pos % 4 == 3, and the sigmoid gates leave as G ([24 x 256] fp32, a head a row, as Q) for program 2. One
program checked at pos 0, 1, 2, 3, 5, 7, 100, 1003 and 8191 in turn: every phase as before (H1 6.9e-4, the
projection 3.4e-4, q 1.7e-4, G exact, the pooled key 1.1-1.6e-4 at block closes and untouched otherwise, the scores
9.4-9.7e-4, the top min(512, pools) the same); 186.4 us traced. The scores cover the pools before the token's block:
when the token closes its block (pos % 4 == 3), that pool's score is not among them (its pooled key is written in
the same phase).
Then (the same day) the pool role also scores the block the token closes from its new key (the score tensor's word
2048; the link's page carries it in that pool's place, the host's no-link path patches it): llama.cpp ranks the
just-closed pool with the others. Checked at pos 3 ... 8191 (2051, 4095, 6003 added: beyond 2,048 cells the top 512
prune): that score within 6e-5-8.7e-4 of float64 on the new key, everything else as before; 195.5 us traced (the
pool core's N work is now the key, then the q heads and the dot: +9 us; its cache reads and rope rows moved before
the barrier did not change that).
16,384 cells (4,096 pools). The 64 score cores take blocks k and k + 64, the second only once a
pool of it lies before the token's block (the reader tells the compute through cb 22), so below 8K cells program 1
does what it did; past 2,048 pools the scores leave as a second 8 KB link page (pos // 4 * 4 // 8192 + 1 pages, on
both sides). Checked at pos 0 ... 7999, 8191, 8195, 8199, 8327, 12003, 12290, 16383: as at 8K (the scores 5.6-6.0e-4,
the pooled key 2.3-7.4e-4, the own block's score 2.5e-4-1.2e-3); the link's pages equal the score tensor and the
selection g4_host's rule at 8191, 8192, 8195, 12003, 16383. 197.4 us traced at 16383 (N 45,381 cycles, as at 8K). At
7,900-8,191 every output bit-identical to the 8K build's on the same inputs.

Attn1(m_max=4) builds the same program for a verify pass of up to 4 rows (fattn1_reader.cpp M_MAX: rows at
pos .. pos + M - 1, M from the step tensor, set_step / new_pass): the mix and X as the DeltaNet program's M rows, PA in
mp-row blocks, N per row; the scores [m_max x (NPOOL + 16)], Q bf16 and G a head-row a page; the block a row of the
pass closes scored from its new key for every row (row_scores puts it in place for the rows that see it complete).
Checked and timed by g6_attn_m.py (errors_m per row), opened with WORKER_L1_M. m_max = 1 (the default, the token
loop) compiles the one-row source unchanged.
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
from fused.grid import NOC1_ROWS, Grid, profile_plumbing, split_weighted, traced_us  # noqa: E402
from g2_layers import MODEL, Gguf, rope_matrix  # noqa: E402
from g3_gdn import EPS, nan_tensor  # noqa: E402
from g3_hc import HC, LO, LR, model_tag, stream  # noqa: E402
from g3_layer import WORKER_L1, bf, blocks, mp_of, rel, unblocks  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
LAYER = 3
D, NH, NKV, HD, IH, IHD = 2560, 24, 2, 256, 4, 128
# the cells of the attention caches: the build's context (TT_CTX, the profile's ttcard.ctx), 8K steps up to 256K:
# the 64 score cores take up to CTX/8192 blocks of 32 pools each (fattn1_writer.cpp's stage, cb 62: 128 B a block)
CTX, KPOOL = int(os.environ.get("TT_CTX", 16384)), 4
assert CTX % 8192 == 0 and 8192 <= CTX <= 262144, f"TT_CTX={CTX}: a multiple of 8192 up to 262144"
NPOOL = CTX // KPOOL
NSCORE = 64                                  # score cores: k takes pool blocks k, k + 64, ... once the token is past them
NT = 436                                     # q|gate 384, k 16, v 16, the indexer's q 16 and k 4
T_K, T_V, T_IQ, T_IK = 384, 400, 416, 432
BATCH, DEPTH = 8, 4
IDLE = 0xFFFFFFFF
R_SCORE, R_QHEAD, R_KVHEAD, R_POOL = range(4)
EXACT = (3, 6, 9, 11, 12, 13, 23, 24, 25, 27, 32, 33)
_ROPE = {}                                    # id(device) -> (device, its RoPE table): Attn1's
# the raw indexer keys' ring (256K plan step 2): row p % RAWK holds position p's; only the open block's are read. Two
# blocks, as a 4-row verify pass writes into the next block before the one it closes is pooled, and a rejected row
# must not overwrite a key the next pass reads
RAWK = 8


def kv_caches(dev, r, zero: bool):
    """The K and V caches [1, NKV, NPOOL, 4 HD] bf16 row-major, and their host copies: random (the checks' inputs), or
    zeros made on the card with no host copies (a model's build, which keeps them zero: 4 x 134M draws an attention
    layer took ~9 s at 256K)."""
    shape = (1, NKV, NPOOL, KPOOL * HD)
    if zero:
        return (None, None) + tuple(ttnn.zeros(list(shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                               device=dev) for _ in range(2))
    k0, v0 = r(NKV, NPOOL, KPOOL * HD).to(torch.bfloat16), r(NKV, NPOOL, KPOOL * HD).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t.reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,  # noqa: E731
                                   device=dev)
    return k0, v0, up(k0), up(v0)


class Ring:
    """The raw-key ring [RAWK, IHD] read by position: ring[p] is row p % RAWK, ring[a:b] the rows of a..b-1."""
    def __init__(self, t):
        self.t = t

    def __getitem__(self, i):
        if isinstance(i, slice):
            return torch.stack([self.t[p % RAWK] for p in range(i.start, i.stop)])
        return self.t[i % RAWK]
POSITIONS = tuple(p for p in (0, 1, 2, 3, 5, 7, 100, 1003, 2051, 4095, 6003, 7999, 8191, 8195, 8199, 8327, 12003, 12290,
                              16383, 16387, 24579, 32768, 40003, 49155, 57347, 65539, 98307, 131075, 196611,
                              229379) if p < CTX - 1) + (CTX - 1,)
                                                                     # checked in this order, with one program


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def x_tensors(dev, x, mm: int):
    """X [mp, 4 x 2560] (M rows) -> the M-row programs' X tensors, fp32 and bf16, a stream a page in fused/hc_dm.h's
    mp-row blocks (as g3_layer.Layer.x_tensors); dev None: host tensors (for copy_host_to_device_tensor)."""
    mp = x.shape[0]
    st = torch.zeros(HC, mm * D)
    for s in range(HC):
        st[s, :mp * D] = blocks(x[:, s * D:(s + 1) * D].float(), mp)
    kw = dict(layout=ttnn.ROW_MAJOR_LAYOUT) | ({} if dev is None else dict(device=dev))
    return (ttnn.from_torch(st.reshape(1, 1, HC, -1), dtype=ttnn.float32, **kw),
            ttnn.from_torch(st.to(torch.bfloat16).reshape(1, 1, HC, -1), dtype=ttnn.bfloat16, **kw))


def x_rows(t, mp: int):
    """An X tensor of the M-row layout -> [mp, 4 x 2560] (float64)."""
    st = ttnn.to_torch(t).double().reshape(HC, -1)
    return torch.cat([unblocks(st[s], D // 32, mp) for s in range(HC)], 1)


def step_words(pos: int, M: int = 1) -> torch.Tensor:
    """The step tensor's words: 0 pos, 1..4 M, n_keep (1), parity (0), Mp; attention reads pos, M, Mp."""
    w = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
    w[..., 0] = pos
    w[0, 0, 0, 1:5] = torch.tensor([M, 1, 0, mp_of(M)])
    return w


class Attn1:
    def __init__(self, dev, g: Gguf, rng, pos=CTX - 1, layer=LAYER, back=True, m_max=1, zero_caches=False):
        self.zero_caches = zero_caches       # zero K/V caches made on the card (kv_caches): a model's build
        """back=False: no float64 host copies of the weights (the token loop; errors() then cannot run).
        m_max > 1: the M-row program (fattn1_reader.cpp M_MAX) for up to m_max (2 or 4) rows a pass at positions
        pos .. pos + M - 1, M from the step tensor (set_step), the owner arena, X tensors a stream a page."""
        assert m_max in (1, 2, 4), "M_MAX: 1, 2 or 4 (M = 3 runs in the 4-row layout)"
        self.m_max = m_max
        self.dev, self.pos, self.layer = dev, pos, layer
        self.grid = Grid(dev)
        self.noc1_rows = NOC1_ROWS   # rows streaming through NOC1 (fused/grid.py)
        self.gprof, self.gprof_n = None, 32  # grid.h's per-core clocks (fused/grid.py profile_tensor; g4_phases.py)
        self.defines = []            # the host link's (EXCHANGE, the socket config addresses): the scores out,
        self.sel_out = None          # the host's selection back into this tensor (program 2's selection)
        p = f"blk.{layer}."
        tag = model_tag(g)
        key = lambda name: f"{tag}/{p}{name}"                                   # noqa: E731  (g3_hc.stream's cache)
        norm = g(f"{p}hc_attn_norm.weight")

        def wd():
            down, inject = g(f"{p}hc_attn_down.weight"), g(f"{p}hc_attn_inject.weight")
            w = torch.zeros(HC * D, LO)
            w[:, :LR + HC] = (torch.cat([down, inject]) * norm[None, :]).T
            return w
        self.wd, self.wdq = stream(dev, wd, back, key("hc_attn_wd"))
        self.wu, self.wuq = stream(dev, lambda: g(f"{p}hc_attn_up.weight").T, back, key("hc_attn_wu"))
        self.hnorm = norm
        self.hnorm_t = self.rm(norm)
        w = lambda: torch.cat([g(p + "attn_q.weight"), g(p + "attn_k.weight"), g(p + "attn_v.weight"),  # noqa: E731
                               g(p + "indexer.q_proj.weight"), g(p + "indexer.k_proj.weight")]).T  # [2560, 13952]
        self.wp, self.wpq = stream(dev, w, back, key("wp"))
        self.n = {k: g(p + f) for k, f in (("q", "attn_q_norm.weight"), ("k", "attn_k_norm.weight"),
                                          ("iq", "indexer.q_norm.weight"), ("ik", "indexer.k_norm.weight"))}
        self.n_t = {k: self.rm(v) for k, v in self.n.items()}
        # the RoPE table of every position: rows 4 t + (cos t, sin t, cos t0, sin t0), t0 = t - 3 (the block's first);
        # one per device, as every layer's is the same (512 B a cell: 12 copies were 1.6 GB at 256K, 256K plan step 0)
        if _ROPE.get(id(dev), (None,))[0] is not dev:
            t = torch.arange(CTX, dtype=torch.float64)[:, None]
            th = lambda t: t * 1e7 ** (-torch.arange(32, dtype=torch.float64) * 2 / 64)   # noqa: E731
            _ROPE[id(dev)] = (dev, self.rm(torch.stack([torch.cos(th(t)), torch.sin(th(t)), torch.cos(th(t - 3)),
                                                        torch.sin(th(t - 3))], 1).reshape(CTX * 4, 32)))
        self.rope_t = _ROPE[id(dev)][1]
        # the step tensor (uint32 [16]; word 0 the token's position): the per-token input of a program built once
        self.step = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        o, a = {}, 0
        mm = m_max
        if mm == 1:
            regions = (("x32", HC * D * 4), ("x16", HC * D * 2), ("p1", HC * LO * 4), ("rs", 64),
                       ("hna", D * 2), ("lo", 640), ("wa", 64), ("pa", NT * 128))
        else:   # the owner layout (fattn1_reader.cpp M rows): x32 / x16 an H1 core's stream, rs the slot of xown
            # (the H2 core's tile of the 4 streams), p1 then the r_s, lo then w, by M
            regions = (("x32", mm * D * 4), ("x16", mm * D * 2), ("p1", mm * (HC * LO * 4 + 256)),
                       ("rs", mm * HC * 128), ("hna", mm * D * 2), ("lo", mm * 704), ("wa", 0), ("pa", mm * NT * 128))
        for name, size in regions:
            o[name] = a
            a += size
        self.off, self.arena = o, a
        self.set_inputs(rng)
        if mm == 1:
            self.set_pos(pos)
        else:
            self.set_step(pos, mm)

    def set_pos(self, pos):
        """The token's position, into the step tensor (an in-queue write: the program is not rebuilt)."""
        self.pos = pos
        self.rope = {"main": rope_matrix(pos, HD).double(), "idx": rope_matrix(pos, IHD).double(),
                     "pool": rope_matrix(pos - KPOOL + 1, IHD).double()}
        st = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
        st[..., 0] = pos
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(st, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.step)

    def set_step(self, pos, M):
        """The M-row program's pass: M rows at positions pos .. pos + M - 1 (the step tensor, in queue)."""
        assert self.m_max > 1 and 1 <= M <= self.m_max and pos + M <= CTX
        self.pos, self.M, self.mp = pos, M, mp_of(M)
        self.mc = (3 - pos % KPOOL) % KPOOL          # the row that closes a block, when < M
        self.ropes = [{"main": rope_matrix(pos + m, HD).double(), "idx": rope_matrix(pos + m, IHD).double()}
                      for m in range(M)]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(pos, M), dtype=ttnn.uint32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT), self.step)

    def run(self, prog, serve=None):
        """One launch (serve(): the host's side of the link, between the launch and the sync), with the caches'
        state before it kept for the checks and the outputs reset to NaN."""
        self.rawk_before = Ring(ttnn.to_torch(self.rawk).double().reshape(RAWK, IHD))
        pooled = ttnn.to_torch(self.pooled).double().reshape(NPOOL // 32, 4, 32, 32).permute(0, 2, 1, 3)
        self.pk_before = pooled.reshape(NPOOL, IHD)
        for t in (self.scores, self.q, self.g_out) + (tuple(self.chk.values()) if self.m_max > 1 else ()):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full(tuple(t.shape), float("nan")),
                                                            dtype=t.dtype, layout=ttnn.ROW_MAJOR_LAYOUT), t)
        prog()
        if serve:
            serve()
        ttnn.synchronize_device(self.dev)

    def rm(self, t):
        t = t.float().contiguous()
        return ttnn.from_torch(t.reshape(1, 1, -1, t.shape[-1]) if t.dim() > 1 else t.reshape(1, 1, 1, -1),
                               dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)

    def set_inputs(self, rng):
        if self.m_max > 1:
            return self.set_inputs_m(rng)
        dev = self.dev
        r = lambda *s: torch.from_numpy(rng.standard_normal(s).astype(np.float32))           # noqa: E731
        self.x = r(HC * D).double()
        self.x32_t = self.rm(self.x)
        self.x16_t = ttnn.from_torch(self.x.float().to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.kc0, self.vc0, self.kc, self.vc = kv_caches(dev, r, self.zero_caches)
        self.rawk0 = r(RAWK, IHD)
        self.rawk = self.rm(self.rawk0)
        pk = r(NPOOL, IHD)
        self.pk0 = (pk * torch.rsqrt(pk.pow(2).mean(-1, keepdim=True))).double()
        tiles = self.pk0.float().reshape(NPOOL // 32, 32, 4, 32).permute(0, 2, 1, 3).reshape(1, 1, -1, 32)
        self.pooled = ttnn.from_torch(tiles.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        n = lambda *shape: nan_tensor(dev, shape)                                          # noqa: E731
        self.chk = dict(p1=n(1, 1, 1, HC * LO), hn=n(1, 1, 1, D), w=n(1, 1, 1, 16), pa=n(1, 1, 1, NT * 32))
        # the scores: the 4096 pools', then (word 4096) the score of the block the token closes, from its new key
        self.scores, self.q, self.g_out = n(1, 1, 1, NPOOL + 16), n(1, 1, NH, HD), n(1, 1, NH, HD)  # G: the gates
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def set_inputs_m(self, rng, M=None, caches=True):
        """M rows (m_max > 1): fresh X rows (rows M..mp-1 copies of row M - 1) and NaN-filled outputs; with caches,
        fresh caches too (random, as set_inputs), else the ones the last pass left. Outputs at M rows: the scores
        [m_max x (NPOOL + 16)] (row m's), Q (bf16) and G [m_max x 24, 256] (row m head h at 24 m + h)."""
        dev, mm = self.dev, self.m_max
        self.set_step(self.pos, M or mm)
        M, mp = self.M, self.mp
        r = lambda *s: torch.from_numpy(rng.standard_normal(s).astype(np.float32))           # noqa: E731
        x = r(M, HC * D)
        x = torch.cat([x, x[-1:].expand(mp - M, -1)]) if mp > M else x
        self.x = x.double()
        self.x32_t, self.x16_t = x_tensors(dev, x, mm)
        if caches:
            self.kc0, self.vc0, self.kc, self.vc = kv_caches(dev, r, self.zero_caches)
            self.rawk = self.rm(r(RAWK, IHD))
            pk = r(NPOOL, IHD)
            pk = (pk * torch.rsqrt(pk.pow(2).mean(-1, keepdim=True))).float()
            tiles = pk.reshape(NPOOL // 32, 32, 4, 32).permute(0, 2, 1, 3).reshape(1, 1, -1, 32)
            self.pooled = ttnn.from_torch(tiles.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        n = lambda *shape: nan_tensor(dev, shape)                                          # noqa: E731
        self.chk = dict(p1=n(1, 1, 1, mm * HC * LO), hn=n(1, 1, 1, mm * D), w=n(1, 1, 1, mm * 16),
                        pa=n(1, 1, 1, mm * NT * 32))
        self.scores, self.g_out = n(1, 1, mm, NPOOL + 16), n(1, 1, mm * NH, HD)
        self.q = ttnn.from_torch(torch.full((1, 1, mm * NH, HD), float("nan")), dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.prof = ttnn.from_torch(torch.zeros(1, 1, 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def new_pass(self, rng, pos, M):
        """The next pass of a verify loop: M rows at pos, fresh X, the caches kept (rows past an earlier pass's
        accepted ones are rewritten by position). The outputs stay the same tensors (the program is built once)."""
        self.set_step(pos, M)
        r = torch.from_numpy(rng.standard_normal((M, HC * D)).astype(np.float32))
        x = torch.cat([r, r[-1:].expand(self.mp - M, -1)]) if self.mp > M else r
        self.x = x.double()
        for h, d in zip(x_tensors(None, x, self.m_max), (self.x32_t, self.x16_t)):
            ttnn.copy_host_to_device_tensor(h, d)

    def roles(self, k):
        """Phase N's role of the core taking item k."""
        if k < NSCORE:
            return R_SCORE, k
        if k < NSCORE + NH:
            return R_QHEAD, k - NSCORE
        if k < NSCORE + NH + NKV:
            return R_KVHEAD, k - NSCORE - NH
        if k == NSCORE + NH + NKV:           # acts when the token closes its block (the kernel reads pos)
            return R_POOL, 0
        return IDLE, 0

    def program(self):
        g, o = self.grid, self.off
        offs = [o[k] for k in ("x32", "x16", "p1", "rs", "hna", "lo", "wa", "pa")]
        A = lambda t: t.buffer_address()                                                    # noqa: E731
        # the projection split by each core's share of card memory bandwidth, the other phases on the NOC0 rows first
        ps, order = split_weighted(NT, g.speeds(self.noc1_rows)), g.order(self.noc1_rows)
        gp, gdef, gbytes = profile_plumbing(self.gprof, self.gprof_n, 63, 64)  # in cb 63, after the 16 stamps
        mr = self.m_max > 1
        rows = [("M_MAX", str(self.m_max))] if mr else []     # every kernel of the program takes it
        pack = lambda k: g.noc[order[k]].x | g.noc[order[k]].y << 16                # noqa: E731
        h1_cores = [pack(i) for i in range((LO // 32) * HC)] if mr else []   # M rows: X's streams to H1's cores
        rargs, crt, wargs = {}, ttnn.RuntimeArgs(), {}
        for idx, (cx, cy) in enumerate(g.cores):
            k = order.index(idx)
            h_item = k if k < (LO // 32) * HC else IDLE
            h_tile = k if k <= D // 32 else IDLE
            role, index = self.roles(k)
            pf, pc = ps[idx]
            assert not mr or pc <= 6, "the M-row writer's stage holds 6 projection tiles' blocks"
            rargs[(cx, cy)] = [A(self.x32_t), A(self.x16_t)] + g.args(idx) + offs + [
                A(self.wd), A(self.wu), A(self.hnorm_t), h_item, h_tile, A(self.wp), pf, pc, role, index, A(self.step),
                A(self.pooled), A(self.rawk), A(self.rope_t)] + [A(self.n_t[k]) for k in ("q", "k", "iq", "ik")] + [
                A(self.prof), A(self.scores), A(self.sel_out) if self.sel_out is not None else 0] + h1_cores + gp(idx)
            crt[cx][cy] = [h_item, h_tile, pf, pc, role, index]
            wargs[(cx, cy)] = g.args(idx) + offs + [A(self.chk[c]) for c in ("p1", "hn", "w", "pa")] + [
                A(self.scores), A(self.q), A(self.kc), A(self.vc), A(self.rawk), A(self.pooled), A(self.g_out),
                h_item, h_tile, pf, pc, role, index, A(self.step), NPOOL] + gp(idx)
        acc = list(ttnn.TensorAccessorArgs(self.wp).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("G4_FIDELITY", "HiFi4")), fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in EXACT:
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fattn1_compute.cpp"), core_ranges=g.ranges,
                                         compile_time_args=[BATCH], runtime_args=crt, config=cfg,
                                         **(dict(defines=rows) if mr else {}))]
        kernels += g.dm_kernels(os.path.join(KDIR, "fattn1_reader.cpp"), os.path.join(KDIR, "fattn1_writer.cpp"),
                                [BATCH, DEPTH] + acc, acc, rargs, wargs,
                                self.defines + gdef + rows + [("NPOOL", str(NPOOL)), ("RAWK", str(RAWK))],
                                noc1_rows=self.noc1_rows)
        f32, bf16, u8 = ttnn.float32, ttnn.bfloat16, ttnn.uint8
        T4, T2 = 4096, 2048
        spec = [(0, ttnn.bfloat8_b, 1088, BATCH * DEPTH), (1, bf16, T2, 80), (2, bf16, T2, 1),
                (3, f32, T4, 3), (4, f32, T4, 3), (5, f32, T4, 1), (6, f32, T4, 1), (7, f32, T4, 2), (8, f32, T4, 1),
                (9, f32, T4, 4), (10, bf16, T2, 10), (11, f32, T4, 4), (12, f32, T4, 4), (13, f32, T4, 4),
                (14, bf16, T2, 1), (15, f32, T4, 1), (21, f32, T4, 4), (22, f32, T4, 1),
                (23, f32, T4, 16), (24, f32, T4, 8), (25, f32, T4, 4), (26, f32, T4, 8), (27, f32, T4, 1),
                (28, f32, T4, 16), (29, f32, T4, 8), (30, f32, T4, 8), (31, bf16, T2, 8), (32, f32, T4, 8),
                (33, f32, T4, 4), (60, u8, T4, 1), (61, u8, self.arena, 1), (62, u8, max(1024, CTX // 64), 1),
                (63, u8, 64 + gbytes, 1)]
        if mr:  # M rows: the ones tile + Sel + the step's words, H1's raw X chunks double-buffered, a 4 KB stage
            size = {2: 3, 3: 6}
            spec = [(i, t, 4096 if i == 62 else pg, size.get(i, k)) for i, t, pg, k in spec]
        self.l1_bytes = sum(size * n for _, _, size, n in spec)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        ins = [self.x32_t, self.x16_t, self.wd, self.wu, self.hnorm_t, self.wp, self.pooled, self.rawk,
               self.rope_t, self.step] + list(self.n_t.values()) + [self.prof] + list(self.chk.values()) + [
               self.scores, self.q, self.g_out, self.kc, self.vc] + ([] if self.sel_out is None else [self.sel_out]) + (
                   [] if self.gprof is None else [self.gprof])
        return lambda: ttnn.generic_op(ins, prog)

    def get(self, t):
        return ttnn.to_torch(t).double().flatten()

    def errors(self) -> dict:
        res = {}
        X, X16 = self.x.reshape(HC, D), bf(self.x).reshape(HC, D)
        r = torch.rsqrt(X.pow(2).mean(-1) + EPS)
        p_ref = torch.stack([r[s] * (X16[s] @ self.wdq[s * D:(s + 1) * D]) for s in range(HC)])
        p1 = self.get(self.chk["p1"]).reshape(HC, LO)
        res["H1"] = rel(p1[:, :LR + HC], p_ref[:, :LR + HC])
        lo = p1.sum(0)
        gate = (bf(torch.nn.functional.silu(lo[:LR] / HC)) @ self.wuq).reshape(HC, D)
        hn = (torch.sigmoid(gate) * X * self.hnorm.double().reshape(HC, D) * r[:, None]).mean(0)
        res["H2_hn"] = rel(self.get(self.chk["hn"]), hn)
        # P from the card's hn
        y = bf(self.get(self.chk["hn"])) @ self.wpq
        y[:384 * 32].reshape(NH, 2, HD)[:, 1] = torch.sigmoid(y[:384 * 32].reshape(NH, 2, HD)[:, 1])
        pa = self.get(self.chk["pa"])
        res["P"] = rel(pa, y)
        # N from the card's PA
        q = pa[:384 * 32].reshape(NH, 2, HD)[:, 0]
        qr = rms(q, self.n["q"].double()) @ self.rope["main"] / 16
        res["N_q"] = rel(self.get(self.q).reshape(NH, HD), qr)
        res["N_gate"] = rel(self.get(self.g_out).reshape(NH, HD), pa[:384 * 32].reshape(NH, 2, HD)[:, 1])
        k = pa[T_K * 32:T_V * 32].reshape(NKV, HD)
        kr = rms(k, self.n["k"].double()) @ self.rope["main"]
        v = pa[T_V * 32:T_IQ * 32].reshape(NKV, HD)
        kc = ttnn.to_torch(self.kc).double().reshape(NKV, NPOOL, KPOOL, HD)[:, self.pos // 4, self.pos % 4]
        vc = ttnn.to_torch(self.vc).double().reshape(NKV, NPOOL, KPOOL, HD)[:, self.pos // 4, self.pos % 4]
        res["N_kcache"], res["N_vcache"] = rel(kc, bf(kr)), rel(vc, bf(v))
        ik = pa[T_IK * 32:NT * 32]
        res["N_rawk"] = rel(Ring(ttnn.to_torch(self.rawk).double().reshape(RAWK, IHD))[self.pos], ik)
        pooled = ttnn.to_torch(self.pooled).double().reshape(NPOOL // 32, 4, 32, 32).permute(0, 2, 1, 3)
        pooled = pooled.reshape(NPOOL, IHD)
        blk = self.pos // 4
        iq = rms(pa[T_IQ * 32:T_IK * 32].reshape(IH, IHD), self.n["iq"].double()) @ self.rope["idx"]
        if self.pos % KPOOL == KPOOL - 1:
            mean = torch.cat([self.rawk_before[self.pos - 3:self.pos], ik[None]]).mean(0)
            k_new = rms(mean, self.n["ik"].double()) @ self.rope["pool"]
            res["N_pool"] = rel(pooled[blk], k_new)
            # the closed block's own score from the new key (llama.cpp ranks it with the others)
            own_ref = torch.relu(iq @ k_new).sum() * IHD ** -0.5
            res["N_own_score"] = float((self.get(self.scores)[NPOOL] - own_ref).abs() / own_ref.abs().clamp(min=1e-3))
        else:   # the pool role ran but must not have written
            res["N_pool_untouched"] = bool(torch.equal(pooled[blk], self.pk_before[blk]))
        P = self.pos // KPOOL   # the complete pools before the token's block
        if P:
            sc_ref = torch.relu(iq @ self.pk_before[:P].T).sum(0) * IHD ** -0.5
            sc = self.get(self.scores)[:P]
            res["N_scores"] = rel(sc, sc_ref)
            k = min(512, P)
            top_ref = set(torch.topk(sc_ref, k).indices.tolist())
            res["N_top_agree"] = f"{len(set(torch.topk(sc, k).indices.tolist()) & top_ref)}/{k}"
        return res

    def pooled_key(self, block: int) -> torch.Tensor:
        """A block's pooled key as the card's cache holds it."""
        pooled = ttnn.to_torch(self.pooled).double().reshape(NPOOL // 32, 4, 32, 32).permute(0, 2, 1, 3)
        return pooled.reshape(NPOOL, IHD)[block]

    def row_scores(self) -> list[np.ndarray]:
        """M rows: each row's pool scores as the host takes them (float32 [NPOOL]): the block the pass closes (row
        mc's) scored from its new key for the rows m >= mc (word NPOOL of the row), in place of the stale score."""
        sc = ttnn.to_torch(self.scores).float().reshape(self.m_max, NPOOL + 16).numpy()
        out = []
        for m in range(self.M):
            s = sc[m, :NPOOL].copy()
            if self.mc <= m:
                s[(self.pos + self.mc) // KPOOL] = sc[m, NPOOL]
            out.append(s)
        return out

    def errors_m(self) -> dict:
        """M rows: every phase of every row m < M against float64 of the single-token step at pos + m, from the card's
        own previous outputs (as errors()), the caches before the pass for the cells before it; the block the pass
        closes from the cache's cells before pos and PA's rows after. Returns the worst row per check and every row's
        ("rows"), and "ref_scores" (each row's reference scores as the host takes them, for the selection)."""
        M, mp, pos, mm = self.M, self.mp, self.pos, self.m_max
        un = lambda k, n: unblocks(self.get(self.chk[k]), n, mp)                        # noqa: E731
        p1, hn, pa = un("p1", HC * LO // 32), un("hn", D // 32), un("pa", NT)
        w = self.get(self.chk["w"])[:mp * 16].reshape(mp, 16)
        Q = ttnn.to_torch(self.q).double().reshape(mm, NH, HD)
        G = ttnn.to_torch(self.g_out).double().reshape(mm, NH, HD)
        sc = ttnn.to_torch(self.scores).double().reshape(mm, NPOOL + 16)
        kc = ttnn.to_torch(self.kc).double().reshape(NKV, NPOOL * KPOOL, HD)
        vc = ttnn.to_torch(self.vc).double().reshape(NKV, NPOOL * KPOOL, HD)
        rawk = Ring(ttnn.to_torch(self.rawk).double().reshape(RAWK, IHD))
        pooled = ttnn.to_torch(self.pooled).double().reshape(NPOOL // 32, 4, 32, 32).permute(0, 2, 1, 3)
        pooled = pooled.reshape(NPOOL, IHD)
        close = self.mc < M
        pc = pos + (self.mc if close else 0)
        if close:   # the block's new key: its cells before the pass from the cache, the rest from PA's rows
            keys = [self.rawk_before[c] if c < pos else pa[c - pos][T_IK * 32:NT * 32] for c in range(pc - 3, pc + 1)]
            k_new = rms(torch.stack(keys).mean(0), self.n["ik"].double()) @ rope_matrix(pc - 3, IHD).double()
        rows, ref_scores, got = [], [], self.row_scores()
        for m in range(M):
            res, rope = {}, self.ropes[m]
            X, X16 = self.x[m].reshape(HC, D), bf(self.x[m]).reshape(HC, D)
            r = torch.rsqrt(X.pow(2).mean(-1) + EPS)
            p_ref = torch.stack([r[s] * (X16[s] @ self.wdq[s * D:(s + 1) * D]) for s in range(HC)])
            pm = p1[m].reshape(HC, LO)
            res["H1"] = rel(pm[:, :LR + HC], p_ref[:, :LR + HC])
            lo = pm.sum(0)
            gate = (bf(torch.nn.functional.silu(lo[:LR] / HC)) @ self.wuq).reshape(HC, D)
            hn_ref = (torch.sigmoid(gate) * X * self.hnorm.double().reshape(HC, D) * r[:, None]).mean(0)
            res["H2_hn"] = rel(hn[m], hn_ref)
            res["H2_w"] = rel(w[m][:HC], 2 * torch.sigmoid(lo[LR:LR + HC] / HC))
            y = bf(hn[m]) @ self.wpq
            y[:384 * 32].reshape(NH, 2, HD)[:, 1] = torch.sigmoid(y[:384 * 32].reshape(NH, 2, HD)[:, 1])
            res["P"] = rel(pa[m], y)
            q = pa[m][:384 * 32].reshape(NH, 2, HD)[:, 0]
            res["N_q"] = rel(Q[m], rms(q, self.n["q"].double()) @ rope["main"] / 16)   # (bf16)
            res["N_gate"] = rel(G[m], pa[m][:384 * 32].reshape(NH, 2, HD)[:, 1])
            k = pa[m][T_K * 32:T_V * 32].reshape(NKV, HD)
            kr = rms(k, self.n["k"].double()) @ rope["main"]
            v = pa[m][T_V * 32:T_IQ * 32].reshape(NKV, HD)
            res["N_kcache"], res["N_vcache"] = rel(kc[:, pos + m], kr), rel(vc[:, pos + m], bf(v))   # (k: bf16)
            res["N_rawk"] = rel(rawk[pos + m], pa[m][T_IK * 32:NT * 32])
            iq = rms(pa[m][T_IQ * 32:T_IK * 32].reshape(IH, IHD), self.n["iq"].double()) @ rope["idx"]
            P = (pos + m) // KPOOL   # the complete pools before row m's block
            s_ref = None
            if P:
                s_ref = torch.relu(iq @ self.pk_before[:P].T).sum(0) * IHD ** -0.5
                # the closed block's stale score, which the host replaces for these rows: past a score core's second
                # block the pooled keys are read after the barrier, so it may see the pool role's new key
                live = torch.ones(P, dtype=torch.bool)
                if close and m >= self.mc and pc // KPOOL < P:
                    live[pc // KPOOL] = False
                res["N_scores"] = rel(sc[m, :P][live], s_ref[live])
            full = torch.zeros(NPOOL, dtype=torch.float64)
            if s_ref is not None:
                full[:P] = s_ref
            if close and m >= self.mc:   # the closed block's score from its new key (relative to the row's scores)
                own_ref = torch.relu(iq @ k_new).sum() * IHD ** -0.5
                scale = max(float(own_ref.abs()), float(s_ref.abs().max()) if s_ref is not None else 0.0, 1e-3)
                res["N_own_score"] = float((sc[m, NPOOL] - own_ref).abs()) / scale
                full[pc // KPOOL] = own_ref
            ref_scores.append(full.float().numpy())
            n = min(512, (pos + m + 1) // KPOOL)
            if n:
                top_ref = set(np.argsort(-ref_scores[-1][:(pos + m + 1) // KPOOL])[:n].tolist())
                top = set(np.argsort(-got[m][:(pos + m + 1) // KPOOL])[:n].tolist())
                res["N_top_agree"] = len(top & top_ref) / n
            rows.append(res)
        out = {}
        for key in sorted({k for r in rows for k in r}):
            vals = [r[key] for r in rows if key in r]
            out[key] = min(vals) if key == "N_top_agree" else max(vals)
        if close:
            out["N_pool"] = rel(pooled[pc // KPOOL], k_new)
        keep = torch.ones(NPOOL, dtype=torch.bool)
        keep[pc // KPOOL] = not close
        out["N_pool_others_untouched"] = bool(torch.equal(pooled[keep], self.pk_before[keep]))
        out["rows"] = rows
        return out, ref_scores


def link_check(dev, A1) -> dict:
    """Program 1 with -DEXCHANGE: the host takes the scores page and replies with the selection words (g4_host's
    rule), which the coordinator writes into program 2's selection tensor. A page per 2,048 pools up to the token's."""
    from u1_sockets import pinned
    from g4_host import Host
    from g4_model import select_words
    core = ttnn.CoreCoord(*A1.grid.cores[0])                       # the coordinator
    where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core)
    h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 64 << 10, ttnn.H2DMode.DEVICE_PULL))
    h2d.set_page_size(D * 4)
    d2h = pinned(lambda: ttnn.D2HSocket(dev, where, 64 << 10))
    d2h.set_page_size(8192)
    A1.defines = [("EXCHANGE", "1"), ("H2D_CONFIG", str(h2d.get_config_buffer_address())),
                  ("D2H_CONFIG", str(d2h.get_config_buffer_address())), ("XCHG_OUT_BYTES", "8192")]
    nsel_words = 512 + 16
    A1.sel_out = ttnn.from_torch(torch.full((1, 1, 1, nsel_words), -1, dtype=torch.int32), dtype=ttnn.uint32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    prog = A1.program()
    pages = torch.zeros(NPOOL * 4 // 8192, 8192, dtype=torch.uint8)
    reply = torch.zeros(D * 4, dtype=torch.uint8)
    words = reply.numpy().view(np.uint32)
    res = {}

    parts = np.zeros(3)

    same, check_ref = [], [True]

    def serve():
        t0 = time.perf_counter()
        n = A1.pos // 4 * 4 // 8192 + 1                            # g4_model.Link.serve_select's pages
        for k in range(n):
            d2h.read_tensor(pages[k])
        sc = pages[:n].numpy().reshape(-1).view(np.float32)
        t1 = time.perf_counter()
        nsel, own = select_words(A1.pos, sc, words)              # the token loop's (g4_model.Link.serve_select)
        t2 = time.perf_counter()
        h2d.write_tensor(reply)
        parts[:] += (t1 - t0, t2 - t1, time.perf_counter() - t2)
        if not check_ref[0]:
            return
        sel, n_ref, own_ref = Host.select(A1.pos, sc)            # the reference rule: the same set
        same.append(n_ref == nsel and own_ref == own and set(sel[:n_ref].tolist()) == set(words[:nsel].tolist()))

    for pos in [p for p in (5, 5, 100, 1003, 8191, 8192, 8195, 12003, 16387, 24579, 40003, 57347, 65539, 131075,
                            196611) if p < CTX - 1] + [
            CTX - 1]:                                                # the first launch compiles
        A1.set_pos(pos)
        A1.run(prog, serve)
        sc = ttnn.to_torch(A1.scores).float().flatten()
        if pos % 4 == 3:   # the link sends the closed block's own score in place of the stale one
            sc[pos // 4] = sc[NPOOL]
        sent = pages.reshape(-1).view(torch.float32)[:pos // 4 + 1]
        got = ttnn.to_torch(A1.sel_out).flatten().numpy().view(np.uint32)
        err = A1.errors()
        res[pos] = dict(scores_vs_tensor=float((sent - sc[:pos // 4 + 1]).abs().nan_to_num(0).max()),
                        selection_same=bool((got == words[:nsel_words]).all()), nsel=int(words[512]),
                        own_valid=int(words[513]), scores_err=err["N_scores"],
                        top_agree=err.get("N_top512_agree"))
    # the time of a launch with the host servicing the link, against the same launch without it
    A1.set_pos(CTX - 1)
    ts = []
    parts[:] = 0
    check_ref[0] = False
    for _ in range(10):
        t0 = time.perf_counter()
        prog()
        serve()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e6)
    A1.defines, sel_keep, A1.sel_out = [], A1.sel_out, None
    plain = A1.program()
    A1.sel_out = sel_keep
    plain()
    ttnn.synchronize_device(dev)
    tp = []
    for _ in range(10):
        t0 = time.perf_counter()
        plain()
        ttnn.synchronize_device(dev)
        tp.append((time.perf_counter() - t0) * 1e6)
    # in the loop: program 2 (here a second program 1 stands in for it) queued behind program 1, against the host
    # sync path (sync, read the scores, select, write the selection, launch the next program)
    sel_t = ttnn.from_torch(torch.zeros(1, 1, 1, nsel_words, dtype=torch.int32), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    old, new = [], []
    for _ in range(12):
        t0 = time.perf_counter()
        plain()
        ttnn.synchronize_device(dev)
        sc = ttnn.to_torch(A1.scores).float().flatten().numpy()
        if A1.pos % 4 == 3:
            sc[A1.pos // 4] = sc[NPOOL]
        select_words(A1.pos, sc, words)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.from_numpy(words[:nsel_words].view(np.int32).copy())
                                                        .reshape(1, 1, 1, -1), dtype=ttnn.uint32,
                                                        layout=ttnn.ROW_MAJOR_LAYOUT), sel_t)
        plain()
        ttnn.synchronize_device(dev)
        old.append((time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter()
        prog()
        plain()
        serve()
        ttnn.synchronize_device(dev)
        new.append((time.perf_counter() - t0) * 1e6)
    res["pair_us_sync_path"], res["pair_us_link"] = float(np.median(old[2:])), float(np.median(new[2:]))
    res["launch_us_with_link"], res["launch_us_without"] = float(np.median(ts[2:])), float(np.median(tp[2:]))
    A1.sel_out = None
    res["select_same_as_host_rule"] = all(same)
    res["serve_us"] = dict(zip(("wait for the page", "selection", "write the reply"), (parts / 10 * 1e6).tolist()))
    print("host link: " + "; ".join(f"pos {p}: scores {r['scores_vs_tensor']:.1e}, selection same "
                                    f"{r['selection_same']} (nsel {r['nsel']}, own {r['own_valid']})"
                                    for p, r in res.items() if isinstance(p, int))
          + f"; the selection the same as g4_host's rule: {res['select_same_as_host_rule']}"
          + f"; one launch + sync {res['launch_us_with_link']:.0f} us with the host serving it, "
          f"{res['launch_us_without']:.0f} us without the link; two programs back to back {res['pair_us_link']:.0f} "
          f"us through the link, {res['pair_us_sync_path']:.0f} us through a host sync; the host's side (us): "
          + ", ".join(f"{k} {v:.0f}" for k, v in res["serve_us"].items()), flush=True)
    return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": LAYER}
    try:
        A1 = Attn1(dev, g, rng, pos=0)
        prog = A1.program()
        print(f"weights in {time.time() - t0:.0f} s; circular buffers {A1.l1_bytes / 1024:.0f} KB per core",
              flush=True)
        res["max_rel_err"] = {}
        for pos in POSITIONS:       # one program; only the step tensor changes
            A1.set_pos(pos)
            A1.run(prog)
            err = res["max_rel_err"][pos] = A1.errors()
            print(f"pos {pos}: " + ", ".join(f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}"
                                             for k, v in err.items()), flush=True)
        A1.set_pos(CTX - 1)
        res["traced_us"] = traced_us(dev, A1.program())
        st = ttnn.to_torch(A1.prof).flatten().long().tolist()
        names = ["X", "H1", "HL", "H2", "P", "N"]
        cyc = [(st[i + 1] - st[i]) & 0xFFFFFFFF for i in range(len(names))]
        res["phase_cycles"] = dict(zip(names, cyc))
        print(f"program 1 traced {res['traced_us']:.1f} us; phases (cycles): "
              + ", ".join(f"{n} {c}" for n, c in zip(names, cyc)), flush=True)
        A1.noc1_rows = ()                                          # for comparison: every row on NOC0
        res["traced_us_noc0_only"] = traced_us(dev, A1.program())
        A1.noc1_rows = NOC1_ROWS
        print(f"traced {res['traced_us']:.1f} us with rows {NOC1_ROWS} on NOC1, {res['traced_us_noc0_only']:.1f} us "
              f"on NOC0 alone", flush=True)
        res["link"] = {str(k): v for k, v in link_check(dev, A1).items()}
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g3_attn1.json" if CTX == 16384 else f"g3_attn1_ctx{CTX}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
