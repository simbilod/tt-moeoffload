"""G2 for GLM-5.3-Flash Maya-L: one KDA and one MLA (+ indexer) layer step on the p150a, decode-shaped.

One token through a whole layer except its routed experts: the mHC pre-mix (with the 20-step Sinkhorn), the RMSNorm,
the mixer, the mHC post-mix, again for the FFN half, the router with its top-8, the shared expert and the add of the
experts' sum (a stand-in tensor here: the CPU lane's and the card's experts are priced from Strata's log and the card's measured units).
Real weights of the staged GGUF: blk.4 (KDA + MoE) and blk.3 (MLA with its full k-pool indexer + MoE), dequantized
on the host and uploaded as bfp8_b (Q6_K / Q8_0 have no card form yet; bfp8_b is ttnn's decode weight format), the
small F32 tensors as bf16 / fp32. The big projections are DRAM-sharded decode matmuls (tt_transformers' recipe:
weight width-sharded over the 8 banks, activation width-sharded in L1).

Each step is checked once against a float32 torch reference of llama.cpp's glm5-next graph (src/models/glm5-next.cpp)
on the same dequantized weights and synthetic state (random hidden streams, KDA state, conv window, an 8K latent
cache and its 2,048 pooled indexer keys), then traced and replayed 46 times per sync, like G1; medians of 20.

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/glm/g2_layers.py
"""
from __future__ import annotations

import glob
import json
import math
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import statistics
import sys
import time

import numpy as np
import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp", "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/GLM-5.3-Flash Maya-L")   # engine.py --model-dir sets it
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KDA_LAYER, MLA_LAYER, DENSE_LAYER = 4, 3, 2
D, HC, NH, HD = 4096, 4, 64, 128          # hidden, mHC streams, heads, KDA head dim
NX, NE, TOPK = 288, 8, 8                  # experts, padded below to 512 for the router matmul
RMS_EPS, HC_EPS, LN_EPS, L2_EPS = 1e-5, 1e-6, 1e-6, 1e-6
SINKHORN, KDA_LB = 20, -5.0
QL, KVL, HK = 1536, 512, 256              # q lora, kv lora (the latent), MLA nope head dim
IH, IHD, KPOOL, ITOP = 32, 128, 4, 2048   # indexer heads, head dim, pool size, tokens kept
CTX = int(os.environ.get("G2_CTX", 8192))
LAYERS, TOKENS = 46, 20
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
BF16, BFP8, FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
MM_CK = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)
HI_CK = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                         fp32_dest_acc_en=True, packer_l1_acc=True)


class Gguf:
    def __init__(self, folder: str):
        self.t = {}
        for f in sorted(glob.glob(os.path.join(folder, "*.gguf"))):
            for t in GGUFReader(f).tensors:
                self.t[t.name] = t
        self.bytes = 0

    def __call__(self, name: str) -> torch.Tensor:
        """float32, numpy order (ne[-1], ..., ne[0]): a 2-D weight comes back [out, in]."""
        t = self.t[name]
        self.bytes += int(t.n_bytes)
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        return torch.from_numpy(a.reshape([int(x) for x in reversed(t.shape)]).copy())


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


class Card:
    def __init__(self, dev):
        self.dev = dev

    def put(self, t: torch.Tensor, dtype=BF16, layout=TILE, mc=None) -> ttnn.Tensor:
        return ttnn.from_torch(t.contiguous(), device=self.dev, layout=layout, dtype=dtype,
                               memory_config=mc or ttnn.DRAM_MEMORY_CONFIG)

    def row(self, v: torch.Tensor, dtype=BF16) -> ttnn.Tensor:
        return self.put(v.reshape(1, 1, 1, -1), dtype)

    def norm_w(self, v: torch.Tensor) -> ttnn.Tensor:
        return self.put(v.reshape(1, 1, -1, 32), BF16, RM)

    def mm(self, w_kn: torch.Tensor, dtype=BFP8) -> "DramMM":
        return DramMM(self, w_kn, dtype)


# Derived from tt-metal's models/tt_transformers/tt/model_config.py (create_dram_sharded_mem_config, find_grid_k_n,
# find_largest_divisor, dram_decode_in0_block_w), Copyright 2026 Tenstorrent USA, Inc., Apache License 2.0
# (LICENSES/tt-metal-Apache-2.0.txt); condensed into DramMM here.
class DramMM:
    """x [1,1,1,k] @ w [k,n] with w width-sharded over the DRAM banks and x width-sharded in L1."""

    def __init__(self, card: Card, w: torch.Tensor, dtype):
        dev = card.dev
        k, n = w.shape
        banks = dev.dram_grid_size().x
        n_pad = math.ceil(n / (32 * banks)) * 32 * banks
        self.n, self.n_pad, self.bytes = n, n_pad, k * n * (1.0625 if dtype == BFP8 else 2)
        kt, nt = k // 32, n_pad // 32
        cores = max(c for c in range(1, 65) if kt % c == 0 and nt % c == 0)
        rows = next(r for r in range(1, 9) if cores % r == 0 and cores // r <= 8)
        grid = ttnn.CoreGrid(y=rows, x=cores // rows)
        wp = torch.zeros(k, n_pad)
        wp[:, :n] = w
        dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        wmc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                ttnn.ShardSpec(dram, (k, n_pad // banks), ttnn.ShardOrientation.ROW_MAJOR))
        self.w = card.put(wp, dtype, mc=wmc)
        shard = lambda width: ttnn.create_sharded_memory_config(  # noqa: E731
            shape=(32, width), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        self.in_mc, self.out_mc = shard(k // cores), shard(n_pad // cores)
        per_core_k, per_worker_n = kt // cores, math.ceil(nt / banks)
        bw = max(d for d in range(1, 9) if per_core_k % d == 0)
        for cand in range(min(16, kt), bw, -1):  # tt_transformers' dram_decode_in0_block_w
            if kt % cand or (per_core_k % cand and cand % per_core_k) or cand * per_worker_n > 100:
                continue
            if cand * per_worker_n <= 800 * 1024 // (3 * 1088):
                bw = cand
                break
        self.pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=bw, per_core_M=1, per_core_N=nt // cores, fused_activation=None)

    def __call__(self, x: ttnn.Tensor, dtype=None) -> ttnn.Tensor:
        y = ttnn.matmul(ttnn.to_memory_config(x, self.in_mc), self.w, program_config=self.pc,
                        memory_config=self.out_mc, compute_kernel_config=MM_CK, dtype=dtype)
        y = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
        return y if self.n == self.n_pad else ttnn.slice(y, [0, 0, 0, 0], [1, 1, 1, self.n])


# ---------------------------------------------------------------- mHC (shared by both layer kinds)

def hc_consts() -> dict:
    """Constant matrices over the 32-wide mix row: cols 0-3 pre, 4-7 post, 8-23 comb (8 + dst + 4*src)."""
    e_pre = torch.zeros(32, HC * D)
    for i in range(HC):
        e_pre[i, i * D:(i + 1) * D] = 1
    inside = torch.zeros(32)
    inside[8:24] = 1
    g_dst, g_src = torch.zeros(32, 32), torch.zeros(32, 32)
    for i in range(8, 24):
        for j in range(8, 24):
            g_dst[i, j] = float((i - 8) // 4 == (j - 8) // 4)   # same src: the softmax / row groups
            g_src[i, j] = float((i - 8) % 4 == (j - 8) % 4)     # same dst: the column groups
    return dict(e_pre=e_pre, inside=inside, g_dst=g_dst, g_src=g_src)


def hc_ref(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    """llama.cpp build_hc_pre: x [HC, D] -> (h [D], post [HC], comb [dst, src])."""
    flat = x.flatten()
    mixes = fn @ (flat * torch.rsqrt(flat.pow(2).mean() + RMS_EPS))     # fn [24, 16384]
    pre = torch.sigmoid(mixes[:4] * scale[0] + base[:4]) + HC_EPS
    post = 2 * torch.sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb = (mixes[8:24] * scale[2] + base[8:24]).reshape(HC, HC)         # [src, dst] (ne0 = dst)
    comb = torch.softmax(comb, dim=1) + HC_EPS
    comb = comb / (comb.sum(0, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(0, keepdim=True) + HC_EPS)
    return (pre[:, None] * x).sum(0), post, comb.T                       # comb.T [dst, src]


def hc_post_ref(out, x, post, comb):
    return post[:, None] * out[None, :] + comb @ x


class HcCard:
    """One mHC site (attn or ffn) of a layer on the card. The streams travel flat: X [1,1,1,HC*D] bf16."""

    def __init__(self, card: Card, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, k: dict):
        w = torch.zeros(HC * D, 32)
        w[:, :24] = fn.T
        self.fn = card.put(w, BF16)
        sv, bv = torch.zeros(32), torch.zeros(32)
        sv[:4], sv[4:8], sv[8:24] = scale[0], scale[1], scale[2]
        bv[:24] = base
        self.scale, self.base = card.row(sv, FP32), card.row(bv, FP32)
        self.e_pre = card.put(k["e_pre"], FP32)
        self.inside = card.row(k["inside"], FP32)
        self.g_dst, self.g_src = card.put(k["g_dst"], FP32), card.put(k["g_src"], FP32)
        self.one_out = card.row(1 - k["inside"], FP32)                   # denominators 1 outside the comb
        self.eps_in = card.row(k["inside"] * HC_EPS + (1 - k["inside"]), FP32)
        self.eps_add = card.row(k["inside"] * HC_EPS, FP32)

    def mix(self, X):
        """-> a (affine mixes, fp32 [1,1,1,32]), s = sigmoid(a)."""
        fn = ttnn.rms_norm(X, epsilon=RMS_EPS)
        mixes = ttnn.matmul(fn, self.fn, dtype=FP32, compute_kernel_config=HI_CK)
        a = ttnn.add(ttnn.multiply(mixes, self.scale), self.base)
        return a, ttnn.sigmoid(a)

    def sinkhorn(self, a):
        c = ttnn.multiply(ttnn.exp(a), self.inside)
        c = ttnn.divide(c, ttnn.linear(c, self.g_dst, bias=self.one_out, compute_kernel_config=HI_CK))
        c = ttnn.add(c, self.eps_add)
        c = ttnn.divide(c, ttnn.linear(c, self.g_src, bias=self.eps_in, compute_kernel_config=HI_CK))
        for _ in range(SINKHORN - 1):
            c = ttnn.divide(c, ttnn.linear(c, self.g_dst, bias=self.eps_in, compute_kernel_config=HI_CK))
            c = ttnn.divide(c, ttnn.linear(c, self.g_src, bias=self.eps_in, compute_kernel_config=HI_CK))
        return c

    def pre(self, X, s):
        pre = ttnn.add(ttnn.matmul(s, self.e_pre, compute_kernel_config=HI_CK), HC_EPS)
        xw = ttnn.multiply(X, pre)
        return ttnn.sum(ttnn.reshape(xw, [1, 1, HC, D]), dim=-2, keepdim=True)

    @staticmethod
    def post(out, X, s, c):
        cm = ttnn.reshape(ttnn.slice(c, [0, 0, 0, 8], [1, 1, 1, 24]), [1, 1, HC, HC])   # [src, dst]
        y = ttnn.matmul(cm, ttnn.typecast(ttnn.reshape(X, [1, 1, HC, D]), FP32), transpose_a=True,
                        compute_kernel_config=HI_CK)
        post = ttnn.multiply(ttnn.reshape(ttnn.slice(s, [0, 0, 0, 4], [1, 1, 1, 8]), [1, 1, HC, 1]), 2.0)
        y = ttnn.add(y, ttnn.multiply(post, ttnn.typecast(out, FP32)))
        return ttnn.typecast(ttnn.reshape(y, [1, 1, 1, HC * D]), BF16)


# ---------------------------------------------------------------- the FFN half (router + shared expert)

class MoeCard:
    def __init__(self, card: Card, g: Gguf, il: int, ref: dict):
        router = g(f"blk.{il}.ffn_gate_inp.weight")                      # [288, 4096]
        bias = g(f"blk.{il}.exp_probs_b.bias")
        gate, up = g(f"blk.{il}.ffn_gate_shexp.weight"), g(f"blk.{il}.ffn_up_shexp.weight")   # [2048, 4096]
        down = g(f"blk.{il}.ffn_down_shexp.weight")                      # [4096, 2048]
        ref.update(router=router, probs_b=bias, sh_gate=gate, sh_up=up, sh_down=down)
        self.router = card.mm(router.T, BF16)
        bp = torch.full((512,), -1e4)
        bp[:NX] = bias
        self.bias = card.row(bp)
        self.sh_gu = card.mm(torch.cat([gate.T, up.T], dim=1))
        self.sh_down = card.mm(down.T)
        self.routed = card.row(torch.zeros(D))                           # the experts' sum, from the CPU lane

    def __call__(self, hn):
        probs = ttnn.sigmoid(ttnn.pad(self.router(hn), [(0, 0), (0, 0), (0, 0), (0, 512 - NX)], 0.0))
        sel = ttnn.add(probs, self.bias)
        vals, idx = ttnn.topk(sel, k=TOPK, dim=-1)
        kth = ttnn.slice(vals, [0, 0, 0, TOPK - 1], [1, 1, 1, TOPK])
        w = ttnn.multiply(probs, ttnn.ge(sel, kth))                      # the 8 router weights, dense over 512
        w = ttnn.multiply(ttnn.divide(w, ttnn.sum(w, dim=-1, keepdim=True)), 2.5)
        gu = self.sh_gu(hn)
        gate = ttnn.clamp(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 1, 2048]), max=10.0)
        up = ttnn.clamp(ttnn.slice(gu, [0, 0, 0, 2048], [1, 1, 1, 4096]), min=-10.0, max=10.0)
        sh = self.sh_down(ttnn.multiply(ttnn.silu(gate), up))
        return ttnn.add(sh, self.routed), idx, w


def moe_ref(hn, r):
    probs = torch.sigmoid(r["router"] @ hn)
    idx = torch.topk(probs + r["probs_b"], TOPK).indices
    w = probs[idx] / probs[idx].sum() * 2.5
    gate, up = (r["sh_gate"] @ hn).clamp(max=10), (r["sh_up"] @ hn).clamp(-10, 10)
    return r["sh_down"] @ (torch.nn.functional.silu(gate) * up), idx, w


# ---------------------------------------------------------------- KDA layer

class KdaCard:
    def __init__(self, card: Card, g: Gguf, il: int, k: dict, rng: torch.Generator):
        p = f"blk.{il}."
        r = {}
        for site in ("attn", "ffn"):
            r[f"hc_{site}"] = (g(p + f"hc_{site}_fn.weight"), g(p + f"hc_{site}_scale.weight"),
                               g(p + f"hc_{site}_base.weight"))
        self.hc_attn, self.hc_ffn = HcCard(card, *r["hc_attn"], k), HcCard(card, *r["hc_ffn"], k)
        r["attn_norm"], r["ffn_norm"] = g(p + "attn_norm.weight"), g(p + "ffn_norm.weight")
        self.attn_norm, self.ffn_norm = card.norm_w(r["attn_norm"]), card.norm_w(r["ffn_norm"])
        wq, wk, wv = (g(p + f"attn_{x}.weight") for x in "qkv")         # [8192, 4096]
        taps = [g(p + f"ssm_conv1d_{x}.weight").reshape(NH * HD, 4).T for x in "qkv"]   # [4, 8192]
        r.update(wq=wq, wk=wk, wv=wv, taps=torch.cat(taps, dim=1))
        for x in ("f_a", "f_b", "g_a", "g_b", "beta"):
            r[x] = g(p + f"ssm_{x}.weight")
        r["A"], r["dt"], r["o_norm"] = g(p + "ssm_a"), g(p + "ssm_dt.bias"), g(p + "ssm_norm.weight")
        r["wo"] = g(p + "attn_output.weight")                            # [4096, 8192]
        self.moe = MoeCard(card, g, il, r)
        self.qkv = card.mm(torch.cat([wq.T, wk.T, wv.T], dim=1))
        self.taps = card.put(r["taps"].reshape(1, 1, 4, -1))
        self.f_a, self.f_b = card.put(r["f_a"].T, BFP8), card.put(r["f_b"].T, BFP8)
        self.g_a, self.g_b = card.put(r["g_a"].T, BFP8), card.put(r["g_b"].T, BFP8)
        self.beta = card.put(r["beta"].T, BFP8)
        self.neg_a = card.row(-r["A"].repeat_interleave(HD))             # ssm_a holds -exp(A_log)
        self.dt = card.row(r["dt"])
        self.o_norm = card.norm_w(r["o_norm"])
        self.wo = card.mm(r["wo"].T)
        r["S0"] = torch.randn(NH, HD, HD, generator=rng) * 0.05          # [head, k, v]
        r["conv0"] = torch.randn(3, 3 * NH * HD, generator=rng) * 0.5    # x_{t-3}, x_{t-2}, x_{t-1}
        self.S = card.put(r["S0"].reshape(1, NH, HD, HD))
        self.conv = card.put(r["conv0"].reshape(1, 1, 3, -1))
        self.ref = r

    def mixer(self, hn):
        qkv = self.qkv(hn)
        win = ttnn.concat([self.conv, qkv], dim=2)                       # [1,1,4,24576]
        ttnn.copy(ttnn.slice(win, [0, 0, 1, 0], [1, 1, 4, 3 * NH * HD]), self.conv)
        qkv = ttnn.silu(ttnn.sum(ttnn.multiply(win, self.taps), dim=2, keepdim=True))
        part = lambda i: ttnn.slice(qkv, [0, 0, 0, i * NH * HD], [1, 1, 1, (i + 1) * NH * HD])  # noqa: E731
        l2 = lambda t, s: ttnn.multiply(ttnn.rms_norm(ttnn.reshape(t, [1, 1, NH, HD]),  # noqa: E731
                                                      epsilon=L2_EPS / HD), s)
        q = ttnn.reshape(l2(part(0), HD ** -1.0), [1, NH, 1, HD])        # l2 norm and the 1/sqrt(HD) scale
        kr = ttnn.reshape(l2(part(1), HD ** -0.5), [1, NH, 1, HD])
        v = ttnn.reshape(part(2), [1, NH, 1, HD])
        g1 = ttnn.add(ttnn.matmul(ttnn.matmul(hn, self.f_a), self.f_b), self.dt)
        decay = ttnn.exp(ttnn.multiply(ttnn.sigmoid(ttnn.multiply(g1, self.neg_a)), KDA_LB))
        decay = ttnn.transpose(ttnn.reshape(decay, [1, NH, 1, HD]), -2, -1)              # [1,64,128,1]
        beta = ttnn.reshape(ttnn.sigmoid(ttnn.matmul(hn, self.beta)), [1, NH, 1, 1])
        S = ttnn.multiply(self.S, decay)                                 # per-channel decay of the k rows
        delta = ttnn.multiply(ttnn.subtract(v, ttnn.matmul(kr, S, compute_kernel_config=HI_CK)), beta)
        S = ttnn.add(S, ttnn.matmul(ttnn.transpose(kr, -2, -1), delta, compute_kernel_config=HI_CK))
        o = ttnn.matmul(q, S, compute_kernel_config=HI_CK)
        ttnn.copy(S, self.S)
        on = ttnn.rms_norm(ttnn.reshape(o, [1, 1, NH, HD]), weight=self.o_norm, epsilon=RMS_EPS)
        g2 = ttnn.sigmoid(ttnn.matmul(ttnn.matmul(hn, self.g_a), self.g_b))
        return self.wo(ttnn.multiply(ttnn.reshape(on, [1, 1, 1, NH * HD]), g2))

    def mixer_ref(self, hn):
        r = self.ref
        x = torch.cat([r["wq"] @ hn, r["wk"] @ hn, r["wv"] @ hn])
        win = torch.cat([r["conv0"], x[None]])
        x = torch.nn.functional.silu((win * r["taps"]).sum(0))
        q, k, v = (t.reshape(NH, HD) for t in x.split(NH * HD))
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + L2_EPS) * HD ** -0.5
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + L2_EPS)
        g1 = (r["f_b"] @ (r["f_a"] @ hn) + r["dt"]).reshape(NH, HD) * r["A"][:, None]
        decay = torch.exp(torch.sigmoid(-g1) * KDA_LB)
        beta = torch.sigmoid(r["beta"] @ hn)
        S = r["S0"] * decay[:, :, None]
        delta = (v - torch.einsum("hk,hkv->hv", k, S)) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        o = torch.einsum("hk,hkv->hv", q, S)
        on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + RMS_EPS) * r["o_norm"]
        g2 = torch.sigmoid(r["g_b"] @ (r["g_a"] @ hn)).reshape(NH, HD)
        return r["wo"] @ (on * g2).flatten()


# ---------------------------------------------------------------- MLA layer with the k-pool indexer

class MlaCard:
    def __init__(self, card: Card, g: Gguf, il: int, k: dict, rng: torch.Generator):
        p = f"blk.{il}."
        r = {}
        for site in ("attn", "ffn"):
            r[f"hc_{site}"] = (g(p + f"hc_{site}_fn.weight"), g(p + f"hc_{site}_scale.weight"),
                               g(p + f"hc_{site}_base.weight"))
        self.hc_attn, self.hc_ffn = HcCard(card, *r["hc_attn"], k), HcCard(card, *r["hc_ffn"], k)
        r["attn_norm"], r["ffn_norm"] = g(p + "attn_norm.weight"), g(p + "ffn_norm.weight")
        self.attn_norm, self.ffn_norm = card.norm_w(r["attn_norm"]), card.norm_w(r["ffn_norm"])
        for x in ("q_a", "q_b", "kv_a_mqa", "k_b", "v_b", "output", "q_a_norm", "kv_a_norm"):
            r[x] = g(p + f"attn_{x}.weight")
        for x in ("attn_k", "attn_q_b", "proj", "k_norm"):
            r["i_" + x] = g(p + f"indexer.{x}.weight")
        r["i_k_norm_b"] = g(p + "indexer.k_norm.bias")
        r["i_gate"], r["i_ape"] = g(p + "indexer_compressor_gate.weight"), g(p + "indexer_compressor_ape.weight")
        self.moe = MoeCard(card, g, il, r)
        self.q_a, self.q_b, self.kv_a = card.mm(r["q_a"].T), card.mm(r["q_b"].T), card.mm(r["kv_a_mqa"].T)
        self.q_a_norm, self.kv_a_norm = card.norm_w(r["q_a_norm"]), card.norm_w(r["kv_a_norm"])
        self.k_b = card.put(r["k_b"].transpose(1, 2).reshape(1, NH, HK, KVL), BFP8)       # q_h [256] -> [512]
        self.v_b = card.put(r["v_b"].transpose(1, 2).reshape(1, NH, KVL, HK), BFP8)       # o_h [512] -> [256]
        self.wo = card.mm(r["output"].T)
        self.i_qb, self.i_k, self.i_gate = card.mm(r["i_attn_q_b"].T), card.mm(r["i_attn_k"].T), card.mm(r["i_gate"].T)
        self.i_proj = card.put(r["i_proj"].T / math.sqrt(IHD * IH), BFP8)
        self.i_kn_w, self.i_kn_b = card.norm_w(r["i_k_norm"]), card.norm_w(r["i_k_norm_b"])
        self.ape = card.put(r["i_ape"].reshape(1, 1, KPOOL, IHD))
        n_pool = CTX // KPOOL
        r["lat"] = torch.randn(CTX, KVL, generator=rng)                   # the latent cache (post kv_a_norm)
        r["pooled"] = torch.randn(n_pool, IHD, generator=rng)             # the indexer's pooled keys
        r["tail_k"] = torch.randn(KPOOL - 1, IHD, generator=rng)          # the open pool's keys and gates
        r["tail_g"] = torch.randn(KPOOL - 1, IHD, generator=rng)
        self.lat = card.put(r["lat"].reshape(1, 1, CTX, KVL), BF16, RM)
        self.lat_tile = card.put(r["lat"].reshape(1, 1, CTX, KVL))        # stand-in for the cache write
        self.pooled = card.put(r["pooled"].reshape(1, 1, n_pool, IHD))
        self.tail_k = card.put(r["tail_k"].reshape(1, 1, KPOOL - 1, IHD))
        self.tail_g = card.put(r["tail_g"].reshape(1, 1, KPOOL - 1, IHD))
        self.pool_mask = card.row(torch.zeros(n_pool))
        self.offs = card.row(torch.arange(float(KPOOL)), FP32)
        self.ref = r
        self.sel = None

    def indexer(self, hn, qr):
        """-> token indices [1,1,1,2048] uint32 row-major: the top 512 pools x 4 tokens."""
        iq = ttnn.reshape(self.i_qb(qr), [1, 1, IH, IHD])                # 32 heads as rows
        kq = ttnn.relu(ttnn.matmul(iq, self.pooled, transpose_b=True, compute_kernel_config=HI_CK))
        score = ttnn.add(ttnn.matmul(ttnn.matmul(hn, self.i_proj), kq, compute_kernel_config=HI_CK), self.pool_mask)
        _, top = ttnn.topk(score, k=ITOP // KPOOL, dim=-1)
        col = ttnn.reshape(ttnn.typecast(ttnn.typecast(top, ttnn.uint32), FP32), [1, 1, ITOP // KPOOL, 1])
        tok = ttnn.reshape(ttnn.add(ttnn.multiply(col, float(KPOOL)), self.offs), [1, 1, 1, ITOP])
        # this token's key and pool gate; the pooled key of the pool it closes (every 4th token, done every token here)
        ik = ttnn.layer_norm(self.i_k(hn), weight=self.i_kn_w, bias=self.i_kn_b, epsilon=LN_EPS)
        win_k = ttnn.concat([self.tail_k, ik], dim=2)
        win_g = ttnn.add(ttnn.concat([self.tail_g, self.i_gate(hn)], dim=2), self.ape)
        pooled = ttnn.sum(ttnn.multiply(ttnn.softmax(win_g, dim=2), win_k), dim=2, keepdim=True)
        ttnn.copy(ttnn.slice(win_k, [0, 0, 1, 0], [1, 1, KPOOL, IHD]), self.tail_k)
        self.new_pooled = pooled
        return ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), RM), top

    def mixer(self, hn):
        qr = ttnn.rms_norm(self.q_a(hn), weight=self.q_a_norm, epsilon=RMS_EPS)
        q = ttnn.reshape(self.q_b(qr), [1, NH, 1, HK])
        kv = ttnn.rms_norm(self.kv_a(hn), weight=self.kv_a_norm, epsilon=RMS_EPS)
        ttnn.update_cache(self.lat_tile, ttnn.reshape(kv, [1, 1, 1, KVL]), CTX - 1)
        q_abs = ttnn.matmul(q, self.k_b, compute_kernel_config=MM_CK)  # [1,64,1,512]
        idx, self.sel = self.indexer(hn, qr)
        o = ttnn.transformer.sparse_sdpa(ttnn.to_layout(q_abs, RM), self.lat, idx, KVL,
                                         kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=HK ** -0.5)
        o = ttnn.matmul(ttnn.to_layout(o, TILE), self.v_b, compute_kernel_config=MM_CK)   # [1,64,1,256]
        return self.wo(ttnn.reshape(o, [1, 1, 1, NH * HK]))

    def mixer_ref(self, hn):
        r = self.ref
        rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w  # noqa: E731
        qr = rms(r["q_a"] @ hn, r["q_a_norm"])
        q = (r["q_b"] @ qr).reshape(NH, HK)
        q_abs = torch.einsum("hlk,hk->hl", r["k_b"], q)                  # k_b [64, 512, 256]
        iq = (r["i_attn_q_b"] @ qr).reshape(IH, IHD)
        w = r["i_proj"] @ hn / math.sqrt(IHD * IH)
        score = (torch.relu(iq @ r["pooled"].T) * w[:, None]).sum(0)
        pools = torch.topk(score, ITOP // KPOOL).indices
        tok = (pools[:, None] * KPOOL + torch.arange(KPOOL)).flatten()
        lat = r["lat"][tok]
        p = torch.softmax(q_abs @ lat.T * HK ** -0.5, dim=-1)
        o = torch.einsum("hvl,hl->hv", r["v_b"], p @ lat)                # v_b [64, 256, 512]
        self.ref_pools = pools
        return r["output"] @ o.flatten()


# ---------------------------------------------------------------- one layer step and its reference

def step(L, X):
    a, s = L.hc_attn.mix(X)
    c = L.hc_attn.sinkhorn(a)
    hn = ttnn.rms_norm(L.hc_attn.pre(X, s), weight=L.attn_norm, epsilon=RMS_EPS)
    X = HcCard.post(L.mixer(hn), X, s, c)
    a, s = L.hc_ffn.mix(X)
    c = L.hc_ffn.sinkhorn(a)
    hn = ttnn.rms_norm(L.hc_ffn.pre(X, s), weight=L.ffn_norm, epsilon=RMS_EPS)
    out, idx, w = L.moe(hn)
    return HcCard.post(out, X, s, c), idx, w


def step_ref(L, x):
    r = L.ref
    rms = lambda v, w: v * torch.rsqrt(v.pow(2).mean() + RMS_EPS) * w  # noqa: E731
    h, post, comb = hc_ref(x, *r["hc_attn"])
    attn = L.mixer_ref(rms(h, r["attn_norm"]))
    x = hc_post_ref(attn, x, post, comb)
    h, post, comb = hc_ref(x, *r["hc_ffn"])
    out, idx, w = moe_ref(rms(h, r["ffn_norm"]), r)
    return hc_post_ref(out, x, post, comb), attn, idx, w


def traced_us(dev, fn, reps=LAYERS, tokens=TOKENS) -> float:
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    keep = fn()  # noqa: F841  (the trace writes these buffers; hold them until it is released)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    samples = []
    for i in range(tokens + 2):
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        if i >= 2:
            samples.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(dev, tid)
    return statistics.median(samples)


def check(name, L, card, x0, results):
    X = card.row(x0.flatten())
    Xo, idx, w = step(L, X)
    xr, attn_r, idx_r, w_r = step_ref(L, x0)
    got = ttnn.to_torch(Xo).float().reshape(HC, D)
    ids = set(ttnn.to_torch(idx).flatten().long().tolist())
    res = {"pcc_x_out": pcc(got, xr), "pcc_delta": pcc(got - x0, xr - x0),
           "experts_agree": len(ids & set(idx_r.tolist()))}
    if isinstance(L, MlaCard):
        pools = set(ttnn.to_torch(L.sel).flatten().long().tolist())
        res["pools_agree_of_512"] = len(pools & set(L.ref_pools.tolist()))
    print(f"{name}: " + ", ".join(f"{k} {v:.5f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
    results[name]["check"] = res


def time_layer(name, L, card, x0, results):
    dev = card.dev
    X = card.row(x0.flatten())
    a, s = L.hc_attn.mix(X)
    c = L.hc_attn.sinkhorn(a)
    hn = ttnn.rms_norm(L.hc_attn.pre(X, s), weight=L.attn_norm, epsilon=RMS_EPS)
    out = L.mixer(hn)
    blocks = {
        "layer step": lambda: step(L, X),
        "mHC pre (mix, weights, sum, norm) x1": lambda: ttnn.rms_norm(
            L.hc_attn.pre(X, L.hc_attn.mix(X)[1]), weight=L.attn_norm, epsilon=RMS_EPS),
        "Sinkhorn (20 steps) x1": lambda: L.hc_attn.sinkhorn(a),
        "mHC post x1": lambda: HcCard.post(out, X, s, c),
        "mixer": lambda: L.mixer(hn),
        "router + top-8 + shared expert": lambda: L.moe(hn),
    }
    if isinstance(L, KdaCard):
        blocks["  q|k|v projection (4096 x 24576)"] = lambda: L.qkv(hn)
        x_o = card.row(torch.randn(NH * HD))
        blocks["  o projection (8192 x 4096)"] = lambda: L.wo(x_o)
    else:
        qr = ttnn.rms_norm(L.q_a(hn), weight=L.q_a_norm, epsilon=RMS_EPS)
        blocks["  indexer"] = lambda: L.indexer(hn, qr)
        q_abs = ttnn.matmul(ttnn.reshape(L.q_b(qr), [1, NH, 1, HK]), L.k_b)
        idx, _ = L.indexer(hn, qr)
        q_rm = ttnn.to_layout(q_abs, RM)
        blocks["  sparse attention (2,048 of the latents)"] = lambda: ttnn.transformer.sparse_sdpa(
            q_rm, L.lat, idx, KVL, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=HK ** -0.5)
        x_o = card.row(torch.randn(NH * HK))
        blocks["  o projection (16384 x 4096)"] = lambda: L.wo(x_o)
    blocks["  shared expert gate|up (4096 x 4096)"] = lambda: L.moe.sh_gu(hn)
    times = {}
    for k, fn in blocks.items():
        times[k] = traced_us(dev, fn)
        print(f"  {times[k]:8.1f} us  {name}: {k}")
    results[name]["us"] = times


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    g = Gguf(MODEL)
    rng = torch.Generator().manual_seed(0)
    k = hc_consts()
    results = {"ctx": CTX, "date": time.strftime("%Y-%m-%d %H:%M"), "kda": {}, "mla": {}, "other": {}}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=256 << 20)
    try:
        card = Card(dev)
        kda = KdaCard(card, g, KDA_LAYER, k, rng)
        kda_bytes, g.bytes = g.bytes, 0
        mla = MlaCard(card, g, MLA_LAYER, k, rng)
        mla_bytes = g.bytes
        results["kda"]["gguf_bytes"], results["mla"]["gguf_bytes"] = kda_bytes, mla_bytes
        print(f"weights loaded in {time.time() - t0:.0f} s: KDA layer {kda_bytes / 1e6:.1f} MB, "
              f"MLA layer {mla_bytes / 1e6:.1f} MB as stored (non-expert)")
        x0 = torch.randn(HC, D, generator=rng)
        check("kda", kda, card, x0, results)
        check("mla", mla, card, x0, results)
        time_layer("kda", kda, card, x0, results)
        time_layer("mla", mla, card, x0, results)
        # the rest of a token: one small op (the per-op floor), the 3 leading dense FFNs, the LM head
        x = card.row(torch.randn(HC * D))
        results["other"]["one add [1 x 16384]"] = traced_us(dev, lambda: ttnn.add(x, x))
        hn = card.row(torch.randn(D))
        gate, up, down = (g(f"blk.{DENSE_LAYER}.ffn_{n}.weight") for n in ("gate", "up", "down"))
        gu, dn = card.mm(torch.cat([gate.T, up.T], dim=1)), card.mm(down.T)
        del gate, up, down

        def dense():
            y = gu(hn)
            return dn(ttnn.multiply(ttnn.silu(ttnn.slice(y, [0, 0, 0, 0], [1, 1, 1, 12288])),
                                    ttnn.slice(y, [0, 0, 0, 12288], [1, 1, 1, 24576])))
        results["other"]["dense FFN (blk.2, 3 x 4096 x 12288)"] = traced_us(dev, dense)
        head = [card.mm(torch.randn(D, 7744) * 0.02) for _ in range(20)]   # 154,880 = 20 x 7,744; random weights
        results["other"]["LM head (4096 x 154880, 20 matmuls)"] = traced_us(dev, lambda: [h(hn) for h in head])
        for kk, v in results["other"].items():
            print(f"  {v:8.1f} us  {kk}")
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g2_ctx{CTX}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
