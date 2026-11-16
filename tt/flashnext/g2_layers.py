"""G2 for Qwen3.8-Flash-Next UD-IQ4_XS: one gated DeltaNet and one gated full-attention (QSA) layer step on the p150a.

Decode-shaped: one token through a whole layer except its routed experts. Per layer, llama.cpp's qwen4exp graph
(src/models/qwen4exp.cpp): the hyper-connection mix before the mixer (4 streams: grouped RMSNorm, the 10240 -> 320
-> 10240 gate, the mean over streams, the 4 inject weights), the mixer, the combine, again for the FFN half, the
router (softmax, top-10 of 512), the shared expert with its sigmoid gate and the add of the experts' sum (a stand-in
tensor: the card's and the CPU's experts are priced separately, from the card's measured units and Strata's CPU lane).

  DeltaNet (blk.0): qkv|z|alpha|beta in one projection, the causal conv (kernel 4), L2 norms, the 16 key heads
    repeated to the 48 value heads, qwen36's fused recurrent gated-delta-rule op (one core per head, fp32 state),
    the gated RMSNorm (sigmoid gate: qwen4exp's one difference from Qwen3.5) and ssm_out.
  Full attention (blk.3): q|gate, k, v and the QSA indexer's q and k in one projection, q/k norms, RoPE on 64 of
    256 dims (a matmul by the position's rotation here), the indexer (4 heads over the pooled keys of blocks of 4
    cells, top-512 blocks = 2,048 cells + the 3-cell tail), sparse_sdpa over the selected cells of an 8K cache
    (both KV heads in one call: each row is [v0 | v1 | k0 | k1], each query zero outside its own k), the sigmoid
    output gate and wo. The cache and pooled-key writes are done every token (a pool closes every 4th).

Real weights from the HDD GGUF (read-only; nothing staged), dequantized on the host and uploaded as bfp8_b (the
Q8_0 projections) or bf16 (the F32 router, as GLM's G2). The big projections are DRAM-sharded decode matmuls
(tt_transformers' recipe). Each step is checked once against a float32 torch reference on the same weights and
synthetic state, then traced and replayed 48 times per sync; medians of 20.

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/flashnext/g2_layers.py
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

MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models", "qwen3.8-flash-next-ud-iq4xs", "UD-IQ4_XS")   # engine.py --model-dir sets it
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GDN_LAYER, ATTN_LAYER, PLE_LAYER = 0, 3, 1
D, HC, LR = 2560, 4, 320                  # hidden, hyper-connection streams, their gate's low rank
NK, NV, HK = 16, 48, 128                  # DeltaNet key heads, value heads, head dim
QKV, ZD = 2 * NK * HK + NV * HK, NV * HK  # 10240 conv channels, 6144 gate
NH, NKV, HD, NROT = 24, 2, 256, 64        # attention heads, KV heads, head dim, rotated dims
IH, IHD, KPOOL, ITOP = 4, 128, 4, 2048    # indexer heads, head dim, pool size, cells kept
NX, TOPK, FF = 512, 10, 640               # experts, used, expert / shared FFN
EPS, ROPE_BASE = 1e-6, 1e7
CTX = int(os.environ.get("G2_CTX", 8192))
SEL = 2176                                # 2,048 + 3 tail cells, padded to sparse_sdpa's 128-cell chunks
LAYERS, TOKENS = 48, 20
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
        self.n, self.n_pad = n, n_pad
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

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        y = ttnn.matmul(ttnn.to_memory_config(x, self.in_mc), self.w, program_config=self.pc,
                        memory_config=self.out_mc, compute_kernel_config=MM_CK)
        y = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
        return y if self.n == self.n_pad else ttnn.slice(y, [0, 0, 0, 0], [1, 1, 1, self.n])


def cut(t, a, b):
    """Columns [a, b) of a [1,1,1,n] row."""
    return ttnn.slice(t, [0, 0, 0, a], [1, 1, 1, b])


def rope_matrix(pos: int, dim: int) -> torch.Tensor:
    """x @ M = NEOX RoPE of the first NROT dims at pos (text tokens: IMRoPE's sections all see the same position)."""
    half = NROT // 2
    theta = pos * ROPE_BASE ** (-torch.arange(half, dtype=torch.float64) * 2 / NROT)
    c, s = torch.cos(theta).float(), torch.sin(theta).float()
    m = torch.eye(dim)
    i = torch.arange(half)
    m[i, i], m[i + half, i + half] = c, c
    m[i + half, i], m[i, i + half] = -s, s     # y_i = x_i c - x_{i+h} s, y_{i+h} = x_i s + x_{i+h} c
    return m


def rms(x: torch.Tensor, w=None) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if w is None else y * w


def l2n(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + EPS)


# ---------------------------------------------------------------- hyper-connections (both halves of every layer)

class HcCard:
    """One HC site. The 4 streams travel flat: X [1,1,1,4*2560] bf16."""

    def __init__(self, card: Card, g: Gguf, p: str, ref: dict):
        norm, down, up, inject = (g(p + x) for x in ("norm.weight", "down.weight", "up.weight", "inject.weight"))
        ref[p] = (norm, down, up, inject)                                # norm [10240] (1 + w folded in)
        self.norm = card.row(norm)
        self.down_inj = card.mm(torch.cat([down.T, inject.T], dim=1))    # [10240, 320 + 4]
        self.up = card.mm(up.T)                                          # [320, 10240]

    def mix(self, X):
        """-> (the mixed stream [1,1,1,2560], the 4 inject weights [1,1,1,4])."""
        xn = ttnn.reshape(ttnn.rms_norm(ttnn.reshape(X, [1, 1, HC, D]), epsilon=EPS), [1, 1, 1, HC * D])
        xn = ttnn.multiply(xn, self.norm)
        lo = self.down_inj(xn)
        gate = self.up(ttnn.silu(ttnn.multiply(cut(lo, 0, LR), 1.0 / HC)))
        gated = ttnn.reshape(ttnn.multiply(xn, ttnn.sigmoid(gate)), [1, 1, HC, D])
        return ttnn.multiply(ttnn.sum(gated, dim=-2, keepdim=True), 1.0 / HC), cut(lo, LR, LR + HC)

    @staticmethod
    def combine(X, out, inj):
        w = ttnn.reshape(ttnn.multiply(ttnn.sigmoid(ttnn.multiply(inj, 1.0 / HC)), 2.0), [1, 1, HC, 1])
        return ttnn.add(X, ttnn.reshape(ttnn.multiply(w, out), [1, 1, 1, HC * D]))


def hc_mix_ref(x, norm, down, up, inject):
    xn = (rms(x) * norm.reshape(HC, D)).flatten()                        # x [4, 2560]
    gate = up @ torch.nn.functional.silu(down @ xn / HC)
    return (xn * torch.sigmoid(gate)).reshape(HC, D).mean(0), inject @ xn


def hc_combine_ref(x, out, inj):
    return x + 2 * torch.sigmoid(inj / HC)[:, None] * out[None, :]


# ---------------------------------------------------------------- the FFN half (router + shared expert)

class MoeCard:
    def __init__(self, card: Card, g: Gguf, il: int, ref: dict):
        p = f"blk.{il}."
        router, sh_gate = g(p + "ffn_gate_inp.weight"), g(p + "ffn_gate_inp_shexp.weight")   # [512, 2560], [2560]
        gate, up, down = (g(p + f"ffn_{x}_shexp.weight") for x in ("gate", "up", "down"))   # [640,2560] x2, [2560,640]
        ref.update(router=router, sh_g=sh_gate, sh_gate=gate, sh_up=up, sh_down=down)
        self.router = card.mm(torch.cat([router.T, sh_gate[:, None]], dim=1), BF16)        # 512 logits + the gate
        self.sh_gu = card.mm(torch.cat([gate.T, up.T], dim=1))
        self.sh_down = card.mm(down.T)
        self.routed = card.row(torch.zeros(D))                           # the experts' sum, from the card + CPU

    def __call__(self, hn):
        r = self.router(hn)
        probs = ttnn.softmax(cut(r, 0, NX), dim=-1)
        vals, idx = ttnn.topk(probs, k=TOPK, dim=-1)
        w = ttnn.divide(vals, ttnn.sum(vals, dim=-1, keepdim=True))      # the 10 router weights
        gu = self.sh_gu(hn)
        sh = self.sh_down(ttnn.multiply(ttnn.silu(cut(gu, 0, FF)), cut(gu, FF, 2 * FF)))
        sh = ttnn.multiply(sh, ttnn.sigmoid(cut(r, NX, NX + 1)))
        return ttnn.add(sh, self.routed), idx, w


def moe_ref(hn, r):
    probs = torch.softmax(r["router"] @ hn, -1)
    vals, idx = torch.topk(probs, TOPK)
    sh = r["sh_down"] @ (torch.nn.functional.silu(r["sh_gate"] @ hn) * (r["sh_up"] @ hn))
    return sh * torch.sigmoid(r["sh_g"] @ hn), idx, vals / vals.sum()


# ---------------------------------------------------------------- gated DeltaNet layer

class GdnCard:
    def __init__(self, card: Card, g: Gguf, il: int, rng: torch.Generator):
        p = f"blk.{il}."
        r = {}
        self.hc_attn, self.hc_ffn = HcCard(card, g, p + "hc_attn_", r), HcCard(card, g, p + "hc_ffn_", r)
        self.hc_keys = (p + "hc_attn_", p + "hc_ffn_")
        r["qkv"], r["z"] = g(p + "attn_qkv.weight"), g(p + "attn_gate.weight")             # [10240|6144, 2560]
        r["alpha"], r["beta"] = g(p + "ssm_alpha.weight"), g(p + "ssm_beta.weight")        # [48, 2560]
        r["taps"] = g(p + "ssm_conv1d.weight").T                         # [4, 10240], row 0 = oldest
        r["A"], r["dt"], r["o_norm"] = g(p + "ssm_a"), g(p + "ssm_dt.bias"), g(p + "ssm_norm.weight")
        r["wo"] = g(p + "ssm_out.weight")                                # [2560, 6144]
        self.moe = MoeCard(card, g, il, r)
        self.proj = card.mm(torch.cat([r["qkv"].T, r["z"].T, r["alpha"].T, r["beta"].T], dim=1))   # n = 16480
        self.taps = card.put(r["taps"].reshape(1, 1, 4, QKV))
        self.neg_a, self.dt = card.row(r["A"]), card.row(r["dt"])        # ssm_a holds -exp(A_log)
        self.o_norm = card.norm_w(r["o_norm"])
        self.wo = card.mm(r["wo"].T)
        r["S0"] = torch.randn(NV, HK, HK, generator=rng) * 0.05          # [head, k, v]
        r["conv0"] = torch.randn(3, QKV, generator=rng) * 0.5            # x_{t-3}, x_{t-2}, x_{t-1}
        self.S = card.put(r["S0"].reshape(1, NV, HK, HK), FP32)
        self.conv = card.put(r["conv0"].reshape(1, 1, 3, QKV))
        self.ref = r

    def mixer(self, hn):
        y = self.proj(hn)
        win = ttnn.concat([self.conv, ttnn.reshape(cut(y, 0, QKV), [1, 1, 1, QKV])], dim=2)   # [1,1,4,10240]
        ttnn.copy(ttnn.slice(win, [0, 0, 1, 0], [1, 1, 4, QKV]), self.conv)
        x = ttnn.silu(ttnn.sum(ttnn.multiply(win, self.taps), dim=2, keepdim=True))
        l2 = lambda t: ttnn.multiply(ttnn.rms_norm(ttnn.reshape(t, [1, 1, NK, HK]), epsilon=EPS / HK),  # noqa: E731
                                     HK ** -0.5)
        q, k = l2(cut(x, 0, NK * HK)), l2(cut(x, NK * HK, 2 * NK * HK))
        q = ttnn.typecast(ttnn.concat([q, q, q], dim=2), FP32)           # v head h reads k head h % 16 (ggml_repeat)
        k = ttnn.typecast(ttnn.concat([k, k, k], dim=2), FP32)
        v = ttnn.typecast(ttnn.reshape(cut(x, 2 * NK * HK, QKV), [1, 1, NV, HK]), FP32)
        a = cut(y, QKV + ZD, QKV + ZD + NV)
        gdec = ttnn.reshape(ttnn.typecast(ttnn.multiply(ttnn.softplus(ttnn.add(a, self.dt)), self.neg_a), FP32),
                            [1, 1, NV])
        beta = ttnn.reshape(ttnn.typecast(ttnn.sigmoid(cut(y, QKV + ZD + NV, QKV + ZD + 2 * NV)), FP32), [1, 1, NV])
        o, S = ttnn.transformer.fused_recurrent_gated_delta_rule(q, k, v, gdec, beta, scale=HK ** -0.5,
                                                                 initial_state=self.S, output_final_state=True)
        ttnn.copy(S, self.S)
        on = ttnn.rms_norm(ttnn.typecast(ttnn.reshape(o, [1, 1, NV, HK]), BF16), weight=self.o_norm, epsilon=EPS)
        z = ttnn.reshape(cut(y, QKV, QKV + ZD), [1, 1, NV, HK])
        return self.wo(ttnn.reshape(ttnn.multiply(on, ttnn.sigmoid(z)), [1, 1, 1, ZD]))

    def mixer_ref(self, hn):
        r = self.ref
        win = torch.cat([r["conv0"], (r["qkv"] @ hn)[None]])
        x = torch.nn.functional.silu((win * r["taps"]).sum(0))
        q = l2n(x[:NK * HK].reshape(NK, HK)).repeat(3, 1) * HK ** -0.5
        k = l2n(x[NK * HK:2 * NK * HK].reshape(NK, HK)).repeat(3, 1)
        v = x[2 * NK * HK:].reshape(NV, HK)
        decay = torch.exp(torch.nn.functional.softplus(r["alpha"] @ hn + r["dt"]) * r["A"])
        beta = torch.sigmoid(r["beta"] @ hn)
        S = r["S0"] * decay[:, None, None]
        delta = (v - torch.einsum("hk,hkv->hv", k, S)) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        o = torch.einsum("hk,hkv->hv", q, S)
        z = (r["z"] @ hn).reshape(NV, HK)
        return r["wo"] @ (rms(o, r["o_norm"]) * torch.sigmoid(z)).flatten()


# ---------------------------------------------------------------- gated full attention with the QSA indexer

class AttnCard:
    def __init__(self, card: Card, g: Gguf, il: int, rng: torch.Generator):
        p = f"blk.{il}."
        r = {}
        self.hc_attn, self.hc_ffn = HcCard(card, g, p + "hc_attn_", r), HcCard(card, g, p + "hc_ffn_", r)
        self.hc_keys = (p + "hc_attn_", p + "hc_ffn_")
        for x in ("q", "k", "v", "output", "q_norm", "k_norm"):
            r[x] = g(p + f"attn_{x}.weight")                             # q [12288 = 24 x (q|gate), 2560]
        for x in ("q_proj", "k_proj", "q_norm", "k_norm"):
            r["i_" + x] = g(p + f"indexer.{x}.weight")
        self.moe = MoeCard(card, g, il, r)
        self.pos = CTX - 1
        self.n_pool = CTX // KPOOL
        r["rope"], r["rope_i"] = rope_matrix(self.pos, HD), rope_matrix(self.pos, IHD)
        r["rope_pool"] = rope_matrix(self.pos - KPOOL + 1, IHD)          # a pooled key turns at its first member
        w = torch.cat([r["q"].T, r["k"].T, r["v"].T, r["i_q_proj"].T, r["i_k_proj"].T], dim=1)   # n = 13952
        self.proj = card.mm(w)
        self.q_norm, self.k_norm = card.norm_w(r["q_norm"]), card.norm_w(r["k_norm"])
        self.iq_norm, self.ik_norm = card.norm_w(r["i_q_norm"]), card.norm_w(r["i_k_norm"])
        self.rope, self.rope_i, self.rope_pool = (card.put(r[x]) for x in ("rope", "rope_i", "rope_pool"))
        self.wo = card.mm(r["output"].T)
        # the KV cache, one row per cell: [v0 | v1 | k0 | k1]; queries of KV head g are zero outside k_g
        r["k_cache"] = torch.randn(CTX, NKV, HD, generator=rng)
        r["v_cache"] = torch.randn(CTX, NKV, HD, generator=rng)
        r["pooled"] = rms(torch.randn(self.n_pool, IHD, generator=rng))  # normed, rotated pooled keys
        r["tail_k"] = torch.randn(KPOOL - 1, IHD, generator=rng)         # raw keys of the open block
        rows = torch.cat([r["v_cache"].reshape(CTX, -1), r["k_cache"].reshape(CTX, -1)], dim=1)
        self.kv = card.put(rows.reshape(1, 1, CTX, 4 * HD), BF16, RM)
        self.kv_tile = card.put(rows.reshape(CTX, 4, HD).transpose(0, 1).reshape(1, 4, CTX, HD))   # stand-in write
        self.pooled = card.put(r["pooled"].reshape(1, 1, self.n_pool, IHD))
        self.tail_k = card.put(r["tail_k"].reshape(1, 1, KPOOL - 1, IHD))
        self.pool_mask = card.row(torch.zeros(self.n_pool))
        self.offs = card.row(torch.arange(float(KPOOL)), FP32)
        tail = torch.full((SEL - ITOP,), 0xFFFFFFFF, dtype=torch.int64)
        tail[:KPOOL - 1] = torch.arange(CTX - KPOOL + 1, CTX)            # the open block's 3 cells, this one last
        self.tail_idx = card.put(torch.from_numpy(tail.numpy().astype(np.uint32).view(np.int32)).reshape(1, 1, 1, -1),
                                 ttnn.uint32, RM)
        hm = torch.zeros(32, 4 * HD)                                     # head h's [0 | 0 | k0 | k1] placement
        for h in range(NH):
            hm[h, 2 * HD + (h // (NH // NKV)) * HD:][:HD] = 1
        self.q_place = card.put(hm.reshape(1, 1, 32, 4 * HD))
        vm = torch.zeros(32, 2 * HD)                                     # and its own v out of [v0 | v1]
        for h in range(NH):
            vm[h, (h // (NH // NKV)) * HD:][:HD] = 1
        self.v_pick = card.put(vm.reshape(1, 1, 32, 2 * HD))
        self.ref = r
        self.sel = None

    def indexer(self, iq, ik):
        """-> cell indices [1,1,1,2176] uint32 row-major: the top 512 blocks x 4 cells, the tail, the padding."""
        iq = ttnn.matmul(ttnn.rms_norm(ttnn.reshape(iq, [1, 1, IH, IHD]), weight=self.iq_norm, epsilon=EPS),
                         self.rope_i, compute_kernel_config=HI_CK)
        kq = ttnn.relu(ttnn.matmul(iq, self.pooled, transpose_b=True, compute_kernel_config=HI_CK))
        score = ttnn.add(ttnn.multiply(ttnn.sum(kq, dim=-2, keepdim=True), IHD ** -0.5), self.pool_mask)
        _, top = ttnn.topk(score, k=ITOP // KPOOL, dim=-1)
        col = ttnn.reshape(ttnn.typecast(ttnn.typecast(top, ttnn.uint32), FP32), [1, 1, ITOP // KPOOL, 1])
        cell = ttnn.reshape(ttnn.add(ttnn.multiply(col, float(KPOOL)), self.offs), [1, 1, 1, ITOP])
        idx = ttnn.concat([ttnn.to_layout(ttnn.typecast(cell, ttnn.uint32), RM), self.tail_idx], dim=-1)
        # this token's raw key joins the open block; the block's pooled key (mean, norm, RoPE), every token here
        win = ttnn.concat([self.tail_k, ttnn.reshape(ik, [1, 1, 1, IHD])], dim=2)
        ttnn.copy(ttnn.slice(win, [0, 0, 1, 0], [1, 1, KPOOL, IHD]), self.tail_k)
        pk = ttnn.rms_norm(ttnn.multiply(ttnn.sum(win, dim=2, keepdim=True), 1.0 / KPOOL), weight=self.ik_norm,
                           epsilon=EPS)
        ttnn.update_cache(self.pooled, ttnn.matmul(pk, self.rope_pool, compute_kernel_config=HI_CK),
                          self.n_pool - 1)
        return idx, top

    def mixer(self, hn):
        y = self.proj(hn)
        qg = ttnn.reshape(cut(y, 0, 2 * NH * HD), [1, 1, NH, 2 * HD])
        q = ttnn.rms_norm(ttnn.slice(qg, [0, 0, 0, 0], [1, 1, NH, HD]), weight=self.q_norm, epsilon=EPS)
        gate = ttnn.slice(qg, [0, 0, 0, HD], [1, 1, NH, 2 * HD])
        o = 2 * NH * HD
        k = ttnn.rms_norm(ttnn.reshape(cut(y, o, o + NKV * HD), [1, 1, NKV, HD]), weight=self.k_norm, epsilon=EPS)
        v = cut(y, o + NKV * HD, o + 2 * NKV * HD)
        q = ttnn.matmul(q, self.rope, compute_kernel_config=HI_CK)
        k = ttnn.matmul(k, self.rope, compute_kernel_config=HI_CK)
        ttnn.update_cache(self.kv_tile, ttnn.reshape(ttnn.concat([ttnn.reshape(v, [1, 1, NKV, HD]), k], dim=2),
                                                     [1, 2 * NKV, 1, HD]), CTX - 1)
        i0 = o + 2 * NKV * HD
        idx, self.sel = self.indexer(cut(y, i0, i0 + IH * IHD), cut(y, i0 + IH * IHD, i0 + IH * IHD + IHD))
        qp = ttnn.multiply(ttnn.repeat(ttnn.pad(q, [(0, 0), (0, 0), (0, 32 - NH), (0, 0)], 0.0), [1, 1, 1, 4]),
                           self.q_place)                                 # [1,1,32,1024]
        a = ttnn.transformer.sparse_sdpa(ttnn.to_layout(ttnn.reshape(qp, [1, 32, 1, 4 * HD]), RM), self.kv, idx,
                                         2 * HD, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=HD ** -0.5)
        a = ttnn.multiply(ttnn.reshape(ttnn.to_layout(a, TILE), [1, 1, 32, 2 * HD]), self.v_pick)
        a = ttnn.add(ttnn.slice(a, [0, 0, 0, 0], [1, 1, NH, HD]), ttnn.slice(a, [0, 0, 0, HD], [1, 1, NH, 2 * HD]))
        return self.wo(ttnn.reshape(ttnn.multiply(a, ttnn.sigmoid(gate)), [1, 1, 1, NH * HD]))

    def mixer_ref(self, hn):
        r = self.ref
        qg = (r["q"] @ hn).reshape(NH, 2 * HD)
        q = rms(qg[:, :HD], r["q_norm"]) @ r["rope"]
        iq = rms((r["i_q_proj"] @ hn).reshape(IH, IHD), r["i_q_norm"]) @ r["rope_i"]
        score = torch.relu(iq @ r["pooled"].T).sum(0) * IHD ** -0.5
        pools = torch.topk(score, ITOP // KPOOL).indices
        cells = torch.cat([(pools[:, None] * KPOOL + torch.arange(KPOOL)).flatten(),
                           torch.arange(CTX - KPOOL + 1, CTX)])
        kc, vc = r["k_cache"], r["v_cache"]                              # the card's cache write is a stand-in
        out = []
        for h in range(NH):
            gi = h // (NH // NKV)
            p = torch.softmax(kc[cells, gi] @ q[h] * HD ** -0.5, -1)
            out.append(p @ vc[cells, gi])
        self.ref_pools = pools
        return r["output"] @ (torch.stack(out) * torch.sigmoid(qg[:, HD:])).flatten()


# ---------------------------------------------------------------- one layer step and its reference

def step(L, X):
    hn, inj = L.hc_attn.mix(X)
    X = HcCard.combine(X, L.mixer(hn), inj)
    hn, inj = L.hc_ffn.mix(X)
    out, idx, w = L.moe(hn)
    return HcCard.combine(X, out, inj), idx, w


def step_ref(L, x):
    r = L.ref
    ka, kf = L.hc_keys
    hn, inj = hc_mix_ref(x, *r[ka])
    x = hc_combine_ref(x, L.mixer_ref(hn), inj)
    hn, inj = hc_mix_ref(x, *r[kf])
    out, idx, w = moe_ref(hn, r)
    return hc_combine_ref(x, out, inj), idx, w


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
    """One step from x0 on fresh copies of the state; the state is restored afterwards, so timing starts the same."""
    X = card.row(x0.flatten())
    Xo, idx, w = step(L, X)
    xr, idx_r, w_r = step_ref(L, x0)
    got = ttnn.to_torch(Xo).float().reshape(HC, D)
    ids = ttnn.to_torch(idx).flatten().long().tolist()
    res = {"pcc_x_out": pcc(got, xr), "pcc_delta": pcc(got - x0, xr - x0),
           "experts_agree": len(set(ids) & set(idx_r.tolist()))}
    if isinstance(L, AttnCard):
        blocks = set(ttnn.to_torch(L.sel).flatten().long().tolist())
        res["blocks_agree_of_512"] = len(blocks & set(L.ref_pools.tolist()))
    print(f"{name}: " + ", ".join(f"{k} {v:.5f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
    results[name]["check"] = res


def time_layer(name, L, card, x0, results):
    dev = card.dev
    X = card.row(x0.flatten())
    hn, inj = L.hc_attn.mix(X)
    out = L.mixer(hn)
    blocks = {
        "layer step": lambda: step(L, X),
        "HC mix x1": lambda: L.hc_attn.mix(X),
        "HC combine x1": lambda: HcCard.combine(X, out, inj),
        "mixer": lambda: L.mixer(hn),
        "router + top-10 + shared expert": lambda: L.moe(hn),
        "  mixer's input projection": lambda: L.proj(hn),
    }
    if isinstance(L, GdnCard):
        z = card.put(torch.randn(1, 1, NV, HK) * 0.1, FP32)
        zg = ttnn.reshape(card.put(torch.zeros(1, 1, 1, NV), FP32), [1, 1, NV])
        blocks["  fused recurrent delta rule (48 heads)"] = lambda: ttnn.transformer.fused_recurrent_gated_delta_rule(
            z, z, z, zg, zg, scale=HK ** -0.5, initial_state=L.S, output_final_state=True)
        x_o = card.row(torch.randn(ZD))
        blocks["  ssm_out (6144 x 2560)"] = lambda: L.wo(x_o)
    else:
        y = L.proj(hn)
        i0 = 2 * NH * HD + 2 * NKV * HD
        iq, ik = cut(y, i0, i0 + IH * IHD), cut(y, i0 + IH * IHD, i0 + IH * IHD + IHD)
        blocks["  indexer (score 2,048 blocks, top-512, pool)"] = lambda: L.indexer(iq, ik)
        idx, _ = L.indexer(iq, ik)
        qp = ttnn.to_layout(card.put(torch.randn(1, 32, 1, 4 * HD)), RM)
        blocks["  sparse attention (2,051 of 8K cells)"] = lambda: ttnn.transformer.sparse_sdpa(
            qp, L.kv, idx, 2 * HD, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=HD ** -0.5)
        x_o = card.row(torch.randn(NH * HD))
        blocks["  wo (6144 x 2560)"] = lambda: L.wo(x_o)
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
    results = {"ctx": CTX, "date": time.strftime("%Y-%m-%d %H:%M"), "gdn": {}, "attn": {}, "other": {}}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=256 << 20)
    try:
        card = Card(dev)
        gdn = GdnCard(card, g, GDN_LAYER, rng)
        gdn_bytes, g.bytes = g.bytes, 0
        attn = AttnCard(card, g, ATTN_LAYER, rng)
        attn_bytes = g.bytes
        results["gdn"]["gguf_bytes"], results["attn"]["gguf_bytes"] = gdn_bytes, attn_bytes
        print(f"weights loaded in {time.time() - t0:.0f} s: DeltaNet layer {gdn_bytes / 1e6:.1f} MB, "
              f"attention layer {attn_bytes / 1e6:.1f} MB as stored (non-expert)")
        x0 = torch.randn(HC, D, generator=rng)
        for name, L in (("gdn", gdn), ("attn", attn)):
            S0 = [ttnn.clone(t) for t in ((L.S, L.conv) if name == "gdn" else (L.tail_k, L.pooled))]
            check(name, L, card, x0, results)
            for t, s in zip((L.S, L.conv) if name == "gdn" else (L.tail_k, L.pooled), S0):
                ttnn.copy(s, t)
        time_layer("gdn", gdn, card, x0, results)
        time_layer("attn", attn, card, x0, results)
        # the rest of a token: one small op (the floor), PLE's two projections (blk.1 only), the final HC mixer and
        # the LM head (2560 x 248320, as 25 matmuls of random weights: the vocabulary padded to 256,000)
        x = card.row(torch.randn(HC * D))
        results["other"]["one add [1 x 10240]"] = traced_us(dev, lambda: ttnn.add(x, x))
        hn = card.row(torch.randn(D))
        ple = card.mm(torch.cat([g(f"blk.{PLE_LAYER}.ple_key.weight").T, g(f"blk.{PLE_LAYER}.ple_value.weight").T],
                                dim=1))
        results["other"]["PLE key|value (2560 x 12800, blk.1 only)"] = traced_us(dev, lambda: ple(hn))
        head = [card.mm(torch.randn(D, 10240) * 0.02) for _ in range(25)]
        results["other"]["LM head (2560 x 248320, 25 matmuls)"] = traced_us(dev, lambda: [h(hn) for h in head])
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
