"""GLM's prefill on T rows, layer by layer, against llama.cpp's own prefill tensors
($TT_MOEOFFLOAD_DATA/glm_ref: the 89-token prompt in one ubatch, the streams after every layer l_out-<il>, hc_init, ffn_out).

Shared, not ported: Tenstorrent's prefill modules for this family in the container's own tt-metal commit
(third_party/tt-metal/models/demos/deepseek_v3_d_p): TtMHCWrap (DeepSeek-V4's mHC: GLM's formula, with GLM's norm eps
1e-5), ttKDA (Kimi's chunked KDA; GLM's bounded gate is Kimi-K3's: lb sigmoid(exp(A_log) (g + dt_b)), A_log =
log(-ssm_a)). Rows past the prompt are padding (actual_end); the streams [1, 1, T, 4 x 4096] fp32, stream-major.

Modes (python g6_prefill.py <mode>):
  layer0   blk.0 (KDA + the dense FFN) on the prompt's rows from hc_init, against ffn_out-0 and l_out-0
  layers   blk.0 .. GP_LAYERS - 1 (default 5), each from llama.cpp's input and chained from ours (the CPU lane loaded)
  handoff  the prefill handed to the speculative decoder, against llama.cpp's logits (GP_REF, GP_FROM, GP_N, GP_PROF)
  experts  the card's routed experts (g6_experts, 4.4) against the lane's on random rows (GP_XSCALE 8: past the clamp)
  split    the prefill's time by prompt size and card share (split_groups' table)
The routed experts (4.4, 2026-11-13): GP_EXPERTS=card (default) takes the card's share by prompt size where card memory
leaves room for the experts' two stream slots (card_experts); beside the decoder that needs GP_WDT=bfp8 (bf16 prefill
weights leave 5.3 GB: the lane alone then).

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((64 << 30)) --engine tt/glm/g6_prefill.py -- layer0
"""
from __future__ import annotations

import copy
import json
import math
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys
import time

import numpy as np
import torch

# the card's experts stream from pinned host memory (g7_experts.CardExperts): tt-metal's pinned-memory cache holds them
# all (34 layers x 3.75 GB, g6_experts)
os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", str(160 << 30))
import ttnn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g2_layers import D, HC, HD, HK, IH, IHD, KDA_LB, KPOOL, KVL, LN_EPS, MODEL, NH, OUT, RMS_EPS, Gguf, pcc  # noqa: E402
from fused.host_rows import HostOut, HostRows  # noqa: E402
from fused.qsa import qsa_mask  # noqa: E402
from tune import tuned  # noqa: E402
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig  # noqa: E402
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap  # noqa: E402

REF = os.path.expanduser(os.environ.get("GP_REF", os.path.join(DATA, "glm_ref")))
BF16, FP32, TILE, RM = ttnn.bfloat16, ttnn.float32, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
N_DENSE = 3
SPARSE_FROM = int(os.environ.get("GP_SPARSE_FROM", 2048))   # chunks of more rows: the indexer's selection
SPARSE_TOPK = 2176                    # >= 512 x 4 + 3, a multiple of sparse_sdpa's k chunk (128); g7_prefill's
SPARSE_CHECK = os.environ.get("GP_SPARSE_CHECK", "0") == "1"
LANE_M = 2048
# rows a MoE layer's routed-experts call (whole chunks; P1, bitwise): 2,048 / 4,096 / 8,192 / 16,384 at 29,514 tokens
# 66.3 / 60.6 / 56.3 / 54.5 s (least free card memory 7.2 -> 5.3 GB); a new group size compiles its shapes once (+18 s)
MOE_ROWS = int(os.environ.get("GP_MOE_ROWS", 16384))
# the routed experts: "card" (g6_experts' card form, split with the lane by prompt size: split_groups) or
# "lane" (the CPU lane alone, as before 4.4)
EXPERTS = os.environ.get("GP_EXPERTS", "card")
KEEP_ROUTED = False                   # the layers mode's check of ffn_moe_out
PROF = os.environ.get("GP_PROF", "0") == "1"   # the prefill's time by part (a device sync between parts)
TIMES: dict = {}


def tick(dev, part: str, t0: list) -> None:
    """GP_PROF: the time since t0[0] to part (synchronized), t0 moved on"""
    if PROF:
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        TIMES[part] = TIMES.get(part, 0.0) + t - t0[0]
        t0[0] = t
W_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("GP_WDT", "bf16")]   # big weights
SWIGLU_LIMIT = 10.0          # llama.cpp's swiglu_clamp_shexp: every FFN with a gate (glm5-next), dense ones too
MHC = MHCConfig(dim=D, n=HC, sinkhorn_iters=20, eps=1e-6, norm_eps=RMS_EPS)
KDA = KDAConfig(hidden_size=D, num_heads=NH, head_k_dim=HD, head_v_dim=HD, conv_kernel_size=4, norm_eps=RMS_EPS,
                gate_lower_bound=KDA_LB)
CK = ttnn.init_device_compute_kernel_config(ttnn.device.Arch.BLACKHOLE, math_fidelity=ttnn.MathFidelity.HiFi4,
                                            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
# the indexer score's matmuls: the k-tiles summed in Dest, as iscore() sums them (CK's packer accumulation rounds
# differently: 23% of a head's scores off by an ulp, generated/iscore_mm.py 2026-11-14)
CK_DEST = ttnn.init_device_compute_kernel_config(ttnn.device.Arch.BLACKHOLE, math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)


def kda_state_dict(g: Gguf, il: int) -> dict:
    """GLM's GGUF names -> Kimi's layer-local keys (shapes as tt/kda's expected_kda_weight_shapes)."""
    w = lambda n: g(f"blk.{il}.{n}")  # noqa: E731
    sd = {"q_proj.weight": w("attn_q.weight"), "k_proj.weight": w("attn_k.weight"), "v_proj.weight": w("attn_v.weight"),
          "f_a_proj.weight": w("ssm_f_a.weight"), "f_b_proj.weight": w("ssm_f_b.weight"),
          "g_a_proj.weight": w("ssm_g_a.weight"), "g_b_proj.weight": w("ssm_g_b.weight"),
          "b_proj.weight": w("ssm_beta.weight"), "dt_bias": w("ssm_dt.bias"), "o_norm.weight": w("ssm_norm.weight"),
          "o_proj.weight": w("attn_output.weight"),
          "A_log": torch.log(-w("ssm_a"))}                                 # ssm_a holds -exp(A_log)
    for x in "qkv":
        sd[f"{x}_conv1d.weight"] = w(f"ssm_conv1d_{x}.weight").reshape(NH * HD, 1, 4)   # tap 0 = the oldest row
    return sd


class Card:
    def __init__(self, dev):
        self.dev = dev
        # the lane's rows to the host and back, in buffers kept per shape (Flash-Next's, fused/host_rows.py)
        self.rows, self.outs = HostRows(dev), HostOut(dev, D)

    def put(self, t: torch.Tensor, dtype=BF16, layout=TILE) -> ttnn.Tensor:
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.dev)

    def w(self, t: torch.Tensor) -> ttnn.Tensor:
        """a projection's weight, W_DTYPE"""
        return self.put(t, W_DTYPE)

    def scalar(self, v: int) -> ttnn.Tensor:
        return self.put(torch.tensor([v], dtype=torch.int64), ttnn.uint32, RM)

    def norm(self, v: torch.Tensor) -> ttnn.Tensor:
        return self.put(v.reshape(1, 1, 1, -1), BF16)


KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
HCPOST = os.environ.get("GP_HCPOST", "1") != "0"   # 0: the stock post (4 multiplies, 16 addcmul, a concat)


def _hc_mix(X, o, w, pre: bool):
    """tt/kernels/glm_hcpost_*.cpp over X fp32 [1, 1, T, 4 D]: pre, h = sum_s w_s X_s -> [1, 1, T, D]; else the new
    streams post_j o + sum_s comb_sj X_s -> [1, 1, T, 4 D]. w: the row coefficients widened to tiles [1, 4 | 20, T, 32]."""
    dev, T = X.device(), X.shape[-2]
    rt, ct = T // 32, D // 32
    Y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, D if pre else HC * D]), FP32, TILE, dev,
                                       ttnn.DRAM_MEMORY_CONFIG)
    o = X if pre else o
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    per, extra = divmod(rt * ct, len(cores))
    f = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [X.buffer_address(), o.buffer_address(), w.buffer_address(), f, n]
        crt[cx][cy] = [f, n]
        wrt[cx][cy] = [Y.buffer_address(), f, n]
        f += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    mode = [ttnn.UnpackToDestMode.Default] * 64
    mode[0] = mode[1] = mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = mode
    src = lambda part: os.path.join(KDIR, f"glm_hcpost_{part}.cpp")  # noqa: E731
    kernels = [
        ttnn.KernelDescriptor(kernel_source=src("reader"), core_ranges=grid,
                              compile_time_args=[rt, ct, int(pre)] + acc(X) + acc(o) + acc(w), runtime_args=rrt,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("writer"), core_ranges=grid,
                              compile_time_args=[rt, ct, int(pre)] + acc(Y), runtime_args=wrt,
                              config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("compute"), core_ranges=grid, compile_time_args=[ct, int(pre)],
                              runtime_args=crt, config=cfg)]
    cbs = [ttnn.CBDescriptor(total_size=4096 * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=FP32, page_size=4096)])
        for i, n in ((0, 8), (1, 2), (2, 8 if pre else 40), (16, 2 if pre else 8))]
    ttnn.generic_op([X, o, w, Y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return Y


def _widen(t):
    """[1, 1, T, k] -> [1, k, T, 32]: each column a row-broadcast tile"""
    return ttnn.repeat(ttnn.permute(t, [0, 3, 2, 1]), [1, 1, 1, 32])


def hc_post(X, o, post, comb):
    """new_j = post_j o + sum_s comb_sj X_s in one program: X fp32 [1, 1, T, 4 D], o fp32 [1, 1, T, D], post [1, 1,
    T, 4], comb [1, 1, T, 16]; X and o read once, the new streams written once. Bitwise tt_mhc.hc_post's."""
    return _hc_mix(X, o, _widen(ttnn.concat([post, comb], dim=-1)), False)


def hc_pre_mix(X, pre):
    """h = sum_s pre_s X_s in one program (pre [1, 1, T, 4]); tt_mhc._mix's order"""
    return _hc_mix(X, None, _widen(pre), True)


ISCORE = os.environ.get("GP_ISCORE", "1") == "1"   # the indexer score in one program (else the per-head ops)


def iscore(iq, K, w, _stage=(32, 1)):
    """tt/kernels/glm_iscore_*.cpp: the indexer score [1, 1, T, P] fp32 = sum_h relu(iq_h . K) w_h in one program, the
    per-head chain's order (MlaMixer.sparse): iq [1, 32, T, 128] (bf16 or fp32), K [1, 1, P, 128] fp32, w [1, 1, T, 32]
    fp32. The score tiles split over the cores as runs of (query tile, pool tile), row-major; each written once."""
    dev, T, P = iq.device(), iq.shape[-2], K.shape[-2]
    QT, PT = T // 32, P // 32
    assert iq.shape[1] == IH and iq.shape[-1] == IHD and K.dtype == FP32 and P % 32 == 0, (iq.shape, K.shape)
    wd = _widen(w)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, P]), FP32, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    per, extra = divmod(QT * PT, len(cores))
    f = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [iq.buffer_address(), wd.buffer_address(), K.buffer_address(), f, n]
        crt[cx][cy] = [f, n]
        wrt[cx][cy] = [out.buffer_address(), f, n]
        f += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    mode = [ttnn.UnpackToDestMode.Default] * 64
    mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32      # the gates copied into Dest exactly
    cfg.unpack_to_dest_mode = mode
    ib = 4096 if iq.dtype == FP32 else 2048
    src = lambda part: os.path.join(KDIR, f"glm_iscore_{part}.cpp")  # noqa: E731
    kernels = [
        ttnn.KernelDescriptor(kernel_source=src("reader"), core_ranges=grid,
                              compile_time_args=[QT, PT, ib] + acc(iq) + acc(wd) + acc(K), runtime_args=rrt,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("writer"), core_ranges=grid, compile_time_args=[PT] + acc(out),
                              runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("compute"), core_ranges=grid, compile_time_args=[PT, *_stage],
                              runtime_args=crt, config=cfg)]
    cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=size)])
        for i, fmt, size, n in ((0, iq.dtype, ib, 128), (1, FP32, 4096, 8), (2, FP32, 4096, 32), (16, FP32, 4096, 2))]
    ttnn.generic_op([iq, wd, K, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    ttnn.deallocate(wd)
    return out


class Site:
    """One mHC site on T rows (TtMHCWrap) with its sublayer's input RMSNorm."""

    def __init__(self, c: Card, g: Gguf, il: int, site: str):
        w = lambda n: g(f"blk.{il}.hc_{site}_{n}")  # noqa: E731
        self.hc = TtMHCWrap(c.dev, MHC, w("fn.weight"), w("base.weight"), w("scale.weight"))
        self.norm = c.norm(g(f"blk.{il}.{site}_norm.weight"))

    def pre(self, X):
        if HCPOST:
            hc, T = self.hc, X.shape[-2]
            pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(hc.project(X), hc.consts, hc.n,
                                                                                    hc.iters, hc.eps)
            r4 = lambda t: ttnn.reshape(t, [1, 1, T, t.shape[-1]])  # noqa: E731
            h, post, comb = hc_pre_mix(X, r4(pre)), r4(post), r4(comb)
        else:
            h, post, comb = self.hc.hc_pre(X)
        return ttnn.typecast(ttnn.rms_norm(h, weight=self.norm, epsilon=RMS_EPS, compute_kernel_config=CK), BF16), \
            post, comb

    def post(self, out, X, post, comb):
        o = out if out.dtype == FP32 else ttnn.typecast(out, FP32)
        return hc_post(X, o, post, comb) if HCPOST else self.hc.hc_post(o, X, post, comb)


class DenseFfn:
    def __init__(self, c: Card, g: Gguf, il: int):
        w = lambda n: c.w(g(f"blk.{il}.ffn_{n}.weight").T)  # noqa: E731
        self.gate, self.up, self.down = w("gate"), w("up"), w("down")

    def __call__(self, hn):
        gt = ttnn.clamp(ttnn.linear(hn, self.gate, dtype=FP32, compute_kernel_config=CK), max=SWIGLU_LIMIT)
        up = ttnn.clamp(ttnn.linear(hn, self.up, dtype=FP32, compute_kernel_config=CK), min=-SWIGLU_LIMIT,
                        max=SWIGLU_LIMIT)
        h = ttnn.typecast(ttnn.multiply(ttnn.silu(gt), up), BF16)
        return ttnn.linear(h, self.down, dtype=FP32, compute_kernel_config=CK)


class MlaMixer:
    """GLM's MLA on T rows (llama.cpp's build_dsa_layer): q = q_b(rms(q_a hn)) [64 x 256], absorbed through k_b into
    the 512-wide latent (no RoPE in this model), lat = rms(kv_a hn), causal attention over the latents
    (flash_mla_prefill, scale 256^-1/2), v_b per head, then the output projection. Dense causal: below 2,049 positions
    every pool is selected (the top 512 of <= 512), so the indexer's choice changes nothing there."""

    def __init__(self, c: Card, g: Gguf, il: int):
        w = lambda n: g(f"blk.{il}.{n}")  # noqa: E731
        # q_b's heads reordered for nlp_create_q_heads_split (its 32 "heads" of 512 split into first | second): slot
        # 2 j holds head j, 2 j + 1 head 32 + j, so the two halves come out as heads 0..31 and 32..63 (exact)
        slots = [s // 2 if s % 2 == 0 else NH // 2 + s // 2 for s in range(NH)]
        q_b = w("attn_q_b.weight").reshape(NH, HK, -1)[slots].reshape(NH * HK, -1)
        self.q_a, self.q_b = c.w(w("attn_q_a.weight").T), c.w(q_b.T)
        self.kv_a, self.out = c.w(w("attn_kv_a_mqa.weight").T), c.w(w("attn_output.weight").T)
        self.q_a_norm, self.kv_a_norm = c.norm(w("attn_q_a_norm.weight")), c.norm(w("attn_kv_a_norm.weight"))
        self.k_b = c.put(w("attn_k_b.weight").transpose(1, 2).reshape(1, NH, HK, KVL))   # [h, k, l]: q @ it
        self.v_b = c.put(w("attn_v_b.weight").transpose(1, 2).reshape(1, NH, KVL, HK))   # [h, l, v]
        # the indexer's keys and gates per row (the decode's K and host ring are filled from them at the hand-off)
        self.i_kg = c.put(torch.cat([w("indexer.attn_k.weight"), w("indexer_compressor_gate.weight")]).T)  # [D, 256]
        self.k_norm = (w("indexer.k_norm.weight"), w("indexer.k_norm.bias"))
        self.ape = w("indexer_compressor_ape.weight").reshape(KPOOL, IHD)
        self.i_qb = c.put(w("indexer.attn_q_b.weight").T)                    # [1536, 32 x 128]
        self.i_proj = c.put(w("indexer.proj.weight").T * (IHD * IH) ** -0.5)  # [D, 32], llama.cpp's scale folded
        self.c, self.lat, self.kg, self.sel_stats = c, None, None, None
        self.reset()

    def reset(self) -> None:
        """a new prompt: the chunks' latent rows (row-major [1, 1, rows, 512], sparse_sdpa's form) and indexer rows
        (the host ring's form) empty"""
        self.lat_rows, self.lat_host, self.kg_host = None, None, torch.zeros(0, 2 * IHD)
        self.pk_host = torch.zeros(0, IHD)     # the closed pools' keys so far (a closed pool never changes)

    def kg_rows(self, n: int) -> torch.Tensor:
        """the chunk's rows' normalized indexer key and raw gate [n, 2 x 128] fp32 (the host ring's form):
        LayerNorm in float64 on the host (eps 1e-6, glm_loop.cpp's)"""
        kg = torch.from_numpy(self.c.rows(self.kg, n)).double()
        k = kg[:, :IHD]
        k = (k - k.mean(-1, keepdim=True)) / torch.sqrt(k.var(-1, unbiased=False, keepdim=True) + LN_EPS)
        return torch.cat([k * self.k_norm[0] + self.k_norm[1], kg[:, IHD:]], dim=1).float()

    def pooled(self, kg: torch.Tensor) -> torch.Tensor:
        """the closed pools' keys [n // 4, 128] (per channel the softmax over the pool's 4 rows of gate + ape)"""
        q = kg.shape[0] // KPOOL
        k, gt = (kg[:q * KPOOL].double().reshape(q, KPOOL, 2 * IHD).split(IHD, dim=-1))
        return (torch.softmax(gt + self.ape.double(), dim=1) * k).sum(1).float()

    def __call__(self, hn, p0: int = 0, t0=None):
        """the chunk at positions p0 .. p0 + T - 1 (its rows after the prompt's end: padding, never visible to a
        real query), its latent and indexer rows appended to the prompt's"""
        T = hn.shape[-2]
        assert p0 == (0 if self.lat_rows is None else self.lat_rows.shape[-2]), "chunks in order"
        t0 = t0 or [time.perf_counter()]
        dev = hn.device()
        lin = lambda x, wt: ttnn.linear(x, wt, compute_kernel_config=CK)  # noqa: E731
        qr = ttnn.rms_norm(lin(hn, self.q_a), weight=self.q_a_norm, epsilon=RMS_EPS, compute_kernel_config=CK)
        # head-major [1, 64, T, 256]: nlp_create_q_heads_split's two halves (heads 0..31, 32..63: q_b's slots),
        # concatenated (a permute of the [1, T, 64, 256] rows cost ~0.1 s a layer at 4.6K rows)
        qh = ttnn.experimental.nlp_create_q_heads_split(lin(qr, self.q_b), num_heads=NH // 2, split_head_dim=HK)
        q = ttnn.concat(list(qh), dim=1)
        qa = ttnn.matmul(q, self.k_b, compute_kernel_config=CK)                              # [1, 64, T, 512]
        self.cache_rows(hn, p0, t0)
        if p0 > 0 or T > SPARSE_FROM:
            a = self.sparse(qa, qr, hn, T, p0, t0)
        else:
            a = ttnn.transformer.flash_mla_prefill(qa, self.lat, head_dim_v=KVL, scale=HK ** -0.5, is_causal=True,
                                                   compute_kernel_config=CK)
            tick(dev, "mla: attention", t0)
        o = ttnn.matmul(a, self.v_b, compute_kernel_config=CK)                               # [1, 64, T, 256]
        # [1, 1, T, 16384] by nlp_concat_heads in two halves of 32 heads (64 x 256 outgrow its L1 buffers)
        o = ttnn.concat([ttnn.experimental.nlp_concat_heads(ttnn.slice(o, [0, h, 0, 0], [1, h + NH // 2, T, HK]))
                         for h in (0, NH // 2)], dim=-1)
        return ttnn.linear(o, self.out, dtype=FP32, compute_kernel_config=CK)


    def hold(self, n: int) -> None:
        """after a read: the prompt's n latent rows to the host (bf16, the last chunk's padding dropped), none left on
        the card (memory allocated after the decoder's traces were captured is theirs to overwrite: a held tensor
        there came back changed after verify passes, 2026-11-13); the hand-off and an extension read them there"""
        self.lat_host = ttnn.to_torch(self.lat_rows).reshape(-1, KVL)[:n].clone()
        self.lat_rows, self.kg_host, self.pk_host = None, self.kg_host[:n], self.pk_host[:n // KPOOL]

    def resume(self) -> None:
        """an extension: the held latent rows back on the card"""
        self.lat_rows = self.c.put(self.lat_host.reshape(1, 1, -1, KVL), BF16, RM)

    def cache_rows(self, hn, p0: int, t0=None) -> None:
        """the chunk's latent rows (lat = rms(kv_a hn)) and indexer rows (the keys, the gates) appended to the
        prompt's: all a later row reads of this layer (MtpPrefill takes only this)"""
        T, dev = hn.shape[-2], hn.device()
        t0 = t0 or [time.perf_counter()]
        self.lat = ttnn.rms_norm(ttnn.linear(hn, self.kv_a, compute_kernel_config=CK), weight=self.kv_a_norm,
                                 epsilon=RMS_EPS, compute_kernel_config=CK)
        self.kg = ttnn.linear(hn, self.i_kg, dtype=FP32, compute_kernel_config=CK)
        tick(dev, "mla: projections", t0)
        self.kg_host = torch.cat([self.kg_host, self.kg_rows(T)])
        tick(dev, "mla: indexer rows to the host", t0)
        rows = ttnn.reshape(ttnn.to_layout(self.lat, RM), [1, 1, T, KVL])
        # shortcut: the rows concatenated a chunk at a time (a copy of the prompt's so far); the decode's cache
        # written in place (Flash-Next's INTO) when long prompts' memory is ranked
        self.lat_rows = rows if self.lat_rows is None else ttnn.concat([self.lat_rows, rows], dim=2)
        tick(dev, "mla: latent rows", t0)

    def sparse(self, qa, qr, hn, T: int, p0: int, t0):
        """Past 2,048 positions: per query the top 512 closed pools by the indexer (score = sum_h w_h relu(iq_h . k_p),
        w = proj hn / sqrt(128 x 32): the decode's host rule, glm_loop.cpp) as cells, and its open pool's cells up to
        itself (fused/qsa.py's kernel, signed scores), then sparse_sdpa over those latent rows (k = v = the latent).
        shortcut: the pooled keys made on the host (T x 256 floats down, the new closed pools added, all up), the score
        score by iscore() (GP_ISCORE=0: 32 per-head matmuls, multiplies and adds)."""
        L = -(-(p0 + T) // 128) * 128          # whole tiles of pools (an extension's p0 is any position): the pools
        P = L // KPOOL                          # past p0 + T zero, never visible to the chunk's rows
        K = torch.zeros(P, IHD)
        have, q = len(self.pk_host), (p0 + T) // KPOOL      # only the pools closed since (O(n), not O(n^2 / T))
        if q > have:
            self.pk_host = torch.cat([self.pk_host, self.pooled(self.kg_host[have * KPOOL:q * KPOOL])])
        K[:q] = self.pk_host[:q]
        K = self.c.put(K.reshape(1, 1, P, IHD), FP32)
        tick(qa.device(), "mla: pooled keys (host)", t0)
        iq = ttnn.permute(ttnn.reshape(ttnn.linear(qr, self.i_qb, compute_kernel_config=CK), [1, T, IH, IHD]),
                          [0, 2, 1, 3])                                                     # [1, 32, T, 128]
        w = ttnn.linear(hn, self.i_proj, dtype=FP32, compute_kernel_config=CK)              # [1, 1, T, 32]
        score = iscore(iq, K, w) if ISCORE else None
        for h in range(0 if ISCORE else IH):
            s = ttnn.matmul(ttnn.slice(iq, [0, h, 0, 0], [1, h + 1, T, IHD]), K, transpose_b=True, dtype=FP32,
                            compute_kernel_config=CK_DEST, activation="relu")
            s = ttnn.multiply(s, ttnn.slice(w, [0, 0, 0, h], [1, 1, T, h + 1]))
            score = s if score is None else ttnn.add(score, s)
        tick(qa.device(), "mla: score", t0)
        idx, want = qsa_mask(score, L, p0, topk=SPARSE_TOPK, signed=True)
        tick(qa.device(), "mla: select", t0)
        if SPARSE_CHECK:
            wt = ttnn.to_torch(want).reshape(T, 16)[:, :2].to(torch.int64)
            self.sel_stats = {"rows": T, "tie_rows": int((wt[:, 0] != wt[:, 1]).sum()), "max_cells": int(wt[:, 1].max())}
            # the kernel's cells against the host rule (glm_loop.cpp) on the same scores, rows past 2,047
            sc = ttnn.to_torch(score).double().reshape(T, P)
            ix = ttnn.to_torch(idx).reshape(T, SPARSE_TOPK).to(torch.int64) & 0xFFFFFFFF
            r0 = max(0, 2047 - p0)
            bad, rows = 0, list(range(r0, T, max(1, (T - r0) // 64)))
            for r in rows:
                nv = (p0 + r + 1) // KPOOL
                order = sorted(range(nv), key=lambda q: (-float(sc[r, q]), q))[:512]
                want_c = {q * KPOOL + j for q in order for j in range(KPOOL)} | \
                    {p0 + r - t for t in range((p0 + r + 1) % KPOOL)}
                got_c = set(ix[r][ix[r] != 0xFFFFFFFF].tolist())
                bad += got_c != want_c
            self.sel_stats.update({"checked_rows": len(rows), "rows_differing_from_host_rule": bad})
            self.sel_stats = dict(self.sel_stats, p0=p0)
        q_rm = ttnn.to_layout(qa, RM)
        a = ttnn.transformer.sparse_sdpa(q_rm, self.lat_rows, idx, KVL, kv_format=ttnn.transformer.SparseKVFormat.BF16,
                                         scale=HK ** -0.5, compute_kernel_config=CK)
        a = ttnn.to_layout(a, TILE)
        tick(qa.device(), "mla: attention", t0)
        return a


class MoeFfn:
    """A MoE layer's FFN on T rows: the shared expert on the card (SwiGLU clamped), the routed experts on the CPU
    lane: GLM's routing per row (cl_glm_route: sigmoid, the top 8 of it + the bias, x 2.5, the decode's bits) and
    cl_moe_m over the rows (each expert of the union read once for all of them)."""

    def __init__(self, c: Card, g: Gguf, il: int, lane, experts=None):
        self.c, self.il, self.lane, self.experts = c, il, lane, experts
        w = lambda n: c.w(g(f"blk.{il}.ffn_{n}_shexp.weight").T)  # noqa: E731
        self.gate, self.up, self.down = w("gate"), w("up"), w("down")
        lane.router_load(il, g(f"blk.{il}.ffn_gate_inp.weight").numpy(), g(f"blk.{il}.exp_probs_b.bias").numpy())
        self.routed = None

    def __call__(self, hn, n: int, t0=None):
        T = hn.shape[-2]
        t0 = t0 or [time.perf_counter()]
        gt = ttnn.clamp(ttnn.linear(hn, self.gate, dtype=FP32, compute_kernel_config=CK), max=SWIGLU_LIMIT)
        up = ttnn.clamp(ttnn.linear(hn, self.up, dtype=FP32, compute_kernel_config=CK), min=-SWIGLU_LIMIT,
                        max=SWIGLU_LIMIT)
        sh = ttnn.linear(ttnn.typecast(ttnn.multiply(ttnn.silu(gt), up), BF16), self.down, dtype=FP32,
                         compute_kernel_config=CK)
        tick(hn.device(), "moe: shared expert", t0)
        E = self.experts if self.experts is not None and self.experts.takes(self.il) else None
        if E is not None:
            E.pump()            # a streamed layer's copy while the card runs the layer's queued dense work
            tick(hn.device(), "moe: card experts' stream", t0)
        x = self.c.rows(hn, n)                                  # fp32 [n, D] (bf16 hn widened)
        tick(hn.device(), "moe: hn down", t0)
        ids, wts = self.lane.glm_route_m(self.il, x)           # every row cl_glm_route's, to the bit
        if KEEP_ROUTED:   # the layers mode: checked against the one-row routing
            for t in range(n):
                i1, w1, _ = self.lane.glm_route(self.il, x[t])
                assert np.array_equal(i1, ids[t]) and np.array_equal(w1.view(np.uint32), wts[t].view(np.uint32)), t
        tick(hn.device(), "moe: routing (lane)", t0)
        if E is not None:
            # the padding rows routed as row 0 (finite sums, unused: the down's rows for them are written, not garbage)
            pad = lambda a: np.concatenate([a, np.repeat(a[:1], T - n, 0)])  # noqa: E731
            routed = E.routed(self.il, hn, pad(ids), pad(wts), n, x=x)
            E.pump()            # the next streamed layer's copy while the card runs this one's experts
            tick(hn.device(), "moe: card experts (+ the lane's share)", t0)
            if KEEP_ROUTED:
                self.routed = ttnn.to_torch(routed).float().reshape(T, D)[:n]
            return ttnn.add(sh, routed)
        out = self.c.outs.buffer(T, n)
        for i in range(0, n, LANE_M):   # cl_moe_m takes <= MAXM (2,048) rows a call
            j = min(n, i + LANE_M)
            self.lane.moe_m(self.il, np.ascontiguousarray(x[i:j]), np.ascontiguousarray(ids[i:j]),
                            np.ascontiguousarray(wts[i:j]), out[i:j])
        tick(hn.device(), "moe: lane experts", t0)
        if KEEP_ROUTED:
            self.routed = torch.from_numpy(out[:n].copy())
        return ttnn.add(sh, self.c.outs.upload(T))


# the card's share of a MoE layer's routed experts by prompt size (Flash-Next's policy, tt/serve/engine.py use_experts):
# (most tokens, groups of 32 experts on the card of 9). From `split` (2026-11-13, bfp8 weights, warm, no decoder; s for
# the prefill at 0 / 3 / 5 / 7 / 9 groups): 512 tokens 2.57 / 3.34 / 4.47 / 5.60 / 6.70, 1,024 4.65 / 4.12 / 5.16 / 6.35
# / 7.42, 2,048 9.08 / 7.50 / 7.12 / 8.23 / 9.24, 4,486 20.24 / 16.56 / 14.00 / 13.11 / 14.09; bounds where the lines
# cross. A group costs ~0.55 s a prompt in copies (34 x 0.435 GB over PCIe), so short prompts stay on the lane; past
# 4.5K the share stays 7: the card's ~19 us a token a layer for all experts against the lane's ~77 put ~80% on the card
# (not measured past 4,486 yet). GP_CARD_GROUPS overrides.
# 2026-11-14 (P6 + the 8 Q5_K layers + fused mHC; card groups 5 / 6 / 7 / 8 / 9): 4,487 tokens 16.5 / 17.0 / 17.1 / 18.4 /
# 19.1 s, 29,514 82.0 / 72.9 / 66.3 / 68.2 / 71.8 s; then with MOE_ROWS 8,192 4,487 tokens 13.6 (5) / 13.1 s (7): the
# grouped calls make the card's share cheaper again, so 3,400 stays the bound
SPLIT = tuned("GLM_SPLIT", [(800, 0), (1800, 3), (3400, 5), (float("inf"), 7)])   # tune.json: Infinity for inf
# shortcut: the least int8-down groups (of 9) for which the Q5_K layers go to the card; 2 measured no gain (lane-bound),
# unmeasured between 3 and 9: measure when card memory is freed
Q8_MIN = int(os.environ.get("GP_Q8_MIN", 5))


def split_groups(n: int, E) -> int:
    """the groups of a card layer's experts the card takes for a prompt of n tokens (the rest: the lane's), at most
    what E's stream slots hold"""
    if os.environ.get("GP_CARD_GROUPS"):
        g = int(os.environ["GP_CARD_GROUPS"])
    else:
        g = next((g for bound, g in SPLIT if n <= bound), E.n_groups)
    return min(g, E.slot_groups)


def card_experts(dev, reserve_gb: float | None = None):
    """g6_experts' CardExperts in the card memory left (built after everything else): its 2 stream slots hold as many
    groups of 32 experts as fit beside GP_CARD_RESERVE_GB (default 5: the prefill's and the experts' run buffers at
    4.6K rows), and of them as many with room for an int8 down (the Q5_K layers') as fit after -> None when not one
    group fits (the lane alone)"""
    from g6_experts import SHAPE, CardExperts
    reserve = float(os.environ.get("GP_CARD_RESERVE_GB", 5)) if reserve_gb is None else reserve_gb
    folder = SHAPE.out_dir()
    size = lambda il, k: os.path.getsize(os.path.join(folder, f"blk.{il}.p0.{k}.npy"))  # noqa: E731
    q8 = [il for il in SHAPE.layers if os.path.exists(os.path.join(folder, f"blk.{il}.p0.dn_v.npy"))]
    group = sum(size(next(il for il in SHAPE.layers if il not in q8), k) for k in ("gu_c", "gu_s", "dn_c", "dn_s"))
    dn_v = size(q8[0], "dn_v") if q8 else 0
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    free = mv.largest_contiguous_bytes_free_per_bank * 8 - reserve * 1e9
    # at most SPLIT's largest share a slot when Q8 downs stream: the room left goes to their dn_v groups
    sg = int(os.environ.get("GP_SLOT_GROUPS", min(SHAPE.NG, SPLIT[-1][1] if q8 else SHAPE.NG, int(free // (2 * group)))))
    qg = int(os.environ.get("GP_Q8_GROUPS", min(sg, int((free - 2 * sg * group) // (2 * dn_v))) if q8 else 0))
    S = SHAPE
    if q8 and qg < Q8_MIN:
        # too small a share to pay (2 of 9 groups: the lane still takes 7 and the layer stays the lane's length; 29.5K
        # 90.0 -> 91.1 s, 2026-11-14): the Q5_K layers stay the lane's, the slots without dn_v
        print(f"card experts: {qg} int8-down groups fit (< {Q8_MIN}): layers {q8} stay on the lane", flush=True)
        S = copy.copy(SHAPE)
        S.layers = [il for il in SHAPE.layers if il not in q8]
        q8, qg = [], 0
        sg = int(os.environ.get("GP_SLOT_GROUPS", min(SHAPE.NG, int(free // (2 * group)))))
    print(f"card experts: {(free + reserve * 1e9) / 1e9:.1f} GB free, {group / 1e9:.2f} GB a group: {sg} groups a slot"
          + (f", {qg} with the int8 down of layers {q8} ({dn_v / 1e9:.2f} GB more each)" if q8 else ""), flush=True)
    return CardExperts(dev, [], n_layers=45, shape=S, slot_groups=sg, q8_groups=qg) if sg > 0 else None


class Layer:
    def __init__(self, c: Card, g: Gguf, il: int, T: int, lane, experts=None):
        self.il = il
        self.sa, self.sf = Site(c, g, il, "attn"), Site(c, g, il, "ffn")
        self.mla = il % 4 == 3
        self.mixer = MlaMixer(c, g, il) if self.mla else ttKDA(c.dev, KDA, kda_state_dict(g, il), active_seq_len=T,
                                                                    program_config=KDAProgramConfig(projection_dtype=W_DTYPE))
        self.ffn = DenseFfn(c, g, il) if il < N_DENSE else MoeFfn(c, g, il, lane, experts)
        self.reset()

    def hold(self, n: int) -> None:
        """after a read: the states to the host (MlaMixer.hold's reason), the KDA carries as their raw tensors"""
        if self.mla:
            self.mixer.hold(n)
        else:
            self.held = (ttnn.to_torch(self.new_state.convolution), ttnn.to_torch(self.new_state.recurrent))
            self.state = self.new_state = None

    def resume(self) -> None:
        """an extension: the held states back on the card, bit for bit"""
        if self.mla:
            self.mixer.resume()
            return
        st = self.mixer.allocate_state()
        for ref, t in zip((st.convolution, st.recurrent), self.held):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=ref.dtype, layout=ref.layout), ref)
        self.state = self.new_state = st

    def reset(self) -> None:
        """a new prompt: the KDA carries zero, the MLA caches empty"""
        if self.mla:
            self.mixer.reset()
        else:
            self.state = self.new_state = self.mixer.allocate_state()

    def front(self, X, n: int, start, end, p0: int = 0) -> dict:
        """a chunk of T rows at positions p0 .., its first n real (start, end: the device scalars p0, p0 + n), up to the
        FFN's input: the attention half and the FFN site's pre"""
        T = X.shape[-2]
        dev, t0 = X.device(), [time.perf_counter()]
        hn, post, comb = self.sa.pre(X)
        tick(dev, "mhc", t0)
        if self.mla:
            o = self.mixer(hn, p0, t0)
            tick(dev, "mla: v_b and the output", t0)
        else:
            o, self.new_state = self.mixer.forward(ttnn.reshape(hn, [1, T, D]), self.state, start, end)
            self.state = self.new_state              # the next chunk's carry (forward only reads its input state)
            o = ttnn.reshape(o, [1, 1, T, D])
            tick(dev, "kda", t0)
        X1 = self.sa.post(o, X, post, comb)
        hn, post, comb = self.sf.pre(X1)
        tick(dev, "mhc", t0)
        return {"attn_out": o, "X1": X1, "hn": hn, "post": post, "comb": comb, "t0": t0, "n": n}

    def back(self, f: dict, y) -> dict:
        """the FFN's output y through the FFN site's post"""
        out = {"attn_out": f["attn_out"], "ffn_out": y, "l_out": self.sf.post(y, f["X1"], f["post"], f["comb"])}
        tick(y.device(), "mhc", f["t0"])
        return out

    def __call__(self, X, n: int, start, end, p0: int = 0, warm=()) -> dict:
        f = self.front(X, n, start, end, p0)
        for k in warm:      # Prefill.warm_ks: the grouped calls' programs, on k copies of this chunk's rows (unused)
            hb = ttnn.concat([f["hn"]] * k, dim=-2)
            T = X.shape[-2]
            y = self.ffn(hb, (k - 1) * T + n, f["t0"])
            ttnn.deallocate(hb)
            for j in range(k):
                ttnn.deallocate(ttnn.slice(y, [0, 0, j * T, 0], [1, 1, (j + 1) * T, D]))
            ttnn.deallocate(y)
        y = self.ffn(f["hn"]) if isinstance(self.ffn, DenseFfn) else self.ffn(f["hn"], n, f["t0"])
        tick(X.device(), "dense ffn" if isinstance(self.ffn, DenseFfn) else "moe: lane up", f["t0"])
        return self.back(f, y)

    def group(self, chs: list) -> None:
        """a MoE layer over consecutive chunks ([X, n, start, end, p0], all full but the last) with one routed-experts
        call for all their rows (GP_MOE_ROWS): an expert's weights decoded once for them; each chunk's X replaced"""
        fs = [self.front(*ch) for ch in chs]
        T = chs[0][0].shape[-2]
        hb = ttnn.concat([f["hn"] for f in fs], dim=-2)
        y = self.ffn(hb, T * (len(chs) - 1) + chs[-1][1], fs[-1]["t0"])
        ttnn.deallocate(hb)
        for k, (ch, f) in enumerate(zip(chs, fs)):
            f["t0"][0] = time.perf_counter()     # GP_PROF: the post's own time, not since this chunk's front
            ch[0] = self.back(f, ttnn.slice(y, [0, 0, k * T, 0], [1, 1, (k + 1) * T, D]))["l_out"]
        ttnn.deallocate(y)


MLA_W = ("q_a", "q_b", "kv_a", "out", "q_a_norm", "kv_a_norm", "k_b", "v_b", "i_kg", "i_qb", "i_proj")
SHEXP_W = ("gate", "up", "down")


def _weights(L: "Layer"):
    """A layer's streamable weight groups: (kind, its device tensors, rebind(slot layer): its users pointed at the slot
    layer's tensors). The mHC sites and the dense FFNs (layers 0-2) stay resident (small, or unlike the rest)."""
    if L.mla:
        yield "mla", [getattr(L.mixer, n) for n in MLA_W], \
            lambda S: [setattr(L.mixer, n, getattr(S.mixer, n)) for n in MLA_W]
    else:
        w = L.mixer.weights
        yield "kda", [w.input_projection, w.decay_output_projection, w.output_projection, w.decay_scale_flat,
                      w.decay_bias_flat, w.norm, *w.convolution_taps], lambda S: setattr(L.mixer, "weights", S.mixer.weights)
    if isinstance(L.ffn, MoeFfn):
        yield "shexp", [getattr(L.ffn, n) for n in SHEXP_W], \
            lambda S: [setattr(L.ffn, n, getattr(S.ffn, n)) for n in SHEXP_W]


class DenseStream:
    """The prefill's dense weights streamed like the experts. The prefill runs layer by layer, so a
    layer's mixer and shared-expert weights are needed once a prompt: two slots a kind (KDA mixer, MLA mixer, shared
    expert; the first two layers' own tensors), every layer's weights on the host, copied into its slot on the second
    command queue before the layer runs (the copy into a slot waits on a device event after the layer that used it
    last). Frees the other layers' ~7.5 GB of card memory for the expert slots. Built before the decoder's traces are
    captured (memory freed or allocated after that is the replays' to overwrite)."""

    def __init__(self, dev):
        self.dev = dev
        self.q0, self.q1 = ttnn.QueueId(0), ttnn.QueueId(1)
        self.order, self.slots, self.entries, self.at = {}, {}, {}, {}
        self.seconds = 0.0

    def add(self, i: int, L: "Layer") -> None:
        """layer i, just built: its weights to the host, and from each kind's third layer on its device tensors freed
        before the next layer is built (which then reuses their room: freed afterwards they left holes no expert group
        fit, 2026-11-14) and its users pointed at the slot layer's"""
        for kind, ts, rebind in _weights(L):
            j = len(self.order.setdefault(kind, []))
            self.order[kind].append(i)
            host = [ttnn.from_device(t) for t in ts]
            if j < 2:
                self.slots.setdefault(kind, []).append(L)
            slot = self.slots[kind][j % 2]
            if j >= 2:
                rebind(slot)
                for t in ts:
                    ttnn.deallocate(t)
            tgt = next(t for k, t, _ in _weights(slot) if k == kind)
            self.entries[(kind, j)] = {"host": host, "slot": tgt, "ev": None}
            self.at.setdefault(i, []).append((kind, j))

    def _copy(self, key, after=None) -> None:
        e = self.entries[key]
        t0 = time.perf_counter()
        if after is not None:
            ttnn.wait_for_event(self.q1, after)     # the slot's last user done on queue 0
        for h, t in zip(e["host"], e["slot"]):
            ttnn.copy_host_to_device_tensor(h, t, cq_id=self.q1)
        e["ev"] = ttnn.record_event(self.dev, self.q1)
        self.seconds += time.perf_counter() - t0

    def start(self) -> None:
        """a read's start: each kind's first two layers into their slots"""
        for kind in self.order:
            for j in (0, 1):
                if (kind, j) in self.entries:
                    self._copy((kind, j))

    def before(self, i: int) -> None:
        for key in self.at.get(i, []):
            ttnn.wait_for_event(self.q0, self.entries[key]["ev"])

    def after(self, i: int) -> None:
        """layer i queued on queue 0 (all chunks): the next layer of each kind sharing its slots, after it"""
        nxt = [(kind, j + 2) for kind, j in self.at.get(i, []) if (kind, j + 2) in self.entries]
        if nxt:
            ev = ttnn.record_event(self.dev, self.q0)
            for key in nxt:
                self._copy(key, ev)


def mla_hand_off(mx: "MlaMixer", n: int, lat_t, K_t, state: int, lib) -> None:
    """An MLA layer's decode state after a prefill's n rows: the latent rows into lat_t (row-major [ctx, 512]), the
    closed pools' keys into K_t (tile [1, 1, n_pool, 128]), the host state at position n (gl_mla_set: the open pool's
    ring rows)"""
    # bf16 on the host throughout (an fp32 [ctx, 512] converted per layer cost ~0.15 s)
    lat = torch.zeros(tuple(lat_t.shape), dtype=torch.bfloat16).reshape(-1, KVL)
    lat[:n] = mx.lat_host[:n]
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(lat.reshape(tuple(lat_t.shape)), dtype=BF16, layout=RM), lat_t)
    kg = mx.kg_host[:n]
    K = torch.zeros(tuple(K_t.shape), dtype=torch.bfloat16).reshape(-1, IHD)
    K[:n // KPOOL] = mx.pooled(kg).to(torch.bfloat16)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(K.reshape(tuple(K_t.shape)), dtype=BF16, layout=TILE), K_t)
    rows = min(8, n)
    ring = np.ascontiguousarray(kg[n - rows:].numpy(), np.float32)
    assert lib.gl_mla_set(state, n, ring.ctypes.data, rows) == 0


class MtpPrefill:
    """GLM's MTP block (blk.45, g5_mtp.Mtp's graph) over a prompt's rows in chunks of T: row j reads the trunk's
    post-norm hidden of position j and the embedding of token j + 1; the front (eh_proj over rms(e) enorm | rms(h)
    hnorm), then the MLA's latent and indexer rows from rms(x) attn_norm (MlaMixer.cache_rows on blk.45's weights).
    Nothing else: a row's caches depend only on its own front (each row reads the trunk's h, not the MTP's previous
    output), so the attention, the FFN and the head of these rows change no later draft. hand_off() writes the caches
    into the MTP's (g5_mtp.Mtp: lat, K, the host state)."""

    def __init__(self, c: Card, g: Gguf, lane, T: int, layer: int = 45):
        w = lambda n: g(f"blk.{layer}.{n}")  # noqa: E731
        self.c, self.T = c, T
        self.enorm, self.hnorm = c.norm(w("nextn.enorm.weight")), c.norm(w("nextn.hnorm.weight"))
        self.attn_norm = c.norm(w("attn_norm.weight"))
        self.eh = c.w(w("nextn.eh_proj.weight").T)                                     # [8192, 4096]
        self.mla = MlaMixer(c, g, layer)
        self.n = None

    def __call__(self, emb: np.ndarray, ids, hid: torch.Tensor, n: int, r0: int = 0) -> None:
        """rows r0 .. n - 1: hid[j - r0] with token ids[j + 1]; r0 > 0 continues the rows held (self.n == r0)"""
        T, c = self.T, self.c
        norm = lambda x, wt: ttnn.rms_norm(x, weight=wt, epsilon=RMS_EPS, compute_kernel_config=CK)  # noqa: E731
        if r0 == 0:
            self.mla.reset()
        else:
            self.mla.resume()
        assert r0 == (self.n if r0 else 0), (r0, getattr(self, "n", None))
        self.n = None                           # until the rows are all in
        for p0 in range(r0, n, T):
            m = min(T, n - p0)
            e, h = torch.zeros(T, D), torch.zeros(T, D)
            e[:m] = torch.from_numpy(emb[np.asarray(ids[p0 + 1:p0 + 1 + m])])
            h[:m] = hid[p0 - r0:p0 - r0 + m]
            E, H = (c.put(t.reshape(1, 1, T, D), FP32) for t in (e, h))
            cat = ttnn.concat([norm(E, self.enorm), norm(H, self.hnorm)], dim=-1)
            x = ttnn.linear(ttnn.typecast(cat, BF16), self.eh, dtype=FP32, compute_kernel_config=CK)
            self.mla.cache_rows(ttnn.typecast(norm(x, self.attn_norm), BF16), p0)
        self.mla.hold(n)
        self.n = n

    def hand_off(self, mtp, lib) -> None:
        mla_hand_off(self.mla, self.n, mtp.lat, mtp.K, mtp.state, lib)


class Prefill:
    """The prompt's first n rows through the 45 layers in chunks of T rows, layer by layer (every chunk through a layer
    before the next layer, the chunks' residual streams kept: a streamed layer's experts cross PCIe once a prompt,
    Flash-Next's order): the KDA layers carry their states from chunk to chunk, the MLA layers keep the chunks' latent
    and indexer rows and attend over all of them (dense causal for a lone first chunk of <= 2,048 rows, else the
    indexer's selection and sparse_sdpa); the routed experts on the card (experts: g7_experts.CardExperts of
    g6_experts.SHAPE, its layers) with a share on the lane by prompt size (split_groups), or the lane's; then
    hand_off() writes the decode's states."""

    def __init__(self, c: Card, g: Gguf, lane, T: int, n_layer: int = 45, experts=None, stream: bool = False):
        """stream: the dense weights streamed (DenseStream; needs the second command queue; before the experts are
        sized and the decoder's traces captured)"""
        self.c, self.T, self.lane, self.experts = c, T, lane, experts
        self.stream = DenseStream(c.dev) if stream else None
        self.layers = []
        for il in range(n_layer):
            self.layers.append(Layer(c, g, il, T, lane, experts))
            if stream:
                self.stream.add(il, self.layers[-1])
        self.out_norm = c.norm(g("output_norm.weight"))
        self.emb = None
        self.n = None
        self.warm_ks = ()   # the engine's warm-up: the group sizes (chunks a call) to compile, on each kind of MoE layer

    def set_experts(self, E) -> None:
        """the routed experts on the card from now on (CardExperts built after the prefill's weights)"""
        self.experts = E
        for L in self.layers:
            if isinstance(L.ffn, MoeFfn):
                L.ffn.experts = E

    def __call__(self, emb: np.ndarray, ids, n: int, start: int = 0) -> torch.Tensor:
        """emb: the token embedding [VOCAB, D] -> the trunk's post-norm hidden rows [n - start, D] of positions
        start .. n - 1 (the MTP's h); start > 0 continues the states of this prefill's last read (self.n == start: the
        KDA carries, the MLA rows), as its chunks carry them. start 32-aligned (ttKDA's actual_start contract: from
        2,499 the rows were off from blk.0 on, from 2,560 bit for bit); the sparse path's pools padded to tiles"""
        T, c, E = self.T, self.c, self.experts
        assert start == (self.n if start else 0) and start % 32 == 0, (start, self.n)
        self.n = None                           # until the read is done (a failed read continues nothing)
        for L in self.layers:
            L.resume() if start else L.reset()
        if E is not None:
            g = split_groups(n - start, E)
            E.lane, E.card_groups = (self.lane if g < E.n_groups else None), g
            if g:
                E.plan()
        chunks = []
        for p0 in range(start, n, T):
            m = min(T, n - p0)
            x = torch.zeros(T, HC * D)
            x[:m] = torch.from_numpy(emb[np.asarray(ids[p0:p0 + m])]).repeat(1, HC)   # mhc_expand: every stream the row
            chunks.append([c.put(x.reshape(1, 1, T, HC * D), FP32), m, c.scalar(p0), c.scalar(p0 + m), p0])
        st = self.stream
        if st is not None:
            st.start()
        seen = set()
        for i, L in enumerate(self.layers):
            if st is not None:
                st.before(i)
            xr = max(1, MOE_ROWS // T) if isinstance(L.ffn, MoeFfn) else 1
            warm = ()
            if self.warm_ks and isinstance(L.ffn, MoeFfn):
                kind = E is not None and E.takes(i) and (E.layers[i].gu_q, E.layers[i].dn_q)
                warm = () if kind in seen else self.warm_ks
                seen.add(kind)
            for g0 in range(0, len(chunks), xr):
                grp = chunks[g0:g0 + xr]
                if len(grp) > 1:
                    L.group(grp)
                    continue
                X, m, start, end, p0 = grp[0]
                grp[0][0] = L(X, m, start, end, p0, warm=warm if g0 == 0 else ())["l_out"]
            if st is not None:
                st.after(i)
        if E is not None:
            E.finish()
        hid = []
        for X, m, _, _, p0 in chunks:
            # the streams' mean by slices (a [1, T, 4, D] tile reshape pads the 4 streams to 32: 2.4 GB at 4.6K rows)
            h = ttnn.slice(X, [0, 0, 0, 0], [1, 1, T, D])
            for i in range(1, HC):
                h = ttnn.add(h, ttnn.slice(X, [0, 0, 0, i * D], [1, 1, T, (i + 1) * D]))
            hn = ttnn.rms_norm(ttnn.multiply(h, 1.0 / HC), weight=self.out_norm, epsilon=RMS_EPS,
                               compute_kernel_config=CK)
            hid.append(ttnn.to_torch(hn).float().reshape(T, D)[:m])
        for L in self.layers:
            L.hold(n)
        self.n = n
        return torch.cat(hid)

    def hand_off(self, tok, lib) -> None:
        """the decode's states after the prompt's n rows (g4_token.Token, its M-row build): per KDA layer the conv
        history and S (Mixer.load_states), per MLA layer the latent rows, the closed pools' keys and the host state
        at position n (gl_mla_set: the open pool's ring rows)"""
        n = self.n
        mla_i = 0
        for L, D_ in zip(self.layers, tok.layers):
            if L.mla:
                mla_hand_off(L.mixer, n, D_["lat"], D_["K"], tok.mla_states[mla_i], lib)
                mla_i += 1
            else:
                conv = L.held[0].float().reshape(3, -1)
                S = L.held[1].float().reshape(NH, HD, HD)
                D_["kda"].load_states(conv, S)


def dram_gb(dev) -> float:
    """card memory (DRAM) allocated now, GB (g7_prefill's)"""
    return round(ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank * 8 / 1e9, 2)


def compare(name: str, got: torch.Tensor, want: torch.Tensor) -> dict:
    got, want = got.double(), want.double()
    rel = (got - want).norm(dim=-1) / want.norm(dim=-1)
    r = {"pcc": round(pcc(got, want), 7), "row_rel_l2_median": float(rel.median()), "row_rel_l2_max": float(rel.max()),
         "max_abs_rel": float((got - want).abs().max() / want.abs().max())}
    print(f"  {name}: {json.dumps(r)}", flush=True)
    return r


def layer0() -> None:
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32)
    n = len(ids)
    T = 32 * math.ceil(n / 32)
    X0 = torch.from_numpy(np.fromfile(os.path.join(REF, "tensors", "hc_init.f32"), np.float32).reshape(n, HC * D))
    want = {k: torch.from_numpy(np.fromfile(os.path.join(REF, "tensors", f"{k}-0.f32"), np.float32))
            for k in ("ffn_out", "l_out")}
    want["ffn_out"] = want["ffn_out"].reshape(n, D)
    want["l_out"] = want["l_out"].reshape(n, HC * D)
    g = Gguf(MODEL)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ref": REF, "tokens": n, "T": T}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=1536)
    try:
        c = Card(dev)
        t0 = time.perf_counter()
        sa, sf = Site(c, g, 0, "attn"), Site(c, g, 0, "ffn")
        kda = ttKDA(dev, KDA, kda_state_dict(g, 0), active_seq_len=T)
        ffn = DenseFfn(c, g, 0)
        res["build_s"] = round(time.perf_counter() - t0, 1)
        state = kda.allocate_state()
        start, end = c.scalar(0), c.scalar(n)
        X = c.put(torch.cat([X0, torch.zeros(T - n, HC * D)]).reshape(1, 1, T, HC * D), FP32)

        def run():
            hn, post, comb = sa.pre(X)
            o, st = kda.forward(ttnn.reshape(hn, [1, T, D]), state, start, end)
            X1 = sa.post(ttnn.reshape(o, [1, 1, T, D]), X, post, comb)
            hn, post, comb = sf.pre(X1)
            y = ffn(hn)
            return y, sf.post(y, X1, post, comb), st

        y, X2, st = run()
        to = lambda t, w: ttnn.to_torch(t).float().reshape(T, w)[:n]  # noqa: E731
        res["ffn_out"] = compare("ffn_out-0", to(y, D), want["ffn_out"])
        res["l_out"] = compare("l_out-0", to(X2, HC * D), want["l_out"])
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(5):
            run()
        ttnn.synchronize_device(dev)
        res["layer_ms_eager"] = round((time.perf_counter() - t0) / 5 * 1e3, 2)
        print(f"  blk.0 on {T} rows: {res['layer_ms_eager']} ms eager", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, "g6_prefill_layer0.json"), "w"), indent=1)


def ref_tensor(name: str, n: int, width: int) -> torch.Tensor | None:
    p = os.path.join(REF, "tensors", f"{name}.f32")
    return torch.from_numpy(np.fromfile(p, np.float32).reshape(n, width)) if os.path.exists(p) else None


def layers() -> None:
    """blk.0 .. GP_LAYERS - 1 on the prompt's rows, each layer twice: from llama.cpp's own input (l_out of the layer
    before: the layer's own error) and chained from ours (the prefill's error so far), against l_out-<il>,
    attn_out / kv_cmpr (MLA), ffn_moe_out."""
    from g3_overlap import cpulane
    global KEEP_ROUTED
    KEEP_ROUTED = True
    n_layers = int(os.environ.get("GP_LAYERS", 5))
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32)
    n = len(ids)
    T = 32 * math.ceil(n / 32)
    g = Gguf(MODEL)
    lane = cpulane.Lane()
    t0 = time.perf_counter()
    lane.load(MODEL, layers=set(range(N_DENSE, n_layers)))
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ref": REF, "tokens": n, "T": T, "layers": {},
           "lane_load_s": round(time.perf_counter() - t0, 1)}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=1536)
    try:
        c = Card(dev)
        start, end = c.scalar(0), c.scalar(n)
        pad = lambda x: c.put(torch.cat([x, torch.zeros(T - n, HC * D)]).reshape(1, 1, T, HC * D), FP32)  # noqa
        to = lambda t, w: ttnn.to_torch(t).float().reshape(T, w)[:n]  # noqa: E731
        x_ours = pad(ref_tensor("hc_init", n, HC * D))
        for il in range(n_layers):
            t1 = time.perf_counter()
            L = Layer(c, g, il, T, lane)
            build = time.perf_counter() - t1
            r = {"build_s": round(build, 1), "mla": L.mla}
            inp = ref_tensor("hc_init" if il == 0 else f"l_out-{il - 1}", n, HC * D)
            for how, X in (("isolated", pad(inp)), ("chained", x_ours)):
                ttnn.synchronize_device(dev)
                t1 = time.perf_counter()
                L.reset()
                out = L(X, n, start, end)
                ttnn.synchronize_device(dev)
                r[f"{how}_ms"] = round((time.perf_counter() - t1) * 1e3, 1)
                print(f" blk.{il} {how}:", flush=True)
                r[how] = {"l_out": compare(f"l_out-{il}", to(out["l_out"], HC * D), ref_tensor(f"l_out-{il}", n, HC * D))}
                if how == "isolated":
                    for k, w, got in (("attn_out", D, lambda: to(out["attn_out"], D)),
                                      ("kv_cmpr", KVL, lambda: to(L.mixer.lat, KVL)),
                                      ("ffn_moe_out", D, lambda: L.ffn.routed)):
                        want = ref_tensor(f"{k}-{il}", n, w)
                        if want is not None:
                            r[how][k] = compare(f"{k}-{il}", got(), want)
                else:
                    x_ours = out["l_out"]
            res["layers"][il] = r
            del L
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, f"g6_prefill_layers{n_layers}.json"), "w"), indent=1)


def handoff() -> None:
    """The prompt's first GP_N rows (default all but the reference's last 9) by the prefill, handed to the speculative
    decoder (g5_spec.Spec: the trunk's M-row build + the MTP, its caches caught up over the prompt), then the rest of
    the reference teacher-forced in verify passes of 4 (all kept) against llama.cpp's logits; then the same positions
    with the prompt through the decoder's own passes (the earlier path), for the distance between the two."""
    import ctypes
    from g5_spec import M_MAX, Spec
    from g3_fkda import WORKER_L1_M
    from g4_token import VOCAB, _LIB, cpulane
    from g5_mtp import MTP
    _LIB.gl_mla_set.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
    ids = [int(x) for x in np.fromfile(os.path.join(REF, "ids.i32"), np.int32)]
    frm = int(os.environ.get("GP_FROM", 0))      # the first position logits.f32 holds (the long reference: 4,486)
    ref_logits = np.memmap(os.path.join(REF, "logits.f32"), np.float32, "r").reshape(-1, VOCAB)
    n = int(os.environ.get("GP_N", len(ids) - 9))
    assert n >= frm
    # the chunk (GP_CHUNK, a multiple of 128: the selection's pools a whole number of tiles); default the prompt
    T = int(os.environ.get("GP_CHUNK", 0)) or 128 * math.ceil(n / 128)
    assert T % 128 == 0
    g = Gguf(MODEL)
    lane = cpulane.Lane()
    lane.load(MODEL, layers=set(range(N_DENSE, 45)) | {MTP})
    lane.router_load(MTP, g(f"blk.{MTP}.ffn_gate_inp.weight").numpy(), g(f"blk.{MTP}.exp_probs_b.bias").numpy())
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ref": REF, "n": n, "T": T}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=1 << 30, worker_l1_size=WORKER_L1_M,
                                l1_small_size=1536, num_command_queues=2 if EXPERTS == "card" else 1)
    try:
        s = Spec(dev, g, lane)
        res["card_gb"] = {"decoder": dram_gb(dev)}
        t0 = time.perf_counter()
        pf = Prefill(Card(dev), g, lane, T)
        res["card_gb"]["prefill"] = dram_gb(dev)
        res["prefill_build_s"] = round(time.perf_counter() - t0, 1)
        if EXPERTS == "card":
            t0 = time.perf_counter()
            E = card_experts(dev)
            if E is not None:
                pf.set_experts(E)
                res["slot_groups"] = E.slot_groups
                res["card_gb"]["experts"] = dram_gb(dev)
            res["experts_build_s"] = round(time.perf_counter() - t0, 1)
        res["experts"] = EXPERTS
        res["card_groups"] = split_groups(n, pf.experts) if pf.experts is not None else 0
        print("card memory (GB, allocated):", json.dumps(res["card_gb"]), "card groups", res["card_groups"], flush=True)

        def tail(how: str) -> dict:
            kl, top1 = [], []
            while s.pos < len(ids) - 1:
                rows = ids[s.pos:min(s.pos + M_MAX, len(ids) - 1)]
                s.verify(rows)
                lg = ttnn.to_torch(s.tok.logits).double().reshape(-1, VOCAB)[:len(rows)]
                for m in range(len(rows)):
                    rf = torch.from_numpy(ref_logits[s.pos + m - frm].astype(np.float64))
                    lp, lq = torch.log_softmax(rf, -1), torch.log_softmax(lg[m], -1)
                    kl.append(float((lp.exp() * (lp - lq)).sum()))
                    top1.append(bool(rf.argmax() == lg[m].argmax()))
                s.keep(len(rows))
            r = {"positions": [n, len(ids) - 1], "kl_median": float(np.median(kl)), "kl_mean": float(np.mean(kl)),
                 "kl": [round(x, 5) for x in kl], "top1": float(np.mean(top1))}
            print(f"{how}:", json.dumps(r), flush=True)
            return r

        # the trunk's trace first: Spec.verify's first pass is eager and the capture after it rebinds tok.logits to the
        # trace's output (read after it, the eager pass's logits are the unexecuted capture's buffer)
        s.reset()
        s.verify(ids[:M_MAX])
        # (a) the prefill, the hand-off, the MTP caught up over the prompt's rows from the prefill's hidden rows
        s.reset()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        hid = pf(s.emb, ids, n)
        ttnn.synchronize_device(dev)
        res["prefill_s"] = round(time.perf_counter() - t0, 2)
        if PROF:
            res["prefill_parts_s"] = {k: round(v, 2) for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1])}
            print("parts:", json.dumps(res["prefill_parts_s"]), flush=True)
        res["selection"] = {L.il: L.mixer.sel_stats for L in pf.layers if L.mla and L.mixer.sel_stats}   # last chunk's
        if res["selection"]:
            print("selection:", json.dumps(res["selection"]), flush=True)
        pf.hand_off(s.tok, _LIB)
        s.pos, s.n_keep, s.parity = n, 1, 0
        s.mtp_catch_up(0, ids[1:n + 1], hid, 0)
        res["prefill"] = tail("after the prefill")
        if os.environ.get("GP_PASSES", "1") != "0":   # (GP_PASSES=0, profiling runs: the prefill only)
            # (b) the same positions with the prompt through the decoder's passes
            s.reset()
            t0 = time.perf_counter()
            while s.pos < n:
                rows = ids[s.pos:min(s.pos + M_MAX, n)]
                s.verify(rows)
                s.keep(len(rows))
            res["passes_s"] = round(time.perf_counter() - t0, 2)
            res["passes"] = tail("after the decoder's passes")
            print(f"prompt of {n}: prefill {res['prefill_s']} s (eager, untuned), passes {res['passes_s']} s", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, f"g6_prefill_handoff_n{n}" + (f"_c{T}" if T < n else "") +
                              (f"_{EXPERTS}{res['card_groups']}" if EXPERTS == "card" else "") +
                              ("_bfp8" if W_DTYPE == ttnn.bfloat8_b else "") +
                              ("_prof" if PROF else "") + ".json"), "w"),
              indent=1)


def split() -> None:
    """The prefill's time by prompt size and card share, for split_groups' table: glm_ref_long's first n tokens for n in
    GP_SWEEP_N (default 512, 1024, 2048, 4486) with GP_SWEEP_G groups of 9 on the card (default 0, 3, 6, 9; 0 = the
    lane alone), each timed on its second run (the first compiles); the decoder not built (its card memory free)."""
    from g3_fkda import WORKER_L1_M
    from g4_token import cpulane
    from g6_experts import SHAPE, CardExperts
    ref = os.path.join(DATA, "glm_ref_long")
    ids = [int(x) for x in np.fromfile(os.path.join(ref, "ids.i32"), np.int32)]
    sizes = [int(v) for v in os.environ.get("GP_SWEEP_N", "512,1024,2048,4486").split(",")]
    groups = [int(v) for v in os.environ.get("GP_SWEEP_G", "0,3,6,9").split(",")]
    g = Gguf(MODEL)
    emb = g("token_embd.weight").float().numpy()
    lane = cpulane.Lane()
    lane.load(MODEL, layers=set(range(N_DENSE, 45)))
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "mt": SHAPE.mt, "mt_dn": SHAPE.mt_dn, "s": {}}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), worker_l1_size=WORKER_L1_M, l1_small_size=1536,
                                num_command_queues=2)
    try:
        E = CardExperts(dev, [], n_layers=45, shape=SHAPE)
        for n in sizes:
            T = 128 * math.ceil(n / 128)
            pf = Prefill(Card(dev), g, lane, T, experts=E)
            for k in groups:
                os.environ["GP_CARD_GROUPS"] = str(k)
                for _ in range(2):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    pf(emb, ids, n)
                    ttnn.synchronize_device(dev)
                    dt = time.perf_counter() - t0
                res["s"][f"{n}/{k}"] = round(dt, 2)
                print(f"{n} tokens, {k} groups on the card: {dt:.2f} s", flush=True)
            del pf
    finally:
        os.environ.pop("GP_CARD_GROUPS", None)
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, "g6_prefill_split.json"), "w"), indent=1)


def experts() -> None:
    """The card's routed experts (g6_experts' card form) against the lane's on the same rows and routes, for GP_XLAYERS
    (default 7, 20, 40) and GP_XROWS rows (default 4,486, T rounded to 128): hn rows drawn N(0, GP_XSCALE^2) (the
    routing's spread is what the grouping sees; a scale of ~8 takes gate and up past the SwiGLU clamp's 10), routed by cl_glm_route_m; the card alone and split with the lane
    (GP_CARD_GROUPS, default 4 of 9). Per row rel L2 of card - lane against the lane."""
    import copy
    from g3_overlap import cpulane
    from g3_fkda import WORKER_L1_M
    from g6_experts import SHAPE, CardExperts
    ils = [int(v) for v in os.environ.get("GP_XLAYERS", "7,20,40").split(",")]
    n = int(os.environ.get("GP_XROWS", 4486))
    T = 128 * math.ceil(n / 128)
    S = copy.copy(SHAPE)
    S.layers = ils
    g = Gguf(MODEL)
    lane = cpulane.Lane()
    lane.load(MODEL, layers=set(ils))
    for il in ils:
        lane.router_load(il, g(f"blk.{il}.ffn_gate_inp.weight").numpy(), g(f"blk.{il}.exp_probs_b.bias").numpy())
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "n": n, "T": T, "mt": S.mt, "mt_dn": S.mt_dn, "layers": {}}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), worker_l1_size=WORKER_L1_M, l1_small_size=1536,
                                num_command_queues=2)
    try:
        c = Card(dev)
        E = CardExperts(dev, [], n_layers=45, shape=S)
        rng = np.random.default_rng(0)
        cg = int(os.environ.get("GP_CARD_GROUPS", 4))
        inputs = {}
        for il in ils:                       # the rows, their routes and the lane's sums per layer
            x = np.zeros((T, D), np.float32)
            x[:n] = rng.standard_normal((n, D), dtype=np.float32) * float(os.environ.get("GP_XSCALE", 1))
            hn = c.put(torch.from_numpy(x).reshape(1, 1, T, D), BF16)
            xb = c.rows(hn, n).copy()                            # the bf16 rows, widened: both sides read these
            ids, wts = lane.glm_route_m(il, xb)
            want = np.zeros((n, D), np.float32)
            for i in range(0, n, LANE_M):
                j = min(n, i + LANE_M)
                lane.moe_m(il, np.ascontiguousarray(xb[i:j]), np.ascontiguousarray(ids[i:j]),
                           np.ascontiguousarray(wts[i:j]), want[i:j])
            inputs[il] = (hn, xb, ids, wts, want)
        pad = lambda a: np.concatenate([a, np.repeat(a[:1], T - n, 0)])  # noqa: E731
        # a pass = a plan (its layers streamed in order); the card alone twice (the first compiles), then the split
        for how, ln, groups in (("card (cold)", None, E.n_groups), ("card", None, E.n_groups),
                                (f"split{cg}", lane, cg)):
            E.lane, E.card_groups = ln, groups
            E.plan()
            for il in ils:
                hn, xb, ids, wts, want = inputs[il]
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                got = E.routed(il, hn, pad(ids), pad(wts), n, x=xb)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                got = ttnn.to_torch(got).float().numpy().reshape(T, D)[:n]
                rel = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
                r = {"rel_l2_median": float(np.median(rel)), "rel_l2_max": float(rel.max()),
                     "finite": bool(np.isfinite(got).all()), "s": round(dt, 3)}
                print(f"blk.{il} {how}:", json.dumps(r), flush=True)
                res["layers"].setdefault(il, {})[how] = r
            E.finish()
    finally:
        ttnn.close_mesh_device(dev)
    res["xscale"] = float(os.environ.get("GP_XSCALE", 1))
    json.dump(res, open(os.path.join(OUT, f"g6_prefill_experts_n{n}_x{res['xscale']:g}.json"), "w"), indent=1)


if __name__ == "__main__":
    {"layer0": layer0, "layers": layers, "handoff": handoff, "experts": experts, "split": split}[sys.argv[1] if len(sys.argv) > 1 else "layer0"]()
