"""Chunked prefill into the card engine. A prompt's tokens go through the 48 layers a chunk at a time (stock ttnn
ops plus G2's kernels, real weights), and the states the decode programs read are written into g4_model.Model's own
tensors (the programs hold their addresses): per DeltaNet layer the recurrent state S and the conv state, per
attention layer the K and V caches, the raw indexer keys and the pooled keys, the PLE block's history ring. The next
decode step (Model.step / step_traced at position len(ids)) then continues the sequence.

Per chunk of T tokens at positions p0 .. p0 + T - 1 (T a multiple of 128; the last chunk padded, its pad rows after
the real ones, so causal attention and the DeltaNet's valid_len keep them out):
  X        the residual streams, fp32 [1, 4, T, 2560] (stream-major), from the host's embedding rows
  HC mix   rms_norm per stream, the down|inject projection (norm folded in, 4-batch matmul), the gate, the
           stream sum (hn) and the inject weights; the combine X_c += 2 sigmoid(inject_c / 4) * out, in fp32
  DeltaNet the projection [q|k|v | z | alpha | beta] (value heads permuted to FLA's GQA order: the op reads key head
           h // 3, the GGUF's graph h % 16), G2's causal conv kernel (its state the previous chunk's last rows),
           qwen36's fused chunked delta rule (in-kernel L2 norms, the state carried in its own layout), the gated
           RMSNorm head-major, ssm_out
  attention the projection, q / k / indexer-q RMSNorm + NEOX RoPE (64 of 256 dims) at the absolute positions, the
           chunk's raw indexer keys and pooled keys (mean of 4, RMSNorm, RoPE at the block's first cell), the
           indexer's scores over every pool before the chunk's end, QSA's selection as a mask (the top min(512,
           complete pools) per query, the incomplete own block's cells up to the query; g4_host.Host.select's rule),
           dense flash attention under it over the cells before the chunk's end, the sigmoid gate, wo
  PLE      before layer 1: key|value of the host's n-gram embedding, the per-stream gate, the gated value's norm,
           the dilated conv over the 9-token history, X += gated + silu(conv)
  FFN half the router (bf16 weights, fp32 logits), the shared expert, the routed experts, the combine
Layer by layer: every chunk of the prompt through a layer before the next layer (the residual streams of all chunks
kept), so a streamed layer's experts cross PCIe once per prompt. Routed experts: g7_experts.CardExperts (the card,
native codes; the layers it does not hold streamed from pinned host memory on command queue 1) or CpuExperts (the
CPU lane's cl_moe_m, the decode's own experts, M <= 8 tokens per call: each token's sum is cl_moe's to the bit).

The hand-off (Prefill.hand_off, page copies on the card, pf_pagecopy.cpp): S from the op's [48, 128, 128] (FLA head
order) to the decode's tiles (GGUF order), the conv state = the last 3 tokens' q|k|v projection rows (fp32, GGUF
channel order), the K / V caches row-major [2, CTX, 256] bf16, the raw keys [CTX, 128] fp32, the pooled keys in the
decode's tile order, the PLE ring's rows q % 17 for the last 9 tokens. CTX follows g3_attn1.CTX.

Modes (python g7_prefill.py <mode>; W51_EXPERTS card | cpu, W51_CHUNK, results/g7_prefill_<mode>_<experts>.json):
  refstates  the decode's own states after g4_ref's 46 / 91 and g4_ref_long's 512 / 4,775 tokens, saved once to
             $TT_MOEOFFLOAD_DATA/w51_ref_states (the checks compare against them)
  states     the fast loop: the first W51_LAYERS layers over W51_REFN's tokens, every state against the decode's
  check      g4_ref: the prefill layer by layer against llama.cpp's tensors; check (b); prefill 46 + decode 46
  long       g4_ref_long: check (a) (prefill 4,775, the hand-off, decode 256 against llama.cpp), check (d)
  sens       step 0 of the expert-kernel scope: timing-only variants of the experts (sens())
  long16     g4_ref_16k: check (c) (prefill 13,244, the hand-off, decode 256 against llama.cpp and the card)
  gdn        where the DeltaNet's inputs and state depart from float64
  time       prefill tok/s from ids to the decode-ready state, random ids (W51_PROMPTS), the per-phase breakdown
  mem        card memory by holder after the build, and at the peak of a W51_MEM_PROMPT-token prompt (256K plan, step 0)

  python3 tt_moeoffload/launch.py --cpus all --memory-max $((220 << 30)) --env W51_EXPERTS=card \
      --engine tt/flashnext/g7_prefill.py -- long
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import types
import statistics
import sys
import time

import numpy as np
import torch

# torch's pool, woken by the host-side ops (the hand-off's conv rows, tensors made for upload), would sit on the lane's
# spinning CPUs: a 1.3 MB host tensor took 12 ms, the hand-off's conv rows 410 ms
torch.set_num_threads(1)
# the streamed experts stay pinned (g7_experts.CardExperts): tt-metal's pinned-memory cache holds them (75.7 GB with
# the 5 Q8_0 layers resident, 87.3 GB with every layer streamed)
os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", str(96 << 30))
import ttnn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from g2_conv import CausalConv  # noqa: E402
from g2_layers import Gguf  # noqa: E402
from g3_attn1 import CTX, NPOOL, RAWK  # noqa: E402
from g3_hc import model_tag  # noqa: E402
from g4_model import is_attention  # noqa: E402
from g7_experts import HostOut, HostRows  # noqa: E402
from fused import qsa as _qsa  # noqa: E402
from fused.qsa import qsa_alloc  # noqa: E402  (re-exported)

OUT = os.path.join(HERE, "results")
KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
WCACHE = os.path.join(DATA, "w51_weights")
D, HC, LR, LO = 2560, 4, 320, 352
NK, NV, HK = 16, 48, 128
QKV, ZD, NA = 2 * NK * HK + NV * HK, NV * HK, 64          # alpha, beta padded to 64 columns each (2 tiles)
NH, NKV, HD, NROT = 24, 2, 256, 64
IH, IHD, KPOOL, NSEL = 4, 128, 4, 512
EXT_LPAD = 8192                       # a follow-up's attention cells: a multiple of this (Chunk's lpad)
INTO = os.environ.get("W51_INTO", "1") == "1"   # mem, long16, time: the prefill writes into the decode's caches
# a query sub-block's queries x cells in Attn.select: its fp32 [Tq, L] temporaries at most 268 MB (at 8192 x 16384,
# 537 MB, a 65,503-token prompt's last chunks found no 67 MB a bank among 288 MB free, 2026-11-06)
SUB_CELLS = 8192 * 8192
NX, TOPK, FF = 512, 10, 640
EPS, ROPE_BASE = 1e-6, 1e7
RING = 17                                                   # g3_ple.RING


def resident(model_dir: str) -> list[int]:
    """the layers whose experts stay in card memory: the Q8_0-down layers (2.33 GB each; the others stream), from the
    GGUF's types (the validated UD-IQ4_XS: 2, 4, 30, 46, 47); W51_RESIDENT (a list, maybe empty) overrides"""
    if os.environ.get("W51_RESIDENT") is not None:
        return [int(v) for v in os.environ["W51_RESIDENT"].split(",") if v]
    from expert_types import expert_types
    return [il for il, (_, _, d) in expert_types(model_dir).items() if d == "Q8_0"]


# a prompt of several chunks released the traces and their buckets' buffers first (2026-11-07: at the 64K build a
# 65,503-token prompt found no block for a 512 MB mask); since the sparse path (no [Tq, L] mask past 20K cells) a
# 126K prompt fits beside them on the 128K build (81.8 s), so they stay (2026-11-10), released only if a read runs
# out of card memory (Prefill.prefill); 1: released first, as before
DROP_TRACES = os.environ.get("W51_DROP_TRACES", "0") == "1"
PERM = [j // 3 + NK * (j % 3) for j in range(NV)]           # FLA value head j <- GGUF value head PERM[j]
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
# HiFi3 by default for the prefill's matmuls, SDPAs and experts (2026-11-10, the user's call): 126K 87.1 -> 85.3 s; KL
# against llama.cpp g4_ref_64k 0.0267 -> 0.0229, _long 0.0174 -> 0.0138, _16k 0.0222 -> 0.0197, g4_ref 0.0312 -> 0.0718
# (within g4_ref's sum-order sensitivity: HiFi4 at key chunk 32 / 64 gives 0.0751 / 0.0384). HI (selection) stays HiFi4
BF16, BFP8, FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
CK = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("W51_FIDELITY", "HiFi3")),
                                      math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
HI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True, packer_l1_acc=False)
# the masked SDPA's fidelity and key chunk (2026-11-08, scope 1: HiFi2 + k 256 measured 39.5 -> 30.3 ms a 2,048 x 32K
# sub-block, not bitwise; the parity sweep decides)
SDPA_CK = ttnn.WormholeComputeKernelConfig(
    math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("W51_SDPA_FID", "HiFi3")), math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=False)
SDPA_K = int(os.environ.get("W51_SDPA_K", 128))


def cut(t, a, b):
    """Columns [a, b) of the last dim."""
    s = list(t.shape)
    return ttnn.slice(t, [0] * (len(s) - 1) + [a], s[:-1] + [b])


def rows(t, a, b):
    """Rows [a, b) of dim -2."""
    s = list(t.shape)
    return ttnn.slice(t, [0] * (len(s) - 2) + [a, 0], s[:-2] + [b, s[-1]])


def sum4(x):
    """[1, 4, T, W] -> the sum over dim 1, [1, 1, T, W] (ttnn.sum over dim 1 pads it to a tile row: 8x the memory)."""
    _, h, T, W = x.shape
    p = [ttnn.slice(x, [0, i, 0, 0], [1, i + 1, T, W]) for i in range(h)]
    return ttnn.add(ttnn.add(p[0], p[1]), ttnn.add(p[2], p[3]))


def lin(x, w, dtype=FP32):
    return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=CK)


TILEMOVE = os.environ.get("W51_TILEMOVE", "1") != "0"   # 0: heads / tokens through row-major (the stock ops)
TILEMOVE_MIN = 8192          # tiles: below, the stock ops' round trip is as fast (the kernel's launch dominates)


def tile_move(x, to_heads: bool, n: int, d: int, c0: int = 0):
    """tt/kernels/pf_tilemove.cpp: token-major x's columns c0 .. c0 + n d -> head-major [1, n, T, d] (to_heads), or
    head-major x -> token-major [1, 1, T, n d]: whole tiles re-indexed (a tile is 32 tokens x 32 dims of one head in
    both), the values those of the row-major round trip to the bit."""
    dev, T = x.device(), x.shape[-2]
    shape = [1, n, T, d] if to_heads else [1, 1, T, n * d]
    y = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), x.dtype, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    total = n * (T // 32) * (d // 32)
    per, extra = divmod(total, len(cores))
    rt, f = ttnn.RuntimeArgs(), 0
    used = []
    for k, (cx, cy) in enumerate(cores):
        c = per + (1 if k < extra else 0)
        if c:
            rt[cx][cy] = [x.buffer_address(), y.buffer_address(), f, c, T // 32, x.shape[-1] // 32, c0 // 32]
            used.append((cx, cy))
        f += c
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in used])
    pb = page_bytes(x)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    k = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_tilemove.cpp"), core_ranges=ranges,
                              compile_time_args=[pb, n, d // 32] + acc(x) + acc(y),
                              defines=[("HEADS", "1")] if to_heads else [], runtime_args=rt,
                              config=ttnn.ReaderConfigDescriptor())
    cb = ttnn.CBDescriptor(total_size=16 * pb, core_ranges=ranges, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint8, page_size=pb)])
    ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))
    return y


MEMLOG = os.environ.get("W51_MEMLOG") == "1"       # card memory before each layer's experts (debug)
# a layer's routed experts over consecutive chunks of up to this many rows at once (Prefill.run): units of up to MT_MAX
# tiles span the chunks, so an expert's weights are decoded once for all of them (offline on real routes, 2 x 8K: -20%
# units, 4 x 8K -29%; 16K on the card 11.37 -> 9.49 s, bit for bit). Bounded by the down's output, 10 fp32 rows a
# token in one buffer: 1.7 GB at 16K; 32K's 3.4 GB found no block at the 64K build (2026-11-09). 0: a chunk a call
XROWS = int(os.environ.get("W51_XROWS", 16384))
MASKSTAT = {int(v) for v in os.environ.get("W51_MASKSTAT", "").split(",") if v}   # layers whose masks are measured
# W51_MASKSTAT_AT: only the first sub-block of the chunks holding these positions (a 128K prompt's masks read back whole
# would be ~16 GB a layer); unset: every sub-block
MASKSTAT_AT = [int(v) for v in os.environ.get("W51_MASKSTAT_AT", "").split(",") if v]


def mask_stats(il: int, chunk, sub, mask) -> None:
    """How sparse a sub-block's QSA mask is in blocks (W51_MASKSTAT): per query block of q rows and key block of k
    cells, whether any cell is attended; the share of such blocks a query block needs, and the attended cells' share,
    for the block sizes a block-sparse attention would use (2026-11-08, before choosing one). Read back: slow."""
    m = ttnn.to_torch(mask).reshape(mask.shape[-2], mask.shape[-1]) == 0
    Tq, L = m.shape
    vis = chunk.p0 + sub[1]                      # the cells any of these queries can see
    out = {"layer": il, "p0": chunk.p0, "rows": [chunk.p0 + sub[0], chunk.p0 + sub[1]], "L": L,
           "cells_attended": round(float(m.float().mean()) * L / max(1, vis), 4)}
    for q, k in ((128, 128), (32, 128), (128, 32), (32, 32)):
        b = m[: Tq // q * q, : L // k * k].reshape(Tq // q, q, L // k, k).any(dim=3).any(dim=1).float()
        seen = [-(-(chunk.p0 + sub[0] + (i + 1) * q) // k) for i in range(Tq // q)]   # a block's visible key blocks
        out[f"q{q}_k{k}"] = round(sum(float(b[i, :n].sum()) / n for i, n in enumerate(seen)) / len(seen), 4)
    print(f"[maskstat] {json.dumps(out)}", flush=True)


# W51_SPARSE=1 (2026-11-09): a long chunk's attention by ttnn's sparse_sdpa over each query's selected cells
# (pf_qsa_mask.cpp IDX) instead of the dense SDPA under the mask; sparse_probe.py: 7.9x at 128K, closer to float64.
# Per KV head the rows [V | K] and the queries [0 | q] (its 12 heads padded to 32): q' . row = q . K, V the rows'
# leading 256 columns. W51_SPARSE_CHECK=1: each sub-block's wanted cell counts read back (SPARSE_STATS).
# W51_GNORM_OP (on; 0: the op chain): the DeltaNet gated norm by ttnn.experimental.kda.sigmoid_gated_rms_norm, 32K
# 22.31 -> 21.56 s, against float64 on real rows as the chain (mean error 9.0e-5 against 8.9e-5 at layer 24)
GNORM_OP = os.environ.get("W51_GNORM_OP", "1") == "1"
GNORM_F64 = int(os.environ.get("W51_GNORM_F64", -1))   # a layer: both gated norms against float64 once, printed
SPARSE = os.environ.get("W51_SPARSE", "1") == "1"   # on since 2026-11-09 (the user's call: 126K -32%, closer to f64)
SPARSE_CHECK = os.environ.get("W51_SPARSE_CHECK", "0") == "1"
SPARSE_TOPK = 2176                                   # >= 512 x 4 + 3, a multiple of sparse_sdpa's k chunk (128)
# sparse_sdpa costs ~12.5 us a query whatever the context (gather-bound: 2,176 rows of 1 KB a KV head), the dense SDPA
# ~0.6 us per 1K cells (sparse_probe plumb: 25.5 against 39.5 ms a 2,048-query sub-block at 32K): the crossover is
# ~21K, so the chunks whose context L passes this go sparse (24K and up; 16K stays dense)
SPARSE_MIN_L = int(os.environ.get("W51_SPARSE_MIN_L", 20480))
# sparse_sdpa's fidelity (2026-11-10): HiFi3 drops only phase 3, a bf16 input's low 3 mantissa bits times the other's
# last bit (<= ~2^-11 a product, about HiFi4's own alignment loss); not bitwise, the parity sweep decides
SPARSE_CK = ttnn.WormholeComputeKernelConfig(
    math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("W51_SPARSE_FID", "HiFi3")), math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=False)
SPARSE_STATS = {"rows": 0, "tie_rows": 0, "max_mask_cells": 0, "max_cells": 0}   # tie: the mask kept more than 512
# W51_SPARSE_F64=layer,position: on that layer's sub-block holding the position, the dense SDPA's output and the sparse
# one against float64 over each sampled row's selected cells (on the sparse indices; rows where the two rules differ
# skipped), relative to the reference's largest value; into SPARSE_STATS["f64"]
SPARSE_F64 = [int(v) for v in os.environ.get("W51_SPARSE_F64", "").split(",") if v]
QSAK = os.environ.get("W51_QSAK", "1") == "1"      # the selection's mask by pf_qsa_mask.cpp (0: the ttnn ops)
QSAK_ROWS = int(os.environ.get("W51_QSAK_ROWS", 0))  # query rows a unit (8, 4 or 2; 0: by L, or IDX's by Tq)
# the sparse path's selection, all bit for bit (2026-11-09, sel_long2.py at 128K, an 8K chunk a layer 390 -> 191 ms):
# the score by per-head matmuls with relu fused (the same sum order), IDX's rows a unit by Tq, sub-blocks 4x larger
SEL_FAST = os.environ.get("W51_SEL_FAST", "1") == "1"
# the score by pf_qsa_score_*.cpp, bit for bit (2026-11-10, sel_score.py: an 8K chunk a layer at 128K 77.3 -> 10.3 ms)
SCOREK = os.environ.get("W51_SCOREK", "1") == "1"


def qsa_score(iq, pk, s0: int, s1: int):
    """tt/kernels/pf_qsa_score_*.cpp: the sub-block's indexer score [1, 1, Tq, P] fp32 = ((r0 + r1) + (r2 + r3)) /
    sqrt(128), r_h = relu(iq_h . pk), iq [1, 4, T, 128] fp32 (or bf16; rows s0 .. s1), pk [1, 1, P, 128] fp32; the pool tiles
    split over the cores, each core's held in L1, the query tiles two at a time."""
    import struct
    dev, QT, Tq, P = iq.device(), iq.shape[-2] // 32, s1 - s0, pk.shape[-2]
    PT = P // 32
    assert iq.dtype in (BF16, FP32) and pk.dtype == FP32 and iq.shape[1] == IH and Tq % 64 == 0 and s0 % 32 == 0, (iq, pk)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Tq, P]), FP32, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    per, extra = divmod(PT, len(cores))
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    p0 = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [iq.buffer_address(), pk.buffer_address(), p0, n, s0 // 32, Tq // 32]
        crt[cx][cy] = [n, Tq // 32]
        wrt[cx][cy] = [out.buffer_address(), p0, n, Tq // 32]
        p0 += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    scale = struct.unpack("<I", struct.pack("<f", IHD ** -0.5))[0]
    ib = 4096 if iq.dtype == FP32 else 2048
    kernels = [
        ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_qsa_score_reader.cpp"), core_ranges=grid,
                              compile_time_args=[QT, ib] + acc(iq) + acc(pk), runtime_args=rrt,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_qsa_score_writer.cpp"), core_ranges=grid,
                              compile_time_args=[PT] + acc(out), runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_qsa_score_compute.cpp"), core_ranges=grid,
                              compile_time_args=[scale], runtime_args=crt, config=cfg)]
    npm = per + (1 if extra else 0)
    cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=size)])
        for i, f, size, n in ((0, iq.dtype, ib, 64), (1, FP32, 4096, 4 * max(1, npm)), (16, FP32, 4096, 4))]
    ttnn.generic_op([iq, pk, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def qsa_mask(score, L: int, pos0: int, R: int = QSAK_ROWS, out=None, topk: int = 0):
    """fused/qsa.py's (shared with GLM since 2026-11-13) with this file's W51_QSAK_ROWS / W51_SEL_FAST: IDX's rows a
    unit by Tq whenever W51_QSAK_ROWS is 0, as before the move"""
    if topk and SEL_FAST and not QSAK_ROWS:
        R = 0
    return _qsa.qsa_mask(score, L, pos0, R, out, topk, sel_fast=SEL_FAST)


def heads(x, n, d, c0: int = 0):
    """Columns c0 .. c0 + n d of token-major x [1, 1, T, W] -> head-major [1, n, T, d]."""
    if TILEMOVE and n * x.shape[-2] * d >= TILEMOVE_MIN * 1024:
        return tile_move(x, True, n, d, c0)
    if c0 or x.shape[-1] != n * d:
        x = cut(x, c0, c0 + n * d)
    T = x.shape[-2]          # through row-major (a tile reshape pads n to 32 rows per token)
    return ttnn.to_layout(ttnn.permute(ttnn.reshape(ttnn.to_layout(x, RM), [1, T, n, d]), [0, 2, 1, 3]), TILE)


def tokens(x):
    """Head-major [1, n, T, d] -> [1, 1, T, n * d]: bf16 by ttnn's nlp_concat_heads (the dedup audit's K2: bit for bit,
    0.61 against tile_move's 1.15 ms at [1, 24, 8192, 256])."""
    _, n, T, d = x.shape
    if x.dtype == BF16 and d % 32 == 0:
        return ttnn.experimental.nlp_concat_heads(x)
    if TILEMOVE and n * T * d >= TILEMOVE_MIN * 1024:
        return tile_move(x, False, n, d)
    return ttnn.to_layout(ttnn.reshape(ttnn.permute(ttnn.to_layout(x, RM), [0, 2, 1, 3]), [1, 1, T, n * d]), TILE)


def softplus(x):
    """log(1 + e^x) = max(x, 0) + log1p(e^-|x|): the LLK softplus is 0 below -5."""
    return ttnn.add(ttnn.relu(x), ttnn.log(ttnn.add(ttnn.exp(ttnn.neg(ttnn.abs(x))), 1.0)))


def to_np(t) -> np.ndarray:
    return ttnn.to_torch(t).float().numpy()


class Weights:
    """The prefill's copies of a layer's weights in ttnn's [k, n] tile layout (the decode's are n-major streams),
    converted once from the GGUF and kept in WCACHE (ttnn tensor files)."""

    def __init__(self, dev, g: Gguf):
        self.dev, self.g, self.tag = dev, g, model_tag(g)

    def __call__(self, key: str, make, dtype=BFP8, layout=TILE) -> ttnn.Tensor:
        path = os.path.join(WCACHE, self.tag, key + ".tensorbin")
        if os.path.exists(path):
            return ttnn.load_tensor(path, device=self.dev)
        t = make()
        host = ttnn.from_torch(t.contiguous().float(), dtype=dtype, layout=layout)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path[:-len('.tensorbin')]}.{os.getpid()}.tmp.tensorbin"
        ttnn.dump_tensor(tmp, host)
        os.replace(tmp, path)
        return ttnn.to_device(host, self.dev)

    def row(self, v: torch.Tensor, dtype=FP32) -> ttnn.Tensor:
        """A small constant [1, 1, 1, n] (or any shape given as [.., n]), not cached."""
        return ttnn.from_torch(v.float().reshape([1] * (4 - v.dim()) + list(v.shape)).contiguous(), dtype=dtype,
                               layout=TILE, device=self.dev)


def gate_sum(dev, r, gate, norm_t):
    """hn = sum_c r_c norm_c sigmoid(gate_c) over the 4 streams, r and gate fp32 [1, 4, T, 2560], into bf16 [1, 1, T,
    2560]: tt/kernels/hcmix_*.cpp (G2's), its CBs fp32 and unpacked to Dest as fp32, the output tiles bf16."""
    T = r.shape[-2]
    rt, ct = T // 32, D // 32
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, D]), BF16, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    per, extra = divmod(rt * ct, len(cores))
    f = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [r.buffer_address(), gate.buffer_address(), norm_t.buffer_address(), f, n]
        crt[cx][cy] = [f, n]
        wrt[cx][cy] = [y.buffer_address(), f, n]
        f += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    mode = [ttnn.UnpackToDestMode.Default] * 64
    mode[0] = mode[1] = mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = mode
    src = lambda part: os.path.join(KDIR, f"hcmix_{part}.cpp")  # noqa: E731
    kernels = [
        ttnn.KernelDescriptor(kernel_source=src("reader"), core_ranges=grid,
                              compile_time_args=[rt, ct] + acc(r) + acc(gate) + acc(norm_t), runtime_args=rrt,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("writer"), core_ranges=grid, compile_time_args=[rt, ct] + acc(y),
                              runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("compute"), core_ranges=grid, compile_time_args=[rt],
                              runtime_args=crt, config=cfg)]
    cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=size)])
        for i, fmt, size, n in ((0, FP32, 4096, 8), (1, FP32, 4096, 8), (2, FP32, 4096, 8), (16, BF16, 2048, 2))]
    ttnn.generic_op([r, gate, norm_t, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


HCGATE = os.environ.get("W51_HCGATE", "1") != "0"   # 0: the up matmul (on a repeated 4 times) and gate_sum
HCGATE_NC = int(os.environ.get("W51_HCGATE_NC", 0))  # output columns a core's item covers (0: by T)


def hc_gate(dev, a, up, xn, norm_t):
    """hn = sum_c xn_c norm_c sigmoid(a up_c) into bf16 [1, 1, T, 2560] (tt/kernels/hcgate_*.cpp): a bf16 [1, 1, T,
    320], up bfp8 [1, 4, 320, 2560], xn fp32 [1, 4, T, 2560]; the gate tiles stay in Dest. A core's items are a group
    of NC columns over a run of row tiles, the group's up tiles kept in L1."""
    T = xn.shape[-2]
    rt, ct = T // 32, D // 32
    nc = HCGATE_NC or (4 if rt >= 64 else 1)
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, D]), BF16, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    per, extra = divmod(rt * (ct // nc), len(cores))
    f = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [a.buffer_address(), up.buffer_address(), xn.buffer_address(), norm_t.buffer_address(), f, n]
        crt[cx][cy] = [f, n]
        wrt[cx][cy] = [y.buffer_address(), f, n]
        f += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    mode = [ttnn.UnpackToDestMode.Default] * 64
    mode[2] = mode[3] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = mode
    src = lambda part: os.path.join(KDIR, f"hcgate_{part}.cpp")  # noqa: E731
    kernels = [
        ttnn.KernelDescriptor(kernel_source=src("reader"), core_ranges=grid,
                              compile_time_args=[rt, nc] + acc(a) + acc(up) + acc(xn) + acc(norm_t), runtime_args=rrt,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("writer"), core_ranges=grid, compile_time_args=[rt, nc] + acc(y),
                              runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("compute"), core_ranges=grid, compile_time_args=[rt, nc],
                              runtime_args=crt, config=cfg)]
    cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=size)])
        for i, fmt, size, n in ((0, BF16, 2048, 20), (1, BFP8, 1088, 40 * nc), (2, FP32, 4096, 8 * nc),
                                (3, FP32, 4096, 4 * nc), (16, BF16, 2048, 2 * nc))]
    ttnn.generic_op([a, up, xn, norm_t, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


def hc_lo(dev, p):
    """The 4 streams' down|inject partials p fp32 [1, 4, T, 352] -> (a = bf16 silu(lo / 4) [1, 1, T, 320], the inject
    tile 2 sigmoid(lo / 4) fp32 [1, 1, T, 32], its first 4 columns the weights), lo = sum4(p) (tt/kernels/hclo_*.cpp)."""
    T = p.shape[-2]
    rt = T // 32
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, LR]), BF16, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 32]), FP32, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    per, extra = divmod(rt * (LO // 32), len(cores))
    f = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [p.buffer_address(), f, n]
        crt[cx][cy] = [f, n]
        wrt[cx][cy] = [a.buffer_address(), w.buffer_address(), f, n]
        f += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    mode = [ttnn.UnpackToDestMode.Default] * 64
    mode[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = mode
    src = lambda part: os.path.join(KDIR, f"hclo_{part}.cpp")  # noqa: E731
    kernels = [
        ttnn.KernelDescriptor(kernel_source=src("reader"), core_ranges=grid, compile_time_args=[rt] + acc(p),
                              runtime_args=rrt, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("writer"), core_ranges=grid, compile_time_args=acc(a) + acc(w),
                              runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("compute"), core_ranges=grid, compile_time_args=[],
                              runtime_args=crt, config=cfg)]
    cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=size)])
        for i, fmt, size, n in ((0, FP32, 4096, 8), (16, BF16, 2048, 2), (17, FP32, 4096, 2))]
    ttnn.generic_op([p, a, w], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return a, w


HCCOMB = os.environ.get("W51_HCCOMB", "1") != "0"   # 0: the combine as a broadcast multiply and an add


def hc_combine(X, out, w4):
    """X_c += w4_c * out over the 4 streams, in place (tt/kernels/hccomb_*.cpp): X fp32 [1, 4, T, 2560], out [1, 1, T,
    2560], w4 [1, 4, T, 1]; out read once for the 4 streams, X once each way."""
    dev, T = X.device(), X.shape[-2]
    rt, ct = T // 32, D // 32
    w = ttnn.repeat(w4, [1, 1, 1, 32])                     # each row's weight across a tile
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    per, extra = divmod(rt * ct, len(cores))
    f = 0
    for k, (cx, cy) in enumerate(cores):
        n = per + (1 if k < extra else 0)
        rrt[cx][cy] = [X.buffer_address(), out.buffer_address(), w.buffer_address(), f, n]
        crt[cx][cy] = [f, n]
        wrt[cx][cy] = [X.buffer_address(), f, n]
        f += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                       dst_full_sync_en=True)
    mode = [ttnn.UnpackToDestMode.Default] * 64
    mode[0] = mode[1] = mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = mode
    src = lambda part: os.path.join(KDIR, f"hccomb_{part}.cpp")  # noqa: E731
    kernels = [
        ttnn.KernelDescriptor(kernel_source=src("reader"), core_ranges=grid,
                              compile_time_args=[rt, ct] + acc(X) + acc(out) + acc(w), runtime_args=rrt,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("writer"), core_ranges=grid, compile_time_args=[rt, ct] + acc(X),
                              runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=src("compute"), core_ranges=grid, compile_time_args=[ct],
                              runtime_args=crt, config=cfg)]
    cbs = [ttnn.CBDescriptor(total_size=4096 * n, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=FP32, page_size=4096)])
        for i, n in ((0, 8), (1, 2), (2, 8), (16, 8))]
    ttnn.generic_op([X, out, w, X], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return X


class Hc:
    """One hyper-connection site (blk.il.hc_{site}_*), the 4 streams stream-major: X [1, 4, T, 2560] fp32."""

    def __init__(self, W: Weights, il: int, site: str):
        g, p = W.g, f"blk.{il}.hc_{site}_"
        norm = g(p + "norm.weight")                                           # [10240]

        def wdi():
            w = torch.zeros(HC * D, LO)
            w[:, :LR + HC] = (torch.cat([g(p + "down.weight"), g(p + "inject.weight")]) * norm[None, :]).T
            return w.reshape(1, HC, D, LO)
        self.wdi = W(f"blk.{il}.hc_{site}_wdi", wdi)
        self.up = W(f"blk.{il}.hc_{site}_up", lambda: g(p + "up.weight").T.reshape(LR, HC, D).permute(1, 0, 2)
                    .reshape(1, HC, LR, D))
        # the gate-sum's norm rows (1/4 folded in), row-broadcast tiles [1, 4, 32, 2560] fp32
        self.norm_t = W.row((norm / HC).reshape(HC, 1, D).expand(HC, 32, D).reshape(1, HC, 32, D))
        self.dev = W.dev

    def mix(self, X):
        """-> hn in bf16 [1, 1, T, 2560] (what its users read), the inject weights w4 [1, 4, T, 1] (2 sigmoid(inject
        / 4)). With HCGATE: lo's sum4, silu and inject weights in one kernel (hc_lo), the up projection, sigmoid and
        gate-sum in another (hc_gate: the gate never leaves Dest); else the stock ops and G2's gate-sum (hcmix)."""
        xn = ttnn.rms_norm(X, epsilon=EPS, compute_kernel_config=HI)
        p = ttnn.matmul(ttnn.typecast(xn, BF16), self.wdi, dtype=FP32,   # bf16 as the decode
                        compute_kernel_config=CK)                       # the 4 streams' partials [1, 4, T, 352]
        if HCGATE:
            a, w = hc_lo(self.dev, p)
            hn = hc_gate(self.dev, a, self.up, xn, self.norm_t)
            return hn, ttnn.permute(cut(w, 0, HC), [0, 3, 2, 1])
        lo = sum4(p)
        a = ttnn.typecast(ttnn.silu(ttnn.multiply(cut(lo, 0, LR), 1.0 / HC)), BF16)
        gate = ttnn.matmul(ttnn.repeat(a, [1, HC, 1, 1]), self.up, dtype=FP32, compute_kernel_config=CK)
        hn = gate_sum(self.dev, xn, gate, self.norm_t)
        w4 = ttnn.multiply(ttnn.sigmoid(ttnn.multiply(cut(lo, LR, LR + HC), 1.0 / HC)), 2.0)   # [1, 1, T, 4]
        return hn, ttnn.permute(w4, [0, 3, 2, 1])

    @staticmethod
    def combine(X, out, w4):
        """X + w4 * out (with HCCOMB in place: X itself)."""
        if HCCOMB and X.dtype == FP32 and out.dtype == FP32:
            return hc_combine(X, out, w4)
        return ttnn.add(X, ttnn.multiply(w4, out))


class Ffn:
    """The FFN half's dense part: the router (512 logits + the shared expert's gate logit) and the shared expert."""

    def __init__(self, W: Weights, il: int):
        g, p = W.g, f"blk.{il}."

        def router():
            w = torch.zeros(D, 544)
            w[:, :NX], w[:, NX] = g(p + "ffn_gate_inp.weight").T, g(p + "ffn_gate_inp_shexp.weight")
            return w
        self.router = W(f"blk.{il}.router", router, BF16)
        self.gu = W(f"blk.{il}.sh_gu", lambda: torch.cat([g(p + "ffn_gate_shexp.weight").T,
                                                          g(p + "ffn_up_shexp.weight").T], dim=1))
        self.down = W(f"blk.{il}.sh_down", lambda: g(p + "ffn_down_shexp.weight").T)

    def __call__(self, hb):
        """hb: hn in bf16 -> (the router's logits [1, 1, T, 544] fp32, the gated shared expert [1, 1, T, 2560])."""
        logits = lin(hb, self.router)
        gu = lin(hb, self.gu, BF16)
        h = ttnn.multiply(ttnn.silu(cut(gu, 0, FF)), cut(gu, FF, 2 * FF))
        sh = ttnn.multiply(lin(h, self.down), ttnn.sigmoid(cut(logits, NX, NX + 1)))
        return logits, sh


class Gdn:
    """A DeltaNet layer's mixer for chunks, its states carried between them."""
    tick = staticmethod(lambda phase: None)          # the breakdown's synchronized clock (Prefill.run sets it)

    def __init__(self, W: Weights, il: int, t_max: int):
        g, p = W.g, f"blk.{il}."
        self.dev, self.il = W.dev, il
        vperm = torch.tensor(PERM)
        cperm = torch.cat([torch.arange(2 * NK * HK), 2 * NK * HK + (vperm[:, None] * HK + torch.arange(HK)).flatten()])
        zperm = (vperm[:, None] * HK + torch.arange(HK)).flatten()
        self.cperm = cperm

        def proj():
            w = torch.zeros(D, QKV + ZD + 2 * NA)
            w[:, :QKV] = g(p + "attn_qkv.weight").T[:, cperm]
            w[:, QKV:QKV + ZD] = g(p + "attn_gate.weight").T[:, zperm]
            w[:, QKV + ZD:QKV + ZD + NV] = g(p + "ssm_alpha.weight").T[:, vperm]
            w[:, QKV + ZD + NA:QKV + ZD + NA + NV] = g(p + "ssm_beta.weight").T[:, vperm]
            return w
        self.proj = W(f"blk.{il}.gdn_proj", proj)
        self.wo = W(f"blk.{il}.gdn_wo", lambda: g(p + "ssm_out.weight").T[zperm])
        taps = g(p + "ssm_conv1d.weight").T[:, cperm]                              # [4, 10240], row 0 the oldest
        self.conv = CausalConv(W.dev, t_max, QKV, taps, torch.zeros(3, QKV))
        self.taps32 = taps
        self.dt = W.row(g(p + "ssm_dt.bias")[vperm])
        self.neg_a = W.row(g(p + "ssm_a")[vperm])                                 # ssm_a holds -exp(A_log)
        self.o_norm = W.row(g(p + "ssm_norm.weight"))
        self.o_norm16 = ttnn.from_torch(g(p + "ssm_norm.weight").float().reshape(-1), dtype=BF16, layout=TILE,
                                        device=W.dev)                              # (GNORM_OP's weight, rank 1)
        # the carried states, written in place (a traced chunk reads their addresses): S and the conv kernel's
        # history (rows 29..31 of the 32 it reads); a sequence's first chunk reads the zero ones instead (fresh)
        self.S = ttnn.zeros([1, NV, HK, HK], dtype=FP32, layout=TILE, device=self.dev)
        self.st = self.conv.st
        self.zero = None            # (zero S, zero history, the delta rule's constant tiles): Prefill's, shared
        self.reset()

    def reset(self):
        self.fresh = True
        self.tails = []

    def gnorm_f64(self, o, y, T: int, real: int, n: int = 256):
        """W51_GNORM_F64: the chain's and the op's gated norm against float64 on n of the chunk's real rows."""
        z = ttnn.sigmoid(heads(y, NV, HK, QKV))
        chain = tokens(ttnn.typecast(ttnn.multiply(ttnn.multiply(ttnn.rms_norm(ttnn.reshape(o, [1, NV, T, HK]),
            epsilon=EPS, compute_kernel_config=HI), self.o_norm), z), BF16))
        op = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            ttnn.reshape(o, [NV, T, HK]), ttnn.reshape(y, [1, T, y.shape[-1]]), self.o_norm16, num_heads=NV,
            epsilon=EPS, gate_column_offset=QKV, output_dtype=BF16, compute_kernel_config=HI)
        r = torch.linspace(0, real - 1, n).long()                               # (padded rows are all zero)
        ot = ttnn.to_torch(o).double().reshape(NV, T, HK)[:, r]                  # [NV, n, HK]
        zt = ttnn.to_torch(y).double().reshape(T, -1)[r, QKV:QKV + NV * HK].reshape(n, NV, HK).transpose(0, 1)
        w = ttnn.to_torch(self.o_norm).double().reshape(-1)
        ref = ot / torch.sqrt((ot * ot).mean(-1, keepdim=True) + EPS) * w * torch.sigmoid(zt)
        ref = ref.transpose(0, 1).reshape(n, NV * HK)
        out = {}
        for name, t in (("chain", chain), ("op", op)):
            got = ttnn.to_torch(t).double().reshape(T, NV * HK)[r]
            e = (got - ref).abs() / ref.abs().amax(-1, keepdim=True)
            out[name] = {"max_rel": round(float(e.max()), 5), "mean_rel": round(float(e.mean()), 6)}
        print(f"gated norm f64 check (layer {self.il}, {n} rows): {out}", flush=True)

    def mixer(self, hb, ch: "Chunk", more: bool = False):
        """hb [1, 1, T, 2560] bf16 -> ssm_out's output [1, 1, T, 2560] fp32; ch.valid masks the padding rows (beta and
        g zeroed past the real rows, the adapter's valid_len masking, from a tensor the run wrote: no upload, so a trace
        can hold it; that alone makes the padded rows identity updates, and causality keeps them from the real rows:
        q|k|v are not masked, bit for bit the same, 2026-11-09). more: a chunk follows (the conv history is carried)."""
        T = hb.shape[-2]
        zS, zst, consts = self.zero
        y = lin(hb, self.proj, BF16)                                               # [1, 1, T, 16512]
        self.tick("projection")
        self.conv.t, self.conv.rt = T, T // 32
        self.conv.st = zst if self.fresh else self.st
        x = self.conv(y, ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, QKV]), BF16, TILE, self.dev,
                                                        ttnn.DRAM_MEMORY_CONFIG))
        x = ttnn.reshape(x, [1, T, QKV])
        self.tick("conv")
        m = ch.valid
        dram = ttnn.DRAM_MEMORY_CONFIG
        q, k, v = (cut(x, 0, NK * HK), cut(x, NK * HK, 2 * NK * HK), cut(x, 2 * NK * HK, QKV))
        a = ttnn.typecast(cut(y, QKV + ZD, QKV + ZD + NV), FP32)
        b = ttnn.typecast(cut(y, QKV + ZD + NA, QKV + ZD + NA + NV), FP32)
        gl = ttnn.reshape(ttnn.multiply(softplus(ttnn.add(a, self.dt)), self.neg_a), [1, T, NV])
        beta = ttnn.reshape(ttnn.sigmoid(b), [1, T, NV])
        beta, gl = ttnn.multiply(beta, m, memory_config=dram), ttnn.multiply(gl, m, memory_config=dram)
        self.tick("decays, betas")
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter
        s0 = zS if self.fresh else self.S
        if getattr(self, "capture", False):      # check_gdn: the op's inputs and initial state
            self.cap = [ttnn.to_torch(t).double() for t in (x, gl, beta, s0, hb, y)]
        o, S = chunk_gated_delta_rule_fused_adapter(q, k, v, beta, gl, scale=HK ** -0.5, initial_state=s0,
                                                    device=self.dev, qkv_head_dims=(NK, HK, NV, HK),
                                                    return_o_bh=True, const_tiles=consts)
        ttnn.copy(S, self.S)
        ttnn.deallocate(S)
        self.tick("delta rule")
        if self.il == GNORM_F64 and not getattr(Gdn, "f64_done", False):
            Gdn.f64_done = True
            self.gnorm_f64(o, y, T, ch.n)
        if GNORM_OP:      # one op: z read in place from y, o head-first in, token-major out (the dedup audit's C2)
            on = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                ttnn.reshape(o, [NV, T, HK]), ttnn.reshape(y, [1, T, y.shape[-1]]), self.o_norm16, num_heads=NV,
                epsilon=EPS, gate_column_offset=QKV, output_dtype=BF16, compute_kernel_config=HI)
            on = ttnn.reshape(on, [1, 1, T, NV * HK])
        else:
            z = ttnn.sigmoid(heads(y, NV, HK, QKV))
            on = ttnn.multiply(ttnn.multiply(ttnn.rms_norm(ttnn.reshape(o, [1, NV, T, HK]), epsilon=EPS,
                                                           compute_kernel_config=HI), self.o_norm), z)
            on = tokens(ttnn.typecast(on, BF16))                                  # [1, 1, T, 6144], FLA order
        self.tick("gated norm")
        out = lin(on, self.wo)
        self.tick("ssm_out")
        if more:
            ttnn.copy(rows(cut(y, 0, QKV), T - 32, T), self.st)
        return out

    def tail(self, hb, n: int, now: bool):
        """After a chunk (its fresh state spent): the tile rows holding its last 3 real tokens' q|k|v rows (two
        chunks' when this one has fewer than 3), projected again in fp32 as the decode keeps them. now (a traced run):
        read to the host at once (a tensor kept on the card past a trace's replay could sit in its temporaries); else
        at the hand-off (reading here waits for the layer: the expert stream's copies then no longer overlap its dense
        work, an 8K prompt 7.3 -> 8.5 s)."""
        self.fresh = False
        lo = max(n - 3, 0) // 32 * 32
        t = cut(lin(rows(hb, lo, -(-n // 32) * 32), self.proj), 0, QKV)
        self.tails = (self.tails + [(ttnn.to_torch(t).float().reshape(-1, QKV) if now else t, lo, n)])[-2:]

    def conv_state(self) -> torch.Tensor:
        """The last 3 real tokens' q|k|v projection rows in the GGUF's channel order, fp32 [3, 10240] (zeros before
        the prompt's first token)."""
        host = lambda w: w if isinstance(w, torch.Tensor) else ttnn.to_torch(w).float().reshape(-1, QKV)  # noqa: E731
        got = torch.cat([host(w)[:n - lo] for w, lo, n in self.tails])[-3:]
        out = torch.zeros(3, QKV)
        out[3 - len(got):] = got
        st = torch.zeros_like(out)
        st[:, self.cperm] = out
        return st

    def state(self) -> torch.Tensor:
        """S in the GGUF's value-head order, [48, 128 k, 128 v] fp32."""
        s = ttnn.to_torch(self.S).float().reshape(NV, HK, HK)
        out = torch.empty_like(s)
        out[torch.tensor(PERM)] = s
        return out


def rope_table(pos: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [len(pos), 64] (NEOX: the 32 frequencies repeated for both halves) at the given positions."""
    th = torch.from_numpy(pos.astype(np.float64))[:, None] * ROPE_BASE ** (
        -torch.arange(NROT // 2, dtype=torch.float64) * 2 / NROT)
    return torch.cos(th).repeat(1, 2).float(), torch.sin(th).repeat(1, 2).float()


def rope(x, cos, sin):
    """NEOX RoPE on the first 64 dims of head-major x [1, h, T, d]: x1 c - x2 s | x2 c + x1 s."""
    d = x.shape[-1]
    x1, x2 = cut(x, 0, NROT // 2), cut(x, NROT // 2, NROT)
    y = ttnn.add(ttnn.multiply(cut(x, 0, NROT), cos), ttnn.multiply(ttnn.concat([ttnn.neg(x2), x1], dim=-1), sin))
    return ttnn.concat([y, cut(x, NROT, d)], dim=-1)


class Attn:
    """A QSA attention layer's mixer for chunks; the chunks' K, V (bf16, tiles), raw indexer keys and pooled keys
    (fp32) kept for the next chunk and the hand-off."""
    tick = staticmethod(lambda phase: None)          # the breakdown's synchronized clock (Prefill.run sets it)

    def __init__(self, W: Weights, il: int):
        g, p = W.g, f"blk.{il}."
        self.dev, self.il = W.dev, il
        self.proj = W(f"blk.{il}.attn_proj", lambda: torch.cat([g(p + "attn_q.weight"), g(p + "attn_k.weight"),
                                                                g(p + "attn_v.weight"), g(p + "indexer.q_proj.weight"),
                                                                g(p + "indexer.k_proj.weight")]).T)   # [2560, 13952]
        # q | gate | k | v projected to bf16 (each is normalized or rounded to bf16 next), the indexer's q | k in fp32
        # (its raw keys are the decode's fp32 cache)
        m = 2 * NH * HD + 2 * NKV * HD
        self.proj_m, self.proj_i = cut(self.proj, 0, m), cut(self.proj, m, m + IH * IHD + IHD)
        ttnn.deallocate(self.proj)
        self.wo = W(f"blk.{il}.attn_wo", lambda: g(p + "attn_output.weight").T)
        self.n = {k: W.row(g(p + f)) for k, f in (("q", "attn_q_norm.weight"), ("k", "attn_k_norm.weight"),
                                                  ("iq", "indexer.q_norm.weight"), ("ik", "indexer.k_norm.weight"))}
        self.into = None            # the decode's attention program 1 (A1) this layer writes into (Prefill.prefill)
        self.reset()

    def reset(self):
        self.K = self.V = self.rawk = self.pk = None

    def pre(self, hb, chunk: "Chunk") -> dict:
        """The chunk's queries, gate, keys and values, its raw indexer keys and pooled keys (no position-dependent
        writes: traceable)."""
        T = hb.shape[-2]
        y = lin(hb, self.proj_m, BF16)                                            # [1, 1, T, 13312]
        yi = lin(hb, self.proj_i)                                                 # [1, 1, T, 640] fp32
        self.tick("projection")
        o = 2 * NH * HD
        # q and gate head-major [1, 24, T, 256] each in one op (nlp_create_q_heads_split: bit for bit, 1.04 against
        # heads + 2 cuts' 3.15 ms at an 8K chunk; the dedup audit's K2)
        qh, gh = ttnn.experimental.nlp_create_q_heads_split(cut(y, 0, o), num_heads=NH, split_head_dim=HD)
        q = rope(ttnn.multiply(ttnn.rms_norm(qh, epsilon=EPS, compute_kernel_config=HI), self.n["q"]),
                 chunk.cos, chunk.sin)
        q = ttnn.typecast(ttnn.multiply(q, 1.0 / 16), BF16)
        gate = ttnn.sigmoid(gh)
        k = rope(ttnn.multiply(ttnn.rms_norm(heads(y, NKV, HD, o), epsilon=EPS,
                                             compute_kernel_config=HI), self.n["k"]), chunk.cos, chunk.sin)
        v = heads(y, NKV, HD, o + NKV * HD)
        iq = rope(ttnn.multiply(ttnn.rms_norm(heads(yi, IH, IHD), epsilon=EPS,
                                              compute_kernel_config=HI), self.n["iq"]), chunk.cos, chunk.sin)
        ik = cut(yi, IH * IHD, IH * IHD + IHD)                                    # [1, 1, T, 128] raw
        # the chunk's pooled keys: mean of 4 raw keys, RMSNorm, RoPE at the block's first cell
        pk = ttnn.multiply(ttnn.sum(ttnn.reshape(ttnn.to_layout(ik, RM), [1, T // KPOOL, KPOOL, IHD]), dim=-2,
                                    keepdim=True), 1.0 / KPOOL)
        pk = ttnn.to_layout(ttnn.reshape(pk, [1, 1, T // KPOOL, IHD]), TILE)
        pk = rope(ttnn.multiply(ttnn.rms_norm(pk, epsilon=EPS, compute_kernel_config=HI), self.n["ik"]),
                  chunk.pcos, chunk.psin)
        self.tick("norms, rope, pooled keys")
        return dict(q=q, gate=gate, k=ttnn.typecast(k, BF16), v=ttnn.typecast(v, BF16), iq=iq, ik=ik, pk=pk)

    def write(self, t: dict, chunk: "Chunk"):
        """The chunk's keys kept (eager: the page copies' jobs depend on its position and length). 256K plan step 1:
        with self.into (the decode's A1) its K and V (all T cells, padding included), the raw keys of its last RAWK
        positions and its pools go straight into the decode's caches; without, onto the prefill's own."""
        A, dev, p0, T, n = self.into, self.dev, chunk.p0, chunk.T, chunk.n
        if A is None:
            cat = lambda a, b: b if a is None else ttnn.concat([a, b], dim=2)  # noqa: E731
            self.K, self.V = cat(self.K, t["k"]), cat(self.V, t["v"])
            self.rawk, self.pk = cat(self.rawk, t["ik"]), cat(self.pk, t["pk"])
        else:
            for src, dst in ((t["k"], A.kc), (t["v"], A.vc)):
                r = ttnn.to_layout(src, RM)
                page_copy(dev, r, dst, [(h * T, h * CTX + p0, T) for h in range(NKV)], HD * 2)
                ttnn.deallocate(r)
            r = ttnn.to_layout(t["ik"], RM)
            page_copy(dev, r, A.rawk, [(q - p0, q % RAWK, 1) for q in range(max(p0, p0 + n - RAWK), p0 + n)],
                      IHD * 4)
            ttnn.deallocate(r)
            # the pools in whole tiles of 32: a chunk from p0 inside a tile (a follow-up from any multiple of 4,
            # 2026-11-10) gets the tile's held pools in front (read back, written again as they were)
            m, pk, tiles = p0 // KPOOL, t["pk"], T // KPOOL // 32
            if m % 32:
                head = ttnn.empty([1, 1, 32, IHD], FP32, TILE, dev)
                page_copy(dev, A.pooled, head, [(m // 32 * 4, 0, 4)], 4096)
                h = ttnn.to_layout(head, RM)
                pk = ttnn.to_layout(ttnn.concat([rows(h, 0, m % 32), ttnn.to_layout(pk, RM), rows(h, 0, 32 - m % 32)],
                                                dim=2), TILE)
                ttnn.deallocate(head)
                tiles = min(tiles + 1, page_count(A.pooled) // 4 - m // 32)   # (a tile past the context: none)
            page_copy(dev, pk, A.pooled, [(0, m // 32 * 4, tiles * 4)], 4096)
        self.tick("caches")

    def stage_rows(self, L: int):
        """The sparse path's [V_h | K_h] rows, row-major [1, 1, L, 2 HD] per KV head, straight from the decode's caches
        (no tile copies of K and V: at 104K those held ~213 MB beside these, and the served engine's prefill ran out
        of memory for the experts' 839 MB buffers after a run of requests, 2026-11-09), and the pooled keys."""
        A, dev = self.into, self.dev
        # the rows allocated before the layer's temporaries (as qsa_alloc's mask) and filled in place: a concatenation's
        # output allocated after its inputs left the banks' largest free block 4 MB short of the experts' 839 MB
        kvp = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, L, 2 * HD]), BF16, RM, dev, ttnn.DRAM_MEMORY_CONFIG)
               for _ in range(NKV)]
        for h in range(NKV):          # V row i to half-row 2 i, K row i to 2 i + 1: written in place, no temporaries
            page_copy(dev, A.vc, kvp[h], [(h * CTX, 0, L)], HD * 2, dst_stride=2)
            page_copy(dev, A.kc, kvp[h], [(h * CTX, 1, L)], HD * 2, dst_stride=2)
        p = ttnn.empty([1, 1, L // KPOOL, IHD], FP32, TILE, dev)
        page_copy(dev, A.pooled, p, [(0, 0, L // KPOOL // 32 * 4)], 4096)
        return kvp, p

    def stage(self, L: int):
        """The layer's prefix [0, L) staged back from the decode's caches in the tile layouts SDPA and the selection
        read: the bytes a concatenation held, in temporaries freed after the layer, instead of a second copy of every
        layer's context."""
        A, dev = self.into, self.dev
        staged = []
        for src in (A.kc, A.vc):
            r = ttnn.empty([1, NKV, L, HD], BF16, RM, dev)
            page_copy(dev, src, r, [(h * CTX, h * L, L) for h in range(NKV)], HD * 2)
            staged.append(ttnn.to_layout(r, TILE))
            ttnn.deallocate(r)
        p = ttnn.empty([1, 1, L // KPOOL, IHD], FP32, TILE, dev)
        page_copy(dev, A.pooled, p, [(0, 0, L // KPOOL // 32 * 4)], 4096)
        return staged[0], staged[1], p

    def post(self, t: dict, chunk: "Chunk"):
        """Attention over the prefix [0, L) (its masks from the chunk's position tensors: traceable) -> wo's output."""
        q, iq, L = t["q"], t["iq"], chunk.L
        sparse = SPARSE and not chunk.causal_only and L >= SPARSE_MIN_L
        if sparse and self.into is not None and not SPARSE_F64:
            kvp, PK = self.stage_rows(L)
            K = V = None
        else:
            K, V, PK = self.stage(L) if self.into is not None else (self.K, self.V, self.pk)
            kvp = None
        sdpa = lambda q, mask: ttnn.transformer.scaled_dot_product_attention(  # noqa: E731
            q, K, V, attn_mask=mask, is_causal=False, scale=1.0, program_config=chunk.sdpa_pc,
            compute_kernel_config=SDPA_CK)
        cell = cut(chunk.cells[0], 0, L)
        if sparse:
            if kvp is None:                               # the rows [V_h | K_h] from the tile K, V (no caches / F64)
                kvp = []
                for h in range(NKV):
                    hk = lambda x: ttnn.to_layout(ttnn.slice(x, [0, h, 0, 0], [1, h + 1, L, HD]), RM)  # noqa: E731
                    kvp.append(ttnn.concat([hk(V), hk(K)], dim=-1))
            if SPARSE_F64:   # the check's inputs (held only then: on every layer they kept ~200 MB alive, 2026-11-09)
                self.f64_K, self.f64_V, self.f64_iq, self.f64_pk = K, V, iq, PK
            parts = [self.sparse(q, iq, PK, kvp, chunk, sub, cell) for sub in chunk.subs]
            for x in kvp:
                ttnn.deallocate(x)
            a = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        elif chunk.causal_only:
            mask = ttnn.typecast(ttnn.multiply(ttnn.subtract(ttnn.le(cell, chunk.t_c), 1.0), 1e4), BF16)
            self.tick("selection")
            a = sdpa(q, mask)
            self.tick("flash attention")
        else:   # the queries in sub-blocks (Chunk.subs: each one's masks [Tq, L] at most 8K x 16K cells)
            parts = []
            for sub in chunk.subs:
                mask = self.select(iq, PK, chunk, sub, cell)
                self.tick("selection")
                if self.il in MASKSTAT and (not MASKSTAT_AT or sub[0] == 0 and any(
                        chunk.p0 <= a < chunk.p0 + chunk.T for a in MASKSTAT_AT)):
                    mask_stats(self.il, chunk, sub, mask)
                parts.append(sdpa(rows(q, sub[0], sub[1]), mask))
                self.tick("flash attention")
            a = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        if self.into is not None:                    # the staged prefix: this layer's only
            for x in (K, V, PK):
                if x is not None:
                    ttnn.deallocate(x)
        a = ttnn.multiply(a, t["gate"])                                          # [1, 24, T, 256]
        out = lin(tokens(ttnn.typecast(a, BF16)), self.wo)
        self.tick("gate, wo")
        return out

    def sparse(self, q, iq, pk, kvp: list, chunk: "Chunk", sub: tuple, cell):
        """W51_SPARSE: the sub-block's attention [1, NH, Tq, HD] by sparse_sdpa per KV head over each query's selected
        cells (select's indices); q the chunk's [1, NH, T, HD] (tile), kvp the layer's [V_h | K_h] rows."""
        idx, want = self.select(iq, pk, chunk, sub, cell, topk=SPARSE_TOPK)
        if SPARSE_CHECK:
            w = ttnn.to_torch(want).reshape(-1, 16)[:, :2].to(torch.int64)
            SPARSE_STATS["rows"] += len(w)
            SPARSE_STATS["tie_rows"] += int((w[:, 0] != w[:, 1]).sum())
            SPARSE_STATS["max_mask_cells"] = max(SPARSE_STATS["max_mask_cells"], int(w[:, 0].max()))
            SPARSE_STATS["max_cells"] = max(SPARSE_STATS["max_cells"], int(w[:, 1].max()))
        ttnn.deallocate(want)
        self.tick("selection")
        s0, s1 = sub
        Tq, G = s1 - s0, NH // NKV
        # [0 | q] for the sub-block's queries, row-major (a head's rows contiguous): its temporaries ~50 MB at 60K,
        # where the whole chunk's held ~600 MB and fragmented the banks before the experts' 839 MB buffers
        # [0 | q] and the heads to 32 by device-side pads (row-major: front padding) — no host-made zeros: a traced
        # follow-up's capture refused their writes ("Writes are not supported during trace capture", 2026-11-09); and
        # qs is never deallocated (with one sub-block over the chunk it is q itself)
        qz = ttnn.pad(ttnn.to_layout(rows(q, s0, s1), RM), [(0, 0), (0, 0), (0, 0), (HD, 0)], 0.0)  # [1, NH, Tq, 2 HD]
        outs = []
        for h in range(NKV):
            qh = ttnn.pad(ttnn.slice(qz, [0, h * G, 0, 0], [1, (h + 1) * G, Tq, 2 * HD]),
                          [(0, 0), (0, 32 - G), (0, 0), (0, 0)], 0.0)
            o = ttnn.transformer.sparse_sdpa(qh, kvp[h], idx, HD, kv_format=ttnn.transformer.SparseKVFormat.BF16,
                                             scale=1.0, compute_kernel_config=SPARSE_CK)
            ttnn.deallocate(qh)
            outs.append(ttnn.slice(o, [0, 0, 0, 0], [1, G, Tq, HD]))
            ttnn.deallocate(o)
        a = ttnn.to_layout(ttnn.concat(outs, dim=1), TILE)
        if SPARSE_F64 and self.il == SPARSE_F64[0] and chunk.p0 + s0 <= SPARSE_F64[1] < chunk.p0 + s1 and "f64" not in SPARSE_STATS:
            self.f64_check(q, kvp, chunk, sub, cell, idx, a)
        for x in (idx, qz):
            ttnn.deallocate(x)
        self.tick("flash attention")
        return a

    def f64_check(self, q, kvp, chunk, sub, cell, idx, sparse_out, n: int = 64):
        """W51_SPARSE_F64: the sub-block's dense SDPA (its mask from select) and sparse_out against float64 on n rows."""
        s0, s1 = sub
        mask = self.select(self.f64_iq, self.f64_pk, chunk, sub, cell)
        qs = rows(q, s0, s1)
        K = ttnn.to_torch(self.f64_K).double()[0]                          # [NKV, L, HD]
        V = ttnn.to_torch(self.f64_V).double()[0]
        dense = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
            qs, self.f64_K, self.f64_V, attn_mask=mask, is_causal=False, scale=1.0, program_config=chunk.sdpa_pc,
            compute_kernel_config=SDPA_CK)).double()[0]                     # [NH, Tq, HD]
        keep = ttnn.to_torch(mask).reshape(s1 - s0, -1) == 0
        sp = ttnn.to_torch(sparse_out).double()[0]
        qh = ttnn.to_torch(qs).double()[0]
        ix = ttnn.to_torch(idx).reshape(s1 - s0, -1).to(torch.int64) & 0xFFFFFFFF
        G, errs = NH // NKV, {"dense": [], "sparse": []}
        for r in torch.linspace(0, s1 - s0 - 1, n).long().tolist():
            c = ix[r][ix[r] != 0xFFFFFFFF]
            if int(keep[r].sum()) != len(c):                               # a tie row: the two rules differ
                continue
            for h in range(NH):
                kh = h // G
                p = torch.softmax(qh[h, r] @ K[kh, c].T, 0)
                ref = p @ V[kh, c]
                den = ref.abs().max()
                errs["dense"].append(float((dense[h, r] - ref).abs().max() / den))
                errs["sparse"].append(float((sp[h, r] - ref).abs().max() / den))
        SPARSE_STATS["f64"] = {"layer": self.il, "rows": [chunk.p0 + s0, chunk.p0 + s1], "L": chunk.L,
                               "max_rel": {k: round(max(v), 5) for k, v in errs.items()},
                               "mean_rel": {k: round(sum(v) / len(v), 6) for k, v in errs.items()},
                               "sparse_closer_share": round(sum(b < a for a, b in zip(errs["dense"], errs["sparse"]))
                                                            / len(errs["dense"]), 3), "samples": len(errs["dense"])}
        print(f"sparse f64 check: {SPARSE_STATS['f64']}", flush=True)
        ttnn.deallocate(mask)

    def select(self, iq, pk, chunk: "Chunk", sub: tuple, cell, topk: int = 0):
        """The QSA mask [1, 1, Tq, L] of the sub-block's queries (0 attended, -1e4 not): per query the top min(512,
        complete pools) pools by the indexer's score (sum over 4 heads of relu(q_h . k_p) / sqrt(128)), as cells, and
        its own incomplete block's cells up to itself. The masks' parts are built per sub-block (at 64K cells
        a chunk's would take ~2 GB), the pools' picks repeated to cells."""
        s0, s1 = sub
        L = chunk.L
        t_c, t_pool, t_open = (rows(x, s0, s1) for x in (chunk.t_c, chunk.t_pool, chunk.t_open))
        pool_end, cell_pool = cut(chunk.cells[2], 0, L // KPOOL), cut(chunk.cells[1], 0, L)
        out = qsa_alloc(iq.device(), s1 - s0, L) if QSAK and not topk else None   # first: under the temporaries
        assert QSAK or not topk, "the sparse attention's indices need pf_qsa_mask.cpp (W51_QSAK=1)"
        if SCOREK:
            score, kq = qsa_score(iq, pk, s0, s1), None
        elif SEL_FAST:   # per head, relu in the matmul's packer, pairs summed as they come: (h0 + h1) + (h2 + h3)
            iqs = rows(iq, s0, s1)
            head = lambda h: ttnn.matmul(ttnn.slice(iqs, [0, h, 0, 0], [1, h + 1, s1 - s0, IHD]), pk,  # noqa: E731
                                         transpose_b=True, dtype=FP32, compute_kernel_config=HI, activation="relu")
            pair = []
            for h in (0, 2):
                a, b = head(h), head(h + 1)
                pair.append(ttnn.add(a, b))
                ttnn.deallocate(a)
                ttnn.deallocate(b)
            kq = ttnn.add(pair[0], pair[1])
            for x in pair:
                ttnn.deallocate(x)
            score = ttnn.multiply(kq, IHD ** -0.5)       # [1, 1, Tq, L / 4]
        else:
            kq = ttnn.relu(ttnn.matmul(rows(iq, s0, s1), pk, transpose_b=True, dtype=FP32, compute_kernel_config=HI))
            score = ttnn.multiply(sum4(kq), IHD ** -0.5)     # [1, 1, Tq, L / 4]
        if QSAK:                                         # the rest in one kernel (2026-11-08)
            if kq is not None:
                ttnn.deallocate(kq)
            mask = qsa_mask(score, L, chunk.p0 + s0, out=out, topk=topk)
            ttnn.deallocate(score)
            return mask
        vis = ttnn.le(pool_end, t_c)                     # the complete pools
        score = ttnn.add(score, ttnn.multiply(ttnn.subtract(vis, 1.0), 1e9))
        vals, _ = ttnn.topk(score, k=NSEL, dim=-1)
        kth = ttnn.typecast(cut(vals, NSEL - 1, NSEL), FP32)
        pick = ttnn.typecast(ttnn.multiply(ttnn.ge(score, kth), vis), BF16)
        cells = ttnn.repeat_interleave(pick, KPOOL, dim=3)                       # [1, 1, Tq, L]
        own = ttnn.typecast(ttnn.multiply(ttnn.multiply(ttnn.eq(cell_pool, t_pool), ttnn.le(cell, t_c)),
                                          t_open), BF16)
        return ttnn.typecast(ttnn.multiply(ttnn.subtract(ttnn.maximum(cells, own), 1.0), 1e4), BF16)


class MtpKv:
    """The MTP draft layer's catch-up K/V over many cells at once (2026-11-10, engine.SpecModel.draft): its front
    (tt/spec/mtp.py Mtp.front), its attention's hyper-connection read and k / v as this file's ops over blocks of rows,
    written into the MTP's caches (MtpCard.A1's) as Attn.write does. g6_mtp's 4-row programs cost 0.25 ms of card time
    a call (every weight read for 4 rows), 0.44 s at 6,300 cells. Not their bits: the target's own prefill K/V differ
    from its decode programs' the same way."""
    # rows a block, the largest that fits first: a few shapes (each new one compiled ~5-9 s in the request path)
    BLOCKS = (2048, 256, 32)           # (2048: X fp32 [1, 4, 2048, 2560] 84 MB)

    def __init__(self, dev, A1, host):
        import g6_mtp
        dense = g6_mtp.M.load_dense()
        g, L = g6_mtp.MtpGguf(dense), g6_mtp.L
        W = Weights(dev, g)                # g's model tag is the MTP's own (its dense.bin): its own cache folder
        t = lambda k: torch.from_numpy(dense[k])  # noqa: E731
        self.dev, self.A1, self.host = dev, A1, host
        self.hc = Hc(W, 0, "attn")
        self.wh = W("mtp_fc_hidden", lambda: t("fc_hidden.weight").T)              # [2560, 2560]
        self.we = W("mtp_fc_embedding", lambda: t("fc_embedding.weight").T)
        self.kv = W("mtp_attn_kv", lambda: torch.cat([g(L + "attn_k.weight"), g(L + "attn_v.weight")]).T)  # [2560, 1024]
        self.nh, self.ne = W.row(t("pre_fc_norm_hidden.weight")), W.row(t("pre_fc_norm_embedding.weight"))
        self.nk = W.row(g(L + "attn_k_norm.weight"))

    def __call__(self, R, cf: int, c0: int, n: int, nxt) -> None:
        """Cells c0 .. c0 + n - 1: R the target's rows on the card from cell cf (Prefill.rows_card's form, cf a multiple
        of 32, c0 - cf < 32), nxt the tokens at c0 + 1 .. c0 + n."""
        A, dev, rn = self.A1, self.dev, HC * D // 32          # 320 pages a row
        a, end = c0 - cf - (c0 - cf) % 32, -(-(c0 - cf + n) // 32) * 32
        while a < end:
            T = next(s for s in self.BLOCKS if s <= end - a)
            b = a + T
            w0, w1 = max(c0, cf + a), min(c0 + n, cf + b)                      # the cells written
            toks = [nxt[c - c0] if c0 <= c < c0 + n else 0 for c in range(cf + a, cf + b)]
            e = ttnn.from_torch(torch.from_numpy(np.stack([self.host.embed(v)[:D] for v in toks])).reshape(1, 1, T, D),
                                dtype=FP32, layout=TILE, device=dev)
            e = lin(ttnn.typecast(ttnn.multiply(ttnn.rms_norm(e, epsilon=EPS, compute_kernel_config=HI), self.ne),
                                  BF16), self.we)
            r = ttnn.to_layout(ttnn.reshape(ttnn.slice(R, [0, 0, a * rn, 0], [1, 1, b * rn, 32]), [1, 1, T, HC * D]),
                               TILE)
            hn = ttnn.typecast(ttnn.multiply(ttnn.rms_norm(r, epsilon=EPS, compute_kernel_config=HI), self.nh), BF16)
            ttnn.deallocate(r)
            X = ttnn.concat([ttnn.add(lin(cut(hn, s * D, (s + 1) * D), self.wh), e) for s in range(HC)], dim=1)
            hb, _ = self.hc.mix(X)                                             # [1, 1, T, 2560] bf16
            y = lin(hb, self.kv, BF16)
            cos, sin = (ttnn.from_torch(v.reshape(1, 1, T, NROT), dtype=FP32, layout=TILE, device=dev)
                        for v in rope_table(np.arange(cf + a, cf + b)))
            k = rope(ttnn.multiply(ttnn.rms_norm(heads(y, NKV, HD), epsilon=EPS, compute_kernel_config=HI), self.nk),
                     cos, sin)
            for src, dst in ((k, A.kc), (heads(y, NKV, HD, NKV * HD), A.vc)):
                rm = ttnn.to_layout(ttnn.typecast(src, BF16), RM)
                page_copy(dev, rm, dst, [(h * T + w0 - cf - a, h * CTX + w0, w1 - w0) for h in range(NKV)], HD * 2)
                ttnn.deallocate(rm)
            a = b

    def warm(self) -> None:
        """Each block shape once (its programs compiled): zero rows into cells 0 .. 2,335 of the MTP's caches (a
        sequence's catch-up writes the cells its window reads)."""
        n = sum(self.BLOCKS)
        R = ttnn.zeros([1, 1, n * HC * D // 32, 32], dtype=FP32, layout=RM, device=self.dev)
        self(R, 0, 0, n, [0] * n)
        ttnn.deallocate(R)


class Chunk:
    """A chunk's geometry, shared by its layers: positions, RoPE rows, the padding mask, the attention masks' parts."""

    def __init__(self, dev, p0: int, T: int, n: int, cells: list, lpad: int = 0, B: dict | None = None):
        """lpad: the attention's cells L rounded up to a multiple of it (cells past p0 + T masked): a follow-up's prefix
        length is any multiple of 128, and each new one compiled ~6 s of programs; rounded, a few shapes, compiled once
        (tt-metal's cache on disk keeps them across processes). cells: Prefill's rows over the whole context (cell,
        its pool, each pool's last cell), sliced to L by the layers. B: the tensors are written into B's of the same
        name when it has them (a traced run's buffers: the traces read their addresses), else made there."""
        self.p0, self.T, self.n, self.L = p0, T, n, (-(-(p0 + T) // lpad) * lpad if lpad else p0 + T)
        assert self.L <= cells[0].shape[-1], (self.L, cells[0].shape)
        self.cells, B = cells, ({} if B is None else B)

        def put(name, a, dt=FP32):
            h = ttnn.from_torch(a.contiguous(), dtype=dt, layout=TILE)
            if name in B:
                ttnn.copy_host_to_device_tensor(h, B[name])
            else:
                B[name] = ttnn.to_device(h, dev)
            return B[name]
        pos = np.arange(p0, p0 + T)
        cos, sin = rope_table(pos)
        self.cos, self.sin = put("cos", cos.reshape(1, 1, T, NROT)), put("sin", sin.reshape(1, 1, T, NROT))
        pcos, psin = rope_table(np.arange(p0, p0 + T, KPOOL))
        self.pcos, self.psin = put("pcos", pcos.reshape(1, 1, -1, NROT)), put("psin", psin.reshape(1, 1, -1, NROT))
        col = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32).reshape(1, 1, -1, 1)  # noqa: E731
        self.t_c, self.t_pool = put("t_c", col(pos)), put("t_pool", col(pos // KPOOL))
        self.t_open = put("t_open", col((pos % KPOOL != KPOOL - 1).astype(np.float32)))
        valid = (torch.arange(T) < n).float().reshape(1, T, 1)          # the delta rule's padding mask
        self.valid = put("valid", valid)
        # every query's visible pools number at most ((p0 + T) / 4) <= 512: QSA keeps them all, the mask is causal
        self.causal_only = (p0 + T) // KPOOL <= NSEL
        self.subs = []
        if not self.causal_only:
            # a sub-block's largest temporaries are fp32 [Tq, L], several at once: an eighth of the largest free
            # block now (256K plan step 1: a 61K prompt at 256K failed allocating one with 2 GB free in 35 MB pieces;
            # the layers' own temporaries fragment it further after this reading)
            mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
            room = mv.largest_contiguous_bytes_free_per_bank * mv.num_banks // 8 // (4 * self.L)
            # the sparse path builds no [Tq, L] mask: its peak is the score's ~3 L bytes a query (SEL_FAST), not 4 L
            # bytes a temporary: 4x the queries a sub-block
            k = 4 if SEL_FAST and SPARSE and self.L >= SPARSE_MIN_L else 1
            tq = max(128, min(T, k * SUB_CELLS // self.L, k * room) // 128 * 128)
            self.subs = [(s0, min(T, s0 + tq)) for s0 in range(0, T, tq)]
        self.sdpa_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
                                              q_chunk_size=128, k_chunk_size=SDPA_K, exp_approx_mode=False)

    def key(self) -> tuple:
        """What a traced attention layer's second part bakes in."""
        return (self.T, self.L, self.causal_only, tuple(self.subs))


def context_cells(dev, ctx: int) -> list:
    """Rows [1, 1, 1, ctx] (fp32): every cell's index and its pool's, and [1, 1, 1, ctx / 4] each pool's last cell."""
    row = lambda v: ttnn.from_torch(torch.as_tensor(v, dtype=torch.float32).reshape(1, 1, 1, -1), dtype=FP32,  # noqa: E731
                                    layout=TILE, device=dev)
    cell = np.arange(ctx)
    return [row(cell), row(cell // KPOOL), row(np.arange(ctx // KPOOL) * KPOOL + KPOOL - 1)]


class Ple:
    """Layer 1's PLE block for chunks; the last 9 tokens' n carried for the dilated conv."""

    def __init__(self, W: Weights, layer: int = 1):
        g, p = W.g, f"blk.{layer}."
        self.dev = W.dev
        self.w = W(f"blk.{layer}.ple_kv", lambda: torch.cat([g(p + "ple_key.weight"), g(p + "ple_value.weight")]).T)
        self.nk, self.nq, self.nc = (W.row(g(p + f"ple_norm_{x}.weight").reshape(1, HC, 1, D))
                                     for x in ("key", "query", "conv"))
        taps = g(p + "ple_conv1d.weight").T.contiguous()                          # [4, 10240], row 0 the oldest
        self.taps = [W.row(taps[k].reshape(1, HC, 1, D)) for k in range(4)]
        # the last 9 tokens' n, written in place (a traced run's later layers stay off its address); a sequence's
        # first chunk reads the zero rows
        self.hist = ttnn.zeros([1, HC, 9, D], dtype=FP32, layout=RM, device=self.dev)
        self.zhist = ttnn.zeros([1, HC, 9, D], dtype=FP32, layout=RM, device=self.dev)
        self.reset()

    def reset(self):
        self.fresh = True

    def __call__(self, X, emb, n: int):
        """X [1, 4, T, 2560] fp32, emb [1, 1, T, 2560] bf16 (the host's gathered rows) -> X after the block."""
        T = X.shape[-2]
        y = lin(emb, self.w)                                                      # [1, 1, T, 12800]
        key = heads(y, HC, D)                                                     # [1, 4, T, 2560]
        value = cut(y, HC * D, HC * D + D)
        s = ttnn.sum(ttnn.multiply(ttnn.multiply(ttnn.rms_norm(key, epsilon=EPS, compute_kernel_config=HI), self.nk),
                                   ttnn.multiply(ttnn.rms_norm(X, epsilon=EPS, compute_kernel_config=HI), self.nq)),
                     dim=-1, keepdim=True)
        s = ttnn.multiply(s, D ** -0.5)                                           # [1, 4, T, 1]
        gate = ttnn.sigmoid(ttnn.multiply(ttnn.sign(s), ttnn.sqrt(ttnn.maximum(ttnn.abs(s), 1e-6))))
        gated = ttnn.multiply(value, gate)                                        # [1, 4, T, 2560]
        nn = ttnn.multiply(ttnn.rms_norm(gated, epsilon=EPS, compute_kernel_config=HI), self.nc)
        hist = ttnn.concat([self.zhist if self.fresh else self.hist, ttnn.to_layout(nn, RM)], dim=2)  # [1, 4, 9 + T, 2560]
        conv = None
        for k in range(4):
            term = ttnn.multiply(ttnn.to_layout(ttnn.slice(hist, [0, 0, 3 * k, 0], [1, HC, 3 * k + T, D]), TILE),
                                 self.taps[k])
            conv = term if conv is None else ttnn.add(conv, term)
        ttnn.copy(ttnn.slice(hist, [0, 0, n, 0], [1, HC, n + 9, D]), self.hist)    # the last 9 real tokens' n
        self.fresh = False
        return ttnn.add(X, ttnn.add(gated, ttnn.silu(conv)))

    def ring_rows(self) -> torch.Tensor:
        """The last 9 real tokens' n, oldest first, fp32 [9, 10240] (stream-major per token)."""
        return ttnn.to_torch(self.hist).float().reshape(HC, 9, D).permute(1, 0, 2).reshape(9, HC * D)


class CpuExperts:
    """The routed experts on the CPU lane (cl_moe_m, up to the lane's max_m tokens a call, each expert of a call's
    union read once for all its tokens), routed by the lane's cl_route from the card's logits: every token's routes
    and sum are the decode lane's to the bit, whatever the call's size."""

    def __init__(self, dev, lane):
        self.dev, self.lane = dev, lane
        self.rows, self.outs = HostRows(dev), HostOut(dev)
        self.seconds = 0.0
        self.parts = dict(read=0.0, route=0.0, lane=0.0, write=0.0)   # host seconds by part (the read waits for the card)

    def __call__(self, il: int, hb, logits, n: int):
        t0 = time.perf_counter()
        T = hb.shape[-2]
        # tiles to rows on the card, so the host only copies (its untilize was ~40 ms a layer at 512 tokens), into
        # buffers kept per shape (HostRows, HostOut: fresh arrays' page faults slowed the reads and the lane)
        x = self.rows(hb, n)
        lg = self.rows(logits, n)
        t1 = time.perf_counter()
        ids, w = self.lane.route_m(lg, TOPK)          # cl_route: the decode's routing, to the bit
        out = self.outs.buffer(T, n)
        t2 = time.perf_counter()
        step = getattr(self.lane, "max_m", 8)
        for i in range(0, n, step):
            j = min(n, i + step)
            self.lane.moe_m(il, np.ascontiguousarray(x[i:j]), np.ascontiguousarray(ids[i:j]),
                            np.ascontiguousarray(w[i:j]), out[i:j])
        t3 = time.perf_counter()
        y = self.outs.upload(T)
        t4 = time.perf_counter()
        for k, v in zip(("read", "route", "lane", "write"), (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            self.parts[k] += v
        self.seconds += t4 - t0
        return y


INPUT_THREADS = int(os.environ.get("W51_INPUT_THREADS", 32))   # token_inputs' (the lane's CPUs idle then)


def token_inputs(host, ids: list[int], p0: int, n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """X's init (the embedding row) and the PLE input of tokens p0 .. p0 + n - 1 of ids, fp32 [T, 2560] each (rows
    past n zero): Host's C++ gather (cpulane.Gather, the decode's own bits) called directly, 8 threads (ctypes lets
    the GIL go during each call: the PLE rows' page faults overlap)."""
    from concurrent.futures import ThreadPoolExecutor
    G = host.gather
    assert G is not None, "token_inputs: the lane's gather library is not built"
    x0, emb = np.zeros((T, D), np.float32), np.zeros((T, D), np.float32)
    a = np.ascontiguousarray(ids, np.int64)
    if INPUT_THREADS and hasattr(G.lib, "gt_inputs"):          # one call, its own threads (0: the Python loop)
        if G.lib.gt_inputs(a.ctypes.data, p0, n, G.n_prev, x0.ctypes.data, emb.ctypes.data, INPUT_THREADS):
            raise IndexError("token_inputs: a token outside the embedding")
        return x0, emb

    def part(r):
        bad, x4 = 0, np.empty(HC * D, np.float32)           # gt_embed writes the 4 streams; one is kept
        for i in r:
            q = p0 + i
            tok, s = int(a[q]), max(0, q - G.n_prev)
            bad |= G.lib.gt_embed(tok, x4.ctypes.data)
            x0[i] = x4[:D]
            bad |= G.lib.gt_ple(tok, a.ctypes.data + 8 * s, q - s, emb.ctypes.data + i * D * 4)
        return bad
    with ThreadPoolExecutor(8) as ex:
        if any(ex.map(part, [range(k, n, 8) for k in range(8)])):
            raise IndexError("token_inputs: a token outside the embedding")
    return x0, emb


class Prefill:
    """The prompt's chunks through the 48 layers, and the hand-off into a g4_model.Model's tensors."""

    def __init__(self, dev, g: Gguf, experts, chunk: int = 4096, log=print, n_layers: int = 48):
        """n_layers: the first n layers only (the checks' fast loop: their states against the reference's)."""
        t0 = time.time()
        self.dev, self.chunk, self.experts = dev, chunk, experts
        W = Weights(dev, g)
        self.layers = []
        for il in range(n_layers):
            mixer = Attn(W, il) if is_attention(il) else Gdn(W, il, chunk)
            self.layers.append(dict(il=il, attn=Hc(W, il, "attn"), ffn=Hc(W, il, "ffn"), mixer=mixer,
                                    dense=Ffn(W, il)))
        self.ple = Ple(W)
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles
        zero = (ttnn.zeros([1, NV, HK, HK], dtype=FP32, layout=TILE, device=dev),
                ttnn.zeros([1, 1, 32, QKV], dtype=BF16, layout=TILE, device=dev), build_fused_const_tiles(dev))
        for L in self.layers:
            if isinstance(L["mixer"], Gdn):
                L["mixer"].zero = zero
        self.cells = context_cells(dev, CTX)
        # traced runs (a single chunk of a bucket's rows, into a decode Model): a layer's ops replayed from a ttnn trace
        # instead of issued one by one. Every tensor a trace reads or writes outside itself was made before any trace
        # was captured (weights, states, the decode's caches) or is a bucket's buffer (bufs[T]: X, the layers' outputs,
        # the chunk's geometry) written in place each run; what else a run makes is freed before the next replay.
        self.trace_t = sorted(int(v) for v in os.environ.get("W51_TRACE_T", "128,256,512,1024,2048").split(",") if v)
        self.traces = {}            # key -> None (seen once: eager, it compiles) or the trace id
        self.bufs = {}              # T -> the bucket's buffers
        self.trace_full = False     # a capture ran out of trace region: no more are captured
        self.log = log
        self.build_s = time.time() - t0
        log(f"prefill weights in {self.build_s:.0f} s")
        self.on_layer = None        # checks: on_layer(il, X, parts) after every layer, parts the layer's outputs
        self.prof = None            # a dict: synchronized seconds per (layer kind, phase) (the breakdown's runs)
        self.mem = None             # mem(kind, phase, il): called at every phase (mem mode: card memory's peak)
        self.walls = None           # a dict: run()'s host seconds by phase, unsynchronized (traced runs' breakdown)
        self.pad = 128              # the last chunk's rows rounded up to a multiple of this (a divisor of chunk; the
                                    # serving engine takes fewer shapes, as each new one compiles)
        self.rows_out = 0           # run() keeps X after the last layer at positions rows_from .. rows_out - 1 in h_rows
        self.rows_from = 0          # (the MTP's window: only its last 8,800 cells are read)
                                    # (fp32 [n, 4 x 2560]: the MTP draft layer's inputs)
        self.h_rows = None
        self.rows_card = False      # h_rows kept on the card instead: (its first cell, rows_from // 32 x 32; row-major
                                    # fp32 [1, 1, rows x 320, 32], g6_mtp's rin page form; the cells rows_from ..
                                    # rows_out - 1 in it) (engine.SpecModel)
        self.zero_m = None          # the verify build's DeltaNet state of a sequence's start (hand_off, made once)

    def reset(self):
        for L in self.layers:
            L["mixer"].reset()
        self.ple.reset()

    def run(self, ids: list[int], host, start: int = 0, held=None) -> torch.Tensor:
        """The prompt from position 0, layer by layer (every chunk through a layer before the next layer: a streamed
        layer's experts come over PCIe once per prompt, not once per chunk); returns X after the last layer at the
        last real position (fp32 [4 x 2560])."""
        self.reset()
        self.xgroups = {}                    # chunks a routed-experts call -> calls (XROWS)
        if held is not None:                 # a follow-up: the carried states from the decode's (resume)
            self.resume(held[0], start, held[1])
        dev = self.dev
        clock = [time.perf_counter()]

        def tick(kind, phase):
            if self.mem is not None:
                self.mem(kind, phase, il_now[0])
            if self.prof is not None:
                ttnn.synchronize_device(dev)
                now = time.perf_counter()
                key = f"{kind}: {phase}"
                self.prof[key] = self.prof.get(key, 0.0) + now - clock[0]
                clock[0] = now
        chunks, il_now = [], [-1]
        on = self.prof is not None or self.mem is not None
        lap_t = [time.perf_counter()]

        def lap(phase):                      # walls: host seconds by phase, unsynchronized (a phase that reads from
            if self.walls is not None:       # the card holds the wait for what was queued before it)
                now = time.perf_counter()
                self.walls[phase] = self.walls.get(phase, 0.0) + now - lap_t[0]
                lap_t[0] = now
        n_all = len(ids) - start
        bucket = next((t for t in self.trace_t if t >= n_all and t <= self.chunk), None)
        traced = (bucket is not None and n_all <= self.chunk and not on and self.on_layer is None
                  and all(L["mixer"].into is not None for L in self.layers if isinstance(L["mixer"], Attn)))
        if n_all > self.chunk and DROP_TRACES:
            self.drop_traces()
        for p0 in range(start, len(ids), self.chunk):
            part = ids[p0:p0 + self.chunk]
            n = len(part)
            T = bucket if traced else -(-n // self.pad) * self.pad
            B = self.bufs.setdefault(T, {}) if traced else {}
            x0, emb = token_inputs(host, self.ids, p0, n, T)
            tick("prompt", "token inputs (host)")
            ch = Chunk(dev, p0, T, n, self.cells, lpad=EXT_LPAD if start else 0, B=B)
            tick("prompt", "chunk geometry")
            tdt = {FP32: torch.float32, BF16: torch.bfloat16}       # bf16: torch's rounding (the decode's input's)
            up = lambda a, dt: ttnn.to_layout(ttnn.from_torch(  # noqa: E731
                torch.from_numpy(a).reshape(1, 1, T, D).to(tdt[dt]), dtype=dt, layout=RM, device=dev), TILE)
            c = dict(ch=ch, B=B, p0=p0, n=n, more=p0 + self.chunk < len(ids))
            self.keep(c, "X", ttnn.repeat(up(x0, FP32), [1, HC, 1, 1]))
            self.keep(c, "emb", up(emb, BF16))
            chunks.append(c)
            tick("prompt", "uploads")
        lap("inputs")
        if hasattr(self.experts, "plan"):
            self.experts.plan()
        for L in self.layers:
            k = "attention" if is_attention(L["il"]) else "deltanet"
            L["mixer"].tick = (lambda phase, k=k: tick(f"{k} mixer", phase)) if on else (lambda phase: None)
        if hasattr(self.experts, "tick"):
            self.experts.tick = (lambda phase: tick("experts", phase)) if on else (lambda phase: None)
        for L in self.layers:
            il = il_now[0] = L["il"]
            kind = "attention" if is_attention(il) else "deltanet"
            mx = L["mixer"]
            for c in chunks:
                B, ch, n = c["B"], c["ch"], c["n"]
                if il == 1:
                    self.keep(c, "X", self.ple(B["X"], B["emb"], n))
                    self.drop(c, ["emb"])
                    tick("ple", "block")
                    lap("ple")

                def ffn_half(out, w4):       # the mixer's combine, the FFN's hc mix, the router and shared expert
                    self.keep(c, "X", Hc.combine(B["X"], out, w4))
                    hb, w4 = L["ffn"].mix(B["X"])
                    tick(kind, "combine + hc mix (ffn)")
                    logits, sh = L["dense"](hb)
                    for name, t in (("hb", hb), ("w4", w4), ("logits", logits), ("sh", sh)):
                        self.keep(c, name, t)
                    tick(kind, "router + shared expert")
                if isinstance(mx, Attn):
                    def first():
                        hb, w4 = L["attn"].mix(B["X"])
                        tick(kind, "hc mix (attn)")
                        self.keep(c, "w4a", w4)
                        for name, t in mx.pre(hb, ch).items():
                            self.keep(c, "a_" + name, t)
                    self.seg(c, ("attn pre", il, ch.T), first)
                    lap("attention: issue")
                    t = {name[2:]: B[name] for name in B if name.startswith("a_")}
                    mx.write(t, ch)
                    lap("attention: cache writes")
                    self.seg(c, ("attn post", il) + ch.key(), lambda: ffn_half(mx.post(t, ch), B["w4a"]))
                    # without the decode's caches (into) write() keeps a chunk's k, v, ik, pk as the prefill's own
                    self.drop(c, ["w4a"] + ["a_" + name for name in t
                                            if mx.into is not None or name not in ("k", "v", "ik", "pk")])
                    lap("attention: issue")
                else:
                    def whole():
                        hb, w4 = L["attn"].mix(B["X"])
                        tick(kind, "hc mix (attn)")
                        self.keep(c, "hb_a", hb)
                        out = mx.mixer(B["hb_a"], ch, c["more"])
                        tick(kind, "mixer")
                        ffn_half(out, w4)
                    self.seg(c, ("deltanet", il, ch.T, mx.fresh), whole)
                    lap("deltanet: issue")
                    mx.tail(B["hb_a"], n, B is self.bufs.get(ch.T))
                    self.drop(c, ["hb_a"])
                    lap("deltanet: conv rows (waits for the card)")
                if MEMLOG:
                    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
                    self.log(f"memlog chunk {c['p0']} layer {il}: allocated {mv.total_bytes_allocated_per_bank / 1e6:.1f} "
                             f"MB a bank, largest free {mv.largest_contiguous_bytes_free_per_bank / 1e6:.1f} MB")
            # the routed experts of consecutive chunks (up to XROWS rows) in one call: an expert's weights decoded
            # once for all their rows; a group whose buffers do not fit (fits()) goes in halves
            groups = []
            for c in chunks:
                if groups and sum(g["ch"].T for g in groups[-1]) + c["ch"].T <= XROWS:
                    groups[-1].append(c)
                else:
                    groups.append([c])
            groups = (part for g in groups for part in self.halves(g))
            for grp in groups:
                self.xgroups[len(grp)] = self.xgroups.get(len(grp), 0) + 1
                one = len(grp) == 1
                cat = lambda name: grp[0]["B"][name] if one else ttnn.concat(  # noqa: E731
                    [c["B"][name] for c in grp], dim=-2)
                hb, logits = cat("hb"), cat("logits")
                routed = self.experts(il, hb, logits, sum(c["ch"].T for c in grp[:-1]) + grp[-1]["n"])
                if not one:
                    ttnn.deallocate(hb)
                    ttnn.deallocate(logits)
                tick(kind, "routed experts")
                lap("routed experts")
                r0 = 0
                for c in grp:
                    B, T = c["B"], c["ch"].T
                    rc = routed if one else ttnn.slice(routed, [0, 0, r0, 0], [1, 1, r0 + T, D])
                    r0 += T
                    self.keep(c, "X", Hc.combine(B["X"], ttnn.add(B["sh"], rc), B["w4"]))
                    tick(kind, "combine")
                    lap("combine")
                    if self.on_layer:
                        self.on_layer(il, B["X"], dict(ffn_moe_out=rc), c["p0"], c["n"])
                if not one:
                    ttnn.deallocate(routed)
        ttnn.synchronize_device(dev)
        if hasattr(self.experts, "finish"):
            self.experts.finish()
        self.h_rows = None
        if self.rows_out and self.rows_card:
            # on the card (2026-11-10): rows cut at 32-row bounds, untilized and permuted to [cell, HC x D] in the
            # MTP's rin page form (128 B pages, 320 a row); the host copy cost ~0.76 s at 6,300 rows
            a, pieces, c_first = self.rows_from, [], None
            for c in chunks:
                if not (c["p0"] < self.rows_out and c["p0"] + c["n"] > a):
                    continue
                lo = max(0, a - c["p0"]) // 32 * 32
                hi = -(-min(c["n"], self.rows_out - c["p0"]) // 32) * 32
                c_first = c["p0"] + lo if c_first is None else c_first
                t = ttnn.to_layout(ttnn.slice(c["B"]["X"], [0, 0, lo, 0], [1, HC, hi, D]), RM)
                p = ttnn.permute(t, [0, 2, 1, 3])
                ttnn.deallocate(t)
                pieces.append(ttnn.reshape(p, [1, 1, (hi - lo) * HC * D // 32, 32]))
            R = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
            if len(pieces) > 1:
                for p in pieces:
                    ttnn.deallocate(p)
            self.h_rows = (c_first, R, self.rows_out - a)
            tick("prompt", "X rows out")
        elif self.rows_out:
            a = self.rows_from
            parts = [ttnn.to_torch(c["B"]["X"])[0, :, max(0, a - c["p0"]):min(c["n"], self.rows_out - c["p0"])]
                     for c in chunks if c["p0"] < self.rows_out and c["p0"] + c["n"] > a]
            self.h_rows = torch.cat(parts, 1).float().permute(1, 0, 2).reshape(-1, HC * D).numpy()
            tick("prompt", "X rows out")
        X, n = chunks[-1]["B"]["X"], chunks[-1]["n"]
        last = ttnn.to_torch(rows(X, (n - 1) // 32 * 32, (n - 1) // 32 * 32 + 32)).float().reshape(HC, 32, D)
        lap("end (sync, rows out)")
        return last[:, (n - 1) % 32].double()

    def halves(self, grp: list):
        """grp as it is when its merged call fits, else each half the same way (down to a chunk a call)."""
        if len(grp) == 1 or self.fits(grp):
            yield grp
        else:
            yield from self.halves(grp[:len(grp) // 2])
            yield from self.halves(grp[len(grp) // 2:])

    def fits(self, grp: list) -> bool:
        """Whether a merged routed-experts call over grp's chunks finds its card memory: its buffers allocated (and
        freed) in the call's order (the concatenated inputs; CardExperts' row-major hb, gather, SwiGLU h, the down's
        10 rows a token, the sum; R at its most), so a group that would not fit runs in halves. The 128K build's
        64K pieces leave ~0.15-0.2 GB a bank in blocks (2026-11-09: two merged 8K chunks' down needs 0.21)."""
        T = sum(c["ch"].T for c in grp)
        R = -(-(10 * T // 32 + 512) // 128) * 4096
        shapes = [("+", [T, D], BF16), ("+", [T, grp[0]["B"]["logits"].shape[-1]], FP32), ("+", [T, D], BF16),
                  ("+", [R, D], BF16), ("-", 2), ("+", [R, 768], FP32), ("-", 3), ("+", [10 * T, D], FP32),
                  ("-", 4), ("+", [T, D], FP32)]
        made = []
        try:
            for op, *a in shapes:
                if op == "+":
                    # each buffer's pages a bank against the largest free block first: a failed allocation logs a
                    # TT_FATAL Out of Memory (the earlier buffers fragment the block a first check saw, 2026-11-09)
                    mv = ttnn.get_memory_view(self.dev, ttnn.BufferType.DRAM)
                    page = 2048 if a[1] == BF16 else 4096
                    if -(-(a[0][0] * a[0][1] // 1024) // mv.num_banks) * page > mv.largest_contiguous_bytes_free_per_bank:
                        return False
                    made.append(ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1] + a[0]), a[1], TILE, self.dev,
                                                               ttnn.DRAM_MEMORY_CONFIG))
                else:
                    ttnn.deallocate(made[a[0]])
            return True
        except RuntimeError:
            return False
        finally:
            for t in made:
                if t.is_allocated():
                    ttnn.deallocate(t)

    def drop_traces(self) -> None:
        """The traces and the buckets' buffers released (the next short prompts issue, then capture, again): a prompt
        of several chunks needs card memory in large blocks, and the buffers sit between them (a 65,503-token
        prompt at the 64K build failed to place a 512 MB mask with 1.9 GB free)."""
        for tid in self.captured():
            ttnn.release_trace(self.dev, tid)
        if self.bufs:
            self.log(f"prefill traces: {len(self.captured())} released, and {len(self.bufs)} buckets' buffers")
        self.traces, self.bufs, self.trace_full = {}, {}, False

    def captured(self) -> list:
        return [v for v in self.traces.values() if v is not None and v is not False]   # (an id may be 0)

    def keep(self, c: dict, name: str, t) -> None:
        """t as the chunk's `name`: a traced run's bucket buffer of that name is written in place (its address is
        what the traces read), an untraced run's entry replaced."""
        B = c["B"]
        if B.get(name) is t:                 # (written in place: Hc.combine)
            return
        if name not in B or B is not self.bufs.get(c["ch"].T):
            B[name] = t
        else:
            ttnn.copy(t, B[name])

    def drop(self, c: dict, names: list) -> None:
        """An untraced run's chunk tensors freed after their last use (the embeddings after layer 1, the attention's
        and the DeltaNet's mixer inputs after the mixer: ~2.5 GB at a 64K piece, room for fits()); a traced run's
        bucket buffers stay (the traces read them)."""
        B = c["B"]
        if B is self.bufs.get(c["ch"].T):
            return
        for k in names:
            t = B.pop(k, None)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)

    def seg(self, c: dict, key: tuple, fn) -> None:
        """fn's ops (their results kept in the chunk's buffers): in an untraced run issued; in a traced run issued on
        the key's first use (it compiles them), captured as a trace on its second, replayed from then on."""
        if c["B"] is not self.bufs.get(c["ch"].T):
            return fn()
        tid = self.traces.get(key, "new")
        if tid == "new":
            self.traces[key] = None
            return fn()
        if tid is None and self.trace_full:
            tid = self.traces[key] = False
        if tid is False:                     # no room for its trace: issued
            return fn()
        if tid is None:
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            try:
                try:
                    fn()
                finally:
                    ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            except RuntimeError as e:        # the trace region full (~1 MB a trace): this key and later ones issued
                if "TRACE buffer" not in str(e):
                    raise
                ttnn.release_trace(self.dev, tid)
                self.trace_full, self.traces[key] = True, False
                self.log(f"prefill traces: the trace region is full after {len(self.captured())} traces; the rest "
                         f"issued")
                return fn()
            self.traces[key] = tid
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)

    def prefill(self, ids: list[int], host, into=None, start: int = 0, n_keep: int = 1) -> torch.Tensor:
        """into: the decode Model whose attention caches the attention layers write into as they go (256K plan step
        1: no second copy of the context; hand_off(into) then skips them). None: the prefill's own caches, which the
        state checks (prefill_states) read. start > 0 (a follow-up, into required): positions start .. from the
        state `into` holds for ids[:start] (its last verify pass's first n_keep rows kept), a multiple of KPOOL (whole
        pools: Attn.write fills a part tile of pooled keys) with at least 3 tokens after it."""
        self.ids = list(ids)
        assert not start or (into is not None and start % KPOOL == 0 and len(ids) - start >= 3), (start, len(ids))
        A1 = {} if into is None else {what: p.A1 for (kind, what), p in zip(into.segs, into.progs)
                                      if kind not in ("run", "ple")}
        for L in self.layers:
            if isinstance(L["mixer"], Attn):
                L["mixer"].into = A1.get(L["il"])
        held = (into, n_keep) if start else None
        try:
            return self.run(self.ids, host, start=start, held=held)
        except RuntimeError as e:            # the traces' buffers (and the experts' cache) in the way: released, the
            cached = any(c[3] is not None for c in getattr(self.experts, "cache", {}).values())  # read again (nothing
            if "Out of Memory" not in str(e) or not (self.bufs or cached):   # is in the decode's states before the
                raise                                                        # hand-off; the cache writes are redone)
            self.log(f"prefill of {len(ids) - start} tokens: out of card memory with the traces"
                     f"{' and the experts cache' if cached else ''} kept, again without")
            self.drop_traces()
            if cached:
                self.experts.drop_cache()
            return self.run(self.ids, host, start=start, held=held)

    def resume(self, M, start: int, n_keep: int) -> None:
        """The prefill's carried states from Model M's at position start: each DeltaNet layer's S (the verify build's:
        its last pass's pending rows committed for n_keep, g3_layer.commit, in float64) and conv history (the last 3
        rows: q|k|v, rounded to bf16 as the prefill's own), the PLE's last 9 rows from M's ring. The attention layers
        need nothing: they read their prefix from M's caches."""
        from g3_layer import commit
        by_il = {L["il"]: L["mixer"] for L in self.layers}
        for (kind, what), p in zip(M.segs, M.progs):
            if kind == "run":
                for L in p.keep:
                    G = by_il[L.layer]
                    if L.m_max > 1:
                        cs, st, K, U, d = L.states_m(M.parity)
                        S, conv = commit(st, K, U, d, n_keep), cs[n_keep - 1:n_keep + 2]
                    else:
                        S = ttnn.to_torch(L.S).double().reshape(NV, 4, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(
                            NV, HK, HK)
                        conv = ttnn.to_torch(L.cstate).double().reshape(3, QKV)
                    put = lambda t, v, dt: ttnn.copy_host_to_device_tensor(  # noqa: E731
                        ttnn.from_torch(v.contiguous(), dtype=dt, layout=t.layout), t)
                    put(G.S, S[torch.tensor(PERM)].float().reshape(1, NV, HK, HK), FP32)   # GGUF -> FLA value-head order
                    st = torch.zeros(32, QKV)
                    st[29:] = conv.float()[:, G.cperm]                        # rows 29..31 of the kernel's history
                    put(G.st, st.reshape(1, 1, 32, QKV), BF16)
                    G.fresh = False
            elif kind == "ple":
                ring = ttnn.to_torch(p.ring_t).float().reshape(RING, HC * D)
                rows9 = torch.stack([ring[q % RING] if q >= 0 else torch.zeros(HC * D) for q in range(start - 9, start)])
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(rows9.reshape(9, HC, D).permute(1, 0, 2).reshape(
                    1, HC, 9, D).contiguous(), dtype=FP32, layout=RM), self.ple.hist)
                self.ple.fresh = False

    def hand_off(self, M) -> None:
        """Every state the decode reads, into Model M's tensors (in place): S, the K / V caches, the raw and pooled
        keys by page copies on the card (pf_pagecopy.cpp: their layouts match page for page, S's value heads
        permuted back), the conv states and the PLE ring (small, permuted channels) from the host. A verify build
        (M.m_max > 1) gets g3_layer.Layer.set_states_m's layout, parity 0, for a first pass with n_keep = 1: its S
        tensor a sequence start's (both parities zero, the pending page committing nothing), then S into parity 0's
        tiles; the conv history at rows 0..2 of parity 0's buffer."""
        if os.environ.get("W51_HANDOFF") == "host":
            return self.hand_off_host(M)
        n, dev = len(self.ids), self.dev
        by_il = {L["il"]: L["mixer"] for L in self.layers}

        def put(t, v: torch.Tensor):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(v.reshape(tuple(t.shape)).float(), dtype=t.dtype,
                                                            layout=t.layout), t)
        tiles, zero = (HK // 32) ** 2, None
        for (kind, what), p in zip(M.segs, M.progs):
            if kind == "run":
                for L in p.keep:
                    G = by_il[L.layer]
                    assert G.S.layout == TILE and G.S.dtype == FP32
                    if L.m_max > 1:
                        if zero is None:
                            zero = self.zero_state_m(L).to(dev)
                        page_copy(dev, zero, L.S, [(0, 0, page_count(zero))], 4096)
                    page_copy(dev, G.S, L.S, [(j * tiles, PERM[j] * tiles, tiles) for j in range(NV)], 4096)
                    conv = G.conv_state()
                    if L.m_max > 1:
                        conv = torch.cat([conv.reshape(3, QKV), torch.zeros(2 * (2 + L.m_max) - 3, QKV)])
                    put(L.cstate, conv)
            elif kind == "ple":
                ring = torch.zeros(RING, HC * D)
                r = self.ple.ring_rows()
                for i in range(9):
                    if n - 9 + i >= 0:
                        ring[(n - 9 + i) % RING] = r[i]
                put(p.ring_t, ring)
            else:
                A, at = p.A1, by_il[what]
                if at.into is A:                                      # written during the run (prefill(into=M))
                    continue
                L = at.K.shape[-2]
                for src, dst in ((at.K, A.kc), (at.V, A.vc)):
                    page_copy(dev, ttnn.to_layout(src, RM), dst, [(h * L, h * CTX, n) for h in range(NKV)],
                              HD * 2)
                page_copy(dev, ttnn.to_layout(at.rawk, RM), A.rawk,           # the ring: the last RAWK positions
                          [(q, q % RAWK, 1) for q in range(max(0, n - RAWK), n)], IHD * 4)
                full = n // KPOOL                                     # complete pools; the open one: the decode's
                if full:
                    page_copy(dev, at.pk, A.pooled, [(0, 0, -(-full // 32) * 4)], 4096)
        M.parity = 0
        ttnn.synchronize_device(dev)

    def zero_state_m(self, L):
        """The verify build's S tensor at a sequence's start (Layer.set_states_m of zero states), on the card."""
        if self.zero_m is None:     # (host side only: a device copy made after a trace's capture could sit in its
            from g3_layer import raw_tiles, state_tiles      # temporaries)
            pend = torch.zeros(2, NV, 2, 1024)
            pend[0, :, 1, 0] = 1.0
            tiles = torch.cat([state_tiles(torch.zeros(2, NV, HK, HK)), raw_tiles(pend)]).reshape(1, 1, -1, 32)
            self.zero_m = ttnn.from_torch(tiles, dtype=FP32, layout=TILE)
        assert tuple(self.zero_m.shape) == tuple(L.S.shape), (self.zero_m.shape, L.S.shape)
        return self.zero_m

    def hand_off_host(self, M) -> None:
        """hand_off through the host (the reference path)."""
        put_states(M, prefill_states(self, len(self.ids)), len(self.ids))


def put_states(M, st: dict, n: int) -> None:
    """States in decode_states' form after n tokens (any subset of its keys; the KV and key caches' valid parts) into
    Model M's tensors, through the host; the caches' cells past n zero."""
    def put(t, v: torch.Tensor):
        dt = {ttnn.bfloat16: torch.bfloat16}.get(t.dtype, torch.float32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(v.reshape(tuple(t.shape)).to(dt), dtype=t.dtype,
                                                        layout=t.layout), t)

    def cells(v, total):
        out = torch.zeros(*v.shape[:-2], total, v.shape[-1])
        out[..., :v.shape[-2], :] = v
        return out
    for (kind, what), p in zip(M.segs, M.progs):
        if kind == "run":
            for L in p.keep:
                if f"S.{L.layer}" in st:                       # [48, k, v] -> tiles (h, k-tile, v-tile)
                    put(L.S, st[f"S.{L.layer}"].reshape(NV, 4, 32, 4, 32).permute(0, 1, 3, 2, 4))
                if f"conv.{L.layer}" in st:
                    put(L.cstate, st[f"conv.{L.layer}"])
        elif kind == "ple":
            if "ring" in st:
                put(p.ring_t, st["ring"])
        else:
            A = p.A1
            for k, t in (("kc", A.kc), ("vc", A.vc)):
                if f"{k}.{what}" in st:
                    put(t, cells(st[f"{k}.{what}"][:, :n], CTX))
            if f"rawk.{what}" in st:
                put(A.rawk, rawk_ring(st[f"rawk.{what}"], n))
            if f"pooled.{what}" in st:                      # complete pools; the open one: the decode's
                pk = cells(st[f"pooled.{what}"][:n // KPOOL], NPOOL)
                put(A.pooled, pk.reshape(NPOOL // 32, 32, 4, 32).permute(0, 2, 1, 3))
    ttnn.synchronize_device(M.dev)


def page_count(t) -> int:
    """An interleaved tensor's pages (tiles, or rows of a row-major one)."""
    n = 1
    for d in t.shape:
        n *= int(d)
    return n // 1024 if t.layout == TILE else n // int(t.shape[-1])


def page_bytes(t) -> int:
    size = {FP32: 4, BF16: 2, ttnn.uint32: 4, ttnn.uint8: 1}
    if t.layout == TILE:
        return 1088 if t.dtype == BFP8 else 1024 * size[t.dtype]
    return int(t.shape[-1]) * size[t.dtype]


J_MAX = 32                        # page_copy's jobs per core


def page_copy(dev, src, dst, jobs, unit: int, dst_stride: int = 1, n_cores: int = 0) -> None:
    """Units of `unit` bytes from src to dst (both interleaved in card memory): jobs [(first src unit, first dst unit,
    count)], split over the cores (pf_pagecopy.cpp; n_cores: only the first that many, for small copies whose host
    cost is the per-core arguments); dst_stride: a job's consecutive units land that many dst units apart."""
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)][:n_cores or None]
    total = sum(c for _, _, c in jobs)
    share = -(-total // len(cores))
    per, cur = [[]], 0
    for a, b, c in jobs:
        while c:
            k = min(c, share - cur)
            per[-1].append((a, b, k))
            a, b, c, cur = a + k, b + k * dst_stride, c - k, cur + k
            if cur == share:
                per.append([])
                cur = 0
    per = [p for p in per if p]
    used = cores[:len(per)]
    rt = ttnn.RuntimeArgs()
    for (cx, cy), mine in zip(used, per):
        assert len(mine) <= J_MAX, (len(mine), J_MAX)
        # a fixed count: generic_op's program cache keeps the first call's
        rt[cx][cy] = [src.buffer_address(), dst.buffer_address(), len(mine)] + [v for j in mine for v in j] + [0] * (
            3 * (J_MAX - len(mine)))
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in used])
    ps, pd = page_bytes(src), page_bytes(dst)
    assert ps % unit == 0 and pd % unit == 0, (ps, pd, unit)
    k = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_pagecopy.cpp"), core_ranges=ranges,
                              compile_time_args=[unit, ps, pd] + list(ttnn.TensorAccessorArgs(src)
                                                                    .get_compile_time_args()),
                              runtime_args=rt, config=ttnn.ReaderConfigDescriptor(),
                              defines=[("DST_STRIDE", str(dst_stride))] if dst_stride != 1 else [])
    cb = ttnn.CBDescriptor(total_size=8 * unit, core_ranges=ranges, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint8, page_size=unit)])
    ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))


# ---------------------------------------------------------------------------------------------------- the checks

def decode_states(M) -> dict:
    """Every state the decode leaves behind, on the host (float64), keyed by name."""
    out = {}
    get = lambda t: ttnn.to_torch(t).double()  # noqa: E731
    for (kind, what), p in zip(M.segs, M.progs):
        if kind == "run":
            for L in p.keep:
                out[f"S.{L.layer}"] = get(L.S).reshape(NV, 4, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(NV, HK, HK)
                out[f"conv.{L.layer}"] = get(L.cstate).reshape(3, QKV)
        elif kind == "ple":
            out["ring"] = get(p.ring_t).reshape(RING, HC * D)
        else:
            A = p.A1
            out[f"kc.{what}"] = get(A.kc).reshape(NKV, CTX, HD)
            out[f"vc.{what}"] = get(A.vc).reshape(NKV, CTX, HD)
            out[f"rawk.{what}"] = get(A.rawk).reshape(RAWK, IHD)
            out[f"pooled.{what}"] = get(A.pooled).reshape(NPOOL // 32, 4, 32, 32).permute(0, 2, 1, 3).reshape(
                NPOOL, IHD)
    return out


def kl_top1(ref: np.ndarray, lg: torch.Tensor) -> tuple[float, bool]:
    r = torch.from_numpy(ref).double()
    lp, lq = torch.log_softmax(r, -1), torch.log_softmax(lg.double(), -1)
    return float((lp.exp() * (lp - lq)).sum()), int(r.argmax()) == int(lg.argmax())


def dram_gb(dev) -> float:
    """Card memory (DRAM) allocated now, GB."""
    return ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank * 8 / 1e9


def inventory(dev, roots: dict) -> dict:
    """Card memory (DRAM) by holder, GB: every allocated buffer (ttnn's reports) named by the first attribute path, from
    the roots, of a device tensor at its address; list indices folded (the 12 layers' kc are one line). What no root
    holds is "unattributed": a phase's temporaries, tensors only closures hold, programs' own buffers."""
    banks = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).num_banks
    size = {b.address: b.max_size_per_bank * banks for b in ttnn._ttnn.reports.get_buffers(dev)
            if b.buffer_type == ttnn.BufferType.DRAM}
    held, seen = {}, set()

    def walk(o, path, depth):
        if isinstance(o, ttnn.Tensor):
            if o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                held.setdefault(o.buffer_address(), path)
            return
        if depth > 10 or id(o) in seen or type(o).__module__.split(".")[0] in ("torch", "numpy", "ttnn", "builtins"
                                                                                 ) and not isinstance(o, (dict, list, tuple)):
            return
        seen.add(id(o))
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if isinstance(k, str) else path, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, path if isinstance(v, (ttnn.Tensor, dict, list, tuple)) else f"{path}[{type(v).__name__}]",
                     depth + 1)
        elif hasattr(o, "__dict__") and not isinstance(o, (type, types.ModuleType, types.FunctionType,
                                                          types.MethodType)):
            for k, v in vars(o).items():
                walk(v, f"{path}.{k}", depth + 1)
    for name, o in roots.items():
        walk(o, name, 0)
    by = {}
    for a, b in size.items():
        k = held.get(a, "unattributed")
        by[k] = by.get(k, 0) + b
    return dict(total_gb=sum(size.values()) / 1e9, buffers=len(size),
                by_gb={k: round(v / 1e9, 4) for k, v in sorted(by.items(), key=lambda kv: -kv[1])})


def mem_inventory(dev, res: dict, chunk: int) -> None:
    """Step 0 of the 256K-context work: card memory by holder after
    the build (card experts, W51_RESIDENT as the engine sets it), then, with W51_MEM_PROMPT=n, during an n-token prompt
    of random ids: the inventory at each new peak (50 MB steps) and the phase it came at; an out-of-memory is recorded,
    not raised (the peak before it is the answer)."""
    from g4_gate import reset
    lane, host, g, M, P = build(dev, chunk, "card")
    roots = lambda: {"decode": M, "experts": P.experts, "prefill": P}  # noqa: E731
    res["resident"] = sorted(P.experts.resident)
    res["build"] = inventory(dev, roots())
    show = lambda inv: [print(f"  {v:8.3f} GB  {k}", flush=True) for k, v in list(inv["by_gb"].items())[:25]]  # noqa
    print(f"after the build: {res['build']['total_gb']:.2f} GB in {res['build']['buffers']} buffers", flush=True)
    show(res["build"])
    n = int(os.environ.get("W51_MEM_PROMPT", 0))
    if not n:
        return
    peak = dict(gb=0.0)

    def mem(kind, phase, il):
        gb = dram_gb(dev)
        if gb > peak["gb"] + 0.05:   # the callers' locals too (run()'s chunks, a mixer's temporaries), after the roots
            frames, f = {}, sys._getframe(1)
            while f is not None and f.f_code.co_name != "mem_inventory":
                frames[f"{f.f_code.co_name}()"] = dict(f.f_locals)
                f = f.f_back
            peak.update(gb=gb, at=f"layer {il}, {kind}: {phase}", inventory=inventory(dev, {**roots(), **frames}))
    P.mem = mem
    reset(M)
    t0 = time.perf_counter()
    try:
        P.prefill(np.random.default_rng(0).integers(0, 200000, n).tolist(), host, into=M if INTO else None)
        P.hand_off(M)
        res["prompt"] = dict(tokens=n, ok=True, s=time.perf_counter() - t0)
    except RuntimeError as e:
        res["prompt"] = dict(tokens=n, ok=False, s=time.perf_counter() - t0, error=str(e)[-800:])
    P.mem = None
    res["peak"] = peak
    print(f"prompt {res['prompt']}", flush=True)
    print(f"peak {peak['gb']:.2f} GB at {peak.get('at')}", flush=True)
    if "inventory" in peak:
        show(peak["inventory"])


def build(dev, chunk: int, experts: str = "cpu", n_layers: int = 48, model: bool = True):
    """The CPU lane (n_layers' experts), the host's gathers, the decode Model (model=True; 48 layers) and a Prefill."""
    from cpulane import Lane
    from g4_host import Host
    from g4_model import SSD_MODEL, Model
    t0 = time.time()
    lane = Lane()
    lane.load(SSD_MODEL, layers=range(n_layers))
    print(f"lane: experts loaded in {time.time() - t0:.0f} s", flush=True)
    host = Host(SSD_MODEL)
    g = Gguf(SSD_MODEL)
    M = None
    if model:
        t0 = time.time()
        M = Model(dev, g, lane, log=lambda s: None)
        print(f"decode model built in {time.time() - t0:.0f} s (card memory {dram_gb(dev):.1f} GB)", flush=True)
    if experts == "card":
        from g7_experts import CardExperts
        E = CardExperts(dev, [il for il in resident(SSD_MODEL) if il < n_layers], n_layers=n_layers)
        cg = os.environ.get("W51_CARD_GROUPS", "").split(",")[0]
        if cg and int(cg) < E.n_groups:         # the split for the checks (time mode sweeps its own)
            E.lane, E.card_groups = lane, int(cg)
    else:
        E = CpuExperts(dev, lane)
    print(f"experts ready (card memory {dram_gb(dev):.1f} GB)", flush=True)
    P = Prefill(dev, g, E, chunk=chunk, n_layers=n_layers)
    print(f"prefill weights ready (card memory {dram_gb(dev):.1f} GB)", flush=True)
    return lane, host, g, M, P


# ---------------------------------------------------------------------------------- the decode's reference states
# $TT_MOEOFFLOAD_DATA/w51_ref_states/<ref>_<n>/: what the decode (g4_model, traced, teacher-forced from position 0) leaves
# after a reference's first n ids, saved once (refstates mode) so the prefill's checks compare tensors directly:
#   S.<il>.npy [48, 128 k, 128 v] fp32 (GGUF value-head order), conv.<il>.npy [3, 10240] fp32, ring.npy [17, 10240]
#   (token q's n at row q % 17), kc.<il>.npy / vc.<il>.npy [2, n, 256] (bf16 values as fp32), rawk.<il>.npy [n, 128] (saved before 2026-11-06; the ring [8, 128] since),
#   pooled.<il>.npy [n // 4, 128] (complete pools), logits.npy: the decode's logits at position n (its next step)
REF_STATES = os.path.join(DATA, "w51_ref_states")
REFS = {"g4_ref": (os.path.join(DATA, "g4_ref"), (46, 91)),
        "g4_ref_long": (os.path.join(DATA, "g4_ref_long"), (512, 4775))}


def ref_ids(name: str) -> list[int]:
    return np.fromfile(os.path.join(REFS[name][0], "ids.i32"), np.int32).tolist()


def rawk_ring(v: torch.Tensor, n: int) -> torch.Tensor:
    """The decode's raw-key ring (row p % RAWK: position p) after n positions, from raw keys given by position
    ([>= n, IHD]: the prefill's, references saved before the ring) or already as the ring ([RAWK, IHD])."""
    if v.shape[0] == RAWK:
        return v
    ring = torch.zeros(RAWK, v.shape[-1], dtype=v.dtype)
    for q in range(max(0, n - RAWK), n):
        ring[q % RAWK] = v[q]
    return ring


def valid_part(k: str, v: torch.Tensor, n: int) -> torch.Tensor:
    """A decode state's part a prefill of n tokens must reproduce (decode_states' full tensors)."""
    kind = k.split(".")[0]
    if kind in ("kc", "vc"):
        return v[:, :n]
    if kind == "rawk":
        return rawk_ring(v, n)
    if kind == "pooled":
        return v[:n // KPOOL]
    return v


def save_ref_states(dev, res: dict) -> None:
    """The decode over each reference's ids (traced, as g4_gate's parities), its states saved at every listed n."""
    from g4_gate import reset
    lane, host, g, M, _ = build(dev, 128, n_layers=48)
    M.set_head("logits")
    for name, (_, points) in REFS.items():
        ids = ref_ids(name)
        reset(M)
        t0 = time.time()
        for pos in range(max(points) + 1):
            if M.tid is None:                     # the first token untraced: it loads the programs, then the capture
                M.step(host.embed(ids[pos]), host.ple(ids[pos], ids[:pos]), pos)
                M.capture()
            else:
                M.step_traced(host.embed(ids[pos]), host.ple(ids[pos], ids[:pos]), pos)
            if pos + 1 in points:
                n = pos + 1
                out = os.path.join(REF_STATES, f"{name}_{n}")
                os.makedirs(out, exist_ok=True)
                for k, v in decode_states(M).items():
                    np.save(os.path.join(out, k + ".npy"), valid_part(k, v, n).float().numpy())
                # the decode's own next step from these states (its logits at position n): the hand-off's target
                st = state_tensors(M)
                keep = {k: [ttnn.to_torch(t) for t in v] for k, v in st.items()}
                np.save(os.path.join(out, "logits.npy"),
                        M.step_traced(host.embed(ids[n]), host.ple(ids[n], ids[:n]), n).float().numpy())
                for k, v in st.items():                           # back to the states after n tokens
                    for t, h in zip(v, keep[k]):
                        ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=t.dtype, layout=t.layout), t)
                print(f"{name}: states after {n} tokens -> {out} ({time.time() - t0:.0f} s)", flush=True)
        res[name] = dict(points=list(points), seconds=round(time.time() - t0, 1))


def load_ref_states(name: str, n: int) -> dict:
    d = os.path.join(REF_STATES, f"{name}_{n}")
    return {f[:-4]: torch.from_numpy(np.load(os.path.join(d, f))).double() for f in sorted(os.listdir(d))
            if f.endswith(".npy") and f != "logits.npy"}


def prefill_states(P, n: int) -> dict:
    """The prefill's states after n tokens in decode_states' form (the valid parts), from P's own tensors."""
    out = {}
    for L in P.layers:
        il, G = L["il"], L["mixer"]
        if isinstance(G, Gdn):
            out[f"S.{il}"] = G.state().double()
            out[f"conv.{il}"] = G.conv_state().double()
        else:
            out[f"kc.{il}"] = ttnn.to_torch(G.K).double()[0, :, :n]
            out[f"vc.{il}"] = ttnn.to_torch(G.V).double()[0, :, :n]
            out[f"rawk.{il}"] = rawk_ring(ttnn.to_torch(G.rawk).double().reshape(-1, IHD), n)
            out[f"pooled.{il}"] = ttnn.to_torch(G.pk).double().reshape(-1, IHD)[:n // KPOOL]
    if len(P.layers) > 1:
        ring = torch.zeros(RING, HC * D, dtype=torch.float64)
        r = P.ple.ring_rows().double()
        for i in range(9):
            if n - 9 + i >= 0:
                ring[(n - 9 + i) % RING] = r[i]
        out["ring"] = ring
    return out


def state_errors(ref: dict, got: dict, n: int) -> dict:
    """Per state: the Frobenius norm of the difference relative to the reference's, and the max error relative to its
    max (the ring: its last min(n, 9) rows)."""
    res = {}
    for k, y in got.items():
        x = ref[k]
        if k.startswith("rawk."):                     # references saved by position before the ring
            x = rawk_ring(x, n)
        if k == "ring":
            sl = [q % RING for q in range(max(0, n - 9), n)]
            x, y = x[sl], y[sl]
        res[k] = dict(fro=float((x - y).norm() / x.norm().clamp(min=1e-30)),
                      max=float((x - y).abs().max() / x.abs().max().clamp(min=1e-30)))
        if k.split(".")[0] in ("kc", "rawk"):      # by position, 512 at a time (the chunk and selection boundaries)
            a, b = x.reshape(-1, x.shape[-2], x.shape[-1]), y.reshape(-1, y.shape[-2], y.shape[-1])
            res[k]["by_512"] = [round(float((a[:, i:i + 512] - b[:, i:i + 512]).norm() / a[:, i:i + 512].norm()), 4)
                                for i in range(0, a.shape[1], 512)]
    return res


def check_states(dev, res: dict, chunk: int) -> None:
    """The fast loop: the first W51_LAYERS layers (default 8) prefilled over W51_REFN (default g4_ref_long:512) in
    chunks of `chunk`, every state against the decode's saved one (save_ref_states); W51_CHUNKS lists more chunk sizes
    for the same process."""
    n_layers = int(os.environ.get("W51_LAYERS", 8))
    name, n = os.environ.get("W51_REFN", "g4_ref_long:512").split(":")
    n = int(n)
    chunks = [int(c) for c in os.environ.get("W51_CHUNKS", str(chunk)).split(",")]
    lane, host, g, _, P = build(dev, max(chunks), os.environ.get("W51_EXPERTS", "cpu"), n_layers=n_layers, model=False)
    ref = load_ref_states(name, n)
    ids = ref_ids(name)[:n]
    res["states"] = {}
    G0 = P.layers[0]["mixer"]
    for c in chunks:
        P.chunk = c
        G0.capture = c >= n                       # one chunk: layer 0's S in float64 from its captured hn
        t0 = time.time()
        P.prefill(ids, host)
        got = prefill_states(P, n)
        err = state_errors(ref, got, n)
        res["states"][c] = dict(seconds=round(time.time() - t0, 1), err=err)
        print(f"{name}[:{n}], {n_layers} layers, chunks of {c}: " + ", ".join(
            f"{k} {v['fro']:.1e}/{v['max']:.1e}" for k, v in err.items()), flush=True)
        for k, v in err.items():
            if "by_512" in v:
                print(f"  {k} by position (512 at a time): {v['by_512']}", flush=True)
        if G0.capture:
            x, gl, beta = gdn_inputs64(G0, G0.cap, n)
            S = torch.empty(NV, HK, HK, dtype=torch.float64)
            S[torch.tensor(PERM)] = recurrence((x, gl, beta, G0.cap[3]), n)
            e = {"prefill": state_errors({"S": S}, {"S": got["S.0"]}, n)["S"],
                 "decode": state_errors({"S": S}, {"S": ref["S.0"]}, n)["S"]}
            bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
            for name_, a, b in (("x_bf16", 0, QKV), ("k_bf16", NK * HK, 2 * NK * HK), ("v_bf16", 2 * NK * HK, QKV)):
                xr = x.clone()
                xr[:, a:b] = bf(xr[:, a:b])
                Sr = torch.empty_like(S)
                Sr[torch.tensor(PERM)] = recurrence((xr, gl, beta, G0.cap[3]), n)
                e[f"float64, {name_}"] = state_errors({"S": S}, {"S": Sr}, n)["S"]
            res["states"][c]["S0_vs_float64"] = e
            print("  layer 0's S against float64 from the prefill's hn: " + ", ".join(
                f"{k} {v['fro']:.1e}/{v['max']:.1e}" for k, v in e.items()), flush=True)


def hand_off_check(M, P, host, ids: list[int], n: int, name: str, res: dict) -> torch.Tensor:
    """After P's prefill of ids[:n]: the hand-off into M, every decode state against the decode's own after the same
    n tokens (w51_ref_states), and the next decode step (position n) against the decode's next step there. Returns
    that step's logits."""
    P.hand_off(M)
    err = state_errors(load_ref_states(name, n), {k: valid_part(k, v, n) for k, v in decode_states(M).items()}, n)
    worst = {}
    for k, v in err.items():
        kind = k.split(".")[0]
        if v["fro"] > worst.get(kind, {}).get("fro", -1):
            worst[kind] = dict(v, at=k)
    lg = M.step(host.embed(ids[n]), host.ple(ids[n], ids[:n]), n).clone()
    kl, same = kl_top1(np.load(os.path.join(REF_STATES, f"{name}_{n}", "logits.npy")), lg)
    res[f"hand_off_{name}_{n}"] = dict(worst=worst, per_state=err, next_step_kl_vs_decode=kl, same_top1=same)
    print(f"{name}: hand-off after {n} tokens, worst state error (fro/max) per kind: " + ", ".join(
        f"{k} {v['fro']:.1e}/{v['max']:.1e} ({v['at']})" for k, v in worst.items()) +
        f"; next step KL(decode || prefill + decode) {kl:.4f}, same top-1 {same}", flush=True)
    return lg


def decode_parity(M, host, ids, start, ref_logits, first=None) -> dict:
    """Teacher-forced decode from `start` (the prefill's hand-off before it) to the end; logits against llama.cpp's
    (ref_logits[i] is position start + i's). first: position start's logits when already stepped."""
    kls, top1 = [], []
    for pos in range(start, len(ids)):
        lg = first if pos == start and first is not None else M.step(host.embed(ids[pos]),
                                                                    host.ple(ids[pos], ids[:pos]), pos)
        kl, t = kl_top1(ref_logits[pos - start], lg)
        kls.append(kl)
        top1.append(t)
    return dict(decoded=len(kls), top1=sum(top1) / len(top1), kl_mean=statistics.mean(kls), kl_max=max(kls),
                kl=kls)


def head_logits(dev, M, x: torch.Tensor) -> torch.Tensor:
    """The decode's head (logits mode) on one position's X [4 x 2560]."""
    from g4_model import set_x
    set_x(dev, M.x32, M.x16, x.float())
    M.head.run()
    ttnn.synchronize_device(dev)
    return M.head.get_logits()


def check_short(dev, res: dict, chunk: int, experts: str) -> None:
    """g4_ref (92 tokens, llama.cpp's logits and per-layer tensors): (1) the prefill of all 92 tokens layer by layer
    against llama.cpp's residual streams and routed sums, every position's logits through the decode's head; (2) check
    (b): prefill 91, the hand-off (states against the decode's), decode position 91: positions 0-90 from the prefill
    and 91 from the decode against llama.cpp; (3) prefill 46, decode the other 46."""
    from g4_gate import reset
    ref = REFS["g4_ref"][0]
    lane, host, g, M, P = build(dev, chunk, experts)
    M.set_head("logits")
    ids = ref_ids("g4_ref")
    N = len(ids)
    ref_logits = np.fromfile(os.path.join(ref, "logits.f32"), np.float32).reshape(N, -1)
    tens = lambda name, il, w: np.fromfile(os.path.join(ref, "tensors", f"{name}-{il}.f32"),  # noqa: E731
                                           np.float32).reshape(N, w)
    errs = {"l_last": {}, "ffn_moe_out": {}}
    x47 = []

    def rel_rows(a, b):
        """Per position: max |a - b| / max |b| -> (median, max) over positions."""
        e = np.abs(a - b).max(1) / np.abs(b).max(1)
        return float(np.median(e)), float(e.max())

    def on_layer(il, X, parts, p0, n):
        x = ttnn.to_torch(X).float().numpy()[0, :, :n].transpose(1, 0, 2).reshape(n, HC * D)
        errs["l_last"][il] = rel_rows(x, tens("l_last", il, HC * D)[p0:p0 + n])
        errs["ffn_moe_out"][il] = rel_rows(to_np(parts["ffn_moe_out"]).reshape(-1, D)[:n],
                                           tens("ffn_moe_out", il, D)[p0:p0 + n])
        if il == 47:
            x47.append(x)
    P.on_layer = on_layer
    P.prefill(ids, host)
    P.on_layer = None
    res["layers_vs_llamacpp_median_max"] = errs
    print(f"prefill of {N} tokens vs llama.cpp, per-position rel err median per layer: X " +
          " ".join(f"{errs['l_last'][i][0]:.1e}" for i in range(48)), flush=True)
    print("  routed sums: " + " ".join(f"{errs['ffn_moe_out'][i][0]:.1e}" for i in range(48)), flush=True)
    x47 = np.concatenate(x47)
    each = [kl_top1(ref_logits[pos], head_logits(dev, M, torch.from_numpy(x47[pos]))) for pos in range(N)]
    kls, t1s = [k for k, _ in each], [t for _, t in each]
    res["prefill_all_positions"] = dict(top1=sum(t1s) / N, kl_mean=statistics.mean(kls), kl_max=max(kls), kl=kls)
    print(f"prefill of {N}: every position's logits (the decode's head) vs llama.cpp: top-1 "
          f"{100 * sum(t1s) / N:.1f}%, KL mean {statistics.mean(kls):.4f}, max {max(kls):.3f}", flush=True)
    # check (b): prefill N - 1, decode the last
    reset(M)
    P.prefill(ids[:N - 1], host)
    lg = hand_off_check(M, P, host, ids, N - 1, "g4_ref", res)
    kl, t = kl_top1(ref_logits[N - 1], lg)
    b_kls, b_t1 = kls[:N - 1] + [kl], t1s[:N - 1] + [t]
    res["check_b"] = dict(prefill=N - 1, decoded=1, top1=sum(b_t1) / N, kl_mean=statistics.mean(b_kls),
                          decoded_kl=kl, decoded_top1=t)
    print(f"check (b): prefill {N - 1} + decode 1: top-1 {100 * sum(b_t1) / N:.1f}%, KL mean "
          f"{statistics.mean(b_kls):.4f} (the decoded position's KL {kl:.4f})", flush=True)
    half = 46
    reset(M)
    P.prefill(ids[:half], host)
    lg = hand_off_check(M, P, host, ids, half, "g4_ref", res)
    r = res["prefill_half_then_decode"] = decode_parity(M, host, ids, half, ref_logits[half:], first=lg)
    print(f"prefill {half} + decode {N - half}: top-1 {100 * r['top1']:.1f}%, KL mean {r['kl_mean']:.4f}", flush=True)


def check_16k(dev, res: dict, chunk: int, experts: str) -> None:
    """Check (c), g4_ref_16k (13,500 tokens, llama.cpp's logits from 13,244): prefill 0..13,243 (past 8K: two chunks),
    the hand-off, decode 13,244..13,499 teacher-forced against llama.cpp's logits and against the card's decode-only
    run (logits_card.f32). W51_REF names another reference of the same form (e.g. g4_ref_64k, 61,000 tokens
    from 60,744, under TT_CTX=65536); its README's "from" is where the decode starts."""
    from g4_gate import reset
    ref = os.path.expanduser(os.environ.get("W51_REF", os.path.join(DATA, "g4_ref_16k")))
    with open(os.path.join(ref, "README.json")) as f:
        start = json.load(f)["from"]
    res["ref"] = ref
    lane, host, g, M, P = build(dev, chunk, experts)
    ids = np.fromfile(os.path.join(ref, "ids.i32"), np.int32).tolist()
    M.set_head("logits")
    reset(M)
    t0 = time.perf_counter()
    P.prefill(ids[:start], host, into=M if INTO else None)
    P.hand_off(M)
    res["prefill_s"] = time.perf_counter() - t0
    out = {}
    for pos in range(start, len(ids)):
        out[pos] = M.step(host.embed(ids[pos]), host.ple(ids[pos], ids[:pos]), pos).clone()
    for name in [n for n in ("logits", "logits_card") if os.path.exists(os.path.join(ref, n + ".f32"))]:
        rl = np.fromfile(os.path.join(ref, name + ".f32"), np.float32).reshape(-1, 248320)
        kls, top1 = zip(*(kl_top1(rl[pos - start], out[pos]) for pos in range(start, len(ids))))
        r = res[f"check_c_vs_{name}"] = dict(decoded=len(kls), top1=sum(top1) / len(top1),
                                             kl_mean=statistics.mean(kls), kl_max=max(kls))
        print(f"check (c): prefill {start} + decode {len(kls)} against {name}: top-1 {100 * r['top1']:.1f}%, KL mean "
              f"{r['kl_mean']:.4f} (max {r['kl_max']:.3f})", flush=True)


def check_long(dev, res: dict, chunk: int, experts: str) -> None:
    """g4_ref_long (5,031 tokens, llama.cpp's logits from 4,775): check (a): prefill 0..4,774, the hand-off (states
    against the decode's), decode 4,775..5,030 teacher-forced: top-1, KL; check (d): prefill 0..4,775 and its last
    position's logits through the head against llama.cpp's."""
    from g4_gate import reset
    start = 4775
    lane, host, g, M, P = build(dev, chunk, experts)
    ids = ref_ids("g4_ref_long")
    ref_logits = np.fromfile(os.path.join(REFS["g4_ref_long"][0], "logits.f32"), np.float32).reshape(-1, 248320)
    M.set_head("logits")
    reset(M)
    t0 = time.perf_counter()
    P.prefill(ids[:start], host)
    res["prefill_s"] = time.perf_counter() - t0
    print(f"prefill of {start} tokens: {res['prefill_s']:.2f} s (the first: compiles)", flush=True)
    lg = hand_off_check(M, P, host, ids, start, "g4_ref_long", res)
    st = state_tensors(M)
    snap = {k: [ttnn.to_torch(t) for t in v] for k, v in st.items()}      # the hand-off's states
    r = res["check_a"] = decode_parity(M, host, ids, start, ref_logits, first=lg)
    print(f"check (a): prefill {start} + decode {r['decoded']}: top-1 {100 * r['top1']:.1f}%, KL mean "
          f"{r['kl_mean']:.4f} (max {r['kl_max']:.3f})", flush=True)
    if os.environ.get("W51_MIX") == "1":           # which state kind costs the parity: one kind from the decode's
        ref = load_ref_states("g4_ref_long", start)
        res["mix"] = {}
        for name, kinds in (("all from the decode", ("S", "conv", "kc", "vc", "rawk", "pooled", "ring")),
                            ("S", ("S",)), ("conv", ("conv",)), ("kv", ("kc", "vc")), ("idx", ("rawk", "pooled")),
                            ("ring", ("ring",))):
            for k, v in st.items():
                for t, h in zip(v, snap[k]):
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=t.dtype, layout=t.layout), t)
            put_states(M, {k: v for k, v in ref.items() if k.split(".")[0] in kinds}, start)
            m = res["mix"][name] = decode_parity(M, host, ids, start, ref_logits)
            print(f"  {name} from the decode, the rest from the prefill: top-1 {100 * m['top1']:.1f}%, KL mean "
                  f"{m['kl_mean']:.4f} (max {m['kl_max']:.3f})", flush=True)
    x_last = P.prefill(ids[:start + 1], host)
    kl, t = kl_top1(ref_logits[0], head_logits(dev, M, x_last))
    res["check_d"] = dict(pos=start, kl=kl, top1=t)
    print(f"check (d): prefill of {start + 1}, the last position's logits vs llama.cpp: KL {kl:.4f}, top-1 {t}",
          flush=True)


def state_tensors(M) -> dict:
    """Every decode state tensor by kind: S, conv, kv (K and V caches), idx (raw and pooled keys), ring."""
    out = {k: [] for k in ("S", "conv", "kv", "idx", "ring")}
    for (kind, what), p in zip(M.segs, M.progs):
        if kind == "run":
            for L in p.keep:
                out["S"].append(L.S)
                out["conv"].append(L.cstate)
        elif kind == "ple":
            out["ring"].append(p.ring_t)
        else:
            out["kv"] += [p.A1.kc, p.A1.vc]
            out["idx"] += [p.A1.rawk, p.A1.pooled]
    return out


def recurrence(ins, n: int) -> torch.Tensor:
    """The gated delta rule's final state in float64 from the op's inputs (the conv's output, log decays, betas,
    initial state), their n real rows; FLA head order, [48, k, v]."""
    x, gl, beta, S0 = ins
    x, gl, beta = x.reshape(-1, QKV)[:n], gl.reshape(-1, NV)[:n], beta.reshape(-1, NV)[:n]
    l2 = lambda t: t / torch.sqrt((t * t).sum(-1, keepdim=True) + EPS)  # noqa: E731
    k = l2(x[:, NK * HK:2 * NK * HK].reshape(n, NK, HK)).repeat_interleave(3, 1)   # FLA: value head j reads j // 3
    v = x[:, 2 * NK * HK:].reshape(n, NV, HK)
    S = S0.reshape(NV, HK, HK).clone()
    for t in range(n):
        S = S * torch.exp(gl[t])[:, None, None]
        S = S + k[t][:, :, None] * ((v[t] - torch.einsum("hk,hkv->hv", k[t], S)) * beta[t][:, None])[:, None, :]
    return S


def gdn_inputs64(G, cap, n: int, y=None):
    """The delta rule's inputs in float64 from a chunk's captured hn (or the given projection y), the card's weights
    and the fp32 taps: x (conv + SiLU), the log decays, the betas; the conv state zero (a prompt's first chunk)."""
    if y is None:
        y = cap[4].reshape(-1, D)[:n] @ ttnn.to_torch(G.proj).double().reshape(D, -1)
    yp = torch.cat([torch.zeros(3, QKV, dtype=torch.float64), y[:n, :QKV]])
    c = sum(G.taps32[j].double() * yp[j:j + n] for j in range(4))
    a = y[:n, QKV + ZD:QKV + ZD + NV] + ttnn.to_torch(G.dt).double().reshape(NV)
    gl = torch.nn.functional.softplus(a) * ttnn.to_torch(G.neg_a).double().reshape(NV)
    return c * torch.sigmoid(c), gl, torch.sigmoid(y[:n, QKV + ZD + NA:QKV + ZD + NA + NV])


def check_gdn(dev, res: dict, chunk: int) -> None:
    """Where the prefill's DeltaNet state departs from float64: per DeltaNet layer, the card's projection y, conv
    output x, log decays and betas against float64 from the same hn (x and the gates also from the card's own y), and
    the final state S from float64 recurrences on mixes of the card's and float64 inputs, against all-float64."""
    lane, host, g, M, P = build(dev, chunk)
    ids = np.fromfile(os.path.join(DATA, "g4_ref/ids.i32"), np.int32).tolist()
    gdns = [L["mixer"] for L in P.layers if isinstance(L["mixer"], Gdn)]
    for G in gdns:
        G.capture = True
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
    res["gdn"] = {}
    for n in (46,):
        P.prefill(ids[:n], host)
        for G in gdns:
            x, gl, beta, S0, hb, y = G.cap
            x, gl, beta = x.reshape(-1, QKV)[:n], gl.reshape(-1, NV)[:n], beta.reshape(-1, NV)[:n]
            y = y.reshape(-1, y.shape[-1])[:n]
            x64, gl64, b64 = gdn_inputs64(G, G.cap, n)
            y64 = hb.reshape(-1, D)[:n] @ ttnn.to_torch(G.proj).double().reshape(D, -1)
            xc, glc, bc = gdn_inputs64(G, G.cap, n, y=y)              # float64 from the card's y
            S64 = recurrence((x64, gl64, b64, S0), n)
            r = dict(y=rel(y, y64), x=rel(x, x64), x_from_card_y=rel(x, xc), gl=rel(gl, gl64),
                     gl_from_card_y=rel(gl, glc), beta=rel(beta, b64), beta_from_card_y=rel(beta, bc))
            for name, ins in (("S_op", None), ("S_card_inputs", (x, gl, beta)), ("S_card_x", (x, gl64, b64)),
                              ("S_card_gates", (x64, gl, beta)), ("S_card_y", (xc, glc, bc)),
                              ("S_x_bf16", (x64.to(torch.bfloat16).double(), gl64, b64))):
                S = ttnn.to_torch(G.S).double().reshape(NV, HK, HK) if ins is None else recurrence((*ins, S0), n)
                r[name] = rel(S, S64)
            res["gdn"][f"{n}.{G.il}"] = r
            print(f"{n} tokens, layer {G.il}: " + " ".join(f"{k} {v:.1e}" for k, v in r.items()), flush=True)


def warm_table(host, name: str, threads: int = 16) -> float:
    """A GGUF tensor's bytes read once into the page cache (the gather maps per_layer_token_embd: its rows are read
    in place, a cold page from the SSD); returns the seconds."""
    from concurrent.futures import ThreadPoolExecutor
    t = host.t[name]
    off, size, step = int(t.data_offset), int(t.n_bytes), 64 << 20
    fd = os.open(host.path[name], os.O_RDONLY)
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda a: len(os.pread(fd, min(step, off + size - a), a)), range(off, off + size, step)))
    finally:
        os.close(fd)
    print(f"{name}: {size / 1e9:.1f} GB in the page cache in {time.perf_counter() - t0:.1f} s", flush=True)
    return time.perf_counter() - t0


def traced_check(dev, res: dict, chunk: int, experts: str) -> None:
    """Traced prefills against untraced ones, bit for bit: prompts of W51_PROMPTS tokens (default 100,500,1000,2000:
    each one's bucket is its untraced row count too) from position 0, then follow-ups (W51_FOLLOW, default
    1024+500,4096+500: held+new; the second's attention selects). Each case untraced twice (the first compiles), then
    traced four times (the first issues and compiles, the second captures, the others replay); compared: the last X,
    every DeltaNet layer's S and conv rows, the PLE's rows; walls timed from ids to the decode-ready state."""
    from g4_gate import reset
    sizes = [int(v) for v in os.environ.get("W51_PROMPTS", "100,500,1000,2000").split(",") if v]
    follows = [tuple(int(x) for x in v.split("+")) for v in os.environ.get("W51_FOLLOW", "1024+500,4096+500").split(",")
               if v]
    lane, host, g, M, P = build(dev, chunk, experts)
    warm_table(host, "per_layer_token_embd.weight")
    rng = np.random.default_rng(0)
    gdns = [L["mixer"] for L in P.layers if isinstance(L["mixer"], Gdn)]
    buckets = P.trace_t

    def once(ids, start, traced):
        reset(M)
        if start:                                   # the held state: an untraced prefill of the first `start` ids
            P.trace_t = []
            P.prefill(ids[:start], host, into=M)
            P.hand_off(M)
            ttnn.synchronize_device(dev)
        P.trace_t = buckets if traced else []
        t0 = time.perf_counter()
        x = P.prefill(ids, host, into=M, start=start)
        P.hand_off(M)
        ttnn.synchronize_device(dev)
        wall = round(time.perf_counter() - t0, 3)
        P.trace_t = buckets
        return wall, [x.clone(), torch.stack([G.state() for G in gdns]), torch.stack([G.conv_state() for G in gdns]),
                      P.ple.ring_rows()]

    cases = [(n, 0) for n in sizes] + [(h + f, h) for h, f in follows]
    for n, start in cases:
        ids = rng.integers(0, 200000, n).tolist()
        r = {}
        plain = [once(ids, start, False) for _ in range(2)]
        traced = [once(ids, start, True) for _ in range(3)]
        P.walls = {}
        if hasattr(P.experts, "parts"):
            P.experts.parts = dict.fromkeys(P.experts.parts, 0.0)
        traced.append(once(ids, start, True))
        r["walls"] = {k: round(v, 3) for k, v in sorted(P.walls.items(), key=lambda kv: -kv[1])}
        r["expert_parts"] = {k: round(v, 3) for k, v in getattr(P.experts, "parts", {}).items()}
        P.walls = None
        ref = plain[-1][1]
        r["untraced_s"], r["traced_s"] = [w for w, _ in plain], [w for w, _ in traced]
        r["bitwise"] = [all(torch.equal(a, b) for a, b in zip(got, ref)) for _, got in traced]
        r["max_abs"] = [max(float((a - b).abs().max()) for a, b in zip(got, ref)) for _, got in traced]
        res[f"{start}+{n - start}"] = r
        print(f"{start}+{n - start} tokens: untraced {r['untraced_s']} s, traced {r['traced_s']} s; bitwise {r['bitwise']}"
              f" (max abs {r['max_abs']})", flush=True)
        print(f"  last traced run's walls: {r['walls']}; experts' parts {r['expert_parts']}", flush=True)
    res["traces"] = len(P.captured())
    # the round trips a traced run still makes per layer, and the hand-off, timed alone (ms, mean of 10)
    B = P.bufs[min(P.bufs)]

    def ms(f, k=10):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(k):
            f()
        return round(1e3 * (time.perf_counter() - t0) / k, 2)
    res["alone_ms"] = r = {
        "synchronize": ms(lambda: ttnn.synchronize_device(dev)),
        "hb: to row-major + read": ms(lambda: ttnn.to_torch(ttnn.to_layout(B["hb"], RM))),
        "logits: to row-major + read": ms(lambda: ttnn.to_torch(ttnn.to_layout(B["logits"], RM))),
        "hb: read tiles": ms(lambda: ttnn.to_torch(B["hb"])),
        "X: read tiles": ms(lambda: ttnn.to_torch(B["X"])),
        "write [T, 2560] fp32 row-major + tilize": ms(lambda: ttnn.to_layout(ttnn.from_torch(
            torch.zeros(1, 1, B["X"].shape[-2], D), dtype=FP32, layout=RM, device=dev), TILE)),
        "hand_off": ms(lambda: P.hand_off(M), 3),
    }
    T0 = min(P.bufs)
    for key in (("deltanet", 0, T0, True), ("deltanet", 1, T0, True), ("attn pre", 3, T0)) + tuple(
            k for k in P.traces if k[0] == "attn post" and k[1] == 3 and k[2] == T0):
        tid = P.traces.get(key)
        if tid is not None and tid is not False:
            r[f"replay {key[:2]}"] = ms(lambda: (ttnn.execute_trace(dev, tid, cq_id=0, blocking=False),
                                                 ttnn.synchronize_device(dev)))
    E = P.experts
    if isinstance(E, CpuExperts):
        hb, lg = B["hb"], B["logits"]
        r["cpu experts, layer 0, whole"] = ms(lambda: (E(0, hb, lg, T0 - 28), ttnn.synchronize_device(dev)))
    T0 = B["X"].shape[-2]
    host = ttnn.from_torch(torch.zeros(1, 1, T0, D), dtype=FP32, layout=RM)
    dst = ttnn.to_device(host, dev)
    r["routed: host tensor made"] = ms(lambda: ttnn.from_torch(torch.zeros(1, 1, T0, D), dtype=FP32, layout=RM))
    r["routed: copy_host_to_device_tensor"] = ms(lambda: (ttnn.copy_host_to_device_tensor(host, dst),
                                                          ttnn.synchronize_device(dev)))
    r["routed: from_torch to the device"] = ms(lambda: (ttnn.from_torch(torch.zeros(1, 1, T0, D), dtype=FP32,
                                                                        layout=RM, device=dev),
                                                        ttnn.synchronize_device(dev)))
    print(f"alone (ms): {r}", flush=True)
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        P.hand_off(M)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(18)


def op_times(dev, res: dict, chunk: int) -> None:
    """The hyper-connection mix and combine and the DeltaNet's gated norm op by op at W51_T rows (default 8192):
    layer 0's weights on random inputs, each op synchronized alone (median of 5, ms), against the bytes it moves."""
    lane, host, g, M, P = build(dev, chunk, "cpu", n_layers=1, model=False)
    T = int(os.environ.get("W51_T", 8192))
    L, G = P.layers[0], P.layers[0]["mixer"]
    rnd = lambda shape, dt=FP32, sc=0.5: ttnn.from_torch(torch.randn(*shape) * sc, dtype=dt, layout=TILE,  # noqa: E731
                                                        device=dev)
    X, out = rnd([1, HC, T, D]), rnd([1, 1, T, D])
    times = {}

    def op(name, f):
        ms = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = f()
            ttnn.synchronize_device(dev)
            ms.append(1e3 * (time.perf_counter() - t0))
        times[name] = round(statistics.median(ms), 2)
        return r
    H = L["attn"]
    xn = op("mix: rms_norm X", lambda: ttnn.rms_norm(X, epsilon=EPS, compute_kernel_config=HI))
    xb = op("mix: typecast xn bf16", lambda: ttnn.typecast(xn, BF16))
    mm = op("mix: matmul down|inject (4 streams)", lambda: ttnn.matmul(xb, H.wdi, dtype=FP32, compute_kernel_config=CK))
    lo = op("mix: sum4", lambda: sum4(mm))
    a = op("mix: silu(lo / 4) bf16", lambda: ttnn.typecast(ttnn.silu(ttnn.multiply(cut(lo, 0, LR), 1.0 / HC)), BF16))
    ar = op("mix: repeat a", lambda: ttnn.repeat(a, [1, HC, 1, 1]))
    gate = op("mix: matmul up (4 streams)", lambda: ttnn.matmul(ar, H.up, dtype=FP32, compute_kernel_config=CK))
    hs = op("mix: gate_sum kernel", lambda: gate_sum(dev, xn, gate, H.norm_t))
    a2, w2 = op("mix: hc_lo kernel", lambda: hc_lo(dev, mm))
    w4r = ttnn.multiply(ttnn.sigmoid(ttnn.multiply(cut(lo, LR, LR + HC), 1.0 / HC)), 2.0)
    for name, ref_t, got_t in (("a", a, a2), ("inject", w4r, cut(w2, 0, HC))):
        ref, got = ttnn.to_torch(ref_t).float(), ttnn.to_torch(got_t).float()
        dd = (got - ref).abs()
        res[f"hc_lo_{name}"] = dict(bitwise=bool(torch.equal(got, ref)), max_abs=float(dd.max()),
                                    differ=float((dd > 0).float().mean()))
        print(f"hc_lo {name} against the stock ops: {res[f'hc_lo_{name}']}", flush=True)
    global HCGATE_NC
    for nc in (1, 2, 4, 8):
        HCGATE_NC = nc
        hg = op(f"mix: hc_gate kernel, NC {nc}", lambda: hc_gate(dev, a, H.up, xn, H.norm_t))
        ref, got = ttnn.to_torch(hs).float(), ttnn.to_torch(hg).float()
        dd = (got - ref).abs()
        res[f"hc_gate_nc{nc}"] = dict(max_abs=float(dd.max()), mean_abs=float(dd.mean()),
                                      max_rel=float(dd.max() / ref.abs().max()),
                                      differ=float((dd > 0).float().mean()))
        print(f"hc_gate NC {nc} against up matmul + gate_sum: {res[f'hc_gate_nc{nc}']}", flush=True)
    HCGATE_NC = 0
    w4 = op("mix: inject weights", lambda: ttnn.permute(ttnn.multiply(ttnn.sigmoid(ttnn.multiply(
        cut(lo, LR, LR + HC), 1.0 / HC)), 2.0), [0, 3, 2, 1]))
    op("mix: whole", lambda: H.mix(X))
    wo = op("combine: multiply w4 out", lambda: ttnn.multiply(w4, out))
    op("combine: add", lambda: ttnn.add(X, wo))
    def dev_ms(name, f):                # the card's time alone: f captured once, replayed
        f()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        f()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ms = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ms.append(1e3 * (time.perf_counter() - t0))
        ttnn.release_trace(dev, tid)
        times[f"[card] {name}"] = round(statistics.median(ms), 2)
    Xd = ttnn.clone(X)
    dev_ms("combine: multiply + add", lambda: ttnn.add(X, ttnn.multiply(w4, out)))
    dev_ms("combine: kernel", lambda: hc_combine(Xd, out, w4))
    dev_ms("mix: whole", lambda: H.mix(X))
    dev_ms("mix: gate_sum kernel", lambda: gate_sum(dev, xn, gate, H.norm_t))
    dev_ms("mix: up matmul + gate_sum", lambda: gate_sum(dev, xn, ttnn.matmul(ar, H.up, dtype=FP32,
                                                                          compute_kernel_config=CK), H.norm_t))
    dev_ms("mix: hc_gate kernel", lambda: hc_gate(dev, a, H.up, xn, H.norm_t))
    dev_ms("mix: sum4 + silu + inject (stock)", lambda: (ttnn.typecast(ttnn.silu(ttnn.multiply(cut(
        sum4(mm), 0, LR), 1.0 / HC)), BF16), ttnn.permute(ttnn.multiply(ttnn.sigmoid(ttnn.multiply(
            cut(sum4(mm), LR, LR + HC), 1.0 / HC)), 2.0), [0, 3, 2, 1])))
    dev_ms("mix: hc_lo kernel + permute", lambda: ttnn.permute(cut(hc_lo(dev, mm)[1], 0, HC), [0, 3, 2, 1]))
    dev_ms("mix: matmul down (fp32 xn)", lambda: ttnn.matmul(xn, H.wdi, dtype=FP32, compute_kernel_config=CK))
    dev_ms("mix: typecast + matmul down", lambda: ttnn.matmul(ttnn.typecast(xn, BF16), H.wdi, dtype=FP32,
                                                              compute_kernel_config=CK))
    dev_ms("mix: rms_norm X", lambda: ttnn.rms_norm(X, epsilon=EPS, compute_kernel_config=HI))
    yy = rnd([1, 1, T, QKV + ZD + 2 * NA], BF16)
    dev_ms("tile move: heads gdn z", lambda: heads(yy, NV, HK, QKV))
    ref = ttnn.to_torch(ttnn.add(X, ttnn.multiply(w4, out)))
    Xc = ttnn.clone(X)
    op("combine: kernel, in place", lambda: hc_combine(Xc, out, w4))
    Xc = hc_combine(ttnn.clone(X), out, w4)
    d = (ttnn.to_torch(Xc) - ref).abs()
    res["combine_kernel"] = dict(bitwise=bool(torch.equal(ttnn.to_torch(Xc), ref)), max_abs=float(d.max()),
                                 max_rel=float(d.max() / ref.abs().max()))
    print(f"combine kernel against multiply + add: {res['combine_kernel']}", flush=True)
    y = rnd([1, 1, T, QKV + ZD + 2 * NA], BF16)
    o = rnd([1, NV, T, HK])
    z = op("gated norm: z = sigmoid(heads(y's gate))", lambda: ttnn.sigmoid(heads(y, NV, HK, QKV)))
    on = op("gated norm: rms_norm o", lambda: ttnn.rms_norm(o, epsilon=EPS, compute_kernel_config=HI))
    on = op("gated norm: * o_norm * z", lambda: ttnn.multiply(ttnn.multiply(on, G.o_norm), z))
    ob = op("gated norm: typecast bf16", lambda: ttnn.typecast(on, BF16))
    op("gated norm: tokens (head-major -> token rows)", lambda: tokens(ob))
    # the chunk's token inputs: the batched gather against the Python loop's, timed and bit for bit
    global TILEMOVE, INPUT_THREADS
    ids = np.random.default_rng(0).integers(0, 200000, T).tolist()
    warm_table(host, "per_layer_token_embd.weight")
    got = {}
    for k in (32, 0):
        INPUT_THREADS = k
        token_inputs(host, ids, 0, T, T)
        t0 = time.perf_counter()
        got[k] = token_inputs(host, ids, 0, T, T)
        times[f"token inputs, {k or 'the Python loop'} threads" if k else "token inputs, the Python loop"] = round(
            1e3 * (time.perf_counter() - t0), 2)
    INPUT_THREADS = 32
    res["token_inputs_bitwise"] = all(np.array_equal(a, b) for a, b in zip(got[32], got[0]))
    print(f"token inputs, batched against the Python loop, bitwise: {res['token_inputs_bitwise']}", flush=True)
    # the tile moves against the row-major round trip, every shape the prefill uses, bit for bit
    same = {}
    for name, (W, n, d, c0, dt) in {"q|gate": (13312, NH, 2 * HD, 0, BF16), "k": (13312, NKV, HD, 2 * NH * HD, BF16),
                                    "v": (13312, NKV, HD, 2 * NH * HD + NKV * HD, BF16), "iq": (640, IH, IHD, 0, FP32),
                                    "gdn z": (QKV + ZD + 2 * NA, NV, HK, QKV, BF16), "ple key": (12800, HC, D, 0, FP32)}.items():
        x = rnd([1, 1, T, W], dt)
        TILEMOVE = True
        a, ta = heads(x, n, d, c0), op(f"tile move: heads {name}", lambda: heads(x, n, d, c0))
        b, tb = tokens(a), op(f"tile move: tokens {name}", lambda: tokens(a))
        TILEMOVE = False
        a0 = op(f"row-major: heads {name}", lambda: heads(x, n, d, c0))
        b0 = op(f"row-major: tokens {name}", lambda: tokens(a0))
        TILEMOVE = True
        same[name] = bool(torch.equal(ttnn.to_torch(a), ttnn.to_torch(a0)) and torch.equal(ttnn.to_torch(b), ttnn.to_torch(b0)))
    res["tile_move_bitwise"] = same
    print(f"tile moves against the row-major round trip, bitwise: {same}", flush=True)
    gb = 4 * HC * T * D / 1e9
    res.update(T=T, ms=times, x_gb=gb)
    for k, v in times.items():
        print(f"  {v:8.2f} ms  {k}", flush=True)
    print(f"X is {gb:.3f} GB (fp32, 4 streams): one pass at 0.5 TB/s {1e3 * gb / 500:.2f} ms", flush=True)


def regress(dev, res: dict, chunk: int) -> None:
    """Prompts of W51_PROMPTS tokens (default 4096) with the card's experts: CardExperts' SwiGLU buffer a bare allocation
    (its writer zeroes the padding) against zero-filled per call (ttnn.zeros, the regression's cause); walls, and the
    outputs (the last X, every DeltaNet S) compared bit for bit. W51_REGRESS names a switch instead, on against off:
    padskip (g7_experts.PAD_SKIP, the down's padded k-tiles skipped), gnorm (GNORM_OP, the gated norm by ttnn's op), xrows (XROWS, the experts over several chunks)."""
    import g7_experts
    from g4_gate import reset
    what = os.environ.get("W51_REGRESS", "")
    # (module, name, the value on, off, the default restored after)
    flag = {"padskip": (g7_experts, "PAD_SKIP", True, False, True),
            "gnorm": (sys.modules[__name__], "GNORM_OP", True, False, True),
            "gexpbatch": (g7_experts, "READ_BATCH", 5, 1, g7_experts.READ_BATCH),
            "w16": (g7_experts, "W16", True, False, g7_experts.W16),
            "xrows": (sys.modules[__name__], "XROWS", XROWS or 16384, 0, XROWS),
            "mtdn": (g7_experts, "MT_DN", 8, 5, g7_experts.MT_DN),
            "selfast": (sys.modules[__name__], "SEL_FAST", True, False, SEL_FAST),
            "scorek": (sys.modules[__name__], "SCOREK", True, False, SCOREK)}.get(what)
    names = ("on", "off", "on again") if flag else ("bare", "zero-filled", "bare again")
    sizes = [int(v) for v in os.environ.get("W51_PROMPTS", "4096").split(",")]
    lane, host, g, M, P = build(dev, chunk, "card")
    warm_table(host, "per_layer_token_embd.weight")
    rng = np.random.default_rng(0)
    E = P.experts
    gdns = [L["mixer"] for L in P.layers if isinstance(L["mixer"], Gdn)]
    for n in sizes:
        ids = rng.integers(0, 200000, n).tolist()
        r, outs = {}, {}
        for name in names:
            if flag:
                setattr(flag[0], flag[1], flag[3] if name == "off" else flag[2])
            else:
                E.zero_fill = name == "zero-filled"
            walls = []
            for _ in range(2):
                reset(M)
                t0 = time.perf_counter()
                x = P.prefill(ids, host, into=M)
                P.hand_off(M)
                ttnn.synchronize_device(dev)
                walls.append(round(time.perf_counter() - t0, 3))
            outs[name] = [x.clone(), torch.stack([G.state() for G in gdns])]
            r[name] = walls
            print(f"{n} {name}: {walls} s; chunks an expert call: calls {P.xgroups}", flush=True)
        E.zero_fill = False
        if flag:
            setattr(flag[0], flag[1], flag[4])                    # back to the default
        r["bitwise"] = all(torch.equal(a, b) for a, b in zip(outs[names[0]], outs[names[1]]))
        r["repeat_bitwise"] = all(torch.equal(a, b) for a, b in zip(outs[names[0]], outs[names[2]]))
        r["max_rel"] = [float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))
                        for a, b in zip(outs[names[0]], outs[names[1]])]   # [last X, DeltaNet S]
        r["diff_share"] = [float((a != b).double().mean()) for a, b in zip(outs[names[0]], outs[names[1]])]
        print(f"{n}: max |diff| / max |x| (X, S) {r['max_rel']}, share of values differing {r['diff_share']}", flush=True)
        print(f"{n}: {names[0]} vs {names[1]} bitwise {r['bitwise']} (repeat {r['repeat_bitwise']})", flush=True)
        res[f"prompt_{n}"] = r


def dispatch_prefill(dev, res: dict, chunk: int) -> None:
    """Where a short prefill's time goes (W51_PROMPTS, default 128,512,2048 tokens; card and CPU experts): the wall from
    ids to the decode-ready state; the host's time to issue it (the ops are queued, the host waits only at reads and
    the final sync); the ttnn calls it made, counted by wrapping the module's functions for one run, by name."""
    import collections
    import functools
    from g4_gate import reset
    sizes = [int(v) for v in os.environ.get("W51_PROMPTS", "128,512,2048").split(",")]
    lane, host, g, M, P = build(dev, chunk, "card")
    card, cpu = P.experts, CpuExperts(dev, lane)
    warm_table(host, "per_layer_token_embd.weight")
    rng = np.random.default_rng(0)
    counts = collections.Counter()
    names = [k for k in dir(ttnn) if not k.startswith("_") and callable(getattr(ttnn, k))
             and k not in ("Tensor", "Shape", "CoreCoord", "CoreRange", "CoreRangeSet", "MeshShape", "MeshCoordinate")
             and not isinstance(getattr(ttnn, k), type)]
    orig = {k: getattr(ttnn, k) for k in names}

    def counting(on):
        for k, f in orig.items():
            if on:
                @functools.wraps(f)
                def w(*a, _f=f, _k=k, **kw):
                    counts[_k] += 1
                    return _f(*a, **kw)
                setattr(ttnn, k, w)
            else:
                setattr(ttnn, k, f)
    for n in sizes:
        for name, E in (("card", card), ("cpu", cpu)):
            P.experts = E
            walls = []
            for rep in range(3):
                reset(M)
                ids = rng.integers(0, 200000, n).tolist()
                counts.clear()
                counting(rep == 2)
                t0 = time.perf_counter()
                P.prefill(ids, host, into=M)
                t1 = time.perf_counter()
                P.hand_off(M)
                ttnn.synchronize_device(dev)
                t2 = time.perf_counter()
                counting(False)
                walls.append((round(t1 - t0, 3), round(t2 - t0, 3)))
            if rep == 2 and n == sizes[0]:           # the calls by phase: one more counted run, the counter read at each tick
                counting(True)
                by, last = collections.Counter(), [0]

                def at_tick(kind, phase, il):
                    c = sum(counts.values())
                    by[f"{'attention' if is_attention(il) else 'deltanet'}: {kind}: {phase}" if il >= 0 else
                       f"{kind}: {phase}"] += c - last[0]
                    last[0] = c
                P.mem = at_tick
                reset(M)
                counts.clear()
                P.prefill(rng.integers(0, 200000, n).tolist(), host, into=M)
                P.mem = None
                counting(False)
                res[f"{name}_{n}_by_phase"] = by.most_common()
                print(f"  {name} calls by phase (all layers): {by.most_common(14)}", flush=True)
            top = counts.most_common(12)
            res[f"{name}_{n}"] = dict(walls_issue_then_total=walls, ttnn_calls=sum(counts.values()), top=top)
            print(f"{n} {name}: (prefill returned, total) {walls}; ttnn calls {sum(counts.values())} "
                  f"({sum(counts.values()) / 48:.0f} a layer); top {top[:8]}", flush=True)


def short_prefill(dev, res: dict, chunk: int) -> None:
    """Short prompts (W51_PROMPTS, default 128,256,512,1024,2048,4096 tokens of random ids): the prefill's wall from ids
    to the decode-ready state with the card's experts (every expert layer streamed over PCIe) and with the CPU lane's
    (cl_moe_m, up to the lane's max_m tokens a call), twice each (the first may compile); then one 512-token prompt
    through the CPU experts at 8 tokens a call and at max_m: the outputs must be equal bit for bit."""
    from g4_gate import reset
    sizes = [int(v) for v in os.environ.get("W51_PROMPTS", "128,256,512,1024,2048,4096").split(",")]
    lane, host, g, M, P = build(dev, chunk, "card")
    card, cpu = P.experts, CpuExperts(dev, lane)
    res["lane_max_m"] = getattr(lane, "max_m", 8)
    warm_table(host, "per_layer_token_embd.weight")
    rng = np.random.default_rng(0)
    for n in sizes:
        r = {}
        for name, E in (("card", card), ("cpu", cpu)):
            P.experts, walls = E, []
            for _ in range(2):
                reset(M)
                ids = rng.integers(0, 200000, n).tolist()
                t0 = time.perf_counter()
                P.prefill(ids, host, into=M)
                P.hand_off(M)
                ttnn.synchronize_device(dev)
                walls.append(round(time.perf_counter() - t0, 3))
            r[name] = walls
        if n in (512, 2048):                          # where the time goes: a synchronized run per kind
            for name, E in (("card", card), ("cpu", cpu)):
                P.experts, P.prof = E, {}
                reset(M)
                P.prefill(rng.integers(0, 200000, n).tolist(), host, into=M)
                b = P.prof
                P.prof = None
                g = {}
                for k, v in b.items():
                    g[k.split(":")[-1].strip() if k.startswith("experts") else k.split(":")[0]] = g.get(
                        k.split(":")[-1].strip() if k.startswith("experts") else k.split(":")[0], 0.0) + v
                r[f"breakdown_{name}"] = {k: round(v, 3) for k, v in sorted(b.items(), key=lambda kv: -kv[1])}
                if name == "cpu":
                    cpu.parts = dict.fromkeys(cpu.parts, 0.0)
                    reset(M)
                    P.prefill(rng.integers(0, 200000, n).tolist(), host, into=M)
                    ttnn.synchronize_device(dev)
                    r["cpu_parts_s"] = {k: round(v, 3) for k, v in cpu.parts.items()}
                    print(f"  {n} cpu experts' host parts (unsynchronized run): {r['cpu_parts_s']}", flush=True)
                print(f"  {n} {name}: synchronized {sum(b.values()):.2f} s: " + ", ".join(
                    f"{k} {v:.2f}" for k, v in sorted(g.items(), key=lambda kv: -kv[1])[:8]), flush=True)
        res[f"prompt_{n}"] = r
        print(f"{n} tokens: card experts {r['card']} s, CPU experts {r['cpu']} s", flush=True)
    P.experts = cpu
    ids = rng.integers(0, 200000, 512).tolist()
    outs = []
    for m in (8, res["lane_max_m"]):
        lane.max_m = m
        reset(M)
        outs.append(P.prefill(ids, host, into=M).clone())
    lane.max_m = res["lane_max_m"]
    res["cpu_bits_equal_8_vs_max_m"] = bool(torch.equal(outs[0], outs[1]))
    print(f"CPU experts, 8 vs {res['lane_max_m']} tokens a call: equal {res['cpu_bits_equal_8_vs_max_m']}", flush=True)
    P.experts = card


def time_prefill(dev, res: dict, chunk: int) -> None:
    """Prefill tok/s from ids to the decode-ready state (the hand-off into the decode Model included), card experts,
    new random ids every prompt (the card's time depends on shapes, not tokens; the host's PLE gather reads 16 rows a
    token of a 28.8 GB table, kept in the page cache: warm_table) for W51_PROMPTS (default 4096,8192) tokens; then one
    synchronized run of the first prompt size per phase (the breakdown: card work per phase, the expert stream's
    host-blocking copies apart)."""
    from g4_gate import reset
    sizes = [int(v) for v in os.environ.get("W51_PROMPTS", "4096,8192").split(",")]
    lane, host, g, M, P = build(dev, chunk, "card")
    res["resident"] = sorted(P.experts.resident)
    res["dram_after_build_gb"] = dram_gb(dev)
    res["ple_table_warm_s"] = warm_table(host, "per_layer_token_embd.weight")
    rng = np.random.default_rng(0)
    from g7_experts import NG
    E = P.experts
    # W51_CARD_GROUPS: the streamed layers' expert groups (of NG) on the card, the rest on the lane (SplitExperts)
    for cg in [int(v) for v in os.environ.get("W51_CARD_GROUPS", str(NG)).split(",")]:
        E.lane, E.card_groups = (lane if cg < NG else None), cg
        for n in sizes:
            walls = []
            for rep in range(int(os.environ.get("W51_REPS", 3))):     # the first also compiles
                ids = rng.integers(0, 200000, n).tolist()
                reset(M)
                E.seconds = {k: 0.0 for k in E.seconds}
                E.cpu_s = {k: 0.0 for k in E.cpu_s}
                t0 = time.perf_counter()
                P.prefill(ids, host, into=M if INTO else None)
                P.hand_off(M)
                walls.append(time.perf_counter() - t0)
                print(f"{n} tokens in chunks of {chunk}, {cg} of {NG} expert groups on the card: {walls[-1]:.2f} s "
                      f"-> {n / walls[-1]:.0f} tok/s; host {dict((k, round(v, 2)) for k, v in E.seconds.items())}"
                      + (f", lane {dict((k, round(v, 2)) for k, v in E.cpu_s.items())}" if E.lane else ""), flush=True)
            res[f"prompt_{n}" + (f"_g{cg}" if cg < NG else "")] = dict(
                walls_s=walls, tok_s=n / min(walls[1:]), host_s=dict(E.seconds), lane_s=dict(E.cpu_s))
    E.lane, E.card_groups = None, NG
    P.prof = {}
    reset(M)
    P.experts.seconds = {k: 0.0 for k in P.experts.seconds}
    P.prefill(rng.integers(0, 200000, sizes[0]).tolist(), host)
    res["breakdown_synchronized_s"] = dict(sorted(P.prof.items(), key=lambda kv: -kv[1]))
    res["breakdown_host_s"] = dict(P.experts.seconds)
    P.prof = None
    for k, v in res["breakdown_synchronized_s"].items():
        print(f"  {1e3 * v:9.1f} ms  {k}", flush=True)
    print(f"  of which the experts' host side: {res['breakdown_host_s']}", flush=True)
    # the expert stream alone: every streamed layer into one slot, nothing else on the card
    streamed = [L for L in E.layers.values() if L.il not in E.resident]
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for L in streamed:
        for p, h in zip(E.slot_tensors[0], L.host):
            for k, t in h.items():
                ttnn.copy_host_to_device_tensor(t, p[k], cq_id=ttnn.QueueId(1))
    ttnn.synchronize_device(dev)
    s = time.perf_counter() - t0
    gb = sum(t.volume() * t.element_size() for L in streamed for h in L.host for t in h.values()) / 1e9
    res["stream_alone"] = dict(layers=len(streamed), gb=gb, s=s, gb_s=gb / s)
    print(f"the expert stream alone: {len(streamed)} layers, {gb:.1f} GB in {s:.2f} s -> {gb / s:.1f} GB/s", flush=True)


def sens(dev, res: dict, chunk: int) -> None:
    """Step 0 of the expert-kernel scope: one build, then per W51_VARIANTS (default base, nostream, nodecode, noscale,
    nomatmul, noscatter; CardExperts.skip, timing only, wrong outputs; "+" joins several, nodecode+nomatmul) and W51_PROMPTS size (default 8192), W51_REPS
    prompts (the first compiles: the best of the rest), one synchronized breakdown run, the cores' balance (the
    busiest core's LPT cost over the mean core's, summed over the calls). W51_OUT_PT: base's output for one fixed prompt
    saved there (torch.save), to compare builds bit for bit."""
    from g4_gate import reset
    sizes = [int(v) for v in os.environ.get("W51_PROMPTS", "8192").split(",")]
    variants = os.environ.get("W51_VARIANTS", "base,nostream,nodecode,noscale,nomatmul,noscatter").split(",")
    lane, host, g, M, P = build(dev, chunk, "card")
    E = P.experts
    res["resident"] = sorted(E.resident)
    warm_table(host, "per_layer_token_embd.weight")
    rng = np.random.default_rng(0)
    fixed = rng.integers(0, 200000, sizes[0]).tolist()
    for n, v in [(n, v) for n in sizes for v in variants]:
        key = v if len(sizes) == 1 else f"{v}_{n}"
        E.skip = set() if v == "base" else set(v.split("+"))      # skips combine: nodecode+nomatmul
        walls = []
        for rep in range(int(os.environ.get("W51_REPS", 3))):
            reset(M)
            E.seconds = {k: 0.0 for k in E.seconds}
            E.balance = {}
            t0 = time.perf_counter()
            P.prefill(rng.integers(0, 200000, n).tolist(), host)
            ttnn.synchronize_device(dev)
            walls.append(time.perf_counter() - t0)
        host_s, bal = dict(E.seconds), {k: round(a / b, 3) for k, (a, b) in E.balance.items()}
        P.prof = {}
        reset(M)
        P.prefill(rng.integers(0, 200000, n).tolist(), host)
        prof, P.prof = dict(sorted(P.prof.items(), key=lambda kv: -kv[1])), None
        card = sum(t for k, t in prof.items() if "(host" not in k and not k.endswith(": route"))
        res[key] = dict(walls_s=walls, best_s=min(walls[1:] or walls), host_s=host_s, balance=bal,
                      breakdown_synchronized_s=prof, synchronized_card_s=card)
        top = ", ".join(f"{k} {1e3 * t:.0f}" for k, t in prof.items() if k.startswith("experts"))
        print(f"{v:10s} {n} tokens: best {res[key]['best_s']:.2f} s (walls {[round(w, 2) for w in walls]}); "
              f"synchronized card {card:.2f} s; balance {bal}; host {dict((k, round(t, 2)) for k, t in host_s.items())}"
              f"\n           experts ms: {top}", flush=True)
        if v == "base" and n == sizes[0] and os.environ.get("W51_OUT_PT"):
            reset(M)
            torch.save(P.prefill(fixed, host).clone(), os.environ["W51_OUT_PT"])
    E.skip = set()


def main() -> None:
    from g3_layer import WORKER_L1
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    chunk = int(os.environ.get("W51_CHUNK", 8192))
    os.makedirs(OUT, exist_ok=True)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "mode": mode, "chunk": chunk, "ctx": CTX}
    experts = "card" if mode in ("time", "mem", "short", "dispatch", "regress", "sens") else os.environ.get("W51_EXPERTS", "cpu")
    # the card's experts stream on a second command queue
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=int(os.environ.get("W51_TRACE_MB", 64)) << 20,
                                worker_l1_size=WORKER_L1,
                                num_command_queues=2 if experts == "card" else 1)
    try:
        res["experts"] = experts
        if mode == "refstates":
            save_ref_states(dev, res)
        elif mode == "states":
            check_states(dev, res, chunk)
        elif mode == "check":
            check_short(dev, res, chunk, experts)
        elif mode == "gdn":
            check_gdn(dev, res, chunk)
        elif mode == "long16":
            check_16k(dev, res, chunk, experts)
        elif mode == "long":
            check_long(dev, res, chunk, experts)
        elif mode == "time":
            time_prefill(dev, res, chunk)
        elif mode == "mem":
            mem_inventory(dev, res, chunk)
        elif mode == "short":
            short_prefill(dev, res, chunk)
        elif mode == "dispatch":
            dispatch_prefill(dev, res, chunk)
        elif mode == "traced":
            traced_check(dev, res, chunk, experts)
        elif mode == "ops":
            op_times(dev, res, chunk)
        elif mode == "sens":
            sens(dev, res, chunk)
        elif mode == "regress":
            regress(dev, res, chunk)
    finally:
        ttnn.close_mesh_device(dev)
    tag = os.environ.get("W51_TAG")
    path = os.path.join(OUT, f"g7_prefill_{mode}" + ("" if mode == "time" else f"_{res['experts']}") +
                        (f"_{tag}" if tag else "") + ".json")
    res["fidelity"] = os.environ.get("W51_FIDELITY", "HiFi3")
    if SPARSE:
        res["sparse"] = dict(SPARSE_STATS, checked=SPARSE_CHECK)
        print(f"sparse attention: {res['sparse']}", flush=True)
    res["sdpa"] = [os.environ.get("W51_SDPA_FID", "HiFi3"), SDPA_K]
    res["sparse_fidelity"] = os.environ.get("W51_SPARSE_FID", "HiFi3")
    res["experts_fidelity"] = os.environ.get("W51_EXP_FID", "HiFi3")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
