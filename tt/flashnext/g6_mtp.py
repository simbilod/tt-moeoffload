"""Strata's MTP draft layer on the card, one draft step as five programs, chained up to 4 steps a pass:

  front     tt/kernels/fmtp_front_*.cpp: R (the target's final residual streams, or the layer's own) and the token's
            embedding -> X = fc_hidden(rms_10240(R) w_h) + fc_embedding(rms(e) w_e) per stream, M <= 4 rows
  attn 1    g3_attn1.py's M-row program built from the draft layer's weights (MtpAttn1: no indexer): the
            hyper-connection read, q|gate, k, v into the layer's own KV cache at the target's positions
  attn 2    tt/kernels/fmtp_attn_*.cpp (MtpAttn2): g3_attn2.py's M-row program with dense attention over every cell
            (Strata's runtime choice) on the whole grid and no selection tensor, then o_proj, the FFN read, the
            router's logits and the shared expert (routed sum 0)
  experts   tt/kernels/fmtp_exp_*.cpp: the router's top 10 and weights on the card, the 10 experts from Q2_0's codes
            as bfp8_b tiles of (code - 1) and fp32 scales (exact), X_s += w_f,s y for the pass's last row -> R
  head      g3_head.Head from the layer's final mixer and the target's LM head rows (the subset head by default:
            Strata's draft vocabulary draft_vocab_fr.bin, 46,211 tokens; the full head, 248,320, as an option), the
            top 64 on the card (word 0 = the draft, an index into the head's rows)

A pass (the verify loop's): step 0 = the catch-up, M rows at the cells c0 .. c0 + M - 1 (cell c pairs the target's R_c
with the token at c + 1; their K/V written, the draft from the last row), then chained steps at the cells after it (the
previous step's R and draft; the front maps the head's row to its token through the vocabulary map). Every step's
programs are built once with their own step tensor (set_pass writes the pass's rows, tokens and step words in place
into host tensors made once, then copies them), so a pass replays as one trace. Dense attention covers cells up to
CTX_DENSE (8,799).

  python3 tt_moeoffload/launch.py --memory-max $((64 << 30)) --env G6_STAGE=check --engine tt/flashnext/g6_mtp.py

G6_STAGE (G6_HEADS: fr,full; results/g6_mtp_<stage>.json):
  build    the card forms into $TT_MOEOFFLOAD_DATA/w43f_mtp (experts, the bf16 embedding table in 64-byte pages, weights)
  ref      the float64 references (tt/spec/mtp.py) without the card; G6_EMU=1: the card's precision on the host
           (CardMtp, after G6_STAGE=weights dumped the card's weights)
  layer    each program against float64 and against references at the card's own precision and inputs; G6_LAYER_AT
           from a later cell of LONG (the attention's O by head; G6_PARTS=1: both combine stages from the
           card's partials, on the host)
  check    drafts by depth, step 0's logits, the KV catch-up, acceptance against the target's greedy tokens, subset
           and full heads, against float64 (and the emulation when its reference exists); the A prompts' cells only
  long     LONG's 4,785-cell prompt caught up and 16 chained drafts against float64, at cell offsets 0 and
           3,900 (to 8,699 cells: the attention's multi-block partials and two-stage combine); G6_LONG_CELLS a shorter
           prompt (its prefix), G6_LONG_OFFSETS the offsets: 450 / 800 cells reach the combine's groups of 6-7
           (2026-11-07: cb 34 read past its end there, every draft wrong at 9..47 blocks)
  replay   the card's drafts at every decode position of the census A prompts, E per pass against float32
  time     traced: one step and a 3-step chain at cells 100 and 7,900, each program alone; set_pass's host time
  profile  one traced step per program and phase from grid.h's per-core clocks, against each program's floor

Measured 2026-10-26: one step 0.927 / 1.004 ms (subset head) and 2.025 / 2.101 ms (full) at cells 100 / 7,900,
a 3-step chain 2.770 / 2.998 and 6.058 / 6.285 ms (earlier: 1.02 / 1.10, 2.11 / 2.20; chains 3.04 / 3.29, 6.33 / 6.58);
per program at 7,900, front 67.6, attn 1 171.2, attn 2 248.0, experts 198.8, head 353.6 (full 1,451.1) us traced
alone; set_pass 95 us (4-row catch-up) / 105 us (1 row + 3 steps), from 600 / 1,031. E at --spec 4 2.441 (card) vs
2.454 (float32), full head 2.628 vs 2.640; drafts equal float64 95.8 / 86.5 / 87.5 / 84.4% by depth on the A prompts,
100 / 100 / 93.8 / 93.8% at 5.8K and 8.7K cells (long). End to end at ~7.5K cells (g6_spec.py, G6S_GROUP=long):
drafting 6.43 -> 3.43 ms a pass (subset head), 58.7 tok/s.
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tt", "spec"))
import g3_attn1  # noqa: E402
import g3_hc  # noqa: E402
import g3_head  # noqa: E402
import mtp as M  # noqa: E402
from fused.grid import (NOC1_ROWS, Grid, profile_defines, profile_plumbing, profile_tensor, profile_words,  # noqa: E402
                        traced_us)
from g3_attn1 import NH, NKV, R_KVHEAD, R_QHEAD, T_IQ, Attn1, x_rows  # noqa: E402
from g3_attn2 import BATCH as A2_BATCH, DEPTH as A2_DEPTH, EXACT, HD, Attn2  # noqa: E402
from g3_hc import LO  # noqa: E402
from g3_head import Head  # noqa: E402
from g3_layer import N_ROUTER, WORKER_L1_M, mp_of, rel  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(DATA, "w43f_mtp")
XDATA = os.path.join(REPO, "third_party", "strata", "data")
D, HC, FF, NX, TOPK, VOCAB = 2560, 4, 640, 512, 10, 248320
M_MAX, S_MAX = 4, 4                   # rows a catch-up takes; steps a pass runs (the catch-up + 3 chained drafts)
NAPC, NBMAX = 55, 5                   # fmtp_attn_*.cpp: AP cores per KV head, blocks of 32 cells per core
WINDOW = NAPC * NBMAX * 32            # the dense attention's window: the last 8,800 cells (block-aligned)
CTX_DENSE = g3_attn1.CTX - 1          # the drafts' last cell: the K/V cache's (a window past WINDOW cells)
A2_EXACT = tuple(c for c in EXACT if c != 22)   # (cb 22: the bf16 mask tiles, a matmul input, at M rows)
IDLE = 0xFFFFFFFF
PROGRAMS = ("front", "attn1", "attn2", "exp", "head")   # one step's programs, in launch order
CLOCK_MHZ = 1350.0                    # the cores' clock (g4_phases.py's)
FRONT_PRE = 160                       # the front's W tiles read before its barrier (of 160: all; fewer were slower)
NREP = 8                              # copies of the step and words tensors (one per card memory bank)
LR_INJ = 324                          # a hyper-connection read's down columns and its 4 inject columns (g3_hc LO's 352)
L = "blk.0."
NAMES = {   # the target's names (g3_attn1/2, g3_head) -> the draft layer's (tt/spec/mtp.py load_dense)
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight", "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}
MIXER = {"output_hc_norm.weight": "hyper_connection_mixer.hc_norm.weight",
         "output_hc_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
         "output_hc_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight"}


class MtpGguf:
    """The draft layer's weights under the target's GGUF names for g3_attn1/2's and g3_head's builders (layer 0's
    prefix; output_hc_* the layer's final mixer; output.weight the head's rows). The indexer's projection
    (index_qk_proj: 512 q rows, then 128 k rows, assumed) only feeds program 1's pool scores, which dense attention
    does not read."""

    def __init__(self, dense: dict, head_rows=None):
        self.d, self.head_rows = dense, head_rows
        self.t = {"mtp-rt": SimpleNamespace(n_bytes=os.path.getsize(os.path.join(M.RT, "dense.bin")))}

    def __call__(self, name: str) -> torch.Tensor:
        if name == "output.weight":
            return torch.from_numpy(np.ascontiguousarray(self.head_rows))
        if name in MIXER:
            return torch.from_numpy(self.d[MIXER[name]])
        name = name[len(L):]
        if name.startswith("indexer.") and name.endswith("_proj.weight"):
            qk = self.d["self_attn.indexer.index_qk_proj.weight"]
            return torch.from_numpy(qk[:512] if name == "indexer.q_proj.weight" else qk[512:])
        w = torch.from_numpy(self.d[NAMES[name]])
        return w.reshape(-1) if name == "ffn_gate_inp_shexp.weight" else w


@contextmanager
def weight_cache(sub: str):
    """g3_hc.stream's cache pointed at CACHE/<sub> while the draft layer's programs are built (its keys carry the
    model's tag and the target's tensor names, so the draft layer's must live apart)."""
    old = g3_hc.CACHE
    g3_hc.CACHE = os.path.join(CACHE, sub)
    try:
        yield
    finally:
        g3_hc.CACHE = old


@contextmanager
def head_tiles(nt: int):
    """g3_head's LM-head tile count for a head of nt output tiles (shortcut: g3_head reads its module constant at
    build time; a head of its own size would need g3_head to take it as an argument)."""
    old = g3_head.NT
    g3_head.NT = nt
    try:
        yield
    finally:
        g3_head.NT = old


def cached(dev, name: str, build):
    """A card tensor from CACHE/<name>.tensorbin, or build() (a host ttnn tensor) dumped there first."""
    path = os.path.join(CACHE, name + ".tensorbin")
    if not os.path.exists(path):
        t = build()
        os.makedirs(CACHE, exist_ok=True)
        tmp = os.path.join(CACHE, f"{name}.{os.getpid()}.tmp.tensorbin")
        ttnn.dump_tensor(tmp, t)
        os.replace(tmp, path)
        return ttnn.to_device(t, dev)
    return ttnn.load_tensor(path, device=dev)


def head_vocab(head: str):
    """The head's token ids (int32), None for the whole vocabulary: fr / cyr = Strata's draft_vocab_*.bin, en = the
    first 40,525 ids both share (tt/flashnext/w4_draft.heads)."""
    if head == "full":
        return None
    from w4_draft import heads
    return heads()[head]


def head_rows(head: str):
    """The head's LM rows (the target's output.weight, f32 [rows, 2560]; a subset padded to a whole tile with copies
    of its first row: a tie goes to the lower index, that row) and its token ids (None: the whole vocabulary)."""
    vocab = head_vocab(head)
    rows = M.gguf_tensor("output.weight")
    if vocab is not None:
        rows = np.asarray(rows[np.concatenate([vocab, np.full(-len(vocab) % 32, vocab[0])])])
    return rows, vocab


def embedding_table():
    """The target's token_embd.weight as a bf16 row-major host tensor in 64-byte pages [VOCAB x 80, 32] (Q8_0 rounded
    to bf16; token v's k-tile t at page 80 v + t: the front's 80 cores read one token's row from every bank)."""
    old = os.path.join(CACHE, "token_embd_bf16.tensorbin")   # [VOCAB, 2560] form: the same bytes
    w = (ttnn.to_torch(ttnn.load_tensor(old)) if os.path.exists(old) else
         torch.from_numpy(np.asarray(M.gguf_tensor("token_embd.weight"))).to(torch.bfloat16))
    return ttnn.from_torch(w.reshape(1, 1, VOCAB * D // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def front_w(dense) -> torch.Tensor:
    """The front's W = [fc_hidden^T; fc_embedding^T] [5120, 2560]."""
    return torch.cat([torch.from_numpy(dense["fc_hidden.weight"]).T, torch.from_numpy(dense["fc_embedding.weight"]).T])


# ---------------------------------------------------------------- the experts in card form

def expert_codes(which):
    """experts.bin (tt/spec/mtp.py decode_experts' layout) -> Q2_0 codes (uint8 0..3) and scales (fp32, one per 64):
    gate|up [n, 1280, 2560] (rows 0..639 gate, 640..1279 up) and [n, 1280, 40], down [n, 2560, 640] and [n, 2560, 10]."""
    mm = np.memmap(os.path.join(M.RT, "experts.bin"), dtype=np.uint8, mode="r").reshape(NX, M.BLOB)
    x = np.asarray(mm[np.asarray(which)])
    n = len(x)
    a = 2 * FF * (D // 64) * 16
    b = a + D * (FF // 64) * 16
    c = b + 2 * FF * (D // 64) * 2

    def codes(raw, rows, cols):
        q = raw.reshape(n, rows, cols // 64, 16)
        return np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=4).reshape(n, rows, cols)

    gu = codes(x[:, :a], 2 * FF, D)
    gus = x[:, b:c].copy().view(np.float16).reshape(n, 2 * FF, D // 64).astype(np.float32)
    order = np.concatenate([np.arange(0, 2 * FF, 2), np.arange(1, 2 * FF, 2)])   # interleaved -> gate rows, up rows
    return (gu[:, order], gus[:, order], codes(x[:, a:b], D, FF),
            x[:, c:M.BLOB].copy().view(np.float16).reshape(n, D, FF // 64).astype(np.float32))


def expert_tiles(which):
    """The card form of experts `which` (fmtp_exp_reader.cpp): value tiles (code - 1, [k][n]) and scale tiles
    ([b][n], row b = k-tile b's scale), gate|up [n x 40][80] / [n x 40][3], down [n x 80][20] / [n x 80][1]."""
    gu, gus, dn, dns = expert_codes(which)
    n = len(gu)
    gv = (gu.astype(np.int8) - 1).reshape(n, 40, 32, 80, 32).transpose(0, 1, 3, 4, 2)
    dv = (dn.astype(np.int8) - 1).reshape(n, 80, 32, 20, 32).transpose(0, 1, 3, 4, 2)

    def scales(s, groups):        # [n, N, K / 64] -> [n, N / 32, groups, 32 (k-tile b), 32 (column)]
        kt = np.repeat(s, 2, axis=2)
        pad = np.zeros((n, s.shape[1], 32 * groups), np.float32)
        pad[:, :, :kt.shape[2]] = kt
        return pad.reshape(n, s.shape[1] // 32, 32, groups, 32).transpose(0, 1, 3, 4, 2)
    return gv, scales(gus, 3), dv, scales(dns, 1)


def card_experts(dev, chunk: int = 32):
    """All 512 experts on the card (cached in CACHE): gate|up values (bfp8_b), scales (fp32), down values, scales."""
    names = ("gu_vals", "gu_scales", "d_vals", "d_scales")
    paths = [os.path.join(CACHE, f"experts_{k}.tensorbin") for k in names]
    if not all(os.path.exists(p) for p in paths):
        parts = [[], [], [], []]
        for a in range(0, NX, chunk):
            for i, (p, t) in enumerate(zip(parts, expert_tiles(range(a, a + chunk)))):
                p.append(torch.from_numpy(np.ascontiguousarray(t)).reshape(-1, 32).to(
                    torch.bfloat16 if i in (0, 2) else torch.float32))
        os.makedirs(CACHE, exist_ok=True)
        for i, (p, path) in enumerate(zip(parts, paths)):
            t = ttnn.from_torch(torch.cat(p).reshape(1, 1, -1, 32), dtype=ttnn.bfloat8_b if i in (0, 2) else
                                ttnn.float32, layout=ttnn.TILE_LAYOUT)
            tmp = path.replace(".tensorbin", f".{os.getpid()}.tmp.tensorbin")
            ttnn.dump_tensor(tmp, t)
            os.replace(tmp, path)
            parts[i] = None
    return [ttnn.load_tensor(p, device=dev) for p in paths]


def experts_exact(which=(0, 1, 255, 511)) -> dict:
    """bfp8_b holds the value tiles exactly: experts `which` through ttnn's bfp8_b packing and back, against the codes;
    and the card form's arithmetic (values x scales) against tt/spec/mtp.py's decode (gguf-py's Q2_0)."""
    gv, gs, dv, ds = expert_tiles(which)
    out = {}
    for name, v in (("gate_up", gv), ("down", dv)):
        t = torch.from_numpy(np.ascontiguousarray(v)).reshape(1, 1, -1, 32).to(torch.bfloat16)
        back = ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        out[f"{name}_bfp8_mismatches"] = int((back != t.float()).sum())
    g, u, d = M.decode_experts(which=list(which))
    n = len(which)
    w_gu = (gv.astype(np.float32) * np.repeat(gs.reshape(n, 40, 3 * 32, 32)[:, :, :80], 32, axis=2).reshape(
        n, 40, 80, 32, 32)).transpose(0, 1, 4, 2, 3).reshape(n, 2 * FF, D)
    w_d = (dv.astype(np.float32) * np.repeat(ds.reshape(n, 80, 32, 32)[:, :, :20], 32, axis=2).reshape(
        n, 80, 20, 32, 32)).transpose(0, 1, 4, 2, 3).reshape(n, D, FF)
    out["values_x_scales_vs_q2_0_decode_max_abs"] = float(max(np.abs(w_gu[:, :FF] - g).max(),
                                                              np.abs(w_gu[:, FF:] - u).max(), np.abs(w_d - d).max()))
    out["experts"] = list(map(int, which))
    return out


# ---------------------------------------------------------------- the layer on the card

def u32(dev, n: int, rows: int = 1):
    return ttnn.from_torch(torch.zeros(1, 1, rows, n, dtype=torch.int32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def host_of(t: ttnn.Tensor):
    """A row-major host tensor of card tensor t's shape and dtype, and a writable numpy view of its buffer (float32 or
    int32: uint32 bits)."""
    h = ttnn.from_torch(torch.zeros(tuple(t.shape), dtype=torch.int32 if t.dtype == ttnn.uint32 else torch.float32),
                        dtype=t.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    v = torch.from_dlpack(h.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0)))
    return h, v.view(torch.int32 if t.dtype == ttnn.uint32 else torch.float32).numpy().reshape(-1)


def put(t: ttnn.Tensor, a: np.ndarray) -> None:
    """Host array a into the card tensor t in place (its dtype and shape)."""
    dt = {ttnn.uint32: torch.int32, ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}[t.dtype]
    h = torch.from_numpy(np.ascontiguousarray(a).view(np.int32) if dt == torch.int32 else
                         np.ascontiguousarray(a, np.float32)).to(dt).reshape(tuple(t.shape))
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=t.dtype, layout=t.layout), t)


def nbytes(t: ttnn.Tensor) -> int:
    n = int(np.prod(tuple(t.shape)))
    return n * {ttnn.bfloat8_b: 1088 / 1024, ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint32: 4}[t.dtype]


class MtpAttn1(Attn1):
    """g3_attn1's M-row program without the indexer (its projection tiles, pool scores and pooled keys feed only the
    selection, which dense attention does not take): phase N's roles are the 24 q heads and the 2 KV heads, the
    projection stops at the indexer's tiles (their PA rows, and the raw key KV head 0 writes from them, are stale)."""

    def roles(self, k):
        if k < NH:
            return R_QHEAD, k
        if k < NH + NKV:
            return R_KVHEAD, k - NH
        return IDLE, 0

    def program(self):
        old = g3_attn1.NT   # shortcut: program() splits the module's tile count; an argument would need g3_attn1 changed
        g3_attn1.NT = T_IQ
        try:
            return super().program()
        finally:
            g3_attn1.NT = old


class MtpAttn2(Attn2):
    """g3_attn2's M-row program with dense attention over every cell (tt/kernels/fmtp_attn_*.cpp): no selection
    tensor; AP on NAPC cores per KV head (the whole grid), up to NBMAX blocks of 32 cells each; AC in batches of O
    tiles. The rest as g3_attn2's."""

    def program(self):
        g, mm = self.grid, self.m_max
        A = lambda t: t.buffer_address()                                                    # noqa: E731
        order = g.order(self.noc1_rows)
        gp, gdef, gbytes = profile_plumbing(self.gprof, self.gprof_n, 63, 64)
        rows = [("M_MAX", str(mm))]
        t_max = (12 * mm + 31) // 32
        o, a = {}, 0    # the arena: g3_attn2's M-row regions without the selection's
        for name, size in (("x32", mm * D * 4), ("x16", mm * D * 2), ("p1", mm * (HC * LO * 4 + 256)),
                           ("rs", mm * HC * 128), ("wa", mm * 64), ("o", mm * NH * HD * 2), ("hnf", mm * D * 2),
                           ("lof", mm * 704), ("wf", 0), ("l", mm * (N_ROUTER * 128 + FF * 2)), ("sh", 0)):
            o[name] = a
            a += size
        pack = lambda k: g.noc[order[k]].x | g.noc[order[k]].y << 16                # noqa: E731
        h1_cores = [pack(i) for i in range((LO // 32) * HC)]
        rargs, crt, wargs = {}, ttnn.RuntimeArgs(), {}
        for idx, (cx, cy) in enumerate(g.cores):
            k = order.index(idx)
            ap = k if k < NKV * NAPC else IDLE
            ac = k                                     # (the combine's roles: fmtp_attn_compute.cpp)
            o_tile = s_tile = k if k < D // 32 else IDLE
            h_item = k if k < (LO // 32) * HC else IDLE
            h_tile = k if k <= D // 32 else IDLE
            r_item = k if k < N_ROUTER + FF // 32 else IDLE
            rargs[(cx, cy)] = [A(self.x32_t), A(self.x16_t), A(self.q_t), A(self.g_t), A(self.wa_t), A(self.step_t)] + \
                g.args(idx) + [o[r] for r in ("x32", "x16", "p1", "rs", "wa", "o", "hnf", "lof", "l")] + [
                ap, A(self.kc), A(self.vc), ac, A(self.parts), A(self.wo), o_tile, A(self.wd), A(self.wu),
                A(self.hnorm_t), h_item, h_tile, A(self.wr), A(self.wgu), A(self.wsd), A(self.routed_t), r_item, s_tile,
                A(self.prof)] + h1_cores + gp(idx)
            crt[cx][cy] = [ap, ac, o_tile, h_item, h_tile, r_item, s_tile]
            wargs[(cx, cy)] = g.args(idx) + [o[r] for r in ("x32", "x16", "p1", "rs", "o", "hnf", "lof", "wf", "l",
                                                             "sh")] + [
                A(self.parts)] + [A(self.chk[c]) for c in ("o", "x1", "p1", "hn", "w", "l", "h")] + [
                A(self.x32_t) if self.inplace else A(self.chk["x2"])] + [
                ap, ac, o_tile, h_item, h_tile, r_item, s_tile, A(self.x16_t) if self.inplace else 0] + h1_cores + [
                A(self.step_t)] + gp(idx)
        acc = list(ttnn.TensorAccessorArgs(self.wo).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in A2_EXACT:
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fmtp_attn_compute.cpp"), core_ranges=g.ranges,
                                         compile_time_args=[A2_BATCH], runtime_args=crt, config=cfg, defines=rows)]
        kernels += g.dm_kernels(os.path.join(KDIR, "fmtp_attn_reader.cpp"), os.path.join(KDIR, "fmtp_attn_writer.cpp"),
                                [A2_BATCH, A2_DEPTH] + acc, acc, rargs, wargs,
                                # the K/V caches' pools a KV head: A1's, sized by TT_CTX (the reader's 4,096 was 16K's)
                                gdef + rows + [("NPOOL", str(g3_attn1.NPOOL))], self.noc1_rows)
        f32, bf16, u8 = ttnn.float32, ttnn.bfloat16, ttnn.uint8
        T4, T2, t, ns = 4096, 2048, t_max, (NAPC + 31) // 32
        nb = NBMAX
        spec = [(0, ttnn.bfloat8_b, 1088, A2_BATCH * A2_DEPTH), (1, bf16, T2, 80), (2, bf16, T2, 3),
                (3, f32, T4, 6), (4, f32, T4, 3), (5, f32, T4, 1), (6, f32, T4, 1), (7, f32, T4, 2), (8, f32, T4, 1),
                (9, f32, T4, 4), (10, bf16, T2, 10), (11, f32, T4, 4), (12, f32, T4, 4), (13, f32, T4, 4),
                (14, bf16, T2, 1), (15, f32, T4, 1),
                (16, f32, T4, 1), (17, bf16, T2, 8), (18, bf16, T2, 8), (19, bf16, T2, 8 * nb), (20, bf16, T2, 8 * nb),
                (21, bf16, T2, 9 * t), (48, bf16, T2, nb * t), (58, bf16, T2, 9 * t), (59, f32, T4, nb * t),
                (22, bf16, T2, 4), (23, f32, T4, nb * t), (24, f32, T4, t), (25, bf16, T2, nb * t), (26, f32, T4, t),
                (27, f32, T4, 8), (28, f32, T4, ns), (29, f32, T4, 8), (30, f32, T4, ns), (31, f32, T4, 1),
                (32, f32, T4, 1), (33, f32, T4, 1), (34, f32, T4, 8), (35, f32, T4, ns), (36, f32, T4, 2 * ns),
                (37, f32, T4, 1), (38, bf16, T2, 1), (39, f32, T4, 3),
                (40, bf16, T2, 2 * A2_BATCH), (41, f32, T4, 1), (42, f32, T4, 4), (43, f32, T4, 4), (44, f32, T4, 4),
                (45, bf16, T2, 4), (46, bf16, T2, 2 * A2_BATCH), (47, f32, T4, 1), (49, bf16, T2, 1), (50, f32, T4, 1),
                (51, f32, T4, 1), (52, f32, T4, 1), (53, f32, T4, 4), (54, f32, T4, 4), (55, f32, T4, 1),
                (56, f32, T4, 4), (57, bf16, T2, 4),
                (60, u8, T4, 1), (61, u8, a, 1), (62, u8, 3072, 1), (63, u8, 64 + gbytes, 1)]
        self.l1_bytes = sum(size * n for _, _, size, n in spec)
        assert self.l1_bytes <= WORKER_L1_M, (self.l1_bytes, WORKER_L1_M)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        ins = [self.x32_t, self.x16_t, self.q_t, self.g_t, self.wa_t, self.step_t, self.kc, self.vc, self.parts,
               self.wo, self.wd, self.wu, self.hnorm_t, self.wr, self.wgu, self.wsd, self.routed_t, self.prof] + \
            list(self.chk.values()) + ([] if self.gprof is None else [self.gprof])
        return lambda: ttnn.generic_op(ins, prog)


class MtpCard:
    """The draft layer on the card and its programs for passes of up to S_MAX steps. heads: the draft heads to build
    (subset names or "full"); use_head() picks the one a pass runs."""

    def __init__(self, dev, heads=("fr",), log=print, back=False, prof_n=0):
        """prof_n > 0: every program with grid.h's per-core clocks (prof_n barriers kept) in self.gprof[kind]."""
        self.dev, self.grid, self.log = dev, Grid(dev), log
        self.prof_n, self.front_pre = prof_n, FRONT_PRE
        self.gprof = {k: profile_tensor(dev, prof_n) for k in PROGRAMS} if prof_n else {}
        t0 = time.time()
        dense = M.load_dense()
        rng = np.random.default_rng(0)
        g = MtpGguf(dense)
        with weight_cache("layer"):
            A1 = MtpAttn1(dev, g, rng, pos=0, layer=0, back=back, m_max=M_MAX, zero_caches=True)
            A2 = MtpAttn2(dev, g, rng, pos=0, layer=0, back=back, m_max=M_MAX, zero_caches=True)
            # W = [fc_hidden^T; fc_embedding^T] [5120, 2560], bfp8_b n-major (page n * 160 + k)
            self.w_front, _ = g3_hc.stream(dev, lambda: front_w(dense), False, "mtp/front_w")
        z = lambda shape, dt, lay: ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=lay, device=dev)  # noqa
        A1.rawk = z(tuple(A1.rawk.shape), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        A1.pooled = z(tuple(A1.pooled.shape), ttnn.float32, ttnn.TILE_LAYOUT)
        A2.x32_t, A2.x16_t, A2.kc, A2.vc = A1.x32_t, A1.x16_t, A1.kc, A1.vc
        A2.q_t, A2.g_t, A2.wa_t = A1.q, A1.g_out, A1.chk["w"]
        A2.routed_t = z(tuple(A2.routed_t.shape), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)   # the experts come after
        # the partials (55 a KV head, 10 T_MAX tiles each) and the two-stage combine's sub-partials (3 tiles each)
        A2.parts = z((1, 1, 32 * (2 * NAPC * 20 + 2 * 2 * 8 * 6 * 3), 32), ttnn.float32, ttnn.TILE_LAYOUT)
        A2.inplace = True
        if prof_n:
            A1.gprof, A2.gprof, A1.gprof_n, A2.gprof_n = self.gprof["attn1"], self.gprof["attn2"], prof_n, prof_n
        self.A1, self.A2 = A1, A2
        # w_h / w_e tiles: tile t (< 80) row 4 s + m = w_h[s * 2560 + 32 t ..], tile 80 + t rows = w_e[32 t ..]
        wt = torch.zeros(160, 32, 32)
        wh = torch.from_numpy(dense["pre_fc_norm_hidden.weight"]).reshape(HC, D // 32, 32)
        we = torch.from_numpy(dense["pre_fc_norm_embedding.weight"]).reshape(D // 32, 32)
        for s in range(HC):
            wt[:80, 4 * s:4 * s + 4] = wh[s][:, None, :]
        wt[80:, :16] = we[:, None, :]
        self.whe = ttnn.from_torch(wt.reshape(1, 1, -1, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        self.emb = cached(dev, "token_embd_bf16_p64", embedding_table)
        self.experts = card_experts(dev)
        # per pass: the catch-up's rows (128-byte pages: row m stream s k-tile t at page (4 m + s) 80 + t) and tokens;
        # per step: its step tensor (step and words in NREP copies, the front's cores reading theirs);
        # R (the head's x, one page) and its copy in 128-byte pages (the next step's front reads that)
        self.rin = z((1, 1, M_MAX * HC * D // 32, 32), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        self.words = u32(dev, 16, NREP)
        self.steps = [u32(dev, 16, NREP) for _ in range(S_MAX)]
        self.r32 = z((1, 1, 1, HC * D), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        self.r16 = z((1, 1, 1, HC * D), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.rp = z((1, 1, HC * D // 32, 32), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        # host tensors made once, written in place (set_pass): the pass's inputs without a ttnn.from_torch per write
        self.host = {id(t): host_of(t) for t in [self.rin, self.words] + self.steps}
        self.p_layer = []
        for k in range(S_MAX):
            A1.step, A2.step_t = self.steps[k], self.steps[k]
            self.p_layer.append((A1.program(), A2.program(), self.exp_program(self.steps[k])))
        self.p_catchup = self.front_program(self.steps[0], self.rin, 0, self.words, None)
        self.heads = {}
        for h in heads:
            self.add_head(h, dense)
        log(f"draft layer built in {time.time() - t0:.0f} s")

    # ---- heads
    def add_head(self, name: str, dense: dict) -> None:
        rows, vocab = head_rows(name)
        nt = len(rows) // 32
        with weight_cache(f"head_{name}"), head_tiles(nt):
            H = Head(self.dev, MtpGguf(dense, rows), back=False, m_max=1)
            H.x32_t, H.x16_t = self.r32, self.r16
            if self.prof_n:
                H.gprof, H.gprof_n = self.gprof["head"], self.prof_n
            progs = {"ids": H.program(m=1, logits=False, ids=True), "both": H.program(m=1, logits=True, ids=True)}
        vmap = None
        if vocab is not None:
            v = np.zeros(-(-len(vocab) // 16) * 16, np.int32)
            v[:len(vocab)] = vocab
            vmap = ttnn.from_torch(torch.from_numpy(v).reshape(1, 1, -1, 16), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
        fronts = [None] + [self.front_program(self.steps[k], self.rp, 1, H.ids_t, vmap) for k in range(1, S_MAX)]
        self.heads[name] = SimpleNamespace(H=H, vocab=vocab, vmap=vmap, progs=progs, fronts=fronts, nt=nt,
                                           rows=VOCAB if vocab is None else len(vocab))
        self.use_head(name)

    def use_head(self, name: str) -> None:
        self.hd = self.heads[name]

    # ---- programs
    def front_program(self, step, rin, mode: int, src, vmap):
        g, A = self.grid, (lambda t: t.buffer_address() if t is not None else 0)   # noqa: E731
        order = g.order(NOC1_ROWS)
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            k = order.index(idx)
            t = n = k if k < 80 else IDLE
            rargs[(cx, cy)] = g.args(idx) + [A(step), A(rin), mode, A(src), A(vmap), A(self.emb), A(self.w_front),
                                             A(self.whe), t, n, self.front_pre, idx % NREP] + self._gp("front", idx)
            wargs[(cx, cy)] = g.args(idx) + [A(step), A(self.A1.x32_t), A(self.A1.x16_t), t, n] + self._gp("front", idx)
            crt[cx][cy] = [t, n]
        acc = list(ttnn.TensorAccessorArgs(self.w_front).get_compile_time_args())
        return self._program("fmtp_front", "front", acc, [M_MAX] + acc, rargs, wargs, crt, exact=(3, 5, 10, 14), cbs=[
            (0, ttnn.bfloat8_b, 1088, 160), (1, ttnn.bfloat16, 2048, 86), (2, ttnn.bfloat16, 2048, 1),
            (3, ttnn.float32, 4096, 1), (4, ttnn.bfloat16, 2048, 1), (5, ttnn.float32, 4096, 2),
            (6, ttnn.bfloat16, 2048, 1), (7, ttnn.bfloat16, 2048, 1), (8, ttnn.float32, 4096, 2),
            (9, ttnn.float32, 4096, 2), (10, ttnn.float32, 4096, 1), (11, ttnn.bfloat16, 2048, 1),
            (12, ttnn.bfloat16, 2048, 2), (13, ttnn.float32, 4096, 1), (14, ttnn.float32, 4096, 2),
            (15, ttnn.float32, 4096, 1), (16, ttnn.bfloat16, 2048, 1), (60, ttnn.uint8, 4096, 1),
            (62, ttnn.uint8, 8192, 1)], ins=[step, rin, src, self.emb, self.w_front, self.whe, self.A1.x32_t,
                                             self.A1.x16_t] + ([vmap] if vmap is not None else []))

    def exp_program(self, step):
        g, A, A2 = self.grid, (lambda t: t.buffer_address()), self.A2   # noqa: E731
        order = g.order(NOC1_ROWS)
        C = len(g.cores)
        ex = self.experts
        # E1's 200 units: one on the coordinator (P0 is its) and on the cores with the least card memory bandwidth,
        # two on the rest
        speed = g.speeds(NOC1_ROWS)
        ones = set([0] + sorted(range(1, C), key=lambda i: speed[i])[:2 * C - 20 * TOPK - 1])
        first = np.cumsum([0] + [1 if i in ones else 2 for i in range(C)])
        assert first[-1] == 20 * TOPK
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            k = order.index(idx)
            col = k if k < D // 32 else IDLE
            v0, v1 = int(first[idx]), int(first[idx + 1])
            rargs[(cx, cy)] = g.args(idx) + [A(step), A(A2.chk["l"]), A(A2.chk["hn"]), A(A2.chk["w"]),
                                             A(A2.x32_t)] + [A(t) for t in ex] + [v0, v1, col, idx % NREP] + \
                self._gp("exp", idx)
            wargs[(cx, cy)] = g.args(idx) + [A(self.r32), A(self.r16), A(self.rp), v0, v1, col] + self._gp("exp", idx)
            crt[cx][cy] = [v0, v1, col, 1 if idx == 0 else 0]
        acc = list(ttnn.TensorAccessorArgs(ex[0]).get_compile_time_args())
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        return self._program("fmtp_exp", "exp", [M_MAX] + acc, acc, rargs, wargs, crt, exact=(3, 4, 5, 6, 10, 11, 13, 16),
                             cbs=[
            (0, ttnn.bfloat8_b, 1088, 96), (1, bf16, 2048, 80), (2, bf16, 2048, 1), (3, f32, 4096, 8),
            (4, f32, 4096, TOPK), (5, f32, 4096, TOPK), (6, f32, 4096, 1), (7, f32, 4096, 2), (8, bf16, 2048, 2),
            (9, bf16, 2048, 40), (10, f32, 4096, 1), (11, f32, 4096, 4), (12, f32, 4096, 1), (13, f32, 4096, 4),
            (14, f32, 4096, 4), (15, bf16, 2048, 4), (16, f32, 4096, 3), (17, bf16, 2048, 3),
            (60, ttnn.uint8, 4096, 1), (61, ttnn.uint8, 8192 + TOPK * FF * 2, 1), (62, ttnn.uint8, 4096, 1)],
            ins=[step, A2.chk["l"], A2.chk["hn"], A2.chk["w"], A2.x32_t] + list(ex) + [self.r32, self.r16, self.rp])

    def _gp(self, kind: str, idx: int) -> list:
        """A program's profile args (grid.h's: the tensor, this core's slot), none without the profile."""
        return [self.gprof[kind].buffer_address(), idx] if self.prof_n else []

    def _program(self, name, kind, reader_ct, writer_ct, rargs, wargs, crt, exact, cbs, ins):
        g = self.grid
        defines = profile_defines(63, self.prof_n) if self.prof_n else []
        if self.prof_n:
            cbs, ins = cbs + [(63, ttnn.uint8, 16 * self.prof_n, 1)], ins + [self.gprof[kind]]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in exact:
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, f"{name}_compute.cpp"), core_ranges=g.ranges,
                                         compile_time_args=[], runtime_args=crt, config=cfg)]
        kernels += g.dm_kernels(os.path.join(KDIR, f"{name}_reader.cpp"), os.path.join(KDIR, f"{name}_writer.cpp"),
                                reader_ct, writer_ct, rargs, wargs, defines, NOC1_ROWS)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*c) for c in cbs])
        return lambda: ttnn.generic_op(ins, prog)

    # ---- passes
    def set_pass(self, c0: int, R_rows: np.ndarray, toks, n_steps: int) -> None:
        """A pass of n_steps steps: step 0 the catch-up, M = len(toks) rows at the cells c0 .. c0 + M - 1 (R_rows
        [M, 10240] the target's final residual streams there, toks the tokens at c0 + 1 ..; None: already in rin on the
        card, engine.SpecModel.draft's page copy); steps k >= 1 at the cells c0 + M - 1 + k. In-queue writes of the
        pass's inputs and its steps' step tensors."""
        Mr = len(toks)
        assert 1 <= Mr <= M_MAX and 1 <= n_steps <= S_MAX and c0 + Mr - 1 + n_steps <= CTX_DENSE
        rin = self.host[id(self.rin)][1].reshape(M_MAX, HC * D)
        words = self.host[id(self.words)][1].reshape(NREP, 16)
        todo = [self.words]
        if R_rows is not None:
            rin[:Mr] = R_rows                     # (rows Mr.. are never read: the front repeats row Mr - 1)
            todo.append(self.rin)
        words[:] = 0
        words[:, :Mr] = toks
        for k in range(n_steps):
            pos, m = (c0, Mr) if k == 0 else (c0 + Mr - 1 + k, 1)
            st = self.host[id(self.steps[k])][1].reshape(NREP, 16)
            st[:, :5] = (pos, m, 1, 0, mp_of(m))  # g3_attn1.step_words' words
            todo.append(self.steps[k])
        for t in todo:
            ttnn.copy_host_to_device_tensor(self.host[id(t)][0], t)

    def launch(self, k: int, kv_only: bool = False, logits: bool = False) -> None:
        """Step k's programs (kv_only: the catch-up's front and program 1, the K/V of its rows and nothing else)."""
        p1, p2, pe = self.p_layer[k]
        (self.p_catchup if k == 0 else self.hd.fronts[k])()
        p1()
        if kv_only:
            return
        p2()
        pe()
        self.hd.progs["both" if logits else "ids"]()

    def run(self, n_steps: int):
        """A pass already set: its steps' programs in one go (the traced path)."""
        for k in range(n_steps):
            self.launch(k)

    def draft(self) -> tuple[int, float]:
        """The last step's draft (a token id) and its logit."""
        ids, vals = self.hd.H.get_ids(1)
        j = int(ids[0, 0])
        return (j if self.hd.vocab is None else int(self.hd.vocab[j])), float(vals[0, 0])

    def logits(self) -> np.ndarray:
        """The last step's logits over the head's rows (a launch with logits)."""
        return ttnn.to_torch(self.hd.H.logits).flatten()[:self.hd.rows].double().numpy()

    def kv(self, cells) -> tuple[np.ndarray, np.ndarray]:
        """The layer's K and V cache rows at `cells`: [n, 2, 256] each (K normed and rotated, bf16 on the card)."""
        f = lambda t: ttnn.to_torch(t).double().reshape(2, -1, 256)[:, np.asarray(cells)].permute(1, 0, 2).numpy()  # noqa
        return f(self.A1.kc), f(self.A1.vc)

    def zero_kv(self) -> None:
        for t in (self.A1.kc, self.A1.vc):
            put(t, np.zeros(tuple(t.shape), np.float32))

    def memory(self) -> dict:
        """Card memory the draft layer holds (MiB), by part."""
        A1, A2 = self.A1, self.A2
        mib = lambda ts: round(sum(nbytes(t) for t in ts if t is not None) / 2**20, 1)   # noqa: E731
        out = {"attention_weights": mib([A1.wd, A1.wu, A1.wp, A2.wd, A2.wu, A2.wo, A2.wr, A2.wgu, A2.wsd]),
               "front_weights": mib([self.w_front, self.whe]),
               "kv_and_index_caches": mib([A1.kc, A1.vc, A1.rawk, A1.pooled]),
               "experts": mib(self.experts), "embedding_table": mib([self.emb])}
        for n, h in self.heads.items():
            out[f"head_{n}"] = mib([h.H.wl_t, h.H.wd_t, h.H.wu_t, h.vmap])
        out["total"] = round(sum(out.values()), 1)
        return out


# ---------------------------------------------------------------- the card's precision on the host

def dump_card_weights(dev, head: str) -> None:
    """The draft layer's weights as the card holds them (bfp8_b values read back, float32, the target's tensor names
    mapped back to the draft layer's; the hyper-connection reads' with their norms folded in, as the card computes
    them) into CACHE/card_weights_<head>.npz, for CardMtp."""
    dense = M.load_dense()
    tag = g3_hc.model_tag(MtpGguf(dense))
    rb = lambda key: g3_hc.stream(dev, None, True, key)[1].float().numpy()   # noqa: E731  (a cache hit: read back)
    out = {}
    with weight_cache("layer"):
        for pre, site in (("attn_hyper_connection.", "hc_attn"), ("mlp_hyper_connection.", "hc_ffn")):
            out[pre + "wd"] = rb(f"{tag}/{L}{site}_wd")[:, :LR_INJ]
            out[pre + "up"] = rb(f"{tag}/{L}{site}_wu")
        wp = rb(f"{tag}/{L}wp")
        out["self_attn.q_proj.weight"], out["self_attn.k_proj.weight"] = wp[:, :12288].T, wp[:, 12288:12800].T
        out["self_attn.v_proj.weight"] = wp[:, 12800:13312].T
        out["self_attn.o_proj.weight"] = rb(f"{tag}/{L}wo").T
        gu = rb(f"{tag}/{L}wgu").reshape(D, FF // 32, 2, 32)
        out["mlp.shared_expert.gate_proj.weight"] = gu[:, :, 0].reshape(D, FF).T
        out["mlp.shared_expert.up_proj.weight"] = gu[:, :, 1].reshape(D, FF).T
        out["mlp.shared_expert.down_proj.weight"] = rb(f"{tag}/{L}wsd").T
        out["front"] = rb("mtp/front_w")
    with weight_cache(f"head_{head}"):
        out["hyper_connection_mixer.wd"] = rb(f"{tag}/output_hc_wd")[:, :LR_INJ]
        out["hyper_connection_mixer.up"] = rb(f"{tag}/output_hc_wu")
        out["head"] = rb(f"{tag}/output").T
    np.savez(os.path.join(CACHE, f"card_weights_{head}.npz"), **{k: np.ascontiguousarray(v, np.float32)
                                                                  for k, v in out.items()})


class CardMtp(M.Mtp):
    """tt/spec/mtp.py's layer at the card's precision: the card's weights (dump_card_weights), GEMV inputs in bf16
    with fp32 sums (Mtp's bf16 mode; bfp8_b values are bf16-exact), the experts exact (fp32), the embedding in bf16,
    and the card's rounding points in the front and the hyper-connection reads (bf16 of the raw streams, the norms
    folded into the weights, the RMS scalars applied after the GEMV: g3_attn1's errors() reference)."""

    def __init__(self, head: str, threads: int = 8):
        z = np.load(os.path.join(CACHE, f"card_weights_{head}.npz"))
        super().__init__("bf16", experts="lazy", vocab=head_vocab(head), threads=threads)
        for k in z.files:
            if k.endswith(".weight"):
                self.w[k] = self._t(z[k])
        self.head = self._t(z["head"][:len(self.head)])
        self.hcw = {p: (torch.from_numpy(z[p + "wd"]), torch.from_numpy(z[p + "up"]))
                    for p in ("attn_hyper_connection.", "mlp_hyper_connection.", "hyper_connection_mixer.")}
        self.front_w = torch.from_numpy(z["front"])

    def expert(self, e: int):
        if e not in self._exp:
            self._exp[e] = tuple(torch.from_numpy(x[0].astype(np.float32)) for x in M.decode_experts(which=[e]))
        return self._exp[e]

    def hc_read(self, R, pre: str, inject: bool, apply=None):
        if apply is not None:
            bo, inj0 = apply
            R = R + bo[:, None, :] * (2 * torch.sigmoid(inj0 / HC))[:, :, None]
        wd, up = self.hcw[pre]
        b16 = lambda x: x.to(torch.bfloat16).float()   # noqa: E731
        r = torch.rsqrt(R.pow(2).mean(-1, keepdim=True) + M.EPS)
        lo = sum(r[:, s] * (b16(R[:, s]) @ wd[s * D:(s + 1) * D]) for s in range(HC))
        gate = torch.sigmoid(b16(torch.nn.functional.silu(lo[:, :M.LR] / HC)) @ up).view(-1, HC, D)
        hn = (gate * R * self.wn[pre + "hc_norm.weight"].view(HC, D) * r).mean(1)
        return R, hn, (lo[:, M.LR:M.LR + HC] if inject else None)

    def front(self, R_in, tok, pos, hnorm_stream: bool = False):
        b16 = lambda x: x.to(torch.bfloat16).float()   # noqa: E731
        B = R_in.shape[0]
        e = b16(torch.from_numpy(np.asarray(self.embd[tok.numpy()], np.float32)))
        Rv = R_in.float().reshape(B, HC * D)
        r = torch.rsqrt(Rv.pow(2).mean(-1) + M.EPS)
        q = torch.rsqrt(e.pow(2).mean(-1) + M.EPS)
        h = b16(Rv * self.wn["pre_fc_norm_hidden.weight"]).view(B, HC, D) @ self.front_w[:D]
        R = r[:, None, None] * h + (q[:, None] * (b16(e * self.wn["pre_fc_norm_embedding.weight"]) @
                                                  self.front_w[D:]))[:, None, :]
        R, mixed, inj = self.hc_read(R, "attn_hyper_connection.", True)
        k = self.mm(mixed, self.w["self_attn.k_proj.weight"]).view(B, M.NKV, M.HD)
        v = self.mm(mixed, self.w["self_attn.v_proj.weight"]).view(B, M.NKV, M.HD)
        k = self.rope(self.rms(k, self.wn["self_attn.k_norm.weight"]), pos)
        return R, mixed, inj, k, v


# ---------------------------------------------------------------- checks

PROMPTS = ("A.chat.18", "A.code.0", "A.math.12", "A.zh.40")   # precision positions (w4_draft.check)


def reference(seqs, head: str, emu: bool = False):
    """float64 (tt/spec/mtp.py) chains of S_MAX steps at positions (the first 24 decode cells of each prompt),
    step 0's logits, the catch-up K/V over each prompt's cells before them, and step 0 at every such cell (the drafts
    of the prefix's catch-up passes). Cached in CACHE/ref_<head>.npz (G6_STAGE=ref computes it without the card)."""
    import w4_draft as W
    path = os.path.join(CACHE, f"ref_{'emu_' if emu else ''}{head}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {name: {k: z[f"{name}/{k}"] for k in ("q", "d", "p", "lg", "K", "V", "d_pre")} for name, *_ in seqs}
    th = int(os.environ.get("G6_THREADS", 8))
    m = CardMtp(head, th) if emu else M.Mtp("float64", experts="lazy", vocab=head_vocab(head), threads=th)
    out = {}
    for name, meta, ids, R in seqs:
        q0 = meta["n_prompt"] - 1
        q = np.arange(q0, q0 + 24)
        d, p, lg = W.chain(m, R, ids, 0, q, steps=S_MAX)
        Rt = torch.from_numpy(np.ascontiguousarray(R[:q0])).to(m.dt)
        cache, (_, lg_pre) = M.teacher_forced(m, Rt, ids[1:q0 + 1], 0, rest_rows=np.arange(q0))
        d_pre, _ = m.pick(lg_pre)
        out[name] = dict(q=q, d=d, p=p, lg=lg.double().numpy(), K=cache.K.double().numpy(), V=cache.V.double().numpy(),
                         d_pre=d_pre.numpy())
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **{f"{n}/{k}": v for n, r in out.items() for k, v in r.items()})
    return out


def card_run(C: MtpCard, seqs, chunks=(1, 2, 3, 4), logits=True):
    """The card's drafts at the reference's positions: per prompt, the catch-up over the cells before them in passes
    of M = 1, 2, 3, 4, ... rows (each pass's draft from its last row), then at each position a pass of one catch-up
    row and S_MAX - 1 chained steps (step 0's logits read)."""
    out = {}
    for name, meta, ids, R in seqs:
        C.zero_kv()
        q0 = meta["n_prompt"] - 1
        c, i, pre = 0, 0, {}
        while c < q0:
            Mr = min(chunks[i % len(chunks)], q0 - c)
            C.set_pass(c, np.asarray(R[c:c + Mr]), ids[c + 1:c + 1 + Mr], 1)
            C.launch(0)
            ttnn.synchronize_device(C.dev)
            pre[c + Mr - 1] = C.draft()[0]
            c, i = c + Mr, i + 1
        K, V = C.kv(range(q0))
        drafts, lgs = [], []
        for q in range(q0, q0 + 24):
            C.set_pass(q, np.asarray(R[q:q + 1]), ids[q + 1:q + 2], S_MAX)
            row = []
            for k in range(S_MAX):
                C.launch(k, logits=logits and k == 0)
                ttnn.synchronize_device(C.dev)
                row.append(C.draft()[0])
                if logits and k == 0:
                    lgs.append(C.logits())
            drafts.append(row)
        out[name] = dict(d=np.array(drafts), lg=np.array(lgs) if lgs else None, K=K, V=V, pre=pre)
    return out


def accept(drafts: dict, seqs) -> list:
    """The drafts' acceptance against the target's greedy tokens (the census continuations): P(d_0 .. d_j all equal
    the tokens at q + 2 .. q + 2 + j) by depth j, over the positions q (w4_draft's float64_cumulative_accept_by_depth)."""
    hit, n = np.zeros(S_MAX), 0
    for name, meta, ids, _ in seqs:
        q0 = meta["n_prompt"] - 1
        for i, row in enumerate(drafts[name]):
            hit += np.cumprod(np.asarray(row) == ids[q0 + i + 2:q0 + i + 2 + S_MAX])
            n += 1
    return (hit / n).round(4).tolist()


def compare(ref: dict, card: dict) -> dict:
    """Drafts equal by depth, step 0's logits (max rel and rel rms, top-1 agreement), the catch-up's K/V and drafts."""
    eq, n, lrel, lrms, krel, vrel, pre_eq, pre_n = np.zeros(S_MAX), 0, 0.0, [], 0.0, 0.0, 0, 0
    per = {}
    for name in ref:
        r, c = ref[name], card[name]
        e = (c["d"] == r["d"]).sum(0)
        eq += e
        n += len(r["q"])
        per[name] = {"drafts_equal_by_depth": e.tolist()}
        if c["lg"] is not None:
            a, b = c["lg"], r["lg"]
            diff = np.nonzero(c["d"][:, 0] != r["d"][:, 0])[0]   # step 0's misses: float64's own margin there
            per[name]["step0_misses"] = [dict(q=int(r["q"][i]), card=int(c["d"][i, 0]), f64=int(r["d"][i, 0]),
                                              f64_top2_gap=float(np.sort(b[i])[-1] - np.sort(b[i])[-2]),
                                              f64_gap_to_card=float(b[i].max() - b[i][int(np.argmax(a[i]))]))
                                         for i in diff]
            lrel = max(lrel, float(np.abs(a - b).max() / np.abs(b).max()))
            lrms.append(float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())))
        K, V = r["K"], r["V"]
        krel = max(krel, rel(torch.from_numpy(c["K"]), torch.from_numpy(K)))
        vrel = max(vrel, rel(torch.from_numpy(c["V"]), torch.from_numpy(V)))
        for cell, dft in c["pre"].items():
            pre_eq += int(dft == r["d_pre"][cell])
            pre_n += 1
    return {"positions": n, "drafts_equal_float64_by_depth": (eq / n).round(4).tolist(),
            "drafts_differ": int(n * S_MAX - eq.sum()), "step0_logits_rel_max": lrel,
            "step0_logits_rel_rms": float(np.mean(lrms)) if lrms else None,
            "catchup_K_rel_max": krel, "catchup_V_rel_max": vrel,
            "catchup_pass_drafts_equal": f"{pre_eq}/{pre_n}", "per_prompt": per}


def stage_check(dev) -> dict:
    import w4_draft as W
    seqs = [s for s in W.sequences() if s[0] in PROMPTS]
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "experts_exact": experts_exact()}
    print("experts in card form:", res["experts_exact"], flush=True)
    heads = os.environ.get("G6_HEADS", "fr,full").split(",")
    C = MtpCard(dev, heads=heads)
    res["card_memory_mib"] = C.memory()
    for h in heads:
        t0 = time.time()
        ref = reference(seqs, h)
        print(f"float64 reference ({h} head) in {time.time() - t0:.0f} s", flush=True)
        C.use_head(h)
        t0 = time.time()
        card = card_run(C, seqs)
        res[h] = compare(ref, card)
        res[h]["accept_vs_greedy_by_depth"] = {"float64": accept({k: v["d"] for k, v in ref.items()}, seqs),
                                               "card": accept({k: v["d"] for k, v in card.items()}, seqs)}
        res[h]["card_seconds"] = round(time.time() - t0, 1)
        print(f"{h} head: " + json.dumps({k: v for k, v in res[h].items() if k != "per_prompt"}), flush=True)
        if os.path.exists(os.path.join(CACHE, f"ref_emu_{h}.npz")):   # the card's precision on the host (CardMtp)
            emu = reference(seqs, h, emu=True)
            as_card = {n: dict(d=e["d"], lg=e["lg"], K=e["K"], V=e["V"],
                               pre={c: int(e["d_pre"][c]) for c in card[n]["pre"]}) for n, e in emu.items()}
            res[f"{h}_card_vs_emulation"] = compare(emu, card)
            res[f"{h}_emulation_vs_float64"] = compare(ref, as_card)
            res[h]["accept_vs_greedy_by_depth"]["emulation"] = accept({k: v["d"] for k, v in emu.items()}, seqs)
            for k in (f"{h}_card_vs_emulation", f"{h}_emulation_vs_float64"):
                print(f"{k}: " + json.dumps({a: b for a, b in res[k].items() if a != "per_prompt"}), flush=True)
    return res


def ref_detail(m, R_rows, toks, c: int) -> dict:
    """float64 step 0 at cell c (cells 0..c teacher-forced) with its intermediates: the front's X, program 2's hn_f and
    router logits, the experts, the layer's output R and the head's logits (tt/spec/mtp.py's rest(), unrolled)."""
    tok = torch.from_numpy(np.asarray(toks[:c + 1], np.int64))
    pos = torch.arange(c + 1)
    R, mixed, inj, k, v = m.front(torch.from_numpy(np.ascontiguousarray(R_rows[:c + 1])).double(), tok, pos)
    sl = slice(c, c + 1)
    qf = m.mm(mixed[sl], m.w["self_attn.q_proj.weight"]).view(1, M.NH, 2 * M.HD)
    q = m.rope(m.rms(qf[..., :M.HD], m.wn["self_attn.q_norm.weight"]), pos[sl])
    a = m.attend(q, k, v) * torch.sigmoid(qf[..., M.HD:])
    bo = m.mm(a.reshape(1, -1), m.w["self_attn.o_proj.weight"])
    R1, mixed2, inj2 = m.hc_read(R[sl], "mlp_hyper_connection.", True, apply=(bo, inj[sl]))
    y, ids = m.moe(mixed2)
    Rout = R1 + y[:, None, :] * (2 * torch.sigmoid(inj2 / HC))[:, :, None]
    _, sample, _ = m.hc_read(Rout, "hyper_connection_mixer.", False)
    return dict(o=a[0].numpy(), X=R[c].flatten().numpy(), hn=mixed2[0].numpy(), rl=m.mm(mixed2, m.w["mlp.gate.weight"])[0].numpy(),
                ids=ids[0].numpy(), R=Rout[0].flatten().numpy(), lg=m.mm(sample, m.head)[0].numpy())


def stage_layer(dev) -> dict:
    """Each program's output against float64 (ref_detail) along one prompt's first cells, catch-up passes of
    M = 1, 2, 3, 4 rows (the last row's): the front's X, program 2's hn_f and router logits, the experts' routes, R,
    the head's logits and draft."""
    import w4_draft as W
    from g3_layer import unblocks
    at = int(os.environ.get("G6_LAYER_AT", 0))         # a later start: LONG's cells before it caught up first
    name, meta, ids, R = [s for s in W.sequences() if s[0] == ("A.code.0" if not at else LONG)][0]
    C = MtpCard(dev, heads=("fr",))
    m = M.Mtp("float64", experts="lazy", vocab=head_vocab("fr"), threads=int(os.environ.get("G6_THREADS", 8)))
    T = lambda t: ttnn.to_torch(t).double().flatten()   # noqa: E731
    bf = lambda a: torch.from_numpy(np.asarray(a, np.float64)).to(torch.bfloat16).double().numpy()   # noqa: E731
    dense = M.load_dense()
    with weight_cache("layer"):     # the front's weights as the card holds them (bfp8_b), for the kernel's own check
        _, wq = g3_hc.stream(dev, lambda: front_w(dense), True, "mtp/front_w")
    wq = wq.numpy()
    emb = M.gguf_tensor("token_embd.weight")
    wh, we = dense["pre_fc_norm_hidden.weight"].astype(np.float64), dense["pre_fc_norm_embedding.weight"]
    res, c = {"rows": [], "sequence": name, "at": at}, at
    C.zero_kv()
    for c0 in range(0, at, 4):
        Mr = min(4, at - c0)
        C.set_pass(c0, np.asarray(R[c0:c0 + Mr]), ids[c0 + 1:c0 + 1 + Mr], 1)
        C.launch(0, kv_only=True)
    ttnn.synchronize_device(dev)
    for Mr in (1, 1, 2, 3, 4, 1, 4, 3):
        C.set_pass(c, np.asarray(R[c:c + Mr]), ids[c + 1:c + 1 + Mr], 1)
        mp = 1 if Mr == 1 else 2 if Mr == 2 else 4
        C.p_catchup()
        ttnn.synchronize_device(dev)
        X = x_rows(C.A1.x32_t, mp)[Mr - 1].numpy()             # (program 2 rewrites X in place)
        Rl, e = np.asarray(R[c + Mr - 1], np.float64), bf(emb[ids[c + Mr]]).astype(np.float64)
        hq = bf(Rl * wh).reshape(HC, D) @ wq[:D] / np.sqrt((Rl ** 2).mean() + M.EPS)
        X_card = (hq + (bf(e * we) @ wq[D:]) / np.sqrt((e ** 2).mean() + M.EPS)).reshape(-1)
        p1, p2, pe = C.p_layer[0]
        for fn in (p1, p2, pe, C.hd.progs["both"]):
            fn()
        ttnn.synchronize_device(dev)
        last = c + Mr - 1
        r = ref_detail(m, R, ids[1:], last)
        hn = unblocks(T(C.A2.chk["hn"]), D // 32, mp)[Mr - 1].numpy()
        rl = unblocks(T(C.A2.chk["l"]), 17, mp)[Mr - 1, :512].numpy()
        Rc, lg = T(C.r32).numpy(), C.logits()
        f = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())   # noqa: E731
        top = np.argsort(-rl, kind="stable")[:TOPK]
        # the experts' program from its own inputs (program 2's hn_f in bf16, its logits, X and w_f), exact weights
        x2 = x_rows(C.A2.x32_t, mp)[Mr - 1].numpy().reshape(HC, D)
        wf = T(C.A2.chk["w"]).numpy()[16 * (Mr - 1):16 * (Mr - 1) + HC]
        ew = np.exp(rl[top] - rl[top].max())
        ew /= ew.sum()
        g_, u_, d_ = M.decode_experts(which=top)
        xb = bf(hn)
        y = np.zeros(D)
        for j in range(TOPK):
            gg, uu = g_[j].astype(np.float64) @ xb, u_[j].astype(np.float64) @ xb
            y += d_[j].astype(np.float64) @ bf(ew[j] * gg / (1 + np.exp(-gg)) * uu)
        R_card = (x2 + wf[:, None] * y[None, :]).reshape(-1)
        row = dict(cell=last, M=Mr, X_vs_card_precision=f(X, X_card), R_vs_card_inputs=f(Rc, R_card),
                   X=f(X, r["X"]), hn=f(hn, r["hn"]), router_logits=f(rl, r["rl"]),
                   routes_same=bool(set(top.tolist()) == set(r["ids"].tolist())), R=f(Rc, r["R"]),
                   logits=f(lg, r["lg"]), draft=C.draft()[0], draft_f64=int(head_vocab("fr")[int(np.argmax(r["lg"]))]))
        rr = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))   # noqa: E731
        sc = lambda a, b: float((a * b).sum() / (b * b).sum())                     # noqa: E731
        o = T(C.A2.chk["o"]).numpy()[:M.NH * 8 * mp * 32].reshape(M.NH, 8, mp, 32)[:, :, Mr - 1].reshape(M.NH, -1)
        row["o_rel_by_head"] = [round(f(o[h], r["o"][h]), 4) for h in range(M.NH)]
        if at and Mr == 1 and os.environ.get("G6_PARTS"):   # the combine's two stages from the partials, on the host
            pt = T(C.A2.parts).numpy().reshape(-1, 32, 32)        # tile i (untilized)
            nb = (last + 1 + 31) // 32
            P = min(nb, 55)
            Q = 1 if P <= 8 else min((P + 7) // 8, 6)
            g, t, d, ps = 0, 0, 0, 20
            O = np.stack([pt[(g * 55 + p) * ps + d] for p in range(P)])          # [P, rows, dims]
            m_ = np.stack([pt[(g * 55 + p) * ps + 16][0] for p in range(P)])    # [P, cols = q rows]
            l_ = np.stack([pt[(g * 55 + p) * ps + 18][0] for p in range(P)])
            sub0 = 2 * 55 * ps
            dbg = {"P": P, "Q": Q}
            for q in range(Q):
                p0, p1 = q * P // Q, (q + 1) * P // Q
                Mq = m_[p0:p1].max(0)
                A = np.exp(m_[p0:p1] - Mq)
                Lq = (A * l_[p0:p1]).sum(0)
                Oq = (A[:, :, None] * O[p0:p1]).sum(0)
                s0 = sub0 + (((g * 2 + t) * 8 + d) * 6 + q) * 3
                cO, cm, cl = pt[s0][:12], pt[s0 + 1][0, :12], pt[s0 + 2][0, :12]
                dbg[f"q{q}"] = dict(n=p1 - p0, M=f(cm, Mq[:12]), L=f(cl, Lq[:12]), O=f(cO, Oq[:12]),
                                    O_rows=[round(f(cO[r], Oq[r]), 3) for r in range(12)])
            row["parts_check"] = dbg
            print("parts", json.dumps(dbg), flush=True)
        row.update({f"rms_{k}": rr(a, r[k]) for k, a in (("X", X), ("hn", hn), ("rl", rl), ("R", Rc), ("lg", lg))})
        row.update({f"scale_{k}": sc(a, r[k]) for k, a in (("hn", hn), ("R", Rc), ("lg", lg))})
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        c += Mr
    return res


LONG = "B.strata-drafter"   # the census's one agent prompt that fits the dense attention (4,785 cells)


def stage_long(dev) -> dict:
    """Long contexts (the A prompts reach ~130 cells: one AP block per core, a one-stage combine): LONG's prompt
    caught up in passes of 4 rows (kv_only but the last 8, whose drafts are compared), then a chain of S_MAX steps at
    each of its first 16 decode cells, against float64 (tt/spec/mtp.py teacher_forced + chain_steps, the same cells);
    placed at cell offsets 0 and 3,900 (the cells before it empty: zero keys score 0 against the layer's 400-700, so
    they weigh nothing) to reach 8,700 cells: AP's 5-block partials and both combine stages at T = 1 and 2."""
    import w4_draft as W
    name, meta, ids, R = [s for s in W.sequences() if s[0] == LONG][0]
    heads = os.environ.get("G6_HEADS", "fr").split(",")
    C = MtpCard(dev, heads=heads)
    cut = int(os.environ.get("G6_LONG_CELLS", 0)) or meta["n_prompt"]   # a shorter prompt: its prefix (the rest teacher-forced)
    q0 = cut - 1
    qs = np.arange(q0, q0 + 16)
    ends = np.arange(3, q0, 4)[-8:]                    # the last 8 catch-up passes' last rows
    rows = np.concatenate([ends, qs])
    th = int(os.environ.get("G6_THREADS", 16))
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "sequence": name, "cells": int(qs[-1] + 1)}
    for h in heads:
        C.use_head(h)
        m = None
        for off in [int(x) for x in os.environ.get("G6_LONG_OFFSETS", "0,3900").split(",")]:
            path = os.path.join(CACHE, f"ref_long_{h}_{off}" + (f"_{cut}" if cut != meta["n_prompt"] else "") + ".npz")
            if not os.path.exists(path):
                t0 = time.time()
                m = m or M.Mtp("float64", experts="lazy", vocab=head_vocab(h), threads=th)
                Rt = torch.from_numpy(np.ascontiguousarray(R[:qs[-1] + 1])).to(m.dt)
                cache, (R0, lg0) = M.teacher_forced(m, Rt, ids[1:qs[-1] + 2], off, rest_rows=rows)
                d0, _ = m.pick(lg0)
                d = [d0] + [x for x, _ in M.chain_steps(m, cache, rows + off, R0, d0, S_MAX - 1)]
                np.savez(path, d=torch.stack(d, 1).numpy(), lg=lg0[len(ends):].double().numpy(),
                         K=cache.K[:q0].double().numpy(), V=cache.V[:q0].double().numpy())
                print(f"float64 reference ({h}, offset {off}) in {time.time() - t0:.0f} s", flush=True)
            z = np.load(path)
            C.zero_kv()
            pre = []
            for c in range(0, q0, 4):
                Mr = min(4, q0 - c)
                C.set_pass(off + c, np.asarray(R[c:c + Mr]), ids[c + 1:c + 1 + Mr], 1)
                C.launch(0, kv_only=c + Mr - 1 not in ends)
                if c + Mr - 1 in ends:
                    ttnn.synchronize_device(dev)
                    pre.append(C.draft()[0])
            K, V = C.kv(range(off, off + q0))
            drafts, lgs = [], []
            for q in qs:
                C.set_pass(off + q, np.asarray(R[q:q + 1]), ids[q + 1:q + 2], S_MAX)
                row = []
                for k in range(S_MAX):
                    C.launch(k, logits=k == 0)
                    ttnn.synchronize_device(dev)
                    row.append(C.draft()[0])
                    if k == 0:
                        lgs.append(C.logits())
                drafts.append(row)
            d, lg, rd = np.array(drafts), np.array(lgs), z["d"][len(ends):]
            out = {"drafts_equal_float64_by_depth": (d == rd).mean(0).round(4).tolist(),
                   "catchup_pass_drafts_equal": f"{int((np.array(pre) == z['d'][:len(ends), 0]).sum())}/{len(ends)}",
                   "step0_logits_rel_max": float(np.abs(lg - z["lg"]).max() / np.abs(z["lg"]).max()),
                   "step0_logits_rel_rms": float(np.sqrt(((lg - z["lg"]) ** 2).mean() / (z["lg"] ** 2).mean())),
                   "step0_top1_equal": int((lg.argmax(1) == z["lg"].argmax(1)).sum()),
                   "catchup_K_rel_max": rel(torch.from_numpy(K), torch.from_numpy(z["K"])),
                   "catchup_V_rel_max": rel(torch.from_numpy(V), torch.from_numpy(z["V"]))}
            res[f"{h}_offset{off}_cells{cut}"] = out
            print(f"{h} head, offset {off} (cells to {off + qs[-1]}): " + json.dumps(out), flush=True)
    return res


def stage_replay(dev) -> dict:
    """The card's drafts (a chain of S_MAX steps) at every decode position of the census A prompts (replay
    data, tt/flashnext/w4_draft.py; the B prompts pass the dense attention's CTX_DENSE cells), into
    CACHE/drafts_card_<head>/ (w4_draft's format), and E per pass at --spec M (M - 1 MTP drafts every pass) against
    float32 drafts on the same sequences."""
    import w4_draft as W
    seqs = [s for s in W.sequences() if s[0].startswith("A.")]
    heads = os.environ.get("G6_HEADS", "fr").split(",")
    C = MtpCard(dev, heads=heads)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "sequences": len(seqs)}
    cost = W.costs()
    for h in heads:
        C.use_head(h)
        out = os.path.join(CACHE, f"drafts_card_{h}")
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        for name, meta, ids, R in seqs:
            C.zero_kv()
            q0 = meta["n_prompt"] - 1
            for c in range(0, q0, M_MAX):
                Mr = min(M_MAX, q0 - c)
                C.set_pass(c, np.asarray(R[c:c + Mr]), ids[c + 1:c + 1 + Mr], 1)
                C.launch(0, kv_only=True)
            d = []
            for q in range(q0, len(ids) - 1):
                C.set_pass(q, np.asarray(R[q:q + 1]), ids[q + 1:q + 2], S_MAX)
                row = []
                for k in range(S_MAX):
                    C.launch(k)
                    row.append(C.draft()[0])
                d.append(row)
            np.savez(os.path.join(out, name + ".npz"), q0=np.array(q0), d=np.array(d, np.int32),
                     p=np.ones((len(d), S_MAX), np.float32))
        card = {f[:-4]: (int(z["q0"]), z["d"], z["p"]) for f in sorted(os.listdir(out))
                for z in [np.load(os.path.join(out, f))]}
        f32 = W.load_drafts(h)
        res[h] = {"card_seconds": round(time.time() - t0, 1),
                  "E_by_rows": {Mr: {"card": W.run(W.fixed_mtp(Mr), seqs, card, cost["mtp_subset"])["E"],
                                     "float32": W.run(W.fixed_mtp(Mr), seqs, f32, cost["mtp_subset"])["E"]}
                                for Mr in range(2, S_MAX + 2)},
                  "drafts_equal_float32_by_depth": (sum((card[n][1] == f32[n][1]).sum(0) for n in card) /
                                                    sum(len(card[n][1]) for n in card)).round(4).tolist()}
        print(f"{h} head: " + json.dumps(res[h]), flush=True)
    return res


# ---------------------------------------------------------------- timing

def stage_time(dev) -> dict:
    import w4_draft as W
    heads = os.environ.get("G6_HEADS", "fr,full").split(",")
    C = MtpCard(dev, heads=heads)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "card_memory_mib": C.memory(), "traced_us": {}}
    name, meta, ids, R = [s for s in W.sequences() if s[0] == "A.code.0"][0]
    for h in heads:
        C.use_head(h)
        for pos in (100, 7900):
            for steps in (1, 3):
                C.set_pass(pos, np.asarray(R[100:101]), ids[101:102], steps)
                us = traced_us(dev, lambda: C.run(steps))
                res["traced_us"][f"{h}_pos{pos}_steps{steps}"] = round(us, 1)
                print(f"{h} head, pos {pos}, {steps} step(s): {us:.1f} us traced", flush=True)
        # one step's programs alone (pos 7900)
        C.set_pass(7900, np.asarray(R[100:101]), ids[101:102], 1)
        p1, p2, pe = C.p_layer[0]
        for pname, fn in (("front", C.p_catchup), ("attn1", p1), ("attn2", p2), ("experts", pe),
                          ("head", C.hd.progs["ids"])):
            res["traced_us"][f"{h}_part_{pname}"] = round(traced_us(dev, fn), 1)
        print(f"{h} head parts: " + ", ".join(f"{k} {v}" for k, v in res["traced_us"].items() if "_part_" in k
                                              and k.startswith(h)), flush=True)
    # the host's side of a pass: set_pass for a 4-row catch-up and 3 chained steps (its in-queue writes), synced
    res["set_pass_us"] = {}
    for Mr, steps in ((4, 1), (1, 3), (4, 4)):
        ts = []
        for _ in range(22):
            t0 = time.perf_counter()
            C.set_pass(7900, np.asarray(R[100:100 + Mr]), ids[101:101 + Mr], steps)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e6)
        res["set_pass_us"][f"M{Mr}_steps{steps}"] = round(float(np.median(ts[2:])), 1)
    print("set_pass (us, synced):", res["set_pass_us"], flush=True)
    return res


PHASE_NAMES = {"front": ["F1"], "attn1": ["X", "H1", "HL", "H2", "P", "N"],
               "attn2": ["in", "AP", "AC", "oproj", "H1f", "HLf", "H2f", "R1", "R2"], "exp": ["P0", "E1"],
               "head": ["X", "H1", "HL", "H2", "LM"]}     # the phases before each barrier; "post" after the last
WRITER_END = ("front", "exp")   # programs whose writers stamp their end (else the coordinator reader's end mark)


def program_bytes(C: MtpCard, cells: int) -> dict:
    """Card memory bytes each program of one step streams (weights, the KV cache's cells)."""
    A1, A2, H = C.A1, C.A2, C.hd.H
    s = lambda ts: sum(nbytes(t) for t in ts)   # noqa: E731
    per_expert = 40 * 80 * 1088 + 40 * 3 * 4096 + 80 * 20 * 1088 + 80 * 4096
    return {"front": s([C.w_front, C.whe]), "attn1": s([A1.wd, A1.wu, A1.wp]),
            "attn2": s([A2.wd, A2.wu, A2.wo, A2.wr, A2.wgu, A2.wsd]) + 2 * 2 * cells * 256 * 2,
            "exp": TOPK * per_expert, "head": s([H.wl_t, H.wd_t, H.wu_t])}


def step_profile(C: MtpCard, reps: int = 12) -> dict:
    """One traced step's programs from grid.h's per-core clocks (medians over reps replays): per program its start
    after the step's (the coordinator's clock), its phases' walls (barrier to barrier on the coordinator) and the
    slowest core's busy time in each (its release to its writer's arrival), the last barrier to the end (the slowest
    writer's end where the writer stamps it, else the coordinator reader's)."""
    n, dev = C.prof_n, C.dev
    us = lambda d: ((d % (1 << 32)) - ((d % (1 << 32)) >= (1 << 31)) * (1 << 32)) / CLOCK_MHZ   # noqa: E731
    C.run(1)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    C.run(1)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    runs = []
    for _ in range(reps + 2):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        runs.append({k: profile_words(C.gprof[k], len(C.grid), n) for k in PROGRAMS})
    ttnn.release_trace(dev, tid)
    runs = runs[2:]
    med = lambda v: round(float(np.median(v)), 2)   # noqa: E731
    out = {}
    for k in PROGRAMS:
        names = PHASE_NAMES[k]       # (the profile's unstamped words hold whatever L1 held: not a barrier count)
        nb = len(names)
        ph = {}
        for b in range(1, nb + 1):
            walls, busy = [], []
            for r in runs:
                w = r[k]
                rel, arr, mk = w[:, n:2 * n], w[:, 2 * n:3 * n], w[:, 3 * n:]
                prev = mk[:, 0] if b == 1 else rel[:, b - 1]
                walls.append(float(us(rel[0, b] - prev[0])))
                busy.append(float(us(arr[:, b] - prev).clamp(min=0).max()))
            ph[names[b - 1]] = dict(wall_us=med(walls), busy_max_us=med(busy))
        post, start = [], []
        for r in runs:
            w = r[k]
            rel, arr, mk = w[:, n:2 * n], w[:, 2 * n:3 * n], w[:, 3 * n:]
            post.append(float(us(arr[:, n - 1] - rel[:, nb]).max()) if k in WRITER_END else
                        float(us(mk[0, n - 1] - rel[0, nb])))
            start.append(float(us(mk[0, 0] - r[PROGRAMS[0]][0, 3 * n])))
        out[k] = dict(start_us=med(start), phases=ph, post_us=med(post))
    for a, b in zip(PROGRAMS, PROGRAMS[1:]):          # program a's start to b's on the coordinator: its share
        out[a]["span_us"] = round(out[b]["start_us"] - out[a]["start_us"], 2)
    return out


def stage_profile(dev) -> dict:
    """One step's programs per phase (step_profile) at cells 100 and 7,900, subset head, against each program's
    floor (its card memory bytes at 490 GB/s), and the traced step for comparison."""
    import w4_draft as W
    name, meta, ids, R = [s for s in W.sequences() if s[0] == "A.code.0"][0]
    C = MtpCard(dev, heads=("fr",), prof_n=32)
    res = {"date": time.strftime("%Y-%m-%d %H:%M")}
    for pos in (100, 7900):
        C.set_pass(pos, np.asarray(R[100:101]), ids[101:102], 1)
        p = step_profile(C)
        for k, v in program_bytes(C, pos + 1).items():
            p[k]["bytes"], p[k]["floor_us"] = int(v), round(v / 490e3, 2)
        p["traced_step_us"] = round(traced_us(dev, lambda: C.run(1)), 1)
        res[f"pos{pos}"] = p
        print(f"pos {pos}: step {p['traced_step_us']} us traced", flush=True)
        for k in PROGRAMS:
            print(f"  {k:6s} start {p[k]['start_us']} span {p[k].get('span_us', '-')} floor {p[k]['floor_us']} "
                  f"post {p[k]['post_us']}: " + ", ".join(f"{a} {b['wall_us']}/{b['busy_max_us']}"
                                                         for a, b in p[k]["phases"].items()), flush=True)
    return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    stage = os.environ.get("G6_STAGE", "check")
    if stage == "ref":   # the float64 references, without the card
        import w4_draft as W
        seqs = [s for s in W.sequences() if s[0] in PROMPTS]
        for h in os.environ.get("G6_HEADS", "fr,full").split(","):
            t0 = time.time()
            reference(seqs, h, emu=os.environ.get("G6_EMU") == "1")
            print(f"reference ({h} head, emu {os.environ.get('G6_EMU')}) in {time.time() - t0:.0f} s", flush=True)
        return
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, worker_l1_size=WORKER_L1_M)
    try:
        if stage == "weights":
            for h in os.environ.get("G6_HEADS", "fr").split(","):
                dump_card_weights(dev, h)
            return
        if stage == "build":
            C = MtpCard(dev, heads=os.environ.get("G6_HEADS", "fr,full").split(","))
            res = {"card_memory_mib": C.memory()}
            print(res, flush=True)
            return
        res = {"check": stage_check, "layer": stage_layer, "replay": stage_replay,
               "time": stage_time, "profile": stage_profile, "long": stage_long}[stage](dev)
    finally:
        ttnn.close_mesh_device(dev)
    tag = "_cells" + os.environ["G6_LONG_CELLS"] if stage == "long" and os.environ.get("G6_LONG_CELLS") else ""
    tag = "_at" + os.environ["G6_LAYER_AT"] if stage == "layer" and os.environ.get("G6_LAYER_AT") else tag
    with open(os.path.join(OUT, f"g6_mtp_{stage}{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
