"""Strata's MTP draft layer for Qwen3.8-Flash-Next, on the host: one draft step as Strata's src/core/mtp.cpp runs it
(record_front + record_rest), from the runtime files Strata loads ($TT_MOEOFFLOAD_DATA/strata/mtp/rt: dense.txt, dense.bin,
experts.bin; written by third_party/strata/tools/mtp_rt.py) and the target's embedding and LM head (the GGUF's
token_embd.weight and output.weight, which the draft layer shares).

One step, for a cell at position `pos` (mtp.hpp: cell i pairs the target's final multi-stream residual R_i, 4 x 2560,
with the token at i + 1; its output predicts the token at i + 2 and its own residual feeds the next step):

  e  = fc_embedding @ rms(embed(tok)) * w_e                    (pre_fc_norm_embedding; dense.bin stores 1 + w)
  h  = fc_hidden @ (rms_10240(R_in) * w_h), per stream          (ONE RMS over all four streams: Strata's default;
                                                                 --mtp-hnorm stream / llama.cpp's graph: per stream)
  R  = h + e (to every stream)
  attention hyper-connection read (gr_read: per-stream RMS, low-rank gate, mean over streams, 4 injections)
  attention: q|gate (24 heads x 2 x 256), k, v (2 KV heads); q/k RMSNorm per head; NEOX RoPE on 64 of 256 dims at
             base 1e7 (pos = the cell); dense attention over every cell the layer has written (mtp.hpp: Strata's
             runtime choice, identical to the model's sparse selection below 2,051 cells); sigmoid gate; o_proj
  MLP hyper-connection read with the attention's write folded in (R += o * 2 sigmoid(inj / 4))
  MoE: router (bf16) softmax over 512, top 10 (stable, ties by index), renormalized (clamp 2^-14); Q2_0 experts
       silu(gate) * up -> down; shared expert * sigmoid(shared_expert_gate . x); R += y * 2 sigmoid(inj2 / 4)
  final mixer read (hyper_connection_mixer, no inject) -> the target's LM head (the whole vocabulary, or a subset:
       rt/draft_vocab.bin / --mtp-draft-vocab) -> argmax = the draft; its softmax probability (over the head's rows)
       gates the chain (--spec-min-p)

Every weight is held exactly (Q8_0, Q2_0 and Q6_K values and bf16 are exact in float32); the activations run in the
dtype asked for (float64 = the reference). Strata's own runtime rounds activations to q8_1 for its MMVQ GEMVs and keeps
its K/V as INT8 under --kv int8: those are its precision choices, not modelled here (see the f32/bf16 error study in
tt/flashnext/w4_draft.py).
"""
from __future__ import annotations

import glob
import math
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models

import numpy as np
import torch

RT = os.path.join(DATA, "strata/mtp/rt")
MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/qwen3.8-flash-next-ud-iq4xs/UD-IQ4_XS")   # engine.py --model-dir sets it
CACHE = os.path.join(DATA, "w4_draft")
D, HC, LR = 2560, 4, 320
NH, NKV, HD, NROT, ROPE_BASE = 24, 2, 256, 64, 1e7
NX, TOPK, FF = 512, 10, 640
EPS = 1e-6
VOCAB = 248320
BLOB = 3 * (D * FF * 18 // 64)


# ---------------------------------------------------------------- loading
# the Q8_0 and Q2_0 dequantizations follow Strata's tools/mtp_pack.py (MIT, LICENSES/Strata-MIT.txt)

def _q8_0(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(rows, cols)


def load_dense(rt: str = RT) -> dict[str, np.ndarray]:
    """dense.txt / dense.bin -> {name: float32 [rows, cols] (1-D norms: [cols], 1 + w already applied)}."""
    blob = np.fromfile(os.path.join(rt, "dense.bin"), dtype=np.uint8)
    out = {}
    for line in open(os.path.join(rt, "dense.txt")):
        name, kind, rows, cols, off, nb = line.split()
        rows, cols, off, nb = int(rows), int(cols), int(off), int(nb)
        raw = blob[off:off + nb]
        if kind == "q8_0":
            out[name] = _q8_0(raw, rows, cols)
        elif kind == "bf16":
            out[name] = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
        elif kind == "f32":
            out[name] = raw.view(np.float32).reshape(cols).copy()
        else:
            raise ValueError(f"{name}: kind {kind}")
    return out


def decode_experts(rt: str = RT, which=None) -> np.ndarray:
    """experts.bin (mtp_rt.py blob_of: gate/up rows interleaved 2r = gate r, 2r + 1 = up r, then down rows; the Q2_0
    codes in one plane, the fp16 scales in another; value = (code - 1) * d, 64 per block) -> float16 (exact)
    [n, 3, ...]: returns (gate [n, 640, 2560], up [n, 640, 2560], down [n, 2560, 640])."""
    mm = np.memmap(os.path.join(rt, "experts.bin"), dtype=np.uint8, mode="r").reshape(NX, BLOB)
    idx = np.arange(NX) if which is None else np.asarray(which)
    n = len(idx)
    gate = np.empty((n, FF, D), np.float16)
    up = np.empty((n, FF, D), np.float16)
    down = np.empty((n, D, FF), np.float16)
    a = 2 * FF * (D // 64) * 16                 # interleaved gate/up codes
    b = a + D * (FF // 64) * 16                 # down codes
    c = b + 2 * FF * (D // 64) * 2              # interleaved scales
    for k, e in enumerate(idx):
        x = np.asarray(mm[e])

        def dec(codes, scales, rows, cols):
            q = codes.reshape(rows, cols // 64, 16)
            v = np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=3).reshape(rows, cols // 64, 64)
            d = scales.view(np.float16).reshape(rows, cols // 64, 1).astype(np.float32)
            return ((v.astype(np.float32) - 1.0) * d).reshape(rows, cols)
        gu = dec(x[:a], x[b:c], 2 * FF, D)
        gate[k], up[k] = gu[0::2], gu[1::2]
        down[k] = dec(x[a:b], x[c:BLOB], D, FF)
    return gate, up, down


def bf16_experts(which, folder: str = os.path.join(os.path.dirname(RT), "tensors")):
    """The checkpoint's BF16 experts (tools/mtp_fetch.py's raw tensors: gate_up_proj [512, 1280, 2560] with gate rows
    0..639, up rows 640..1279; down_proj [512, 2560, 640]) -> float32 (gate, up, down) for the experts in `which`."""
    gu = np.memmap(os.path.join(folder, "mtp.layers.0.mlp.experts.gate_up_proj.bin"), dtype="<u2", mode="r",
                   shape=(NX, 2 * FF, D))
    dn = np.memmap(os.path.join(folder, "mtp.layers.0.mlp.experts.down_proj.bin"), dtype="<u2", mode="r",
                   shape=(NX, D, FF))
    f = lambda a: (np.asarray(a).astype(np.uint32) << 16).view(np.float32)   # noqa: E731
    w = np.asarray(which)
    g = f(gu[w])
    return g[:, :FF], g[:, FF:], f(dn[w])


def gguf_tensor(name: str, folder: str = MODEL) -> np.ndarray:
    """A target tensor dequantized to float32, numpy order ([rows, cols] for a 2-D weight), cached in CACHE."""
    path = os.path.join(CACHE, f"target_{name}.f32.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    from gguf import GGUFReader
    from gguf.quants import dequantize
    for f in sorted(glob.glob(os.path.join(folder, "*.gguf"))):
        for t in GGUFReader(f).tensors:
            if t.name == name:
                if len(t.shape) == 1:
                    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
                rows = int(t.shape[1])
                out = np.empty((rows, int(t.shape[0])), np.float32)
                step = 8192
                for r in range(0, rows, step):           # by row chunks: bounded memory
                    out[r:r + step] = dequantize(t.data[r:r + step], t.tensor_type).astype(np.float32)
                os.makedirs(CACHE, exist_ok=True)
                np.save(path, out)
                return np.load(path, mmap_mode="r")
    raise KeyError(name)


def draft_vocab(path: str | None) -> np.ndarray | None:
    """A draft head's token subset (int32 ids, tools/draft_vocab.py's format), or None for the whole vocabulary."""
    return None if path is None else np.fromfile(path, dtype=np.int32)


# ---------------------------------------------------------------- the layer

class Mtp:
    """The draft layer's weights as torch tensors of `dtype` (float64 for the reference, float32, or 'bf16': weights
    and every GEMV input rounded to bfloat16, float32 accumulation), and its K/V.

    experts: 'all' decodes the 512 experts up front, 'lazy' decodes each on first use. expert_src: 'q2_0' (rt/experts.bin,
    what Strata runs) or 'bf16' (the checkpoint's BF16 experts, $TT_MOEOFFLOAD_DATA/strata/mtp/tensors: a variant only)."""

    def __init__(self, dtype="float64", rt: str = RT, experts: str = "lazy", vocab: np.ndarray | None = None,
                 head: np.ndarray | None = None, embd: np.ndarray | None = None, threads: int | None = None,
                 expert_src: str = "q2_0"):
        if threads:
            torch.set_num_threads(threads)
        self.bf16 = dtype == "bf16"
        self.dt = torch.float32 if dtype in ("float32", "bf16") else torch.float64
        w = load_dense(rt)
        self.w = {k: self._t(v) for k, v in w.items()}
        self.wn = {k: torch.from_numpy(v).to(self.dt) for k, v in w.items() if v.ndim == 1}   # norms: not rounded
        self.embd = embd if embd is not None else gguf_tensor("token_embd.weight")
        head = head if head is not None else gguf_tensor("output.weight")
        self.vocab = None if vocab is None else torch.from_numpy(np.asarray(vocab, np.int64))
        hv = head if vocab is None else head[np.asarray(vocab)]
        self.head = self._t(np.ascontiguousarray(hv))
        self.rt, self.src = rt, expert_src
        self._exp = {}
        if experts == "all":
            for a in range(0, NX, 64):
                g, u, d = self._decode(list(range(a, a + 64)))
                for k in range(64):
                    self._exp[a + k] = tuple(self._t(x[k]) for x in (g, u, d))
        half = NROT // 2
        self.inv_freq = ROPE_BASE ** (-torch.arange(half, dtype=torch.float64) * 2 / NROT)

    def _t(self, a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(a, dtype=np.float32))
        if self.bf16:
            return t.to(torch.bfloat16)
        return t.to(self.dt)

    def _decode(self, which):
        return decode_experts(self.rt, which) if self.src == "q2_0" else bf16_experts(which)

    def expert(self, e: int):
        if e not in self._exp:
            g, u, d = self._decode([e])
            self._exp[e] = (self._t(g[0]), self._t(u[0]), self._t(d[0]))
        return self._exp[e]

    # ---- primitives
    def mm(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """x [..., in] @ W[out, in].T in the layer's precision."""
        if self.bf16:
            return (x.to(torch.bfloat16).float() @ W.float().T)
        return x @ W.T

    @staticmethod
    def rms(x, w=None):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        return y if w is None else y * w

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """NEOX RoPE of the first 64 dims; x [B, H, 256], pos [B]."""
        half = NROT // 2
        th = pos.to(torch.float64)[:, None] * self.inv_freq[None, :]
        c, s = torch.cos(th).to(x.dtype)[:, None, :], torch.sin(th).to(x.dtype)[:, None, :]
        a, b = x[..., :half], x[..., half:NROT]
        return torch.cat([a * c - b * s, a * s + b * c, x[..., NROT:]], dim=-1)

    def hc_read(self, R, pre: str, inject: bool, apply=None):
        """gr_read / fused_gr_read: R [B, 4, 2560] -> (R', mixed [B, 2560], inj [B, 4] or None); `apply` = (bo, inj)
        of the previous half folds its write in first."""
        if apply is not None:
            bo, inj0 = apply
            R = R + bo[:, None, :] * (2 * torch.sigmoid(inj0 / HC))[:, :, None]
        xn = self.rms(R) * self.wn[pre + "hc_norm.weight"].view(HC, D)
        flat = xn.reshape(R.shape[0], HC * D)
        lo = torch.nn.functional.silu(self.mm(flat, self.w[pre + "input_mix_weight_down.weight"]) / HC)
        gate = torch.sigmoid(self.mm(lo, self.w[pre + "input_mix_weight_up.weight"])).view(-1, HC, D)
        inj = self.mm(flat, self.w[pre + "block_inject_weight.weight"]) if inject else None
        return R, (xn * gate).mean(1), inj

    def front(self, R_in: torch.Tensor, tok: torch.Tensor, pos: torch.Tensor, hnorm_stream: bool = False):
        """record_front for B rows: -> (R [B,4,2560], mixed, inj, k [B,2,256] roped, v [B,2,256])."""
        B = R_in.shape[0]
        emb = torch.from_numpy(np.asarray(self.embd[tok.numpy()], np.float32)).to(self.dt)
        e2 = self.mm(self.rms(emb, self.wn["pre_fc_norm_embedding.weight"]), self.w["fc_embedding.weight"])
        Rv = R_in.to(self.dt).reshape(B, HC, D)
        if hnorm_stream:
            hn = self.rms(Rv) * self.wn["pre_fc_norm_hidden.weight"].view(HC, D)
        else:
            hn = self.rms(Rv.reshape(B, HC * D), self.wn["pre_fc_norm_hidden.weight"]).view(B, HC, D)
        R = self.mm(hn, self.w["fc_hidden.weight"]) + e2[:, None, :]
        R, mixed, inj = self.hc_read(R, "attn_hyper_connection.", True)
        k = self.mm(mixed, self.w["self_attn.k_proj.weight"]).view(B, NKV, HD)
        v = self.mm(mixed, self.w["self_attn.v_proj.weight"]).view(B, NKV, HD)
        k = self.rope(self.rms(k, self.wn["self_attn.k_norm.weight"]), pos)
        return R, mixed, inj, k, v

    def attend(self, q, K, V, mask=None, Kx=None, Vx=None):
        """q [B, 24, 256]; K, V [C, 2, 256] shared cells with mask [B, C] (True = visible); Kx, Vx [B, J, 2, 256] each
        row's own extra cells (all visible). Softmax over both, scale 1/sqrt(256), GQA 12 heads per KV head."""
        B = q.shape[0]
        qg = q.view(B, NKV, NH // NKV, HD)
        parts_s, parts_v = [], []
        if K is not None:
            s = torch.einsum("bgnd,cgd->bgnc", qg, K) / math.sqrt(HD)
            if mask is not None:
                s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
            parts_s.append(s)
            parts_v.append(("shared", V))
        if Kx is not None:
            parts_s.append(torch.einsum("bgnd,bjgd->bgnj", qg, Kx) / math.sqrt(HD))
            parts_v.append(("own", Vx))
        s = torch.cat(parts_s, dim=-1)
        p = torch.softmax(s, dim=-1)
        out, o = 0, 0
        for (kind, Vv), sp in zip(parts_v, parts_s):
            n = sp.shape[-1]
            pp = p[..., o:o + n]
            o += n
            out = out + (torch.einsum("bgnc,cgd->bgnd", pp, Vv) if kind == "shared"
                         else torch.einsum("bgnj,bjgd->bgnd", pp, Vv))
        return out.reshape(B, NH, HD)

    def moe(self, x: torch.Tensor):
        """router + routed experts + shared expert for B rows -> (y [B, 2560], ids [B, 10])."""
        B = x.shape[0]
        logits = self.mm(x, self.w["mlp.gate.weight"])
        probs = torch.softmax(logits.to(torch.float64) if self.dt == torch.float64 else logits, dim=-1)
        # stable descending order, ties by index (router_top10.hpp)
        order = torch.sort(-probs, dim=-1, stable=True).indices[:, :TOPK]
        vals = torch.gather(probs, 1, order)
        wts = (vals / torch.clamp(vals.sum(-1, keepdim=True), min=2.0 ** -14)).to(x.dtype)
        y = torch.zeros_like(x)
        flat_ids = order.reshape(-1)
        for e in torch.unique(flat_ids).tolist():
            rows, slots = (order == e).nonzero(as_tuple=True)
            g, u, d = self.expert(e)
            xe = x[rows]
            h = torch.nn.functional.silu(self.mm(xe, g)) * self.mm(xe, u)
            y.index_add_(0, rows, self.mm(h, d) * wts[rows, slots][:, None])
        sh = self.mm(torch.nn.functional.silu(self.mm(x, self.w["mlp.shared_expert.gate_proj.weight"])) *
                     self.mm(x, self.w["mlp.shared_expert.up_proj.weight"]), self.w["mlp.shared_expert.down_proj.weight"])
        sg = torch.sigmoid(self.mm(x, self.w["mlp.shared_expert_gate.weight"]))
        return y + sh * sg, order

    def rest(self, R, mixed, inj, pos, K=None, V=None, mask=None, Kx=None, Vx=None):
        """record_rest for B rows (their K/V already appended: the shared cells K, V under `mask` and/or each row's own
        Kx, Vx) -> (R_out [B, 4, 2560], logits [B, head rows])."""
        B = R.shape[0]
        qf = self.mm(mixed, self.w["self_attn.q_proj.weight"]).view(B, NH, 2 * HD)
        q = self.rope(self.rms(qf[..., :HD], self.wn["self_attn.q_norm.weight"]), pos)
        a = self.attend(q, K, V, mask, Kx, Vx) * torch.sigmoid(qf[..., HD:])
        bo = self.mm(a.reshape(B, NH * HD), self.w["self_attn.o_proj.weight"])
        R, mixed2, inj2 = self.hc_read(R, "mlp_hyper_connection.", True, apply=(bo, inj))
        y, _ = self.moe(mixed2)
        R = R + y[:, None, :] * (2 * torch.sigmoid(inj2 / HC))[:, :, None]
        _, sample, _ = self.hc_read(R, "hyper_connection_mixer.", False)
        return R, self.mm(sample, self.head)

    def pick(self, logits: torch.Tensor):
        """-> (draft ids [B] in the vocabulary, their probability under the head's softmax [B])."""
        lp = torch.log_softmax(logits.to(torch.float64), dim=-1)
        top = torch.argmax(logits, dim=-1)
        prob = torch.exp(torch.gather(lp, 1, top[:, None])[:, 0])
        ids = top if self.vocab is None else self.vocab[top]
        return ids, prob


class Cache:
    """The draft layer's K/V over consecutive cells [c0, c0 + n)."""

    def __init__(self, c0: int, K: torch.Tensor, V: torch.Tensor):
        self.c0, self.K, self.V = c0, K, V


def teacher_forced(m: Mtp, R_rows: torch.Tensor, next_tok: np.ndarray, c0: int, rest_rows=None, chunk: int = 256,
                   hnorm_stream: bool = False):
    """The verified cells [c0, c0 + n): cell c pairs R_c with the token at c + 1 (prefill + catch-up). Returns the
    K/V cache and, for the cells in `rest_rows` (offsets), step 0 of a draft chain from that cell: (R_out, logits)
    with attention over cells <= c (causal)."""
    n = R_rows.shape[0]
    tok = torch.from_numpy(np.asarray(next_tok, np.int64))
    pos = torch.arange(c0, c0 + n)
    Ks, Vs, fr = [], [], {}
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        R, mixed, inj, k, v = m.front(R_rows[a:b], tok[a:b], pos[a:b], hnorm_stream)
        Ks.append(k)
        Vs.append(v)
        fr[a] = (R, mixed, inj)
    K, V = torch.cat(Ks), torch.cat(Vs)
    out = None
    if rest_rows is not None and len(rest_rows):
        rr = np.asarray(rest_rows)
        R_outs, logits = [], []
        for a in range(0, len(rr), chunk):
            sel = rr[a:a + chunk]
            Rs = torch.cat([fr[(i // chunk) * chunk][0][i % chunk][None] for i in sel])
            Ms = torch.cat([fr[(i // chunk) * chunk][1][i % chunk][None] for i in sel])
            Is = torch.cat([fr[(i // chunk) * chunk][2][i % chunk][None] for i in sel])
            hi = int(sel.max()) + 1
            mask = torch.arange(hi)[None, :] <= torch.from_numpy(sel)[:, None]
            Ro, lg = m.rest(Rs, Ms, Is, pos[torch.from_numpy(sel)], K[:hi], V[:hi], mask)
            R_outs.append(Ro)
            logits.append(lg)
        out = (torch.cat(R_outs), torch.cat(logits))
    return Cache(c0, K, V), out


def chain_steps(m: Mtp, cache: Cache, starts: np.ndarray, R0: torch.Tensor, d0: torch.Tensor, steps: int,
                chunk: int = 256, hnorm_stream: bool = False):
    """Draft steps 1..steps of the chains that start at cells `starts` (absolute), from step 0's residual R0 and draft
    d0. Step j writes cell start + j from (previous residual, previous draft) and attends to the verified cells
    <= start plus its own chain's cells start + 1 .. start + j (mtp.cpp: draft j's cell p + a + j). Returns lists
    over j = 1..steps of (draft ids [B], probabilities [B])."""
    outs = []
    st = torch.from_numpy(np.asarray(starts, np.int64))
    R_prev, d_prev = R0, d0
    Kx = Vx = None
    for j in range(1, steps + 1):
        Rn, ids, prob, kx, vx = [], [], [], [], []
        for a in range(0, len(st), chunk):
            sl = slice(a, a + chunk)
            pos = st[sl] + j
            R, mixed, inj, k, v = m.front(R_prev[sl].reshape(-1, HC * D), d_prev[sl], pos, hnorm_stream)
            Kj = k[:, None] if Kx is None else torch.cat([Kx[sl], k[:, None]], 1)
            Vj = v[:, None] if Vx is None else torch.cat([Vx[sl], v[:, None]], 1)
            hi = int(st[sl].max()) - cache.c0 + 1
            mask = torch.arange(cache.c0, cache.c0 + hi)[None, :] <= st[sl][:, None]
            Ro, lg = m.rest(R, mixed, inj, pos, cache.K[:hi], cache.V[:hi], mask, Kj, Vj)
            di, pi = m.pick(lg)
            Rn.append(Ro)
            ids.append(di)
            prob.append(pi)
            kx.append(Kj)
            vx.append(Vj)
        R_prev, d_prev = torch.cat(Rn), torch.cat(ids)
        Kx, Vx = torch.cat(kx), torch.cat(vx)
        outs.append((d_prev.clone(), torch.cat(prob)))
    return outs
