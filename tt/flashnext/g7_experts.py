"""The routed experts of a prefill chunk on the card, in the GGUF's native codes (gexpq's card form, G2), and
their card form cached on the SSD.

Card form (build(), $TT_MOEOFFLOAD_DATA/w51_experts/<model tag>/blk.<il>.p<j>.<part>.npy, raw tile bytes in pieces of 32
experts, joined per layer when loaded):
  gu_c   gate|up [K 2560 x N 1280] as code tiles [expert][40 n-tiles][10 code tiles] uint32: nibble t of word [k][n]
         = the code of k-tile 8 c + t (IQ3_S repacked exactly: f = 2 c - 15; layer 2's IQ4_XS: the IQ4_NL table)
  gu_s   the scales compact, a page per column tile [expert][40]: the GGUF blocks' fp16 d [10][32 columns], then the
         int8 m [80 k-blocks][32] (scale = d m, exact: IQ3_S a / 2 = d (1 + 2 s), IQ4_XS d (ls - 32)), 3,200 bytes;
         the reader rebuilds the fp32 scale tiles (row r of tile q = k-block 32 q + r's scale per column) from them
  dn_c   down [640 (padded to 768) x 2560] IQ4_NL code tiles [expert][80][3]; Q8_0 layers (4, 30, 46, 47, 2): dn_v,
         its int8 values as bfp8_b tiles [expert][80][24] (exact)
  dn_s   down's scales compact: fp16 d [20 k-blocks][32 columns] a column tile (scale = d), 1,280 bytes
A layer is 1.46 GB (2.03 with Q8_0 down) against 1.15 GB as stored in the GGUF; the fp32 scale tiles took 1.76 GB
(the cache w51_experts, before 2026-11-07).
The shapes above are Flash-Next's (FLASHNEXT); a Shape carries another model's (2026-11-13): GLM's
(tt/glm/g6_experts.py: D 4096, FF 2048, 288 experts, top 8) has an IQ4_XS down whose compact pages hold d per 256
weights and an m per 32 as gate|up's (HAS_M on the down), and no padded k-tiles (FF a multiple of 256).

CardExperts per layer: the routes (the top 10 of the router's softmax, renormalized: ttnn.topk on the fp32 logits,
ids and values read back) grouped by expert on the host into units of <= MT_MAX token tiles; the tokens gathered into
expert order (ttnn.embedding); the grouped matmuls (tt/kernels/pf_gexp_*.cpp: one program for gate|up with SwiGLU
folded in, one for down whose writer puts each output row in its route's row of a token-ordered buffer), each unit
reading its expert from the layer's tensors (resident in card memory, or streamed into a slot over PCIe on the second
command queue, ahead of use); the routes' weighted sum per token in fp32 (pf_combine_*.cpp).

  python g7_experts.py build        (no card: any container with the tt env; --workers N)
"""
from __future__ import annotations

import functools
import glob
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "formats")))
sys.path.insert(0, HERE)
from fused.host_rows import HostOut as _HostOut, HostRows  # noqa: E402,F401  (HostRows re-exported)

# the compact scales (m as bfp8_b rows); w51_experts: the fp32 scale tiles, w51_experts_c: m as int8 (2026-11-07)
ECACHE = os.path.join(DATA, "w51_experts_c2")
# caches whose files build() links instead of writing when the bytes are the same (the code tiles; the down's d)
PREV_CACHES = [os.path.join(DATA, c) for c in ("w51_experts", "w51_experts_c")]
KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
PE = 32                                      # experts per piece
MIN_FREE = 20 << 30                          # build() stops below this much free SSD
# Flash-Next's layer's tensors in NG groups of experts (pieces joined): 16 (one piece each, 64 ~28 MB copies a layer)
# took an 8K prompt from 0 at a 64K build 6.02 -> 5.53 s against 4 groups' 16 copies, bit for bit (2026-11-07, sens)
NG = int(os.environ.get("W51_EXPERT_GROUPS", 16))
# the expert kernels' fidelity (scope 2, 2026-11-08: HiFi2 ~1.7x g2_matmul's rate; not bitwise, the parity sweep decides)
EXP_FID = os.environ.get("W51_EXP_FID", "HiFi3")
# W51_W16=1 (probe): the decoded weights packed bf16 (cb 26) instead of fp32: half the pack / unpack bytes of the units'
# W materialization, the floor of step 0 (not bitwise: 7 mantissa bits against the FPU's tf32 10)
W16 = os.environ.get("W51_W16", "0") == "1"
READ_BATCH = int(os.environ.get("W51_GEXP_BATCH", 1))   # pf_gexp_reader.cpp: code tiles read under one barrier
PAD_SKIP = os.environ.get("W51_PAD_SKIP", "1") == "1"   # the down's padded k-tiles skipped (pf_gexp_compute.cpp kr)
MT_MAX = int(os.environ.get("W51_MT", 5))     # token tiles per unit (cb 1 holds MT_MAX x 80 bf16 tiles for gate|up)
# the down's units (its x is h, 24 fp32 k-tiles a token tile: 768 KB of L1 at 8; Dest holds 8 fp32 tiles), 2026-11-09
MT_DN = int(os.environ.get("W51_MT_DN", 8))
U_MAX = 48                                   # units per core: runtime args padded to it (generic_op's program cache
U_CLASSES = (8, 16, 32, U_MAX)          # the runtime args' padding: the smallest class holding the busiest core
                                             # keeps the first call's argument counts)


class Shape:
    """A model's routed experts as the card form and the kernels see them (2026-11-13: was module constants,
    Flash-Next's). KP: the down's K padded to whole code tiles (8 k-tiles); GU / DN: the two programs' tile counts
    (rows: the k-blocks with a scale, per: k-blocks per fp16 d (the GGUF block), m: an int8 m per k-block); gu_q /
    dn_q: the GGUF types the card form holds (gu_q per layer, a dict over the GGUF's own; a Q8_0 down is found by its
    dn_v files); mt / mt_dn: token tiles per unit (None: the module's MT_MAX / MT_DN, which g7_prefill's flags set);
    layers: the layers with a card form; cache: the card form's folder (the model's tag below it); swiglu_limit:
    llama.cpp's swiglu_clamp_exp on gate|up (GLM's 10; None: plain SwiGLU, Flash-Next's)."""

    def __init__(self, D: int, FF: int, NX: int, TOPK: int, layers, folder, cache: str, ng: int, dn_m: bool = False,
                 gu_q: dict | None = None, dn_q: str = "IQ4_NL", mt: int | None = None, mt_dn: int | None = None,
                 swiglu_limit: float | None = None):
        self.D, self.FF, self.NX, self.TOPK = D, FF, NX, TOPK
        self.KP = -(-FF // 256) * 256
        self.NPIECE, self.NG = NX // PE, ng
        assert NX % PE == 0 and self.NPIECE % ng == 0, (NX, ng)
        self.GE = NX // ng                   # experts per group
        self.GU = dict(K=D, N=2 * FF, kt=D // 32, nt=2 * FF // 32, kc=D // 256, kq=-(-(D // 32) // 32), rows=D // 32,
                       per=8, m=True)
        self.DN = dict(K=FF, N=D, kt=self.KP // 32, nt=D // 32, kc=self.KP // 256, kq=-(-(self.KP // 32) // 32),
                       rows=FF // 32, per=8 if dn_m else 1, m=dn_m)
        self.layers, self.folder, self.cache = list(layers), folder, cache
        self.gu_q, self.dn_q, self.mt, self.mt_dn = gu_q or {}, dn_q, mt, mt_dn
        self.swiglu_limit = swiglu_limit

    def gu_type(self, il: int) -> str:
        from expert_types import expert_types
        return self.gu_q.get(il) or expert_types(self.model_dir())[il][0]

    def model_dir(self) -> str:
        return self.folder() if callable(self.folder) else self.folder

    def out_dir(self) -> str:
        return os.path.join(self.cache, tag_of(self.model_dir()))


def model_dir() -> str:
    from g4_model import SSD_MODEL
    return SSD_MODEL


# Flash-Next (Qwen3.8-Flash-Next): gate|up and down types from the GGUF (expert_types; the validated UD-IQ4_XS: layer 2's
# gate|up IQ4_XS, the rest IQ3_S; down IQ4_NL, Q8_0 on 2, 4, 30, 46, 47)
FLASHNEXT = Shape(2560, 640, 512, 10, range(48), model_dir, ECACHE, NG)
DEFINES = {"IQ3_S": ["DECODE_LINEAR"], "IQ4_NL": ["DECODE_IQ4NL", "IQ4_MAGIC", "NOMASK", "PAIR"],
           "IQ4_XS": ["DECODE_IQ4NL", "IQ4_MAGIC", "NOMASK", "PAIR"], "Q8_0": ["VALUES"]}


@functools.lru_cache(maxsize=None)    # reads every GGUF header (seconds a call); Shape.out_dir() asks it often
def tag_of(folder: str) -> str:
    """g3_hc.model_tag's: the tensor count and bytes (without opening the Gguf class's dequantizer)."""
    from gguf import GGUFReader
    n, b = 0, 0
    for f in sorted(glob.glob(os.path.join(folder, "*.gguf"))):
        for t in GGUFReader(f).tensors:
            n += 1
            b += int(t.n_bytes)
    return f"t{n}-b{b}"


def code_tiles(codes: np.ndarray, a: np.ndarray, K: int, kpad: int) -> tuple[np.ndarray, np.ndarray]:
    """codes uint8 [E, N, K], per-block scales a [E, N, K / 32] -> words [E, nt, kc, 32, 32] uint32 (word [k][n]'s
    nibble t = k-tile 8 c + t's code), scales [E, nt, kq, 32, 32] fp32 (row r of tile q = k-tile 32 q + r)."""
    E, N, _ = codes.shape
    nt, kt = N // 32, kpad // 32
    kc, kq = kt // 8, -(-kt // 32)
    c = np.zeros((E, N, kpad), np.uint8)
    c[:, :, :K] = codes
    cv = c.reshape(E, nt, 32, kc, 8, 32)                                       # [e][nt][nl][c][t][kl]
    words = np.zeros((E, nt, kc, 32, 32), np.uint32)
    for t in range(8):
        words |= cv[:, :, :, :, t, :].transpose(0, 1, 3, 4, 2).astype(np.uint32) << np.uint32(4 * t)
    s = np.zeros((E, N, kq * 32), np.float32)
    s[:, :, :K // 32] = a
    return words, s.reshape(E, nt, 32, kq, 32).transpose(0, 1, 3, 4, 2)


def compact_pages(d: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    """Scales d [E, N, nd] float16 (and m [E, N, nk] int8) -> a page per column tile [E nt, 64 nd (+ 34 nk)] uint8:
    d [nd][32 columns]; then m as bfp8_b rows: the exponents [nk][2] (columns 0-15, 16-31), the sign-magnitude
    mantissas [nk][32] (pf_gexp_writer.cpp copies them into row 0 of a bfp8_b tile; exact for |m| <= 127)."""
    E, N, nd = d.shape
    pg = [np.ascontiguousarray(d.reshape(E, N // 32, 32, nd).transpose(0, 1, 3, 2)).view(np.uint8).reshape(-1, 64 * nd)]
    if m is not None:
        v = m.reshape(E, N // 32, 32, -1).transpose(0, 1, 3, 2).astype(np.int32)   # [E][nt][k-block][32 columns]
        g = np.abs(v).reshape(*v.shape[:3], 2, 16)                                # 16-column groups
        mx = g.max(-1)
        e = np.where(mx > 0, np.floor(np.log2(np.maximum(mx, 1))), 0).astype(np.int32)
        mag = g * (1 << (6 - e))[..., None]                                       # |m| / 2^(e - 6), exact
        assert mag.max() <= 127 and np.array_equal(mag // (1 << (6 - e))[..., None], g), "m not exact in bfp8_b"
        mant = (mag | ((v.reshape(g.shape) < 0) << 7)).astype(np.uint8)
        exp = np.where(mx > 0, e + 127, 0).astype(np.uint8)
        pg += [exp.reshape(E * (N // 32), -1), mant.reshape(E * (N // 32), -1)]
    return np.ascontiguousarray(np.concatenate(pg, 1))


def tiles32(a: np.ndarray) -> np.ndarray:
    """[..., 32, 32] 4-byte values -> tile bytes in the card's face order, one row a tile: [n, 1024]."""
    return np.ascontiguousarray(a.reshape(-1, 2, 16, 2, 16).transpose(0, 1, 3, 2, 4).reshape(-1, 1024))


def bfp8_tiles(v: np.ndarray) -> np.ndarray:
    """[..., 32, 32] float32 small integers (|v| <= 127, exact) -> bfp8_b tiles [n, 1088] uint8: per face row of 16
    a shared exponent (the largest's), then sign | 7-bit mantissa with the explicit leading bit, faces in order
    (tt-metal's blockfloat packer: 64 exponent bytes, then 1024 mantissa bytes)."""
    f = v.reshape(-1, 2, 16, 2, 16).transpose(0, 1, 3, 2, 4).reshape(-1, 4, 16, 16)
    a = np.abs(f)
    mx = a.max(-1)
    e = np.floor(np.log2(np.where(mx > 0, mx, 1))).astype(np.int32)
    m = np.rint(a / (2.0 ** (e - 6))[..., None]).astype(np.int32)
    assert m.max() <= 127 and np.array_equal(m * (2.0 ** (e - 6))[..., None], a), "not exact in bfp8_b"
    byte = (m | ((f < 0).astype(np.int32) << 7)).astype(np.uint8)
    exp = np.where(mx > 0, e + 127, 0).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([exp.reshape(-1, 64), byte.reshape(-1, 1024)], 1))


def q5k_as_q8(raw: np.ndarray, E: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Q5_K matrix [E, N, K] (GLM's down on layers 3-6, 41-45) as int8 values with GLM's down scale pages: fp16 d per
    256 weights and an int8 m per 32 (scale = d m, the IQ4_XS pages' layout), read by pf_gexp's VALUES + HAS_M.
    shortcut: re-quantized, not exact (Q5_K's w = d sc q - dmin m has an offset per 32 the kernel does not add):
    llama.cpp's quantize_row_q8_0 rounding with the scale rounded up to d m; exact would need the offset term.
    -> d [E, N, K / 256] fp16, m [E, N, K / 32] int8 (1..127), values [E, N, K] float32 integers in -127..127"""
    from gguf import GGMLQuantizationType as QT
    from gguf.quants import dequantize
    b = dequantize(raw, QT.Q5_K).astype(np.float32).reshape(E, N, K // 256, 8, 32)
    a = np.abs(b).max(-1)                                              # [E, N, K / 256, 8] each 32's amax
    want = a.max(-1) / (127 * 127)                                     # d m >= amax / 127 with m <= 127
    d = want.astype(np.float16)
    d = np.where(d.astype(np.float32) < want, np.nextafter(d, np.float16(np.inf)), d)
    df = d.astype(np.float32)[..., None]
    m = np.clip(np.ceil(a / (127 * np.where(df > 0, df, 1))), 0, 127)
    s = (df * m)[..., None]
    v = np.clip(np.rint(b / np.where(s > 0, s, 1)), -127, 127)
    return d, m.astype(np.int8).reshape(E, N, K // 32), v.reshape(E, N, K)


def piece(job) -> str:
    """One piece of one layer's experts into the cache (host only, no ttnn): raw tile bytes per part, .npy."""
    il, j, folder, out, S = job
    D, FF, KP = S.D, S.FF, S.KP
    import repack as rp
    from gguf import GGMLQuantizationType as QT
    from gguf import GGUFReader
    t = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.gguf"))):
        for x in GGUFReader(f).tensors:
            if x.name in (f"blk.{il}.ffn_gate_exps.weight", f"blk.{il}.ffn_up_exps.weight",
                          f"blk.{il}.ffn_down_exps.weight"):
                t[x.name.split(".")[2]] = x
    e0 = j * PE

    def gu_part(x):
        raw = np.asarray(x.data[e0:e0 + PE])
        r = rp.repack(x.tensor_type, raw)
        a = r.a / (2.0 if x.tensor_type == QT.IQ3_S else 1.0)                 # IQ3_S: w = (a / 2)(2c - 15)
        d, m = rp.scale_factors(x.tensor_type, raw)
        assert np.array_equal(rp.dm_bits(d[:, None], m), a.astype(np.float32).reshape(m.shape).view(np.uint32))
        return r.codes.reshape(PE, FF, D), d.reshape(PE, FF, D // 256), m.reshape(PE, FF, D // 32)
    cg, dg, mg = gu_part(t["ffn_gate_exps"])
    cu, du, mu = gu_part(t["ffn_up_exps"])
    words, _ = code_tiles(np.concatenate([cg, cu], 1), np.zeros((PE, 2 * FF, D // 32), np.float32), D, D)
    parts = {"gu_c": tiles32(words), "gu_s": compact_pages(np.concatenate([dg, du], 1), np.concatenate([mg, mu], 1))}
    dn = t["ffn_down_exps"]
    raw = np.asarray(dn.data[e0:e0 + PE])
    if dn.tensor_type in (QT.Q8_0, QT.Q5_K):
        m = None
        if dn.tensor_type == QT.Q8_0:
            blk = raw.reshape(-1, 34)
            d = blk[:, :2].copy().view(np.float16).reshape(PE, D, FF // 32)
            v = blk[:, 2:].copy().view(np.int8).astype(np.float32).reshape(PE, D, FF)
        else:
            assert S.DN["m"], "a Q5_K down: GLM's (HAS_M) pages only"
            d, m, v = q5k_as_q8(raw, PE, D, FF)
        assert np.isfinite(d).all()
        vals = np.zeros((PE, D, KP), np.float32)
        vals[:, :, :FF] = v
        parts["dn_v"] = bfp8_tiles(vals.reshape(PE, D // 32, 32, KP // 32, 32).transpose(0, 1, 3, 4, 2))  # [e][nt][kt]
    else:
        r = rp.repack(dn.tensor_type, raw)
        d, m = rp.scale_factors(dn.tensor_type, raw)
        assert (m is not None) == S.DN["m"], (dn.tensor_type, S.DN["m"])
        assert np.array_equal(rp.dm_bits(d if m is None else d[:, None], m),
                              r.a.astype(np.float32).reshape(-1 if m is None else m.shape).view(np.uint32))
        w, _ = code_tiles(r.codes.reshape(PE, D, FF), r.a.reshape(PE, D, FF // 32), FF, KP)
        parts["dn_c"] = tiles32(w)
        # IQ4_XS (GLM): d per 256 weights with an m per 32, as gate|up's; IQ4_NL: d per 32
        d = d.reshape(PE, D, FF // (256 if m is not None else 32))
        m = None if m is None else m.reshape(PE, D, FF // 32)
    parts["dn_s"] = compact_pages(d, m)
    for k, a in parts.items():
        path = os.path.join(out, f"blk.{il}.p{j}.{k}.npy")
        prevs = [os.path.join(c, os.path.basename(out), os.path.basename(path)) for c in PREV_CACHES]
        prev = next((q for q in prevs if os.path.exists(q) and np.load(q, mmap_mode="r").shape == a.shape and
                     np.array_equal(np.load(q, mmap_mode="r"), a)), None)
        if prev is not None:                       # the same bytes as an earlier cache's: one copy on disk
            if not os.path.exists(path):
                os.link(prev, path)
            continue
        tmp = f"{path[:-4]}.{os.getpid()}.tmp.npy"
        np.save(tmp, a)
        os.replace(tmp, path)
    return f"blk.{il}.p{j}"


def build(workers: int, S: Shape = FLASHNEXT) -> None:
    from multiprocessing import Pool
    folder = S.model_dir()
    out = S.out_dir()
    os.makedirs(out, exist_ok=True)
    jobs = [(il, j, folder, out, S) for il in S.layers for j in range(S.NPIECE)
            if not os.path.exists(os.path.join(out, f"blk.{il}.p{j}.dn_s.npy"))]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, name in enumerate(pool.imap_unordered(piece, jobs)):
            if shutil.disk_usage(out).free < MIN_FREE:   # the SSD also holds the OS: stop, a rerun resumes
                pool.terminate()
                sys.exit(f"stopped after {i + 1} of {len(jobs)} pieces: under {MIN_FREE >> 30} GiB free on the SSD")
            if i % 32 == 31:
                print(f"{i + 1} of {len(jobs)} pieces ({time.time() - t0:.0f} s)", flush=True)
    print(f"{len(jobs)} pieces in {time.time() - t0:.0f} s -> {out}")
    for old in PREV_CACHES if S is FLASHNEXT else []:   # complete: the older layouts were link sources only
        if os.path.isdir(old):
            print(f"removing the older layout {old}", flush=True)
            shutil.rmtree(old)
    root = os.path.dirname(out)
    others = [t for t in os.listdir(root) if t != os.path.basename(out)]
    if others:                  # another GGUF's (an older quant, or a second staged model): only a person knows
        print(f"other model tags in {root}: {', '.join(others)} (delete by hand when unused)", flush=True)


# ------------------------------------------------------------------------------------------------- the card side

def split_lpt(costs: list[float], bins: int) -> list[list[int]]:
    """Items to bins, largest first onto the least loaded (an LPT schedule)."""
    import heapq
    h = [(0.0, b) for b in range(bins)]
    out = [[] for _ in range(bins)]
    for i in sorted(range(len(costs)), key=lambda i: -costs[i]):
        load, b = heapq.heappop(h)
        out[b].append(i)
        heapq.heappush(h, (load + costs[i], b))
    return out


def host_tensor(a: np.ndarray):
    """Raw tile bytes [n, page] (uint32, fp32 or uint8) as a row-major ttnn host tensor: a row is a page."""
    import ttnn
    if a.dtype == np.uint32:
        return ttnn.from_torch(torch.from_numpy(a.view(np.int32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    dt = {np.dtype(np.float32): ttnn.float32, np.dtype(np.uint8): ttnn.uint8}[a.dtype]
    return ttnn.from_torch(torch.from_numpy(a), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)


def pad64(a: np.ndarray) -> np.ndarray:
    """Rows (pages) padded to a multiple of 64 bytes, the DRAM page alignment (gate|up's compact page is 3,360)."""
    w = a.shape[1] * a.itemsize
    return a if w % 64 == 0 else np.pad(a, ((0, 0), (0, (64 - w % 64) // a.itemsize)))


def load_joined(paths: list[str]) -> np.ndarray:
    """The .npy pieces' arrays (one dtype and row shape) joined by rows, each read straight into its rows of one buffer:
    the bytes np.concatenate([np.load(p) ...]) gives, without its second copy of the layer."""
    heads = []
    for p in paths:
        with open(p, "rb") as f:
            v = np.lib.format.read_magic(f)
            shape, fortran, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else
                                  np.lib.format.read_array_header_2_0)(f)
            assert not fortran, p
            heads.append((p, shape, dt, f.tell()))
    out = np.empty((sum(h[1][0] for h in heads),) + tuple(heads[0][1][1:]), heads[0][2])
    r = 0
    for p, shape, dt, off in heads:
        assert dt == out.dtype and tuple(shape[1:]) == out.shape[1:], (p, shape, dt)
        with open(p, "rb") as f:
            f.seek(off)
            view = memoryview(out[r:r + shape[0]]).cast("B")
            assert f.readinto(view) == view.nbytes, p
        r += shape[0]
    return out


class HostOut(_HostOut):
    """fused/host_rows.py's (shared with GLM since 2026-11-13), at Flash-Next's width by default"""

    def __init__(self, dev, d: int = FLASHNEXT.D):
        super().__init__(dev, d)


class Layer:
    """A layer's card form: its tensors, the cache's pieces joined into NG groups of GE experts (one tensor per part
    and group: a layer streams in 4 NG copies), host
    copies and the card tensors they are in now. A streamed layer copies only the groups the card computes."""

    def __init__(self, folder: str, il: int, S: Shape):
        self.il = il
        self.q8 = os.path.exists(os.path.join(folder, f"blk.{il}.p0.dn_v.npy"))
        self.parts = ("gu_c", "gu_s", "dn_v" if self.q8 else "dn_c", "dn_s")
        self.gu_q, self.dn_q = S.gu_type(il), "Q8_0" if self.q8 else S.dn_q
        pg = S.NPIECE // S.NG
        self.host = [{k: host_tensor(pad64(load_joined([os.path.join(folder, f"blk.{il}.p{j}.{k}.npy")
                                                        for j in range(g * pg, (g + 1) * pg)]))) for k in self.parts}
                     for g in range(S.NG)]
        self.card = None                     # [{part: device tensor}] when resident


class CardExperts:
    """The routed experts of a chunk on the card. resident: the layers whose card form stays in card memory; the
    others are streamed into `slots` sets of tensors (each holding a layer of either form) on command queue 1, ahead
    of use."""

    def __init__(self, dev, resident: list[int], slots: int = 2, folder: str | None = None, log=print,
                 n_layers: int = 48, shape: Shape = FLASHNEXT, slot_groups: int | None = None,
                 q8_groups: int | None = None):
        """n_layers: the shape's layers below it (Flash-Next's checks build fewer); slot_groups: the groups a stream
        slot holds (default all: the card may take a whole layer; fewer when card memory is short, the rest of a
        streamed layer then the lane's); q8_groups: of those, the groups with room for a Q8_0 down (dn_v, 2.2x
        dn_c's bytes), at most a Q8_0-down layer's card share (default all; GLM's re-quantized Q5_K downs: fewer)"""
        import ttnn
        self.ttnn = ttnn
        self.dev = dev
        self.S = S = shape
        folder = folder or S.out_dir()
        t0 = time.time()
        from concurrent.futures import ThreadPoolExecutor   # file reads and the bulk copies release the GIL
        with ThreadPoolExecutor(int(os.environ.get("W51_LOAD_THREADS", 16))) as ex:
            ils = [il for il in S.layers if il < n_layers]
            self.layers = dict(zip(ils, ex.map(lambda il: Layer(folder, il, S), ils)))   # il -> Layer, in order
        self.n_layers = n_layers
        log(f"experts' card form: {len(self.layers)} layers' host tensors in {time.time() - t0:.0f} s")
        t0 = time.time()
        self.resident = set(resident)
        for il in resident:
            L = self.layers[il]
            L.card = [{k: ttnn.to_device(h, dev) for k, h in p.items()} for p in L.host]
        self.slot_tensors = []
        streamed = [L for L in self.layers.values() if L.il not in self.resident]
        self.slot_groups = min(S.NG, slot_groups or S.NG)
        self.q8_groups = self.slot_groups if q8_groups is None else min(self.slot_groups, q8_groups)
        # a slot holds either form: gu_c, gu_s, dn_s (the same shapes in both; layer 2's IQ4_XS gate|up as IQ3_S's)
        # and both dn_c (IQ4_NL) and dn_v (Q8_0) when layers of each form stream; a layer fills its own parts
        proto = {}
        for L in streamed:
            for k, h in L.host[0].items():
                assert proto.setdefault(k, h).shape == h.shape, (L.il, k, h.shape, proto[k].shape)
        for _ in range(slots if streamed else 0):
            self.slot_tensors.append([{k: ttnn.allocate_tensor_on_device(h.shape, h.dtype, h.layout, dev,
                                                                         ttnn.DRAM_MEMORY_CONFIG)
                                       for k, h in proto.items() if k != "dn_v" or g < self.q8_groups}
                                      for g in range(self.slot_groups)])   # a set per group
        self.events = {}                     # queue entry -> its copy's event (CQ 1)
        # every streamed layer copied once now: a host buffer's first copy pins it (~10 GB/s), later ones are the
        # card's DMA from pinned memory (28.5 GB/s) while tt-metal's pinned-memory cache holds them all
        # (TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES above the streamed bytes)
        for L in streamed:
            for p, h in zip(self.slot_tensors[0], L.host):
                for k, t in h.items():
                    if k in p:                # a Q8_0 down's groups past q8_groups: never the card's
                        ttnn.copy_host_to_device_tensor(t, p[k], cq_id=ttnn.QueueId(1))
        ttnn.synchronize_device(dev)
        log(f"experts: {len(resident)} layers resident, {len(streamed)} streamed through {slots} slots, in "
            f"{time.time() - t0:.0f} s")
        self.seconds = dict(route=0.0, group=0.0, stream=0.0, launch=0.0)
        self.tick = lambda phase: None        # the breakdown's synchronized clock (Prefill.run sets it)
        # timing-only variants (g7_prefill's sens mode; wrong outputs): nostream (no weight copies), nodecode, noscale,
        # nomatmul (that part of pf_gexp_compute skipped), noscatter (the down's rows written as whole tiles), norows
        # (the writer's scale rows pushed unconverted)
        self.skip = set()
        self.balance = {}                     # kind -> [sum of the busiest core's cost, sum of the mean core's]
        self.cur = (None, None)              # the layer whose weights the last call used, and them
        self.lane = None                     # SplitExperts: the CPU lane, for the experts past card_groups
        self.card_groups = S.NG              # a streamed layer's groups on the card (set before plan())
        self.n_groups = S.NG
        self.fenced = None                   # the threads fence() moved, while a prompt shares the lane
        self.cpu_s = dict(read=0.0, lane=0.0, wait=0.0, write=0.0)
        self.folder = folder
        self.cache = {}                      # il -> (on the card [NX] bool, slot of an expert [NX], host groups, card
        self.use_cache = False               # groups or None): set_cache's; use_cache: a call takes the cached experts

    # ---- the short reads' cache (2026-11-10): experts kept in card memory, computed on the card beside the lane
    def set_cache(self, slots: int, counts: dict) -> None:
        """The `slots` (layer, expert) pairs of most prompt routes (counts: il -> [NX], census.json's prompt counts)
        in card form, a layer's in groups of GE experts as its streamed groups (the same kernels read them: a unit's
        (group, expert in group) is (slot // GE, slot % GE)); the host groups kept for load_cache."""
        S = self.S
        GE, NX, NPIECE = S.GE, S.NX, S.NPIECE
        ranked = sorted(((int(c), il, e) for il, v in counts.items() for e, c in enumerate(v)
                         if il in self.layers and il not in self.resident), reverse=True)[:slots]
        share = {}
        for _, il, e in ranked:
            share[il] = share.get(il, 0) + 1
        by = {}                              # each layer's share rounded to whole groups (a part group is padding)
        for il, k in share.items():
            k = int(round(k / GE)) * GE
            if k:
                by[il] = list(np.argsort(-np.asarray(counts[il]), kind="stable")[:k])
        self.drop_cache()
        self.cache, self.cache_bytes = {}, 0
        for il, es in by.items():
            es = sorted(es)
            L = self.layers[il]
            src = {k: [np.load(os.path.join(self.folder, f"blk.{il}.p{j}.{k}.npy"), mmap_mode="r")
                       for j in range(NPIECE)] for k in L.parts}
            groups = []
            for g0 in range(0, len(es), GE):
                part = {}
                for k, pieces in src.items():
                    per = pieces[0].shape[0] // PE                     # an expert's rows (pages) in a piece
                    rows = [pieces[e // PE][(e % PE) * per:(e % PE + 1) * per] for e in es[g0:g0 + GE]]
                    a = np.concatenate(rows + [np.zeros((per * (GE - len(rows)),) + rows[0].shape[1:], rows[0].dtype)])
                    part[k] = host_tensor(pad64(a))
                    self.cache_bytes += pad64(a).nbytes
                groups.append(part)
            on, slot = np.zeros(NX, bool), np.zeros(NX, np.int32)
            on[es], slot[es] = True, np.arange(len(es), dtype=np.int32)
            self.cache[il] = [on, slot, groups, None]
        self.load_cache()

    def load_cache(self) -> None:
        for c in self.cache.values():
            if c[3] is None:
                c[3] = [{k: self.ttnn.to_device(h, self.dev) for k, h in g.items()} for g in c[2]]

    def drop_cache(self) -> None:
        for c in self.cache.values():
            for g in c[3] or []:
                for t in g.values():
                    self.ttnn.deallocate(t)
            c[3] = None

    def takes(self, il: int) -> bool:
        """Whether the card computes any of layer il's experts (a card-form layer, a share on the card)."""
        return il in self.layers and (self.lane is None or self.limit(il) > 0)

    def limit(self, il: int) -> int:
        """Layer il's experts below this are the card's, the rest the lane's (resident layers: all the card's)."""
        if il in self.resident:
            return self.S.NX
        assert self.lane is not None or self.slot_groups == self.S.NG, "a slot holds part of a layer: the lane's needed"
        assert self.lane is not None or not self.layers[il].q8 or self.q8_groups == self.S.NG, "Q8_0 part: lane"
        cap = self.q8_groups if self.layers[il].q8 else self.slot_groups
        return self.S.NX if self.lane is None else min(self.card_groups, cap) * self.S.GE

    # ---- streaming
    def plan(self) -> None:
        """A prompt's start: the streamed layers in order of use (once each: Prefill.run goes layer by layer)."""
        self.fence()
        self.queue = [il for il in self.layers if il not in self.resident]
        self.cur = (None, None)
        self.next_copy = 0                   # the next queue entry to copy
        self.in_slot = [None] * len(self.slot_tensors)   # queue index held by each slot
        self.use = 0                         # the queue entry the next streamed layer's use takes
        self.use_done = 0                    # the queue entries before this one are done on the card

    def pump(self, most: int = 1) -> None:
        """Copy up to `most` of the next queued layers into free slots (a slot is free when its layer's use was
        launched and the host has since synchronized past it: a routing read). Called before a layer's routing
        read, when the card has that layer's dense work queued: the copy (the host blocks ~70 ms a layer) overlaps
        it. On this thread: a copier thread lost (g7_copythread.py: tt-metal's device lock holds queue 0's ops for
        each copy, and the GIL starves the copier between copies; 8K 6.35 s at 4 groups, 8.2 s at 16)."""
        ttnn = self.ttnn
        while self.next_copy < len(self.queue) and most > 0:
            most -= 1
            free = [s for s, q in enumerate(self.in_slot) if q is None or q < self.use_done]
            if not free:
                return
            s = free[0]
            il = self.queue[self.next_copy]
            t0 = time.perf_counter()
            for p, h in list(zip(self.slot_tensors[s], self.layers[il].host))[:self.limit(il) // self.S.GE]:
                for k, t in h.items():
                    if "nostream" not in self.skip:
                        ttnn.copy_host_to_device_tensor(t, p[k], cq_id=ttnn.QueueId(1))
            self.events[self.next_copy] = ttnn.record_event(self.dev, ttnn.QueueId(1))
            self.seconds["stream"] += time.perf_counter() - t0
            self.in_slot[s] = self.next_copy
            self.next_copy += 1

    def weights(self, il: int):
        """The card tensors of layer il's pieces (waiting, on queue 0, for its copy when streamed)."""
        if il in self.resident:
            return self.layers[il].card
        qi = self.use
        assert self.queue[qi] == il, (qi, il)
        while qi >= self.next_copy:
            self.pump()
        s = self.in_slot.index(qi)
        self.ttnn.wait_for_event(self.ttnn.QueueId(0), self.events.pop(qi))
        self.use += 1
        return self.slot_tensors[s]

    # ---- one layer
    def route(self, logits, n: int):
        """The top 10 of the router's softmax for the first n rows (cl_route's rule): the card's top 10 by fp32 logit
        (ttnn.topk on the fp32 logits; only ids and values cross to the host); ids [n, 10] int32, weights [n, 10]
        fp32."""
        ttnn = self.ttnn
        T = logits.shape[-2]
        NX, TOPK = self.S.NX, self.S.TOPK
        vals, idx = ttnn.topk(ttnn.slice(logits, [0, 0, 0, 0], [1, 1, T, NX]), k=128, dim=-1)
        top = lambda t: ttnn.typecast(ttnn.slice(t, [0, 0, 0, 0], [1, 1, T, TOPK]), ttnn.float32)  # noqa: E731
        both = ttnn.to_torch(ttnn.concat([top(vals), top(idx)], dim=-1)).reshape(T, 2 * TOPK)[:n].numpy()  # one read
        v, ids = both[:, :TOPK], both[:, TOPK:].astype(np.int32)
        w = np.exp(v - v[:, :1])
        return ids, (w / w.sum(1, keepdims=True)).astype(np.float32)

    def combine(self, y, w, T: int, skip: bool = False):
        """The routes' weighted sum per token, fp32 [1, 1, T, 2560]: y [1, 1, 10 T, 2560] (route j's outputs in rows
        j T ..), w row-major [T, 32] (token t's 10 weights); tt/kernels/pf_combine_*.cpp, hcmix_writer.cpp."""
        ttnn = self.ttnn
        D, TOPK = self.S.D, self.S.TOPK
        rt, nt = T // 32, D // 32
        g = self.dev.compute_with_storage_grid_size()
        cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, D]), ttnn.float32, ttnn.TILE_LAYOUT, self.dev,
                                             ttnn.DRAM_MEMORY_CONFIG)
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        per, extra = divmod(rt * nt, len(cores))
        f = 0
        for k, (cx, cy) in enumerate(cores):
            n = per + (1 if k < extra else 0)
            rrt[cx][cy] = [y.buffer_address(), w.buffer_address(), f, n]
            crt[cx][cy] = [f, n]
            wrt[cx][cy] = [out.buffer_address(), f, n]
            f += n
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[0] = mode[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_combine_reader.cpp"), core_ranges=grid,
                                  compile_time_args=[rt, nt, TOPK] + acc(y) + acc(w), runtime_args=rrt,
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "hcmix_writer.cpp"), core_ranges=grid,
                                  compile_time_args=[1, nt] + acc(out), runtime_args=wrt,   # rt 1: page = item
                                  config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_combine_compute.cpp"), core_ranges=grid,
                                  compile_time_args=[nt, TOPK], runtime_args=crt, config=cfg,
                                  defines=[("SKIP", "1")] if skip else [])]
        cbs = [ttnn.CBDescriptor(total_size=4096 * n, core_ranges=grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)])
            for i, n in ((0, 2 * TOPK), (2, TOPK), (3, 1), (16, 2))]
        ttnn.generic_op([y, w, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def program(self, kind: str, qtype: str, x, y, units, wt, dest=None):
        t0 = time.perf_counter()
        try:
            return self._program(kind, qtype, x, y, units, wt, dest, t0)
        finally:
            self.seconds["program"] = self.seconds.get("program", 0.0) + time.perf_counter() - t0

    def _program(self, kind: str, qtype: str, x, y, units, wt, dest, t0):
        """The grouped matmul over `units` [(piece, expert in piece, first tile, tiles)], cores by LPT; with dest
        (uint32 [R / 32, 32]) each output row goes to y's row dest[row] (pf_gexp_writer.cpp's SCATTER)."""
        ttnn = self.ttnn
        S = self.S
        P = S.GU if kind == "gu" else S.DN
        kt, nt, kc, kq = P["kt"], P["nt"], P["kc"], P["kq"]
        mt_max = (S.mt or MT_MAX) if kind == "gu" else (S.mt_dn or MT_DN)
        g = self.dev.compute_with_storage_grid_size()
        cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        bins = split_lpt([2.0 + u[3] for u in units], len(cores))
        loads = [sum(2.0 + units[i][3] for i in b) for b in bins]
        bal = self.balance.setdefault(kind, [0.0, 0.0])
        bal[0] += max(loads)
        bal[1] += sum(loads) / len(cores)
        cpart, spart = ("gu_c", "gu_s") if kind == "gu" else (("dn_v" if qtype == "Q8_0" else "dn_c"), "dn_s")
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        # the pieces' addresses once (a buffer_address() per unit was most of the ~3 ms a call), and the runtime args
        # padded to the smallest of U_CLASSES holding the busiest core (the cached program a class: the kernels read
        # only their n_units entries; always U_MAX was ~5x the args to convert and to dispatch), 2026-11-09
        ca = [p[cpart].buffer_address() if cpart in p else 0 for p in wt]   # a group past q8_groups: no unit reads it
        sa = [p[spart].buffer_address() for p in wt]
        xa, ya, da = x.buffer_address(), y.buffer_address(), dest.buffer_address() if dest is not None else 0
        per = [[] if "nounits" in self.skip else [units[i] for i in sorted(b)] for b in bins]   # (timing: launch only)
        um = next(c for c in U_CLASSES if c >= max(len(m) for m in per))
        for (cx, cy), mine in zip(cores, per):
            r, w = [xa, len(mine)], [ya, da, len(mine)]
            for p, e, x0, mt in mine:
                r += [ca[p], sa[p], e, x0, mt]
                w += [x0, mt, sa[p], e]
            pad = um - len(mine)
            rrt[cx][cy] = r + [0] * (5 * pad)
            crt[cx][cy] = [len(mine)] + [u[3] for u in mine] + [0] * pad
            wrt[cx][cy] = w + [0] * (4 * pad)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        w0 = wt[0][cpart]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, EXP_FID), fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        mode[0] = ttnn.UnpackToDestMode.UnpackToDestFp32      # the code words, copied to Dest exactly
        cfg.unpack_to_dest_mode = mode
        swiglu = kind == "gu"                  # gate|up's output is h = silu(gate) * up, [R, 768] (pad zero)
        skips = [f"SKIP_{v.upper()}" for v in ("decode", "scale", "matmul", "rows", "x", "codes", "scales", "out", "bcast", "pack", "rowpush")
                 if "no" + v in self.skip]
        if dest is not None and "noscatter" in self.skip:
            skips.append("SKIP_SCATTER")
        # the code tiles' batch: a divisor of kc (a batch never wraps cb 0's 2 batches), at most W51_GEXP_BATCH
        rb = max(b for b in range(1, READ_BATCH + 1) if kc % b == 0)
        defines = [(d, "1") for d in DEFINES[qtype] + skips + (["SCATTER"] if dest is not None else []) +
                   (["SWIGLU"] if swiglu else []) + (["HAS_M"] if P["m"] else [])] + (
                   [("READ_BATCH", str(rb))] if rb != 1 else []) + [("U_PAD", str(um))] + (   # a cached program a class
                   [("SWIGLU_LIMIT", f"0x{np.float32(S.swiglu_limit).view(np.uint32):08x}u")]
                   if swiglu and S.swiglu_limit is not None else []) + (
                   [("TT_MOEOFFLOAD_ZONES", "1")] if os.environ.get("W51_ZONES") == "1" else [])
        qpage = -(-(64 * (P["rows"] // P["per"]) + (34 * P["rows"] if P["m"] else 0)) // 64) * 64  # pad64's page
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_gexp_reader.cpp"), core_ranges=grid,
                                  compile_time_args=[kt, nt, mt_max, kc, kq] + acc(w0) + acc(x), runtime_args=rrt,
                                  defines=defines, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_gexp_writer.cpp"), core_ranges=grid,
                                  compile_time_args=([S.FF // 32, S.KP // 32] if swiglu else [nt, nt]) + acc(y) +
                                  (acc(dest) if dest is not None else []) + [nt, kc, P["rows"], P["per"]] +
                                  acc(wt[0][spart]),
                                  runtime_args=wrt, defines=defines,
                                  config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "pf_gexp_compute.cpp"), core_ranges=grid,
                                  compile_time_args=[kt, nt, mt_max, kc, P["K"] // 32 if PAD_SKIP else kt],
                                  defines=defines, runtime_args=crt,
                                  config=cfg),
        ]
        xs = 4096 if x.dtype == ttnn.float32 else 2048            # gate|up reads bf16 hn, down fp32 h
        # cb 25 for Q8_0's bfp8_b values; cb 21 / 22: the writer's m rows (bfp8_b) and d rows (fp32), 8 each (two
        # passes of 4 k-tiles); cb 27: the writer's compact page
        spec = [(27, ttnn.uint8, qpage, 1), (0, ttnn.uint32, 4096, 2 * rb), (1, x.dtype, xs, mt_max * kt),
                (22, ttnn.float32, 4096, 8)] + [(21, ttnn.bfloat8_b, 1088, 8)] * P["m"] + [
                (25, ttnn.bfloat8_b, 1088, 16)] * (qtype == "Q8_0") + [
                (26, ttnn.bfloat16 if W16 else ttnn.float32, 2048 if W16 else 4096, kt),
                (16, ttnn.float32, 4096, 2 * mt_max), (17, ttnn.uint32, 128, mt_max)] + (
                    [(18, ttnn.float32, 4096, mt_max), (19, ttnn.float32, 4096, mt_max),
                     (20, ttnn.float32, 4096, 1)] if swiglu else [])
        cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=size)]) for i, f, size, n in spec]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ins = [x, y] + [p[k] for p in wt for k in (cpart, spart) if k in p] + ([dest] if dest is not None else [])
        self.seconds["program: build"] = self.seconds.get("program: build", 0.0) + time.perf_counter() - t0
        ttnn.generic_op(ins, prog)

    def __call__(self, il: int, hb, logits, n: int):
        """Flash-Next's: the routes by route() (the card's top 10 of the router's logits), then routed()."""
        ttnn = self.ttnn
        T = hb.shape[-2]
        if not (self.use_cache and il in self.cache):
            self.pump()
        self.tick("stream (host copies)")
        t0 = time.perf_counter()
        table = ttnn.to_layout(ttnn.reshape(hb, [T, self.S.D]), ttnn.ROW_MAJOR_LAYOUT)   # before the route's sync
        self.tick("gather in: hn row-major")
        ids, w = self.route(logits, T)          # synchronizes: everything before this layer's experts is done
        return self.routed(il, hb, ids, w, n, table=table, t0=t0)

    def routed(self, il: int, hb, ids, w, n: int, table=None, t0=None, x=None):
        """Layer il's routed experts for the chunk hb (bf16 [1, 1, T, D]) given its routes (ids int32 / w fp32
        [T, TOPK], the padding rows' too: finite, unused) -> fp32 [1, 1, T, D]. Called after a host sync past the
        layer's dense work (a routing read: route(), or GLM's routing on the lane). x: hn's first n rows fp32 on the
        host when the caller has them (the lane's share reads them otherwise)."""
        ttnn = self.ttnn
        S = self.S
        NX, TOPK, D, GE = S.NX, S.TOPK, S.D, S.GE
        T = hb.shape[-2]
        cached = self.cache.get(il) if self.use_cache else None
        if t0 is None:
            t0 = time.perf_counter()
        if table is None:
            table = ttnn.to_layout(ttnn.reshape(hb, [T, D]), ttnn.ROW_MAJOR_LAYOUT)
            self.tick("gather in: hn row-major")
        # the streamed layers used so far have finished, but this one when an earlier chunk took its slot
        self.use_done = self.use - (1 if self.cur[0] == il and il not in self.resident else 0)
        self.tick("route")
        lim = self.limit(il)
        # the card's experts: below lim (its streamed groups), or the cached ones (their slots for the units)
        on = cached[0] if cached is not None else (np.arange(NX) < lim if lim < NX else None)
        if self.use_cache and cached is None:
            on = np.zeros(NX, bool)              # a layer without cached experts: all the lane's
        lane = self.lane_start(il, hb, ids, w, n, on, x) if on is not None else None
        t1 = time.perf_counter()
        # group (every row, the padding's too: their sums are finite and unused): the card's routes in expert order,
        # each expert's rows from a tile boundary, units of <= MT_MAX tiles; per row its token (the gather) and its
        # route's row j * T + t in the down's output (dest; the padding rows 0xFFFFFFFF)
        # (int32 throughout and the rows by np.repeat: 16K tokens' grouping 8.6 -> ~3 ms, the card idle meanwhile)
        flat = ids.reshape(-1).astype(np.int32)
        sel = np.flatnonzero(on[flat]).astype(np.int32) if on is not None else None   # the card's routes (flat indices)
        fe = flat if sel is None else flat[sel]
        if cached is not None:
            fe = cached[1][fe]                   # expert -> its slot in the layer's cache groups
        if sel is not None and sel.size == 0:    # nothing on the card: the lane's share alone
            ttnn.deallocate(table)
            return self.lane_end(lane, T)
        order = np.argsort(fe.astype(np.int16), kind="stable").astype(np.int32)  # (a radix sort for 16-bit keys)
        counts = np.bincount(fe, minlength=NX)
        nt_e = -(-counts // 32)
        tile0 = np.cumsum(nt_e) - nt_e
        first = np.cumsum(counts) - counts
        # a route's row: its expert's first tile's, plus its place among the expert's routes (fe[order] is sorted)
        row = np.repeat((tile0 * 32 - first).astype(np.int32), counts) + np.arange(fe.size, dtype=np.int32)
        if sel is not None:
            order = sel[order]
        R = max(1, -(-int(nt_e.sum()) // 128)) * 4096   # rounded up to 4K rows: the row-shaped ops see few shapes
        tok, j = np.divmod(order, np.int32(TOPK))
        gidx = np.zeros(R, np.int32)
        gidx[row] = tok
        # the gather first (it needs only gidx): dest, the units and the stream's wait while the card gathers
        up = lambda a: ttnn.from_torch(torch.from_numpy(a.view(np.int32)), dtype=ttnn.uint32,  # noqa: E731
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
        gdev = up(gidx.reshape(1, R))
        self.tick("gather in: upload")
        xg = ttnn.reshape(ttnn.embedding(gdev, table, layout=ttnn.TILE_LAYOUT), [1, 1, R, D])
        ttnn.deallocate(table)
        self.tick("gather in: embedding")
        dest = np.full(R, 0xFFFFFFFF, np.uint32)
        dest[row] = j * np.int32(T) + tok

        def cut(mt):                               # units of <= mt tiles: (piece, expert in piece, first tile, tiles)
            used = np.flatnonzero(counts)
            nu = -(-nt_e[used] // mt)
            e = np.repeat(used, nu)
            m0 = (np.arange(int(nu.sum())) - np.repeat(np.cumsum(nu) - nu, nu)) * mt
            return list(zip((e // GE).tolist(), (e % GE).tolist(), (tile0[e] + m0).tolist(),
                            np.minimum(mt, nt_e[e] - m0).tolist()))
        units = cut(S.mt or MT_MAX)
        self.tick("group (host)")
        t2 = time.perf_counter()
        if cached is not None:
            wts = cached[3]
        else:
            if self.cur[0] != il:              # a layer's chunks come one after another (Prefill.run)
                self.cur = (il, self.weights(il))
            wts = self.cur[1]
        self.tick("wait for the stream")
        L = self.layers[il]
        alloc = lambda shape: ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32,  # noqa: E731
                                                             ttnn.TILE_LAYOUT, self.dev, ttnn.DRAM_MEMORY_CONFIG)
        # SwiGLU's h [R, KP]: its writer fills the FF columns and zeroes the padding tiles of the units' rows (the rows
        # the down reads), so a bare allocation per call (a zero fill of it took ~10 s a 4K prompt; a cached one,
        # made after the prefill captured its traces, could sit in their temporaries)
        h = ttnn.zeros([1, 1, R, S.KP], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.dev) if getattr(
            self, "zero_fill", False) else alloc([1, 1, R, S.KP])          # (zero_fill: g7_prefill's regress mode)
        self.program("gu", L.gu_q, xg, h, units, wts)
        ttnn.deallocate(xg)
        self.tick("gate|up + swiglu")
        y = alloc([1, 1, TOPK * T, D])          # route j's outputs in rows j * T .. (token order)
        self.program("dn", L.dn_q, h, y, cut(S.mt_dn or MT_DN), wts, dest=up(dest.reshape(R // 32, 32)))
        ttnn.deallocate(h)
        self.tick("down")
        wrow = np.zeros((T, 32), np.float32)   # the weights row-major, a token a row (pf_combine_reader widens them)
        wrow[:, :TOPK] = w if lane is None else np.where(on[ids], w, 0)   # the lane's routes weigh 0 here
        wdev = ttnn.from_torch(torch.from_numpy(wrow), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.dev)
        self.tick("sum of the routes: weights")
        out = self.combine(y, wdev, T, skip=lane is not None)
        ttnn.deallocate(y)
        self.tick("sum of the routes")
        if lane is not None:
            if cached is None:
                self.pump()                      # the next layer's copy while the lane runs
            out = ttnn.add(out, self.lane_end(lane, T))
            self.tick("the lane's share")
        t3 = time.perf_counter()
        self.seconds["route"] += t1 - t0
        self.seconds["group"] += t2 - t1
        self.seconds["launch"] += t3 - t2
        return out


    # ---- the lane's share (SplitExperts)
    def lane_start(self, il: int, hb, ids, w, n: int, on, x=None):
        """Layer il's routes to experts not `on` the card on the CPU lane, in a thread (ctypes lets the GIL go): hn's
        rows read to the host (the route's read synchronized the card) unless given (x), the card's routes skipped
        (id -1)."""
        import threading
        ttnn = self.ttnn
        t0 = time.perf_counter()
        T = hb.shape[-2]
        if not hasattr(self, "rows"):
            self.rows, self.outs = HostRows(self.dev), HostOut(self.dev, self.S.D)
        if x is None:
            x = self.rows(hb, n)
        cid = np.ascontiguousarray(np.where(on[ids[:n]], -1, ids[:n]).astype(np.int32))
        cw = np.ascontiguousarray(w[:n])
        out = self.outs.buffer(T, n)
        self.cpu_s["read"] += time.perf_counter() - t0
        step = getattr(self.lane, "max_m", 8)

        err = []

        def run():
            if self.lane.cpus:                   # the lane's thread 0 is its caller: cl_init's CPU for it
                os.sched_setaffinity(0, {self.lane.cpus[0]})
            t = time.perf_counter()
            try:
                for i in range(0, n, step):
                    j = min(n, i + step)
                    self.lane.moe_m(il, x[i:j], cid[i:j], cw[i:j], out[i:j])
            except BaseException as e:           # re-raised by lane_end on the prefill's thread
                err.append(e)
            self.cpu_s["lane"] += time.perf_counter() - t
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th, out, err

    def fence(self) -> None:
        """For a prompt the lane shares: the process's other threads (the prefill's, tt-metal's, pinned or not: some
        spin) off the lane's CPUs, so none shares a lane worker's CPU while the lane runs beside the card (its phases
        wait for their slowest thread); finish() puts them back."""
        if self.fenced is None and self.lane is not None and self.lane.cpus:
            self.fenced = self.lane.fence_others()

    def finish(self) -> None:
        """A prompt's end (Prefill.run)."""
        if self.fenced is not None:
            self.lane.unfence(self.fenced)
            self.fenced = None

    def lane_end(self, lane, T: int):
        ttnn = self.ttnn
        th, out, err = lane
        t0 = time.perf_counter()
        th.join()
        if err:
            raise err[0]
        t1 = time.perf_counter()
        y = self.outs.upload(T)
        self.cpu_s["wait"] += t1 - t0
        self.cpu_s["write"] += time.perf_counter() - t1
        return y


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
        build(w)
