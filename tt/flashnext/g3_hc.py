"""G3 step 3: Flash-Next's hyper-connection mix (4 residual streams -> the mixer's input) as two fused phases on the
grid (tt/kernels/fhc_*.cpp), real blk.0 hc_attn weights.

  H1  lo = sum_s r_s (X_s W'_s), W' = diag(norm) [down | inject] (10240 -> 324): 44 cores, one (column tile, stream)
      each, r_s = rsqrt(mean X_s^2 + eps) computed on the item's core
  H2  gate = up silu(lo[:320] / 4); hn = 1/4 sum_s sigmoid(gate_s) X_s norm_s r_s: 80 cores, one channel tile each
      (the 4 streams' gate tiles of that channel); w = 2 sigmoid(lo[320:324] / 4) on core 80

Checked against float64 per phase from the card's own previous outputs (X's bf16 copy as the GEMV reads it), on
NaN-filled outputs with fresh inputs, then timed (traced, one launch).

  python3 tt_moeoffload/launch.py --memory-max $((16 << 30)) --engine tt/flashnext/g3_hc.py

Measured 2026-10-23: H1's partials within 5.5e-4, hn 4.5e-4, w 3.5e-8 of float64; 78.2 us traced (with X's load and
broadcast, which the layer program does not need) against 137.0 us for G2's stock ops. Every core reading X from card
memory itself took 174.7 us: 110 reads of the same 60 KB funnel through one or two DRAM banks.
"""
from __future__ import annotations

import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import sys
import time

import numpy as np
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from fused.grid import Grid, traced_us  # noqa: E402
from g2_layers import MODEL, Gguf  # noqa: E402
from g3_gdn import nan_tensor  # noqa: E402

KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
SITE = "blk.0.hc_attn_"
D, HC, LR, LO = 2560, 4, 320, 352
W_PAGE, BATCH, DEPTH = 1088, 10, 4
IDLE = 0xFFFFFFFF
EPS = 1e-6


CACHE = os.path.expanduser(os.environ.get("G4_WEIGHT_CACHE", os.path.join(DATA, "g4_weights")))
LAYOUT = "v1"   # bump when stream()'s tile order changes: every cached entry then misses


def model_tag(g: Gguf) -> str:
    """A model's cache folder name: its tensor count and bytes (the same GGUF files give the same tag)."""
    return f"t{len(g.t)}-b{sum(int(t.n_bytes) for t in g.t.values())}"


def stream(dev, w, back=True, key=None):
    """[k, n] -> bfp8_b tiles n-major (page n * kt + k), and the dequantized weights back as float64 [k, n] (None
    with back=False: the token loop keeps no host copies).

    w is the matrix or a function returning it. With key (model_tag(g) + "/" + a name for the matrix as built,
    e.g. "blk.0.wa"), the converted tensor is kept in CACHE and loaded from there the next time: w is then not
    built (no dequantization), and back's copy is read back from the card."""
    path = os.path.join(CACHE, LAYOUT, key + ".tensorbin") if key else None
    if path and os.path.exists(path):
        t = ttnn.load_tensor(path, device=dev)
        k, n = None, None
    else:
        w = w() if callable(w) else w
        k, n = w.shape
        kt, nt = k // 32, n // 32
        st = w.contiguous().to(torch.bfloat16).reshape(kt, 32, nt, 32).permute(2, 0, 1, 3).reshape(1, 1, -1, 32)
        t = ttnn.from_torch(st, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path[:-len('.tensorbin')]}.{os.getpid()}.tmp.tensorbin"   # dump_tensor wants the suffix
            ttnn.dump_tensor(tmp, t)
            with open(path + ".shape", "w") as f:
                f.write(f"{k} {n}\n")
            os.replace(tmp, path)
    if not back:
        return t, None
    if k is None:
        with open(path + ".shape") as f:
            k, n = (int(v) for v in f.read().split())
    kt, nt = k // 32, n // 32
    return t, ttnn.to_torch(t).double().reshape(nt, kt, 32, 32).permute(1, 2, 0, 3).reshape(k, n)


class HcMix:
    def __init__(self, dev, g: Gguf, rng):
        self.dev = dev
        self.grid = Grid(dev)
        norm, down, up, inject = (g(SITE + x) for x in ("norm.weight", "down.weight", "up.weight", "inject.weight"))
        self.norm = norm                                                       # [10240] (1 + w folded in)
        wd = torch.zeros(HC * D, LO)
        wd[:, :LR + HC] = (torch.cat([down, inject]) * norm[None, :]).T        # W' = diag(norm) [down | inject]
        self.wd, self.wdq = stream(dev, wd)
        self.wu, self.wuq = stream(dev, up.T)                                  # [320, 10240]
        rm = lambda t: ttnn.from_torch(t.float().reshape(1, 1, 1, -1).contiguous(), dtype=ttnn.float32,  # noqa: E731
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.norm_t = rm(norm)
        x = torch.from_numpy(rng.standard_normal(HC * D).astype(np.float32))
        self.x = x.double()
        self.x16 = x.to(torch.bfloat16).double()
        self.x32_t = rm(x)
        self.x16_t = ttnn.from_torch(x.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.chk_p = nan_tensor(dev, (1, 1, 1, HC * LO))
        self.chk_hn = nan_tensor(dev, (1, 1, 1, D))
        self.chk_w = nan_tensor(dev, (1, 1, 1, 16))
        o = {}
        a = 0
        for name, size in (("x32", HC * D * 4), ("x16", HC * D * 2), ("p1", HC * LO * 4), ("rs", 64),
                           ("hn", D * 2), ("w", 16)):
            o[name] = a
            a += size
        self.off, self.arena = o, a

    def program(self):
        g, o = self.grid, self.off
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            item = idx if idx < (LO // 32) * HC else IDLE                      # (n, s) = (idx // 4, idx % 4)
            tile = idx if idx <= D // 32 else IDLE                              # 80: the inject weights
            rrt[cx][cy] = [self.x32_t.buffer_address(), self.x16_t.buffer_address(), self.wd.buffer_address(),
                           self.wu.buffer_address(), self.norm_t.buffer_address()] + g.args(idx) + \
                [o["x32"], o["x16"], o["p1"], o["rs"], o["hn"], o["w"], item, tile]
            crt[cx][cy] = [item, tile]
            wrt[cx][cy] = [self.chk_p.buffer_address(), self.chk_hn.buffer_address(),
                           self.chk_w.buffer_address()] + g.args(idx) + [o["p1"], o["rs"], o["hn"], o["w"], item, tile]
        acc = list(ttnn.TensorAccessorArgs(self.wd).get_compile_time_args())
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (4, 5, 6, 7, 8, 9):
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fhc_reader.cpp"), core_ranges=g.ranges,
                                  compile_time_args=[BATCH] + acc, runtime_args=rrt,
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fhc_writer.cpp"), core_ranges=g.ranges,
                                  compile_time_args=acc, runtime_args=wrt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "fhc_compute.cpp"), core_ranges=g.ranges,
                                  compile_time_args=[BATCH], runtime_args=crt, config=cfg),
        ]
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        spec = [(0, ttnn.bfloat8_b, W_PAGE, BATCH * DEPTH), (1, bf16, 2048, 80), (2, bf16, 2048, 1),
                (3, bf16, 2048, 10), (4, f32, 4096, 3), (5, f32, 4096, 8), (6, f32, 4096, 4), (7, f32, 4096, 4),
                (8, f32, 4096, 4), (9, f32, 4096, 1), (10, f32, 4096, 3), (11, f32, 4096, 1), (17, f32, 4096, 2),
                (18, f32, 4096, 1), (19, bf16, 2048, 1), (20, f32, 4096, 1), (29, ttnn.uint8, 4096, 1),
                (30, ttnn.uint8, self.arena, 1), (31, ttnn.uint8, 512, 1)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        ins = [self.x32_t, self.x16_t, self.wd, self.wu, self.norm_t, self.chk_p, self.chk_hn, self.chk_w]
        return lambda: ttnn.generic_op(ins, prog)

    def errors(self) -> dict:
        rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()  # noqa: E731
        X, X16 = self.x.reshape(HC, D), self.x16.reshape(HC, D)
        r = torch.rsqrt(X.pow(2).mean(-1) + EPS)                                # [4]
        p_ref = torch.stack([r[s] * (X16[s] @ self.wdq[s * D:(s + 1) * D]) for s in range(HC)])   # [4, 352]
        p = ttnn.to_torch(self.chk_p).double().reshape(HC, LO)
        res = {"H1_partials": rel(p[:, :LR + HC], p_ref[:, :LR + HC])}
        lo = p.sum(0)                                                           # H2 from the card's partials
        xg = torch.nn.functional.silu(lo[:LR] / HC).to(torch.bfloat16).double()
        gate = (xg @ self.wuq).reshape(HC, D)
        hn = (torch.sigmoid(gate) * X * self.norm.double().reshape(HC, D) * r[:, None]).mean(0)
        res["H2_hn"] = rel(ttnn.to_torch(self.chk_hn).double().flatten(), hn)
        w = 2 * torch.sigmoid(lo[LR:LR + HC] / HC)
        res["H2_w"] = rel(ttnn.to_torch(self.chk_w).double().flatten()[:HC], w)
        return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(int(time.time()))
    g = Gguf(MODEL)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "site": SITE}
    try:
        m = HcMix(dev, g, rng)
        m.program()()
        ttnn.synchronize_device(dev)
        res["max_rel_err"] = m.errors()
        print("max rel err: " + ", ".join(f"{k} {v:.1e}" for k, v in res["max_rel_err"].items()), flush=True)
        res["traced_us"] = traced_us(dev, m.program())
        print(f"HC mix traced {res['traced_us']:.1f} us ({(m.wd.volume() + m.wu.volume()) * 1.0625 / 1e6:.1f} MB)",
              flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g3_hc.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
