"""GLM's MLA output as one program (tt/kernels/glm_fmo_*.cpp), real blk.3 weights:
the attention's latent outputs o_lat [64 x 512] (two flash-MLA output groups of 32 heads, bf16 tiles) -> v_b per head
(512 -> 256) on 64 cores -> o (16384 -> 4096) split by K in FMO_S parts over all 110 cores, the partials summed on the
cores' RISC-V FPUs into a row-major fp32 [4096] (the mHC site's input).

Checked against float64 on the card's weights (o rounded to bf16 as the card's region B holds it), the output
pre-filled with NaN; timed traced against the stock route of g3_mla_layer.py (v_b's batched matmul + o's
DRAM-sharded one: 671.0 us).

  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/glm/g3_fmo.py
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
TT = os.path.dirname(HERE)
KDIR = os.path.join(TT, "kernels")
sys.path.insert(0, HERE)
sys.path.insert(1, TT)
from g2_layers import MLA_LAYER, MODEL, OUT, Gguf, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
from fused.grid import NOC1_ROWS, Grid, split, split_weighted, traced_us  # noqa: E402
from fused.mrows import mp_of, step_words, unblocks  # noqa: E402

D, NH, KVL, HV = 4096, 64, 512, 256
IDLE, W_PAGE = 0xFFFFFFFF, 1088
BATCH, DEPTH = int(os.environ.get("FMO_BATCH", 8)), int(os.environ.get("FMO_DEPTH", 8))
S = int(os.environ.get("FMO_S", 4))               # o's K split


def stream(dev, w, check=True):
    """[k, n] -> bfp8_b tiles n-major (page n * kt + k), and the dequantized weights back as float64 [k, n] (None
    without check: a whole model's would take ~75 GB)."""
    k, n = w.shape
    kt, nt = k // 32, n // 32
    st = w.contiguous().to(torch.bfloat16).reshape(kt, 32, nt, 32).permute(2, 0, 1, 3).reshape(1, 1, -1, 32)
    t = ttnn.from_torch(st, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    if not check:
        return t, None
    return t, ttnn.to_torch(t).double().reshape(nt, kt, 32, 32).permute(1, 2, 0, 3).reshape(k, n)


class MlaOut:
    """v_b [64, 256, 512] (head, out, in) and o [4096, 16384] as llama.cpp holds them; og0, og1 the flash-MLA output
    groups (bf16 tiles [1, 1, 32, 512]); out the row-major fp32 [1, 1, 1, 4096] it writes."""

    def __init__(self, dev, v_b, wo, og0, og1, out, check=True, m_max=1, step=None):
        """m_max > 1: the M-row build (glm_fmo_reader.cpp): og [1, M_MAX, 32, 512], out mp-row blocks of
        fp32 [M_MAX * 4096], the pass's rows from step (tt/fused/mrows.py)."""
        self.dev, self.og, self.out, self.m_max = dev, (og0, og1), out, m_max
        g = self.grid = Grid(dev)
        vb = [stream(dev, v_b[h].T.contiguous()) for h in range(NH)]                  # [512 k, 256 n] each
        self.vbq = torch.stack([q for _, q in vb]) if check else None
        self.vb = ttnn.from_torch(torch.cat([ttnn.to_torch(t).reshape(-1, 32) for t, _ in vb]).reshape(1, 1, -1, 32),
                                  dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        for t, _ in vb:
            ttnn.deallocate(t)
        self.wo, self.woq = stream(dev, wo.T.contiguous(), check)                       # [16384, 4096]
        speeds = g.speeds(NOC1_ROWS)
        items, sums = split_weighted(128 * S, speeds), split(128, len(g))
        self.rx, self.rb = 0, 1024 * m_max
        self.rp = self.rb + NH * HV * 2 * m_max
        arena = self.rp + (S * 128 * 128 if m_max == 1 else S * max(n for _, n in sums) * m_max * 128)
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            head = idx if idx < NH else IDLE
            (i0, ni), (s0, ns) = items[idx], sums[idx]
            rargs[(cx, cy)] = g.args(idx) + [og0.buffer_address(), og1.buffer_address(), self.vb.buffer_address(),
                                             self.wo.buffer_address(), out.buffer_address(), head, i0, ni, s0, ns,
                                             self.rx, self.rb, self.rp]
            wargs[(cx, cy)] = g.args(idx) + [head, i0, ni, self.rb, self.rp]
            if m_max > 1:   # E2's partial (tile n, part) to the owner of tile n (fused::ksplit_sum_m)
                rargs[(cx, cy)] += [step.buffer_address()]
                wargs[(cx, cy)] += [step.buffer_address()]
                for it in range(i0, i0 + ni):
                    n, part = divmod(it, S)
                    o = next(j for j, (a0, na) in enumerate(sums) if a0 <= n < a0 + na)
                    a0, na = sums[o]
                    wargs[(cx, cy)] += [g.noc[o].x | g.noc[o].y << 16, part * na + n - a0]
            crt[cx][cy] = [head, ni]
        acc = list(ttnn.TensorAccessorArgs(self.wo).get_compile_time_args())
        mdef = [("M_MAX", str(m_max))] if m_max > 1 else []
        kernels = g.dm_kernels(os.path.join(KDIR, "glm_fmo_reader.cpp"), os.path.join(KDIR, "glm_fmo_writer.cpp"),
                               [BATCH, DEPTH, S] + acc, [S] + (acc if m_max > 1 else []), rargs, wargs, defines=mdef)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 64
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_fmo_compute.cpp"),
                                             core_ranges=g.ranges, compile_time_args=[BATCH, S], runtime_args=crt,
                                             config=cfg))
        spec = [(0, ttnn.bfloat8_b, W_PAGE, BATCH * DEPTH), (1, ttnn.bfloat16, 2048, 2 * BATCH),
                (2, ttnn.uint8, 16384 * m_max, 1),
                (16, ttnn.bfloat16, 2048, 8), (17, ttnn.float32, 4096, 2), (30, ttnn.uint8, arena, 1),
                (31, ttnn.uint8, 1024 * m_max, 1)]
        self.cb_kb = sum(s * n for _, _, s, n in spec) / 1024
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        self.ios = [og0, og1, self.vb, self.wo, out] + ([step] if m_max > 1 else [])

    def __call__(self):
        ttnn.generic_op(self.ios, self.prog)

    def reference(self, o_lat: torch.Tensor) -> torch.Tensor:
        """float64 from o_lat [64, 512] (bf16 values), o rounded to bf16 as region B holds it."""
        o = torch.einsum("hk,hkn->hn", o_lat.double(), self.vbq).to(torch.bfloat16).double()
        return o.flatten() @ self.woq


def main_m() -> None:
    """The MLA output at M rows (FMO_M = M_MAX), every row against float64, timed at M = 1..M_MAX."""
    g = Gguf(MODEL)
    p = f"blk.{MLA_LAYER}."
    v_b, wo = g(p + "attn_v_b.weight"), g(p + "attn_output.weight")
    mm = M_MAX
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": MLA_LAYER, "S": S, "m_max": mm, "rows": {}}
    RMj = ttnn.ROW_MAJOR_LAYOUT
    try:
        rng = np.random.default_rng(int(time.time()))
        og = [ttnn.from_torch(torch.zeros(1, mm, 32, KVL), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
              for _ in range(2)]
        out = ttnn.from_torch(torch.full((1, 1, 1, D * mm), float("nan")), dtype=ttnn.float32, layout=RMj, device=dev)
        step = ttnn.from_torch(step_words(1), dtype=ttnn.uint32, layout=RMj, device=dev)
        m = MlaOut(dev, v_b, wo, og[0], og[1], out, m_max=mm, step=step)
        for M in range(1, mm + 1):
            o_lat = torch.from_numpy(rng.standard_normal((mm, NH, KVL)).astype(np.float32)).to(torch.bfloat16)
            for i, t in enumerate(og):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(o_lat[:, 32 * i:32 * i + 32].reshape(1, mm, 32, KVL),
                                                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), t)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(M), dtype=ttnn.uint32, layout=RMj), step)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full((1, 1, 1, D * mm), float("nan")),
                                                            dtype=ttnn.float32, layout=RMj), out)
            m()
            ttnn.synchronize_device(dev)
            got = unblocks(ttnn.to_torch(out).double().flatten(), D // 32, mp_of(M))[:M]
            want = torch.stack([m.reference(o_lat[r]) for r in range(M)])
            e = {"max_rel_err": float((got - want).abs().max() / want.abs().max()),
                 "traced_us": round(traced_us(dev, m), 1)}
            res["rows"][M] = e
            print(f"M {M}:", json.dumps(e), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_fmo_m{mm}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


M_MAX = int(os.environ.get("FMO_M", 1))


def main() -> None:
    if M_MAX > 1:
        return main_m()
    g = Gguf(MODEL)
    p = f"blk.{MLA_LAYER}."
    v_b, wo = g(p + "attn_v_b.weight"), g(p + "attn_output.weight")                 # [64, 256, 512], [4096, 16384]
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": MLA_LAYER, "S": S}
    try:
        rng = np.random.default_rng(int(time.time()))
        o_lat = torch.from_numpy(rng.standard_normal((NH, KVL)).astype(np.float32)).to(torch.bfloat16)
        og = [ttnn.from_torch(o_lat[i:i + 32].reshape(1, 1, 32, KVL), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=dev) for i in (0, 32)]
        out = ttnn.from_torch(torch.full((1, 1, 1, D), float("nan")), dtype=ttnn.float32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        m = MlaOut(dev, v_b, wo, og[0], og[1], out)
        m()
        ttnn.synchronize_device(dev)
        got, want = ttnn.to_torch(out).double().flatten(), m.reference(o_lat)
        res["check"] = {"max_rel_err": float((got - want).abs().max() / want.abs().max()), "pcc": round(pcc(got, want), 7)}
        print("check:", json.dumps(res["check"]), f"(circular buffers {m.cb_kb:.0f} KB a core)", flush=True)
        res["traced_us"] = round(traced_us(dev, m), 1)
        mb = (m.vb.volume() + m.wo.volume()) * 1.0625 / 1e6
        print(f"S {S}: traced {res['traced_us']} us ({mb:.1f} MB of weights; the stock v_b + o 671.0 us)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, "g3_fmo.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
