"""GLM's KDA decode step on the card (tt/kernels/glm_kda_*.cpp): Flash-Next's fused delta
rule with KDA's per-key decay, one value head per core on 64 cores, the state exact in fp32 on the SFPU.

Inputs as the mixer's projections would give them (random, the real shapes): q l2-normalized and scaled 1/sqrt(128), k
l2-normalized, v, sigmoid(g2) in (0, 1), the decays in (exp(-5), 1) (GLM's lower-bounded gate), beta in (0, 1), the
state; the output norm's weight is blk.4's real ssm_norm. Checked against float64 (G2's mixer_ref recurrence and
llama.cpp's KDA output: RMSNorm(o) eps 1e-5 times sigmoid(g2)): the outputs and the new state. Timed traced, against
ttnn's GDN op route (g2_kda_probe.py: 147.7 us).

  python3 tt_moeoffload/launch.py --memory-max $((16 << 30)) --engine tt/glm/g3_kda.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.dirname(HERE)
KDIR = os.path.join(TT, "kernels")
sys.path.insert(0, HERE)
sys.path.insert(1, TT)
from g2_layers import HD, KDA_LB, MODEL, NH, OUT, RMS_EPS, Gguf, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
from fused.grid import Grid, traced_us  # noqa: E402

IDLE = 0xFFFFFFFF


class KdaStep:
    """The KDA step's program (glm_kda_*.cpp) over fixed tensors: S (fp32 tiles [NH, HD, HD]), rows (row-major fp32
    [5 NH, HD]: q, k, v, sigmoid(g2), d), par (row-major fp32 [2, HD]: the output norm's weight, beta), S_out (S itself
    is fine: each core reads its head's whole state before writing any), out (row-major fp32 [NH, HD])."""

    def __init__(self, dev, S, rows, par, S_out, out):
        g = Grid(dev)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            h = idx if idx < NH else IDLE
            rargs[(cx, cy)] = [S.buffer_address(), rows.buffer_address(), par.buffer_address(), h]
            wargs[(cx, cy)] = [S_out.buffer_address(), out.buffer_address(), h]
            crt[cx][cy] = [0 if h == IDLE else 1]
        kernels = g.dm_kernels(os.path.join(KDIR, "glm_kda_reader.cpp"), os.path.join(KDIR, "glm_kda_writer.cpp"),
                               acc(S) + acc(rows) + acc(par), acc(S_out) + acc(out), rargs, wargs)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        mode = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (6, 8, 10, 13, 14, 24, 26):           # exact: loaded straight into Dest in fp32 (fgdn's, + Dbc)
            mode[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = mode
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_kda_compute.cpp"),
                                             core_ranges=g.ranges, compile_time_args=[], runtime_args=crt,
                                             config=cfg))
        f32, bf = ttnn.float32, ttnn.bfloat16
        spec = [(2, bf, 2048, 1), (5, f32, 4096, 4), (6, f32, 4096, 1), (7, f32, 4096, 4), (8, f32, 4096, 16),
                (9, f32, 4096, 16), (10, f32, 4096, 4), (11, f32, 4096, 4), (12, f32, 4096, 4), (13, f32, 4096, 10),
                (14, f32, 4096, 4), (15, f32, 4096, 4), (19, f32, 4096, 8), (20, f32, 4096, 16),
                (22, f32, 4096, 4), (23, f32, 4096, 4), (24, f32, 4096, 4), (25, f32, 4096, 4), (26, f32, 4096, 4),
                (29, ttnn.uint8, 4096, 1), (31, ttnn.uint8, 1024, 1)]
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[g.cb(*x) for x in spec])
        self.ios = list(dict.fromkeys([S, rows, par, S_out, out]))

    def __call__(self):
        ttnn.generic_op(self.ios, self.prog)


def main() -> None:
    gen = torch.Generator().manual_seed(10)
    l2 = lambda t: t / t.norm(dim=-1, keepdim=True)  # noqa: E731
    q = l2(torch.randn(NH, HD, generator=gen)) * HD ** -0.5
    k = l2(torch.randn(NH, HD, generator=gen))
    v = torch.randn(NH, HD, generator=gen)
    sg = torch.rand(NH, HD, generator=gen)
    d = torch.exp(torch.rand(NH, HD, generator=gen) * KDA_LB)
    beta = torch.rand(NH, generator=gen)
    S0 = torch.randn(NH, HD, HD, generator=gen) * 0.05
    w = Gguf(MODEL)("blk.4.ssm_norm.weight")
    S = S0.double() * d.double()[:, :, None]
    delta = (v.double() - torch.einsum("hk,hkv->hv", k.double(), S)) * beta.double()[:, None]
    S_ref = S + k.double()[:, :, None] * delta[:, None, :]
    o = torch.einsum("hk,hkv->hv", q.double(), S_ref)
    out_ref = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w.double() * sg.double()

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M")}
    try:
        tile = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
        rm = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,  # noqa
                                       device=dev)
        S_dev, S_out = tile(S0.reshape(1, NH, HD, HD)), tile(torch.zeros(1, NH, HD, HD))
        rows = rm(torch.cat([q, k, v, sg, d]).reshape(1, 1, 5 * NH, HD))
        par = torch.zeros(2, HD)
        par[0], par[1, :NH] = w, beta
        par_dev, out_dev = rm(par.reshape(1, 1, 2, HD)), rm(torch.zeros(1, 1, NH, HD))
        run = KdaStep(dev, S_dev, rows, par_dev, S_out, out_dev)
        run()
        ttnn.synchronize_device(dev)
        out_c = ttnn.to_torch(out_dev).double().reshape(NH, HD)
        S_c = ttnn.to_torch(S_out).double().reshape(NH, HD, HD)
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
        res["check"] = {"out_max_rel_err": rel(out_c, out_ref), "out_pcc": round(pcc(out_c, out_ref), 7),
                        "state_max_rel_err": rel(S_c, S_ref), "state_pcc": round(pcc(S_c, S_ref), 7)}
        print("check:", json.dumps(res["check"]), flush=True)
        res["traced_us"] = round(traced_us(dev, run, reps=34, tokens=10), 1)
        print(f"KDA step, 64 heads: {res['traced_us']} us traced (ttnn's route 147.7)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, "g3_kda.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "g3_kda.json"))


if __name__ == "__main__":
    main()
