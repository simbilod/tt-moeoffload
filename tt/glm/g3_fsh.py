"""GLM's FFN half (MoE layers) as one program (tt/kernels/glm_fsh_*.cpp): hn goes to the
host at once, which routes in fp32 on the lane's threads (cpulane cl_glm_route: GLM's sigmoid + bias top 8, the
weights normalized x 2.5) and runs the lane, while the card runs the shared expert (gate|up with GLM's SwiGLU clamp on
64 cores, down split by K over all 110); the lane's sum comes back at the last barrier and each core adds it to its
output tiles straight into the mHC site's input. It replaces g3_kda_layer.FfnHalf's tile conversions, the card's bf16
router, the page building and the sum after the reply.

Checked here without the lane: the host replies with a random vector, the output against float64 (the card's
weights, h rounded to bf16 as region H holds it) plus that vector; cl_glm_route against G2's moe_ref routing on the
real router (blk.4). Timed traced with the host replying at once (the card's own time). FSH_DENSE=1: blk.0's dense
FFN (FF 12288, no lane, the same program) against float64 alone.

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((24 << 30)) --engine tt/glm/g3_fsh.py
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
os.environ.setdefault("CL_LIB", os.path.join(TT, "cpulane", "build", "libcpulane_glm.so"))
sys.path.insert(0, HERE)
sys.path.insert(1, TT)
sys.path.append(os.path.join(TT, "flashnext"))                      # u1_sockets only
from g2_layers import KDA_LAYER, MODEL, OUT, Gguf, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
from fused.grid import NOC1_ROWS, Grid, split, split_weighted  # noqa: E402
from fused.mrows import blocks, mp_of, pad_rows, step_words, unblocks  # noqa: E402
from g3_fmo import stream  # noqa: E402
from u1_sockets import pinned  # noqa: E402
import cpulane  # noqa: E402

D, FF, NT = 4096, 2048, 128
IDLE, W_PAGE = 0xFFFFFFFF, 1088
BATCH, DEPTH = 16, 4
S = int(os.environ.get("FSH_S", 2))                # down's K split
REPS = int(os.environ.get("FSH_REPS", 50))
DENSE = int(os.environ.get("FSH_DENSE", 0))        # 1: blk.0's dense FFN (FF 12288, no lane) instead


class FfnFused:
    """gate, up [FF, 4096], down [4096, FF] as llama.cpp holds them: the shared expert (FF 2048) of a MoE layer, whose
    routed experts run on the lane (lane=True), or a dense layer's FFN (FF 12288, lane=False); hn the input (row-major
    bf16 [1, 1, 1, 4096]); out the row-major fp32 [1, 1, 1, 4096] it writes. With the lane, a socket pair on core
    (0, 0), its own or link's (another FfnFused's (h2d, d2h)): build it before any trace capture. serve(fn) is the
    host's side: fn(hn bf16 as uint16 [4096], reply fp32 [4096]). check: keep the float64 copies (reference()).
    pf: (an L1 prefetch buffer, pages, the next layer's mixer): with the lane, after R2 each core reads the mixer's
    first pages for that core (mixer.a_ranges) into the buffer while the lane works."""

    def __init__(self, dev, gate, up, down, hn, out, lane=True, link=None, check=True, s=None, pf=None, m_max=1,
                 step=None):
        """m_max > 1: the M-row build (glm_fsh_reader.cpp): hn and out as mp-row blocks of m_max rows of room,
        the host's page and reply blocks too (experts.cpp's cl_glm_exchange_m), M from step (the step tensor)."""
        self.dev, self.hn, self.out, self.lane, self.m_max = dev, hn, out, lane, m_max
        ff = self.ff = gate.shape[0]
        S_ = self.S = s or (S if ff == FF else 4)                         # KD / S a multiple of BATCH
        assert (ff // 32) % (S_ * BATCH) == 0
        self.h2d = self.d2h = None
        if lane and link is None:
            where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
            h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 2 * m_max * D * 4,
                                                ttnn.H2DMode.DEVICE_PULL))
            h2d.set_page_size(m_max * D * 4)
            d2h = pinned(lambda: ttnn.D2HSocket(dev, where, 2 * m_max * D * 2))
            d2h.set_page_size(m_max * D * 2)
            link = (h2d, d2h)
        if lane:
            self.h2d, self.d2h = link
        gu = torch.stack([gate.T.reshape(D, ff // 32, 32), up.T.reshape(D, ff // 32, 32)], dim=2)   # (k, j, g|u, 32)
        self.wgu, self.wguq = stream(dev, gu.reshape(D, 2 * ff), check)   # column tiles g0, u0, g1, u1, ..
        self.wd, self.wdq = stream(dev, down.T.contiguous(), check)      # [FF, 4096]
        g = self.grid = Grid(dev)
        speeds = g.speeds(NOC1_ROWS)
        items, sums = split_weighted(NT * S_, speeds), split(NT, len(g))
        units = ([(i, 1) if i < ff // 32 else (0, 0) for i in range(len(g))] if ff // 32 <= len(g)
                 else split_weighted(ff // 32, speeds))
        self.rx, self.rh = 0, m_max * D * 2
        self.rp = self.rh + m_max * ff * 2
        arena = self.rp + (S_ + 1) * NT * 128
        if m_max > 1:   # the partials by owner (S x its tiles), then RT: the routed rows' blocks
            self.rt = self.rp + S_ * max(n for _, n in sums) * m_max * 128
            arena = self.rt + m_max * D * 4
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, (cx, cy) in enumerate(g.cores):
            (u0, nu), (i0, ni), (s0, ns) = units[idx], items[idx], sums[idx]
            rargs[(cx, cy)] = g.args(idx) + [hn.buffer_address(), self.wgu.buffer_address(), self.wd.buffer_address(),
                                             out.buffer_address(), u0, nu, i0, ni, s0, ns, self.rx, self.rh, self.rp] + (
                [step.buffer_address(), self.rt] if m_max > 1 else [])
            if pf is not None and lane:
                buf, n, nxt = pf
                rg = nxt.a_ranges(idx, n)
                assert len(rg) <= 4
                rargs[(cx, cy)] += [buf.buffer_address(), nxt.a_src.buffer_address(), len(rg)] + \
                    [v for r in rg for v in r]
            else:
                rargs[(cx, cy)] += [0, 0, 0]
            wargs[(cx, cy)] = g.args(idx) + [u0, nu, i0, ni, self.rh, self.rp]
            if m_max > 1:   # R2's partial (tile n, part) to the owner of tile n (fused::ksplit_sum_m)
                wargs[(cx, cy)] += [step.buffer_address()]
                for it in range(i0, i0 + ni):
                    n, part = divmod(it, S_)
                    o = next(j for j, (a0, na) in enumerate(sums) if a0 <= n < a0 + na)
                    a0, na = sums[o]
                    wargs[(cx, cy)] += [g.noc[o].x | g.noc[o].y << 16, part * na + n - a0]
            crt[cx][cy] = [nu, ni]
        acc = list(ttnn.TensorAccessorArgs(self.wd).get_compile_time_args())
        defines = [("FSH_LANE", "1"), ("H2D_CONFIG", str(self.h2d.get_config_buffer_address())),
                   ("D2H_CONFIG", str(self.d2h.get_config_buffer_address()))] if lane else []
        if m_max > 1:
            defines = defines + [("M_MAX", str(m_max))]
        kernels = g.dm_kernels(os.path.join(KDIR, "glm_fsh_reader.cpp"), os.path.join(KDIR, "glm_fsh_writer.cpp"),
                               [BATCH, DEPTH, S_, ff // 32] + acc, [S_] + (acc if m_max > 1 else []), rargs, wargs,
                               defines=defines)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 64
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_fsh_compute.cpp"),
                                             core_ranges=g.ranges, compile_time_args=[BATCH, S_, ff // 32],
                                             runtime_args=crt,
                                             config=cfg))
        spec = [(0, ttnn.bfloat8_b, W_PAGE, BATCH * DEPTH), (1, ttnn.bfloat16, 2048, 2 * BATCH),
                (16, ttnn.bfloat16, 2048, 1), (17, ttnn.float32, 4096, 2), (30, ttnn.uint8, arena, 1),
                (31, ttnn.uint8, 1024 * m_max, 1)]
        self.cb_kb = sum(s * n for _, _, s, n in spec) / 1024
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=[g.cb(*x) for x in spec])
        self.ios = [hn, self.wgu, self.wd, out] + ([pf[0], pf[2].a_src] if pf is not None and lane else []) + (
            [step] if m_max > 1 else [])
        self.page = torch.zeros(m_max * D * 2, dtype=torch.uint8)
        self.reply = torch.zeros(m_max * D * 4, dtype=torch.uint8)
        self.page_np, self.reply_np = self.page.numpy(), self.reply.numpy()

    def __call__(self):
        ttnn.generic_op(self.ios, self.prog)

    def serve(self, fn):
        self.d2h.read_tensor(self.page)
        fn(self.page_np.view(np.uint16), self.reply_np.view(np.float32))
        self.h2d.write_tensor(self.reply)

    def reference(self, hn: torch.Tensor) -> torch.Tensor:
        """The shared expert in float64 on the card's weights, h rounded to bf16 as region H holds it."""
        gu = (hn.double() @ self.wguq).reshape(self.ff // 32, 2, 32)
        g, u = gu[:, 0].flatten().clamp(max=10.0), gu[:, 1].flatten().clamp(-10.0, 10.0)
        h = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).double()
        return h @ self.wdq


def main_m() -> None:
    """The FFN half at M rows (FSH_M = M_MAX; FSH_DENSE as main's), every row against float64, the host
    side a stand-in (random routed rows, the page checked to be hn's blocks)."""
    gg = Gguf(MODEL)
    if DENSE:
        gate, up, down = (gg(f"blk.0.ffn_{n}.weight") for n in ("gate", "up", "down"))
    else:
        gate, up, down = (gg(f"blk.{KDA_LAYER}.ffn_{n}_shexp.weight") for n in ("gate", "up", "down"))
    mm, RMj = M_MAX, ttnn.ROW_MAJOR_LAYOUT
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "dense": bool(DENSE), "m_max": mm, "rows": {}}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    try:
        rng = np.random.default_rng(int(time.time()))
        hn = ttnn.from_torch(torch.zeros(1, 1, 1, mm * D, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=RMj,
                             device=dev)
        out = ttnn.from_torch(torch.full((1, 1, 1, mm * D), float("nan")), dtype=ttnn.float32, layout=RMj, device=dev)
        step = ttnn.from_torch(step_words(1), dtype=ttnn.uint32, layout=RMj, device=dev)
        f = FfnFused(dev, gate, up, down, hn, out, lane=not DENSE, m_max=mm, step=step)
        for M in range(1, mm + 1):
            mp = mp_of(M)
            hn_h = torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32)).to(torch.bfloat16)
            routed = torch.zeros(M, D) if DENSE else torch.from_numpy(rng.standard_normal((M, D)).astype(np.float32))
            full = torch.zeros(mm * D, dtype=torch.bfloat16)
            fl = blocks(pad_rows(hn_h), mp)
            full[:fl.numel()] = fl
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(full.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                                            layout=RMj), hn)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(M), dtype=ttnn.uint32, layout=RMj), step)
            rb = torch.zeros(mm * D)
            fr = blocks(pad_rows(routed), mp)
            rb[:fr.numel()] = fr
            seen = {}

            def host(page, reply):
                seen["hn"] = page.copy()
                reply[:] = rb.numpy()

            serve = (lambda: None) if DENSE else (lambda: f.serve(host))
            f()
            serve()
            ttnn.synchronize_device(dev)
            got = unblocks(ttnn.to_torch(out).double().flatten(), NT, mp)[:M]
            want = torch.stack([f.reference(hn_h[m]) for m in range(M)]) + routed.double()
            e = {"max_rel_err": float((got - want).abs().max() / want.abs().max())}
            if not DENSE:
                e["page_is_hn"] = bool(np.array_equal(seen["hn"][:fl.numel()], fl.view(torch.int16).numpy().view(np.uint16)))
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            f()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            for n, key in ((5, None), (REPS, "traced_us")):
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                    serve()
                ttnn.synchronize_device(dev)
                if key:
                    e[key] = round((time.perf_counter() - t) / n * 1e6, 1)
            ttnn.release_trace(dev, tid)
            res["rows"][M] = e
            print(f"M {M}:", json.dumps(e), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_fsh{'_dense' if DENSE else ''}_m{mm}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


M_MAX = int(os.environ.get("FSH_M", 1))


def main() -> None:
    if M_MAX > 1:
        return main_m()
    # the lane first, as every script here starts it: started after the Q6_K dequantization below, its first
    # router phase never finished (2026-11-11; cause not found, the same calls in any other order work)
    lane = cpulane.Lane()
    gg = Gguf(MODEL)
    p = f"blk.{KDA_LAYER}."
    gate, up, down = (gg(p + f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down"))
    if DENSE:
        gate, up, down = (gg(f"blk.0.ffn_{n}.weight") for n in ("gate", "up", "down"))
    router, bias = gg(p + "ffn_gate_inp.weight"), gg(p + "exp_probs_b.bias")               # [288, 4096], [288]
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": 0 if DENSE else KDA_LAYER, "S": S, "dense": bool(DENSE)}
    # the lane's routing against G2's moe_ref formula on the real router
    lane.router_load(KDA_LAYER, router.numpy(), bias.numpy())
    rng = np.random.default_rng(int(time.time()))
    agree, wmax = [], 0.0
    for it in range(20):
        x = rng.standard_normal(D).astype(np.float32)
        ids, w, _ = lane.glm_route(KDA_LAYER, x)
        pr = torch.sigmoid(router.double() @ torch.from_numpy(x).double())
        top = torch.topk(pr + bias.double(), 8).indices
        wr = pr[top] / pr[top].sum() * 2.5
        agree.append(len(set(ids.tolist()) & set(top.tolist())))
        order = {int(e): i for i, e in enumerate(top)}
        if agree[-1] == 8:
            wmax = max(wmax, float(np.abs(w - wr.numpy()[[order[int(e)] for e in ids]]).max()))
    res["route_check"] = {"experts_agree_min_of_8": min(agree), "weights_max_abs_err": wmax}
    print("routing:", json.dumps(res["route_check"]), flush=True)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    try:
        hn_h = torch.from_numpy(rng.standard_normal(D).astype(np.float32)).to(torch.bfloat16)
        routed = rng.standard_normal(D).astype(np.float32)
        hn = ttnn.from_torch(hn_h.reshape(1, 1, 1, D), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        out = ttnn.from_torch(torch.full((1, 1, 1, D), float("nan")), dtype=ttnn.float32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        f = FfnFused(dev, gate, up, down, hn, out, lane=not DENSE)
        res["S"] = f.S
        if DENSE:
            routed[:] = 0
        seen = {"hn": hn_h.view(torch.int16).numpy().view(np.uint16)}

        def host(page, reply):
            seen["hn"] = page.copy()
            reply[:] = routed

        serve = (lambda: None) if DENSE else (lambda: f.serve(host))
        f()
        serve()
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(out).double().flatten()
        want = f.reference(hn_h) + torch.from_numpy(routed).double()
        res["check"] = {"max_rel_err": float((got - want).abs().max() / want.abs().max()), "pcc": round(pcc(got, want), 7),
                        "page_is_hn": bool(np.array_equal(seen["hn"], hn_h.view(torch.int16).numpy().view(np.uint16)))}
        print("check:", json.dumps(res["check"]), f"(circular buffers {f.cb_kb:.0f} KB a core)", flush=True)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        f()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for n, key in ((5, None), (REPS, "traced_us")):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                serve()
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t) / n * 1e6, 1)
        ttnn.release_trace(dev, tid)
        print(f"FF {f.ff}, S {f.S}: traced {res['traced_us']} us with an instant host (the card's own time)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, "g3_fsh_dense.json" if DENSE else "g3_fsh.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
