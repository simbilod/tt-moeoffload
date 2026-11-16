"""One whole KDA layer of GLM on the card + the CPU lane, the real blk.4 weights.

The card's pieces put together, one token through blk.4:
  attn half   mHC pre (glm_hc_*, phases 1: the host's parameters over the sockets) -> h; RMSNorm; the mixer's
              projections and small ops on stock ttnn (G2's: q|k|v, the conv window, silu, the l2 norms, the gates);
              the KDA step (glm_kda_*, the state in place, fp32); the o projection; mHC post (phases 2)
  ffn half    mHC pre -> h; RMSNorm; the router's sigmoids and hn to the host (glm_xsend); the shared expert while
              the host routes (top 8 with the bias) and runs the real lane (tt/cpulane, layer 4's experts); the
              experts' sum back (glm_xrecv) and added; mHC post
Three host round trips a layer (two mHC parameter sets, the lane), served in order while the card's queue runs.

Checked against float32 torch (G2's hc_ref / mixer_ref / moe_ref, llama.cpp's glm5-next graph) with the lane's own
output on the reference's route as the routed experts: the attention output, the KDA state, the experts chosen, the
new streams. Timed traced with the host serving; the mixer alone (no sockets) for the breakdown.

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((40 << 30)) --engine tt/glm/g3_kda_layer.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g3_overlap import Xchg, cpulane  # noqa: E402  (sets CL_LIB to the GLM lane first)
from g2_layers import (BF16, D, FP32, HC, HD, KDA_LAYER, KDA_LB, L2_EPS, MODEL, NH, NX, OUT, RM, RMS_EPS, TILE,  # noqa
                       Card, Gguf, KdaCard, hc_consts, hc_post_ref, hc_ref, moe_ref, pcc, traced_us)
from g3_hc import HcSite, sockets  # noqa: E402
from g3_kda import KdaStep  # noqa: E402
from g3_fkda import Mixer  # noqa: E402
from g3_fsh import FfnFused  # noqa: E402
from fused.grid import Grid  # noqa: E402

TOPK = 8
REPS = int(os.environ.get("KL_REPS", 100))
FUSED = int(os.environ.get("KL_FUSED", 1))        # 1: the mixer as one program (g3_fkda.py), 0: G2's stock ops
LOOP = int(os.environ.get("KL_LOOP", 1))          # 1: the host's round trips in C++ (glm_loop.py; needs FFN_FUSED)
FFN_FUSED = int(os.environ.get("KL_FFN_FUSED", 1))  # 1: the FFN half as one program (g3_fsh.py), the router on the lane


class FfnHalf:
    """A layer's FFN half on the card + the lane (shared by the KDA and MLA layers): the streams from the attention
    half's mHC result (its columns D..2D) into this site's x; mHC pre with the FFN norm (hn); the router's sigmoids
    and hn to the host (glm_xsend); the shared expert while the host routes and runs the lane; the experts' sum back
    (glm_xrecv) and added; mHC post; the new streams to x_next. serve() is the host's side, in order. With FFN_FUSED
    (the default) the middle is g3_fsh.FfnFused: hn straight to the host, which routes in fp32 on the lane's threads,
    the shared expert and the sum in one program."""

    def __init__(self, dev, grid, L, layer, lane, xc, h2d, d2h):
        r = L.ref
        put = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa: E731
        self.L, self.layer, self.lane, self.xc, self.r = L, layer, lane, xc, r
        self.hn = put(torch.zeros(1, 1, 1, D), BF16, RM)
        self.x_next = put(torch.zeros(1, 1, 32, D), FP32, TILE)       # the post's new streams: the next layer's x
        self.hs = HcSite(dev, grid, *r["hc_ffn"], h2d, d2h, hn=self.hn, norm=r["ffn_norm"], xn=self.x_next)
        self.bias = r["probs_b"].numpy().astype(np.float32)
        self.lane_out = np.zeros(D, np.float32)
        self.ids = None
        if FFN_FUSED:
            lane.router_load(layer, r["router"].numpy(), r["probs_b"].numpy())
            self.fsh = FfnFused(dev, r["sh_gate"], r["sh_up"], r["sh_down"], self.hn, self.hs.out)
            self.ids_buf, self.w_buf = np.zeros(TOPK, np.int32), np.zeros(TOPK, np.float32)

    def shexp(self, hn):
        gu = self.L.moe.sh_gu(hn)
        gate = ttnn.clamp(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 1, 2048]), max=10.0)
        up = ttnn.clamp(ttnn.slice(gu, [0, 0, 0, 2048], [1, 1, 1, 4096]), min=-10.0, max=10.0)
        return self.L.moe.sh_down(ttnn.multiply(ttnn.silu(gate), up))

    def run(self):
        """hs.x holds the attention site's new streams (its post writes them there)"""
        hs, xc = self.hs, self.xc
        hs.run(hs.pre_prog)
        if FFN_FUSED:
            self.fsh()
            hs.run(hs.post_prog)
            return
        hn = ttnn.to_layout(self.hn, TILE)
        probs = ttnn.sigmoid(ttnn.typecast(self.L.moe.router(hn), FP32))
        row = ttnn.concat([ttnn.typecast(hn, FP32), ttnn.pad(probs, [(0, 0), (0, 0), (0, 0), (0, 512 - NX)], 0.0)],
                          dim=-1)
        ttnn.copy(ttnn.to_layout(row, RM), xc.page)
        xc.send()
        sh = self.shexp(hn)
        xc.recv()
        routed = ttnn.to_layout(ttnn.to_memory_config(xc.reply, ttnn.DRAM_MEMORY_CONFIG), TILE)
        ttnn.copy(ttnn.to_layout(ttnn.add(ttnn.typecast(sh, FP32), routed), RM), hs.out)
        hs.run(hs.post_prog)

    def serve(self):
        xc = self.xc
        self.hs.serve()
        if FFN_FUSED:
            self.fsh.serve(lambda page, reply: self.lane.glm_exchange(self.layer, page, reply, self.ids_buf, self.w_buf))
            self.ids = self.ids_buf
            return
        xc.d2h.read_tensor(xc.host_out)
        f = xc.host_out.numpy().view(np.float32)
        hn, probs = f[:D], f[D:D + NX]
        ids = np.argpartition(-(probs + self.bias), TOPK - 1)[:TOPK].astype(np.int32)
        w = probs[ids]
        w = (w / w.sum() * 2.5).astype(np.float32)
        self.lane.moe(self.layer, np.ascontiguousarray(hn), ids, w, self.lane_out)
        xc.host_in.numpy().view(np.float32)[:] = self.lane_out
        xc.h2d.write_tensor(xc.host_in)
        self.ids = ids

    def reference(self, x1):
        """float32 torch from the attention half's new streams x1 [HC, D], the lane on the reference's route: the
        new streams and the experts chosen."""
        r = self.r
        h, post, comb = hc_ref(x1, *r["hc_ffn"])
        hn = h * torch.rsqrt(h.pow(2).mean() + RMS_EPS) * r["ffn_norm"]
        sh, idx, w = moe_ref(hn, r)
        routed = np.zeros(D, np.float32)
        self.lane.moe(self.layer, hn.numpy().astype(np.float32), idx.numpy().astype(np.int32),
                      w.numpy().astype(np.float32), routed)
        return hc_post_ref(sh + torch.from_numpy(routed), x1, post, comb).double(), idx


def main() -> None:
    lane = cpulane.Lane()
    lane.load(MODEL, layers={KDA_LAYER})
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    me = threading.get_native_id()
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": KDA_LAYER}
    try:
        # the sockets and their L1 pages before any program is built or traced (g3_overlap.py's trap)
        h2d, d2h = sockets(dev)
        xc = None if FFN_FUSED else Xchg(dev)          # the fused FFN half has its own sockets
        card, g = Card(dev), Gguf(MODEL)
        L = KdaCard(card, g, KDA_LAYER, hc_consts(), torch.Generator().manual_seed(0))
        r = L.ref
        grid = Grid(dev)
        put = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa: E731
        # each mHC pre launch ends with its sublayer's input norm (HC_NORM): hn straight into its consumer's input
        hn_a = put(torch.zeros(1, 1, 1, D), BF16, RM)
        ffn = FfnHalf(dev, grid, L, KDA_LAYER, lane, xc, h2d, d2h)
        hs_a = HcSite(dev, grid, *r["hc_attn"], h2d, d2h, xn=ffn.hs.x,
                      **(dict(hn=hn_a, norm=r["attn_norm"]) if FUSED else {}))
        hs_f, x_next = ffn.hs, ffn.x_next
        S = put(r["S0"].reshape(1, NH, HD, HD), FP32, TILE)                       # the state, fp32, in place
        rows = put(torch.zeros(1, 1, 5 * NH, HD), FP32, RM)
        par = put(torch.zeros(1, 1, 2, HD), FP32, RM)
        kout = put(torch.zeros(1, 1, NH, HD), FP32, RM)
        kda = KdaStep(dev, S, rows, par, S, kout)
        onorm = put(r["o_norm"].reshape(1, 1, 1, HD), FP32, TILE)

        def conv(qkv):
            win = ttnn.concat([L.conv, qkv], dim=2)                               # [1,1,4,24576]
            ttnn.copy(ttnn.slice(win, [0, 0, 1, 0], [1, 1, 4, 3 * NH * HD]), L.conv)
            qkv = ttnn.silu(ttnn.sum(ttnn.multiply(win, L.taps), dim=2, keepdim=True))
            part = lambda i: ttnn.reshape(ttnn.slice(qkv, [0, 0, 0, i * NH * HD], [1, 1, 1, (i + 1) * NH * HD]),  # noqa
                                          [1, 1, NH, HD])
            l2 = lambda t, s: ttnn.multiply(ttnn.rms_norm(t, epsilon=L2_EPS / HD), s)  # noqa: E731
            return l2(part(0), HD ** -1.0), l2(part(1), HD ** -0.5), part(2)

        def gates(hn):
            g1 = ttnn.add(ttnn.matmul(ttnn.matmul(hn, L.f_a), L.f_b, dtype=FP32), ttnn.typecast(L.dt, FP32))
            decay = ttnn.exp(ttnn.multiply(ttnn.sigmoid(ttnn.multiply(g1, ttnn.typecast(L.neg_a, FP32))), KDA_LB))
            beta = ttnn.sigmoid(ttnn.matmul(hn, L.beta, dtype=FP32))              # [1,1,1,64]
            g2 = ttnn.sigmoid(ttnn.matmul(ttnn.matmul(hn, L.g_a), L.g_b, dtype=FP32))
            return decay, beta, g2

        def pack(q, k, v, decay, beta, g2):
            f = lambda t: ttnn.typecast(t, FP32) if t.dtype != FP32 else t  # noqa: E731
            cat = ttnn.concat([f(q), f(k), f(v), ttnn.reshape(g2, [1, 1, NH, HD]),
                               ttnn.reshape(decay, [1, 1, NH, HD])], dim=2)         # [1,1,320,128]
            ttnn.copy(ttnn.to_layout(cat, RM), rows)
            p = ttnn.concat([onorm, ttnn.pad(beta, [(0, 0), (0, 0), (0, 0), (0, HD - NH)], 0.0)], dim=2)
            ttnn.copy(ttnn.to_layout(p, RM), par)

        def o_proj():
            return L.wo(ttnn.typecast(ttnn.to_layout(ttnn.reshape(kout, [1, 1, 1, NH * HD]), TILE), BF16))

        def mixer(hn):
            q, k, v = conv(L.qkv(hn))
            pack(q, k, v, *gates(hn))
            kda()
            return o_proj()

        fm = Mixer(dev, g, None, states=(r["conv0"], r["S0"]), out=hs_a.out, in_place=True, hn=hn_a) if FUSED else None
        fused_run = fm.program() if FUSED else None

        def attn_mixer(hn):
            """the attention half's mixer from hn into hs_a.out"""
            if FUSED:
                fused_run()                                  # hn is already in fm.hn (hs_a's pre launch)
            else:
                to_out(mixer(hn), hs_a)

        def h_of(hs, w):
            return ttnn.rms_norm(ttnn.typecast(ttnn.slice(hs.res, [0, 0, 0, 0], [1, 1, 1, D]), BF16), weight=w,
                                 epsilon=RMS_EPS)

        def to_out(t, hs):
            ttnn.copy(ttnn.to_layout(t if t.dtype == FP32 else ttnn.typecast(t, FP32), RM), hs.out)

        def layer():
            hs_a.run(hs_a.pre_prog)
            attn_mixer(None if FUSED else h_of(hs_a, L.attn_norm))
            hs_a.run(hs_a.post_prog)
            ffn.run()

        if LOOP:
            from glm_loop import GlmLoop
            loop = GlmLoop(lane.lib)
            loop.hc(hs_a)
            loop.hc(ffn.hs)
            loop.lane(ffn.fsh, KDA_LAYER)

        def host():
            if LOOP:
                loop.run()
                ffn.ids = loop.ids()[0]
                return
            hs_a.serve()
            ffn.serve()

        # check: one step from the fresh state
        x0 = torch.randn(HC, D, generator=torch.Generator().manual_seed(14))
        xt = torch.zeros(32, D)
        xt[:HC] = x0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(xt.reshape(1, 1, 32, D), dtype=FP32, layout=TILE), hs_a.x)
        layer()
        host()
        ttnn.synchronize_device(dev)
        attn_c = ttnn.to_torch(hs_a.out).double().flatten()
        S_c = ttnn.to_torch(fm.S if FUSED else S).double().reshape(NH, HD, HD)
        y_c = ttnn.to_torch(x_next).double().reshape(32, D)[:HC]
        ids_c = set(ffn.ids.tolist())

        rms = lambda v, w: v * torch.rsqrt(v.pow(2).mean() + RMS_EPS) * w  # noqa: E731
        h, post, comb = hc_ref(x0, *r["hc_attn"])
        attn = L.mixer_ref(rms(h, r["attn_norm"]))
        x1 = hc_post_ref(attn, x0, post, comb)
        y_ref, idx = ffn.reference(x1)
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
        res["check"] = {"attn_pcc": round(pcc(attn_c, attn.double()), 6), "attn_max_rel_err": rel(attn_c, attn.double()),
                        "experts_agree_of_8": len(ids_c & set(idx.tolist())),
                        "streams_pcc": round(pcc(y_c, y_ref), 6), "streams_max_rel_err": rel(y_c, y_ref),
                        "delta_pcc": round(pcc(y_c - x0.double(), y_ref - x0.double()), 6),
                        "state_changed": bool((S_c - r["S0"].double()).abs().max() > 0)}
        print("check:", json.dumps(res["check"]), flush=True)

        # timed: the whole layer traced, the host serving its three round trips
        layer()
        host()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        layer()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        res["fenced_threads"] = len(lane.fence_others(keep={me}))
        if LOOP:
            loop.random_routes(True)                  # the timing's lane draws fresh experts, as decode does
            res["random_routes"] = True
        os.sched_setaffinity(0, {lane.cpus[0]})       # the caller is the lane's thread 0
        hosts, ends = [], []
        for n, key in ((5, None), (REPS, "layer_us_traced")):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            ends = []
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                s = time.perf_counter()
                host()
                ends.append(time.perf_counter())
                hosts.append(ends[-1] - s)
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t) / n * 1e6, 1)
        # the median period between consecutive host loops' ends: a layer's steady time without the outliers
        res["layer_us_median"] = round(statistics.median(np.diff(ends)) * 1e6, 1)
        ttnn.release_trace(dev, tid)
        res["host_wall_us"] = round(statistics.median(hosts) * 1e6, 1)
        hn = h_of(hs_a, L.attn_norm)
        res["fused_mixer"], res["host_loop"] = bool(FUSED), bool(LOOP)
        res["mixer_alone_us"] = round(traced_us(dev, lambda: attn_mixer(hn), reps=REPS, tokens=10), 1)
        qkv = L.qkv(hn)
        qkv_v, gts = conv(qkv), gates(hn)
        hf = h_of(hs_f, L.ffn_norm)
        pieces = {"q|k|v projection": lambda: L.qkv(hn), "conv, silu, l2 norms": lambda: conv(qkv),
                  "gates (f, beta, g: 5 matmuls)": lambda: gates(hn), "pack rows + par": lambda: pack(*qkv_v, *gts),
                  "KDA step": kda, "o projection": o_proj, "h + RMSNorm": lambda: h_of(hs_a, L.attn_norm),
                  "router + sigmoid": lambda: ttnn.sigmoid(ttnn.typecast(L.moe.router(hf), FP32)),
                  "shared expert": lambda: ffn.shexp(hf),
                  "mHC post": lambda: hs_a.run(hs_a.post_prog)}
        res["pieces_us"] = {}
        for name, fn in pieces.items():
            res["pieces_us"][name] = round(traced_us(dev, fn, reps=REPS, tokens=10), 1)
            print(f"  {res['pieces_us'][name]:8.1f} us  {name}", flush=True)
        res["kda_step_alone_us"] = res["pieces_us"]["KDA step"]
        print(f"KDA layer {KDA_LAYER}: {res['layer_us_traced']} us traced with the host (median {res['layer_us_median']}) (its wall from the "
              f"launch {res['host_wall_us']} us); the mixer alone {res['mixer_alone_us']} us, of it the KDA step "
              f"{res['kda_step_alone_us']} us", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, "g3_kda_layer.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
