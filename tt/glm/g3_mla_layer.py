"""One whole MLA layer of GLM on the card + the CPU lane, the real blk.3 weights, at
G2_CTX (8K default: the latent cache and the pooled indexer keys random, as G2's).

  attn half   mHC pre with the attention norm (glm_hc_*, HC_NORM) -> hn; G2's stock projections: q_a + its norm,
              q_b, kv_a + its norm (the cache write left out, as G2's reference reads the cache as given), q absorbed
              by k_b; the indexer's inputs (i_q_b, i_proj) and its scores on the card with the host's top 512
              (glm_idx_*, the 2,048 token ids written back for the gather: IDX_SEL); this token's indexer key and pool
              gate (G2's stock ops); the attention as 2d.3b's (ttnn.embedding of the 2,048 latents, then
              flash_multi_latent_attention_decode in 2 calls of 32 heads); v_b, o; mHC post
  ffn half    g3_kda_layer.FfnHalf: the router, the shared expert under the real lane (layer 3's experts), mHC post
Four host round trips a layer (two mHC parameter sets, the indexer's selection, the lane), served in order.

Checked against float32 torch (G2's hc_ref / MlaCard.mixer_ref / moe_ref, the lane on the reference's route): the
attention output, the pools chosen (of 512), the experts chosen, the new streams. Timed traced with the host
serving; the stock pieces alone for the breakdown.

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((48 << 30)) --engine tt/glm/g3_mla_layer.py
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
from g2_layers import (BF16, CTX, D, FP32, HC, HK, IH, IHD, ITOP, KPOOL, KVL, LN_EPS, MLA_LAYER, MODEL, NH, OUT,  # noqa
                       RM, RMS_EPS, TILE, Card, Gguf, MlaCard, hc_consts, hc_post_ref, hc_ref, pcc, traced_us)
from g3_hc import HcSite, sockets  # noqa: E402
from g3_idx import IdxStep  # noqa: E402
from g3_kda_layer import FFN_FUSED, LOOP, FfnHalf  # noqa: E402
from g3_fmo import MlaOut  # noqa: E402
from g3_fmq import NS, MlaQuery  # noqa: E402
from fused.grid import Grid  # noqa: E402

REPS = int(os.environ.get("ML_REPS", 100))
FUSED_OUT = int(os.environ.get("ML_FUSED_OUT", 1))   # 1: v_b + o as one program (g3_fmo.py), 0: G2's stock ops
FUSED_Q = int(os.environ.get("ML_FUSED_Q", 1))       # 1: the query side + indexer as one program (g3_fmq.py)
HG = 32                                            # heads a flash-MLA call (2d.3b: its buffers outgrow L1 at 64)


class MlaAttention:
    """2d.3b's attention over the selection: sel's NS latents gathered from lat_2d (row-major bf16 [ctx, 512]) by
    ttnn.embedding, then flash_multi_latent_attention_decode per group of 32 heads over the first cpos + 1 of them
    (cpos: int32 [1] on the card, so one trace serves every token). The slots past the count point at row 0."""

    def __init__(self, sel, cpos, lat_2d):
        self.sel, self.cpos, self.lat = sel, cpos, lat_2d
        self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                   fp32_dest_acc_en=True, packer_l1_acc=False)
        self.pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=0,
                                         k_chunk_size=64, exp_approx_mode=False)

    def __call__(self, qs, outs=None):
        """qs: the query groups (bf16 tiles [1, 1, 32, 512]); outs: tensors to copy the groups' outputs into (then
        None is returned), else their concat [1, 1, 64, 512]."""
        kv = ttnn.reshape(ttnn.embedding(self.sel, self.lat, layout=TILE), [1, 1, NS, KVL])
        res = [ttnn.transformer.flash_multi_latent_attention_decode(
            q, kv, None, KVL, cur_pos_tensor=self.cpos, scale=HK ** -0.5, compute_kernel_config=self.ck,
            program_config=self.pc) for q in qs]
        res = [o if list(o.shape) == [1, 1, HG, KVL] else ttnn.slice(o, [0, 0, 0, 0], [1, 1, HG, KVL]) for o in res]
        if outs is None:
            return ttnn.concat(res, dim=2)
        for o, dst in zip(res, outs):
            ttnn.copy(o, dst)
        return None


def main() -> None:
    lane = cpulane.Lane()
    lane.load(MODEL, layers={MLA_LAYER})
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    me = threading.get_native_id()
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": MLA_LAYER, "ctx": CTX}
    try:
        h2d, d2h = sockets(dev)                    # every socket before any program is built or traced
        xc = None if FFN_FUSED else Xchg(dev)          # the fused FFN half has its own sockets
        put = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa: E731
        sel = put(torch.zeros(1, NS, dtype=torch.int32), ttnn.uint32, RM)
        card, g = Card(dev), Gguf(MODEL)
        L = MlaCard(card, g, MLA_LAYER, hc_consts(), torch.Generator().manual_seed(0))
        r = L.ref
        n_pool = CTX // KPOOL
        og = [put(torch.zeros(1, 1, HG, KVL), BF16, TILE) for _ in range(2)]      # flash-MLA query groups (FUSED_Q)
        hn_a = put(torch.zeros(1, 1, 1, D), BF16, RM)
        lat_2d = put(r["lat"].reshape(CTX, KVL), BF16, RM)
        cpos = put(torch.full((1,), ITOP - 1, dtype=torch.int32), ttnn.int32, RM)   # the stateless selection's
        ix = None if FUSED_Q else IdxStep(dev, L.pooled, n_pool, sel=sel)
        fmq = MlaQuery(dev, r, L.pooled, n_pool, hn_a, og, sel, cpos, lat_2d) if FUSED_Q else None
        grid = Grid(dev)
        ffn = FfnHalf(dev, grid, L, MLA_LAYER, lane, xc, h2d, d2h)
        hs_a = HcSite(dev, grid, *r["hc_attn"], h2d, d2h, hn=hn_a, norm=r["attn_norm"], xn=ffn.hs.x)
        att = MlaAttention(sel, cpos, lat_2d)

        def proj(hn):
            qr = ttnn.rms_norm(L.q_a(hn), weight=L.q_a_norm, epsilon=RMS_EPS)
            q = ttnn.reshape(L.q_b(qr), [1, NH, 1, HK])
            kv = ttnn.rms_norm(L.kv_a(hn), weight=L.kv_a_norm, epsilon=RMS_EPS)  # noqa: F841 (the cache write: out)
            return qr, ttnn.matmul(q, L.k_b)                                     # q_abs [1, 64, 1, 512]

        def idx_inputs(hn, qr):
            iq = ttnn.reshape(L.i_qb(qr), [1, 1, IH, IHD])
            ttnn.copy(ttnn.transpose(iq, -2, -1), ix.iqT)
            ttnn.copy(ttnn.matmul(hn, L.i_proj), ix.w)

        def idx_tail(hn):
            """this token's indexer key and pool gate, the pooled key of the pool it closes (G2's stock ops)"""
            ik = ttnn.layer_norm(L.i_k(hn), weight=L.i_kn_w, bias=L.i_kn_b, epsilon=LN_EPS)
            win_k = ttnn.concat([L.tail_k, ik], dim=2)
            win_g = ttnn.add(ttnn.concat([L.tail_g, L.i_gate(hn)], dim=2), L.ape)
            pooled = ttnn.sum(ttnn.multiply(ttnn.softmax(win_g, dim=2), win_k), dim=2, keepdim=True)
            ttnn.copy(ttnn.slice(win_k, [0, 0, 1, 0], [1, 1, KPOOL, IHD]), L.tail_k)
            return pooled

        qo =[put(torch.zeros(1, 1, HG, KVL), BF16, TILE) for _ in range(2)]      # the attention's outputs (FUSED_OUT)

        def attention(q_abs):
            if q_abs is None:
                qs = og                                      # g3_fmq.py wrote the groups
            else:
                q4 = ttnn.reshape(q_abs, [1, 1, NH, KVL])
                qs = [ttnn.slice(q4, [0, 0, i, 0], [1, 1, i + HG, KVL]) for i in range(0, NH, HG)]
            return att(qs, qo if FUSED_OUT else None)

        fmo = MlaOut(dev, r["v_b"], r["output"], qo[0], qo[1], hs_a.out) if FUSED_OUT else None

        def out_proj(o):
            o = ttnn.matmul(ttnn.reshape(o, [1, NH, 1, KVL]), L.v_b)                # [1, 64, 1, 256]
            return L.wo(ttnn.reshape(o, [1, 1, 1, NH * HK]))

        def layer():
            hs_a.run(hs_a.pre_prog)
            if FUSED_Q:
                fmq()                                        # q groups, the selection (the host's top 512)
                q_abs = None
            else:
                hn = ttnn.to_layout(hn_a, TILE)
                qr, q_abs = proj(hn)
                idx_inputs(hn, qr)
                ix.run()
                idx_tail(hn)
            o = attention(q_abs)
            if FUSED_OUT:
                fmo()                                        # v_b + o straight into hs_a.out
            else:
                ttnn.copy(ttnn.to_layout(ttnn.typecast(out_proj(o), FP32), RM), hs_a.out)
            hs_a.run(hs_a.post_prog)
            ffn.run()

        if LOOP:
            from glm_loop import GlmLoop
            loop = GlmLoop(lane.lib)
            loop.hc(hs_a)
            loop.sel(fmq)
            loop.hc(ffn.hs)
            loop.lane(ffn.fsh, MLA_LAYER)

        def host():
            if LOOP:
                loop.run()
                ffn.ids, fmq.pools = loop.ids()[0], loop.pools()
                return
            hs_a.serve()
            (fmq if FUSED_Q else ix).serve()
            ffn.serve()

        # check: one step
        x0 = torch.randn(HC, D, generator=torch.Generator().manual_seed(15))
        xt = torch.zeros(32, D)
        xt[:HC] = x0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(xt.reshape(1, 1, 32, D), dtype=FP32, layout=TILE), hs_a.x)
        layer()
        host()
        ttnn.synchronize_device(dev)
        attn_c = ttnn.to_torch(hs_a.out).double().flatten()
        y_c = ttnn.to_torch(ffn.x_next).double().reshape(32, D)[:HC]
        pools_c, ids_c = set((fmq if FUSED_Q else ix).pools.tolist()), set(ffn.ids.tolist())

        h, post, comb = hc_ref(x0, *r["hc_attn"])
        attn = L.mixer_ref(h * torch.rsqrt(h.pow(2).mean() + RMS_EPS) * r["attn_norm"])
        x1 = hc_post_ref(attn, x0, post, comb)
        y_ref, idx = ffn.reference(x1)
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
        res["check"] = {"attn_pcc": round(pcc(attn_c, attn.double()), 6), "attn_max_rel_err": rel(attn_c, attn.double()),
                        "pools_agree_of_512": len(pools_c & set(L.ref_pools.tolist())),
                        "experts_agree_of_8": len(ids_c & set(idx.tolist())),
                        "streams_pcc": round(pcc(y_c, y_ref), 6), "streams_max_rel_err": rel(y_c, y_ref),
                        "delta_pcc": round(pcc(y_c - x0.double(), y_ref - x0.double()), 6)}
        print("check:", json.dumps(res["check"]), flush=True)

        # timed: the whole layer traced, the host serving its four round trips
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
        res["fused_out"], res["fused_q"], res["host_loop"] = bool(FUSED_OUT), bool(FUSED_Q), bool(LOOP)

        hn = ttnn.to_layout(hn_a, TILE)
        qr, q_abs = proj(hn)
        o = attention(None if FUSED_Q else q_abs)
        hf = ttnn.to_layout(ffn.hn, TILE)
        pieces = {"hn to tiles": lambda: ttnn.to_layout(hn_a, TILE), "q_a, q_b, kv_a, norms, absorb": lambda: proj(hn),
                  "indexer inputs (i_q_b, i_proj)": lambda: idx_inputs(hn, qr), "indexer tail (key, pool gate)":
                  lambda: idx_tail(hn), "attention (gather + 2 flash-MLA)": lambda: attention(q_abs),
                  "v_b + o": fmo if FUSED_OUT else (lambda: out_proj(o)), "router + sigmoid": lambda: ttnn.sigmoid(ttnn.typecast(
                      L.moe.router(hf), FP32)), "shared expert": lambda: ffn.shexp(hf),
                  "mHC post": lambda: hs_a.run(hs_a.post_prog)}
        if FUSED_Q:                                          # one program with its host round trip: not timed alone
            for k in ("hn to tiles", "q_a, q_b, kv_a, norms, absorb", "indexer inputs (i_q_b, i_proj)",
                      "indexer tail (key, pool gate)"):
                pieces.pop(k)
        res["pieces_us"] = {}
        for name, fn in pieces.items():
            res["pieces_us"][name] = round(traced_us(dev, fn, reps=REPS, tokens=10), 1)
            print(f"  {res['pieces_us'][name]:8.1f} us  {name}", flush=True)
        print(f"MLA layer {MLA_LAYER} at {CTX}: {res['layer_us_traced']} us traced with the host (median {res['layer_us_median']}) (its wall from the "
              f"launch {res['host_wall_us']} us)", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_mla_layer_ctx{CTX}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
