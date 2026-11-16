"""GLM's MTP draft block (blk.45, llama.cpp's glm5-next graph_mtp) on the card, one row a step,
from the one-row programs: the front by ttnn's ops (rms_norm(e) enorm | rms_norm(h) hnorm -> eh_proj, 8192 -> 4096),
then blk.45's MLA layer (g3_fmq.MlaQuery, the attention, g3_fmo.MlaOut; its own host state and caches) and residual,
its FFN half (g3_fsh.FfnFused: the shared expert under the lane's routed experts) and residual, shared_head_norm (the
next step's h) and the trunk's head (output.weight) with the top 64 (g5_head_m.py's ttnn.topk). No mHC: a plain
residual block. Its host round trips (the selection, the lane) through one GlmLoop.

Checked step by step from position 0 against float64 of the same block on the GGUF's weights: the MLA by
g3_mla_decode.Ref (its own caches), the shared expert in float64, the routed experts by the lane's cl_glm_exchange
(bitwise against itself, g5_lane_m.py) on the reference's rounded input; e = token_embd rows of random tokens, h
random (the trunk's post-norm hidden stands in), each step's h the card's own previous one (so errors do not chain).
Then timed: one step traced with the loop serving.

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((64 << 30)) --engine tt/glm/g5_mtp.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g3_overlap import cpulane  # noqa: E402  (sets CL_LIB to the GLM lane first)
from g2_layers import BF16, BFP8, CTX, D, FP32, IHD, KPOOL, MODEL, OUT, RM, RMS_EPS, TILE, Gguf, pcc  # noqa: E402
from g3_fmq import NS, MlaQuery, _LIB, mla_state  # noqa: E402
from g3_fmo import MlaOut  # noqa: E402
from g3_fsh import FfnFused  # noqa: E402
from g3_mla_decode import Ref  # noqa: E402
from g3_mla_layer import HG, MlaAttention  # noqa: E402
from glm_loop import GlmLoop  # noqa: E402

MTP, KVL, K = 45, 512, 64
STEPS = int(os.environ.get("MTP_STEPS", 40))
REPS = int(os.environ.get("MTP_REPS", 30))


class Mtp:
    """One MTP step a run(): e (the draft input token's embedding) and h (the previous row's post-norm hidden) in, the
    next h and the top 64 out. The GlmLoop (loop) serves its two round trips."""

    def __init__(self, dev, g: Gguf, head, ctx=CTX, seq=0):
        put = lambda t, dt, lay=TILE: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa
        w = lambda n: g(f"blk.{MTP}." + n)  # noqa: E731
        vec = lambda n: put(w(n).reshape(1, 1, 1, D).to(torch.bfloat16), BF16)  # noqa: E731
        self.dev, self.head = dev, head
        self.e, self.h = put(torch.zeros(1, 1, 1, D), FP32), put(torch.zeros(1, 1, 1, D), FP32)
        self.enorm, self.hnorm = vec("nextn.enorm.weight"), vec("nextn.hnorm.weight")
        self.attn_norm, self.ffn_norm = vec("attn_norm.weight"), vec("ffn_norm.weight")
        self.sh_norm = vec("nextn.shared_head_norm.weight")
        self.eh = put(w("nextn.eh_proj.weight").T.contiguous().to(torch.bfloat16), BFP8)   # [8192, 4096]
        r = {x: w(f"attn_{x}.weight") for x in ("q_a", "q_b", "kv_a_mqa", "k_b", "v_b", "output", "q_a_norm",
                                               "kv_a_norm")}
        for x in ("attn_k", "attn_q_b", "proj", "k_norm"):
            r["i_" + x] = w(f"indexer.{x}.weight")
        r["i_k_norm_b"] = w("indexer.k_norm.bias")
        r["i_gate"], r["i_ape"] = w("indexer_compressor_gate.weight"), w("indexer_compressor_ape.weight")
        self.r = r
        n_pool = ctx // KPOOL
        self.hn_a, self.hn_f = put(torch.zeros(1, 1, 1, D), BF16, RM), put(torch.zeros(1, 1, 1, D), BF16, RM)
        self.out_a, self.out_f = put(torch.zeros(1, 1, 1, D), FP32, RM), put(torch.zeros(1, 1, 1, D), FP32, RM)
        self.K = put(torch.zeros(1, 1, n_pool, IHD), BF16)
        self.lat = put(torch.zeros(ctx, KVL), BF16, RM)
        self.sel = put(torch.zeros(1, NS, dtype=torch.int32), ttnn.uint32, RM)
        self.cpos = put(torch.zeros(1, dtype=torch.int32), ttnn.int32, RM)
        self.og = [put(torch.zeros(1, 1, HG, KVL), BF16) for _ in range(2)]
        self.qo = [put(torch.zeros(1, 1, HG, KVL), BF16) for _ in range(2)]
        self.state, self.keep = mla_state(r)
        self.fmq = MlaQuery(dev, r, self.K, n_pool, self.hn_a, self.og, self.sel, self.cpos, self.lat, self.state,
                            check=False)
        self.att = MlaAttention(self.sel, self.cpos, self.lat)
        self.fmo = MlaOut(dev, r["v_b"], r["output"], self.qo[0], self.qo[1], self.out_a, check=False)
        self.ffn = FfnFused(dev, *(w(f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down")), self.hn_f, self.out_f,
                            lane=True, check=False)
        self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                   fp32_dest_acc_en=True, packer_l1_acc=True)
        self.ck_head = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                        fp32_dest_acc_en=True, packer_l1_acc=True)
        self.loop = GlmLoop(_LIB, seq)   # (seq: its own exchange list beside a trunk's)
        self.loop.sel(self.fmq)
        if self.ffn.lane:
            self.loop.lane(self.ffn, MTP)
        self.o = {}

    def norm(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=RMS_EPS, compute_kernel_config=self.ck)

    def to_rm(self, x, dst):
        """an fp32 tile vector [1, 1, 1, D] into dst (row-major bf16, a program's input)"""
        ttnn.copy(ttnn.to_layout(ttnn.typecast(x, BF16), RM), dst)

    def run(self, part=15):
        """part 1: the front and the MLA's query side (its round trip next); 2: the attention, the output side, the
        FFN program (the lane's next); 4: the residual and shared_head_norm (the next h); 8: the head and its top 64.
        All in a trace: the loop serves both round trips after it. A catch-up row needs no head: parts 1 | 2 | 4."""
        if part & 1:
            cat = ttnn.concat([self.norm(self.e, self.enorm), self.norm(self.h, self.hnorm)], dim=-1)
            self.o["x"] = ttnn.linear(ttnn.typecast(cat, BF16), self.eh, dtype=FP32, compute_kernel_config=self.ck)
            self.to_rm(self.norm(self.o["x"], self.attn_norm), self.hn_a)
            self.fmq()
        if part & 2:
            self.att(self.og, self.qo)
            self.fmo()
            self.o["x2"] = ttnn.add(self.o["x"], ttnn.to_layout(self.out_a, TILE))
            self.to_rm(self.norm(self.o["x2"], self.ffn_norm), self.hn_f)
            self.ffn()
        if part & 4:
            x3 = ttnn.add(self.o["x2"], ttnn.to_layout(self.out_f, TILE))
            self.o["h"] = self.norm(x3, self.sh_norm)
        if part & 8:
            lg = ttnn.linear(ttnn.typecast(self.o["h"], BF16), self.head, dtype=FP32, compute_kernel_config=self.ck_head)
            self.o["top_v"], self.o["top_i"] = ttnn.topk(ttnn.typecast(lg, BF16), K, dim=-1, largest=True, sorted=True)

    def first(self):
        """the first step eager, a part at a time with each round trip served at once (eager, the ops after a program
        can block the main thread before the host serves it)"""
        lib = self.loop.lib
        lane = (lambda: self.loop.lane(self.ffn, MTP)) if self.ffn.lane else None
        for part, add in ((1, lambda: self.loop.sel(self.fmq)), (2, lane), (12, None)):
            self.run(part)
            if add:
                self.loop.clear()
                add()
                self.loop.run()
            ttnn.synchronize_device(self.dev)
        self.loop.clear()
        self.loop.sel(self.fmq)
        if self.ffn.lane:
            self.loop.lane(self.ffn, MTP)


def main() -> None:
    lane = cpulane.Lane()
    lane.load(MODEL, layers={MTP})
    g = Gguf(MODEL)
    lane.router_load(MTP, g(f"blk.{MTP}.ffn_gate_inp.weight").numpy(), g(f"blk.{MTP}.exp_probs_b.bias").numpy())
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "steps": STEPS}
    try:
        head = ttnn.from_torch(g("output.weight").T.contiguous().to(torch.bfloat16), dtype=BFP8, layout=TILE,
                               device=dev)
        _LIB.gl_mla_clear()
        m = Mtp(dev, g, head)
        emb = g("token_embd.weight")
        w = lambda n: g(f"blk.{MTP}." + n).double()  # noqa: E731
        eh, enorm, hnorm = w("nextn.eh_proj.weight"), w("nextn.enorm.weight"), w("nextn.hnorm.weight")
        an, fnorm, shn = w("attn_norm.weight"), w("ffn_norm.weight"), w("nextn.shared_head_norm.weight")
        gate, up, down = (w(f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down"))
        head_w = g("output.weight").double()
        ref_mla = Ref(m.r)
        rms = lambda x, wt: x * torch.rsqrt(x.pow(2).mean() + RMS_EPS) * wt  # noqa: E731
        rng = np.random.default_rng(int(time.time()))
        h_card = torch.from_numpy(rng.standard_normal(D).astype(np.float32))
        errs, agree, top1, tid = {"x": [], "x2": [], "h": []}, [], 0, None
        for t in range(STEPS):
            tok = int(rng.integers(0, emb.shape[0]))
            e = emb[tok].float()
            for src, dst in ((e, m.e), (h_card, m.h)):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(src.reshape(1, 1, 1, D), dtype=FP32, layout=TILE),
                                                dst)
            if tid is None:
                m.first()
            else:
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                m.loop.run()
            card = {k: ttnn.to_torch(m.o[k]).double().flatten()[:D] for k in ("x", "x2", "h")}
            top_i = ttnn.to_torch(m.o["top_i"]).long().flatten()[:K]
            if tid is None:
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                m.run()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
            # float64 from the same e and h
            x = eh @ torch.cat([rms(e.double(), enorm), rms(h_card.double(), hnorm)])
            a, _ = ref_mla(rms(x, an))
            x2 = x + a
            f_in = rms(x2, fnorm).to(torch.bfloat16)
            gu = torch.nn.functional.silu((gate @ f_in.double()).clamp(max=10.0)) * (up @ f_in.double()).clamp(-10, 10)
            routed = np.zeros(D, np.float32)
            lane.glm_exchange(MTP, f_in.view(torch.int16).numpy().view(np.uint16), routed)
            x3 = x2 + down @ gu + torch.from_numpy(routed).double()
            h_ref = rms(x3, shn)
            lg_ref = head_w @ h_ref
            for k, want in (("x", x), ("x2", x2), ("h", h_ref)):
                errs[k].append(float((card[k] - want).abs().max() / want.abs().max()))
            want_top = torch.topk(lg_ref, K).indices
            agree.append(len(set(top_i.tolist()) & set(want_top.tolist())))
            top1 += int(top_i[0]) == int(want_top[0])
            h_card = card["h"].float()
            if t % 10 == 0:
                print(f"  step {t}: x {errs['x'][-1]:.2e}, x2 {errs['x2'][-1]:.2e}, h {errs['h'][-1]:.2e}, "
                      f"top64 agree {agree[-1]}, top1 {int(top_i[0]) == int(want_top[0])}", flush=True)
        res["check"] = {k + "_max_rel_err_worst": max(v) for k, v in errs.items()}
        res["check"].update({k + "_max_rel_err_median": statistics.median(v) for k, v in errs.items()})
        res["check"].update({"top64_agree_min": min(agree), "top1_agree": top1 / STEPS})
        print("check:", json.dumps(res["check"]), flush=True)
        for n, key in ((5, None), (REPS, "step_us_traced")):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                m.loop.run()
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t0) / n * 1e6, 1)
        ttnn.release_trace(dev, tid)
        print(f"one MTP step traced: {res['step_us_traced']} us (the loop serving its selection and lane)", flush=True)
        if os.environ.get("MTP_PROF"):   # eager, each piece synchronized (after the trace's warm-up)
            lib = m.loop.lib
            pieces = {
                "front": lambda: (m.o.__setitem__("x", ttnn.linear(ttnn.typecast(ttnn.concat(
                    [m.norm(m.e, m.enorm), m.norm(m.h, m.hnorm)], dim=-1), BF16), m.eh, dtype=FP32,
                    compute_kernel_config=m.ck)), m.to_rm(m.norm(m.o["x"], m.attn_norm), m.hn_a)),
                "query+SEL": lambda: (m.fmq(), m.loop.clear(), m.loop.sel(m.fmq), m.loop.run()),
                "attention": lambda: m.att(m.og, m.qo),
                "out side": lambda: m.fmo(),
                "residual+norm": lambda: m.to_rm(m.norm(ttnn.add(m.o["x"], ttnn.to_layout(m.out_a, TILE)), m.ffn_norm),
                                                 m.hn_f),
                "ffn+LANE": lambda: (m.ffn(), m.loop.clear(), m.loop.lane(m.ffn, MTP), m.loop.run()),
                "head": lambda: ttnn.linear(ttnn.typecast(m.o["h"], BF16), m.head, dtype=FP32,
                                            compute_kernel_config=m.ck_head),
            }
            prof = {}
            for name, fn in pieces.items():
                ws = []
                for _ in range(6):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    fn()
                    ttnn.synchronize_device(dev)
                    ws.append((time.perf_counter() - t0) * 1e6)
                prof[name] = round(statistics.median(ws[1:]), 1)
            res["eager_us"] = prof
            print("eager pieces (us):", json.dumps(prof), flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, "g5_mtp.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "g5_mtp.json"))


if __name__ == "__main__":
    main()
