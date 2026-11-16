"""A whole GLM token on the card + the CPU lane, one trace from the embedding to the
logits, the host's round trips served by the C++ loop (glm_loop.cpp), teacher-forced through llama.cpp's own token ids
against its logits and per-layer streams ($TT_MOEOFFLOAD_DATA/glm_ref, tools/llama's dump-logits on llama.cpp).

The trunk (llama.cpp's glm5-next graph; blk.45 is the MTP block, not run here): the embedding (the host's row of
token_embd) into the 4 streams; per layer il < 45: mHC pre with the attention norm, the KDA mixer (g3_fkda.Mixer, its
conv and recurrent states in place) or, at il % 4 == 3, the MLA mixer (g3_fmq.MlaQuery with the host's decode state,
g3_mla_layer.MlaAttention, g3_fmo.MlaOut), mHC post; mHC pre with the FFN norm, the FFN (g3_fsh.FfnFused: blk.0-2's
dense FFN alone, else the shared expert with the routed experts on the lane), mHC post. Each post writes the next
site's streams. Then the streams' mean, output_norm and the LM head (G2's DramMM, fp32 logits).

Round trips a token: 90 mHC sites, 11 MLA selections, 42 lane exchanges, through 3 socket pairs on core (0, 0).

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((320 << 30)) --engine tt/glm/g4_token.py

Environment: GT_LAYERS=n (the first n layers only, no head: the bisect), GT_TOKENS (positions, default all of the
reference), GT_REPS (traced tokens timed after the parity, default 50), GT_REF (the reference folder), GT_FROM (the
first position its logits.f32 holds: dump-logits --dump-from; the long reference $TT_MOEOFFLOAD_DATA/glm_ref_long).
The per-layer streams are compared where the reference has them (tensors/l_out-<il>.f32).
"""
from __future__ import annotations

import hashlib
import json
import os
DATA = os.environ.get("TT_MOEOFFLOAD_DATA", os.path.expanduser("~/.cache/tt-moeoffload"))   # the data root: caches, references, models
import statistics
import sys
import time

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g3_overlap import cpulane  # noqa: E402  (sets CL_LIB to the GLM lane first)
from g2_layers import BF16, CTX, D, FP32, HC, IHD, KPOOL, MODEL, OUT, RM, RMS_EPS, TILE, BFP8, Card, Gguf  # noqa
from g3_hc import HcSite, sockets  # noqa: E402
from g3_fkda import CONV, HK as KHK, NV, Mixer  # noqa: E402
from g3_fmq import NS, MlaQuery, _LIB, mla_state  # noqa: E402
from g3_fmo import MlaOut  # noqa: E402
from g3_fsh import FfnFused  # noqa: E402
from g3_mla_layer import HG, MlaAttention  # noqa: E402
from glm_loop import GlmLoop  # noqa: E402
from fused.grid import Grid  # noqa: E402
from fused.mrows import step_words  # noqa: E402
from g5_mla_attn_m import attend_m  # noqa: E402

N_LAYER, N_DENSE, KVL, VOCAB = 45, 3, 512, 154880
PF_PAGES = int(os.environ.get("GT_PF", 150))   # prefetched pages a core (0: off): ~160 KB beside the KDA mixer
STALL_S = 120                                         # a position slower than this: a program failed, exit
REF = os.path.expanduser(os.environ.get("GT_REF", os.path.join(DATA, "glm_ref")))
LAYERS = int(os.environ.get("GT_LAYERS", N_LAYER))
REPS = int(os.environ.get("GT_REPS", 50))
FROM = int(os.environ.get("GT_FROM", 0))


def is_mla(il: int) -> bool:
    return il % 4 == 3


class Token:
    """The trunk's programs over the first n_layer layers (all: with the head). run() enqueues one token (the trace
    captures it); the GlmLoop serves its round trips; set_embedding(id) writes the first site's streams."""

    def __init__(self, dev, g: Gguf, lane, n_layer=N_LAYER, ctx=CTX, m_max=1):
        """m_max > 1: a verify pass's rows: every program's M-row build, one step tensor (self.step, words
        1..4 M, n_keep, parity, mp: set_pass), row m's streams in rows 4 m + s, the head's top 64 a row on the card."""
        self.dev, self.g, self.n_layer, self.m_max = dev, g, n_layer, m_max
        put = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa: E731
        self.put = put
        hc_h2d, hc_d2h = sockets(dev, m_max)                   # every socket before any program is built or traced
        mm = dict(m_max=m_max, step=None)
        if m_max > 1:
            self.step = put(step_words(1), ttnn.uint32, RM)
            mm["step"] = self.step
        self.grid = grid = Grid(dev)
        # the streams between sites: xs[2 il] into layer il's attention site, xs[2 il + 1] its FFN site's
        self.xs = [put(torch.zeros(1, 1, 32, D), FP32, TILE) for _ in range(2 * n_layer + 1)]
        n_pool = ctx // KPOOL
        self.sel = put(torch.zeros(m_max, NS, dtype=torch.int32), ttnn.uint32, RM)      # shared by the MLA layers
        self.cpos = put(torch.zeros(m_max, dtype=torch.int32), ttnn.int32, RM)
        self.og = [put(torch.zeros(1, m_max, HG, KVL), BF16, TILE) for _ in range(2)]
        self.qo = [put(torch.zeros(1, m_max, HG, KVL), BF16, TILE) for _ in range(2)]
        self.mla_states = []
        self.keep, self.layers = [], []
        sel_link = lane_link = None
        loop = self.loop = GlmLoop(_LIB)
        _LIB.gl_mla_clear()
        t0 = time.perf_counter()
        # the prefetch buffer (every program runs beside it): each MoE layer's FFN program fills it, while the lane
        # works, with the next layer's mixer's first PF_PAGES pages a core (its phase A's weights)
        self.pf = None
        if PF_PAGES and m_max == 1:   # (the M-row KDA program's CBs leave no room for it)
            rows = PF_PAGES * 1088 // 64 + 32 - (PF_PAGES * 1088 // 64) % 32
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(grid.ranges, (rows, 32), ttnn.ShardOrientation.ROW_MAJOR))
            self.pf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows * len(grid), 32]), BF16, TILE, dev, mc)
        pf_in = lambda il: (self.pf, PF_PAGES) if self.pf is not None and il - 1 >= N_DENSE else None  # noqa: E731
        w_of = lambda il: (lambda n: g(f"blk.{il}." + n))  # noqa: E731
        hc_of = lambda il, site: tuple(w_of(il)(f"hc_{site}_{x}.weight") for x in ("fn", "scale", "base"))  # noqa
        # pass 1: every layer's attention half (the mixers first: an FFN program prefetches for the next one)
        for il in range(n_layer):
            w = w_of(il)
            hn_a, hn_f = (put(torch.zeros(1, 1, 1, m_max * D), BF16, RM) for _ in range(2))
            hs_a = HcSite(dev, grid, *hc_of(il, "attn"), hc_h2d, hc_d2h, hn=hn_a, norm=w("attn_norm.weight"),
                          xn=self.xs[2 * il + 1], x=self.xs[2 * il], **mm)
            L = {"hs_a": hs_a, "hn_f": hn_f}
            if is_mla(il):
                r = {x: w(f"attn_{x}.weight") for x in ("q_a", "q_b", "kv_a_mqa", "k_b", "v_b", "output", "q_a_norm",
                                                        "kv_a_norm")}
                for x in ("attn_k", "attn_q_b", "proj", "k_norm"):
                    r["i_" + x] = w(f"indexer.{x}.weight")
                r["i_k_norm_b"] = w("indexer.k_norm.bias")
                r["i_gate"], r["i_ape"] = w("indexer_compressor_gate.weight"), w("indexer_compressor_ape.weight")
                K = put(torch.zeros(1, 1, n_pool, IHD), BF16, TILE)
                lat = put(torch.zeros(ctx, KVL), BF16, RM)
                state, ws = mla_state(r)
                self.keep.append(ws)
                self.mla_states.append(state)
                L["fmq"] = MlaQuery(dev, r, K, n_pool, hn_a, self.og, self.sel, self.cpos, lat, state,
                                    link=sel_link, check=False, pf=pf_in(il), **mm)
                sel_link = (L["fmq"].h2d, L["fmq"].d2h)
                L["att"] = MlaAttention(self.sel, self.cpos, lat)
                L["fmo"] = MlaOut(dev, r["v_b"], r["output"], self.qo[0], self.qo[1], hs_a.out, check=False, **mm)
                L["K"], L["lat"], L["mixer"] = K, lat, L["fmq"]
                del r
            else:
                states = (torch.zeros(3, CONV), torch.zeros(NV, KHK, KHK))
                m = Mixer(dev, g, None, states=states, out=hs_a.out, in_place=True, hn=hn_a, layer=il, check=False,
                          pf=pf_in(il), **mm)
                L["kda"], L["kda_run"], L["mixer"] = m, m.program(), m
            self.layers.append(L)
            if il % 5 == 4:
                print(f"  built layer {il}'s attention half ({time.perf_counter() - t0:.0f} s)", flush=True)
        # pass 2: every layer's FFN half, a MoE layer's prefetching for the next layer's mixer
        for il, L in enumerate(self.layers):
            w, hn_f = w_of(il), L["hn_f"]
            hs_f = HcSite(dev, grid, *hc_of(il, "ffn"), hc_h2d, hc_d2h, hn=hn_f, norm=w("ffn_norm.weight"),
                          xn=self.xs[2 * il + 2], x=self.xs[2 * il + 1], **mm)
            L["hs_f"] = hs_f
            if il < N_DENSE:
                L["ffn"] = FfnFused(dev, *(w(f"ffn_{n}.weight") for n in ("gate", "up", "down")), hn_f, hs_f.out,
                                    lane=False, check=False, **mm)
            else:
                lane.router_load(il, w("ffn_gate_inp.weight").numpy(), w("exp_probs_b.bias").numpy())
                nxt = self.layers[il + 1]["mixer"] if il + 1 < n_layer and self.pf is not None else None
                L["ffn"] = FfnFused(dev, *(w(f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down")), hn_f,
                                    hs_f.out, lane=True, link=lane_link, check=False,
                                    pf=(self.pf, PF_PAGES, nxt) if nxt is not None else None, **mm)
                lane_link = (L["ffn"].h2d, L["ffn"].d2h)
            if il % 5 == 4:
                print(f"  built layer {il}'s FFN half ({time.perf_counter() - t0:.0f} s)", flush=True)
        for il, L in enumerate(self.layers):                   # the host's round trips in launch order
            loop.hc(L["hs_a"])
            if is_mla(il):
                loop.sel(L["fmq"])
            loop.hc(L["hs_f"])
            if il >= N_DENSE:
                loop.lane(L["ffn"], il)
        self.head = None
        if n_layer == N_LAYER:
            self.out_norm = put(g("output_norm.weight").reshape(1, 1, 1, D).to(torch.bfloat16), BF16, TILE)
            # shortcut: stock ttnn.linear, fp32 logits, 1.93 ms traced (G2's DRAM-sharded DramMM outgrows L1 at this
            # width); a 110-core GEMV with the top k on the card when the token's time is ranked
            self.head = put(g("output.weight").T.contiguous().to(torch.bfloat16), BFP8, TILE)   # [4096, 154880]
            self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                       fp32_dest_acc_en=True, packer_l1_acc=True)
            self.logits = None
        emb = g("token_embd.weight")                                       # [154880, 4096] float32 on the host
        self.emb = emb.numpy()
        self.x0 = torch.zeros(1, 1, 32, D)
        self.build_s = round(time.perf_counter() - t0, 1)

    def set_pass(self, toks, n_keep: int, parity: int) -> None:
        """a verify pass (m_max > 1): row m's streams = token toks[m]'s embedding (rows past M: the last token's,
        finite), the step words, the host's rows"""
        M = len(toks)
        for m in range(self.m_max):
            self.x0[0, 0, 4 * m:4 * m + HC] = torch.from_numpy(self.emb[toks[min(m, M - 1)]])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self.x0, dtype=FP32, layout=TILE), self.xs[0])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(step_words(M, n_keep, parity), dtype=ttnn.uint32, layout=RM),
                                        self.step)
        assert _LIB.gl_rows(M) == 0

    def set_embedding(self, tok: int) -> None:
        self.x0[0, 0, :HC] = torch.from_numpy(self.emb[tok])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self.x0, dtype=FP32, layout=TILE), self.xs[0])

    def run(self) -> None:
        for il, L in enumerate(self.layers):
            hs_a, hs_f = L["hs_a"], L["hs_f"]
            hs_a.run(hs_a.pre_prog)
            if "kda" in L:
                L["kda_run"]()
            elif self.m_max > 1:   # the rows as flash-MLA's batch (g5_mla_attn_m.py)
                L["fmq"]()
                att = L["att"]
                for src, dst in zip(attend_m(self.sel, self.cpos, att.lat, self.og, att.ck, att.pc, self.m_max),
                                    self.qo):
                    ttnn.copy(src, dst)
                L["fmo"]()
            else:
                L["fmq"]()
                L["att"](self.og, self.qo)
                L["fmo"]()
            hs_a.run(hs_a.post_prog)
            hs_f.run(hs_f.pre_prog)
            L["ffn"]()
            hs_f.run(hs_f.post_prog)
        if self.head is not None and self.m_max > 1:   # row m's streams in rows 4 m + s: their mean, the head, the top 64
            x = ttnn.reshape(ttnn.slice(self.xs[-1], [0, 0, 0, 0], [1, 1, HC * self.m_max, D]), [1, self.m_max, HC, D])
            h = ttnn.reshape(ttnn.multiply(ttnn.sum(x, dim=2, keepdim=True), 1.0 / HC), [1, 1, self.m_max, D])
            self.hn_rows = hn = ttnn.rms_norm(h, weight=self.out_norm, epsilon=RMS_EPS, compute_kernel_config=self.ck)
            self.logits = ttnn.linear(ttnn.typecast(hn, BF16), self.head, dtype=FP32, compute_kernel_config=self.ck)
            self.top_v, self.top_i = ttnn.topk(ttnn.typecast(self.logits, BF16), 64, dim=-1, largest=True, sorted=True)
        elif self.head is not None:
            x = ttnn.slice(self.xs[-1], [0, 0, 0, 0], [1, 1, HC, D])
            h = ttnn.multiply(ttnn.sum(x, dim=2, keepdim=True), 1.0 / HC)
            hn = ttnn.rms_norm(h, weight=self.out_norm, epsilon=RMS_EPS, compute_kernel_config=self.ck)   # fp32 in
            self.logits = ttnn.linear(ttnn.typecast(hn, BF16), self.head, dtype=FP32, compute_kernel_config=self.ck)

    def streams(self, il: int) -> torch.Tensor:
        """layer il's output streams [HC, D] (after the trace ran)"""
        return ttnn.to_torch(self.xs[2 * il + 2]).double().reshape(32, D)[:HC]


def main() -> None:
    lane = cpulane.Lane()                # the lane first (g3_fsh.py's trap)
    t0 = time.perf_counter()
    lane.load(MODEL, layers=set(range(N_DENSE, min(LAYERS, N_LAYER))))
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layers": LAYERS, "ctx": CTX, "ref": REF, "from": FROM,
           "lane_load_s": round(time.perf_counter() - t0, 1)}
    ids = np.fromfile(os.path.join(REF, "ids.i32"), np.int32)
    n_tok = min(int(os.environ.get("GT_TOKENS", len(ids))), len(ids))
    ref_logits = np.memmap(os.path.join(REF, "logits.f32"), np.float32, "r").reshape(len(ids) - FROM, VOCAB)
    have_streams = os.path.exists(os.path.join(REF, "tensors", "l_out-0.f32"))
    lout = {il: np.memmap(os.path.join(REF, "tensors", f"l_out-{il}.f32"), np.float32, "r").reshape(len(ids), HC, D)
            for il in range(min(LAYERS, N_LAYER))} if have_streams else {}
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=1 << 30)
    try:
        g = Gguf(MODEL)
        tok = Token(dev, g, lane, n_layer=min(LAYERS, N_LAYER))
        res["build_s"] = tok.build_s
        print(f"built in {tok.build_s} s (the lane's experts {res['lane_load_s']} s)", flush=True)
        loop, tid = tok.loop, None
        rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())  # noqa: E731
        layer_err, kl, top1, worst, digests = [], [], [], [], []
        # ttnn reports a failed program from its dispatch thread; the loop then waits forever on a round trip
        import threading
        beat = [time.monotonic()]

        def watchdog():
            while True:
                time.sleep(5)
                if time.monotonic() - beat[0] > STALL_S:
                    print(f"watchdog: no progress in {STALL_S} s, exiting", flush=True)
                    os._exit(2)
        threading.Thread(target=watchdog, daemon=True).start()
        for t in range(n_tok):
            beat[0] = time.monotonic()
            tok.set_embedding(int(ids[t]))
            if tid is None:
                tok.run()                                  # position 0 eager (compiles), then one trace serves all
            else:
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            loop.run()
            errs = [rel(tok.streams(il).numpy(), lout[il][t].astype(np.float64)) for il in range(tok.n_layer)] \
                if lout else [0.0]
            layer_err.append(errs)
            if tok.head is not None and t >= FROM:
                lt = ttnn.to_torch(tok.logits).flatten()[:VOCAB]
                digests.append(hashlib.sha1(lt.float().numpy().tobytes()).hexdigest()[:16])   # a refactor's bitwise check
                lg = lt.double()
                rf = torch.from_numpy(ref_logits[t - FROM].astype(np.float64))
                lp, lq = torch.log_softmax(rf, -1), torch.log_softmax(lg, -1)
                kl.append(float((lp.exp() * (lp - lq)).sum()))
                top1.append(bool(rf.argmax() == lg.argmax()))
            if tid is None:
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                tok.run()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
            worst_il = int(np.argmax(errs))
            worst.append((errs[worst_il], worst_il))
            if (t % 10 == 0 and (lout or t >= FROM)) or t % 500 == 0 or t == n_tok - 1:
                msg = f"  p {t}: streams rel err last layer {errs[-1]:.2e} (worst {errs[worst_il]:.2e} at {worst_il})"
                if kl:
                    msg += f", KL {kl[-1]:.4f}, top-1 {top1[-1]}"
                print(msg, flush=True)
        le = np.array(layer_err)
        res["check"] = {"positions": n_tok, "streams_rel_err_median_by_layer": [round(float(x), 5) for x in
                                                                                  np.median(le, 0)],
                        "streams_rel_err_max_last_layer": float(le[:, -1].max()),
                        "streams_rel_err": [[round(float(x), 5) for x in row] for row in le]}
        if digests:
            res["logits_sha1"] = hashlib.sha1("".join(digests).encode()).hexdigest()
            res["logits_sha1_by_position"] = digests
        if kl:
            res["check"].update({"kl_mean": float(np.mean(kl)), "kl_p99": float(np.percentile(kl, 99)),
                                 "kl_max": float(np.max(kl)), "top1": float(np.mean(top1)),
                                 "kl_median": float(np.median(kl)), "kl": [round(x, 5) for x in kl],
                                 "top1_by_position": top1,
                                 "kl_over_0.1_at": [FROM + i for i, x in enumerate(kl) if x > 0.1]})
        print("check:", json.dumps({k: v for k, v in res["check"].items() if k not in (
            "streams_rel_err_median_by_layer", "streams_rel_err", "kl", "top1_by_position")}),
              flush=True)
        print("  streams rel err by layer (median):", res["check"]["streams_rel_err_median_by_layer"], flush=True)

        # timed: the trace with the C++ loop serving, positions continuing
        os.sched_setaffinity(0, {lane.cpus[0]})
        for n, key in ((3, None), (REPS, "token_ms")):
            ttnn.synchronize_device(dev)
            t1, ends = time.perf_counter(), []
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                loop.run()
                ends.append(time.perf_counter())
                beat[0] = time.monotonic()
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t1) / n * 1e3, 2)
                res["token_ms_median"] = round(statistics.median(np.diff(ends)) * 1e3, 2)
        ttnn.release_trace(dev, tid)
        print(f"token ({tok.n_layer} layers{', head' if tok.head else ''}): {res['token_ms']} ms traced "
              f"(median {res['token_ms_median']})", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g4_token_l{LAYERS}{'_long' if FROM else ''}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # a failure mid-token leaves programs waiting on round trips nobody serves: closing the device would hang
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
