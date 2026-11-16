"""An MLA layer's decode state on the card, the real blk.3 weights, stepped from
position 0 with a fresh random hn each token: g3_fmq.py's query side (the host's side glm_loop.cpp's mla_step: the
latent normed into row p of the cache, the indexer key normed and kept with its gate, the pooled key of each pool of 4
as it closes, the top min(nv, 512) visible pools plus the open pool's tokens, cur_pos), the attention over the
selection (g3_mla_layer.MlaAttention, cur_pos on the card), v_b + o (g3_fmo.py). One trace serves every token.

Checked every token against a float64 decode of llama.cpp's glm5-next graph on the GGUF's weights from the same hn
(its own caches): the output, the pools chosen; at the end the card's latent rows and pooled keys against the
reference's. Then timed traced with the C++ loop serving.

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --memory-max $((24 << 30)) --engine tt/glm/g3_mla_decode.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g2_layers import CTX, IH, IHD, KPOOL, LN_EPS, MLA_LAYER, MODEL, OUT, RMS_EPS, Gguf, pcc  # noqa: E402
from g3_fmq import NS, NSEL, MlaQuery, _LIB, mla_state  # noqa: E402
from g3_fmo import MlaOut  # noqa: E402
from g3_mla_layer import HG, MlaAttention  # noqa: E402
from glm_loop import GlmLoop  # noqa: E402

D, NH, HK, KVL = 4096, 64, 256, 512
STEPS = int(os.environ.get("DEC_STEPS", 2200))     # past 2,048 tokens: more than 512 pools visible
REPS = int(os.environ.get("DEC_REPS", 100))


class Ref:
    """float64 decode of one MLA layer (llama.cpp's build_dsa_layer + build_kpool_select), its own caches."""

    def __init__(self, r):
        self.r = {k: v.double() for k, v in r.items()}
        self.lat, self.keys, self.gates, self.pooled, self.p = [], [], [], [], 0

    def __call__(self, hn):
        r, p = self.r, self.p
        rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean() + RMS_EPS) * w  # noqa: E731
        qr = rms(r["q_a"] @ hn, r["q_a_norm"])
        q_abs = torch.einsum("hlk,hk->hl", r["k_b"], (r["q_b"] @ qr).reshape(NH, HK))
        self.lat.append(rms(r["kv_a_mqa"] @ hn, r["kv_a_norm"]))
        ik = r["i_attn_k"] @ hn
        ik = (ik - ik.mean()) / torch.sqrt(ik.var(unbiased=False) + LN_EPS) * r["i_k_norm"] + r["i_k_norm_b"]
        self.keys.append(ik)
        self.gates.append(r["i_gate"] @ hn)
        if (p + 1) % KPOOL == 0:
            k, g = torch.stack(self.keys[-KPOOL:]), torch.stack(self.gates[-KPOOL:]) + r["i_ape"].reshape(KPOOL, IHD)
            self.pooled.append((torch.softmax(g, 0) * k).sum(0))
        nv, tail = (p + 1) // KPOOL, (p + 1) % KPOOL
        pools = torch.zeros(0, dtype=torch.long)
        if nv:
            iq = (r["i_attn_q_b"] @ qr).reshape(IH, IHD)
            w = r["i_proj"] @ hn / math.sqrt(IHD * IH)
            score = (torch.relu(iq @ torch.stack(self.pooled).T) * w[:, None]).sum(0)
            pools = torch.topk(score, min(nv, NSEL)).indices
        tok = torch.cat([(pools[:, None] * KPOOL + torch.arange(KPOOL)).flatten(), p - torch.arange(tail)])
        lat = torch.stack(self.lat)[tok]
        a = torch.softmax(q_abs @ lat.T * HK ** -0.5, dim=-1)
        o = torch.einsum("hvl,hl->hv", r["v_b"], a @ lat)
        self.p += 1
        return r["output"] @ o.flatten(), pools


def main() -> None:
    g = Gguf(MODEL)
    p = f"blk.{MLA_LAYER}."
    r = {x: g(p + f"attn_{x}.weight") for x in ("q_a", "q_b", "kv_a_mqa", "k_b", "v_b", "output", "q_a_norm",
                                                "kv_a_norm")}
    for x in ("attn_k", "attn_q_b", "proj", "k_norm"):
        r["i_" + x] = g(p + f"indexer.{x}.weight")
    r["i_k_norm_b"] = g(p + "indexer.k_norm.bias")
    r["i_gate"], r["i_ape"] = g(p + "indexer_compressor_gate.weight"), g(p + "indexer_compressor_ape.weight")
    n_pool = CTX // KPOOL
    assert STEPS + REPS + 7 <= CTX
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "layer": MLA_LAYER, "ctx": CTX, "steps": STEPS}
    try:
        put = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa: E731
        BF, RM, TILE = ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        hn = put(torch.zeros(1, 1, 1, D), BF, RM)
        K = put(torch.zeros(1, 1, n_pool, IHD), BF, TILE)
        lat = put(torch.zeros(CTX, KVL), BF, RM)
        sel = put(torch.zeros(1, NS, dtype=torch.int32), ttnn.uint32, RM)
        cpos = put(torch.zeros(1, dtype=torch.int32), ttnn.int32, RM)
        og = [put(torch.zeros(1, 1, HG, KVL), BF, TILE) for _ in range(2)]
        qo = [put(torch.zeros(1, 1, HG, KVL), BF, TILE) for _ in range(2)]
        out = put(torch.zeros(1, 1, 1, D), ttnn.float32, RM)
        state, keep = mla_state(r)
        fmq = MlaQuery(dev, r, K, n_pool, hn, og, sel, cpos, lat, state)
        att = MlaAttention(sel, cpos, lat)
        fmo = MlaOut(dev, r["v_b"], r["output"], qo[0], qo[1], out)
        loop = GlmLoop(_LIB)
        loop.sel(fmq)

        def mixer():
            fmq()
            att(og, qo)
            fmo()

        ref = Ref(r)
        rng = np.random.default_rng(int(time.time()))
        errs, pccs, agree, worst = [], [], [], (0.0, -1)
        tid = None
        for t in range(STEPS):
            h = torch.from_numpy(rng.standard_normal(D).astype(np.float32)).to(torch.bfloat16)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(h.reshape(1, 1, 1, D), dtype=BF, layout=RM), hn)
            if tid is None:
                mixer()                                   # position 0 eager (compiles), then the trace serves all
            else:
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            loop.run()
            y = ttnn.to_torch(out).double().flatten()
            if tid is None:
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                mixer()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
            y_ref, pools_ref = ref(h.double())
            e = float((y - y_ref).abs().max() / y_ref.abs().max())
            errs.append(e)
            pccs.append(pcc(y, y_ref))
            n = len(pools_ref)
            agree.append(1.0 if n == 0 else len(set(loop.pools().tolist()) & set(pools_ref.tolist())) / n)
            if e > worst[0]:
                worst = (e, t)
            if t % 500 == 0 or t == STEPS - 1:
                print(f"  p {t}: rel err {e:.2e}, pcc {pccs[-1]:.6f}, pools agree {agree[-1]:.3f}", flush=True)
        lat_c = ttnn.to_torch(lat).double()[:STEPS]
        nk = STEPS // KPOOL
        K_c = ttnn.to_torch(K).double().reshape(n_pool, IHD)[:nk]
        lat_r, K_r = torch.stack(ref.lat), torch.stack(ref.pooled)
        res["check"] = {"out_max_rel_err_median": statistics.median(errs), "out_max_rel_err_worst": worst[0],
                        "worst_at": worst[1], "out_pcc_min": min(pccs), "pools_agree_mean": float(np.mean(agree)),
                        "pools_agree_min": min(agree),
                        "lat_rows_rel_err": float((lat_c - lat_r).abs().max() / lat_r.abs().max()),
                        "pooled_keys_rel_err": float((K_c - K_r).abs().max() / K_r.abs().max()),
                        "cur_pos_last": int(ttnn.to_torch(cpos).flatten()[0])}
        print("check:", json.dumps(res["check"]), flush=True)

        # timed: the trace with the C++ loop serving, positions continuing
        for n, key in ((5, None), (REPS, "mixer_us_traced")):
            ttnn.synchronize_device(dev)
            t0, ends = time.perf_counter(), []
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                loop.run()
                ends.append(time.perf_counter())
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - t0) / n * 1e6, 1)
                res["mixer_us_median"] = round(statistics.median(np.diff(ends)) * 1e6, 1)
        ttnn.release_trace(dev, tid)
        print(f"MLA mixer at p ~{STEPS}: {res['mixer_us_traced']} us traced (median {res['mixer_us_median']})",
              flush=True)
        del keep
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, f"g3_mla_decode_ctx{CTX}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
