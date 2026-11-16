"""A probe: MLA attention for a verify pass of M rows with ttnn's stock ops, the rows as the batch.

Each row gathers its own selection (ttnn.embedding of its NS token ids from the latent cache) and attends over its
first cur_pos[m] + 1 slots with flash_multi_latent_attention_decode, q [1, M, 32, 512] per group of 32 heads, kv
[M, 1, NS, 512], cur_pos one per row. No union of the rows' selections: each row reads its own 2.2 MB.

Checked against float64 per row (random latents, ids and queries; the rows' counts differ), timed traced at M = 1..4
against today's one-row call (g3_mla_layer.MlaAttention).

  python3 tt_moeoffload/launch.py --memory-max $((16 << 30)) --engine tt/glm/g5_mla_attn_m.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g2_layers import BF16, HK, KVL, OUT, RM, TILE, traced_us  # noqa: E402
from g3_fmq import NS  # noqa: E402
from g3_mla_layer import HG, MlaAttention  # noqa: E402

CTX = 8192


def attend_m(sel, cpos, lat, qs, ck, pc, M):
    """sel uint32 [M, NS], cpos int32 [M], lat [ctx, 512], qs: the 2 head groups' queries [1, M, 32, 512]"""
    kv = ttnn.reshape(ttnn.embedding(sel, lat, layout=TILE), [M, 1, NS, KVL])
    return [ttnn.transformer.flash_multi_latent_attention_decode(
        q, kv, None, KVL, cur_pos_tensor=cpos, scale=HK ** -0.5, compute_kernel_config=ck, program_config=pc)
        for q in qs]


def main() -> None:
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    put = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev)  # noqa: E731
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ctx": CTX, "ns": NS}
    try:
        g = torch.Generator().manual_seed(3)
        lat_h = torch.randn(CTX, KVL, generator=g).to(torch.bfloat16)
        lat = put(lat_h, BF16, RM)
        one = MlaAttention(None, None, lat)
        for M in (1, 2, 3, 4):
            ids = torch.randint(0, CTX, (M, NS), generator=g, dtype=torch.int32)
            counts = torch.tensor([2048 + 1 + m for m in range(M)])      # row m sees its own 2048 + 1 + m slots
            q_h = [torch.randn(1, M, HG, KVL, generator=g).to(torch.bfloat16) for _ in range(2)]
            sel = put(ids, ttnn.uint32, RM)
            cpos = put((counts - 1).to(torch.int32), ttnn.int32, RM)
            qs = [put(q, BF16, TILE) for q in q_h]
            outs = attend_m(sel, cpos, lat, qs, one.ck, one.pc, M)
            worst = 0.0
            for grp in range(2):
                got = ttnn.to_torch(outs[grp]).double().reshape(-1)[:M * HG * KVL].reshape(M, HG, KVL)
                for m in range(M):
                    kv = lat_h[ids[m, :counts[m]].long()].double()                     # [n, 512], K = V
                    s = (q_h[grp][0, m].double() @ kv.T) * HK ** -0.5
                    want = torch.softmax(s, -1) @ kv
                    worst = max(worst, ((got[m] - want).abs().max() / want.abs().max()).item())
            us = traced_us(dev, lambda: attend_m(sel, cpos, lat, qs, one.ck, one.pc, M))
            res[f"M{M}"] = {"max_rel_err": worst, "traced_us": round(us, 1)}
            print(f"M {M}: max rel err {worst:.2e}, traced {us:.1f} us", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    with open(os.path.join(OUT, "g5_mla_attn_m.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
