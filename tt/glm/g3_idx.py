"""GLM's decode indexer: the pool scores on the card (tt/kernels/glm_idx_*.cpp, the pooled
keys split over all cores), the top 512 on the host (tt/cpulane/hc.cpp cl_top_k) over the sockets, as Flash-Next
selects (g4_model.select_words); the ids come back and are multicast. One MLA layer, one decode token, at IDX_CTX
(8K / 32K / 128K: 2,048 / 8,192 / 32,768 pools). Random bf16 inputs of the real shapes (iq 32 heads x 128, the
pooled keys, w scaled 1 / sqrt(128 * 32)), as g2_indexer.py.

Checked against float64 on the same bf16 inputs: the scores (max rel err) and the selected 512 against float64's
top 512. Timed traced with the host serving each round trip; the host's selection alone. Against G2's stock chain
(score + topk on the card: 48 / 75 / 242 us, g2_indexer.json).

  python3 tt_moeoffload/launch.py --memory-max $((16 << 30)) --env IDX_CTX=131072 --engine tt/glm/g3_idx.py
"""
from __future__ import annotations

import ctypes
import json
import os
import statistics
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
sys.path.append(os.path.join(TT, "flashnext"))                      # u1_sockets only
from g2_layers import IH, IHD, ITOP, KPOOL, OUT, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE
from fused.grid import Grid, split  # noqa: E402
from u1_sockets import pinned  # noqa: E402

CTX = int(os.environ.get("IDX_CTX", 8192))
NP = CTX // KPOOL
NSEL = ITOP // KPOOL                              # 512 pools
REPS = int(os.environ.get("IDX_REPS", 50))
_LIB = ctypes.CDLL(os.path.join(TT, "cpulane", "build", "libcpulane_glm.so"))
_LIB.cl_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


class IdxStep:
    """The indexer's program (glm_idx_*.cpp) over fixed tensors, with its own socket pair on core (0, 0) (built
    before any trace capture): K, the pooled keys (bf16 tiles [n_pools, IHD]); iqT (bf16 tiles [IHD, IH]) and w (one
    bf16 tile, row 0 the 32 head weights), which the caller fills each token. sel: the 2,048 selected token ids also go
    to sel (row-major uint32 [1, 2048]), the host replying with them instead of the 512 pool ids."""

    def __init__(self, dev, K, n_pools, sel=None):
        self.np, self.sel = n_pools, sel
        self.score_bytes, self.ids = n_pools * 4, (ITOP if sel is not None else NSEL) * 4
        where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
        self.h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, 4 * self.ids,
                                                  ttnn.H2DMode.DEVICE_PULL))
        self.h2d.set_page_size(self.ids)
        self.d2h = pinned(lambda: ttnn.D2HSocket(dev, where, 2 * self.score_bytes))
        self.d2h.set_page_size(self.score_bytes)
        tile = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
        self.K, self.iqT, self.w = K, tile(torch.zeros(1, 1, IHD, IH)), tile(torch.zeros(1, 1, 1, 32))
        g = Grid(dev)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        rargs, wargs, crt = {}, {}, ttnn.RuntimeArgs()
        for idx, ((cx, cy), (t0, nt)) in enumerate(zip(g.cores, split(n_pools // 32, len(g.cores)))):
            rargs[(cx, cy)] = g.args(idx) + [K.buffer_address(), self.iqT.buffer_address(), self.w.buffer_address(),
                                             t0, nt]
            wargs[(cx, cy)] = g.args(idx) + [t0, nt]
            crt[cx][cy] = [nt]
        defines = [("H2D_CONFIG", str(self.h2d.get_config_buffer_address())),
                   ("D2H_CONFIG", str(self.d2h.get_config_buffer_address())), ("IDX_IDS", str(self.ids))]
        ct = [self.score_bytes] + acc(K) + acc(self.iqT) + acc(self.w)
        if sel is not None:
            defines.append(("IDX_SEL", str(sel.buffer_address())))
            ct += acc(sel)
        kernels = g.dm_kernels(os.path.join(KDIR, "glm_idx_reader.cpp"), os.path.join(KDIR, "glm_idx_writer.cpp"),
                               ct, [], rargs, wargs, defines=defines)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 64
        kernels.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "glm_idx_compute.cpp"),
                                             core_ranges=g.ranges, compile_time_args=[], runtime_args=crt,
                                             config=cfg))
        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [g.cb(0, bf, 2048, 8), g.cb(1, bf, 2048, 4), g.cb(2, bf, 2048, 1), g.cb(3, f32, 4096, 2),
               g.cb(4, f32, 4096, 2), g.cb(8, ttnn.uint8, self.score_bytes + self.ids, 1), g.cb(16, f32, 4096, 2)]
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=g.semaphores(), cbs=cbs)
        self.ios = [K, self.iqT, self.w] + ([sel] if sel is not None else [])
        self.page = torch.zeros(self.score_bytes, dtype=torch.uint8)
        self.reply = torch.zeros(self.ids, dtype=torch.uint8)
        self.page_np, self.reply_np = self.page.numpy(), self.reply.numpy()
        self.pools = np.zeros(NSEL, np.uint32)

    def run(self):
        ttnn.generic_op(self.ios, self.prog)

    def select(self):
        """The host's top 512 over the scores in page_np (into self.pools), expanded to token ids when sel is set."""
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        _LIB.cl_top_k(p(self.page_np), self.np, NSEL, p(self.pools if self.sel is not None else self.reply_np))
        if self.sel is not None:
            tok = self.reply_np.view(np.uint32).reshape(NSEL, KPOOL)
            tok[:] = self.pools[:, None] * KPOOL + np.arange(KPOOL, dtype=np.uint32)

    def serve(self):
        """The host's side of a launch: the scores in, the selection out."""
        self.d2h.read_tensor(self.page)
        self.select()
        self.h2d.write_tensor(self.reply)


def main() -> None:
    gen = torch.Generator().manual_seed(12)
    iq = torch.randn(IH, IHD, generator=gen).bfloat16()
    K = torch.randn(NP, IHD, generator=gen).bfloat16()
    w = (torch.randn(IH, generator=gen) / (IHD * IH) ** 0.5).bfloat16()
    ref = (torch.relu(K.double() @ iq.double().T) * w.double()).sum(1)            # [NP]
    ref_top = set(torch.topk(ref, NSEL).indices.tolist())
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "ctx": CTX, "pools": NP}
    try:
        tile = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
        ix = IdxStep(dev, tile(K.reshape(1, 1, NP, IHD)), NP)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(iq.T.contiguous().reshape(1, 1, IHD, IH), dtype=ttnn.bfloat16,
                                                        layout=ttnn.TILE_LAYOUT), ix.iqT)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(w.reshape(1, 1, 1, 32), dtype=ttnn.bfloat16,
                                                        layout=ttnn.TILE_LAYOUT), ix.w)
        page_np, reply_np, serve = ix.page_np, ix.reply_np, ix.serve
        ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731

        ix.run()
        serve()
        ttnn.synchronize_device(dev)
        got = torch.from_numpy(page_np.view(np.float32).copy()).double()
        sel = set(reply_np.view(np.uint32).tolist())
        res["check"] = {"score_max_rel_err": float((got - ref).abs().max() / ref.abs().max()),
                        "score_pcc": round(pcc(got, ref), 7), "top512_agree_with_f64": len(sel & ref_top)}
        print("check:", json.dumps(res["check"]), flush=True)
        t = []
        for _ in range(REPS):
            s = time.perf_counter()
            _LIB.cl_top_k(ptr(page_np), NP, NSEL, ptr(reply_np))
            t.append((time.perf_counter() - s) * 1e6)
        res["host_top512_us"] = round(statistics.median(t), 1)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ix.run()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        for n, key in ((5, None), (REPS, "layer_us_traced")):
            ttnn.synchronize_device(dev)
            s = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                serve()
            ttnn.synchronize_device(dev)
            if key:
                res[key] = round((time.perf_counter() - s) / n * 1e6, 1)
        ttnn.release_trace(dev, tid)
        g2 = json.load(open(os.path.join(OUT, "g2_indexer.json"))).get(f"ctx{CTX}", {}).get("chain", {}).get("us")
        res["g2_chain_us"] = g2
        print(f"ctx {CTX}: {res['layer_us_traced']} us a layer traced with the host's top 512 "
              f"({res['host_top512_us']} us of it on the host); G2's stock chain {g2} us", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    json.dump(res, open(os.path.join(OUT, f"g3_idx_ctx{CTX}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
