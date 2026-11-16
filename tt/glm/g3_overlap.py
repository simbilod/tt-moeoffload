"""Does the card overlap with the CPU lane as the ledger prices it? Stock ttnn ops, the real
GLM-shaped lane.

A layer here is GLM's KDA layer's big matmuls at their real sizes (bfp8_b): q, k, v (4096 x 8192 each), o (8192 x
4096), the router (4096 x 288), the shared expert (gate|up 4096 x 4096, down 2048 x 4096), and the routed experts on
the CPU lane (tt/cpulane, CL_SHAPE=glm, layer 10's experts, 8 fresh random routes a call). Two weight sets alternate,
so a prefetch always copies the next layer's own weights. Inputs are fixed rows: this times the schedule, not the
model's math.

  serial   the dense part (all from DRAM), the router, its logits to the host, the lane, the routed sum back, the
           shared expert, the add
  overlap  the dense part with OV_L1's matrices from L1 (copied there during the previous layer's lane); after the
           logits are read, the shared expert and the next layer's L1 copies are queued on the card, and the host
           runs the lane while the card works through them

Per mode: wall a layer (median over OV_LAYERS layers); alone: the dense part, the lane, the card's work during the
lane. Check: the L1 matmuls' outputs against the DRAM ones (PCC).

  CL_SHAPE=glm tt/cpulane/build.sh   (in the container: tt/env/setup.sh builds it)
  python3 tt_moeoffload/launch.py --cpus all --memory-max $((40 << 30)) --engine tt/glm/g3_overlap.py
"""
from __future__ import annotations

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
os.environ.setdefault("CL_LIB", os.path.join(HERE, "..", "cpulane", "build", "libcpulane_glm.so"))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
sys.path.append(os.path.join(os.path.dirname(HERE), "flashnext"))   # u1_sockets only: GLM's own g2_layers comes first
from g2_layers import BF16, BFP8, FP32, MM_CK, OUT, RM, TILE, Card, pcc  # noqa: E402
assert os.path.dirname(sys.modules["g2_layers"].__file__) == HERE, "GLM's g2_layers, not Flash-Next's"
import cpulane  # noqa: E402
from u1_sockets import pinned  # noqa: E402

MODEL = os.environ.get("TT_MOEOFFLOAD_MODEL_DIR") or os.path.join(DATA, "models/GLM-5.3-Flash Maya-L")   # engine.py --model-dir sets it
LAYERS = int(os.environ.get("OV_LAYERS", 40))
L1_SET = [v for v in os.environ.get("OV_L1", "q").split(",") if v]
L1_CORES = 64
LANE_LAYER, D, NX, K = 10, 4096, 288, 8
SHAPES = {"q": (4096, 8192), "k": (4096, 8192), "v": (4096, 8192), "o": (8192, 4096)}
KDIR = os.path.join(os.path.dirname(HERE), "kernels")
PAGE_OUT, PAGE_IN = (D + 512) * 4, D * 4   # hn + the router's sigmoids (fp32, 288 padded to 512); the experts' sum
FIFO_PAGES = 4                             # each socket's FIFO a whole number of its pages


class Xchg:
    """The socket hand-off on core (0, 0): glm_xsend.cpp sends `page` (an L1 tensor on that core), glm_xrecv.cpp
    reads the host's reply into `reply`; each its own program, so the card's queue runs other work in between."""

    def __init__(self, dev):
        core = ttnn.CoreCoord(0, 0)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core)
        self.h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, FIFO_PAGES * PAGE_IN,
                                                  ttnn.H2DMode.DEVICE_PULL))
        self.h2d.set_page_size(PAGE_IN)
        self.d2h = pinned(lambda: ttnn.D2HSocket(dev, where, FIFO_PAGES * PAGE_OUT))
        self.d2h.set_page_size(PAGE_OUT)
        mc = lambda w: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,  # noqa: E731
                                         ttnn.ShardSpec(self.cores, (1, w), ttnn.ShardOrientation.ROW_MAJOR))
        self.page = ttnn.from_torch(torch.zeros(1, 1, 1, D + 512), device=dev, layout=RM, dtype=FP32,
                                    memory_config=mc(D + 512))
        self.reply = ttnn.from_torch(torch.zeros(1, 1, 1, D), device=dev, layout=RM, dtype=FP32, memory_config=mc(D))
        self.host_out = torch.zeros(PAGE_OUT, dtype=torch.uint8)
        self.host_in = torch.zeros(PAGE_IN, dtype=torch.uint8)

    def _prog(self, src: str, config: int, page: int, addr: int) -> None:
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [addr]
        k = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=[config, page], runtime_args=rt,
                                  config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op([self.page, self.reply], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))

    def send(self) -> None:
        self._prog("glm_xsend.cpp", self.d2h.get_config_buffer_address(), PAGE_OUT, self.page.buffer_address())

    def recv(self) -> None:
        self._prog("glm_xrecv.cpp", self.h2d.get_config_buffer_address(), PAGE_IN, self.reply.buffer_address())


class L1Slot:
    """A matrix's place in L1 (width-sharded over L1_CORES) and its 1-D multicast matmul."""

    def __init__(self, dev, k: int, n: int):
        grid = ttnn.num_cores_to_corerangeset(L1_CORES, dev.compute_with_storage_grid_size(), True)
        self.mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                    ttnn.ShardSpec(grid, (k, n // L1_CORES), ttnn.ShardOrientation.ROW_MAJOR))
        per_n = n // 32 // L1_CORES
        self.pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), in0_block_w=8, out_subblock_h=1,
            out_subblock_w=max(d for d in range(1, 9) if per_n % d == 0), per_core_M=1, per_core_N=per_n,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        self.omc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                     ttnn.ShardSpec(grid, (32, n // L1_CORES), ttnn.ShardOrientation.ROW_MAJOR))

    def __call__(self, x32, w_l1):
        return ttnn.matmul(x32, w_l1, program_config=self.pc, memory_config=self.omc, compute_kernel_config=MM_CK)


class Layer:
    def __init__(self, card: Card, g: torch.Generator):
        self.w = {n: torch.randn(*s, generator=g) / s[0] ** 0.5 for n, s in SHAPES.items()}
        self.dram = {n: card.mm(w) for n, w in self.w.items()}                       # G2's DRAM-sharded matmuls
        self.src = {n: ttnn.from_torch(self.w[n], device=card.dev, layout=TILE, dtype=BFP8,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG) for n in L1_SET}   # copy sources
        self.router = card.mm(torch.randn(D, NX, generator=g) / 64, BF16)
        self.sh_gu = card.mm(torch.randn(D, 4096, generator=g) / 64)
        self.sh_down = card.mm(torch.randn(2048, D, generator=g) / 45)


def capture(dev, fn):
    """fn's ops as a trace (run once first to compile); -> (trace id, fn's outputs, held while the trace lives)."""
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    return tid, out


def main() -> None:
    lane = cpulane.Lane()
    lane.load(MODEL, layers={LANE_LAYER})
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    card = Card(dev)
    import threading
    me = threading.get_native_id()
    fenced = lane.fence_others(keep={me})         # tt-metal's own threads (some spin) off the lane's CPUs
    os.sched_setaffinity(0, {lane.cpus[0]})       # the caller is the lane's thread 0: its own CPU, not the crowd
    g = torch.Generator().manual_seed(11)
    rng = np.random.default_rng(11)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "l1_set": L1_SET, "layers": LAYERS,
           "fenced_threads": len(fenced)}
    try:
        layers = [Layer(card, g), Layer(card, g)]
        slots = {n: L1Slot(dev, *SHAPES[n]) for n in L1_SET}
        X = {n: ttnn.to_memory_config(layers[0].src[n], slots[n].mc) for n in L1_SET}   # one L1 buffer a matrix
        rows = {k: card.row(torch.randn(k, generator=g)) for k in (2048, 4096, 8192)}
        x32 = {k: ttnn.from_torch(torch.nn.functional.pad(torch.randn(1, k, generator=g), (0, 0, 0, 31))
                                  .reshape(1, 1, 32, k), device=dev, layout=TILE, dtype=BF16,
                                  memory_config=ttnn.L1_MEMORY_CONFIG) for k in (4096, 8192)}
        routed_dev = ttnn.from_torch(torch.zeros(1, 1, 1, D), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=BF16)
        # the sockets and their L1 pages before any program is built or traced: a trace replays its programs' static
        # circular buffers without the clash check, so L1 allocated after a capture can be overwritten by them
        xc = Xchg(dev)
        hn_dev = rows[4096]
        x_host = rng.standard_normal(D).astype(np.float32)
        out_host = np.zeros(D, np.float32)

        def dense(L, use_l1):
            outs = [slots[n](x32[k], X[n]) if use_l1 and n in X else L.dram[n](rows[k])
                    for n, (k, _) in SHAPES.items()]
            return outs, L.router(rows[4096])

        def shexp(L):
            gu = L.sh_gu(rows[4096])
            return L.sh_down(ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 1, 2048])),
                                           ttnn.slice(gu, [0, 0, 0, 2048], [1, 1, 1, 4096])))

        def during(L, nxt):
            sh = shexp(L)
            for n in L1_SET:
                ttnn.copy(nxt.src[n], X[n])
            return sh

        # check: an L1 matmul against torch, with the weights put there by the traced copy's op
        ttnn.copy(layers[1].src[L1_SET[0]], X[L1_SET[0]])
        n0 = L1_SET[0]
        k0 = SHAPES[n0][0]
        b = ttnn.to_torch(ttnn.sharded_to_interleaved(slots[n0](x32[k0], X[n0]), ttnn.DRAM_MEMORY_CONFIG)).float()
        res["check"] = {n0: round(pcc(b.reshape(32, -1)[0], ttnn.to_torch(x32[k0]).float().reshape(32, k0)[0]
                                      @ layers[1].w[n0]), 6)}
        print("check:", json.dumps(res["check"]), flush=True)

        T = {}
        for i, L in enumerate(layers):
            nxt = layers[1 - i]
            T["dense_l1", i] = capture(dev, lambda L=L: dense(L, True))
            T["dense_dram", i] = capture(dev, lambda L=L: dense(L, False))
            T["sh", i] = capture(dev, lambda L=L: shexp(L))
            T["during", i] = capture(dev, lambda L=L, nxt=nxt: during(L, nxt))
            T["add_sh", i] = capture(dev, lambda i=i: ttnn.add(T["sh", i][1], ttnn.to_layout(routed_dev, TILE)))
            T["add_during", i] = capture(dev, lambda i=i: ttnn.add(T["during", i][1],
                                                                   ttnn.to_layout(routed_dev, TILE)))

        def dense_send(L):
            outs, logits = dense(L, True)
            probs = ttnn.pad(ttnn.sigmoid(logits), [(0, 0), (0, 0), (0, 0), (0, 512 - NX)], 0.0)
            row = ttnn.concat([ttnn.typecast(hn_dev, FP32), ttnn.typecast(probs, FP32)], dim=-1)
            ttnn.copy(ttnn.to_layout(row, RM), xc.page)
            xc.send()
            return outs

        def during_recv(L, nxt):
            sh = during(L, nxt)
            xc.recv()
            r = ttnn.typecast(ttnn.to_layout(ttnn.to_memory_config(xc.reply, ttnn.DRAM_MEMORY_CONFIG), TILE), BF16)
            return ttnn.add(sh, r)

        def host_xchg(dbg=False):
            xc.d2h.read_tensor(xc.host_out)                                  # waits for the card's page
            if dbg:
                print("    page read", flush=True)
            f = xc.host_out.numpy().view(np.float32)
            hn, probs = f[:D], f[D:D + NX]
            ids = np.argpartition(-probs, K - 1)[:K].astype(np.int32)        # GLM: sigmoid (+ bias, 0 here), top 8
            w = probs[ids]
            w = (w / w.sum() * 2.5).astype(np.float32)
            lane.moe(LANE_LAYER, hn, ids, w, out_host)
            xc.host_in.numpy().view(np.float32)[:] = out_host
            if dbg:
                print("    lane done", flush=True)
            xc.h2d.write_tensor(xc.host_in)

        for i, L in enumerate(layers):                                      # warm the socket programs by hand,
            dense_send(L)                                                   # serving the host side, then capture
            during_recv(L, layers[1 - i])
            host_xchg()
            ttnn.synchronize_device(dev)
        for i, L in enumerate(layers):
            for name, fn in (("dense_send", lambda L=L: dense_send(L)),
                             ("during_recv", lambda L=L, nxt=layers[1 - i]: during_recv(L, nxt))):
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                out = fn()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                T[name, i] = (tid, out)
        # check: one exchange's reply against the host's own sum
        ttnn.execute_trace(dev, T["dense_send", 0][0], cq_id=0, blocking=False)
        ttnn.execute_trace(dev, T["during_recv", 0][0], cq_id=0, blocking=False)
        host_xchg()
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(xc.reply).float().flatten()
        res["check"]["reply_exact"] = bool(torch.equal(got, torch.from_numpy(out_host.copy())))
        f = xc.host_out.numpy().view(np.float32)
        res["check"]["page_hn_pcc"] = round(pcc(torch.from_numpy(f[:D].copy()),
                                                ttnn.to_torch(hn_dev).float().flatten()[:D]), 6)
        print("socket check:", json.dumps(res["check"]), flush=True)

        def run(t):
            ttnn.execute_trace(dev, T[t][0], cq_id=0, blocking=False)

        def lane_only():
            ids = rng.choice(NX, K, replace=False).astype(np.int32)
            lane.moe(LANE_LAYER, x_host, ids, np.full(K, 1 / K, np.float32), out_host)

        def lane_call():
            lane_only()
            return ttnn.from_torch(torch.from_numpy(out_host).reshape(1, 1, 1, D), dtype=BF16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT)

        DEBUG = int(os.environ.get("OV_DEBUG", 0))

        def layer(i, mode):
            j = i % 2
            if mode == "l1_sock":
                run(("dense_send", j))
                run(("during_recv", j))
                if i < DEBUG:
                    print(f"  sock layer {i}: queued", flush=True)
                host_xchg(i < DEBUG)
                if i < DEBUG:
                    print(f"  sock layer {i}: replied", flush=True)
                return
            run(("dense_l1" if mode == "l1" else "dense_dram", j))
            ttnn.to_torch(T["dense_l1" if mode == "l1" else "dense_dram", j][1][1])    # the logits: the hand-off
            if mode == "serial":
                host = lane_call()
                ttnn.copy_host_to_device_tensor(host, routed_dev)
                run(("sh", j))
                run(("add_sh", j))
            else:
                run(("during" if mode == "l1" else "sh", j))                          # the card works ...
                host = lane_call()                                                    # ... while the lane runs
                ttnn.copy_host_to_device_tensor(host, routed_dev)
                run(("add_during" if mode == "l1" else "add_sh", j))

        def timed(fn, n):
            fn(0)
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for i in range(n):
                fn(i)
            ttnn.synchronize_device(dev)
            return round((time.perf_counter() - t) / n * 1e6, 1)

        res["fenced_threads_after_setup"] = len(lane.fence_others(keep={me}))   # the sockets' and traces' threads
        for mode in os.environ.get("OV_MODES", "serial,shexp,l1,l1_sock").split(","):
            res[f"{mode}_us"] = timed(lambda i, m=mode: layer(i, m), LAYERS)
            print(mode, res[f"{mode}_us"], "us a layer", flush=True)

        def alone(t):
            def f(i):
                run((t, i % 2))
                ttnn.synchronize_device(dev)
            return f
        res["alone_us"] = {t: timed(alone(t), 20) for t in ("dense_dram", "dense_l1", "sh", "during")}
        res["alone_us"]["lane"] = timed(lambda i: lane_only(), 20)
        res["alone_us"]["lane_and_host_tensor"] = timed(lambda i: lane_call(), 20)
        res["alone_us"]["logits_read"] = timed(lambda i: ttnn.to_torch(T["dense_dram", i % 2][1][1]), 20)
        print(json.dumps(res), flush=True)
        for t in T.values():
            ttnn.release_trace(dev, t[0])
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, "g3_overlap.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
