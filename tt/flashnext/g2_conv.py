"""G2 prefill follow-up: the DeltaNet's causal depthwise conv (kernel 4) + SiLU for a 4K chunk, against qwen36's
_causal_conv1d_fir (9.1 ms: it re-tiles three shifted copies of the [4096 x 10240] input).

Each core takes a run of column tiles and walks down the T axis (tt/kernels/conv_*.cpp): the taps scale the current
tile by a row broadcast, and the FPU shifts and sums with 0/1 shift tiles (exact), carrying the previous tile's last
rows; SiLU on the SFPU. The chunk's first rows read the previous chunk's last 3 tokens (the conv state). The input may
be the wider projection output, read in place. Checked against float64, then timed traced (4 replays per sync,
medians of 10) and as the stock op.

  python3 tt_moeoffload/launch.py --memory-max $((16 << 30)) --engine tt/flashnext/g2_conv.py
"""
from __future__ import annotations

import json
import os
import statistics
import time

import torch

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.normpath(os.path.join(HERE, "..", "kernels"))
OUT = os.path.join(HERE, "results")
T, C = 4096, 10240


def split(n: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(n, parts)
    out, s = [], 0
    for i in range(parts):
        c = base + (1 if i < extra else 0)
        out.append((s, c))
        s += c
    return out


def shift_tiles() -> torch.Tensor:
    """I, L1..L3 (row r takes row r - s), U1..U3 (row r < s takes the previous tile's row 32 - s + r)."""
    t = torch.zeros(7, 32, 32)
    t[0] = torch.eye(32)
    for s in range(1, 4):
        for r in range(s, 32):
            t[s, r, r - s] = 1
        for r in range(s):
            t[3 + s, r, 32 - s + r] = 1
    return t.reshape(1, 1, 7 * 32, 32)


class CausalConv:
    """y = silu(causal depthwise conv(x)) over [T, C]; taps [4, C] and the 3-token state as host tensors."""

    def __init__(self, dev, t, c, taps: torch.Tensor, state: torch.Tensor):
        self.dev, self.t, self.c, self.rt, self.ct = dev, t, c, t // 32, c // 32
        put = lambda a, dt=ttnn.bfloat16: ttnn.from_torch(a, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt)  # noqa: E731
        tap_rows = torch.zeros(4 * 32, c, dtype=torch.bfloat16)
        tap_rows[0::32] = taps.to(torch.bfloat16)
        st = torch.zeros(32, c, dtype=torch.bfloat16)
        st[29:] = state.to(torch.bfloat16)
        self.w, self.st = put(tap_rows.reshape(1, 1, -1, c)), put(st.reshape(1, 1, 32, c))
        self.sh = put(shift_tiles(), ttnn.float32)
        gs = dev.compute_with_storage_grid_size()
        self.cores = [(a, b) for b in range(gs.y) for a in range(gs.x)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gs.x - 1, gs.y - 1))])

    def __call__(self, x, y, reps=1, batch=4):
        """x [1, 1, T, >= C] (its first C columns are the input), y [1, 1, T, C]: built per call (addresses)."""
        x_ct = x.shape[-1] // 32
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (a, b), (j0, jn) in zip(self.cores, split(self.ct, len(self.cores))):
            rrt[a][b] = [x.buffer_address(), self.w.buffer_address(), self.st.buffer_address(),
                         self.sh.buffer_address(), j0, jn]
            crt[a][b] = [j0, jn]
            wrt[a][b] = [y.buffer_address(), j0, jn]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True)
        cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 64
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "conv_reader.cpp"), core_ranges=self.grid,
                                  compile_time_args=[reps, self.rt, self.ct, x_ct, batch] + acc(x) + acc(self.w)
                                  + acc(self.st) + acc(self.sh), runtime_args=rrt,
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "conv_writer.cpp"), core_ranges=self.grid,
                                  compile_time_args=[reps, self.rt, self.ct] + acc(y), runtime_args=wrt,
                                  config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "conv_compute.cpp"), core_ranges=self.grid,
                                  compile_time_args=[reps, self.rt], runtime_args=crt, config=cfg),
        ]
        spec = [(0, ttnn.bfloat16, 2048, 2 * batch), (1, ttnn.bfloat16, 2048, 4), (2, ttnn.bfloat16, 2048, 1),
                (5, ttnn.float32, 4096, 7), (24, ttnn.float32, 4096, 4), (25, ttnn.float32, 4096, 6),
                (16, ttnn.bfloat16, 2048, 2)]
        cbs = [ttnn.CBDescriptor(total_size=size * n, core_ranges=self.grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=size)]) for i, f, size, n in spec]
        ttnn.generic_op([x, self.w, self.st, self.sh, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[],
                                                                                 cbs=cbs))
        return y


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    put = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt)  # noqa: E731
    res = {"date": time.strftime("%Y-%m-%d %H:%M")}
    g = torch.Generator().manual_seed(0)
    try:
        x = torch.randn(T, C, generator=g).to(torch.bfloat16)
        taps = (torch.randn(4, C, generator=g) * 0.5).to(torch.bfloat16)
        state = torch.randn(3, C, generator=g).to(torch.bfloat16)
        xt = put(x.reshape(1, 1, T, C))
        yt = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                            ttnn.DRAM_MEMORY_CONFIG)
        conv = CausalConv(dev, T, C, taps, state)
        print("launching the conv kernel", flush=True)
        conv(xt, yt)
        ttnn.synchronize_device(dev)
        print("  done, checking", flush=True)
        xp = torch.cat([state, x]).double()
        ref = sum(taps[k].double() * xp[k:k + T] for k in range(4))
        ref = ref * torch.sigmoid(ref)
        got = ttnn.to_torch(yt).double().reshape(T, C)
        res["max_rel_err"] = float((got - ref).abs().max() / ref.abs().max())
        print(f"  max rel err vs float64 {res['max_rel_err']:.1e}", flush=True)

        def traced_ms(f, reps=4, samples=10):
            f()
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            f()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            s = []
            for i in range(samples + 2):
                t0 = time.perf_counter()
                for _ in range(reps):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                if i >= 2:
                    s.append((time.perf_counter() - t0) / reps * 1e3)
            ttnn.release_trace(dev, tid)
            return statistics.median(s)

        res["kernel_traced_ms"] = traced_ms(lambda: conv(xt, yt))
        print(f"  {res['kernel_traced_ms']:7.2f} ms  conv kernel, traced", flush=True)
        x3 = ttnn.reshape(xt, [1, T, C])
        taps_dev = [put(taps[k].reshape(1, 1, C)) for k in range(4)]
        conv_state = put(state.reshape(1, 3, C))
        stock = lambda: _causal_conv1d_fir(x3, None, None, 4, dev, conv_state=conv_state, weight_taps=taps_dev)  # noqa: E731
        res["stock_traced_ms"] = traced_ms(stock)
        print(f"  {res['stock_traced_ms']:7.2f} ms  qwen36's _causal_conv1d_fir, traced", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    path = os.path.join(OUT, "g2_conv.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
