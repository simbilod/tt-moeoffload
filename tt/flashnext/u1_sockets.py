"""The per-layer hand-off floor through H2D / D2H sockets.

G1 measured ~68 us per layer for ttnn's hand-off (traced compute, per-layer sync, reads and an upload through the
command queue). A persistent kernel talking to the host through sockets avoids the command queue, launches and
syncs: the host writes a page, the card's core forwards it into pinned host RAM, the host polls. This times that
round trip with no compute, one page each way, for:

  64 B      the protocol floor
  5,184 B   a mode-1 layer: x (2560 bf16) + 10 expert ids and router weights, rounded up to 64 B
  16,384 B  room to spare

in both H2D modes (HOST_PUSH: the host writes into the core's L1; DEVICE_PULL: the core reads pinned host RAM).
Host-side Python timing, so it includes a few us of pybind overhead per call that a C++ worker would not have.

  python3 tt_moeoffload/launch.py --cpus 64-71 --memory-max $((16 << 30)) --engine tt/flashnext/u1_sockets.py

Measured 2026-10-21 (2,000 iterations, payload checked; the card idle at ~61 C), p50 / p99 us:

  device_pull     64 B   5.6 / 6.6     host_push     64 B   5.9 / 7.7
  device_pull  5,184 B   6.1 / 10.8    host_push  5,184 B  25.2 / 30.8
                                       host_push 16,384 B  59.9 / 65.0

DEVICE_PULL carries a layer's page at about the protocol floor: ~6 us vs G1's ~68 us. HOST_PUSH is bound by the
host's uncached TLB stores (~200 B/us). device_pull at 16 KB: no lucky pin in 3 tries.

The sockets' host FIFOs must be pinned for the card's DMA, and with iommu=pt tt-kmd pins only
physically contiguous memory (memory.c: "pages discontiguous"); tt-metal's hugepage fallback is Wormhole-only. So
socket creation works only when a 68 KB anonymous mmap happens to be contiguous (pinned() retries; ~2 sockets per
fresh process). For real use the card's IOMMU group (the card alone) must be translated: unbind tenstorrent,
`echo DMA-FQ > /sys/kernel/iommu_groups/<group>/type`, rebind (root; other devices stay identity), or drop iommu=pt at boot.

Measured 2026-10-22 with group 49 in DMA-FQ: every config in one process, no retries needed, p50 / p99 us:

  device_pull     64 B   5.5 / 6.3     host_push     64 B   5.9 / 6.7
  device_pull  5,184 B   6.1 / 6.8     host_push  5,184 B  21.6 / 22.4
  device_pull 16,384 B   7.6 / 12.8    host_push 16,384 B  59.7 / 65.8
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

import ttnn

KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "socket_loopback.cpp")
ITERS, WARMUP = 2000, 100
FIFO = 64 << 10


def pinned(make):
    """With iommu=pt, tt-kmd pins only physically contiguous host memory, and the sockets' FIFOs are plain heap
    allocations; GLIBC_TUNABLES=glibc.malloc.hugetlb=1 puts most of them inside one 2 MB THP page. Retry the rest
    with some ballast in between to move the allocator along."""
    ballast = []
    for _ in range(32):
        try:
            return make()
        except RuntimeError as e:
            if "Failed to pin pages" not in str(e):
                raise
            ballast.append(bytearray(96 << 10))
    raise SystemExit("could not get a physically contiguous socket FIFO (needs IOMMU translation or THP)")


def run(dev, mode: str, page: int) -> list[float]:
    core = ttnn.CoreCoord(0, 0)
    where = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), core)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    h2d_mode = ttnn.H2DMode.DEVICE_PULL if mode == "device_pull" else ttnn.H2DMode.HOST_PUSH
    h2d = pinned(lambda: ttnn.H2DSocket(dev, where, ttnn.BufferType.L1, FIFO, h2d_mode))
    h2d.set_page_size(page)
    d2h = pinned(lambda: ttnn.D2HSocket(dev, where, FIFO))
    d2h.set_page_size(page)

    kernel = ttnn.KernelDescriptor(
        kernel_source=os.path.abspath(KERNEL),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[h2d.get_config_buffer_address(), d2h.get_config_buffer_address(), page, WARMUP + ITERS],
        defines=[("H2D_DEVICE_PULL", "1")] if mode == "device_pull" else [],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint8, page_size=page)],
    )
    # generic_op wants an input and an output; the kernel touches neither
    dummies = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                              ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
    ttnn.generic_op(dummies, ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))

    src = torch.randint(0, 256, (page,), dtype=torch.uint8)
    dst = torch.zeros(page, dtype=torch.uint8)
    samples = []
    for i in range(WARMUP + ITERS):
        src[:8] = torch.tensor(list(i.to_bytes(8, "little")), dtype=torch.uint8)
        t0 = time.perf_counter_ns()
        h2d.write_tensor(src)
        d2h.read_tensor(dst)
        t1 = time.perf_counter_ns()
        if i >= WARMUP:
            samples.append((t1 - t0) / 1e3)
        if i in (0, WARMUP + ITERS - 1) and not torch.equal(src, dst):
            raise SystemExit(f"{mode} {page} B: the page came back different (iteration {i})")
    h2d.barrier(timeout_ms=5000)
    d2h.barrier(timeout_ms=5000)
    ttnn.synchronize_device(dev)
    return samples


def main() -> None:
    # optional: one mode and page sizes, e.g. `device_pull 5184` (a fresh process per config gets more lucky pins
    # while iommu=pt; see pinned())
    modes = sys.argv[1:2] or ["host_push", "device_pull"]
    pages = [int(p) for p in sys.argv[2:]] or [64, 5184, 16384]
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    rows = []
    try:
        for mode in modes:
            for page in pages:
                s = sorted(run(dev, mode, page))
                pct = lambda p: s[min(len(s) - 1, int(p * len(s)))]
                rows.append((mode, page, s[0], statistics.median(s), pct(0.99), s[-1]))
                print(f"  {mode:11s} {page:6d} B  min {s[0]:6.1f}  p50 {rows[-1][3]:6.1f}  p99 {rows[-1][4]:7.1f}  "
                      f"max {s[-1]:8.1f} us", flush=True)
    finally:
        ttnn.close_mesh_device(dev)
    print(f"\nround trip host -> card core -> host, one page each way, {ITERS} iterations (us):")
    print(f"  {'mode':11s} {'page':>8s} {'min':>7s} {'p50':>7s} {'p99':>8s} {'max':>9s}")
    for mode, page, mn, p50, p99, mx in rows:
        print(f"  {mode:11s} {page:6d} B {mn:7.1f} {p50:7.1f} {p99:8.1f} {mx:9.1f}")


if __name__ == "__main__":
    main()
