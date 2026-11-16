"""Host side of tt/kernels/fused/grid.h: the worker grid, its multicast rectangles, the barrier's arguments, and the
small helpers every fused program needs (work splits, circular buffers, semaphores, traced timing)."""
from __future__ import annotations

import statistics
import time

import torch

import ttnn


def split(n: int, parts: int) -> list[tuple[int, int]]:
    """n items over parts workers as (first, count), the first n % parts workers taking one more."""
    base, extra = divmod(n, parts)
    out, s = [], 0
    for i in range(parts):
        c = base + (1 if i < extra else 0)
        out.append((s, c))
        s += c
    return out


# Card memory reads are not shared fairly between cores: with all 110 cores streaming the same bytes through NOC0, a
# core's time depends on its logical row (row 0 the slowest, ~3.3x row 9) and the card delivers ~400 GB/s; with some
# rows reading through NOC1 instead, ~480-490 GB/s and far more even (g3_stream.py, 2026-10-23: us per 400 bfp8_b
# pages, by row, keyed by the rows on NOC1). Work split in proportion to 1 / time ends together.
ROW_US = {
    (): (119.6, 120.3, 117.3, 110.6, 103.2, 86.6, 71.5, 56.8, 46.6, 35.6),
    (0, 1, 2): (80.1, 80.2, 80.2, 97.2, 100.1, 97.1, 88.3, 86.1, 83.6, 77.7),
    (0, 1, 9): (84.9, 84.6, 97.6, 96.0, 93.9, 96.6, 91.4, 89.9, 87.6, 84.8),
    (0, 3, 6, 9): (85.8, 76.9, 94.5, 86.6, 93.3, 97.5, 86.6, 95.2, 91.6, 86.2),
}


# the rows that read through NOC1 in the fused programs (the reader on NOC1, the writer on NOC0 there)
NOC1_ROWS = (0, 1, 9)


def split_weighted(n: int, weights: list[float]) -> list[tuple[int, int]]:
    """n items over len(weights) workers as (first, count), counts in proportion to weights (largest remainders)."""
    total = sum(weights)
    exact = [n * w / total for w in weights]
    counts = [int(e) for e in exact]
    for i in sorted(range(len(weights)), key=lambda i: counts[i] - exact[i])[:n - sum(counts)]:
        counts[i] += 1
    out, s = [], 0
    for c in counts:
        out.append((s, c))
        s += c
    return out


class Grid:
    """All worker cores, row-major in logical coordinates; core 0 coordinates the barrier."""

    def __init__(self, dev, arrivals_per_core: int = 1):
        self.dev = dev
        g = dev.compute_with_storage_grid_size()
        self.cores = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        self.noc = [dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x, y in self.cores]
        # rectangles: runs of consecutive NOC x, all rows (the grid's NOC columns have a gap)
        xs, ys = sorted({n.x for n in self.noc}), sorted({n.y for n in self.noc})
        assert len(xs) * len(ys) == len(self.cores)
        runs, run = [], [xs[0]]
        for v in xs[1:]:
            if v == run[-1] + 1:
                run.append(v)
            else:
                runs.append(run)
                run = [v]
        runs.append(run)
        self.rects = [(r[0], ys[0], r[-1], ys[-1], len(r) * len(ys)) for r in runs]
        self.arrivals = arrivals_per_core

    def __len__(self) -> int:
        return len(self.cores)

    def speeds(self, noc1_rows=()) -> list[float]:
        """Each core's share of card memory bandwidth when every core streams, the rows noc1_rows reading through NOC1
        (relative; ROW_US)."""
        t = ROW_US[tuple(sorted(noc1_rows))]
        return [t[0] / t[y] for _, y in self.cores]

    def order(self, noc1_rows=NOC1_ROWS) -> list[int]:
        """Core indices, the NOC0 rows first: a phase on fewer cores than the grid takes them (NOC1's reads from card
        memory are slow per core). order()[k] is the core of a phase's item k."""
        return sorted(range(len(self.cores)), key=lambda i: self.cores[i][1] in noc1_rows)

    def dm_kernels(self, reader: str, writer: str, reader_ct: list, writer_ct: list, rargs: dict, wargs: dict,
                   defines=(), noc1_rows=NOC1_ROWS) -> list:
        """The reader (RISCV_1) and writer (RISCV_0) kernels: reader on NOC0 and writer on NOC1, except on the rows
        noc1_rows (the other way around). rargs, wargs: {(x, y): runtime args}."""
        out = []
        for swap in (False, True):
            cores = [(x, y) for x, y in self.cores if (y in noc1_rows) == swap]
            if not cores:
                continue
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
            n0, n1 = (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0) if swap else (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)
            for src, ct, args, proc, noc in ((reader, reader_ct, rargs, ttnn.DataMovementProcessor.RISCV_1, n0),
                                             (writer, writer_ct, wargs, ttnn.DataMovementProcessor.RISCV_0, n1)):
                dm = ttnn.DataMovementConfigDescriptor()
                dm.processor, dm.noc = proc, noc
                rt = ttnn.RuntimeArgs()
                for x, y in cores:
                    rt[x][y] = args[(x, y)]
                out.append(ttnn.KernelDescriptor(kernel_source=src, core_ranges=ranges, compile_time_args=ct,
                                                 runtime_args=rt, defines=list(defines), config=dm))
        return out

    def args(self, idx: int) -> list[int]:
        """grid.h's argument block for core idx."""
        c, me = self.noc[0], self.noc[idx]
        own = next(r for r, (x0, _, x1, _, _) in enumerate(self.rects) if x0 <= me.x <= x1)
        out = [1 if idx == 0 else 0, c.x, c.y, len(self.cores) * self.arrivals, own, len(self.rects)]
        for r in self.rects:
            out += list(r)
        return out

    def semaphores(self) -> list:
        return [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.ranges, initial_value=0) for i in (0, 1, 2)]

    def cb(self, index: int, dtype, page: int, pages: int):
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=self.ranges, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)])


# grid.h's profile (-DGRID_PROFILE): per core 16 n bytes in card memory, n barriers kept per program (32 by default; the
# token's programs, g4_phases.py, keep 64: a DeltaNet run of 3 layers crosses 37). Word layout per core: the reader's
# wait entries [0, n), its releases [n, 2 n), the writer's arrivals [2 n, 3 n), the reader's marks [3 n, 4 n).
PROFILE_N = 64
MARK_START, MARK_END = 0, -1    # the marks every program's reader keeps: its start (mark 0) and end (mark n - 1)


def profile_tensor(dev, n: int = 32):
    """The profile's card memory: 128 slots of 16 n bytes, one row-major page (uint32, zeroed)."""
    return ttnn.from_torch(torch.zeros(1, 1, 1, 128 * 4 * n, dtype=torch.int32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def profile_defines(cb: int, n: int = 32, off: int = 0) -> list:
    """The kernels' defines for the profile in CB cb (16 n bytes from byte off of it)."""
    return [("GRID_PROFILE", str(cb)), ("GRID_PROFILE_N", str(n))] + ([("GRID_PROFILE_OFF", str(off))] if off else [])


def profile_plumbing(t, n: int, cb: int, off: int = 0):
    """A program builder's profile pieces for tensor t (None: no profile): the reader's and writer's extra runtime
    args for core idx, the kernels' defines, the profile's bytes in CB cb (from byte off)."""
    if t is None:
        return (lambda idx: []), [], 0
    return (lambda idx: [t.buffer_address(), idx]), profile_defines(cb, n, off), 16 * n


def profile_words(t, cores: int, n: int = 32) -> torch.Tensor:
    """The profile tensor read back: [cores, 4 n] clocks (int64, the 32-bit counters as written)."""
    return ttnn.to_torch(t).flatten().long()[:cores * 4 * n].reshape(cores, 4 * n) & 0xFFFFFFFF


def traced_us(dev, fn, reps: int = 48, tokens: int = 20) -> float:
    """G2's method: one capture of fn, replayed reps times per sync; the median over tokens of the time per replay."""
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    keep = fn()  # noqa: F841  (the trace writes these buffers; hold them until it is released)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    samples = []
    for i in range(tokens + 2):
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        if i >= 2:
            samples.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(dev, tid)
    return statistics.median(samples)
