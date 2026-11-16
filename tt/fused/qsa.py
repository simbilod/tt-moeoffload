"""The prefill's top-512-pools selection (tt/kernels/pf_qsa_mask.cpp), shared by Flash-Next's QSA (g7_prefill.Attn.select)
and GLM's DSA indexer (tt/glm/g6_prefill.py): per query row the complete pools p < (pos + 1) / 4, all of them when
<= 512, else the 512 largest scores (ties to the lowest pool), as 4 cells each, then the row's own open pool's cells up
to itself. Moved from g7_prefill.py (2026-11-13), which re-exports it with its W51_QSAK_ROWS / W51_SEL_FAST."""
from __future__ import annotations

import os

import ttnn

KDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
KPOOL = 4
BF16, FP32, TILE, RM = ttnn.bfloat16, ttnn.float32, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def qsa_alloc(dev, Tq: int, L: int):
    """qsa_mask's output, allocated before the selection's temporaries (Attn.select): the mask lives through the SDPA,
    the temporaries are freed under it, so they leave one free region (a 61K prompt at the 64K build: the experts'
    two 839 MB buffers after layer 47 found 199 MB a bank in one piece instead of the ttnn path's 224, 2026-11-08)."""
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Tq, L]), BF16, TILE, dev, ttnn.DRAM_MEMORY_CONFIG)


def qsa_mask(score, L: int, pos0: int, R: int = 0, out=None, topk: int = 0, sel_fast: bool = True,
             signed: bool = False):
    """score [1, 1, Tq, L / 4] fp32 (the indexer's sum over heads, before any visibility mask) -> the QSA mask
    [1, 1, Tq, L] bf16 (0 attended, -1e4 not) of queries at positions pos0 .., as Attn.select's ops build it, bit for
    bit (select_ops.py): units of R query rows over both data-movement cores of every core. topk > 0 (the sparse
    attention, IDX): instead (indices [1, 1, Tq, topk] uint32 row-major, each row's attended cells ascending then
    0xFFFFFFFF; wanted [1, 1, Tq, 16] uint32, word 0 the row's cell count before the cap). signed: scores of any sign
    (GLM's; Flash-Next's are relu sums, >= 0, and keep the kernel's unsigned keys). R 0: by L, or IDX's by Tq."""
    dev, Tq = score.device(), score.shape[-2]
    PT = score.shape[-1] // 32
    auto = not R
    R = R or (4 if L <= 32768 else 8)        # select_ops.py: 38.6 / 48.0 ms at 32K, 100.8 / 97.2 at 64K
    if topk and sel_fast and auto:           # IDX at 128K, ms an 8K chunk (R 8 / 4 / 2): Tq 512 291 / 176 / 183,
        R = 2 if Tq >= 1024 else 4           # Tq 1024 149 / 172 / 136, Tq 2048 147 / 128 / 113
    assert score.dtype == FP32 and Tq % 32 == 0 and score.shape[-1] * KPOOL == L, (score.shape, L)
    if topk:
        mask = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Tq, topk]), ttnn.uint32, RM, dev,
                                              ttnn.DRAM_MEMORY_CONFIG)
        wanted = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Tq, 16]), ttnn.uint32, RM, dev,
                                                ttnn.DRAM_MEMORY_CONFIG)
    else:
        mask = out if out is not None else qsa_alloc(dev, Tq, L)
    g = dev.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    units = Tq // 32 * (32 // R)
    slots = [(c, w) for w in (0, 1) for c in cores]          # (core, reader 0 / writer 1)
    per, extra = divmod(units, len(slots))
    rts, u = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()], 0
    used = [set(), set()]
    for k, ((cx, cy), w) in enumerate(slots):
        n = per + (1 if k < extra else 0)
        if n:
            rts[w][cx][cy] = [score.buffer_address(), mask.buffer_address(), u, n, pos0, PT] + (
                [wanted.buffer_address()] if topk else [])
            used[w].add((cx, cy))
        u += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    rng = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy))  # noqa: E731
                                        for cx, cy in sorted(cs)])
    kernels, cbs = [], []
    lst = (2048 if R >= 4 else 4096) if sel_fast else 0   # pf_qsa_mask.cpp LIST: a row's pivot-bin keys (IDX: and pools) in L1
    for w, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
        if not used[w]:
            continue
        r = rng(used[w])
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "pf_qsa_mask.cpp"), core_ranges=r,
            compile_time_args=[R] + acc(score) + acc(mask) + (acc(wanted) if topk else []),
            defines=([("WRITER", "1")] if w else []) + ([("IDX", "1"), ("TOPK", str(topk))] if topk else []) +
                    ([("LIST", str(lst))] if lst else []) + ([("SIGNED", "1")] if signed else []),
            runtime_args=rts[w], config=cfg))
        b = 3 * w
        for idx, size in ((b, 16 * 128 * R), (b + 1, R * 2048 * 2),
                          (b + 2, R * (topk * 4 + 64) if topk else 4 * 64 * R)) + (((6 + w, R * lst * (8 if topk else 4)),) if lst else ()):
            cbs.append(ttnn.CBDescriptor(total_size=size, core_ranges=r, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint8, page_size=size)]))
    ttnn.generic_op([score, mask] + ([wanted] if topk else []),
                    ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return (mask, wanted) if topk else mask
