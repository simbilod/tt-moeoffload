"""The M-row layout's host side (a verify pass's rows; fused/hc_dm.h's mp-row blocks), shared by the models' programs.

The same functions as tt/flashnext/g3_layer.py's (mp_of, blocks, unblocks, step_words), which keeps its own copies until
Flash-Next moves to these (a duplication to remove)."""
import torch


def mp_of(m: int) -> int:
    """The M-row layout's rows for M rows (1, 2, 4: M = 3 runs as 4 with row 3 a copy of row 2)."""
    return 1 if m == 1 else 2 if m == 2 else 4


def blocks(rows, mp: int):
    """[mp, n * 32] -> the M-row layout: per tile its face-0 pieces of rows 0..mp-1, then its face-1 pieces, flat."""
    n = rows.shape[1] // 32
    return rows.reshape(mp, n, 2, 16).permute(1, 2, 0, 3).reshape(-1)


def unblocks(flat, n: int, mp: int):
    """The M-row layout's n tiles (flat) -> [mp, n * 32]."""
    return flat[:n * 2 * mp * 16].reshape(n, 2, mp, 16).permute(2, 0, 1, 3).reshape(mp, n * 32)


def pad_rows(rows):
    """[M, n] -> [mp, n]: M = 3 gets row 3 a copy of row 2."""
    mp = mp_of(rows.shape[0])
    return torch.cat([rows] + [rows[-1:]] * (mp - rows.shape[0]))


def step_words(M, n_keep=1, parity=0, pos=0):
    """The step tensor's host form (uint32 [16]): word 0 the position, words 1..4 M, n_keep, parity, mp."""
    w = torch.zeros(1, 1, 1, 16, dtype=torch.int32)
    w[0, 0, 0, :5] = torch.tensor([pos, M, n_keep, parity, mp_of(M)])
    return w
