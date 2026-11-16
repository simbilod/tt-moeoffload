"""GLM's KDA at M rows, float64: the chunked form with a deferred commit that the card program follows.

Flash-Next's DeltaNet (flayer_compute.cpp phase_b_m, 8f572b9) has a scalar decay per row, so its coefficient tiles hold
products of scalars. KDA's decay is per key channel (d = exp(l), l = -5 sigmoid(..), one 128-vector per row and head), so
the ratios become vectors. They are built from log-decays summed over the rows (L_t = l_0 + .. + l_t; the reader's fp32
adds) and exps on the card: G = exp(L) and Gi = exp(-L) per row, and exp(L_a - L_j) for the commit. With M <= 4 the
ratios stay within e^20 (l >= -5), well inside fp32.

One row (glm_kda_compute.cpp): S = diag(d) S; u = beta (v - k^T S); S += k u^T; o = q^T S.
A pass of M rows from S_a (the state after the rows kept so far), rows t = 0..M-1, Kg = K * G, Qg = Q * G, Ki = K * Gi:
  R = diag(beta) (V - Kg S_a)
  N = diag(beta) strict_lower(Kg Ki^T)          (u_t = R_t - sum_{j<t} N_tj u_j)
  U = (I - N)(I + N^2) R                       ((I + N)^-1; N^4 = 0 at M <= 4)
  O = Qg S_a + lower(Qg Ki^T) U
The pass keeps rows 0..a-1 (n_keep = a >= 1: row 0 is the accepted token); the next pass first commits them:
  S_next = diag(G_{a-1}) S_a + (Ki[:a] * G_{a-1})^T U[:a]
so a pass carries its state S_a (committed) and pending rows (K, U, L) to the next one, as Flash-Next's S tensor does.
The conv history: the window is (s0, s1, s2, y_0 .. y_{M-1}); the next pass's history is rows [a, a + 3) of it.
"""
from __future__ import annotations

import numpy as np

KDA_LB = -5.0


def step(S, q, k, v, beta, l):
    """one row, heads batched: S [H, K, V], q k l [H, K], v [H, V], beta [H] -> o [H, V], the new S."""
    S = S * np.exp(l)[:, :, None]
    u = beta[:, None] * (v - np.einsum("hk,hkv->hv", k, S))
    S = S + k[:, :, None] * u[:, None, :]
    return np.einsum("hk,hkv->hv", q, S), S


def commit(S, pend, a):
    """the previous pass's first a rows onto its state: pend = (K [H, M, K], U [H, M, V], L [H, M, K])."""
    if pend is None:
        return S
    K, U, L = pend
    last = L[:, a - 1]
    Kc = K[:, :a] * np.exp(last[:, None] - L[:, :a])
    return S * np.exp(last)[:, :, None] + np.einsum("hjk,hjv->hkv", Kc, U[:, :a])


def rows(S, Q, K, V, beta, l):
    """M rows from S (heads batched: Q K l [H, M, K], V [H, M, V], beta [H, M]) -> O [H, M, V], pend (K, U, L)."""
    M = Q.shape[1]
    L = np.cumsum(l, axis=1)
    G, Gi = np.exp(L), np.exp(-L)
    Kg, Qg, Ki = K * G, Q * G, K * Gi
    R = beta[:, :, None] * (V - np.einsum("hmk,hkv->hmv", Kg, S))
    N = beta[:, :, None] * np.tril(np.einsum("hmk,hjk->hmj", Kg, Ki), -1)
    I = np.eye(M)
    U = (I - N) @ (I + N @ N) @ R
    O = np.einsum("hmk,hkv->hmv", Qg, S) + np.tril(np.einsum("hmk,hjk->hmj", Qg, Ki)) @ U
    return O, (K, U, L)


def conv(hist, Y, taps):
    """the causal conv's M rows: hist [3, C], Y [M, C], taps [4, C] -> pre-silu [M, C], the window [3 + M, C]."""
    win = np.concatenate([hist, Y])
    return np.stack([(win[t:t + 4] * taps).sum(0) for t in range(len(Y))]), win


def keep_history(win, a):
    """the next pass's conv history after keeping a rows."""
    return win[a:a + 3]
