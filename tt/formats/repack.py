"""GGUF expert quants -> the card's form: per weight a code, per 32-weight block an affine (a, b).

Every format met so far dequantizes as w = a * f(code) + b within a block of 32 weights, with one of three code
transforms the SFPU computes exactly (tt/kernels/iq_decode_sfpu.h, tt/kernels/format_decode.cpp):

  LINEAR  f(c) = c                     Q4_K, Q5_K (5-bit: c = lo + 16 * hi), IQ3_S repacked (c = (v + 15) / 2)
  IQ4NL   f(c) = kvalues_iq4nl[c]      IQ4_NL, IQ4_XS (degree-4 polynomial, rounded)
  E2M1    f(c) = 2 x E2M1(c), signed   MXFP4 (degree-3 polynomial on the magnitude, rounded, then the sign)

repack(qtype, raw) parses whole GGUF rows (numpy uint8, a multiple of the block size) and returns a Repacked with
codes (uint8, one per weight, in GGUF's weight order), a and b (float32, one per 32 weights) and the transform.
reconstruct() rebuilds the weights from that; card_formats.py checks it against gguf-py's dequantizer, which is
llama.cpp's reference. a and b are exact in fp32: products of an fp16 scale and small integers.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp",
                                "gguf-py"))
from gguf import GGMLQuantizationType as QT  # noqa: E402
from gguf import quants as gq  # noqa: E402

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
E2M1_X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)  # ggml's kvalues_mxfp4


@dataclass
class Repacked:
    transform: str  # LINEAR | IQ4NL | E2M1
    bits: int  # 4, or 5 (Q5_K: the code's top bit travels separately on the card)
    codes: np.ndarray  # uint8 [n]
    a: np.ndarray  # float32 [n / 32]
    b: np.ndarray  # float32 [n / 32]


def f16(raw: np.ndarray, off: int) -> np.ndarray:
    return raw[:, off:off + 2].copy().view(np.float16).astype(np.float32)[:, 0]


def _scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml's get_scale_min_k4 for all 8 sub-blocks: 6-bit scales and mins packed in 12 bytes. [nb, 8] each."""
    q = scales.astype(np.int32)
    sc = np.empty((q.shape[0], 8), np.int32)
    mn = np.empty_like(sc)
    sc[:, :4] = q[:, 0:4] & 63
    mn[:, :4] = q[:, 4:8] & 63
    sc[:, 4:] = (q[:, 8:12] & 0xF) | ((q[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)
    return sc, mn


def _q4k_like(raw: np.ndarray, with_hi: bool) -> Repacked:
    nb = raw.shape[0]
    d, dmin = f16(raw, 0), f16(raw, 2)
    sc, mn = _scale_min_k4(raw[:, 4:16])
    off = 16
    if with_hi:
        qh = raw[:, off:off + 32]
        off += 32
    qs = raw[:, off:off + 128].reshape(nb, 4, 32)  # 4 chunks of 64 weights: low nibbles, then high nibbles
    lo = np.stack([qs & 0xF, qs >> 4], axis=2).reshape(nb, 256)  # [chunk][half][32]
    codes = lo.astype(np.uint8)
    if with_hi:
        bit = np.arange(8).reshape(1, 4, 2, 1)  # chunk j, half h -> bit 2j + h of qh[l]
        hi = (qh.reshape(nb, 1, 1, 32) >> bit) & 1
        codes = (codes + 16 * hi.reshape(nb, 256)).astype(np.uint8)
    a = (d[:, None] * sc.astype(np.float32)).reshape(-1)
    b = -(dmin[:, None] * mn.astype(np.float32)).reshape(-1)
    return Repacked("LINEAR", 5 if with_hi else 4, codes.reshape(-1), a.astype(np.float32), b.astype(np.float32))


def _iq4xs(raw: np.ndarray) -> Repacked:
    nb = raw.shape[0]
    d = f16(raw, 0)
    sh = raw[:, 2:4].copy().view(np.uint16)[:, 0].astype(np.int32)
    sl = raw[:, 4:8].astype(np.int32)
    ib = np.arange(8)
    ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
    qs = raw[:, 8:136].reshape(nb, 8, 16)
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(-1).astype(np.uint8)
    a = (d[:, None] * (ls - 32).astype(np.float32)).reshape(-1)
    return Repacked("IQ4NL", 4, codes, a.astype(np.float32), np.zeros_like(a, np.float32))


def _iq4nl(raw: np.ndarray) -> Repacked:
    d = f16(raw, 0)
    qs = raw[:, 2:18]
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=1).reshape(-1).astype(np.uint8)
    return Repacked("IQ4NL", 4, codes, d.astype(np.float32), np.zeros_like(d, np.float32))


def _mxfp4(raw: np.ndarray) -> Repacked:
    e = raw[:, 0].astype(np.int32)
    qs = raw[:, 1:17]
    codes = np.concatenate([qs & 0xF, qs >> 4], axis=1).reshape(-1).astype(np.uint8)
    a = np.ldexp(np.float32(1.0), e - 128).astype(np.float32)  # ggml's E8M0 half scale: the codes are doubled
    return Repacked("E2M1", 4, codes, a, np.zeros_like(a, np.float32))


def _iq3s(raw: np.ndarray) -> Repacked:
    """IQ3_S: |v| = grid byte (only 1, 3, ..., 15), sign separate, so v = 2c - 15 for a 4-bit c, exactly."""
    nb = raw.shape[0]
    blocks = raw.reshape(-1)
    v = gq.dequantize(blocks, QT.IQ3_S).reshape(nb, 8, 32)  # = db * v; recover db and v per sub-block
    d = f16(raw, 0)
    sc = raw[:, 106:110].astype(np.int32)
    ib = np.arange(8)
    db = d[:, None] * (1 + 2 * ((sc[:, ib // 2] >> (4 * (ib % 2))) & 0xF)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        vi = np.where(db[:, :, None] != 0, v / db[:, :, None], 1.0)
    vi = np.rint(vi).astype(np.int32)
    assert np.all(np.abs(vi) % 2 == 1) and np.all(np.abs(vi) <= 15), "IQ3_S grid outside odd 1..15"
    codes = ((vi + 15) // 2).reshape(-1).astype(np.uint8)
    a = (2 * db).reshape(-1)
    b = (-15 * db).reshape(-1)
    return Repacked("LINEAR", 4, codes, a.astype(np.float32), b.astype(np.float32))


BLOCK_BYTES = {QT.Q4_K: 144, QT.Q5_K: 176, QT.IQ4_XS: 136, QT.IQ4_NL: 18, QT.MXFP4: 17, QT.IQ3_S: 110}
BLOCK_WEIGHTS = {QT.Q4_K: 256, QT.Q5_K: 256, QT.IQ4_XS: 256, QT.IQ4_NL: 32, QT.MXFP4: 32, QT.IQ3_S: 256}


def repack(qtype: QT, raw: np.ndarray) -> Repacked:
    raw = np.asarray(raw, np.uint8).reshape(-1, BLOCK_BYTES[qtype])
    if qtype == QT.Q4_K:
        return _q4k_like(raw, False)
    if qtype == QT.Q5_K:
        return _q4k_like(raw, True)
    if qtype == QT.IQ4_XS:
        return _iq4xs(raw)
    if qtype == QT.IQ4_NL:
        return _iq4nl(raw)
    if qtype == QT.MXFP4:
        return _mxfp4(raw)
    if qtype == QT.IQ3_S:
        return _iq3s(raw)
    raise NotImplementedError(f"no card form for {qtype.name} yet")


def scale_factors(qtype: QT, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """The scales a of repack(qtype, raw) as the GGUF block keeps them: a = d * m, d the block's fp16 scale (one per
    block), m a small integer per 32 weights (None: m = 1, a = d). d as float16 [n_blocks], m as int8
    [n_blocks, block weights / 32]. The card's compact stream ships these (tt/kernels/pf_gexp_reader.cpp, COMPACT)
    and rebuilds a in fp32 exactly (dm_bits). IQ3_S's are a / 2 (the card's f = 2c - 15). Q8_0's d: the caller's."""
    raw = np.asarray(raw, np.uint8).reshape(-1, BLOCK_BYTES[qtype])
    d = raw[:, 0:2].copy().view(np.float16)[:, 0]
    ib = np.arange(8)
    if qtype == QT.IQ3_S:
        sc = raw[:, 106:110].astype(np.int32)
        return d, (1 + 2 * ((sc[:, ib // 2] >> (4 * (ib % 2))) & 0xF)).astype(np.int8)
    if qtype == QT.IQ4_XS:
        sh = raw[:, 2:4].copy().view(np.uint16)[:, 0].astype(np.int32)
        sl = raw[:, 4:8].astype(np.int32)
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
        return d, (ls - 32).astype(np.int8)
    if qtype == QT.IQ4_NL:
        return d, None
    raise NotImplementedError(f"no compact scales for {qtype.name} yet (Q4_K / Q5_K: d * sc with a b = -dmin * mn "
                              f"term; MXFP4: an E8M0 exponent instead of d)")


def dm_bits(d: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    """The fp32 bits of d * m (float16 d, int8 m, broadcast) as the card's reader builds them, in integers only (its
    RISC-V cores have no FPU): d = M 2^E (M <= 2047), P = M |m| < 2^18 normalized. Equal to numpy's float32 product
    bit for bit (the product needs at most 18 bits); d must be finite."""
    h = d.view(np.uint16).astype(np.int64)
    m = np.ones_like(h) if m is None else m.astype(np.int64)
    e, f = (h >> 10) & 31, h & 1023
    assert not np.any(e == 31), "d not finite"
    M = np.where(e == 0, f, 1024 | f)
    E = np.where(e == 0, -24, e - 25)
    P = M * np.abs(m)
    sign = ((h >> 15) ^ (m < 0)) << 31
    b = np.floor(np.log2(np.maximum(P, 1))).astype(np.int64)
    bits = sign | ((E + b + 127) << 23) | ((P << (23 - b)) & 0x7FFFFF)
    return np.where(P == 0, sign, bits).astype(np.uint32)


def code_values(r: Repacked) -> np.ndarray:
    """f(code) per weight, exactly (small integers)."""
    if r.transform == "LINEAR":
        return r.codes.astype(np.float32)
    if r.transform == "IQ4NL":
        return KVALUES_IQ4NL[r.codes]
    return E2M1_X2[r.codes]


def reconstruct(r: Repacked) -> np.ndarray:
    """w = a * f(code) + b, one rounding (as the card's fused multiply-add)."""
    a = np.repeat(r.a.astype(np.float64), 32)
    b = np.repeat(r.b.astype(np.float64), 32)
    return (a * code_values(r).astype(np.float64) + b).astype(np.float32)
