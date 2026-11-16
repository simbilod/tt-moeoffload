// Derived from Strata's src/kernels/cpu/iq_avx512.cpp in the form its fork Xolotl had it (gu_rows with the SwiGLU
// clamp), Copyright (c) 2026 Niko1221 and the Strata contributors, MIT License (LICENSES/, NOTICE).
// IQ3_S gate|up rows on AVX-512 VBMI (the CPU lane's GLM path): Strata's iq_avx512.cpp rows (as its
// Xolotl fork had them: Fmt<21> + row_dot + gu_rows) with the grid lookup vectorized. Strata's decode builds each 64-value chunk from 16
// scalar loads of iq3s_grid (9-bit index: qs byte | the qh bit << 8) assembled by _mm512_set_epi32: ~100 cycles a
// 256-value block on the development host, the lane's gate|up at ~217 GB/s against down's ~320 (2.86 GB/s a thread, the same hot
// in L2 as from memory: compute-bound). Here a block's 64 grid entries at once: for each byte position k of an
// entry a 512-byte table T_k (iq3s_grid's bytes are 1..15), 4 vpermi2b over its 128-byte quarters (index bits 0-6),
// blended by bit 7 (the qs byte's sign bit) and bit 8 (qh, one 64-bit mask); then each chunk's 64 bytes interleaved
// from the 4 results (2 vpermi2b and a blend). Signs, scales and the integer dots are Strata's, in its order, so
// every result is its row_dot's to the bit.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ggml-common.h"   // block_q8_K, iq3s_grid (the includer defines GGML_COMMON_DECL_* / IMPL_*)

namespace cl3s {

struct Tables {
    alignas(64) uint8_t t[4][512];       // t[k][i] = byte k of iq3s_grid[i]
    alignas(64) uint8_t ilv[4][64];      // chunk j: byte 4e + k <- entry 16 j + e of t_k's result (k odd: + 64)
    Tables() {
        for (int i = 0; i < 512; i++)
            for (int k = 0; k < 4; k++) t[k][i] = (uint8_t) (iq3s_grid[i] >> (8 * k));
        for (int j = 0; j < 4; j++)
            for (int e = 0; e < 16; e++)
                for (int k = 0; k < 4; k++) ilv[j][4 * e + k] = (uint8_t) (16 * j + e + (k & 1 ? 64 : 0));
    }
};
inline const Tables& tables() {
    static const Tables T;
    return T;
}

// software prefetch distance in bytes (CL_IQ3S_PREFETCH, default Strata's 2048)
inline int prefetch_ahead() {
    static const int v = getenv("CL_IQ3S_PREFETCH") ? atoi(getenv("CL_IQ3S_PREFETCH")) : 2048;
    return v;
}

inline uint64_t u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (Strata's scales4)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// a block's 4 chunks of grid magnitudes (64 unsigned bytes each, value order)
inline void decode(const uint8_t* b, __m512i g[4]) {
    const Tables& T = tables();
    const __m512i idx = _mm512_loadu_si512((const void*) (b + 2));
    const __mmask64 m7 = _mm512_movepi8_mask(idx), m8 = _cvtu64_mask64(u64(b + 66));
    __m512i r[4];
    for (int k = 0; k < 4; k++) {
        const __m512i* t = (const __m512i*) T.t[k];
        const __m512i a = _mm512_permutex2var_epi8(_mm512_load_si512(t + 0), idx, _mm512_load_si512(t + 1));
        const __m512i c = _mm512_permutex2var_epi8(_mm512_load_si512(t + 2), idx, _mm512_load_si512(t + 3));
        const __m512i e = _mm512_permutex2var_epi8(_mm512_load_si512(t + 4), idx, _mm512_load_si512(t + 5));
        const __m512i f = _mm512_permutex2var_epi8(_mm512_load_si512(t + 6), idx, _mm512_load_si512(t + 7));
        r[k] = _mm512_mask_blend_epi8(m8, _mm512_mask_blend_epi8(m7, a, c), _mm512_mask_blend_epi8(m7, e, f));
    }
    const __mmask64 hi2 = _cvtu64_mask64(0xCCCCCCCCCCCCCCCCull);   // bytes 2, 3 of every 4: t_2, t_3's
    for (int j = 0; j < 4; j++) {
        const __m512i il = _mm512_load_si512((const void*) T.ilv[j]);
        g[j] = _mm512_mask_blend_epi8(hi2, _mm512_permutex2var_epi8(r[0], il, r[1]),
                                      _mm512_permutex2var_epi8(r[2], il, r[3]));
    }
}

// Strata's row_dot<21, NT> with the decode above
template <int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    constexpr int BYTES = 110;
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * BYTES;
        if (const int pf = prefetch_ahead()) {
            _mm_prefetch((const char*) blk + pf, _MM_HINT_T0);
            _mm_prefetch((const char*) blk + pf + 64, _MM_HINT_T0);
        }
        __m512i g[4];
        decode(blk, g);
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __mmask64 m = _cvtu64_mask64(u64(blk + 74 + 8 * j));
            const uint8_t s = blk[106 + j];
            const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
            const __m512i sc = scales4(sa, sa, sb, sb);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g[j], ys), sc));
            }
        }
        const float dx = h2f((uint16_t) (blk[0] | blk[1] << 8));
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

// Strata's gu_rows<21, NT>: ff[t][r] = silu(min(gate . y_t, lim)) * clamp(up . y_t, -lim, lim)
template <int NT>
inline void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
                    int r0, int r1, float lim) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) {
            const float gc = std::min(g[t], lim), uc = std::min(std::max(u[t], -lim), lim);
            ff[t][r] = (gc / (1.f + std::exp(-gc))) * uc;
        }
    }
}

inline void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       float* const* ff, int r0, int r1, float lim) {
    switch (nt) {
        case 1: gu_rows<1>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        case 2: gu_rows<2>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        case 3: gu_rows<3>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        case 4: gu_rows<4>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        case 5: gu_rows<5>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        case 6: gu_rows<6>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        case 7: gu_rows<7>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
        default: gu_rows<8>(blob, gu_row, up_off, n, act, ff, r0, r1, lim); break;
    }
}

}  // namespace cl3s
