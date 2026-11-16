// SFPU decode of 4-bit expert codes for the decode-time expert kernels (dexp_compute.cpp), both of Flash-Next's
// expert formats in one kernel (iq_decode_sfpu.h picks one per compile): codes in Dest 0 (one uint32 word = 8 codes,
// nibble t = k-tile 8c + t of a code tile), the f(code) values of nibbles 4 PASS + 0..3 into Dest 4..7.
//   linear  2c - 15: IQ3_S repacked exactly (odd magnitudes 1..15, the sign folded into the code)
//   iq4nl   kvalues_iq4nl[c]: a degree-4 polynomial in c with a0 = -127 + 1.5 * 2^23, so the last multiply-add
//           rounds to an integer by itself (iq_decode_sfpu.h's IQ4_MAGIC); its top three coefficients live in the
//           programmable constant registers: iq4nl_init() before a phase's first decode (other SFPU inits may
//           overwrite them)
// Nibbles come out with two shifts (iq_decode_sfpu.h's NOMASK); two chains are written side by side (its PAIR).
#pragma once

#include "sfpi.h"

namespace ckernel::sfpu::dexp {

using sfpi::dst_reg;
using sfpi::vFloat;
using sfpi::vUInt;

constexpr int TR = 32;                 // one tile in SFPU row addressing
constexpr float MAGIC = 12582912.0f;   // 1.5 * 2^23
constexpr float P4 = 0.0014480184763669968f, P3 = 0.03627324849367142f, P2 = -1.4385637044906616f,
                P1 = 24.500598907470703f, P0 = -127.0f + MAGIC;

inline void iq4nl_init() {
    sfpi::vConstFloatPrgm0 = P4;
    sfpi::vConstFloatPrgm1 = P3;
    sfpi::vConstFloatPrgm2 = P2;
}

template <int S>
sfpi_inline vUInt nibble(vUInt w) {
    if constexpr (S == 28) {
        return w >> 28;
    } else {
        return (w << (28 - S)) >> 28;
    }
}

sfpi_inline vFloat to_f(vUInt c) { return sfpi::convert<vFloat>(sfpi::as<sfpi::vSMag>(c), sfpi::RoundMode::Nearest); }

template <int PASS>
inline void decode_linear() {
    const vFloat two = 2.0f, m15 = -15.0f;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat a = to_f(nibble<16 * PASS + 0>(w)), b = to_f(nibble<16 * PASS + 4>(w));
        dst_reg[4 * TR] = a * two + m15;
        dst_reg[5 * TR] = b * two + m15;
        a = to_f(nibble<16 * PASS + 8>(w));
        b = to_f(nibble<16 * PASS + 12>(w));
        dst_reg[6 * TR] = a * two + m15;
        dst_reg[7 * TR] = b * two + m15;
        dst_reg++;
    }
}

template <int SA, int SB>
sfpi_inline void iq4nl_pair(vUInt w, vFloat p1, vFloat p0, vFloat& a, vFloat& b) {
    const vFloat fa = to_f(nibble<SA>(w)), fb = to_f(nibble<SB>(w));
    vFloat pa = sfpi::vConstFloatPrgm0 * fa + sfpi::vConstFloatPrgm1;
    vFloat pb = sfpi::vConstFloatPrgm0 * fb + sfpi::vConstFloatPrgm1;
    pa = pa * fa + sfpi::vConstFloatPrgm2;
    pb = pb * fb + sfpi::vConstFloatPrgm2;
    pa = pa * fa + p1;
    pb = pb * fb + p1;
    a = (pa * fa + p0) - MAGIC;
    b = (pb * fb + p0) - MAGIC;
}

template <int PASS>
inline void decode_iq4nl() {
    const vFloat p1 = P1, p0 = P0;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat a, b;
        iq4nl_pair<16 * PASS + 0, 16 * PASS + 4>(w, p1, p0, a, b);
        dst_reg[4 * TR] = a;
        dst_reg[5 * TR] = b;
        iq4nl_pair<16 * PASS + 8, 16 * PASS + 12>(w, p1, p0, a, b);
        dst_reg[6 * TR] = a;
        dst_reg[7 * TR] = b;
        dst_reg++;
    }
}

// all 8 nibbles in one pass: codes in Dest 0, nibbles 0..6 into Dest 1..7, nibble 7 over Dest 0 (the word is read
// first; Dest is fp32, 8 tiles)
inline void decode8_linear() {
    const vFloat two = 2.0f, m15 = -15.0f;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat a = to_f(nibble<0>(w)), b = to_f(nibble<4>(w));
        dst_reg[1 * TR] = a * two + m15;
        dst_reg[2 * TR] = b * two + m15;
        a = to_f(nibble<8>(w));
        b = to_f(nibble<12>(w));
        dst_reg[3 * TR] = a * two + m15;
        dst_reg[4 * TR] = b * two + m15;
        a = to_f(nibble<16>(w));
        b = to_f(nibble<20>(w));
        dst_reg[5 * TR] = a * two + m15;
        dst_reg[6 * TR] = b * two + m15;
        a = to_f(nibble<24>(w));
        b = to_f(nibble<28>(w));
        dst_reg[7 * TR] = a * two + m15;
        dst_reg[0] = b * two + m15;
        dst_reg++;
    }
}

inline void decode8_iq4nl() {
    const vFloat p1 = P1, p0 = P0;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat a, b;
        iq4nl_pair<0, 4>(w, p1, p0, a, b);
        dst_reg[1 * TR] = a;
        dst_reg[2 * TR] = b;
        iq4nl_pair<8, 12>(w, p1, p0, a, b);
        dst_reg[3 * TR] = a;
        dst_reg[4 * TR] = b;
        iq4nl_pair<16, 20>(w, p1, p0, a, b);
        dst_reg[5 * TR] = a;
        dst_reg[6 * TR] = b;
        iq4nl_pair<24, 28>(w, p1, p0, a, b);
        dst_reg[7 * TR] = a;
        dst_reg[0] = b;
        dst_reg++;
    }
}

// half-sync Dest (4 fp32 tiles): codes in Dest 0, nibbles F .. F + N - 1 into Dest 1 .. N (N <= 3)
template <int F, int N>
inline void decode3_linear() {
    const vFloat two = 2.0f, m15 = -15.0f;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat a = to_f(nibble<4 * F>(w));
        if constexpr (N > 1) {
            vFloat b = to_f(nibble<4 * (F + 1)>(w));
            dst_reg[2 * TR] = b * two + m15;
        }
        if constexpr (N > 2) {
            vFloat c = to_f(nibble<4 * (F + 2)>(w));
            dst_reg[3 * TR] = c * two + m15;
        }
        dst_reg[1 * TR] = a * two + m15;
        dst_reg++;
    }
}

// the three passes (a two-argument template does not fit through the SFPU call macro's arguments)
inline void decode3_pass0() { decode3_linear<0, 3>(); }
inline void decode3_pass1() { decode3_linear<3, 3>(); }
inline void decode3_pass2() { decode3_linear<6, 2>(); }

// M rows (fused/dexp_cmp.h): Dest 2 = Dest 0 * S, S = rows 0..7 of Dest 1 repeated in each of the tile's four 8-row
// blocks (a block of a face is 4 SFPU rows: block m of tile rows sits in faces 2 (m / 2) + c, SFPU rows
// 4 (m % 2) .. + 3 of them). One call for the whole tile (no face loop)
inline void rowmul0() {
#pragma GCC unroll 2
    for (int c = 0; c < 2; c++) {
#pragma GCC unroll 4
        for (int i = 0; i < 4; i++) {
            const vFloat s = dst_reg[TR + 8 * c + i];
#pragma GCC unroll 4
            for (int m = 0; m < 4; m++) {
                const int o = 8 * ((m / 2) * 2 + c) + 4 * (m % 2) + i;
                dst_reg[2 * TR + o] = dst_reg[o] * s;
            }
        }
    }
}

}  // namespace ckernel::sfpu::dexp
