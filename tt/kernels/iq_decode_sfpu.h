// SFPU decode of 4-bit expert codes to fp32, shared by iq_decode.cpp and expert_stream.cpp.
// One 32-bit word holds 8 codes (nibble t at bits 4t..4t+3); the decode handles one 32-lane row of words at a time.
//   DECODE_LINEAR  2c - 15: IQ3_S repacked exactly (its grid holds only odd magnitudes 1..15, the sign is separate)
//   DECODE_IQ4NL   kvalues_iq4nl[c]: a degree-4 polynomial in c, rounded to the nearest integer (exact for all 16)
//   DECODE_MXFP4   2 x E2M1 (0, 1, 2, 3, 4, 6, 8, 12 with the sign bit on top): a degree-3 polynomial in the
//                  3-bit magnitude, rounded, then the code's sign; the x2 folds into MXFP4's power-of-two block scale
// Block scales are left out (1.0); a kernel applies them once per block of partial sums.
//
// Options:
//   FOLD_OFFSET  linear: decode to c only; 2c - 15 is applied per block (2 * sum(c x) - 15 * sum(x))
//   IQ4_MAGIC    IQ4_NL: a fit with a0 = -127 exactly, so a0 + 1.5 * 2^23 is an fp32 integer and the last MAD
//                rounds to an integer by itself; one subtraction takes the magic back off (MXFP4 always does this)
//   NOMASK       extract a nibble with two shifts instead of shift + mask (same count, one LReg free)
#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

using sfpi::dst_reg;
using sfpi::vFloat;
using sfpi::vUInt;

constexpr uint32_t TILE_ROWS = 32;  // one tile in SFPU row addressing
constexpr float MAGIC = 12582912.0f;  // 1.5 * 2^23: an fp32 add at this magnitude rounds to an integer

#if defined(DECODE_IQ4NL)
#ifdef IQ4_MAGIC
constexpr float P4 = 0.0014480184763669968f, P3 = 0.03627324849367142f, P2 = -1.4385637044906616f,
                P1 = 24.500598907470703f, P0 = -127.0f + MAGIC;
#else
constexpr float P4 = 0.0015860428102314472f, P3 = 0.03108643926680088f, P2 = -1.37192702293396f,
                P1 = 24.173830032348633f, P0 = -126.56700897216797f;
#endif
#elif defined(DECODE_MXFP4)
// 2 x E2M1 magnitude = round(P3 c^3 + P2 c^2 + P1 c) for c = 0..7 (max error 0.23)
constexpr float P3 = 0.0469696968793869f, P2 = -0.30000001192092896f, P1 = 1.4803030490875244f, P0 = MAGIC;
#endif

// Constants live in registers for the whole face: as immediates, each use costs one SFPLOADI (fp32: two), which
// doubled the decode's cost. The polynomials' highest coefficients sit in the programmable constant registers
// (decode_init; Blackhole has three); the rest, and the mask, in LRegs.
struct Consts {
#ifndef NOMASK
    vUInt mask;
#endif
#if defined(DECODE_IQ4NL) || defined(DECODE_MXFP4)
    vFloat p1, p0;
#elif !defined(FOLD_OFFSET)
    vFloat two, minus15;
#endif
};

sfpi_inline Consts make_consts() {
#if defined(NOMASK)
#if defined(DECODE_IQ4NL) || defined(DECODE_MXFP4)
    return Consts{P1, P0};
#elif defined(FOLD_OFFSET)
    return Consts{};
#else
    return Consts{2.0f, -15.0f};
#endif
#else
#if defined(DECODE_IQ4NL) || defined(DECODE_MXFP4)
    return Consts{0xF, P1, P0};
#elif defined(FOLD_OFFSET)
    return Consts{0xF};
#else
    return Consts{0xF, 2.0f, -15.0f};
#endif
#endif
}

inline void decode_init() {
#if defined(DECODE_IQ4NL)
    sfpi::vConstFloatPrgm0 = P4;
    sfpi::vConstFloatPrgm1 = P3;
    sfpi::vConstFloatPrgm2 = P2;
#elif defined(DECODE_MXFP4)
    sfpi::vConstFloatPrgm0 = P3;
    sfpi::vConstFloatPrgm1 = P2;
#endif
}

template <int SHIFT>
sfpi_inline vUInt code(vUInt w, const Consts& k) {
#ifdef NOMASK
    // two shifts cost the same as a shift and a mask, and free the mask's LReg
    if constexpr (SHIFT == 28) {
        return w >> SHIFT;
    } else {
        return (w << (28 - SHIFT)) >> 28;
    }
#else
    if constexpr (SHIFT == 0) {
        return w & k.mask;
    } else if constexpr (SHIFT == 28) {
        return w >> SHIFT;  // a logical shift leaves only the top nibble
    } else {
        return (w >> SHIFT) & k.mask;
    }
#endif
}

sfpi_inline vFloat to_float(vUInt c) {
    return sfpi::convert<vFloat>(sfpi::as<sfpi::vSMag>(c), sfpi::RoundMode::Nearest);
}

#ifdef DECODE_IQ4NL
sfpi_inline vFloat poly_head(vFloat f, const Consts& k) {
    return ((sfpi::vConstFloatPrgm0 * f + sfpi::vConstFloatPrgm1) * f + sfpi::vConstFloatPrgm2) * f + k.p1;
}

sfpi_inline vFloat poly_tail(vFloat p, vFloat f, const Consts& k) {
#ifdef IQ4_MAGIC
    return (p * f + k.p0) - MAGIC;
#else
    p = p * f + k.p0;  // the fit's error is under 0.43, so no ties
    return sfpi::convert<vFloat>(sfpi::convert<sfpi::vSMag16>(p, sfpi::RoundMode::Nearest), sfpi::RoundMode::Nearest);
#endif
}
#endif

sfpi_inline vFloat value(vFloat f, const Consts& k) {
#if defined(DECODE_IQ4NL)
    return poly_tail(poly_head(f, k), f, k);
#elif defined(DECODE_LINEAR) && !defined(FOLD_OFFSET)
    return f * k.two + k.minus15;
#else
    return f;  // FOLD_OFFSET; MXFP4 decodes in nib()
#endif
}

template <int SHIFT>
sfpi_inline vFloat nib(vUInt w, const Consts& k) {
#if defined(DECODE_MXFP4)
    // the code to the top of the word: its sign at bit 31, its magnitude at bits 28..30
    vUInt s;
    if constexpr (SHIFT == 28) {
        s = w;
    } else {
        s = w << (28 - SHIFT);
    }
    vFloat f = to_float((s << 1) >> 29);
    vFloat p = ((sfpi::vConstFloatPrgm0 * f + sfpi::vConstFloatPrgm1) * f + k.p1) * f + k.p0;
    return sfpi::copysgn(p - MAGIC, sfpi::as<vFloat>(s));
#else
    return value(to_float(code<SHIFT>(w, k)), k);
#endif
}

// Two nibbles, written step by step side by side so the two dependency chains interleave.
template <int SA, int SB>
sfpi_inline void pair(vUInt w, const Consts& k, vFloat& a, vFloat& b) {
#if defined(DECODE_MXFP4)
    a = nib<SA>(w, k);
    b = nib<SB>(w, k);
#else
    vFloat fa = to_float(code<SA>(w, k));
    vFloat fb = to_float(code<SB>(w, k));
#if defined(DECODE_IQ4NL)
    vFloat pa = sfpi::vConstFloatPrgm0 * fa + sfpi::vConstFloatPrgm1;
    vFloat pb = sfpi::vConstFloatPrgm0 * fb + sfpi::vConstFloatPrgm1;
    pa = pa * fa + sfpi::vConstFloatPrgm2;
    pb = pb * fb + sfpi::vConstFloatPrgm2;
    pa = pa * fa + k.p1;
    pb = pb * fb + k.p1;
    a = poly_tail(pa, fa, k);
    b = poly_tail(pb, fb, k);
#else
    a = value(fa, k);
    b = value(fb, k);
#endif
#endif
}

}  // namespace ckernel::sfpu
