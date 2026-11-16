// Card formats test (tt/formats/card_formats.py): one tile of 32-bit code words (8 codes, all from one 32-weight
// block), each word's block affine (a, b) as fp32 tiles, and for 5-bit codes a tile of top bits (bit t = code t's).
// Decodes w = a * f(code) + b on the SFPU, f one of (tt/formats/repack.py):
//   FMT_LINEAR   f(c) = c                    Q4_K, Q5_K (HI_BIT), IQ3_S repacked
//   FMT_IQ4NL    f(c) = kvalues_iq4nl[c]     IQ4_NL, IQ4_XS: degree-4 polynomial, magic rounding
//   FMT_E2M1     f(c) = 2 x E2M1(c)          MXFP4: degree-3 polynomial on the magnitude, magic rounding, sign
// Dest: 0 codes, 1 a, 2 b, 3 top bits; 4..7 outputs. Two passes of 4 nibbles (Dest holds 8 fp32 tiles).
// REPS == 0: correctness (8 output tiles, nibble order). REPS > 0: pass 0 looped over Dest, for cycles per row.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {

using sfpi::dst_reg;
using sfpi::vFloat;
using sfpi::vUInt;

constexpr uint32_t TILE_ROWS = 32;
constexpr float MAGIC = 12582912.0f;  // 1.5 * 2^23: an fp32 add at this magnitude rounds to an integer

#if defined(FMT_IQ4NL)
// kvalues_iq4nl[c] = round(P4 c^4 + P3 c^3 + P2 c^2 + P1 c - 127), exact for c = 0..15 (max error 0.44)
constexpr float P4 = 0.0014480184763669968f, P3 = 0.03627324849367142f, P2 = -1.4385637044906616f,
                P1 = 24.500598907470703f, P0 = -127.0f + MAGIC;
#elif defined(FMT_E2M1)
// 2 x E2M1 magnitude = round(P3 c^3 + P2 c^2 + P1 c) for c = 0..7 (max error 0.23)
constexpr float P3 = 0.0469696968793869f, P2 = -0.30000001192092896f, P1 = 1.4803030490875244f, P0 = MAGIC;
#endif

inline void format_init() {
#if defined(FMT_IQ4NL)
    sfpi::vConstFloatPrgm0 = P4;
    sfpi::vConstFloatPrgm1 = P3;
    sfpi::vConstFloatPrgm2 = P2;
#elif defined(FMT_E2M1)
    sfpi::vConstFloatPrgm0 = P3;
    sfpi::vConstFloatPrgm1 = P2;
#endif
}

sfpi_inline vFloat to_float(vUInt c) {
    return sfpi::convert<vFloat>(sfpi::as<sfpi::vSMag>(c), sfpi::RoundMode::Nearest);
}

// nibble T of w (and, for 5-bit codes, bit T of h) -> w = a * f(code) + b
template <int T>
sfpi_inline vFloat weight(vUInt w, vUInt h, vFloat a, vFloat b, vFloat p1, vFloat p0) {
    constexpr int s = 4 * T;
    vUInt top;  // code T at bits 28..31
    if constexpr (s == 28) {
        top = w;
    } else {
        top = w << (28 - s);
    }
#if defined(FMT_E2M1)
    vFloat f = to_float((top << 1) >> 29);
    vFloat p = ((sfpi::vConstFloatPrgm0 * f + sfpi::vConstFloatPrgm1) * f + p1) * f + p0;
    return sfpi::copysgn(p - MAGIC, sfpi::as<vFloat>(top)) * a;
#else
    vUInt c = top >> 28;
#ifdef HI_BIT
    if constexpr (T == 7) {
        c = c + ((h >> 7) << 4);
    } else {
        c = c + (((h << (31 - T)) >> 31) << 4);
    }
#endif
    vFloat f = to_float(c);
#if defined(FMT_IQ4NL)
    vFloat p = (((sfpi::vConstFloatPrgm0 * f + sfpi::vConstFloatPrgm1) * f + sfpi::vConstFloatPrgm2) * f + p1) * f +
               p0;
    return (p - MAGIC) * a;
#else
    return f * a + b;
#endif
#endif
}

template <int PASS>
inline void format_face() {
#if defined(FMT_IQ4NL) || defined(FMT_E2M1)
    const vFloat p1 = P1, p0 = P0;
#else
    const vFloat p1 = 0.0f, p0 = 0.0f;
#endif
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat a = dst_reg[1 * TILE_ROWS];
#if defined(FMT_LINEAR)
        vFloat b = dst_reg[2 * TILE_ROWS];
#else
        vFloat b = 0.0f;
#endif
#ifdef HI_BIT
        vUInt h = dst_reg[3 * TILE_ROWS].mode<sfpi::DataLayout::U32>();
#else
        vUInt h = 0;
#endif
        dst_reg[4 * TILE_ROWS] = weight<4 * PASS + 0>(w, h, a, b, p1, p0);
        dst_reg[5 * TILE_ROWS] = weight<4 * PASS + 1>(w, h, a, b, p1, p0);
        dst_reg[6 * TILE_ROWS] = weight<4 * PASS + 2>(w, h, a, b, p1, p0);
        dst_reg[7 * TILE_ROWS] = weight<4 * PASS + 3>(w, h, a, b, p1, p0);
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t cb_codes = 0, cb_a = 1, cb_b = 2, cb_hi = 3, cb_out = 16;

    compute_kernel_hw_startup(cb_codes, cb_out);
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup; no tanh state is used
    MATH((ckernel::sfpu::format_init()));
    cb_wait_front(cb_codes, 1);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_wait_front(cb_hi, 1);
    cb_reserve_back(cb_out, 8);

    for (uint32_t pass = 0; pass < 2; pass++) {
        tile_regs_acquire();
        copy_init(cb_codes);
        copy_tile(cb_codes, 0, 0);
        copy_init(cb_a);
        copy_tile(cb_a, 0, 1);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 2);
        copy_init(cb_hi);
        copy_tile(cb_hi, 0, 3);
        if constexpr (reps == 0) {
            if (pass == 0) {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, format_face<0>, 0, VectorMode::RC)));
            } else {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, format_face<1>, 0, VectorMode::RC)));
            }
        } else {
            for (uint32_t r = 0; r < reps; r++) {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, format_face<0>, 0, VectorMode::RC)));
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 4; t < 8; t++) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();
    }
    cb_push_back(cb_out, 8);
    cb_pop_front(cb_codes, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    cb_pop_front(cb_hi, 1);
}
