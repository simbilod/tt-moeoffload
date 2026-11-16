// Prefill experts compute (gexpq_compute.cpp's, for units of up to MT_MAX token tiles). Per unit and n-tile:
//   decode   each code tile on the SFPU (iq_decode_sfpu.h: DECODE_LINEAR 2c - 15 for IQ3_S, DECODE_IQ4NL the IQ4_NL
//            table, also IQ4_XS's) into 8 F tiles in Dest, 4 at a time; VALUES (Q8_0): F are the reader's bfp8_b value
//            tiles (cb 25, int8 exactly), copied in. Then W = F m d exactly on the SFPU: each k-tile's m row (cb 21,
//            bfp8_b; HAS_M) and its block's d row (cb 22, fp32) broadcast down a Dest tile (unary_bcast<ROW>) and
//            multiplied in (the writer makes the rows from the compact scales), into cb 26 in fp32: the n-tile's
//            whole column of kt weight tiles, decoded once per unit (step 3: the GGUF's weights exactly, where the
//            FPU's scale multiply took the scale as a tf32 operand)
//   matmul   the kt x mt matmuls x_u[m, k] W[k, n] accumulated in Dest 0..mt-1, packed one page a push (cb 16)
//   SWIGLU   (gate|up, its columns in pairs: pf_gexp_reader.cpp) the gate tiles packed to cb 18, the up tiles to
//            cb 19, then per token tile h = silu(gate) * up on the SFPU into cb 16: the output is h [R, FF];
//            SWIGLU_LIMIT (the bits of a float, GLM's 10): gate clamped above, up both ways first (fused/cmp.h)
// cb 0 codes (uint32, unpacked to Dest), cb 21 / 22 the m / d rows, cb 26 W (fp32), cb 1 x (bf16).
// Runtime args: unit count, then per unit its token tiles. Compile args: kt, nt, mt_max, kc, kr: the k-tiles that
// hold weights (down: 20 of 24, K 640 padded to 768; the padding's scale rows still taken from the writer, its
// decode, pack and matmul skipped, cb 26's places for them pushed unwritten: h is zero there, so each sum is the
// same; 2026-11-08, scope item a).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#ifdef SWIGLU_LIMIT
#include "fused/cmp.h"                           // clamp_gu, SFPU_CALL
#endif

#if defined(TRISC_MATH) && !defined(VALUES)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "iq_decode_sfpu.h"

namespace ckernel::sfpu {

// codes in Dest 0 -> f(code) of nibbles 4 * PASS + 0..3 in Dest 4..7 (gexpq_compute.cpp's), each times the scale
// row broadcast there beforehand (HAS_M: m_t, and the block's d in Dest 1; else d_t): W = f m d exactly, in lanes
#ifdef HAS_M
#define SCALED(t, v) ((v) * vFloat(dst_reg[(t) * TILE_ROWS]) * dd)
#else
#define SCALED(t, v) ((v) * vFloat(dst_reg[(t) * TILE_ROWS]))
#endif
template <int PASS>
inline void decode_face() {
    const Consts k = make_consts();
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
#ifdef HAS_M
        const vFloat dd = dst_reg[1 * TILE_ROWS];
#endif
#ifdef PAIR
        vFloat a, b;
        pair<16 * PASS + 0, 16 * PASS + 4>(w, k, a, b);
        dst_reg[4 * TILE_ROWS] = SCALED(4, a);
        dst_reg[5 * TILE_ROWS] = SCALED(5, b);
        pair<16 * PASS + 8, 16 * PASS + 12>(w, k, a, b);
        dst_reg[6 * TILE_ROWS] = SCALED(6, a);
        dst_reg[7 * TILE_ROWS] = SCALED(7, b);
#else
        dst_reg[4 * TILE_ROWS] = SCALED(4, nib<16 * PASS + 0>(w, k));
        dst_reg[5 * TILE_ROWS] = SCALED(5, nib<16 * PASS + 4>(w, k));
        dst_reg[6 * TILE_ROWS] = SCALED(6, nib<16 * PASS + 8>(w, k));
        dst_reg[7 * TILE_ROWS] = SCALED(7, nib<16 * PASS + 12>(w, k));
#endif
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

// zones for tt-metal's device profiler (TT_MOEOFFLOAD_ZONES, with TT_MOEOFFLOAD_TTNN=tracy and TT_METAL_DEVICE_PROFILER=1): the first
// unit's first two n-tiles' decode (GX-DEC) and matmul + pack (GX-MM); nothing otherwise (2026-11-09)
#ifdef TT_MOEOFFLOAD_ZONES
#include "api/debug/kernel_profiler.hpp"
#endif

constexpr uint32_t cb_codes = 0, cb_x = 1, cb_mr = 21, cb_dr = 22, cb_v = 25, cb_w = 26, cb_out = 16;
constexpr uint32_t cb_gate = 18, cb_up = 19;

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t mt_max = get_compile_time_arg_val(2);
    constexpr uint32_t kc = get_compile_time_arg_val(3);
    constexpr uint32_t kr = get_compile_time_arg_val(4);
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    for (uint32_t u = 0; u < n_units; u++) {
        const uint32_t mt = get_arg_val<uint32_t>(1 + u);
        cb_wait_front(cb_x, mt_max * kt);
        for (uint32_t n = 0; n < nt; n++) {
            auto decode_col = [&]() {
            for (uint32_t c = 0; c < kc; c++) {
#ifdef VALUES
                cb_wait_front(cb_v, 8);
#else
                cb_wait_front(cb_codes, 1);
#endif
                for (uint32_t pass = 0; pass < 2; pass++) {
                    if (8 * c + 4 * pass >= kr) {  // the padding's k-tiles: their scale rows taken, nothing made
#if defined(SKIP_ROWPUSH) && !defined(VALUES)
#elif defined(HAS_M)
                        cb_wait_front(cb_mr, 4);
                        cb_pop_front(cb_mr, 4);
                        cb_wait_front(cb_dr, 1);
                        cb_pop_front(cb_dr, 1);
#else
                        cb_wait_front(cb_dr, 4);
                        cb_pop_front(cb_dr, 4);
#endif
                        // their 4 places in cb 26 pushed unwritten (never read): the CB's rounds stay kt tiles, so
                        // the next n-tile's tiles do not wrap past its end (a matmul's tile index does not wrap)
                        cb_reserve_back(cb_w, 4);
                        cb_push_back(cb_w, 4);
                        continue;
                    }
                    tile_regs_acquire();
#if defined(VALUES) && defined(HAS_M)
                    // GLM's re-quantized Q5_K downs (g7_experts.piece): F (bfp8_b values, exact) into Dest 0..3, the
                    // m rows broadcast into 4..7, F m (exact in fp32), then the block's d row into Dest 0 and W = F m d
                    cb_wait_front(cb_mr, 4);
                    cb_wait_front(cb_dr, 1);
                    reconfig_data_format(cb_mr, cb_mr);
                    unary_bcast_init<BroadcastType::ROW>(cb_mr);
                    for (uint32_t i = 0; i < 4; i++) {
                        unary_bcast<BroadcastType::ROW>(cb_mr, i, 4 + i);
                    }
                    reconfig_data_format_srca(cb_v);
                    copy_init(cb_v);
                    for (uint32_t i = 0; i < 4; i++) {
                        copy_tile(cb_v, 4 * pass + i, i);
                    }
                    mul_binary_tile_init();
                    for (uint32_t i = 0; i < 4; i++) {
                        mul_binary_tile(4 + i, i, 4 + i);
                    }
                    reconfig_data_format(cb_dr, cb_dr);
                    unary_bcast_init<BroadcastType::ROW>(cb_dr);
                    unary_bcast<BroadcastType::ROW>(cb_dr, 0, 0);
                    mul_binary_tile_init();
                    for (uint32_t i = 0; i < 4; i++) {
                        mul_binary_tile(4 + i, 0, 4 + i);
                    }
                    cb_pop_front(cb_mr, 4);
                    cb_pop_front(cb_dr, 1);
#elif defined(VALUES)
                    // F (bfp8_b values, exact) into Dest 0..3, the d rows broadcast into 4..7, multiplied on the SFPU
                    cb_wait_front(cb_dr, 4);
                    reconfig_data_format(cb_dr, cb_dr);
                    unary_bcast_init<BroadcastType::ROW>(cb_dr);
                    for (uint32_t i = 0; i < 4; i++) {
                        unary_bcast<BroadcastType::ROW>(cb_dr, i, 4 + i);
                    }
                    reconfig_data_format_srca(cb_v);
                    copy_init(cb_v);
                    for (uint32_t i = 0; i < 4; i++) {
                        copy_tile(cb_v, 4 * pass + i, i);
                    }
#ifndef SKIP_SCALE
                    mul_binary_tile_init();
                    for (uint32_t i = 0; i < 4; i++) {
                        mul_binary_tile(4 + i, i, 4 + i);
                    }
#endif
                    cb_pop_front(cb_dr, 4);
#else
                    // the scale rows of k-tiles 8 c + 4 pass + 0..3 broadcast into Dest 4..7 (m_t, or d_t), HAS_M's
                    // block d into Dest 1; then the codes into Dest 0 and the decode writes f m d over them
#ifdef SKIP_ROWPUSH                     // timing only: no scale rows from the writer at all (nor their broadcasts)
#elif defined(HAS_M)
                    cb_wait_front(cb_mr, 4);
                    cb_wait_front(cb_dr, 1);
#ifndef SKIP_BCAST                       // timing only: the scale rows not broadcast
                    reconfig_data_format(cb_mr, cb_mr);
                    unary_bcast_init<BroadcastType::ROW>(cb_mr);
                    for (uint32_t i = 0; i < 4; i++) {
                        unary_bcast<BroadcastType::ROW>(cb_mr, i, 4 + i);
                    }
                    reconfig_data_format(cb_dr, cb_dr);
                    unary_bcast_init<BroadcastType::ROW>(cb_dr);
                    unary_bcast<BroadcastType::ROW>(cb_dr, 0, 1);
#endif
#else
                    cb_wait_front(cb_dr, 4);
#ifndef SKIP_BCAST
                    reconfig_data_format(cb_dr, cb_dr);
                    unary_bcast_init<BroadcastType::ROW>(cb_dr);
                    for (uint32_t i = 0; i < 4; i++) {
                        unary_bcast<BroadcastType::ROW>(cb_dr, i, 4 + i);
                    }
#endif
#endif
                    reconfig_data_format_srca(cb_codes);
                    copy_init(cb_codes);
                    copy_tile(cb_codes, 0, 0);
                    MATH((SFPU_UNARY_INIT(tanh)));       // generic SFPU address setup; no tanh state is used
                    MATH((ckernel::sfpu::decode_init()));
#ifdef SKIP_DECODE
                    if (false) {
#else
                    if (pass == 0) {
#endif
                        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, decode_face<0>, 0,
                                                               VectorMode::RC)));
                    } else {
#ifndef SKIP_DECODE
                        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, decode_face<1>, 0,
                                                               VectorMode::RC)));
#endif
                    }
#ifdef SKIP_ROWPUSH
#elif defined(HAS_M)
                    cb_pop_front(cb_mr, 4);
                    cb_pop_front(cb_dr, 1);
#else
                    cb_pop_front(cb_dr, 4);
#endif
#endif
                    tile_regs_commit();
                    cb_reserve_back(cb_w, 4);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_w);
#ifndef SKIP_PACK                        // timing only: the decoded tiles not packed (cb 26 pushed as it was)
                    for (uint32_t t = 4; t < 8; t++) {
                        pack_tile(t, cb_w);
                    }
#endif
                    tile_regs_release();
                    cb_push_back(cb_w, 4);
                }
#ifdef VALUES
                cb_pop_front(cb_v, 8);
#else
                cb_pop_front(cb_codes, 1);
#endif
            }

            };
#ifdef TT_MOEOFFLOAD_ZONES
            if (u == 0 && n < 2) {
                DeviceZoneScopedN("GX-DEC");
                decode_col();
            } else {
                decode_col();
            }
#else
            decode_col();
#endif
            auto matmul_col = [&]() {
            cb_wait_front(cb_w, kt);
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_w);
            matmul_init(cb_x, cb_w);
#ifndef SKIP_MATMUL                        // timing only: Dest left as it was
            for (uint32_t k = 0; k < kr; k++) {
                for (uint32_t m = 0; m < mt; m++) {
                    matmul_tiles(cb_x, cb_w, m * kt + k, k, m);
                }
            }
#endif
            tile_regs_commit();
            cb_pop_front(cb_w, kt);
            };
#ifdef TT_MOEOFFLOAD_ZONES
            if (u == 0 && n < 2) {
                DeviceZoneScopedN("GX-MM");
                matmul_col();
            } else {
                matmul_col();
            }
#else
            matmul_col();
#endif
            tile_regs_wait();
#ifdef SWIGLU
            // whole mt_max pushes (cb 18 and 19 hold mt_max tiles): a reservation of mt tiles wrapped past the CB's end
            // when mt did not divide mt_max, and the packer wrote on into the next CB in L1 (2026-11-07)
            const uint32_t cb_half = (n & 1) ? cb_up : cb_gate;
            cb_reserve_back(cb_half, mt_max);
            pack_reconfig_data_format(cb_half);
            for (uint32_t m = 0; m < mt; m++) {
                pack_tile(m, cb_half);
            }
            tile_regs_release();
            cb_push_back(cb_half, mt_max);
            if (n & 1) {
                cb_wait_front(cb_gate, mt_max);
                cb_wait_front(cb_up, mt_max);
                for (uint32_t m = 0; m < mt; m++) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_gate);
                    copy_init(cb_gate);
                    copy_tile(cb_gate, m, 0);
                    copy_init(cb_up);
                    copy_tile(cb_up, m, 1);
#ifdef SWIGLU_LIMIT                              // GLM: llama.cpp's swiglu_clamp_exp, gate <= lim, up in [-lim, lim]
                    MATH((SFPU_UNARY_INIT(tanh)));       // generic SFPU address setup; no tanh state is used
                    SFPU_CALL(clamp_gu, SWIGLU_LIMIT);
#endif
                    silu_tile_init();
                    silu_tile(0);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 0);
                    tile_regs_commit();
                    cb_reserve_back(cb_out, 1);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_out);
                    pack_tile(0, cb_out);
                    tile_regs_release();
                    cb_push_back(cb_out, 1);
                }
                cb_pop_front(cb_gate, mt_max);
                cb_pop_front(cb_up, mt_max);
            }
#else
            pack_reconfig_data_format(cb_out);
            for (uint32_t m = 0; m < mt; m++) {
                cb_reserve_back(cb_out, 1);
                pack_tile(m, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
#endif
        }
        cb_pop_front(cb_x, mt_max * kt);
    }
}
