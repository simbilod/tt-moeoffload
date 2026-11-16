// GLM's KDA decode step, compute: Flash-Next's fused delta rule (fgdn_compute.cpp's phase B) with
// KDA's decay per key channel instead of GDN's scalar per head. One value head per core, its state S [128 k x 128 v]
// as fp32 tiles (j, c) at 4 j + c, kept twice as fgdn does (cb 8 exact for the update, unpacked to Dest in fp32; cb 9
// the FPU's operand copy). d [128] the head's decays (row 0 of 4 tiles), b its beta:
//   k' = -b (d * k)                    (from the zero-padded k: k' only feeds the FPU)
//   delta = b v + k'^T S               (= b (v - k^T D S))
//   S = S * Dbc + k delta^T            the SFPU in fp32, the outer product on the FPU; Dbc_j holds d[32 j + r] across
//                                      row r (built exactly by the reader), so D S is never rounded to the FPU's TF32
//   o = q^T S                          (q carries GLM's 1 / sqrt(128))
//   out = o / rms(o) * w * sigmoid(g2) (GLM's KDA output norm, eps 1e-5)
// fgdn's state_update4 does d S + P for 4 tiles of a row block at once; with Dbc in Dest too, a row block goes in two
// halves (Dest holds 8 fp32 tiles in full-sync mode).
// The new state goes out exactly (cb 19, 2 tiles at a time) and as the operand for o (cb 20); out to cb 22 (fp32).
// Runtime args: active (0: idle core).
#include <cstdint>

#include "fused/cmp.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Dest 0..3 *= -(Dest 4) (beta, filled across row 0)
inline void neg_by4() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat b = sfpi::dst_reg[4 * 32];
        const sfpi::vFloat s = -b;
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * s;
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * s;
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * s;
        sfpi::dst_reg[96] = sfpi::dst_reg[96] * s;
        sfpi::dst_reg++;
    }
}
// Dest 0..1 = S, 2..3 = P, 4 = Dbc: Dest 0..1 = S * Dbc + P
inline void kda_update2() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat d = sfpi::dst_reg[4 * 32];
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * d + sfpi::dst_reg[2 * 32];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * d + sfpi::dst_reg[3 * 32];
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

using namespace fused;

constexpr uint32_t cb_ones = 2, cb_sq = 5, cb_r = 6, cb_kp = 7, cb_sx = 8, cb_sop = 9, cb_nw = 10, cb_q = 11;
constexpr uint32_t cb_k = 12, cb_ex = 13, cb_bv = 14, cb_kc = 15, cb_sxo = 19, cb_sop2 = 20, cb_hchk = 22;
constexpr uint32_t cb_delta = 23, cb_o = 24, cb_kd = 25, cb_dbc = 26;
constexpr uint32_t F_EPS5 = 0x3727c5ac;  // 1e-5

void kernel_main() {
    if (!get_arg_val<uint32_t>(0)) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kp, cb_sop, cb_delta);
    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_q, 4);
    cb_wait_front(cb_k, 4);
    cb_wait_front(cb_ex, 10);  // v 0..3, sigmoid(g2) 4..7, beta 8 (across row 0), 9 unused
    cb_wait_front(cb_nw, 4);
    cb_wait_front(cb_kd, 4);

    // k' = -b (d * k)
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; t++) load(cb_k, t, t);
    for (uint32_t t = 0; t < 4; t++) load(cb_kd, t, 4 + t);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) mul_binary_tile(t, 4 + t, t);
    load(cb_ex, 8, 4);
    SFPU_CALL(neg_by4);
    finish(cb_kp, 4);
    cb_pop_front(cb_kd, 4);

    // b v (exact)
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; t++) load(cb_ex, t, t);
    load(cb_ex, 8, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) mul_binary_tile(t, 4, t);
    finish(cb_bv, 4);

    // k as columns
    tile_regs_acquire();
    reconfig_data_format_srca(cb_k);
    transpose_init(cb_k);
    for (uint32_t t = 0; t < 4; t++) transpose_tile(cb_k, t, t);
    finish(cb_kc, 4);
    cb_pop_front(cb_k, 4);

    // delta = b v + k'^T S
    cb_wait_front(cb_bv, 4);
    cb_wait_front(cb_kp, 4);
    cb_wait_front(cb_sop, 16);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; t++) load(cb_bv, t, t);
    matmul_start(cb_kp, cb_sop);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) matmul_tiles(cb_kp, cb_sop, j, 4 * j + c, c);
    }
    tile_regs_commit();
    cb_pop_front(cb_bv, 4);
    cb_pop_front(cb_kp, 4);
    cb_pop_front(cb_sop, 16);
    tile_regs_wait();
    pack_n(cb_delta, 4);
    tile_regs_release();

    // S = S * Dbc + k delta^T, half a row block (2 state tiles) at a time, into both copies
    cb_wait_front(cb_delta, 4);
    cb_wait_front(cb_kc, 4);
    cb_wait_front(cb_sx, 16);
    cb_wait_front(cb_dbc, 4);
    for (uint32_t j = 0; j < 4; j++) {
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t c0 = 2 * h;
            tile_regs_acquire();
            load(cb_sx, 4 * j + c0, 0);
            load(cb_sx, 4 * j + c0 + 1, 1);
            load(cb_dbc, j, 4);
            matmul_start(cb_kc, cb_delta);
            matmul_tiles(cb_kc, cb_delta, j, c0, 2);
            matmul_tiles(cb_kc, cb_delta, j, c0 + 1, 3);
            SFPU_CALL(kda_update2);
            tile_regs_commit();
            tile_regs_wait();
            pack_n(cb_sxo, 2);
            pack_n(cb_sop2, 2);
            tile_regs_release();
        }
    }
    cb_pop_front(cb_delta, 4);
    cb_pop_front(cb_kc, 4);
    cb_pop_front(cb_sx, 16);
    cb_pop_front(cb_dbc, 4);

    // o = q^T S
    cb_wait_front(cb_sop2, 16);
    tile_regs_acquire();
    matmul_start(cb_q, cb_sop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) matmul_tiles(cb_q, cb_sop2, j, 4 * j + c, c);
    }
    tile_regs_commit();
    cb_pop_front(cb_q, 4);
    cb_pop_front(cb_sop2, 16);
    tile_regs_wait();
    pack_n(cb_o, 4);
    tile_regs_release();

    // out = o * rsqrt(mean o^2 + 1e-5) * w * sigmoid(g2)
    cb_wait_front(cb_o, 4);
    rsqrt_of_sum(cb_o, 4, F_INV128, cb_sq, cb_ones, F_EPS5);
    finish(cb_r, 1);
    cb_wait_front(cb_r, 1);
    for (uint32_t c = 0; c < 4; c++) {
        tile_regs_acquire();
        load(cb_o, c, 0);
        load(cb_nw, c, 1);
        load(cb_ex, 4 + c, 2);
        load(cb_r, 0, 3);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(0, 3, 0);
        finish(cb_hchk, 1);
    }
    cb_pop_front(cb_r, 1);
    cb_pop_front(cb_o, 4);
    cb_pop_front(cb_ex, 10);
    cb_pop_front(cb_nw, 4);
    cb_pop_front(cb_ones, 1);
}
