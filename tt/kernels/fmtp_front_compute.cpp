// The MTP draft layer's front, compute (the phases are in fmtp_front_reader.cpp). Sums of squares go through the
// all-ones tile: Q 1 puts a row's sum in every column of it; (S o I) then 1 (S o I) moves row r's sum to column r of
// every row (I the identity, o elementwise), so a core's 16 row sums leave as one row piece; 1 SS sums those over the
// cores (rows), G sums a row m's four streams (columns 4 s + m), and (U o I) 1 moves the scalars back to rows.
// Runtime args: F1's item t, F2's tile n.
#include <cstdint>
#include "fused/cmp.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Dest 0 = Dest 0 * Dest 2 + (Dest 1 * Dest 3, rows 16..31 of Dest 1 onto rows 0..15): X = r A + q B, B's rows a face
// pair down (a tile is 32 SFPU rows, 8 a face; rows 16..31 of the result are left garbage)
inline void fold() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * sfpi::dst_reg[64] + sfpi::dst_reg[32 + 16] * sfpi::dst_reg[96];
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_r = 3, cb_e = 4, cb_whe = 5, cb_xh = 6, cb_xe = 7;
constexpr uint32_t cb_sq = 8, cb_d = 9, cb_id = 10, cb_g = 11, cb_v = 12, cb_c = 13, cb_rq = 14, cb_o32 = 15;
constexpr uint32_t cb_o16 = 16;
constexpr uint32_t KD = 80, SS = 80, CH = 16, IDLE = 0xffffffff;
constexpr uint32_t F_INV10240 = 0x38cccccd;  // 1 / 10240

// Dest 0..n-1 = (Dest 0..n-1 o I) after the caller's matmuls; I from cb_id into Dest 7
inline void times_identity(uint32_t n) {
    load(cb_id, 0, 7);
    mul_binary_tile_init();
    for (uint32_t k = 0; k < n; k++) {
        mul_binary_tile(k, 7, k);
    }
}

// rsqrt(column sums * scale + eps) of the SS tiles (ones on the left), through G when grouped, as row scalars: the
// result's row r = the scalar of column r, into cb_rq
inline void row_scalars(uint32_t first, bool grouped, uint32_t scale_bits) {
    tile_regs_acquire();
    matmul_start(cb_ones, cb_x);
    for (uint32_t k = 0; k < 3; k++) {
        matmul_tiles(cb_ones, cb_x, 0, first + k, 0);
    }
    if (grouped) {
        finish(cb_c, 1);
        cb_wait_front(cb_c, 1);
        tile_regs_acquire();
        matmul_start(cb_c, cb_g);
        matmul_tiles(cb_c, cb_g, 0, 0, 0);
        cb_pop_front(cb_c, 1);
    }
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup (a binary SFPU op ran since)
    SFPU_CALL(affine, scale_bits, F_EPS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    times_identity(1);
    finish(cb_c, 1);
    cb_wait_front(cb_c, 1);
    tile_regs_acquire();
    matmul_start(cb_c, cb_ones);
    matmul_tiles(cb_c, cb_ones, 0, 0, 0);
    cb_pop_front(cb_c, 1);
    finish(cb_rq, 1);
}

void kernel_main() {
    const uint32_t t = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o32);
    cb_wait_front(cb_ones, 1);

    // ---- F1: x_h = R o w_h, x_e = E o w_e (bf16), the row sums of R^2 and E^2 as row pieces (bf16)
    if (t != IDLE) {
        cb_wait_front(cb_r, 1);
        cb_wait_front(cb_e, 1);
        cb_wait_front(cb_whe, 2);
        tile_regs_acquire();
        load(cb_r, 0, 0);
        load(cb_whe, 0, 1);
        load(cb_e, 0, 2);
        load(cb_whe, 1, 3);
        load(cb_r, 0, 4);
        load(cb_e, 0, 5);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 3, 2);
        mul_binary_tile(4, 4, 4);
        mul_binary_tile(5, 5, 5);
        tile_regs_commit();
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_e, 1);
        cb_pop_front(cb_whe, 2);
        tile_regs_wait();
        pack_n(cb_xh, 1, 0);
        pack_n(cb_xe, 1, 2);
        pack_n(cb_sq, 2, 4);
        tile_regs_release();
        cb_wait_front(cb_sq, 2);
        tile_regs_acquire();
        matmul_start(cb_sq, cb_ones);
        matmul_tiles(cb_sq, cb_ones, 0, 0, 0);
        matmul_tiles(cb_sq, cb_ones, 1, 0, 1);
        cb_pop_front(cb_sq, 2);
        times_identity(2);
        finish(cb_d, 2);
        cb_wait_front(cb_d, 2);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_d);
        matmul_tiles(cb_ones, cb_d, 0, 0, 0);
        matmul_tiles(cb_ones, cb_d, 0, 1, 1);
        cb_pop_front(cb_d, 2);
        finish(cb_v, 2);
    }

    // ---- F2: X's tile n = r_m (x_h W_h) + q_m (x_e W_e), fp32 and bf16
    if (n != IDLE) {
        cb_wait_front(cb_id, 1);
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_x, SS + 6);
        row_scalars(SS, true, F_INV10240);
        row_scalars(SS + 3, false, F_INV2560);
        cb_wait_front(cb_rq, 2);
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        for (uint32_t k = 0; k < 2 * KD; k++) {
            if (k % CH == 0) {
                cb_wait_front(cb_w, k + CH);  // W arrives in chunks (the reader's)
            }
            matmul_tiles(cb_x, cb_w, k % KD, k, k < KD ? 0 : 1);
        }
        load(cb_rq, 0, 2);
        load(cb_rq, 1, 3);
        MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup
        SFPU_CALL(fold);
        tile_regs_commit();
        cb_pop_front(cb_w, 2 * KD);
        cb_pop_front(cb_rq, 2);
        cb_pop_front(cb_x, SS + 6);
        tile_regs_wait();
        pack_n(cb_o32, 1, 0);
        pack_n(cb_o16, 1, 0);
        tile_regs_release();
    }
}
