// The hyper-connection mix's two phases, compute side (fused/hc_dm.h has the phases and the data movement).
// H1: r_s = rsqrt(mean X_s^2 + eps) through two matmuls with the all-ones tile, then lo[s][n] = r_s (X_s W'_s).
// HL (tile t < 11): lo's tile t = sum_s P_s, then silu(lo / 4) as the gate GEMV's x, or for t = 10 w = 2 sigmoid(lo / 4).
// H2: hn = 1/4 sum_s sigmoid(gate_s) X_s norm_s r_s.
#pragma once

#include "fused/cmp.h"

namespace fused::hc {

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2;
constexpr uint32_t cb_hd = 3, cb_hsq = 4, cb_ht1 = 5, cb_hr = 6, cb_hout = 7, cb_hrout = 8, cb_hp = 9, cb_hx2 = 10;
constexpr uint32_t cb_hxs = 11, cb_hnrm = 12, cb_hrs = 13, cb_hhn = 14, cb_hchk = 15, cb_hlo = 14;
constexpr uint32_t D = 2560, KS = 80, KD = 80, KHU = 10, LO = 352, IDLE = 0xffffffff;

// H1 then H2 for this core's item and tile (fused/hc_dm.h has the phases)
inline void compute(uint32_t item, uint32_t tile, Weights& w) {
    if (item != IDLE) {
        cb_wait_front(cb_hd, 3);
        tile_regs_acquire();
        load_n(cb_hd, 3);
        square_tile_init();
        for (uint32_t t = 0; t < 3; t++) {
            square_tile(t);
        }
        cb_pop_front(cb_hd, 3);
        finish(cb_hsq, 3);
        cb_wait_front(cb_hsq, 3);
        tile_regs_acquire();
        matmul_start(cb_hsq, cb_ones);
        for (uint32_t t = 0; t < 3; t++) {
            matmul_tiles(cb_hsq, cb_ones, t, 0, 0);
        }
        cb_pop_front(cb_hsq, 3);
        finish(cb_ht1, 1);
        cb_wait_front(cb_ht1, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_ht1);
        matmul_tiles(cb_ones, cb_ht1, 0, 0, 0);
        SFPU_CALL(affine, F_INV2560, F_EPS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        cb_pop_front(cb_ht1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_hr, 1);
        if ((item >> 2) == 0) {
            pack_n(cb_hrout, 1);
        }
        tile_regs_release();

        cb_wait_front(cb_x, KD);
        cb_wait_front(cb_hr, 1);
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        w.gemv(cb_x, KS, 0);
        w.end();
        load(cb_hr, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        cb_pop_front(cb_x, KD);
        cb_pop_front(cb_hr, 1);
        finish(cb_hout, 1);
    }
    if (tile != IDLE && tile < LO / 32) {  // HL: lo's tile t from its 4 partials
        cb_wait_front(cb_hp, 4);
        tile_regs_acquire();
        load_n(cb_hp, 4);
        SFPU_CALL(sum4);
        SFPU_CALL(affine, F_QUARTER, F_ZERO);
        cb_pop_front(cb_hp, 4);
        if (tile == LO / 32 - 1) {  // the inject weights
            sigmoid_tile_init();
            sigmoid_tile(0);
            SFPU_CALL(affine, F_TWO, F_ZERO);
            finish(cb_hout, 1);
        } else {  // the gate GEMV's x
            silu_tile_init();
            silu_tile(0);
            finish(cb_hlo, 1);
        }
    }
    if (tile != IDLE && tile < D / 32) {  // H2
        cb_wait_front(cb_hx2, KHU);
        cb_wait_front(cb_hxs, 4);
        cb_wait_front(cb_hnrm, 4);
        cb_wait_front(cb_hrs, 4);
        tile_regs_acquire();
        matmul_start(cb_hx2, cb_w);
        for (uint32_t s = 0; s < 4; s++) {
            w.gemv(cb_hx2, KHU, s);
        }
        w.end();
        sigmoid_tile_init();
        for (uint32_t s = 0; s < 4; s++) {
            sigmoid_tile(s);
        }
        for (uint32_t s = 0; s < 4; s++) {
            load(cb_hxs, s, 4);
            load(cb_hnrm, s, 5);
            load(cb_hrs, s, 6);
            if (s == 0) {
                SFPU_CALL(mul3<0>);
            } else if (s == 1) {
                SFPU_CALL(mul3<1>);
            } else if (s == 2) {
                SFPU_CALL(mul3<2>);
            } else {
                SFPU_CALL(mul3<3>);
            }
        }
        SFPU_CALL(sum4);
        SFPU_CALL(affine, F_QUARTER, F_ZERO);
        cb_pop_front(cb_hx2, KHU);
        cb_pop_front(cb_hxs, 4);
        cb_pop_front(cb_hnrm, 4);
        cb_pop_front(cb_hrs, 4);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_hhn, 1);
        pack_n(cb_hchk, 1);
        tile_regs_release();
    }
}

// H1's r_s at mp rows (fused/hc_dm.h read_m): X_s's mp-row region as raw tiles, mp chunks of 3; tile row R of any of
// them holds only row R % mp's values, so the squares summed elementwise over the tiles (Q, fp32 in Dest), then
// Sel @ (Q @ ones) (cb_ones tile 1: Sel[i][R] = 1 when R % mp == i) gives row i's sum in row i. Into Dest 0, the tile
// regs left acquired.
inline void h1_rs_m(uint32_t mp) {
    tile_regs_acquire();
    for (uint32_t c = 0; c < mp; c++) {
        const uint32_t at = c ? 1 : 0;  // chunk 0 into Dest 0..2, the others into 1..3, summed into 0
        cb_wait_front(cb_hd, 3);
        load_n(cb_hd, 3, at);
        square_tile_init();
        for (uint32_t t = 0; t < 3; t++) {
            square_tile(at + t);
        }
        add_binary_tile_init();
        for (uint32_t t = 1; t < 4; t++) {
            if (t < at + 3) {
                add_binary_tile(0, t, 0);
            }
        }
        cb_pop_front(cb_hd, 3);
    }
    finish(cb_hsq, 1);
    cb_wait_front(cb_hsq, 1);
    tile_regs_acquire();
    matmul_start(cb_hsq, cb_ones);
    matmul_tiles(cb_hsq, cb_ones, 0, 0, 0);
    cb_pop_front(cb_hsq, 1);
    finish(cb_ht1, 1);
    cb_wait_front(cb_ht1, 1);
    tile_regs_acquire();
    matmul_start(cb_ones, cb_ht1);
    matmul_tiles(cb_ones, cb_ht1, 1, 0, 0);
    SFPU_CALL(affine, F_INV2560, F_EPS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    cb_pop_front(cb_ht1, 1);
}

// the same at mp rows (fused/hc_dm.h read_m, write_m), H1's r_s per row
inline void compute_m(uint32_t item, uint32_t tile, Weights& w, uint32_t mp) {
    if (item != IDLE) {
        h1_rs_m(mp);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_hr, 1);
        if ((item >> 2) == 0) {
            pack_n(cb_hrout, 1);
        }
        tile_regs_release();

        cb_wait_front(cb_x, KD);
        cb_wait_front(cb_hr, 1);
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        w.gemv(cb_x, KS, 0);
        w.end();
        load(cb_hr, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        cb_pop_front(cb_x, KD);
        cb_pop_front(cb_hr, 1);
        finish(cb_hout, 1);
    }
    if (tile != IDLE && tile < LO / 32) {  // HL: lo's tile t from its 4 partials
        cb_wait_front(cb_hp, 4);
        tile_regs_acquire();
        load_n(cb_hp, 4);
        SFPU_CALL(sum4);
        SFPU_CALL(affine, F_QUARTER, F_ZERO);
        cb_pop_front(cb_hp, 4);
        if (tile == LO / 32 - 1) {  // the inject weights
            sigmoid_tile_init();
            sigmoid_tile(0);
            SFPU_CALL(affine, F_TWO, F_ZERO);
            finish(cb_hout, 1);
        } else {  // the gate GEMV's x
            silu_tile_init();
            silu_tile(0);
            finish(cb_hlo, 1);
        }
    }
    if (tile != IDLE && tile < D / 32) {  // H2
        cb_wait_front(cb_hx2, KHU);
        cb_wait_front(cb_hxs, 4);
        cb_wait_front(cb_hnrm, 4);
        cb_wait_front(cb_hrs, 4);
        tile_regs_acquire();
        matmul_start(cb_hx2, cb_w);
        for (uint32_t s = 0; s < 4; s++) {
            w.gemv(cb_hx2, KHU, s);
        }
        w.end();
        sigmoid_tile_init();
        for (uint32_t s = 0; s < 4; s++) {
            sigmoid_tile(s);
        }
        for (uint32_t s = 0; s < 4; s++) {
            load(cb_hxs, s, 4);
            load(cb_hnrm, s, 5);
            load(cb_hrs, s, 6);
            if (s == 0) {
                SFPU_CALL(mul3<0>);
            } else if (s == 1) {
                SFPU_CALL(mul3<1>);
            } else if (s == 2) {
                SFPU_CALL(mul3<2>);
            } else {
                SFPU_CALL(mul3<3>);
            }
        }
        SFPU_CALL(sum4);
        SFPU_CALL(affine, F_QUARTER, F_ZERO);
        cb_pop_front(cb_hx2, KHU);
        cb_pop_front(cb_hxs, 4);
        cb_pop_front(cb_hnrm, 4);
        cb_pop_front(cb_hrs, 4);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_hhn, 1);
        pack_n(cb_hchk, 1);
        tile_regs_release();
    }
}

}  // namespace fused::hc
