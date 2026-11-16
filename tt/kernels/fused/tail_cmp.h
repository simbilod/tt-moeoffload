// A layer's tail, compute side (the phases are described in fused/tail_dm.h).
#pragma once

#include "fused/cmp.h"

namespace fused::tail {

constexpr uint32_t cb_w = 0, cb_x = 1;
constexpr uint32_t cb_cx = 40, cb_cy = 41, cb_cxs = 42, cb_cw = 43, cb_cout32 = 44, cb_cout16 = 45;
constexpr uint32_t cb_rw = 46, cb_rlog = 47, cb_rh16 = 49, cb_rh32 = 50;
constexpr uint32_t cb_sgl = 51, cb_srt = 52, cb_sxs = 53, cb_sw = 54, cb_sy = 55, cb_sout32 = 56, cb_sout16 = 57;
constexpr uint32_t KD = 80, KC = 192, KR2 = 20, N_ROUTER = 17, IDLE = 0xffffffff;

// X_s += w_s y for the 4 streams, X in cb xs (4), w across row 0 in cb wcb (4), y in cb ycb: fp32 and bf16 out
inline void combine(uint32_t xs, uint32_t wcb, uint32_t ycb, uint32_t out32, uint32_t out16) {
    cb_wait_front(xs, 4);
    cb_wait_front(wcb, 4);
    cb_wait_front(ycb, 1);
    for (uint32_t half = 0; half < 2; half++) {
        tile_regs_acquire();
        load(xs, 2 * half, 0);
        copy_tile(xs, 2 * half + 1, 1);
        load(wcb, 2 * half, 4);
        copy_tile(wcb, 2 * half + 1, 5);
        load(ycb, 0, 6);
        SFPU_CALL(combine2<0>);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(out32, 2);
        pack_n(out16, 2);
        tile_regs_release();
    }
    cb_pop_front(xs, 4);
    cb_pop_front(wcb, 4);
    cb_pop_front(ycb, 1);
}

// oproj: x streamed with the weights (one output tile), then the combine
inline void oproj(uint32_t tile, uint32_t batch) {
    if (tile == IDLE) {
        return;
    }
    tile_regs_acquire();
    matmul_start(cb_cx, cb_w);
    for (uint32_t k = 0; k < KC; k += batch) {
        cb_wait_front(cb_cx, batch);
        cb_wait_front(cb_w, batch);
        for (uint32_t j = 0; j < batch; j++) {
            matmul_tiles(cb_cx, cb_w, j, j, 0);
        }
        cb_pop_front(cb_cx, batch);
        cb_pop_front(cb_w, batch);
    }
    finish(cb_cy, 1);
    combine(cb_cxs, cb_cw, cb_cy, cb_cout32, cb_cout16);
}

// R1
inline void ffn_r1(uint32_t r_item, Weights& w, Weights& rw) {
    if (r_item != IDLE) {
        cb_wait_front(cb_x, KD);
        tile_regs_acquire();
        if (r_item < N_ROUTER) {
            matmul_start(cb_x, cb_rw);
            rw.gemv(cb_x, KD, 0);
            rw.end();
            cb_pop_front(cb_x, KD);
            finish(cb_rlog, 1);
        } else {  // h = silu(g) * u
            matmul_start(cb_x, cb_w);
            w.gemv(cb_x, KD, 0);
            w.gemv(cb_x, KD, 1);
            w.end();
            cb_pop_front(cb_x, KD);
            silu_tile_init();
            silu_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_n(cb_rh16, 1);
            pack_n(cb_rh32, 1);
            tile_regs_release();
        }
    }
}

// R2: sh = down(h) * sigmoid(gate logit) + routed (+ the card's n partial rows of this tile from cb 16, padded to its
// batch of 7: fused/dexp_dm.h), then the combine
inline void ffn_r2(uint32_t s_tile, Weights& w, uint32_t n_card = 0) {
    constexpr uint32_t cb_part = 16;
    if (s_tile != IDLE) {
        cb_wait_front(cb_x, KD);
        cb_wait_front(cb_sgl, 1);
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        w.gemv(cb_x, KR2, 0);
        w.end();
        cb_pop_front(cb_x, KD);
        load(cb_sgl, 0, 1);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        cb_wait_front(cb_srt, 1);  // the routed sum last: the GEMV can run while the host computes it
        load(cb_srt, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        for (uint32_t j = 0; j < n_card; j++) {
            cb_wait_front(cb_part, 1);
            load(cb_part, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(cb_part, 1);
        }
        cb_pop_front(cb_sgl, 1);
        cb_pop_front(cb_srt, 1);
        finish(cb_sy, 1);
        combine(cb_sxs, cb_sw, cb_sy, cb_sout32, cb_sout16);
        if (n_card % 7) {
            cb_wait_front(cb_part, 7 - n_card % 7);
            cb_pop_front(cb_part, 7 - n_card % 7);
        }
    }
}

#if defined(M_MAX) && M_MAX > 1
// R2 with the card's experts at M rows: their n partial row tiles before the routed sum (n may pass cb 16's 14 pages,
// and the reader pushes the routed sums after the partial rows), else ffn_r2's
inline void ffn_r2_m(uint32_t s_tile, Weights& w, uint32_t n_card) {
    constexpr uint32_t cb_part = 16;
    if (s_tile != IDLE) {
        cb_wait_front(cb_x, KD);
        cb_wait_front(cb_sgl, 1);
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        w.gemv(cb_x, KR2, 0);
        w.end();
        cb_pop_front(cb_x, KD);
        load(cb_sgl, 0, 1);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        for (uint32_t j = 0; j < n_card; j++) {
            cb_wait_front(cb_part, 1);
            load(cb_part, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(cb_part, 1);
        }
        cb_wait_front(cb_srt, 1);
        load(cb_srt, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        cb_pop_front(cb_sgl, 1);
        cb_pop_front(cb_srt, 1);
        finish(cb_sy, 1);
        combine(cb_sxs, cb_sw, cb_sy, cb_sout32, cb_sout16);
        if (n_card % 7) {
            cb_wait_front(cb_part, 7 - n_card % 7);
            cb_pop_front(cb_part, 7 - n_card % 7);
        }
    }
}
#endif

// R1 then R2
inline void ffn(uint32_t r_item, uint32_t s_tile, Weights& w, Weights& rw) {
    ffn_r1(r_item, w, rw);
    ffn_r2(s_tile, w);
}

}  // namespace fused::tail
