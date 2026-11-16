// GLM's mHC site on the card, compute (glm_hc_reader.cpp has the phases): A, the mixes' partial: the
// HC * nt bf16 matmuls x_i fn_i into Dest 0, one fp32 tile out; B, PRE @ X_t per column tile (h); C, C' @ X'_t per
// column tile (the new streams). Runtime args: nt (0 off the working cores), phases (bit 0: A and B, bit 1: C).
#include <cstdint>

#include "fused/cmp.h"
#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_X = 4, cb_coef = 5, cb_out = 16, HC = 4;
constexpr uint32_t cb_ones = 10, cb_sqt = 11, cb_sq = 6;   // M rows: the rows' sums of squares here

void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0), phases = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    if (!nt) return;
    cb_wait_front(cb_X, 2);
    if (phases & 1) {
        fused::matmul_start(cb_x, cb_w);
        cb_wait_front(cb_w, HC * nt);
        cb_wait_front(cb_x, HC * nt);
        tile_regs_acquire();
        for (uint32_t i = 0; i < HC * nt; i++) matmul_tiles(cb_x, cb_w, i, i, 0);
        cb_pop_front(cb_w, HC * nt);
#if M_MAX > 1
        fused::finish(cb_out, 1);
        // the rows' sums of squares: each x tile squared (row m: row m's stream s), then their row sums by a matmul with
        // ones, accumulated over the tiles: row m of cb 6's tile = row m's partial sum (every column)
        for (uint32_t i = 0; i < HC * nt; i++) {
            tile_regs_acquire();
            fused::load(cb_x, i, 0);
            square_tile_init();
            square_tile(0);
            fused::finish(cb_sqt, 1);
        }
        cb_pop_front(cb_x, HC * nt);
        cb_wait_front(cb_ones, 1);
        cb_wait_front(cb_sqt, HC * nt);
        tile_regs_acquire();
        fused::matmul_start(cb_sqt, cb_ones);
        for (uint32_t i = 0; i < HC * nt; i++) matmul_tiles(cb_sqt, cb_ones, i, 0, 0);
        cb_pop_front(cb_sqt, HC * nt);
        cb_pop_front(cb_ones, 1);
        fused::finish(cb_sq, 1);
        fused::matmul_start(cb_x, cb_w);
#else
        cb_pop_front(cb_x, HC * nt);
        fused::finish(cb_out, 1);
#endif
    }
    fused::matmul_start(cb_coef, cb_X);
    for (uint32_t phase = 0; phase < 2; phase++) {        // B: PRE, C: C' (each its own coefficient tile)
        if (!(phases & (1u << phase))) continue;
        cb_wait_front(cb_coef, 1);
        for (uint32_t j = 0; j < nt; j++) {
            tile_regs_acquire();
            matmul_tiles(cb_coef, cb_X, 0, j, 0);
            fused::finish(cb_out, 1);
        }
        cb_pop_front(cb_coef, 1);
    }
    cb_pop_front(cb_X, 2);
}
