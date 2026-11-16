// Decode QSA partial compute: one core's NB blocks of 32 selected cells of one KV head, for its 12 query heads (padded
// to 32 columns). Per block b: tilize K and V (cells x dims); S^T_b = mask_b + sum_d K_d q^T_d (cells x heads; the mask
// copied into Dest first, the matmuls accumulate on it). Then m = the column max over all blocks, P^T_b = exp(S^T_b -
// m) (a row broadcast), l = the column sum of P^T, P_b = transpose(P^T_b) and O_d = sum_b P_b V_b,d (heads x dims).
// Out: O (8 tiles, fp32), m and l (row tiles: row 0, a value per head) for the combine.
// cb 0 q^T, cb 2 / 3 K / V row-major blocks, cb 4 masks, cb 5 scaler, cb 24 / 25 K / V tiles (all blocks), cb 26 S^T
// (fp32), cb 27 m, cb 28 P^T, cb 29 l, cb 30 P, cb 16 out (O, m, l). Runtime args: kv head (0xffffffff = idle).
// Compile args: nb.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_qt = 0, cb_krm = 2, cb_vrm = 3, cb_mask = 4, cb_sc = 5, cb_k = 24, cb_v = 25, cb_s = 26,
                   cb_m = 27, cb_pt = 28, cb_l = 29, cb_p = 30, cb_out = 16;

void pack_to(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; i++) {
        pack_tile(i, cb);
    }
    tile_regs_release();
    cb_push_back(cb, n);
}

void kernel_main() {
    constexpr uint32_t nb = get_compile_time_arg_val(0);
    if (get_arg_val<uint32_t>(0) == 0xffffffff) {
        return;
    }
    compute_kernel_hw_startup(cb_krm, cb_k);     // the tilize's order; the matmuls reconfigure to Reverse
    cb_wait_front(cb_qt, 8);
    cb_wait_front(cb_mask, nb);
    cb_wait_front(cb_sc, 1);

    for (uint32_t b = 0; b < nb; b++) {
        cb_wait_front(cb_krm, 8);
        // fast tilize, with both sources reconfigured: after a matmul, a plain tilize read bf16 rows as fp32
        cb_reserve_back(cb_k, 8);
        fast_tilize_init_with_dt(cb_krm, 8, cb_k);
        fast_tilize_block(cb_krm, 8, cb_k);
        fast_tilize_uninit(cb_krm, cb_k, 8);
        cb_push_back(cb_k, 8);
        cb_pop_front(cb_krm, 8);
        cb_wait_front(cb_vrm, 8);
        cb_reserve_back(cb_v, 8);
        fast_tilize_init_with_dt(cb_vrm, 8, cb_v);
        fast_tilize_block(cb_vrm, 8, cb_v);
        fast_tilize_uninit(cb_vrm, cb_v, 8);
        cb_push_back(cb_v, 8);
        cb_pop_front(cb_vrm, 8);

        cb_wait_front(cb_k, (b + 1) * 8);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_mask);
        copy_init(cb_mask);
        copy_tile(cb_mask, b, 0);
        reconfig_data_format<SrcOrder::Reverse>(cb_k, cb_qt);
        matmul_init(cb_k, cb_qt);
        for (uint32_t d = 0; d < 8; d++) {
            matmul_tiles(cb_k, cb_qt, b * 8 + d, d, 0);
        }
        tile_regs_commit();
        pack_to(cb_s, 1);
    }
    cb_wait_front(cb_s, nb);

    // m: the column max over the blocks
    tile_regs_acquire();
    reconfig_data_format(cb_s, cb_sc);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_s, cb_sc, cb_m);
    for (uint32_t b = 0; b < nb; b++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_s, cb_sc, b, 0, 0);
    }
    reduce_uninit();
    tile_regs_commit();
    pack_to(cb_m, 1);
    cb_wait_front(cb_m, 1);

    // P^T = exp(S^T - m)
    for (uint32_t b = 0; b < nb; b++) {
        tile_regs_acquire();
        reconfig_data_format(cb_s, cb_m);
        sub_bcast_rows_init(cb_s, cb_m);
        sub_tiles_bcast_rows(cb_s, cb_m, b, 0, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        pack_to(cb_pt, 1);
    }
    cb_wait_front(cb_pt, nb);

    // l: the column sum of P^T
    tile_regs_acquire();
    reconfig_data_format(cb_pt, cb_sc);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_pt, cb_sc, cb_l);
    for (uint32_t b = 0; b < nb; b++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_pt, cb_sc, b, 0, 0);
    }
    reduce_uninit();
    tile_regs_commit();
    pack_to(cb_l, 1);

    // P = transpose(P^T): heads x cells
    reconfig_data_format_srca(cb_pt);
    transpose_init(cb_pt);
    for (uint32_t b = 0; b < nb; b++) {
        tile_regs_acquire();
        transpose_tile(cb_pt, b, 0);
        tile_regs_commit();
        pack_to(cb_p, 1);
    }
    cb_wait_front(cb_p, nb);
    cb_wait_front(cb_v, nb * 8);

    // O_d = sum_b P_b V_b,d, in two halves of 4 tiles (Dest half-sync: fast tilize needs it)
    reconfig_data_format<SrcOrder::Reverse>(cb_p, cb_v);
    matmul_init(cb_p, cb_v);
    for (uint32_t half = 0; half < 2; half++) {
        tile_regs_acquire();
        for (uint32_t d = 4 * half; d < 4 * half + 4; d++) {
            for (uint32_t b = 0; b < nb; b++) {
                matmul_tiles(cb_p, cb_v, b, b * 8 + d, d - 4 * half);
            }
        }
        tile_regs_commit();
        pack_to(cb_out, 4);
    }

    // m and l out after O
    cb_wait_front(cb_l, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_m);
    copy_init(cb_m);
    copy_tile(cb_m, 0, 0);
    copy_init(cb_l);
    copy_tile(cb_l, 0, 1);
    tile_regs_commit();
    pack_to(cb_out, 2);
}
