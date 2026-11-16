// Decode QSA combine compute, one core per (KV head, dims tile d): with the partials' m and l as the rows of one
// statistics tile each (heads along the columns),
//   M     = the column max of the m tile                         (reduce)
//   A     = exp(m - M)                                           (a row broadcast, the SFPU's exp)
//   1 / L = 1 / the column sum of A * l                          (reduce, reciprocal)
//   W     = A / L                                                (a row broadcast)
//   D_i   = diag(row i of W) = the identity times W's row i      (bcast_row_idx = i)
//   O_d   = sum_i D_i O_i,d                                      (matmuls accumulating in Dest)
// so the softmax statistics cost a few tile operations, not one per partial.
// cb 0 O_i,d (np), cb 1 m tile, cb 2 l tile, cb 3 identity, cb 5 scaler, cb 24 M, cb 25 A, cb 26 A * l, cb 27 1 / L,
// cb 28 W, cb 29 D (np), cb 16 out. Runtime args: kv head (0xffffffff = idle). Compile args: partials per head.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_o = 0, cb_m = 1, cb_l = 2, cb_eye = 3, cb_sc = 5, cb_mx = 24, cb_a = 25, cb_al = 26,
                   cb_rl = 27, cb_w = 28, cb_d = 29, cb_out = 16;

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
    constexpr uint32_t np = get_compile_time_arg_val(0);
    if (get_arg_val<uint32_t>(0) == 0xffffffff) {
        return;
    }
    compute_kernel_hw_startup(cb_m, cb_sc, cb_mx);
    cb_wait_front(cb_m, 1);
    cb_wait_front(cb_l, 1);
    cb_wait_front(cb_sc, 1);
    cb_wait_front(cb_eye, 1);

    tile_regs_acquire();
    reconfig_data_format(cb_m, cb_sc);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_m, cb_sc, cb_mx);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_m, cb_sc, 0, 0, 0);
    reduce_uninit();
    tile_regs_commit();
    pack_to(cb_mx, 1);
    cb_wait_front(cb_mx, 1);

    tile_regs_acquire();
    reconfig_data_format(cb_m, cb_mx);
    sub_bcast_rows_init(cb_m, cb_mx);
    sub_tiles_bcast_rows(cb_m, cb_mx, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();
    pack_to(cb_a, 1);
    cb_wait_front(cb_a, 1);

    tile_regs_acquire();
    reconfig_data_format(cb_a, cb_l);
    mul_init(cb_a, cb_l);
    mul_tiles(cb_a, cb_l, 0, 0, 0);
    tile_regs_commit();
    pack_to(cb_al, 1);
    cb_wait_front(cb_al, 1);

    tile_regs_acquire();
    reconfig_data_format(cb_al, cb_sc);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_al, cb_sc, cb_rl);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_al, cb_sc, 0, 0, 0);
    reduce_uninit();
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    pack_to(cb_rl, 1);
    cb_wait_front(cb_rl, 1);

    tile_regs_acquire();
    reconfig_data_format(cb_a, cb_rl);
    mul_bcast_rows_init(cb_a, cb_rl);
    mul_tiles_bcast_rows(cb_a, cb_rl, 0, 0, 0);
    tile_regs_commit();
    pack_to(cb_w, 1);
    cb_wait_front(cb_w, 1);

    reconfig_data_format(cb_eye, cb_w);
    mul_bcast_rows_init(cb_eye, cb_w);
    for (uint32_t i = 0; i < np; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_eye, cb_w, 0, 0, 0, i);
        tile_regs_commit();
        pack_to(cb_d, 1);
    }
    cb_wait_front(cb_d, np);
    cb_wait_front(cb_o, np);

    tile_regs_acquire();
    reconfig_data_format<SrcOrder::Reverse>(cb_d, cb_o);
    matmul_init(cb_d, cb_o);
    for (uint32_t i = 0; i < np; i++) {
        matmul_tiles(cb_d, cb_o, i, i, 0);
    }
    tile_regs_commit();
    pack_to(cb_out, 1);
}
