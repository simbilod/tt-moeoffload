// Kept over ttnn.experimental.kda.qkv_causal_conv1d_silu (the ttnn dedup audit, 2026-11-09): with the history
// carried row-major it was not bit for bit (the last X up to 6% of its max apart at 4K-20K) and only ~0.5% faster
// over a whole prefill (20K tokens 12.96 -> 12.89-12.94 s); results/g7_prefill_regress_card_conv.json.
// Causal depthwise conv compute (G2 prefill): y[t] = silu(sum_k w_k * x[t - 3 + k]) per channel, a column tile at a time
// down the T axis. For row tile i: A_k = x_i * w_k (a row broadcast of tap k, k = 0..3) into cb 24, and A_0..A_2 again
// into cb 25 as the next tile's carries B. Then, accumulated in Dest by the FPU with 0/1 shift tiles (exact):
//   y_i = I A_3 + sum_{s=1..3} (L_s A_{3-s} + U_s B_{3-s})   (L_s: rows shift down by s; U_s: the previous tile's last
//                                                            s rows to the top)
// and SiLU on the SFPU. cb 25 holds two carry sets (the current one at the front while the next is pushed), so its
// index reads never wrap; the first set of a column comes from the conv state tile.
// cb 0 x, cb 1 taps, cb 2 state, cb 5 shifts (fp32), cb 24 A (fp32), cb 25 B (fp32), cb 16 y.
// Runtime args: first column tile, count. Compile args: reps, rt.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_st = 2, cb_sh = 5, cb_a = 24, cb_b = 25, cb_y = 16;

void scale(uint32_t cb_in, uint32_t n) {
    // Dest k = in * tap k, k < n
    reconfig_data_format(cb_in, cb_w);
    mul_bcast_rows_init(cb_in, cb_w);
    for (uint32_t k = 0; k < n; k++) {
        mul_tiles_bcast_rows(cb_in, cb_w, 0, k, k);
    }
}

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t rt = get_compile_time_arg_val(1);
    const uint32_t jn = get_arg_val<uint32_t>(1);
    if (jn == 0) {
        return;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sh, cb_a, cb_y);
    cb_wait_front(cb_sh, 7);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t j = 0; j < jn; j++) {
            cb_wait_front(cb_w, 4);
            cb_wait_front(cb_st, 1);
            tile_regs_acquire();
            scale(cb_st, 3);
            tile_regs_commit();
            cb_pop_front(cb_st, 1);
            cb_reserve_back(cb_b, 3);
            tile_regs_wait();
            pack_reconfig_data_format(cb_b);
            for (uint32_t k = 0; k < 3; k++) {
                pack_tile(k, cb_b);
            }
            tile_regs_release();
            cb_push_back(cb_b, 3);

            for (uint32_t i = 0; i < rt; i++) {
                cb_wait_front(cb_x, 1);
                tile_regs_acquire();
                scale(cb_x, 4);
                tile_regs_commit();
                cb_pop_front(cb_x, 1);
                cb_reserve_back(cb_a, 4);
                cb_reserve_back(cb_b, 3);
                tile_regs_wait();
                pack_reconfig_data_format(cb_a);
                for (uint32_t k = 0; k < 4; k++) {
                    pack_tile(k, cb_a);
                }
                for (uint32_t k = 0; k < 3; k++) {
                    pack_tile(k, cb_b);
                }
                tile_regs_release();
                cb_push_back(cb_a, 4);
                cb_push_back(cb_b, 3);

                cb_wait_front(cb_a, 4);
                cb_wait_front(cb_b, 6);
                tile_regs_acquire();
                reconfig_data_format<SrcOrder::Reverse>(cb_sh, cb_a);
                matmul_init(cb_sh, cb_a);
                matmul_tiles(cb_sh, cb_a, 0, 3, 0);
                for (uint32_t s = 1; s <= 3; s++) {
                    matmul_tiles(cb_sh, cb_a, s, 3 - s, 0);
                    matmul_tiles(cb_sh, cb_b, 3 + s, 3 - s, 0);
                }
                silu_tile_init();
                silu_tile(0);
                tile_regs_commit();
                cb_pop_front(cb_a, 4);
                cb_pop_front(cb_b, 3);
                cb_reserve_back(cb_y, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_y);
                pack_tile(0, cb_y);
                tile_regs_release();
                cb_push_back(cb_y, 1);
            }
            cb_pop_front(cb_b, 3);
            cb_pop_front(cb_w, 4);
        }
    }
}
