// GLM's fused MLA query side, compute (the phases in glm_fmq_reader.cpp):
//   A       each tile's 128 matmuls into Dest 0, fp32 to cb 16
//   B       each tile's 48 matmuls, then times r (cb 2, across row 0; loaded after the matmuls: an fp32 load before
//           them in one acquire doubled their sum, glm_fkda_compute.cpp), bf16 to cb 17
//   C       the absorb: the head's 16 output tiles, 8 matmuls each, bf16 to cb 17; the scores: iq's 4 tiles
//           transposed (cb 4 -> cb 8), then glm_idx_compute.cpp's loop (cb 3 keys, cb 5 w) to cb 16
// Runtime args: A's tile count, B's, head (0xffffffff: none), the pool tile count. Compile args: batch.
#include <cstdint>

#include "fused/cmp.h"
#include "api/compute/eltwise_unary/relu.h"
#ifndef M_MAX
#define M_MAX 1   // > 1: the M-row build (glm_fmq_reader.cpp): A, B and the absorb on whole tiles, the scores per row
#endif

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_r = 2, cb_k = 3, cb_iq = 4, cb_wt = 5, cb_p = 6, cb_pt = 7, cb_iqt = 8;
constexpr uint32_t cb_out = 16, cb_ob = 17, KA = 128, KB = 48, KH = 8, IDLE = 0xffffffff;

inline void gemv_ring(uint32_t kt, uint32_t batch) {
    for (uint32_t c = 0; c < kt; c += batch) {
        cb_wait_front(cb_w, batch);
        cb_wait_front(cb_x, batch);
        for (uint32_t j = 0; j < batch; j++) matmul_tiles(cb_x, cb_w, j, j, 0);
        cb_pop_front(cb_w, batch);
        cb_pop_front(cb_x, batch);
    }
}

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t na = get_arg_val<uint32_t>(0), nb = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2), np = get_arg_val<uint32_t>(3);
    if (!na && !nb && head == IDLE && !np) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);

    for (uint32_t t = 0; t < na; t++) {
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        gemv_ring(KA, batch);
        finish(cb_out, 1);
    }

    if (nb) {
        cb_wait_front(cb_r, 1);
        for (uint32_t t = 0; t < nb; t++) {
            tile_regs_acquire();
            matmul_start(cb_x, cb_w);
            gemv_ring(KB, batch);
            load(cb_r, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            finish(cb_ob, 1);
        }
        cb_pop_front(cb_r, 1);
    }

    if (head != IDLE) {
        for (uint32_t n = 0; n < 16; n++) {
            tile_regs_acquire();
            matmul_start(cb_x, cb_w);
            gemv_ring(KH, batch);
            finish(cb_ob, 1);
        }
    }

    if (np) {
        cb_wait_front(cb_iq, 4 * M_MAX);
        for (uint32_t m = 0; m < M_MAX; m++) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_iq);
            transpose_init(cb_iq);
            for (uint32_t d = 0; d < 4; d++) transpose_tile(cb_iq, 4 * m + d, d);
            finish(cb_iqt, 4);
        }
        cb_pop_front(cb_iq, 4 * M_MAX);
        cb_wait_front(cb_iqt, 4 * M_MAX);
        cb_wait_front(cb_wt, M_MAX);
        for (uint32_t t = 0; t < np; t++) {
            cb_wait_front(cb_k, 4);
            for (uint32_t m = 0; m < M_MAX; m++) {   // row m's scores of the pool tile
            tile_regs_acquire();
            matmul_start(cb_k, cb_iqt);
            for (uint32_t d = 0; d < 4; d++) matmul_tiles(cb_k, cb_iqt, d, 4 * m + d, 0);
            relu_tile_init();
            relu_tile(0);
            finish(cb_p, 1);
            cb_wait_front(cb_p, 1);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_p);
            transpose_init(cb_p);
            transpose_tile(cb_p, 0, 0);
            finish(cb_pt, 1);
            cb_pop_front(cb_p, 1);
            cb_wait_front(cb_pt, 1);
            tile_regs_acquire();
            matmul_start(cb_wt, cb_pt);
            matmul_tiles(cb_wt, cb_pt, m, 0, 0);
            finish(cb_out, 1);
            cb_pop_front(cb_pt, 1);
            }
            cb_pop_front(cb_k, 4);
        }
        cb_pop_front(cb_iqt, 4 * M_MAX);
        cb_pop_front(cb_wt, M_MAX);
    }
}
