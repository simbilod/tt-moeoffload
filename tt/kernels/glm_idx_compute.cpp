// GLM's decode indexer, compute (the data in glm_idx_reader.cpp): per pool tile, P = K_t iq^T over the
// 4 dim tiles (32 pools x 32 heads), relu, transposed (heads x pools), then w @ that: row 0 = the 32 pools' scores.
// Runtime args: the pool tile count (0: idle).
#include <cstdint>

#include "fused/cmp.h"
#include "api/compute/eltwise_unary/relu.h"

using namespace fused;

constexpr uint32_t cb_k = 0, cb_iq = 1, cb_w = 2, cb_p = 3, cb_pt = 4, cb_out = 16;

void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0);
    if (!nt) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_k, cb_iq, cb_out);
    cb_wait_front(cb_iq, 4);
    cb_wait_front(cb_w, 1);
    for (uint32_t t = 0; t < nt; t++) {
        cb_wait_front(cb_k, 4);
        tile_regs_acquire();
        matmul_start(cb_k, cb_iq);
        for (uint32_t d = 0; d < 4; d++) matmul_tiles(cb_k, cb_iq, d, d, 0);
        relu_tile_init();
        relu_tile(0);
        finish(cb_p, 1);
        cb_pop_front(cb_k, 4);
        cb_wait_front(cb_p, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_p);
        transpose_init(cb_p);
        transpose_tile(cb_p, 0, 0);
        finish(cb_pt, 1);
        cb_pop_front(cb_p, 1);
        cb_wait_front(cb_pt, 1);
        tile_regs_acquire();
        matmul_start(cb_w, cb_pt);
        matmul_tiles(cb_w, cb_pt, 0, 0, 0);
        finish(cb_out, 1);
        cb_pop_front(cb_pt, 1);
    }
    cb_pop_front(cb_iq, 4);
    cb_pop_front(cb_w, 1);
}
