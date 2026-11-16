// GLM's fused MLA output, compute (the phases in glm_fmo_reader.cpp): E1, a head's 8 output tiles
// of v_b (16 k-tiles each) to cb 16 as bf16 (E2's x); E2, each item's KO / S matmuls into Dest 0, the partial row to
// cb 17 (fp32). x from the reader's ring, a batch beside each weight batch.
// Runtime args: head (0xffffffff: none), E2's item count. Compile args: batch, S.
#include <cstdint>

#include "fused/cmp.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ob = 16, cb_out = 17, K1 = 16, KO = 512, IDLE = 0xffffffff;

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
    constexpr uint32_t S = get_compile_time_arg_val(1);
    const uint32_t head = get_arg_val<uint32_t>(0), n_it = get_arg_val<uint32_t>(1);
    if (head == IDLE && !n_it) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    if (head != IDLE) {
        for (uint32_t n = 0; n < 8; n++) {
            tile_regs_acquire();
            fused::matmul_start(cb_x, cb_w);
            gemv_ring(K1, batch);
            fused::finish(cb_ob, 1);
        }
    }
    for (uint32_t it = 0; it < n_it; it++) {
        tile_regs_acquire();
        fused::matmul_start(cb_x, cb_w);
        gemv_ring(KO / S, batch);
        fused::finish(cb_out, 1);
    }
}
