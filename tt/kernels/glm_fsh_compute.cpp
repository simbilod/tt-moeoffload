// GLM's fused FFN half, compute (the phases in glm_fsh_reader.cpp): R1, each unit's gate (Dest 0)
// and up (Dest 1) tiles, 128 matmuls each, then h = silu(min(g, 10)) * clamp(u, -10, 10), bf16 to cb 16; R2, each
// item's KD / S matmuls into Dest 0, fp32 to cb 17.
// Runtime args: R1's unit count, R2's item count. Compile args: batch, S, KD.
#include <cstdint>

#include "fused/cmp.h"

// clamp_gu: fused/cmp.h

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_h = 16, cb_out = 17, KA = 128;
constexpr uint32_t F_LIM = 0x41200000;   // 10

inline void gemv_ring(uint32_t kt, uint32_t dst, uint32_t batch) {
    for (uint32_t c = 0; c < kt; c += batch) {
        cb_wait_front(cb_w, batch);
        cb_wait_front(cb_x, batch);
        for (uint32_t j = 0; j < batch; j++) matmul_tiles(cb_x, cb_w, j, j, dst);
        cb_pop_front(cb_w, batch);
        cb_pop_front(cb_x, batch);
    }
}

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t KD = get_compile_time_arg_val(2);
    const uint32_t nu = get_arg_val<uint32_t>(0), n_it = get_arg_val<uint32_t>(1);
    if (!nu && !n_it) return;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    for (uint32_t u = 0; u < nu; u++) {
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        gemv_ring(KA, 0, batch);
        gemv_ring(KA, 1, batch);
        SFPU_CALL(clamp_gu, F_LIM);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        finish(cb_h, 1);
    }
    for (uint32_t it = 0; it < n_it; it++) {
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        gemv_ring(KD / S, 0, batch);
        finish(cb_out, 1);
    }
}
