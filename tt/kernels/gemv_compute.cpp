// G2 GEMV compute: per output tile, the kt matmuls x_k W_kn accumulated in Dest 0 (row 0 is y's 32 values), packed.
// cb 1 x (bf16, resident), cb 0 W (bfp8_b, BATCH tiles at a time), cb 16 y.
// Runtime args: first output tile, output tile count. Compile args: reps, kt, batch.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_out = 16;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    const uint32_t n_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_init(cb_x, cb_w);
    cb_wait_front(cb_x, kt);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t n = 0; n < n_count; n++) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < kt; c += batch) {
                cb_wait_front(cb_w, batch);
                for (uint32_t j = 0; j < batch; j++) {
                    matmul_tiles(cb_x, cb_w, c + j, j, 0);
                }
                cb_pop_front(cb_w, batch);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
    }
}
