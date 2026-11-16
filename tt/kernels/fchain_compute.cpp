// F1 chain compute: per phase and output tile, the kt matmuls x_k W_kn accumulated in Dest 0 (row 0 is y's 32
// values), packed twice: fp32 (cb 16) and bf16 (cb 17, the next phase's x). Weights arrive in batches of BATCH
// tiles (cb 0), consumed as one flat stream across the phase's output tiles; a padded last batch is popped whole.
// Runtime args: per phase kt, output tile count. Compile args: reps, phases, batch, max kt.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_y32 = 16, cb_y16 = 17;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t phases = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr uint32_t max_kt = get_compile_time_arg_val(3);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_y32);
    matmul_init(cb_x, cb_w);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t p = 0; p < phases; p++) {
            const uint32_t kt = get_arg_val<uint32_t>(2 * p);
            const uint32_t n_count = get_arg_val<uint32_t>(2 * p + 1);
            if (n_count == 0) {
                continue;
            }
            cb_wait_front(cb_x, max_kt);
            uint32_t used = 0;
            for (uint32_t n = 0; n < n_count; n++) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < kt; k++) {
                    if (used == 0) {
                        cb_wait_front(cb_w, batch);
                    }
                    matmul_tiles(cb_x, cb_w, k, used, 0);
                    if (++used == batch) {
                        cb_pop_front(cb_w, batch);
                        used = 0;
                    }
                }
                tile_regs_commit();
                cb_reserve_back(cb_y32, 1);
                cb_reserve_back(cb_y16, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_y32);
                pack_tile(0, cb_y32);
                pack_reconfig_data_format(cb_y16);
                pack_tile(0, cb_y16);
                tile_regs_release();
                cb_push_back(cb_y32, 1);
                cb_push_back(cb_y16, 1);
            }
            if (used) {
                cb_pop_front(cb_w, batch);
            }
            cb_pop_front(cb_x, max_kt);
        }
    }
}
