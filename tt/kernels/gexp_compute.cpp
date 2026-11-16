// Grouped expert compute (G2 prefill): per expert and output tile n, the kt x MT matmuls x_e[m, k] W_e[k, n]
// accumulated in Dest tiles 0..MT-1 (fp32), then packed. cb 1 x (bf16, one expert at a time), cb 0 W (BATCH tiles at a
// time), cb 16 y. Runtime args: first expert, expert count. Compile args: reps, kt, nt, mt, mt_max, batch.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_out = 16;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t mt_max = get_compile_time_arg_val(4);
    constexpr uint32_t batch = get_compile_time_arg_val(5);
    const uint32_t e_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_init(cb_x, cb_w);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t e = 0; e < e_count; e++) {
            cb_wait_front(cb_x, mt_max * kt);
            for (uint32_t n = 0; n < nt; n++) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < kt; k += batch) {
                    cb_wait_front(cb_w, batch);
                    for (uint32_t j = 0; j < batch; j++) {
                        for (uint32_t m = 0; m < mt; m++) {
                            matmul_tiles(cb_x, cb_w, m * kt + k + j, j, m);
                        }
                    }
                    cb_pop_front(cb_w, batch);
                }
                tile_regs_commit();
                cb_reserve_back(cb_out, mt);
                tile_regs_wait();
                for (uint32_t m = 0; m < mt; m++) {
                    pack_tile(m, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, mt);
            }
            cb_pop_front(cb_x, mt_max * kt);
        }
    }
}
