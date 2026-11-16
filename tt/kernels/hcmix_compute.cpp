// Hyper-connection gate-sum compute (G2 prefill): mixed = sum over the 4 streams of r_c * norm_c * sigmoid(gate_c), per
// output tile, with the 1/4 of the stream mean folded into norm. Per stream: r, gate and norm copied to Dest 1..3, the
// sigmoid on Dest 2, then the SFPU accumulates Dest 0 += Dest 1 * Dest 2 * Dest 3 (fp32). Replaces a sigmoid, two
// multiplies and a sum over [4, T, 2560] that each read and wrote the whole tensor.
// cb 0 r, cb 1 gate, cb 2 norm, cb 16 mixed. Runtime args: first item, count. Compile args: rt.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool FIRST>
inline void hc_acc() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat v = sfpi::dst_reg[32] * sfpi::dst_reg[2 * 32] * sfpi::dst_reg[3 * 32];
        if constexpr (FIRST) {
            sfpi::dst_reg[0] = v;
        } else {
            sfpi::dst_reg[0] = sfpi::dst_reg[0] + v;
        }
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_r = 0, cb_g = 1, cb_n = 2, cb_y = 16;

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    const uint32_t first = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    unary_op_init_common(cb_r, cb_y);
    sigmoid_tile_init();
    uint32_t j_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t j = it / rt;
        if (j != j_prev) {
            if (j_prev != 0xffffffff) {
                cb_pop_front(cb_n, 4);
            }
            cb_wait_front(cb_n, 4);
            j_prev = j;
        }
        cb_wait_front(cb_r, 4);
        cb_wait_front(cb_g, 4);
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; c++) {
            copy_init(cb_r);
            copy_tile(cb_r, c, 1);
            copy_init(cb_g);
            copy_tile(cb_g, c, 2);
            copy_init(cb_n);
            copy_tile(cb_n, c, 3);
            sigmoid_tile(2);
            if (c == 0) {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hc_acc<true>, 0, VectorMode::RC)));
            } else {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hc_acc<false>, 0, VectorMode::RC)));
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_r, 4);
        cb_pop_front(cb_g, 4);
        cb_reserve_back(cb_y, 1);
        tile_regs_wait();
        pack_tile(0, cb_y);
        tile_regs_release();
        cb_push_back(cb_y, 1);
    }
    cb_pop_front(cb_n, 4);
}
