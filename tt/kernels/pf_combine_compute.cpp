// Prefill: the routed experts' sum per token (pf_combine_reader.cpp): per output tile, for each route j, y_j and
// w_j copied to Dest 1 and 2 (unpacked to Dest as fp32), then the SFPU accumulates Dest 0 += Dest 1 * Dest 2.
// SKIP: a zero weight's product is 0 whatever y holds (SplitExperts: the CPU's routes, rows unwritten).
// cb 0 y (K per item), cb 2 w (K per token tile), cb 16 out. Runtime args: first item, count. Compile args: nt, K.
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
inline void route_acc() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat w = sfpi::dst_reg[2 * 32];
        sfpi::vFloat v = sfpi::dst_reg[32] * w;
#ifdef SKIP
        v_if (w == 0.0f) { v = 0.0f; }   // a route another engine took: its y rows were never written
        v_endif;
#endif
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

constexpr uint32_t cb_y = 0, cb_w = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t nt = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    const uint32_t first = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    unary_op_init_common(cb_y, cb_out);
    uint32_t i_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / nt;
        if (i != i_prev) {
            if (i_prev != 0xffffffff) {
                cb_pop_front(cb_w, K);
            }
            cb_wait_front(cb_w, K);
            i_prev = i;
        }
        cb_wait_front(cb_y, K);
        tile_regs_acquire();
        for (uint32_t j = 0; j < K; j++) {
            copy_init(cb_y);
            copy_tile(cb_y, j, 1);
            copy_init(cb_w);
            copy_tile(cb_w, j, 2);
            if (j == 0) {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, route_acc<true>, 0,
                                                       VectorMode::RC)));
            } else {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, route_acc<false>, 0,
                                                       VectorMode::RC)));
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_y, K);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_w, K);
}
