// Hyper-connection combine compute (prefill): per item the 4 streams' X_c + out * w_c (fp32, the SFPU: X_c in Dest c,
// out in 4, w_c in 5 in turn) into cb 16, one Dest pass an item. Replaces a broadcast multiply and an add that each read and
// wrote the 4 streams. cb 0 X (4 tiles an item), cb 1 out, cb 2 w (4 tiles a row), cb 16 the new X.
// Runtime args: first item, count. Compile args: ct.
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

// Dest tile C += Dest 4 (out) * Dest 5 (w_C), called with Dest tile C as the base
template <int C>
inline void hc_comb() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat p = sfpi::dst_reg[(4 - C) * 32] * sfpi::dst_reg[(5 - C) * 32];
        sfpi::dst_reg[0] = sfpi::dst_reg[0] + p;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_o = 1, cb_w = 2, cb_y = 16;

void kernel_main() {
    constexpr uint32_t ct = get_compile_time_arg_val(0);
    const uint32_t first = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    unary_op_init_common(cb_x, cb_y);
    uint32_t i_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / ct;
        if (i != i_prev) {
            if (i_prev != 0xffffffff) {
                cb_pop_front(cb_w, 4);
            }
            cb_wait_front(cb_w, 4);
            i_prev = i;
        }
        cb_wait_front(cb_o, 1);
        cb_wait_front(cb_x, 4);
        tile_regs_acquire();
        copy_init(cb_x);
        for (uint32_t c = 0; c < 4; c++) {
            copy_tile(cb_x, c, c);
        }
        copy_init(cb_o);
        copy_tile(cb_o, 0, 4);
        copy_init(cb_w);
#define COMB(C)                                                                                                    \
    copy_tile(cb_w, C, 5);                                                                                         \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hc_comb<C>, C, VectorMode::RC)));
        COMB(0)
        COMB(1)
        COMB(2)
        COMB(3)
#undef COMB
        tile_regs_commit();
        cb_reserve_back(cb_y, 4);
        tile_regs_wait();
        for (uint32_t c = 0; c < 4; c++) {
            pack_tile(c, cb_y);
        }
        tile_regs_release();
        cb_push_back(cb_y, 4);
        cb_pop_front(cb_o, 1);
        cb_pop_front(cb_x, 4);
    }
    cb_pop_front(cb_w, 4);
}
