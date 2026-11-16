// GLM mHC post compute (prefill): per item new_j = post_j * o + sum_s comb_sj * X_s, in the stock path's order
// (tt_mhc.hc_post: a multiply, then addcmul for s = 0..3), fp32 on the SFPU. Dest 0-3 the 4 accumulators, Dest 4 o then
// X_s in turn, Dest 5 the coefficient tile in turn. cb 0 X (4 tiles an item), cb 1 o, cb 2 w (20 tiles a row: post 0-3,
// comb 4 + 4 s + j), cb 16 the new streams. Runtime args: first item, count. Compile args: ct, pre.
// pre = 1 (the pre mix): h = pre_0 X_0, then += pre_s X_s (tt_mhc._mix's order) in Dest 0; w 4 tiles a row, no o.
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

// Dest tile J = Dest 4 * Dest 5, called with Dest tile J as the base
template <int J>
inline void hp_mul() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[(4 - J) * 32] * sfpi::dst_reg[(5 - J) * 32];
        sfpi::dst_reg++;
    }
}

// Dest tile J += Dest 4 * Dest 5
template <int J>
inline void hp_mac() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat p = sfpi::dst_reg[(4 - J) * 32] * sfpi::dst_reg[(5 - J) * 32];
        sfpi::dst_reg[0] = sfpi::dst_reg[0] + p;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_x = 0, cb_o = 1, cb_w = 2, cb_y = 16;

void kernel_main() {
    constexpr uint32_t ct = get_compile_time_arg_val(0);
    constexpr uint32_t pre = get_compile_time_arg_val(1);
    constexpr uint32_t NQ = pre ? 4 : 20, NO = pre ? 1 : 4;
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
                cb_pop_front(cb_w, NQ);
            }
            cb_wait_front(cb_w, NQ);
            i_prev = i;
        }
        cb_wait_front(cb_x, 4);
        tile_regs_acquire();
        if constexpr (pre) {
            copy_init(cb_x);
            copy_tile(cb_x, 0, 4);
            copy_init(cb_w);
            copy_tile(cb_w, 0, 5);
            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hp_mul<0>, 0, VectorMode::RC)));
            for (uint32_t c = 1; c < 4; c++) {
                copy_init(cb_x);
                copy_tile(cb_x, c, 4);
                copy_init(cb_w);
                copy_tile(cb_w, c, 5);
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hp_mac<0>, 0, VectorMode::RC)));
            }
        } else {
            cb_wait_front(cb_o, 1);
            copy_init(cb_o);
            copy_tile(cb_o, 0, 4);
            copy_init(cb_w);
#define MUL(J)                                                                                                      \
    copy_tile(cb_w, J, 5);                                                                                          \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hp_mul<J>, J, VectorMode::RC)));
            MUL(0)
            MUL(1)
            MUL(2)
            MUL(3)
#undef MUL
#define MAC(S, J)                                                                                                   \
    copy_tile(cb_w, 4 + 4 * S + J, 5);                                                                              \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, hp_mac<J>, J, VectorMode::RC)));
#define STREAM(S)                                                                                                   \
    copy_init(cb_x);                                                                                                \
    copy_tile(cb_x, S, 4);                                                                                          \
    copy_init(cb_w);                                                                                                \
    MAC(S, 0) MAC(S, 1) MAC(S, 2) MAC(S, 3)
            STREAM(0)
            STREAM(1)
            STREAM(2)
            STREAM(3)
#undef STREAM
#undef MAC
            cb_pop_front(cb_o, 1);
        }
        tile_regs_commit();
        cb_reserve_back(cb_y, NO);
        tile_regs_wait();
        for (uint32_t c = 0; c < NO; c++) {
            pack_tile(c, cb_y);
        }
        tile_regs_release();
        cb_push_back(cb_y, NO);
        cb_pop_front(cb_x, 4);
    }
    cb_pop_front(cb_w, NQ);
}
