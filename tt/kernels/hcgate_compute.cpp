// Hyper-connection gate (prefill): hn = sum_c xn_c norm_c sigmoid(a up_c) per output tile, the 4 streams' gate tiles
// computed in Dest (Dest c, KU tiles of a each) and never written out, then the gate-sum as hcmix_compute.cpp's
// (Dest 6 += (xn_c sigmoid(gate_c)) norm_c, xn and norm unpacked to Dest 4, 5 as fp32). Replaces the repeat of a, the
// [4, T, 2560] fp32 gate's matmul (written once, read once) and the gate-sum kernel.
// cb 0 a (bf16), cb 1 up (the group's), cb 2 xn, cb 3 norm (the group's), cb 16 hn (bf16).
// Runtime args: first item, count. Compile args: rt, NC.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Dest 6 (=|+=) (Dest 4 * Dest C) * Dest 5
template <int C>
inline void gate_acc() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[4 * 32] * sfpi::dst_reg[C * 32] * sfpi::dst_reg[5 * 32];
        if constexpr (C == 0) {
            sfpi::dst_reg[6 * 32] = v;
        } else {
            sfpi::dst_reg[6 * 32] = sfpi::dst_reg[6 * 32] + v;
        }
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_a = 0, cb_w = 1, cb_x = 2, cb_n = 3, cb_y = 16;
constexpr uint32_t KU = 10, IDLE = 0xffffffff;

#define SFPU_CALL(fn) MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, 0, VectorMode::RC)))

inline void load(uint32_t src, uint32_t itile, uint32_t idst) {
    reconfig_data_format_srca(src);
    copy_init(src);
    copy_tile(src, itile, idst);
}

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    const uint32_t first = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_y);
    pack_reconfig_data_format(cb_y);
    uint32_t g_prev = IDLE;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t g = it / rt;
        if (g != g_prev) {
            if (g_prev != IDLE) {
                cb_pop_front(cb_w, 4 * KU * NC);
                cb_pop_front(cb_n, 4 * NC);
            }
            cb_wait_front(cb_w, 4 * KU * NC);
            cb_wait_front(cb_n, 4 * NC);
            g_prev = g;
        }
        cb_wait_front(cb_a, KU);
        cb_wait_front(cb_x, 4 * NC);
        for (uint32_t j = 0; j < NC; j++) {
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_w);
            matmul_init(cb_a, cb_w);
            for (uint32_t c = 0; c < 4; c++) {
                for (uint32_t k = 0; k < KU; k++) {
                    matmul_tiles(cb_a, cb_w, k, (j * 4 + c) * KU + k, c);
                }
            }
            sigmoid_tile_init();
            for (uint32_t c = 0; c < 4; c++) {
                sigmoid_tile(c);
            }
            for (uint32_t c = 0; c < 4; c++) {
                load(cb_x, j * 4 + c, 4);
                load(cb_n, j * 4 + c, 5);
                if (c == 0) {
                    SFPU_CALL(gate_acc<0>);
                } else if (c == 1) {
                    SFPU_CALL(gate_acc<1>);
                } else if (c == 2) {
                    SFPU_CALL(gate_acc<2>);
                } else {
                    SFPU_CALL(gate_acc<3>);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_y, 1);
            tile_regs_wait();
            pack_tile(6, cb_y);
            tile_regs_release();
            cb_push_back(cb_y, 1);
        }
        cb_pop_front(cb_a, KU);
        cb_pop_front(cb_x, 4 * NC);
    }
    cb_pop_front(cb_w, 4 * KU * NC);
    cb_pop_front(cb_n, 4 * NC);
}
