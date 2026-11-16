// Hyper-connection low-rank outputs (prefill): lo = the sum of the 4 streams' down|inject partials, (p0 + p1) +
// (p2 + p3) as sum4; tiles 0..9 a = bf16 silu(lo / 4) (the up projection's input, cb 16), tile 10 the inject weights
// 2 sigmoid(lo / 4) (fp32, cb 17). Replaces sum4's slices and adds, the silu's and the inject weights' ops.
// cb 0 the partials. Runtime args: first item, count.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// Dest 0 = ((Dest 0 + Dest 1) + (Dest 2 + Dest 3)) * 0.25
inline void sum4q() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[32], c = sfpi::dst_reg[64], d = sfpi::dst_reg[96];
        sfpi::dst_reg[0] = ((a + b) + (c + d)) * 0.25f;
        sfpi::dst_reg++;
    }
}

// Dest 0 *= 2
inline void twice() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * 2.0f;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_p = 0, cb_a = 16, cb_w = 17, KO = 11;

#define SFPU_CALL(fn) MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, 0, VectorMode::RC)))

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    unary_op_init_common(cb_p, cb_a);
    for (uint32_t it = first; it < first + count; it++) {
        const bool inject = it % KO == KO - 1;
        cb_wait_front(cb_p, 4);
        tile_regs_acquire();
        copy_init(cb_p);
        for (uint32_t c = 0; c < 4; c++) {
            copy_tile(cb_p, c, c);
        }
        SFPU_CALL(sum4q);
        if (inject) {
            sigmoid_tile_init();
            sigmoid_tile(0);
            SFPU_CALL(twice);
        } else {
            silu_tile_init();
            silu_tile(0);
        }
        tile_regs_commit();
        cb_pop_front(cb_p, 4);
        const uint32_t out = inject ? cb_w : cb_a;
        cb_reserve_back(out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(out);
        pack_tile(0, out);
        tile_regs_release();
        cb_push_back(out, 1);
    }
}
