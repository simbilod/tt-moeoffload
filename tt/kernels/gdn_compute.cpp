// Split delta rule compute (G2 follow-up): one token of the gated delta rule on one (head, 64-column slice) of the
// state S [128 k x 64 v], fp32 tiles at 2j + c (k-tile j, column tile c). The columns update independently, so the
// 48 heads x 2 slices run on 96 cores with no exchange. Per token, with d = exp(g):
//   delta = beta v - beta d k^T S          FPU: Dest starts at beta v, accumulates k'^T S (k' = -beta d k)
//   S     = d S + k delta^T                FPU for the outer product P; the SFPU applies d S + P in fp32
//   o     = q^T S                          FPU (q carries the 1/sqrt(128) scale)
// The state is kept twice: an exact copy, unpacked straight to Dest in fp32 for the SFPU update (so the state is never
// rounded to the FPU's 19-bit operands), and the FPU's operand copy. A buffer unpacked to Dest cannot also feed the
// matmul (its results were garbage), and each buffer has one producer (a second one desynchronizes its counters and
// hangs): the reader fills cb 23 / 22 once, the packer ping-pongs cb 24 / 25 (exact) and cb 27 / 28 (operand).
// cb 0 k' | k columns | q (12 tiles), cb 2 beta v, cb 26 delta, cb 16 o, cb 17 the final state.
// Runtime args: item (0xffffffff = idle), d as fp32 bits. Compile args: reps.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// Dest tiles 0, 1 = S, 2, 3 = P -> tiles 0, 1 = d S + P (one face per call)
inline void state_update(uint32_t d_bits) {
    const sfpi::vFloat d = Converter::as_float(d_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * d + sfpi::dst_reg[2 * 32];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * d + sfpi::dst_reg[3 * 32];
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_in = 0, cb_bv = 2, cb_o = 16, cb_sout = 17, cb_delta = 26;
constexpr uint32_t cb_m0 = 22, cb_x0 = 23, cb_xa = 24, cb_xb = 25, cb_ma = 27, cb_mb = 28;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    if (get_arg_val<uint32_t>(0) == 0xffffffff) {
        return;
    }
    const uint32_t d_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in, cb_m0, cb_o);
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup; no tanh state is used

    uint32_t xs = cb_x0, ms = cb_m0;  // this token's state: exact, operand
    for (uint32_t rep = 0; rep < reps; rep++) {
        const uint32_t xd = rep % 2 ? cb_xb : cb_xa, md = rep % 2 ? cb_mb : cb_ma;
        cb_wait_front(cb_in, 12);
        cb_wait_front(cb_bv, 2);
        cb_wait_front(ms, 8);

        // delta = beta v + k'^T S
        tile_regs_acquire();
        reconfig_data_format_srca(cb_bv);
        copy_init(cb_bv);
        copy_tile(cb_bv, 0, 0);
        copy_tile(cb_bv, 1, 1);
        reconfig_data_format<SrcOrder::Reverse>(cb_in, ms);
        matmul_init(cb_in, ms);
        for (uint32_t c = 0; c < 2; c++) {
            for (uint32_t j = 0; j < 4; j++) {
                matmul_tiles(cb_in, ms, j, 2 * j + c, c);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_bv, 2);
        cb_pop_front(ms, 8);
        cb_reserve_back(cb_delta, 2);
        tile_regs_wait();
        pack_tile(0, cb_delta);
        pack_tile(1, cb_delta);
        tile_regs_release();
        cb_push_back(cb_delta, 2);

        // S = d S + k delta^T, one k-tile (2 state tiles) at a time, into both copies
        cb_wait_front(cb_delta, 2);
        cb_wait_front(xs, 8);
        for (uint32_t j = 0; j < 4; j++) {
            tile_regs_acquire();
            reconfig_data_format_srca(xs);
            copy_init(xs);
            copy_tile(xs, 2 * j, 0);
            copy_tile(xs, 2 * j + 1, 1);
            reconfig_data_format<SrcOrder::Reverse>(cb_in, cb_delta);
            matmul_init(cb_in, cb_delta);
            matmul_tiles(cb_in, cb_delta, 4 + j, 0, 2);
            matmul_tiles(cb_in, cb_delta, 4 + j, 1, 3);
            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, state_update, 0, VectorMode::RC,
                                                   d_bits)));
            tile_regs_commit();
            cb_reserve_back(xd, 2);
            cb_reserve_back(md, 2);
            tile_regs_wait();
            pack_tile(0, xd);
            pack_tile(1, xd);
            pack_tile(0, md);
            pack_tile(1, md);
            tile_regs_release();
            cb_push_back(xd, 2);
            cb_push_back(md, 2);
        }
        cb_pop_front(cb_delta, 2);
        cb_pop_front(xs, 8);

        // o = q^T S
        cb_wait_front(md, 8);
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_in, md);
        matmul_init(cb_in, md);
        for (uint32_t c = 0; c < 2; c++) {
            for (uint32_t j = 0; j < 4; j++) {
                matmul_tiles(cb_in, md, 8 + j, 2 * j + c, c);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_in, 12);
        cb_reserve_back(cb_o, 2);
        tile_regs_wait();
        pack_tile(0, cb_o);
        pack_tile(1, cb_o);
        tile_regs_release();
        cb_push_back(cb_o, 2);

        xs = xd;
        ms = md;
    }

    // the final state out, exactly (the operand copy is left in place: the program ends)
    cb_wait_front(xs, 8);
    tile_regs_acquire();
    reconfig_data_format_srca(xs);
    copy_init(xs);
    for (uint32_t i = 0; i < 8; i++) {
        copy_tile(xs, i, i);
    }
    tile_regs_commit();
    cb_pop_front(xs, 8);
    cb_reserve_back(cb_sout, 8);
    tile_regs_wait();
    for (uint32_t i = 0; i < 8; i++) {
        pack_tile(i, cb_sout);
    }
    tile_regs_release();
    cb_push_back(cb_sout, 8);
}
