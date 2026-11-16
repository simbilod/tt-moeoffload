// The prefill selection's indexer score (2026-11-10): score[q, p] = ((r0 + r1) + (r2 + r3)) * s, r_h = relu(iq_h[q] .
// pk[p]), s = 128^-0.5, fp32, as g7_prefill.Attn.select's ttnn ops make it (SEL_FAST's per-head matmuls, relu in the
// packer, then three adds and a multiply over [Tq, L / 4] fp32 in DRAM). Here per (pair of query tiles, pool tile)
// the 4 heads x 2 query tiles accumulate over the 4 k-tiles in Dest 0..7 (pk's tiles transposed in the unpacker),
// then relu, the sums and the scale on the SFPU, and only the 2 score tiles leave. cb 0 iq (fp32 as served, or bf16; 32 tiles a pair),
// cb 1 pk (fp32, 4 a pool tile, resident), cb 16 the score (fp32). Runtime args: pool tiles, query tiles.
// Compile args: the scale's fp32 bits.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

constexpr uint32_t cb_iq = 0, cb_pk = 1, cb_out = 16;

void kernel_main() {
    constexpr uint32_t SCALE = get_compile_time_arg_val(0);
    const uint32_t np = get_arg_val<uint32_t>(0), nq = get_arg_val<uint32_t>(1);
    if (np == 0) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_iq, cb_pk, cb_out);
    cb_wait_front(cb_pk, 4 * np);
    for (uint32_t q = 0; q < nq; q += 2) {
        cb_wait_front(cb_iq, 32);
        for (uint32_t p = 0; p < np; p++) {
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_iq, cb_pk);
            matmul_init(cb_iq, cb_pk, 1);
            for (uint32_t d = 0; d < 8; d++) {              // d = 4 m + h
                for (uint32_t k = 0; k < 4; k++) {
                    matmul_tiles(cb_iq, cb_pk, 4 * d + k, 4 * p + k, d);
                }
            }
            relu_tile_init();
            for (uint32_t d = 0; d < 8; d++) {
                relu_tile(d);
            }
            add_binary_tile_init();
            for (uint32_t m = 0; m < 8; m += 4) {
                add_binary_tile(m, m + 1, m);
                add_binary_tile(m + 2, m + 3, m + 2);
                add_binary_tile(m, m + 2, m);
            }
            binop_with_scalar_tile_init();
            mul_unary_tile(0, SCALE);
            mul_unary_tile(4, SCALE);
            tile_regs_commit();
            cb_reserve_back(cb_out, 2);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            pack_tile(4, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 2);
        }
        cb_pop_front(cb_iq, 32);
    }
}
