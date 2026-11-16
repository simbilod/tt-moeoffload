// One matvec unit, 32 outputs x 1,024 inputs (32 blocks of 32), from card-form codes
// (tt/formats/README.md). The SFPU decodes, the FPU multiplies, the SFPU applies the block affines:
//
//   1. decode: each code tile (8 codes per 32-bit word) -> 8 tiles F_b of f(code) values [32 k x 32 n], packed
//      to cb 24 as bf16 (exact: small integers). Code tile c, nibble t = block 8c + t.
//   2. matmul: P += X_b . F_b for b = 0..31, all into Dest 0. X_b holds block b's x in its row b (zeros elsewhere),
//      so row b of P is block b's partial sums for the 32 outputs.
//   3. affine: Y = A * P + B * SX on the SFPU, A[b][n] = a and B[b][n] = b of (output n, block b), SX[b][.] = the
//      sum of block b's x. The outputs are the column sums of Y (the host sums them here; on-card next).
//
// Inputs: cb 0 codes (uint32, 4 tiles), cb 1 X (bf16, 32), cb 2 A, cb 3 B, cb 4 SX (fp32, 1 each).
// Output cb 16: P and Y (fp32). REPS > 1 repeats phases 1-3 over the same inputs (timing); only the last is packed.
// Decode: DECODE_LINEAR + FOLD_OFFSET (f = c) or DECODE_IQ4NL + IQ4_MAGIC (f = kvalues_iq4nl[c]); see
// iq_decode_sfpu.h. Timing breakdown: SKIP_DECODE (codes copied and packed, no SFPU decode), SKIP_MATMUL.
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
#include "iq_decode_sfpu.h"

namespace ckernel::sfpu {

// codes in Dest 0 -> f(code) of nibbles 4 * PASS + 0..3 in Dest 4..7
template <int PASS>
inline void decode_face() {
    const Consts k = make_consts();
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
#ifdef PAIR
        vFloat a, b;
        pair<16 * PASS + 0, 16 * PASS + 4>(w, k, a, b);
        dst_reg[4 * TILE_ROWS] = a;
        dst_reg[5 * TILE_ROWS] = b;
        pair<16 * PASS + 8, 16 * PASS + 12>(w, k, a, b);
        dst_reg[6 * TILE_ROWS] = a;
        dst_reg[7 * TILE_ROWS] = b;
#else
        dst_reg[4 * TILE_ROWS] = nib<16 * PASS + 0>(w, k);
        dst_reg[5 * TILE_ROWS] = nib<16 * PASS + 4>(w, k);
        dst_reg[6 * TILE_ROWS] = nib<16 * PASS + 8>(w, k);
        dst_reg[7 * TILE_ROWS] = nib<16 * PASS + 12>(w, k);
#endif
        dst_reg++;
    }
}

// Y = A * P + B * SX: P in Dest 0, A 1, B 2, SX 3 -> Y in Dest 4
inline void affine_face() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat p = dst_reg[0], a = dst_reg[1 * TILE_ROWS], b = dst_reg[2 * TILE_ROWS], sx = dst_reg[3 * TILE_ROWS];
        dst_reg[4 * TILE_ROWS] = p * a + b * sx;
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_codes = 0, cb_x = 1, cb_a = 2, cb_b = 3, cb_sx = 4, cb_f = 24, cb_out = 16;
constexpr uint32_t n_code_tiles = 4, n_blocks = 32;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_f, cb_out);
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup; no tanh state is used
    MATH((ckernel::sfpu::decode_init()));
    cb_wait_front(cb_codes, n_code_tiles);
    cb_wait_front(cb_x, n_blocks);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_wait_front(cb_sx, 1);

    for (uint32_t rep = 0; rep < reps; rep++) {
        // 1. decode
        reconfig_data_format_srca(cb_codes);
        copy_init(cb_codes);
        pack_reconfig_data_format(cb_f);
        for (uint32_t c = 0; c < n_code_tiles; c++) {
            for (uint32_t pass = 0; pass < 2; pass++) {
                cb_reserve_back(cb_f, 4);
                tile_regs_acquire();
                copy_tile(cb_codes, c, 0);
#ifdef SKIP_DECODE
                if (false) {
#else
                if (pass == 0) {
#endif
                    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, decode_face<0>, 0, VectorMode::RC)));
                } else {
#ifndef SKIP_DECODE
                    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, decode_face<1>, 0, VectorMode::RC)));
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 4; t < 8; t++) {
                    pack_tile(t, cb_f);
                }
                tile_regs_release();
                cb_push_back(cb_f, 4);
            }
        }

        // 2. matmul, 3. affine
        cb_wait_front(cb_f, n_blocks);
        reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_f);
        matmul_init(cb_x, cb_f);
        tile_regs_acquire();
#ifndef SKIP_MATMUL
        for (uint32_t b = 0; b < n_blocks; b++) {
            matmul_tiles(cb_x, cb_f, b, b, 0);
        }
#endif
        reconfig_data_format_srca(cb_a);
        copy_init(cb_a);
        copy_tile(cb_a, 0, 1);
        copy_tile(cb_b, 0, 2);
        copy_tile(cb_sx, 0, 3);
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, affine_face, 0, VectorMode::RC)));
        tile_regs_commit();
        cb_pop_front(cb_f, n_blocks);
        if (rep + 1 == reps) {
            cb_reserve_back(cb_out, 2);
            pack_reconfig_data_format(cb_out);
            tile_regs_wait();
            pack_tile(0, cb_out);
            pack_tile(4, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 2);
        } else {
            tile_regs_wait();
            tile_regs_release();
        }
    }
    cb_pop_front(cb_codes, n_code_tiles);
    cb_pop_front(cb_x, n_blocks);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    cb_pop_front(cb_sx, 1);
}
