// Can the SFPU decode i-quant codes fast enough? One tile of packed 32-bit words is unpacked into Dest (fp32),
// and each word's 8 nibbles are decoded to fp32 values, one 32-lane row at a time (decodes and their options:
// iq_decode_sfpu.h). Layouts on top of the decode, each a define:
//   PAIR         decode two nibbles side by side, so one chain's MAD latency hides behind the other's
//   GEMV         one-token matvec, x loaded for every nibble, one accumulator stored per row
//   GEMV_REG     one-token matvec, x loaded once per row (or per 2 nibbles), accumulators kept in LRegs across rows
//   GEMV_PAIR    GEMV_REG for IQ4_NL with paired decodes (needs NOMASK for registers)
//   UNROLL       unroll the 8 rows of a face (no loop overhead between rows)
//
// REPS == 0: correctness. Nibble t of every word goes to output tile t (Dest tiles 1..7 = nibbles 0..6, nibble 7
//            overwrites the input tile 0), packed to cb 16 in nibble order.
// REPS > 0:  throughput. The decode runs REPS times over the tile already in Dest; the 8 tiles are packed once.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "iq_decode_sfpu.h"

namespace ckernel::sfpu {

inline void iq_decode_face() {
    const Consts k = make_consts();
#if defined(GEMV_REG) || defined(GEMV_PAIR)
    vFloat acc0 = 0.0f;
#ifdef DECODE_LINEAR
    vFloat acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#endif
#endif
#ifdef UNROLL
#pragma GCC unroll 8
#else
#pragma GCC unroll 0
#endif
    for (int d = 0; d < 8; d++) {
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
#if defined(GEMV_REG) && defined(DECODE_LINEAR)
        // a word = 4 outputs x 2 k-subgroups: two x loads per row, four accumulators
        vFloat x = dst_reg[1 * TILE_ROWS];
        acc0 = nib<0>(w, k) * x + acc0;
        acc1 = nib<4>(w, k) * x + acc1;
        acc2 = nib<8>(w, k) * x + acc2;
        acc3 = nib<12>(w, k) * x + acc3;
        x = dst_reg[2 * TILE_ROWS];
        acc0 = nib<16>(w, k) * x + acc0;
        acc1 = nib<20>(w, k) * x + acc1;
        acc2 = nib<24>(w, k) * x + acc2;
        acc3 = nib<28>(w, k) * x + acc3;
#elif defined(GEMV_REG)
        // a word = 2 outputs x 4 k-subgroups: four x loads per row. One accumulator: the polynomial's constants
        // leave no LReg for a second.
        vFloat x = dst_reg[1 * TILE_ROWS];
        acc0 = nib<0>(w, k) * x + acc0;
        acc0 = nib<4>(w, k) * x + acc0;
        x = dst_reg[2 * TILE_ROWS];
        acc0 = nib<8>(w, k) * x + acc0;
        acc0 = nib<12>(w, k) * x + acc0;
        x = dst_reg[3 * TILE_ROWS];
        acc0 = nib<16>(w, k) * x + acc0;
        acc0 = nib<20>(w, k) * x + acc0;
        x = dst_reg[4 * TILE_ROWS];
        acc0 = nib<24>(w, k) * x + acc0;
        acc0 = nib<28>(w, k) * x + acc0;
#elif defined(GEMV_PAIR)
        // as GEMV_REG for IQ4_NL, with nibbles decoded in pairs; x is loaded once the pair's chains are done
        vFloat a, b, x;
        pair<0, 4>(w, k, a, b);
        x = dst_reg[1 * TILE_ROWS];
        acc0 = a * x + acc0;
        acc0 = b * x + acc0;
        pair<8, 12>(w, k, a, b);
        x = dst_reg[2 * TILE_ROWS];
        acc0 = a * x + acc0;
        acc0 = b * x + acc0;
        pair<16, 20>(w, k, a, b);
        x = dst_reg[3 * TILE_ROWS];
        acc0 = a * x + acc0;
        acc0 = b * x + acc0;
        pair<24, 28>(w, k, a, b);
        x = dst_reg[4 * TILE_ROWS];
        acc0 = a * x + acc0;
        acc0 = b * x + acc0;
#elif defined(GEMV)
        vFloat acc = nib<0>(w, k) * dst_reg[1 * TILE_ROWS];
        acc = nib<4>(w, k) * dst_reg[2 * TILE_ROWS] + acc;
        acc = nib<8>(w, k) * dst_reg[3 * TILE_ROWS] + acc;
        acc = nib<12>(w, k) * dst_reg[4 * TILE_ROWS] + acc;
        acc = nib<16>(w, k) * dst_reg[5 * TILE_ROWS] + acc;
        acc = nib<20>(w, k) * dst_reg[6 * TILE_ROWS] + acc;
        acc = nib<24>(w, k) * dst_reg[7 * TILE_ROWS] + acc;
        acc = nib<28>(w, k) * dst_reg[1 * TILE_ROWS] + acc;
        dst_reg[7 * TILE_ROWS] = acc;
#elif defined(PAIR)
        vFloat a, b;
        pair<0, 4>(w, k, a, b);
        dst_reg[1 * TILE_ROWS] = a;
        dst_reg[2 * TILE_ROWS] = b;
        pair<8, 12>(w, k, a, b);
        dst_reg[3 * TILE_ROWS] = a;
        dst_reg[4 * TILE_ROWS] = b;
        pair<16, 20>(w, k, a, b);
        dst_reg[5 * TILE_ROWS] = a;
        dst_reg[6 * TILE_ROWS] = b;
        pair<24, 28>(w, k, a, b);
        dst_reg[7 * TILE_ROWS] = a;
        dst_reg[0] = b;
#else
        dst_reg[1 * TILE_ROWS] = nib<0>(w, k);
        dst_reg[2 * TILE_ROWS] = nib<4>(w, k);
        dst_reg[3 * TILE_ROWS] = nib<8>(w, k);
        dst_reg[4 * TILE_ROWS] = nib<12>(w, k);
        dst_reg[5 * TILE_ROWS] = nib<16>(w, k);
        dst_reg[6 * TILE_ROWS] = nib<20>(w, k);
        dst_reg[7 * TILE_ROWS] = nib<24>(w, k);
        dst_reg[0] = nib<28>(w, k);
#endif
        dst_reg++;
    }
#if defined(GEMV_REG) || defined(GEMV_PAIR)
    // the face's partial sums, once (a real kernel keeps them across the K loop and reduces at the end)
    dst_reg[5 * TILE_ROWS] = acc0;
#ifdef DECODE_LINEAR
    dst_reg[6 * TILE_ROWS] = acc1;
    dst_reg[7 * TILE_ROWS] = acc2 + acc3;
#endif
#endif
}

}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup; no tanh state is used
    MATH((ckernel::sfpu::decode_init()));

    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    if constexpr (reps == 0) {
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, iq_decode_face, 0, VectorMode::RC)));
    } else {
        for (uint32_t r = 0; r < reps; r++) {
            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, iq_decode_face, 0, VectorMode::RC)));
        }
    }
    tile_regs_commit();
    cb_pop_front(cb_in, 1);

    cb_reserve_back(cb_out, 8);
    tile_regs_wait();
    if constexpr (reps == 0) {
        for (uint32_t t = 1; t < 8; t++) {
            pack_tile(t, cb_out);
        }
        pack_tile(0, cb_out);
    } else {
        for (uint32_t t = 0; t < 8; t++) {
            pack_tile(t, cb_out);
        }
    }
    tile_regs_release();
    cb_push_back(cb_out, 8);
}
