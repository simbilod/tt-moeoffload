// A one-token matvec over weights streamed from DRAM. Each code tile from cb 0 is unpacked into Dest tile 0
// and decoded on the SFPU (iq_decode_sfpu.h); its rows multiply x and add into the accumulator tile 5, which is
// packed once at the end. x is four constant tiles (1, 1/2, 1/4, 2) so the result can be checked exactly.
//   DECODE_MXFP4 / DECODE_LINEAR (+ FOLD_OFFSET) ...   a uint32 tile = 8 codes per element
//   DECODE_NONE   the tile is already weights (bfp8_b), unpacked by hardware: one MAD per row
//   READ_ONLY     pop the tiles and do nothing: the streaming rate alone
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifndef DECODE_NONE
#include "iq_decode_sfpu.h"
#else
#include "sfpi.h"
#endif

namespace ckernel::sfpu {

#ifdef DECODE_NONE
using sfpi::dst_reg;
using sfpi::vFloat;
constexpr uint32_t TILE_ROWS = 32;
#endif

inline void init_x_face() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        dst_reg[1 * TILE_ROWS] = 1.0f;
        dst_reg[2 * TILE_ROWS] = 0.5f;
        dst_reg[3 * TILE_ROWS] = 0.25f;
        dst_reg[4 * TILE_ROWS] = 2.0f;
        dst_reg[5 * TILE_ROWS] = 0.0f;
        dst_reg++;
    }
}

inline void stream_face() {
#ifndef DECODE_NONE
    const Consts k = make_consts();
#endif
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat acc = dst_reg[5 * TILE_ROWS];
#ifdef DECODE_NONE
        vFloat w = dst_reg[0], x = dst_reg[1 * TILE_ROWS];
        acc = w * x + acc;
#else
        vUInt w = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vFloat x = dst_reg[1 * TILE_ROWS];
        acc = nib<0>(w, k) * x + acc;
        acc = nib<4>(w, k) * x + acc;
        x = dst_reg[2 * TILE_ROWS];
        acc = nib<8>(w, k) * x + acc;
        acc = nib<12>(w, k) * x + acc;
        x = dst_reg[3 * TILE_ROWS];
        acc = nib<16>(w, k) * x + acc;
        acc = nib<20>(w, k) * x + acc;
        x = dst_reg[4 * TILE_ROWS];
        acc = nib<24>(w, k) * x + acc;
        acc = nib<28>(w, k) * x + acc;
#endif
        dst_reg[5 * TILE_ROWS] = acc;
        dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup; no tanh state is used
#ifndef DECODE_NONE
    MATH((ckernel::sfpu::decode_init()));
#endif

    tile_regs_acquire();
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, init_x_face, 0, VectorMode::RC)));
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in, 1);
#ifndef READ_ONLY
        copy_tile(cb_in, 0, 0);
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, stream_face, 0, VectorMode::RC)));
#endif
        cb_pop_front(cb_in, 1);
    }
    tile_regs_commit();

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(5, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
