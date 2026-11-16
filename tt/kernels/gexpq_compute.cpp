// Grouped expert compute, native codes (G2 prefill). Per expert and n-tile:
//   decode   each code tile on the SFPU (iq_decode_sfpu.h: DECODE_LINEAR gives 2c - 15 for IQ3_S, DECODE_IQ4NL the
//            IQ4_NL table) into 8 F tiles, then W = F * scale by a row broadcast of the block's row of the scale tile
//            (bcast_row_idx), into cb 26 in fp32: the n-tile's whole column of kt weight tiles, decoded once
//   matmul   the kt x MT matmuls x_e[m, k] W[k, n] accumulated in Dest 0..MT-1, packed
// so each decode serves all of the expert's token tiles. IQ3_S's scale is a / 2 (w = a c + b with b = -7.5 a), IQ4_NL's
// is d. cb 0 codes (uint32, unpacked to Dest), cb 3 scales (fp32), cb 24 F (bf16, exact: odd 1..15 or the table),
// cb 26 W (fp32), cb 1 x (bf16), cb 16 y. Runtime args: first expert, count. Compile args: reps, kt, nt, mt, mt_max, kc.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "iq_decode_sfpu.h"

namespace ckernel::sfpu {

// codes in Dest 0 -> f(code) of nibbles 4 * PASS + 0..3 in Dest 4..7 (decode, one face per call)
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

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_codes = 0, cb_x = 1, cb_s = 3, cb_f = 24, cb_w = 26, cb_out = 16;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t mt_max = get_compile_time_arg_val(4);
    constexpr uint32_t kc = get_compile_time_arg_val(5);
    const uint32_t e_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup; no tanh state is used
    MATH((ckernel::sfpu::decode_init()));
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t e = 0; e < e_count; e++) {
            cb_wait_front(cb_x, mt_max * kt);
            for (uint32_t n = 0; n < nt; n++) {
                for (uint32_t c = 0; c < kc; c++) {
                    cb_wait_front(cb_codes, 1);
                    reconfig_data_format_srca(cb_codes);
                    copy_init(cb_codes);
                    pack_reconfig_data_format(cb_f);
                    for (uint32_t pass = 0; pass < 2; pass++) {
                        cb_reserve_back(cb_f, 4);
                        tile_regs_acquire();
                        copy_tile(cb_codes, 0, 0);
                        if (pass == 0) {
                            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, decode_face<0>, 0,
                                                                   VectorMode::RC)));
                        } else {
                            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, decode_face<1>, 0,
                                                                   VectorMode::RC)));
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t t = 4; t < 8; t++) {
                            pack_tile(t, cb_f);
                        }
                        tile_regs_release();
                        cb_push_back(cb_f, 4);
                    }
                    cb_pop_front(cb_codes, 1);

                    // W = F * the block's scale row (blocks 8c .. 8c + 7 sit in rows 8 (c % 4) + t of the scale tile)
                    cb_wait_front(cb_f, 8);
                    cb_wait_front(cb_s, 1);
                    tile_regs_acquire();
                    reconfig_data_format(cb_f, cb_s);
                    mul_bcast_rows_init(cb_f, cb_s);
                    for (uint32_t t = 0; t < 8; t++) {
                        mul_tiles_bcast_rows(cb_f, cb_s, t, 0, t, 8 * (c % 4) + t);
                    }
                    tile_regs_commit();
                    cb_pop_front(cb_f, 8);
                    if (c % 4 == 3 || c == kc - 1) {
                        cb_pop_front(cb_s, 1);
                    }
                    cb_reserve_back(cb_w, 8);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_w);
                    for (uint32_t t = 0; t < 8; t++) {
                        pack_tile(t, cb_w);
                    }
                    tile_regs_release();
                    cb_push_back(cb_w, 8);
                }

                cb_wait_front(cb_w, kt);
                tile_regs_acquire();
                reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_w);
                matmul_init(cb_x, cb_w);
                for (uint32_t k = 0; k < kt; k++) {
                    for (uint32_t m = 0; m < mt; m++) {
                        matmul_tiles(cb_x, cb_w, m * kt + k, k, m);
                    }
                }
                tile_regs_commit();
                cb_pop_front(cb_w, kt);
                cb_reserve_back(cb_out, mt);
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                for (uint32_t m = 0; m < mt; m++) {
                    pack_tile(m, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, mt);
            }
            cb_pop_front(cb_x, mt_max * kt);
        }
    }
}
