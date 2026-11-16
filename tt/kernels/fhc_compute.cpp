// Fused hyper-connection mix compute (G3 step 3; the math and phases are in fhc_reader.cpp). Row-0 tiles as in
// fgdn_compute.cpp; the sum of a dense vector's squares through two matmuls with the all-ones tile J:
// (sum_t D_t^2) J holds each row's sum, J ((sum_t D_t^2) J) the total in every element.
// H1  r_s (cb 9, also cb 18 for the writer when n = 0), then lo[s][n] = r_s (X_s W'_s) to cb 17
// H2  lo's tiles: silu((sum_s P_s) / 4) as the gate GEMV's x (cb 3, bf16); the 4 gate tiles; then
//     hn = 1/4 sum_s sigmoid(gate_s) X_s norm_s r_s to cb 19 (bf16) and cb 20 (fp32); core 80: w = 2 sigmoid(lo / 4)
//     of lo's tile 10 to cb 17
// Runtime args: the H1 item (n, s; 0xffffffff none), the H2 tile (j; 80: w). Compile args: batch.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// Dest 0 = Dest 0 * scale + add (the bits of two floats)
inline void affine(uint32_t scale_bits, uint32_t add_bits) {
    const sfpi::vFloat scale = Converter::as_float(scale_bits), add = Converter::as_float(add_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * scale + add;
        sfpi::dst_reg++;
    }
}

// Dest 0 = Dest 0 + Dest 1 + Dest 2 + Dest 3
inline void sum4() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[32], c = sfpi::dst_reg[64], d = sfpi::dst_reg[96];
        sfpi::dst_reg[0] = (a + b) + (c + d);
        sfpi::dst_reg++;
    }
}

// Dest S = Dest S * Dest 4 * Dest 5 * Dest 6
template <int S>
inline void mul3() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat x = sfpi::dst_reg[S * 32], a = sfpi::dst_reg[4 * 32], b = sfpi::dst_reg[5 * 32];
        const sfpi::vFloat c = sfpi::dst_reg[6 * 32];
        sfpi::dst_reg[S * 32] = x * a * b * c;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_x2 = 3, cb_d = 4, cb_p = 5, cb_xs = 6, cb_nrm = 7;
constexpr uint32_t cb_rs = 8, cb_r = 9, cb_sq = 10, cb_t1 = 11;
constexpr uint32_t cb_out = 17, cb_rout = 18, cb_hn = 19, cb_hchk = 20;
constexpr uint32_t D = 2560, KS = 80, KU = 10, IDLE = 0xffffffff;
constexpr uint32_t F_ONE = 0x3f800000, F_ZERO = 0, F_TWO = 0x40000000, F_QUARTER = 0x3e800000;
constexpr uint32_t F_EPS = 0x358637bd;       // 1e-6
constexpr uint32_t F_INV2560 = 0x39cccccd;   // 1 / 2560

#define SFPU_CALL(fn, ...) \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, 0, VectorMode::RC, ##__VA_ARGS__)))

inline void load(uint32_t src, uint32_t itile, uint32_t idst) {
    reconfig_data_format_srca(src);
    copy_init(src);
    copy_tile(src, itile, idst);
}

inline void pack_n(uint32_t cb, uint32_t n, uint32_t from = 0) {
    cb_reserve_back(cb, n);
    pack_reconfig_data_format(cb);
    for (uint32_t t = 0; t < n; t++) {
        pack_tile(from + t, cb);
    }
    cb_push_back(cb, n);
}

inline void matmul_start(uint32_t in0, uint32_t in1) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
}

inline void gemv(uint32_t xcb, uint32_t kt, uint32_t dst, uint32_t batch, uint32_t& used) {
    for (uint32_t k = 0; k < kt; k++) {
        if (used == 0) {
            cb_wait_front(cb_w, batch);
        }
        matmul_tiles(xcb, cb_w, k, used, dst);
        if (++used == batch) {
            cb_pop_front(cb_w, batch);
            used = 0;
        }
    }
}

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t item = get_arg_val<uint32_t>(0), tile = get_arg_val<uint32_t>(1);
    if (item == IDLE && tile == IDLE) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_ones, 1);
    uint32_t used = 0;

    // ---- H1
    if (item != IDLE) {
        cb_wait_front(cb_d, 3);
        tile_regs_acquire();
        load(cb_d, 0, 0);
        copy_tile(cb_d, 1, 1);
        copy_tile(cb_d, 2, 2);
        square_tile_init();
        for (uint32_t t = 0; t < 3; t++) {
            square_tile(t);
        }
        tile_regs_commit();
        cb_pop_front(cb_d, 3);
        tile_regs_wait();
        pack_n(cb_sq, 3);
        tile_regs_release();
        cb_wait_front(cb_sq, 3);
        tile_regs_acquire();
        matmul_start(cb_sq, cb_ones);
        for (uint32_t t = 0; t < 3; t++) {
            matmul_tiles(cb_sq, cb_ones, t, 0, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_sq, 3);
        tile_regs_wait();
        pack_n(cb_t1, 1);
        tile_regs_release();
        cb_wait_front(cb_t1, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_t1);
        matmul_tiles(cb_ones, cb_t1, 0, 0, 0);
        SFPU_CALL(affine, F_INV2560, F_EPS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        cb_pop_front(cb_t1, 1);
        tile_regs_wait();
        pack_n(cb_r, 1);
        if ((item >> 2) == 0) {
            pack_n(cb_rout, 1);
        }
        tile_regs_release();

        cb_wait_front(cb_x, KS);
        cb_wait_front(cb_r, 1);
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        gemv(cb_x, KS, 0, batch, used);
        load(cb_r, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_x, KS);
        cb_pop_front(cb_r, 1);
        tile_regs_wait();
        pack_n(cb_out, 1);
        tile_regs_release();
    }

    // ---- H2
    if (tile == D / 32) {  // the inject weights
        cb_wait_front(cb_p, 4);
        tile_regs_acquire();
        load(cb_p, 0, 0);
        for (uint32_t s = 1; s < 4; s++) {
            copy_tile(cb_p, s, s);
        }
        SFPU_CALL(sum4);
        SFPU_CALL(affine, F_QUARTER, F_ZERO);
        sigmoid_tile_init();
        sigmoid_tile(0);
        SFPU_CALL(affine, F_TWO, F_ZERO);
        tile_regs_commit();
        cb_pop_front(cb_p, 4);
        tile_regs_wait();
        pack_n(cb_out, 1);
        tile_regs_release();
    } else if (tile != IDLE) {
        for (uint32_t t = 0; t < KU; t++) {
            cb_wait_front(cb_p, 4);
            tile_regs_acquire();
            load(cb_p, 0, 0);
            for (uint32_t s = 1; s < 4; s++) {
                copy_tile(cb_p, s, s);
            }
            SFPU_CALL(sum4);
            SFPU_CALL(affine, F_QUARTER, F_ZERO);
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            cb_pop_front(cb_p, 4);
            tile_regs_wait();
            pack_n(cb_x2, 1);
            tile_regs_release();
        }
        cb_wait_front(cb_x2, KU);
        cb_wait_front(cb_xs, 4);
        cb_wait_front(cb_nrm, 4);
        cb_wait_front(cb_rs, 4);
        tile_regs_acquire();
        matmul_start(cb_x2, cb_w);
        for (uint32_t s = 0; s < 4; s++) {
            gemv(cb_x2, KU, s, batch, used);
        }
        sigmoid_tile_init();
        for (uint32_t s = 0; s < 4; s++) {
            sigmoid_tile(s);
        }
        for (uint32_t s = 0; s < 4; s++) {
            load(cb_xs, s, 4);
            load(cb_nrm, s, 5);
            load(cb_rs, s, 6);
            if (s == 0) {
                SFPU_CALL(mul3<0>);
            } else if (s == 1) {
                SFPU_CALL(mul3<1>);
            } else if (s == 2) {
                SFPU_CALL(mul3<2>);
            } else {
                SFPU_CALL(mul3<3>);
            }
        }
        SFPU_CALL(sum4);
        SFPU_CALL(affine, F_QUARTER, F_ZERO);
        tile_regs_commit();
        cb_pop_front(cb_x2, KU);
        cb_pop_front(cb_xs, 4);
        cb_pop_front(cb_nrm, 4);
        cb_pop_front(cb_rs, 4);
        tile_regs_wait();
        pack_n(cb_hn, 1);
        pack_n(cb_hchk, 1);
        tile_regs_release();
    }
}
