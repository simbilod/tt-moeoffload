// Fused DeltaNet mixer compute (G3 step 2; the phases are described in fgdn_reader.cpp). Row-0 tiles throughout:
// a vector travels 32 values per tile in row 0 (faces 0 and 1); the other rows carry garbage that only ever reaches
// other rows, except where a tile is zero outside row 0 (q, k from the reader).
//
// A  per unit, the GEMV of its output tiles into Dest, packed raw (cb 16), then its epilogue in fp32:
//      v tile        x = silu(t0 s0 + t1 s1 + t2 s2 + t3 y), the new conv state (s1, s2, y) to cb 18
//      q / k head    the same for its 4 tiles, then x / sqrt(sum x^2 + eps) (q also / sqrt(128)); the sum through
//                    matmuls with an all-ones tile (row 0 of x^2 J holds the sum in every column)
//      z, beta       sigmoid
//      alpha         decay = exp(log(1 + exp(y + dt)) * neg_a)
//    results to cb 17 (fp32).
// B  one value head, its state S [128 k x 128 v] as tiles (j, c) at 4 j + c, d = decay, b = beta:
//      k' = -b d k, b v                               SFPU
//      k as columns                                   transpose of the zero-padded k tiles
//      delta = b v + k'^T S                           FPU, Dest starting at b v
//      S = d S + k delta^T                            FPU for the outer product, the SFPU applies d S + P in fp32
//      o = q^T S                                      FPU (q carries the 1/sqrt(128))
//      out = o / rms(o) * w * sigmoid(z)              the sum through the ones tile again
//    the state is kept twice, as the split delta-rule kernel does (gdn_compute.cpp): an exact copy unpacked to Dest
//    in fp32 for the update, and the FPU's operand copy; the new state goes out exactly (cb 19) and as the operand
//    for o (cb 20). out to cb 21 (bf16, C's x) and cb 22 (fp32, the check).
// C  the output projection's tiles, KC matmuls each into Dest 0, to cb 17.
// Runtime args: unit count, per unit (first tile, count, type), head (0xffffffff: none), C's tile count.
// Compile args: batch, phases.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// Dest 0 = y, 1..3 = s0..s2, 4..7 = t0..t3: Dest 1 = t0 s0 + t1 s1 + t2 s2 + t3 y (s0 is read before it is written)
inline void conv4() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::vFloat acc = sfpi::dst_reg[1 * 32] * sfpi::dst_reg[4 * 32];
        acc = acc + sfpi::dst_reg[2 * 32] * sfpi::dst_reg[5 * 32];
        acc = acc + sfpi::dst_reg[3 * 32] * sfpi::dst_reg[6 * 32];
        sfpi::dst_reg[1 * 32] = acc + sfpi::dst_reg[0] * sfpi::dst_reg[7 * 32];
        sfpi::dst_reg++;
    }
}

// Dest 0 = Dest 0 * scale + add (the bits of two floats)
inline void affine(uint32_t scale_bits, uint32_t add_bits) {
    const sfpi::vFloat scale = Converter::as_float(scale_bits), add = Converter::as_float(add_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * scale + add;
        sfpi::dst_reg++;
    }
}

// Dest 0..3 = -(Dest 0..3 * Dest 4 * Dest 5): k' = -b d k
inline void neg_scale4() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat s = -(sfpi::dst_reg[4 * 32] * sfpi::dst_reg[5 * 32]);
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * s;
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * s;
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * s;
        sfpi::dst_reg[96] = sfpi::dst_reg[96] * s;
        sfpi::dst_reg++;
    }
}

// Dest 0..3 = S, 4..7 = P: Dest 0..3 = d S + P
inline void state_update4(uint32_t d_bits) {
    const sfpi::vFloat d = Converter::as_float(d_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * d + sfpi::dst_reg[4 * 32];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * d + sfpi::dst_reg[5 * 32];
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * d + sfpi::dst_reg[6 * 32];
        sfpi::dst_reg[96] = sfpi::dst_reg[96] * d + sfpi::dst_reg[7 * 32];
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_c = 3, cb_xc = 4, cb_sq = 5, cb_r = 6;
constexpr uint32_t cb_kp = 7, cb_sx = 8, cb_sop = 9, cb_nw = 10, cb_q = 11, cb_k = 12, cb_ex = 13;
constexpr uint32_t cb_bv = 14, cb_kc = 15, cb_raw = 16, cb_out = 17, cb_snew = 18, cb_sxo = 19, cb_sop2 = 20;
constexpr uint32_t cb_hout = 21, cb_hchk = 22, cb_delta = 23, cb_o = 24;
constexpr uint32_t KA = 80, KC = 192;
constexpr uint32_t T_V = 0, T_Q = 1, T_K = 2, T_Z = 3, T_A = 4, T_B = 5;
constexpr uint32_t IDLE = 0xffffffff;
constexpr uint32_t F_ONE = 0x3f800000, F_ZERO = 0;
constexpr uint32_t F_EPS = 0x358637bd;      // 1e-6
constexpr uint32_t F_RSQRT128 = 0x3db504f3;  // 1 / sqrt(128)
constexpr uint32_t F_INV128 = 0x3c000000;    // 1 / 128

#define SFPU_CALL(fn, ...) \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, 0, VectorMode::RC, ##__VA_ARGS__)))

// one tile of cb src into Dest idst (exactly when the CB unpacks straight to Dest in fp32)
inline void load(uint32_t src, uint32_t itile, uint32_t idst) {
    reconfig_data_format_srca(src);
    copy_init(src);
    copy_tile(src, itile, idst);
}

// Dest 0..n-1 into cb (reserve, pack, push); the caller has committed
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

// the GEMV stream: KT matmuls of x tile k with the next weight tile into Dest dst; batches of the weight buffer
inline void gemv_tile(uint32_t kt, uint32_t dst, uint32_t batch, uint32_t& used) {
    for (uint32_t k = 0; k < kt; k++) {
        if (used == 0) {
            cb_wait_front(cb_w, batch);
        }
        matmul_tiles(cb_x, cb_w, k, used, dst);
        if (++used == batch) {
            cb_pop_front(cb_w, batch);
            used = 0;
        }
    }
}

// the sum of squares of n row-0 tiles of cb src through the ones tile, -> Dest 0 = rsqrt(sum * scale + eps)
inline void rsqrt_of_sum(uint32_t src, uint32_t n, uint32_t scale_bits) {
    tile_regs_acquire();
    load(src, 0, 0);
    for (uint32_t t = 1; t < n; t++) {
        copy_tile(src, t, t);
    }
    square_tile_init();
    for (uint32_t t = 0; t < n; t++) {
        square_tile(t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_sq, n);
    tile_regs_release();

    cb_wait_front(cb_sq, n);
    tile_regs_acquire();
    matmul_start(cb_sq, cb_ones);
    for (uint32_t t = 0; t < n; t++) {
        matmul_tiles(cb_sq, cb_ones, t, 0, 0);
    }
    SFPU_CALL(affine, scale_bits, F_EPS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    cb_pop_front(cb_sq, n);
}

void phase_a(uint32_t n_units, uint32_t batch, uint32_t& used) {
    cb_wait_front(cb_x, KC);
    for (uint32_t u = 0; u < n_units; u++) {
        const uint32_t count = get_arg_val<uint32_t>(2 + 3 * u);
        const uint32_t type = get_arg_val<uint32_t>(3 + 3 * u);

        // the GEMV: tile t of the unit into Dest t
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        for (uint32_t t = 0; t < count; t++) {
            gemv_tile(KA, t, batch, used);
        }
        if (type == T_Z || type == T_B) {
            sigmoid_tile_init();
            for (uint32_t t = 0; t < count; t++) {
                sigmoid_tile(t);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_n((type == T_Z || type == T_B) ? cb_out : cb_raw, count);
        tile_regs_release();
        if (type == T_Z || type == T_B) {
            continue;
        }

        cb_wait_front(cb_raw, count);
        if (type == T_A) {
            for (uint32_t t = 0; t < count; t++) {
                cb_wait_front(cb_c, 7);
                tile_regs_acquire();
                load(cb_raw, t, 0);
                load(cb_c, 0, 1);  // dt
                load(cb_c, 1, 2);  // neg_a
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                // softplus as log(1 + exp(x)): the LLK softplus returns 0 below x = -5 (its bf16 polynomial)
                exp_tile_init();
                exp_tile(0);
                SFPU_CALL(affine, F_ONE, F_ONE);
                log_tile_init();
                log_tile(0);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                exp_tile_init();
                exp_tile(0);
                tile_regs_commit();
                cb_pop_front(cb_c, 7);
                tile_regs_wait();
                pack_n(cb_out, 1);
                tile_regs_release();
            }
            cb_pop_front(cb_raw, count);
            continue;
        }

        // the causal conv + silu, tile by tile; a head's results wait in cb 4 for the norm
        const bool head = type == T_Q || type == T_K;
        for (uint32_t t = 0; t < count; t++) {
            cb_wait_front(cb_c, 7);
            tile_regs_acquire();
            load(cb_raw, t, 0);
            reconfig_data_format_srca(cb_c);
            copy_init(cb_c);
            for (uint32_t j = 0; j < 7; j++) {
                copy_tile(cb_c, j, 1 + j);
            }
            SFPU_CALL(conv4);
            silu_tile_init();
            silu_tile(1);
            tile_regs_commit();
            cb_pop_front(cb_c, 7);
            tile_regs_wait();
            pack_n(head ? cb_xc : cb_out, 1, 1);
            cb_reserve_back(cb_snew, 3);
            pack_reconfig_data_format(cb_snew);
            pack_tile(2, cb_snew);  // new s0 = old s1
            pack_tile(3, cb_snew);  // new s1 = old s2
            pack_tile(0, cb_snew);  // new s2 = y
            cb_push_back(cb_snew, 3);
            tile_regs_release();
        }
        cb_pop_front(cb_raw, count);
        if (!head) {
            continue;
        }

        // the head's L2 norm
        cb_wait_front(cb_xc, count);
        rsqrt_of_sum(cb_xc, count, F_ONE);
        if (type == T_Q) {
            SFPU_CALL(affine, F_RSQRT128, F_ZERO);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_r, 1);
        tile_regs_release();

        cb_wait_front(cb_r, 1);
        tile_regs_acquire();
        load(cb_r, 0, 4);
        load(cb_xc, 0, 0);
        for (uint32_t t = 1; t < count; t++) {
            copy_tile(cb_xc, t, t);
        }
        mul_binary_tile_init();
        for (uint32_t t = 0; t < count; t++) {
            mul_binary_tile(t, 4, t);
        }
        tile_regs_commit();
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_xc, count);
        tile_regs_wait();
        pack_n(cb_out, count);
        tile_regs_release();
    }
    cb_pop_front(cb_x, KC);
}

void phase_b() {
    cb_wait_front(cb_q, 4);
    cb_wait_front(cb_k, 4);
    cb_wait_front(cb_ex, 10);  // v 0..3, sigmoid(z) 4..7, beta 8, decay 9
    cb_wait_front(cb_nw, 4);
    const uint32_t d_bits = read_tile_value(cb_ex, 9, 0);

    // k' = -b d k (from the zero-padded k: the FPU's operand precision, as k' only feeds the FPU)
    tile_regs_acquire();
    load(cb_k, 0, 0);
    for (uint32_t t = 1; t < 4; t++) {
        copy_tile(cb_k, t, t);
    }
    load(cb_ex, 8, 4);
    copy_tile(cb_ex, 9, 5);
    SFPU_CALL(neg_scale4);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_kp, 4);
    tile_regs_release();

    // b v (exact)
    tile_regs_acquire();
    load(cb_ex, 0, 0);
    for (uint32_t t = 1; t < 4; t++) {
        copy_tile(cb_ex, t, t);
    }
    copy_tile(cb_ex, 8, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) {
        mul_binary_tile(t, 4, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_bv, 4);
    tile_regs_release();

    // k as columns
    tile_regs_acquire();
    reconfig_data_format_srca(cb_k);
    transpose_init(cb_k);
    for (uint32_t t = 0; t < 4; t++) {
        transpose_tile(cb_k, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_kc, 4);
    tile_regs_release();
    cb_pop_front(cb_k, 4);

    // delta = b v + k'^T S
    cb_wait_front(cb_bv, 4);
    cb_wait_front(cb_kp, 4);
    cb_wait_front(cb_sop, 16);
    tile_regs_acquire();
    load(cb_bv, 0, 0);
    for (uint32_t t = 1; t < 4; t++) {
        copy_tile(cb_bv, t, t);
    }
    matmul_start(cb_kp, cb_sop);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_kp, cb_sop, j, 4 * j + c, c);
        }
    }
    tile_regs_commit();
    cb_pop_front(cb_bv, 4);
    cb_pop_front(cb_kp, 4);
    cb_pop_front(cb_sop, 16);
    tile_regs_wait();
    pack_n(cb_delta, 4);
    tile_regs_release();

    // S = d S + k delta^T, one k-tile (4 state tiles) at a time, into both copies
    cb_wait_front(cb_delta, 4);
    cb_wait_front(cb_kc, 4);
    cb_wait_front(cb_sx, 16);
    for (uint32_t j = 0; j < 4; j++) {
        tile_regs_acquire();
        load(cb_sx, 4 * j, 0);
        for (uint32_t c = 1; c < 4; c++) {
            copy_tile(cb_sx, 4 * j + c, c);
        }
        matmul_start(cb_kc, cb_delta);
        for (uint32_t c = 0; c < 4; c++) {
            matmul_tiles(cb_kc, cb_delta, j, c, 4 + c);
        }
        SFPU_CALL(state_update4, d_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_sxo, 4);
        pack_n(cb_sop2, 4);
        tile_regs_release();
    }
    cb_pop_front(cb_delta, 4);
    cb_pop_front(cb_kc, 4);
    cb_pop_front(cb_sx, 16);

    // o = q^T S, and its squares
    cb_wait_front(cb_sop2, 16);
    tile_regs_acquire();
    matmul_start(cb_q, cb_sop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_q, cb_sop2, j, 4 * j + c, c);
        }
    }
    tile_regs_commit();
    cb_pop_front(cb_q, 4);
    cb_pop_front(cb_sop2, 16);
    tile_regs_wait();
    pack_n(cb_o, 4);
    tile_regs_release();

    // out = o * rsqrt(mean o^2 + eps) * w * sigmoid(z)
    cb_wait_front(cb_o, 4);
    rsqrt_of_sum(cb_o, 4, F_INV128);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_r, 1);
    tile_regs_release();
    cb_wait_front(cb_r, 1);
    for (uint32_t c = 0; c < 4; c++) {
        tile_regs_acquire();
        load(cb_o, c, 0);
        load(cb_nw, c, 1);
        load(cb_ex, 4 + c, 2);
        load(cb_r, 0, 3);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(0, 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_hout, 1);
        pack_n(cb_hchk, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_r, 1);
    cb_pop_front(cb_o, 4);
    cb_pop_front(cb_ex, 10);
    cb_pop_front(cb_nw, 4);
}

void phase_c(uint32_t count, uint32_t batch, uint32_t& used) {
    cb_wait_front(cb_x, KC);
    for (uint32_t n = 0; n < count; n++) {
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        gemv_tile(KC, 0, batch, used);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_out, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_x, KC);
}

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t phases = get_compile_time_arg_val(1);
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    uint32_t a_tiles = 0;
    for (uint32_t u = 0; u < n_units; u++) {
        a_tiles += get_arg_val<uint32_t>(2 + 3 * u);
    }
    const uint32_t head = get_arg_val<uint32_t>(1 + 3 * n_units);
    const uint32_t c_count = get_arg_val<uint32_t>(2 + 3 * n_units);
    if (a_tiles == 0 && head == IDLE && c_count == 0) {
        return;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_raw);
    cb_wait_front(cb_ones, 1);
    uint32_t used = 0;
    if ((phases & 1) && a_tiles) {
        phase_a(n_units, batch, used);
    }
    if ((phases & 2) && head != IDLE) {
        phase_b();
    }
    if ((phases & 4) && c_count) {
        phase_c(c_count, batch, used);
    }
}
