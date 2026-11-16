// Compute-side helpers for the fused layer programs (row-0 tiles: a vector travels 32 values per tile in row 0, faces
// 0 and 1; the other rows carry garbage that only reaches other rows, except where a tile is zero outside row 0).
#pragma once

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

// Dest 0 = min(Dest 0, lim), Dest 1 = clamp(Dest 1, -lim, lim): llama.cpp's SwiGLU clamp (GLM: lim 10) on gate and up
// (glm_fsh_compute.cpp's, shared with pf_gexp_compute.cpp's SWIGLU_LIMIT since 2026-11-13)
inline void clamp_gu(uint32_t lim_bits) {
    const sfpi::vFloat lim = Converter::as_float(lim_bits);
    const sfpi::vFloat nlim = -lim;
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::vFloat g = sfpi::dst_reg[0];
        v_if (g > lim) { g = lim; }
        v_endif;
        sfpi::dst_reg[0] = g;
        sfpi::vFloat u = sfpi::dst_reg[32];
        v_if (u > lim) { u = lim; }
        v_endif;
        v_if (u < nlim) { u = nlim; }
        v_endif;
        sfpi::dst_reg[32] = u;
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

// the hyper-connection combine for two streams (four do not fit with their weights and y in 8 Dest tiles):
// Dest A, A + 1 = X_s, Dest 4, 5 = w_s, Dest 6 = y: X_s += w_s y
template <int A>
inline void combine2() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat y = sfpi::dst_reg[6 * 32];
        sfpi::dst_reg[A * 32] = sfpi::dst_reg[A * 32] + sfpi::dst_reg[4 * 32] * y;
        sfpi::dst_reg[(A + 1) * 32] = sfpi::dst_reg[(A + 1) * 32] + sfpi::dst_reg[5 * 32] * y;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace fused {

constexpr uint32_t F_ONE = 0x3f800000, F_ZERO = 0, F_TWO = 0x40000000, F_QUARTER = 0x3e800000;
constexpr uint32_t F_EPS = 0x358637bd;        // 1e-6
constexpr uint32_t F_RSQRT128 = 0x3db504f3;   // 1 / sqrt(128)
constexpr uint32_t F_INV128 = 0x3c000000;     // 1 / 128
constexpr uint32_t F_INV2560 = 0x39cccccd;    // 1 / 2560

#define SFPU_CALL(fn, ...) \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, 0, VectorMode::RC, ##__VA_ARGS__)))
// the same with Dest tile idst as the function's tile 0
#define SFPU_CALL_AT(fn, idst, ...) \
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, idst, VectorMode::RC, ##__VA_ARGS__)))

// one tile of cb src into Dest idst (exactly when the CB unpacks straight to Dest in fp32)
inline void load(uint32_t src, uint32_t itile, uint32_t idst) {
    reconfig_data_format_srca(src);
    copy_init(src);
    copy_tile(src, itile, idst);
}

// tiles 0..n-1 of cb src into Dest from..from+n-1
inline void load_n(uint32_t src, uint32_t n, uint32_t from = 0) {
    load(src, 0, from);
    for (uint32_t t = 1; t < n; t++) {
        copy_tile(src, t, from + t);
    }
}

// Dest from..from+n-1 into cb (reserve, pack, push); the caller has committed and waited
inline void pack_n(uint32_t cb, uint32_t n, uint32_t from = 0) {
    cb_reserve_back(cb, n);
    pack_reconfig_data_format(cb);
    for (uint32_t t = 0; t < n; t++) {
        pack_tile(from + t, cb);
    }
    cb_push_back(cb, n);
}

// commit, pack Dest from..from+n-1 into cb, release
inline void finish(uint32_t cb, uint32_t n, uint32_t from = 0) {
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb, n, from);
    tile_regs_release();
}

inline void matmul_start(uint32_t in0, uint32_t in1) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
}

// A weight stream consumed in batches of the weight CB: kt matmuls of x tile k with the next weight tile into Dest dst.
// end() pops a phase's padded last batch (the reader pushes whole batches).
struct Weights {
    uint32_t cb, batch, used = 0;
    void gemv(uint32_t xcb, uint32_t kt, uint32_t dst) {
        for (uint32_t k = 0; k < kt; k++) {
            if (used == 0) {
                cb_wait_front(cb, batch);
            }
            matmul_tiles(xcb, cb, k, used, dst);
            if (++used == batch) {
                cb_pop_front(cb, batch);
                used = 0;
            }
        }
    }
    void end() {
        if (used) {
            cb_pop_front(cb, batch);
            used = 0;
        }
    }
};

// the sum of squares of n row-0 tiles of cb src (exact) through the ones tile into Dest 0, then rsqrt(sum * scale +
// eps); leaves the tile regs acquired with the result in Dest 0. sq: a Default-mode scratch CB of n tiles.
// eps_bits: the norm's epsilon (Qwen3.8's 1e-6 by default; GLM's KDA output norm takes its f_norm_rms_eps, 1e-5)
inline void rsqrt_of_sum(uint32_t src, uint32_t n, uint32_t scale_bits, uint32_t sq, uint32_t ones,
                         uint32_t eps_bits = F_EPS) {
    tile_regs_acquire();
    load_n(src, n);
    square_tile_init();
    for (uint32_t t = 0; t < n; t++) {
        square_tile(t);
    }
    finish(sq, n);
    cb_wait_front(sq, n);
    tile_regs_acquire();
    matmul_start(sq, ones);
    for (uint32_t t = 0; t < n; t++) {
        matmul_tiles(sq, ones, t, 0, 0);
    }
    SFPU_CALL(affine, scale_bits, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    cb_pop_front(sq, n);
}

}  // namespace fused
