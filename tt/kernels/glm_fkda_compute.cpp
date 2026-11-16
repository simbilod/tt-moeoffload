// GLM's fused KDA mixer, compute (the phases in glm_fkda_reader.cpp). Row-0 tiles throughout.
//
// A  per unit, the GEMV of its output tiles into Dest (x from the reader's ring, a batch with each weight batch),
//    packed raw (cb 16), then its epilogue in fp32:
//      v tile        x = silu(t0 s0 + t1 s1 + t2 s2 + t3 y), the new conv state (s1, s2, y) to cb 18
//      q / k head    the same for its 4 tiles, then x / sqrt(sum x^2 + 1e-6) (q also / sqrt(128))
//      plain         none (f_lo, g_lo)
//      beta          sigmoid
//    results to cb 17 (fp32).
// B  one value head:
//      d = exp(-5 sigmoid(neg_a (f_lo f_b_h) + dt neg_a))    the 16 matmuls on the FPU, the rest on the SFPU; to
//                                                           cb 25 (for k') and cb 33 (the reader builds Dbc from it)
//      g = sigmoid(g_lo g_b_h)                              to cb 34
//      then glm_kda_compute.cpp's step: k' = -b (d * k), delta = b v + k'^T S, S = S * Dbc + k delta^T (SFPU, fp32),
//      o = q^T S, out = o / rms(o) (eps 1e-5) * w * g, to cb 21 (bf16, C's x) and cb 22 (fp32, the check); the new
//      state exactly to cb 19
// C  the output projection's items (an output tile's KC / S k-tiles), the matmuls into Dest 0, to cb 17.
// Runtime args: unit count, per unit (first tile, count, type), head (0xffffffff: none), neg_a's bits (the head's),
// C's item count. Compile args: batch, phases, S.
#include <cstdint>

#include "fused/cmp.h"
#ifndef M_MAX
#define M_MAX 1
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Dest 0..3 *= -(Dest 4) (beta, filled across row 0)
inline void neg_by4() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat b = sfpi::dst_reg[4 * 32];
        const sfpi::vFloat s = -b;
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * s;
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * s;
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * s;
        sfpi::dst_reg[96] = sfpi::dst_reg[96] * s;
        sfpi::dst_reg++;
    }
}
// Dest 0..1 = S, 2..3 = P, 4 = Dbc: Dest 0..1 = S * Dbc + P
inline void kda_update2() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat d = sfpi::dst_reg[4 * 32];
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * d + sfpi::dst_reg[2 * 32];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * d + sfpi::dst_reg[3 * 32];
        sfpi::dst_reg++;
    }
}
// Dest 0..3 = Dest 0..3 * scale + Dest 4..7 (neg_a f + dt neg_a)
inline void scale_add4(uint32_t scale_bits) {
    const sfpi::vFloat s = Converter::as_float(scale_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * s + sfpi::dst_reg[4 * 32];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * s + sfpi::dst_reg[5 * 32];
        sfpi::dst_reg[64] = sfpi::dst_reg[64] * s + sfpi::dst_reg[6 * 32];
        sfpi::dst_reg[96] = sfpi::dst_reg[96] * s + sfpi::dst_reg[7 * 32];
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_c = 3, cb_xc = 4, cb_sq = 5, cb_r = 6;
constexpr uint32_t cb_kp = 7, cb_sx = 8, cb_sop = 9, cb_nw = 10, cb_q = 11, cb_k = 12, cb_ex = 13;
constexpr uint32_t cb_bv = 14, cb_kc = 15, cb_raw = 16, cb_out = 17, cb_snew = 18, cb_sxo = 19, cb_sop2 = 20;
constexpr uint32_t cb_hout = 21, cb_hchk = 22, cb_delta = 23, cb_o = 24, cb_kd = 25, cb_dbc = 26, cb_lw = 27;
constexpr uint32_t cb_xlo = 28, cb_dtn = 32, cb_dsrc = 33, cb_g = 34;
constexpr uint32_t KA = 128, KC = 256;
constexpr uint32_t T_V = 0, T_Q = 1, T_K = 2, T_P = 3, T_B = 5;
constexpr uint32_t IDLE = 0xffffffff;
constexpr uint32_t F_EPS5 = 0x3727c5ac;   // 1e-5, GLM's KDA output norm
constexpr uint32_t F_LB = 0xc0a00000;     // -5, GLM's decay bound
constexpr uint32_t F_MINUS1 = 0xbf800000;

// kt matmuls into Dest dst, x from the ring: a batch of x tiles with each batch of weights
inline void gemv_ring(uint32_t kt, uint32_t dst, uint32_t batch) {
    for (uint32_t c = 0; c < kt; c += batch) {
        cb_wait_front(cb_w, batch);
        cb_wait_front(cb_x, batch);
        for (uint32_t j = 0; j < batch; j++) matmul_tiles(cb_x, cb_w, j, j, dst);
        cb_pop_front(cb_w, batch);
        cb_pop_front(cb_x, batch);
    }
}

void phase_a(uint32_t n_units, uint32_t batch) {
    for (uint32_t u = 0; u < n_units; u++) {
        const uint32_t count = get_arg_val<uint32_t>(2 + 3 * u);
        const uint32_t type = get_arg_val<uint32_t>(3 + 3 * u);
        const bool done = type == T_P || type == T_B;      // no epilogue beyond the sigmoid

        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        for (uint32_t t = 0; t < count; t++) gemv_ring(KA, t, batch);
        if (type == T_B) {
            sigmoid_tile_init();
            for (uint32_t t = 0; t < count; t++) sigmoid_tile(t);
        }
        finish(done ? cb_out : cb_raw, count);
        if (done) continue;

        // the causal conv + silu, tile by tile; a head's results wait in cb 4 for the norm
        const bool head = type == T_Q || type == T_K;
#if M_MAX > 1
        // M rows: y went to the writer, which hands back the window (cb 3: y, W0..W2, the taps) per tile
        for (uint32_t t = 0; t < count; t++) {
            cb_wait_front(cb_c, 8);
            tile_regs_acquire();
            load_n(cb_c, 8);
            SFPU_CALL(conv4);
            silu_tile_init();
            silu_tile(1);
            tile_regs_commit();
            cb_pop_front(cb_c, 8);
            tile_regs_wait();
            pack_n(head ? cb_xc : cb_out, 1, 1);
            tile_regs_release();
        }
#else
        cb_wait_front(cb_raw, count);
        for (uint32_t t = 0; t < count; t++) {
            cb_wait_front(cb_c, 7);
            tile_regs_acquire();
            load(cb_raw, t, 0);
            reconfig_data_format_srca(cb_c);
            copy_init(cb_c);
            for (uint32_t j = 0; j < 7; j++) copy_tile(cb_c, j, 1 + j);
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
#endif
        if (!head) continue;

        // the head's L2 norm
        cb_wait_front(cb_xc, count);
        rsqrt_of_sum(cb_xc, count, F_ONE, cb_sq, cb_ones);
        if (type == T_Q) SFPU_CALL(affine, F_RSQRT128, F_ZERO);
        finish(cb_r, 1);
        cb_wait_front(cb_r, 1);
        tile_regs_acquire();
        load(cb_r, 0, 4);
        load_n(cb_xc, count);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < count; t++) mul_binary_tile(t, 4, t);
        tile_regs_commit();
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_xc, count);
        tile_regs_wait();
        pack_n(cb_out, count);
        tile_regs_release();
    }
}

void phase_b(uint32_t neg_a_bits) {
    // d and g from the low-rank halves (x: cb 28 tiles 0..3 f_lo, 4..7 g_lo; W: cb 27 pages 16 g + 4 n + k)
    cb_wait_front(cb_xlo, 8);
    cb_wait_front(cb_lw, 32);
    cb_wait_front(cb_dtn, 4);
#ifdef FK_DEBUG
    // 35: f alone (the matmuls into a fresh Dest); 36: the pre-sigmoid argument as the real block computes it
    tile_regs_acquire();
    matmul_start(cb_xlo, cb_lw);
    for (uint32_t n = 0; n < 4; n++) {
        for (uint32_t k = 0; k < 4; k++) matmul_tiles(cb_xlo, cb_lw, k, 4 * n + k, n);
    }
    finish(35, 4);
    // the matmuls into a fresh Dest, then dt * neg_a beside them: with the fp32 load first, the matmuls' sum came
    // out doubled (measured 2026-11-11: 2 f neg_a + dt neg_a)
    tile_regs_acquire();
    matmul_start(cb_xlo, cb_lw);
    for (uint32_t n = 0; n < 4; n++) {
        for (uint32_t k = 0; k < 4; k++) matmul_tiles(cb_xlo, cb_lw, k, 4 * n + k, n);
    }
    load_n(cb_dtn, 4, 4);
    SFPU_CALL(scale_add4, neg_a_bits);
    finish(36, 4);
#endif
    // the matmuls into a fresh Dest, then dt * neg_a beside them: with the fp32 load first, the matmuls' sum came
    // out doubled (measured 2026-11-11: 2 f neg_a + dt neg_a)
    tile_regs_acquire();
    matmul_start(cb_xlo, cb_lw);
    for (uint32_t n = 0; n < 4; n++) {
        for (uint32_t k = 0; k < 4; k++) matmul_tiles(cb_xlo, cb_lw, k, 4 * n + k, n);
    }
    load_n(cb_dtn, 4, 4);
    SFPU_CALL(scale_add4, neg_a_bits);
    sigmoid_tile_init();
    for (uint32_t t = 0; t < 4; t++) sigmoid_tile(t);
    for (uint32_t t = 0; t < 4; t++) SFPU_CALL_AT(affine, t, F_LB, F_ZERO);
    exp_tile_init();
    for (uint32_t t = 0; t < 4; t++) exp_tile(t);
    tile_regs_commit();
    cb_pop_front(cb_dtn, 4);
    tile_regs_wait();
    pack_n(cb_kd, 4);
    pack_n(cb_dsrc, 4);
    tile_regs_release();

    tile_regs_acquire();
    matmul_start(cb_xlo, cb_lw);
    for (uint32_t n = 0; n < 4; n++) {
        for (uint32_t k = 0; k < 4; k++) matmul_tiles(cb_xlo, cb_lw, 4 + k, 16 + 4 * n + k, n);
    }
    sigmoid_tile_init();
    for (uint32_t t = 0; t < 4; t++) sigmoid_tile(t);
    finish(cb_g, 4);
    cb_pop_front(cb_xlo, 8);
    cb_pop_front(cb_lw, 32);

    // the KDA step (glm_kda_compute.cpp)
    cb_wait_front(cb_q, 4);
    cb_wait_front(cb_k, 4);
    cb_wait_front(cb_ex, 10);  // v 0..3, beta 8 (across row 0)
    cb_wait_front(cb_nw, 4);
    cb_wait_front(cb_kd, 4);

    // k' = -b (d * k)
    tile_regs_acquire();
    load_n(cb_k, 4);
    load_n(cb_kd, 4, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) mul_binary_tile(t, 4 + t, t);
    load(cb_ex, 8, 4);
    SFPU_CALL(neg_by4);
    finish(cb_kp, 4);
    cb_pop_front(cb_kd, 4);

    // b v (exact)
    tile_regs_acquire();
    load_n(cb_ex, 4);
    load(cb_ex, 8, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) mul_binary_tile(t, 4, t);
    finish(cb_bv, 4);

    // k as columns
    tile_regs_acquire();
    reconfig_data_format_srca(cb_k);
    transpose_init(cb_k);
    for (uint32_t t = 0; t < 4; t++) transpose_tile(cb_k, t, t);
    finish(cb_kc, 4);
    cb_pop_front(cb_k, 4);

    // delta = b v + k'^T S
    cb_wait_front(cb_bv, 4);
    cb_wait_front(cb_kp, 4);
    cb_wait_front(cb_sop, 16);
    tile_regs_acquire();
    load_n(cb_bv, 4);
    matmul_start(cb_kp, cb_sop);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) matmul_tiles(cb_kp, cb_sop, j, 4 * j + c, c);
    }
    tile_regs_commit();
    cb_pop_front(cb_bv, 4);
    cb_pop_front(cb_kp, 4);
    cb_pop_front(cb_sop, 16);
    tile_regs_wait();
    pack_n(cb_delta, 4);
    tile_regs_release();

    // S = S * Dbc + k delta^T, half a row block (2 state tiles) at a time, into both copies
    cb_wait_front(cb_delta, 4);
    cb_wait_front(cb_kc, 4);
    cb_wait_front(cb_sx, 16);
    cb_wait_front(cb_dbc, 4);
    for (uint32_t j = 0; j < 4; j++) {
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t c0 = 2 * h;
            tile_regs_acquire();
            load(cb_sx, 4 * j + c0, 0);
            load(cb_sx, 4 * j + c0 + 1, 1);
            load(cb_dbc, j, 4);
            matmul_start(cb_kc, cb_delta);
            matmul_tiles(cb_kc, cb_delta, j, c0, 2);
            matmul_tiles(cb_kc, cb_delta, j, c0 + 1, 3);
            SFPU_CALL(kda_update2);
            tile_regs_commit();
            tile_regs_wait();
            pack_n(cb_sxo, 2);
            pack_n(cb_sop2, 2);
            tile_regs_release();
        }
    }
    cb_pop_front(cb_delta, 4);
    cb_pop_front(cb_kc, 4);
    cb_pop_front(cb_sx, 16);
    cb_pop_front(cb_dbc, 4);

    // o = q^T S
    cb_wait_front(cb_sop2, 16);
    tile_regs_acquire();
    matmul_start(cb_q, cb_sop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) matmul_tiles(cb_q, cb_sop2, j, 4 * j + c, c);
    }
    tile_regs_commit();
    cb_pop_front(cb_q, 4);
    cb_pop_front(cb_sop2, 16);
    tile_regs_wait();
    pack_n(cb_o, 4);
    tile_regs_release();

    // out = o * rsqrt(mean o^2 + 1e-5) * w * g
    cb_wait_front(cb_o, 4);
    cb_wait_front(cb_g, 4);
    rsqrt_of_sum(cb_o, 4, F_INV128, cb_sq, cb_ones, F_EPS5);
    finish(cb_r, 1);
    cb_wait_front(cb_r, 1);
    for (uint32_t c = 0; c < 4; c++) {
        tile_regs_acquire();
        load(cb_o, c, 0);
        load(cb_nw, c, 1);
        load(cb_g, c, 2);
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
    cb_pop_front(cb_g, 4);
    cb_pop_front(cb_ex, 10);
    cb_pop_front(cb_nw, 4);
}

#if M_MAX > 1
// ---- B at M rows (the math in tt/glm/kda_m.py): rows 0..M-1 of every tile, the reader's coefficient tiles
// zero outside them. CBs: 35 the commit's Ki_p 0..3 and G_p[a-1] down rows 4..7, 36 its U_p, 37 L (the reader's sums of
// this pass's log-decays), 38 G, 39 G for the writer, 40 Gi, 41 Kg (operand), 42 Kg (exact), 43 Qg, 44 Ki (transpose
// source), 45 Ki for the writer, 46 Ki^T, 47 NL 0, P 1, NL^2 2 (3 its copy), 48 R (operand), 49 R (exact), 50 R2
// (operand), 51 R2 (exact), 52 U (operand), 53 U for the writer; cb 13: v 0..3, beta down rows 4, -beta 5, CL 6
// (-beta_t at j < t < M), CP 7 (1 at j <= t < M).
constexpr uint32_t cb_pk = 35, cb_pu = 36, cb_L = 37, cb_G = 38, cb_Gw = 39, cb_Gi = 40, cb_kg = 41, cb_kgx = 42;
constexpr uint32_t cb_qg = 43, cb_ki = 44, cb_kiw = 45, cb_kit = 46, cb_nl = 47, cb_ro = 48, cb_rx = 49;
constexpr uint32_t cb_r2o = 50, cb_r2x = 51, cb_uo = 52, cb_uw = 53;

// Dest 0..3 = Dest 0..3 * Dest 4..7
inline void mul4() {
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) mul_binary_tile(t, 4 + t, t);
}
// Dest 0..3 = Dest 0..3 * Dest 4
inline void mul4_by(uint32_t d) {
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) mul_binary_tile(t, d, t);
}

void phase_b_m(uint32_t neg_a_bits) {
    // l = -5 sigmoid(neg_a (f_lo f_b_h) + dt neg_a) per row (no exp: the reader sums them over the rows), g as at one row
    cb_wait_front(cb_xlo, 8);
    cb_wait_front(cb_lw, 32);
    cb_wait_front(cb_dtn, 4);
    tile_regs_acquire();
    matmul_start(cb_xlo, cb_lw);
    for (uint32_t n = 0; n < 4; n++) {
        for (uint32_t k = 0; k < 4; k++) matmul_tiles(cb_xlo, cb_lw, k, 4 * n + k, n);
    }
    load_n(cb_dtn, 4, 4);
    SFPU_CALL(scale_add4, neg_a_bits);
    sigmoid_tile_init();
    for (uint32_t t = 0; t < 4; t++) sigmoid_tile(t);
    for (uint32_t t = 0; t < 4; t++) SFPU_CALL_AT(affine, t, F_LB, F_ZERO);
    finish(cb_dsrc, 4);
    cb_pop_front(cb_dtn, 4);
    tile_regs_acquire();
    matmul_start(cb_xlo, cb_lw);
    for (uint32_t n = 0; n < 4; n++) {
        for (uint32_t k = 0; k < 4; k++) matmul_tiles(cb_xlo, cb_lw, 4 + k, 16 + 4 * n + k, n);
    }
    sigmoid_tile_init();
    for (uint32_t t = 0; t < 4; t++) sigmoid_tile(t);
    finish(cb_g, 4);
    cb_pop_front(cb_xlo, 8);
    cb_pop_front(cb_lw, 32);

    // the commit: Kc = Ki_p * G_p[a-1] (rows j < a), S_a = S_c * Dbc_a + Kc^T U_p, into cb 19 (exact, the writer's) and
    // cb 20 (the operand)
    cb_wait_front(cb_pk, 8);
    tile_regs_acquire();
    load_n(cb_pk, 8);
    mul4();
    finish(cb_kp, 4);
    cb_pop_front(cb_pk, 8);
    cb_wait_front(cb_kp, 4);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_kp);
    transpose_init(cb_kp);
    for (uint32_t t = 0; t < 4; t++) transpose_tile(cb_kp, t, t);
    finish(cb_kc, 4);
    cb_pop_front(cb_kp, 4);
    cb_wait_front(cb_kc, 4);
    cb_wait_front(cb_pu, 4);
    cb_wait_front(cb_sx, 16);
    cb_wait_front(cb_dbc, 4);
    for (uint32_t j = 0; j < 4; j++) {
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t c0 = 2 * h;
            tile_regs_acquire();
            load(cb_sx, 4 * j + c0, 0);
            load(cb_sx, 4 * j + c0 + 1, 1);
            load(cb_dbc, j, 4);
            matmul_start(cb_kc, cb_pu);
            matmul_tiles(cb_kc, cb_pu, j, c0, 2);
            matmul_tiles(cb_kc, cb_pu, j, c0 + 1, 3);
            SFPU_CALL(kda_update2);
            tile_regs_commit();
            tile_regs_wait();
            pack_n(cb_sxo, 2);
            pack_n(cb_sop2, 2);
            tile_regs_release();
        }
    }
    cb_pop_front(cb_kc, 4);
    cb_pop_front(cb_pu, 4);
    cb_pop_front(cb_sx, 16);
    cb_pop_front(cb_dbc, 4);

    // G = exp(L), Gi = exp(-L)
    cb_wait_front(cb_L, 4);
    tile_regs_acquire();
    load_n(cb_L, 4);
    exp_tile_init();
    for (uint32_t t = 0; t < 4; t++) exp_tile(t);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_G, 4);
    pack_n(cb_Gw, 4);
    tile_regs_release();
    tile_regs_acquire();
    load_n(cb_L, 4);
    for (uint32_t t = 0; t < 4; t++) SFPU_CALL_AT(affine, t, F_MINUS1, F_ZERO);
    exp_tile_init();
    for (uint32_t t = 0; t < 4; t++) exp_tile(t);
    finish(cb_Gi, 4);
    cb_pop_front(cb_L, 4);

    // Kg = K G (operand and exact), Qg = Q G, Ki = K Gi (the transpose's and the writer's)
    cb_wait_front(cb_q, 4);
    cb_wait_front(cb_k, 4);
    cb_wait_front(cb_G, 4);
    cb_wait_front(cb_Gi, 4);
    tile_regs_acquire();
    load_n(cb_k, 4);
    load_n(cb_G, 4, 4);
    mul4();
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_kg, 4);
    pack_n(cb_kgx, 4);
    tile_regs_release();
    tile_regs_acquire();
    load_n(cb_q, 4);
    load_n(cb_G, 4, 4);
    mul4();
    finish(cb_qg, 4);
    tile_regs_acquire();
    load_n(cb_k, 4);
    load_n(cb_Gi, 4, 4);
    mul4();
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_ki, 4);
    pack_n(cb_kiw, 4);
    tile_regs_release();
    cb_pop_front(cb_q, 4);
    cb_pop_front(cb_k, 4);
    cb_pop_front(cb_G, 4);
    cb_pop_front(cb_Gi, 4);

    cb_wait_front(cb_ex, 10);  // v 0..3, beta 4, -beta 5, CL 6, CP 7
    tile_regs_acquire();       // Kb = -beta Kg (cb 7)
    cb_wait_front(cb_kgx, 4);
    load_n(cb_kgx, 4);
    load(cb_ex, 5, 4);
    mul4_by(4);
    finish(cb_kp, 4);
    cb_pop_front(cb_kgx, 4);
    tile_regs_acquire();       // beta V (cb 14, exact)
    load_n(cb_ex, 4);
    load(cb_ex, 4, 4);
    mul4_by(4);
    finish(cb_bv, 4);
    tile_regs_acquire();       // Ki as columns
    cb_wait_front(cb_ki, 4);
    reconfig_data_format_srca(cb_ki);
    transpose_init(cb_ki);
    for (uint32_t t = 0; t < 4; t++) transpose_tile(cb_ki, t, t);
    finish(cb_kit, 4);
    cb_pop_front(cb_ki, 4);

    // R = beta V + Kb S_a
    cb_wait_front(cb_bv, 4);
    cb_wait_front(cb_kp, 4);
    cb_wait_front(cb_sop2, 16);
    tile_regs_acquire();
    load_n(cb_bv, 4);
    matmul_start(cb_kp, cb_sop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) matmul_tiles(cb_kp, cb_sop2, j, 4 * j + c, c);
    }
    tile_regs_commit();
    cb_pop_front(cb_bv, 4);
    cb_pop_front(cb_kp, 4);
    tile_regs_wait();
    pack_n(cb_ro, 4);
    pack_n(cb_rx, 4);
    tile_regs_release();

    // NL = CL * (Kg Ki^T), P = CP * (Qg Ki^T)
    cb_wait_front(cb_kg, 4);
    cb_wait_front(cb_qg, 4);
    cb_wait_front(cb_kit, 4);
    tile_regs_acquire();
    matmul_start(cb_kg, cb_kit);
    for (uint32_t j = 0; j < 4; j++) {
        matmul_tiles(cb_kg, cb_kit, j, j, 0);
        matmul_tiles(cb_qg, cb_kit, j, j, 1);
    }
    load(cb_ex, 6, 2);
    copy_tile(cb_ex, 7, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 3, 1);
    finish(cb_nl, 2);
    cb_pop_front(cb_kg, 4);
    cb_pop_front(cb_kit, 4);

    cb_wait_front(cb_nl, 2);  // NL^2: tile 2 (tile 3 a copy, so the pushes add up to the CB's size)
    tile_regs_acquire();
    matmul_start(cb_nl, cb_nl);
    matmul_tiles(cb_nl, cb_nl, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_nl, 2);
    pack_reconfig_data_format(cb_nl);
    pack_tile(0, cb_nl);
    pack_tile(0, cb_nl);
    cb_push_back(cb_nl, 2);
    tile_regs_release();

    // R2 = R + NL^2 R, U = R2 + NL R2  (= (I + N)^-1 R, N = -NL)
    cb_wait_front(cb_nl, 4);
    cb_wait_front(cb_ro, 4);
    cb_wait_front(cb_rx, 4);
    tile_regs_acquire();
    load_n(cb_rx, 4);
    matmul_start(cb_nl, cb_ro);
    for (uint32_t c = 0; c < 4; c++) matmul_tiles(cb_nl, cb_ro, 2, c, c);
    tile_regs_commit();
    cb_pop_front(cb_ro, 4);
    cb_pop_front(cb_rx, 4);
    tile_regs_wait();
    pack_n(cb_r2o, 4);
    pack_n(cb_r2x, 4);
    tile_regs_release();
    cb_wait_front(cb_r2o, 4);
    cb_wait_front(cb_r2x, 4);
    tile_regs_acquire();
    load_n(cb_r2x, 4);
    matmul_start(cb_nl, cb_r2o);
    for (uint32_t c = 0; c < 4; c++) matmul_tiles(cb_nl, cb_r2o, 0, c, c);
    tile_regs_commit();
    cb_pop_front(cb_r2o, 4);
    cb_pop_front(cb_r2x, 4);
    tile_regs_wait();
    pack_n(cb_uo, 4);
    pack_n(cb_uw, 4);
    tile_regs_release();

    // O = Qg S_a + P U
    cb_wait_front(cb_uo, 4);
    tile_regs_acquire();
    matmul_start(cb_qg, cb_sop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) matmul_tiles(cb_qg, cb_sop2, j, 4 * j + c, c);
    }
    matmul_start(cb_nl, cb_uo);
    for (uint32_t c = 0; c < 4; c++) matmul_tiles(cb_nl, cb_uo, 1, c, c);
    tile_regs_commit();
    cb_pop_front(cb_qg, 4);
    cb_pop_front(cb_sop2, 16);
    cb_pop_front(cb_nl, 4);
    cb_pop_front(cb_uo, 4);
    tile_regs_wait();
    pack_n(cb_o, 4);
    tile_regs_release();

    // out = o * rsqrt(mean o^2 + 1e-5) * w * g, row by row
    cb_wait_front(cb_o, 4);
    cb_wait_front(cb_g, 4);
    cb_wait_front(cb_nw, 4);
    rsqrt_of_sum(cb_o, 4, F_INV128, cb_sq, cb_ones, F_EPS5);
    finish(cb_r, 1);
    cb_wait_front(cb_r, 1);
    for (uint32_t c = 0; c < 4; c++) {
        tile_regs_acquire();
        load(cb_o, c, 0);
        load(cb_nw, c, 1);
        load(cb_g, c, 2);
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
    cb_pop_front(cb_g, 4);
    cb_pop_front(cb_ex, 10);
    cb_pop_front(cb_nw, 4);
}
#endif

void phase_c(uint32_t count, uint32_t batch) {
    for (uint32_t n = 0; n < count; n++) {
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        gemv_ring(KC / get_compile_time_arg_val(2), 0, batch);
        finish(cb_out, 1);
    }
}

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t phases = get_compile_time_arg_val(1);
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    uint32_t a_tiles = 0;
    for (uint32_t u = 0; u < n_units; u++) a_tiles += get_arg_val<uint32_t>(2 + 3 * u);
    const uint32_t head = get_arg_val<uint32_t>(1 + 3 * n_units);
    const uint32_t neg_a_bits = get_arg_val<uint32_t>(2 + 3 * n_units);
    const uint32_t c_count = get_arg_val<uint32_t>(3 + 3 * n_units);
    if (a_tiles == 0 && head == IDLE && c_count == 0) return;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_raw);
    cb_wait_front(cb_ones, 1);
    if ((phases & 1) && a_tiles) phase_a(n_units, batch);
#if M_MAX > 1
    if ((phases & 2) && head != IDLE) phase_b_m(neg_a_bits);
#else
    if ((phases & 2) && head != IDLE) phase_b(neg_a_bits);
#endif
    if ((phases & 4) && c_count) phase_c(c_count, batch);
}
