// Fused DeltaNet layer compute (G3 step 4; the phases are listed in flayer_reader.cpp). The math of each phase is
// the standalone kernels' (fhc_compute.cpp, fgdn_compute.cpp), on the layer program's own circular buffers; the
// hyper-connection mix comes from fused/hc_cmp.h, ssm_out + the combine and the FFN half from
// fused/tail_cmp.h, the card's experts between R1 and R2 (CARD_EXPERTS) from fused/dexp_cmp.h.
// Runtime args: the H1 item, the H2 tile, A's unit count and units, B's head, C's tile, the R1 item, R2's tile, this
// core's index, the core count.
// Compile args: batch.
// M rows (-DM_MAX > 1, flayer_reader.cpp): M and the layout's rows mp from the reader (cb 2, tile 2); A's conv window
// from the writer (cb 22: y and the three shifted tiles), B's chunked delta rule with the previous pass's commit
// (phase_b_m). The M-row code is preprocessed out at M_MAX = 1 (the decode build compiles exactly the one-row source).
#include <cstdint>
#include "fused/cmp.h"
#include "fused/hc_cmp.h"
#include "fused/tail_cmp.h"
#ifdef CARD_EXPERTS
#include "fused/dexp_cmp.h"
#endif

using namespace fused;

// zones for tt-metal's device profiler (TT_MOEOFFLOAD_ZONES, with TT_MOEOFFLOAD_TTNN=tracy and TT_METAL_DEVICE_PROFILER=1); nothing
// otherwise, so the normal builds compile exactly the source without them (2026-11-08)
#ifdef TT_MOEOFFLOAD_ZONES
#include "api/debug/kernel_profiler.hpp"
#define ZONE(n) DeviceZoneScopedN(n)
#else
#define ZONE(n)
#endif

#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2;
constexpr uint32_t cb_ac = 16, cb_araw = 17, cb_axc = 18, cb_asq = 19, cb_ar = 20, cb_aout = 21, cb_asnew = 22;
#if M_MAX > 1
constexpr uint32_t cb_awin = 22;  // M rows: the conv's window (the writer's)
constexpr uint32_t cb_bsw = 63;   // M rows: the rows' U for the writer (the next pass's commit)
constexpr uint32_t cb_bpend = 59;  // M rows: the previous pass's kept rows of k (tiles 0..3) and U (4..7)
#endif
constexpr uint32_t cb_bsx = 23, cb_bsop = 24, cb_bnw = 25, cb_bq = 26, cb_bk = 27, cb_bex = 28, cb_bkp = 29;
constexpr uint32_t cb_bbv = 30, cb_bkc = 31, cb_bdelta = 32, cb_bsxo = 33, cb_bsop2 = 34, cb_bo = 35, cb_bsq = 36;
constexpr uint32_t cb_br = 37, cb_bhout = 38, cb_bhchk = 39;
constexpr uint32_t cb_cx = 40, cb_cy = 41, cb_cxs = 42, cb_cw = 43, cb_cout32 = 44, cb_cout16 = 45;
constexpr uint32_t cb_rw = 46, cb_rlog = 47, cb_rraw = 48, cb_rh16 = 49, cb_rh32 = 50;
constexpr uint32_t cb_sgl = 51, cb_srt = 52, cb_sxs = 53, cb_sw = 54, cb_sy = 55, cb_sout32 = 56, cb_sout16 = 57;
constexpr uint32_t D = 2560, KD = 80, KS = 80, KHU = 10, KC = 192, KR2 = 20;
constexpr uint32_t T_V = 0, T_Q = 1, T_K = 2, T_Z = 3, T_A = 4, T_B = 5;
constexpr uint32_t IDLE = 0xffffffff, N_ROUTER = 17;

// A: the input projection with its epilogues (fgdn_compute.cpp's phase A, the epilogue per tile)
void phase_a(uint32_t n_units, uint32_t ua, Weights& w) {
    {
        ZONE("A_wait_x");
        cb_wait_front(cb_x, KD);
    }
    for (uint32_t u = 0; u < n_units; u++) {
        const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
        const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
        const bool sig = type == T_Z || type == T_B, head = type == T_Q || type == T_K;
        for (uint32_t t = 0; t < count; t++) {
            ZONE("A_tile");
            tile_regs_acquire();
            {
                ZONE("A_gemv");
                matmul_start(cb_x, cb_w);
                w.gemv(cb_x, KD, 0);
            }
            if (sig) {
                sigmoid_tile_init();
                sigmoid_tile(0);
                finish(cb_aout, 1);
                continue;
            }
#if M_MAX > 1
            {
                if (type == T_A) {  // y stays in Dest 0
                    cb_wait_front(cb_ac, 7);
                    load(cb_ac, 0, 1);  // dt
                    load(cb_ac, 1, 2);  // neg_a
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    exp_tile_init();
                    exp_tile(0);
                    SFPU_CALL(affine, F_ONE, F_ONE);
                    log_tile_init();
                    log_tile(0);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 2, 0);
                    exp_tile_init();
                    exp_tile(0);
                    cb_pop_front(cb_ac, 7);
                    finish(cb_aout, 1);
                    continue;
                }
                finish(cb_araw, 1);  // y to the writer, which builds the window
                cb_wait_front(cb_awin, 4);
                cb_wait_front(cb_ac, 7);
                tile_regs_acquire();
                load_n(cb_awin, 4);  // y, then the window's tiles (rows m: inputs m, m + 1, m + 2 of history + y)
                load(cb_ac, 3, 4);   // the taps
                for (uint32_t j = 1; j < 4; j++) {
                    copy_tile(cb_ac, 3 + j, 4 + j);
                }
                SFPU_CALL(conv4);
                silu_tile_init();
                silu_tile(1);
                cb_pop_front(cb_awin, 4);
                cb_pop_front(cb_ac, 7);
                tile_regs_commit();
                tile_regs_wait();
                pack_n(head ? cb_axc : cb_aout, 1, 1);
                tile_regs_release();
                continue;
            }
#endif
            finish(cb_araw, 1);
            cb_wait_front(cb_araw, 1);
            cb_wait_front(cb_ac, 7);
            tile_regs_acquire();
            if (type == T_A) {
                load(cb_araw, 0, 0);
                load(cb_ac, 0, 1);  // dt
                load(cb_ac, 1, 2);  // neg_a
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
                cb_pop_front(cb_araw, 1);
                cb_pop_front(cb_ac, 7);
                finish(cb_aout, 1);
                continue;
            }
            load(cb_araw, 0, 0);
            load_n(cb_ac, 7, 1);
            SFPU_CALL(conv4);
            silu_tile_init();
            silu_tile(1);
            cb_pop_front(cb_araw, 1);
            cb_pop_front(cb_ac, 7);
            tile_regs_commit();
            tile_regs_wait();
            pack_n(head ? cb_axc : cb_aout, 1, 1);
            cb_reserve_back(cb_asnew, 3);
            pack_reconfig_data_format(cb_asnew);
            pack_tile(2, cb_asnew);  // new s0 = old s1
            pack_tile(3, cb_asnew);  // new s1 = old s2
            pack_tile(0, cb_asnew);  // new s2 = y
            cb_push_back(cb_asnew, 3);
            tile_regs_release();
        }
        if (!head) {
            continue;
        }
        // the head's L2 norm
        ZONE("A_norm");
        cb_wait_front(cb_axc, 4);
        rsqrt_of_sum(cb_axc, 4, F_ONE, cb_asq, cb_ones);
        if (type == T_Q) {
            SFPU_CALL(affine, F_RSQRT128, F_ZERO);
        }
        finish(cb_ar, 1);
        cb_wait_front(cb_ar, 1);
        tile_regs_acquire();
        load(cb_ar, 0, 4);
        load_n(cb_axc, 4);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < 4; t++) {
            mul_binary_tile(t, 4, t);
        }
        cb_pop_front(cb_ar, 1);
        cb_pop_front(cb_axc, 4);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < 4; t++) {  // one tile a push: across passes a 4-tile push could straddle the end
            pack_n(cb_aout, 1, t);
        }
        tile_regs_release();
    }
    {
        ZONE("A_end");
        w.end();
    }
    cb_pop_front(cb_x, KD);
}

#if M_MAX == 1
// B: the delta rule and the gated norm for one value head (fgdn_compute.cpp's phase B)
void phase_b() {
    ZONE("B");
    constexpr uint32_t sop = cb_bsop;
    constexpr bool last = true;
    cb_wait_front(cb_bq, 4);
    cb_wait_front(cb_bk, 4);
    cb_wait_front(cb_bex, 10);  // v 0..3, sigmoid(z) 4..7, beta 8, decay 9
    cb_wait_front(cb_bnw, 4);
    const uint32_t d_bits = read_tile_value(cb_bex, 9, 0);

    tile_regs_acquire();  // k' = -b d k
    load_n(cb_bk, 4);
    load(cb_bex, 8, 4);
    copy_tile(cb_bex, 9, 5);
    SFPU_CALL(neg_scale4);
    finish(cb_bkp, 4);

    tile_regs_acquire();  // b v
    load_n(cb_bex, 4);
    copy_tile(cb_bex, 8, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) {
        mul_binary_tile(t, 4, t);
    }
    finish(cb_bbv, 4);

    tile_regs_acquire();  // k as columns
    reconfig_data_format_srca(cb_bk);
    transpose_init(cb_bk);
    for (uint32_t t = 0; t < 4; t++) {
        transpose_tile(cb_bk, t, t);
    }
    cb_pop_front(cb_bk, 4);
    finish(cb_bkc, 4);

    cb_wait_front(cb_bbv, 4);  // delta = b v + k'^T S
    cb_wait_front(cb_bkp, 4);
    cb_wait_front(sop, 16);
    tile_regs_acquire();
    load_n(cb_bbv, 4);
    matmul_start(cb_bkp, sop);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_bkp, sop, j, 4 * j + c, c);
        }
    }
    cb_pop_front(cb_bbv, 4);
    cb_pop_front(cb_bkp, 4);
    cb_pop_front(sop, 16);
    finish(cb_bdelta, 4);

    cb_wait_front(cb_bdelta, 4);  // S = d S + k delta^T, into both copies
    cb_wait_front(cb_bkc, 4);
    cb_wait_front(cb_bsx, 16);
    for (uint32_t j = 0; j < 4; j++) {
        tile_regs_acquire();
        load(cb_bsx, 4 * j, 0);
        for (uint32_t c = 1; c < 4; c++) {
            copy_tile(cb_bsx, 4 * j + c, c);
        }
        matmul_start(cb_bkc, cb_bdelta);
        for (uint32_t c = 0; c < 4; c++) {
            matmul_tiles(cb_bkc, cb_bdelta, j, c, 4 + c);
        }
        SFPU_CALL(state_update4, d_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_bsxo, 4);
        pack_n(cb_bsop2, 4);
        tile_regs_release();
    }
    cb_pop_front(cb_bdelta, 4);
    cb_pop_front(cb_bkc, 4);
    cb_pop_front(cb_bsx, 16);

    cb_wait_front(cb_bsop2, 16);  // o = q^T S
    tile_regs_acquire();
    matmul_start(cb_bq, cb_bsop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_bq, cb_bsop2, j, 4 * j + c, c);
        }
    }
    cb_pop_front(cb_bq, 4);
    if (last) {
        cb_pop_front(cb_bsop2, 16);
    }
    finish(cb_bo, 4);

    cb_wait_front(cb_bo, 4);  // out = o * rsqrt(mean o^2 + eps) * w * sigmoid(z)
    rsqrt_of_sum(cb_bo, 4, F_INV128, cb_bsq, cb_ones);
    finish(cb_br, 1);
    cb_wait_front(cb_br, 1);
    for (uint32_t c = 0; c < 4; c++) {
        tile_regs_acquire();
        load(cb_bo, c, 0);
        load(cb_bnw, c, 1);
        load(cb_bex, 4 + c, 2);
        load(cb_br, 0, 3);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(0, 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_bhout, 1);
        pack_n(cb_bhchk, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_br, 1);
    cb_pop_front(cb_bo, 4);
    cb_pop_front(cb_bex, 10);
    if (last) {
        cb_pop_front(cb_bnw, 4);
    }
}
#else
// B at M rows for one value head: the delta rule over the pass's rows t < M in the chunked (WY) form, from the state
// the previous pass left with that pass's kept rows committed onto it first (flayer_reader.cpp has the inputs):
//   commit  S_a = gamma_a S + K_p^T diag(c) U_p: the one-row update block with K_p's columns as k, c U_p as delta
//   rows    R = beta V - diag(beta gamma) K S_a,  NL = CL * (K K^T),  U = (I + NL)(I + NL^2) R = (I - NL)^-1 R (NL is
//           strictly lower and M <= 4, so NL^4 = 0),  O = diag(gamma) Q S_a + (CP * (Q K^T)) U,  then the gated norm
// U (cb 63) and S_a (cb 33) go to the writer, the next pass's commit inputs. A matmul operand comes from a Default-mode
// CB, a term added exactly from an fp32 one, so R and R2 are packed twice.
void phase_b_m() {
    ZONE("Bm");
    cb_wait_front(cb_bpend, 8);
    cb_wait_front(cb_bnw, 5);
    const uint32_t ga_bits = read_tile_value(cb_bnw, 4, 512);

    tile_regs_acquire();  // the kept rows' k as columns
    reconfig_data_format_srca(cb_bpend);
    transpose_init(cb_bpend);
    for (uint32_t t = 0; t < 4; t++) {
        transpose_tile(cb_bpend, t, t);
    }
    finish(cb_bkc, 4);

    tile_regs_acquire();  // c U_p
    load(cb_bpend, 4, 0);
    for (uint32_t t = 1; t < 4; t++) {
        copy_tile(cb_bpend, 4 + t, t);
    }
    load(cb_bnw, 4, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) {
        mul_binary_tile(t, 4, t);
    }
    cb_pop_front(cb_bpend, 8);
    finish(cb_bdelta, 4);

    cb_wait_front(cb_bdelta, 4);  // S_a = gamma_a S + K_p^T (c U_p), into both copies
    cb_wait_front(cb_bkc, 4);
    cb_wait_front(cb_bsx, 16);
    for (uint32_t j = 0; j < 4; j++) {
        tile_regs_acquire();
        load(cb_bsx, 4 * j, 0);
        for (uint32_t c = 1; c < 4; c++) {
            copy_tile(cb_bsx, 4 * j + c, c);
        }
        matmul_start(cb_bkc, cb_bdelta);
        for (uint32_t c = 0; c < 4; c++) {
            matmul_tiles(cb_bkc, cb_bdelta, j, c, 4 + c);
        }
        SFPU_CALL(state_update4, ga_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_bsxo, 4);
        pack_n(cb_bsop2, 4);
        tile_regs_release();
    }
    cb_pop_front(cb_bdelta, 4);
    cb_pop_front(cb_bkc, 4);
    cb_pop_front(cb_bsx, 16);

    cb_wait_front(cb_bq, 4);
    cb_wait_front(cb_bk, 4);
    cb_wait_front(cb_bex, 12);  // v 0..3, sigmoid(z) 4..7, beta 8, gamma 9, CL 10, CP 11 (rows t < M)

    tile_regs_acquire();  // k' = -beta gamma k
    load_n(cb_bk, 4);
    load(cb_bex, 8, 4);
    copy_tile(cb_bex, 9, 5);
    SFPU_CALL(neg_scale4);
    finish(cb_bkp, 4);

    tile_regs_acquire();  // beta v
    load_n(cb_bex, 4);
    copy_tile(cb_bex, 8, 4);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; t++) {
        mul_binary_tile(t, 4, t);
    }
    finish(cb_bbv, 4);

    tile_regs_acquire();  // k as columns
    reconfig_data_format_srca(cb_bk);
    transpose_init(cb_bk);
    for (uint32_t t = 0; t < 4; t++) {
        transpose_tile(cb_bk, t, t);
    }
    finish(cb_bkc, 4);

    cb_wait_front(cb_bbv, 4);  // R = beta v + k' S_a: cb 32 (an operand), cb 30 (exact)
    cb_wait_front(cb_bkp, 4);
    cb_wait_front(cb_bsop2, 16);
    tile_regs_acquire();
    load_n(cb_bbv, 4);
    matmul_start(cb_bkp, cb_bsop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_bkp, cb_bsop2, j, 4 * j + c, c);
        }
    }
    cb_pop_front(cb_bbv, 4);
    cb_pop_front(cb_bkp, 4);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_bdelta, 4);
    pack_n(cb_bbv, 4);
    tile_regs_release();

    cb_wait_front(cb_bkc, 4);  // NL = CL * (K K^T), PQK = CP * (Q K^T): cb 36 tiles 0, 1
    tile_regs_acquire();
    matmul_start(cb_bk, cb_bkc);
    for (uint32_t j = 0; j < 4; j++) {
        matmul_tiles(cb_bk, cb_bkc, j, j, 0);
        matmul_tiles(cb_bq, cb_bkc, j, j, 1);
    }
    load(cb_bex, 10, 2);
    copy_tile(cb_bex, 11, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    mul_binary_tile(1, 3, 1);
    cb_pop_front(cb_bk, 4);
    cb_pop_front(cb_bkc, 4);
    finish(cb_bsq, 2);

    cb_wait_front(cb_bsq, 2);  // NL^2: tile 2 (tile 3 a copy, so cb 36's pushes add up to its size)
    tile_regs_acquire();
    matmul_start(cb_bsq, cb_bsq);
    matmul_tiles(cb_bsq, cb_bsq, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_bsq, 2);
    pack_reconfig_data_format(cb_bsq);
    pack_tile(0, cb_bsq);
    pack_tile(0, cb_bsq);
    cb_push_back(cb_bsq, 2);
    tile_regs_release();

    cb_wait_front(cb_bsq, 4);  // R2 = R + NL^2 R: cb 29 (an operand), cb 35 (exact)
    cb_wait_front(cb_bdelta, 4);
    cb_wait_front(cb_bbv, 4);
    tile_regs_acquire();
    load_n(cb_bbv, 4);
    matmul_start(cb_bsq, cb_bdelta);
    for (uint32_t c = 0; c < 4; c++) {
        matmul_tiles(cb_bsq, cb_bdelta, 2, c, c);
    }
    cb_pop_front(cb_bbv, 4);
    cb_pop_front(cb_bdelta, 4);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_bkp, 4);
    pack_n(cb_bo, 4);
    tile_regs_release();

    cb_wait_front(cb_bkp, 4);  // U = R2 + NL R2: cb 32 (an operand), cb 63 (the writer's)
    cb_wait_front(cb_bo, 4);
    tile_regs_acquire();
    load_n(cb_bo, 4);
    matmul_start(cb_bsq, cb_bkp);
    for (uint32_t c = 0; c < 4; c++) {
        matmul_tiles(cb_bsq, cb_bkp, 0, c, c);
    }
    cb_pop_front(cb_bo, 4);
    cb_pop_front(cb_bkp, 4);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_bdelta, 4);
    pack_n(cb_bsw, 4);
    tile_regs_release();

    cb_wait_front(cb_bdelta, 4);  // o = gamma (q S_a) + PQK U
    tile_regs_acquire();
    matmul_start(cb_bq, cb_bsop2);
    for (uint32_t c = 0; c < 4; c++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_bq, cb_bsop2, j, 4 * j + c, c);
        }
    }
    load(cb_bex, 9, 4);
    mul_binary_tile_init();
    for (uint32_t c = 0; c < 4; c++) {
        mul_binary_tile(c, 4, c);
    }
    matmul_start(cb_bsq, cb_bdelta);
    for (uint32_t c = 0; c < 4; c++) {
        matmul_tiles(cb_bsq, cb_bdelta, 1, c, c);
    }
    cb_pop_front(cb_bq, 4);
    cb_pop_front(cb_bsop2, 16);
    cb_pop_front(cb_bsq, 4);
    cb_pop_front(cb_bdelta, 4);
    finish(cb_bo, 4);

    cb_wait_front(cb_bo, 4);  // out = o * rsqrt(mean o^2 + eps) * w * sigmoid(z), row by row
    rsqrt_of_sum(cb_bo, 4, F_INV128, cb_bsq, cb_ones);
    finish(cb_br, 1);
    cb_wait_front(cb_br, 1);
    for (uint32_t c = 0; c < 4; c++) {
        tile_regs_acquire();
        load(cb_bo, c, 0);
        load(cb_bnw, c, 1);
        load(cb_bex, 4 + c, 2);
        load(cb_br, 0, 3);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(0, 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_bhout, 1);
        pack_n(cb_bhchk, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_br, 1);
    cb_pop_front(cb_bo, 4);
    cb_pop_front(cb_bex, 12);
    cb_pop_front(cb_bnw, 5);
}
#endif

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t layers = get_compile_time_arg_val(1);
    uint32_t i = 0;
    const uint32_t h_item = get_arg_val<uint32_t>(i++), h_tile = get_arg_val<uint32_t>(i++);
    const uint32_t n_units = get_arg_val<uint32_t>(i++), ua = i;
    i += 3 * n_units;
    const uint32_t head = get_arg_val<uint32_t>(i++), c_tile = get_arg_val<uint32_t>(i++);
    const uint32_t r_item = get_arg_val<uint32_t>(i++), s_tile = get_arg_val<uint32_t>(i++);
    const uint32_t me = get_arg_val<uint32_t>(i++), cores = get_arg_val<uint32_t>(i++);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_aout);
#if M_MAX > 1
    cb_wait_front(cb_ones, 3);  // the step's M and the layout's rows: the reader's words in cb 2, tile 2
#ifdef CARD_EXPERTS
    constexpr uint32_t w0 = 256;  // (rows 0..3 of tile 2: the card experts' Sel)
#else
    constexpr uint32_t w0 = 0;
#endif
    const uint32_t M = read_tile_value(cb_ones, 2, w0), mp = read_tile_value(cb_ones, 2, w0 + 3);
#else
    cb_wait_front(cb_ones, 1);
#endif
    Weights w{cb_w, batch}, rw{cb_rw, batch};

    for (uint32_t layer = 0; layer < layers; layer++) {
#if M_MAX > 1
        hc::compute_m(h_item, h_tile, w, mp);
        if (n_units) {
            phase_a(n_units, ua, w);
        }
        if (head != IDLE) {
            phase_b_m();
        }
        tail::oproj(c_tile, batch);
        hc::compute_m(h_item, h_tile, w, mp);
#ifdef CARD_EXPERTS
        tail::ffn_r1(r_item, w, rw);
        const uint32_t n_card = dexp::compute_m(me, cores);
        tail::ffn_r2_m(s_tile, w, n_card);
#else
        tail::ffn(r_item, s_tile, w, rw);
#endif
#else
        hc::compute(h_item, h_tile, w);
        if (n_units) {
            phase_a(n_units, ua, w);
        }
        if (head != IDLE) {
            phase_b();
        }
        tail::oproj(c_tile, batch);
        hc::compute(h_item, h_tile, w);
#ifdef CARD_EXPERTS
        tail::ffn_r1(r_item, w, rw);
        const uint32_t n_card = dexp::compute(me, cores);
        tail::ffn_r2(s_tile, w, n_card);
#else
        tail::ffn(r_item, s_tile, w, rw);
#endif
#endif
    }
}
