// Fused QSA attention layer, program 1 compute (the phases are described in fattn1_reader.cpp).
// Runtime args: the H1 item, the H2 tile, P's first tile and count, N's role and index. Compile args: batch.
// M rows (-DM_MAX > 1, fattn1_reader.cpp): the mix at M rows (fused/hc_cmp.h compute_m, M and mp from the reader's
// cb 2 tile 2); P and N as at one row, on tiles with M rows (the norms, RoPE and the scores are row-wise); the q heads
// leave as bf16 (program 2's q^T is bf16). Preprocessed out at M_MAX = 1.
#include <cstdint>
#include "fused/cmp.h"
#include "fused/hc_cmp.h"
#include "api/compute/eltwise_unary/relu.h"

#ifndef M_MAX
#define M_MAX 1
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// Dest 0, 1 = x (dims 0..31, 32..63), Dest 2 = c, Dest 3 = s: Dest 0 = (x0 c - x1 s) k, Dest 1 = (x0 s + x1 c) k
inline void rope2(uint32_t k_bits) {
    const sfpi::vFloat k = Converter::as_float(k_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat x0 = sfpi::dst_reg[0], x1 = sfpi::dst_reg[32], c = sfpi::dst_reg[64], s = sfpi::dst_reg[96];
        sfpi::dst_reg[0] = (x0 * c - x1 * s) * k;
        sfpi::dst_reg[32] = (x0 * s + x1 * c) * k;
        sfpi::dst_reg++;
    }
}

// Dest 0 = Dest 0 * Dest 1 * Dest 2 * k
inline void mul2k(uint32_t k_bits) {
    const sfpi::vFloat k = Converter::as_float(k_bits);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[32], c = sfpi::dst_reg[64];
        sfpi::dst_reg[0] = a * b * c * k;
        sfpi::dst_reg++;
    }
}

// Dest 0, 1 = Dest 0, 1 * Dest 4 (r) * Dest 2, 3 (w)
inline void norm2() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat q = sfpi::dst_reg[4 * 32];
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * q * sfpi::dst_reg[64];
        sfpi::dst_reg[32] = sfpi::dst_reg[32] * q * sfpi::dst_reg[96];
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_aout = 21;
constexpr uint32_t cb_nin = 23, cb_nw = 24, cb_nr = 25, cb_nsq = 26, cb_nr1 = 27, cb_niq = 28, cb_npk = 29;
constexpr uint32_t cb_nout = 30, cb_nout16 = 31, cb_nv = 32, cb_nmean = 33, cb_meta = 22;
constexpr uint32_t KD = 80;
constexpr uint32_t R_SCORE = 0, R_QHEAD = 1, R_KVHEAD = 2, R_POOL = 3, IDLE = 0xffffffff;
constexpr uint32_t F_INV256 = 0x3b800000;      // 1 / 256
constexpr uint32_t F_RSQRT256 = 0x3d800000;    // 1 / 16

// x * rsqrt(mean x^2 + eps) * w for n row-0 tiles of src (exact), RoPE on tiles 0 and 1 (c, s: tiles 0, 1 of cb_nr),
// everything times k; into out. Pops src (n), and unless keep, w (n) and the rope tiles.
void norm_rope(uint32_t src, uint32_t n, uint32_t mean_bits, uint32_t k_bits, uint32_t out, bool keep = false) {
    cb_wait_front(src, n);
    cb_wait_front(cb_nw, n);
    cb_wait_front(cb_nr, 2);
    rsqrt_of_sum(src, n, mean_bits, cb_nsq, cb_ones);
    finish(cb_nr1, 1);
    cb_wait_front(cb_nr1, 1);
    tile_regs_acquire();
    load_n(src, 2);
    load_n(cb_nw, 2, 2);
    load(cb_nr1, 0, 4);
    SFPU_CALL(norm2);
    load_n(cb_nr, 2, 2);
    SFPU_CALL(rope2, k_bits);
    finish(out, 2);
    for (uint32_t t = 2; t < n; t++) {
        tile_regs_acquire();
        load(src, t, 0);
        load(cb_nw, t, 1);
        load(cb_nr1, 0, 2);
        SFPU_CALL(mul2k, k_bits);
        finish(out, 1);
    }
    cb_pop_front(src, n);
    cb_pop_front(cb_nr1, 1);
    if (!keep) {
        cb_pop_front(cb_nw, n);
        cb_pop_front(cb_nr, 2);
    }
}

// sum_h relu(q_h k_p^T) / sqrt(128) for the 32 pools p of cb_npk's 4 fp32 tiles (row p: pool p's key, read
// transposed), the q heads' normed, rotated tiles in cb_niq (16); pops both (unless keep: cb_niq for another block),
// the scores (row 0) into cb_nout
void pool_scores(bool keep = false) {
    cb_wait_front(cb_niq, 16);
    cb_wait_front(cb_npk, 4);
    tile_regs_acquire();
    reconfig_data_format<SrcOrder::Reverse>(cb_niq, cb_npk);
    matmul_init(cb_niq, cb_npk, 1);
    for (uint32_t h = 0; h < 4; h++) {
        for (uint32_t j = 0; j < 4; j++) {
            matmul_tiles(cb_niq, cb_npk, 4 * h + j, j, h);
        }
    }
    relu_tile_init();
    for (uint32_t h = 0; h < 4; h++) {
        relu_tile(h);
    }
    SFPU_CALL(sum4);
    SFPU_CALL(affine, F_RSQRT128, F_ZERO);
    if (!keep) {
        cb_pop_front(cb_niq, 16);
    }
    cb_pop_front(cb_npk, 4);
    finish(cb_nout, 1);
}

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t h_item = get_arg_val<uint32_t>(0), h_tile = get_arg_val<uint32_t>(1);
    const uint32_t p_first = get_arg_val<uint32_t>(2), p_count = get_arg_val<uint32_t>(3);
    const uint32_t role = get_arg_val<uint32_t>(4);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_aout);
#if M_MAX > 1
    cb_wait_front(cb_ones, 3);  // the layout's rows: the reader's words in cb 2, tile 2
    const uint32_t mp = read_tile_value(cb_ones, 2, 3);
    Weights w{cb_w, batch};
    hc::compute_m(h_item, h_tile, w, mp);
#else
    cb_wait_front(cb_ones, 1);
    Weights w{cb_w, batch};
    hc::compute(h_item, h_tile, w);
#endif

    // ---- P: sigmoid on the gate tiles (8..15 of each q head's 16)
    if (p_count) {
        cb_wait_front(cb_x, KD);
        for (uint32_t n = p_first; n < p_first + p_count; n++) {
            tile_regs_acquire();
            matmul_start(cb_x, cb_w);
            w.gemv(cb_x, KD, 0);
            if (n < 384 && n % 16 >= 8) {
                sigmoid_tile_init();
                sigmoid_tile(0);
            }
            finish(cb_aout, 1);
        }
        w.end();
        cb_pop_front(cb_x, KD);
    }

    // ---- N
    if (role == R_SCORE) {
        cb_wait_front(cb_meta, 1);  // the blocks to score (the reader's: 2 once the token is past block index + 64)
        const uint32_t nblk = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
        for (uint32_t h = 0; h < 4; h++) {  // the indexer's q heads, sharing the norm weights and the rope tiles
            norm_rope(cb_nin, 4, F_INV128, F_ONE, cb_niq, h < 3);
        }
        for (uint32_t b = 0; b < nblk; b++) {
            pool_scores(b + 1 < nblk);
        }
    } else if (role == R_QHEAD) {
#if M_MAX > 1
        norm_rope(cb_nin, 8, F_INV256, F_RSQRT256, cb_nout16);
#else
        norm_rope(cb_nin, 8, F_INV256, F_RSQRT256, cb_nout);
#endif
    } else if (role == R_KVHEAD) {
        norm_rope(cb_nin, 8, F_INV256, F_ONE, cb_nout16);
        cb_wait_front(cb_nv, 8);
        tile_regs_acquire();
        load_n(cb_nv, 8);
        cb_pop_front(cb_nv, 8);
        finish(cb_nout16, 8);
    } else if (role == R_POOL) {
        for (uint32_t t = 0; t < 4; t++) {  // the mean of the block's 4 raw keys
            cb_wait_front(cb_nin, 4);
            tile_regs_acquire();
            load_n(cb_nin, 4);
            SFPU_CALL(sum4);
            SFPU_CALL(affine, F_QUARTER, F_ZERO);
            cb_pop_front(cb_nin, 4);
            finish(cb_nmean, 1);
        }
        // the new pooled key (row 0 of 4 tiles) to the writer, kept in cb_npk for the block's own score
        norm_rope(cb_nmean, 4, F_INV128, F_ONE, cb_npk);
        cb_wait_front(cb_npk, 4);
        tile_regs_acquire();
        load_n(cb_npk, 4);
        finish(cb_nout, 4);
        // the score of the pool the token closes: the score cores read the pooled cache before this key is in it,
        // so the pool role scores it (element (0, 0) of the score tile: the key in row 0 stands for pool 0)
        for (uint32_t h = 0; h < 4; h++) {
            norm_rope(cb_nin, 4, F_INV128, F_ONE, cb_niq, h < 3);
        }
        pool_scores();
    }
}
