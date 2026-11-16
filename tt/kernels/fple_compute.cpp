// Fused PLE block compute (the math and phases are in fple_reader.cpp). A row-0 tile's 32 values sum through one matmul
// with the all-ones tile J (row 0 of v J holds the sum in every column); a tile's 1024 through two (J (T J)).
// Runtime args: P1's first tile and count, P2's stream, P3's tile count. Compile args: batch.
#include <cstdint>
#include "fused/cmp.h"
#include "api/compute/copy_dest_values.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// Dest 4 = Dest 0 * Dest 1 * Dest 2 * Dest 3 (k X nk nq), Dest 5 = Dest 0^2
inline void ple_partials() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat k = sfpi::dst_reg[0], x = sfpi::dst_reg[32], a = sfpi::dst_reg[64], b = sfpi::dst_reg[96];
        sfpi::dst_reg[4 * 32] = k * x * a * b;
        sfpi::dst_reg[5 * 32] = k * k;
        sfpi::dst_reg++;
    }
}

// Dest 1 = Dest 0^2
inline void ple_sq() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[32] = v * v;
        sfpi::dst_reg++;
    }
}

// Dest 0 = sum A, Dest 1 = rsqrt(B / 2560 + eps), Dest 3 = rsqrt(X^2 / 2560 + eps): Dest 0 = s = A * Dest 1 * Dest 3
// / sqrt(2560), Dest 4 = max(|s|, 1e-6)
inline void ple_s() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat s = sfpi::dst_reg[0] * sfpi::dst_reg[32] * sfpi::dst_reg[96] * 0.019764235f;
        sfpi::dst_reg[0] = s;
        sfpi::vFloat m = sfpi::setsgn(s, 0);
        v_if(m < 1e-6f) { m = 1e-6f; }
        v_endif;
        sfpi::dst_reg[4 * 32] = m;
        sfpi::dst_reg++;
    }
}

// Dest 0 = s, Dest 4 = max(|s|, 1e-6), Dest 5 = rsqrt(Dest 4): Dest 0 = sign(s) sqrt(m) = sign(s) m rsqrt(m)
inline void ple_signed_sqrt() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat s = sfpi::dst_reg[0];
        sfpi::vFloat out = sfpi::dst_reg[4 * 32] * sfpi::dst_reg[5 * 32];
        v_if(s < 0.0f) { out = -out; }
        v_endif;
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

// Dest 0 = gate, Dest 2 = sum V: Dest 6 = gate^2 V / 2560 + eps
inline void ple_gnorm() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat g = sfpi::dst_reg[0];
        sfpi::dst_reg[6 * 32] = g * g * sfpi::dst_reg[64] * 0.000390625f + 1e-6f;
        sfpi::dst_reg++;
    }
}

// Dest 7 = Dest 0 * Dest 6
inline void ple_g() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        sfpi::dst_reg[7 * 32] = sfpi::dst_reg[0] * sfpi::dst_reg[6 * 32];
        sfpi::dst_reg++;
    }
}

// Dest 0 = v, 1 = gate, 2 = g, 3 = nc: Dest 4 = v gate (gated), Dest 5 = v g nc (n)
inline void ple_gate() {
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[4 * 32] = v * sfpi::dst_reg[32];
        sfpi::dst_reg[5 * 32] = v * sfpi::dst_reg[64] * sfpi::dst_reg[96];
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_pc = 3, cb_pprod = 5, cb_psum = 6, cb_pv = 7;
constexpr uint32_t cb_qd = 8, cb_qsq = 9, cb_qs = 11, cb_qrow = 12, cb_qgate = 13, cb_qg = 26;
constexpr uint32_t cb_rv = 14, cb_rg = 15, cb_rnc = 16, cb_rw = 17, cb_rh = 18, cb_rx = 19, cb_rn = 20, cb_rnw = 21;
constexpr uint32_t cb_rgated = 22, cb_rout = 23, cb_rx32 = 24, cb_rx16 = 25;
constexpr uint32_t KD = 80, NKEY = 320;
constexpr uint32_t IDLE = 0xffffffff;

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t p1_first = get_arg_val<uint32_t>(0), p1_count = get_arg_val<uint32_t>(1);
    const uint32_t stream = get_arg_val<uint32_t>(2), p3_count = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_psum);
    cb_wait_front(cb_ones, 1);
    Weights w{cb_w, batch};

    // ---- P1: key tiles' A and B, value tiles and their V
    if (p1_count) {
        cb_wait_front(cb_x, KD);
        for (uint32_t n = p1_first; n < p1_first + p1_count; n++) {
            tile_regs_acquire();
            matmul_start(cb_x, cb_w);
            w.gemv(cb_x, KD, 0);
            if (n < NKEY) {
                cb_wait_front(cb_pc, 3);
                load_n(cb_pc, 3, 1);
                SFPU_CALL(ple_partials);
                cb_pop_front(cb_pc, 3);
                finish(cb_pprod, 2, 4);
            } else {
                SFPU_CALL(ple_sq);
                tile_regs_commit();
                tile_regs_wait();
                pack_n(cb_pv, 1);
                pack_n(cb_pprod, 1, 1);
                tile_regs_release();
            }
            const uint32_t m = n < NKEY ? 2 : 1;
            cb_wait_front(cb_pprod, m);
            tile_regs_acquire();
            matmul_start(cb_pprod, cb_ones);
            for (uint32_t t = 0; t < m; t++) {
                matmul_tiles(cb_pprod, cb_ones, t, 0, t);
            }
            cb_pop_front(cb_pprod, m);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < m; t++) {
                pack_n(cb_psum, 1, t);
            }
            tile_regs_release();
        }
        w.end();
        cb_pop_front(cb_x, KD);
    }

    // ---- P2: the stream's gate and g
    if (stream != IDLE) {
        cb_wait_front(cb_qd, 3);  // X_c^2 summed by row: Dest 3 of the row-sum tiles
        tile_regs_acquire();
        load_n(cb_qd, 3);
        square_tile_init();
        for (uint32_t t = 0; t < 3; t++) {
            square_tile(t);
        }
        cb_pop_front(cb_qd, 3);
        finish(cb_qsq, 3);
        cb_wait_front(cb_qsq, 3);
        cb_wait_front(cb_qs, 3);
        tile_regs_acquire();  // the row sums of A, B, V and X^2
        matmul_start(cb_qs, cb_ones);
        for (uint32_t t = 0; t < 3; t++) {
            matmul_tiles(cb_qs, cb_ones, t, 0, t);
        }
        matmul_start(cb_qsq, cb_ones);
        for (uint32_t t = 0; t < 3; t++) {
            matmul_tiles(cb_qsq, cb_ones, t, 0, 3);
        }
        cb_pop_front(cb_qs, 3);
        cb_pop_front(cb_qsq, 3);
        finish(cb_qrow, 4);
        cb_wait_front(cb_qrow, 4);
        tile_regs_acquire();  // the totals, in every element
        matmul_start(cb_ones, cb_qrow);
        for (uint32_t t = 0; t < 4; t++) {
            matmul_tiles(cb_ones, cb_qrow, 0, t, t);
        }
        cb_pop_front(cb_qrow, 4);
        SFPU_CALL_AT(affine, 1, F_INV2560, F_EPS);
        SFPU_CALL_AT(affine, 3, F_INV2560, F_EPS);
        rsqrt_tile_init();
        rsqrt_tile(1);
        rsqrt_tile(3);
        SFPU_CALL(ple_s);
        copy_dest_values_init();
        copy_dest_values(4, 5);
        rsqrt_tile_init();
        rsqrt_tile(5);
        SFPU_CALL(ple_signed_sqrt);
        sigmoid_tile_init();
        sigmoid_tile(0);
        SFPU_CALL(ple_gnorm);
        rsqrt_tile_init();
        rsqrt_tile(6);
        SFPU_CALL(ple_g);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_qgate, 1, 0);
        pack_n(cb_qg, 1, 7);
        tile_regs_release();
    }

    // ---- P3, per tile: gated and n; the conv; X += gated + silu(conv)
    for (uint32_t n = 0; n < p3_count; n++) {
        cb_wait_front(cb_rv, 1);
        cb_wait_front(cb_rg, 2);
        cb_wait_front(cb_rnc, 1);
        tile_regs_acquire();
        load(cb_rv, 0, 0);
        load_n(cb_rg, 2, 1);
        load(cb_rnc, 0, 3);
        SFPU_CALL(ple_gate);
        cb_pop_front(cb_rv, 1);
        cb_pop_front(cb_rg, 2);
        cb_pop_front(cb_rnc, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_rn, 1, 5);
        pack_n(cb_rnw, 1, 5);
        pack_n(cb_rgated, 1, 4);
        tile_regs_release();

        cb_wait_front(cb_rn, 1);
        cb_wait_front(cb_rh, 3);
        cb_wait_front(cb_rw, 4);
        tile_regs_acquire();
        load(cb_rn, 0, 0);
        load_n(cb_rh, 3, 1);
        load_n(cb_rw, 4, 4);
        SFPU_CALL(conv4);
        silu_tile_init();
        silu_tile(1);
        cb_pop_front(cb_rn, 1);
        cb_pop_front(cb_rh, 3);
        cb_pop_front(cb_rw, 4);
        finish(cb_rout, 1, 1);

        cb_wait_front(cb_rx, 1);
        cb_wait_front(cb_rgated, 1);
        cb_wait_front(cb_rout, 1);
        tile_regs_acquire();
        load(cb_rx, 0, 0);
        load(cb_rgated, 0, 1);
        load(cb_rout, 0, 2);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        cb_pop_front(cb_rx, 1);
        cb_pop_front(cb_rgated, 1);
        cb_pop_front(cb_rout, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_rx32, 1);
        pack_n(cb_rx16, 1);
        tile_regs_release();
    }
}
