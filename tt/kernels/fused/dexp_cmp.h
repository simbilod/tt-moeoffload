// The card's share of a layer's routed experts inside a fused layer program, compute side (the phases
// and buffers are described in fused/dexp_dm.h; the standalone program and its measurements are dexp_*.cpp,
// g5_experts.py). One token; the arithmetic of dexp_compute.cpp:
//   E1  per unit (expert j, 32-output tile p): gate column p, then up column p, each four 24-k-tile groups: the codes
//       decoded on the SFPU into bfp8_b F tiles (one Dest pass per code tile), P = sum X_k F_k (x's k-tile k in row
//       k % 32 of its tile), Y += A * P (A: the group's scale tile, row k % 32 = block k's scales) through the
//       packer's L1 accumulation; then h = w_j silu(g) u from the column sums of the two Y, bf16 to the writer
//   E2  per unit (j, output column): down from bfp8_b tiles of the decoded values streamed tile by tile, one group of
//       20 k-tiles, x = h_j in row k; the column sums (fp32) to the writer
// Unpack-to-dest (codes into Dest 0, scales into 1, w_j into 5) only waits for the math thread: Y comes out of Dest 2,
// sums and h out of Dest 7 (none of them unpacked into), and a decode's next codes wait for the previous code tile's
// F push.
#pragma once

#include "fused/cmp.h"

#ifdef TRISC_MATH
#include "dexp_sfpu.h"
#endif

namespace fused::dexp {

constexpr uint32_t cb_codes = 63, cb_f = 59, cb_x = 1, cb_ones = 2, cb_s = 13, cb_y = 19, cb_out = 7, cb_h16 = 49;
constexpr uint32_t cb_w = 52, cb_meta = 3, cb_down = 0;  // cb_meta: read_tile_value takes CBs 0..31; the mixes' 3-batch
constexpr uint32_t FT = 24, KD = 20, GROUPS = 4;   // F tiles per group (24 k-tiles); down's k-tiles; groups per column

inline uint32_t lo(uint32_t n, uint32_t k, uint32_t c) { return n * k / c; }
inline uint32_t group_codes(uint32_t g) { return g == GROUPS - 1 ? 1 : 3; }  // 80 k-tiles = 24 + 24 + 24 + 8

// nc code tiles into cb_f (k order), padded to FT tiles
inline void decode(uint32_t nc) {
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup (a binary SFPU op ran since)
    reconfig_data_format_srca(cb_codes);
    copy_init(cb_codes);
    pack_reconfig_data_format(cb_f);
    for (uint32_t c = 0; c < nc; c++) {
        if (c) {
            cb_wait_front(cb_f, 8 * c);  // the previous code tile's pack (it read Dest 0) is done
        }
        cb_wait_front(cb_codes, 1);
        cb_reserve_back(cb_f, 8);
        tile_regs_acquire();
        copy_tile(cb_codes, 0, 0);
        SFPU_CALL(dexp::decode8_linear);
        tile_regs_commit();
        cb_pop_front(cb_codes, 1);
        tile_regs_wait();
        for (uint32_t t = 1; t < 8; t++) {
            pack_tile(t, cb_f);
        }
        pack_tile(0, cb_f);
        tile_regs_release();
        cb_push_back(cb_f, 8);
    }
    if (nc * 8 < FT) {
        cb_reserve_back(cb_f, FT - nc * 8);
        cb_push_back(cb_f, FT - nc * 8);
    }
}

// Y (+)= A * P into slot `slot` of cb_y; P from the group's nk F tiles of cb_f (by index) or, when streamed, from nk
// tiles of cb_down taken one at a time; x from k-tile kx of cb_x. Y leaves from Dest 2: the next decode's codes are
// unpacked straight into Dest 0, which may still be packing
template <bool STREAMED>
inline void group(uint32_t kx, uint32_t nk, uint32_t slot, bool first) {
    constexpr uint32_t fcb = STREAMED ? cb_down : cb_f;
    if constexpr (!STREAMED) {
        cb_wait_front(cb_f, FT);
    }
    tile_regs_acquire();
    matmul_start(cb_x, fcb);
    for (uint32_t t = 0; t < nk; t++) {
        if constexpr (STREAMED) {
            cb_wait_front(cb_down, 1);
            matmul_tiles(cb_x, cb_down, kx + t, 0, 0);
            cb_pop_front(cb_down, 1);
        } else {
            matmul_tiles(cb_x, cb_f, kx + t, t, 0);
        }
    }
    cb_wait_front(cb_s, 1);
    load(cb_s, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 2);
    tile_regs_commit();
    if constexpr (!STREAMED) {
        cb_pop_front(cb_f, FT);
    }
    cb_pop_front(cb_s, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_y);
    pack_reconfig_l1_acc(first ? 0 : 1);
    pack_tile<true>(2, cb_y, slot);
    pack_reconfig_l1_acc(0);
    tile_regs_release();
}

// E1 and E2 for this core (index me of cores); returns the route count n (R2 adds n partials)
inline uint32_t compute(uint32_t me, uint32_t cores) {
    cb_wait_front(cb_meta, 3);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 3);
    if (n == 0) {
        return 0;
    }
    // ---- E1
    const uint32_t u1 = lo(20 * n, me + 1, cores) - lo(20 * n, me, cores);
    const uint32_t u2 = lo(80 * n, me + 1, cores) - lo(80 * n, me, cores);
    cb_wait_front(cb_x, 80);
    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        cb_reserve_back(cb_y, 2);
        for (uint32_t half = 0; half < 2; half++) {
            for (uint32_t g = 0; g < GROUPS; g++) {
                decode(group_codes(g));
                group<false>(FT * g, 8 * group_codes(g), half, g == 0);
            }
        }
        cb_push_back(cb_y, 2);
        cb_wait_front(cb_y, 2);
        cb_wait_front(cb_w, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_y);
        matmul_tiles(cb_ones, cb_y, 0, 0, 6);  // every row: g's column sums
        matmul_tiles(cb_ones, cb_y, 0, 1, 7);  // u's
        load(cb_w, 0, 5);
        silu_tile_init();
        silu_tile(6);
        mul_binary_tile_init();
        mul_binary_tile(6, 7, 6);
        mul_binary_tile(6, 5, 7);
        tile_regs_commit();
        cb_pop_front(cb_y, 2);
        cb_pop_front(cb_w, 1);
        tile_regs_wait();
        pack_n(cb_h16, 1, 7);
        tile_regs_release();
    }
    cb_pop_front(cb_x, 80);
    // ---- E2
    uint32_t cur = 0xffffffff;
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        const uint32_t j = w / 80;
        if (j != cur) {
            if (cur != 0xffffffff) {
                cb_pop_front(cb_x, 80);
            }
            cb_wait_front(cb_x, 80);  // h_j in rows 0..19 of tiles 0..19
            cur = j;
        }
        cb_reserve_back(cb_y, 1);
        group<true>(0, KD, 0, true);
        cb_push_back(cb_y, 1);
        cb_wait_front(cb_y, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_y);
        matmul_tiles(cb_ones, cb_y, 0, 0, 7);
        cb_pop_front(cb_y, 1);
        finish(cb_out, 1, 7);
    }
    if (cur != 0xffffffff) {
        cb_pop_front(cb_x, 80);
    }
    // the pads that leave the borrowed buffers where their owners' batches need them (fused/dexp_dm.h): the reader's
    // on the weight stream (8) and the scales (4), this kernel's own on Y (cb 19: A's batches of 4)
    if ((4 * u2) % 8) {
        cb_wait_front(cb_down, 4);
        cb_pop_front(cb_down, 4);
    }
    if (u2 % 4) {
        cb_wait_front(cb_s, 4 - u2 % 4);
        cb_pop_front(cb_s, 4 - u2 % 4);
    }
    if (const uint32_t y = (2 * u1 + u2) % 4) {
        cb_reserve_back(cb_y, 4 - y);
        cb_push_back(cb_y, 4 - y);
        cb_wait_front(cb_y, 4 - y);
        cb_pop_front(cb_y, 4 - y);
    }
    return n;
}

#if defined(M_MAX) && M_MAX > 1
#pragma GCC push_options
#pragma GCC optimize("Os")
// ---- M rows (a verify pass, fused/dexp_dm.h): x's k-tile k holds row m's 32 values in tile row 8 m + k % 8, so a
// code tile's 8 k-tiles (one per row of each 8-row block) accumulate into P without mixing; the reader brings each
// code tile's 8 scale rows into rows 0..7 of a scale page, which multiply P repeated over its four 8-row blocks
// (dexp::rowmul0 on the SFPU); Sel (cb 2 tile 2, rows 0..3: row m sums rows 8 m .. 8 m + 7) then gives row m's column
// sums. The codes in cb 58; F in cb 34 (B's S rows, fp32: exact; idle from B's barrier to the next pass's B, and the
// compute both its producer and its consumer there too: a buffer must keep its producer and consumer threads, whose
// counters are their own). Compiled for size, the helpers out of line: the program's kernels share one config buffer,
// which unrolled loops overflow.
constexpr uint32_t cb_codes_m = 58, cb_f_m = 34, SEL_M = 2;

// one code tile into cb_f_m (8 F tiles, k order); the previous one's F was consumed by its matmul, so its pack (which
// read Dest 0) is done before the codes are unpacked into Dest 0
__attribute__((noinline, noclone)) void decode_m() {
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup (a binary SFPU op ran since)
    reconfig_data_format_srca(cb_codes_m);
    copy_init(cb_codes_m);
    pack_reconfig_data_format(cb_f_m);
    cb_wait_front(cb_codes_m, 1);
    cb_reserve_back(cb_f_m, 8);
    tile_regs_acquire();
    copy_tile(cb_codes_m, 0, 0);
    SFPU_CALL(dexp::decode8_linear);
    tile_regs_commit();
    cb_pop_front(cb_codes_m, 1);
    tile_regs_wait();
    for (uint32_t t = 1; t < 8; t++) {
        pack_tile(t, cb_f_m);
    }
    pack_tile(0, cb_f_m);
    tile_regs_release();
    cb_push_back(cb_f_m, 8);
}

// load()'s one copy (each call site of an inline load with its own constants is a copy of its unpack and math code)
__attribute__((noinline, noclone)) void load_m(uint32_t cb, uint32_t dst) { load(cb, 0, dst); }

// Y (+)= P * S into slot `slot` of cb_y: P from nk k-tiles of x from kx with F tiles of cb f (streamed: one at a time
// from the front, else 8 by index), S the scale page at the front of cb_s (popped); Y leaves from Dest 2
__attribute__((noinline, noclone)) void block_m(uint32_t f, bool streamed, uint32_t kx, uint32_t nk, uint32_t slot,
                                                bool first) {
    if (!streamed) {
        cb_wait_front(f, 8);
    }
    tile_regs_acquire();
    matmul_start(cb_x, f);
    for (uint32_t t = 0; t < nk; t++) {
        if (streamed) {
            cb_wait_front(f, 1);
        }
        matmul_tiles(cb_x, f, kx + t, streamed ? 0 : t, 0);
        if (streamed) {
            cb_pop_front(f, 1);
        }
    }
    cb_wait_front(cb_s, 1);
    load_m(cb_s, 1);
    MATH((SFPU_UNARY_INIT(tanh)));
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, dexp::rowmul0, 0, VectorMode::None)));
    tile_regs_commit();
    if (!streamed) {
        cb_pop_front(f, 8);
    }
    cb_pop_front(cb_s, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_y);
    pack_reconfig_l1_acc(first ? 0 : 1);
    pack_tile<true>(2, cb_y, slot);
    pack_reconfig_l1_acc(0);
    tile_regs_release();
}

// n tiles of cb_y through Sel into Dest 6.. (row m: row m's column sums); the tile regs left acquired
__attribute__((noinline, noclone)) void sel_sums(uint32_t n) {
    cb_push_back(cb_y, n);
    cb_wait_front(cb_y, n);
    tile_regs_acquire();
    matmul_start(cb_ones, cb_y);
    for (uint32_t t = 0; t < n; t++) {
        matmul_tiles(cb_ones, cb_y, SEL_M, t, 6 + t);
    }
    cb_pop_front(cb_y, n);
}

// E1 and E2 at M rows for this core; returns n. E1's x is always pushed (the reader builds it before it knows n)
inline uint32_t compute_m(uint32_t me, uint32_t cores) {
    cb_wait_front(cb_meta, 3);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 3);
    cb_wait_front(cb_x, 80);
    // ---- E1: per unit (j, p) gate column p then up column p, code tile by code tile; h = w silu(g) u by row
    const uint32_t u1 = lo(20 * n, me + 1, cores) - lo(20 * n, me, cores);
    const uint32_t u2 = lo(80 * n, me + 1, cores) - lo(80 * n, me, cores);
    for (uint32_t v = 0; v < u1; v++) {
        cb_reserve_back(cb_y, 2);
#pragma GCC unroll 1
        for (uint32_t c = 0; c < 20; c++) {  // gate's 10 code tiles into slot 0, then up's into slot 1
            decode_m();
            block_m(cb_f_m, false, 8 * (c % 10), 8, c / 10, c % 10 == 0);
        }
        cb_wait_front(cb_w, 1);
        sel_sums(2);
        load_m(cb_w, 5);  // row m: row m's weight for this expert
        silu_tile_init();
        silu_tile(6);
        mul_binary_tile_init();
        mul_binary_tile(6, 7, 6);
        mul_binary_tile(6, 5, 7);
        tile_regs_commit();
        cb_pop_front(cb_w, 1);
        tile_regs_wait();
        pack_n(cb_h16, 1, 7);
        tile_regs_release();
    }
    cb_pop_front(cb_x, 80);
    // ---- E2: per unit (j, output column) down's 20 k-tiles as blocks of 8, 8, 4, a scale page each
    uint32_t cur = 0xffffffff;
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        const uint32_t j = w / 80;
        if (j != cur) {
            if (cur != 0xffffffff) {
                cb_pop_front(cb_x, 80);
            }
            cb_wait_front(cb_x, 80);  // h_j's rows in tiles 0..19
            cur = j;
        }
        cb_reserve_back(cb_y, 1);
#pragma GCC unroll 1
        for (uint32_t b = 0; b < 3; b++) {
            block_m(cb_down, true, 8 * b, b < 2 ? 8 : 4, 0, b == 0);
        }
        sel_sums(1);
        finish(cb_out, 1, 6);
    }
    if (cur != 0xffffffff) {
        cb_pop_front(cb_x, 80);
    }
    // the borrowed buffers' pads, as compute()'s (the scale pages: 20 a unit in E1, 3 in E2)
    if ((4 * u2) % 8) {
        cb_wait_front(cb_down, 4);
        cb_pop_front(cb_down, 4);
    }
    if (const uint32_t s = (3 * u2) % 4) {
        cb_wait_front(cb_s, 4 - s);
        cb_pop_front(cb_s, 4 - s);
    }
    if (const uint32_t y = (2 * u1 + u2) % 4) {
        cb_reserve_back(cb_y, 4 - y);
        cb_push_back(cb_y, 4 - y);
        cb_wait_front(cb_y, 4 - y);
        cb_pop_front(cb_y, 4 - y);
    }
    return n;
}
#pragma GCC pop_options
#endif

}  // namespace fused::dexp
