// The card's share of a layer's routed experts inside a fused layer program, reader and writer sides
// (compute in fused/dexp_cmp.h). Between R1 and R2, with the host link (fused/link.h):
//   the coordinator sends hn_f and the logits; the host routes, and replies first with the card's routes (n <= 10
//   cache slots and weights: words 0, 1..10, 11..20 of the first 128 bytes) -> region RQ, broadcast at barrier X1
//   E1   units (j, 32-output tile p), 20 per expert, split evenly over the cores: gate|up from the cache slot's codes
//        and scales, h_j into region H (bf16 [n x 640]) of the coordinator's arena, broadcast at barrier X2
//   E2   units (j, output column), 80 per expert: down from its decoded values; partial row 80 j + column (fp32 32) to
//        the partial rows in card memory
//   barrier X3 (every core's partial rows written): R2's cores read the n partial rows of their tile while the host
//   still computes; the host then replies with the CPU's experts' sum (region RT, barrier X4), and R2 adds the
//   partial rows to it
// Buffers borrowed from the layer program's other phases (each left at the position its owner's batches need):
// x (cb 1, 80 bf16 tiles: x's k-tile k in row k % 32; only row 0 is ever written by the other phases, so the rest
// stays zero after one fill), the weight stream (cb 0, down's values, padded to a batch of 8), H2's r_s (cb 13: the
// scales, padded to 4), A's per-tile constants (cb 16: R2's partials, padded to 7), the router's h (cb 49), the mix
// outputs (cb 7), the mixes' first input (cb 3: the route count for the compute), the routed tile (cb 52: w_j) and,
// compute side, A's squares (cb 19: Y, padded to 4). Own: codes (cb 63, uint32) and F (cb 59, compute side). A cache
// slot in card memory (g5_cache.py): gate|up codes [40][10] and scales [40][4] (24-k-tile groups), down values
// [80][20] (bfp8_b) and scales [80][1].
#pragma once

#include "fused/dm.h"
#if defined(M_MAX) && M_MAX > 1
#include "fused/hc_dm.h"  // the mp-row helpers
#endif

namespace fused::dexp {

constexpr uint32_t cb_down = 0, cb_x = 1, cb_s = 13, cb_part = 16, cb_meta = 3, cb_w = 52, cb_codes = 63;
constexpr uint32_t cb_out = 7, cb_h16 = 49;
constexpr uint32_t D = 2560, FF = 640, KCG = 10, KQG = 4, KD = 20, GROUPS = 4, TILE = 4096, F_PAGE = 1088;

inline uint32_t lo(uint32_t n, uint32_t k, uint32_t c) { return n * k / c; }
inline uint32_t group_codes(uint32_t g) { return g == GROUPS - 1 ? 1 : 3; }

// n pages into cb in one batch, each page's address wrapped at the buffer's end, pushed one at a time (cb_push_back
// wraps only when it lands exactly on the end); src(p) = page p's NOC address
template <typename F>
inline void batch_read(uint32_t cb, uint32_t n, uint32_t bytes, F src) {
    cb_reserve_back(cb, n);
    const auto& ci = get_local_cb_interface(cb);
    const uint32_t base = ci.fifo_limit - ci.fifo_size;
    uint32_t a = get_write_ptr(cb);
    for (uint32_t p = 0; p < n; p++) {
        noc_async_read(src(p), a, bytes);
        a += bytes;
        if (a >= base + ci.fifo_size) {
            a = base;
        }
    }
    noc_async_read_barrier();
    for (uint32_t p = 0; p < n; p++) {
        cb_push_back(cb, 1);
    }
}
inline void pad_pages(uint32_t cb, uint32_t n) {
    for (uint32_t p = 0; p < n; p++) {
        cb_reserve_back(cb, 1);
        cb_push_back(cb, 1);
    }
}

inline uint32_t cb_base(uint32_t cb) {
    const auto& ci = get_local_cb_interface(cb);
    return ci.fifo_limit - ci.fifo_size;
}

// once per program: the x buffer all zero (the other phases write only row 0 of its tiles)
inline void zero_x(uint32_t zero) {
    const uint32_t b = cb_base(cb_x);
    for (uint32_t k = 0; k < 80; k++) {
        l1_copy(zero, b + k * 2048, 2048);
    }
    noc_async_read_barrier();
}

// x's k-tiles 0..n-1 from the bf16 vector at L1 src into row k % 32 of tile k (row 0 zeroed where it is not that
// row), as one push of the whole buffer (80 pages: every other user pushes 80, so tile k stays at page k)
inline void x_rows(uint32_t src, uint32_t n, uint32_t zero) {
    cb_reserve_back(cb_x, 80);
    const uint32_t b = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < n; k++) {
        const uint32_t r = k % 32, t = b + k * 2048;
        if (r) {
            l1_copy(zero, t, 32);
            l1_copy(zero, t + 512, 32);
        }
        const uint32_t row = t + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        l1_copy(src + k * 64, row, 32);
        l1_copy(src + k * 64 + 32, row + 512, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, 80);
}

// a layer's cache tensors (its table row) and the partial rows
struct Cache {
    uint32_t gu_codes, gu_scales, d_vals, d_scales;
};

// reader, after X1: the route count to the compute (cb_meta), then E1's inputs; returns n
template <typename Args>
uint32_t e1_read(const Args& d_args, const Cache& c, uint32_t rq, uint32_t hn, uint32_t zero, uint32_t me,
                 uint32_t cores) {
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rq);
    const uint32_t n = words[0];
    cb_reserve_back(cb_meta, 3);  // the count in word 0; three pages, the mixes' batch on this buffer
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = n;
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb_meta, 3);
    if (n == 0) {
        return 0;
    }
    const auto codes = TensorAccessor(d_args, c.gu_codes, TILE), scales = TensorAccessor(d_args, c.gu_scales, TILE);
    x_rows(hn, 80, zero);
    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        const uint32_t j = v / 20, p = v % 20, slot = words[1 + j];
        for (uint32_t col : {p, 20 + p}) {
            for (uint32_t g = 0; g < GROUPS; g++) {
                const uint32_t c0 = (slot * 40 + col) * KCG + 3 * g, q = (slot * 40 + col) * KQG + g;
                batch_read(cb_s, 1, TILE, [&](uint32_t) { return scales.get_noc_addr(q); });
                batch_read(cb_codes, group_codes(g), TILE, [&](uint32_t k) { return codes.get_noc_addr(c0 + k); });
            }
        }
        cb_reserve_back(cb_w, 1);
        fill_row0(get_write_ptr(cb_w), rq + (11 + j) * 4);
        cb_push_back(cb_w, 1);
    }
    return n;
}

// reader, after X2: E2's inputs (h_j from region H), then the pads that leave cb 0 and cb 13 aligned
template <typename Args>
void e2_read(const Args& d_args, const Cache& c, uint32_t n, uint32_t rq, uint32_t h, uint32_t zero, uint32_t me,
             uint32_t cores) {
    if (n == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rq);
    const auto vals = TensorAccessor(d_args, c.d_vals, F_PAGE), scales = TensorAccessor(d_args, c.d_scales, TILE);
    uint32_t cur = 0xffffffff, units = 0;
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++, units++) {
        const uint32_t j = w / 80, col = w % 80, slot = words[1 + j];
        if (j != cur) {
            x_rows(h + j * FF * 2, KD, zero);
            cur = j;
        }
        batch_read(cb_s, 1, TILE, [&](uint32_t) { return scales.get_noc_addr(slot * 80 + col); });
        batch_read(cb_down, KD, F_PAGE, [&](uint32_t k) { return vals.get_noc_addr((slot * 80 + col) * KD + k); });
    }
    pad_pages(cb_down, (4 * units) % 8);  // 20 pages a unit; the weight stream pushes batches of 8
    pad_pages(cb_s, (4 - units % 4) % 4);  // E1 pushes 8 a unit; H2 pushes 4
}

// reader, after X3: the n partial rows of R2's output tile `tile` into cb 16 (row 0), padded to its batch of 7
template <typename Args>
void part_read(const Args& d_args, uint32_t part, uint32_t n, uint32_t tile) {
    if (n == 0) {
        return;
    }
    const auto rows = TensorAccessor(d_args, part, 128);
    for (uint32_t j = 0; j < n; j++) {
        cb_reserve_back(cb_part, 1);
        row0_from_dram(rows, j * 80 + tile, 0, get_write_ptr(cb_part));
        noc_async_read_barrier();
        cb_push_back(cb_part, 1);
    }
    pad_pages(cb_part, (7 - n % 7) % 7);
}

// writer, after arriving at X1: wait for its release (region RQ is then in place), the route count
inline uint32_t routes_seen(uint32_t epoch, uint32_t rq) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), epoch);
    invalidate_l1_cache();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rq)[0];
}

// writer: E1's h rows into region H of the coordinator's arena
inline void e1_write(const Grid& grid, uint32_t stage, uint32_t n, uint32_t h, uint32_t me, uint32_t cores) {
    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        stage_rows(cb_h16, 1, 32, stage);
        grid.send(stage, h + (v / 20) * FF * 2 + (v % 20) * 64, 64);
        noc_async_write_barrier();
    }
}

// writer: E2's partial rows to card memory
template <typename Args>
void e2_write(const Args& d_args, uint32_t stage, uint32_t n, uint32_t part, uint32_t me, uint32_t cores) {
    if (n == 0) {
        return;
    }
    const auto rows = TensorAccessor(d_args, part, 128);
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        stage_rows(cb_out, 1, 64, stage);
        noc_async_write(stage, rows.get_noc_addr(w), 128);
        noc_async_write_barrier();
    }
}

#if defined(M_MAX) && M_MAX > 1
#pragma GCC push_options
#pragma GCC optimize("Os")
// ---- M rows (a verify pass; compute in fused/dexp_cmp.h compute_m). The host's first reply (region RQ, RQ_BYTES):
// word 0 n <= NC, words 1..n the slots, word W0 + NC m + j row m's weight for expert j (0: row m does not route it).
// x's k-tile k holds row m's 32 values in tile row 8 m + k % 8 (block m); a tile's rows 0..mp-1 belong to the other
// phases' x, so E1 clears those but its own. Each code tile (and each 8-k-tile block of down) gets its own scale page:
// the 8 scale rows of its k-tiles (512 bytes in each of two faces of the slot's scale tile) in rows 0..7. E1's h of
// unit (j, p) stays in its core's region A (block li of H_BLOCK bytes, li its index among the core's units: the mp-row
// layout, bf16), and E2 reads h_j's 20 blocks from their cores. E2's partial row block of unit w (fp32, mp rows) goes
// to page w of the partial rows (PART_PAGE). The codes in cb 58 (two pages: two code tiles a batch with their scales).
// Compiled for size, the helpers out of line where a loop would unroll (the kernels share one config buffer).
constexpr uint32_t NC = 24, W0 = 1 + NC, RQ_BYTES = 512, H_BLOCK = 256, PART_PAGE = 512, cb_codes_m = 58;
constexpr uint32_t cb_f_m = 34;  // the compute's F (fused/dexp_cmp.h): busy until X2

// the NOC address of l1 on core idx (fused/grid.py's row-major order: its rectangles are runs of NOC columns over the
// same consecutive rows; g3_layer.py checks the mapping)
inline uint64_t core_addr(const Grid& g, uint32_t idx, uint32_t l1) {
    uint32_t gx = 0;
    for (uint32_t r = 0; r < g.nrects; r++) {
        gx += get_arg_val<uint32_t>(g.rect_arg + 5 * r + 2) - get_arg_val<uint32_t>(g.rect_arg + 5 * r) + 1;
    }
    uint32_t lx = idx % gx;
    const uint32_t ly = idx / gx;
    for (uint32_t r = 0;; r++) {
        const uint32_t x0 = get_arg_val<uint32_t>(g.rect_arg + 5 * r);
        const uint32_t w = get_arg_val<uint32_t>(g.rect_arg + 5 * r + 2) - x0 + 1;
        if (lx < w || r + 1 == g.nrects) {
            return get_noc_addr(x0 + lx, get_arg_val<uint32_t>(g.rect_arg + 5 * r + 1) + ly, l1);
        }
        lx -= w;
    }
}

// x tile at L1 `tile` (t = k % 8) from the mp-row block at NOC address s (row m's face-0 piece at + 32 m, its face-1
// piece at + 32 (mp + m)) into rows 8 m + t; clear: rows 0..mp-1 but t zeroed first
__attribute__((noinline, noclone)) void x_tile_m(uint32_t tile, uint32_t t, uint64_t s, uint32_t mp, uint32_t zero,
                                                 bool clear) {
    if (clear && t < mp) {
        if (t) {
            l1_copy(zero, tile, t * 32);
            l1_copy(zero, tile + 512, t * 32);
        }
        if (t + 1 < mp) {
            l1_copy(zero, tile + (t + 1) * 32, (mp - t - 1) * 32);
            l1_copy(zero, tile + 512 + (t + 1) * 32, (mp - t - 1) * 32);
        }
    } else if (clear) {
        l1_copy(zero, tile, mp * 32);
        l1_copy(zero, tile + 512, mp * 32);
    }
    for (uint32_t m = 0; m < mp; m++) {
        const uint32_t r = 8 * m + t, row = tile + (r / 16) * 1024 + (r % 16) * 32;
        noc_async_read(s + 32 * m, row, 32);
        noc_async_read(s + 32 * (mp + m), row + 512, 32);
    }
}

// reader, after R1's barrier (before the exchange: it needs no route): E1's x from hn_f's mp-row blocks, one push of 80
inline void e1_x(uint32_t hn, uint32_t mp, uint32_t zero) {
    cb_reserve_back(cb_x, 80);
    const uint32_t b = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < 80; k++) {
        x_tile_m(b + k * 2048, k % 8, get_noc_addr(hn + k * 64 * mp), mp, zero, true);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, 80);
}

// page p after cb's write pointer (wrapping at the buffer's end)
inline uint32_t page_at(uint32_t cb, uint32_t p, uint32_t bytes) {
    const auto& ci = get_local_cb_interface(cb);
    const uint32_t base = ci.fifo_limit - ci.fifo_size, a = get_write_ptr(cb) + p * bytes;
    return a >= base + ci.fifo_size ? a - ci.fifo_size : a;
}

// scale rows 8 b .. 8 b + 7 of tile page q into rows 0..7 of the scale page at L1 dst
template <typename A>
inline void scale_rows(const A& scales, uint32_t q, uint32_t b, uint32_t dst) {
    const uint32_t f0 = (b / 2) * 2048 + (b % 2) * 512;
    noc_async_read(scales.get_noc_addr(q, f0), dst, 512);
    noc_async_read(scales.get_noc_addr(q, f0 + 1024), dst + 1024, 512);
}

// one gate|up column of a cache slot: per pair of code tiles one batch of their scale rows and codes
template <typename A>
__attribute__((noinline, noclone)) void column_m(const A& codes, const A& scales, uint32_t sc) {
    for (uint32_t c = 0; c < KCG; c += 2) {
        cb_reserve_back(cb_s, 2);
        cb_reserve_back(cb_codes_m, 2);
        for (uint32_t i = 0; i < 2; i++) {
            scale_rows(scales, sc * KQG + (c + i) / 3, (c + i) % 4, page_at(cb_s, i, TILE));
            noc_async_read(codes.get_noc_addr(sc * KCG + c + i), page_at(cb_codes_m, i, TILE), TILE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < 2; i++) {
            cb_push_back(cb_s, 1);
            cb_push_back(cb_codes_m, 1);
        }
    }
}

// reader, after X1: the route count to the compute, then E1's inputs per unit (row m's weight in row m of cb 52's
// tile, rows mp..3 0), pf() before each column (the caller's share of other reads for that column's first batch);
// returns n
template <typename Args, typename F>
uint32_t e1_read_m(const Args& d_args, const Cache& c, uint32_t rq, uint32_t me, uint32_t cores, F pf) {
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rq);
    const uint32_t n = words[0];
    cb_reserve_back(cb_meta, 3);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = n;
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb_meta, 3);
    const auto codes = TensorAccessor(d_args, c.gu_codes, TILE), scales = TensorAccessor(d_args, c.gu_scales, TILE);
    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        const uint32_t j = v / 20, p = v % 20, slot = words[1 + j];
        pf();
        column_m(codes, scales, slot * 40 + p);  // gate
        pf();
        column_m(codes, scales, slot * 40 + 20 + p);  // up
        cb_reserve_back(cb_w, 1);
        const uint32_t t = get_write_ptr(cb_w);
        for (uint32_t m = 0; m < 4; m++) {
            fill_row(t, m, rq + (W0 + NC * m + j) * 4);
        }
        cb_push_back(cb_w, 1);
    }
    return n;
}

// E2's inputs of one unit: down's three scale pages and 20 value pages, one batch
template <typename A, typename B>
__attribute__((noinline, noclone)) void down_m(const A& scales, const B& vals, uint32_t sc) {
    cb_reserve_back(cb_s, 3);
    cb_reserve_back(cb_down, KD);
    for (uint32_t b = 0; b < 3; b++) {
        scale_rows(scales, sc, b, page_at(cb_s, b, TILE));
    }
    for (uint32_t k = 0; k < KD; k++) {
        noc_async_read(vals.get_noc_addr(sc * KD + k), page_at(cb_down, k, F_PAGE), F_PAGE);
    }
    noc_async_read_barrier();
    for (uint32_t b = 0; b < 3; b++) {
        cb_push_back(cb_s, 1);
    }
    for (uint32_t k = 0; k < KD; k++) {
        cb_push_back(cb_down, 1);
    }
}

// reader, after X2: E2's inputs (h_j's blocks from the cores of units 20 j .. 20 j + 19; pf() before each unit's
// batch, as e1_read_m's), then the pads
template <typename Args, typename F>
void e2_read_m(const Args& d_args, const Grid& grid, const Cache& c, uint32_t n, uint32_t rq, uint32_t h,
               uint32_t mp, uint32_t me, uint32_t cores, F pf) {
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rq);
    const auto vals = TensorAccessor(d_args, c.d_vals, F_PAGE), scales = TensorAccessor(d_args, c.d_scales, TILE);
    uint32_t cur = 0xffffffff, units = 0;
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++, units++) {
        const uint32_t j = w / 80, slot = words[1 + j];
        if (j != cur) {
            cb_reserve_back(cb_x, 80);
            const uint32_t b = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < KD; k++) {
                const uint32_t v = 20 * j + k, owner = ((v + 1) * cores - 1) / (20 * n);  // lo()'s inverse
                x_tile_m(b + k * 2048, k % 8, core_addr(grid, owner, h + (v - lo(20 * n, owner, cores)) * H_BLOCK),
                         mp, 0, false);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, 80);
            cur = j;
        }
        pf();
        down_m(scales, vals, slot * 80 + w % 80);
    }
    pad_pages(cb_down, (4 * units) % 8);  // 20 pages a unit; the weight stream pushes batches of 8
    pad_pages(cb_s, (4 - (3 * units) % 4) % 4);  // E1 pushes 20 a unit, E2 3; H2 pushes 4
}

// reader, after X3: the n partial row blocks of R2's output tile `tile` into cb 16 (rows 0..mp-1), padded to 7
template <typename Args>
void part_read_m(const Args& d_args, uint32_t part, uint32_t n, uint32_t tile, uint32_t mp) {
    const auto rows = TensorAccessor(d_args, part, PART_PAGE);
    for (uint32_t j = 0; j < n; j++) {
        cb_reserve_back(cb_part, 1);
        rows_from_dram(rows, j * 80 + tile, 0, get_write_ptr(cb_part), mp);
        noc_async_read_barrier();
        cb_push_back(cb_part, 1);
    }
    pad_pages(cb_part, (7 - n % 7) % 7);
}

// writer: E1's h rows into this core's region A (block li of H_BLOCK bytes)
inline void e1_write_m(uint32_t n, uint32_t h, uint32_t mp, uint32_t me, uint32_t cores) {
    for (uint32_t li = 0; li < lo(20 * n, me + 1, cores) - lo(20 * n, me, cores); li++) {
        stage_rows_m(cb_h16, 1, 32, h + li * H_BLOCK, mp);
    }
}

// writer: E2's partial row blocks to card memory
template <typename Args>
void e2_write_m(const Args& d_args, uint32_t stage, uint32_t n, uint32_t part, uint32_t mp, uint32_t me,
                uint32_t cores) {
    const auto rows = TensorAccessor(d_args, part, PART_PAGE);
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        stage_rows_m(cb_out, 1, 64, stage, mp);
        noc_async_write(stage, rows.get_noc_addr(w), mp * 128);
        noc_async_write_barrier();
    }
}
#pragma GCC pop_options
#endif

}  // namespace fused::dexp
