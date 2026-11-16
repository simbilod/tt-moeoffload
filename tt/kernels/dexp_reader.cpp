// Decode-time routed experts on the card, reader. Phases (compute in dexp_compute.cpp, writer in
// dexp_writer.cpp), one grid barrier after each (fused/grid.h):
//   E1  gate|up and h: units (j, 32-output tile p), 20 per expert: gate column p and up column p over all of K (three
//       32-k-tile groups each, from the codes), then h = w_j silu(g) u on the same core; h into the coordinator's
//       arena (bf16 [n x 640]), which the barrier broadcasts
//   E2  down: units (j, output column 0..79), 80 per expert, j-major (an expert's x tiles are built once per core),
//       from bfp8_b tiles of the decoded IQ4_NL values (exact integers: the card keeps down decoded, its polynomial
//       decode costing ~60% of the SFPU's time); each writes one partial row at page w = 80 j + column, the experts'
//       sum being the sum over j of those rows
// The route list (row-major uint32 [32]): word 0 the count n (<= 10), words 1..10 the cache slots, 11..20 the
// weights (fp32 bits). A cache slot holds an expert in card form: gate|up codes [40 columns][10 code tiles] and scales
// [40][3 groups] (uint32 / fp32 tiles), down values [80 columns][20 k-tiles] (bfp8_b) and scales [80][1] (fp32).
// Runtime args: grid.h's block; gate|up codes, gate|up scales, down values, down scales, hn (bf16 [2560]), routes,
// the partials; the arena's H offset; this core's index, the core count. Compile args: a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_codes = 0, cb_x = 1, cb_ones = 2, cb_s = 3, cb_w2 = 9, cb_xd = 11, cb_meta = 12;
constexpr uint32_t cb_zero = 13, cb_rt = 14, cb_hn = 15, cb_arena = 16, cb_fd = 18;
constexpr uint32_t D = 2560, FF = 640, KCG = 10, KQG = 3, KD = 20, FT = 32, TILE = 4096, F_PAGE = 1088;

inline uint32_t lo(uint32_t n, uint32_t k, uint32_t c) { return n * k / c; }

// n pages into cb in one batch: page p's L1 address wraps at the buffer's end (a batch need not start aligned), and
// the pages are pushed one at a time (cb_push_back wraps its pointer only when it lands exactly on the end)
template <typename F>
inline void batch_read(uint32_t cb, uint32_t n, uint32_t bytes, F src) {
    cb_reserve_back(cb, n);
    const auto& ci = get_local_cb_interface(cb);
    const uint32_t base = ci.fifo_limit - ci.fifo_size;
    uint32_t a = get_write_ptr(cb);
    for (uint32_t p = 0; p < n; p++) {
#ifndef DEXP_NO_READ  // timing breakdown: the weights left out
        noc_async_read(src(p), a, bytes);
#endif
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

// n bf16 x tiles from the bf16 vector at L1 src into cb, k-tile k in row k % 32 (zeros elsewhere)
inline void x_rows(uint32_t cb, uint32_t src, uint32_t n, uint32_t zero) {
    cb_reserve_back(cb, n);
    const uint32_t xt = get_write_ptr(cb);
    for (uint32_t k = 0; k < n; k++) {
        fused::l1_copy(zero, xt + k * 2048, 2048);
    }
    noc_async_read_barrier();
    for (uint32_t k = 0; k < n; k++) {
        const uint32_t r = k % 32, row = xt + k * 2048 + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        fused::l1_copy(src + k * 64, row, 32);
        fused::l1_copy(src + k * 64 + 32, row + 512, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    const auto gu_codes = TensorAccessor(d_args, arg(), TILE), gu_scales = TensorAccessor(d_args, arg(), TILE);
    const auto d_vals = TensorAccessor(d_args, arg(), F_PAGE), d_scales = TensorAccessor(d_args, arg(), TILE);
    const auto hn = TensorAccessor(d_args, arg(), D * 2), routes = TensorAccessor(d_args, arg(), 128);
    arg();  // the partials (the writer's)
    const uint32_t r_h = get_write_ptr(cb_arena) + arg();
    const uint32_t me = arg(), cores = arg();

    const uint32_t zero = get_write_ptr(cb_zero), rt = get_write_ptr(cb_rt), xv = get_write_ptr(cb_hn);
    fused::fill_words(zero, 512, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
    noc_async_read(routes.get_noc_addr(0), rt, 128);
    noc_async_read(hn.get_noc_addr(0), xv, D * 2);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt);
    const uint32_t n = words[0];
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = n;
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb_meta, 1);

    // ---- E1: per unit, gate column p then up column p, each group's scale tile ahead of its codes; then w_j
    x_rows(cb_x, xv, 80, zero);
    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        const uint32_t j = v / 20, p = v % 20, slot = words[1 + j];
        for (uint32_t col : {p, 20 + p}) {
            for (uint32_t g = 0; g < 3; g++) {
                const uint32_t c0 = (slot * 40 + col) * KCG + 4 * g, q = (slot * 40 + col) * KQG + g;
                batch_read(cb_s, 1, TILE, [&](uint32_t) { return gu_scales.get_noc_addr(q); });
                batch_read(cb_codes, g == 2 ? 2 : 4, TILE, [&](uint32_t c) { return gu_codes.get_noc_addr(c0 + c); });
            }
        }
        cb_reserve_back(cb_w2, 1);
        fused::fill_row0(get_write_ptr(cb_w2), rt + (11 + j) * 4);
        cb_push_back(cb_w2, 1);
    }
    grid.wait(1, r_h, n * FF * 2);

    // ---- E2: down, F tiles straight from card memory (padded to FT pushes, so the compute's indexed reads align)
    uint32_t cur = 0xffffffff;
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        const uint32_t j = w / 80, col = w % 80, slot = words[1 + j];
        if (j != cur) {
            x_rows(cb_xd, r_h + j * FF * 2, KD, zero);
            cur = j;
        }
        batch_read(cb_s, 1, TILE, [&](uint32_t) { return d_scales.get_noc_addr(slot * 80 + col); });
        batch_read(cb_fd, KD, F_PAGE, [&](uint32_t k) { return d_vals.get_noc_addr((slot * 80 + col) * KD + k); });
        cb_reserve_back(cb_fd, FT - KD);
        cb_push_back(cb_fd, FT - KD);
    }
    grid.wait(2);
}
