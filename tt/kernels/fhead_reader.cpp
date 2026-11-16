// Fused decode head (G3 step 9): the final hyper-connection mix (the model's output norm: hc_head norm, down, up, no
// inject) and the LM head (2560 -> 248,320 logits), on the whole grid (compute fhead_compute.cpp, writer
// fhead_writer.cpp). The mix is fused/hc_dm.h's (its inject column zero); the LM head a GEMV over each core's run of
// output tiles, W's bfp8_b tiles n-major (page n * 80 + k), x = hn's row-0 tiles.
// Runtime args: X fp32, X bf16; grid.h's block; arena offsets X32, X16, P1, RS, HN, LO, W; W', up, norm, the H1 item,
// the H2 tile; the LM head's W, first tile, count; the stamps tensor; [GRID_PROFILE: the profile]; [HEAD_M: m, the
// arena offset HM, the extra rows' tensor ([m - 1 x 2560] bf16, rows 1..m-1 of x; 0 none)]. Compile args: batch, then a
// DRAM accessor.
// With HEAD_M, x's tiles carry m rows: row 0 hn, rows 1..m-1 from region HM, read from card memory at the start (the
// test's rows).
// With X_BLOCKS=n (and HEAD_M; a verify pass): X in the M-row block layout of the layer programs (X tensors [4 x n
// 2560], a stream a page, fused/hc_dm.h's mp-row blocks), M and Mp from the step tensor (words 1, 4), whose address
// takes HEAD_M's m argument (the HM arguments unused); the mix at M rows as attention program 1's (fattn1_reader.cpp
// M rows: the owner layout, X's streams read by H1's cores 1/11 each and exchanged, H1's 44 cores as runtime args at
// the end), hn's mp-row blocks the LM head's x (row m of a tile row m's). Preprocessed out without X_BLOCKS.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_zero = 60, cb_arena = 61, cb_prof = 63;
constexpr uint32_t W_PAGE = 1088, D = 2560, KD = 80;

#ifndef X_BLOCKS
void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    using Acc = decltype(TensorAccessor(d_args, 0u, 0u));
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const auto x32 = acc(4 * D * 4), x16 = acc(4 * D * 2);
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_hn = arena + arg(), r_lo = arena + arg(), r_w = arena + arg();
    const auto wd = acc(W_PAGE), wu = acc(W_PAGE), norm = acc(4 * D * 4);
    const uint32_t h_item = arg(), h_tile = arg();
    const auto wl = acc(W_PAGE);
    const uint32_t first = arg(), count = arg();
    const auto prof = acc(64);
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
#ifdef HEAD_M
    const uint32_t m = arg(), r_hm = arena + arg(), hm_addr = arg();
    const auto hm = TensorAccessor(d_args, hm_addr, D * 2);
    if (m > 1 && hm_addr) {
        for (uint32_t r = 1; r < m; r++) {
            noc_async_read(hm.get_noc_addr(r - 1), r_hm + (r - 1) * D * 2, D * 2);
        }
        noc_async_read_barrier();
    }
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    fused::fill_words(zero, 1024, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };
    if (grid.coordinator) {
        noc_async_read(x32.get_noc_addr(0), r_x32, 4 * D * 4);
        noc_async_read(x16.get_noc_addr(0), r_x16, 4 * D * 2);
        noc_async_read_barrier();
    }
    const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd, h_item);  // H1's first weights before the first barrier
    uint32_t epoch = 1;
    wait(epoch, r_x32, r_p1 - r_x32);
    const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
    fused::hc::read(ws, wd, wu, norm, h_item, h_tile, hreg, r_hn, r_lo, r_w, zero, wait, epoch, h1_pre);

    // ---- the LM head
#ifndef HEAD_M
    fused::x_tiles(cb_x, r_hn, KD, KD);
#else
    cb_reserve_back(cb_x, KD);
    const uint32_t xt = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < KD; k++) {  // row r of a bf16 tile: 32 bytes at r * 32 (face 0) and 512 + r * 32 (face 1)
        for (uint32_t r = 0; r < m; r++) {
            const uint32_t src = (r ? r_hm + (r - 1) * D * 2 : r_hn) + k * 64;
            fused::l1_copy(src, xt + k * 2048 + r * 32, 32);
            fused::l1_copy(src + 32, xt + k * 2048 + 512 + r * 32, 32);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, KD);
#endif
    ws.read(wl, first * KD, count * KD);
    ws.flush();
    wait(++epoch, 0, 0);
    if (grid.coordinator) {
        noc_async_write(get_write_ptr(cb_prof), prof.get_noc_addr(0), 64);
        noc_async_write_barrier();
    }
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the program's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
#else
constexpr uint32_t IDLE = 0xffffffff;

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    using Acc = decltype(TensorAccessor(d_args, 0u, 0u));
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const auto x32 = acc(X_BLOCKS * D * 4), x16 = acc(X_BLOCKS * D * 2);  // a stream a page
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_xown = arena + arg();
    const uint32_t r_hn = arena + arg(), r_lo = arena + arg();
    arg();  // (W: after LO at M rows)
    const auto wd = acc(W_PAGE), wu = acc(W_PAGE), norm = acc(4 * D * 4);
    const uint32_t h_item = arg(), h_tile = arg();
    const auto wl = acc(W_PAGE);
    const uint32_t first = arg(), count = arg();
    const auto prof = acc(64);
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    const auto step = acc(64);  // (HEAD_M's m)
    arg(), arg();               // (HEAD_M's HM, its rows)
    const uint32_t h1_cores = i;  // H1's 44 cores by item 4 n + s, NOC x | y << 16

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    noc_async_read(step.get_noc_addr(0), zero, 64);  // the step's words, through the zero tile before it is cleared
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
    const uint32_t M = sw[1], mp = sw[4];
    fused::fill_words(zero, 1024, 0);
    // the ones tile, Sel (fused/hc_cmp.h h1_rs_m) and the step's words for the compute (tile 2: M, -, -, mp)
    cb_reserve_back(cb_ones, 3);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    {
        const uint32_t sel = get_write_ptr(cb_ones) + 2048;
        fused::fill_words(sel, 1024, 0);
        volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sel);
        for (uint32_t r = 0; r < mp; r++) {
            for (uint32_t c = r; c < 32; c += mp) {
                e[(c / 16) * 256 + r * 16 + c % 16] = 0x3f80;
            }
        }
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel + 2048);
        t[0] = M, t[3] = mp;
        asm volatile("fence rw, rw" ::: "memory");
    }
    cb_push_back(cb_ones, 3);
    const uint32_t rs_m = r_p1 + mp * 4 * fused::hc::LO * 4;  // the r_s after P1 (fused/hc_dm.h RegionsM)
    const uint32_t blk = mp * 128;                            // an fp32 tile's mp-row block

    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };

    // ---- X: each of a stream's 11 H1 cores reads 1/11 of its blocks and sends it to the other 10, then arrives at
    // the first barrier (the writer does not); the H2 cores read their tile's blocks of the 4 streams (xown)
    if (h_item != IDLE) {
        const uint32_t s = h_item & 3, n = h_item >> 2, t0 = 80 * n / 11, t1 = 80 * (n + 1) / 11;
        const uint32_t o32 = r_x32 + t0 * blk, o16 = r_x16 + t0 * mp * 64;
        noc_async_read(x32.get_noc_addr(s, t0 * blk), o32, (t1 - t0) * blk);
        noc_async_read(x16.get_noc_addr(s, t0 * mp * 64), o16, (t1 - t0) * mp * 64);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 11; k++) {
            const uint32_t c = get_arg_val<uint32_t>(h1_cores + 4 * k + s);
            if (k != n) {
                noc_async_write(o32, get_noc_addr(c & 0xffff, c >> 16, o32), (t1 - t0) * blk);
                noc_async_write(o16, get_noc_addr(c & 0xffff, c >> 16, o16), (t1 - t0) * mp * 64);
            }
        }
        noc_async_write_barrier();
    }
    grid.arrive(1);
    if (h_tile != IDLE && h_tile < D / 32) {
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(x32.get_noc_addr(s, h_tile * blk), r_xown + s * blk, blk);
        }
    }
    const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd, h_item);  // H1's first weights before the first barrier
    uint32_t epoch = 1;
    wait(epoch, 0, 0);
    const fused::hc::RegionsM hreg{r_x32, r_x16, r_xown, r_p1, rs_m, mp};
    fused::hc::read_m(ws, wd, wu, norm, h_item, h_tile, hreg, r_hn, r_lo, zero, wait, epoch, h1_pre);

    // ---- the LM head on hn's mp rows
    fused::x_tiles_m(cb_x, r_hn, KD, KD, mp);
    ws.read(wl, first * KD, count * KD);
    ws.flush();
    wait(++epoch, 0, 0);
    if (grid.coordinator) {
        noc_async_write(get_write_ptr(cb_prof), prof.get_noc_addr(0), 64);
        noc_async_write_barrier();
    }
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the program's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
#endif
