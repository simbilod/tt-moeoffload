// The hyper-connection mix's two phases, reader and writer sides (the math is in fused/hc_cmp.h; standalone test in
// tt/flashnext/g3_hc.py). Shared by the fused layer programs, which give these circular buffers the same indices.
//   H1  item (n, s) = 4 n + s on 44 cores: r_s from X_s, lo[s][n] = r_s (X_s W'_s); P1 gets the partials, RS the r_s
//   HL  lo's tile t on 11 cores (the H2 tile index t < 11): the 4 partials summed, silu(lo / 4) as the gate GEMV's
//       x (bf16) to region LO for t < 10, w = 2 sigmoid(lo / 4) to region W for t = 10. Once, instead of on every
//       H2 core (which cost ~30 us of serial tile operations per mix).
//   H2  tile j on 80 cores: the gate tiles (s, j) from LO, hn's tile j
// Regions are L1 addresses in the arena; P1 and RS are contiguous (H1's broadcast), as are LO and W (HL's: W =
// LO + 640).
#pragma once

#include "fused/dm.h"

namespace fused::hc {

constexpr uint32_t cb_w = 0, cb_x = 1, cb_hd = 3, cb_hout = 7, cb_hrout = 8, cb_hp = 9, cb_hx2 = 10, cb_hxs = 11;
constexpr uint32_t cb_hnrm = 12;
constexpr uint32_t cb_hrs = 13, cb_hhn = 14, cb_hchk = 15, cb_hlo = 14;  // HL and H2 take turns on cb 14
constexpr uint32_t D = 2560, KS = 80, KW = 320, KHU = 10, LO = 352, IDLE = 0xffffffff;
#ifndef HC_PREFETCH_BATCHES
#define HC_PREFETCH_BATCHES 4  // the weight buffer's depth in batches (every program using this has 8 x 4 pages)
#endif

// reader, before the barrier that precedes H1 (the previous phase's weights all issued): H1's first weights, as many
// pages as the weight buffer holds; read() takes the count and streams the rest
template <typename Acc>
uint32_t prefetch_h1(Stream<Acc>& ws, const Acc& wd, uint32_t item) {
    if (item == IDLE) {
        return 0;
    }
    const uint32_t pre = ws.batch * HC_PREFETCH_BATCHES;
    ws.read(wd, (item >> 2) * KW + (item & 3) * KS, pre);
    return pre;
}

struct Regions {
    uint32_t x32, x16, p1, rs;  // X fp32, X bf16, the partials, the r_s (16 bytes each)
};

// reader: H1's inputs, barrier, HL's and H2's inputs, barrier, H2's x and weights, barrier (wait(n, region, bytes) is
// the program's barrier wait); h1_pre: H1's pages already issued by prefetch_h1()
template <typename Acc, typename Wait>
void read(Stream<Acc>& ws, const Acc& wd, const Acc& wu, const Acc& norm, uint32_t item, uint32_t tile,
          const Regions& r, uint32_t hn, uint32_t lo, uint32_t w, uint32_t zero, Wait& wait, uint32_t& epoch,
          uint32_t h1_pre = 0) {
    if (item != IDLE) {
        const uint32_t n = item >> 2, s = item & 3;
        cb_reserve_back(cb_hd, 3);
        const uint32_t d = get_write_ptr(cb_hd);
        l1_copy(r.x32 + s * D * 4, d, 8192);
        l1_copy(r.x32 + s * D * 4 + 8192, d + 8192, 2048);
        l1_copy(zero, d + 10240, 2048);
        noc_async_read_barrier();
        cb_push_back(cb_hd, 3);
        x_tiles(cb_x, r.x16 + s * D * 2, KS, KS);
        ws.read(wd, n * KW + s * KS + h1_pre, KS - h1_pre);  // h1_pre: pages prefetch_h1() issued
        ws.flush();
    }
    if (tile != IDLE && tile < D / 32) {
        cb_reserve_back(cb_hnrm, 4);
        const uint32_t d = get_write_ptr(cb_hnrm);
        for (uint32_t s = 0; s < 4; s++) {
            row0_from_dram(norm, 0, (s * D + tile * 32) * 4, d + s * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_hnrm, 4);
    }
    wait(++epoch, r.p1, r.rs + 64 - r.p1);
    if (tile != IDLE && tile <= LO / 32 - 1) {  // HL: lo's tile t's 4 partials
        cb_reserve_back(cb_hp, 4);
        const uint32_t d = get_write_ptr(cb_hp);
        for (uint32_t s = 0; s < 4; s++) {
            row0_from_l1(r.p1 + (s * LO + tile * 32) * 4, d + s * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_hp, 4);
    }
    // H2's first weights before HL's barrier: as many pages as the weight buffer holds (the compute drains them only
    // once H2's x is in, after the barrier; one page more would wait for it: deadlock)
    const uint32_t pre = ws.batch * HC_PREFETCH_BATCHES;
    if (tile != IDLE && tile < D / 32) {
        for (uint32_t s = 0, left = pre; s < 4 && left; s++) {
            const uint32_t n = left < KHU ? left : KHU;
            ws.read(wu, (s * D / 32 + tile) * KHU, n);
            left -= n;
        }
    }
    if (tile != IDLE && tile < D / 32) {  // H2's inputs that need no activation
        cb_reserve_back(cb_hxs, 4);
        cb_reserve_back(cb_hrs, 4);
        const uint32_t d = get_write_ptr(cb_hxs), rr = get_write_ptr(cb_hrs);
        for (uint32_t s = 0; s < 4; s++) {
            row0_from_l1(r.x32 + (s * D + tile * 32) * 4, d + s * 4096);
            fill_row0(rr + s * 4096, r.rs + s * 16);
        }
        noc_async_read_barrier();
        cb_push_back(cb_hxs, 4);
        cb_push_back(cb_hrs, 4);
    }
    wait(++epoch, lo, w + 64 - lo);
    if (tile != IDLE && tile < D / 32) {
        x_tiles(cb_hx2, lo, KHU, KHU);
        for (uint32_t s = 0, done = pre; s < 4; s++) {  // the rest of the 40 pages
            const uint32_t skip = done < KHU ? done : KHU;
            if (skip < KHU) {
                ws.read(wu, (s * D / 32 + tile) * KHU + skip, KHU - skip);
            }
            done -= skip;
        }
        ws.flush();
    }
    wait(++epoch, hn, D * 2);
}

// writer: H1's partial and r_s, arrive; HL's lo tile (or w), arrive; H2's hn tile, arrive.
// Checks: P1 [4 x 352], HN [2560], W [16] fp32 (none with -DNO_CHECKS: the token loop reads none of them).
template <typename Acc>
void write(const Grid& grid, uint32_t stage, const Acc& chk_p1, const Acc& chk_hn, const Acc& chk_w, uint32_t item,
           uint32_t tile, const Regions& r, uint32_t hn, uint32_t lo, uint32_t w, uint32_t& epoch) {
    if (item != IDLE) {
        const uint32_t n = item >> 2, s = item & 3;
        if (n == 0) {
            cb_wait_front(cb_hrout, 1);
            l1_copy(get_read_ptr(cb_hrout), stage + 512, 16);
            noc_async_read_barrier();
            cb_pop_front(cb_hrout, 1);
            grid.send(stage + 512, r.rs + s * 16, 16);
        }
        stage_rows(cb_hout, 1, 64, stage);
        grid.send(stage, r.p1 + (s * LO + n * 32) * 4, 128);
#ifndef NO_CHECKS
        noc_async_write(stage, chk_p1.get_noc_addr(0, (s * LO + n * 32) * 4), 128);
#endif
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);
    if (tile == LO / 32 - 1) {
        stage_rows(cb_hout, 1, 64, stage);
        grid.send(stage, w, 16);
#ifndef NO_CHECKS
        noc_async_write(stage, chk_w.get_noc_addr(0), 16);
#endif
        noc_async_write_barrier();
    } else if (tile != IDLE && tile < LO / 32 - 1) {
        stage_rows(cb_hlo, 1, 32, stage);
        grid.send(stage, lo + tile * 64, 64);
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);
    if (tile != IDLE && tile < D / 32) {
        stage_rows(cb_hhn, 1, 32, stage);
        grid.send(stage, hn + tile * 64, 64);
        noc_async_write_barrier();
#ifdef NO_CHECKS
        cb_wait_front(cb_hchk, 1);
        cb_pop_front(cb_hchk, 1);
#else
        stage_rows(cb_hchk, 1, 64, stage);
        noc_async_write(stage, chk_hn.get_noc_addr(0, tile * 128), 128);
        noc_async_write_barrier();
#endif
    }
    grid.arrive(++epoch);
}

}  // namespace fused::hc

// ---- M rows (a verify pass: rows 0..M-1 of every tile, M <= 4, the layout's row count Mp = 1, 2 or 4; M = 3 runs in
// the Mp = 4 layout with row 3 a copy of row 2). A vector region of n tiles holds each tile as a block of 2 Mp
// pieces: its face-0 parts of rows 0..Mp-1, then its face-1 parts (a piece: 16 values, 64 bytes fp32, 32 bf16), so a
// tile's rows go into a CB tile with one copy per face (rows 0..Mp-1 of a face are contiguous there too). At Mp = 1
// this is the row-0 layout above.
namespace fused {

// rows 0..mp-1 of an fp32 tile at dst from the block at L1 src
inline void rows_from_l1(uint32_t src, uint32_t dst, uint32_t mp) {
    l1_copy(src, dst, mp * 64);
    l1_copy(src + mp * 64, dst + 1024, mp * 64);
}

// the same from card memory (page `page` of src, from byte off)
template <typename Accessor>
inline void rows_from_dram(const Accessor& src, uint32_t page, uint32_t off, uint32_t dst, uint32_t mp) {
    noc_async_read(src.get_noc_addr(page, off), dst, mp * 64);
    noc_async_read(src.get_noc_addr(page, off + mp * 64), dst + 1024, mp * 64);
}

// n bf16 x tiles of mp rows from the region at L1 src into cb, pushed as `pushed` tiles
inline void x_tiles_m(uint32_t cb, uint32_t src, uint32_t n, uint32_t pushed, uint32_t mp) {
    cb_reserve_back(cb, pushed);
    const uint32_t xt = get_write_ptr(cb), b = mp * 32;
    for (uint32_t k = 0; k < n; k++) {
        l1_copy(src + k * 2 * b, xt + k * 2048, b);
        l1_copy(src + k * 2 * b + b, xt + k * 2048 + 512, b);
    }
    noc_async_read_barrier();
    cb_push_back(cb, pushed);
}

// row m of the fp32 tile at dst = the float bits at L1 addr
inline void fill_row(uint32_t dst, uint32_t m, uint32_t addr) {
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    const uint32_t bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t j = 0; j < 16; j++) {
        t[16 * m + j] = bits;
        t[256 + 16 * m + j] = bits;
    }
    asm volatile("fence rw, rw" ::: "memory");
}

// rows 0..mp-1 of the fp32 tile at dst: row m = the word at addr + m * stride
inline void fill_rows(uint32_t dst, uint32_t addr, uint32_t stride, uint32_t mp) {
    for (uint32_t m = 0; m < mp; m++) {
        fill_row(dst, m, addr + m * stride);
    }
}

// rows 0..mp-1 of the fp32 tile at dst = the row at L1 src (128 bytes: its face-0 half, then its face-1 half)
inline void row_down(uint32_t src, uint32_t dst, uint32_t mp) {
    for (uint32_t r = 0; r < mp; r++) {
        l1_copy(src, dst + r * 64, 64);
        l1_copy(src + 64, dst + 1024 + r * 64, 64);
    }
}

// the causal conv's window at mp rows (the writers' of Flash-Next's flayer and GLM's fkda): 4 fp32 tiles
// at dst, y (rows 0..mp-1 of the fp32 tile at y), then W_j (j = 0..2) with row r = input j + r of the history's 3 rows
// (at L1 hist, 128 bytes a row as row_down's) followed by y's rows. Complete after noc_async_read_barrier().
inline void conv_window_m(uint32_t y, uint32_t hist, uint32_t dst, uint32_t mp) {
    l1_copy(y, dst, mp * 64);
    l1_copy(y + 1024, dst + 1024, mp * 64);
    for (uint32_t j = 0; j < 3; j++) {
        const uint32_t t = dst + (1 + j) * 4096, r0 = 3 - j;  // rows from r0 on: y's rows 0..
        for (uint32_t r = 0; r < mp && r < r0; r++) {
            l1_copy(hist + (j + r) * 128, t + r * 64, 64);
            l1_copy(hist + (j + r) * 128 + 64, t + 1024 + r * 64, 64);
        }
        if (r0 < mp) {
            l1_copy(y, t + r0 * 64, (mp - r0) * 64);
            l1_copy(y + 1024, t + 1024 + r0 * 64, (mp - r0) * 64);
        }
    }
}

// writer: rows 0..mp-1 of n tiles of cb (fp32: piece 64, bf16: piece 32) as blocks into the run at stage, tile by tile
inline void stage_rows_m(uint32_t cb, uint32_t n, uint32_t piece, uint32_t stage, uint32_t mp) {
    const uint32_t tile = piece * 64, b = mp * piece;
    for (uint32_t k = 0; k < n; k++) {
        cb_wait_front(cb, 1);
        const uint32_t t = get_read_ptr(cb);
        l1_copy(t, stage + k * 2 * b, b);
        l1_copy(t + tile / 4, stage + k * 2 * b + b, b);
        noc_async_read_barrier();
        cb_pop_front(cb, 1);
    }
}

}  // namespace fused

namespace fused::hc {

// The mix at M rows. Regions (fused/hc_dm.h's above, mp-row blocks): xs32, xs16 this core's stream of X (H1's, fp32
// and bf16, 80 tile blocks), xown X's tile of this core's H2 item for the 4 streams (fp32, block s), p1 the partials
// (block 11 s + n) then rs (stream s: r_s of row m in word 16 m of mp * 64 bytes), lo (10 bf16 blocks) then w (row m
// in words 16 m .. 16 m + 3).
struct RegionsM {
    uint32_t xs32, xs16, xown, p1, rs, mp;
};

// reader, as read() above. H1's sums of squares: X_s's mp-row region as raw fp32 tiles in chunks of 3 (zero-padded),
// mp chunks (hc_cmp.h compute_m); the norm, r_s and inject weights on rows 0..mp-1.
template <typename Acc, typename Wait>
void read_m(Stream<Acc>& ws, const Acc& wd, const Acc& wu, const Acc& norm, uint32_t item, uint32_t tile,
            const RegionsM& r, uint32_t hn, uint32_t lo, uint32_t zero, Wait& wait, uint32_t& epoch,
            uint32_t h1_pre = 0) {
    const uint32_t mp = r.mp;
    if (item != IDLE) {
        const uint32_t n = item >> 2, s = item & 3, bytes = mp * D * 4;
        for (uint32_t c = 0; c < mp; c++) {
            cb_reserve_back(cb_hd, 3);
            const uint32_t d = get_write_ptr(cb_hd);
            for (uint32_t t = 0; t < 3; t++) {
                const uint32_t at = (3 * c + t) * 4096;
                const uint32_t real = at >= bytes ? 0 : (bytes - at < 4096 ? bytes - at : 4096);
                if (real) {
                    l1_copy(r.xs32 + at, d + t * 4096, real);
                }
                if (real < 4096) {
                    l1_copy(zero, d + t * 4096 + real, 4096 - real);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_hd, 3);
        }
        x_tiles_m(cb_x, r.xs16, KS, KS, mp);
        ws.read(wd, n * KW + s * KS + h1_pre, KS - h1_pre);
        ws.flush();
    }
    if (tile != IDLE && tile < D / 32) {
        cb_reserve_back(cb_hnrm, 4);
        const uint32_t d = get_write_ptr(cb_hnrm);
        for (uint32_t s = 0; s < 4; s++) {  // row 0 from card memory (one page: 80 cores read it), then copied down
            row0_from_dram(norm, 0, (s * D + tile * 32) * 4, d + s * 4096);
        }
        noc_async_read_barrier();
        for (uint32_t s = 0; s < 4; s++) {
            for (uint32_t m = 1; m < mp; m++) {
                l1_copy(d + s * 4096, d + s * 4096 + m * 64, 64);
                l1_copy(d + s * 4096 + 1024, d + s * 4096 + 1024 + m * 64, 64);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_hnrm, 4);
    }
    wait(++epoch, r.p1, mp * (4 * LO * 4 + 256));
    if (tile != IDLE && tile <= LO / 32 - 1) {
        cb_reserve_back(cb_hp, 4);
        const uint32_t d = get_write_ptr(cb_hp);
        for (uint32_t s = 0; s < 4; s++) {
            rows_from_l1(r.p1 + (s * (LO / 32) + tile) * mp * 128, d + s * 4096, mp);
        }
        noc_async_read_barrier();
        cb_push_back(cb_hp, 4);
    }
    const uint32_t pre = ws.batch * HC_PREFETCH_BATCHES;
    if (tile != IDLE && tile < D / 32) {
        for (uint32_t s = 0, left = pre; s < 4 && left; s++) {
            const uint32_t n = left < KHU ? left : KHU;
            ws.read(wu, (s * D / 32 + tile) * KHU, n);
            left -= n;
        }
        cb_reserve_back(cb_hxs, 4);
        cb_reserve_back(cb_hrs, 4);
        const uint32_t d = get_write_ptr(cb_hxs), rr = get_write_ptr(cb_hrs);
        for (uint32_t s = 0; s < 4; s++) {
            rows_from_l1(r.xown + s * mp * 128, d + s * 4096, mp);
            fill_rows(rr + s * 4096, r.rs + s * mp * 64, 64, mp);
        }
        noc_async_read_barrier();
        cb_push_back(cb_hxs, 4);
        cb_push_back(cb_hrs, 4);
    }
    wait(++epoch, lo, mp * 704);
    if (tile != IDLE && tile < D / 32) {
        x_tiles_m(cb_hx2, lo, KHU, KHU, mp);
        for (uint32_t s = 0, done = pre; s < 4; s++) {
            const uint32_t skip = done < KHU ? done : KHU;
            if (skip < KHU) {
                ws.read(wu, (s * D / 32 + tile) * KHU + skip, KHU - skip);
            }
            done -= skip;
        }
        ws.flush();
    }
    wait(++epoch, hn, mp * D * 2);
}

// writer, as write() above, mp-row blocks (the checks too: P1 [44 blocks], HN [80], W [mp x 16] fp32)
template <typename Acc>
void write_m(const Grid& grid, uint32_t stage, const Acc& chk_p1, const Acc& chk_hn, const Acc& chk_w, uint32_t item,
             uint32_t tile, const RegionsM& r, uint32_t hn, uint32_t lo, uint32_t& epoch) {
    const uint32_t mp = r.mp, w = lo + mp * 640;
    if (item != IDLE) {
        const uint32_t n = item >> 2, s = item & 3;
        if (n == 0) {
            stage_rows_m(cb_hrout, 1, 64, stage, mp);
            grid.send(stage, r.rs + s * mp * 64, mp * 64);
            noc_async_write_barrier();
        }
        stage_rows_m(cb_hout, 1, 64, stage, mp);
        const uint32_t at = (s * (LO / 32) + n) * mp * 128;
        grid.send(stage, r.p1 + at, mp * 128);
#ifndef NO_CHECKS
        noc_async_write(stage, chk_p1.get_noc_addr(0, at), mp * 128);
#endif
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);
    if (tile == LO / 32 - 1) {
        stage_rows_m(cb_hout, 1, 64, stage, mp);
        grid.send(stage, w, mp * 64);
#ifndef NO_CHECKS
        noc_async_write(stage, chk_w.get_noc_addr(0), mp * 64);
#endif
        noc_async_write_barrier();
    } else if (tile != IDLE && tile < LO / 32 - 1) {
        stage_rows_m(cb_hlo, 1, 32, stage, mp);
        grid.send(stage, lo + tile * mp * 64, mp * 64);
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);
    if (tile != IDLE && tile < D / 32) {
        stage_rows_m(cb_hhn, 1, 32, stage, mp);
        grid.send(stage, hn + tile * mp * 64, mp * 64);
        noc_async_write_barrier();
#ifdef NO_CHECKS
        cb_wait_front(cb_hchk, 1);
        cb_pop_front(cb_hchk, 1);
#else
        stage_rows_m(cb_hchk, 1, 64, stage, mp);
        noc_async_write(stage, chk_hn.get_noc_addr(0, tile * mp * 128), mp * 128);
        noc_async_write_barrier();
#endif
    }
    grid.arrive(++epoch);
}

}  // namespace fused::hc
