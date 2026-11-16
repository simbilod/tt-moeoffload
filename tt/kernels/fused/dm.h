// Dataflow-side helpers for the fused layer programs (reader and writer). Row-0 tiles: 32 values in row 0, as two
// pieces in faces 0 and 1 (fp32: 64 bytes at 0 and 1024; bf16: 32 bytes at 0 and 512).
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

namespace fused {

// a weight stream into a CB in batches; flush() pushes a phase's partial last batch (the compute pops it whole)
template <typename Accessor>
struct Stream {
    uint32_t cb, batch, page, in_batch = 0, dst = 0;
    void read(const Accessor& src, uint32_t first, uint32_t pages) {
        for (uint32_t p = first; p < first + pages; p++) {
            if (in_batch == 0) {
                cb_reserve_back(cb, batch);
                dst = get_write_ptr(cb);
            }
            noc_async_read(src.get_noc_addr(p), dst + in_batch * page, page);
            if (++in_batch == batch) {
                noc_async_read_barrier();
                cb_push_back(cb, batch);
                in_batch = 0;
            }
        }
    }
    // one page from any NOC address (e.g. this core's L1, where it was prefetched), batched as read() does
    void read_page(uint64_t noc_addr) {
        if (in_batch == 0) {
            cb_reserve_back(cb, batch);
            dst = get_write_ptr(cb);
        }
        noc_async_read(noc_addr, dst + in_batch * page, page);
        if (++in_batch == batch) {
            noc_async_read_barrier();
            cb_push_back(cb, batch);
            in_batch = 0;
        }
    }
    void flush() {
        if (in_batch) {
            noc_async_read_barrier();
            cb_push_back(cb, batch);
            in_batch = 0;
        }
    }
};

// row 0 of an fp32 tile at dst from 32 consecutive floats at L1 src
inline void row0_from_l1(uint32_t src, uint32_t dst) {
    l1_copy(src, dst, 64);
    l1_copy(src + 64, dst + 1024, 64);
}

// row 0 of an fp32 tile at dst from 32 consecutive floats of page `row` of a row-major tensor, from byte offset off
template <typename Accessor>
inline void row0_from_dram(const Accessor& src, uint32_t row, uint32_t off, uint32_t dst) {
    noc_async_read(src.get_noc_addr(row, off), dst, 64);
    noc_async_read(src.get_noc_addr(row, off + 64), dst + 1024, 64);
}

// n bf16 x tiles (row 0) from the bf16 vector at L1 src into cb, pushed as `pushed` tiles (>= n: padding keeps a
// buffer read by index aligned)
inline void x_tiles(uint32_t cb, uint32_t src, uint32_t n, uint32_t pushed) {
    cb_reserve_back(cb, pushed);
    const uint32_t xt = get_write_ptr(cb);
    for (uint32_t k = 0; k < n; k++) {
        l1_copy(src + k * 64, xt + k * 2048, 32);
        l1_copy(src + k * 64 + 32, xt + k * 2048 + 512, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb, pushed);
}

// the x ring (GLM's programs): row 0 of the x tiles k0 .. k0 + n from the bf16 vector at L1 xrow (64 bytes a tile),
// copied by this RISC-V into n tiles of cb (faces 0 and 1) and pushed: a ring of 2 x batch tiles in L1 instead of
// all K / 32 resident (x_tiles' NoC copies, by word)
inline void x_ring(uint32_t cb, uint32_t xrow, uint32_t k0, uint32_t n) {
    cb_reserve_back(cb, n);
    const uint32_t xt = get_write_ptr(cb);
    for (uint32_t j = 0; j < n; j++) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xrow + (k0 + j) * 64);
        volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xt + j * 2048);
        volatile tt_l1_ptr uint32_t* f1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xt + j * 2048 + 512);
        for (uint32_t e = 0; e < 8; e++) {
            f0[e] = src[e];
            f1[e] = src[8 + e];
        }
    }
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb, n);
}

// the x ring at mp rows (mp > 1) from the bf16 region at L1 xblk in fused/hc_dm.h's block layout (tile k: its face-0
// parts of rows 0..mp-1, then its face-1 parts), by the NoC (x_ring's word copies, ~11 ns a word, cost +500 us per KDA
// mixer at 4 rows): reserves n tiles of cb and issues their copies; the caller pushes them after its next noc_async_read_barrier()
// (the weight batch's: fused::Stream waits at every batch)
inline void x_ring_m_issue(uint32_t cb, uint32_t xblk, uint32_t k0, uint32_t n, uint32_t mp) {
    cb_reserve_back(cb, n);
    const uint32_t xt = get_write_ptr(cb), b = mp * 32;
    for (uint32_t j = 0; j < n; j++) {
        l1_copy(xblk + (k0 + j) * 2 * b, xt + j * 2048, b);
        l1_copy(xblk + (k0 + j) * 2 * b + b, xt + j * 2048 + 512, b);
    }
}

// x_tiles (mp = 1) or hc_dm.h's x_tiles_m by the RISC-V's own stores, the same bytes: no read barrier, so reads this
// core still has in flight (the next pass's A prefetch) do not hold it up
inline void x_tiles_cpu(uint32_t cb, uint32_t src, uint32_t n, uint32_t pushed, uint32_t mp = 1) {
    cb_reserve_back(cb, pushed);
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");
    const uint32_t w = mp * 8;  // a face's mp rows of 16 bf16, in words
    const volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t j = 0; j < w; j++) {
            t[k * 512 + j] = s[k * 2 * w + j];
            t[k * 512 + 128 + j] = s[k * 2 * w + w + j];
        }
    }
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb, pushed);
}

// the float bits at L1 addr across row 0 of the fp32 tile at dst (the RISC-V copies the word; no float math)
inline void fill_row0(uint32_t dst, uint32_t addr) {
    invalidate_l1_cache();
    const uint32_t bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t j = 0; j < 16; j++) {
        t[j] = bits;
        t[256 + j] = bits;
    }
    asm volatile("fence rw, rw" ::: "memory");  // a precaution: the stores land before a CB push hands the tile on
}

inline void fill_words(uint32_t dst, uint32_t words, uint32_t bits) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t j = 0; j < words; j++) {
        t[j] = bits;
    }
}

// writer: row 0 of n tiles of cb (fp32: piece 64, bf16: piece 32) into the run at stage, 2 * piece bytes per tile;
// tile by tile, so a run that wraps around the buffer's end reads right (indexed access does not wrap)
inline void stage_rows(uint32_t cb, uint32_t n, uint32_t piece, uint32_t stage) {
    const uint32_t tile = piece * 64;
    for (uint32_t k = 0; k < n; k++) {
        cb_wait_front(cb, 1);
        const uint32_t t = get_read_ptr(cb);
        l1_copy(t, stage + k * 2 * piece, piece);
        l1_copy(t + tile / 4, stage + k * 2 * piece + piece, piece);
        noc_async_read_barrier();
        cb_pop_front(cb, 1);
    }
}

}  // namespace fused
