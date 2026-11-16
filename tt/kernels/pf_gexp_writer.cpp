// Prefill experts writer (gexp_writer.cpp's, for units): each unit's output tiles (cb 16, one page a push: the
// units' tile counts differ, and a push must not straddle the buffer's end) to y's pages (x0 + m) * nt + n.
// SCATTER (down's output, fp32): each row of a unit's tile to its own row of y instead, the row d = dest[its row]
// (a route's row j * T + t; 0xFFFFFFFF: a padding row, not written), as the two 64-byte halves of the row in their
// faces (rows 0-15 in faces 0 and 1, 16-31 in faces 2 and 3): y then holds every route's output in token order and
// the sum over the routes needs no gather. dest: uint32 [R / 32, 32] row-major (a page per tile row of rows), read
// a unit at a time into cb 17.
// The scales too (step 3): per n-tile of the compute, in its order, the column's compact scale page (the GGUF
// blocks' fp16 d [rows / per][32], then HAS_M m as bfp8_b rows; g7_experts.compact_pages) read into cb 27, and per
// 4 k-tiles of a code tile their rows for the compute's row broadcasts: HAS_M (gate|up) 4 m rows (cb 21, bfp8_b) and
// the block's d row (cb 22, fp32), else 4 d rows (zero past `rows`). Copies and fp16 -> fp32 only; while a
// row CB is full the finished output tiles are written, so neither side waits on the other for long.
// Runtime args: y address, dest address (SCATTER; else unused), unit count, then per unit: first token tile, token
// tiles, its compact scales' base, the expert within the piece. Compile args: nt, y's row of tiles (>= nt: SwiGLU's h
// is written into the down's padded [R, 768], the padding tiles of each unit's rows written zero from cb 20, one
// page: y needs no zero fill), y's accessor; SCATTER: dest's accessor; then the compute's n-tiles, kc, rows, per,
// the compact scales' accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "compact_scales.h"
#ifdef TT_MOEOFFLOAD_ZONES
#include "api/debug/kernel_profiler.hpp"
#endif

constexpr uint32_t cb_out = 16, cb_mr = 21, cb_dr = 22, cb_q = 27;

void kernel_main() {
    constexpr uint32_t nt = get_compile_time_arg_val(0);
    constexpr uint32_t ns = get_compile_time_arg_val(1);
    constexpr auto y_args = TensorAccessorArgs<2>();
#ifdef SCATTER
    constexpr auto d_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t o = d_args.next_compile_time_args_offset();
#else
    constexpr uint32_t o = y_args.next_compile_time_args_offset();
#endif
    constexpr uint32_t NC = get_compile_time_arg_val(o), kc = get_compile_time_arg_val(o + 1);
    constexpr uint32_t rows = get_compile_time_arg_val(o + 2), per = get_compile_time_arg_val(o + 3);
    constexpr auto s_args = TensorAccessorArgs<o + 4>();
    const uint32_t pb = get_local_cb_interface(cb_out).fifo_page_size;
    const uint32_t qb = get_local_cb_interface(cb_q).fifo_page_size;
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0), pb);
    const uint32_t n_units = get_arg_val<uint32_t>(2);
    lead_table();
    cb_reserve_back(cb_q, 1);                 // the compact page's scratch (never pushed)
    const uint32_t qp = get_write_ptr(cb_q);
    const uint16_t* qd = reinterpret_cast<const uint16_t*>(qp);
#ifdef HAS_M
    const uint8_t* qe = reinterpret_cast<const uint8_t*>(qp + 64 * (rows / per));    // m's bfp8_b exponents [rows][2]
    const uint32_t* qm = reinterpret_cast<const uint32_t*>(qp + 64 * (rows / per) + 2 * rows);  // mantissas [rows][32]
#endif
#ifdef SCATTER
    const auto dest = TensorAccessor(d_args, get_arg_val<uint32_t>(1), 128);
    const uint32_t drows = get_write_ptr(17);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drows);
#else
    uint32_t zero = 0;
    if constexpr (ns > nt) {
        zero = get_write_ptr(20);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
        for (uint32_t i = 0; i < pb / 4; i++) {
            z[i] = 0;
        }
    }
#endif
    for (uint32_t u = 0; u < n_units; u++) {
        const uint32_t x0 = get_arg_val<uint32_t>(3 + 4 * u), mt = get_arg_val<uint32_t>(4 + 4 * u);
        const auto scales = TensorAccessor(s_args, get_arg_val<uint32_t>(5 + 4 * u), qb);
        const uint32_t e = get_arg_val<uint32_t>(6 + 4 * u);
#ifdef SCATTER
        for (uint32_t m = 0; m < mt; m++) {
            noc_async_read(dest.get_noc_addr(x0 + m), drows + m * 128, 128);
        }
        noc_async_read_barrier();
        asm volatile("fence rw, rw" ::: "memory");
#endif
        // the unit's output tiles, in the compute's order (tile k: n = k / mt, m = k % mt), as they come
        const uint32_t total = nt * mt;
        uint32_t k = 0;
        auto emit = [&]() {
            const uint32_t n = k / mt, m = k % mt;
            const uint32_t src = get_read_ptr(cb_out);
#ifdef SKIP_OUT                                 // timing only: the output tiles not written
            (void)n;
            (void)src;
#elif defined(SCATTER)
#ifdef SKIP_SCATTER                         // timing only: the tile whole to the page of its first route row
            for (uint32_t r = 0; r < 32; r++) {
                if (d[m * 32 + r] != 0xFFFFFFFF) {
                    noc_async_write(src, y.get_noc_addr((d[m * 32 + r] >> 5) * nt + n), pb);
                    break;
                }
            }
#else
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t row = d[m * 32 + r];
                if (row == 0xFFFFFFFF) {
                    continue;
                }
                const uint32_t page = (row >> 5) * nt + n, dr = row & 31;
                for (uint32_t h = 0; h < 2; h++) {
                    const uint32_t so = (((r >> 4) << 1) + h) * 1024 + (r & 15) * 64;
                    const uint32_t dof = (((dr >> 4) << 1) + h) * 1024 + (dr & 15) * 64;
                    noc_async_write(src + so, y.get_noc_addr(page, dof), 64);
                }
            }
#endif
#else
            noc_async_write(src, y.get_noc_addr((x0 + m) * ns + n), pb);
#endif
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
            k++;
        };
        auto drain = [&]() {
            while (k < total && cb_pages_available_at_front(cb_out, 1)) {
                emit();
            }
        };
        auto reserve = [&](uint32_t cb) {
            while (!cb_pages_reservable_at_back(cb, 1)) {
                drain();
            }
            cb_reserve_back(cb, 1);
            return get_write_ptr(cb);
        };
        for (uint32_t n = 0; n < NC; n++) {
#ifdef SWIGLU
            const uint32_t col = e * NC + ((n & 1) ? NC / 2 + n / 2 : n / 2);   // pf_gexp_reader.cpp's order
#else
            const uint32_t col = e * NC + n;
#endif
            auto rows_col = [&]() {
#if !(defined(SKIP_ROWPUSH) && !defined(VALUES))   // timing only: no rows (the compute takes none)
#ifndef SKIP_SCALES                            // timing only: the column's scale page not read
            noc_async_read(scales.get_noc_addr(col), qp, qb);
            noc_async_read_barrier();
#endif
            invalidate_l1_cache();            // the page came over the NOC: no stale cached lines of the last one
            asm volatile("fence rw, rw" ::: "memory");
            for (uint32_t c = 0; c < kc; c++) {
                for (uint32_t pass = 0; pass < 2; pass++) {
                    const uint32_t kt0 = 8 * c + 4 * pass;
#ifdef HAS_M
                    for (uint32_t i = 0; i < 4; i++) {   // m rows: row 0 of a bfp8_b tile, copied as they come
                        uint8_t* r = reinterpret_cast<uint8_t*>(reserve(cb_mr));
                        const uint32_t kb = kt0 + i;
#ifndef SKIP_ROWS                    // timing only: the rows pushed unconverted
                        r[0] = qe[2 * kb];               // face 0 row 0's exponent, face 1 row 0's
                        r[16] = qe[2 * kb + 1];
                        uint32_t* f0 = reinterpret_cast<uint32_t*>(r + 64);         // the mantissas, faces 0 and 1
                        uint32_t* f1 = reinterpret_cast<uint32_t*>(r + 64 + 256);
                        const uint32_t* src = qm + 8 * kb;
                        for (uint32_t j = 0; j < 4; j++) {
                            f0[j] = src[j];
                            f1[j] = src[4 + j];
                        }
#endif
                        asm volatile("fence rw, rw" ::: "memory");
                        cb_push_back(cb_mr, 1);
                    }
                    {                                    // the block's d row: row 0 of an fp32 tile
                        uint32_t* r = reinterpret_cast<uint32_t*>(reserve(cb_dr));
                        const uint16_t* dr = qd + (kt0 / per) * 32;
#ifndef SKIP_ROWS                    // timing only: the rows pushed unconverted
                        for (uint32_t j = 0; j < 16; j++) {
                            r[j] = f16_bits(dr[j]);
                            r[256 + j] = f16_bits(dr[16 + j]);
                        }
#endif
                        asm volatile("fence rw, rw" ::: "memory");
                        cb_push_back(cb_dr, 1);
                    }
#else
                    for (uint32_t i = 0; i < 4; i++) {   // d rows (per = 1), zero for the padding k-tiles
                        uint32_t* r = reinterpret_cast<uint32_t*>(reserve(cb_dr));
                        const uint32_t kb = kt0 + i;
                        const uint16_t* dr = qd + kb * 32;
#ifndef SKIP_ROWS                    // timing only: the rows pushed unconverted
                        for (uint32_t j = 0; j < 16; j++) {
                            r[j] = kb < rows ? f16_bits(dr[j]) : 0;
                            r[256 + j] = kb < rows ? f16_bits(dr[16 + j]) : 0;
                        }
#endif
                        asm volatile("fence rw, rw" ::: "memory");
                        cb_push_back(cb_dr, 1);
                    }
#endif
                }
            }
#endif
            };
#ifdef TT_MOEOFFLOAD_ZONES
            if (u == 0 && n < 2) {
                DeviceZoneScopedN("GX-W");
                rows_col();
            } else {
                rows_col();
            }
#else
            rows_col();
#endif
            drain();
        }
        while (k < total) {
            cb_wait_front(cb_out, 1);
            emit();
        }
#if !defined(SCATTER) && !defined(SKIP_OUT)
        for (uint32_t n = nt; n < ns; n++) {
            for (uint32_t m = 0; m < mt; m++) {
                noc_async_write(zero, y.get_noc_addr((x0 + m) * ns + n), pb);
            }
        }
#endif
    }
    noc_async_write_barrier();
}
