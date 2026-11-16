// Prefill experts reader (gexpq_reader.cpp's, for a chunk's routed tokens): one core's run of units, a unit =
// (expert, up to MT_MAX token tiles of it). Per unit: its mt token tiles x kt k-tiles into cb 1 (padded to MT_MAX x kt
// pushes, so the index-read buffer always sees whole batches from its start); then per n-tile its weights from the
// unit's own weight tensors (a piece of the layer's experts, resident or streamed in: base addresses per unit) in
// the card's form of the GGUF quant: kc code tiles into cb 0 (uint32 words, nibble t of word [k][n] = the code of
// k-tile 8c + t). VALUES (Q8_0): instead of a code tile, its 8 k-tiles' values as bfp8_b tiles into cb 25. The
// scales are the writer's (pf_gexp_writer.cpp: the compact scales' m and d rows).
// SWIGLU (gate|up): the column tiles in the order gate 0, up 0, gate 1, up 1, ...
// Runtime args: x address, unit count, then per unit: codes (or values) base, scales base (unused here), expert
// within the piece, first token tile, token tiles. Compile args: kt, nt, mt_max, kc, kq, the weights' accessor (any
// interleaved DRAM tensor: the pieces are found by base address), x's.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#ifdef TT_MOEOFFLOAD_ZONES
#include "api/debug/kernel_profiler.hpp"
#endif

#ifndef READ_BATCH
#define READ_BATCH 1
#endif

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t mt_max = get_compile_time_arg_val(2);
    constexpr uint32_t kc = get_compile_time_arg_val(3);
    constexpr uint32_t kq = get_compile_time_arg_val(4);
    constexpr auto w_args = TensorAccessorArgs<5>();
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t n_units = get_arg_val<uint32_t>(1);

    Noc noc;
    DataflowBuffer dx(1);
#ifdef VALUES
    DataflowBuffer dv(25);
    const uint32_t vb = get_local_cb_interface(25).fifo_page_size;
#else
    DataflowBuffer dc(0);
    const uint32_t cb = get_local_cb_interface(0).fifo_page_size;
#endif
    const uint32_t xb = get_local_cb_interface(1).fifo_page_size;
    for (uint32_t u = 0; u < n_units; u++) {
        const uint32_t a = 2 + 5 * u;
        const uint32_t e = get_arg_val<uint32_t>(a + 2), x0 = get_arg_val<uint32_t>(a + 3);
        const uint32_t mt = get_arg_val<uint32_t>(a + 4);
#ifdef VALUES
        const auto codes = TensorAccessor(w_args, get_arg_val<uint32_t>(a), vb);
#else
        const auto codes = TensorAccessor(w_args, get_arg_val<uint32_t>(a), cb);
#endif
        dx.reserve_back(mt_max * kt);
        auto read_x = [&]() {
#ifndef SKIP_X                                 // timing only: x's tiles not read
            for (uint32_t i = 0; i < mt * kt; i++) {
                noc.async_read(x, dx, xb, {.page_id = x0 * kt + i}, {.offset_bytes = i * xb});
            }
#endif
            noc.async_read_barrier();
        };
#ifdef TT_MOEOFFLOAD_ZONES
        if (u == 0) {
            DeviceZoneScopedN("GX-X");
            read_x();
        } else {
            read_x();
        }
#else
        read_x();
#endif
        dx.push_back(mt_max * kt);
        for (uint32_t n = 0; n < nt; n++) {
#ifdef SWIGLU
            // gate and up column tiles in turn (gate n / 2, then up n / 2: the compute's SwiGLU pairs them)
            const uint32_t col = e * nt + ((n & 1) ? nt / 2 + n / 2 : n / 2);
#else
            const uint32_t col = e * nt + n;
#endif
#ifdef VALUES
            for (uint32_t c = 0; c < kc; c++) {
                dv.reserve_back(8);
                for (uint32_t t = 0; t < 8; t++) {
                    noc.async_read(codes, dv, vb, {.page_id = col * kt + 8 * c + t}, {.offset_bytes = t * vb});
                }
                noc.async_read_barrier();
                dv.push_back(8);
            }
#else
            // READ_BATCH code tiles in flight under one barrier (2026-11-09: one 4 KB read a barrier left each core's
            // reads latency-bound); READ_BATCH divides kc and cb 0 holds 2 READ_BATCH pages, so no batch wraps it
            static_assert(kc % READ_BATCH == 0, "READ_BATCH divides kc");
            for (uint32_t c = 0; c < kc; c += READ_BATCH) {
                constexpr uint32_t b = READ_BATCH;
                dc.reserve_back(b);
#ifndef SKIP_CODES                             // timing only: the code tiles not read
                for (uint32_t t = 0; t < b; t++) {
                    noc.async_read(codes, dc, cb, {.page_id = col * kc + c + t}, {.offset_bytes = t * cb});
                }
#endif
                noc.async_read_barrier();
                dc.push_back(b);
            }
#endif
        }
    }
}
