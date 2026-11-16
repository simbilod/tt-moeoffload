// GLM's multi-phase one-token GEMV, reader: P phases in one program, each y_p = x_p W_p over the grid's
// cores, a barrier between phases (fused/grid.h). Phase p's x is the DRAM input row (p = 0) or an earlier phase's y,
// gathered in every core's arena by the barrier's multicast. W: all phases' bfp8_b tiles in one tensor, each phase's in
// the cores' read order (page wbase + n * kt + k = W_p's tile (k, n)). x's tiles are not kept: for each weight batch
// the reader writes row 0 of BATCH x tiles in a ring (64 bytes a tile; rows 1-31 reach only output rows nobody reads),
// so x costs 2 * BATCH tiles of L1 at any K (gemv_reader.cpp keeps all K / 32). The first weight batch of a phase is
// read before its barrier is released.
// Runtime args: the grid block (grid.h), w address, x address, then per phase 8 words: kt, first output tile, count,
// w base page, x source (0: the DRAM row, 1: the arena), x's arena offset (bytes), and the barrier's multicast: the
// arena offset and bytes of the previous phase's y. Compile args: P, BATCH, then w's and x's accessor args.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "fused/grid.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_xrow = 3, cb_arena = 8;

void kernel_main() {
    constexpr uint32_t P = get_compile_time_arg_val(0);
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    constexpr auto w_args = TensorAccessorArgs<2>();
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const fused::Grid g = fused::Grid::from_args(0);
    uint32_t a = fused::Grid::args(g.nrects);
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(a));
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(a + 1));
    a += 2;
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t xrow_dram = get_write_ptr(cb_xrow);
    const uint32_t page_bytes = get_local_cb_interface(cb_w).fifo_page_size;
    Noc noc;
    DataflowBuffer dw(cb_w);

    auto read_w = [&](uint32_t page0) {
        dw.reserve_back(batch);
        for (uint32_t j = 0; j < batch; j++) {
            noc.async_read(w, dw, page_bytes, {.page_id = page0 + j}, {.offset_bytes = j * page_bytes});
        }
    };

    for (uint32_t p = 0; p < P; p++, a += 8) {
        const uint32_t kt = get_arg_val<uint32_t>(a), first = get_arg_val<uint32_t>(a + 1);
        const uint32_t count = get_arg_val<uint32_t>(a + 2), wbase = get_arg_val<uint32_t>(a + 3);
        const uint32_t xsrc = get_arg_val<uint32_t>(a + 4), xoff = get_arg_val<uint32_t>(a + 5);
        const uint32_t mc_off = get_arg_val<uint32_t>(a + 6), mc_bytes = get_arg_val<uint32_t>(a + 7);
        const uint32_t page0 = wbase + first * kt;
        if (count) read_w(page0);                     // the first batch before the barrier: it needs no x
        if (p > 0) g.wait(p, arena + mc_off, mc_bytes);
        uint32_t xrow = arena + xoff;
        if (xsrc == 0) {
            noc_async_read(x.get_noc_addr(0), xrow_dram, kt * 64);
            noc_async_read_barrier();
            xrow = xrow_dram;
        }
        for (uint32_t n = 0; n < count; n++) {
            for (uint32_t c = 0; c < kt; c += batch) {
                if (n || c) read_w(page0 + n * kt + c);
                cb_reserve_back(cb_x, batch);
                const uint32_t xt = get_write_ptr(cb_x);
                for (uint32_t j = 0; j < batch; j++) {
                    volatile tt_l1_ptr uint32_t* src =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xrow + (c + j) * 64);
                    volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xt + j * 2048);
                    volatile tt_l1_ptr uint32_t* f1 =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xt + j * 2048 + 512);
                    for (uint32_t i = 0; i < 8; i++) {
                        f0[i] = src[i];
                        f1[i] = src[8 + i];
                    }
                }
                noc.async_read_barrier();
                dw.push_back(batch);
                cb_push_back(cb_x, batch);
            }
        }
    }
}
