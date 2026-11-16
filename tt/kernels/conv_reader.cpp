// Causal depthwise conv reader (G2 prefill): one core's run of column tiles of x [T, C] (the DeltaNet's q|k|v before
// the conv). Once: the 7 shift tiles (I, L1..L3, U1..U3) into cb 5. Per column tile j: the 4 tap rows (tile k * ct + j,
// tap k in row 0) into cb 1, the conv state tile (the previous chunk's last 3 tokens in rows 29..31) into cb 2, then
// x's tiles (i, j) down the column into cb 0, BATCH per barrier. REPS times over.
// Runtime args: x, taps, state, shift addresses; first column tile, count. Compile args: reps, rt, ct, x_ct (x's row
// stride in tiles: x may be the wider projection output, its first ct column tiles read in place), batch, accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <typename Accessor>
void read_pages(Noc& noc, const Accessor& s, DataflowBuffer& d, uint32_t page_bytes, uint32_t first, uint32_t stride,
                uint32_t n) {
    d.reserve_back(n);
    for (uint32_t i = 0; i < n; i++) {
        noc.async_read(s, d, page_bytes, {.page_id = first + i * stride}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    d.push_back(n);
}

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t rt = get_compile_time_arg_val(1);
    constexpr uint32_t ct = get_compile_time_arg_val(2);
    constexpr uint32_t x_ct = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sh_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const auto st = TensorAccessor(s_args, get_arg_val<uint32_t>(2));
    const auto sh = TensorAccessor(sh_args, get_arg_val<uint32_t>(3));
    const uint32_t j0 = get_arg_val<uint32_t>(4), jn = get_arg_val<uint32_t>(5);
    if (jn == 0) {
        return;
    }

    Noc noc;
    DataflowBuffer dx(0), dw(1), ds(2), dsh(5);
    const uint32_t xb = get_local_cb_interface(0).fifo_page_size, wb = get_local_cb_interface(1).fifo_page_size;
    const uint32_t sb = get_local_cb_interface(2).fifo_page_size, shb = get_local_cb_interface(5).fifo_page_size;
    read_pages(noc, sh, dsh, shb, 0, 1, 7);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t j = j0; j < j0 + jn; j++) {
            read_pages(noc, w, dw, wb, j, ct, 4);
            read_pages(noc, st, ds, sb, j, 1, 1);
            for (uint32_t i = 0; i < rt; i += batch) {
                read_pages(noc, x, dx, xb, i * x_ct + j, x_ct, batch);
            }
        }
    }
}
