// Hyper-connection combine reader (prefill): X_c += w_c * out over the 4 streams, X fp32 [1, 4, T, 2560] stream-
// major (tile (c, i, j) at page (c * rt + i) * ct + j), out [1, 1, T, 2560], w [1, 4, T, 32] (each stream's inject
// weight per row, widened to a tile: page c * rt + i). One core's run of items (i, j) = out's tiles, row-major (item =
// i * ct + j): per new row i the 4 streams' w tiles into cb 2; per item out's tile into cb 1 and the 4 streams' X
// tiles into cb 0. Runtime args: X, out, w addresses; first item, count. Compile args: rt, ct, accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto o = TensorAccessor(o_args, get_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2));
    const uint32_t first = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);

    Noc noc;
    DataflowBuffer dx(0), dout(1), dw(2);
    const uint32_t pb = get_local_cb_interface(0).fifo_page_size;
    uint32_t i_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / ct, j = it % ct;
        if (i != i_prev) {
            dw.reserve_back(4);
            for (uint32_t c = 0; c < 4; c++) {
                noc.async_read(w, dw, pb, {.page_id = c * rt + i}, {.offset_bytes = c * pb});
            }
            noc.async_read_barrier();
            dw.push_back(4);
            i_prev = i;
        }
        dout.reserve_back(1);
        dx.reserve_back(4);
        noc.async_read(o, dout, pb, {.page_id = i * ct + j}, {});
        for (uint32_t c = 0; c < 4; c++) {
            noc.async_read(x, dx, pb, {.page_id = (c * rt + i) * ct + j}, {.offset_bytes = c * pb});
        }
        noc.async_read_barrier();
        dout.push_back(1);
        dx.push_back(4);
    }
}
