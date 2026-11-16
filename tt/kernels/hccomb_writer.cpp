// Hyper-connection combine writer (prefill): each item's 4 stream tiles (cb 16, in stream order) to X's pages
// (c * rt + i) * ct + j, in place (a tile is read and written by the same core, the read first).
// Runtime args: X address, first item, count. Compile args: rt, ct, accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr auto y_args = TensorAccessorArgs<2>();
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    const uint32_t pb = get_local_cb_interface(16).fifo_page_size;

    Noc noc;
    DataflowBuffer dy(16);
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / ct, j = it % ct;
        dy.wait_front(4);
        for (uint32_t c = 0; c < 4; c++) {
            noc.async_write(dy, y, pb, {.offset_bytes = c * pb}, {.page_id = (c * rt + i) * ct + j});
        }
        noc.async_writes_flushed();
        dy.pop_front(4);
    }
    noc.async_write_barrier();
}
