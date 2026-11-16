// GLM mHC post writer (prefill): each item's 4 new stream tiles (cb 16, in stream order) to Y's pages
// (i * 4 + s) * ct + j (Y laid out as X). Runtime args: Y address, first item, count. Compile args: rt, ct, pre,
// accessor. pre = 1: one tile an item, to h's page i * ct + j.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t pre = get_compile_time_arg_val(2);
    constexpr uint32_t NO = pre ? 1 : 4;
    constexpr auto y_args = TensorAccessorArgs<3>();
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    const uint32_t pb = get_local_cb_interface(16).fifo_page_size;

    Noc noc;
    DataflowBuffer dy(16);
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / ct, j = it % ct;
        dy.wait_front(NO);
        for (uint32_t s = 0; s < NO; s++) {
            noc.async_write(dy, y, pb, {.offset_bytes = s * pb}, {.page_id = pre ? i * ct + j : (i * 4 + s) * ct + j});
        }
        noc.async_writes_flushed();
        dy.pop_front(NO);
    }
    noc.async_write_barrier();
}
