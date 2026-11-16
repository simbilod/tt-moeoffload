// Hyper-connection gate-sum writer (G2 prefill): each item's tile (cb 16) to mixed's page i * ct + j (item = j * rt + i).
// Runtime args: y address, first item, count. Compile args: rt, ct, accessor.
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
        dy.wait_front(1);
        noc.async_write(dy, y, pb, {}, {.page_id = (it % rt) * ct + it / rt});
        noc.async_writes_flushed();
        dy.pop_front(1);
    }
    noc.async_write_barrier();
}
