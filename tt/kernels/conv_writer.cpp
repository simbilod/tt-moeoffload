// Causal depthwise conv writer (G2 prefill): y's tiles (i, j) down each column tile, REPS times over.
// Runtime args: y address, first column tile, count. Compile args: reps, rt, ct, accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t rt = get_compile_time_arg_val(1);
    constexpr uint32_t ct = get_compile_time_arg_val(2);
    constexpr auto y_args = TensorAccessorArgs<3>();
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const uint32_t j0 = get_arg_val<uint32_t>(1), jn = get_arg_val<uint32_t>(2);
    const uint32_t pb = get_local_cb_interface(16).fifo_page_size;

    Noc noc;
    DataflowBuffer dy(16);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t j = j0; j < j0 + jn; j++) {
            for (uint32_t i = 0; i < rt; i++) {
                dy.wait_front(1);
                noc.async_write(dy, y, pb, {}, {.page_id = i * ct + j});
                noc.async_writes_flushed();
                dy.pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
}
