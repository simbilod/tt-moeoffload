// G2 GEMV writer: each output tile (cb 16) to its page, REPS times over (the same pages).
// Runtime args: output address, first output tile, output tile count. Compile args: reps, out.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t n_first = get_arg_val<uint32_t>(1);
    const uint32_t n_count = get_arg_val<uint32_t>(2);
    const uint32_t page_bytes = get_local_cb_interface(cb_out).fifo_page_size;

    Noc noc;
    DataflowBuffer dfb(cb_out);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t n = n_first; n < n_first + n_count; n++) {
            dfb.wait_front(1);
            noc.async_write(dfb, out, page_bytes, {}, {.page_id = n});
            noc.async_writes_flushed();
            dfb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
