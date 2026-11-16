// Grouped expert writer (G2 prefill): each (expert, output tile n)'s MT tiles (cb 16) to y's pages (e * mt + m) * nt + n,
// REPS times over. Runtime args: y address, first expert, expert count. Compile args: reps, nt, mt, accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t mt = get_compile_time_arg_val(2);
    constexpr auto y_args = TensorAccessorArgs<3>();
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const uint32_t e_first = get_arg_val<uint32_t>(1);
    const uint32_t e_count = get_arg_val<uint32_t>(2);
    const uint32_t pb = get_local_cb_interface(16).fifo_page_size;

    Noc noc;
    DataflowBuffer dy(16);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t e = e_first; e < e_first + e_count; e++) {
            for (uint32_t n = 0; n < nt; n++) {
                dy.wait_front(mt);
                for (uint32_t m = 0; m < mt; m++) {
                    noc.async_write(dy, y, pb, {.offset_bytes = m * pb}, {.page_id = (e * mt + m) * nt + n});
                }
                noc.async_writes_flushed();
                dy.pop_front(mt);
            }
        }
    }
    noc.async_write_barrier();
}
