// Stream n_pages consecutive pages of an interleaved DRAM tensor into cb 0, BATCH pages per barrier (ttnn's
// unary reader waits after every page, so each page pays a full DRAM round trip).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);  // a multiple of BATCH
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id);
    for (uint32_t i = 0; i < num_pages; i += batch) {
        dfb.reserve_back(batch);
        for (uint32_t j = 0; j < batch; j++) {
            noc.async_read(s, dfb, page_bytes, {.page_id = start_id + i + j}, {.offset_bytes = j * page_bytes});
        }
        noc.async_read_barrier();
        dfb.push_back(batch);
    }
}
