// GLM indexer score writer: glm_iscore_compute.cpp's score tiles (cb 16, one an item (q, p)) to score [T, P]'s
// pages. Runtime args: score address, first item, count. Compile args: PT, accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PT = get_compile_time_arg_val(0);
    constexpr auto s_args = TensorAccessorArgs<1>();
    const auto score = TensorAccessor(s_args, get_arg_val<uint32_t>(0), 4096);
    const uint32_t first = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    for (uint32_t it = first; it < first + count; it++) {
        cb_wait_front(16, 1);
        noc_async_write(get_read_ptr(16), score.get_noc_addr(it), 4096);   // item (q, p) is page q PT + p
        noc_async_writes_flushed();
        cb_pop_front(16, 1);
    }
    noc_async_write_barrier();
}
