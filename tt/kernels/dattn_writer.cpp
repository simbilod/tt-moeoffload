// Decode QSA partial writer: the core's O (8 tiles, fp32), m and l (row tiles) to partials pages part * 10 + 0..9.
// Runtime args: partials address, partial index (0xffffffff = idle). Compile args: accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    const auto o = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t part = get_arg_val<uint32_t>(1);
    if (part == 0xffffffff) {
        return;
    }
    const uint32_t pb = get_local_cb_interface(16).fifo_page_size;
    Noc noc;
    DataflowBuffer dy(16);
    dy.wait_front(8);
    for (uint32_t i = 0; i < 8; i++) {
        noc.async_write(dy, o, pb, {.offset_bytes = i * pb}, {.page_id = part * 10 + i});
    }
    noc.async_writes_flushed();
    dy.pop_front(8);
    dy.wait_front(2);
    for (uint32_t i = 0; i < 2; i++) {
        noc.async_write(dy, o, pb, {.offset_bytes = i * pb}, {.page_id = part * 10 + 8 + i});
    }
    noc.async_writes_flushed();
    dy.pop_front(2);
    noc.async_write_barrier();
}
