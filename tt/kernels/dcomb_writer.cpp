// Decode QSA combine writer: the (KV head g, dims tile d) output tile (its 12 heads in rows 0..11) to out page g * 8 + d.
// Runtime args: out address, kv head (0xffffffff = idle), d. Compile args: accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    const auto o = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t g = get_arg_val<uint32_t>(1), d = get_arg_val<uint32_t>(2);
    if (g == 0xffffffff) {
        return;
    }
    const uint32_t pb = get_local_cb_interface(16).fifo_page_size;
    Noc noc;
    DataflowBuffer dy(16);
    dy.wait_front(1);
    noc.async_write(dy, o, pb, {}, {.page_id = g * 8 + d});
    noc.async_writes_flushed();
    dy.pop_front(1);
    noc.async_write_barrier();
}
