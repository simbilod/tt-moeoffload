// Hyper-connection low-rank outputs (prefill) writer: item (i, k) = i * 11 + k: k < 10 cb 16's tile to a [1, 1, T,
// 320] page i * 10 + k, k = 10 cb 17's to w [1, 1, T, 32] page i.
// Runtime args: a, w addresses, first item, count. Compile args: the 2 accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t KO = 11;
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const uint32_t first = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    const uint32_t pa = get_local_cb_interface(16).fifo_page_size, pw = get_local_cb_interface(17).fifo_page_size;
    Noc noc;
    DataflowBuffer da(16), dw(17);
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / KO, k = it % KO;
        if (k == KO - 1) {
            dw.wait_front(1);
            noc.async_write(dw, w, pw, {}, {.page_id = i});
            noc.async_writes_flushed();
            dw.pop_front(1);
        } else {
            da.wait_front(1);
            noc.async_write(da, a, pa, {}, {.page_id = i * (KO - 1) + k});
            noc.async_writes_flushed();
            da.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
