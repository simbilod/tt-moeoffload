// Split delta rule writer (G2 follow-up): each token's o (2 tiles, cb 16) to its pages, REPS times over; at the end the
// slice's final state (8 tiles, cb 17). Runtime args: o, state-out addresses; item (0xffffffff = idle core).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <uint32_t CB, typename Accessor>
void write_run(Noc& noc, const Accessor& d, uint32_t first_page, uint32_t n) {
    const uint32_t page_bytes = get_local_cb_interface(CB).fifo_page_size;
    DataflowBuffer dfb(CB);
    dfb.wait_front(n);
    for (uint32_t i = 0; i < n; i++) {
        noc.async_write(dfb, d, page_bytes, {.offset_bytes = i * page_bytes}, {.page_id = first_page + i});
    }
    noc.async_writes_flushed();
    dfb.pop_front(n);
}

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr auto s_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t item = get_arg_val<uint32_t>(2);
    if (item == 0xffffffff) {
        return;
    }
    const auto o = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const auto s = TensorAccessor(s_args, get_arg_val<uint32_t>(1));

    Noc noc;
    for (uint32_t rep = 0; rep < reps; rep++) {
        write_run<16>(noc, o, item * 2, 2);
    }
    write_run<17>(noc, s, item * 8, 8);
    noc.async_write_barrier();
}
