// Card formats test: page 0 of four interleaved tensors (codes, a, b, top bits) into cbs 0..3.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <uint32_t CB, typename Args>
void read_one(Noc& noc, const Args& args, uint32_t addr) {
    const auto s = TensorAccessor(args, addr);
    DataflowBuffer dfb(CB);
    dfb.reserve_back(1);
    noc.async_read(s, dfb, get_local_cb_interface(CB).fifo_page_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb.push_back(1);
}

void kernel_main() {
    constexpr auto args0 = TensorAccessorArgs<0>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    Noc noc;
    read_one<0>(noc, args0, get_arg_val<uint32_t>(0));
    read_one<1>(noc, args1, get_arg_val<uint32_t>(1));
    read_one<2>(noc, args2, get_arg_val<uint32_t>(2));
    read_one<3>(noc, args3, get_arg_val<uint32_t>(3));
}
