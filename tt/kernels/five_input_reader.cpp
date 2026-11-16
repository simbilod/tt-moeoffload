// Expert unit test: n_i pages of five interleaved tensors into cbs 0..4 (runtime args: addr_i, n_i for i = 0..4).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <uint32_t CB, typename Args>
void read_pages(Noc& noc, const Args& args, uint32_t addr, uint32_t n) {
    const auto s = TensorAccessor(args, addr);
    const uint32_t page_bytes = get_local_cb_interface(CB).fifo_page_size;
    DataflowBuffer dfb(CB);
    dfb.reserve_back(n);
    for (uint32_t i = 0; i < n; i++) {
        noc.async_read(s, dfb, page_bytes, {.page_id = i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    dfb.push_back(n);
}

void kernel_main() {
    constexpr auto args0 = TensorAccessorArgs<0>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    constexpr auto args4 = TensorAccessorArgs<args3.next_compile_time_args_offset()>();
    Noc noc;
    read_pages<0>(noc, args0, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    read_pages<1>(noc, args1, get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    read_pages<2>(noc, args2, get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    read_pages<3>(noc, args3, get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    read_pages<4>(noc, args4, get_arg_val<uint32_t>(8), get_arg_val<uint32_t>(9));
}
