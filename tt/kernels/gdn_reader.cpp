// Split delta rule reader (G2 follow-up): one core's (head, 64-column slice) of a gated DeltaNet's recurrent state.
// Once: the slice's state, 8 fp32 tiles (k-tile j, column tile c at 2j + c), into cb 23 (read exactly by the SFPU
// update) and cb 22 (the FPU's operand); each buffer has one producer, so the compute's state buffers are its own.
// REPS times: the token's 12
// matmul operand tiles into cb 0 (k' = -beta d k in row 0 x4, k in column 0 x4, scaled q in row 0 x4) and the 2 tiles
// of beta v (row 0) into cb 2. With RESIDENT, the token's inputs come from card memory only the first time (the
// compute time alone).
// Runtime args: state, inputs, beta v addresses; item (0xffffffff = idle core). Compile args: reps, then accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

template <uint32_t CB, typename Accessor>
void read_run(Noc& noc, const Accessor& s, uint32_t first_page, uint32_t n, bool fetch) {
    const uint32_t page_bytes = get_local_cb_interface(CB).fifo_page_size;
    DataflowBuffer dfb(CB);
    dfb.reserve_back(n);
    if (fetch) {
        for (uint32_t i = 0; i < n; i++) {
            noc.async_read(s, dfb, page_bytes, {.page_id = first_page + i}, {.offset_bytes = i * page_bytes});
        }
        noc.async_read_barrier();
    }
    dfb.push_back(n);
}

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr auto s_args = TensorAccessorArgs<1>();
    constexpr auto in_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto bv_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const uint32_t item = get_arg_val<uint32_t>(3);
    if (item == 0xffffffff) {
        return;
    }
    const auto s = TensorAccessor(s_args, get_arg_val<uint32_t>(0));
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(1));
    const auto bv = TensorAccessor(bv_args, get_arg_val<uint32_t>(2));

    Noc noc;
    read_run<23>(noc, s, item * 8, 8, true);
    read_run<22>(noc, s, item * 8, 8, true);
    for (uint32_t rep = 0; rep < reps; rep++) {
#ifdef RESIDENT
        const bool fetch = rep == 0;
#else
        const bool fetch = true;
#endif
        read_run<0>(noc, in, item * 12, 12, fetch);
        read_run<2>(noc, bv, item * 2, 2, fetch);
    }
}
