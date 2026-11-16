// Grouped expert reader (G2 prefill): one core's run of experts, y_e = x_e W_e for each. Per expert: its MT token tiles
// x kt k-tiles of x into cb 1, kept for the expert (padded to MT_MAX x kt pushes, so the index-read buffer always
// sees whole batches from its start); then its W, stored in read order (page e * nt * kt + n * kt + k = W_e's tile
// (k, n)), streamed into cb 0, BATCH pages per barrier (reader). REPS times over.
// Runtime args: w, x addresses; first expert, expert count. Compile args: reps, kt, nt, mt, mt_max, batch, accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t mt_max = get_compile_time_arg_val(4);
    constexpr uint32_t batch = get_compile_time_arg_val(5);
    constexpr auto w_args = TensorAccessorArgs<6>();
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(1));
    const uint32_t e_first = get_arg_val<uint32_t>(2);
    const uint32_t e_count = get_arg_val<uint32_t>(3);

    Noc noc;
    DataflowBuffer dw(0), dx(1);
    const uint32_t wb = get_local_cb_interface(0).fifo_page_size, xb = get_local_cb_interface(1).fifo_page_size;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t e = e_first; e < e_first + e_count; e++) {
            dx.reserve_back(mt_max * kt);
            for (uint32_t i = 0; i < mt * kt; i++) {
                noc.async_read(x, dx, xb, {.page_id = e * mt * kt + i}, {.offset_bytes = i * xb});
            }
            noc.async_read_barrier();
            dx.push_back(mt_max * kt);
            const uint32_t w0 = e * nt * kt;
            for (uint32_t i = 0; i < nt * kt; i += batch) {
                dw.reserve_back(batch);
                for (uint32_t j = 0; j < batch; j++) {
                    noc.async_read(w, dw, wb, {.page_id = w0 + i + j}, {.offset_bytes = j * wb});
                }
                noc.async_read_barrier();
                dw.push_back(batch);
            }
        }
    }
}
