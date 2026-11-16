// G2 GEMV reader: one core's run of output tiles of y = x W, W in bfp8_b tiles stored in the order the cores read them
// (page n * kt + k = W's tile (k, n)), so each core streams consecutive pages, BATCH per barrier (reader).
// Once: x (one row-major row of K bf16) into row 0 of kt tiles in cb 1; rows 1-31 are left as they are (they only
// reach rows 1-31 of the output, which nobody reads). REPS times over: the run's weight pages into cb 0.
// Runtime args: w address, x address, first output tile, output tile count. Compile args: reps, kt, batch, w, x.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_xrow = 3;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr auto w_args = TensorAccessorArgs<3>();
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(1));
    const uint32_t n_first = get_arg_val<uint32_t>(2);
    const uint32_t n_count = get_arg_val<uint32_t>(3);

    // x: its one page (K bf16), then 32 + 32 bytes into faces 0 and 1 of each tile's row 0
    const uint32_t xrow = get_write_ptr(cb_xrow);
    noc_async_read(x.get_noc_addr(0), xrow, kt * 64);
    noc_async_read_barrier();
    cb_reserve_back(cb_x, kt);
    const uint32_t xt = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < kt; k++) {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xrow + k * 64);
        volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xt + k * 2048);
        volatile tt_l1_ptr uint32_t* f1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xt + k * 2048 + 512);
        for (uint32_t i = 0; i < 8; i++) {
            f0[i] = src[i];
            f1[i] = src[8 + i];
        }
    }
    cb_push_back(cb_x, kt);

    const uint32_t page_bytes = get_local_cb_interface(cb_w).fifo_page_size;
    const uint32_t first = n_first * kt, count = n_count * kt;
    Noc noc;
    DataflowBuffer dfb(cb_w);
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t i = 0; i < count; i += batch) {
            dfb.reserve_back(batch);
            for (uint32_t j = 0; j < batch; j++) {
                noc.async_read(w, dfb, page_bytes, {.page_id = first + i + j}, {.offset_bytes = j * page_bytes});
            }
            noc.async_read_barrier();
            dfb.push_back(batch);
        }
    }
}
