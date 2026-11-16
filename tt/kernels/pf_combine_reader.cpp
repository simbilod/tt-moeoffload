// Prefill: the routed experts' sum per token, out = sum_j w_j * y_j over the K routes. y holds route j's outputs
// in token order (fp32 [K T, 2560]: pf_gexp_writer.cpp's SCATTER), w the routes' weights row-major fp32 [T, 32] (row
// t: token t's K weights, then padding). Items (token tile i, column tile n) numbered i * nt + n: per new i the 32
// weight rows into cb 3, made into K tiles in cb 2 whose every column is the row's weight j (face order: element
// (r, c) at face 2 (r / 16) + c / 16, (r % 16) * 16 + c % 16); per item the K y tiles into cb 0.
// Runtime args: y, w addresses; first item, count. Compile args: rt, nt, K, accessors (y, w).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto y_args = TensorAccessorArgs<3>();
    constexpr auto w_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1), 128);
    const uint32_t first = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);

    Noc noc;
    DataflowBuffer dy(0);
    const uint32_t pb = get_local_cb_interface(0).fifo_page_size;
    const uint32_t rows = get_write_ptr(3);
    volatile tt_l1_ptr float* wr = reinterpret_cast<volatile tt_l1_ptr float*>(rows);
    uint32_t i_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / nt, n = it % nt;
        if (i != i_prev) {
            for (uint32_t r = 0; r < 32; r++) {
                noc_async_read(w.get_noc_addr(i * 32 + r), rows + r * 128, 128);
            }
            noc_async_read_barrier();
            asm volatile("fence rw, rw" ::: "memory");
            cb_reserve_back(2, K);
            volatile tt_l1_ptr float* t = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(2));
            for (uint32_t j = 0; j < K; j++) {
                for (uint32_t r = 0; r < 32; r++) {
                    const float v = wr[r * 32 + j];
                    volatile tt_l1_ptr float* f = t + j * 1024 + (r >> 4) * 512 + (r & 15) * 16;
                    for (uint32_t c = 0; c < 16; c++) {
                        f[c] = v;
                        f[256 + c] = v;
                    }
                }
            }
            cb_push_back(2, K);
            i_prev = i;
        }
        dy.reserve_back(K);
        for (uint32_t j = 0; j < K; j++) {
            noc.async_read(y, dy, pb, {.page_id = (j * rt + i) * nt + n}, {.offset_bytes = j * pb});
        }
        noc.async_read_barrier();
        dy.push_back(K);
    }
}
