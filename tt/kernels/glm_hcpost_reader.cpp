// GLM mHC post reader (prefill): new_j = post_j * o + sum_i comb_ij * X_i over the 4 streams, fp32. X [1, 1, T,
// 4 * ct * 32] streams side by side (tile (i, s, j) at page i * 4 ct + s ct + j), o [1, 1, T, ct * 32] fp32, w [1, 20,
// T, 32] the row coefficients widened to a tile (q 0-3 post_j, 4 + 4 s + j comb_sj; page q * rt + i). One core's run of
// items (i, j) = o's tiles, row-major: per new row i its 20 w tiles into cb 2; per item o's tile into cb 1 and the 4
// streams' X tiles into cb 0. Runtime args: X, o, w addresses; first item, count. Compile args: rt, ct, pre, accessors.
// pre = 1 (the site's pre mix, h = sum_s pre_s X_s): w [1, 4, T, 32], no o read (its accessor is X's).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t pre = get_compile_time_arg_val(2);
    constexpr uint32_t NQ = pre ? 4 : 20;
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto o = TensorAccessor(o_args, get_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2));
    const uint32_t first = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);

    Noc noc;
    DataflowBuffer dx(0), dout(1), dw(2);
    const uint32_t pb = get_local_cb_interface(0).fifo_page_size;
    uint32_t i_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / ct, j = it % ct;
        if (i != i_prev) {
            dw.reserve_back(NQ);
            for (uint32_t q = 0; q < NQ; q++) {
                noc.async_read(w, dw, pb, {.page_id = q * rt + i}, {.offset_bytes = q * pb});
            }
            noc.async_read_barrier();
            dw.push_back(NQ);
            i_prev = i;
        }
        if constexpr (!pre) {
            dout.reserve_back(1);
            noc.async_read(o, dout, pb, {.page_id = i * ct + j}, {});
        }
        dx.reserve_back(4);
        for (uint32_t s = 0; s < 4; s++) {
            noc.async_read(x, dx, pb, {.page_id = (i * 4 + s) * ct + j}, {.offset_bytes = s * pb});
        }
        noc.async_read_barrier();
        if constexpr (!pre) {
            dout.push_back(1);
        }
        dx.push_back(4);
    }
}
