// Hyper-connection gate (prefill): the up projection, its sigmoid and the gate-sum in one pass (hcgate_compute.cpp).
// One core's run of items (g, i), numbered g * rt + i: row tile i of a group g of NC output columns. Per new group
// the group's up tiles of the 4 streams (cb 1, tile (j * 4 + c) * KU + k) and norm tiles (cb 3, tile j * 4 + c); per
// item row tile i of a (cb 0, KU tiles) and the 4 streams' xn tiles of the group (cb 2, tile j * 4 + c).
// Pages: a [1, 1, T, 320] i * KU + k; up [1, 4, 320, 2560] (c * KU + k) * CT + n; xn [1, 4, T, 2560]
// (c * rt + i) * CT + n; norm [1, 4, 32, 2560] c * CT + n.
// Runtime args: a, up, xn, norm addresses; first item, count. Compile args: rt, NC, then the 4 accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t CT = 80, KU = 10;
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto u_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto up = TensorAccessor(u_args, get_arg_val<uint32_t>(1));
    const auto xn = TensorAccessor(x_args, get_arg_val<uint32_t>(2));
    const auto nw = TensorAccessor(n_args, get_arg_val<uint32_t>(3));
    const uint32_t first = get_arg_val<uint32_t>(4), count = get_arg_val<uint32_t>(5);

    Noc noc;
    DataflowBuffer da(0), dw(1), dx(2), dn(3);
    const uint32_t pa = get_local_cb_interface(0).fifo_page_size, pw = get_local_cb_interface(1).fifo_page_size;
    const uint32_t px = get_local_cb_interface(2).fifo_page_size;
    uint32_t g_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t g = it / rt, i = it % rt;
        if (g != g_prev) {
            dw.reserve_back(4 * KU * NC);
            dn.reserve_back(4 * NC);
            for (uint32_t j = 0; j < NC; j++) {
                const uint32_t n = g * NC + j;
                for (uint32_t c = 0; c < 4; c++) {
                    for (uint32_t k = 0; k < KU; k++) {
                        noc.async_read(up, dw, pw, {.page_id = (c * KU + k) * CT + n},
                                       {.offset_bytes = ((j * 4 + c) * KU + k) * pw});
                    }
                    noc.async_read(nw, dn, px, {.page_id = c * CT + n}, {.offset_bytes = (j * 4 + c) * px});
                }
            }
            noc.async_read_barrier();
            dw.push_back(4 * KU * NC);
            dn.push_back(4 * NC);
            g_prev = g;
        }
        da.reserve_back(KU);
        dx.reserve_back(4 * NC);
        for (uint32_t k = 0; k < KU; k++) {
            noc.async_read(a, da, pa, {.page_id = i * KU + k}, {.offset_bytes = k * pa});
        }
        for (uint32_t j = 0; j < NC; j++) {
            for (uint32_t c = 0; c < 4; c++) {
                noc.async_read(xn, dx, px, {.page_id = (c * rt + i) * CT + g * NC + j},
                               {.offset_bytes = (j * 4 + c) * px});
            }
        }
        noc.async_read_barrier();
        da.push_back(KU);
        dx.push_back(4 * NC);
    }
}
