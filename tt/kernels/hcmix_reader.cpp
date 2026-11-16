// Hyper-connection gate-sum reader (G2 prefill): one core's run of output tiles (i, j) of mixed [T, D], numbered
// column-major (item = j * rt + i) so a core meets few columns: per new column j the 4 streams' norm tiles (row-
// broadcast already, 1/4 folded in) into cb 2; per item the 4 streams' r tiles (the RMSNorm output) into cb 0 and
// gate tiles (before the sigmoid) into cb 1. Streams are stream-major: tile (c, i, j) at page (c * rt + i) * ct + j.
// Runtime args: r, gate, norm addresses; first item, count. Compile args: rt, ct, accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr auto r_args = TensorAccessorArgs<2>();
    constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto r = TensorAccessor(r_args, get_arg_val<uint32_t>(0));
    const auto g = TensorAccessor(g_args, get_arg_val<uint32_t>(1));
    const auto nw = TensorAccessor(n_args, get_arg_val<uint32_t>(2));
    const uint32_t first = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);

    Noc noc;
    DataflowBuffer dr(0), dg(1), dn(2);
    const uint32_t pb = get_local_cb_interface(0).fifo_page_size;
    uint32_t j_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t j = it / rt, i = it % rt;
        if (j != j_prev) {
            dn.reserve_back(4);
            for (uint32_t c = 0; c < 4; c++) {
                noc.async_read(nw, dn, pb, {.page_id = c * ct + j}, {.offset_bytes = c * pb});
            }
            noc.async_read_barrier();
            dn.push_back(4);
            j_prev = j;
        }
        dr.reserve_back(4);
        dg.reserve_back(4);
        for (uint32_t c = 0; c < 4; c++) {
            noc.async_read(r, dr, pb, {.page_id = (c * rt + i) * ct + j}, {.offset_bytes = c * pb});
            noc.async_read(g, dg, pb, {.page_id = (c * rt + i) * ct + j}, {.offset_bytes = c * pb});
        }
        noc.async_read_barrier();
        dr.push_back(4);
        dg.push_back(4);
    }
}
