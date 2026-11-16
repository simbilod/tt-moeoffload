// Hyper-connection low-rank outputs (prefill, hclo_compute.cpp) reader: item (i, k), numbered i * 11 + k, the 4
// streams' down|inject partial tiles (c, i, k) of p [1, 4, T, 352] fp32 (page (c * rt + i) * 11 + k) into cb 0.
// Runtime args: p address, first item, count. Compile args: rt, accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t KO = 11;
    constexpr auto p_args = TensorAccessorArgs<1>();
    const auto p = TensorAccessor(p_args, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer dp(0);
    const uint32_t pb = get_local_cb_interface(0).fifo_page_size;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t i = it / KO, k = it % KO;
        dp.reserve_back(4);
        for (uint32_t c = 0; c < 4; c++) {
            noc.async_read(p, dp, pb, {.page_id = (c * rt + i) * KO + k}, {.offset_bytes = c * pb});
        }
        noc.async_read_barrier();
        dp.push_back(4);
    }
}
