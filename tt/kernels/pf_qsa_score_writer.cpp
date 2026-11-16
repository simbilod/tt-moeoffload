// pf_qsa_score_compute.cpp's score tiles (cb 16, 2 a pool tile: query tiles q, q + 1) to score [Tq, P]'s pages.
// Runtime args: score address, first pool tile, pool tiles, query tiles. Compile args: PT (pool tiles a row), accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PT = get_compile_time_arg_val(0);
    constexpr auto s_args = TensorAccessorArgs<1>();
    const auto score = TensorAccessor(s_args, get_arg_val<uint32_t>(0), 4096);
    const uint32_t p0 = get_arg_val<uint32_t>(1), np = get_arg_val<uint32_t>(2), nq = get_arg_val<uint32_t>(3);
    for (uint32_t q = 0; q < nq && np; q += 2) {
        for (uint32_t p = 0; p < np; p++) {
            cb_wait_front(16, 2);
            const uint32_t r = get_read_ptr(16);
            noc_async_write(r, score.get_noc_addr(q * PT + p0 + p), 4096);
            noc_async_write(r + 4096, score.get_noc_addr((q + 1) * PT + p0 + p), 4096);
            noc_async_writes_flushed();
            cb_pop_front(16, 2);
        }
    }
    noc_async_write_barrier();
}
