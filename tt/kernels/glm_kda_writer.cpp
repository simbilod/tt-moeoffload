// GLM's KDA decode step, writer: the new exact state (cb 19, two tiles at a time, in tile order
// 4 j + c) to its pages of S_out, then the head's 128 outputs (row 0 of 4 fp32 tiles, cb 22) to row h of out.
// Runtime args: S_out address, out address, head (0xffffffff: idle). Compile args: the accessor args of S_out, out.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t cb_sxo = 19, cb_hchk = 22, TILE = 4096, ROW = 512, IDLE = 0xffffffff;

void kernel_main() {
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto So = TensorAccessor(s_args, get_arg_val<uint32_t>(0), TILE);
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(1), ROW);
    const uint32_t h = get_arg_val<uint32_t>(2);
    if (h == IDLE) return;
    for (uint32_t t = 0; t < 16; t += 2) {
        cb_wait_front(cb_sxo, 2);
        const uint32_t src = get_read_ptr(cb_sxo);
        noc_async_write(src, So.get_noc_addr(h * 16 + t), TILE);
        noc_async_write(src + TILE, So.get_noc_addr(h * 16 + t + 1), TILE);
        noc_async_write_barrier();
        cb_pop_front(cb_sxo, 2);
    }
    for (uint32_t c = 0; c < 4; c++) {
        cb_wait_front(cb_hchk, 1);
        const uint32_t src = get_read_ptr(cb_hchk);
        noc_async_write(src, out.get_noc_addr(h, c * 128), 64);
        noc_async_write(src + 1024, out.get_noc_addr(h, c * 128 + 64), 64);
        noc_async_write_barrier();
        cb_pop_front(cb_hchk, 1);
    }
}
