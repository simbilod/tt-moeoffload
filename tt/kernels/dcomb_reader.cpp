// Decode QSA combine reader, one core per (KV head, dims tile d): the partials' d-th O tiles into cb 0, and the
// partials' m and l rows gathered into one statistics tile each (row i = partial i: 64-byte reads of row 0, heads
// 0..15, from each partial's row tile; rows past the partials keep the init tile's -1e30 / 0) in cb 1 and cb 2. Once:
// the constants (identity, -1e30 tile, zero tile, scaler) into cb 3 (identity) and cb 5 (scaler).
// Runtime args: partials, constants addresses; kv head (0xffffffff = idle), d. Compile args: partials per head,
// accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t np = get_compile_time_arg_val(0);
    constexpr auto p_args = TensorAccessorArgs<1>();
    constexpr auto c_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto parts = TensorAccessor(p_args, get_arg_val<uint32_t>(0));
    const auto consts = TensorAccessor(c_args, get_arg_val<uint32_t>(1));
    const uint32_t g = get_arg_val<uint32_t>(2), d = get_arg_val<uint32_t>(3);
    if (g == 0xffffffff) {
        return;
    }
    Noc noc;
    DataflowBuffer d_o(0), d_m(1), d_l(2), d_i(3), d_s(5);
    const uint32_t fb = get_local_cb_interface(0).fifo_page_size;
    d_i.reserve_back(1);
    noc.async_read(consts, d_i, fb, {.page_id = 0}, {});
    d_s.reserve_back(1);
    noc.async_read(consts, d_s, fb, {.page_id = 3}, {});
    d_m.reserve_back(1);
    noc.async_read(consts, d_m, fb, {.page_id = 1}, {});
    d_l.reserve_back(1);
    noc.async_read(consts, d_l, fb, {.page_id = 2}, {});
    d_o.reserve_back(np);
    for (uint32_t i = 0; i < np; i++) {
        noc.async_read(parts, d_o, fb, {.page_id = (g * np + i) * 10 + d}, {.offset_bytes = i * fb});
    }
    noc.async_read_barrier();
    // row i of an fp32 tile: face 0 (rows 0..15) at i * 64 bytes, face 2 (rows 16..31) at 2048 + (i - 16) * 64
    for (uint32_t i = 0; i < np; i++) {
        const uint32_t off = i < 16 ? i * 64 : 2048 + (i - 16) * 64;
        noc.async_read(parts, d_m, 64, {.page_id = (g * np + i) * 10 + 8}, {.offset_bytes = off});
        noc.async_read(parts, d_l, 64, {.page_id = (g * np + i) * 10 + 9}, {.offset_bytes = off});
    }
    noc.async_read_barrier();
    d_i.push_back(1);
    d_s.push_back(1);
    d_m.push_back(1);
    d_l.push_back(1);
    d_o.push_back(np);
}
