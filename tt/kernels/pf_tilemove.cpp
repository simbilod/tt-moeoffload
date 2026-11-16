// Head-major <-> token-major by whole tiles (g7_prefill.heads / tokens): a 32 x 32 tile holds 32 tokens x 32 dims of
// one head in both layouts, so the conversion is a re-indexing of tiles, no data inside one changes. Every output tile
// of the core's range [first, first + count), its source tile's page read, 16 a batch through cb 0, then written.
//   HEADS (src token-major [.., T, SRC_CT tiles], the heads' columns from tile SRC_C0; dst [1, NH, T, CPH tiles]):
//     dst tile (h * RT + rt) * CPH + c  <-  src tile rt * SRC_CT + SRC_C0 + h * CPH + c
//   else (src head-major [1, NH, T, CPH tiles]; dst token-major [1, 1, T, NH * CPH tiles]):
//     dst tile rt * NH * CPH + h * CPH + c  <-  src tile (h * RT + rt) * CPH + c
// Runtime args: src, dst addresses, first dst tile, count, RT, SRC_CT, SRC_C0 (they change with T and the call: as
// compile args every prompt length was its own JIT build). Compile args: page bytes, NH, CPH, the src accessor's, the
// dst accessor's.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PB = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t CPH = get_compile_time_arg_val(2);
    constexpr auto sargs = TensorAccessorArgs<3>();
    constexpr auto dargs = TensorAccessorArgs<sargs.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(sargs, get_arg_val<uint32_t>(0), PB);
    const auto dst = TensorAccessor(dargs, get_arg_val<uint32_t>(1), PB);
    const uint32_t first = get_arg_val<uint32_t>(2), count = get_arg_val<uint32_t>(3);
    const uint32_t RT = get_arg_val<uint32_t>(4), SRC_CT = get_arg_val<uint32_t>(5), SRC_C0 = get_arg_val<uint32_t>(6);
    const uint32_t buf = get_write_ptr(0);
    for (uint32_t i = 0; i < count; i += 16) {
        const uint32_t k = count - i < 16 ? count - i : 16;
        for (uint32_t t = 0; t < k; t++) {
            const uint32_t o = first + i + t;
#ifdef HEADS
            const uint32_t c = o % CPH, hr = o / CPH, rt = hr % RT, h = hr / RT;
            const uint32_t s = rt * SRC_CT + SRC_C0 + h * CPH + c;
#else
            const uint32_t col = o % (NH * CPH), rt = o / (NH * CPH), h = col / CPH, c = col % CPH;
            const uint32_t s = (h * RT + rt) * CPH + c;
#endif
            noc_async_read(src.get_noc_addr(s), buf + t * PB, PB);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < k; t++) {
            noc_async_write(buf + t * PB, dst.get_noc_addr(first + i + t), PB);
        }
        noc_async_write_barrier();
    }
}
