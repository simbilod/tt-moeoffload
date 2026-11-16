// Hand-off: copies between two interleaved DRAM tensors in units of U bytes (a page of each, or a part of one:
// U divides both page sizes), so the prefill's tensors land in the decode's whatever their shapes: unit u of a tensor
// is at page u * U / P, offset u * U % P. Per job, `count` units from src unit a.. to dst unit b.., through cb 0
// (8 units a batch).
// DST_STRIDE (define, default 1): consecutive src units go to every DST_STRIDE-th dst unit (the sparse attention's
// [V | K] rows: V row i to unit 2 i, K row i to 2 i + 1, no temporaries; 2026-11-09).
// Runtime args: src, dst addresses, job count, then per job: first src unit, first dst unit, count.
// Compile args: U, src page size, dst page size, the accessor args (any interleaved DRAM tensor).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifndef DST_STRIDE
#define DST_STRIDE 1
#endif

void kernel_main() {
    constexpr uint32_t U = get_compile_time_arg_val(0);
    constexpr uint32_t PS = get_compile_time_arg_val(1);
    constexpr uint32_t PD = get_compile_time_arg_val(2);
    constexpr auto args = TensorAccessorArgs<3>();
    const auto src = TensorAccessor(args, get_arg_val<uint32_t>(0), PS);
    const auto dst = TensorAccessor(args, get_arg_val<uint32_t>(1), PD);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    const uint32_t buf = get_write_ptr(0);
    for (uint32_t j = 0; j < jobs; j++) {
        const uint32_t a = get_arg_val<uint32_t>(3 + 3 * j), b = get_arg_val<uint32_t>(4 + 3 * j);
        const uint32_t count = get_arg_val<uint32_t>(5 + 3 * j);
        for (uint32_t i = 0; i < count; i += 8) {
            const uint32_t k = count - i < 8 ? count - i : 8;
            for (uint32_t t = 0; t < k; t++) {
                const uint32_t s = (a + i + t) * U, d = buf + t * U;
                noc_async_read(src.get_noc_addr(s / PS, s % PS), d, U);
            }
            noc_async_read_barrier();
            for (uint32_t t = 0; t < k; t++) {
                const uint32_t d = (b + (i + t) * DST_STRIDE) * U;
                noc_async_write(buf + t * U, dst.get_noc_addr(d / PD, d % PD), U);
            }
            noc_async_write_barrier();
        }
    }
}
