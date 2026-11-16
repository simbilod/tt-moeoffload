// F1 chain writer: per phase, row 0 of each output tile (fp32 from cb 16, bf16 from cb 17) into a staging run (cb
// 31), sent into the coordinator's arena (cb 30) at the phase's fp32 and bf16 offsets (its barrier broadcasts them),
// the fp32 run also to row p of the DRAM output (for checking); then this core arrives at the barrier.
// Runtime args: out address, then grid.h's block, then per phase: first output tile, output tile count, fp32 arena
// offset, bf16 arena offset. Compile args: reps, phases, staging bf16 offset, then the out accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

constexpr uint32_t cb_y32 = 16, cb_y16 = 17, cb_arena = 30, cb_stage = 31;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t phases = get_compile_time_arg_val(1);
    constexpr uint32_t stage16 = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const fused::Grid grid = fused::Grid::from_args(1);
    const uint32_t pa = 1 + fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t s32 = get_write_ptr(cb_stage), s16 = s32 + stage16;

    [[maybe_unused]] uint32_t epoch = 0;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t p = 0; p < phases; p++) {
            const uint32_t a = pa + 4 * p;
            const uint32_t n_first = get_arg_val<uint32_t>(a), n_count = get_arg_val<uint32_t>(a + 1);
            const uint32_t off32 = get_arg_val<uint32_t>(a + 2), off16 = get_arg_val<uint32_t>(a + 3);
            for (uint32_t n = 0; n < n_count; n++) {
                cb_wait_front(cb_y32, 1);
                cb_wait_front(cb_y16, 1);
                const uint32_t t32 = get_read_ptr(cb_y32), t16 = get_read_ptr(cb_y16);
                fused::l1_copy(t32, s32 + n * 128, 64);
                fused::l1_copy(t32 + 1024, s32 + n * 128 + 64, 64);
                fused::l1_copy(t16, s16 + n * 64, 32);
                fused::l1_copy(t16 + 512, s16 + n * 64 + 32, 32);
                noc_async_read_barrier();
                cb_pop_front(cb_y32, 1);
                cb_pop_front(cb_y16, 1);
            }
            if (n_count) {
#ifndef NO_BARRIER
                grid.send(s32, arena + off32 + n_first * 128, n_count * 128);
                grid.send(s16, arena + off16 + n_first * 64, n_count * 64);
#endif
                noc_async_write(s32, out.get_noc_addr(p, n_first * 128), n_count * 128);
            }
#ifndef NO_BARRIER
            grid.arrive(++epoch);
#else
            noc_async_write_barrier();
#endif
        }
    }
}
