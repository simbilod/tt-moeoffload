// GLM's fused FFN half, writer (the phases in glm_fsh_reader.cpp): R1, each unit j's h row (cb 16,
// bf16) to region H at 64 j, barrier 1; R2, each item's partial row (cb 17, fp32) to region P at (part, n), barrier 2.
// Runtime args: grid.h's block, R1's first unit and count, R2's first item and count, the arena offsets of H, P.
// Compile args: S.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#ifndef M_MAX
#define M_MAX 1
#endif
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif

constexpr uint32_t cb_h = 16, cb_out = 17, cb_arena = 30, cb_stage = 31, NT = 128;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    const fused::Grid grid = fused::Grid::from_args(0);
    uint32_t i = fused::Grid::args(grid.nrects);
    const uint32_t u0 = get_arg_val<uint32_t>(i++), nu = get_arg_val<uint32_t>(i++);
    const uint32_t it0 = get_arg_val<uint32_t>(i++), n_it = get_arg_val<uint32_t>(i++);
    const uint32_t H = get_arg_val<uint32_t>(i++), P = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
#if M_MAX > 1
    // M rows (glm_fsh_reader.cpp): runtime args after P: the step tensor, then per R2 item the owner's NOC x | y << 16
    // and the partial's slot there (fused::ksplit_sum_m's layout)
    constexpr auto d_args = TensorAccessorArgs<1>();
    const uint32_t step_addr = get_arg_val<uint32_t>(i++), dest = i;
    noc_async_read(TensorAccessor(d_args, step_addr, 64).get_noc_addr(0), stage, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[4], blk = mp * 128;
    for (uint32_t u = u0; u < u0 + nu; u++) {
        fused::stage_rows_m(cb_h, 1, 32, stage, mp);
        grid.send(stage, arena + H + u * mp * 64, mp * 64);
        noc_async_write_barrier();
    }
    grid.arrive(1);
    for (uint32_t j = 0; j < n_it; j++) {
        fused::stage_rows_m(cb_out, 1, 64, stage, mp);
        const uint32_t xy = get_arg_val<uint32_t>(dest + 2 * j), slot = get_arg_val<uint32_t>(dest + 2 * j + 1);
        noc_async_write(stage, get_noc_addr(xy & 0xffff, xy >> 16, arena + P + slot * blk), blk);
        noc_async_write_barrier();
    }
    grid.arrive(2);
    return;
#endif
    for (uint32_t u = u0; u < u0 + nu; u++) {
        fused::stage_rows(cb_h, 1, 32, stage);
        grid.send(stage, arena + H + u * 64, 64);
        noc_async_write_barrier();
    }
    grid.arrive(1);
    for (uint32_t it = it0; it < it0 + n_it; it++) {
        fused::stage_rows(cb_out, 1, 64, stage);
        grid.send(stage, arena + P + ((it % S) * NT + it / S) * 128, 128);
        noc_async_write_barrier();
    }
    grid.arrive(2);
}
