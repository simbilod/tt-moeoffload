// The MTP draft layer's routed experts, writer (the phases are in fmtp_exp_reader.cpp): arrive at barrier 1; E1's h
// rows into region H of the coordinator; arrive at barrier 2; E2's tile of R (4 streams) to R's fp32 and bf16 tensors.
// Runtime args: grid.h's block; R fp32, R bf16 ([4 x 2560], one page); this core's index, the core count, E2's tile.
// Compile args: a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_h16 = 8, cb_o32 = 14, cb_o16 = 15, cb_arena = 61, cb_stage = 62;
constexpr uint32_t D = 2560, FF = 640, TOPK = 10, H_OFF = 8192, IDLE = 0xffffffff;

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    const auto r32 = TensorAccessor(d_args, arg(), 4 * D * 4), r16 = TensorAccessor(d_args, arg(), 4 * D * 2);
    const auto rp = TensorAccessor(d_args, arg(), 128);
    const uint32_t v0 = arg(), v1 = arg(), col = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    const uint32_t stage = get_write_ptr(cb_stage) + 2048, h = get_write_ptr(cb_arena) + H_OFF;

    grid.arrive(1);
    for (uint32_t v = v0; v < v1; v++) {
        fused::stage_rows(cb_h16, 1, 32, stage);
        grid.send(stage, h + (v / 20) * FF * 2 + (v % 20) * 64, 64);
        noc_async_write_barrier();
    }
    grid.arrive(2);
    if (col != IDLE) {
        fused::stage_rows(cb_o32, 4, 64, stage);
        fused::stage_rows(cb_o16, 4, 32, stage + 512);
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_write(stage + s * 128, r32.get_noc_addr(0, (s * D + 32 * col) * 4), 128);
            noc_async_write(stage + 512 + s * 64, r16.get_noc_addr(0, (s * D + 32 * col) * 2), 64);
            noc_async_write(stage + s * 128, rp.get_noc_addr(s * 80 + col), 128);  // (the next front's copy)
        }
        noc_async_write_barrier();
    }
#ifdef GRID_PROFILE
    fused::Grid::stamp(2, GRID_PROFILE_N - 1);  // the writer's end (its last arrival slot)
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
