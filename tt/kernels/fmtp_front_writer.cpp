// The MTP draft layer's front, writer (the phases are in fmtp_front_reader.cpp): F1's x tiles (rows 0..15) and row
// sums into the coordinator's cb 1; arrive; F2's X tile, rows 4 s .. 4 s + mp - 1 for stream s, to the X tensors.
// Runtime args: grid.h's block; step, X fp32, X bf16 (a stream a page of M_MAX x 2560); F1's item t, F2's tile n.
// Compile args: M_MAX, a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_x = 1, cb_xh = 6, cb_xe = 7, cb_v = 12, cb_o32 = 15, cb_o16 = 16, cb_stage = 62;
constexpr uint32_t D = 2560, KD = 80, SS = 80, IDLE = 0xffffffff;

void kernel_main() {
    constexpr uint32_t m_max = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    uint32_t i = 0;
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    const auto step = TensorAccessor(d_args, arg(), 64);
    const auto x32 = TensorAccessor(d_args, arg(), m_max * D * 4), x16 = TensorAccessor(d_args, arg(), m_max * D * 2);
    const uint32_t t = arg(), n = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    const uint32_t stage = get_write_ptr(cb_stage) + 4096, x_base = get_write_ptr(cb_x);

    if (t != IDLE) {
        cb_wait_front(cb_xh, 1);
        grid.send(get_read_ptr(cb_xh), x_base + t * 2048, 1024);
        cb_wait_front(cb_xe, 1);
        grid.send(get_read_ptr(cb_xe), x_base + t * 2048 + 1024, 1024);  // x_e's rows 0..15 as rows 16..31
        cb_wait_front(cb_v, 2);  // row 0's columns 0..15 (32 bytes) to row t of SS tiles t / 32 and 3 + t / 32
        const uint32_t v = get_read_ptr(cb_v), row = (t % 32 < 16 ? 0 : 1024) + (t % 16) * 32;
        grid.send(v, x_base + (SS + t / 32) * 2048 + row, 32);
        grid.send(v + 2048, x_base + (SS + 3 + t / 32) * 2048 + row, 32);
        noc_async_write_barrier();
        cb_pop_front(cb_xh, 1);
        cb_pop_front(cb_xe, 1);
        cb_pop_front(cb_v, 2);
    }
    grid.arrive(1);

    if (n != IDLE) {
        noc_async_read(step.get_noc_addr(0), stage, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
        const uint32_t mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[4];
        const uint32_t b32 = mp * 128, b16 = mp * 64;
        cb_wait_front(cb_o32, 1);
        cb_wait_front(cb_o16, 1);
        const uint32_t o32 = get_read_ptr(cb_o32), o16 = get_read_ptr(cb_o16);
        for (uint32_t s = 0; s < 4; s++) {  // stream s's block: face 0's rows 0..mp-1, then face 1's
            fused::l1_copy(o32 + s * 4 * 64, stage + s * b32, mp * 64);
            fused::l1_copy(o32 + 1024 + s * 4 * 64, stage + s * b32 + mp * 64, mp * 64);
            fused::l1_copy(o16 + s * 4 * 32, stage + 2048 + s * b16, mp * 32);
            fused::l1_copy(o16 + 512 + s * 4 * 32, stage + 2048 + s * b16 + mp * 32, mp * 32);
        }
        noc_async_read_barrier();
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_write(stage + s * b32, x32.get_noc_addr(s, n * b32), b32);
            noc_async_write(stage + 2048 + s * b16, x16.get_noc_addr(s, n * b16), b16);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_o32, 1);
        cb_pop_front(cb_o16, 1);
    }
#ifdef GRID_PROFILE
    fused::Grid::stamp(2, GRID_PROFILE_N - 1);  // the writer's end (its last arrival slot)
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
