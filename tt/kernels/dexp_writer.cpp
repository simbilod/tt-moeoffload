// Decode-time routed experts on the card, writer (the phases are in dexp_reader.cpp): E1's h rows to
// the coordinator's arena and the H check, E2's partial rows to card memory; then this core arrives.
// Runtime args: grid.h's block; the partials, the H check (bf16 [10 x 640]), routes; the arena's H offset; this
// core's index, the core count. Compile args: a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_out = 6, cb_h16 = 10, cb_rt = 14, cb_arena = 16, cb_stage = 17;
constexpr uint32_t FF = 640;

inline uint32_t lo(uint32_t n, uint32_t k, uint32_t c) { return n * k / c; }

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    const auto part = TensorAccessor(d_args, arg(), 128);
    const auto chk_h = TensorAccessor(d_args, arg(), 10 * FF * 2), routes = TensorAccessor(d_args, arg(), 128);
    const uint32_t r_h = get_write_ptr(cb_arena) + arg();
    const uint32_t me = arg(), cores = arg();
    const uint32_t stage = get_write_ptr(cb_stage), rt = get_write_ptr(cb_rt) + 128;  // the reader's copy is below

    noc_async_read(routes.get_noc_addr(0), rt, 128);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt)[0];

    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        const uint32_t off = (v / 20) * FF * 2 + (v % 20) * 64;
        fused::stage_rows(cb_h16, 1, 32, stage);
        grid.send(stage, r_h + off, 64);
        noc_async_write(stage, chk_h.get_noc_addr(0, off), 64);
        noc_async_write_barrier();
    }
    grid.arrive(1);

    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        fused::stage_rows(cb_out, 1, 64, stage);
        noc_async_write(stage, part.get_noc_addr(w), 128);
        noc_async_write_barrier();
    }
    grid.arrive(2);
}
