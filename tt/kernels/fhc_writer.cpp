// Fused hyper-connection mix writer (G3 step 3; fhc_reader.cpp). H1: the partial lo[s][n] (cb 17, row 0) to region
// P1 of the coordinator's arena and P1's check; r_s (cb 18, its first 16 bytes) to region RS when n = 0; arrive at
// barrier 2. H2: hn's tile j (cb 19 bf16, row 0) to region HN, (cb 20 fp32) to HN's check; core 80: w (cb 17, its
// first 16 bytes) to region W and W's check; arrive at barrier 3 (barrier 1 is X's broadcast).
// Runtime args: P1 check, HN check, W check, grid.h's block, the arena offsets P1, RS, HN, W, the H1 item, the H2
// tile. Compile args: a DRAM accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

constexpr uint32_t cb_out = 17, cb_rout = 18, cb_hn = 19, cb_hchk = 20, cb_arena = 30, cb_stage = 31;
constexpr uint32_t D = 2560, LO = 352, IDLE = 0xffffffff;

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto chk_p = TensorAccessor(d_args, get_arg_val<uint32_t>(0), 4 * LO * 4);
    const auto chk_hn = TensorAccessor(d_args, get_arg_val<uint32_t>(1), D * 4);
    const auto chk_w = TensorAccessor(d_args, get_arg_val<uint32_t>(2), 64);
    const fused::Grid grid = fused::Grid::from_args(3);
    uint32_t i = 3 + fused::Grid::args(grid.nrects);
    const uint32_t r_p1 = get_arg_val<uint32_t>(i++), r_rs = get_arg_val<uint32_t>(i++);
    const uint32_t r_hn = get_arg_val<uint32_t>(i++), r_w = get_arg_val<uint32_t>(i++);
    const uint32_t item = get_arg_val<uint32_t>(i++), tile = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);

    auto row0 = [&](uint32_t cb, uint32_t piece) {  // row 0 of cb's front tile into the staging run
        cb_wait_front(cb, 1);
        const uint32_t t = get_read_ptr(cb);
        fused::l1_copy(t, stage, piece);
        fused::l1_copy(t + piece * 16, stage + piece, piece);
        noc_async_read_barrier();
        cb_pop_front(cb, 1);
    };

    grid.arrive(1);  // X's broadcast
    if (item != IDLE) {
        const uint32_t n = item >> 2, s = item & 3;
        if (n == 0) {
            cb_wait_front(cb_rout, 1);
            fused::l1_copy(get_read_ptr(cb_rout), stage + 256, 16);
            noc_async_read_barrier();
            cb_pop_front(cb_rout, 1);
            grid.send(stage + 256, arena + r_rs + s * 16, 16);
        }
        row0(cb_out, 64);
        grid.send(stage, arena + r_p1 + (s * LO + n * 32) * 4, 128);
        noc_async_write(stage, chk_p.get_noc_addr(0, (s * LO + n * 32) * 4), 128);
        noc_async_write_barrier();
    }
    grid.arrive(2);

    if (tile == D / 32) {
        row0(cb_out, 64);
        grid.send(stage, arena + r_w, 16);
        noc_async_write(stage, chk_w.get_noc_addr(0), 64);
        noc_async_write_barrier();
    } else if (tile != IDLE) {
        row0(cb_hn, 32);
        grid.send(stage, arena + r_hn + tile * 64, 64);
        noc_async_write_barrier();
        row0(cb_hchk, 64);
        noc_async_write(stage, chk_hn.get_noc_addr(0, tile * 128), 128);
        noc_async_write_barrier();
    }
    grid.arrive(3);
}
