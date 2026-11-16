// Fused DeltaNet mixer writer (G3 step 2; the phases are described in fgdn_reader.cpp). Row 0 of a result tile is
// two 64-byte pieces (fp32) or two 32-byte pieces (bf16), faces 0 and 1.
//   A  per unit: for conv tiles the 3 new conv-state rows (cb 18) to the conv state; row 0 of the unit's results
//      (cb 17) to region A of the coordinator's arena and to A's check tensor; arrive at barrier 1
//   B  the head's new state (cb 19, 16 tiles) to card memory; its 128 outputs (cb 21 bf16) to region B, (cb 22
//      fp32) to B's check tensor; arrive at barrier 2
//   C  per output tile (cb 17), row 0 to region C and C's check tensor; arrive at barrier 3
// Runtime args: conv state out, A check, grid.h's block, region A, unit count, per unit (first tile, count, type),
// head, state out, B check, region B, C's first tile and count, C check, region C. Compile args: phases, then a
// DRAM accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

constexpr uint32_t cb_out = 17, cb_snew = 18, cb_sxo = 19, cb_hout = 21, cb_hchk = 22, cb_arena = 30, cb_stage = 31;
constexpr uint32_t CONV = 10240, NT_A = 516, HK = 128;
constexpr uint32_t T_K = 2, IDLE = 0xffffffff;

void kernel_main() {
    constexpr uint32_t phases = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    const auto state = TensorAccessor(d_args, get_arg_val<uint32_t>(0), CONV * 4);
    const auto check_a = TensorAccessor(d_args, get_arg_val<uint32_t>(1), NT_A * 128);
    const fused::Grid grid = fused::Grid::from_args(2);
    uint32_t i = 2 + fused::Grid::args(grid.nrects);
    const uint32_t region_a = get_arg_val<uint32_t>(i++);
    const uint32_t n_units = get_arg_val<uint32_t>(i++);
    const uint32_t ua = i;
    i += 3 * n_units;
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const auto S_out = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 4096);
    const auto check_b = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 48 * HK * 4);
    const uint32_t region_b = get_arg_val<uint32_t>(i++);
    const uint32_t c_first = get_arg_val<uint32_t>(i++), c_count = get_arg_val<uint32_t>(i++);
    const auto check_c = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 80 * 128);
    const uint32_t region_c = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);

    // row 0 of n tiles of cb (fp32 or bf16) into the staging run
    auto stage_rows = [&](uint32_t cb, uint32_t n, uint32_t piece) {
        cb_wait_front(cb, n);
        const uint32_t t = get_read_ptr(cb), tile = piece * 64;
        for (uint32_t k = 0; k < n; k++) {
            fused::l1_copy(t + k * tile, stage + k * 2 * piece, piece);
            fused::l1_copy(t + k * tile + tile / 4, stage + k * 2 * piece + piece, piece);
        }
        noc_async_read_barrier();
        cb_pop_front(cb, n);
    };

    // ---- A
    if (phases & 1) {
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            if (type <= T_K) {
                for (uint32_t n = first; n < first + count; n++) {
                    cb_wait_front(cb_snew, 3);
                    const uint32_t t = get_read_ptr(cb_snew);
                    for (uint32_t j = 0; j < 3; j++) {
                        noc_async_write(t + j * 4096, state.get_noc_addr(j, n * 128), 64);
                        noc_async_write(t + j * 4096 + 1024, state.get_noc_addr(j, n * 128 + 64), 64);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_snew, 3);
                }
            }
            stage_rows(cb_out, count, 64);
            grid.send(stage, arena + region_a + first * 128, count * 128);
            noc_async_write(stage, check_a.get_noc_addr(0, first * 128), count * 128);
            noc_async_write_barrier();  // the staging run is reused next
        }
        grid.arrive(1);
    }

    // ---- B
    if (phases & 2) {
        if (head != IDLE) {
            for (uint32_t j = 0; j < 4; j++) {
                cb_wait_front(cb_sxo, 4);
                const uint32_t t = get_read_ptr(cb_sxo);
                for (uint32_t c = 0; c < 4; c++) {
                    noc_async_write(t + c * 4096, S_out.get_noc_addr(head * 16 + 4 * j + c), 4096);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_sxo, 4);
            }
            stage_rows(cb_hout, 4, 32);
            grid.send(stage, arena + region_b + head * HK * 2, HK * 2);
            noc_async_write_barrier();
            stage_rows(cb_hchk, 4, 64);
            noc_async_write(stage, check_b.get_noc_addr(0, head * HK * 4), HK * 4);
            noc_async_write_barrier();
        }
        grid.arrive(2);
    }

    // ---- C
    if (phases & 4) {
        for (uint32_t n = c_first; n < c_first + c_count; n++) {
            stage_rows(cb_out, 1, 64);
            grid.send(stage, arena + region_c + n * 128, 128);
            noc_async_write(stage, check_c.get_noc_addr(0, n * 128), 128);
            noc_async_write_barrier();
        }
        grid.arrive(3);
    }
}
