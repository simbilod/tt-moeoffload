// F1 chain reader: P GEMV phases y_p = x_p W_p in one program on the whole grid, REPS times over, a grid barrier
// before every phase but the first (tt/kernels/fused/grid.h). Each phase's x is a bf16 vector in the arena (cb 30,
// the same L1 address on every core); W_p's bfp8_b tiles are stored in the order the cores read them (page
// n * kt + k = tile (k, n), as G2's GEMV).
// Per phase: the first DEPTH weight batches are read before the barrier (they need no x), then x's row 0 pieces into
// kt tiles (cb 1: always MAX_KT pushed so the buffer ends where it began), then the rest of the weights (cb 0).
// Weights travel in batches of BATCH pages; a phase's last batch is padded (pushed whole, partly read). The
// coordinator's barrier broadcasts the previous phase's outputs (gathered into its arena by the writers).
// Runtime args: x0 address, then grid.h's block, then per phase: W address, kt, first output tile, output tile
// count, x's arena byte offset, the phase's output region (arena offset, bytes). Compile args: reps, phases, batch,
// depth, max kt, K0, then the W and x0 accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_arena = 30;
constexpr uint32_t W_PAGE = 1088;

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t phases = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr uint32_t depth = get_compile_time_arg_val(3);
    constexpr uint32_t max_kt = get_compile_time_arg_val(4);
    constexpr uint32_t k0 = get_compile_time_arg_val(5);
    constexpr auto w_args = TensorAccessorArgs<6>();
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x0 = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const fused::Grid grid = fused::Grid::from_args(1);
    const uint32_t pa = 1 + fused::Grid::args(grid.nrects);  // first phase arg
    const uint32_t arena = get_write_ptr(cb_arena);

    noc_async_read(x0.get_noc_addr(0), arena, k0 * 2);
    noc_async_read_barrier();

    [[maybe_unused]] uint32_t epoch = 0, region = 0, region_bytes = 0;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t p = 0; p < phases; p++) {
            const uint32_t a = pa + 7 * p;
            const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(a), W_PAGE);
            const uint32_t kt = get_arg_val<uint32_t>(a + 1);
            const uint32_t first = get_arg_val<uint32_t>(a + 2) * kt;
            const uint32_t total = get_arg_val<uint32_t>(a + 3) * kt;
            const uint32_t x_off = get_arg_val<uint32_t>(a + 4);
            const uint32_t nb = (total + batch - 1) / batch;

            auto issue = [&](uint32_t b) {
                cb_reserve_back(cb_w, batch);
                const uint32_t dst = get_write_ptr(cb_w);
                const uint32_t n = total - b * batch < batch ? total - b * batch : batch;
                for (uint32_t j = 0; j < n; j++) {
                    noc_async_read(w.get_noc_addr(first + b * batch + j), dst + j * W_PAGE, W_PAGE);
                }
                noc_async_read_barrier();
                cb_push_back(cb_w, batch);
            };
            const uint32_t pre = nb < depth ? nb : depth;
            for (uint32_t b = 0; b < pre; b++) {
                issue(b);
            }
#ifndef NO_BARRIER
            if (rep + p > 0) {
                grid.wait(++epoch, arena + region, region_bytes);
            }
            region = get_arg_val<uint32_t>(a + 5);
            region_bytes = get_arg_val<uint32_t>(a + 6);
#endif
            if (total) {
                // x: 32 + 32 bytes into faces 0 and 1 of each tile's row 0; rows 1-31 are left as they are (they only
                // reach rows 1-31 of the output, which nobody reads)
                cb_reserve_back(cb_x, max_kt);
                const uint32_t xt = get_write_ptr(cb_x);
                for (uint32_t k = 0; k < kt; k++) {
                    fused::l1_copy(arena + x_off + k * 64, xt + k * 2048, 32);
                    fused::l1_copy(arena + x_off + k * 64 + 32, xt + k * 2048 + 512, 32);
                }
                noc_async_read_barrier();
                cb_push_back(cb_x, max_kt);
            }
            for (uint32_t b = pre; b < nb; b++) {
                issue(b);
            }
        }
    }
#ifndef NO_BARRIER
    grid.wait(++epoch, arena + region, region_bytes);  // the last broadcast lands before any core leaves
#endif
}
