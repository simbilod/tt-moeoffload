// GLM's decode indexer, writer: each pool tile's 32 scores (row 0 of an fp32 tile: 64 bytes from face
// 0, 64 from face 1) into the coordinator's arena at 128 bytes a pool tile, in pool order; then barrier 1.
// Runtime args: the grid block, first pool tile, count.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fused/grid.h"

constexpr uint32_t cb_out = 16, cb_arena = 8;

void kernel_main() {
    const fused::Grid g = fused::Grid::from_args(0);
    const uint32_t a = fused::Grid::args(g.nrects);
    const uint32_t t0 = get_arg_val<uint32_t>(a), nt = get_arg_val<uint32_t>(a + 1);
    const uint32_t arena = get_write_ptr(cb_arena);
    for (uint32_t t = t0; t < t0 + nt; t++) {
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        g.send(src, arena + t * 128, 64);
        g.send(src + 1024, arena + t * 128 + 64, 64);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    g.arrive(1);
}
