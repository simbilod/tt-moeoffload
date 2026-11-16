// GLM's multi-phase one-token GEMV, writer (glm_mgemv_reader.cpp): per phase, each output tile's row 0
// (32 bf16: face 0's first row, then face 1's) into the coordinator's arena at the phase's offset, where the next
// barrier multicasts it to every core as the later phases' x; every tile also to its page of the DRAM output (all
// phases' tiles, phase after phase), for the host's check. Then the core arrives at the phase's barrier.
// Runtime args: the grid block, out address, then per phase 4 words: first output tile, count, the arena offset of the
// phase's y (bytes), the phase's first page in out. Compile args: P, then out's accessor args.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "fused/grid.h"

constexpr uint32_t cb_out = 16, cb_arena = 8;

void kernel_main() {
    constexpr uint32_t P = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const fused::Grid g = fused::Grid::from_args(0);
    uint32_t a = fused::Grid::args(g.nrects);
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(a));
    a += 1;
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t page_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    Noc noc;
    DataflowBuffer dfb(cb_out);
    for (uint32_t p = 0; p < P; p++, a += 4) {
        const uint32_t first = get_arg_val<uint32_t>(a), count = get_arg_val<uint32_t>(a + 1);
        const uint32_t yoff = get_arg_val<uint32_t>(a + 2), opage = get_arg_val<uint32_t>(a + 3);
        for (uint32_t n = 0; n < count; n++) {
            dfb.wait_front(1);
            const uint32_t src = get_read_ptr(cb_out);
            const uint32_t dst = arena + yoff + (first + n) * 64;
            g.send(src, dst, 32);
            g.send(src + 512, dst + 32, 32);
            noc.async_write(dfb, out, page_bytes, {}, {.page_id = opage + first + n});
            noc.async_writes_flushed();
            dfb.pop_front(1);
        }
        g.arrive(p + 1);
    }
    noc.async_write_barrier();
}
