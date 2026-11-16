// GLM's decode indexer, reader (compute glm_idx_compute.cpp, writer glm_idx_writer.cpp, host
// tt/glm/g3_idx.py). The score of every complete pool p: s[p] = sum_h relu(iq_h . K_p) w_h (32 heads, 128 dims), with
// the pooled keys K [n_pool, 128] (bf16 tiles, pools as rows: page 4 pt + d = pools 32 pt.., dims 32 d..), iq^T
// [128, 32] (bf16 tiles, page d) and w as row 0 of one bf16 tile. Each core takes a run of pool tiles. The coordinator
// then, at barrier 1, sends every score (gathered in its arena, fp32, pool order) to the host over D2H and takes the
// host's top 512 pool ids back over H2D, which the barrier multicasts.
// Runtime args: the grid block, K address, iq^T address, w address, first pool tile, count. Compile args: SCORE_BYTES
// (n_pool * 4: the D2H page), the accessor args of K, iq^T, w. Defines: H2D_CONFIG, D2H_CONFIG; IDX_IDS, the reply's
// bytes (default 2048: the 512 pool ids); IDX_SEL, a DRAM address (its accessor args after w's): the coordinator also
// writes the reply there (2d.4d: the host sends the 2,048 selected token ids, IDX_IDS 8192, for the attention's
// gather; a row-major uint32 tensor of one page).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"
#include "fused/link.h"

#ifndef IDX_IDS
#define IDX_IDS 2048
#endif
constexpr uint32_t cb_k = 0, cb_iq = 1, cb_w = 2, cb_arena = 8, TILE = 2048, IDS = IDX_IDS;

void kernel_main() {
    constexpr uint32_t SCORE_BYTES = get_compile_time_arg_val(0);
    constexpr auto k_args = TensorAccessorArgs<1>();
    constexpr auto q_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    const fused::Grid g = fused::Grid::from_args(0);
    const uint32_t a = fused::Grid::args(g.nrects);
    const auto K = TensorAccessor(k_args, get_arg_val<uint32_t>(a), TILE);
    const auto iq = TensorAccessor(q_args, get_arg_val<uint32_t>(a + 1), TILE);
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(a + 2), TILE);
    const uint32_t t0 = get_arg_val<uint32_t>(a + 3), nt = get_arg_val<uint32_t>(a + 4);
    const uint32_t arena = get_write_ptr(cb_arena);
    if (nt) {
        cb_reserve_back(cb_iq, 4);
        for (uint32_t d = 0; d < 4; d++) noc_async_read(iq.get_noc_addr(d), get_write_ptr(cb_iq) + d * TILE, TILE);
        cb_reserve_back(cb_w, 1);
        noc_async_read(w.get_noc_addr(0), get_write_ptr(cb_w), TILE);
        noc_async_read_barrier();
        cb_push_back(cb_iq, 4);
        cb_push_back(cb_w, 1);
        for (uint32_t t = t0; t < t0 + nt; t++) {   // a pool tile's 4 key tiles a time
            cb_reserve_back(cb_k, 4);
            for (uint32_t d = 0; d < 4; d++) noc_async_read(K.get_noc_addr(4 * t + d), get_write_ptr(cb_k) + d * TILE, TILE);
            noc_async_read_barrier();
            cb_push_back(cb_k, 4);
        }
    }
    g.wait(1, arena + SCORE_BYTES, IDS, [&] {
        fused::Link link;
        link.open(H2D_CONFIG, D2H_CONFIG, IDS, SCORE_BYTES);
        link.send(arena);
        link.recv(arena + SCORE_BYTES);
        link.close();
#ifdef IDX_SEL
        constexpr auto s_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
        const auto sel = TensorAccessor(s_args, IDX_SEL, IDS);
        noc_async_write(arena + SCORE_BYTES, sel.get_noc_addr(0), IDS);
        noc_async_write_barrier();
#endif
    });
}
