// GLM indexer score reader (prefill; Flash-Next's pf_qsa_score_reader generalized to 32 gated heads). One core's
// run of items (q, p) = score tiles, row-major (q = query tile, p = pool tile): per new q its 32 heads' iq tiles (order
// head, k: 128) into cb 0 and the 32 gate tiles (w_h widened to a row-broadcast tile) into cb 2; per item the pool
// tile's 4 pk tiles into cb 1. iq [1, 32, T, 128] (bf16 or fp32), w [1, 32, T, 32] fp32, pk [1, 1, P, 128] fp32.
// Runtime args: iq, w, pk addresses; first item, count. Compile args: QT, PT, IB (iq's tile bytes), accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_IQ = 0, CB_PK = 1, CB_W = 2;

void kernel_main() {
    constexpr uint32_t QT = get_compile_time_arg_val(0);
    constexpr uint32_t PT = get_compile_time_arg_val(1);
    constexpr uint32_t IB = get_compile_time_arg_val(2);
    constexpr auto i_args = TensorAccessorArgs<3>();
    constexpr auto w_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto iq = TensorAccessor(i_args, get_arg_val<uint32_t>(0), IB);
    const auto wt = TensorAccessor(w_args, get_arg_val<uint32_t>(1), 4096);
    const auto pk = TensorAccessor(p_args, get_arg_val<uint32_t>(2), 4096);
    const uint32_t first = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    uint32_t q_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t q = it / PT, p = it % PT;
        if (q != q_prev) {
            cb_reserve_back(CB_IQ, 128);
            uint32_t a = get_write_ptr(CB_IQ);
            for (uint32_t h = 0; h < 32; h++) {
                for (uint32_t k = 0; k < 4; k++) {
                    noc_async_read(iq.get_noc_addr((h * QT + q) * 4 + k), a + (h * 4 + k) * IB, IB);
                }
            }
            cb_reserve_back(CB_W, 32);
            a = get_write_ptr(CB_W);
            for (uint32_t h = 0; h < 32; h++) {
                noc_async_read(wt.get_noc_addr(h * QT + q), a + h * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(CB_IQ, 128);
            cb_push_back(CB_W, 32);
            q_prev = q;
        }
        cb_reserve_back(CB_PK, 4);
        const uint32_t a = get_write_ptr(CB_PK);
        for (uint32_t k = 0; k < 4; k++) {
            noc_async_read(pk.get_noc_addr(4 * p + k), a + k * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(CB_PK, 4);
    }
}
