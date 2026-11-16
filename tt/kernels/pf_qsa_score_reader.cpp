// The prefill selection's indexer score in one program (pf_qsa_score_compute.cpp, 2026-11-10): a core's run of pooled-
// key tiles (np pools-tiles x 4 k-tiles, fp32) read once into cb 1, then per pair of query tiles the 4 heads' iq tiles
// (2 x 4 x 4, order m, head, k) into cb 0. ttnn has no exact op for it: indexer_score_dsa is bf16 with fp32 Dest
// off, so its picks differ from the decode's fp32 selection (dedup audit C1).
// Runtime args: iq address, pk address, first pool tile, pool tiles, first query tile (of iq's), query tiles (even).
// Compile args: QT (iq's query tiles), IB (iq's tile bytes: fp32 as served, bf16), iq's accessor, pk's.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_IQ = 0, CB_PK = 1;

void kernel_main() {
    constexpr uint32_t QT = get_compile_time_arg_val(0);
    constexpr uint32_t IB = get_compile_time_arg_val(1);
    constexpr auto i_args = TensorAccessorArgs<2>();
    constexpr auto p_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const auto iq = TensorAccessor(i_args, get_arg_val<uint32_t>(0), IB);
    const auto pk = TensorAccessor(p_args, get_arg_val<uint32_t>(1), 4096);
    const uint32_t p0 = get_arg_val<uint32_t>(2), np = get_arg_val<uint32_t>(3);
    const uint32_t q0 = get_arg_val<uint32_t>(4), nq = get_arg_val<uint32_t>(5);
    if (np == 0) {
        return;
    }
    cb_reserve_back(CB_PK, 4 * np);
    uint32_t w = get_write_ptr(CB_PK);
    for (uint32_t i = 0; i < 4 * np; i++) {
        noc_async_read(pk.get_noc_addr(4 * p0 + i), w + i * 4096, 4096);
    }
    noc_async_read_barrier();
    cb_push_back(CB_PK, 4 * np);
    for (uint32_t q = 0; q < nq; q += 2) {
        cb_reserve_back(CB_IQ, 32);
        w = get_write_ptr(CB_IQ);
        for (uint32_t m = 0; m < 2; m++) {
            for (uint32_t h = 0; h < 4; h++) {
                for (uint32_t k = 0; k < 4; k++) {
                    noc_async_read(iq.get_noc_addr((h * QT + q0 + q + m) * 4 + k), w + ((m * 4 + h) * 4 + k) * IB, IB);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_IQ, 32);
    }
}
