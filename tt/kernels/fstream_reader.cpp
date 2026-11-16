// Weight-stream rate per core (g3_stream.py): this core reads its run of bfp8_b tile pages from card memory in batches,
// up to `inflight` batches outstanding (each tagged with its own transaction id, 1..inflight), into a ring in L1, and
// records the cycles from its first read to its last arrival. No compute: the read rate alone.
// Runtime args: W, first page, pages, the out tensor (64 bytes per core), this core's slot. Compile args: batch,
// inflight, then a DRAM accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t cb_ring = 0, cb_stamp = 1;
constexpr uint32_t W_PAGE = 1088;

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t inflight = get_compile_time_arg_val(1);
    constexpr auto d_args = TensorAccessorArgs<2>();
    const auto w = TensorAccessor(d_args, get_arg_val<uint32_t>(0), W_PAGE);
    const uint32_t first = get_arg_val<uint32_t>(1), pages = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(d_args, get_arg_val<uint32_t>(3), 64 * 128);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t ring = get_write_ptr(cb_ring);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t t0 = get_timestamp_32b();
    const uint32_t batches = pages / batch;
    for (uint32_t b = 0; b < batches; b++) {
        const uint32_t s = b % inflight;
        if (b >= inflight) {
            noc_async_read_barrier_with_trid(s + 1);  // the slot's previous batch has landed: reuse it
        }
        noc_async_read_set_trid(s + 1);
        for (uint32_t p = 0; p < batch; p++) {
            noc_async_read(w.get_noc_addr(first + b * batch + p), ring + (s * batch + p) * W_PAGE, W_PAGE);
        }
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
    const uint32_t t1 = get_timestamp_32b();

    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stamp));
    for (uint32_t j = 0; j < 16; j++) {
        st[j] = 0;
    }
    st[0] = t1 - t0;
    st[1] = 1;
    noc_async_write(get_write_ptr(cb_stamp), out.get_noc_addr(0, slot * 64), 64);
    noc_async_write_barrier();
}
