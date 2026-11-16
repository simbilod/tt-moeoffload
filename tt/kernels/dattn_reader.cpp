// Decode QSA partial reader: one core's share of one KV head's selected pools for one token. The cache is pool-major
// row-major ([kv head, pool, 4 cells x 256] bf16, a 2 KB page per pool), so each selected pool is one read for K and
// one for V; 8 pools fill a 32-cell block (32 rows x 256, the 8 pages a tilize takes) in cb 2 (K) and cb 3 (V).
// Once: the KV head's q^T tiles (8: dims x heads, the 1/16 scale folded in) into cb 0, the scaler tile (ones, for the
// reductions) into cb 5, and the core's NB mask tiles (cells x heads, -1e4 on cells it must not see) into cb 4.
// Runtime args: qT, k, v, pool ids, mask, scaler addresses; kv head, first pool slot, mask set. Compile args: nb,
// pools per head (cache), accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t nb = get_compile_time_arg_val(0);
    constexpr uint32_t npool = get_compile_time_arg_val(1);
    constexpr auto q_args = TensorAccessorArgs<2>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto qt = TensorAccessor(q_args, get_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(k_args, get_arg_val<uint32_t>(1));
    const auto vc = TensorAccessor(v_args, get_arg_val<uint32_t>(2));
    const auto ids = TensorAccessor(i_args, get_arg_val<uint32_t>(3));
    const auto mk = TensorAccessor(m_args, get_arg_val<uint32_t>(4));
    const auto sc = TensorAccessor(s_args, get_arg_val<uint32_t>(5));
    const uint32_t g = get_arg_val<uint32_t>(6), slot0 = get_arg_val<uint32_t>(7), mset = get_arg_val<uint32_t>(8);
    if (g == 0xffffffff) {
        return;
    }

    Noc noc;
    DataflowBuffer dq(0), dk(2), dv(3), dm(4), ds(5), did(6);
    const uint32_t tb = get_local_cb_interface(0).fifo_page_size, pb = get_local_cb_interface(2).fifo_page_size;
    dq.reserve_back(8);
    for (uint32_t d = 0; d < 8; d++) {
        noc.async_read(qt, dq, tb, {.page_id = g * 8 + d}, {.offset_bytes = d * tb});
    }
    ds.reserve_back(1);
    noc.async_read(sc, ds, tb, {.page_id = 0}, {});
    dm.reserve_back(nb);
    for (uint32_t b = 0; b < nb; b++) {
        noc.async_read(mk, dm, tb, {.page_id = mset * nb + b}, {.offset_bytes = b * tb});
    }
    // the pool ids: one row-major page of slots, this core's nb * 8 of them
    did.reserve_back(1);
    const uint32_t idb = get_local_cb_interface(6).fifo_page_size;
    noc.async_read(ids, did, idb, {.page_id = 0}, {});
    noc.async_read_barrier();
    dq.push_back(8);
    ds.push_back(1);
    dm.push_back(nb);
    volatile tt_l1_ptr uint32_t* pool = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(6));

    for (uint32_t b = 0; b < nb; b++) {
        dk.reserve_back(8);
        dv.reserve_back(8);
        for (uint32_t s = 0; s < 8; s++) {
            const uint32_t p = pool[slot0 + b * 8 + s];
            noc.async_read(kc, dk, pb, {.page_id = g * npool + p}, {.offset_bytes = s * pb});
            noc.async_read(vc, dv, pb, {.page_id = g * npool + p}, {.offset_bytes = s * pb});
        }
        noc.async_read_barrier();
        dk.push_back(8);
        dv.push_back(8);
    }
}
