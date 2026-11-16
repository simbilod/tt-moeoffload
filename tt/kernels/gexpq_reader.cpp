// Grouped expert reader, native codes (G2 prefill): one core's run of experts, y_e = x_e W_e with W_e in the card's
// form of its GGUF quant (tt/formats/repack.py): per (n-tile, 8 k-tiles) a code tile of uint32 words (nibble t of
// word [k][n] = the code of k-tile 8c + t), per (n-tile, 32 k-tiles) an fp32 scale tile (row b = block b's scale per
// column). Per expert: its MT token tiles x kt k-tiles into cb 1 (padded to MT_MAX x kt pushes); then per n-tile its
// kc code tiles into cb 0 and kq scale tiles into cb 3, each scale tile ahead of its 4 code tiles. REPS times over.
// Runtime args: codes, scales, x addresses; first expert, count. Compile args: reps, kt, nt, mt, mt_max, kc, kq,
// accessors.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t reps = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t mt_max = get_compile_time_arg_val(4);
    constexpr uint32_t kc = get_compile_time_arg_val(5);
    constexpr uint32_t kq = get_compile_time_arg_val(6);
    constexpr auto c_args = TensorAccessorArgs<7>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto codes = TensorAccessor(c_args, get_arg_val<uint32_t>(0));
    const auto scales = TensorAccessor(s_args, get_arg_val<uint32_t>(1));
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(2));
    const uint32_t e_first = get_arg_val<uint32_t>(3);
    const uint32_t e_count = get_arg_val<uint32_t>(4);

    Noc noc;
    DataflowBuffer dc(0), dx(1), ds(3);
    const uint32_t cb = get_local_cb_interface(0).fifo_page_size, xb = get_local_cb_interface(1).fifo_page_size;
    const uint32_t sb = get_local_cb_interface(3).fifo_page_size;
    for (uint32_t rep = 0; rep < reps; rep++) {
        for (uint32_t e = e_first; e < e_first + e_count; e++) {
            dx.reserve_back(mt_max * kt);
            for (uint32_t i = 0; i < mt * kt; i++) {
                noc.async_read(x, dx, xb, {.page_id = e * mt * kt + i}, {.offset_bytes = i * xb});
            }
            noc.async_read_barrier();
            dx.push_back(mt_max * kt);
            for (uint32_t n = 0; n < nt; n++) {
                const uint32_t u = e * nt + n;
                for (uint32_t c = 0; c < kc; c++) {
                    if (c % 4 == 0) {
                        ds.reserve_back(1);
                        noc.async_read(scales, ds, sb, {.page_id = u * kq + c / 4}, {});
                    }
                    dc.reserve_back(1);
                    noc.async_read(codes, dc, cb, {.page_id = u * kc + c}, {});
                    noc.async_read_barrier();
                    if (c % 4 == 0) {
                        ds.push_back(1);
                    }
                    dc.push_back(1);
                }
            }
        }
    }
}
