// Fused decode head compute (fhead_reader.cpp): the hyper-connection mix (fused/hc_cmp.h), then the LM head's output
// tiles, KD matmuls each into Dest 0, to cb 16 (fp32). Runtime args: the H1 item, the H2 tile, the LM head's tile
// count. Compile args: batch.
// With X_BLOCKS (fhead_reader.cpp): the mix at M rows (fused/hc_cmp.h compute_m, mp from the reader's cb 2 tile 2).
#include <cstdint>
#include "fused/cmp.h"
#include "fused/hc_cmp.h"

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_out = 16, KD = 80;

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t h_item = get_arg_val<uint32_t>(0), h_tile = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
#ifdef X_BLOCKS
    cb_wait_front(cb_ones, 3);  // the layout's rows: the reader's words in cb 2, tile 2
    const uint32_t mp = read_tile_value(cb_ones, 2, 3);
    Weights w{cb_w, batch};
    hc::compute_m(h_item, h_tile, w, mp);
#else
    cb_wait_front(cb_ones, 1);
    Weights w{cb_w, batch};
    hc::compute(h_item, h_tile, w);
#endif
    cb_wait_front(cb_x, KD);
    for (uint32_t n = 0; n < count; n++) {
        tile_regs_acquire();
        matmul_start(cb_x, cb_w);
        w.gemv(cb_x, KD, 0);
        finish(cb_out, 1);
    }
    w.end();
    cb_pop_front(cb_x, KD);
}
