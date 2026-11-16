// The MTP draft layer's routed experts, compute (the phases are in fmtp_exp_reader.cpp). A group (dexp_compute.cpp's
// arithmetic, the values streamed as down's are there): P = sum_k X_k F_k over the group's k-tiles, X_k holding x's
// k-tile k in row k % 32, so row b of P is k-tile b's partial; Y (+)= A * P with A the group's scale tile, the groups
// (and E2's experts) summed by the packer's L1 accumulation; the column sums through the all-ones tile.
// Unpack-to-dest (scales into Dest 1, w_j into 5, the logits) waits for the math thread only: Y leaves from Dest 2,
// h from Dest 7, and the weights' second half waits for the first half's pack.
// Runtime args: this core's index, the core count, E2's tile (or IDLE), 1 on the coordinator.
#include <cstdint>
#include "fused/cmp.h"
#include "fused/tail_cmp.h"
#include "api/compute/eltwise_unary/recip.h"

using namespace fused;

constexpr uint32_t cb_f = 0, cb_x = 1, cb_ones = 2, cb_s = 3, cb_lg = 4, cb_w = 5, cb_rs = 6, cb_y = 7;
constexpr uint32_t cb_h16 = 8, cb_xd = 9, cb_yv = 10, cb_xs = 11, cb_meta = 12, cb_wf = 13, cb_o32 = 14;
constexpr uint32_t cb_o16 = 15, cb_hf = 16, cb_hb = 17;
constexpr uint32_t TOPK = 10, KD = 20, IDLE = 0xffffffff;

// Y (+)= A * (sum over nk streamed F tiles of X F) into slot `slot` of cb_y, x from k-tile kx of xcb
inline void group(uint32_t xcb, uint32_t kx, uint32_t nk, uint32_t slot, bool first) {
    tile_regs_acquire();
    matmul_start(xcb, cb_f);
    for (uint32_t t = 0; t < nk; t++) {
        cb_wait_front(cb_f, 1);
        matmul_tiles(xcb, cb_f, kx + t, 0, 0);
        cb_pop_front(cb_f, 1);
    }
    cb_wait_front(cb_s, 1);
    load(cb_s, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 2);
    tile_regs_commit();
    cb_pop_front(cb_s, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_y);
    pack_reconfig_l1_acc(first ? 0 : 1);
    pack_tile<true>(2, cb_y, slot);
    pack_reconfig_l1_acc(0);
    tile_regs_release();
}

// Dest d = exp(Dest d - l_0) (neg: -l_0's bits)
inline void shifted_exp(uint32_t d, uint32_t neg) {
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup (a binary SFPU op ran since)
    SFPU_CALL_AT(affine, d, F_ONE, neg);
    exp_tile_init();
    exp_tile(d);
}

// w_j = exp(l_j - l_0) / sum_i exp(l_i - l_0) from the reader's tiles (row 0 = l_j) into cb_w (10 tiles)
inline void weights() {
    cb_wait_front(cb_meta, 1);
    const uint32_t neg = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    cb_wait_front(cb_lg, TOPK);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TOPK; j++) {
        const uint32_t d = j ? 1 : 0;
        load(cb_lg, j, d);
        shifted_exp(d, neg);
        if (j) {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
    }
    recip_tile_init();
    recip_tile(0);
    finish(cb_rs, 1);
    cb_wait_front(cb_rs, 1);
    for (uint32_t half = 0; half < 2; half++) {
        if (half) {
            cb_wait_front(cb_w, 5);  // the first half's pack of Dest 3..7 is done before they are loaded again
        }
        tile_regs_acquire();
        load(cb_rs, 0, 0);
        for (uint32_t k = 0; k < 5; k++) {
            load(cb_lg, 5 * half + k, 3 + k);
            shifted_exp(3 + k, neg);
        }
        mul_binary_tile_init();
        for (uint32_t k = 0; k < 5; k++) {
            mul_binary_tile(3 + k, 0, 3 + k);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_w, 5, 3);
        tile_regs_release();
    }
    cb_pop_front(cb_rs, 1);
    cb_pop_front(cb_lg, TOPK);
}

void kernel_main() {
    const uint32_t v0 = get_arg_val<uint32_t>(0), v1 = get_arg_val<uint32_t>(1), col = get_arg_val<uint32_t>(2);
    const uint32_t coordinator = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_f, cb_y);
    cb_wait_front(cb_ones, 1);
    if (coordinator) {  // P0: hn_f (fp32, 3 tiles of the contiguous vector) to bf16 for the broadcast
        cb_wait_front(cb_hf, 3);
        tile_regs_acquire();
        load_n(cb_hf, 3);
        cb_pop_front(cb_hf, 3);
        finish(cb_hb, 3);
    }
    weights();

    // ---- E1: gate column p and up column p (three groups each, summed in L1), then h = w_j silu(g) u, bf16
    cb_wait_front(cb_x, 80);
    cb_wait_front(cb_w, TOPK);
    for (uint32_t v = v0; v < v1; v++) {
        cb_reserve_back(cb_y, 2);
        for (uint32_t half = 0; half < 2; half++) {
            for (uint32_t g = 0; g < 3; g++) {
                group(cb_x, 32 * g, g == 2 ? 16 : 32, half, g == 0);
            }
        }
        cb_push_back(cb_y, 2);
        cb_wait_front(cb_y, 2);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_y);
        matmul_tiles(cb_ones, cb_y, 0, 0, 6);  // every row: g's column sums
        matmul_tiles(cb_ones, cb_y, 0, 1, 7);  // u's
        load(cb_w, v / 20, 5);
        silu_tile_init();
        silu_tile(6);
        mul_binary_tile_init();
        mul_binary_tile(6, 7, 6);
        mul_binary_tile(6, 5, 7);
        tile_regs_commit();
        cb_pop_front(cb_y, 2);
        tile_regs_wait();
        pack_n(cb_h16, 1, 7);
        tile_regs_release();
    }
    cb_pop_front(cb_x, 80);
    cb_pop_front(cb_w, TOPK);

    // ---- E2: y's tile col = sum_j down_j h_j, then X_s += w_f,s y
    if (col != IDLE) {
        cb_reserve_back(cb_y, 1);
        for (uint32_t j = 0; j < TOPK; j++) {
            cb_wait_front(cb_xd, KD);
            group(cb_xd, 0, KD, 0, j == 0);
            cb_pop_front(cb_xd, KD);
        }
        cb_push_back(cb_y, 1);
        cb_wait_front(cb_y, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_y);
        matmul_tiles(cb_ones, cb_y, 0, 0, 0);
        cb_pop_front(cb_y, 1);
        finish(cb_yv, 1);
        MATH((SFPU_UNARY_INIT(tanh)));
        tail::combine(cb_xs, cb_wf, cb_yv, cb_o32, cb_o16);
    }
}
