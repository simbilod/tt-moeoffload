// Fused hyper-connection mix reader (G3 step 3): the 4 residual streams X [4 x 2560] -> the mixer's input hn [2560]
// and the 4 inject weights, in two phases on the grid (compute fhc_compute.cpp, writer fhc_writer.cpp):
//   lo = [down | inject] xn with xn_s = X_s / rms(X_s) * norm_s: the norm folded into the matrix W' = diag(norm) W
//        [10240 x 352, 324 used], so lo = sum_s r_s (X_s W'_s), r_s = rsqrt(mean X_s^2 + eps)
//   gate = up silu(lo[:320] / 4)                          [320 -> 10240]
//   hn = 1/4 sum_s sigmoid(gate_s) X_s norm_s r_s,  w_s = 2 sigmoid(lo[320 + s] / 4)
// H1  item (n, s) on core 4 n + s (44 cores): r_s from X_s (3 dense fp32 tiles, cb 4, the last half zero), then
//     the 80 matmuls of X_s's k-tiles with W'_s's column tile n, scaled by r_s: the partial lo[s][n] to region P1;
//     the core with n = 0 also sends r_s.
// H2  channel tile j on core j (80 cores): lo's 10 tiles from the 4 partials each (cb 5), the gate's tiles
//     (s, j) = up's column tile 80 s + j (cb 0), the epilogue with X_s's tile j (cb 6), norm_s (cb 7) and r_s
//     filled across row 0 (cb 8): hn's tile j to region HN; core 80 computes w (lo's tile 10) to region W.
// The arena holds X (fp32 at X32, bf16 at X16; broadcast at barrier 1 here, the layer's previous phase in the full
// program), so H1 ends at barrier 2 and H2 at barrier 3.
// Runtime args: X fp32, X bf16, W', up, norm addresses, grid.h's block, the arena offsets X32, X16, P1, RS, HN, W,
// then the H1 item (n, s; 0xffffffff: none) and the H2 tile (j; 80: w; 0xffffffff: none).
// Compile args: batch (divides 10), then a DRAM accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_d = 4, cb_p = 5, cb_xs = 6, cb_nrm = 7, cb_rs = 8;
constexpr uint32_t cb_zero = 29, cb_arena = 30;
constexpr uint32_t W_PAGE = 1088, D = 2560, KS = 80, KW = 320, KU = 10, LO = 352, IDLE = 0xffffffff;

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    const auto x32 = TensorAccessor(d_args, get_arg_val<uint32_t>(0), 4 * D * 4);
    const auto x16 = TensorAccessor(d_args, get_arg_val<uint32_t>(1), 4 * D * 2);
    const auto wd = TensorAccessor(d_args, get_arg_val<uint32_t>(2), W_PAGE);
    const auto wu = TensorAccessor(d_args, get_arg_val<uint32_t>(3), W_PAGE);
    const auto norm = TensorAccessor(d_args, get_arg_val<uint32_t>(4), 4 * D * 4);
    const fused::Grid grid = fused::Grid::from_args(5);
    uint32_t i = 5 + fused::Grid::args(grid.nrects);
    const uint32_t r_x32 = get_arg_val<uint32_t>(i++), r_x16 = get_arg_val<uint32_t>(i++);
    const uint32_t r_p1 = get_arg_val<uint32_t>(i++), r_rs = get_arg_val<uint32_t>(i++);
    const uint32_t r_hn = get_arg_val<uint32_t>(i++), r_w = get_arg_val<uint32_t>(i++);
    const uint32_t item = get_arg_val<uint32_t>(i++), tile = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t zero = get_write_ptr(cb_zero);

    auto weights = [&](const auto& src, uint32_t first, uint32_t pages) {
        for (uint32_t p = 0; p < pages; p += batch) {
            cb_reserve_back(cb_w, batch);
            const uint32_t d = get_write_ptr(cb_w);
            for (uint32_t j = 0; j < batch; j++) {
                noc_async_read(src.get_noc_addr(first + p + j), d + j * W_PAGE, W_PAGE);
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, batch);
        }
    };
    auto arena_row0 = [&](uint32_t off, uint32_t dst) {
        fused::l1_copy(arena + off, dst, 64);
        fused::l1_copy(arena + off + 64, dst + 1024, 64);
    };
    auto fill_row0 = [&](uint32_t dst, uint32_t bits) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        for (uint32_t j = 0; j < 16; j++) {
            t[j] = bits;
            t[256 + j] = bits;
        }
    };

    // X into every arena (the full program's previous phase leaves it there): read by the coordinator alone and
    // broadcast at barrier 1 (110 cores reading the same pages would funnel ~6.6 MB through one or two banks)
    if (grid.coordinator) {
        noc_async_read(x32.get_noc_addr(0), arena + r_x32, 4 * D * 4);
        noc_async_read(x16.get_noc_addr(0), arena + r_x16, 4 * D * 2);
    }
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
        for (uint32_t j = 0; j < 1024; j++) {
            z[j] = 0;
        }
    }
    noc_async_read_barrier();
    grid.wait(1, arena + r_x32, r_p1 - r_x32);
    if (item != IDLE || tile != IDLE) {
        cb_reserve_back(cb_ones, 1);
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
        for (uint32_t j = 0; j < 512; j++) {
            ones[j] = 0x3f803f80;
        }
        cb_push_back(cb_ones, 1);
    }

    // ---- H1
    if (item != IDLE) {
        const uint32_t n = item >> 2, s = item & 3;
        cb_reserve_back(cb_d, 3);
        const uint32_t d = get_write_ptr(cb_d);
        fused::l1_copy(arena + r_x32 + s * D * 4, d, 8192);
        fused::l1_copy(arena + r_x32 + s * D * 4 + 8192, d + 8192, 2048);
        fused::l1_copy(zero, d + 10240, 2048);
        noc_async_read_barrier();
        cb_push_back(cb_d, 3);
        cb_reserve_back(cb_x, KS);
        const uint32_t xt = get_write_ptr(cb_x);
        for (uint32_t k = 0; k < KS; k++) {
            fused::l1_copy(arena + r_x16 + (s * KS + k) * 64, xt + k * 2048, 32);
            fused::l1_copy(arena + r_x16 + (s * KS + k) * 64 + 32, xt + k * 2048 + 512, 32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, KS);
        weights(wd, n * KW + s * KS, KS);
    }

    // ---- H2's inputs that need no activation, then the barrier
    if (tile != IDLE && tile < D / 32) {
        for (uint32_t s = 0; s < 4; s++) {
            weights(wu, (s * D / 32 + tile) * KU, KU);
        }
        cb_reserve_back(cb_nrm, 4);
        const uint32_t d = get_write_ptr(cb_nrm);
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(norm.get_noc_addr(0, (s * D + tile * 32) * 4), d + s * 4096, 64);
            noc_async_read(norm.get_noc_addr(0, (s * D + tile * 32) * 4 + 64), d + s * 4096 + 1024, 64);
        }
        noc_async_read_barrier();
        cb_push_back(cb_nrm, 4);
    }
    grid.wait(2, arena + r_p1, r_hn - r_p1);

    // ---- H2
    if (tile != IDLE) {
        const uint32_t t0 = tile < D / 32 ? 0 : LO / 32 - 1, t1 = tile < D / 32 ? LO / 32 - 1 : LO / 32;
        for (uint32_t t = t0; t < t1; t++) {  // lo's tiles 0..9 (the gate's input) or 10 (inject)
            cb_reserve_back(cb_p, 4);
            const uint32_t d = get_write_ptr(cb_p);
            for (uint32_t s = 0; s < 4; s++) {
                arena_row0(r_p1 + (s * LO + t * 32) * 4, d + s * 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb_p, 4);
        }
        if (tile < D / 32) {
            cb_reserve_back(cb_xs, 4);
            cb_reserve_back(cb_rs, 4);
            const uint32_t d = get_write_ptr(cb_xs), r = get_write_ptr(cb_rs);
            invalidate_l1_cache();
            for (uint32_t s = 0; s < 4; s++) {
                arena_row0(r_x32 + (s * D + tile * 32) * 4, d + s * 4096);
                fill_row0(r + s * 4096, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arena + r_rs + s * 16));
            }
            noc_async_read_barrier();
            cb_push_back(cb_xs, 4);
            cb_push_back(cb_rs, 4);
        }
    }
    grid.wait(3, arena + r_hn, r_w + 16 - r_hn);
}
