// Fused DeltaNet mixer reader (G3 step 2): one layer's gated DeltaNet mixer, hn -> out, in three phases on the whole
// grid with a barrier between them (tt/kernels/fused/grid.h; compute in fgdn_compute.cpp, writer fgdn_writer.cpp).
//
// A  the input projection y = hn W (2560 -> 16512) on all cores, each core's output tiles followed by their
//    channel-local epilogue. W's bfp8_b tiles are stored n-major (page n * KA + k), its columns q | k | v (10240, the
//    causal conv's channels) | z (6144) | alpha (48, padded to 64) | beta (48, padded to 64). A core's work is a
//    list of units, each a run of output tiles of one type: 0 v tile (conv + silu), 1 q head and 2 k head (4 tiles:
//    conv + silu + L2 norm), 3 z (sigmoid), 4 alpha (decay), 5 beta (sigmoid). The results (fp32) are gathered into
//    the arena's region A.
// B  the delta rule and the gated RMSNorm, one value head per core (48 cores), from region A; the state (128 x 128
//    fp32, 16 tiles) comes from card memory and goes back; the head's 128 outputs (bf16) into region B.
// C  the output projection out = o W_o (6144 -> 2560) on 80 cores, from region B, into region C (fp32).
//
// Reader, in order: hn into the arena and its row-0 pieces into KC x tiles (cb 1, padded to KC so the buffer ends
// where it began), the all-ones tile (cb 2); A's constants (cb 3, 7 fp32 tiles per output tile, row 0 used: the 3
// conv-state rows then the 4 taps for conv tiles, dt and neg_a for alpha); A's weights (cb 0); B's state twice
// (cb 8 exact, cb 9 the FPU's operand) and the norm weight (cb 10); C's first DEPTH weight batches; barrier 1 (A
// broadcast); B's inputs from region A (cb 11 q and cb 12 k, zero outside row 0; cb 13 v, sigmoid(z), beta, decay,
// the last two filled across row 0); barrier 2 (B broadcast); C's x tiles from region B and the rest of its weights;
// barrier 3 (C broadcast; no core leaves before it lands). Weights travel in batches of BATCH, which divides KA, KC.
// Dense arrays are row-major fp32 tensors, a row a page: conv state [3, 10240], taps [4, 10240], [dt; neg_a] [2, 64],
// the norm weight [1, 128]. The DeltaNet state is tiles, head h's tile (j, c) at page 16 h + 4 j + c.
// Runtime args: hn address, grid.h's block, W, conv state, taps, dt|neg_a, unit count, per unit (first tile, count,
// type), then the head (0xffffffff: none), state, norm weight, then W_o, C's first tile and count, then the arena
// offsets of regions A, B, C. Compile args: batch, depth, phases (bit 0 A, 1 B, 2 C), then a DRAM accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/grid.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_c = 3, cb_sx = 8, cb_sop = 9, cb_nw = 10;
constexpr uint32_t cb_q = 11, cb_k = 12, cb_ex = 13, cb_zero = 29, cb_arena = 30;
constexpr uint32_t W_PAGE = 1088, KA = 80, KC = 192, CONV = 10240, NT_A = 516, NV = 48, HK = 128;
constexpr uint32_t T_V = 0, T_Q = 1, T_K = 2, T_Z = 3, T_A = 4, T_B = 5;
constexpr uint32_t IDLE = 0xffffffff;
constexpr uint32_t A_ALPHA = (CONV + 6144) * 4, A_BETA = A_ALPHA + 64 * 4;  // byte offsets in region A

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t phases = get_compile_time_arg_val(2);
    constexpr auto d_args = TensorAccessorArgs<3>();
    const auto hn = TensorAccessor(d_args, get_arg_val<uint32_t>(0), KA * 64);
    const fused::Grid grid = fused::Grid::from_args(1);
    uint32_t i = 1 + fused::Grid::args(grid.nrects);
    const auto w = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto state = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), CONV * 4);
    const auto taps = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), CONV * 4);
    const auto dtna = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 64 * 4);
    const uint32_t n_units = get_arg_val<uint32_t>(i++);
    const uint32_t ua = i;
    i += 3 * n_units;
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const auto S = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 4096);
    const auto nw = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), HK * 4);
    const auto wo = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const uint32_t c_first = get_arg_val<uint32_t>(i++), c_count = get_arg_val<uint32_t>(i++);
    const uint32_t region_a = get_arg_val<uint32_t>(i++), region_b = get_arg_val<uint32_t>(i++);
    const uint32_t region_c = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena);

    // a row-0 fp32 tile from 32 consecutive floats of row `row` of a row-major array, from byte offset off
    auto row0 = [&](const auto& src, uint32_t row, uint32_t off, uint32_t dst) {
        noc_async_read(src.get_noc_addr(row, off), dst, 64);
        noc_async_read(src.get_noc_addr(row, off + 64), dst + 1024, 64);
    };
    // the same from the arena (L1)
    auto arena_row0 = [&](uint32_t off, uint32_t dst) {
        fused::l1_copy(arena + off, dst, 64);
        fused::l1_copy(arena + off + 64, dst + 1024, 64);
    };
    // n x tiles (bf16, row 0) from the bf16 vector at arena offset off, pushed as KC tiles
    auto x_tiles = [&](uint32_t off, uint32_t n) {
        cb_reserve_back(cb_x, KC);
        const uint32_t xt = get_write_ptr(cb_x);
        for (uint32_t k = 0; k < n; k++) {
            fused::l1_copy(arena + off + k * 64, xt + k * 2048, 32);
            fused::l1_copy(arena + off + k * 64 + 32, xt + k * 2048 + 512, 32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, KC);
    };
    // a weight stream over `pages` consecutive pages of src from page first, in batches (the open one persists)
    uint32_t in_batch = 0, wdst = 0;
    auto stream = [&](const auto& src, uint32_t first, uint32_t pages) {
        for (uint32_t p = first; p < first + pages; p++) {
            if (in_batch == 0) {
                cb_reserve_back(cb_w, batch);
                wdst = get_write_ptr(cb_w);
            }
            noc_async_read(src.get_noc_addr(p), wdst + in_batch * W_PAGE, W_PAGE);
            if (++in_batch == batch) {
                noc_async_read_barrier();
                cb_push_back(cb_w, batch);
                in_batch = 0;
            }
        }
    };

    uint32_t a_tiles = 0;
    for (uint32_t u = 0; u < n_units; u++) {
        a_tiles += get_arg_val<uint32_t>(ua + 3 * u + 1);
    }
    const bool any = a_tiles || head != IDLE || c_count;

    noc_async_read(hn.get_noc_addr(0), arena, KA * 64);
    noc_async_read_barrier();
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_zero));
        for (uint32_t j = 0; j < 1024; j++) {
            z[j] = 0;
        }
    }
    if (any) {
        cb_reserve_back(cb_ones, 1);
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
        for (uint32_t j = 0; j < 512; j++) {
            ones[j] = 0x3f803f80;
        }
        cb_push_back(cb_ones, 1);
    }

    // ---- A
    if ((phases & 1) && a_tiles) {
        x_tiles(0, KA);
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            if (type > T_K && type != T_A) {
                continue;
            }
            for (uint32_t n = first; n < first + count; n++) {
                cb_reserve_back(cb_c, 7);
                const uint32_t c = get_write_ptr(cb_c);
                if (type == T_A) {
                    const uint32_t off = (n - CONV / 32 - 192) * 128;  // within the padded alpha block
                    row0(dtna, 0, off, c);
                    row0(dtna, 1, off, c + 4096);
                } else {
                    for (uint32_t j = 0; j < 3; j++) {
                        row0(state, j, n * 128, c + j * 4096);
                    }
                    for (uint32_t j = 0; j < 4; j++) {
                        row0(taps, j, n * 128, c + (3 + j) * 4096);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_c, 7);
            }
        }
        for (uint32_t u = 0; u < n_units; u++) {
            stream(w, get_arg_val<uint32_t>(ua + 3 * u) * KA, get_arg_val<uint32_t>(ua + 3 * u + 1) * KA);
        }
    }

    // ---- B's and C's inputs that need no activation, before the barrier
    if ((phases & 2) && head != IDLE) {
        for (uint32_t cb : {cb_sx, cb_sop}) {
            cb_reserve_back(cb, 16);
            const uint32_t d = get_write_ptr(cb);
            for (uint32_t t = 0; t < 16; t++) {
                noc_async_read(S.get_noc_addr(head * 16 + t), d + t * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb, 16);
        }
        cb_reserve_back(cb_nw, 4);
        const uint32_t d = get_write_ptr(cb_nw);
        for (uint32_t c = 0; c < 4; c++) {
            row0(nw, 0, c * 128, d + c * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_nw, 4);
    }
    const uint32_t c_pages = c_count * KC;
    const uint32_t c_pre = (phases & 4) ? (c_pages < depth * batch ? c_pages : depth * batch) : 0;
    stream(wo, c_first * KC, c_pre);

    if (!(phases & 1)) {
        return;
    }
    grid.wait(1, arena + region_a, NT_A * 128);

    // ---- B
    if (phases & 2) {
        if (head != IDLE) {
            const uint32_t kh = head % 16;  // value head h reads key head h % 16 (ggml_repeat)
            const uint32_t zero = get_write_ptr(cb_zero);
            for (uint32_t cb : {cb_q, cb_k}) {  // zero outside row 0: k becomes a column by transpose
                cb_reserve_back(cb, 4);
                const uint32_t d = get_write_ptr(cb);
                for (uint32_t t = 0; t < 4; t++) {
                    fused::l1_copy(zero, d + t * 4096, 4096);
                }
            }
            noc_async_read_barrier();
            const uint32_t q = get_write_ptr(cb_q), k = get_write_ptr(cb_k);
            for (uint32_t t = 0; t < 4; t++) {
                arena_row0(region_a + (kh * HK + t * 32) * 4, q + t * 4096);
                arena_row0(region_a + (2048 + kh * HK + t * 32) * 4, k + t * 4096);
            }
            cb_reserve_back(cb_ex, 10);
            const uint32_t e = get_write_ptr(cb_ex);
            for (uint32_t t = 0; t < 4; t++) {
                arena_row0(region_a + (4096 + head * HK + t * 32) * 4, e + t * 4096);        // v
                arena_row0(region_a + (CONV + head * HK + t * 32) * 4, e + (4 + t) * 4096);  // sigmoid(z)
            }
            noc_async_read_barrier();
            cb_push_back(cb_q, 4);
            cb_push_back(cb_k, 4);
            // beta and decay across row 0 (the RISC-V copies the word; no float math here)
            invalidate_l1_cache();
            const uint32_t bits[2] = {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arena + region_a + A_BETA + head * 4),
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arena + region_a + A_ALPHA + head * 4)};
            for (uint32_t s = 0; s < 2; s++) {
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e + (8 + s) * 4096);
                for (uint32_t j = 0; j < 16; j++) {
                    t[j] = bits[s];
                    t[256 + j] = bits[s];
                }
            }
            cb_push_back(cb_ex, 10);
        }
        grid.wait(2, arena + region_b, NV * HK * 2);
    }

    // ---- C
    if (phases & 4) {
        if (c_count) {
            x_tiles(region_b, KC);
        }
        stream(wo, c_first * KC + c_pre, c_pages - c_pre);
        grid.wait(3, arena + region_c, 80 * 128);
    }
}
