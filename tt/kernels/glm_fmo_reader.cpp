// GLM's fused MLA output, reader (compute glm_fmo_compute.cpp, writer glm_fmo_writer.cpp, host
// tt/glm/g3_fmo.py): the attention's latent outputs o_lat [64 heads x 512] -> the layer's attention output [4096], in
// two phases on the whole grid with a barrier after each (fused/grid.h):
//
// E1  per head h (cores 0..63): o_h = o_lat_h v_b_h (512 -> 256; v_b_h's 128 bfp8_b tiles at pages 128 h + 16 n + k),
//     o_lat_h read from the flash-MLA output tiles (row h % 32 of group h / 32's 16 bf16 tiles); o (bf16) to region B
// E2  y = o W_o (16384 -> 4096), W_o's tiles n-major (page 512 n + k), split by K in S parts: item i = (n = i / S,
//     part = i % S) covers pages 512 n + (512 / S) part .. + 512 / S, so a core's run of items is one page range;
//     each item's partial row to region P ([S][128 tiles][32] fp32), gathered and broadcast at barrier 2
// SUM each core adds the S partials of its output tiles (this RISC-V's FPU, fused/fpu.h) and writes them to out
//     (row-major fp32 [4096]: the mHC site's input)
//
// x travels as mgemv's ring (a batch of BATCH x tiles beside each weight batch). E2's first pages are read before
// barrier 1 (they need no x); their x batches follow after it, in order.
// Runtime args: grid.h's block, then o_lat group 0 and 1 addresses, v_b, W_o, out addresses, head (0xffffffff:
// none), E2's first item and count, the first output tile this core sums and their count, arena offsets of the x row,
// region B, region P. Compile args: batch, depth, S, then a DRAM accessor.
//
// M rows (-DM_MAX = 2..4; M_MAX = 1 compiles the one-row source): the step tensor's mp (a runtime arg after
// region P), the groups [1, M_MAX, 32, 512] (row m in tile m 16 + t), x and region B as fused/hc_dm.h's mp-row
// blocks, E2's partials sent by owner (glm_fkda's C: fused::ksplit_sum_m, 4 KB a core), out as mp-row blocks.
#include <cstdint>
#ifndef M_MAX
#define M_MAX 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#include "fused/ksum.h"
#include "fused/fpu.h"
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_scratch = 2, cb_arena = 30, cb_stage = 31;
constexpr uint32_t W_PAGE = 1088, K1 = 16, KO = 512, NT = 128, IDLE = 0xffffffff;

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
inline volatile tt_l1_ptr float* f32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); }

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t KP = KO / S;                       // k-tiles an item
    constexpr auto d_args = TensorAccessorArgs<3>();
    const fused::Grid grid = fused::Grid::from_args(0);
    uint32_t i = fused::Grid::args(grid.nrects);
    const auto og0 = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 2048);
    const auto og1 = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 2048);
    const auto vb = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto wo = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto out = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), NT * 128);
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const uint32_t it0 = get_arg_val<uint32_t>(i++), n_it = get_arg_val<uint32_t>(i++);
    const uint32_t s0 = get_arg_val<uint32_t>(i++), n_s = get_arg_val<uint32_t>(i++);
    const uint32_t xrow = get_write_ptr(cb_arena) + get_arg_val<uint32_t>(i++);
    const uint32_t region_b = get_write_ptr(cb_arena) + get_arg_val<uint32_t>(i++);
    const uint32_t region_p = get_write_ptr(cb_arena) + get_arg_val<uint32_t>(i++);

    fused::Stream<decltype(wo)> ws{cb_w, batch, W_PAGE};   // the weight stream (the open batch persists)
#if M_MAX > 1
    {
    const auto step = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 64);
    noc_async_read(step.get_noc_addr(0), get_write_ptr(cb_stage), 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t mp = w32(get_write_ptr(cb_stage))[4];
    if (head != IDLE) {   // E1: rows m < mp of o_lat_h from tile (m, t), as x blocks (bf16: face-0 rows, face-1 rows)
        const auto& og = head < 32 ? og0 : og1;
        const uint32_t r = head % 32, sc = get_write_ptr(cb_scratch);
        for (uint32_t m = 0; m < mp; m++) {
            for (uint32_t t = 0; t < K1; t++) {
                noc_async_read(og.get_noc_addr(m * K1 + t, (r >> 4) * 1024), sc + (m * K1 + t) * 1024, 1024);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t m = 0; m < mp; m++) {
            for (uint32_t t = 0; t < K1; t++) {
                const uint32_t src = sc + (m * K1 + t) * 1024 + (r & 15) * 32, dst = xrow + t * mp * 64 + m * 32;
                for (uint32_t e = 0; e < 8; e++) {
                    w32(dst)[e] = w32(src)[e];                       // cols 0..15
                    w32(dst + mp * 32)[e] = w32(src + 512)[e];       // cols 16..31
                }
            }
        }
        asm volatile("fence rw, rw" ::: "memory");
        for (uint32_t p = 0; p < 8 * K1; p += batch) {
            fused::x_ring_m_issue(cb_x, xrow, p % K1, batch, mp);
            ws.read(vb, head * 8 * K1 + p, batch);
            cb_push_back(cb_x, batch);
        }
    }
    const uint32_t pages = n_it * KP, page0 = it0 * KP;
    const uint32_t pre = pages < depth * batch ? pages : depth * batch;
    ws.read(wo, page0, pre);
    grid.wait(1, region_b, 64 * 8 * mp * 64);
    for (uint32_t p = 0; p < pages; p += batch) {   // E2
        fused::x_ring_m_issue(cb_x, region_b, (page0 + p) % KO, batch, mp);
        if (p >= pre) {
            ws.read(wo, page0 + p, batch);
        } else {
            noc_async_read_barrier();
        }
        cb_push_back(cb_x, batch);
    }
    grid.wait(2);
    fused::ksplit_sum_m(out, region_p, S, s0, n_s, mp * 128, get_write_ptr(cb_stage));
    return;
    }
#endif

    // ---- E1: o_lat_h from its group's tiles: row r = h % 32 lives in faces (r / 16) * 2 and + 1, row r % 16; each
    // tile's two faces (1 KB, aligned) into scratch, then the row's 2 x 32 bytes copied by this RISC-V (a NoC read
    // whose source and destination differ in alignment returns wrong bytes)
    if (head != IDLE) {
        const auto& og = head < 32 ? og0 : og1;
        const uint32_t r = head % 32, sc = get_write_ptr(cb_scratch);
        for (uint32_t t = 0; t < K1; t++) noc_async_read(og.get_noc_addr(t, (r >> 4) * 1024), sc + t * 1024, 1024);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t t = 0; t < K1; t++) {
            for (uint32_t e = 0; e < 8; e++) {
                w32(xrow + t * 64)[e] = w32(sc + t * 1024 + (r & 15) * 32)[e];             // cols 0..15
                w32(xrow + t * 64 + 32)[e] = w32(sc + t * 1024 + 512 + (r & 15) * 32)[e];  // cols 16..31
            }
        }
        for (uint32_t p = 0; p < 8 * K1; p += batch) {
            fused::x_ring(cb_x, xrow, p % K1, batch);
            ws.read(vb, head * 8 * K1 + p, batch);
        }
    }
    // E2's first pages, before the barrier
    const uint32_t pages = n_it * KP, page0 = it0 * KP;
    const uint32_t pre = pages < depth * batch ? pages : depth * batch;
    ws.read(wo, page0, pre);
    grid.wait(1, region_b, 64 * 256 * 2);

    // ---- E2
    for (uint32_t p = 0; p < pages; p += batch) {
        fused::x_ring(cb_x, region_b, (page0 + p) % KO, batch);
        if (p >= pre) ws.read(wo, page0 + p, batch);
    }
    grid.wait(2, region_p, S * NT * 128);

    // ---- SUM
    fused::ksplit_sum(out, region_p, S, NT, s0, n_s, 0, get_write_ptr(cb_stage));
}
