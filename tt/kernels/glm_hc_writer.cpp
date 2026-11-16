// GLM's mHC site on the card, writer (glm_hc_reader.cpp has the phases). A: the partial mixes (row 0
// of the fp32 tile: 64 bytes from face 0, 64 from face 1) and the partial sum of squares (cb_sq) to the coordinator's
// arena (core i at mixes + 128 i, squares + 16 i: NoC writes into L1 land only 16-byte aligned), then barrier 1. B and C: each tile to its DRAM page of `res` (h
// tiles at t, the new stream tiles at 128 + t), for the host's check.
// With HC_NORM (2d.4c): after B the sublayer's input norm, hn = h / rms(h) * w (eps 1e-5) as bf16, on this RISC-V's
// FPU (fused/fpu.h): each core's sum of squares of its h tiles to the arena (16-byte slots after the parameters),
// barrier 2 (the coordinator's reader multicasts them), then each core sums them, takes the rsqrt and writes its 64
// values of hn to the row-major bf16 tensor hn (its consumer's input: the mixer's, or the FFN's).
// Runtime args: the grid block, res address, core index i, first column tile, count, phases (as the reader's), xn (0:
// C's new streams to res's pages 128 + t; else to xn's pages t: the next site's x, so no copy between the sites),
// then with HC_NORM hn's and the norm weight's addresses. Compile args: NC, res's accessor args, then with HC_NORM hn's
// and the weight's.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/fpu.h"
#include "fused/grid.h"
#ifndef M_MAX
#define M_MAX 1   // > 1: the M-row build (glm_hc_reader.cpp); the step tensor a runtime arg after the others
#endif

constexpr uint32_t cb_sq = 6, cb_arena = 8, cb_stage = 9, cb_out = 16, TILE_F32 = 4096, D = 4096;

inline volatile tt_l1_ptr float* f32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); }

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr auto res_args = TensorAccessorArgs<1>();
    constexpr uint32_t SQ = NC * 128;
    const fused::Grid g = fused::Grid::from_args(0);
    const uint32_t a = fused::Grid::args(g.nrects);
    const auto res = TensorAccessor(res_args, get_arg_val<uint32_t>(a), TILE_F32);
    const uint32_t i = get_arg_val<uint32_t>(a + 1), t0 = get_arg_val<uint32_t>(a + 2), nt = get_arg_val<uint32_t>(a + 3);
    const uint32_t phases = get_arg_val<uint32_t>(a + 4), xn_addr = get_arg_val<uint32_t>(a + 5);
    const auto xn = TensorAccessor(res_args, xn_addr, TILE_F32);
    const uint32_t arena = get_write_ptr(cb_arena);
#if M_MAX > 1
    constexpr uint32_t ROW_PAGE = NC * 144, PARAM = M_MAX * ROW_PAGE, SQ2M = PARAM + 128 * M_MAX + 64;
#ifdef HC_NORM
    constexpr uint32_t n_args = 8;
#else
    constexpr uint32_t n_args = 6;
#endif
    const uint32_t st = get_write_ptr(cb_stage);
    noc_async_read(TensorAccessor(res_args, get_arg_val<uint32_t>(a + n_args), 64).get_noc_addr(0), st, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st)[4];
#endif
#ifdef HC_NORM
    constexpr auto hn_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto nw_args = TensorAccessorArgs<hn_args.next_compile_time_args_offset()>();
    constexpr uint32_t SQ2 = SQ + NC * 16 + 128;           // after the squares and the parameters (the reader's PARAM)
    const auto hn = TensorAccessor(hn_args, get_arg_val<uint32_t>(a + 6), M_MAX * D * 2);
    const auto nw = TensorAccessor(nw_args, get_arg_val<uint32_t>(a + 7), D * 4);
    // stage: the h rows (nt x 32 fp32), the weight's (nt x 32 fp32), hn's (nt x 32 bf16)
#if M_MAX > 1
    const uint32_t wrow = get_write_ptr(cb_stage) + 64, hrow = wrow + 256, nrow = hrow + M_MAX * 256;   // h per row
#else
    const uint32_t hrow = get_write_ptr(cb_stage), wrow = hrow + 256, nrow = wrow + 256;
#endif
    if (nt && (phases & 1)) {
        noc_async_read(nw.get_noc_addr(0, t0 * 128), wrow, nt * 128);
        noc_async_read_barrier();
    }
#endif
    if (nt && (phases & 1)) {
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        cb_wait_front(cb_sq, 1);
#if M_MAX > 1
        // row m's mixes and square (the compute's sums tile: row m, column 0 at word 16 m) to row page m
        const uint32_t sq = get_read_ptr(cb_sq);
        for (uint32_t m = 0; m < M_MAX; m++) {
            g.send(src + m * 64, arena + m * ROW_PAGE + i * 128, 64);
            g.send(src + 1024 + m * 64, arena + m * ROW_PAGE + i * 128 + 64, 64);
            g.send(sq + m * 64, arena + m * ROW_PAGE + SQ + i * 16, 16);
        }
#else
        g.send(src, arena + i * 128, 64);
        g.send(src + 1024, arena + i * 128 + 64, 64);
        g.send(get_read_ptr(cb_sq), arena + SQ + i * 16, 16);   // a 16-byte slot: NoC writes to L1 align to 16
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
        cb_pop_front(cb_sq, 1);
    }
    if (phases & 1) g.arrive(1);
    for (uint32_t phase = 0; phase < 2; phase++) {
        if (!(phases & (1u << phase))) continue;
        for (uint32_t j = 0; j < nt; j++) {
            cb_wait_front(cb_out, 1);
            const uint64_t dst = (phase && xn_addr) ? xn.get_noc_addr(t0 + j) : res.get_noc_addr(phase * 128 + t0 + j);
            noc_async_write(get_read_ptr(cb_out), dst, TILE_F32);
#ifdef HC_NORM
            if (phase == 0) {                               // h's row 0: face 0's 16 words, face 1's
                invalidate_l1_cache();
                const uint32_t t = get_read_ptr(cb_out);
#if M_MAX > 1
                for (uint32_t m = 0; m < mp; m++) {         // rows m: row m's h at hrow + 256 m
                    for (uint32_t c = 0; c < 16; c++) {
                        f32(hrow + m * 256 + j * 128)[c] = f32(t + m * 64)[c];
                        f32(hrow + m * 256 + j * 128)[16 + c] = f32(t + 1024 + m * 64)[c];
                    }
                }
#else
                for (uint32_t c = 0; c < 16; c++) {
                    f32(hrow + j * 128)[c] = f32(t)[c];
                    f32(hrow + j * 128)[16 + c] = f32(t + 1024)[c];
                }
#endif
            }
#endif
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
#ifdef HC_NORM
#if M_MAX > 1
        if (phase == 0) {   // the norm per row; hn as mp-row blocks (bf16) of tiles t0 ..
            for (uint32_t m = 0; m < mp && nt; m++) {
                float ss = 0.0f;
                for (uint32_t c = 0; c < nt * 32; c++) {
                    ss = fused::fpu::add(ss, fused::fpu::mul(f32(hrow + m * 256)[c], f32(hrow + m * 256)[c]));
                }
                f32(nrow + m * 16)[0] = ss;
                g.send(nrow + m * 16, arena + SQ2M + (m * NC + i) * 16, 16);
            }
            noc_async_write_barrier();
            g.arrive(2);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 2);
            if (nt) {
                invalidate_l1_cache();
                volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(nrow + 64);
                for (uint32_t m = 0; m < mp; m++) {
                    float tot = 0.0f;
                    for (uint32_t k = 0; k < NC; k++) tot = fused::fpu::add(tot, f32(arena + SQ2M + (m * NC + k) * 16)[0]);
                    const float rs = fused::fpu::rsqrt(fused::fpu::add(fused::fpu::mul(tot, 1.0f / D), 1e-5f));
                    for (uint32_t c = 0; c < nt * 32; c++) {
                        const uint32_t b =
                            fused::fpu::f_bits(fused::fpu::mul(fused::fpu::mul(f32(hrow + m * 256)[c], rs), f32(wrow)[c]));
                        const uint32_t jt = c / 32, cc = c % 32;   // tile jt's block: face-0 rows, then face-1 rows
                        o[jt * mp * 32 + (cc < 16 ? 0 : mp * 16) + m * 16 + (cc & 15)] =
                            static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
                    }
                }
                noc_async_write(nrow + 64, hn.get_noc_addr(0, t0 * mp * 64), nt * mp * 64);
                noc_async_write_barrier();
            }
        }
#else
        if (phase == 0) {
            float ss = 0.0f;
            for (uint32_t c = 0; c < nt * 32; c++) ss = fused::fpu::add(ss, fused::fpu::mul(f32(hrow)[c], f32(hrow)[c]));
            if (nt) {
                f32(nrow)[0] = ss;
                g.send(nrow, arena + SQ2 + i * 16, 16);
                noc_async_write_barrier();
            }
            g.arrive(2);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 2);
            if (nt) {
                invalidate_l1_cache();
                float tot = 0.0f;
                for (uint32_t k = 0; k < NC; k++) tot = fused::fpu::add(tot, f32(arena + SQ2 + k * 16)[0]);
                const float rs = fused::fpu::rsqrt(fused::fpu::add(fused::fpu::mul(tot, 1.0f / D), 1e-5f));
                volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(nrow);
                for (uint32_t c = 0; c < nt * 32; c++) {
                    const uint32_t b = fused::fpu::f_bits(fused::fpu::mul(fused::fpu::mul(f32(hrow)[c], rs), f32(wrow)[c]));
                    o[c] = static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
                }
                noc_async_write(nrow, hn.get_noc_addr(0, t0 * 64), nt * 64);
                noc_async_write_barrier();
            }
        }
#endif
#endif
    }
}
