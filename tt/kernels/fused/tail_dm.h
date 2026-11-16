// A layer's tail, reader and writer sides (compute in fused/tail_cmp.h), shared by the fused layer programs:
//   oproj  the mixer's output projection (6144 -> 2560: ssm_out, or the attention's wo) on 80 cores, one output
//          tile each, x streamed from a bf16 arena region with the weights; its epilogue the hyper-connection
//          combine X_s += w_s * out back into X's regions
//   R1     the router (2560 -> 512 logits + the shared expert's gate logit, bf16 weights) on 17 cores and the shared
//          expert's gate|up (2560 -> 640 | 640, pairs of tiles) on 20 cores, h = silu(g) * u
//   R2     the shared expert's down (640 -> 2560) on 80 cores, * sigmoid(gate logit), + the routed experts' sum,
//          then the combine X_s += w_s * (sh + routed)
#pragma once

#include "fused/dm.h"
#include "fused/hc_dm.h"  // the M-row helpers

namespace fused::tail {

constexpr uint32_t cb_w = 0, cb_x = 1;
constexpr uint32_t cb_cx = 40, cb_cxs = 42, cb_cw = 43, cb_cout32 = 44, cb_cout16 = 45;
constexpr uint32_t cb_rw = 46, cb_rlog = 47, cb_rh16 = 49, cb_rh32 = 50;
constexpr uint32_t cb_sgl = 51, cb_srt = 52, cb_sxs = 53, cb_sw = 54, cb_sout32 = 56, cb_sout16 = 57;
constexpr uint32_t D = 2560, KD = 80, KC = 192, KR2 = 20, N_ROUTER = 17, FF = 640, IDLE = 0xffffffff;

// X's tile n of each stream into cb xs (fp32 row-0 tiles), w_s across row 0 into cb wcb
inline void combine_inputs(uint32_t xs, uint32_t wcb, uint32_t x32, uint32_t w, uint32_t n) {
    cb_reserve_back(xs, 4);
    cb_reserve_back(wcb, 4);
    const uint32_t d = get_write_ptr(xs), ww = get_write_ptr(wcb);
    for (uint32_t s = 0; s < 4; s++) {
        row0_from_l1(x32 + (s * D + n * 32) * 4, d + s * 4096);
        fill_row0(ww + s * 4096, w + s * 4);
    }
    noc_async_read_barrier();
    cb_push_back(xs, 4);
    cb_push_back(wcb, 4);
}

// oproj, reader: the weights that fit the buffer before the barrier; returns how many pages that was
template <typename Acc>
uint32_t oproj_prefetch(Stream<Acc>& ws, const Acc& wo, uint32_t tile, uint32_t pages) {
    if (tile == IDLE) {
        return 0;
    }
    ws.read(wo, tile * KC, pages);
    return pages;
}

// oproj, reader, after the barrier: x from the bf16 region at x16 (6144), streamed with the rest of the weights
template <typename Acc>
void oproj_read(Stream<Acc>& ws, const Acc& wo, uint32_t tile, uint32_t pre, uint32_t x16, uint32_t x32,
                uint32_t w) {
    if (tile == IDLE) {
        return;
    }
    for (uint32_t k = 0; k < KC; k += ws.batch) {
        x_tiles(cb_cx, x16 + k * 64, ws.batch, ws.batch);
        if (k >= pre) {
            ws.read(wo, tile * KC + k, ws.batch);
        }
    }
    combine_inputs(cb_cxs, cb_cw, x32, w, tile);
}

// R1, reader, before the barrier that precedes it (the previous phase's weights all issued): R1's first weights, as
// many pages as their buffer holds (the router's 2 batches, gate|up's 4 in the weight buffer); returns the count
template <typename Acc>
uint32_t r1_prefetch(Stream<Acc>& ws, Stream<Acc>& rs, const Acc& wr, const Acc& wgu, uint32_t r_item) {
    if (r_item == IDLE) {
        return 0;
    }
    if (r_item < N_ROUTER) {
        rs.read(wr, r_item * KD, 2 * rs.batch);
        return 2 * rs.batch;
    }
    ws.read(wgu, (r_item - N_ROUTER) * 2 * KD, 4 * ws.batch);
    return 4 * ws.batch;
}

// R1's inputs (pre: its pages r1_prefetch() issued)
template <typename Acc>
void r1_inputs(Stream<Acc>& ws, Stream<Acc>& rs, const Acc& wr, const Acc& wgu, uint32_t r_item, uint32_t hn,
               uint32_t pre) {
    if (r_item != IDLE) {
        x_tiles(cb_x, hn, KD, KD);
        if (r_item < N_ROUTER) {
            rs.read(wr, r_item * KD + pre, KD - pre);
            rs.flush();
        } else {
            ws.read(wgu, (r_item - N_ROUTER) * 2 * KD + pre, 2 * KD - pre);
            ws.flush();
        }
    }
}

// R1, reader: R1's inputs and R2's weights (they need no activation), then R1's barrier.
// L holds the logits (17 tiles x 32), SH the shared expert's h (bf16, 640); contiguous (R1's broadcast).
template <typename Acc, typename Wait>
void r1_read(Stream<Acc>& ws, Stream<Acc>& rs, const Acc& wr, const Acc& wgu, const Acc& wsd, uint32_t r_item,
             uint32_t s_tile, uint32_t hn, uint32_t l, uint32_t sh, Wait& wait, uint32_t& epoch, uint32_t pre = 0) {
    r1_inputs(ws, rs, wr, wgu, r_item, hn, pre);
    if (s_tile != IDLE) {
        ws.read(wsd, s_tile * KR2, KR2);
        ws.flush();
    }
    wait(++epoch, l, sh + FF * 2 - l);
}

// R1, reader, without R2's weights (the card's experts run between R1 and R2: they use the weight buffer, so R2's
// weights come after them, r2_prefetch)
template <typename Acc, typename Wait>
void r1_read_only(Stream<Acc>& ws, Stream<Acc>& rs, const Acc& wr, const Acc& wgu, uint32_t r_item, uint32_t hn,
                  uint32_t l, uint32_t sh, Wait& wait, uint32_t& epoch, uint32_t pre = 0) {
    r1_inputs(ws, rs, wr, wgu, r_item, hn, pre);
    wait(++epoch, l, sh + FF * 2 - l);
}

template <typename Acc>
void r2_prefetch(Stream<Acc>& ws, const Acc& wsd, uint32_t s_tile) {
    if (s_tile != IDLE) {
        ws.read(wsd, s_tile * KR2, KR2);
        ws.flush();
    }
}

// R2, reader: its inputs but the routed sum (all in after R1's barrier: the compute's GEMV can run while the host
// computes the routed experts); h = false: all but h (flayer's SH_LATE brings it after its own barrier)
inline void r2_inputs(uint32_t s_tile, uint32_t l, uint32_t sh, uint32_t x32, uint32_t w, bool h = true) {
    if (s_tile != IDLE) {
        if (h) {
            x_tiles(cb_x, sh, KR2, KD);
        }
        cb_reserve_back(cb_sgl, 1);
        fill_row0(get_write_ptr(cb_sgl), l + 512 * 4);
        cb_push_back(cb_sgl, 1);
        combine_inputs(cb_sxs, cb_sw, x32, w, s_tile);
    }
}

// R2, reader: the routed experts' sum, then R2's barrier (X fp32 then bf16). The sum comes from the arena
// (routed_l1, fp32 [2560], the host's reply) or, when routed_l1 is 0, from the routed tensor in card memory.
// (bcast = false: the barrier broadcasts nothing, x_out() put X where it is read)
template <typename Acc, typename Wait>
void r2_routed(const Acc& routed, uint32_t routed_l1, uint32_t s_tile, uint32_t x32, Wait& wait, uint32_t& epoch,
               bool bcast = true) {
    if (s_tile != IDLE) {
        cb_reserve_back(cb_srt, 1);
        if (routed_l1) {
            row0_from_l1(routed_l1 + s_tile * 128, get_write_ptr(cb_srt));
        } else {
            row0_from_dram(routed, 0, s_tile * 128, get_write_ptr(cb_srt));
        }
        noc_async_read_barrier();
        cb_push_back(cb_srt, 1);
    }
    wait(++epoch, x32, bcast ? 4 * D * 6 : 0);  // X (fp32 then bf16)
}

// R2, reader: its inputs, then R2's barrier
template <typename Acc, typename Wait>
void r2_read(const Acc& routed, uint32_t routed_l1, uint32_t s_tile, uint32_t l, uint32_t sh, uint32_t x32, uint32_t w,
             Wait& wait, uint32_t& epoch, bool bcast = true) {
    r2_inputs(s_tile, l, sh, x32, w);
    r2_routed(routed, routed_l1, s_tile, x32, wait, epoch, bcast);
}

// R1 + R2, reader, with the routed sum from card memory
template <typename Acc, typename Wait>
void ffn_read(Stream<Acc>& ws, Stream<Acc>& rs, const Acc& wr, const Acc& wgu, const Acc& wsd, const Acc& routed,
              uint32_t r_item, uint32_t s_tile, uint32_t hn, uint32_t l, uint32_t sh, uint32_t x32, uint32_t w,
              Wait& wait, uint32_t& epoch) {
    r1_read(ws, rs, wr, wgu, wsd, r_item, s_tile, hn, l, sh, wait, epoch);
    r2_read(routed, 0, s_tile, l, sh, x32, w, wait, epoch);
}

// writer: X's tile n of the 4 streams (fp32 cb c32, bf16 cb c16) back into X's regions and the check (fp32), and
// to out16 (bf16 [4 x 2560]) when given: a program's last X goes back to the X tensors the next program reads (the
// check then names the fp32 one); out = false: only into X's regions. X's regions: the coordinator's (the barrier
// broadcasts them), or with h1 (the runtime arg index of H1's 44 cores by item 4 m + s, NOC x | y << 16) this core's
// fp32 tile (its own phases read only that) and those of stream s's 11 H1 cores, the only readers of the rest
template <typename Acc>
void x_out(const Grid& grid, uint32_t stage, uint32_t n, uint32_t c32, uint32_t c16, const Acc& chk, uint32_t x32,
           uint32_t x16, const Acc* out16 = nullptr, bool out = true, uint32_t h1 = 0) {
    stage_rows(c32, 4, 64, stage);
    stage_rows(c16, 4, 32, stage + 512);
    for (uint32_t s = 0; s < 4; s++) {
        const uint32_t o32 = x32 + (s * D + n * 32) * 4, o16 = x16 + (s * D + n * 32) * 2;
        if (h1) {
            noc_async_write(stage + s * 128, get_noc_addr(o32), 128);
            for (uint32_t m = 0; m < 11; m++) {
                const uint32_t c = get_arg_val<uint32_t>(h1 + 4 * m + s);
                noc_async_write(stage + s * 128, get_noc_addr(c & 0xffff, c >> 16, o32), 128);
                noc_async_write(stage + 512 + s * 64, get_noc_addr(c & 0xffff, c >> 16, o16), 64);
            }
        } else {
            grid.send(stage + s * 128, o32, 128);
            grid.send(stage + 512 + s * 64, o16, 64);
        }
        if (!out) {
            continue;
        }
        noc_async_write(stage + s * 128, chk.get_noc_addr(0, (s * D + n * 32) * 4), 128);
        if (out16) {
            noc_async_write(stage + 512 + s * 64, out16->get_noc_addr(0, (s * D + n * 32) * 2), 64);
        }
    }
    noc_async_write_barrier();
}

template <typename Acc>
void oproj_write(const Grid& grid, uint32_t stage, uint32_t tile, const Acc& chk, uint32_t x32, uint32_t x16,
                 uint32_t& epoch, uint32_t h1 = 0) {
    if (tile != IDLE) {
#ifdef NO_CHECKS
        x_out(grid, stage, tile, cb_cout32, cb_cout16, chk, x32, x16, static_cast<const Acc*>(nullptr), false, h1);
#else
        x_out(grid, stage, tile, cb_cout32, cb_cout16, chk, x32, x16, static_cast<const Acc*>(nullptr), true, h1);
#endif
    }
    grid.arrive(++epoch);
}

// R1, writer. Checks: L [17 x 32], H [640] fp32 (none with -DNO_CHECKS).
template <typename Acc>
void r1_write(const Grid& grid, uint32_t stage, uint32_t r_item, const Acc& chk_l, const Acc& chk_h, uint32_t l,
              uint32_t sh, uint32_t& epoch) {
    if (r_item != IDLE) {
        if (r_item < N_ROUTER) {
            stage_rows(cb_rlog, 1, 64, stage);
            grid.send(stage, l + r_item * 128, 128);
#ifndef NO_CHECKS
            noc_async_write(stage, chk_l.get_noc_addr(0, r_item * 128), 128);
#endif
            noc_async_write_barrier();
        } else {
            const uint32_t j = r_item - N_ROUTER;
            stage_rows(cb_rh16, 1, 32, stage);
            grid.send(stage, sh + j * 64, 64);
            noc_async_write_barrier();
#ifdef NO_CHECKS
            cb_wait_front(cb_rh32, 1);
            cb_pop_front(cb_rh32, 1);
#else
            stage_rows(cb_rh32, 1, 64, stage);
            noc_async_write(stage, chk_h.get_noc_addr(0, j * 128), 128);
            noc_async_write_barrier();
#endif
        }
    }
    grid.arrive(++epoch);
}

// R2, writer. Check: X after R2 [4 x 2560] fp32 (out = false: none, nor out16).
template <typename Acc>
void r2_write(const Grid& grid, uint32_t stage, uint32_t s_tile, const Acc& chk_x, uint32_t x32, uint32_t x16,
              uint32_t& epoch, const Acc* out16 = nullptr, bool out = true, uint32_t h1 = 0) {
    if (s_tile != IDLE) {
        x_out(grid, stage, s_tile, cb_sout32, cb_sout16, chk_x, x32, x16, out16, out, h1);
    }
    grid.arrive(++epoch);
}

// ---- M rows (fused/hc_dm.h's mp-row blocks; the DeltaNet program's owner layout): X fp32 tile n of the 4 streams in
// its owner's region xown (block s), and each H1 core's stream in xs32 / xs16 (block n)

// X's tile of the 4 streams into cb xs (from xown), w_s on rows 0..mp-1 into cb wcb (row m's w_s: word s of the
// 64 bytes at w + 64 m)
inline void combine_inputs_m(uint32_t xs, uint32_t wcb, uint32_t xown, uint32_t w, uint32_t mp) {
    cb_reserve_back(xs, 4);
    cb_reserve_back(wcb, 4);
    const uint32_t d = get_write_ptr(xs), ww = get_write_ptr(wcb);
    for (uint32_t s = 0; s < 4; s++) {
        rows_from_l1(xown + s * mp * 128, d + s * 4096, mp);
        fill_rows(ww + s * 4096, w + s * 4, 64, mp);
    }
    noc_async_read_barrier();
    cb_push_back(xs, 4);
    cb_push_back(wcb, 4);
}

// writer: X's tile n of the 4 streams (fp32 cb c32, bf16 cb c16) into this core's xown and the xs32 / xs16 of stream
// s's 11 H1 cores (h1: the runtime arg index of their NOC x | y << 16 by item 4 m + s), and with out (a check, or the
// X tensors the next program reads: fp32 x32o, bf16 x16o when given, a stream a page) to card memory. Stage: fp32
// blocks at stage, bf16 at stage + 2048.
template <typename Acc>
void x_out_m(uint32_t stage, uint32_t n, uint32_t c32, uint32_t c16, uint32_t xown, uint32_t xs32, uint32_t xs16,
             uint32_t h1, uint32_t mp, bool out, const Acc& x32o, const Acc* x16o) {
    const uint32_t s16 = stage + 2048, b32 = mp * 128, b16 = mp * 64;
    stage_rows_m(c32, 4, 64, stage, mp);
    stage_rows_m(c16, 4, 32, s16, mp);
    for (uint32_t s = 0; s < 4; s++) {
        noc_async_write(stage + s * b32, get_noc_addr(xown + s * b32), b32);
        for (uint32_t m = 0; m < 11; m++) {
            const uint32_t c = get_arg_val<uint32_t>(h1 + 4 * m + s);
            noc_async_write(stage + s * b32, get_noc_addr(c & 0xffff, c >> 16, xs32 + n * b32), b32);
            noc_async_write(s16 + s * b16, get_noc_addr(c & 0xffff, c >> 16, xs16 + n * b16), b16);
        }
        if (out) {
            noc_async_write(stage + s * b32, x32o.get_noc_addr(s, n * b32), b32);
            if (x16o) {
                noc_async_write(s16 + s * b16, x16o->get_noc_addr(s, n * b16), b16);
            }
        }
    }
    noc_async_write_barrier();
}

template <typename Acc>
void ffn_write(const Grid& grid, uint32_t stage, uint32_t r_item, uint32_t s_tile, const Acc& chk_l,
               const Acc& chk_h, const Acc& chk_x, uint32_t l, uint32_t sh, uint32_t x32, uint32_t x16,
               uint32_t& epoch, const Acc* out16 = nullptr) {
    r1_write(grid, stage, r_item, chk_l, chk_h, l, sh, epoch);
    r2_write(grid, stage, s_tile, chk_x, x32, x16, epoch, out16);
}

}  // namespace fused::tail
