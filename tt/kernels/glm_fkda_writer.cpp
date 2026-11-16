// GLM's fused KDA mixer, writer (the phases in glm_fkda_reader.cpp). Row 0 of a result tile is two
// 64-byte pieces (fp32) or two 32-byte pieces (bf16), faces 0 and 1.
//   A  per unit: for conv tiles the 3 new conv-state rows (cb 18) to the conv state; row 0 of the unit's results
//      (cb 17) to region A of the coordinator's arena and to A's check tensor; arrive at barrier 1
//   B  the head's new state (cb 19, 16 tiles, 2 at a time) to card memory; its 128 outputs (cb 21 bf16) to region B,
//      (cb 22 fp32) to B's check tensor; arrive at barrier 2
//   C  per item (cb 17: a partial row of output tile n = i / S, part i % S) to region P at (part, n); arrive at
//      barrier 3 (the reader sums the partials into out)
// Runtime args: conv state out, A check, grid.h's block, region A, unit count, per unit (first tile, count, type),
// head, state out, B check, region B, C's first item and count, region P. Compile args: phases, S, then a DRAM
// accessor.
//
// M rows (-DM_MAX = 2..4; the reader's comment): M, n_keep, parity and the layout's rows mp from the step
// tensor (words 1..4); vectors as fused/hc_dm.h's mp-row blocks (an fp32 tile's block 128 mp bytes).
//   A  per conv tile its history (3 rows of the parity's buffer from n_keep - 1) and taps (4 rows) into region AC
//      (7 rows of 128 bytes a tile) before any y; then per conv tile the compute's y (cb 16) -> the window (cb 3, 8
//      tiles: y, W0..W2, the taps down rows 0..mp-1; fused::conv_window_m) and the extended history (s1, s2,
//      y_0..y_{M-1}) to the other parity's buffer (rows (1 - parity)(2 + M_MAX) ..). Region A by owner: the head's q, k,
//      v tiles to its B core's region A (blocks 0..3 q, 4..7 k, 8..11 v; core h by the NOC table), f_lo, g_lo, beta
//      to every core's (blocks 12..21, multicast); A's check in blocks [778].
// Runtime args at M rows, after region P: taps, the step tensor, the 64 B cores' NOC x | y << 16.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif
#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_out = 17, cb_snew = 18, cb_sxo = 19, cb_hout = 21, cb_hchk = 22, cb_stage = 31;
constexpr uint32_t CONV = 24576, NT_A = 778, NV = 64, HK = 128, NT_C = 128;
constexpr uint32_t T_K = 2, IDLE = 0xffffffff;
constexpr uint32_t cb_raw = 16, cb_win = 3;

void kernel_main() {
    constexpr uint32_t phases = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr auto d_args = TensorAccessorArgs<2>();
    const auto state = TensorAccessor(d_args, get_arg_val<uint32_t>(0), CONV * 4);
    const auto check_a = TensorAccessor(d_args, get_arg_val<uint32_t>(1), NT_A * 128);
    const fused::Grid grid = fused::Grid::from_args(2);
    uint32_t i = 2 + fused::Grid::args(grid.nrects);
    const uint32_t region_a = get_arg_val<uint32_t>(i++);
    const uint32_t n_units = get_arg_val<uint32_t>(i++);
    const uint32_t ua = i;
    i += 3 * n_units;
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const auto S_out = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 4096);
    const auto check_b = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), NV * HK * 4);
    const uint32_t region_b = get_arg_val<uint32_t>(i++);
    const uint32_t c_first = get_arg_val<uint32_t>(i++), c_count = get_arg_val<uint32_t>(i++);
    const uint32_t region_c = get_arg_val<uint32_t>(i++);
#if M_MAX > 1
    const auto taps = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), CONV * 4);
    const uint32_t step = get_arg_val<uint32_t>(i++), noc_tab = i;
    i += NV;
    constexpr uint32_t PEND = 12 * M_MAX * 128;
    const auto pend = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), PEND);
    const uint32_t c_dest = i;   // C's items: the owner's NOC x | y << 16 and the partial's slot there
    i += 2 * c_count;
#endif
#ifdef GRID_PROFILE
    const uint32_t gp_addr = get_arg_val<uint32_t>(i++), gp_slot = get_arg_val<uint32_t>(i++);
#endif
    const uint32_t arena = get_write_ptr(30), stage = get_write_ptr(cb_stage);

    // ---- A
#if M_MAX > 1
    uint32_t mp = 1, parity = 0;
    {
        const uint32_t sw = stage;  // the step words: 1 M, 2 n_keep, 3 parity, 4 mp (kept in registers)
        noc_async_read(TensorAccessor(d_args, step, 64).get_noc_addr(0), sw, 32);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sw);
        const uint32_t M = w[1], n_keep = w[2];
        parity = w[3], mp = w[4];
        const uint32_t blk = mp * 128, ac = arena + region_a + 22 * M_MAX * 128;   // region AC after region A
        const uint32_t hist_in = parity * (2 + M_MAX) + n_keep - 1, hist_out = (1 - parity) * (2 + M_MAX);
        uint32_t slot = 0;
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            if (get_arg_val<uint32_t>(ua + 3 * u + 2) > T_K) continue;
            for (uint32_t n = first; n < first + count; n++, slot++) {
                for (uint32_t j = 0; j < 3; j++) noc_async_read(state.get_noc_addr(hist_in + j, n * 128), ac + (slot * 7 + j) * 128, 128);
                for (uint32_t j = 0; j < 4; j++) noc_async_read(taps.get_noc_addr(j, n * 128), ac + (slot * 7 + 3 + j) * 128, 128);
            }
        }
        noc_async_read_barrier();
        slot = 0;
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            for (uint32_t n = first; type <= T_K && n < first + count; n++, slot++) {
                const uint32_t c = ac + slot * 7 * 128;
                cb_wait_front(cb_raw, 1);
                const uint32_t y = get_read_ptr(cb_raw);
                cb_reserve_back(cb_win, 8);
                const uint32_t win = get_write_ptr(cb_win);
                fused::conv_window_m(y, c, win, mp);
                for (uint32_t j = 0; j < 4; j++) fused::row_down(c + (3 + j) * 128, win + (4 + j) * 4096, mp);
                noc_async_read_barrier();
                cb_push_back(cb_win, 8);
                noc_async_write(c + 128, state.get_noc_addr(hist_out, n * 128), 128);       // s1
                noc_async_write(c + 256, state.get_noc_addr(hist_out + 1, n * 128), 128);   // s2
                for (uint32_t m = 0; m < M; m++) {                                         // y_m
                    noc_async_write(y + m * 64, state.get_noc_addr(hist_out + 2 + m, n * 128), 64);
                    noc_async_write(y + 1024 + m * 64, state.get_noc_addr(hist_out + 2 + m, n * 128 + 64), 64);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_raw, 1);
            }
            fused::stage_rows_m(cb_out, count, 64, stage, mp);
            for (uint32_t j = 0; j < count; j++) {
                const uint32_t n = first + j, src = stage + j * blk;
                if (n < CONV / 32) {   // q | k | v of head (n % 256) / 4: blocks 4 (n / 256) + n % 4 on its B core
                    const uint32_t xy = get_arg_val<uint32_t>(noc_tab + (n % 256) / 4);
                    noc_async_write(src, get_noc_addr(xy & 0xffff, xy >> 16,
                                                      arena + region_a + (4 * (n / 256) + n % 4) * blk), blk);
                } else {               // f_lo, g_lo, beta: blocks 12..21 everywhere
                    grid.mcast(src, arena + region_a + (12 + n - CONV / 32) * blk, blk);
                }
            }
            noc_async_write(stage, check_a.get_noc_addr(0, first * blk), count * blk);
            noc_async_write_barrier();  // the staging run is reused next
        }
        grid.arrive(1);
    }
    // ---- B at M rows: the committed state S_a (cb 19) to the other parity's state; this pass's rows for the next
    // pass's commit (Ki cb 45, U cb 53, G cb 39: M_MAX-row blocks 0..3, 4..7, 8..11) to the other parity's pending
    // page; the outputs (cb 21 bf16 to region B, cb 22 fp32 to B's check) as mp-row blocks
    if ((phases & 2) && head != IDLE) {
        const uint32_t pg = (1 - parity) * NV + head;
        for (uint32_t t = 0; t < 16; t += 2) {
            cb_wait_front(cb_sxo, 2);
            const uint32_t src = get_read_ptr(cb_sxo);
            noc_async_write(src, S_out.get_noc_addr(pg * 16 + t), 4096);
            noc_async_write(src + 4096, S_out.get_noc_addr(pg * 16 + t + 1), 4096);
            noc_async_writes_flushed();
            cb_pop_front(cb_sxo, 2);
        }
        fused::stage_rows_m(39, 4, 64, stage + 8 * M_MAX * 128, M_MAX);   // G
        fused::stage_rows_m(45, 4, 64, stage, M_MAX);                     // Ki
        fused::stage_rows_m(53, 4, 64, stage + 4 * M_MAX * 128, M_MAX);   // U
        noc_async_write(stage, pend.get_noc_addr(pg), PEND);
        noc_async_write_barrier();
        fused::stage_rows_m(cb_hout, 4, 32, stage, mp);
        grid.send(stage, arena + region_b + head * mp * 256, mp * 256);
        noc_async_write_barrier();
        fused::stage_rows_m(cb_hchk, 4, 64, stage, mp);
        noc_async_write(stage, check_b.get_noc_addr(0, head * mp * 512), mp * 512);
        noc_async_write_barrier();
    }
    if (phases & 2) grid.arrive(2);
#else
    if (phases & 1) {
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            if (type <= T_K) {
                for (uint32_t n = first; n < first + count; n++) {
                    cb_wait_front(cb_snew, 3);
                    const uint32_t t = get_read_ptr(cb_snew);
                    for (uint32_t j = 0; j < 3; j++) {
                        noc_async_write(t + j * 4096, state.get_noc_addr(j, n * 128), 64);
                        noc_async_write(t + j * 4096 + 1024, state.get_noc_addr(j, n * 128 + 64), 64);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_snew, 3);
                }
            }
            fused::stage_rows(cb_out, count, 64, stage);
            grid.send(stage, arena + region_a + first * 128, count * 128);
            noc_async_write(stage, check_a.get_noc_addr(0, first * 128), count * 128);
            noc_async_write_barrier();  // the staging run is reused next
        }
        grid.arrive(1);
    }
#endif

    // ---- B
#if M_MAX == 1
    if (phases & 2) {
        if (head != IDLE) {
#ifdef FK_DEBUG
            // the decay's f (cb 35) and its pre-sigmoid argument (cb 36), row 0 of 4 tiles each, to the debug tensor's row head
            const auto dbg = TensorAccessor(d_args, FK_DEBUG, 256 * 4);
            fused::stage_rows(35, 4, 64, stage);
            fused::stage_rows(36, 4, 64, stage + 512);
            noc_async_write(stage, dbg.get_noc_addr(head), 1024);
            noc_async_write_barrier();
#endif
            for (uint32_t t = 0; t < 16; t += 2) {
                cb_wait_front(cb_sxo, 2);
                const uint32_t src = get_read_ptr(cb_sxo);
                noc_async_write(src, S_out.get_noc_addr(head * 16 + t), 4096);
                noc_async_write(src + 4096, S_out.get_noc_addr(head * 16 + t + 1), 4096);
                noc_async_writes_flushed();
                cb_pop_front(cb_sxo, 2);
            }
            fused::stage_rows(cb_hout, 4, 32, stage);
            grid.send(stage, arena + region_b + head * HK * 2, HK * 2);
            noc_async_write_barrier();
            fused::stage_rows(cb_hchk, 4, 64, stage);
            noc_async_write(stage, check_b.get_noc_addr(0, head * HK * 4), HK * 4);
            noc_async_write_barrier();
        }
        grid.arrive(2);
    }

#endif

    // ---- C
#if M_MAX > 1
    if (phases & 4) {   // the partial block of item it to its output tile's owner (ksplit_sum_m's layout)
        const uint32_t blk = mp * 128;
        for (uint32_t it = 0; it < c_count; it++) {
            fused::stage_rows_m(cb_out, 1, 64, stage, mp);
            const uint32_t xy = get_arg_val<uint32_t>(c_dest + 2 * it), slot = get_arg_val<uint32_t>(c_dest + 2 * it + 1);
            noc_async_write(stage, get_noc_addr(xy & 0xffff, xy >> 16, arena + region_c + slot * blk), blk);
            noc_async_write_barrier();
        }
        grid.arrive(3);
    }
#else
    if (phases & 4) {
        for (uint32_t it = c_first; it < c_first + c_count; it++) {
            fused::stage_rows(cb_out, 1, 64, stage);
            grid.send(stage, arena + region_c + ((it % S) * NT_C + it / S) * 128, 128);
            noc_async_write_barrier();
        }
        grid.arrive(3);
    }
#endif
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
