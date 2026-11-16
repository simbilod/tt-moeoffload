// Fused DeltaNet layer writer (G3 step 4; the phases are listed in flayer_reader.cpp). Each phase's results go to
// the coordinator's arena (its barrier broadcasts them) and to a check tensor; then this core arrives.
// Runtime args, in order: grid.h's block; arena offsets X32, X16, P1, RS, HNA, LOA, WA, HNF, LOF, WF, A, B, L, SH, RT,
// AC, TAB, PROF, RQ; the check tensors P1a, HNa, Wa, A, B, X1 (after C), P1f, HNf, Wf, L, H (the shared expert's h),
// X2 (after R2); conv state out, S out; the H1 item, the H2 tile, A's unit count and units, B's head, C's tile, the R1 item, R2's tile; the
// layer table; X's bf16 out (0 none: with it and X2 = the X fp32 input, X goes back in place for the next program);
// the card's partial rows (CARD_EXPERTS), this core's index, the core count; B's 48 cores by head and H1's 44 by item
// (NOC x | y << 16).
// A's outputs go straight to the B cores that read them, C's and R2's X to this core and H1's (no gather, no broadcast).
// With -DNO_CHECKS (the token loop, which reads none of them) no check tensors are written, and X goes back to card
// memory after the last pass only: a phase's arrival then waits for no card memory write it does not need.
// With M_MAX > 1 (M rows, flayer_reader.cpp) the step tensor after H1's cores: every output in mp-row blocks, the
// conv's window built here from A's y and the history in region AC (cb 22 to the compute), the extended conv history
// (s1, s2, y_0..y_{M-1}) to rows 0..M+1 of the other parity's buffer; B's committed state (cb 33) and its pending
// pages (k, U from cb 63, the decays) to the other parity's; with the host link its barrier after R1's, as at M = 1.
// Compile args: a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"
#include "fused/tail_dm.h"
#ifdef CARD_EXPERTS
#include "fused/dexp_dm.h"
#endif

constexpr uint32_t cb_hout = 7, cb_hrout = 8, cb_hhn = 14, cb_hchk = 15, cb_aout = 21, cb_asnew = 22;
constexpr uint32_t cb_bsxo = 33, cb_bhout = 38, cb_bhchk = 39, cb_cout32 = 44, cb_cout16 = 45;
constexpr uint32_t cb_rlog = 47, cb_rh16 = 49, cb_rh32 = 50, cb_sout32 = 56, cb_sout16 = 57;
constexpr uint32_t cb_arena = 61, cb_stage = 62;
constexpr uint32_t D = 2560, CONV = 10240, HK = 128, LO = 352, NT_A = 516, N_ROUTER = 17, NV = 48;
constexpr uint32_t T_K = 2, T_A = 4, IDLE = 0xffffffff;
// With the host link, without the card's experts (in the layers the layer loop's sh_layer picks): R1 is the
// router's alone. The shared expert's gate|up (20 cores, ~25 us against the router's ~16) runs after the page goes out,
// while the host computes, and its h has its own barrier (SH) before R2 reads it.
#if defined(EXCHANGE) && !defined(CARD_EXPERTS)
#define SH_LATE 1
#endif
#ifndef M_MAX
#define M_MAX 1
#endif
constexpr bool MR = M_MAX > 1;
constexpr uint32_t cb_araw = 17, cb_awin = 22, cb_bsw = 63;  // M rows: A's y, the conv's window, B's U rows
constexpr uint32_t S_PEND = 2 * NV * 16;  // M rows: the S tensor's first pending page (flayer_reader.cpp)

void kernel_main() {
    constexpr uint32_t layers = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const fused::Grid grid = fused::Grid::from_args(0);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_hna = arena + arg(), r_loa = arena + arg(), r_wa = arena + arg(), r_hnf = arena + arg();
    const uint32_t r_lof = arena + arg(), r_wf = arena + arg();
    const uint32_t r_a = arena + arg(), r_b = arena + arg(), r_l = arena + arg(), r_sh = arena + arg();
    arg();  // RT (the reader's)
    const uint32_t r_ac = arena + arg();  // region AC (the reader's: A's per-tile constants)
    const uint32_t r_tab = arena + arg();  // the table rows' scratch (the writer's row at +128)
    arg();  // the profile stamps (the reader's)
    const uint32_t r_rq = arena + arg();  // the card's routes
    const auto chk_p1a = acc(4 * LO * 4), chk_hna = acc(D * 4), chk_wa = acc(64), chk_a = acc(NT_A * 128);
    constexpr uint32_t X_PAGE = MR ? M_MAX * D * 4 : 4 * D * 4;  // M rows: X tensors a stream a page
    const auto chk_b = acc(48 * HK * 4), chk_x1 = acc(X_PAGE), chk_p1f = acc(4 * LO * 4), chk_hnf = acc(D * 4);
    const auto chk_wf = acc(64), chk_l = acc(17 * 128), chk_h = acc(640 * 4), chk_x2 = acc(X_PAGE);
    const uint32_t cstate0 = arg(), S0 = arg();  // the conv state and DeltaNet state written back
    const uint32_t h_item = arg(), h_tile = arg();
    const uint32_t n_units = arg(), ua = i;
    i += 3 * n_units;
    const uint32_t head = arg(), c_tile = arg(), r_item = arg(), s_tile = arg();
    [[maybe_unused]] const bool sh_late = r_item != IDLE && r_item >= N_ROUTER;  // SH_LATE: a gate|up core
    const uint32_t table = arg();  // the reader's layer table: its rows' L_CSTATE (7) and L_S (10) entries
    const auto tab = TensorAccessor(d_args, table, 128);
    const uint32_t x16_out = arg();  // X's bf16 [4 x 2560] after R2 to card memory too (0: none), for the next program
    const auto x16o = TensorAccessor(d_args, x16_out, MR ? M_MAX * D * 2 : 4 * D * 2);
    const uint32_t part = arg(), me = arg(), cores = arg();  // the card's partial rows; this core's index, the count
    const uint32_t b_cores = i;  // B's cores by head: NOC x | y << 16
    i += NV;
    const uint32_t h1_cores = i;  // H1's cores by item, the same way: C's and R2's X goes to them (fused/tail_dm.h)
    i += 4 * 11;
    const uint32_t step_addr = MR ? arg() : 0;  // M rows: words 1..4 M, n_keep, parity, Mp
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    // A's output tile n (row 0 at L1 src) into region A of the B cores that read it, instead of a gather and a 66 KB
    // broadcast: q and k tiles (key head kh) to value heads kh, kh + 16, kh + 32; v and z tiles to their head; the
    // decay (tiles 512, 513) and beta (514, 515) to each head as the 16 bytes holding its word
    auto a_to_b = [&](uint32_t n, uint32_t src) {
        auto to = [&](uint32_t h, uint32_t off, uint32_t bytes) {
            const uint32_t c = get_arg_val<uint32_t>(b_cores + h);
            noc_async_write(src + off, get_noc_addr(c & 0xffff, c >> 16, r_a + n * 128 + off), bytes);
        };
        if (n < 128) {
            for (uint32_t h = (n % 64) / 4; h < NV; h += 16) {
                to(h, 0, 128);
            }
        } else if (n < 512) {
            to(((n - 128) % 192) / 4, 0, 128);
        } else {
            for (uint32_t j = 0, h = 32 * (n % 2); j < 32 && h < NV; j++, h++) {
                to(h, (j * 4) & ~15u, 16);
            }
        }
    };
    // M rows: the step's words; mp the layout's rows; the regions placed after a variable-size one
    uint32_t M = 1, parity = 0, mp = 1;
    if constexpr (MR) {
        const uint32_t st = r_tab + 320;
        noc_async_read(TensorAccessor(d_args, step_addr, 64).get_noc_addr(0), st, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
        volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
        M = sw[1], parity = sw[3], mp = sw[4];
    }
    const uint32_t r_xown = r_rs;  // M rows: the RS slot holds xown
    const uint32_t rs_m = r_p1 + mp * 4 * LO * 4, r_sh_m = r_l + mp * N_ROUTER * 128;
    // M rows: A's output tile n (its mp-row block at L1 src) into region A of the B cores that read it: q and k
    // tiles (key head kh) to value heads kh, kh + 16, kh + 32, v and z to their head, at block 4 kappa + t of
    // (q, k, v, z); the decay's and beta's tiles (512, 513; 514, 515) whole to the 32 heads of each
    auto a_to_b_m = [&](uint32_t n, uint32_t src) {
        const uint32_t blk = mp * 128;
        auto to = [&](uint32_t h, uint32_t b) {
            const uint32_t c = get_arg_val<uint32_t>(b_cores + h);
            noc_async_write(src, get_noc_addr(c & 0xffff, c >> 16, r_a + b * blk), blk);
        };
        if (n < 128) {
            for (uint32_t h = (n % 64) / 4; h < NV; h += 16) {
                to(h, (n < 64 ? 0 : 4) + n % 4);
            }
        } else if (n < 320) {
            to((n - 128) / 4, 8 + n % 4);
        } else if (n < 512) {
            to((n - 320) / 4, 12 + n % 4);
        } else {
            for (uint32_t h = 32 * (n % 2); h < 32 * (n % 2) + 32 && h < NV; h++) {
                to(h, n < 514 ? 16 : 17);
            }
        }
    };
    uint32_t epoch = 1;
    if constexpr (!MR) {
        grid.arrive(epoch);  // (M rows: the reader arrives, after X's pieces are sent)
    }

    for (uint32_t layer = 0; layer < layers; layer++) {
        // SH_LATE's layers (the SH barrier and gate|up after the page): a pass of 2+ rows, or a program's last layer.
        // A one-row pass's other exchanges also prefetch the next layer's A (45 MB in the host's ~100 us: bound by card
        // memory), where gate|up's 3.5 MB cost what R1 saves and h's own barrier queues behind the prefetch's reads
        // (R2 +10 us), so they keep R1 as it was (h with the logits).
        [[maybe_unused]] const bool sh_layer = layer + 1 == layers || mp > 1;
        [[maybe_unused]] const bool late = sh_late && sh_layer;
        uint32_t cs = cstate0, ss = S0;
        if (table) {
            const uint32_t row = r_tab + 128;
            noc_async_read(tab.get_noc_addr(layer), row, 128);
            noc_async_read_barrier();
            invalidate_l1_cache();
            cs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row)[7];
            ss = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row)[10];
        }
        const auto cstate = TensorAccessor(d_args, cs, CONV * 4), S_out = TensorAccessor(d_args, ss, 4096);
        if constexpr (MR) {
            const fused::hc::RegionsM hreg{r_x32, r_x16, r_xown, r_p1, rs_m, mp};
            fused::hc::write_m(grid, stage, chk_p1a, chk_hna, chk_wa, h_item, h_tile, hreg, r_hna, r_loa, epoch);

            // ---- A: per conv tile the window (y and the 3 shifted tiles: row r of tile j = input j + r of the
            // history's 3 rows (region AC) followed by y's rows) to the compute, and the extended history out
            const uint32_t hist = (1 - parity) * (2 + M_MAX);  // the other parity's buffer, row 0
            uint32_t slot = 0;  // region AC's slots, in the reader's order (the v, q, k and alpha tiles)
            for (uint32_t u = 0; u < n_units; u++) {
                const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
                const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
                const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
                if (type == T_A) {
                    slot += count;
                }
                for (uint32_t n = first; type <= T_K && n < first + count; n++) {
                    const uint32_t ac = r_ac + slot++ * 7 * 128;
                    cb_wait_front(cb_araw, 1);
                    const uint32_t y = get_read_ptr(cb_araw);
                    cb_reserve_back(cb_awin, 4);
                    const uint32_t w = get_write_ptr(cb_awin);
                    fused::conv_window_m(y, ac, w, mp);  // y, then the 3 shifted tiles (fused/hc_dm.h)
                    noc_async_read_barrier();
                    cb_push_back(cb_awin, 4);
                    noc_async_write(ac + 128, cstate.get_noc_addr(hist, n * 128), 128);       // s1
                    noc_async_write(ac + 256, cstate.get_noc_addr(hist + 1, n * 128), 128);   // s2
                    for (uint32_t m = 0; m < M; m++) {                                        // y_m
                        noc_async_write(y + m * 64, cstate.get_noc_addr(hist + 2 + m, n * 128), 64);
                        noc_async_write(y + 1024 + m * 64, cstate.get_noc_addr(hist + 2 + m, n * 128 + 64), 64);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_araw, 1);
                }
                fused::stage_rows_m(cb_aout, count, 64, stage, mp);
                for (uint32_t j = 0; j < count; j++) {
                    a_to_b_m(first + j, stage + j * mp * 128);
                }
#ifndef NO_CHECKS
                noc_async_write(stage, chk_a.get_noc_addr(0, first * mp * 128), count * mp * 128);
#endif
                noc_async_write_barrier();
            }
            grid.arrive(++epoch);

            // ---- B: the committed state (cb 33) to the other parity's state; this pass's rows for the next pass's
            // commit to the other parity's pending pages (k from region A, U from cb 63, the decays); the rows'
            // outputs into the head's 4 blocks in the stage (rows M..mp-1 copies of row M - 1), then to region B
            if (head != IDLE) {
                const uint32_t b16 = mp * 64, a_blk = mp * 128;
                const uint32_t s0 = ((1 - parity) * NV + head) * 16, pg = S_PEND + ((1 - parity) * NV + head) * 2;
                for (uint32_t j = 0; j < 4; j++) {
                    cb_wait_front(cb_bsxo, 4);
                    const uint32_t t = get_read_ptr(cb_bsxo);
                    for (uint32_t c = 0; c < 4; c++) {
                        noc_async_write(t + c * 4096, S_out.get_noc_addr(s0 + 4 * j + c), 4096);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_bsxo, 4);
                }
                // (region A is complete once U is in: the commit runs before A's barrier, U after it)
                cb_wait_front(cb_bsw, 4);
                for (uint32_t t = 0; t < 4; t++) {
                    for (uint32_t f = 0; f < 2; f++) {
                        noc_async_write(r_a + (4 + t) * a_blk + f * mp * 64,
                                        S_out.get_noc_addr(pg, t * 512 + f * 256), mp * 64);
                    }
                }
                {
                    invalidate_l1_cache();
                    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
                    const uint32_t j = head % 32, dec = r_a + 16 * a_blk + (j / 16) * mp * 64 + (j % 16) * 4;
                    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 2048);
                    for (uint32_t m = 0; m < 16; m++) {
                        w[m] = m < M ? *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dec + m * 64) : 0;
                    }
                    asm volatile("fence rw, rw" ::: "memory");
                    noc_async_write(stage + 2048, S_out.get_noc_addr(pg + 1), 64);
                }
                const uint32_t u = get_read_ptr(cb_bsw);
                for (uint32_t t = 0; t < 4; t++) {
                    for (uint32_t f = 0; f < 2; f++) {
                        noc_async_write(u + t * 4096 + f * 1024, S_out.get_noc_addr(pg, 2048 + t * 512 + f * 256), 256);
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_bsw, 4);
                cb_wait_front(cb_bhout, 4);
                const uint32_t t = get_read_ptr(cb_bhout);
                for (uint32_t c = 0; c < 4; c++) {
                    fused::l1_copy(t + c * 2048, stage + c * b16, mp * 32);
                    fused::l1_copy(t + c * 2048 + 512, stage + c * b16 + mp * 32, mp * 32);
                }
                noc_async_read_barrier();
                cb_pop_front(cb_bhout, 4);
                cb_wait_front(cb_bhchk, 4);
#ifndef NO_CHECKS
                const uint32_t k = get_read_ptr(cb_bhchk);
                for (uint32_t c = 0; c < 4; c++) {
                    const uint32_t at = (head * 4 + c) * mp * 128;
                    noc_async_write(k + c * 4096, chk_b.get_noc_addr(0, at), mp * 64);
                    noc_async_write(k + c * 4096 + 1024, chk_b.get_noc_addr(0, at + mp * 64), mp * 64);
                }
                noc_async_writes_flushed();
#endif
                cb_pop_front(cb_bhchk, 4);
                for (uint32_t m = M; m < mp; m++) {
                    for (uint32_t c = 0; c < 4; c++) {
                        fused::l1_copy(stage + c * b16 + (M - 1) * 32, stage + c * b16 + m * 32, 32);
                        fused::l1_copy(stage + c * b16 + mp * 32 + (M - 1) * 32, stage + c * b16 + mp * 32 + m * 32,
                                       32);
                    }
                }
                noc_async_read_barrier();
                grid.send(stage, r_b + head * 4 * b16, 4 * b16);
                noc_async_write_barrier();
            }
            grid.arrive(++epoch);

            // ---- C (+ the combine): X's tile to its owner and its H1 cores (fused/tail_dm.h x_out_m)
#ifdef NO_CHECKS
            constexpr bool chk = false;
#else
            constexpr bool chk = true;
#endif
            if (c_tile != IDLE) {
                fused::tail::x_out_m(stage, c_tile, cb_cout32, cb_cout16, r_xown, r_x32, r_x16, h1_cores, mp, chk,
                                     chk_x1, static_cast<const decltype(chk_x1)*>(nullptr));
            }
            grid.arrive(++epoch);

            fused::hc::write_m(grid, stage, chk_p1f, chk_hnf, chk_wf, h_item, h_tile, hreg, r_hnf, r_lof, epoch);

            // ---- R1 (logits and the shared expert's h), R2 (+ the combine)
#ifdef SH_LATE
            if (late) {
                grid.arrive(++epoch);  // R1: the router's alone; this core's h is SH's
            }
#endif
            if (r_item != IDLE && r_item < N_ROUTER) {
                fused::stage_rows_m(cb_rlog, 1, 64, stage, mp);
                grid.send(stage, r_l + r_item * mp * 128, mp * 128);
#ifndef NO_CHECKS
                noc_async_write(stage, chk_l.get_noc_addr(0, r_item * mp * 128), mp * 128);
#endif
                noc_async_write_barrier();
            } else if (r_item != IDLE) {
                const uint32_t j = r_item - N_ROUTER;
                fused::stage_rows_m(cb_rh16, 1, 32, stage, mp);
                grid.send(stage, r_sh_m + j * mp * 64, mp * 64);
                noc_async_write_barrier();
#ifdef NO_CHECKS
                cb_wait_front(cb_rh32, 1);
                cb_pop_front(cb_rh32, 1);
#else
                fused::stage_rows_m(cb_rh32, 1, 64, stage, mp);
                noc_async_write(stage, chk_h.get_noc_addr(0, j * mp * 128), mp * 128);
                noc_async_write_barrier();
#endif
            }
            grid.arrive(++epoch);
#ifdef SH_LATE
            if (sh_layer && !late) {
                grid.arrive(++epoch);  // SH
            }
#endif
#if defined(CARD_EXPERTS) && M_MAX > 1  // (the M-row card functions exist at M rows only)
            grid.arrive(++epoch);  // X1: the card's routes
            {
                const uint32_t n_card = fused::dexp::routes_seen(epoch, r_rq);
                fused::dexp::e1_write_m(n_card, r_a, mp, me, cores);
                grid.arrive(++epoch);  // X2: h in this core's region A
                fused::dexp::e2_write_m(d_args, stage, n_card, part, mp, me, cores);
                grid.arrive(++epoch);  // X3: the partial rows written
            }
            grid.arrive(++epoch);  // X4: the rows' CPU sums
#elif defined(EXCHANGE)
            grid.arrive(++epoch);  // the host link's barrier
#endif
            if (s_tile != IDLE) {
                fused::tail::x_out_m(stage, s_tile, cb_sout32, cb_sout16, r_xown, r_x32, r_x16, h1_cores, mp,
                                     chk || layer + 1 == layers, chk_x2, x16_out ? &x16o : nullptr);
            }
            grid.arrive(++epoch);
            continue;
        }
        const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
        fused::hc::write(grid, stage, chk_p1a, chk_hna, chk_wa, h_item, h_tile, hreg, r_hna, r_loa, r_wa, epoch);

        // ---- A
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            if (type <= T_K) {
                for (uint32_t n = first; n < first + count; n++) {
                    cb_wait_front(cb_asnew, 3);
                    const uint32_t t = get_read_ptr(cb_asnew);
                    for (uint32_t j = 0; j < 3; j++) {
                        noc_async_write(t + j * 4096, cstate.get_noc_addr(j, n * 128), 64);
                        noc_async_write(t + j * 4096 + 1024, cstate.get_noc_addr(j, n * 128 + 64), 64);
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_asnew, 3);
                }
            }
            fused::stage_rows(cb_aout, count, 64, stage);
            for (uint32_t j = 0; j < count; j++) {
                a_to_b(first + j, stage + j * 128);
            }
#ifndef NO_CHECKS
            noc_async_write(stage, chk_a.get_noc_addr(0, first * 128), count * 128);
#endif
            noc_async_write_barrier();  // the staging run is reused next
        }
        grid.arrive(++epoch);  // A's barrier broadcasts nothing

        // ---- B
        if (head != IDLE) {
            for (uint32_t j = 0; j < 4; j++) {
                cb_wait_front(cb_bsxo, 4);
                const uint32_t t = get_read_ptr(cb_bsxo);
                for (uint32_t c = 0; c < 4; c++) {
                    noc_async_write(t + c * 4096, S_out.get_noc_addr(head * 16 + 4 * j + c), 4096);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_bsxo, 4);
            }
            fused::stage_rows(cb_bhout, 4, 32, stage);
            grid.send(stage, r_b + head * HK * 2, HK * 2);
            noc_async_write_barrier();
#ifdef NO_CHECKS
            cb_wait_front(cb_bhchk, 4);
            cb_pop_front(cb_bhchk, 4);
#else
            fused::stage_rows(cb_bhchk, 4, 64, stage);
            noc_async_write(stage, chk_b.get_noc_addr(0, head * HK * 4), HK * 4);
            noc_async_write_barrier();
#endif
        }
        grid.arrive(++epoch);

        // ---- C (+ the combine)
        fused::tail::oproj_write(grid, stage, c_tile, chk_x1, r_x32, r_x16, epoch, h1_cores);

        fused::hc::write(grid, stage, chk_p1f, chk_hnf, chk_wf, h_item, h_tile, hreg, r_hnf, r_lof, r_wf, epoch);

        // ---- R1, R2 (+ the combine)
#ifdef SH_LATE
        if (late) {
            grid.arrive(++epoch);  // R1: the router's alone; this core's h is SH's
        }
        fused::tail::r1_write(grid, stage, r_item, chk_l, chk_h, r_l, r_sh, epoch);
        if (sh_layer && !late) {
            grid.arrive(++epoch);  // SH
        }
#else
        fused::tail::r1_write(grid, stage, r_item, chk_l, chk_h, r_l, r_sh, epoch);
#endif
#ifdef CARD_EXPERTS
        grid.arrive(++epoch);  // X1: the card's routes
        const uint32_t n_card = fused::dexp::routes_seen(epoch, r_rq);
        fused::dexp::e1_write(grid, stage, n_card, r_a, me, cores);
        grid.arrive(++epoch);  // X2: h
        fused::dexp::e2_write(d_args, stage, n_card, part, me, cores);
        grid.arrive(++epoch);  // X3: the partial rows written
        grid.arrive(++epoch);  // X4: the host link's CPU sum
#elif defined(EXCHANGE)
        grid.arrive(++epoch);  // the host link's barrier
#endif
#ifdef NO_CHECKS
        const bool out = layer + 1 == layers;  // X back to card memory once, for the next program
#else
        const bool out = true;
#endif
        fused::tail::r2_write(grid, stage, s_tile, chk_x2, r_x32, r_x16, epoch, x16_out ? &x16o : nullptr, out,
                              h1_cores);
    }
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
