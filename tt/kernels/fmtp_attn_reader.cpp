// The MTP draft layer's attention program 2 (compute fmtp_attn_compute.cpp, writer fmtp_attn_writer.cpp):
// fattn2_*.cpp's M-row program (M <= 4 rows at pos .. pos + M - 1, after attention program 1's M rows) with dense
// attention over every cell instead of the host's pool selection (Strata's runtime choice for the MTP layer), so it
// takes no selection tensor:
//   in      the coordinator reads w_a (row m's at word 16 m), broadcast at barrier 1; the oproj cores read X's tile of
//           the 4 streams (xown); the AP cores their KV head's Q rows
//   AP      the window: the last nblk <= NAPC NBMAX blocks of 32 cells (8 pools) up to cell pos + M - 1, from block
//           lo (0 until the context passes 8,800 cells), over NAPC cores per KV head (the whole grid): P =
//           min(nblk, NAPC) partials, partial c the blocks lo + [c nblk / P, (c + 1) nblk / P) (at most NBMAX); the mask as fattn2's ninth dims tile (cell j of a block
//           is row m's when 32 b + j <= pos + m; column M = 1)
//   AC      in two stages when P > 8 (Q = groups_q groups of partials): stage 1 core (g, t, d, q) combines partials
//           [q P / Q, (q + 1) P / Q) of (KV head g, column tile t, dims tile d) unnormalized (their max M_q, sum l_q
//           and O_q = sum exp(m_i - M_q) O_i: a partial again) into sub-slot SUB + ((g T_MAX + t) 8 + d) QMAX + q of
//           the partials (O, m, l: 3 tiles); a barrier; the final stage core (g, t, d), t < T, combines the Q
//           sub-partials (or the P partials when Q = 1), normalized, times the gate. Each reads its partials' m, l rows
//           as the rows of statistics tiles and their O tiles in batches of OB (rows 0..15 only when 12 M <= 16: the
//           rest of the tile is never read). A combine's tile operations (two per partial) bound it, not its reads.
//   oproj, H1f, H2f, R1, R2  as fattn2's M rows
// Runtime args: X fp32, (X bf16), Q, G, w_a, step addresses; grid.h's block; arena offsets X32, X16, P1, XOWN, O,
// HNF, LOF, L; AP: role (g * NAPC + c; 0xffffffff none), K cache, V cache (pool-major pages, NPOOL a KV head); AC: role
// (the core's order index: fmtp_attn_compute.cpp's roles), partials; oproj: W_o, tile; HC: W'_f, up_f, norm_f, item, tile; FFN: router W, gate|up W,
// down W, routed, R1 item, R2 tile; the stamps tensor; H1's 44 cores (the writer's); [GRID_PROFILE: the profile].
// Compile args: batch, depth, then a DRAM accessor. Defines: M_MAX.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"
#include "fused/tail_dm.h"

constexpr uint32_t cb_w = 0, cb_ones = 2;
constexpr uint32_t cb_akrm = 17, cb_avrm = 18;
constexpr uint32_t cb_cm = 28, cb_co = 29, cb_cl = 30, cb_cid = 31, cb_cg = 32;
constexpr uint32_t cb_zero = 60, cb_arena = 61, cb_prof = 63;
#ifndef NPOOL
#define NPOOL 4096  // g3_attn1.NPOOL: the K/V caches' pools a KV head (the host's define: CTX / 4)
#endif
constexpr uint32_t W_PAGE = 1088, R_PAGE = 2048, D = 2560, HD = 256, IDLE = 0xffffffff;
constexpr uint32_t NAPC = 55, NBMAX = 5, T_MAX = (12 * M_MAX + 31) / 32, PSTRIDE = 10 * T_MAX, OB = 4;
constexpr uint32_t cb_aq16 = 58, cb_akx = 22, NH = 24, QMAX = 6, SUB = 2 * NAPC * PSTRIDE;
namespace ft = fused::tail;
constexpr uint16_t B_ONE = 0x3f80, B_P4096 = 0x4580, B_M4096 = 0xc580;  // bf16 1, 4096, -4096

// the combine's roles from the core's order index k (every core has one): stage 1 (only when Q > 1) core (g, t, d, q)
// = k / Q's (g, t, d) and q = k % Q, k < 16 T Q; the final stage core (g, t, d) = (k / 8 T, k / 8 % T, k % 8), k < 16 T
inline uint32_t groups_q(uint32_t P, uint32_t T) {
    const uint32_t q = (P + 7) / 8, qmax = T == 1 ? 6 : 3;
    return P <= 8 ? 1 : q < qmax ? q : qmax;
}

// the fp32 tile element (r, c)'s byte offset
inline uint32_t elem(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 1024 + (r % 16) * 64 + (c % 16) * 4; }

// the bf16 tile element (r, c)'s byte offset
inline uint32_t elem16(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 512 + ((r % 16) * 16 + c % 16) * 2; }

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr auto d_args = TensorAccessorArgs<2>();
    using Acc = decltype(TensorAccessor(d_args, 0u, 0u));
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const auto x32 = acc(M_MAX * D * 4);  // a stream a page
    arg();                                // (X bf16: H1f's streams come from oproj)
    const auto qin = acc(HD * 2), gin = acc(HD * 4);  // a head-row a page: Q bf16, G fp32
    const auto wain = acc(M_MAX * 64), stepin = acc(64);
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_xown = arena + arg();
    const uint32_t r_wa = arena + arg(), r_o = arena + arg(), r_hnf = arena + arg(), r_lof = arena + arg();
    const uint32_t r_l = arena + arg();
    const uint32_t ap = arg();
    const auto kc = acc(2048), vc = acc(2048);
    const uint32_t ac = arg();
    const auto parts = acc(4096);
    const auto wo = acc(W_PAGE);
    const uint32_t o_tile = arg();
    const auto wd = acc(W_PAGE), wu = acc(W_PAGE), hnorm = acc(4 * D * 4);
    const uint32_t h_item = arg(), h_tile = arg();
    const auto wr = acc(R_PAGE), wgu = acc(W_PAGE), wsd = acc(W_PAGE), routed = acc(M_MAX * D * 4);
    const uint32_t r_item = arg(), s_tile = arg();
    const auto prof = acc(64);
    i += 44;  // H1's cores (the writer's)
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    fused::Stream<Acc> rs{ft::cb_rw, batch, R_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    noc_async_read(stepin.get_noc_addr(0), zero, 64);  // the step's words, through the zero tile before it is cleared
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
    const uint32_t pos = sw[0], M = sw[1], mp = sw[4];
    fused::fill_words(zero, 1024, 0);
    const uint32_t T = (12 * M + 31) / 32, blk = mp * 128;
    // the window: the last NAPC NBMAX blocks up to the pass's last cell (Strata's --mtp-window; drafts only)
    const uint32_t nall = (pos + M + 31) / 32, lo = nall > NAPC * NBMAX ? nall - NAPC * NBMAX : 0;
    const uint32_t nblk = nall - lo, P = nblk < NAPC ? nblk : NAPC;
    const uint32_t rs_m = r_p1 + mp * 4 * fused::hc::LO * 4, r_wf = r_lof + mp * 640;
    const uint32_t r_sh = r_l + mp * ft::N_ROUTER * 128;
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };

    // ---- the inputs: w_a (the coordinator's, broadcast at the first barrier), X's tiles (their owners'), q (the AP
    // cores')
    if (grid.coordinator) {
        noc_async_read(wain.get_noc_addr(0), r_wa, mp * 64);
        noc_async_read_barrier();
    }
    if (o_tile != IDLE) {  // complete at a later read barrier
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(x32.get_noc_addr(s, o_tile * blk), r_xown + s * blk, blk);
        }
    }
    const bool ap_on = ap != IDLE && ap % NAPC < P;
    if (ap_on) {
        // the KV head's 12 M q rows (row R = 12 m + h) into region O (idle until AC), then into q's row tiles
        // t = R / 32 (9 per t: the 8 dims tiles, then the mask's: 4096 on column m, -4096 on column M); the rest zero
        const uint32_t g = ap / NAPC, nq = 12 * M;
        for (uint32_t R = 0; R < nq; R++) {
            noc_async_read(qin.get_noc_addr((R / 12) * NH + 12 * g + R % 12), r_o + R * 512, 512);
        }
        cb_reserve_back(cb_aq16, 9 * T);
        const uint32_t q = get_write_ptr(cb_aq16);
        for (uint32_t j = 0; j < 9 * T; j++) {
            fused::l1_copy(zero, q + j * 2048, 2048);
        }
        noc_async_read_barrier();
        for (uint32_t R = 0; R < nq; R++) {
            const uint32_t base = q + (R / 32) * 9 * 2048 + elem16(R % 32, 0);
            for (uint32_t j = 0; j < 8; j++) {
                fused::l1_copy(r_o + R * 512 + j * 64, base + j * 2048, 32);
                fused::l1_copy(r_o + R * 512 + j * 64 + 32, base + j * 2048 + 512, 32);
            }
        }
        for (uint32_t R = 0; R < nq; R++) {
            const uint32_t mask = q + ((R / 32) * 9 + 8) * 2048;
            volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask);
            e[elem16(R % 32, R / 12) / 2] = B_P4096;
            e[elem16(R % 32, M) / 2] = B_M4096;
        }
        noc_async_read_barrier();
        asm volatile("fence rw, rw" ::: "memory");
        cb_push_back(cb_aq16, 9 * T);
    }
    const uint32_t Q = groups_q(P, T);
    const bool s1_on = Q > 1 && ac < 16 * T * Q, ac_on = ac < 16 * T;
    if ((ac_on || s1_on) && 12 * M <= 16) {  // the O tiles' second halves, never read: zero (a stale NaN times 0 is no 0)
        const auto& ci = get_local_cb_interface(cb_co);
        for (uint32_t p = 0; p < 2 * OB; p++) {
            fused::l1_copy(zero, ci.fifo_limit - ci.fifo_size + p * 4096 + 2048, 2048);
        }
        noc_async_read_barrier();
    }
    uint32_t epoch = 1;
    wait(epoch, r_wa, M_MAX * 64);
    // the ones tile, Sel (fused/hc_cmp.h h1_rs_m) and the words for the compute (tile 2: M, nblk, -, mp)
    cb_reserve_back(cb_ones, 3);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    {
        const uint32_t s = get_write_ptr(cb_ones) + 2048;
        fused::fill_words(s, 1024, 0);
        volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(s);
        for (uint32_t r = 0; r < mp; r++) {
            for (uint32_t c = r; c < 32; c += mp) {
                e[(c / 16) * 256 + r * 16 + c % 16] = 0x3f80;
            }
        }
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + 2048);
        t[0] = M, t[1] = nblk, t[3] = mp;
        asm volatile("fence rw, rw" ::: "memory");
    }
    cb_push_back(cb_ones, 3);

    // ---- AP: partial c of the KV head, its blocks' mask tiles (the cells' rows) and K, V pages
    if (ap_on) {
        const uint32_t g = ap / NAPC, c = ap % NAPC, b0 = c * nblk / P, b1 = (c + 1) * nblk / P;
        for (uint32_t b = lo + b0; b < lo + b1; b++) {
            cb_reserve_back(cb_akx, 1);
            const uint32_t x = get_write_ptr(cb_akx);
            fused::l1_copy(zero, x, 2048);
            cb_reserve_back(cb_akrm, 8);
            cb_reserve_back(cb_avrm, 8);
            const uint32_t dk = get_write_ptr(cb_akrm), dv = get_write_ptr(cb_avrm);
            noc_async_read_barrier();  // (the mask tile zeroed)
            for (uint32_t s = 0; s < 8; s++) {
                noc_async_read(kc.get_noc_addr(g * NPOOL + 8 * b + s), dk + s * 2048, 2048);
                noc_async_read(vc.get_noc_addr(g * NPOOL + 8 * b + s), dv + s * 2048, 2048);
            }
            volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x);
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t m = 0; m < M; m++) {
                    if (32 * b + r <= pos + m) {
                        e[elem16(r, m) / 2] = B_ONE;
                    }
                }
                e[elem16(r, M) / 2] = B_ONE;
            }
            asm volatile("fence rw, rw" ::: "memory");
            cb_push_back(cb_akx, 1);
            noc_async_read_barrier();
            cb_push_back(cb_akrm, 8);
            cb_push_back(cb_avrm, 8);
        }
    }
    wait(++epoch, 0, 0);

    // ---- AC: a combine's inputs: the identity, the statistics tiles (row p % 32 of tile p / 32: partial p's m or l
    // row of column tile t), with gate the gate tile (q row R = 32 t + r: row R / 12's head R % 12, dims tile d), the
    // O tiles in batches of OB (each tile's first half alone when only rows 0..15 are q rows); n partials, partial p's
    // O, m, l tiles at slot(p) + oo, + om, + ol
    auto combine_inputs = [&](uint32_t g, uint32_t t, uint32_t d, uint32_t n, auto slot, uint32_t oo, uint32_t om,
                              uint32_t ol, bool gate) {
        const uint32_t ns = (n + 31) / 32;
        cb_reserve_back(cb_cid, 1);
        const uint32_t id = get_write_ptr(cb_cid);
        fused::fill_words(id, 1024, 0);
        for (uint32_t r = 0; r < 32; r++) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(id + elem(r, r)) = 0x3f800000;
        }
        cb_push_back(cb_cid, 1);
        cb_reserve_back(cb_cm, ns);
        cb_reserve_back(cb_cl, ns);
        const uint32_t m = get_write_ptr(cb_cm), l = get_write_ptr(cb_cl);
        fused::fill_words(m, 1024 * ns, 0xf149f2ca);  // -1e30
        fused::fill_words(l, 1024 * ns, 0);
        for (uint32_t p = 0; p < n; p++) {
            const uint32_t at = (p / 32) * 4096 + ((p % 32) / 16) * 2048 + (p % 16) * 64;
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(parts.get_noc_addr(slot(p) + om, f * 1024), m + at + f * 1024, 64);
                noc_async_read(parts.get_noc_addr(slot(p) + ol, f * 1024), l + at + f * 1024, 64);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_cm, ns);
        cb_push_back(cb_cl, ns);
        if (gate) {
            cb_reserve_back(cb_cg, 1);
            const uint32_t gt = get_write_ptr(cb_cg);
            fused::l1_copy(zero, gt, 4096);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < 32 && 32 * t + r < 12 * M; r++) {
                const uint32_t R = 32 * t + r, page = (R / 12) * NH + 12 * g + R % 12;
                const uint32_t at = (r / 16) * 2048 + (r % 16) * 64;
                noc_async_read(gin.get_noc_addr(page, 32 * d * 4), gt + at, 64);
                noc_async_read(gin.get_noc_addr(page, 32 * d * 4 + 64), gt + at + 1024, 64);
            }
            noc_async_read_barrier();
            cb_push_back(cb_cg, 1);
        }
        const uint32_t obytes = 12 * M <= 16 ? 2048 : 4096;
        // a batch never past cb 29's end: a reservation that wraps is not split by tt-metal, and after a group of 6
        // or 7 partials the next one wrote past the buffer (the drafts wrong at 9..47 blocks, 2026-11-07)
        const auto& co = get_local_cb_interface(cb_co);
        for (uint32_t p0 = 0, k = 0; p0 < n; p0 += k) {
            const uint32_t room = (co.fifo_limit - co.fifo_wr_ptr) / 4096;
            k = n - p0 < OB ? n - p0 : OB;
            k = k < room ? k : room;
            cb_reserve_back(cb_co, k);
            const uint32_t dst = get_write_ptr(cb_co);
            for (uint32_t p = 0; p < k; p++) {
                noc_async_read(parts.get_noc_addr(slot(p0 + p) + oo), dst + p * 4096, obytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_co, k);
        }
    };
    auto sub = [&](uint32_t g, uint32_t t, uint32_t d) { return SUB + ((g * T_MAX + t) * 8 + d) * QMAX * 3; };
    if (Q > 1) {  // stage 1, then its barrier
        if (s1_on) {
            const uint32_t c = ac / Q, q = ac % Q, g = c / (8 * T), t = c / 8 % T, d = c % 8;
            const uint32_t p0 = q * P / Q, p1 = (q + 1) * P / Q;
            combine_inputs(g, t, d, p1 - p0, [&](uint32_t p) { return (g * NAPC + p0 + p) * PSTRIDE; }, 8 * t + d,
                           8 * T_MAX + t, 9 * T_MAX + t, false);
        }
        wait(++epoch, 0, 0);
    }
    if (ac_on) {
        const uint32_t g = ac / (8 * T), t = ac / 8 % T, d = ac % 8;
        if (Q > 1) {
            const uint32_t s0 = sub(g, t, d);
            combine_inputs(g, t, d, Q, [&](uint32_t q) { return s0 + 3 * q; }, 0, 1, 2, true);
        } else {
            combine_inputs(g, t, d, P, [&](uint32_t p) { return (g * NAPC + p) * PSTRIDE; }, 8 * t + d,
                           8 * T_MAX + t, 9 * T_MAX + t, true);
        }
    }
    const uint32_t c_pre = ft::oproj_prefetch(ws, wo, o_tile, depth * batch);
    wait(++epoch, r_o, NH * HD * 2 * mp);

    // ---- oproj (x streamed with the weights, one output tile) and the combine X += w_a out (fused/tail_dm.h)
    if (o_tile != IDLE) {
        for (uint32_t k = 0; k < ft::KC; k += batch) {
            fused::x_tiles_m(ft::cb_cx, r_o + k * mp * 64, batch, batch, mp);
            if (k >= c_pre) {
                ws.read(wo, o_tile * ft::KC + k, batch);
            }
        }
        ft::combine_inputs_m(ft::cb_cxs, ft::cb_cw, r_xown, r_wa, mp);
    }
    const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd, h_item);  // H1f's first weights before the barrier
    wait(++epoch, r_x32, 0);  // X is in its readers' arenas already (fused/tail_dm.h x_out_m)

    // ---- the FFN half: the mix with R1's first weights before H2f's barrier, then R1 and R2 (flayer_reader.cpp's)
    const uint32_t h2f_barrier = epoch + 3;
    uint32_t r1_pre = 0;
    auto hf_wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        if (n == h2f_barrier) {
            r1_pre = ft::r1_prefetch(ws, rs, wr, wgu, r_item);
        }
        wait(n, region, bytes);
    };
    const fused::hc::RegionsM hreg{r_x32, r_x16, r_xown, r_p1, rs_m, mp};
    fused::hc::read_m(ws, wd, wu, hnorm, h_item, h_tile, hreg, r_hnf, r_lof, zero, hf_wait, epoch, h1_pre);
    if (r_item != IDLE) {
        fused::x_tiles_m(ft::cb_x, r_hnf, ft::KD, ft::KD, mp);
        if (r_item < ft::N_ROUTER) {
            rs.read(wr, r_item * ft::KD + r1_pre, ft::KD - r1_pre);
            rs.flush();
        } else {
            ws.read(wgu, (r_item - ft::N_ROUTER) * 2 * ft::KD + r1_pre, 2 * ft::KD - r1_pre);
            ws.flush();
        }
    }
    if (s_tile != IDLE) {
        ws.read(wsd, s_tile * ft::KR2, ft::KR2);
        ws.flush();
    }
    wait(++epoch, r_l, mp * (ft::N_ROUTER * 128 + ft::FF * 2));
    if (s_tile != IDLE) {
        fused::x_tiles_m(ft::cb_x, r_sh, ft::KR2, ft::KD, mp);
        cb_reserve_back(ft::cb_sgl, 1);  // the shared expert's gate logit (logit 512: tile 16, column 0) by row
        fused::fill_rows(get_write_ptr(ft::cb_sgl), r_l + 16 * mp * 128, 64, mp);
        cb_push_back(ft::cb_sgl, 1);
        ft::combine_inputs_m(ft::cb_sxs, ft::cb_sw, r_xown, r_wf, mp);
        cb_reserve_back(ft::cb_srt, 1);
        fused::rows_from_dram(routed, 0, s_tile * blk, get_write_ptr(ft::cb_srt), mp);
        noc_async_read_barrier();
        cb_push_back(ft::cb_srt, 1);
    }
    wait(++epoch, r_x32, 0);
    if (grid.coordinator) {
        noc_async_write(get_write_ptr(cb_prof), prof.get_noc_addr(0), 64);
        noc_async_write_barrier();
    }
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the program's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
