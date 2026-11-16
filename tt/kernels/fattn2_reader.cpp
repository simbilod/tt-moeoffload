// Fused QSA attention layer, program 2 of 2 (G3 step 6): from the host's pool selection to the next X, on the whole
// grid with a barrier after each phase (compute fattn2_compute.cpp, writer fattn2_writer.cpp).
//
//   barrier 1   X, Q (fp32 [24 x 256], scaled), the gates G (sigmoid, fp32 [24 x 256]), the selection and w_a
//               arrive: the coordinator reads them from card memory (program 1's outputs and the host's selection).
//               The selection tensor is uint32 [528]: 512 pool ids, then word 512 nsel (the valid ones, 0..512; the
//               rest may name any pool) and word 513 own_valid (the own block's cells taken, 0..4); the step tensor
//               (uint32 [16], word 0 pos) gives the token's position: one program serves every token.
//   AP          the attention partials, 34 cores: per KV head g, 16 cores take 32 selected pools each (4 blocks of
//               8 pools = 32 cells) and one core the token's own block. Every block has a mask (-10000 on the cells
//               not taken: core k's block b takes 4 clamp(nsel - 32 k - 8 b, 0, 8) cells, the own block own_valid);
//               a core with no cell taken leaves a partial whose weight in the combine is exp(-10000) = 0. A block's K
//               and V are 8 row-major cache pages (pool-major: 4 cells x 256 bf16), tilized on the card;
//               S_b = q K_b^T (the 12 heads of g in rows, K read transposed by the matmul); m the row max over
//               the blocks, P_b = exp(S_b - m), l the row sum, O = sum_b P_b V_b. The partials (O: 8 tiles, m, l)
//               go to card memory.
//   AC          the combine, 16 cores (g, dims tile d): M = max_i m_i, w_i = exp(m_i - M),
//               O_d = (sum_i diag(w_i) O_i,d) / (sum_i diag(w_i) l_i), times the gate: region O (bf16 6144)
//   oproj       wo (6144 -> 2560) + the combine with w_a (fused/tail_dm.h)
//   H1f, H2f    the FFN half's hyper-connection mix (fused/hc_dm.h)
//   R1, R2      the router, the shared expert, the combine (fused/tail_dm.h); with -DEXCHANGE the host link
//               between them, as the DeltaNet layer's (fused/link.h): hn_f and the logits out, the routed experts'
//               sum back into region RT, broadcast at an extra barrier. The last X goes back to the X tensors
//               (the writer).
//
// Runtime args: X fp32, X bf16, Q, G, selection, w_a, step addresses; grid.h's block; arena offsets X32, X16, P1,
// RS, Q, G, SEL, ST, WA, O, HNF, LOF, WF, L, SH, RT; AP: role (g * 17 + k, k = 16 the own block; 0xffffffff none), K
// cache, V
// cache, the pools per KV head; AC: role (g * 8 + d), partials; oproj: W_o, tile; HC: W'_f, up_f, norm_f, item,
// tile; FFN: router W, gate|up W, down W, routed, R1 item, R2 tile; the profile tensor.
// Compile args: batch, depth, then a DRAM accessor.
//
// M rows (-DM_MAX=n, n <= 4: a verify pass, rows m < M at positions pos + m; M at runtime, the step tensor's words 0,
// 1, 4 = pos, M, Mp; fused/hc_dm.h's mp-row blocks), after program 1's M rows (fattn1_reader.cpp):
//   in      the coordinator reads the selection and w_a (row m's at word 16 m), broadcast at barrier 1; the oproj
//           cores read X's tile of the 4 streams (xown, the owner layout); the AP cores their KV head's Q rows
//   AP      over the union of the rows' selections: the selection tensor is uint32 [SEL_W]: word 0 the union's size
//           U, from word 16 its entries, each a pool id (bits 0..15) and its cell x row mask (bit 16 + 4 j + m: the
//           pool's cell j is row m's), the own blocks' valid cells among them (the host's: a row's own block is the
//           pool of its first own_valid cells). Partial i takes entries 32 i .. 32 i + 31 (4 blocks of 8 pools, the
//           last fewer: P = ceil(ceil(U / 8) / 4) partials per KV head, PMAX at most); core k of a KV head computes
//           partials k, k + NAP, ... q^T has the KV head's 12 M q rows (row 12 m + h: row m's head h) in columns,
//           T = ceil(12 M / 32) column tiles; the mask is a ninth dims tile: K's (cell rows, column m = the cell is
//           row m's, column M = 1) and q^T's (row m = 4096 on row m's columns, row M = -4096), so S^T = K q^T gets
//           0 on the cells a row takes and -4096 on the rest, the mask's product first (exact). Q (bf16, a head-row
//           a page, program 1's) is read once per core into region O (idle until AC) and placed into q's row tiles.
//   AC      32 cores (KV head g, column tile t, dims tile d; t < T): the partials' m, l rows as the rows of ceil(P /
//           32) statistics tiles, the O tiles streamed; the gate rows read from G (program 1's) by these cores
//   oproj, H1f, H2f, R1, R2  the DeltaNet program's M rows (fused/hc_dm.h read_m, fused/tail_dm.h's M-row helpers):
//           X's tiles to their owner and H1's cores, no broadcasts of X; with the host link (EXCHANGE) between R1
//           and R2 the DeltaNet program's M-row exchange (flayer_reader.cpp): the rows' hn_f and logits blocks out as
//           ceil(mp (5120 + 2176) / 8192) pages, M reply pages back (row m's routed sum), staged in region O (idle
//           after oproj) and broadcast at the link's barrier; R2 takes row min(m, M - 1) for row m
// The M-row code is preprocessed out at M_MAX = 1.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"
#include "fused/tail_dm.h"
#ifdef EXCHANGE
#include "fused/link.h"
#endif

#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_w = 0, cb_ones = 2;
constexpr uint32_t cb_aq = 16, cb_akrm = 17, cb_avrm = 18, cb_ak = 19, cb_amask = 22;
constexpr uint32_t cb_cm = 28, cb_co = 29, cb_cl = 30, cb_cid = 31, cb_cg = 32;
constexpr uint32_t cb_zero = 60, cb_arena = 61, cb_prof = 63;
constexpr uint32_t W_PAGE = 1088, R_PAGE = 2048, D = 2560, HD = 256, NB = 4, NPARTS = 17, NSEL = 512;
constexpr uint32_t IDLE = 0xffffffff, MASKED = 0xc61c4000;  // -10000.0f

// the fp32 tile element (r, c)'s byte offset
inline uint32_t elem(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 1024 + (r % 16) * 64 + (c % 16) * 4; }

#if M_MAX > 1
// M rows (g3_attn2.py's): AP cores per KV head, q^T's column tiles, the union's bound, the selection's words, the
// partials per KV head, a partial's tiles (O_t,d at 8 t + d, m_t at 8 T_MAX + t, l_t at 9 T_MAX + t)
constexpr uint32_t NAP = 32, T_MAX = (12 * M_MAX + 31) / 32, NU_MAX = M_MAX * 513;
constexpr uint32_t SEL_W = (16 + NU_MAX + 15) / 16 * 16, PMAX = (NU_MAX + 31) / 32, PSTRIDE = 10 * T_MAX;
constexpr uint32_t cb_aq16 = 58, cb_akx = 22, NH = 24;
namespace ft = fused::tail;
constexpr uint16_t B_ONE = 0x3f80, B_P4096 = 0x4580, B_M4096 = 0xc580;  // bf16 1, 4096, -4096

// the bf16 tile element (r, c)'s byte offset
inline uint32_t elem16(uint32_t r, uint32_t c) {
    return ((r / 16) * 2 + c / 16) * 512 + ((r % 16) * 16 + c % 16) * 2;
}
#endif

#if M_MAX == 1
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
    const auto x32 = acc(4 * D * 4), x16 = acc(4 * D * 2), qin = acc(HD * 4), gin = acc(HD * 4);
    const auto selin = acc((NSEL + 16) * 4), wain = acc(64), stepin = acc(64);
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_q = arena + arg(), r_g = arena + arg(), r_sel = arena + arg(), r_st = arena + arg();
    const uint32_t r_wa = arena + arg();
    const uint32_t r_o = arena + arg(), r_hnf = arena + arg(), r_lof = arena + arg(), r_wf = arena + arg();
    const uint32_t r_l = arena + arg();
    const uint32_t r_sh = arena + arg(), r_rt = arena + arg();
    const uint32_t ap = arg();
    const auto kc = acc(2048), vc = acc(2048);
    const uint32_t npool = arg();
    const uint32_t ac = arg();
    const auto parts = acc(4096);
    const auto wo = acc(W_PAGE);
    const uint32_t o_tile = arg();
    const auto wd = acc(W_PAGE), wu = acc(W_PAGE), hnorm = acc(4 * D * 4);
    const uint32_t h_item = arg(), h_tile = arg();
    const auto wr = acc(R_PAGE), wgu = acc(W_PAGE), wsd = acc(W_PAGE), routed = acc(D * 4);
    const uint32_t r_item = arg(), s_tile = arg();
    const auto prof = acc(64);
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    fused::Stream<Acc> rs{fused::tail::cb_rw, batch, R_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    fused::fill_words(zero, 1024, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };

    // ---- the inputs into every arena
    if (grid.coordinator) {
        noc_async_read(x32.get_noc_addr(0), r_x32, 4 * D * 4);
        noc_async_read(x16.get_noc_addr(0), r_x16, 4 * D * 2);
        for (uint32_t h = 0; h < 24; h++) {  // a row a page: the pages are spread over the DRAM banks
            noc_async_read(qin.get_noc_addr(h), r_q + h * HD * 4, HD * 4);
            noc_async_read(gin.get_noc_addr(h), r_g + h * HD * 4, HD * 4);
        }
        noc_async_read(selin.get_noc_addr(0), r_sel, (NSEL + 16) * 4);
        noc_async_read(stepin.get_noc_addr(0), r_st, 64);
        noc_async_read(wain.get_noc_addr(0), r_wa, 64);
        noc_async_read_barrier();
    }
#ifdef EXCHANGE
    fused::Link link;
    if (grid.coordinator) {
        link.open(H2D_CONFIG, D2H_CONFIG, D * 4, XCHG_OUT_BYTES);
    }
#endif
    uint32_t epoch = 1;
    wait(epoch, r_x32, r_o - r_x32);
    invalidate_l1_cache();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_st)[0];
    const uint32_t nsel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sel)[NSEL];
    const uint32_t own_valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sel)[NSEL + 1];

    // ---- AP
    if (ap != IDLE) {
        const uint32_t g = ap / NPARTS, k = ap % NPARTS;
        const bool own = k == NPARTS - 1;
        cb_reserve_back(cb_aq, 8);  // q: head h of g in row h
        const uint32_t q = get_write_ptr(cb_aq);
        for (uint32_t j = 0; j < 8; j++) {
            fused::l1_copy(zero, q + j * 4096, 4096);
        }
        noc_async_read_barrier();
        for (uint32_t h = 0; h < 12; h++) {
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t src = r_q + ((12 * g + h) * HD + 32 * j) * 4;
                fused::l1_copy(src, q + j * 4096 + h * 64, 64);
                fused::l1_copy(src + 64, q + j * 4096 + 1024 + h * 64, 64);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_aq, 8);
        // a mask per block: 0 on the cells taken (rows), -10000 on the rest
        auto push_mask = [&](uint32_t valid) {
            cb_reserve_back(cb_amask, 1);
            const uint32_t m = get_write_ptr(cb_amask);
            if (valid >= 32) {
                fused::l1_copy(zero, m, 4096);
                noc_async_read_barrier();
            } else {
                for (uint32_t r = 0; r < 32; r++) {  // row r: faces (r / 16) * 2 and + 1, 16 words each
                    const uint32_t v = r < valid ? 0 : MASKED, row = m + (r / 16) * 2048 + (r % 16) * 64;
                    fused::fill_words(row, 16, v);
                    fused::fill_words(row + 1024, 16, v);
                }
            }
            cb_push_back(cb_amask, 1);
        };
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sel);
        for (uint32_t b = 0; b < (own ? 1 : NB); b++) {
            if (own) {
                push_mask(own_valid);
            } else {
                const int32_t taken = (int32_t) nsel - (int32_t) (32 * k + 8 * b);
                push_mask(4 * (uint32_t) (taken < 0 ? 0 : taken > 8 ? 8 : taken));
            }
            cb_reserve_back(cb_akrm, 8);
            cb_reserve_back(cb_avrm, 8);
            const uint32_t dk = get_write_ptr(cb_akrm), dv = get_write_ptr(cb_avrm);
            for (uint32_t s = 0; s < 8; s++) {
                const uint32_t p = own ? (s == 0 ? pos / 4 : 0) : sel[32 * k + 8 * b + s];
                noc_async_read(kc.get_noc_addr(g * npool + p), dk + s * 2048, 2048);
                noc_async_read(vc.get_noc_addr(g * npool + p), dv + s * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(cb_akrm, 8);
            cb_push_back(cb_avrm, 8);
        }
    }
    wait(++epoch, 0, 0);

    // ---- AC
    if (ac != IDLE) {
        const uint32_t g = ac / 8, d = ac % 8;
        cb_reserve_back(cb_cid, 1);
        const uint32_t id = get_write_ptr(cb_cid);
        fused::fill_words(id, 1024, 0);
        for (uint32_t r = 0; r < 32; r++) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(id + elem(r, r)) = 0x3f800000;
        }
        cb_push_back(cb_cid, 1);
        // the partials' m and l rows as the rows of one statistics tile each (row 0 of a row tile: heads 0..15)
        cb_reserve_back(cb_cm, 1);
        cb_reserve_back(cb_cl, 1);
        const uint32_t m = get_write_ptr(cb_cm), l = get_write_ptr(cb_cl);
        fused::fill_words(m, 1024, 0xf149f2ca);  // -1e30
        fused::fill_words(l, 1024, 0);
        for (uint32_t p = 0; p < NPARTS; p++) {
            const uint32_t row = (p / 16) * 2048 + (p % 16) * 64;
            noc_async_read(parts.get_noc_addr((g * NPARTS + p) * 10 + 8), m + row, 64);
            noc_async_read(parts.get_noc_addr((g * NPARTS + p) * 10 + 9), l + row, 64);
        }
        cb_reserve_back(cb_co, NPARTS);
        const uint32_t o = get_write_ptr(cb_co);
        for (uint32_t p = 0; p < NPARTS; p++) {
            noc_async_read(parts.get_noc_addr((g * NPARTS + p) * 10 + d), o + p * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_cm, 1);
        cb_push_back(cb_cl, 1);
        cb_push_back(cb_co, NPARTS);
        cb_reserve_back(cb_cg, 1);  // the gates of g's 12 heads, dims tile d, in rows
        const uint32_t gt = get_write_ptr(cb_cg);
        fused::l1_copy(zero, gt, 4096);
        noc_async_read_barrier();
        for (uint32_t h = 0; h < 12; h++) {
            const uint32_t src = r_g + ((12 * g + h) * HD + 32 * d) * 4;
            fused::l1_copy(src, gt + h * 64, 64);
            fused::l1_copy(src + 64, gt + 1024 + h * 64, 64);
        }
        noc_async_read_barrier();
        cb_push_back(cb_cg, 1);
    }
    const uint32_t c_pre = fused::tail::oproj_prefetch(ws, wo, o_tile, depth * batch);
    wait(++epoch, r_o, 24 * HD * 2);

    // ---- oproj, the FFN half
    fused::tail::oproj_read(ws, wo, o_tile, c_pre, r_o, r_x32, r_wa);
    const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd, h_item);  // H1f's first weights before the barrier
    wait(++epoch, r_x32, 4 * D * 6);
    const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
    fused::hc::read(ws, wd, wu, hnorm, h_item, h_tile, hreg, r_hnf, r_lof, r_wf, zero, wait, epoch, h1_pre);
#ifdef EXCHANGE
    // R1 is the router's alone; the shared expert's gate|up runs after the page goes out, its h has its own
    // barrier (SH), and R2's GEMV runs before the reply instead of after it, all while the host computes
    const bool late = r_item != IDLE && r_item >= fused::tail::N_ROUTER;  // a gate|up core
    if (!late) {
        fused::tail::r1_inputs(ws, rs, wr, wgu, r_item, r_hnf, 0);
        fused::tail::r2_prefetch(ws, wsd, s_tile);
    }
    wait(++epoch, r_l, fused::tail::N_ROUTER * 128);  // R1: the router's logits
    // the host link: hn_f (bf16) and the logits out, the routed experts' sum back into RT (the page is staged in
    // cb 19's L1, AP's K tiles, idle since AP)
    if (grid.coordinator) {
        const uint32_t page = get_write_ptr(cb_ak);
        fused::l1_copy(r_hnf, page, D * 2);
        fused::l1_copy(r_l, page + D * 2, 17 * 128);
        noc_async_read_barrier();
#ifdef GRID_PROFILE
        fused::Grid::mark(epoch + 2);  // the page out: the host's part of the exchange's barrier (after SH's)
#endif
        link.send(page);
    }
    if (late) {
        fused::tail::r1_inputs(ws, rs, wr, wgu, r_item, r_hnf, 0);
        fused::tail::r2_prefetch(ws, wsd, s_tile);
    }
    fused::tail::r2_inputs(s_tile, r_l, r_sh, r_x32, r_wf, false);
    wait(++epoch, r_sh, fused::tail::FF * 2);  // SH: the shared expert's h
    if (s_tile != IDLE) {
        fused::x_tiles_cpu(fused::tail::cb_x, r_sh, fused::tail::KR2, fused::tail::KD);
    }
    if (grid.coordinator) {
        link.recv(r_rt);
    }
    wait(++epoch, r_rt, D * 4);
    fused::tail::r2_routed(routed, r_rt, s_tile, r_x32, wait, epoch);
    if (grid.coordinator) {
        link.close();
    }
#else
    fused::tail::r1_read(ws, rs, wr, wgu, wsd, r_item, s_tile, r_hnf, r_l, r_sh, wait, epoch);
    fused::tail::r2_read(routed, 0, s_tile, r_l, r_sh, r_x32, r_wf, wait, epoch);
#endif
    if (grid.coordinator) {
        noc_async_write(get_write_ptr(cb_prof), prof.get_noc_addr(0), 64);
        noc_async_write_barrier();
    }
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the program's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
#else
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
    const auto selin = acc(SEL_W * 4), wain = acc(M_MAX * 64), stepin = acc(64);
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_xown = arena + arg();
    arg(), arg();  // (Q, G: read by their consumers at M rows)
    const uint32_t r_sel = arena + arg();
    arg();  // (ST)
    const uint32_t r_wa = arena + arg();
    const uint32_t r_o = arena + arg(), r_hnf = arena + arg(), r_lof = arena + arg();
    arg();  // (WF: after LOF at M rows)
    const uint32_t r_l = arena + arg();
    arg(), arg();  // (SH: after L at M rows; RT: the host link's)
    const uint32_t ap = arg();
    const auto kc = acc(2048), vc = acc(2048);
    const uint32_t npool = arg();
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
    const uint32_t M = sw[1], mp = sw[4];  // (word 0, the position: program 1's)
    fused::fill_words(zero, 1024, 0);
    const uint32_t T = (12 * M + 31) / 32, blk = mp * 128;
    const uint32_t rs_m = r_p1 + mp * 4 * fused::hc::LO * 4, r_wf = r_lof + mp * 640;
    const uint32_t r_sh = r_l + mp * ft::N_ROUTER * 128;
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };

    // ---- the inputs: the selection and w_a (the coordinator's, broadcast at the first barrier), X's tiles (their
    // owners'), q (the AP cores')
    if (grid.coordinator) {
        noc_async_read(selin.get_noc_addr(0), r_sel, SEL_W * 4);
        noc_async_read(wain.get_noc_addr(0), r_wa, mp * 64);
        noc_async_read_barrier();
    }
#ifdef EXCHANGE
    fused::Link link;
    if (grid.coordinator) {
        link.open(H2D_CONFIG, D2H_CONFIG, D * 4, XCHG_OUT_BYTES);
    }
#endif
    if (o_tile != IDLE) {  // complete at a later read barrier
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(x32.get_noc_addr(s, o_tile * blk), r_xown + s * blk, blk);
        }
    }
    if (ap != IDLE) {
        // the KV head's 12 M q rows (row R = 12 m + h) into region O (idle until AC), then into q's row tiles
        // t = R / 32
        // (9 per t: the 8 dims tiles, then the mask's: 4096 on column m, -4096 on column M); the rest zero
        const uint32_t g = ap / NAP, nq = 12 * M;
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
    uint32_t epoch = 1;
    wait(epoch, r_sel, r_wa + M_MAX * 64 - r_sel);
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");
    volatile tt_l1_ptr uint32_t* sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sel);
    const uint32_t U = sel[0], nblk = (U + 7) / 8, P = (nblk + 3) / 4;
    // the ones tile, Sel (fused/hc_cmp.h h1_rs_m) and the words for the compute (tile 2: M, the union's blocks, -, mp)
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

    // ---- AP: partials k, k + NAP, ... of the KV head; per block its K mask tile (the cells' rows) and K, V pages
    if (ap != IDLE) {
        const uint32_t g = ap / NAP;
        for (uint32_t p = ap % NAP; p < P; p += NAP) {
            const uint32_t nb = nblk - 4 * p < 4 ? nblk - 4 * p : 4;
            for (uint32_t b = 0; b < nb; b++) {
                cb_reserve_back(cb_akx, 1);
                const uint32_t x = get_write_ptr(cb_akx);
                fused::l1_copy(zero, x, 2048);
                cb_reserve_back(cb_akrm, 8);
                cb_reserve_back(cb_avrm, 8);
                const uint32_t dk = get_write_ptr(cb_akrm), dv = get_write_ptr(cb_avrm);
                noc_async_read_barrier();  // (the mask tile zeroed)
                volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x);
                for (uint32_t s = 0; s < 8; s++) {
                    const uint32_t at = 32 * p + 8 * b + s, w = at < U ? sel[16 + at] : 0, pool = w & 0xffff;
                    noc_async_read(kc.get_noc_addr(g * npool + pool), dk + s * 2048, 2048);
                    noc_async_read(vc.get_noc_addr(g * npool + pool), dv + s * 2048, 2048);
                    for (uint32_t j = 0; j < 4; j++) {
                        const uint32_t r = 4 * s + j;
                        for (uint32_t m = 0; m < M; m++) {
                            if ((w >> (16 + 4 * j + m)) & 1) {
                                e[elem16(r, m) / 2] = B_ONE;
                            }
                        }
                        e[elem16(r, M) / 2] = B_ONE;
                    }
                }
                asm volatile("fence rw, rw" ::: "memory");
                cb_push_back(cb_akx, 1);
                noc_async_read_barrier();
                cb_push_back(cb_akrm, 8);
                cb_push_back(cb_avrm, 8);
            }
        }
    }
    wait(++epoch, 0, 0);

    // ---- AC: core (g, t, d), t < T: the identity, the statistics tiles (row p % 32 of tile p / 32: partial p's m or
    // l row of column tile t), the gate tile (q row R = 32 t + r: row R / 12's head R % 12, dims tile d), the O tiles
    if (ac != IDLE && ac / 8 % T_MAX < T) {
        const uint32_t g = ac / (8 * T_MAX), t = ac / 8 % T_MAX, d = ac % 8, ns = (P + 31) / 32;
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
        for (uint32_t p = 0; p < P; p++) {
            const uint32_t at = (p / 32) * 4096 + ((p % 32) / 16) * 2048 + (p % 16) * 64;
            const uint32_t slot = (g * PMAX + p) * PSTRIDE;
            for (uint32_t f = 0; f < 2; f++) {
                noc_async_read(parts.get_noc_addr(slot + 8 * T_MAX + t, f * 1024), m + at + f * 1024, 64);
                noc_async_read(parts.get_noc_addr(slot + 9 * T_MAX + t, f * 1024), l + at + f * 1024, 64);
            }
        }
        cb_reserve_back(cb_cg, 1);
        const uint32_t gt = get_write_ptr(cb_cg);
        fused::l1_copy(zero, gt, 4096);
        noc_async_read_barrier();
        cb_push_back(cb_cm, ns);
        cb_push_back(cb_cl, ns);
        for (uint32_t r = 0; r < 32 && 32 * t + r < 12 * M; r++) {
            const uint32_t R = 32 * t + r, page = (R / 12) * NH + 12 * g + R % 12;
            const uint32_t at = (r / 16) * 2048 + (r % 16) * 64;
            noc_async_read(gin.get_noc_addr(page, 32 * d * 4), gt + at, 64);
            noc_async_read(gin.get_noc_addr(page, 32 * d * 4 + 64), gt + at + 1024, 64);
        }
        noc_async_read_barrier();
        cb_push_back(cb_cg, 1);
        for (uint32_t p = 0; p < P; p++) {  // the O tiles, one at a time (the compute takes them in groups of 16)
            cb_reserve_back(cb_co, 1);
            noc_async_read(parts.get_noc_addr((g * PMAX + p) * PSTRIDE + 8 * t + d), get_write_ptr(cb_co), 4096);
            noc_async_read_barrier();
            cb_push_back(cb_co, 1);
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
    auto r1_inputs_m = [&]() {
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
    };
#ifdef EXCHANGE
    // As the one-row program's: R1 the router's alone, gate|up after the pages go out, h's own barrier (SH)
    const bool late = r_item != IDLE && r_item >= ft::N_ROUTER;  // a gate|up core
    if (!late) {
        r1_inputs_m();
    }
    wait(++epoch, r_l, mp * ft::N_ROUTER * 128);  // R1: the router's logits
    // the host link: the rows' hn_f and logits blocks out from region O, the M rows' routed sums back into it
    if (grid.coordinator) {
        constexpr uint32_t row_bytes = D * 2 + ft::N_ROUTER * 128;  // a row's hn_f (bf16) and logits (fp32)
        fused::l1_copy(r_hnf, r_o, mp * D * 2);
        fused::l1_copy(r_l, r_o + mp * D * 2, mp * ft::N_ROUTER * 128);
        noc_async_read_barrier();
#ifdef GRID_PROFILE
        fused::Grid::mark(epoch + 2);  // the pages out: the host's part of the exchange's barrier (after SH's)
#endif
        for (uint32_t k = 0; k * XCHG_OUT_BYTES < mp * row_bytes; k++) {
            link.send(r_o + k * XCHG_OUT_BYTES);
        }
    }
    if (late) {
        r1_inputs_m();
    }
    if (s_tile != IDLE) {
        cb_reserve_back(ft::cb_sgl, 1);  // the shared expert's gate logit (logit 512: tile 16, column 0) by row
        fused::fill_rows(get_write_ptr(ft::cb_sgl), r_l + 16 * mp * 128, 64, mp);
        cb_push_back(ft::cb_sgl, 1);
        ft::combine_inputs_m(ft::cb_sxs, ft::cb_sw, r_xown, r_wf, mp);
    }
    wait(++epoch, r_sh, mp * ft::FF * 2);  // SH: the shared expert's h
    if (s_tile != IDLE) {
        fused::x_tiles_cpu(ft::cb_x, r_sh, ft::KR2, ft::KD, mp);
    }
    if (grid.coordinator) {
        for (uint32_t m = 0; m < M; m++) {
            link.recv(r_o + m * D * 4);
        }
    }
#else
    r1_inputs_m();
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
#endif
#ifdef EXCHANGE
    wait(++epoch, r_o, M * D * 4);
    if (s_tile != IDLE) {  // row m's routed sum: the reply's row min(m, M - 1), tile s_tile's 32 values
        cb_reserve_back(ft::cb_srt, 1);
        const uint32_t t = get_write_ptr(ft::cb_srt);
        for (uint32_t m = 0; m < mp; m++) {
            const uint32_t src = r_o + (m < M ? m : M - 1) * D * 4 + s_tile * 128;
            fused::l1_copy(src, t + m * 64, 64);
            fused::l1_copy(src + 64, t + 1024 + m * 64, 64);
        }
        noc_async_read_barrier();
        cb_push_back(ft::cb_srt, 1);
    }
    if (grid.coordinator) {
        link.close();
    }
#endif
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
#endif
