// Fused QSA attention layer, program 1 of 2 (G3 step 5): one decode token from X to the indexer's pool scores, on the
// whole grid with a barrier after each phase (compute fattn1_compute.cpp, writer fattn1_writer.cpp). The host takes
// the top 512 pools from the scores (top-k is a poor fit for the card), then program 2 attends.
//
//   barrier 1   X arrives (this test: the coordinator reads it from card memory)
//   H1a, H2a    the hyper-connection mix (fused/hc_dm.h): hn
//   P           the projection hn -> 436 tiles: per head q (8) | gate (8) x 24, k (8) x 2, v (8) x 2, the indexer's
//               q (4) x 4 and k (4); each core a run of <= 4 tiles; sigmoid on the gate tiles; region PA (fp32)
//   N           per core one role, from region PA:
//                 score b (64 cores)  the indexer's 4 q heads normed (i_q_norm) and rotated, then the 32 pools of
//                                     block b and of blocks b + 64 j once the token is past them (8K cells each;
//                                     up to CTX/8192 blocks, NPOOL from the host's define, 64K pools at 256K
//                                     cells):
//                                     sum_h relu(q_h k_p) / sqrt(128), k_p the pooled keys (fp32 tiles of 32 pools x
//                                     32 dims, read transposed by the matmul)
//                 q head h (24)       RMSNorm (q_norm), RoPE on dims 0..63, / sqrt(256): Q's row h
//                 kv head g (2)       k: RMSNorm (k_norm), RoPE; k and v (bf16) into the cache at the token's cell;
//                                     head 0 also writes the indexer's raw key to the raw-key cache
//                 pool (1)            when the token closes a block of 4: mean of its raw keys, RMSNorm
//                                     (i_k_norm), RoPE at the block's first position, into the pooled-key cache
// RoPE (NEOX on the first 64 dims, IMRoPE's sections all at the token's position): y_i = x_i c_i - x_{i+32} s_i,
// y_{i+32} = x_i s_i + x_{i+32} c_i on row-0 tiles 0 and 1; c, s from the host's table for every position, rows
// 4 pos + (cos t, sin t, cos t0, sin t0), t0 = pos - 3 (the block's first position).
// The token's position comes from the step tensor (uint32 [16], word 0 pos) in card memory, so one program serves
// every token; the pool role is always assigned and its result written only when pos % 4 == 3.
//
// Runtime args: X fp32, X bf16; grid.h's block; arena offsets X32, X16, P1, RS, HNA, LO, WA, PA; HC: W', up, norm, item,
// tile; P: W, first tile, count; N: role, index, the step tensor, pooled cache, raw-key cache, rope table, q_norm,
// k_norm, i_q_norm, i_k_norm; the profile tensor; the score tensor, program 2's selection tensor (both used only with
// -DEXCHANGE: the coordinator then sends the scores to the host over the D2H socket, one page per XCHG_OUT_BYTES / 4
// pools up to the token's block (staged a page at a time, two buffers), and writes the host's reply, the selection words, into program
// 2's selection tensor, so no host sync sits between the two programs).
// Compile args: batch, depth, then a DRAM accessor.
//
// M rows (-DM_MAX=n, n <= 4: a verify pass, rows m < M at positions pos + m; M at runtime, the step tensor's words 0,
// 1, 4 = pos, M, Mp; fused/hc_dm.h's mp-row blocks): X and the mix as the DeltaNet program's M rows (the owner
// layout: X tensors a stream a page, each stream read by its 11 H1 cores 1/11 each and exchanged, X's tile by the H2
// core that owns it; H1's 44 cores by item as runtime args after the selection tensor's); P on mp-row tiles, region
// PA in mp-row blocks (broadcast); N per row: the norm weights read once and copied down the rows, row m's RoPE rows
// at pos + m, the pooled keys read once for all rows (a score tile's row m is row m's scores), Q (bf16, a head-row a
// page: row m head h at page 24 m + h) and G (fp32, the same pages), k, v and the raw key at cell pos + m. The pool
// role takes the block that closes inside the pass (row mc = (3 - pos % 4) % 4, when mc < M): its cells before pos
// from the raw-key cache, the rest from PA's rows, and scores it for every row (the score tensor's word NPOOL of row
// m: the host puts it in place of that pool's stale score for rows m >= mc). The score tensor is [M_MAX x (NPOOL +
// 16)], a row a page. With the host link (EXCHANGE) the coordinator sends each row's scores as the one-row program
// does at position pos + m (rows m >= mc with the closed block's score in its place), staged in region PA, and
// writes the host's reply, the rows' union words (g3_attn2.union_words, SEL_W of them), into program 2's selection
// tensor. The M-row code is preprocessed out at M_MAX = 1.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"
#ifdef EXCHANGE
#include "fused/link.h"
#endif

#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2;
constexpr uint32_t cb_nin = 23, cb_nw = 24, cb_nr = 25, cb_npk = 29, cb_nv = 32;
constexpr uint32_t cb_meta = 22, cb_zero = 60, cb_arena = 61, cb_prof = 63;
constexpr uint32_t W_PAGE = 1088, D = 2560, KD = 80, HD = 256, IHD = 128;
constexpr uint32_t T_IQ = 416, T_IK = 432, T_K = 384, T_V = 400;
constexpr uint32_t R_SCORE = 0, R_QHEAD = 1, R_KVHEAD = 2, R_POOL = 3, IDLE = 0xffffffff;
#ifndef NPOOL
#define NPOOL 4096  // g3_attn1.NPOOL (the host's define: CTX / 4)
#endif
#ifndef RAWK
#define RAWK 8  // g3_attn1.RAWK: the raw-key ring's rows (position p at row p % RAWK)
#endif
constexpr uint32_t NSCORE = 64, NBLK_MAX = NPOOL / 32 / NSCORE;  // the score cores, the blocks one takes at most
static_assert(NBLK_MAX >= 1 && NBLK_MAX <= 32, "the writer's stage holds the blocks' scores (g3_attn1: CTX <= 256K)");
// the blocks score core `index` takes: index + 64 j while a pool of it lies before `last`'s block (at least one)
inline uint32_t score_blocks(uint32_t index, uint32_t last) {
    uint32_t n = 1;
    while (n < NBLK_MAX && 32 * (index + n * NSCORE) < last / 4) n++;
    return n;
}
constexpr uint32_t SCORE_BYTES = NPOOL * 4, REPLY_BYTES = D * 4, SEL_BYTES = (512 + 16) * 4;

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
    const auto x32 = acc(4 * D * 4), x16 = acc(4 * D * 2);
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_hna = arena + arg(), r_lo = arena + arg(), r_wa = arena + arg(), r_pa = arena + arg();
    const auto wd = acc(W_PAGE), wu = acc(W_PAGE), hnorm = acc(4 * D * 4);
    const uint32_t h_item = arg(), h_tile = arg();
    const auto wp = acc(W_PAGE);
    const uint32_t p_first = arg(), p_count = arg();
    const uint32_t role = arg(), index = arg();
    const auto step = acc(64);
    const auto pooled = acc(4096), rawk = acc(IHD * 4), rope = acc(32 * 4);
    const auto qn = acc(HD * 4), kn = acc(HD * 4), iqn = acc(IHD * 4), ikn = acc(IHD * 4);
    const auto prof = acc(64);
    const auto scores = acc(SCORE_BYTES + 64);  // the score tensor (the writers'), program 2's selection: the link's
    const auto sel_out = acc(SEL_BYTES);
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    noc_async_read(step.get_noc_addr(0), zero, 64);  // the step's words, through the zero tile before it is cleared
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero)[0];
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

    if (grid.coordinator) {
        noc_async_read(x32.get_noc_addr(0), r_x32, 4 * D * 4);
        noc_async_read(x16.get_noc_addr(0), r_x16, 4 * D * 2);
        noc_async_read_barrier();
    }
    const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd, h_item);  // H1's first weights before the first barrier
    uint32_t epoch = 1;
    wait(epoch, r_x32, r_p1 - r_x32);
    const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
    fused::hc::read(ws, wd, wu, hnorm, h_item, h_tile, hreg, r_hna, r_lo, r_wa, zero, wait, epoch, h1_pre);

    // ---- P
    if (p_count) {
        fused::x_tiles(cb_x, r_hna, KD, KD);
        ws.read(wp, p_first * KD, p_count * KD);
        ws.flush();
    }
    // N's inputs that need no activation, before the barrier
    auto row0s = [&](const Acc& src, uint32_t row, uint32_t off, uint32_t n, uint32_t dst) {
        for (uint32_t t = 0; t < n; t++) {
            fused::row0_from_dram(src, row, off + t * 128, dst + t * 4096);
        }
    };
    auto push_dram = [&](uint32_t cb, const Acc& src, uint32_t row, uint32_t off, uint32_t n) {
        cb_reserve_back(cb, n);
        row0s(src, row, off, n, get_write_ptr(cb));
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    auto push_arena = [&](uint32_t cb, uint32_t src, uint32_t n) {
        cb_reserve_back(cb, n);
        const uint32_t d = get_write_ptr(cb);
        for (uint32_t t = 0; t < n; t++) {
            fused::row0_from_l1(src + t * 128, d + t * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    // the pooled keys of the score role's blocks index + 64 j, j in [from, to): a block's 4 fp32 tiles at a time
    auto push_pooled = [&](uint32_t from, uint32_t to) {
        for (uint32_t j = from; j < to; j++) {
            cb_reserve_back(cb_npk, 4);
            const uint32_t d = get_write_ptr(cb_npk);
            for (uint32_t t = 0; t < 4; t++) {
                noc_async_read(pooled.get_noc_addr((index + j * NSCORE) * 4 + t), d + t * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb_npk, 4);
        }
    };
#ifdef EXCHANGE
    // a score row's first `pages` pages to the host, a page at a time through two buffers at the start of region PA
    // (the coordinator's; the send two pages back must have left a buffer before it is refilled), word `patch` (the
    // closed block's pool, or none) replaced by `value`
    auto send_pages = [&](fused::Link& link, uint32_t row, uint32_t pages, uint32_t patch, uint32_t value) {
        constexpr uint32_t per = XCHG_OUT_BYTES / 4;
        for (uint32_t p = 0; p < pages; p++) {
            const uint32_t buf = r_pa + (p % 2) * XCHG_OUT_BYTES;
            if (p >= 2) {
                noc_async_write_barrier();
            }
            noc_async_read(scores.get_noc_addr(row, p * XCHG_OUT_BYTES), buf, XCHG_OUT_BYTES);
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (patch / per == p) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf)[patch % per] = value;
            }
            link.send(buf);
        }
    };
#endif
    if (role == R_SCORE) {
        // the blocks index + 64 j once a pool of them lies before the token's block (the compute learns their count
        // from cb_meta); the first two (cb_npk's 8 tiles) before the barrier, the rest once the q heads are in
        cb_reserve_back(cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = score_blocks(index, pos);
        cb_push_back(cb_meta, 1);
        push_dram(cb_nw, iqn, 0, 0, 4);
        push_dram(cb_nr, rope, 4 * pos, 0, 1);
        push_dram(cb_nr, rope, 4 * pos + 1, 0, 1);
        push_pooled(0, score_blocks(index, pos) < 2 ? 1 : 2);
    } else if (role == R_QHEAD || role == R_KVHEAD) {
        push_dram(cb_nw, role == R_QHEAD ? qn : kn, 0, 0, 8);
        push_dram(cb_nr, rope, 4 * pos, 0, 1);
        push_dram(cb_nr, rope, 4 * pos + 1, 0, 1);
    } else if (role == R_POOL) {
        push_dram(cb_nw, ikn, 0, 0, 4);
        push_dram(cb_nr, rope, 4 * pos + 2, 0, 1);
        push_dram(cb_nr, rope, 4 * pos + 3, 0, 1);
        push_dram(cb_nw, iqn, 0, 0, 4);  // then the own block's score's (the indexer's q heads), and their rope rows
        push_dram(cb_nr, rope, 4 * pos, 0, 1);
        push_dram(cb_nr, rope, 4 * pos + 1, 0, 1);
        // the block's earlier raw keys (cells pos - 3 .. pos - 1, from the cache: no activation needed) into the 16
        // tiles of the 4 key groups; the token's own key joins them after the barrier
        cb_reserve_back(cb_nin, 16);
        const uint32_t d = get_write_ptr(cb_nin);
        for (uint32_t t = 0; t < 4; t++) {
            for (uint32_t c = 0; c < 3; c++) {
                fused::row0_from_dram(rawk, (pos >= 3 ? pos - 3 + c : 0) % RAWK, t * 128, d + (4 * t + c) * 4096);
            }
        }
        noc_async_read_barrier();
    }
    wait(++epoch, r_pa, 436 * 128);

    // ---- N
    if (role == R_SCORE) {
        for (uint32_t h = 0; h < 4; h++) {
            push_arena(cb_nin, r_pa + (T_IQ + 4 * h) * 128, 4);
        }
        push_pooled(2, score_blocks(index, pos));
    } else if (role == R_QHEAD) {
        push_arena(cb_nin, r_pa + 16 * index * 128, 8);
    } else if (role == R_KVHEAD) {
        push_arena(cb_nin, r_pa + (T_K + 8 * index) * 128, 8);
        push_arena(cb_nv, r_pa + (T_V + 8 * index) * 128, 8);
    } else if (role == R_POOL) {
        // the block's 4 raw keys, tile by tile: cells pos - 3 .. pos - 1 from the cache, the token's own from PA
        // (computed at every position, written only when the token closes its block; row 0 stands in below 3)
        const uint32_t d = get_write_ptr(cb_nin);  // reserved before the barrier, the cache rows in
        for (uint32_t t = 0; t < 4; t++) {
            fused::row0_from_l1(r_pa + (T_IK + t) * 128, d + (4 * t + 3) * 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_nin, 16);
        // the own block's score: the q heads (once the raw keys are consumed)
        for (uint32_t h = 0; h < 4; h++) {
            push_arena(cb_nin, r_pa + (T_IQ + 4 * h) * 128, 4);
        }
    }
    wait(++epoch, 0, 0);
#ifdef EXCHANGE
    // the host link: the pool scores up to the token's block (complete in card memory: every writer flushed them
    // before arriving) out, a page per XCHG_OUT_BYTES / 4 pools, the host's selection back, into program 2's
    // selection tensor; staged in the coordinator's region PA (no one reads it after the last barrier), a page at a
    // time in two buffers (8 pages at 64K cells would not fit PA)
    if (grid.coordinator) {
        fused::Link link;
        link.open(H2D_CONFIG, D2H_CONFIG, REPLY_BYTES, XCHG_OUT_BYTES);
        const uint32_t reply = r_pa + 2 * XCHG_OUT_BYTES;
        const uint32_t pages = pos / 4 * 4 / XCHG_OUT_BYTES + 1;  // g4_model.Link.serve_select's
        noc_async_read(scores.get_noc_addr(0, SCORE_BYTES), reply, 64);  // the own block's score (the pool role's)
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reply)[0];
#ifdef GRID_PROFILE
        fused::Grid::mark(epoch + 1);  // the host's part: the scores out to the selection back
#endif
        send_pages(link, 0, pages, pos % 4 == 3 ? pos / 4 : 0xffffffff, own);
        link.recv(reply);
#ifdef GRID_PROFILE
        fused::Grid::mark(epoch + 2);
#endif
        noc_async_write(reply, sel_out.get_noc_addr(0), SEL_BYTES);
        noc_async_write_barrier();
        link.close();
    }
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
constexpr uint32_t SEL_W = (16 + M_MAX * 513 + 15) / 16 * 16;  // program 2's union words (g3_attn2.m_sizes)

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<2>();
    using Acc = decltype(TensorAccessor(d_args, 0u, 0u));
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const auto x32 = acc(M_MAX * D * 4), x16 = acc(M_MAX * D * 2);  // a stream a page
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_xown = arena + arg();
    const uint32_t r_hna = arena + arg(), r_lo = arena + arg();
    arg();  // (WA: the inject weights follow LO at M rows)
    const uint32_t r_pa = arena + arg();
    const auto wd = acc(W_PAGE), wu = acc(W_PAGE), hnorm = acc(4 * D * 4);
    const uint32_t h_item = arg(), h_tile = arg();
    const auto wp = acc(W_PAGE);
    const uint32_t p_first = arg(), p_count = arg();
    const uint32_t role = arg(), index = arg();
    const auto step = acc(64);
    const auto pooled = acc(4096), rawk = acc(IHD * 4), rope = acc(32 * 4);
    const auto qn = acc(HD * 4), kn = acc(HD * 4), iqn = acc(IHD * 4), ikn = acc(IHD * 4);
    const auto prof = acc(64);
    const auto scores = acc(SCORE_BYTES + 64);  // a row a page (the writers'), program 2's selection: the link's
    const auto sel_out = acc(SEL_W * 4);
    const uint32_t h1_cores = i;  // H1's 44 cores by item 4 n + s, NOC x | y << 16
    i += 44;
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    noc_async_read(step.get_noc_addr(0), zero, 64);  // the step's words, through the zero tile before it is cleared
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero);
    const uint32_t pos = sw[0], M = sw[1], mp = sw[4];
    fused::fill_words(zero, 1024, 0);
    // the ones tile, Sel (fused/hc_cmp.h h1_rs_m: Sel[i][R] = 1 when R % mp == i, i < mp) and the step's words for the
    // compute (tile 2: M, -, -, mp)
    cb_reserve_back(cb_ones, 3);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    {
        const uint32_t sel = get_write_ptr(cb_ones) + 2048;
        fused::fill_words(sel, 1024, 0);
        volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sel);
        for (uint32_t r = 0; r < mp; r++) {
            for (uint32_t c = r; c < 32; c += mp) {
                e[(c / 16) * 256 + r * 16 + c % 16] = 0x3f80;
            }
        }
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel + 2048);
        t[0] = M, t[3] = mp;
        asm volatile("fence rw, rw" ::: "memory");
    }
    cb_push_back(cb_ones, 3);
    const uint32_t rs_m = r_p1 + mp * 4 * fused::hc::LO * 4;  // the r_s after P1 (fused/hc_dm.h RegionsM)
    const uint32_t blk = mp * 128;                            // an fp32 tile's mp-row block

    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };

    // ---- X (the DeltaNet program's M rows): each of a stream's 11 H1 cores reads 1/11 of its blocks and sends it to
    // the other 10, then arrives at the first barrier (the writer does not); the H2 cores read their tile's blocks of
    // the 4 streams (xown; complete at their next read barrier)
    if (h_item != IDLE) {
        const uint32_t s = h_item & 3, n = h_item >> 2, t0 = 80 * n / 11, t1 = 80 * (n + 1) / 11;
        const uint32_t o32 = r_x32 + t0 * blk, o16 = r_x16 + t0 * mp * 64;
        noc_async_read(x32.get_noc_addr(s, t0 * blk), o32, (t1 - t0) * blk);
        noc_async_read(x16.get_noc_addr(s, t0 * mp * 64), o16, (t1 - t0) * mp * 64);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 11; k++) {
            const uint32_t c = get_arg_val<uint32_t>(h1_cores + 4 * k + s);
            if (k != n) {
                noc_async_write(o32, get_noc_addr(c & 0xffff, c >> 16, o32), (t1 - t0) * blk);
                noc_async_write(o16, get_noc_addr(c & 0xffff, c >> 16, o16), (t1 - t0) * mp * 64);
            }
        }
        noc_async_write_barrier();
    }
    grid.arrive(1);
    if (h_tile != IDLE && h_tile < D / 32) {
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(x32.get_noc_addr(s, h_tile * blk), r_xown + s * blk, blk);
        }
    }
    const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd, h_item);  // H1's first weights before the first barrier
    uint32_t epoch = 1;
    wait(epoch, 0, 0);
    const fused::hc::RegionsM hreg{r_x32, r_x16, r_xown, r_p1, rs_m, mp};
    fused::hc::read_m(ws, wd, wu, hnorm, h_item, h_tile, hreg, r_hna, r_lo, zero, wait, epoch, h1_pre);

    // ---- P
    if (p_count) {
        fused::x_tiles_m(cb_x, r_hna, KD, KD, mp);
        ws.read(wp, p_first * KD, p_count * KD);
        ws.flush();
    }
    // N's inputs that need no activation, before the barrier. Rows past M repeat row M - 1's position.
    auto row_pos = [&](uint32_t m) { return pos + (m < M ? m : M - 1); };
    // n tiles of a norm weight on rows 0..rows-1 (one page read by every core of the role: read once, copied down)
    auto weights_rows = [&](const Acc& src, uint32_t n, uint32_t rows) {
        cb_reserve_back(cb_nw, n);
        const uint32_t d = get_write_ptr(cb_nw);
        for (uint32_t t = 0; t < n; t++) {
            fused::row0_from_dram(src, 0, t * 128, d + t * 4096);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < n; t++) {
            for (uint32_t m = 1; m < rows; m++) {
                fused::l1_copy(d + t * 4096, d + t * 4096 + m * 64, 64);
                fused::l1_copy(d + t * 4096 + 1024, d + t * 4096 + 1024 + m * 64, 64);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_nw, n);
    };
    // the RoPE table's row 4 t + j on rows 0..rows-1 of one tile: t = row m's position, or `at` for every row
    auto rope_rows = [&](uint32_t j, uint32_t rows, uint32_t at) {
        cb_reserve_back(cb_nr, 1);
        const uint32_t d = get_write_ptr(cb_nr);
        for (uint32_t m = 0; m < rows; m++) {
            const uint32_t t = at == IDLE ? row_pos(m) : at;
            noc_async_read(rope.get_noc_addr(4 * t + j, 0), d + m * 64, 64);
            noc_async_read(rope.get_noc_addr(4 * t + j, 64), d + 1024 + m * 64, 64);
        }
        noc_async_read_barrier();
        cb_push_back(cb_nr, 1);
    };
    // n tiles from their mp-row blocks at L1 src on into cb
    auto push_arena_m = [&](uint32_t cb, uint32_t src, uint32_t n) {
        cb_reserve_back(cb, n);
        const uint32_t d = get_write_ptr(cb);
        for (uint32_t t = 0; t < n; t++) {
            fused::rows_from_l1(src + t * blk, d + t * 4096, mp);
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    // the pooled keys of the score role's blocks index + 64 j, j in [from, to): a block's 4 fp32 tiles at a time
    auto push_pooled = [&](uint32_t from, uint32_t to) {
        for (uint32_t j = from; j < to; j++) {
            cb_reserve_back(cb_npk, 4);
            const uint32_t d = get_write_ptr(cb_npk);
            for (uint32_t t = 0; t < 4; t++) {
                noc_async_read(pooled.get_noc_addr((index + j * NSCORE) * 4 + t), d + t * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb_npk, 4);
        }
    };
#ifdef EXCHANGE
    // a score row's first `pages` pages to the host, a page at a time through two buffers at the start of region PA
    // (the coordinator's; the send two pages back must have left a buffer before it is refilled), word `patch` (the
    // closed block's pool, or none) replaced by `value`
    auto send_pages = [&](fused::Link& link, uint32_t row, uint32_t pages, uint32_t patch, uint32_t value) {
        constexpr uint32_t per = XCHG_OUT_BYTES / 4;
        for (uint32_t p = 0; p < pages; p++) {
            const uint32_t buf = r_pa + (p % 2) * XCHG_OUT_BYTES;
            if (p >= 2) {
                noc_async_write_barrier();
            }
            noc_async_read(scores.get_noc_addr(row, p * XCHG_OUT_BYTES), buf, XCHG_OUT_BYTES);
            noc_async_read_barrier();
            invalidate_l1_cache();
            if (patch / per == p) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf)[patch % per] = value;
            }
            link.send(buf);
        }
    };
#endif
    // the pool role's block: the one row mc closes (at pc = pos + mc) when that row is in the pass, else the token's
    // (computed, not written)
    const uint32_t mc = (3 - pos % 4) % 4, pc = pos + (mc < M ? mc : 0);
    if (role == R_SCORE) {
        // the blocks index + 64 j once a pool of them lies before the last row's block (the compute learns their
        // count from cb_meta); the pooled keys once for all rows, the first two blocks before the barrier
        cb_reserve_back(cb_meta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = score_blocks(index, pos + M - 1);
        cb_push_back(cb_meta, 1);
        weights_rows(iqn, 4, mp);
        rope_rows(0, mp, IDLE);
        rope_rows(1, mp, IDLE);
        push_pooled(0, score_blocks(index, pos + M - 1) < 2 ? 1 : 2);
    } else if (role == R_QHEAD || role == R_KVHEAD) {
        weights_rows(role == R_QHEAD ? qn : kn, 8, mp);
        rope_rows(0, mp, IDLE);
        rope_rows(1, mp, IDLE);
    } else if (role == R_POOL) {
        weights_rows(ikn, 4, 1);  // the new key (row 0): its norm weight, its RoPE rows (cos, sin at pc - 3)
        rope_rows(2, 1, pc);
        rope_rows(3, 1, pc);
        weights_rows(iqn, 4, mp);  // then every row's score of it: the indexer's q heads' weight and rope rows
        rope_rows(0, mp, IDLE);
        rope_rows(1, mp, IDLE);
        // the block's raw keys before pos (from the cache: no activation needed) into the 16 tiles of the 4 key groups
        // (tile 4 t + c: cell pc - 3 + c in row 0; cell 0 stands in below 0); the pass's own cells join them after the
        // barrier, from PA
        cb_reserve_back(cb_nin, 16);
        const uint32_t d = get_write_ptr(cb_nin);
        for (uint32_t c = 0; c < 4; c++) {
            if (pc + c < pos + 3) {
                for (uint32_t t = 0; t < 4; t++) {
                    fused::row0_from_dram(rawk, (pc + c >= 3 ? pc + c - 3 : 0) % RAWK, t * 128, d + (4 * t + c) * 4096);
                }
            }
        }
        noc_async_read_barrier();
    }
    wait(++epoch, r_pa, 436 * blk);

    // ---- N
    if (role == R_SCORE) {
        for (uint32_t h = 0; h < 4; h++) {
            push_arena_m(cb_nin, r_pa + (T_IQ + 4 * h) * blk, 4);
        }
        push_pooled(2, score_blocks(index, pos + M - 1));
    } else if (role == R_QHEAD) {
        push_arena_m(cb_nin, r_pa + 16 * index * blk, 8);
    } else if (role == R_KVHEAD) {
        push_arena_m(cb_nin, r_pa + (T_K + 8 * index) * blk, 8);
        push_arena_m(cb_nv, r_pa + (T_V + 8 * index) * blk, 8);
    } else if (role == R_POOL) {
        // the block's cells from pos on: PA's raw key, row cell - pos, into row 0 of their tiles
        const uint32_t d = get_write_ptr(cb_nin);  // reserved before the barrier, the cache rows in
        for (uint32_t c = 0; c < 4; c++) {
            if (pc + c >= pos + 3) {
                const uint32_t r = pc + c - 3 - pos;
                for (uint32_t t = 0; t < 4; t++) {
                    const uint32_t src = r_pa + (T_IK + t) * blk + r * 64;
                    fused::l1_copy(src, d + (4 * t + c) * 4096, 64);
                    fused::l1_copy(src + mp * 64, d + (4 * t + c) * 4096 + 1024, 64);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_nin, 16);
        for (uint32_t h = 0; h < 4; h++) {  // every row's q heads for the score of the new key (once the keys are in)
            push_arena_m(cb_nin, r_pa + (T_IQ + 4 * h) * blk, 4);
        }
    }
    wait(++epoch, 0, 0);
#ifdef EXCHANGE
    // the host link: row m's scores up to its block (the one-row program's pages at pos + m, in card memory: every
    // writer flushed them before arriving), the closed block's score in place for rows m >= mc; the host's union words
    // back into program 2's selection tensor. Staged in the coordinator's region PA: a page at a time in two buffers
    // (send_pages), the rows' closed-block scores and the reply after them
    if (grid.coordinator) {
        fused::Link link;
        link.open(H2D_CONFIG, D2H_CONFIG, REPLY_BYTES, XCHG_OUT_BYTES);
        const uint32_t reply = r_pa + 2 * XCHG_OUT_BYTES, own = reply + REPLY_BYTES;
        for (uint32_t m = 0; m < M; m++) {
            noc_async_read(scores.get_noc_addr(m, SCORE_BYTES), own + m * 64, 64);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
#ifdef GRID_PROFILE
        fused::Grid::mark(epoch + 1);  // the host's part: the scores out to the selection back
#endif
        for (uint32_t m = 0; m < M; m++) {  // (mc < M: a row of the pass closes block (pos + mc) / 4)
            send_pages(link, m, (pos + m) / 4 * 4 / XCHG_OUT_BYTES + 1, m >= mc ? (pos + mc) / 4 : 0xffffffff,
                       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own + m * 64)[0]);
        }
        link.recv(reply);
#ifdef GRID_PROFILE
        fused::Grid::mark(epoch + 2);
#endif
        noc_async_write(reply, sel_out.get_noc_addr(0), SEL_W * 4);
        noc_async_write_barrier();
        link.close();
    }
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
#endif
