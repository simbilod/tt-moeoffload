// Fused QSA attention layer, program 1 writer (the phases are described in fattn1_reader.cpp).
//   P  row 0 of the core's projection tiles (cb 21) to region PA and PA's check; arrive at barrier 4
//   N  score: the block's 32 scores (cb 30) to the score tensor, and those of blocks index + 64 j once the token is
//      past them (up to CTX/8192: the CB 62 stage, CTX/64 bytes); q head: its 256 values (cb 30) to Q's row h, and
//      its gate's (sigmoid, region PA) to G's row h;
//      kv head: k then v (cb 31, bf16) as the token's cell row of the KV cache (pool-major pages, 4 cells of 512 B),
//      head 0 also the indexer's raw key (region PA) to the raw-key cache; pool: the pooled key (cb 30) as row
//      (block % 32) of block (block / 32)'s 4 fp32 tiles, only when the token closes the block, and that block's
//      score from it after the score tensor's 4096 (word 4096); arrive at barrier 5
// Runtime args: grid.h's block; arena offsets X32, X16, P1, RS, HNA, LO, WA, PA; the checks P1, HN, W, PA; scores, Q,
// K cache, V cache, raw-key cache, pooled cache, G; the H1 item, the H2 tile, P's first tile and count, N's role,
// index, the step tensor (word 0 pos); the pools per KV head of the cache. Compile args: a DRAM accessor.
// M rows (-DM_MAX > 1, fattn1_reader.cpp): every output of rows m < M, the mix's and PA in mp-row blocks; scores row
// m to page m of the score tensor ([M_MAX x (NPOOL + 16)]), the q head's row m (bf16) to Q's page 24 m + h and its
// gate's to G's, k, v and the raw key at cell pos + m, the pooled key of the block row mc closes (when mc < M) and
// every row's score of it (word NPOOL of row m). The writer does not arrive at the first barrier (the reader does).
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"

#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_aout = 21, cb_nout = 30, cb_nout16 = 31, cb_arena = 61, cb_stage = 62;
constexpr uint32_t D = 2560, HD = 256, IHD = 128, T_IK = 432;
constexpr uint32_t R_SCORE = 0, R_QHEAD = 1, R_KVHEAD = 2, R_POOL = 3;
#ifndef NPOOL
#define NPOOL 4096  // g3_attn1.NPOOL (the host's define: CTX / 4)
#endif
#ifndef RAWK
#define RAWK 8  // g3_attn1.RAWK: the raw-key ring's rows (position p at row p % RAWK)
#endif
constexpr uint32_t NSCORE = 64, NBLK_MAX = NPOOL / 32 / NSCORE;  // the score cores, the blocks one takes at most
static_assert(NBLK_MAX >= 1 && NBLK_MAX <= 32, "the stage (cb 62, CTX/64 B) holds the blocks' scores (CTX <= 256K)");
// fattn1_reader.cpp's: the blocks score core `index` takes, up to `last`'s block
inline uint32_t score_blocks(uint32_t index, uint32_t last) {
    uint32_t n = 1;
    while (n < NBLK_MAX && 32 * (index + n * NSCORE) < last / 4) n++;
    return n;
}

#if M_MAX == 1
void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const fused::Grid grid = fused::Grid::from_args(0);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_hna = arena + arg(), r_lo = arena + arg(), r_wa = arena + arg(), r_pa = arena + arg();
    const auto chk_p1 = acc(4 * 352 * 4), chk_hn = acc(D * 4), chk_w = acc(64), chk_pa = acc(436 * 128);
    const auto scores = acc(NPOOL * 4 + 64), qout = acc(HD * 4), kc = acc(2048), vc = acc(2048), rawk = acc(IHD * 4);
    const auto pooled = acc(4096), gout = acc(HD * 4);
    const uint32_t h_item = arg(), h_tile = arg(), p_first = arg(), p_count = arg();
    const uint32_t role = arg(), index = arg();
    const auto step = acc(64);
    const uint32_t npool = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    noc_async_read(step.get_noc_addr(0), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[0];

    uint32_t epoch = 1;
    grid.arrive(epoch);
    const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
    fused::hc::write(grid, stage, chk_p1, chk_hn, chk_w, h_item, h_tile, hreg, r_hna, r_lo, r_wa, epoch);

    // ---- P
    if (p_count) {
        fused::stage_rows(cb_aout, p_count, 64, stage);
        grid.send(stage, r_pa + p_first * 128, p_count * 128);
        noc_async_write(stage, chk_pa.get_noc_addr(0, p_first * 128), p_count * 128);
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);

    // ---- N
    if (role == R_SCORE) {
        const uint32_t nblk = score_blocks(index, pos);  // the reader's
        fused::stage_rows(cb_nout, nblk, 64, stage);
        for (uint32_t b = 0; b < nblk; b++) {
            noc_async_write(stage + b * 128, scores.get_noc_addr(0, (index + b * NSCORE) * 128), 128);
        }
    } else if (role == R_QHEAD) {
        fused::stage_rows(cb_nout, 8, 64, stage);
        noc_async_write(stage, qout.get_noc_addr(index), HD * 4);
        noc_async_write(r_pa + (16 * index + 8) * 128, gout.get_noc_addr(index), HD * 4);  // PA: in the arena by now
    } else if (role == R_KVHEAD) {
        const uint32_t page = index * npool + pos / 4, off = (pos % 4) * HD * 2;
        fused::stage_rows(cb_nout16, 8, 32, stage);
        noc_async_write(stage, kc.get_noc_addr(page, off), HD * 2);
        noc_async_write_barrier();
        fused::stage_rows(cb_nout16, 8, 32, stage);
        noc_async_write(stage, vc.get_noc_addr(page, off), HD * 2);
        if (index == 0) {
            noc_async_write(r_pa + T_IK * 128, rawk.get_noc_addr(pos % RAWK), IHD * 4);
        }
    } else if (role == R_POOL) {
        const uint32_t block = pos / 4, row = block % 32;
        const uint32_t face = row < 16 ? 0 : 2048, r = (row % 16) * 64;
        cb_wait_front(cb_nout, 4);
        const uint32_t t = get_read_ptr(cb_nout);
        for (uint32_t j = 0; j < (pos % 4 == 3 ? 4u : 0u); j++) {
            noc_async_write(t + j * 4096, pooled.get_noc_addr((block / 32) * 4 + j, face + r), 64);
            noc_async_write(t + j * 4096 + 1024, pooled.get_noc_addr((block / 32) * 4 + j, face + 1024 + r), 64);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_nout, 4);
        // the own block's score (row 0, element 0 of the next tile) after the 4096 scores: the host or the link
        // puts it in place of the stale one when the token closes the block
        cb_wait_front(cb_nout, 1);
        noc_async_write(get_read_ptr(cb_nout), scores.get_noc_addr(0, NPOOL * 4), 64);
        noc_async_write_barrier();
        cb_pop_front(cb_nout, 1);
    }
    noc_async_write_barrier();
    grid.arrive(++epoch);
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
#else
void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const fused::Grid grid = fused::Grid::from_args(0);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_xown = arena + arg();
    const uint32_t r_hna = arena + arg(), r_lo = arena + arg();
    arg();  // (WA: the inject weights follow LO at M rows)
    const uint32_t r_pa = arena + arg();
    const auto chk_p1 = acc(4 * 352 * 4), chk_hn = acc(D * 4), chk_w = acc(64), chk_pa = acc(436 * 128);
    const auto scores = acc(NPOOL * 4 + 64), qout = acc(HD * 2), kc = acc(2048), vc = acc(2048), rawk = acc(IHD * 4);
    const auto pooled = acc(4096), gout = acc(HD * 4);
    const uint32_t h_item = arg(), h_tile = arg(), p_first = arg(), p_count = arg();
    const uint32_t role = arg(), index = arg();
    const auto step = acc(64);
    const uint32_t npool = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    noc_async_read(step.get_noc_addr(0), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    const uint32_t pos = sw[0], M = sw[1], mp = sw[4];
    const uint32_t blk = mp * 128;  // an fp32 tile's mp-row block

    uint32_t epoch = 1;  // (the reader arrives at the first barrier)
    const fused::hc::RegionsM hreg{r_x32, r_x16, r_xown, r_p1, r_p1 + mp * 4 * fused::hc::LO * 4, mp};
    fused::hc::write_m(grid, stage, chk_p1, chk_hn, chk_w, h_item, h_tile, hreg, r_hna, r_lo, epoch);

    // ---- P: the core's tiles' mp-row blocks to region PA and its check
    if (p_count) {
        fused::stage_rows_m(cb_aout, p_count, 64, stage, mp);
        grid.send(stage, r_pa + p_first * blk, p_count * blk);
        noc_async_write(stage, chk_pa.get_noc_addr(0, p_first * blk), p_count * blk);
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);

    // ---- N
    // row m of n tiles of cb (bf16) into the 64 n byte row at stage + m * 64 n, for the rows m < M (tile by tile)
    auto stage_rows16 = [&](uint32_t cb, uint32_t n) {
        for (uint32_t t = 0; t < n; t++) {
            cb_wait_front(cb, 1);
            const uint32_t s = get_read_ptr(cb);
            for (uint32_t m = 0; m < M; m++) {
                fused::l1_copy(s + m * 32, stage + m * 64 * n + t * 64, 32);
                fused::l1_copy(s + 512 + m * 32, stage + m * 64 * n + t * 64 + 32, 32);
            }
            noc_async_read_barrier();
            cb_pop_front(cb, 1);
        }
    };
    if (role == R_SCORE) {
        const uint32_t nblk = score_blocks(index, pos + M - 1);  // the reader's
        for (uint32_t b = 0; b < nblk; b++) {
            cb_wait_front(cb_nout, 1);
            const uint32_t s = get_read_ptr(cb_nout), at = (index + b * NSCORE) * 128;
            for (uint32_t m = 0; m < M; m++) {
                noc_async_write(s + m * 64, scores.get_noc_addr(m, at), 64);
                noc_async_write(s + 1024 + m * 64, scores.get_noc_addr(m, at + 64), 64);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_nout, 1);
        }
    } else if (role == R_QHEAD) {
        stage_rows16(cb_nout16, 8);
        for (uint32_t m = 0; m < M; m++) {
            noc_async_write(stage + m * 512, qout.get_noc_addr(m * 24 + index), HD * 2);
            for (uint32_t t = 0; t < 8; t++) {  // the gate's (PA: in the arena by now)
                const uint32_t g = r_pa + (16 * index + 8 + t) * blk + m * 64;
                noc_async_write(g, gout.get_noc_addr(m * 24 + index, t * 128), 64);
                noc_async_write(g + mp * 64, gout.get_noc_addr(m * 24 + index, t * 128 + 64), 64);
            }
        }
    } else if (role == R_KVHEAD) {
        for (uint32_t c = 0; c < 2; c++) {  // k, then v
            stage_rows16(cb_nout16, 8);
            for (uint32_t m = 0; m < M; m++) {
                const uint32_t cell = pos + m;
                const auto& cache = c ? vc : kc;
                const uint64_t at = cache.get_noc_addr(index * npool + cell / 4, (cell % 4) * HD * 2);
                noc_async_write(stage + m * 512, at, HD * 2);
            }
            noc_async_write_barrier();
        }
        if (index == 0) {
            for (uint32_t m = 0; m < M; m++) {
                for (uint32_t t = 0; t < 4; t++) {
                    const uint32_t k = r_pa + (T_IK + t) * blk + m * 64;
                    noc_async_write(k, rawk.get_noc_addr((pos + m) % RAWK, t * 128), 64);
                    noc_async_write(k + mp * 64, rawk.get_noc_addr((pos + m) % RAWK, t * 128 + 64), 64);
                }
            }
        }
    } else if (role == R_POOL) {
        const uint32_t mc = (3 - pos % 4) % 4, close = mc < M ? 1 : 0;  // the reader's
        const uint32_t block = (pos + (close ? mc : 0)) / 4, row = block % 32;
        const uint32_t face = row < 16 ? 0 : 2048, r = (row % 16) * 64;
        cb_wait_front(cb_nout, 4);
        const uint32_t t = get_read_ptr(cb_nout);
        for (uint32_t j = 0; j < 4 * close; j++) {
            noc_async_write(t + j * 4096, pooled.get_noc_addr((block / 32) * 4 + j, face + r), 64);
            noc_async_write(t + j * 4096 + 1024, pooled.get_noc_addr((block / 32) * 4 + j, face + 1024 + r), 64);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_nout, 4);
        // every row's score of it (element (m, 0) of the next tile) after its row's 4096 scores
        cb_wait_front(cb_nout, 1);
        for (uint32_t m = 0; m < M; m++) {
            noc_async_write(get_read_ptr(cb_nout) + m * 64, scores.get_noc_addr(m, NPOOL * 4), 64);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_nout, 1);
    }
    noc_async_write_barrier();
    grid.arrive(++epoch);
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
#endif
