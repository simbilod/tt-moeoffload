// The MTP draft layer's front (compute fmtp_front_compute.cpp, writer fmtp_front_writer.cpp): M <= 4 rows, row
// m = (R_m, the token t_m) with R_m the target's final residual streams (4 x 2560) or the draft layer's own, into the
// draft layer's residual streams
//   X_m,s = fc_hidden (rms_10240(R_m) * w_h)_s + fc_embedding (rms_2560(embed(t_m)) * w_e)       (s = 0..3)
// (tt/spec/mtp.py's front: ONE RMS over the four streams, Strata's default). A tile's rows carry the 16 (stream s, row
// m) pairs at row 4 s + m (rows m >= M copies of row M - 1), so one GEMV pass serves every stream and row; the RMS
// scalars are applied after the GEMV (X = r_m H + q_m E).
//   F1  per k-tile t (80 cores): x_h = R * w_h and x_e = E * w_e (bf16 tiles) and the rows' partial sums of squares
//       (a bf16 row vector each) into the coordinator's copy of cb 1: x tile t holds x_h's 16 rows in rows 0..15 and
//       x_e's in rows 16..31 (both halves used: the barrier broadcasts 86 tiles, 172 KB); SS tiles 80..85, row t
//   F2  per output tile n (80 cores): r_m, q_m from the SS tiles; A = sum_t x_t W_t and B = sum_t x_t W_80+t (W's
//       bfp8_b tiles n-major, page n * 160 + k: rows 0..2559 fc_hidden^T, 2560..5119 fc_embedding^T, read before the
//       barrier), X = r A (rows 0..15) + q B (rows 16..31, folded onto 0..15 on the SFPU); X's tile to the M-row X
//       tensors (fp32 and bf16, a stream a page, mp-row blocks: g3_attn1.x_tensors)
// The small inputs every F1 core reads are spread over card memory's banks (one bank serving 80 cores' reads of the
// same page costs ~4 us a round trip): the step and words tensors in NREP copies (core idx reads copy idx % NREP), R
// and the embedding table in 128- and 64-byte pages.
// Runtime args: grid.h's block; step (uint32 [NREP x 16]: word 1 M, word 4 mp), rin (fp32 [rows x 4 x 80, 32]: row m
// stream s k-tile t at page (4 m + s) 80 + t), the token source: mode (0: words tensor [NREP x 16], word m = row m's
// token; 1: the head's ids tensor, word 0 = the draft's index in the head's rows), its address, the vocabulary map
// (uint32, the head's rows -> token ids; 0: none), the embedding table (bf16 [VOCAB x 80, 32]: token v's k-tile t at
// page 80 v + t), W, the w_h / w_e tiles (fp32, tiles t and 80 + t), F1's item t, F2's tile n, W's tiles read before
// the barrier, this core's copy of the replicated tensors. Compile args: a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_r = 3, cb_e = 4, cb_whe = 5, cb_id = 10, cb_g = 11;
constexpr uint32_t cb_zero = 60, cb_stage = 62;
constexpr uint32_t D = 2560, KT = 160, KX = 80, NX = 86, W_PAGE = 1088, IDLE = 0xffffffff;
constexpr uint32_t CH = 16;  // W's tiles per push (the compute's chunk)

// element (r, c) of a tile: its face offset in elements
inline uint32_t at(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + c % 16; }

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    const auto step = TensorAccessor(d_args, arg(), 64), rin = TensorAccessor(d_args, arg(), 128);
    const uint32_t mode = arg(), src_addr = arg(), vmap_addr = arg();
    const auto emb = TensorAccessor(d_args, arg(), 64), w = TensorAccessor(d_args, arg(), W_PAGE);
    const auto whe = TensorAccessor(d_args, arg(), 4096);
    const uint32_t t = arg(), n = arg(), pre = arg(), rep = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    const uint32_t zero = get_write_ptr(cb_zero), stage = get_write_ptr(cb_stage), x_base = get_write_ptr(cb_x);
    // W's tiles k0 .. k1 - 1 (n-major pages), chunk c = k / CH tagged with transaction id c + 1
    auto read_w = [&](uint32_t k0, uint32_t k1) {
        const uint32_t wt = get_write_ptr(cb_w);
        for (uint32_t k = k0; k < k1; k++) {
            if (k % CH == 0 || k == k0) {
                noc_async_read_set_trid(k / CH + 1);
            }
            noc_async_read(w.get_noc_addr(n * KT + k), wt + k * W_PAGE, W_PAGE);
        }
        noc_async_read_set_trid(0);
    };
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    fused::fill_words(zero, 1024, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
    if (grid.coordinator) {  // the parts of cb 1 no F1 core writes: the SS tiles' other faces
        for (uint32_t k = KX; k < NX; k++) {
            fused::l1_copy(zero, x_base + k * 2048 + 512, 512);    // face 1
            fused::l1_copy(zero, x_base + k * 2048 + 1536, 512);   // face 3
            if ((k - KX) % 3 == 2) {
                fused::l1_copy(zero, x_base + k * 2048 + 1024, 512);  // face 2 of the third tile: rows 80..95
            }
        }
        noc_async_read_barrier();
    }

    noc_async_read(step.get_noc_addr(rep), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    const uint32_t M = sw[1];

    // ---- F1: R's and E's 16 rows of k-tile t, the two weight tiles
    if (t != IDLE) {
        uint32_t tok[4];
        noc_async_read(TensorAccessor(d_args, src_addr, 64).get_noc_addr(mode == 0 ? rep : 0), stage, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        asm volatile("fence rw, rw" ::: "memory");
        for (uint32_t m = 0; m < 4; m++) {
            tok[m] = sw[mode == 0 ? (m < M ? m : M - 1) : 0];
        }
        if (mode == 1 && vmap_addr) {  // the head's row -> its token id (a 64-byte read around the word)
            const auto vmap = TensorAccessor(d_args, vmap_addr, 64);
            noc_async_read(vmap.get_noc_addr(tok[0] / 16), stage, 64);
            noc_async_read_barrier();
            invalidate_l1_cache();
            asm volatile("fence rw, rw" ::: "memory");
            const uint32_t id = sw[tok[0] % 16];
            for (uint32_t m = 0; m < 4; m++) {
                tok[m] = id;
            }
        }
        // R: row 4 s + m <- 32 floats of R_min(m, M-1), stream s, at k-tile t (128-byte reads, then the two faces)
        for (uint32_t s = 0; s < 4; s++) {
            for (uint32_t m = 0; m < 4; m++) {
                noc_async_read(rin.get_noc_addr((4 * (m < M ? m : M - 1) + s) * KX + t), stage + (4 * s + m) * 128, 128);
            }
        }
        for (uint32_t m = 0; m < 4; m++) {  // E: the 32 bf16 of row m's embedding at k-tile t
            noc_async_read(emb.get_noc_addr(tok[m] * KX + t), stage + 2048 + m * 64, 64);
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_r, 1);
        cb_reserve_back(cb_e, 1);
        const uint32_t rt = get_write_ptr(cb_r), et = get_write_ptr(cb_e);
        fused::l1_copy(zero, rt + 2048, 2048);
        fused::l1_copy(zero, et + 1024, 1024);
        for (uint32_t r = 0; r < 16; r++) {
            fused::l1_copy(stage + r * 128, rt + r * 64, 64);
            fused::l1_copy(stage + r * 128 + 64, rt + 1024 + r * 64, 64);
            fused::l1_copy(stage + 2048 + (r % 4) * 64, et + r * 32, 32);
            fused::l1_copy(stage + 2048 + (r % 4) * 64 + 32, et + 512 + r * 32, 32);
        }
        cb_reserve_back(cb_whe, 2);
        noc_async_read(whe.get_noc_addr(t), get_write_ptr(cb_whe), 4096);
        noc_async_read(whe.get_noc_addr(80 + t), get_write_ptr(cb_whe) + 4096, 4096);
        noc_async_read_barrier();
        cb_push_back(cb_r, 1);
        cb_push_back(cb_e, 1);
        cb_push_back(cb_whe, 2);
    }

    // ---- F2's inputs: the identity and group tiles, W's 160 tiles (issued before the barrier, pushed after it)
    if (n != IDLE) {
        cb_reserve_back(cb_id, 1);
        cb_reserve_back(cb_g, 1);
        const uint32_t id = get_write_ptr(cb_id), gt = get_write_ptr(cb_g);
        fused::l1_copy(zero, id, 4096);
        fused::l1_copy(zero, gt, 2048);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* idw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(id);
        volatile tt_l1_ptr uint16_t* gw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gt);
        for (uint32_t r = 0; r < 32; r++) {
            idw[at(r, r)] = 0x3f800000;
        }
        for (uint32_t r = 0; r < 16; r++) {  // G[r][j] = 1 where r and j are the same row m (r % 4 == j % 4)
            for (uint32_t j = r % 4; j < 16; j += 4) {
                gw[at(r, j)] = 0x3f80;
            }
        }
        asm volatile("fence rw, rw" ::: "memory");
        cb_push_back(cb_id, 1);
        cb_push_back(cb_g, 1);
        cb_reserve_back(cb_w, KT);
        read_w(0, pre);
    }
    grid.wait(1, x_base, NX * 2048);
    if (n != IDLE) {
        cb_reserve_back(cb_x, NX);
        cb_push_back(cb_x, NX);
        read_w(pre, KT);
        for (uint32_t c = 0; c < KT / CH; c++) {  // each chunk to the compute as it lands
            noc_async_read_barrier_with_trid(c + 1);
            cb_push_back(cb_w, CH);
        }
    }
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the reader's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
