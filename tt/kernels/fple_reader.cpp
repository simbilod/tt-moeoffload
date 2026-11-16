// Fused PLE block (G3 step 10): Flash-Next's per-layer embedding block, at the start of layer 1, for m decode tokens
// at positions pos..pos+m-1 (llama.cpp's qwen4exp build_ple; m > 1: a verify pass's rows), on the whole grid (compute
// fple_compute.cpp, writer fple_writer.cpp). The host gathers each token's embedding emb (16 heads x 160 of the hashed
// n-gram table, 2560).
//   key = W_k emb (2560 -> 10240: 4 streams), value = W_v emb (2560 -> 2560)
//   s_c = sum_i rms_norm(key_c)_i nk_ci rms_norm(X_c)_i nq_ci / sqrt(2560), gate_c = sigmoid(sign(s) sqrt(max(|s|, 1e-6)))
//   gated_c = value gate_c, n_c = rms_norm(gated_c) nc_c, conv_c = silu(sum_k w_k n_c[t - 3 (3 - k)]) (dilated by 3)
//   X_c += gated_c + conv_c
// in three phases, row r of every tile token pos + r's (the tile math is per element, so the compute is the same for
// any m; rows m..31 carry garbage that reaches only themselves):
//   P1  key|value's 400 tiles (runs of <= 4 per core), each key tile's epilogue the partial sums A = sum k nk X nq and
//       B = sum k^2, each value tile's V = sum v^2 and the tile itself (region PV, a row per token); the sums to region
//       PS (per token 720 16-byte slots: A 0..319, B 320..639, V 640..719)
//   P2  item (r, c) = 4 r + c on core 4 r + c: s_c of token r from the sums (gathered into tiles, summed through two
//       matmuls with the ones tile, as H1 sums X_c^2 for rms(X_c)), gate_c and g_c = gate_c / rms(gated_c); to region
//       PG (token r's 128 bytes: gate at 32 c, g at 32 c + 16)
//   P3  tile (c, j) (320, runs per core): gated = v gate_c, n = v g_c nc, the conv over the history, silu,
//       X += gated + conv; X back into its regions
// The history ring: the last tokens' n in card memory, [RING x 10240], token q's at row q % RING. Row r reads tokens
// q = t - 9, t - 6, t - 3 (t = pos + r): from the ring for q < pos, else this pass's row q - pos (the writer hands it
// over through cb 27: the row shift, m > 3 only); the writer writes row r's n to row t % RING. RING = 9 + M_RING (17):
// a pass writes tokens pos..pos+m-1 and reads >= pos - 9; after rejected rows the next pass starts at p_keep > pos and
// reads >= p_keep - 9, while the rejected rows wrote < pos + m <= p_keep + M_RING - 1, so neither meets a slot it needs
// for m <= M_RING (and no ring row is read and written in the same pass, so its reads need no ordering).
// Runtime args: X fp32, X bf16 ([m x 4 x 2560] each), emb (bf16, [m x 2560]); grid.h's block; arena offsets X32, X16,
// EMB, PV, PS, PG; W (key|value), first tile, count; P2's item (0xffffffff none); P3's first tile, count; norms nk, nq,
// nc ([10240] each), taps ([4 x 10240]), the history ring, the step tensor (word 0: pos); the stamps tensor;
// [GRID_PROFILE: the profile]; m. Compile args: batch, then a DRAM accessor.
// X in the M-row layout (-DX_BLOCKS=n: a verify pass between DeltaNet programs at M_MAX = n, flayer_reader.cpp): the
// X tensors [4 x n 2560] a stream a page, each tile a block of its rows (fused/hc_dm.h, Mp = 1, 2 or 4 rows); m and
// Mp from the step tensor (words 1 and 4; the argument m only sizes P2's items: an item of a row >= m gets stand-in
// inputs). X's blocks go to the arena as they are (no X bf16: no phase reads it) with the rows' embeddings after
// them, P1 and P3 take a tile's rows with one copy per face, P2 a row's pieces; X goes back in blocks (rows m..Mp-1
// copies of row m - 1, as the DeltaNet program leaves them), with no gather at the last barrier. At Mp = 1 the blocks
// are the token rows.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_pc = 3, cb_qd = 8, cb_qs = 11;
constexpr uint32_t cb_rv = 14, cb_rg = 15, cb_rnc = 16, cb_rw = 17, cb_rh = 18, cb_rx = 19, cb_shift = 27;
constexpr uint32_t cb_zero = 60, cb_arena = 61, cb_prof = 63;
constexpr uint32_t W_PAGE = 1088, D = 2560, CH = 10240, KD = 80, NKEY = 320, IDLE = 0xffffffff;
constexpr uint32_t RING = 17, XROW = 4 * D * 4, X16ROW = 4 * D * 2, PSROW = 720 * 16;
constexpr uint32_t T1 = XROW + X16ROW + D * 2, T2 = D * 4 + PSROW;  // a token's arena blocks: X32 X16 EMB, PV PS

inline uint32_t elem(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 1024 + (r % 16) * 64 + (c % 16) * 4; }

// row r (< 16) of an fp32 tile at dst from 32 consecutive floats at L1 src / of page `page` of a row-major tensor
inline void row_from_l1(uint32_t src, uint32_t dst, uint32_t r) {
    fused::l1_copy(src, dst + r * 64, 64);
    fused::l1_copy(src + 64, dst + 1024 + r * 64, 64);
}
template <typename Accessor>
inline void row_from_dram(const Accessor& src, uint32_t page, uint32_t off, uint32_t dst, uint32_t r) {
    noc_async_read(src.get_noc_addr(page, off), dst + r * 64, 64);
    noc_async_read(src.get_noc_addr(page, off + 64), dst + 1024 + r * 64, 64);
}
// the float bits at L1 addr across row r of the fp32 tile at dst (fused::fill_row0 for any row)
inline void fill_row(uint32_t dst, uint32_t r, uint32_t addr) {
    invalidate_l1_cache();
    const uint32_t bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + r * 64);
    for (uint32_t j = 0; j < 16; j++) {
        t[j] = bits;
        t[256 + j] = bits;
    }
    asm volatile("fence rw, rw" ::: "memory");
}

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    using Acc = decltype(TensorAccessor(d_args, 0u, 0u));
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
#ifdef X_BLOCKS
    const auto x32 = acc(X_BLOCKS * D * 4), x16 = acc(X_BLOCKS * D * 2), emb = acc(D * 2);  // a stream a page
#else
    const auto x32 = acc(XROW), x16 = acc(X16ROW), emb = acc(D * 2);
#endif
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_emb = arena + arg(), r_pv = arena + arg();
    const uint32_t r_ps = arena + arg(), r_pg = arena + arg();
    const auto w = acc(W_PAGE);
    const uint32_t p1_first = arg(), p1_count = arg(), item = arg(), p3_first = arg(), p3_count = arg();
    const auto nk = acc(CH * 4), nq = acc(CH * 4), nc = acc(CH * 4), taps = acc(CH * 4), ring = acc(CH * 4);
    const auto step = acc(64);  // the token's step tensor: word 0 its position
    const auto prof = acc(64);
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
#ifdef X_BLOCKS
    arg();  // m: the step tensor's (P2's items are the host's for the most rows)
    uint32_t m, mp;
#else
    const uint32_t m = arg();
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    const uint32_t zero = get_write_ptr(cb_zero);
    fused::fill_words(zero, 1024, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_prof));
    noc_async_read(step.get_noc_addr(0), get_write_ptr(cb_prof), 64);  // (the profile's words take it over next)
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t pos = stamps[0];
#ifdef X_BLOCKS
    m = stamps[1], mp = stamps[4];  // before the stamps take the words over
    const uint32_t rx_emb = r_x32 + mp * XROW;  // the rows' embeddings after X's blocks
#endif
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        stamps[n] = get_timestamp_32b();
    };
#ifdef X_BLOCKS
    if (grid.coordinator) {  // stream s's blocks to X32 + s Mp 10240, then the embeddings
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(x32.get_noc_addr(s), r_x32 + s * mp * D * 4, mp * D * 4);
        }
        for (uint32_t r = 0; r < m; r++) {
            noc_async_read(emb.get_noc_addr(r), rx_emb + r * D * 2, D * 2);
        }
        noc_async_read_barrier();
    }
    uint32_t epoch = 1;
    wait(epoch, r_x32, mp * XROW + m * D * 2);
#else
    if (grid.coordinator) {
        for (uint32_t r = 0; r < m; r++) {  // whole rows: 32-byte reads from card memory misread
            noc_async_read(x32.get_noc_addr(r), r_x32 + r * T1, XROW);
            noc_async_read(x16.get_noc_addr(r), r_x16 + r * T1, X16ROW);
            noc_async_read(emb.get_noc_addr(r), r_emb + r * T1, D * 2);
        }
        noc_async_read_barrier();
    }
    uint32_t epoch = 1;
    wait(epoch, r_x32, m * T1);
#endif

    // ---- P1: emb's x tiles (row r token r's), then per output tile its weights and (key tiles) X_c, nk, nq
    if (p1_count) {
        cb_reserve_back(cb_x, KD);
        const uint32_t xt = get_write_ptr(cb_x);
        for (uint32_t k = 0; k < KD; k++) {  // row r of a bf16 tile: 32 bytes at r * 32 (face 0), 512 + r * 32 (face 1)
            for (uint32_t r = 0; r < m; r++) {
#ifdef X_BLOCKS
                fused::l1_copy(rx_emb + r * D * 2 + k * 64, xt + k * 2048 + r * 32, 32);
                fused::l1_copy(rx_emb + r * D * 2 + k * 64 + 32, xt + k * 2048 + 512 + r * 32, 32);
#else
                fused::l1_copy(r_emb + r * T1 + k * 64, xt + k * 2048 + r * 32, 32);
                fused::l1_copy(r_emb + r * T1 + k * 64 + 32, xt + k * 2048 + 512 + r * 32, 32);
#endif
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, KD);
        for (uint32_t n = p1_first; n < p1_first + p1_count; n++) {
            ws.read(w, n * KD, KD);
            if (n < NKEY) {
                cb_reserve_back(cb_pc, 3);
                const uint32_t d = get_write_ptr(cb_pc);
#ifdef X_BLOCKS
                fused::l1_copy(r_x32 + n * mp * 128, d, mp * 64);  // tile n's block (stream n / 80): rows by face
                fused::l1_copy(r_x32 + n * mp * 128 + mp * 64, d + 1024, mp * 64);
#endif
                for (uint32_t r = 0; r < m; r++) {
#ifndef X_BLOCKS
                    row_from_l1(r_x32 + r * T1 + n * 128, d, r);  // stream n / 80, channel tile n % 80
#endif
                    row_from_dram(nk, 0, n * 128, d + 4096, r);
                    row_from_dram(nq, 0, n * 128, d + 8192, r);
                }
                noc_async_read_barrier();
                cb_push_back(cb_pc, 3);
            }
        }
        ws.flush();
    }
    wait(++epoch, r_pv, m * T2);

    // ---- P2: X_c's dense tiles (rms), the sums gathered into tiles (A, B, V: element j at (j / 32, j % 32))
    if (item != IDLE) {
#ifdef X_BLOCKS
        // row r's pieces of stream c's blocks, in order (a row >= Mp: another row's, a stand-in nothing reads)
        const uint32_t r = item / 4, c = item % 4, xr = r_x32 + c * mp * D * 4 + (r % mp) * 64, ps = r_ps + r * T2;
        cb_reserve_back(cb_qd, 3);
        const uint32_t d = get_write_ptr(cb_qd);
        for (uint32_t t = 0; t < D / 32; t++) {
            fused::l1_copy(xr + t * mp * 128, d + t * 128, 64);
            fused::l1_copy(xr + t * mp * 128 + mp * 64, d + t * 128 + 64, 64);
        }
#else
        const uint32_t r = item / 4, c = item % 4, xr = r_x32 + r * T1, ps = r_ps + r * T2;
        cb_reserve_back(cb_qd, 3);
        const uint32_t d = get_write_ptr(cb_qd);
        fused::l1_copy(xr + c * D * 4, d, 8192);
        fused::l1_copy(xr + c * D * 4 + 8192, d + 8192, 2048);
#endif
        fused::l1_copy(zero, d + 10240, 2048);
        noc_async_read_barrier();
        cb_push_back(cb_qd, 3);
        cb_reserve_back(cb_qs, 3);
        const uint32_t q = get_write_ptr(cb_qs);
        for (uint32_t t = 0; t < 3; t++) {
            fused::fill_words(q + t * 4096, 1024, 0);
        }
        invalidate_l1_cache();
        for (uint32_t j = 0; j < 80; j++) {
            const uint32_t at = elem(j / 32, j % 32);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q + at) =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps + (c * 80 + j) * 16);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q + 4096 + at) =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps + (NKEY + c * 80 + j) * 16);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(q + 8192 + at) =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps + (2 * NKEY + j) * 16);
        }
        cb_push_back(cb_qs, 3);
    }
    wait(++epoch, r_pg, 128 * m);

    // ---- P3: per tile (c, j) = 80 c + j its inputs, in the compute's order (with m > 3 the history after the first
    // pass's inputs: its rows from this pass come from that pass's n)
    for (uint32_t n = p3_first; n < p3_first + p3_count; n++) {
        const uint32_t c = n / 80, j = n % 80;
        cb_reserve_back(cb_rh, 3);  // t - 9, t - 6, t - 3: the conv's oldest first; the ring's rows issued now
        const uint32_t h = get_write_ptr(cb_rh);
        bool shifted = false;
        for (uint32_t k = 0; k < 3; k++) {
            for (uint32_t r = 0; r < m; r++) {
                if (r + 3 * k < 9) {
                    row_from_dram(ring, (pos + r + RING - 9 + 3 * k) % RING, n * 128, h + k * 4096, r);
                } else {
                    shifted = true;
                }
            }
        }
        if (!shifted) {  // all from the ring (m < 4): first, as the compute's order has them earliest
            noc_async_read_barrier();
            cb_push_back(cb_rh, 3);
        }
        cb_reserve_back(cb_rv, 1);
        for (uint32_t r = 0; r < m; r++) {
            row_from_l1(r_pv + r * T2 + j * 128, get_write_ptr(cb_rv), r);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rv, 1);
        cb_reserve_back(cb_rg, 2);
        for (uint32_t r = 0; r < m; r++) {
            fill_row(get_write_ptr(cb_rg), r, r_pg + r * 128 + c * 32);
            fill_row(get_write_ptr(cb_rg) + 4096, r, r_pg + r * 128 + c * 32 + 16);
        }
        cb_push_back(cb_rg, 2);
        cb_reserve_back(cb_rnc, 1);
        for (uint32_t r = 0; r < m; r++) {
            row_from_dram(nc, 0, n * 128, get_write_ptr(cb_rnc), r);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rnc, 1);
        if (shifted) {  // this pass's n of the tile (the writer's rows, 128 bytes each), row q - pos to row r
            cb_wait_front(cb_shift, 1);
            const uint32_t s = get_read_ptr(cb_shift);
            for (uint32_t k = 0; k < 3; k++) {
                for (uint32_t r = 0; r < m; r++) {
                    if (r + 3 * k >= 9) {
                        row_from_l1(s + (r + 3 * k - 9) * 128, h + k * 4096, r);
                    }
                }
            }
            noc_async_read_barrier();
            cb_pop_front(cb_shift, 1);
            cb_push_back(cb_rh, 3);
        }
        cb_reserve_back(cb_rw, 4);
        for (uint32_t k = 0; k < 4; k++) {
            for (uint32_t r = 0; r < m; r++) {
                row_from_dram(taps, k, n * 128, get_write_ptr(cb_rw) + k * 4096, r);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_rw, 4);
        cb_reserve_back(cb_rx, 1);
#ifdef X_BLOCKS
        fused::l1_copy(r_x32 + n * mp * 128, get_write_ptr(cb_rx), mp * 64);
        fused::l1_copy(r_x32 + n * mp * 128 + mp * 64, get_write_ptr(cb_rx) + 1024, mp * 64);
#else
        for (uint32_t r = 0; r < m; r++) {
            row_from_l1(r_x32 + r * T1 + n * 128, get_write_ptr(cb_rx), r);
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_rx, 1);
    }
#ifdef X_BLOCKS
    wait(++epoch, 0, 0);
#else
    wait(++epoch, r_x32, (m - 1) * T1 + XROW + X16ROW);  // X's rows
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
