// GLM's fused MLA query side, reader (compute glm_fmq_compute.cpp, writer glm_fmq_writer.cpp, host
// tt/glm/g3_fmq.py): hn -> the attention's inputs, in three phases on the whole grid with a barrier after each
// (fused/grid.h), the last one carrying the host's top 512:
//
// A  y = hn W_A (4096 -> 2336: q_a 1536 | kv_a 512 | i_proj 32 (scaled 1 / sqrt(4096)) | i_k 128 | i_gate 128; 73
//    tiles, n-major pages 128 n + k). The writer sends q_a as bf16 with each tile's sum of squares, the rest as fp32,
//    to region R1: [q_a bf16 1536 | squares 48 x 16 B | the other 25 tiles fp32]
// B  y = q_a W_B (1536 -> 20480: q_b 16384 | i_q_b 4096, the q_a norm's weight folded in: diag(w) W; pages 48 n + k),
//    times r = rsqrt(mean q_a^2 + 1e-5), which every core takes from the 48 squares (this RISC-V's FPU): qr W =
//    r (q_a (diag(w) W)). The outputs (bf16) to region R2: [q 16384 | iq 4096]
// C  on disjoint cores: the absorb, q_abs_h = q_h k_b_h (256 -> 512; head h on core h, k_b_h's 128 tiles at pages
//    128 h + 8 n + k), its 16 output rows written into row h % 32 of group h / 32's flash-MLA input tiles; and the
//    indexer's pool scores (glm_idx_compute.cpp's: the pooled keys K [n_pool, 128] at pages 4 pt + d, iq as 4 tiles
//    of heads x dims built here and transposed by the compute, w from R1) to region SC
// 3  the coordinator appends R1's last 25 tiles (kv_a | i_proj | i_k | i_gate, fp32) and iq (bf16) to the scores,
//    sends the page to the host over D2H and takes the reply back over H2D (glm_loop.cpp's mla_step): the selected
//    token ids to sel (the gather's), cur_pos to cpos (the attention's), the normed latent to row p of lat, and the
//    pooled key of the pool p closes into its row of K (p, the pool: 0xffffffff for none)
//
// x travels as mgemv's ring; each phase's first weight batches are read before the barrier that precedes it.
// Runtime args: grid.h's block, then hn, W_A, W_B, k_b, K, sel, cpos, lat addresses, A's first tile and count, B's first tile and
// count, the head (0xffffffff: none), the first pool tile and count, the arena offsets of X (hn), R1, R2, SC, REPLY,
// the prefetch buffer's L1 address and page count (A's first npf pages, read there by the previous layer's FFN program
// while the lane worked; glm_fsh_reader.cpp; 0: none).
// Compile args: batch, depth, n_pool, then a DRAM accessor. Defines: H2D_CONFIG, D2H_CONFIG.
//
// M rows (-DM_MAX = 2..4; M_MAX = 1 compiles the one-row source): M and the layout's rows mp from the step
// tensor (a runtime arg after the prefetch's; tt/fused/mrows.py), vectors as fused/hc_dm.h's mp-row blocks: hn (X),
// q_a (R1, then each tile's squares per row in M_MAX slots of 16 bytes, then the other 25 tiles fp32), q_b | iq (R2).
// B's r per row; C's absorb at mp rows into row h % 32 of tile (m, ct) of the query groups [1, M_MAX, 32, 512]; the
// scores per row (iq and w of each of M_MAX rows; the compute loops over M_MAX). The host's page is M_MAX row pages
// (SC + m PAGE: the row's scores, then its A2 row and iq row, which A's and B's writers put there), the reply M_MAX
// row replies (glm_loop.cpp's mla_step per row): row m's selection to sel's row m, its latent row and closed pool's
// key to lat and K, the M cur_pos to cpos [M_MAX] in one write.
#include <cstdint>
#ifndef M_MAX
#define M_MAX 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#include "fused/fpu.h"
#include "fused/link.h"
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_r = 2, cb_k = 3, cb_iq = 4, cb_wt = 5, cb_arena = 30, cb_stage = 31;
constexpr uint32_t W_PAGE = 1088, KA = 128, KB = 48, KH = 8, NQA = 48, NA = 73, NB = 640, IDLE = 0xffffffff;
constexpr uint32_t SQ_OFF = NQA * 64, A2_OFF = SQ_OFF + NQA * 16;      // in R1
constexpr uint32_t IQ_OFF = 512 * 64;                                // in R2
constexpr uint32_t T_PROJ = 64, A2_BYTES = (NA - NQA) * 128, IQ_BYTES = 4096 * 2;  // A's tile i_proj; the page's tail
// the reply (glm_loop.cpp's): the selection's NS token ids, the latent row, cur_pos | p | pool, the pooled key's 8
// pieces (64 bytes each, at their destination's alignment)
constexpr uint32_t NS = 2112, LAT_OFF = NS * 4, CTL_OFF = LAT_OFF + 1024, PK_OFF = CTL_OFF + 64, RB = PK_OFF + 512;

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
inline volatile tt_l1_ptr uint16_t* w16(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a); }
inline volatile tt_l1_ptr float* f32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); }

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t n_pool = get_compile_time_arg_val(2);
    constexpr uint32_t PAGE = n_pool * 4 + A2_BYTES + IQ_BYTES;
    constexpr auto d_args = TensorAccessorArgs<3>();
    const fused::Grid grid = fused::Grid::from_args(0);
    uint32_t i = fused::Grid::args(grid.nrects);
    const auto hn = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), KA * 64);
    const auto wa = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto wb = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto kb = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto K = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 2048);
    const auto sel = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), NS * 4);
    const auto cpos = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 4);
    const auto lat = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 1024);
    const uint32_t a0 = get_arg_val<uint32_t>(i++), na = get_arg_val<uint32_t>(i++);
    const uint32_t b0 = get_arg_val<uint32_t>(i++), nb = get_arg_val<uint32_t>(i++);
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const uint32_t p0 = get_arg_val<uint32_t>(i++), np = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t X = arena + get_arg_val<uint32_t>(i++), R1 = arena + get_arg_val<uint32_t>(i++);
    const uint32_t R2 = arena + get_arg_val<uint32_t>(i++), SC = arena + get_arg_val<uint32_t>(i++);
    const uint32_t REPLY = arena + get_arg_val<uint32_t>(i++);
    const uint32_t pf = get_arg_val<uint32_t>(i++), npf = get_arg_val<uint32_t>(i++);
#if M_MAX > 1
    const auto step = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 64);
    noc_async_read(step.get_noc_addr(0), get_write_ptr(cb_stage), 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t M = w32(get_write_ptr(cb_stage))[1], mp = w32(get_write_ptr(cb_stage))[4];
    constexpr uint32_t SQ_M = NQA * M_MAX * 64, A2_M = SQ_M + NQA * M_MAX * 16;   // R1 at M rows
    const uint32_t IQT = arena + get_arg_val<uint32_t>(i++);   // iq's heads x dims tiles per row (B's writers')
    auto xr = [&](uint32_t row, uint32_t k0) { fused::x_ring_m_issue(cb_x, row, k0, batch, mp); };
    auto xpush = [&] { cb_push_back(cb_x, batch); };   // after the batch's read barrier (fused::Stream's)
#else
    auto xr = [&](uint32_t row, uint32_t k0) { fused::x_ring(cb_x, row, k0, batch); };
    auto xpush = [] {};
#endif

    fused::Stream<decltype(wa)> ws{cb_w, batch, W_PAGE};   // the weight stream (the open batch persists)
    // a phase: the weights of tiles [t0, t0 + n) (kt pages each, from page base) with their x batches from row; the
    // first `pre` pages were read before the barrier
    auto phase = [&](const auto& src, uint32_t base, uint32_t t0, uint32_t n, uint32_t kt, uint32_t row,
                     uint32_t pre) {
        for (uint32_t p = 0; p < n * kt; p += batch) {
            xr(row, p % kt);
            if (p >= pre) {
                ws.read(src, base + t0 * kt + p, batch);
            } else if (M_MAX > 1) {
                noc_async_read_barrier();
            }
            xpush();
        }
    };
    auto pre_of = [&](uint32_t pages) { return pages < depth * batch ? pages : depth * batch; };

    // ---- A
#if M_MAX > 1
    noc_async_read(hn.get_noc_addr(0), X, mp * KA * 64);
#else
    noc_async_read(hn.get_noc_addr(0), X, KA * 64);
#endif
    noc_async_read_barrier();
    for (uint32_t p = 0; p < na * KA; p += batch) {          // phase(wa, .., pre 0), the first npf pages from L1
        xr(X, p % KA);
        for (uint32_t j = 0; j < batch; j++) {
            if (p + j < npf) ws.read_page(get_noc_addr(pf + (p + j) * W_PAGE));
            else ws.read(wa, a0 * KA + p + j, 1);
        }
        xpush();
    }
    const uint32_t pre_b = pre_of(nb * KB);
    ws.read(wb, b0 * KB, pre_b);
#if M_MAX > 1
    grid.wait(1, R1, A2_M + (NA - NQA) * mp * 128);
#else
    grid.wait(1, R1, A2_OFF + (NA - NQA) * 128);
#endif

    // ---- B: r from the squares, across row 0 of an fp32 tile
    invalidate_l1_cache();
    if (nb) {
#if M_MAX > 1
        const uint32_t st = get_write_ptr(cb_stage);   // r per row, down rows 0..mp-1
        cb_reserve_back(cb_r, 1);
        for (uint32_t m = 0; m < mp; m++) {
            float ss = 0.0f;
            for (uint32_t t = 0; t < NQA; t++) ss = fused::fpu::add(ss, f32(R1 + SQ_M + (t * M_MAX + m) * 16)[0]);
            f32(st + m * 16)[0] = fused::fpu::rsqrt(fused::fpu::add(fused::fpu::mul(ss, 1.0f / (NQA * 32)), 1e-5f));
            fused::fill_row(get_write_ptr(cb_r), m, st + m * 16);
        }
        cb_push_back(cb_r, 1);
#else
        float ss = 0.0f;
        for (uint32_t t = 0; t < NQA; t++) ss = fused::fpu::add(ss, f32(R1 + SQ_OFF + t * 16)[0]);
        const uint32_t st = get_write_ptr(cb_stage);
        f32(st)[0] = fused::fpu::rsqrt(fused::fpu::add(fused::fpu::mul(ss, 1.0f / (NQA * 32)), 1e-5f));
        cb_reserve_back(cb_r, 1);
        fused::fill_row0(get_write_ptr(cb_r), st);
        cb_push_back(cb_r, 1);
#endif
    }
    phase(wb, 0, b0, nb, KB, R1, pre_b);
    // C's inputs that need no activation: the absorb's first pages, the first pool tiles' keys
    const uint32_t pre_c = head != IDLE ? pre_of(16 * KH) : 0;
    if (head != IDLE) ws.read(kb, head * 16 * KH, pre_c);
    const uint32_t k_pre = np < 2 ? np : 2;
    auto keys = [&](uint32_t t) {
        cb_reserve_back(cb_k, 4);
        for (uint32_t d = 0; d < 4; d++) noc_async_read(K.get_noc_addr(4 * t + d), get_write_ptr(cb_k) + d * 2048, 2048);
        noc_async_read_barrier();
        cb_push_back(cb_k, 4);
    };
    for (uint32_t t = p0; t < p0 + k_pre; t++) keys(t);
#if M_MAX > 1
    grid.wait(2, IQT, 4 * M_MAX * 2048 + NB * mp * 64);   // IQT, then R2 (glm_fmq_writer.cpp)
#else
    grid.wait(2, R2, IQ_OFF + 128 * 64);
#endif

    // ---- C
    if (head != IDLE) {
        for (uint32_t p = 0; p < 16 * KH; p += batch) {
#if M_MAX > 1
            xr(R2 + head * KH * mp * 64, p % KH);
#else
            xr(R2 + head * KH * 64, p % KH);
#endif
            if (p >= pre_c) {
                ws.read(kb, head * 16 * KH + p, batch);
            } else if (M_MAX > 1) {
                noc_async_read_barrier();
            }
            xpush();
        }
    }
#if M_MAX > 1
    if (np) {
        // per row m < M: iq as 4 tiles of heads x dims (tile 4 m + d: row h = iq_m[128 h + 32 d ..]), multicast into
        // region IQT by B's writers; w_m in row 0 of tile m. Rows past M are left as they are (their scores are not
        // sent)
        invalidate_l1_cache();
        cb_reserve_back(cb_iq, 4 * M_MAX);
        cb_reserve_back(cb_wt, M_MAX);
        const uint32_t wt = get_write_ptr(cb_wt);
        fused::l1_copy(IQT, get_write_ptr(cb_iq), 4 * M * 2048);
        fused::fill_words(wt, 512 * M, 0);
        for (uint32_t m = 0; m < M; m++) {   // w: i_proj's 32 fp32 values of row m (R1's A2 block 16), as bf16
            const uint32_t blk = R1 + A2_M + (T_PROJ - NQA) * mp * 128 + m * 64;
            for (uint32_t c = 0; c < 32; c++) {
                const uint32_t b = w32(blk + (c < 16 ? 0 : mp * 64))[c & 15];
                w16(wt + m * 2048 + (c < 16 ? c * 2 : 512 + (c - 16) * 2))[0] =
                    static_cast<uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
            }
        }
        noc_async_read_barrier();
        asm volatile("fence rw, rw" ::: "memory");
        cb_push_back(cb_iq, 4 * M_MAX);
        cb_push_back(cb_wt, M_MAX);
        for (uint32_t t = p0 + k_pre; t < p0 + np; t++) keys(t);
    }
    grid.wait(3, 0, 0, [&] {
        // the row pages are in place (A's and B's writers put each row's A2 and iq there)
        fused::Link link;
        link.open(H2D_CONFIG, D2H_CONFIG, RB * M_MAX, PAGE * M_MAX);
        link.send(SC);
        link.recv(REPLY);
        link.close();
        invalidate_l1_cache();
        const uint32_t cp = REPLY + RB * M_MAX;   // the M cur_pos, gathered for one write
        for (uint32_t m = 0; m < M; m++) {
            const uint32_t rp = REPLY + m * RB;
            w32(cp)[m] = w32(rp + CTL_OFF)[0];
            const uint32_t p = w32(rp + CTL_OFF)[1], pool = w32(rp + CTL_OFF)[2];
            noc_async_write(rp, sel.get_noc_addr(m), NS * 4);
            if (p != IDLE) noc_async_write(rp + LAT_OFF, lat.get_noc_addr(p), 1024);
            if (pool != IDLE) {
                const uint32_t r = pool % 32, off = ((r >> 4) * 512 + (r & 15) * 16) * 2;
                for (uint32_t d = 0; d < 4; d++) {
                    for (uint32_t c = 0; c < 2; c++) {
                        noc_async_write(rp + PK_OFF + (2 * d + c) * 64 + (off & 63),
                                        K.get_noc_addr(4 * (pool / 32) + d, off + c * 512), 32);
                    }
                }
            }
            noc_async_write_barrier();   // (a later row's pool may close in the same K tile)
        }
        // rows past M: row M - 1's cur_pos (their sel rows are an earlier pass's, valid; their outputs unread). Left
        // as L1 garbage they sent flash-MLA's spare batch rows out of range: a hang after ~200 passes (2026-11-12)
        for (uint32_t m = M; m < M_MAX; m++) w32(cp)[m] = w32(cp)[M - 1];
        asm volatile("fence rw, rw" ::: "memory");
        noc_async_write(cp, cpos.get_noc_addr(0), 4 * M_MAX);
        noc_async_write_barrier();
    });
}
#else
    if (np) {
        invalidate_l1_cache();
        // iq as 4 tiles of heads x dims (tile d: row h = iq[128 h + 32 d ..]), the compute transposes them
        cb_reserve_back(cb_iq, 4);
        const uint32_t q = get_write_ptr(cb_iq);
        for (uint32_t d = 0; d < 4; d++) {
            for (uint32_t h = 0; h < 32; h++) {
                const uint32_t src = R2 + IQ_OFF + (h * 128 + d * 32) * 2;
                const uint32_t dst = q + d * 2048 + (((h >> 4) * 2) * 256 + (h & 15) * 16) * 2;
                for (uint32_t e = 0; e < 8; e++) {
                    w32(dst)[e] = w32(src)[e];                // cols 0..15: face (h / 16) * 2
                    w32(dst + 512)[e] = w32(src + 32)[e];     // cols 16..31: the next face
                }
            }
        }
        // w: i_proj's 32 fp32 values, rounded to bf16, in row 0 of a zero tile
        cb_reserve_back(cb_wt, 1);
        const uint32_t wt = get_write_ptr(cb_wt);
        fused::fill_words(wt, 512, 0);
        for (uint32_t c = 0; c < 32; c++) {
            const uint32_t b = w32(R1 + A2_OFF + (T_PROJ - NQA) * 128)[c];
            w16(wt + (c < 16 ? c * 2 : 512 + (c - 16) * 2))[0] = static_cast<uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
        }
        asm volatile("fence rw, rw" ::: "memory");
        cb_push_back(cb_iq, 4);
        cb_push_back(cb_wt, 1);
        for (uint32_t t = p0 + k_pre; t < p0 + np; t++) keys(t);
    }
    grid.wait(3, 0, 0, [&] {
        fused::l1_copy(R1 + A2_OFF, SC + n_pool * 4, A2_BYTES);
        fused::l1_copy(R2 + IQ_OFF, SC + n_pool * 4 + A2_BYTES, IQ_BYTES);
        noc_async_read_barrier();
        fused::Link link;
        link.open(H2D_CONFIG, D2H_CONFIG, RB, PAGE);
        link.send(SC);
        link.recv(REPLY);
        link.close();
        invalidate_l1_cache();
        const uint32_t p = w32(REPLY + CTL_OFF)[1], pool = w32(REPLY + CTL_OFF)[2];
        noc_async_write(REPLY, sel.get_noc_addr(0), NS * 4);
        noc_async_write(REPLY + CTL_OFF, cpos.get_noc_addr(0), 4);
        if (p != IDLE) noc_async_write(REPLY + LAT_OFF, lat.get_noc_addr(p), 1024);
        if (pool != IDLE) {
            // K's row pool % 32 in tiles 4 (pool / 32) + d: cols 0..15 and 16..31 at off and off + 512
            const uint32_t r = pool % 32, off = ((r >> 4) * 512 + (r & 15) * 16) * 2;
            for (uint32_t d = 0; d < 4; d++) {
                for (uint32_t c = 0; c < 2; c++) {
                    noc_async_write(REPLY + PK_OFF + (2 * d + c) * 64 + (off & 63),
                                    K.get_noc_addr(4 * (pool / 32) + d, off + c * 512), 32);
                }
            }
        }
        noc_async_write_barrier();
    });
}
#endif
