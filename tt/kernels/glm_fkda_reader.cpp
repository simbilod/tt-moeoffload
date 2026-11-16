// GLM's fused KDA mixer, reader: one layer's KDA mixer, hn -> out, in three phases on the whole grid
// with a barrier between them (fused/grid.h; compute in glm_fkda_compute.cpp, writer glm_fkda_writer.cpp). The port
// of Flash-Next's fgdn_*.cpp to GLM's shapes:
//
// A  y = hn W (4096 -> 24896) on all cores, each core's output tiles followed by their channel-local epilogue. W's
//    bfp8_b tiles are n-major (page n * KA + k), its columns q | k | v (24576, the causal conv's channels) | f_lo (128,
//    the decay's low-rank half) | g_lo (128, the output gate's) | beta (64). A core's work is a list of units, each a
//    run of output tiles of one type: 0 v tile (conv + silu), 1 q head and 2 k head (4 tiles: conv + silu + L2 norm),
//    3 plain (f_lo, g_lo: no epilogue), 5 beta (sigmoid). The results (fp32) go to the arena's region A.
// B  one value head per core (64 cores), from region A: first the head's decays and gate from the low-rank halves,
//    d = exp(-5 sigmoid((f_lo f_b_h) neg_a + dt neg_a)) and g = sigmoid(g_lo g_b_h) (f_b_h, g_b_h: the head's 128
//    columns, 16 bfp8_b tiles each, read before the barrier); then glm_kda_compute.cpp's step (the state exact in
//    fp32, the per-key decay through Dbc, which this reader builds from the d tiles the compute hands it in cb 33);
//    the head's 128 outputs (bf16) into region B.
// C  out = o W_o (8192 -> 4096) split by K in KS parts (glm_fmo_reader.cpp's E2): item i = (n = i / KS, part = i % KS)
//    covers pages 256 n + (256 / KS) part .., a core's run of items one page range; the partial rows to region P
//    ([KS][128][32] fp32), broadcast at barrier 3; then each core sums its output tiles' partials (this RISC-V's FPU)
//    into out (row-major fp32 [4096]).
//
// x travels as mgemv's ring (glm_mgemv_reader.cpp): for each weight batch the reader writes row 0 of BATCH bf16 x
// tiles (rows 1-31 reach only output rows nobody reads), 2 * BATCH tiles of L1 at any K. C's first weight batches are
// read before barrier 1 (they need no x); their x batches follow after barrier 2, in the same order.
// Dense arrays are row-major fp32 tensors, a row a page: conv state [3, 24576], taps [4, 24576], dt * neg_a [64, 128],
// the norm weight [1, 128]. The KDA state is tiles, head h's tile (j, c) at page 16 h + 4 j + c.
// Runtime args: hn address, grid.h's block, W, conv state, taps, unit count, per unit (first tile, count, type), then
// the head (0xffffffff: none), S, norm weight, low-rank W (page 32 h + 16 g + 4 n + k), dt * neg_a, then W_o, C's
// first item and count, the first output tile it sums and their count, out, then the arena offsets of regions A, B, P,
// then the prefetch buffer's L1 address and its page count: A's first npf pages, in A's order, were read into this
// core's L1 there by the previous layer's FFN program while the lane worked (glm_fsh_reader.cpp; 0: none).
// Compile args: batch, depth, phases (bit 0 A, 1 B, 2 C), KS, then a DRAM accessor.
//
// M rows (-DM_MAX = 2..4: a verify pass of M rows, M at runtime; M_MAX = 1 compiles the one-row source):
// the step tensor (a runtime arg after the prefetch's) holds words 1..4 M, n_keep, parity, mp (the layout's rows: 1, 2
// or 4, M = 3 in the 4-row layout with row 3 a copy of row 2); hn is mp rows in fused/hc_dm.h's block layout
// (bf16, 8 KB a row), x rides the ring at mp rows (fused::x_ring_m_issue, NoC copies). A's conv constants come from the writer, which
// builds the conv window from the compute's y (glm_fkda_writer.cpp); region A is not broadcast (sent by owner).
#include <cstdint>
#ifndef M_MAX
#define M_MAX 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#include "fused/ksum.h"
#include "fused/fpu.h"
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2, cb_c = 3, cb_sx = 8, cb_sop = 9, cb_nw = 10;
constexpr uint32_t cb_q = 11, cb_k = 12, cb_ex = 13, cb_dbc = 26, cb_lw = 27, cb_xlo = 28, cb_zero = 29;
constexpr uint32_t cb_arena = 30, cb_stage = 31, cb_dtn = 32, cb_dsrc = 33;
constexpr uint32_t cb_pk = 35, cb_pu = 36, cb_L = 37;   // M rows (glm_fkda_compute.cpp phase_b_m)
constexpr uint32_t W_PAGE = 1088, KA = 128, KC = 256, CONV = 24576, NT_A = 778, NV = 64, HK = 128;
constexpr uint32_t T_V = 0, T_Q = 1, T_K = 2, T_P = 3, T_B = 5;
constexpr uint32_t IDLE = 0xffffffff;
constexpr uint32_t A_FLO = CONV * 4, A_GLO = A_FLO + 128 * 4, A_BETA = A_GLO + 128 * 4;  // byte offsets in region A

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
inline volatile tt_l1_ptr uint16_t* w16(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a); }

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t phases = get_compile_time_arg_val(2);
    constexpr uint32_t KS = get_compile_time_arg_val(3);  // C's K split
    constexpr uint32_t KP = KC / KS;                        // C's k-tiles an item
    constexpr auto d_args = TensorAccessorArgs<4>();
    const auto hn = TensorAccessor(d_args, get_arg_val<uint32_t>(0), M_MAX * KA * 64);
    const fused::Grid grid = fused::Grid::from_args(1);
    uint32_t i = 1 + fused::Grid::args(grid.nrects);
    const auto w = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto state = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), CONV * 4);
    const auto taps = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), CONV * 4);
    const uint32_t n_units = get_arg_val<uint32_t>(i++);
    const uint32_t ua = i;
    i += 3 * n_units;
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const auto S = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 4096);
    const auto nw = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), HK * 4);
    const auto lw = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto dtn = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), HK * 4);
    const auto wo = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const uint32_t c_first = get_arg_val<uint32_t>(i++), c_count = get_arg_val<uint32_t>(i++);   // C's items
    const uint32_t s0 = get_arg_val<uint32_t>(i++), n_s = get_arg_val<uint32_t>(i++);
    const auto out = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 128 * 128);
    const uint32_t region_a = get_arg_val<uint32_t>(i++), region_b = get_arg_val<uint32_t>(i++);
    const uint32_t region_c = get_arg_val<uint32_t>(i++);
    const uint32_t pf = get_arg_val<uint32_t>(i++), npf = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena);
#if M_MAX > 1
    const auto step = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 64);
    constexpr uint32_t PEND = 12 * M_MAX * 128;   // a pending page: Ki, U, G of the pass's rows, M_MAX-row blocks
    const auto pend = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), PEND);
    noc_async_read(step.get_noc_addr(0), get_write_ptr(cb_stage), 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stage));
    const uint32_t M = sw[1], n_keep = sw[2], parity = sw[3], mp = sw[4];
#endif
#ifdef GRID_PROFILE
    const uint32_t gp_addr = get_arg_val<uint32_t>(i++), gp_slot = get_arg_val<uint32_t>(i++);
    fused::Grid::mark(0);
#endif

    fused::Stream<decltype(wo)> ws{cb_w, batch, W_PAGE};   // the weight stream (the open batch persists)

    uint32_t a_tiles = 0;
    for (uint32_t u = 0; u < n_units; u++) {
        a_tiles += get_arg_val<uint32_t>(ua + 3 * u + 1);
    }
    const bool any = a_tiles || head != IDLE || c_count;

#if M_MAX > 1
    noc_async_read(hn.get_noc_addr(0), arena, mp * KA * 64);
#else
    noc_async_read(hn.get_noc_addr(0), arena, KA * 64);
#endif
    noc_async_read_barrier();
    fused::fill_words(get_write_ptr(cb_zero), 1024, 0);
    if (any) {
        cb_reserve_back(cb_ones, 1);
        fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);   // bf16 1.0 everywhere
        cb_push_back(cb_ones, 1);
    }

    // ---- A: the constants of every conv tile (M rows: the writer's), then the weights with their x batches
    if ((phases & 1) && a_tiles) {
#if M_MAX == 1
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            if (type > T_K) continue;
            for (uint32_t n = first; n < first + count; n++) {
                cb_reserve_back(cb_c, 7);
                const uint32_t c = get_write_ptr(cb_c);
                for (uint32_t j = 0; j < 3; j++) fused::row0_from_dram(state, j, n * 128, c + j * 4096);
                for (uint32_t j = 0; j < 4; j++) fused::row0_from_dram(taps, j, n * 128, c + (3 + j) * 4096);
                noc_async_read_barrier();
                cb_push_back(cb_c, 7);
            }
        }
#endif
        uint32_t a_done = 0;   // A's pages issued: the first npf from this core's prefetch buffer
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            for (uint32_t n = first; n < first + count; n++) {
                for (uint32_t k = 0; k < KA; k += batch) {
#if M_MAX > 1
                    fused::x_ring_m_issue(cb_x, arena, k, batch, mp);   // pushed after the weights' batch
#else
                    fused::x_ring(cb_x, arena, k, batch);
#endif
                    if (a_done + batch <= npf) {
                        for (uint32_t j = 0; j < batch; j++) ws.read_page(get_noc_addr(pf + (a_done + j) * W_PAGE));
                    } else if (a_done >= npf) {
                        ws.read(w, n * KA + k, batch);
                    } else {
                        for (uint32_t j = 0; j < batch; j++) {
                            if (a_done + j < npf) ws.read_page(get_noc_addr(pf + (a_done + j) * W_PAGE));
                            else ws.read(w, n * KA + k + j, 1);
                        }
                    }
                    a_done += batch;
#if M_MAX > 1
                    cb_push_back(cb_x, batch);
#endif
                }
            }
        }
    }

    // ---- B's and C's inputs that need no activation, before the barrier
#if M_MAX > 1
    // M rows: B's input tiles zeroed (each pass fills rows 0..mp-1 only); the parity's committed state (cb 8); the
    // commit's inputs from the parity's pending page (rows j < a = n_keep of Ki_p, U_p; G_p[a-1] down rows 0..a-1 and
    // as Dbc); the norm weight and dt * neg_a down rows 0..mp-1; the low-rank W
    if ((phases & 2) && head != IDLE) {
        const uint32_t zero = get_write_ptr(cb_zero), scr = arena + region_c;   // scratch: C's region, unused in B
        const uint32_t pk = get_write_ptr(cb_pk), pu = get_write_ptr(cb_pu);
        for (uint32_t cb : {cb_q, cb_k, cb_nw, cb_dtn, cb_L, cb_pu}) cb_reserve_back(cb, 4);
        cb_reserve_back(cb_pk, 8);
        cb_reserve_back(cb_ex, 10);
        cb_reserve_back(cb_xlo, 8);
        for (uint32_t cb : {cb_q, cb_k, cb_dtn, cb_L, cb_pu}) {
            for (uint32_t t = 0; t < 4; t++) fused::l1_copy(zero, get_write_ptr(cb) + t * 4096, 4096);
        }
        for (uint32_t t = 0; t < 8; t++) {
            fused::l1_copy(zero, pk + t * 4096, 4096);
            fused::l1_copy(zero, get_write_ptr(cb_ex) + t * 4096, 4096);
        }
        for (uint32_t t = 0; t < 4; t++) fused::l1_copy(zero, get_write_ptr(cb_xlo) + t * 4096, 4096);   // 8 bf16
        cb_reserve_back(cb_sx, 16);
        const uint32_t sx = get_write_ptr(cb_sx), pg = parity * NV + head;
        for (uint32_t t = 0; t < 16; t++) noc_async_read(S.get_noc_addr(pg * 16 + t), sx + t * 4096, 4096);
        noc_async_read(nw.get_noc_addr(0), scr, HK * 4);
        noc_async_read(dtn.get_noc_addr(head), scr + 512, HK * 4);
        noc_async_read_barrier();   // (the zeroing too: the rows land on zeros)
        const uint32_t a = n_keep, half = M_MAX * 64;
        for (uint32_t t = 0; t < 4; t++) {
            noc_async_read(pend.get_noc_addr(pg, t * 2 * half), pk + t * 4096, a * 64);              // Ki_p
            noc_async_read(pend.get_noc_addr(pg, t * 2 * half + half), pk + t * 4096 + 1024, a * 64);
            noc_async_read(pend.get_noc_addr(pg, (4 + t) * 2 * half), pu + t * 4096, a * 64);        // U_p
            noc_async_read(pend.get_noc_addr(pg, (4 + t) * 2 * half + half), pu + t * 4096 + 1024, a * 64);
            noc_async_read(pend.get_noc_addr(pg, (8 + t) * 2 * half + (a - 1) * 64), scr + 1024 + t * 128, 64);
            noc_async_read(pend.get_noc_addr(pg, (8 + t) * 2 * half + half + (a - 1) * 64), scr + 1024 + t * 128 + 64,
                           64);   // G_p[a-1], 32 t.. as a row
            fused::row_down(scr + t * 128, get_write_ptr(cb_nw) + t * 4096, mp);
            fused::row_down(scr + 512 + t * 128, get_write_ptr(cb_dtn) + t * 4096, mp);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < 4; t++) fused::row_down(scr + 1024 + t * 128, pk + (4 + t) * 4096, a);
        cb_reserve_back(cb_dbc, 4);   // Dbc_a (r, c) = G_p[a-1][32 j + r]
        const uint32_t dbc = get_write_ptr(cb_dbc);
        invalidate_l1_cache();
        for (uint32_t j = 0; j < 4; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t bits = w32(scr + 1024 + (32 * j + r) * 4)[0];
                const uint32_t base = dbc + j * 4096 + ((r >> 4) * 2 * 256 + (r & 15) * 16) * 4;
                for (uint32_t c = 0; c < 16; c++) {
                    w32(base)[c] = bits;
                    w32(base + 256 * 4)[c] = bits;
                }
            }
        }
        asm volatile("fence rw, rw" ::: "memory");
        cb_reserve_back(cb_lw, 32);
        for (uint32_t p = 0; p < 32; p++) {
            noc_async_read(lw.get_noc_addr(head * 32 + p), get_write_ptr(cb_lw) + p * W_PAGE, W_PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_sx, 16);
        cb_push_back(cb_pk, 8);
        cb_push_back(cb_pu, 4);
        cb_push_back(cb_dbc, 4);
        cb_push_back(cb_nw, 4);
        cb_push_back(cb_dtn, 4);
        cb_push_back(cb_lw, 32);
    }
#else
    if ((phases & 2) && head != IDLE) {
        // the state once from card memory (cb 8, exact), its operand copy (cb 9) by a local L1 copy
        cb_reserve_back(cb_sx, 16);
        cb_reserve_back(cb_sop, 16);
        const uint32_t sx = get_write_ptr(cb_sx), sop = get_write_ptr(cb_sop);
        for (uint32_t t = 0; t < 16; t++) noc_async_read(S.get_noc_addr(head * 16 + t), sx + t * 4096, 4096);
        noc_async_read_barrier();
        fused::l1_copy(sx, sop, 16 * 4096);
        cb_reserve_back(cb_nw, 4);
        cb_reserve_back(cb_dtn, 4);
        fused::fill_words(get_write_ptr(cb_dtn), 4 * 1024, 0);   // zero outside row 0: d's other rows meet k's zeros
        asm volatile("fence rw, rw" ::: "memory");             // the stores land before the NoC's row 0
        for (uint32_t c = 0; c < 4; c++) {
            fused::row0_from_dram(nw, 0, c * 128, get_write_ptr(cb_nw) + c * 4096);
            fused::row0_from_dram(dtn, head, c * 128, get_write_ptr(cb_dtn) + c * 4096);
        }
        cb_reserve_back(cb_lw, 32);
        for (uint32_t p = 0; p < 32; p++) {
            noc_async_read(lw.get_noc_addr(head * 32 + p), get_write_ptr(cb_lw) + p * W_PAGE, W_PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_sx, 16);
        cb_push_back(cb_sop, 16);
        cb_push_back(cb_nw, 4);
        cb_push_back(cb_dtn, 4);
        cb_push_back(cb_lw, 32);
    }
#endif
    const uint32_t c_pages = c_count * KP, c_page0 = c_first * KP;
    const uint32_t c_pre = (phases & 4) ? (c_pages < depth * batch ? c_pages : depth * batch) : 0;
    ws.read(wo, c_page0, c_pre);

    if (!(phases & 1)) return;
#if M_MAX > 1
    grid.wait(1);
    // ---- B at M rows, from this core's region A (blocks of mp rows: q 0..3, k 4..7, v 8..11 of its head; f_lo
    // 12..15, g_lo 16..19, beta 20, 21 of all heads)
    if ((phases & 2) && head != IDLE) {
        const uint32_t ra = arena + region_a, blk = mp * 128, scr = arena + region_c;
        invalidate_l1_cache();
        const uint32_t xl = get_write_ptr(cb_xlo);   // f_lo, g_lo as bf16 x tiles, rows 0..mp-1 (round to nearest even)
        for (uint32_t t = 0; t < 8; t++) {
            for (uint32_t r = 0; r < mp; r++) {
                for (uint32_t e = 0; e < 32; e++) {
                    const uint32_t b = w32(ra + (12 + t) * blk + (e < 16 ? 0 : mp * 64) + r * 64 + (e & 15) * 4)[0];
                    w16(xl + t * 2048 + (e < 16 ? 0 : 512) + (r * 16 + (e & 15)) * 2)[0] =
                        (b + 0x7fff + ((b >> 16) & 1)) >> 16;
                }
            }
        }
        asm volatile("fence rw, rw" ::: "memory");
        cb_push_back(cb_xlo, 8);
        const uint32_t q = get_write_ptr(cb_q), k = get_write_ptr(cb_k), e = get_write_ptr(cb_ex);
        for (uint32_t t = 0; t < 4; t++) {
            fused::rows_from_l1(ra + t * blk, q + t * 4096, mp);
            fused::rows_from_l1(ra + (4 + t) * blk, k + t * 4096, mp);
            fused::rows_from_l1(ra + (8 + t) * blk, e + t * 4096, mp);   // v
        }
        // the coefficient tiles from beta_t (bits only): 4 beta down row t, 5 -beta, 6 CL (t, j < t) = -beta_t,
        // 7 CP (t, j <= t) = 1, rows t < M (an element (r < 16, c): word (c / 16) 256 + 16 r + c % 16)
        const uint32_t bh = ra + (20 + head / 32) * blk + ((head & 31) < 16 ? 0 : mp * 64) + (head & 15) * 4;
        for (uint32_t t = 0; t < M; t++) {
            const uint32_t bt = w32(bh + t * 64)[0], nb = bt ^ 0x80000000;
            for (uint32_t c = 0; c < 32; c++) {
                const uint32_t w = (c >> 4) * 256 + 16 * t + (c & 15);
                w32(e + 4 * 4096)[w] = bt;
                w32(e + 5 * 4096)[w] = nb;
                if (c < t) w32(e + 6 * 4096)[w] = nb;
                if (c <= t) w32(e + 7 * 4096)[w] = 0x3f800000;
            }
        }
        asm volatile("fence rw, rw" ::: "memory");
        noc_async_read_barrier();
        cb_push_back(cb_q, 4);
        cb_push_back(cb_k, 4);
        cb_push_back(cb_ex, 10);

        // L_t = l_0 + .. + l_t (this RISC-V's float adds) from the compute's l (cb 33), rows t < M
        cb_wait_front(cb_dsrc, 4);
        invalidate_l1_cache();
        const uint32_t ls = get_read_ptr(cb_dsrc), L = get_write_ptr(cb_L);
        for (uint32_t j = 0; j < 4; j++) {
            for (uint32_t c = 0; c < 32; c++) {
                float acc = 0.0f;
                for (uint32_t t = 0; t < M; t++) {
                    const uint32_t w = j * 1024 + (c >> 4) * 256 + 16 * t + (c & 15);
                    acc = fused::fpu::add(acc, reinterpret_cast<volatile tt_l1_ptr float*>(ls)[w]);
                    reinterpret_cast<volatile tt_l1_ptr float*>(L)[w] = acc;
                }
            }
        }
        asm volatile("fence rw, rw" ::: "memory");
        cb_pop_front(cb_dsrc, 4);
        cb_push_back(cb_L, 4);
        (void)scr;
    }
    if (phases & 2) grid.wait(2, arena + region_b, mp * NV * HK * 2);
    // ---- C at M rows: x from region B's blocks; the partials come by owner (the writer's), summed into out's blocks
    if (phases & 4) {
        for (uint32_t p = 0; p < c_pages; p += batch) {
            fused::x_ring_m_issue(cb_x, arena + region_b, (c_page0 + p) % KC, batch, mp);
            if (p >= c_pre) {
                ws.read(wo, c_page0 + p, batch);
            } else {
                noc_async_read_barrier();
            }
            cb_push_back(cb_x, batch);
        }
        grid.wait(3);
        fused::ksplit_sum_m(out, arena + region_c, KS, s0, n_s, mp * 128, get_write_ptr(cb_stage));
    }
    return;
#endif
    grid.wait(1, arena + region_a, NT_A * 128);

    // ---- B
    if (phases & 2) {
        if (head != IDLE) {
            // the low-rank halves as bf16 x tiles (round to nearest even; the RISC-V's integer ops only): f_lo in
            // tiles 0..3, g_lo in 4..7
            invalidate_l1_cache();
            cb_reserve_back(cb_xlo, 8);
            const uint32_t xl = get_write_ptr(cb_xlo);
            fused::fill_words(xl, 8 * 512, 0);                      // zero outside row 0, as cb 32
            for (uint32_t t = 0; t < 8; t++) {
                for (uint32_t e = 0; e < 32; e++) {
                    const uint32_t b = w32(arena + region_a + A_FLO + (t * 32 + e) * 4)[0];
                    const uint16_t h16 = (b + 0x7fff + ((b >> 16) & 1)) >> 16;
                    w16(xl + t * 2048 + (e < 16 ? e * 2 : 512 + (e - 16) * 2))[0] = h16;
                }
            }
            asm volatile("fence rw, rw" ::: "memory");
            cb_push_back(cb_xlo, 8);

            const uint32_t zero = get_write_ptr(cb_zero);
            for (uint32_t cb : {cb_q, cb_k}) {  // zero outside row 0: k becomes a column by transpose
                cb_reserve_back(cb, 4);
                for (uint32_t t = 0; t < 4; t++) fused::l1_copy(zero, get_write_ptr(cb) + t * 4096, 4096);
            }
            noc_async_read_barrier();
            const uint32_t q = get_write_ptr(cb_q), k = get_write_ptr(cb_k);
            cb_reserve_back(cb_ex, 10);
            const uint32_t e = get_write_ptr(cb_ex);
            for (uint32_t t : {0u, 1u, 2u, 3u, 8u}) fused::l1_copy(zero, e + t * 4096, 4096);
            noc_async_read_barrier();
            for (uint32_t t = 0; t < 4; t++) {
                fused::row0_from_l1(arena + region_a + (head * HK + t * 32) * 4, q + t * 4096);
                fused::row0_from_l1(arena + region_a + (8192 + head * HK + t * 32) * 4, k + t * 4096);
                fused::row0_from_l1(arena + region_a + (16384 + head * HK + t * 32) * 4, e + t * 4096);   // v
            }
            noc_async_read_barrier();
            cb_push_back(cb_q, 4);
            cb_push_back(cb_k, 4);
            fused::fill_row0(e + 8 * 4096, arena + region_a + A_BETA + head * 4);                     // beta
            cb_push_back(cb_ex, 10);

            // Dbc_j (r, c) = d[32 j + r] from the compute's d tiles (row 0: d[32 j + r] at face r / 16, word r % 16):
            // face (r / 16) * 2 + c / 16, element (r % 16) * 16 + c % 16
            cb_wait_front(cb_dsrc, 4);
            invalidate_l1_cache();                                   // the packer wrote it: no stale lines
            cb_reserve_back(cb_dbc, 4);
            const uint32_t ds = get_read_ptr(cb_dsrc), dbc = get_write_ptr(cb_dbc);
            for (uint32_t j = 0; j < 4; j++) {
                for (uint32_t r = 0; r < 32; r++) {
                    const uint32_t bits = w32(ds + j * 4096 + ((r >> 4) * 256 + (r & 15)) * 4)[0];
                    const uint32_t base = dbc + j * 4096 + ((r >> 4) * 2 * 256 + (r & 15) * 16) * 4;
                    for (uint32_t c = 0; c < 16; c++) {
                        w32(base)[c] = bits;
                        w32(base + 256 * 4)[c] = bits;
                    }
                }
            }
            asm volatile("fence rw, rw" ::: "memory");
            cb_pop_front(cb_dsrc, 4);
            cb_push_back(cb_dbc, 4);
        }
        grid.wait(2, arena + region_b, NV * HK * 2);
    }

    // ---- C: the x batches in the weights' order (the first c_pre pages were read before barrier 1)
    if (phases & 4) {
        for (uint32_t p = 0; p < c_pages; p += batch) {
            fused::x_ring(cb_x, arena + region_b, (c_page0 + p) % KC, batch);
            if (p >= c_pre) ws.read(wo, c_page0 + p, batch);
        }
        grid.wait(3, arena + region_c, KS * 128 * 128);
        fused::ksplit_sum(out, arena + region_c, KS, 128, s0, n_s, 0, get_write_ptr(cb_stage));
#ifdef GRID_PROFILE
        fused::Grid::mark(1);
        fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
    }
}
