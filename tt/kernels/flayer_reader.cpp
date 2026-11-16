// Fused DeltaNet layer reader (G3 step 4): one decode token through a whole gated-DeltaNet layer of Flash-Next except
// its routed experts, as ONE program on the whole grid: 9 phases with a grid barrier after each (tt/kernels/fused/
// grid.h). Compute in flayer_compute.cpp, writer in flayer_writer.cpp. The 4 residual streams X (fp32 and a bf16 copy)
// live in every core's arena; each phase's outputs are gathered into the coordinator's arena and broadcast (A's go
// straight to the B cores that read them).
//
//   barrier 1   X arrives (this test: the coordinator reads it from card memory)
//   H1a, H2a    the attention half's hyper-connection mix (fhc_*.cpp): hn_a and its 4 inject weights w_a
//   A, B, C     the DeltaNet mixer (fgdn_*.cpp); C's epilogue is the combine X += w_a * out
//   H1f, H2f    the FFN half's hyper-connection mix on the new X: hn_f, w_f
//   R1          the router (2560 -> 512 logits + the shared expert's gate logit, bf16 weights) on 17 cores; the
//               shared expert's gate|up (2560 -> 640 | 640) on 20 cores, h = silu(g) * u as their epilogue
//   R2          the shared expert's down (640 -> 2560) on 80 cores; epilogue sh = down(h) * sigmoid(gate logit),
//               then the combine X += w_f * (sh + routed), routed being the experts' sum (a stand-in here)
//   EXCHANGE    (with -DEXCHANGE) between R1 and R2: the coordinator sends hn_f and the logits to the host over the
//               D2H socket and takes the host's routed sum from the H2D socket into region RT (fused/link.h), which
//               the next barrier broadcasts; R2 reads routed from there. While the host computes, every core
//               streams the next pass's first A weight pages into page slots carved from B's and C's buffers
//               (cb 23..45: idle from here until the next pass's B), and that A reads them from L1
//
// Runtime args, in order (the host side is tt/flashnext/g3_layer.py):
//   X fp32, X bf16 addresses; grid.h's block;
//   arena offsets X32, X16, P1, RS, HNA, LOA, WA, HNF, LOF, WF, A, B, L, SH, RT, AC, TAB, PROF, RQ;
//   HC: W'_a, up_a, norm_a, W'_f, up_f, norm_f, the H1 item (4 n + s), the H2 tile (80: the inject weights);
//   A: W, conv state, taps, dt|neg_a, unit count, units (first tile, count, type); B: head, S, norm weight;
//   C: W_o, C's tile; R1: router W, gate|up W, the R1 item (< 17 router tile, 17 + j gate|up pair j);
//   R2: down W, R2's tile, the routed stand-in; the profile tensor (16 words: the coordinator's wall clock at the
//   start and after each barrier); the layer table; with CARD_EXPERTS (fused/dexp_dm.h, between R1 and R2) the
//   card's partial rows and its routes in card memory (without the host link), this core's index, the core count;
//   with M_MAX > 1 the step tensor. Unused items are 0xffffffff.
// Compile args: batch, depth, then a DRAM accessor.
//
// M rows (-DM_MAX=n, n <= 4: a verify pass; M, the rows kept of the previous pass and a parity from the step tensor,
// words 1..4 M, n_keep, parity, Mp after the position in word 0; M at runtime): every phase on rows 0..M-1 of its
// tiles, the vectors in fused/hc_dm.h's mp-row blocks. The arena is the owner layout: X fp32 only in its tile's owner
// (xown, the RS slot) and in each H1 core for its stream (xs32, xs16 in the X32 and X16 slots); at the start each of a
// stream's 11 H1 cores reads 1/11 of it from the X tensors (a stream a page) and sends it to the other 10. Region A is
// each B core's own head (q, k, v, z blocks, then the decay's and beta's tile blocks). The conv's window is the
// writer's (A's y rows shifted under the history). B runs the delta rule over the rows in the chunked form with a
// deferred commit (flayer_compute.cpp phase_b_m): a pass leaves the state it started from and its rows' k, U and
// decays (the pending page); the next pass first commits the n_keep rows it keeps onto that state. The S tensor holds
// per parity the 48 heads' committed states (16 tiles each), then per parity and head 2 pending pages: page 0 k's and
// U's rows 0..3 (tile t's face-f rows at t * 512 + f * 256, U's tiles after k's), page 1 the decays (words 0..3). A
// pass reads the state and pending pages of the parity it is given, and conv history rows [n_keep - 1, n_keep + 2);
// it writes the other parity's. With the host link (EXCHANGE) the coordinator stages the M rows' hn_f and logits (their
// mp-row blocks: hn_f's 80 bf16 tiles, then the 17 logit tiles) in region B, idle from C's barrier to the next pass's B,
// and sends them as ceil(mp (5120 + 2176) / 8192) pages (at Mp = 1 the M = 1 program's page); the host's reply is M
// pages, row m's routed sum fp32 [2560] each (the M = 1 program's reply), into region B at m 10240, which the next
// barrier broadcasts; R2 takes row min(m, M - 1) for row m. The A prefetch's slots leave out B's input buffers
// (cb 25..28: zeroed once, their unused rows must stay zero). With CARD_EXPERTS at M rows (fused/dexp_dm.h's M-row
// part): E1's x built while the host routes, the host's first reply the card's routes and the rows' weights, the A
// prefetch spread over the card's reads, the partial rows of every row before the rows' CPU sums. The M-row code is
// preprocessed out at M_MAX = 1 (the decode build compiles exactly the one-row source).
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"
#include "fused/tail_dm.h"
#ifdef EXCHANGE
#include "fused/link.h"
#endif
#ifdef CARD_EXPERTS
#include "fused/dexp_dm.h"
#endif

// With the host link, without the card's experts (in the layers the layer loop's sh_layer picks): R1 is the
// router's alone. The shared expert's gate|up (20 cores, ~25 us against the router's ~16) runs after the page goes out,
// while the host computes, and its h has its own barrier (SH) before R2 reads it.
#if defined(EXCHANGE) && !defined(CARD_EXPERTS)
#define SH_LATE 1
#endif
#ifndef M_MAX
#define M_MAX 1
#endif


constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2;
constexpr uint32_t cb_hd = 3, cb_hp = 9, cb_hxs = 11, cb_hnrm = 12, cb_hrs = 13;
constexpr uint32_t cb_ac = 16;
constexpr uint32_t cb_bsx = 23, cb_bsop = 24, cb_bnw = 25, cb_bq = 26, cb_bk = 27, cb_bex = 28;
constexpr uint32_t cb_cx = 40, cb_cxs = 42, cb_cw = 43;
constexpr uint32_t cb_rw = 46;
constexpr uint32_t cb_sgl = 51, cb_srt = 52, cb_sxs = 53, cb_sw = 54;
constexpr uint32_t cb_xout = 58, cb_zero = 60, cb_arena = 61;
constexpr uint32_t W_PAGE = 1088, R_PAGE = 2048, D = 2560, CONV = 10240, HK = 128, LO = 352, NV = 48;
constexpr uint32_t KD = 80, KS = 80, KHU = 10, KC = 192, KR2 = 20;
constexpr uint32_t T_V = 0, T_Q = 1, T_K = 2, T_A = 4;
constexpr uint32_t IDLE = 0xffffffff;
constexpr uint32_t A_ALPHA = (CONV + 6144) * 4, A_BETA = A_ALPHA + 64 * 4;  // byte offsets in region A
constexpr uint32_t NT_A = 516, N_ROUTER = 17;
constexpr uint32_t PF_FIRST = 23, PF_LAST = 45, PF_N = PF_LAST - PF_FIRST + 1;  // the A prefetch's buffers (B's, C's)
constexpr uint32_t CS_FIRST = 16, CS_LAST = 22, CS_N = CS_LAST - CS_FIRST + 1;  // C's prefetch's buffers (A's)
constexpr uint32_t C_SLOTS = 80;  // C's pages read into them during B (more delays B's state writes)
// a layer's tensors, in the order of a layer table row (the conv state and the DeltaNet state are updated in place)
enum : uint32_t {
    L_WD_A, L_WU_A, L_NORM_A, L_WD_F, L_WU_F, L_NORM_F, L_WA, L_CSTATE, L_TAPS, L_DTNA, L_S, L_BNW, L_WO, L_WR, L_WGU,
    L_WSD, L_ROUTED, L_GUC, L_GUS, L_DV, L_DS, L_N  // L_GUC..L_DS: the layer's card expert cache (fused/dexp_dm.h)
};
#if M_MAX > 1
constexpr uint32_t cb_bpend = 59;           // M rows: the previous pass's kept rows of k (tiles 0..3) and U (4..7)
constexpr uint32_t S_PEND = 2 * NV * 16;    // M rows: the S tensor's first pending page (after 2 parities of states)
constexpr uint32_t F_ONE = 0x3f800000;

// a * b for floats >= 0 (beta, the decays: this core has no float unit), rounded to nearest even, results below the
// normal range flushed to 0, subnormal inputs taken as 0
inline uint32_t fmul(uint32_t a, uint32_t b) {
    const uint32_t ea = a >> 23, eb = b >> 23;
    if (ea == 0 || eb == 0) {
        return 0;
    }
    uint64_t p = static_cast<uint64_t>((a & 0x7fffff) | 0x800000) * ((b & 0x7fffff) | 0x800000);  // [2^46, 2^48)
    int32_t e = static_cast<int32_t>(ea + eb) - 127;
    if (p >> 47) {
        e++;
    } else {
        p <<= 1;
    }
    uint32_t m = static_cast<uint32_t>(p >> 24);  // 24 bits, the leading one at bit 23
    const uint32_t rest = static_cast<uint32_t>(p) & 0xffffff;
    if (rest > 0x800000 || (rest == 0x800000 && (m & 1))) {
        if (++m == 0x1000000) {
            m >>= 1;
            e++;
        }
    }
    if (e <= 0) {
        return 0;
    }
    return e >= 255 ? 0x7f800000 : (static_cast<uint32_t>(e) << 23) | (m & 0x7fffff);
}

// the float bits across row r (< 16) of the fp32 tile at dst
inline void fill_row_bits(uint32_t dst, uint32_t r, uint32_t bits) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t j = 0; j < 16; j++) {
        t[16 * r + j] = bits;
        t[256 + 16 * r + j] = bits;
    }
}
#endif

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t layers = get_compile_time_arg_val(2);  // passes through the layer in one launch
    constexpr auto d_args = TensorAccessorArgs<3>();
    using Acc = decltype(TensorAccessor(d_args, 0u, 0u));
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
#if M_MAX > 1
    const auto x32 = acc(M_MAX * D * 4), x16 = acc(M_MAX * D * 2);  // a stream a page
#else
    const auto x32 = acc(4 * D * 4), x16 = acc(4 * D * 2);
#endif
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_hna = arena + arg(), r_loa = arena + arg(), r_wa = arena + arg(), r_hnf = arena + arg();
    const uint32_t r_lof = arena + arg(), r_wf = arena + arg();
    const uint32_t r_a = arena + arg(), r_b = arena + arg(), r_l = arena + arg(), r_sh = arena + arg();
    const uint32_t r_rt = arena + arg(), r_ac = arena + arg();
    // scratch in the arena: the layer table row (the reader's at +0, the writer's at +128), the profile stamps, the
    // card's routes (RQ)
    const uint32_t r_tab = arena + arg(), r_prof = arena + arg(), r_rq = arena + arg();
    // the per-layer tensors' addresses (L_* index them); a layer table in card memory replaces them per layer
    uint32_t base[L_N];
    base[L_WD_A] = arg(), base[L_WU_A] = arg(), base[L_NORM_A] = arg();
    base[L_WD_F] = arg(), base[L_WU_F] = arg(), base[L_NORM_F] = arg();
    const uint32_t h_item = arg(), h_tile = arg();
    base[L_WA] = arg(), base[L_CSTATE] = arg(), base[L_TAPS] = arg(), base[L_DTNA] = arg();
    const uint32_t n_units = arg(), ua = i;
    i += 3 * n_units;
    const uint32_t head = arg();
    base[L_S] = arg(), base[L_BNW] = arg();
    base[L_WO] = arg();
    const uint32_t c_tile = arg();
    base[L_WR] = arg(), base[L_WGU] = arg();
    const uint32_t r_item = arg();
    base[L_WSD] = arg();
    const uint32_t s_tile = arg();
    [[maybe_unused]] const bool sh_late = r_item != IDLE && r_item >= N_ROUTER;  // SH_LATE: a gate|up core
    base[L_ROUTED] = arg();
    const auto prof = acc(64);
    const uint32_t table = arg();  // 0: every pass uses the arguments above
    const auto tab = TensorAccessor(d_args, table, 128);
    // the card's partial rows, its routes in card memory (a test without the host link), this core's index, the count
    const uint32_t part = arg(), croutes = arg(), me = arg(), cores = arg();
    base[L_GUC] = base[L_GUS] = base[L_DV] = base[L_DS] = 0;
#if M_MAX > 1
    const uint32_t step_addr = arg();  // words 1..4 M, n_keep, parity, Mp (word 0: the token's position)
    const uint32_t h1_cores = i;       // H1's cores by item 4 n + s, NOC x | y << 16
    i += 4 * 11;
#endif
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    fused::Stream<Acc> ws{cb_w, batch, W_PAGE};
    fused::Stream<Acc> rs{cb_rw, batch, R_PAGE};
    // A's pages in this core's order (its units' tiles, KD pages each); the prefetch's page slots in cb 23..45
    uint32_t a_pages = 0;
    for (uint32_t u = 0; u < n_units; u++) {
        a_pages += get_arg_val<uint32_t>(ua + 3 * u + 1) * KD;
    }
    uint32_t pf_base[PF_N], pf_slots[PF_N], pf_total = 0;
#if M_MAX > 1
    // M rows: no slots in B's input buffers (zeroed once, below: their unused rows must stay zero) nor in buffers
    // smaller than a page (Slot walks only non-empty segments)
    uint32_t seg = 0;
    auto carve = [&](uint32_t cb) {
        const auto& ci = get_local_cb_interface(cb);
        const uint32_t n = ci.fifo_size / W_PAGE;
        if (n == 0 || (cb >= cb_bnw && cb <= cb_bex)) {
            return;
        }
        pf_base[seg] = ci.fifo_limit - ci.fifo_size;
        pf_slots[seg++] = n;
        pf_total += n;
    };
#ifdef CARD_EXPERTS
    // with the card's experts cb 34's slots last: it holds their F until X2 (fused/dexp_cmp.h), so the prefetch
    // spread over E1's reads stays below pf_safe
    for (uint32_t cb = PF_FIRST; cb <= PF_LAST; cb++) {
        if (cb != fused::dexp::cb_f_m) {
            carve(cb);
        }
    }
    const uint32_t pf_safe = pf_total;
    carve(fused::dexp::cb_f_m);
#else
    for (uint32_t cb = PF_FIRST; cb <= PF_LAST; cb++) {
        carve(cb);
    }
#endif
#else
    for (uint32_t k = 0; k < PF_N; k++) {
        const auto& ci = get_local_cb_interface(PF_FIRST + k);
        pf_base[k] = ci.fifo_limit - ci.fifo_size;
        pf_slots[k] = ci.fifo_size / W_PAGE;
        pf_total += pf_slots[k];
    }
#endif
    const uint32_t pf_pages = a_pages < pf_total ? a_pages : pf_total;
    uint32_t cs_base[CS_N], cs_slots[CS_N], cs_total = 0;  // C's page slots in cb 16..22
    for (uint32_t k = 0; k < CS_N; k++) {
        const auto& ci = get_local_cb_interface(CS_FIRST + k);
        cs_base[k] = ci.fifo_limit - ci.fifo_size;
        cs_slots[k] = ci.fifo_size / W_PAGE;
        cs_total += cs_slots[k];
    }
    struct Slot {  // walks the slots in order
        const uint32_t* base;
        const uint32_t* slots;
        uint32_t seg = 0, idx = 0;
        uint32_t addr() const { return base[seg] + idx * W_PAGE; }
        void next() {
            if (++idx == slots[seg]) {
                seg++;
                idx = 0;
            }
        }
    };
    const uint32_t zero = get_write_ptr(cb_zero);
    fused::fill_words(zero, 1024, 0);
#if M_MAX > 1
    // M rows: the step's words; mp, the layout's rows (1, 2, 4)
    uint32_t M, n_keep, parity, mp;
    {
        const uint32_t st = r_tab + 256;
        noc_async_read(TensorAccessor(d_args, step_addr, 64).get_noc_addr(0), st, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
        volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
        M = sw[1], n_keep = sw[2], parity = sw[3], mp = sw[4];  // word 0: the token's position (other programs)
    }
    // the ones tile; with M rows also Sel (hc_cmp.h h1_rs_m: Sel[i][R] = 1 when R % mp == i, i < mp) and the step's
    // words for the compute (tile 2); with the card's experts tile 2 is also their Sel (rows 0..3: row m sums rows
    // 8 m .. 8 m + 7; the words then at word 256, row 16, where its rows only reach rows nobody reads)
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
#ifdef CARD_EXPERTS
        const uint32_t sel_e = sel + 2048;
        fused::fill_words(sel_e, 512, 0);
        volatile tt_l1_ptr uint16_t* se = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sel_e);
        for (uint32_t m = 0; m < 4; m++) {
            for (uint32_t c = 8 * m; c < 8 * m + 8; c++) {
                se[(c / 16) * 256 + m * 16 + c % 16] = 0x3f80;
            }
        }
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_e) + 256;
#else
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel + 2048);
#endif
        t[0] = M, t[1] = n_keep, t[2] = parity, t[3] = mp;
        asm volatile("fence rw, rw" ::: "memory");
    }
    cb_push_back(cb_ones, 3);
    // M rows: the regions placed after a variable-size one (fused/hc_dm.h RegionsM)
    const uint32_t r_xown = r_rs;  // M rows: the RS slot holds xown
    const uint32_t rs_m = r_p1 + mp * 4 * LO * 4;
    const uint32_t r_wa_m = r_loa + mp * 640, r_wf_m = r_lof + mp * 640;
    const uint32_t r_sh_m = r_l + mp * N_ROUTER * 128;
    // region A's blocks on a B core (M rows): q, k, v, z (4 tiles each), the decay's tile, beta's tile
    const uint32_t a_blk = mp * 128, a_dec = r_a + 16 * a_blk, a_bet = a_dec + a_blk;
    const uint32_t r_dprev = r_tab + 384;  // the previous pass's decays (its pending page 1)
    // B's input buffers zeroed once: each pass fills only rows 0..mp-1 (q, k, v, z), 0..n_keep-1 (the kept rows) or
    // 0..3 (the coefficients), and every push fills a whole buffer, so the rest stays zero
    if (head != IDLE) {
        for (uint32_t cb : {cb_bq, cb_bk, cb_bex, cb_bpend, cb_bnw}) {
            const auto& ci = get_local_cb_interface(cb);
            for (uint32_t off = 0; off < ci.fifo_size; off += 4096) {
                fused::l1_copy(zero, ci.fifo_limit - ci.fifo_size + off, 4096);
            }
        }
        noc_async_read_barrier();
    }
#else
    constexpr uint32_t mp = 1;
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
#endif
#ifdef CARD_EXPERTS
    fused::dexp::zero_x(zero);
#endif

    // ---- X into every arena (M rows: each of a stream's 11 H1 cores reads 1/11 of its blocks and sends it to the
    // other 10, then arrives at the first barrier itself (the writer does not); the owners read their tile, complete at
    // their next read barrier)
#if M_MAX > 1
    {
        if (h_item != IDLE) {
            const uint32_t s = h_item & 3, n = h_item >> 2, t0 = 80 * n / 11, t1 = 80 * (n + 1) / 11;
            const uint32_t b32 = mp * 128, b16 = mp * 64, o32 = r_x32 + t0 * b32, o16 = r_x16 + t0 * b16;
            noc_async_read(x32.get_noc_addr(s, t0 * b32), o32, (t1 - t0) * b32);
            noc_async_read(x16.get_noc_addr(s, t0 * b16), o16, (t1 - t0) * b16);
            noc_async_read_barrier();
            for (uint32_t k = 0; k < 11; k++) {
                const uint32_t c = get_arg_val<uint32_t>(h1_cores + 4 * k + s);
                if (k != n) {
                    noc_async_write(o32, get_noc_addr(c & 0xffff, c >> 16, o32), (t1 - t0) * b32);
                    noc_async_write(o16, get_noc_addr(c & 0xffff, c >> 16, o16), (t1 - t0) * b16);
                }
            }
            noc_async_write_barrier();
        }
        grid.arrive(1);
        if (c_tile != IDLE) {
            for (uint32_t s = 0; s < 4; s++) {
                noc_async_read(x32.get_noc_addr(s, c_tile * mp * 128), r_xown + s * mp * 128, mp * 128);
            }
        }
    }
#else
    if (grid.coordinator) {
        noc_async_read(x32.get_noc_addr(0), r_x32, 4 * D * 4);
        noc_async_read(x16.get_noc_addr(0), r_x16, 4 * D * 2);
        noc_async_read_barrier();
    }
#endif
    // PROFILE: the coordinator's wall clock at the start and after each barrier, to the prof tensor at the end
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_prof);
    stamps[0] = get_timestamp_32b();
    auto wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
        grid.wait(n, region, bytes);
        if (n < 15) {
            stamps[n] = get_timestamp_32b();
        }
        stamps[15] = get_timestamp_32b();  // the last barrier's
    };
#ifdef EXCHANGE
    fused::Link link;
    if (grid.coordinator) {
        link.open(H2D_CONFIG, D2H_CONFIG, D * 4, XCHG_OUT_BYTES);
    }
#endif
    auto row_of = [&](uint32_t layer, uint32_t* ad) {  // a layer's tensor addresses: its table row, or the arguments
        if (table) {
            const uint32_t row = r_tab;
            noc_async_read(tab.get_noc_addr(layer), row, 128);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t k = 0; k < L_N; k++) {
                ad[k] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row)[k];
            }
        } else {
            for (uint32_t k = 0; k < L_N; k++) {
                ad[k] = base[k];
            }
        }
    };
    // A's per-tile constants need no activation: into region AC (7 rows of 32 floats per tile, in this core's tile
    // order) while card memory is otherwise idle (before the first barrier, then during B for the next pass); A
    // builds its constant tiles from there. Complete at the next read barrier.
    auto ac_prefetch = [&](const uint32_t* ad) {
        const auto cstate = TensorAccessor(d_args, ad[L_CSTATE], CONV * 4);
        const auto taps = TensorAccessor(d_args, ad[L_TAPS], CONV * 4), dtna = TensorAccessor(d_args, ad[L_DTNA], 256);
        uint32_t slot = 0;
        for (uint32_t u = 0; u < n_units; u++) {
            const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
            const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
            const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
            if (type > T_K && type != T_A) {
                continue;
            }
            for (uint32_t n = first; n < first + count; n++, slot++) {
                const uint32_t d = r_ac + slot * 7 * 128;
                if (type == T_A) {
                    const uint32_t off = (n - CONV / 32 - 192) * 128;  // within the padded alpha block
                    noc_async_read(dtna.get_noc_addr(0, off), d, 128);
                    noc_async_read(dtna.get_noc_addr(1, off), d + 128, 128);
                    continue;
                }
                for (uint32_t j = 0; j < 3; j++) {
#if M_MAX > 1  // rows [n_keep - 1, n_keep + 2) of the parity's history
                    noc_async_read(cstate.get_noc_addr(parity * (2 + M_MAX) + n_keep - 1 + j, n * 128), d + j * 128,
                                   128);
#else
                    noc_async_read(cstate.get_noc_addr(j, n * 128), d + j * 128, 128);
#endif
                }
                for (uint32_t j = 0; j < 4; j++) {
                    noc_async_read(taps.get_noc_addr(j, n * 128), d + (3 + j) * 128, 128);
                }
            }
        }
    };
    uint32_t ad[L_N], nx[L_N];  // this pass's tensors, the next pass's
    uint32_t pre = 0, pre_next = 0;  // A pages in the prefetch slots for this pass, for the next one
    row_of(0, ad);
    ac_prefetch(ad);
    // H1's first weights before the barrier that precedes it (here the first; then R2's of the previous pass)
    uint32_t h1a_pre = fused::hc::prefetch_h1(ws, TensorAccessor(d_args, ad[L_WD_A], W_PAGE), h_item);
    uint32_t epoch = 1;
#if M_MAX > 1
    wait(epoch, 0, 0);
#else
    wait(epoch, r_x32, r_p1 - r_x32);
#endif

    for (uint32_t layer = 0; layer < layers; layer++) {
        // SH_LATE's layers (the SH barrier and gate|up after the page): a pass of 2+ rows, or a program's last layer.
        // A one-row pass's other exchanges also prefetch the next layer's A (45 MB in the host's ~100 us: bound by card
        // memory), where gate|up's 3.5 MB cost what R1 saves and h's own barrier queues behind the prefetch's reads
        // (R2 +10 us), so they keep R1 as it was (h with the logits).
        [[maybe_unused]] const bool sh_layer = layer + 1 == layers || mp > 1;
        [[maybe_unused]] const bool late = sh_late && sh_layer;
        if (layer) {
            row_of(layer, ad);
        }
        pre = pre_next;
        pre_next = 0;
        auto lacc = [&](uint32_t k, uint32_t page) { return TensorAccessor(d_args, ad[k], page); };
        const auto wd_a = lacc(L_WD_A, W_PAGE), wu_a = lacc(L_WU_A, W_PAGE), norm_a = lacc(L_NORM_A, 4 * D * 4);
        const auto wd_f = lacc(L_WD_F, W_PAGE), wu_f = lacc(L_WU_F, W_PAGE), norm_f = lacc(L_NORM_F, 4 * D * 4);
        const auto wa = lacc(L_WA, W_PAGE), S = lacc(L_S, 4096), bnw = lacc(L_BNW, HK * 4);
        const auto wo = lacc(L_WO, W_PAGE), wr = lacc(L_WR, R_PAGE), wgu = lacc(L_WGU, W_PAGE);
        const auto wsd = lacc(L_WSD, W_PAGE), routed = lacc(L_ROUTED, D * 4);
        // ---- the hyper-connection mix (H1 then H2), for either half (fused/hc_dm.h)
        const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
        // A's pages [p0, p1) of tile n (k: the tile's first page in this core's order): from the prefetch slots while
        // k + p < pre (read during the previous pass's exchange), else from card memory
        Slot pf{pf_base, pf_slots};
        auto a_read = [&](uint32_t n, uint32_t k, uint32_t p0, uint32_t p1) {
            if (k + p0 >= pre) {
                ws.read(wa, n * KD + p0, p1 - p0);
                return;
            }
            for (uint32_t p = p0; p < p1; p++) {
                if (k + p < pre) {
                    ws.read_page(get_noc_addr(pf.addr()));
                    pf.next();
                } else {
                    ws.read_page(wa.get_noc_addr(n * KD + p));
                }
            }
        };
        // A's first weights before H2a's barrier (the mix's last: H2's reads are all issued by then), as many pages
        // as the weight buffer holds (the compute drains H2's first); all in the first tile (KD pages)
        const uint32_t h2a_barrier = epoch + 3;
        uint32_t a_pre = 0;
        auto ha_wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
            if (n == h2a_barrier && n_units && get_arg_val<uint32_t>(ua + 1)) {
                a_pre = 4 * ws.batch;
                a_read(get_arg_val<uint32_t>(ua), 0, 0, a_pre);
            }
            wait(n, region, bytes);
        };
#if M_MAX > 1
        const fused::hc::RegionsM hreg_m{r_x32, r_x16, r_xown, r_p1, rs_m, mp};
        fused::hc::read_m(ws, wd_a, wu_a, norm_a, h_item, h_tile, hreg_m, r_hna, r_loa, zero, ha_wait, epoch,
                          h1a_pre);
#else
        fused::hc::read(ws, wd_a, wu_a, norm_a, h_item, h_tile, hreg, r_hna, r_loa, r_wa, zero, ha_wait, epoch,
                        h1a_pre);
#endif

        // ---- A: the input projection with its epilogues
        uint32_t a_tiles = 0;
        for (uint32_t u = 0; u < n_units; u++) {
            a_tiles += get_arg_val<uint32_t>(ua + 3 * u + 1);
        }
        if (a_tiles) {
#if M_MAX > 1
  // its barrier also completes region AC's and the prefetch's reads
            fused::x_tiles_m(cb_x, r_hna, KD, KD, mp);
#else
            fused::x_tiles(cb_x, r_hna, KD, KD);
#endif
            uint32_t slot = 0, k = 0;
            for (uint32_t u = 0; u < n_units; u++) {
                const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
                const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
                const uint32_t type = get_arg_val<uint32_t>(ua + 3 * u + 2);
                for (uint32_t n = first; n < first + count; n++) {
                    a_read(n, k, k ? 0 : a_pre, KD);
                    k += KD;
                    if (type > T_K && type != T_A) {
                        continue;
                    }
                    cb_reserve_back(cb_ac, 7);  // the tile's constants (from region AC), after its weights
                    const uint32_t c = get_write_ptr(cb_ac), src = r_ac + slot++ * 7 * 128;
#if M_MAX > 1
  // the taps (tiles 3..6) or dt, neg_a (0, 1) on every row; the conv's window
                    for (uint32_t j = type == T_A ? 0 : 3; j < (type == T_A ? 2u : 7u); j++) {  // is the writer's
                        for (uint32_t m = 0; m < mp; m++) {
                            fused::l1_copy(src + j * 128, c + j * 4096 + m * 64, 64);
                            fused::l1_copy(src + j * 128 + 64, c + j * 4096 + 1024 + m * 64, 64);
                        }
                    }
#else
                    for (uint32_t j = 0; j < (type == T_A ? 2u : 7u); j++) {
                        fused::row0_from_l1(src + j * 128, c + j * 4096);
                    }
#endif
                    noc_async_read_barrier();
                    cb_push_back(cb_ac, 7);
                }
            }
            ws.flush();
        }
        // B's state and norm weight need no activation: before the barrier
#if M_MAX > 1
        // M rows: the parity's committed state (cb 23); the previous pass's rows to commit, rows 0..n_keep-1 of its
        // pending page (k into cb 59 tiles 0..3, U into 4..7) and its decays; the norm weight on rows 0..mp-1 (cb 25
        // tiles 0..3); the commit's coefficients (tile 4: c_j = d_{j+1} .. d_a on row j < n_keep, a = n_keep - 1, and
        // gamma_a = d_0 .. d_a at word 512, row 16, where U's rows are zero)
        if (head != IDLE) {
            cb_reserve_back(cb_bsx, 16);
            cb_reserve_back(cb_bpend, 8);
            cb_reserve_back(cb_bnw, 5);
            const uint32_t d = get_write_ptr(cb_bsx), p = get_write_ptr(cb_bpend), nw = get_write_ptr(cb_bnw);
            const uint32_t s0 = (parity * NV + head) * 16, pg = S_PEND + (parity * NV + head) * 2;
            for (uint32_t t = 0; t < 16; t++) {
                noc_async_read(S.get_noc_addr(s0 + t), d + t * 4096, 4096);
            }
            for (uint32_t t = 0; t < 8; t++) {  // the page: tile t's face-f rows 0..3 at t * 512 + f * 256
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_read(S.get_noc_addr(pg, t * 512 + f * 256), p + t * 4096 + f * 1024, n_keep * 64);
                }
            }
            noc_async_read(S.get_noc_addr(pg + 1), r_dprev, 64);
            for (uint32_t c = 0; c < 4; c++) {
                fused::row0_from_dram(bnw, 0, c * 128, nw + c * 4096);
            }
            noc_async_read_barrier();
            for (uint32_t c = 0; c < 4; c++) {
                for (uint32_t m = 1; m < mp; m++) {
                    fused::l1_copy(nw + c * 4096, nw + c * 4096 + m * 64, 64);
                    fused::l1_copy(nw + c * 4096 + 1024, nw + c * 4096 + 1024 + m * 64, 64);
                }
            }
            invalidate_l1_cache();
            asm volatile("fence rw, rw" ::: "memory");
            const volatile tt_l1_ptr uint32_t* dp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_dprev);
            const uint32_t cu = nw + 4 * 4096;
            uint32_t c = F_ONE;
            for (uint32_t j = 4; j-- > 0;) {
                fill_row_bits(cu, j, j < n_keep ? c : 0);
                if (j < n_keep) {
                    c = fmul(c, dp[j]);
                }
            }
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cu)[512] = c;
            asm volatile("fence rw, rw" ::: "memory");
            noc_async_read_barrier();
            cb_push_back(cb_bsx, 16);
            cb_push_back(cb_bpend, 8);
            cb_push_back(cb_bnw, 5);
        }
#else
        if (head != IDLE) {
            for (uint32_t cb : {cb_bsx, cb_bsop}) {
                cb_reserve_back(cb, 16);
                const uint32_t d = get_write_ptr(cb);
                for (uint32_t t = 0; t < 16; t++) {
                    noc_async_read(S.get_noc_addr(head * 16 + t), d + t * 4096, 4096);
                }
                noc_async_read_barrier();
                cb_push_back(cb, 16);
            }
            cb_reserve_back(cb_bnw, 4);
            const uint32_t d = get_write_ptr(cb_bnw);
            for (uint32_t c = 0; c < 4; c++) {
                fused::row0_from_dram(bnw, 0, c * 128, d + c * 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb_bnw, 4);
        }
#endif
        wait(++epoch, r_a, 0);  // A's outputs are in the B cores' region A already (flayer_writer.cpp)

        // ---- B: the delta rule and the gated norm, one value head
#if M_MAX > 1
        // M rows: q, k (cb 26, 27), v and sigmoid(z) (cb 28 tiles 0..3, 4..7) on rows 0..mp-1 from region A's blocks;
        // the chunk's coefficients from beta_t and the decays d_t of rows t < M (zero elsewhere): tile 8 row t beta_t,
        // tile 9 row t gamma_t = d_0 .. d_t, tile 10 (t, j < t) -beta_t P_tj, tile 11 (t, j <= t) P_tj, with
        // P_tj = d_{j+1} .. d_t = gamma_t / gamma_j (products of the decays, no ratio)
        if (head != IDLE) {
            cb_reserve_back(cb_bq, 4);
            cb_reserve_back(cb_bk, 4);
            cb_reserve_back(cb_bex, 12);
            const uint32_t q = get_write_ptr(cb_bq), k = get_write_ptr(cb_bk), e = get_write_ptr(cb_bex);
            for (uint32_t t = 0; t < 4; t++) {
                fused::rows_from_l1(r_a + t * a_blk, q + t * 4096, mp);
                fused::rows_from_l1(r_a + (4 + t) * a_blk, k + t * 4096, mp);
                fused::rows_from_l1(r_a + (8 + t) * a_blk, e + t * 4096, mp);         // v
                fused::rows_from_l1(r_a + (12 + t) * a_blk, e + (4 + t) * 4096, mp);  // sigmoid(z)
            }
            noc_async_read_barrier();
            cb_push_back(cb_bq, 4);
            cb_push_back(cb_bk, 4);
            invalidate_l1_cache();
            asm volatile("fence rw, rw" ::: "memory");
            const uint32_t j = head % 32, w0 = (j / 16) * mp * 64 + (j % 16) * 4;  // the head's word in row 0's piece
            uint32_t bt[4], dt[4];
            for (uint32_t m = 0; m < M; m++) {
                bt[m] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_bet + w0 + m * 64);
                dt[m] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_dec + w0 + m * 64);
            }
            volatile tt_l1_ptr uint32_t* cl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e + 10 * 4096);
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e + 11 * 4096);
            uint32_t g = F_ONE;
            for (uint32_t t = 0; t < 4; t++) {
                const bool row = t < M;
                if (row) {
                    g = fmul(g, dt[t]);
                }
                fill_row_bits(e + 8 * 4096, t, row ? bt[t] : 0);
                fill_row_bits(e + 9 * 4096, t, row ? g : 0);
                uint32_t pr = F_ONE;  // P_tj from j = t down
                for (uint32_t i = 4; i-- > 0;) {
                    const bool in = row && i <= t;
                    cl[16 * t + i] = in && i < t ? fmul(bt[t], pr) | 0x80000000 : 0;
                    cp[16 * t + i] = in ? pr : 0;
                    if (in && i) {
                        pr = fmul(pr, dt[i]);
                    }
                }
            }
            asm volatile("fence rw, rw" ::: "memory");
            cb_push_back(cb_bex, 12);
        }
#else
        if (head != IDLE) {
            const uint32_t kh = head % 16;  // value head h reads key head h % 16 (ggml_repeat)
            for (uint32_t cb : {cb_bq, cb_bk}) {  // zero outside row 0: k becomes a column by transpose
                cb_reserve_back(cb, 4);
                const uint32_t d = get_write_ptr(cb);
                for (uint32_t t = 0; t < 4; t++) {
                    fused::l1_copy(zero, d + t * 4096, 4096);
                }
            }
            noc_async_read_barrier();
            const uint32_t q = get_write_ptr(cb_bq), k = get_write_ptr(cb_bk);
            for (uint32_t t = 0; t < 4; t++) {
                fused::row0_from_l1(r_a + (kh * HK + t * 32) * 4, q + t * 4096);
                fused::row0_from_l1(r_a + (2048 + kh * HK + t * 32) * 4, k + t * 4096);
            }
            cb_reserve_back(cb_bex, 10);
            const uint32_t e = get_write_ptr(cb_bex);
            for (uint32_t t = 0; t < 4; t++) {
                fused::row0_from_l1(r_a + (4096 + head * HK + t * 32) * 4, e + t * 4096);        // v
                fused::row0_from_l1(r_a + (CONV + head * HK + t * 32) * 4, e + (4 + t) * 4096);  // sigmoid(z)
            }
            noc_async_read_barrier();
            cb_push_back(cb_bq, 4);
            cb_push_back(cb_bk, 4);
            fused::fill_row0(e + 8 * 4096, r_a + A_BETA + head * 4);
            fused::fill_row0(e + 9 * 4096, r_a + A_ALPHA + head * 4);
            cb_push_back(cb_bex, 10);
        }
#endif
        if (layer + 1 < layers) {  // the next pass's A constants (A is done with region AC; this pass's conv state is
            row_of(layer + 1, nx);  // written: the writer arrived at A's barrier)
            ac_prefetch(nx);
        }
        // C's first weights before the barrier, a buffer's worth, and the rest into page slots carved from A's buffers
        // (cb 16..22: idle from A's barrier to the next pass's A) while B computes and card memory idles; the slots'
        // reads complete at C's first read barrier
        const uint32_t c_pre = fused::tail::oproj_prefetch(ws, wo, c_tile, depth * batch);
        const uint32_t c_slot = c_tile == IDLE ? 0 : (C_SLOTS < cs_total ? C_SLOTS : cs_total);
        Slot cs{cs_base, cs_slots};
        for (uint32_t p = 0; p < c_slot; p++, cs.next()) {
            noc_async_read(wo.get_noc_addr(c_tile * KC + c_pre + p), cs.addr(), W_PAGE);
        }
#if M_MAX > 1
        wait(++epoch, r_b, mp * NV * HK * 2);
#else
        wait(++epoch, r_b, 48 * HK * 2);
#endif

        // ---- C: ssm_out (x streamed with the weights: one output tile per core) and the combine (fused/tail_dm.h
        // oproj_read, its pages from c_pre on read from the slots)
        if (c_tile != IDLE) {
            cs = Slot{cs_base, cs_slots};
            for (uint32_t k = 0; k < KC; k += batch) {
#if M_MAX > 1
                fused::x_tiles_m(cb_cx, r_b + k * mp * 64, batch, batch, mp);
#else
                fused::x_tiles(cb_cx, r_b + k * 64, batch, batch);
#endif
                for (uint32_t p = k < c_pre ? c_pre : k; p < k + batch; p++) {
                    if (p < c_pre + c_slot) {
                        ws.read_page(get_noc_addr(cs.addr()));
                        cs.next();
                    } else {
                        ws.read_page(wo.get_noc_addr(c_tile * KC + p));
                    }
                }
            }
#if M_MAX > 1
            fused::tail::combine_inputs_m(cb_cxs, cb_cw, r_xown, r_wa_m, mp);
#else
            fused::tail::combine_inputs(cb_cxs, cb_cw, r_x32, r_wa, c_tile);
#endif
        }
        const uint32_t h1_pre = fused::hc::prefetch_h1(ws, wd_f, h_item);
        wait(++epoch, r_x32, 0);  // X is in its readers' arenas already (fused/tail_dm.h x_out, h1)

        // ---- the FFN half
        // R1's first weights before H2f's barrier (the mix's last: H2's reads are all issued by then)
        const uint32_t h2f_barrier = epoch + 3;
        uint32_t r1_pre = 0;
        auto hf_wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
            if (n == h2f_barrier) {
                r1_pre = fused::tail::r1_prefetch(ws, rs, wr, wgu, r_item);
            }
            wait(n, region, bytes);
        };
#if M_MAX > 1
        fused::hc::read_m(ws, wd_f, wu_f, norm_f, h_item, h_tile, hreg_m, r_hnf, r_lof, zero, hf_wait, epoch,
                          h1_pre);
#else
        fused::hc::read(ws, wd_f, wu_f, norm_f, h_item, h_tile, hreg, r_hnf, r_lof, r_wf, zero, hf_wait, epoch,
                        h1_pre);
#endif

        // ---- R1 and R2 (fused/tail_dm.h; M rows: its inputs here, on mp-row blocks)
#if M_MAX > 1
        auto r1_inputs_m = [&]() {
            if (r_item != IDLE) {
                fused::x_tiles_m(cb_x, r_hnf, KD, KD, mp);
                if (r_item < N_ROUTER) {
                    rs.read(wr, r_item * KD + r1_pre, KD - r1_pre);
                    rs.flush();
                } else {
                    ws.read(wgu, (r_item - N_ROUTER) * 2 * KD + r1_pre, 2 * KD - r1_pre);
                    ws.flush();
                }
            }
#ifndef CARD_EXPERTS
            if (s_tile != IDLE) {
                ws.read(wsd, s_tile * KR2, KR2);
                ws.flush();
            }
#endif
        };
#ifdef SH_LATE
        if (!late) {
            r1_inputs_m();
        }
#else
        r1_inputs_m();
#endif
        // R2's inputs but the routed sums (its GEMV runs while the host computes; with the card's experts after
        // theirs, which take the weight stream and x first). SH_LATE's layers: h after its own barrier, the rest here
        auto r2_inputs_m = [&]() {
            if (s_tile == IDLE) {
                return;
            }
#ifdef SH_LATE
            if (!sh_layer)
#endif
                fused::x_tiles_m(cb_x, r_sh_m, KR2, KD, mp);
            cb_reserve_back(cb_sgl, 1);  // the shared expert's gate logit (logit 512: tile 16, column 0) by row
            fused::fill_rows(get_write_ptr(cb_sgl), r_l + 16 * mp * 128, 64, mp);
            cb_push_back(cb_sgl, 1);
            fused::tail::combine_inputs_m(cb_sxs, cb_sw, r_xown, r_wf_m, mp);
#ifndef EXCHANGE
            cb_reserve_back(cb_srt, 1);
            fused::rows_from_dram(routed, 0, s_tile * mp * 128, get_write_ptr(cb_srt), mp);
            noc_async_read_barrier();
            cb_push_back(cb_srt, 1);
#endif
        };
#ifdef SH_LATE
        wait(++epoch, r_l, mp * (N_ROUTER * 128 + (sh_layer ? 0 : 640 * 2)));  // SH's layers: the logits alone
#else
        wait(++epoch, r_l, mp * (N_ROUTER * 128 + 640 * 2));
#endif
#ifndef CARD_EXPERTS
#ifndef SH_LATE
        r2_inputs_m();
#endif
#endif
#else
#ifdef CARD_EXPERTS
        fused::tail::r1_read_only(ws, rs, wr, wgu, r_item, r_hnf, r_l, r_sh, wait, epoch, r1_pre);
#elif defined(SH_LATE)
        if (!late) {
            fused::tail::r1_inputs(ws, rs, wr, wgu, r_item, r_hnf, r1_pre);
            fused::tail::r2_prefetch(ws, wsd, s_tile);
        }
        wait(++epoch, r_l, N_ROUTER * 128 + (sh_layer ? 0 : 640 * 2));  // SH's layers: the logits alone
#else
        fused::tail::r1_read(ws, rs, wr, wgu, wsd, r_item, s_tile, r_hnf, r_l, r_sh, wait, epoch, r1_pre);
#endif
#endif
#ifdef EXCHANGE
        // the host link: hn_f (the experts' input, bf16) and the logits out, the routed experts' sum back into RT
        // the next pass's first A pages into the slots while the host computes (the reads only issue: they complete
        // at the reader's next read barrier, before A reads the slots)
        auto a_prefetch = [&]() {
#ifdef NO_A_PREFETCH
            return;
#endif
            if (layer + 1 >= layers || pf_pages == 0) {
                return;
            }
            const auto wn = TensorAccessor(d_args, nx[L_WA], W_PAGE);
            Slot pf{pf_base, pf_slots};
            uint32_t k = 0;
            for (uint32_t u = 0; u < n_units && k < pf_pages; u++) {
                const uint32_t first = get_arg_val<uint32_t>(ua + 3 * u);
                const uint32_t count = get_arg_val<uint32_t>(ua + 3 * u + 1);
                for (uint32_t n = first; n < first + count && k < pf_pages; n++) {
                    for (uint32_t p = 0; p < KD && k < pf_pages; p++, k++, pf.next()) {
                        noc_async_read(wn.get_noc_addr(n * KD + p), pf.addr(), W_PAGE);
                    }
                }
            }
            pre_next = pf_pages;
        };
#endif
#if M_MAX > 1
#ifdef EXCHANGE
        // the host link: the rows' hn_f and logits blocks out from region B, the M rows' routed sums back into it
        auto send_rows = [&]() {
            constexpr uint32_t row_bytes = D * 2 + N_ROUTER * 128;  // a row's hn_f (bf16) and logits (fp32)
            fused::l1_copy(r_hnf, r_b, mp * D * 2);
            fused::l1_copy(r_l, r_b + mp * D * 2, mp * N_ROUTER * 128);
            noc_async_read_barrier();
#ifdef GRID_PROFILE
#ifdef SH_LATE
            fused::Grid::mark(epoch + (sh_layer ? 2 : 1));  // the pages out: the host's part of the exchange's barrier
#else
            fused::Grid::mark(epoch + 1);  // the pages out: the host's part of the next barrier starts
#endif
#endif
            for (uint32_t k = 0; k * XCHG_OUT_BYTES < mp * row_bytes; k++) {
                link.send(r_b + k * XCHG_OUT_BYTES);
            }
        };
#endif
#ifdef CARD_EXPERTS
        // ---- the card's experts at M rows (fused/dexp_dm.h): the host's first reply = the card's routes (X1), E1
        // (X2: h in the units' cores), E2; R2's weights; the A prefetch spread over the card's reads; R2's inputs and
        // the partial rows once every core's E2 has written them (X3); the rows' CPU sums (X4). E1's x needs no route:
        // every core builds it while the host routes (the coordinator once the pages are out). Without the host link
        // (a test) the routes come from card memory and the routed sums from the routed tensor.
        if (grid.coordinator) {
#ifdef EXCHANGE
            send_rows();
            fused::dexp::e1_x(r_hnf, mp, zero);
            link.recv(r_b);
            fused::l1_copy(r_b, r_rq, fused::dexp::RQ_BYTES);
#else
            fused::dexp::e1_x(r_hnf, mp, zero);
            noc_async_read(TensorAccessor(d_args, croutes, fused::dexp::RQ_BYTES).get_noc_addr(0), r_rq,
                           fused::dexp::RQ_BYTES);
#endif
            noc_async_read_barrier();
        } else {
            fused::dexp::e1_x(r_hnf, mp, zero);
        }
        wait(++epoch, r_rq, fused::dexp::RQ_BYTES);
        {
            const fused::dexp::Cache cache{ad[L_GUC], ad[L_GUS], ad[L_DV], ad[L_DS]};
            // the next pass's A pages spread over the card's read batches, each batch's barrier waiting for its share
            // too (a prefetch issued whole would hold the first one): during E1 into the slots before cb 34's, then
            // the rest during E2
            uint32_t pf_n = 0, pf_k = 0;
#if defined(EXCHANGE) && !defined(NO_A_PREFETCH)
            pf_n = layer + 1 < layers ? pf_pages : 0;
#endif
            const auto wn = TensorAccessor(d_args, nx[L_WA], W_PAGE);
            Slot pfw{pf_base, pf_slots};
            auto pf_to = [&](uint32_t limit, uint32_t cnt) {
                for (; cnt && pf_k < limit; cnt--, pf_k++, pfw.next()) {
                    uint32_t u = 0, k = pf_k;
                    while (k >= get_arg_val<uint32_t>(ua + 3 * u + 1) * KD) {
                        k -= get_arg_val<uint32_t>(ua + 3 * u++ + 1) * KD;
                    }
                    noc_async_read(wn.get_noc_addr((get_arg_val<uint32_t>(ua + 3 * u) + k / KD) * KD + k % KD),
                                   pfw.addr(), W_PAGE);
                }
            };
            invalidate_l1_cache();
            const uint32_t n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_rq)[0];
            const uint32_t u1 = fused::dexp::lo(20 * n, me + 1, cores) - fused::dexp::lo(20 * n, me, cores);
            const uint32_t u2 = fused::dexp::lo(80 * n, me + 1, cores) - fused::dexp::lo(80 * n, me, cores);
            const uint32_t lim1 = pf_n < pf_safe ? pf_n : pf_safe;
            const uint32_t p1 = u1 ? (lim1 + 2 * u1 - 1) / (2 * u1) : 0;
            const uint32_t n_card = fused::dexp::e1_read_m(d_args, cache, r_rq, me, cores, [&]() { pf_to(lim1, p1); });
            wait(++epoch, 0, 0);
            const uint32_t p2 = u2 ? (pf_n - pf_k + u2 - 1) / u2 : 0;
            fused::dexp::e2_read_m(d_args, grid, cache, n_card, r_rq, r_a, mp, me, cores, [&]() { pf_to(pf_n, p2); });
            if (s_tile != IDLE) {
                ws.read(wsd, s_tile * KR2, KR2);
                ws.flush();
            }
            pf_to(pf_n, pf_n);
            pre_next = pf_n;
            wait(++epoch, 0, 0);
            r2_inputs_m();
            if (s_tile != IDLE) {
                fused::dexp::part_read_m(d_args, part, n_card, s_tile, mp);
            }
        }
#ifdef EXCHANGE
        if (grid.coordinator) {
            for (uint32_t m = 0; m < M; m++) {
                link.recv(r_b + m * D * 4);
            }
        }
        wait(++epoch, r_b, M * D * 4);
#else
        wait(++epoch, 0, 0);
#endif
#elif defined(EXCHANGE)  // (SH_LATE)
        if (grid.coordinator) {
            send_rows();
        }
        if (late) {
            r1_inputs_m();
        }
        r2_inputs_m();  // before the prefetch: their read barriers would wait for all of its reads
        a_prefetch();
        if (sh_layer) {
            wait(++epoch, r_sh_m, mp * 640 * 2);  // SH: the shared expert's h
            if (s_tile != IDLE) {
                fused::x_tiles_cpu(cb_x, r_sh_m, KR2, KD, mp);
            }
        }
        if (grid.coordinator) {
            for (uint32_t m = 0; m < M; m++) {
                link.recv(r_b + m * D * 4);
            }
        }
        wait(++epoch, r_b, M * D * 4);
#endif
#ifdef EXCHANGE
        if (s_tile != IDLE) {  // row m's routed sum: the reply's row min(m, M - 1), tile s_tile's 32 values
            cb_reserve_back(cb_srt, 1);
            const uint32_t t = get_write_ptr(cb_srt);
            for (uint32_t m = 0; m < mp; m++) {
                const uint32_t src = r_b + (m < M ? m : M - 1) * D * 4 + s_tile * 128;
                fused::l1_copy(src, t + m * 64, 64);
                fused::l1_copy(src + 64, t + 1024 + m * 64, 64);
            }
            noc_async_read_barrier();
            cb_push_back(cb_srt, 1);
        }
#endif
        h1a_pre = 0;
        if (layer + 1 < layers) {
            h1a_pre = fused::hc::prefetch_h1(ws, TensorAccessor(d_args, nx[L_WD_A], W_PAGE), h_item);
        }
        wait(++epoch, r_x32, 0);
        continue;
#endif
#ifdef CARD_EXPERTS
        // ---- the card's experts (fused/dexp_dm.h): its routes (X1), E1 (X2), E2; R2's weights; the partial rows of
        // R2's tile once every core's E2 has written them (X3), while the host still computes; the CPU's sum (X4).
        // Without the host link (a test) the routes come from card memory and the routed sum from the routed tensor.
        if (grid.coordinator) {
#ifdef EXCHANGE
            const uint32_t page = get_write_ptr(cb_xout);
            fused::l1_copy(r_hnf, page, D * 2);
            fused::l1_copy(r_l, page + D * 2, 17 * 128);
            noc_async_read_barrier();
#ifdef GRID_PROFILE
            fused::Grid::mark(epoch + 1);  // the page out: the host's part of the next barrier starts
#endif
            link.send(page);  // the host starts on the experts
            a_prefetch();
            link.recv(r_rt);
            fused::l1_copy(r_rt, r_rq, 128);  // the first reply: the card's routes
#else
            noc_async_read(TensorAccessor(d_args, croutes, 128).get_noc_addr(0), r_rq, 128);
#endif
            noc_async_read_barrier();
        } else {
#ifdef EXCHANGE
            a_prefetch();
#endif
        }
        wait(++epoch, r_rq, 128);
        const fused::dexp::Cache cache{ad[L_GUC], ad[L_GUS], ad[L_DV], ad[L_DS]};
        const uint32_t n_card = fused::dexp::e1_read(d_args, cache, r_rq, r_hnf, zero, me, cores);
        wait(++epoch, r_a, n_card * 640 * 2);  // region H = region A (B and C are done with it)
        fused::dexp::e2_read(d_args, cache, n_card, r_rq, r_a, zero, me, cores);
        fused::tail::r2_prefetch(ws, wsd, s_tile);
        wait(++epoch, 0, 0);
        if (s_tile != IDLE) {
            fused::dexp::part_read(d_args, part, n_card, s_tile);
        }
#ifdef EXCHANGE
        if (grid.coordinator) {
            link.recv(r_rt);
        }
        wait(++epoch, r_rt, D * 4);
#else
        wait(++epoch, r_rt, 0);
#endif
#elif defined(EXCHANGE)
        if (grid.coordinator) {
            const uint32_t page = get_write_ptr(cb_xout);
            fused::l1_copy(r_hnf, page, D * 2);
            fused::l1_copy(r_l, page + D * 2, 17 * 128);
            noc_async_read_barrier();
#ifdef GRID_PROFILE
            fused::Grid::mark(epoch + (sh_layer ? 2 : 1));  // the page out: the host's part of the exchange's barrier
#endif
            link.send(page);  // the host starts on the experts
        }
        if (late) {  // (SH_LATE)
            fused::tail::r1_inputs(ws, rs, wr, wgu, r_item, r_hnf, r1_pre);
            fused::tail::r2_prefetch(ws, wsd, s_tile);
        }
        fused::tail::r2_inputs(s_tile, r_l, r_sh, r_x32, r_wf, !sh_layer);  // before the prefetch: its read
        a_prefetch();                                                        // barrier would wait for all its reads
        if (sh_layer) {
            wait(++epoch, r_sh, 640 * 2);  // SH: the shared expert's h; R2's GEMV runs while the host computes
            if (s_tile != IDLE) {
                fused::x_tiles_cpu(cb_x, r_sh, KR2, KD);
            }
        }
        if (grid.coordinator) {
            link.recv(r_rt);
        }
        wait(++epoch, r_rt, D * 4);
#endif
        // R2's barrier, with the next pass's first H1 weights issued before it (R2's own were issued during R1)
        h1a_pre = 0;
        auto r2_wait = [&](uint32_t n, uint32_t region, uint32_t bytes) {
            if (layer + 1 < layers) {
                h1a_pre = fused::hc::prefetch_h1(ws, TensorAccessor(d_args, nx[L_WD_A], W_PAGE), h_item);
            }
            wait(n, region, bytes);
        };
#if defined(EXCHANGE) && !defined(CARD_EXPERTS)
        fused::tail::r2_routed(routed, r_rt, s_tile, r_x32, r2_wait, epoch, false);  // its other inputs went in before
#elif defined(EXCHANGE)
        fused::tail::r2_read(routed, r_rt, s_tile, r_l, r_sh, r_x32, r_wf, r2_wait, epoch, false);
#else
        fused::tail::r2_read(routed, 0, s_tile, r_l, r_sh, r_x32, r_wf, r2_wait, epoch, false);
#endif
    }
#ifdef EXCHANGE
    if (grid.coordinator) {
        link.close();
    }
#endif
    if (grid.coordinator) {
        noc_async_write(r_prof, prof.get_noc_addr(0), 64);
        noc_async_write_barrier();
    }
#if M_MAX > 1
    noc_async_write_barrier();  // B's states
#endif
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the program's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
