// Fused QSA attention layer, program 2 writer (the phases are described in fattn2_reader.cpp).
//   AP  the partial (cb 27: O 8 tiles, m, l) to card memory; arrive at barrier 2
//   AC  rows 0..11 (the heads) of the gated O tile (cb 38, bf16) to region O; (cb 39, fp32) to O's check; barrier 3
//   oproj, H1f, H2f, R1, (the host link's barrier, -DEXCHANGE), R2: fused/tail_dm.h, fused/hc_dm.h
// Runtime args: grid.h's block; arena offsets X32, X16, P1, RS, O, HNF, LOF, WF, L, SH; the partials, O's check, X1's
// check, P1, HN, W, L, H, X2's checks (X2's check is X's fp32 tensor in the decode loop); AP's role, AC's role,
// oproj's tile, the H1 item, the H2 tile, the R1 item, R2's tile; X's bf16 tensor for the last X (0: none).
// Compile args: a DRAM accessor.
// M rows (-DM_MAX > 1, fattn2_reader.cpp), then H1's 44 cores by item (NOC x | y << 16), the step tensor, region SEL's
// offset: AP's partials k, k + NAP, ... (P from the union's size in the broadcast selection) to their slots; AC's
// gated O tile, q row R = 12 m + h (row m's head h of KV head g), to row m of region O's block of head 12 g + h's dims
// tile d (rows M..mp-1 copies of row M - 1), the fp32 copy to O's check (a head-row a page); C's and R2's X to its
// owner and H1's cores (fused/tail_dm.h x_out_m), R2's also to the X tensors (a stream a page); with -DEXCHANGE the host
// link's barrier between R1's and R2's. Preprocessed out at M_MAX = 1.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"
#include "fused/tail_dm.h"

#ifndef M_MAX
#define M_MAX 1
#endif

constexpr uint32_t cb_aout = 27, cb_cout16 = 38, cb_cchk = 39, cb_arena = 61, cb_stage = 62;
constexpr uint32_t D = 2560, HD = 256, IDLE = 0xffffffff;

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
    const uint32_t r_o = arena + arg(), r_hnf = arena + arg(), r_lof = arena + arg(), r_wf = arena + arg();
    const uint32_t r_l = arena + arg();
    const uint32_t r_sh = arena + arg();
    const auto parts = acc(4096), chk_o = acc(HD * 4), chk_x1 = acc(4 * D * 4);
    const auto chk_p1 = acc(4 * 352 * 4), chk_hn = acc(D * 4), chk_w = acc(64), chk_l = acc(17 * 128);
    const auto chk_h = acc(640 * 4), chk_x2 = acc(4 * D * 4);
    const uint32_t ap = arg(), ac = arg(), o_tile = arg(), h_item = arg(), h_tile = arg(), r_item = arg();
    const uint32_t s_tile = arg(), x16_out = arg();
    const auto out16 = TensorAccessor(d_args, x16_out, 4 * D * 2);
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif

    uint32_t epoch = 1;
    grid.arrive(epoch);

    // ---- AP
    if (ap != IDLE) {
        cb_wait_front(cb_aout, 10);
        const uint32_t t = get_read_ptr(cb_aout);
        for (uint32_t j = 0; j < 10; j++) {
            noc_async_write(t + j * 4096, parts.get_noc_addr(ap * 10 + j), 4096);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_aout, 10);
    }
    grid.arrive(++epoch);

    // ---- AC: row h of the tile is head 12 g + h's dims 32 d .. 32 d + 31
    if (ac != IDLE) {
        const uint32_t g = ac / 8, d = ac % 8;
        cb_wait_front(cb_cout16, 1);
        const uint32_t t = get_read_ptr(cb_cout16);
        for (uint32_t h = 0; h < 12; h++) {
            fused::l1_copy(t + h * 32, stage + h * 64, 32);
            fused::l1_copy(t + 512 + h * 32, stage + h * 64 + 32, 32);
        }
        noc_async_read_barrier();
        cb_pop_front(cb_cout16, 1);
        for (uint32_t h = 0; h < 12; h++) {
            grid.send(stage + h * 64, r_o + ((12 * g + h) * HD + 32 * d) * 2, 64);
        }
        noc_async_write_barrier();
        cb_wait_front(cb_cchk, 1);
        const uint32_t c = get_read_ptr(cb_cchk);
        for (uint32_t h = 0; h < 12; h++) {
            noc_async_write(c + h * 64, chk_o.get_noc_addr(12 * g + h, 32 * d * 4), 64);
            noc_async_write(c + 1024 + h * 64, chk_o.get_noc_addr(12 * g + h, 32 * d * 4 + 64), 64);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_cchk, 1);
    }
    grid.arrive(++epoch);

    fused::tail::oproj_write(grid, stage, o_tile, chk_x1, r_x32, r_x16, epoch);
    const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
    fused::hc::write(grid, stage, chk_p1, chk_hn, chk_w, h_item, h_tile, hreg, r_hnf, r_lof, r_wf, epoch);
#ifdef EXCHANGE
    const bool late = r_item != IDLE && r_item >= fused::tail::N_ROUTER;  // A gate|up core, its h is SH's
    if (late) {
        grid.arrive(++epoch);  // R1: the router's alone
    }
    fused::tail::r1_write(grid, stage, r_item, chk_l, chk_h, r_l, r_sh, epoch);
    if (!late) {
        grid.arrive(++epoch);  // SH
    }
    grid.arrive(++epoch);  // the host link's barrier
#else
    fused::tail::r1_write(grid, stage, r_item, chk_l, chk_h, r_l, r_sh, epoch);
#endif
    fused::tail::r2_write(grid, stage, s_tile, chk_x2, r_x32, r_x16, epoch, x16_out ? &out16 : nullptr);
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
#else
constexpr uint32_t NAP = 32, T_MAX = (12 * M_MAX + 31) / 32, NU_MAX = M_MAX * 513;  // the reader's
constexpr uint32_t PMAX = (NU_MAX + 31) / 32, PSTRIDE = 10 * T_MAX, NH = 24;
namespace ft = fused::tail;

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const fused::Grid grid = fused::Grid::from_args(0);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_xown = arena + arg();
    const uint32_t r_o = arena + arg(), r_hnf = arena + arg(), r_lof = arena + arg();
    arg();  // (WF: after LOF at M rows)
    const uint32_t r_l = arena + arg();
    arg();  // (SH: after L at M rows)
    const auto parts = acc(4096), chk_o = acc(HD * 4), chk_x1 = acc(M_MAX * D * 4);
    const auto chk_p1 = acc(4 * 352 * 4), chk_hn = acc(D * 4), chk_w = acc(64), chk_l = acc(17 * 128);
    const auto chk_h = acc(640 * 4), chk_x2 = acc(M_MAX * D * 4);
    const uint32_t ap = arg(), ac = arg(), o_tile = arg(), h_item = arg(), h_tile = arg(), r_item = arg();
    const uint32_t s_tile = arg(), x16_out = arg();
    const auto out16 = TensorAccessor(d_args, x16_out, M_MAX * D * 2);
    const uint32_t h1_cores = i;  // H1's cores by item, NOC x | y << 16
    i += 44;
    const auto step = acc(64);
    const uint32_t r_sel = arena + arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    noc_async_read(step.get_noc_addr(0), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    const uint32_t M = sw[1], mp = sw[4], T = (12 * M + 31) / 32;
#ifdef NO_CHECKS
    constexpr bool chk = false;
#else
    constexpr bool chk = true;
#endif

    uint32_t epoch = 1;
    grid.arrive(epoch);

    // ---- AP: the partials' tiles to their slots (O_t,d at 8 t + d, m_t at 8 T_MAX + t, l_t at 9 T_MAX + t)
    if (ap != IDLE) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), 1);  // the selection
        invalidate_l1_cache();
        asm volatile("fence rw, rw" ::: "memory");
        const uint32_t U = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sel)[0];
        const uint32_t P = ((U + 7) / 8 + 3) / 4, g = ap / NAP;
        for (uint32_t p = ap % NAP; p < P; p += NAP) {
            const uint32_t slot = (g * PMAX + p) * PSTRIDE;
            for (uint32_t j = 0; j < 10 * T; j++) {
                const uint32_t to = j < 8 * T ? j : j < 9 * T ? 8 * T_MAX + j - 8 * T : 9 * T_MAX + j - 9 * T;
                cb_wait_front(cb_aout, 1);
                noc_async_write(get_read_ptr(cb_aout), parts.get_noc_addr(slot + to), 4096);
                noc_async_writes_flushed();
                cb_pop_front(cb_aout, 1);
            }
        }
    }
    grid.arrive(++epoch);

    // ---- AC (g, t, d): q row R = 32 t + r of the gated O tile to row R / 12 of head 12 g + R % 12's block d of
    // region O
    if (ac != IDLE && ac / 8 % T_MAX < T) {
        const uint32_t g = ac / (8 * T_MAX), t = ac / 8 % T_MAX, d = ac % 8;
        cb_wait_front(cb_cout16, 1);
        const uint32_t c16 = get_read_ptr(cb_cout16);
        for (uint32_t r = 0; r < 32 && 32 * t + r < 12 * M; r++) {
            const uint32_t R = 32 * t + r, m = R / 12, base = r_o + ((12 * g + R % 12) * 8 + d) * mp * 64;
            const uint32_t src = c16 + (r / 16) * 1024 + (r % 16) * 32;
            for (uint32_t k = m; k < (m + 1 == M ? mp : m + 1); k++) {  // (rows M..mp-1: row M - 1's)
                grid.send(src, base + k * 32, 32);
                grid.send(src + 512, base + mp * 32 + k * 32, 32);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_cout16, 1);
        cb_wait_front(cb_cchk, 1);
        const uint32_t c = get_read_ptr(cb_cchk);
        for (uint32_t r = 0; r < 32 && 32 * t + r < 12 * M; r++) {
            const uint32_t R = 32 * t + r, page = (R / 12) * NH + 12 * g + R % 12;
            const uint32_t src = c + (r / 16) * 2048 + (r % 16) * 64;
            noc_async_write(src, chk_o.get_noc_addr(page, 32 * d * 4), 64);
            noc_async_write(src + 1024, chk_o.get_noc_addr(page, 32 * d * 4 + 64), 64);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_cchk, 1);
    }
    grid.arrive(++epoch);

    // ---- oproj + the combine: X's tile to its owner and H1's cores (and the check)
    if (o_tile != IDLE) {
        ft::x_out_m(stage, o_tile, ft::cb_cout32, ft::cb_cout16, r_xown, r_x32, r_x16, h1_cores, mp, chk, chk_x1,
                    static_cast<const decltype(chk_x1)*>(nullptr));
    }
    grid.arrive(++epoch);
    const fused::hc::RegionsM hreg{r_x32, r_x16, r_xown, r_p1, r_p1 + mp * 4 * fused::hc::LO * 4, mp};
    fused::hc::write_m(grid, stage, chk_p1, chk_hn, chk_w, h_item, h_tile, hreg, r_hnf, r_lof, epoch);

    // ---- R1 (logits and the shared expert's h), R2 (+ the combine; X to the X tensors too)
#ifdef EXCHANGE
    const bool late = r_item != IDLE && r_item >= ft::N_ROUTER;  // A gate|up core, its h is SH's
    if (late) {
        grid.arrive(++epoch);  // R1: the router's alone
    }
#endif
    if (r_item != IDLE && r_item < ft::N_ROUTER) {
        fused::stage_rows_m(ft::cb_rlog, 1, 64, stage, mp);
        grid.send(stage, r_l + r_item * mp * 128, mp * 128);
#ifndef NO_CHECKS
        noc_async_write(stage, chk_l.get_noc_addr(0, r_item * mp * 128), mp * 128);
#endif
        noc_async_write_barrier();
    } else if (r_item != IDLE) {
        const uint32_t j = r_item - ft::N_ROUTER;
        fused::stage_rows_m(ft::cb_rh16, 1, 32, stage, mp);
        grid.send(stage, r_l + mp * ft::N_ROUTER * 128 + j * mp * 64, mp * 64);
        noc_async_write_barrier();
#ifdef NO_CHECKS
        cb_wait_front(ft::cb_rh32, 1);
        cb_pop_front(ft::cb_rh32, 1);
#else
        fused::stage_rows_m(ft::cb_rh32, 1, 64, stage, mp);
        noc_async_write(stage, chk_h.get_noc_addr(0, j * mp * 128), mp * 128);
        noc_async_write_barrier();
#endif
    }
    grid.arrive(++epoch);
#ifdef EXCHANGE
    if (!late) {
        grid.arrive(++epoch);  // SH
    }
    grid.arrive(++epoch);  // the host link's barrier
#endif
    if (s_tile != IDLE) {
        ft::x_out_m(stage, s_tile, ft::cb_sout32, ft::cb_sout16, r_xown, r_x32, r_x16, h1_cores, mp, true, chk_x2,
                    x16_out ? &out16 : nullptr);
    }
    grid.arrive(++epoch);
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
#endif
