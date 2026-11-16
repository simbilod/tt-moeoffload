// The MTP draft layer's routed experts on the card, reader (compute fmtp_exp_compute.cpp, writer
// fmtp_exp_writer.cpp). After attention program 2 (g3_attn2.py, M-row build, routed sum 0) for the pass's last row
// (row M - 1): the router's top 10 of 512 and their weights, the 10 experts of hn_f, and X_s += w_f,s y into the
// layer's output residual streams R (fp32 and bf16 [4 x 2560], the head's x).
//   P0  the coordinator reads row M - 1 of program 2's checks: hn_f (fp32, its compute rounds it to bf16), the
//       router's logits and w_f; barrier 1 broadcasts them (region IN)
//   E0  every core: the top 10 (stable, ties to the lower id: integer keys of the fp32 bits) and, on the compute,
//       w_j = exp(l_j - l_0) / sum exp(l_i - l_0) (tt/spec/mtp.py: the softmax renormalized over the top 10)
//   E1  units (j, 32-output tile p), 20 per expert, split evenly: gate column p and up column p, each three groups of
//       32, 32, 16 k-tiles, then h_j = w_j silu(g) u to region H of the coordinator; barrier 2 broadcasts H
//   E2  output tile c (80 cores): sum over the 10 experts of down's tile c (one 20-k-tile group each, the packer's L1
//       accumulation), then X_s += w_f,s y (fused/tail_cmp.h's combine) from program 2's X (row M - 1)
// The experts in card form (g6_mtp.py card_experts): Q2_0's codes as bfp8_b tiles of (code - 1) (integers -1..2:
// exact), its fp16 scales as fp32 tiles (row b = k-tile b's scale per output column: dexp_*'s groups), all 512 on the
// card: gate|up values [512 x 40 columns][80 k-tiles], scales [512 x 40][3], down values [512 x 80][20], scales
// [512 x 80][1].
// Runtime args: grid.h's block; step (word 1 M, word 4 mp), program 2's L, HN, W checks and X fp32 (a stream a page);
// gate|up values, scales, down values, scales; this core's index, the core count, E2's tile (or IDLE).
// Compile args: M_MAX, a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_f = 0, cb_x = 1, cb_ones = 2, cb_s = 3, cb_lg = 4, cb_xd = 9, cb_xs = 11, cb_meta = 12;
constexpr uint32_t cb_wf = 13, cb_hf = 16, cb_hb = 17, cb_zero = 60, cb_arena = 61, cb_stage = 62;
constexpr uint32_t D = 2560, FF = 640, TOPK = 10, F_PAGE = 1088, TILE = 4096, IDLE = 0xffffffff;
constexpr uint32_t IN_HN = 0, IN_LG = 5120, IN_WF = 7168, IN_BYTES = 7232, H_OFF = 8192;  // arena regions

constexpr uint32_t DEPTH = 4, E2_PRE = 4;  // groups in flight; E2's groups issued before barrier 2 (their pages fit)
constexpr uint32_t T_LOCAL = 15;           // the local copies' transaction id (their barrier waits for nothing else)

// A ring of groups in flight: group = one scale tile (cb s) + nk value tiles (cb f), read under one transaction id
// (1 .. DEPTH), pushed when it lands; up to DEPTH groups outstanding (the buffers' free pages permitting: a group is
// issued only once its pages are free, so a prefetch across a barrier never overruns the compute).
template <uint32_t DEPTH>
struct Ring {
    uint32_t cs, cf, ps, pf;                   // the CBs and their page sizes
    uint32_t n = 0, head = 0, s_out = 0, f_out = 0, nk[DEPTH], landed = 0;  // landed: groups pushed so far

    static uint32_t wrap(uint32_t cb, uint32_t a) {
        const auto& ci = get_local_cb_interface(cb);
        return a >= ci.fifo_limit ? a - ci.fifo_size : a;
    }
    void land() {
        noc_async_read_barrier_with_trid(1 + head);
        cb_push_back(cs, 1);
        for (uint32_t p = 0; p < nk[head]; p++) {
            cb_push_back(cf, 1);  // one at a time: cb_push_back wraps only when it lands exactly on the end
        }
        s_out -= 1;
        f_out -= nk[head];
        head = (head + 1) % DEPTH;
        n--;
        landed++;
    }
    void land_until(uint32_t k) {  // groups 0 .. k - 1 pushed (k <= the groups issued)
        while (landed < k) {
            land();
        }
    }
    template <typename SA, typename FA>
    void issue(const SA& sa, uint32_t sp, const FA& fa, uint32_t f0, uint32_t k) {
        while (n && (n == DEPTH || s_out + 1 > get_local_cb_interface(cs).fifo_num_pages ||
                     f_out + k > get_local_cb_interface(cf).fifo_num_pages)) {
            land();  // (a reservation beyond the buffer would never be granted)
        }
        cb_reserve_back(cs, s_out + 1);
        cb_reserve_back(cf, f_out + k);
        const uint32_t slot = (head + n) % DEPTH;
        noc_async_read_set_trid(1 + slot);
        noc_async_read(sa.get_noc_addr(sp), wrap(cs, get_write_ptr(cs) + s_out * ps), ps);
        uint32_t a = wrap(cf, get_write_ptr(cf) + f_out * pf);
        for (uint32_t p = 0; p < k; p++) {
            noc_async_read(fa.get_noc_addr(f0 + p), a, pf);
            a = wrap(cf, a + pf);
        }
        noc_async_read_set_trid(0);
        nk[slot] = k;
        n++;
        s_out += 1;
        f_out += k;
    }
    void drain() {
        while (n) {
            land();
        }
    }
};

// all n tiles of cb zero, before the program's first barrier (x_rows writes the same rows every time after that)
inline void zero_cb(uint32_t cb, uint32_t n, uint32_t zero) {
    const auto& ci = get_local_cb_interface(cb);
    for (uint32_t k = 0; k < n; k++) {
        fused::l1_copy(zero, ci.fifo_limit - ci.fifo_size + k * 2048, 2048);
    }
    noc_async_read_barrier();
}

// n bf16 x tiles from the bf16 vector at L1 src into cb (zeroed by zero_cb), k-tile k in row k % 32
// (the copies under transaction id T_LOCAL: their barrier does not wait for the weight reads in flight)
inline void x_rows(uint32_t cb, uint32_t src, uint32_t n) {
    cb_reserve_back(cb, n);
    const uint32_t xt = get_write_ptr(cb);
    noc_async_read_set_trid(T_LOCAL);
    for (uint32_t k = 0; k < n; k++) {
        const uint32_t r = k % 32, row = xt + k * 2048 + (r < 16 ? 0 : 1024) + (r % 16) * 32;
        fused::l1_copy(src + k * 64, row, 32);
        fused::l1_copy(src + k * 64 + 32, row + 512, 32);
    }
    noc_async_read_barrier_with_trid(T_LOCAL);
    noc_async_read_set_trid(0);
    cb_push_back(cb, n);
}

void kernel_main() {
#ifdef GRID_PROFILE
    fused::Grid::mark(0);  // the program's start
#endif
    constexpr uint32_t m_max = get_compile_time_arg_val(0);
    constexpr auto d_args = TensorAccessorArgs<1>();
    uint32_t i = 0;
    const fused::Grid grid = fused::Grid::from_args(i);
    i += fused::Grid::args(grid.nrects);
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    const auto step = TensorAccessor(d_args, arg(), 64);
    const auto chk_l = TensorAccessor(d_args, arg(), m_max * 17 * 128);
    const auto chk_hn = TensorAccessor(d_args, arg(), m_max * D * 4);
    const auto chk_w = TensorAccessor(d_args, arg(), m_max * 64);
    const auto x32 = TensorAccessor(d_args, arg(), m_max * D * 4);
    const auto gu_vals = TensorAccessor(d_args, arg(), F_PAGE), gu_scales = TensorAccessor(d_args, arg(), TILE);
    const auto d_vals = TensorAccessor(d_args, arg(), F_PAGE), d_scales = TensorAccessor(d_args, arg(), TILE);
    const uint32_t v0 = arg(), v1 = arg(), col = arg(), rep = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
    const uint32_t zero = get_write_ptr(cb_zero), stage = get_write_ptr(cb_stage);
    const uint32_t in = get_write_ptr(cb_arena), h = in + H_OFF;

    fused::fill_words(zero, 1024, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 512, 0x3f803f80);
    cb_push_back(cb_ones, 1);
    noc_async_read(step.get_noc_addr(rep), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    const uint32_t row = sw[1] - 1, mp = sw[4];

    // ---- P0 (the coordinator): row M - 1's logits, w_f and hn_f (M-row blocks: face 0's piece, then face 1's)
    if (grid.coordinator) {
        for (uint32_t r = 0; r < 16; r++) {
            noc_async_read(chk_l.get_noc_addr(0, r * mp * 128 + row * 64), in + IN_LG + r * 128, 64);
            noc_async_read(chk_l.get_noc_addr(0, r * mp * 128 + mp * 64 + row * 64), in + IN_LG + r * 128 + 64, 64);
        }
        noc_async_read(chk_w.get_noc_addr(0, row * 64), in + IN_WF, 64);
        cb_reserve_back(cb_hf, 3);
        const uint32_t hf = get_write_ptr(cb_hf);
        fused::l1_copy(zero, hf + D * 4, 2048);
        for (uint32_t t = 0; t < D / 32; t++) {
            noc_async_read(chk_hn.get_noc_addr(0, t * mp * 128 + row * 64), hf + t * 128, 64);
            noc_async_read(chk_hn.get_noc_addr(0, t * mp * 128 + mp * 64 + row * 64), hf + t * 128 + 64, 64);
        }
        noc_async_read_barrier();
        cb_push_back(cb_hf, 3);
        cb_wait_front(cb_hb, 3);  // hn_f in bf16 (the compute's packer)
        fused::l1_copy(get_read_ptr(cb_hb), in + IN_HN, D * 2);
        noc_async_read_barrier();
        cb_pop_front(cb_hb, 3);
    }
    auto prepare = [&]() {  // the x and logit buffers zeroed, before the barrier but after the coordinator's P0
        zero_cb(cb_x, D / 32, zero);
        zero_cb(cb_xd, 2 * FF / 32, zero);
        cb_reserve_back(cb_lg, TOPK);
        for (uint32_t j = 0; j < TOPK; j++) {
            fused::l1_copy(zero, get_write_ptr(cb_lg) + j * TILE, TILE);
        }
        noc_async_read_barrier();
    };
    if (!grid.coordinator) {
        prepare();
    }
    grid.wait(1, in, IN_BYTES);
    if (grid.coordinator) {
        prepare();
    }

    // ---- E0: the top 10 of the 512 logits (keys: the fp32 bits made monotone as unsigned), stable
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");
    volatile tt_l1_ptr uint32_t* lg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + IN_LG);
    uint32_t key[TOPK], ex[TOPK], cnt = 0;
    for (uint32_t e = 0; e < 512; e++) {
        const uint32_t b = lg[e], k = (b & 0x80000000) ? ~b : (b | 0x80000000);
        if (cnt == TOPK && k <= key[TOPK - 1]) {
            continue;
        }
        uint32_t p = cnt < TOPK ? cnt++ : TOPK - 1;
        while (p > 0 && key[p - 1] < k) {
            key[p] = key[p - 1];
            ex[p] = ex[p - 1];
            p--;
        }
        key[p] = k;
        ex[p] = e;
    }
    cb_reserve_back(cb_meta, 1);  // -l_0's bits for the compute
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = lg[ex[0]] ^ 0x80000000;
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb_meta, 1);
    const uint32_t lt = get_write_ptr(cb_lg);
    for (uint32_t j = 0; j < TOPK; j++) {
        fused::fill_row0(lt + j * TILE, in + IN_LG + ex[j] * 4);
    }
    cb_push_back(cb_lg, TOPK);

    // ---- E1: per unit, gate column p then up column p, three groups each (a scale tile and its value tiles), up
    // to DEPTH groups in flight; then E2's first groups (their weights need only the routes), before barrier 2
    x_rows(cb_x, in + IN_HN, D / 32);
    Ring<DEPTH> ring{cb_s, cb_f, TILE, F_PAGE};
    const uint32_t e1_groups = 6 * (v1 - v0);
    for (uint32_t v = v0; v < v1; v++) {
        const uint32_t e = ex[v / 20], p = v % 20;
        for (uint32_t c : {p, 20 + p}) {
            for (uint32_t g = 0; g < 3; g++) {
                ring.issue(gu_scales, (e * 40 + c) * 3 + g, gu_vals, (e * 40 + c) * 80 + 32 * g, g == 2 ? 16 : 32);
            }
        }
    }
    auto e2_group = [&](uint32_t j) {
        ring.issue(d_scales, ex[j] * 80 + col, d_vals, (ex[j] * 80 + col) * 20, FF / 32);
    };
    if (col != IDLE) {
        for (uint32_t j = 0; j < E2_PRE; j++) {
            e2_group(j);
        }
    }
    ring.land_until(e1_groups);  // E1's all pushed before the barrier (it waits for every core's h)
    grid.wait(2, h, TOPK * FF * 2);

    // ---- E2: down's tile col of every expert (x = h_j), then program 2's X (row M - 1) and w_f across row 0
    if (col != IDLE) {
        for (uint32_t j = 0; j < TOPK; j++) {
            x_rows(cb_xd, h + j * FF * 2, FF / 32);
            if (j + E2_PRE < TOPK) {
                e2_group(j + E2_PRE);
            }
            ring.land_until(e1_groups + j + 1);  // expert j's group in before its next x tiles need the buffer
        }
    }
    ring.drain();
    if (col != IDLE) {
        cb_reserve_back(cb_xs, 4);
        cb_reserve_back(cb_wf, 4);
        const uint32_t xs = get_write_ptr(cb_xs), wf = get_write_ptr(cb_wf);
        for (uint32_t s = 0; s < 4; s++) {
            noc_async_read(x32.get_noc_addr(s, col * mp * 128 + row * 64), xs + s * TILE, 64);
            noc_async_read(x32.get_noc_addr(s, col * mp * 128 + mp * 64 + row * 64), xs + s * TILE + 1024, 64);
            fused::fill_row0(wf + s * TILE, in + IN_WF + s * 4);
        }
        noc_async_read_barrier();
        cb_push_back(cb_xs, 4);
        cb_push_back(cb_wf, 4);
    }
#ifdef GRID_PROFILE
    fused::Grid::mark(GRID_PROFILE_N - 1);  // the reader's end
    fused::profile_dump(d_args, gp_addr, gp_slot, true);
#endif
}
