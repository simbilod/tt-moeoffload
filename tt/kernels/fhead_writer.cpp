// Fused decode head writer (fhead_reader.cpp): the hyper-connection mix (fused/hc_dm.h), then row 0 of each LM-head
// output tile (cb 16) to its page of the logits ([7760 x 32] fp32, a page per tile); arrive.
// With HEAD_M (g3_head.Head.program(m, logits, ids)): rows 0..m-1 of each output tile, row r to the logits' pages
// r * 7760 + n (when logits), and (when ids) each row's top 64 on the card:
//   scan  every value of the core's tiles through a min-heap of 64 (key, id) per row (key: the float's bits in
//         unsigned order, -0 as +0; the heap's root the worst: lowest key, then highest id), sorted best first
//   merge cores in groups of G: row r's leader of group g (core G g + r) takes its members' lists, row r's final
//         (core G r + r) the leaders'; each k-way merge picks the best head (higher key, then lower id: llama.cpp's
//         argmax takes the lowest id of a tie); semaphores 3 (members in) and 4 (leaders in) count the lists ready
//   out   row r's page of the ids tensor ([m_max x 128] uint32): ids 0..63 best first (id 0 the argmax), then their
//         logits' bits (-0 comes back as +0)
// Runtime args: grid.h's block; arena offsets X32, X16, P1, RS, HN, LO, W; the checks P1, HN, W; the logits; the
// H1 item, the H2 tile, the LM head's first tile and count; [GRID_PROFILE: the profile]; [HEAD_M: m, flags (1 logits,
// 2 ids), the ids tensor, the leaders of rows 0..3 for this core's group (NOC x | y << 16), the row this core leads
// (0xffffffff none), the final core of that row, its group's size and members, the row it finishes (0xffffffff
// none), the leader count and leaders of that row].
// With X_BLOCKS (fhead_reader.cpp): the mix's outputs as fused/hc_dm.h write_m's (mp-row blocks; the reader arrives at
// the first barrier), m the step tensor's M (its address in m's argument).
// Compile args: a DRAM accessor.
#include <cstdint>

#include "fused/dm.h"
#include "fused/hc_dm.h"

constexpr uint32_t cb_out = 16, cb_arena = 61, cb_stage = 62, D = 2560;

#ifdef HEAD_M
constexpr uint32_t cb_topk = 17, NT = 7760, K = 64, M_MAX = 4, LIST = 8 * K, NONE = 0xffffffff;
// cb 17's regions: each row's list, the led row's merged list, the lists read from other cores, the finished row's
// list and its output page
constexpr uint32_t T_LISTS = 0, T_MERGED = M_MAX * LIST, T_IN = T_MERGED + LIST, T_BEST = T_IN + 16 * LIST;
constexpr uint32_t T_OUT = T_BEST + LIST, T_BYTES = T_OUT + LIST;

inline uint32_t key_of(uint32_t bits) {
    if (bits == 0x80000000u) {
        bits = 0;
    }
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
inline uint32_t bits_of(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key; }
// entry a (key ka, id ia) ranks below b
inline bool worse(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) { return ka < kb || (ka == kb && ia > ib); }

// a min-heap of (key, id) pairs at h (words 2 i, 2 i + 1), n entries, the worst at the root
inline void sift_down(uint32_t* h, uint32_t n, uint32_t i) {
    for (;;) {
        uint32_t w = i;
        const uint32_t l = 2 * i + 1, r = l + 1;
        if (l < n && worse(h[2 * l], h[2 * l + 1], h[2 * w], h[2 * w + 1])) {
            w = l;
        }
        if (r < n && worse(h[2 * r], h[2 * r + 1], h[2 * w], h[2 * w + 1])) {
            w = r;
        }
        if (w == i) {
            return;
        }
        const uint32_t k = h[2 * i], d = h[2 * i + 1];
        h[2 * i] = h[2 * w];
        h[2 * i + 1] = h[2 * w + 1];
        h[2 * w] = k;
        h[2 * w + 1] = d;
        i = w;
    }
}
inline void heap_push(uint32_t* h, uint32_t n, uint32_t key, uint32_t id) {  // n: the count before
    uint32_t i = n;
    while (i > 0) {
        const uint32_t p = (i - 1) / 2;
        if (!worse(key, id, h[2 * p], h[2 * p + 1])) {
            break;
        }
        h[2 * i] = h[2 * p];
        h[2 * i + 1] = h[2 * p + 1];
        i = p;
    }
    h[2 * i] = key;
    h[2 * i + 1] = id;
}
// 32 values of a row (two 16-word pieces) into the heap, ids from id0; ids come in ascending order on a core, so once
// the heap is full a value displaces the root only with a strictly higher key
inline void scan(uint32_t* h, uint32_t& n, volatile tt_l1_ptr uint32_t* a, volatile tt_l1_ptr uint32_t* b,
                 uint32_t id0) {
    for (uint32_t j = 0; j < 32; j++) {
        const uint32_t key = key_of(j < 16 ? a[j] : b[j - 16]);
        if (n < K) {
            heap_push(h, n++, key, id0 + j);
        } else if (key > h[0]) {
            h[0] = key;
            h[1] = id0 + j;
            sift_down(h, K, 0);
        }
    }
}
// the heap at l1 sorted best first (worst out to the end), padded to K with entries below every value; sorted in a
// copy on the stack (L1 loads are slower)
inline void finish_list(uint32_t* l1, uint32_t n) {
    uint32_t h[2 * K];
    for (uint32_t e = 0; e < 2 * n; e++) {
        h[e] = l1[e];
    }
    for (uint32_t e = n; e > 1; e--) {
        const uint32_t k = h[0], d = h[1];
        h[0] = h[2 * (e - 1)];
        h[1] = h[2 * (e - 1) + 1];
        h[2 * (e - 1)] = k;
        h[2 * (e - 1) + 1] = d;
        sift_down(h, e - 1, 0);
    }
    for (uint32_t e = n; e < K; e++) {
        h[2 * e] = 0;
        h[2 * e + 1] = NONE;
    }
    for (uint32_t e = 0; e < 2 * K; e++) {
        l1[e] = h[e];
    }
}
// the best K of nl sorted lists at in (LIST bytes apart) into dst, sorted best first
inline void merge(const uint32_t* in, uint32_t nl, uint32_t* dst) {
    uint32_t at[16], hk[16], hi[16];  // each list's head and its key and id (on the stack: L1 loads are slower)
    for (uint32_t l = 0; l < nl; l++) {
        at[l] = 0;
        hk[l] = in[l * 2 * K];
        hi[l] = in[l * 2 * K + 1];
    }
    for (uint32_t e = 0; e < K; e++) {
        uint32_t b = 0, bk = hk[0], bi = hi[0];
        for (uint32_t l = 1; l < nl; l++) {
            if (worse(bk, bi, hk[l], hi[l])) {
                b = l;
                bk = hk[l];
                bi = hi[l];
            }
        }
        dst[2 * e] = bk;
        dst[2 * e + 1] = bi;
        if (++at[b] < K) {
            hk[b] = in[b * 2 * K + 2 * at[b]];
            hi[b] = in[b * 2 * K + 2 * at[b] + 1];
        } else {
            hk[b] = 0;  // spent: below every entry
            hi[b] = NONE;
        }
    }
}
// nl lists of LIST bytes at address src on the cores xy[0..nl) (NOC x | y << 16) into local dst
inline void gather(uint32_t xy_arg, uint32_t nl, uint32_t src, uint32_t dst) {
    for (uint32_t l = 0; l < nl; l++) {
        const uint32_t xy = get_arg_val<uint32_t>(xy_arg + l);
        noc_async_read(get_noc_addr(xy & 0xffff, xy >> 16, src), dst + l * LIST, LIST);
    }
    noc_async_read_barrier();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() without the compiler's memory barrier)
}
#endif

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const fused::Grid grid = fused::Grid::from_args(0);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg(), r_p1 = arena + arg(), r_rs = arena + arg();
    const uint32_t r_hn = arena + arg(), r_lo = arena + arg(), r_w = arena + arg();
    const auto chk_p1 = acc(4 * 352 * 4), chk_hn = acc(D * 4), chk_w = acc(64), logits = acc(128);
    const uint32_t h_item = arg(), h_tile = arg(), first = arg(), count = arg();
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
#ifdef X_BLOCKS
    const auto step = acc(64);
    noc_async_read(step.get_noc_addr(0), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence rw, rw" ::: "memory");  // (invalidate_l1_cache() has no memory clobber)
    const uint32_t m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[1];
    const uint32_t mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[4];
    const uint32_t flags = arg();
#elif defined(HEAD_M)
    const uint32_t m = arg(), flags = arg();
#endif
#ifdef HEAD_M
    const auto ids_out = acc(LIST);
    const uint32_t lead_arg = i;  // this group's leaders of rows 0..3
    i += M_MAX;
    const uint32_t lead_row = arg(), final_xy = arg(), members = arg(), member_arg = i;
    i += members;
    const uint32_t final_row = arg(), leaders = arg(), leader_arg = i;
    const bool want_logits = flags & 1, want_ids = flags & 2;
    const uint32_t topk = get_write_ptr(cb_topk);
    uint32_t* lists = reinterpret_cast<uint32_t*>(topk + T_LISTS);
    uint32_t filled[M_MAX] = {0, 0, 0, 0};
#endif

    uint32_t epoch = 1;
#ifdef X_BLOCKS
    const fused::hc::RegionsM hreg{r_x32, r_x16, r_rs, r_p1, r_p1 + mp * 4 * fused::hc::LO * 4, mp};
    fused::hc::write_m(grid, stage, chk_p1, chk_hn, chk_w, h_item, h_tile, hreg, r_hn, r_lo, epoch);
#else
    grid.arrive(epoch);
    const fused::hc::Regions hreg{r_x32, r_x16, r_p1, r_rs};
    fused::hc::write(grid, stage, chk_p1, chk_hn, chk_w, h_item, h_tile, hreg, r_hn, r_lo, r_w, epoch);
#endif
#ifndef HEAD_M
    for (uint32_t n = first; n < first + count; n++) {
        fused::stage_rows(cb_out, 1, 64, stage + (n % 4) * 128);
        noc_async_write(stage + (n % 4) * 128, logits.get_noc_addr(n), 128);
        if (n % 4 == 3) {
            noc_async_writes_flushed();  // the staging slots are reused every 4 tiles
        }
    }
#else
    for (uint32_t n = first; n < first + count; n++) {
        cb_wait_front(cb_out, 1);
        const uint32_t t = get_read_ptr(cb_out), slot = stage + (n % 4) * M_MAX * 128;
        if (want_logits) {
            for (uint32_t r = 0; r < m; r++) {
                fused::l1_copy(t + r * 64, slot + r * 128, 64);
                fused::l1_copy(t + 1024 + r * 64, slot + r * 128 + 64, 64);
            }
            noc_async_read_barrier();
        }
        if (want_ids) {
            asm volatile("fence rw, rw" ::: "memory");  // the packer's tile, not stale cache lines
            for (uint32_t r = 0; r < m; r++) {
                scan(lists + r * 2 * K, filled[r], reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + r * 64),
                     reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + 1024 + r * 64), n * 32);
            }
        }
        cb_pop_front(cb_out, 1);
        if (want_logits) {
            for (uint32_t r = 0; r < m; r++) {
                noc_async_write(slot + r * 128, logits.get_noc_addr(r * NT + n), 128);
            }
            if (n % 4 == 3) {
                noc_async_writes_flushed();  // the staging slots are reused every 4 tiles
            }
        }
    }
    if (want_ids) {
        for (uint32_t r = 0; r < m; r++) {
            finish_list(lists + r * 2 * K, filled[r]);
        }
        asm volatile("fence rw, rw" ::: "memory");  // the lists in L1 before a leader is told to read them
        for (uint32_t r = 0; r < m; r++) {
            const uint32_t xy = get_arg_val<uint32_t>(lead_arg + r);
            noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(3)), 1);
        }
        noc_async_atomic_barrier();
        const uint32_t* in = reinterpret_cast<const uint32_t*>(topk + T_IN);
        if (lead_row < m) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3)), members);
            gather(member_arg, members, topk + T_LISTS + lead_row * LIST, topk + T_IN);
            merge(in, members, reinterpret_cast<uint32_t*>(topk + T_MERGED));
            asm volatile("fence rw, rw" ::: "memory");
            noc_semaphore_inc(get_noc_addr(final_xy & 0xffff, final_xy >> 16, get_semaphore(4)), 1);
            noc_async_atomic_barrier();
        }
        if (final_row < m) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4)), leaders);
            gather(leader_arg, leaders, topk + T_MERGED, topk + T_IN);
            uint32_t* best = reinterpret_cast<uint32_t*>(topk + T_BEST);
            merge(in, leaders, best);
            volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(topk + T_OUT);
            for (uint32_t e = 0; e < K; e++) {
                o[e] = best[2 * e + 1];
                o[K + e] = bits_of(best[2 * e]);
            }
            asm volatile("fence rw, rw" ::: "memory");
            noc_async_write(topk + T_OUT, ids_out.get_noc_addr(final_row), LIST);
        }
    }
#endif
    noc_async_write_barrier();
    grid.arrive(++epoch);
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
