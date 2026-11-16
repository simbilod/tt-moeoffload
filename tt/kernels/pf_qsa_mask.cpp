// The prefill attention's QSA mask in one data-movement kernel (2026-11-08, scope item e; g7_prefill.Attn.select's
// ops from the score on, bit for bit): per query row, the pools it sees are the complete ones, p < nvis = (pos + 1) / 4;
// it keeps all of them when nvis <= 512, else those whose score is at least the 512th largest's (exactly: the ttnn
// path's topk gives fp32, select_ops.py THR_EXACT 0 cells apart); its cells, and its own open block's cells up to
// itself (pos % 4 != 3), get 0, every other cell -1e4 (bf16 0xC61C, the typecast of -1e4).
// The 512th value exactly, in integers (the DM cores have no FPU): scores are relu sums, >= 0 (a -0 counts as 0), so
// their fp32 bits order as unsigned integers; a radix select over 11, 11 and 10 bits, a histogram pass each.
// A unit is R rows of a 32-row tile row (R = 8 or 4, compile arg): rows R g .. R g + R - 1 lie in faces f0 = 2 (R g /
// 16) and f0 + 1 from face row R g % 16; it reads only those rows of each score tile (two runs of 64 R bytes) and
// writes those rows of each mask tile (two runs of 32 R bytes), 4 mask tiles per score tile (a pool is 4 cells), a
// pool's decision made once and written as its 4 cells (two 32-bit words), the own block's cells fixed after.
// Runtime args: score address, mask address, the first unit, the unit count, the position of the sub-block's row 0,
// PT (score tiles a row: pools / 32), IDX: the wanted address. Compile args: R, the score's accessor args, the mask's.
// CB 0 (or 3: WRITER) the score rows read, CB 1 (4) the counts, CB 2 (5) a score tile's 4 mask tiles' runs.
// IDX (2026-11-09, the sparse attention): instead of the mask, each row's attended cells as indices for ttnn's
// sparse_sdpa: the kept pools' 4 cells in ascending order, then the own open block's cells up to the row, then
// 0xFFFFFFFF to TOPK (define), one page of TOPK words a row. Exactly 512 pools where the mask keeps more (a tie at the
// 512th score, e.g. when it is 0: the mask keeps every tied pool): those above it, then the lowest-indexed ones equal
// to it, as many as the radix select left (rem), the decode's count (g4_host select_words: argpartition's 512).
// The second output (a 16-word page a row) gets in word 0 the mask's cell count for the row, in word 1 the cells
// written, so the host sees where the two rules differ. Runtime arg 1 is then the indices' address, arg 5 the counts'.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#ifdef WRITER
constexpr uint32_t CB_IN = 3, CB_CNT = 4, CB_OUT = 5, CB_LST = 7;
#else
constexpr uint32_t CB_IN = 0, CB_CNT = 1, CB_OUT = 2, CB_LST = 6;
#endif
constexpr uint32_t NSEL = 512, NB = 16;                 // score tiles a batch
constexpr uint32_t KEEP2 = 0x00000000u, DROP2 = 0xC61CC61Cu, DROP = 0xC61Cu;

#ifdef SIGNED
// GLM's indexer score (sum_h w_h relu(q_h . k), w signed) can be negative: every fp32 to an order-preserving key
// (negatives flipped, positives above them; -0 as +0); the radix select is the same over the whole 32 bits
inline uint32_t key_of(uint32_t b) {
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
#else
inline uint32_t key_of(uint32_t b) { return (b & 0x80000000u) ? 0u : b; }
#endif

void kernel_main() {
    const uint32_t PT = get_arg_val<uint32_t>(5);       // the context's length: a compile arg made a JIT build per length
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t G = 32 / R;                      // units a tile row
    constexpr auto s_args = TensorAccessorArgs<1>();
    constexpr auto m_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto score = TensorAccessor(s_args, get_arg_val<uint32_t>(0), 4096);
#ifdef IDX
    constexpr auto w_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto mask = TensorAccessor(m_args, get_arg_val<uint32_t>(1), TOPK * 4);
    const auto wants = TensorAccessor(w_args, get_arg_val<uint32_t>(6), 64);
#else
    const auto mask = TensorAccessor(m_args, get_arg_val<uint32_t>(1), 2048);
#endif
    const uint32_t u0 = get_arg_val<uint32_t>(2), nu = get_arg_val<uint32_t>(3), pos0 = get_arg_val<uint32_t>(4);
    const uint32_t in = get_write_ptr(CB_IN), outb = get_write_ptr(CB_OUT);
    volatile uint16_t* cnt = reinterpret_cast<volatile uint16_t*>(get_write_ptr(CB_CNT));   // [R][2048]
    const uint32_t LT = 4 * PT;
    constexpr uint32_t IN_T = 2 * 64 * R, OUT_T = 2 * 32 * R;   // bytes a staged score tile, mask tile

    for (uint32_t u = u0; u < u0 + nu; u++) {
        const uint32_t tr = u / G, g = u % G, f0 = 2 * (R * g / 16), fr0 = R * g % 16;
        uint32_t pos[R], nvis[R], rem[R], pre[R];
        bool sel[R];
        bool any = false;
        for (uint32_t r = 0; r < R; r++) {
            pos[r] = pos0 + tr * 32 + R * g + r;
            nvis[r] = (pos[r] + 1) / 4;
            if (nvis[r] > 32 * PT) {
                nvis[r] = 32 * PT;
            }
            sel[r] = nvis[r] > NSEL;
            any |= sel[r];
            rem[r] = NSEL;
            pre[r] = 0;
        }
        // the unit's rows of score tiles j0 .. j0 + n - 1: tile t's rows of face f0 at t IN_T, of f0 + 1 after them
        auto read = [&](uint32_t j0, uint32_t n) {
            for (uint32_t t = 0; t < n; t++) {
                const uint32_t page = tr * PT + j0 + t;
                noc_async_read(score.get_noc_addr(page, f0 * 1024 + fr0 * 64), in + t * IN_T, 64 * R);
                noc_async_read(score.get_noc_addr(page, (f0 + 1) * 1024 + fr0 * 64), in + t * IN_T + 64 * R, 64 * R);
            }
            noc_async_read_barrier();
        };
        // row r of staged tile t: columns 0..15 at lo[0..15], 16..31 at lo[16 R ..] (the second face's run)
        auto row = [&](uint32_t t, uint32_t r) {
            return reinterpret_cast<volatile uint32_t*>(in + t * IN_T + r * 64);
        };
        bool two = false;        // IDX + LIST: the select finished after the output scan, over the list
        constexpr uint32_t SH[3] = {21, 10, 0}, NBIN[3] = {2048, 2048, 1024};
        auto zero = [&](uint32_t nb) {
            for (uint32_t r = 0; r < R; r++) {
                if (sel[r]) {
                    volatile uint32_t* c2 = reinterpret_cast<volatile uint32_t*>(cnt + r * 2048);
                    for (uint32_t b = 0; b < nb / 2; b++) {
                        c2[b] = 0;
                    }
                }
            }
        };
        // the bin holding the rem-th largest, from the top; its count (LIST: whether pass 0's bins fit the lists)
        uint32_t binc[R];
        auto find = [&](uint32_t pass) {
            const uint32_t nb = NBIN[pass];
            for (uint32_t r = 0; r < R; r++) {
                if (!sel[r]) {
                    continue;
                }
                uint32_t b = nb - 1;
                while (true) {
                    const uint32_t c = cnt[r * 2048 + b];
                    if (c >= rem[r] || b == 0) {
                        binc[r] = c;
                        break;
                    }
                    rem[r] -= c;
                    b--;
                }
                pre[r] = (pre[r] << (pass == 2 ? 10 : 11)) | b;
            }
        };
        // a histogram pass over the whole rows (DRAM); f(r, k) per visible key
        auto scan = [&](auto f) {
            for (uint32_t j0 = 0; j0 < PT; j0 += NB) {
                const uint32_t n = PT - j0 < NB ? PT - j0 : NB;
                read(j0, n);
                for (uint32_t t = 0; t < n; t++) {
                    const uint32_t p0 = 32 * (j0 + t);
                    for (uint32_t r = 0; r < R; r++) {
                        if (!sel[r] || p0 >= nvis[r]) {
                            continue;
                        }
                        const uint32_t cmax = nvis[r] - p0 < 32 ? nvis[r] - p0 : 32;
                        volatile uint32_t* lo = row(t, r);
                        for (uint32_t c = 0; c < cmax; c++) {
                            f(r, key_of(c < 16 ? lo[c] : lo[16 * R + c - 16]));
                        }
                    }
                }
            }
        };
        auto count = [&](uint32_t pass) {           // pass 0: every key; later: those under the prefix so far
            const uint32_t sh = SH[pass], mb = NBIN[pass] - 1, up = pass == 0 ? 32 : SH[pass - 1];
            return [=](uint32_t r, uint32_t k) {
                if (pass == 0 || (k >> up) == pre[r]) {
                    cnt[r * 2048 + ((k >> sh) & mb)]++;
                }
            };
        };
        if (any) {
            zero(NBIN[0]);
            scan(count(0));
            find(0);
#ifdef LIST
            // (2026-11-10) pass 0's bin of each row gathered once into an L1 list (in scan order), passes 1 and 2 over
            // the lists: three scans of the rows instead of four; the same keys counted, so the same picks. A unit
            // whose bin overflows a list takes the full passes.
            bool fit = true;
            for (uint32_t r = 0; r < R; r++) {
                fit &= !sel[r] || binc[r] <= LIST;
            }
#ifdef IDX
            two = fit;
            if (!fit)
#else
            if (fit) {
                volatile uint32_t* lists = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_LST));
                uint32_t nl[R];
                for (uint32_t r = 0; r < R; r++) {
                    nl[r] = 0;
                }
                scan([&](uint32_t r, uint32_t k) {
                    if ((k >> 21) == pre[r]) {
                        lists[r * LIST + nl[r]++] = k;
                    }
                });
                for (uint32_t pass = 1; pass < 3; pass++) {
                    zero(NBIN[pass]);
                    auto f = count(pass);
                    for (uint32_t r = 0; r < R; r++) {
                        if (sel[r]) {
                            for (uint32_t i = 0; i < nl[r]; i++) {
                                f(r, lists[r * LIST + i]);
                            }
                        }
                    }
                    find(pass);
                }
            } else
#endif
#endif
            {
                for (uint32_t pass = 1; pass < 3; pass++) {
                    zero(NBIN[pass]);
                    scan(count(pass));
                    find(pass);
                }
            }
        }
#if defined(IDX) && defined(LIST)
        // (2026-11-10) two scans: the output scan keeps the pools above pass 0's bin and gathers the bin's keys and
        // pools (ascending) into the lists; passes 1 and 2 then run over them, and the list's picks (the same keys,
        // the same tie order) are merged into the row's cells
        volatile uint32_t* lk = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_LST));
        volatile uint32_t* lp = lk + R * LIST;
        uint32_t nl[R];
        for (uint32_t r = 0; r < R; r++) {
            nl[r] = 0;
        }
#endif
#ifdef IDX
        uint32_t nout[R], want[R], eq[R];                // eq: the threshold's ties still to take
        for (uint32_t r = 0; r < R; r++) {
            nout[r] = want[r] = 0;
            eq[r] = rem[r];
        }
        volatile uint32_t* lst = reinterpret_cast<volatile uint32_t*>(outb);   // [R][TOPK]
#endif
        // the mask: per score tile j, a pool's decision once (32 bits a row), its 4 cells as two words; mask tile
        // 4 j + q holds pools 32 j + 8 q .. + 7, cells 0..15 (pools 8 q .. 8 q + 3) in face f0's run
        for (uint32_t j0 = 0; j0 < PT; j0 += NB) {
            const uint32_t n = PT - j0 < NB ? PT - j0 : NB;
            read(j0, n);
            for (uint32_t t = 0; t < n; t++) {
                const uint32_t j = j0 + t, p0 = 32 * j;
                for (uint32_t r = 0; r < R; r++) {
                    uint32_t bits = 0;
                    if (p0 < nvis[r]) {
                        const uint32_t cmax = nvis[r] - p0 < 32 ? nvis[r] - p0 : 32;
                        if (!sel[r]) {
                            bits = cmax == 32 ? 0xFFFFFFFFu : ((1u << cmax) - 1u);
                        } else {
                            volatile uint32_t* lo = row(t, r);
                            const uint32_t th = pre[r];
                            for (uint32_t c = 0; c < cmax; c++) {
                                const uint32_t k = key_of(c < 16 ? lo[c] : lo[16 * R + c - 16]);
#ifdef IDX
#ifdef LIST
                                if (two) {
                                    const uint32_t h = k >> 21;
                                    if (h > th) {
                                        want[r] += 4;
                                        bits |= 1u << c;
                                    } else if (h == th) {
                                        lk[r * LIST + nl[r]] = k;
                                        lp[r * LIST + nl[r]++] = p0 + c;
                                    }
                                    continue;
                                }
#endif
                                if (k >= th) {
                                    want[r] += 4;             // the mask's count
                                }
                                if (k > th || (k == th && eq[r] && eq[r]--)) {
                                    bits |= 1u << c;
                                }
#else
                                if (k >= th) {
                                    bits |= 1u << c;
                                }
#endif
                            }
                        }
#ifdef IDX
                        if (!sel[r]) {
                            want[r] += 4 * cmax;
                        }
#endif
                    }
#ifdef IDX
                    for (uint32_t c = 0; c < 32; c++) {                    // the kept pools' cells, ascending
                        if ((bits >> c) & 1u) {
                            const uint32_t cell = 4 * (p0 + c);
                            for (uint32_t e = 0; e < 4; e++) {
                                lst[r * TOPK + nout[r]++] = cell + e;
                            }
                        }
                    }
                    continue;
#endif
                    for (uint32_t q = 0; q < 4; q++) {
                        volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(outb + q * OUT_T);
                        for (uint32_t k = 0; k < 8; k++) {        // pool 8 q + k: cells 4 k .. 4 k + 3
                            const uint32_t v = (bits >> (8 * q + k)) & 1u ? KEEP2 : DROP2;
                            const uint32_t at = (k < 4 ? 0 : 8 * R) + r * 8 + 2 * (k % 4);   // words
                            w[at] = v;
                            w[at + 1] = v;
                        }
                    }
                    const uint32_t own = pos[r] / 4;            // the own open block: its cells up to itself kept
                    if (pos[r] % 4 != 3 && own >= p0 && own < p0 + 32) {
                        const uint32_t col = own - p0, q = col / 8, k = col % 8;
                        volatile uint16_t* h = reinterpret_cast<volatile uint16_t*>(outb + q * OUT_T);
                        for (uint32_t e = 0; e <= pos[r] % 4; e++) {
                            h[(k < 4 ? 0 : 16 * R) + r * 16 + 4 * (k % 4) + e] = 0;
                        }
                    }
                }
#ifdef IDX
                continue;
#endif
                for (uint32_t q = 0; q < 4; q++) {
                    const uint32_t page = tr * LT + 4 * j + q;
                    noc_async_write(outb + q * OUT_T, mask.get_noc_addr(page, f0 * 512 + fr0 * 32), 32 * R);
                    noc_async_write(outb + q * OUT_T + 32 * R, mask.get_noc_addr(page, (f0 + 1) * 512 + fr0 * 32),
                                    32 * R);
                }
                noc_async_write_barrier();
            }
        }
#if defined(IDX) && defined(LIST)
        if (two) {
            for (uint32_t pass = 1; pass < 3; pass++) {
                zero(NBIN[pass]);
                auto f = count(pass);
                for (uint32_t r = 0; r < R; r++) {
                    if (sel[r]) {
                        for (uint32_t i = 0; i < nl[r]; i++) {
                            f(r, lk[r * LIST + i]);
                        }
                    }
                }
                find(pass);
            }
            for (uint32_t r = 0; r < R; r++) {
                if (!sel[r]) {
                    continue;
                }
                const uint32_t th = pre[r];
                uint32_t nb = 0, eqr = rem[r];
                for (uint32_t i = 0; i < nl[r]; i++) {
                    const uint32_t k = lk[r * LIST + i];
                    if (k >= th) {
                        want[r] += 4;
                    }
                    if (k > th || (k == th && eqr && eqr--)) {
                        lp[r * LIST + nb++] = lp[r * LIST + i];
                    }
                }
                volatile uint32_t* o = lst + r * TOPK;      // merge from the back: the above-bin cells, the picks
                uint32_t a = nout[r], w = nout[r] + 4 * nb;
                nout[r] = w;
                while (nb) {
                    const uint32_t p = lp[r * LIST + nb - 1];
                    if (a && o[a - 1] > 4 * p) {
                        o[--w] = o[--a];
                    } else {
                        nb--;
                        for (uint32_t e = 4; e-- > 0;) {
                            o[--w] = 4 * p + e;
                        }
                    }
                }
            }
        }
#endif
#ifdef IDX
        // the own open block's cells up to the row (after every complete pool: ascending), the sentinels, the pages
        for (uint32_t r = 0; r < R; r++) {
            if (pos[r] % 4 != 3) {
                for (uint32_t c = pos[r] / 4 * 4; c <= pos[r]; c++) {
                    lst[r * TOPK + nout[r]++] = c;
                    want[r]++;
                }
            }
            for (uint32_t i = nout[r]; i < TOPK; i++) {
                lst[r * TOPK + i] = 0xFFFFFFFFu;
            }
            const uint32_t row = tr * 32 + R * g + r;
            noc_async_write(outb + r * TOPK * 4, mask.get_noc_addr(row), TOPK * 4);
            volatile uint32_t* wv = reinterpret_cast<volatile uint32_t*>(outb + R * TOPK * 4 + r * 64);
            wv[0] = want[r];
            wv[1] = nout[r];
            noc_async_write(outb + R * TOPK * 4 + r * 64, wants.get_noc_addr(row), 64);
        }
        noc_async_write_barrier();
#endif
        (void)two;
    }
    (void)DROP;
}
