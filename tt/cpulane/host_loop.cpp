// introselect (below) is ported from NumPy 1.26's numpy/core/src/npysort/selection.cpp, Copyright (c) 2005-2023,
// NumPy Developers, BSD 3-Clause License (LICENSES/numpy-BSD-3-Clause.txt).
// The host's side of one traced decode token (G4): every exchange of the card's programs, in launch order, in
// one call. The card's coordinator sends a D2H page and waits for the H2D reply (fused/link.h); the host:
//
//   lane exchange (kind 0)     page = hn_f bf16 [D] + the router's logits fp32 [NX]: cl_exchange (routing, the CPU
//                              lane's 10 experts), the reply their sum fp32 [D]       (g4_model.Link.serve)
//   card + lane (kind 2)       the same page; the first reply the card's routes (resident experts, at most card_max, in
//                              routing order: words 0, 1..10, 11..20), the second the lane's sum of the rest
//                                                                                     (Link.serve_card)
//   selection (kind 1)         page = attention program 1's pool scores fp32, 2048 a page up to the token's block (up
//                              to 32 pages at 256K cells); the reply program 2's selection words: the top min(512,
//                              complete pools), the own block (g4_model.select_words)
//   M-row exchange (kind 3)    a verify pass's DeltaNet layer (M = hl_rows' rows): the pages = the rows' hn_f, then
//                              their logits, in mp-row blocks (g4_model.unblock_m); each row routed as cl_exchange
//                              does, cl_moe_m for the M rows, reply m row m's sum   (Link.serve_m)
//   M-row selection (kind 4)   a verify pass's attention program 1 (M rows at pos .. pos + M - 1): per row m its pool
//                              scores as kind 1's at position pos + m (its pages), the reply the
//                              rows' selections as program 2's union words (g3_attn2.union_words: word 0 the union's
//                              size, from word 16 each pool | its cell x row mask << 16, ascending)
//                                                                                     (Link.serve_select_m)
//   M-row card + lane (kind 5) kind 3's pages; the first reply the card's routes for the rows (its cached experts among
//                              the rows' routes, first seen first, at most card_max; each row's weight for each), then
//                              the lane's sums of the rest, reply m row m's       (Link.serve_m_card)
//
// with the same buffers and the same arithmetic as the Python path, so every reply is the same to the bit (checked by
// tt/flashnext/g4_host_loop.py). The sockets are ttnn's own (tt-metal's D2HSocket / H2DSocket C++ objects behind the
// Python ones): their read() and write() are called directly, member functions passed in as addresses with the
// objects' pointers (g4_model.Link.bind_loop). In the test I/O the pages come from an array and the replies go to one.
//
// The selection's top 512 is numpy 1.26's np.argpartition (introselect, numpy/core/src/npysort/selection.cpp) ported
// for float32 with arg = true: the same pools in the same order, so the card's attention sums them in the same order.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

#include <time.h>

#include "mrows.h"

extern "C" {
void cl_route(const float* logits, int k, int32_t* ids, float* w);
int cl_moe(int layer, const float* x, int k, const int32_t* ids, const float* w, float* out);
int cl_exchange(int layer, const uint8_t* page, float* reply, int32_t* ids_out, float* w_out);
int cl_moe_m(int layer, int m, const float* x, int k, const int32_t* ids, const float* w, float* out);
int cl_warm(int layer, int n, const int32_t* ids);
void cl_top_k(const float* score, int n, int k, uint32_t* out);   // hc.cpp: a radix select
}

namespace {

constexpr int D = 2560, NX = 512, LAYERS = 48, K = 10;
constexpr int OUT_PAGE = 8192, IN_PAGE = D * 4;  // g4_model.OUT_PAGE, IN_PAGE
constexpr int NSEL = 512, NPOOL = 65536, MAXITEMS = 256;  // the most of g3_attn1.NPOOL: 256K cells (TT_CTX)
constexpr int PPR = NPOOL * 4 / OUT_PAGE;  // a score row's most pages
constexpr int MAXM = 4, ROW_OUT = 2 * D + 17 * 128;  // g4_model.M_MAX, ROW_OUT: an M-row exchange
constexpr int SEL_W = (16 + MAXM * 513 + 15) / 16 * 16;  // g3_attn2.m_sizes(4)["SEL_W"]: the union words
static_assert(SEL_W * 4 <= IN_PAGE, "the union words fit a reply page");

// tt::tt_metal::distributed::D2HSocket::read(void*, uint32_t, bool), H2DSocket::write(void*, uint32_t): this first
using ReadFn = void (*)(void* self, void* data, uint32_t num_pages, bool notify_sender);
using WriteFn = void (*)(void* self, void* data, uint32_t num_pages);

struct Io {
    void* d2h = nullptr;
    void* h2d = nullptr;
    ReadFn read = nullptr;
    WriteFn write = nullptr;
} io;

// the test I/O: pages[i] in, replies appended
struct TestIo {
    const uint8_t* pages = nullptr;
    uint8_t* replies = nullptr;
    int n_pages = 0, max_replies = 0, next_page = 0, next_reply = 0;
} tio;

void test_read(void*, void* data, uint32_t, bool) {
    if (tio.next_page >= tio.n_pages) throw std::runtime_error("test I/O: out of pages");
    std::memcpy(data, tio.pages + (size_t) tio.next_page++ * OUT_PAGE, OUT_PAGE);
}

void test_write(void*, void* data, uint32_t) {
    if (tio.next_reply >= tio.max_replies) throw std::runtime_error("test I/O: out of reply slots");
    std::memcpy(tio.replies + (size_t) tio.next_reply++ * IN_PAGE, data, IN_PAGE);
}

struct Item {
    int kind, layer;
};
Item plan[MAXITEMS];
int n_items = 0;
const int32_t* slot_of[LAYERS] = {};  // per layer the expert -> card slot map (-1 CPU only), or null
int card_max = 0;
// hl_route_log: every M-row exchange's routes appended (the card's own routing, to simulate the cache on):
// records of 2 + MAXM K int16 (layer, M, the rows' K experts each, -1 past M), up to route_cap records
int16_t* route_buf = nullptr;
int64_t route_cap = 0, route_n = 0;
int card_balance = 0;                                   // hl_card_costs: kind 5 balances the card and the lane
float c_lane_exp = 11.6f, c_lane_use = 1.6f, c_card_exp = 15.1f, c_card_fix = 41.f;   // us (M = 4 fits)
int rows_m = 1;  // kind 3's rows

alignas(64) uint8_t page[PPR * MAXM * OUT_PAGE];  // an M-row selection's up to PPR pages a row (row m at PPR m)
static_assert(PPR * MAXM >= 4, "the page buffer holds an M-row exchange's up to 4 pages");
alignas(64) float out_m[MAXM][D];  // an M-row exchange's replies
alignas(64) uint8_t reply[IN_PAGE];  // one buffer for every reply, as Link's: what a reply leaves stays for the next
char error[512];

inline int64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);  // Python's time.perf_counter()
    return (int64_t) t.tv_sec * 1000000000 + t.tv_nsec;
}

// numpy's introselect for argpartition (arg = true) on float32, pivots off (one kth: the stack only records)
using I = intptr_t;
inline bool lt(float a, float b) { return a < b || (b != b && a == a); }  // npy::float_tag::less: NaN last

void introselect(const float* v, I* ts, I num, I kth);

void median3_swap(const float* v, I* ts, I low, I mid, I high) {
    if (lt(v[ts[high]], v[ts[mid]])) std::swap(ts[high], ts[mid]);
    if (lt(v[ts[high]], v[ts[low]])) std::swap(ts[high], ts[low]);
    if (lt(v[ts[low]], v[ts[mid]])) std::swap(ts[low], ts[mid]);
    std::swap(ts[mid], ts[low + 1]);
}

I median5(const float* v, I* ts) {
    if (lt(v[ts[1]], v[ts[0]])) std::swap(ts[1], ts[0]);
    if (lt(v[ts[4]], v[ts[3]])) std::swap(ts[4], ts[3]);
    if (lt(v[ts[3]], v[ts[0]])) std::swap(ts[3], ts[0]);
    if (lt(v[ts[4]], v[ts[1]])) std::swap(ts[4], ts[1]);
    if (lt(v[ts[2]], v[ts[1]])) std::swap(ts[2], ts[1]);
    if (lt(v[ts[3]], v[ts[2]])) return lt(v[ts[3]], v[ts[1]]) ? 1 : 3;
    return 2;
}

void unguarded_partition(const float* v, I* ts, float pivot, I* ll, I* hh) {
    for (;;) {
        do (*ll)++;
        while (lt(v[ts[*ll]], pivot));
        do (*hh)--;
        while (lt(pivot, v[ts[*hh]]));
        if (*hh < *ll) break;
        std::swap(ts[*ll], ts[*hh]);
    }
}

I median_of_median5(const float* v, I* ts, I num) {
    const I nmed = num / 5;
    for (I i = 0, sub = 0; i < nmed; i++, sub += 5) std::swap(ts[sub + median5(v, ts + sub)], ts[i]);
    if (nmed > 2) introselect(v, ts, nmed, nmed / 2);
    return nmed / 2;
}

void dumb_select(const float* v, I* ts, I num, I kth) {
    for (I i = 0; i <= kth; i++) {
        I mi = i;
        float mv = v[ts[i]];
        for (I k = i + 1; k < num; k++)
            if (lt(v[ts[k]], mv)) {
                mi = k;
                mv = v[ts[k]];
            }
        std::swap(ts[i], ts[mi]);
    }
}

void introselect(const float* v, I* ts, I num, I kth) {
    I low = 0, high = num - 1;
    if (kth - low < 3) {
        dumb_select(v, ts + low, high - low + 1, kth - low);
        return;
    }
    if (kth == num - 1) {
        I mi = low;
        float mv = v[ts[low]];
        for (I k = low + 1; k < num; k++)
            if (!lt(v[ts[k]], mv)) {
                mi = k;
                mv = v[ts[k]];
            }
        std::swap(ts[kth], ts[mi]);
        return;
    }
    int depth = 0;
    for (uint64_t u = num; u >>= 1;) depth++;
    depth *= 2;
    while (low + 1 < high) {
        I ll = low + 1, hh = high;
        if (depth > 0 || hh - ll < 5) {
            median3_swap(v, ts, low, low + (high - low) / 2, high);
        } else {
            const I mid = ll + median_of_median5(v, ts + ll, hh - ll);
            std::swap(ts[mid], ts[low]);
            ll--;
            hh++;
        }
        depth--;
        unguarded_partition(v, ts, v[ts[low]], &ll, &hh);
        std::swap(ts[low], ts[hh]);
        if (hh >= kth) high = hh - 1;
        if (hh <= kth) low = ll;
    }
    if (high == low + 1 && lt(v[ts[high]], v[ts[low]])) std::swap(ts[high], ts[low]);
}

// select_words: pool ids in words[:512], nsel at 512, own_valid at 513, the rest of words[:528] zero
void select_words(int pos, const float* scores, uint32_t* words, int* nsel_out, int* own_out) {
    const int complete = std::min((pos + 1) / 4, NPOOL), own = pos / 4;
    static uint32_t top[NPOOL];
    int n;
    if (complete <= NSEL) {
        for (n = 0; n < complete; n++) top[n] = n;
    } else {
        static float neg[NPOOL];
        static I ts[NPOOL];
        for (int i = 0; i < complete; i++) {
            neg[i] = -scores[i];
            ts[i] = i;
        }
        introselect(neg, ts, complete, NSEL - 1);
        for (n = 0; n < NSEL; n++) top[n] = (uint32_t) ts[n];
    }
    int own_valid;
    if (pos % 4 != 3) {
        own_valid = pos % 4 + 1;
    } else {
        int hit = -1;
        for (int i = 0; i < n && hit < 0; i++)
            if (top[i] == (uint32_t) own) hit = i;
        if (hit >= 0) {
            std::memmove(top + hit, top + hit + 1, (n - hit - 1) * sizeof(uint32_t));
            n--;
            own_valid = 4;
        } else {
            own_valid = 0;
        }
    }
    std::memset(words, 0, (NSEL + 16) * sizeof(uint32_t));
    std::memcpy(words, top, n * sizeof(uint32_t));
    words[NSEL] = n;
    words[NSEL + 1] = own_valid;
    if (nsel_out) *nsel_out = n;
    if (own_out) *own_out = own_valid;
}

inline void get_page() { io.read(io.d2h, page, 1, true); }
inline void put_reply() { io.write(io.h2d, reply, 1); }

// Link.serve_card's arithmetic: returns the card's count
int serve_card(int layer) {
    const float* logits = (const float*) (page + 2 * D);
    int32_t ids[K], ids_cpu[K];
    float w[K], w_cpu[K];
    cl_route(logits, K, ids, w);
    uint32_t* words = (uint32_t*) reply;
    std::memset(words, 0, 21 * sizeof(uint32_t));
    bool card[K] = {};
    int k = 0;
    if (const int32_t* s = slot_of[layer]) {
        for (int i = 0; i < K && k < card_max; i++) {
            if (s[ids[i]] < 0) continue;
            card[i] = true;
            words[1 + k] = (uint32_t) s[ids[i]];
            std::memcpy(&words[11 + k], &w[i], 4);
            k++;
        }
    }
    words[0] = k;
    put_reply();
    float* out = (float*) reply;
    if (k < K) {
        alignas(64) float x[D];
        const uint16_t* hb = (const uint16_t*) page;
        for (int i = 0; i < D; i++) {
            const uint32_t u = (uint32_t) hb[i] << 16;
            std::memcpy(&x[i], &u, 4);
        }
        int n = 0;
        for (int i = 0; i < K; i++)
            if (!card[i]) {
                ids_cpu[n] = ids[i];
                w_cpu[n++] = w[i];
            }
        if (cl_moe(layer, x, n, ids_cpu, w_cpu, out)) throw std::runtime_error("cl_moe: layer not loaded");
    } else {
        std::memset(out, 0, D * sizeof(float));
    }
    return k;
}


// Link.serve_m's pages (out_pages(M) of them)
void get_pages_m() {
    for (int k = 0; k * OUT_PAGE < mp_of(rows_m) * ROW_OUT; k++) io.read(io.d2h, page + k * OUT_PAGE, 1, true);
}

// The warm's guess (hl_warm, experts.cpp cl_warm): after an M-row reply, while the card computes the next exchange,
// the lane's threads read the experts that exchange will likely route to. The score per expert of the next layer:
// this exchange's routes through the census's layer -> layer + 1 transitions (row e of trans[layer]: how often a
// token at layer + 1 went to each expert when it went to e here, normalized), plus half the last WH rows' counts at the
// next layer (normalized); the top warm_n, the card's resident experts left out. Census, 4-row passes, 40 experts: 58%
// of the next union (the history alone 50%, the transitions alone 53%).
constexpr int WH = 16;
const float* trans = nullptr;  // [LAYERS - 1][NX][NX] (null: the history alone)
int warm_n = 0;                // 0: off
int16_t hist[LAYERS][WH][K];
int hist_n[LAYERS];
int32_t last_ids[MAXM * K];
int last_layer = -1, last_m = 0;

void note_routes(int layer, int M, const int32_t* ids) {
    for (int m = 0; m < M; m++) {
        int16_t* h = hist[layer][hist_n[layer]++ % WH];
        for (int i = 0; i < K; i++) h[i] = (int16_t) ids[m * K + i];
    }
    std::memcpy(last_ids, ids, sizeof(int32_t) * M * K);
    last_layer = layer;
    last_m = M;
}

void warm(int next) {
    float s[NX] = {}, hs[NX] = {};
    if (trans && last_layer >= 0 && next == last_layer + 1) {
        float tot = 0;
        for (int i = 0; i < last_m * K; i++) {
            const float* r = trans + ((size_t) last_layer * NX + last_ids[i]) * NX;
            for (int e = 0; e < NX; e++) s[e] += r[e];
        }
        for (int e = 0; e < NX; e++) tot += s[e];
        if (tot > 0)
            for (int e = 0; e < NX; e++) s[e] /= tot;
    }
    const int nh = std::min(hist_n[next], WH);
    for (int r = 0; r < nh; r++)
        for (int i = 0; i < K; i++) hs[hist[next][r][i]] += 1.f;
    for (int e = 0; e < NX; e++) s[e] += nh ? 0.5f * hs[e] / (nh * K) : 0.f;
    int32_t order[NX], n = 0;
    for (int e = 0; e < NX; e++)
        if (s[e] > 0 && !(slot_of[next] && slot_of[next][e] >= 0)) order[n++] = e;
    const int take = std::min(n, warm_n);
    std::partial_sort(order, order + take, order + n, [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    if (take) cl_warm(next, take, order);
}

// Link.serve_m's arithmetic, into out_m
void log_routes(int layer, int M, const int32_t* ids) {
    if (warm_n) note_routes(layer, M, ids);
    if (!route_buf || route_n >= route_cap) return;
    int16_t* r = route_buf + route_n++ * (2 + MAXM * K);
    r[0] = (int16_t) layer;
    r[1] = (int16_t) M;
    for (int i = 0; i < MAXM * K; i++) r[2 + i] = i < M * K ? (int16_t) ids[i] : (int16_t) -1;
}

void serve_m(int layer) {
    const int M = rows_m, mp = mp_of(M);
    alignas(64) float x[MAXM][D], logits[MAXM][NX];
    int32_t ids[MAXM * K];
    float w[MAXM * K];
    const uint16_t* hb = (const uint16_t*) page;
    const float* lg = (const float*) (page + mp * 2 * D);
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < D; i++) {
            const uint32_t u = (uint32_t) hb[mrow_at(i, m, mp)] << 16;
            std::memcpy(&x[m][i], &u, 4);
        }
        for (int i = 0; i < NX; i++) logits[m][i] = lg[mrow_at(i, m, mp)];
        cl_route(logits[m], K, ids + m * K, w + m * K);
    }
    log_routes(layer, M, ids);
    if (cl_moe_m(layer, M, &x[0][0], K, ids, w, &out_m[0][0])) throw std::runtime_error("cl_moe_m: layer not loaded");
}

// a score row's pages at position pos (fattn1_reader.cpp's: a page per 2048 pools up to the token's block) into dst
void get_score_pages(int pos, uint8_t* dst) {
    for (int k = 0; k <= pos / 4 * 4 / OUT_PAGE; k++) io.read(io.d2h, dst + k * OUT_PAGE, 1, true);
}


// kind 5's first reply: the card's routes (fused/dexp_dm.h RQ: word 0 n, words 1..n the slots, word W0 + NC m + j row
// m's weight for card expert j); the lane's routes per row, padded to a common k with zero-weight uses of other lane
// experts (cl_moe_m sums a row's uses in order with fma, and fma(0, v, s) is s: each row's sum the bits of its lane
// routes alone). Returns n.
constexpr int NC = 24, W0 = 1 + NC, RQ_BYTES = 512;  // fused/dexp_dm.h
int serve_m_card(int layer) {
    const int M = rows_m, mp = mp_of(M);
    alignas(64) float x[MAXM][D], logits[MAXM][NX];
    int32_t ids[MAXM * K], lane_ids[MAXM * K];
    float w[MAXM * K], lane_w[MAXM * K];
    const uint16_t* hb = (const uint16_t*) page;
    const float* lg = (const float*) (page + mp * 2 * D);
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < NX; i++) logits[m][i] = lg[mrow_at(i, m, mp)];
        cl_route(logits[m], K, ids + m * K, w + m * K);
    }
    log_routes(layer, M, ids);
    // the card's experts: the rows' resident routes in first-seen order, at most card_max; with card_balance only as
    // many as make the card's and the lane's parts of the layer closest (the more-used ones first: a card expert
    // used by more rows takes more of the lane's uses off it), by the costs hl_card_costs set (model:
    // a layer's routed part is max(lane, card); slot form costs the card more per expert than the lane)
    uint32_t* words = (uint32_t*) reply;
    std::memset(words, 0, RQ_BYTES);
    int8_t on_card[NX];
    std::memset(on_card, -1, sizeof on_card);
    int n = 0;
    if (const int32_t* s = slot_of[layer]) {
        const int cap = std::min(card_max, NC);
        if (!card_balance) {
            for (int i = 0; i < M * K && n < cap; i++) {
                const int e = ids[i];
                if (s[e] < 0 || on_card[e] >= 0) continue;
                on_card[e] = (int8_t) n;
                words[1 + n++] = (uint32_t) s[e];
            }
        } else {
            uint8_t uses[NX] = {};
            int order[MAXM * K], nu = 0, ncand = 0, cand[MAXM * K];
            for (int i = 0; i < M * K; i++) {
                if (uses[ids[i]]++ == 0) order[nu++] = ids[i];
            }
            for (int i = 0; i < nu; i++) {
                if (s[order[i]] >= 0) cand[ncand++] = order[i];
            }
            std::stable_sort(cand, cand + ncand, [&](int a, int b) { return uses[a] > uses[b]; });
            int best_n = 0, sel_uses = 0;
            float best = 1e30f;
            for (int c = 0; c <= std::min(ncand, cap); c++) {
                if (c) sel_uses += uses[cand[c - 1]];
                const int lu = nu - c, lc = M * K - sel_uses;
                const float lane = c_lane_exp * lu + c_lane_use * (lc - lu);
                const float card = c ? c_card_exp * c + c_card_fix : 0.f;
                const float t = std::max(lane, card);
                if (t < best) {
                    best = t;
                    best_n = c;
                }
            }
            for (int c = 0; c < best_n; c++) {
                on_card[cand[c]] = (int8_t) n;
                words[1 + n++] = (uint32_t) s[cand[c]];
            }
        }
    }
    words[0] = n;
    int q[MAXM] = {}, k = 0;
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < K; i++) {
            const int e = ids[m * K + i];
            if (on_card[e] >= 0) {
                std::memcpy(&words[W0 + NC * m + on_card[e]], &w[m * K + i], 4);
            } else {
                lane_ids[m * K + q[m]] = e;
                lane_w[m * K + q[m]++] = w[m * K + i];
            }
        }
        k = std::max(k, q[m]);
    }
    put_reply();
    if (k == 0) {
        std::memset(out_m, 0, sizeof out_m);
        return n;
    }
    // each row's lane routes padded to k from the lane's other experts (the union holds at least k of them)
    int32_t pad_ids[MAXM * K];
    float pad_w[MAXM * K];
    for (int m = 0; m < M; m++) {
        int p = 0;
        for (; p < q[m]; p++) {
            pad_ids[m * k + p] = lane_ids[m * K + p];
            pad_w[m * k + p] = lane_w[m * K + p];
        }
        for (int r = 0; r < M && p < k; r++) {
            for (int i = 0; i < q[r] && p < k; i++) {
                const int e = lane_ids[r * K + i];
                bool seen = false;
                for (int j = 0; j < p; j++) seen |= pad_ids[m * k + j] == e;
                if (seen) continue;
                pad_ids[m * k + p] = e;
                pad_w[m * k + p++] = 0.f;
            }
        }
        for (int i = 0; i < D; i++) {
            const uint32_t u = (uint32_t) hb[mrow_at(i, m, mp)] << 16;
            std::memcpy(&x[m][i], &u, 4);
        }
    }
    if (cl_moe_m(layer, M, &x[0][0], k, pad_ids, pad_w, &out_m[0][0]))
        throw std::runtime_error("cl_moe_m: layer not loaded");
    return n;
}

// kind 4: Link.serve_select_m's pages, row m's (kind 1's at position pos + m) at page PPR m
void get_pages_select_m(int pos) {
    for (int m = 0; m < rows_m; m++) get_score_pages(pos + m, page + PPR * m * OUT_PAGE);
}

// kind 4's arithmetic: each row's selection, their union words into reply. The union lists pools in ascending order,
// so a row's selection matters only as a set: its 512th largest score (the least of cl_top_k's 512, GLM's radix
// select: on random scores 34 / 104 us at 8K / 32K pools against nth_element's 59 / 233; served at 60K a
// 4-row pass's selections 4.07 -> 3.12 ms), then every pool at or above it
// straight into the masks; a row whose ties at that score make the set
// depend on introselect's order (or with a NaN) takes select_words instead. The same sets as select_words' (the
// words bit-identical: g6_verify.py G6V_STAGE=host).
void serve_select_m(int pos) {
    static uint32_t mask[NPOOL];  // pool -> its cell x row mask (bit 4 j + m), up to the last row's own pool
    const int npool = std::min((pos + rows_m - 1) / 4 + 1, NPOOL);
    std::memset(mask, 0, npool * sizeof(uint32_t));
    alignas(64) uint32_t words[NSEL + 16];
    static uint32_t top[NSEL];
    float thr[MAXM];
    bool fast[MAXM] = {};
    int complete[MAXM];
    const float* sc[MAXM];
    for (int m = 0; m < rows_m; m++) {
        sc[m] = (const float*) (page + PPR * m * OUT_PAGE);
        const int c = complete[m] = std::min((pos + m + 1) / 4, NPOOL);
        fast[m] = true;
        if (c <= NSEL) continue;  // every complete pool
        bool nan = false;
        for (int i = 0; i < c; i++) nan |= sc[m][i] != sc[m][i];
        if (nan) {
            fast[m] = false;
            continue;
        }
        cl_top_k(sc[m], c, NSEL, top);  // its keys order -0 below +0; the least as a float is the 512th all the same
        float t = sc[m][top[0]];
        for (int j = 1; j < NSEL; j++) t = std::min(t, sc[m][top[j]]);
        thr[m] = t;
        int ge = 0;
        for (int i = 0; i < c; i++) ge += sc[m][i] >= thr[m];
        fast[m] = ge == NSEL;  // more ties than places: introselect's order picks among them
    }
    for (int m = 0; m < rows_m; m++) {
        if (!fast[m]) continue;
        const uint32_t bits = 0x1111u << m;
        if (complete[m] <= NSEL)
            for (int p = 0; p < complete[m]; p++) mask[p] |= bits;
        else
            for (int p = 0; p < complete[m]; p++) mask[p] |= sc[m][p] >= thr[m] ? bits : 0u;
    }
    for (int m = 0; m < rows_m; m++) {
        const int q = pos + m, own = q / 4;
        int own_valid = q % 4 + 1;
        if (!fast[m]) {
            int n;
            select_words(q, sc[m], words, &n, &own_valid);
            for (int i = 0; i < n; i++) mask[words[i]] |= 0x1111u << m;
        } else if (q % 4 == 3) {  // the block the row closes: its own cells if the selection took it (select_words')
            const bool took = mask[own] >> m & 1;
            mask[own] &= ~(0x1111u << m);
            own_valid = took ? 4 : 0;
        }
        for (int j = 0; j < own_valid; j++) mask[own] |= 1u << (4 * j + m);
    }
    uint32_t* out = (uint32_t*) reply;
    std::memset(out, 0, SEL_W * sizeof(uint32_t));
    int u = 0;
    for (int p = 0; p < npool; p++)
        if (mask[p]) out[16 + u++] = (uint32_t) p | mask[p] << 16;
    out[0] = u;
}

}  // namespace

extern "C" {

// The last error's text (a call returned -1).
const char* hl_error() { return error; }

// The sockets: the C++ objects and the addresses of D2HSocket::read and H2DSocket::write.
void hl_bind(void* d2h, void* h2d, void* read_fn, void* write_fn) {
    io = {d2h, h2d, (ReadFn) read_fn, (WriteFn) write_fn};
}

// The test I/O instead: n_pages pages of OUT_PAGE bytes in, up to max_replies replies of IN_PAGE bytes out.
void hl_bind_test(const uint8_t* pages, int n_pages, uint8_t* replies, int max_replies) {
    tio = {pages, replies, n_pages, max_replies, 0, 0};
    std::memset(reply, 0, sizeof reply);  // as a new Link's
    io = {nullptr, nullptr, test_read, test_write};
}

// Replies written through the test I/O so far.
int hl_test_replies() { return tio.next_reply; }

// The token's exchanges in launch order: kind (0 lane, 1 selection, 2 card + lane, 3 M-row lane, 4 M-row selection,
// 5 M-row card + lane)
// and layer each.
// Returns 0 or -1.
int hl_plan(int n, const int32_t* kind, const int32_t* layer) {
    if (n < 0 || n > MAXITEMS) return std::snprintf(error, sizeof error, "hl_plan: %d items (max %d)", n, MAXITEMS), -1;
    for (int i = 0; i < n; i++) {
        if (kind[i] < 0 || kind[i] > 5 || layer[i] < 0 || layer[i] >= LAYERS)
            return std::snprintf(error, sizeof error, "hl_plan: item %d (%d, %d)", i, kind[i], layer[i]), -1;
        plan[i] = {kind[i], layer[i]};
    }
    n_items = n;
    return 0;
}

// The warm after each M-row exchange (see warm()): n experts (0: off); trans float32 [LAYERS - 1][NX][NX] or null,
// kept alive by the caller. The history starts empty.
void hl_warm(int n, const float* t) {
    warm_n = std::max(0, n);
    trans = t;
    std::fill(hist_n, hist_n + LAYERS, 0);
    last_layer = -1;
}

// Kinds 3's and 4's rows from here on (a verify pass's M, 1..4). Returns 0 or -1.
int hl_rows(int m) {
    if (m < 1 || m > MAXM) return std::snprintf(error, sizeof error, "hl_rows: %d rows (1..%d)", m, MAXM), -1;
    rows_m = m;
    return 0;
}

// The card's expert cache for kinds 2 and 5: layer's expert -> slot map (int32 [NX], -1 CPU only; null: none), the cap.
// The M-row exchanges' routes into buf (int16 [cap][2 + MAXM K]; null: off); hl_route_count: the records so far.
void hl_route_log(int16_t* buf, int64_t cap) {
    route_buf = buf;
    route_cap = cap;
    route_n = 0;
}

int64_t hl_route_count() { return route_n; }

// kind 5's split: on = 0 every resident route of the rows to the card (up to card_max), 1 balanced by these costs (us)
void hl_card_costs(int on, float lane_exp, float lane_use, float card_exp, float card_fix) {
    card_balance = on;
    c_lane_exp = lane_exp;
    c_lane_use = lane_use;
    c_card_exp = card_exp;
    c_card_fix = card_fix;
}

void hl_slots(int layer, const int32_t* slots, int cap) {
    if (layer >= 0 && layer < LAYERS) slot_of[layer] = slots;
    card_max = cap;
}

// select_words alone (checks).
void hl_select(int pos, const float* scores, uint32_t* words, int32_t* nsel_own) {
    select_words(pos, scores, words, &nsel_own[0], &nsel_own[1]);
}

// One token at position pos: every planned exchange. stamps (null: none) gets per item the CLOCK_MONOTONIC ns before
// the page, after it, after the work, after the reply (Link.trace's t0..t3); n_card (null: none) per item the card's
// count (kinds 2, 5). Returns 0, or -1 with hl_error().
int hl_token(int pos, int64_t* stamps, int32_t* n_card) {
    try {
        for (int i = 0; i < n_items; i++) {
            const Item it = plan[i];
            int64_t* st = stamps ? stamps + 4 * i : nullptr;
            if (st) st[0] = now_ns();
            if (it.kind == 3 || it.kind == 5) {
                get_pages_m();
            } else if (it.kind == 4) {
                get_pages_select_m(pos);
            } else if (it.kind == 1) {
                get_score_pages(pos, page);  // serve_select's
            } else {
                get_page();
            }
            if (st) st[1] = now_ns();
            int k = 0;
            if (it.kind == 0) {
                if (cl_exchange(it.layer, page, (float*) reply, nullptr, nullptr))
                    throw std::runtime_error("cl_exchange: layer not loaded");
            } else if (it.kind == 1) {
                select_words(pos, (const float*) page, (uint32_t*) reply, nullptr, nullptr);
            } else if (it.kind == 2) {
                k = serve_card(it.layer);
            } else if (it.kind == 3) {
                serve_m(it.layer);
            } else if (it.kind == 5) {
                k = serve_m_card(it.layer);
            } else {
                serve_select_m(pos);
            }
            if (st) st[2] = now_ns();
            if (it.kind == 3 || it.kind == 5) {
                for (int m = 0; m < rows_m; m++) io.write(io.h2d, out_m[m], 1);
                if (warm_n) {  // the next M-row exchange (the next token's first after the last)
                    int j = (i + 1) % n_items;
                    while (plan[j].kind != 3 && plan[j].kind != 5) j = (j + 1) % n_items;
                    warm(plan[j].layer);
                }
            } else {
                put_reply();
            }
            if (st) st[3] = now_ns();
            if (n_card) n_card[i] = k;
        }
    } catch (const std::exception& e) {
        std::snprintf(error, sizeof error, "hl_token at %d: %s", pos, e.what());
        return -1;
    }
    return 0;
}

}  // extern "C"
