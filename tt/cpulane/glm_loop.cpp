// GLM's host loop (the Python side tt/glm/glm_loop.py): every host round trip of a traced GLM layer
// (later a token) in launch order, in one call, through ttnn's own socket objects (tt-metal's D2HSocket::read and
// H2DSocket::write called directly, as Flash-Next's host_loop.cpp does). Each exchange names its socket pair and kind:
//
//   HC    an mHC site's partials (nc x 32 mixes, then nc 16-byte slots of squares, fp32) -> its 24 parameters
//         (cl_hc_params with the site's scale and base; 128 bytes)                           (g3_hc.HcSite.serve)
//   SEL   an MLA layer's token (mla_step below): the indexer's pool scores (fp32 [n_pool]), kv_a | i_proj | i_k |
//         i_gate (fp32), iq (bf16) -> the selection, cur_pos, the latent row, the closed pool's key (g3_fmq.MlaQuery)
//   LANE  hn (bf16 [D]) -> GLM's routing and the lane's experts' sum (cl_glm_exchange; fp32 [D])
//                                                                                             (g3_fsh.FfnFused.serve)
//
// with the same functions as the Python path, so the replies are the same to the bit. The last LANE exchange's
// routes and the last SEL's pools are kept for checks (gl_ids, gl_pools).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mrows.h"

extern "C" {
void cl_hc_params(const float* page, int nc, int d, int iters, const float* scale, const float* base, float* out);
void cl_top_k(const float* score, int n, int k, uint32_t* out);
int cl_glm_exchange(int layer, const uint16_t* hn, float* reply, int32_t* ids_out, float* w_out);
int cl_glm_exchange_m(int layer, int M, const uint16_t* hn, float* reply, int32_t* ids_out, float* w_out);
int cl_glm_route(int layer, const float* x, int k, float scale, int32_t* ids, float* w, float* logits_out);
int cl_moe(int layer, const float* x, int k, const int32_t* ids, const float* w, float* out);
}

namespace {

using ReadFn = void (*)(void* self, void* data, uint32_t num_pages, bool notify_sender);
using WriteFn = void (*)(void* self, void* data, uint32_t num_pages);

struct Pair {
    void *d2h = nullptr, *h2d = nullptr;
    ReadFn read = nullptr;
    WriteFn write = nullptr;
};

enum Kind { HC = 0, SEL = 1, LANE = 2 };

struct Exchange {
    int kind, pair, layer, nc, n_pool;
    const float *scale, *base;
};

constexpr int MAXPAIRS = 8, D = 4096, SINKHORN = 20, NSEL = 512, KPOOL = 4, IHD = 128, IH = 32, KVL = 512;
// the SEL page after the scores: kv_a 512 | i_proj 32 | i_k 128 | i_gate 128 (fp32), then iq (bf16 [32][128]); the
// reply (glm_fmq_reader.cpp's): NS token ids, the latent row (bf16), cur_pos | p | pool, the pooled key's 8 pieces
constexpr int A2 = KVL + IH + 2 * IHD, PAGE_TAIL = A2 * 4 + IH * IHD * 2;
constexpr int NS = 2112, LAT_OFF = NS * 4, CTL_OFF = LAT_OFF + 1024, PK_OFF = CTL_OFF + 64, RB = PK_OFF + 512;
constexpr uint32_t NONE = 0xffffffffu;
constexpr float KV_EPS = 1e-5f, IK_EPS = 1e-6f;   // kv_a_norm (RMS), the indexer key's layer norm
Pair pairs[MAXPAIRS];
// a few exchange lists (gl_seq selects one for gl_clear / gl_add / gl_run): the trunk's and the MTP's
constexpr int NSEQ = 4;
std::vector<Exchange> seqs[NSEQ];
int cur_seq = 0;
#define seq seqs[cur_seq]
std::vector<uint8_t> page(256 << 10), reply(64 << 10);
int32_t last_ids[8];
float last_w[8];
uint32_t last_pools[NSEL];
int n_last = 0;

// One MLA layer's decode state: the norms' weights and the recent tokens' keys (normed) and gates,
// a ring by position; the caches themselves are on the card (the latent rows, the pooled keys K), written by position,
// so a verify pass's rejected rows need no undo there (Flash-Next's rule: overwritten when the position is decoded
// again). The ring holds RING >= KPOOL + M - 1 positions, so gl_mla_rewind past a pass of M <= 5 rows still finds the
// open pool's accepted keys (with KPOOL slots a rejected row would overwrite one).
constexpr uint32_t RING = 8;
struct Mla {
    const float *kv_w, *k_w, *k_b, *ape;   // kv_a_norm [512], the key norm's weight and bias [128], ape [4][128]
    float kg[RING][2 * IHD];
    uint32_t p;
};
std::vector<Mla> mlas;
std::vector<float> sc;

inline uint16_t bf16(float f) {        // round to nearest even, as the card's packer and glm_fmq_reader.cpp
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t) ((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
inline float from_bf16(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// One token of MLA layer h at its position p, as llama.cpp's glm5-next graph (build_kpool_select, set_input_kpool):
// the pools visible to p are those ending at or before it, nv = (p + 1) / 4, the one p closes included (its score is
// computed here: K gets its key only from this reply); the selection is the top min(nv, 512) of them x 4 tokens, then
// the open pool's (p + 1) % 4 tokens, p's own included. Its order does not matter: no RoPE, and the attention sums.
// h < 0: stateless (the layer scripts' random caches): the top 512 of all n_pool, no tail, no cache writes.
// closed: the pools closed by the earlier rows of this verify pass, their keys reach K only with this
// pass's reply, so the card scored them from stale rows: their scores are computed here too, from this row's iq and w;
// the pool this row closes is appended. -> the selection's count, or -1 past the card's capacity.
struct Closed {
    uint32_t q;
    float pk[IHD];
};
// a pool key's score, as the card's: the sum over the heads of relu(iq_h . key) times w_h (bf16)
float pool_score(const uint16_t* iq, const float* a2, const float* pk) {
    double s = 0;
    for (int hh = 0; hh < IH; hh++) {
        double dot = 0;
        for (int d = 0; d < IHD; d++) dot += (double) from_bf16(iq[hh * IHD + d]) * pk[d];
        s += std::max(dot, 0.0) * from_bf16(bf16(a2[KVL + hh]));
    }
    return (float) s;
}
int mla_step(int h, const uint8_t* pg, int n_pool, uint8_t* rp, std::vector<Closed>* closed = nullptr) {
    const float* score = (const float*) pg;
    const float* a2 = score + n_pool;
    const uint16_t* iq = (const uint16_t*) (a2 + A2);
    uint32_t* ids = (uint32_t*) rp;
    uint32_t* ctl = (uint32_t*) (rp + CTL_OFF);
    std::fill(ids, ids + NS, 0u);
    if (h < 0) {
        cl_top_k(score, n_pool, NSEL, last_pools);
        n_last = NSEL;
        for (int j = 0; j < NSEL; j++) {
            for (int c = 0; c < KPOOL; c++) ids[j * KPOOL + c] = last_pools[j] * KPOOL + c;
        }
        ctl[0] = NSEL * KPOOL - 1;
        ctl[1] = ctl[2] = NONE;
        return NSEL * KPOOL;
    }
    Mla& m = mlas[h];
    const uint32_t p = m.p;
    if (p >= (uint32_t) n_pool * KPOOL) return -1;
    double ss = 0;
    for (int j = 0; j < KVL; j++) ss += (double) a2[j] * a2[j];
    const double rs = 1.0 / std::sqrt(ss / KVL + KV_EPS);
    uint16_t* lat = (uint16_t*) (rp + LAT_OFF);
    for (int j = 0; j < KVL; j++) lat[j] = bf16((float) (a2[j] * rs * m.kv_w[j]));
    const float *ik = a2 + KVL + IH, *ig = ik + IHD;
    double mu = 0, var = 0;
    for (int d = 0; d < IHD; d++) mu += ik[d];
    mu /= IHD;
    for (int d = 0; d < IHD; d++) var += (ik[d] - mu) * (ik[d] - mu);
    const double ri = 1.0 / std::sqrt(var / IHD + IK_EPS);
    float* kg = m.kg[p % RING];
    for (int d = 0; d < IHD; d++) {
        kg[d] = (float) ((ik[d] - mu) * ri * m.k_w[d] + m.k_b[d]);
        kg[IHD + d] = ig[d];
    }
    const uint32_t nv = (p + 1) / KPOOL, tail = (p + 1) % KPOOL;
    sc.assign(score, score + nv);
    if (closed) {
        for (const Closed& c : *closed) {
            if (c.q < nv) sc[c.q] = pool_score(iq, a2, c.pk);
        }
    }
    ctl[2] = NONE;
    if (tail == 0) {
        // the pooled key of pool q: per dimension, the softmax over its 4 tokens of gate + ape weighting their keys
        const uint32_t q = nv - 1;
        float pk[IHD];
        for (int d = 0; d < IHD; d++) {
            double l[KPOOL], mx = -INFINITY, z = 0, acc = 0;
            for (int j = 0; j < KPOOL; j++) {
                mx = std::max(mx, l[j] = (double) m.kg[(q * KPOOL + j) % RING][IHD + d] + m.ape[j * IHD + d]);
            }
            for (int j = 0; j < KPOOL; j++) {
                const double e = std::exp(l[j] - mx);
                z += e;
                acc += e * m.kg[(q * KPOOL + j) % RING][d];
            }
            pk[d] = from_bf16(bf16((float) (acc / z)));   // K holds it as bf16
        }
        sc[q] = pool_score(iq, a2, pk);
        if (closed) {
            closed->push_back({q, {}});
            std::memcpy(closed->back().pk, pk, sizeof(pk));
        }
        // K's row q % 32 in 8 pieces of 16 (tile d's cols 0..15, 16..31), each at its destination's address mod 64
        const uint32_t r = q % 32, at = (r & 1) * 32;
        for (int c = 0; c < 8; c++) {
            uint16_t* dst = (uint16_t*) (rp + PK_OFF + c * 64 + at);
            for (int e = 0; e < 16; e++) dst[e] = bf16(pk[c * 16 + e]);
        }
        ctl[2] = q;
    }
    const uint32_t k = std::min<uint32_t>(nv, NSEL);
    if (k == nv) {
        for (uint32_t j = 0; j < k; j++) last_pools[j] = j;
    } else {
        cl_top_k(sc.data(), (int) nv, (int) k, last_pools);
    }
    n_last = (int) k;
    for (uint32_t j = 0; j < k; j++) {
        for (int c = 0; c < KPOOL; c++) ids[j * KPOOL + c] = last_pools[j] * KPOOL + c;
    }
    for (uint32_t t = 0; t < tail; t++) ids[k * KPOOL + t] = p - t;
    const uint32_t count = k * KPOOL + tail;
    ctl[0] = count - 1;
    ctl[1] = p;
    m.p++;
    return (int) count;
}
// a verify pass's rows of MLA layer h: row m's page at pg + m (n_pool * 4 + PAGE_TAIL), its reply at
// rp + m RB (glm_fmq_reader.cpp's M-row layout); the pools closed by earlier rows rescored per row (mla_step)
int rows = 1;   // the pass's rows (gl_rows)
int mla_rows(int h, const uint8_t* pg, int n_pool, int rows, uint8_t* rp) {
    std::vector<Closed> closed;
    const size_t row_bytes = (size_t) n_pool * 4 + PAGE_TAIL;
    if (rows * row_bytes > page.size() || (size_t) rows * RB > reply.size()) return -1;
    for (int m = 0; m < rows; m++) {
        if (mla_step(h, pg + m * row_bytes, n_pool, rp + (size_t) m * RB, &closed) < 0) return -1;
    }
    return 0;
}

// gl_random_routes (benchmarks): each LANE exchange routes as usual, then runs 8 random distinct experts instead, as
// a decode token's layers would (a traced loop's fixed input routes to the same 8 every time, which stay in cache)
bool random_routes = false;
uint64_t rng = 0x9E3779B97F4A7C15ull;
uint32_t next_rand() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t) (rng >> 32);
}
constexpr int NX = 288;

}  // namespace

extern "C" {

int gl_bind(int pair, void* d2h, void* h2d, void* read, void* write) {
    if (pair < 0 || pair >= MAXPAIRS) return -22;
    pairs[pair] = {d2h, h2d, (ReadFn) read, (WriteFn) write};
    return 0;
}

void gl_clear() { seq.clear(); }

int gl_seq(int k) {
    if (k < 0 || k >= NSEQ) return -22;
    cur_seq = k;
    return 0;
}

void gl_random_routes(int on) { random_routes = on != 0; }

int gl_add(int kind, int pair, int layer, int nc, int n_pool, const float* scale, const float* base) {
    if (pair < 0 || pair >= MAXPAIRS || !pairs[pair].d2h || kind < HC || kind > LANE) return -22;
    if (kind == SEL && ((size_t) n_pool * 4 + PAGE_TAIL > page.size() || layer >= (int) mlas.size())) return -22;
    seq.push_back({kind, pair, layer, nc, n_pool, scale, base});
    return (int) seq.size();
}

// every exchange in order; 0, or -(1 + the failing exchange's index)
int gl_run() {
    for (size_t i = 0; i < seq.size(); i++) {
        const Exchange& e = seq[i];
        Pair& p = pairs[e.pair];
        p.read(p.d2h, page.data(), 1, true);
        if (e.kind == HC) {   // a verify pass's rows: row m's page at m nc 144 bytes, its reply at 128 m
            for (int m = 0; m < rows; m++) {
                cl_hc_params((const float*) (page.data() + (size_t) m * e.nc * 144), e.nc, D, SINKHORN, e.scale,
                             e.base, (float*) (reply.data() + 128 * m));
            }
        } else if (e.kind == SEL && rows > 1) {
            if (mla_rows(e.layer, page.data(), e.n_pool, rows, reply.data()) < 0) return -(int) (1 + i);
        } else if (e.kind == SEL) {
            if (mla_step(e.layer, page.data(), e.n_pool, reply.data()) < 0) return -(int) (1 + i);
        } else if (random_routes) {
            alignas(64) float x[D];
            const uint16_t* hb = (const uint16_t*) page.data();
            for (int j = 0; j < D; j++) {
                const uint32_t u = (uint32_t) hb[j] << 16;
                std::memcpy(&x[j], &u, 4);
            }
            if (cl_glm_route(e.layer, x, 8, 2.5f, last_ids, last_w, nullptr)) return -(int) (1 + i);
            for (int j = 0; j < 8; j++) {
                int c;
                do {
                    c = (int) (next_rand() % NX);
                } while (std::find(last_ids, last_ids + j, c) != last_ids + j);
                last_ids[j] = c;
            }
            if (cl_moe(e.layer, x, 8, last_ids, last_w, (float*) reply.data())) return -(int) (1 + i);
        } else if (rows > 1) {   // a verify pass's rows: the page and reply in mrows.h's blocks
            if (cl_glm_exchange_m(e.layer, rows, (const uint16_t*) page.data(), (float*) reply.data(), nullptr,
                                  nullptr)) {
                return -(int) (1 + i);
            }
        } else if (cl_glm_exchange(e.layer, (const uint16_t*) page.data(), (float*) reply.data(), last_ids, last_w)) {
            return -(int) (1 + i);
        }
        p.write(p.h2d, reply.data(), 1);
    }
    return 0;
}

void gl_ids(int32_t* ids, float* w) {
    std::memcpy(ids, last_ids, sizeof(last_ids));
    std::memcpy(w, last_w, sizeof(last_w));
}

// the last SEL's pools (its count returned)
int gl_pools(uint32_t* pools) {
    std::memcpy(pools, last_pools, sizeof(last_pools));
    return n_last;
}

// a new MLA layer's state at position 0 (the weights must outlive it) -> its handle, the SEL exchange's `layer`
int gl_mla_new(const float* kv_w, const float* k_w, const float* k_b, const float* ape) {
    Mla m{kv_w, k_w, k_b, ape, {}, 0};
    mlas.push_back(m);
    return (int) mlas.size() - 1;
}

void gl_mla_clear() { mlas.clear(); }

// A verify pass kept the rows before position p (its rejected rows' positions get decoded again): the state as if
// they never ran. 0, or -1 when p is ahead of the state or the ring no longer holds p's open pool.
int gl_mla_rewind(int h, uint32_t p) {
    if (h < 0 || h >= (int) mlas.size()) return -22;
    Mla& m = mlas[h];
    // position 0 always (a new sequence: no open pool's keys needed); else the open pool's keys must be in the ring
    if (p > m.p || (p > 0 && m.p - (p - p % KPOOL) > RING)) return -1;
    m.p = p;
    return 0;
}

// A prefill's hand-off (tt/glm/g6_prefill.py): the state at position p, after the prompt's rows 0 .. p - 1; kg
// [rows][2 * IHD] holds positions p - rows .. p - 1 (the normalized key, then the raw gate: the ring's form), at
// least the open pool's p % 4 of them. 0, or -22 for a bad handle or too few / too many rows.
int gl_mla_set(int h, uint32_t p, const float* kg, int rows) {
    if (h < 0 || h >= (int) mlas.size() || rows < (int) (p % KPOOL) || rows > (int) RING || (uint32_t) rows > p)
        return -22;
    Mla& m = mlas[h];
    for (int i = 0; i < rows; i++) std::memcpy(m.kg[(p - rows + i) % RING], kg + i * 2 * IHD, 2 * IHD * sizeof(float));
    m.p = p;
    return 0;
}

// Python's path (g3_fmq.MlaQuery.serve): one SEL exchange's host side
// the rows a pass of the SEL and LANE exchanges from now on (1: the one-row pages and replies)
int gl_rows(int m) {
    if (m < 1 || m > 4 || (size_t) m * RB > reply.size() || (size_t) mp_of(m) * D * 4 > reply.size()) return -22;
    rows = m;
    return 0;
}

int gl_mla_rows(int h, const uint8_t* pg, int n_pool, int rows, uint8_t* rp) {
    if (h >= (int) mlas.size()) return -22;
    return mla_rows(h, pg, n_pool, rows, rp);
}

int gl_mla_step(int h, const uint8_t* pg, int n_pool, uint8_t* rp) {
    if (h >= (int) mlas.size()) return -22;
    return mla_step(h, pg, n_pool, rp);
}

}  // extern "C"
