// Contains code derived from ggml-cpu (llama.cpp: the x86 helpers, the AVX2 vec_dot of Q8_0, IQ4_NL, IQ4_XS and Q5_K,
// and quantize_row_q8_K_ref, here in AVX-512), Copyright (c) 2023-2026 The ggml authors; and from Strata
// (src/kernels/cpu/iq_avx512.cpp's IQ3_S decode, row dot and gate|up rows in the form its fork Xolotl had them, the
// SwiGLU clamp included; native_expert.cpp's iq512_supported), Copyright (c) 2026 Niko1221 and the Strata
// contributors. Both MIT License (LICENSES/, NOTICE).
// The CPU lane of Flash-Next's decode on the card (G4): a layer's routed experts for one token, between the card's
// D2H page (hn_f bf16 + the router's logits fp32, fused/link.h) and its H2D reply (the experts' sum, fp32).
//
// The arithmetic is ggml-cpu's own (ggml_get_type_traits_cpu): the activation quantized to the weight type's
// vec_dot_type and each row one vec_dot, as llama.cpp's CPU backend computes the same tensors. The routing is
// llama.cpp's for qwen4exp: softmax over the 512 logits, the top 10, their weights normalized to sum 1.
//
// The experts live in this library's own memory (2 MiB aligned, transparent huge pages), read once from the GGUF
// files (cl_load; the file's pages dropped from the page cache after each read). A pool of pinned threads spins
// between calls and sleeps after CL_IDLE_MS without work. The rows go to the threads as even shares, the slow ones
// relieved by their CCX (rows()). cl_exchange: [caller] hn to f32, routing; then cl_moe, which is cl_moe_m for one
// token.
//
// cl_moe_m: M <= 8 tokens at once (a verify pass; M = 1 for a decode token): the union of their routes, each distinct
// expert's gate|up and down rows streamed once and applied to every token routed to it (Strata's gate|up rows take
// several activations; down's vec_dot per use). [all] the M activations quantized, gate|up per (expert, rows) range,
// each expert's uses' h quantized by the thread that finishes it; [all] down in blocks of rows, expert by expert within
// a block, the block's sums by the thread that completes it. Each token's sum is added in its own route order, so every
// token's result is cl_moe's for that token alone, to the bit (and the same bits as the lane's one-token path).
//
// Gate/up rows in i-quants go through Strata's AVX-512 kernel (third_party/strata, iq_avx512.cpp: 64 values decoded
// once per lane; ggml-cpu's IQ3_S dot is AVX2), CL_NO_IQ512=1 for ggml-cpu's.
//
// Build: tt/cpulane/build.sh (in the container, as tt/env/setup.sh does: the library is for the container's toolchain). CL_TRACE=1
// build.sh builds build/libcpulane_trace.so instead, with per-call, per-thread TSC stamps (cl_trace); the default
// build has none.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include "mrows.h"
#include <vector>
#include <vector>

#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/cpu/iq_avx512.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP   // the tables (kvalues_iq4nl, iq3s_grid) for the batched rows below
#include "ggml-common.h"
#include "ggml-quants.h"
#ifdef __AVX512VBMI__
#include "iq3s_vbmi.h"   // IQ3_S gate|up with the grid lookup vectorized: 2.2x Strata's rows a thread, the same bits
#endif

// Strata's answer (src/kernels/cpu/native_expert.cpp; not in iq_avx512.cpp, which is compiled for AVX-512)
namespace strata::kernels::cpu {
bool iq512_supported(int type) noexcept { return type == 16 || type == 17 || type == 18 || type == 21 || type == 22; }
// iq_avx512.cpp's default for its grid lookups (expert_layout.cpp's, which would bring Strata's whole CPU expert path):
// gathers give the same words, so the same bits, only their speed differs.
// shortcut: off, as Strata picks on AMD; Strata turns them on for Intel cores with AVX-VNNI (Sapphire Rapids and
// later, 1.3-2x there): port its cpuid test when such a host runs the lane. STRATA_IQ_GATHER=1 forces them.
bool cpu_gather_fast() { return false; }
}  // namespace strata::kernels::cpu

namespace {

// shortcut: the model's shapes are fixed per build (Flash-Next's by default; CL_SHAPE=glm build.sh: GLM-5.3-Flash's),
// make them runtime values when one process must hold two models' lanes
#ifndef CL_D
#define CL_D 2560
#define CL_FF 640
#define CL_NX 512
#define CL_MAXK 16
#define CL_LAYERS 48
#endif
#ifndef CL_SWIGLU_LIM   // llama.cpp's swiglu_clamp_exp: silu(min(gate, lim)) * clamp(up, -lim, lim); none by default
#define CL_SWIGLU_LIM std::numeric_limits<float>::infinity()
#endif
#ifndef CL_XQB          // bytes a quantized activation row (gate|up's vec_dot_type at D) and h row (down's at FF) take
#define CL_XQB 8192
#define CL_HQB 2048
#endif
constexpr int D = CL_D, FF = CL_FF, NX = CL_NX, MAXK = CL_MAXK, LAYERS = CL_LAYERS;
constexpr int CL_IDLE_MS = 200;

struct Proj {
    uint8_t* data = nullptr;  // [NX][rows][row_bytes]
    size_t bytes = 0, row_bytes = 0;
    int type = -1;
    const ggml_type_traits_cpu* tr = nullptr;
};

struct Layer {
    Proj gate, up, down;  // gate and up in one mapping, up right after gate (Strata's gate|up kernel takes up_off)
    uint8_t* gu = nullptr;
    bool iq512 = false;   // gate|up through Strata's AVX-512 i-quant rows (else ggml-cpu's vec_dot)
};

Layer layers[LAYERS];
pthread_mutex_t load_mu = PTHREAD_MUTEX_INITIALIZER;

constexpr size_t huge(size_t b) { return (b + (2u << 20) - 1) & ~(size_t) ((2u << 20) - 1); }

// one cl_moe_m call's state: the union of the tokens' routes, and per use (a token routed to a union expert) its h
// MAXM: tokens a call (the decode's passes take <= 4; a short prefill's chunk up to 512, each expert of the union
// streamed once for all its tokens: 2026-11-06, was 8). The per-use buffers are static, touched only as far as used.
constexpr int MAXM = 2048, MAXP = MAXM * MAXK;
struct JobM {
    int layer = 0, m = 0, k = 0, u = 0, p = 0;  // tokens, routes per token, distinct experts, uses
    int ex[MAXP];                               // the union's experts, in first-seen order (token 0's routes first)
    int first[MAXP + 1];                        // union expert j's uses: first[j] .. first[j + 1] - 1
    int decode = 1;                             // a gate|up row's decode, in token applications (gu_decode)
    int cost[MAXP + 1];                         // gate|up split: prefix sums of FF * (decode + uses) per expert
    int dcost[MAXP + 1];                        // down split: prefix sums of DB * (1 + uses) per expert
    int use_of[MAXM][MAXK];                     // token t's route i -> its use
    int kn[MAXM];                               // token t's routes kept (ids < 0 skipped: the card's)
    float w[MAXM][MAXK];
    const float* x = nullptr;                   // [m][D]
    float* out = nullptr;                       // [m][D]
    const void* act[MAXP];                      // per use: its token's xq
    float* ff[MAXP];                            // per use: its h
    alignas(64) uint8_t xq[MAXM][CL_XQB];
    alignas(64) uint8_t hq[MAXP][CL_HQB];
    alignas(64) float h[MAXP][FF];
    alignas(64) float v[MAXP][D];               // per use: its down rows
} jm;
// a gate|up row's cost in token applications, the threads' shares split by it: Strata's rows decode once for all the
// uses (IQ3_S, 8 threads, one expert hot: 71 us at M = 1, +4.3 us per extra use); ggml-cpu's vec_dot per use
// CL_IQ3S_STRATA=1: IQ3_S through Strata's rows (iq3s_vbmi.h's are the same bits, ~2.2x faster a thread)
const bool iq3s_vbmi = getenv("CL_IQ3S_STRATA") == nullptr;
// the split's IQ3_S decode weight stays 16 uses with the VBMI rows (the results do not depend on it): 8, the decode's
// own cost (~33 us), was 1.5% slower at 1,000-2,000 tokens on Flash-Next's lane (2026-11-11, bench.py --fresh)
inline int gu_decode(const Layer& L);
// ggml's Q8_K quantization (quantize_row_q8_K_ref, ggml-quants.c: the scalar reference, ~5 us a row of 2560 on one
// core, which cl_moe and cl_moe_m waited for) in 512-bit lanes, to the bit: the scale from the first value of the
// largest magnitude, the same float product and add per value (nearest_int; the _round intrinsics are never contracted
// into an FMA), the same int8 truncation and sums. A block with a NaN goes to the reference.
void q8k_block(const float* x, block_q8_K* y, const block_q8_K* was) {
    constexpr int R = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    __m512 am = _mm512_setzero_ps();
    __mmask16 nan = 0;
    for (int j = 0; j < QK_K; j += 16) {
        const __m512 v = _mm512_loadu_ps(x + j);
        am = _mm512_max_ps(am, _mm512_abs_ps(v));
        nan |= _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
    }
    if (nan) {
        std::memcpy(y, was, sizeof(*y));  // the reference may leave bsums too
        return quantize_row_q8_K_ref(x, y, QK_K);
    }
    const float amax = _mm512_reduce_max_ps(am);
    if (!amax) {  // as the reference: bsums left as they were (in `was`, the destination)
        std::memcpy(y->bsums, was->bsums, sizeof(y->bsums));
        y->d = 0;
        std::memset(y->qs, 0, QK_K);
        return;
    }
    float max = 0;
    for (int j = 0; j < QK_K; j += 16) {
        const __mmask16 m = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_loadu_ps(x + j)), _mm512_set1_ps(amax), _CMP_EQ_OQ);
        if (m) {
            max = x[j + __builtin_ctz(m)];
            break;
        }
    }
    const float iscale = -127.f / max;
    const __m512 is = _mm512_set1_ps(iscale), magic = _mm512_set1_ps(12582912.f);
    for (int j = 0; j < QK_K; j += 16) {
        const __m512 f = _mm512_add_round_ps(_mm512_mul_round_ps(is, _mm512_loadu_ps(x + j), R), magic, R);
        __m512i q = _mm512_sub_epi32(_mm512_and_si512(_mm512_castps_si512(f), _mm512_set1_epi32(0x007fffff)),
                                     _mm512_set1_epi32(0x00400000));
        q = _mm512_min_epi32(q, _mm512_set1_epi32(127));
        _mm_storeu_si128((__m128i*) (y->qs + j), _mm512_cvtepi32_epi8(q));
        y->bsums[j / 16] = (int16_t) _mm512_reduce_add_epi32(_mm512_srai_epi32(_mm512_slli_epi32(q, 24), 24));
    }
    y->d = 1 / iscale;
}

// An activation row quantized for gate|up into xq. Every thread reads xq, so a plain store found its lines shared by
// up to 96 cores and waited on their invalidation: ~3 us a row at 96 threads, the arithmetic ~0.8 (prefetchw at the
// previous call's end did not help). The row is quantized into this thread's own buffer and streamed to xq whole.
void quantize_row(const ggml_type_traits_cpu* tr, const float* x, uint8_t* xq) {
    alignas(64) static thread_local uint8_t tmp[CL_XQB];
    if (tr->vec_dot_type == GGML_TYPE_Q8_K) {
        for (int b = 0; b < D / QK_K; b++) q8k_block(x + b * QK_K, (block_q8_K*) tmp + b, (const block_q8_K*) xq + b);
    } else {
        ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x, tmp, D);
    }
    for (size_t o = 0; o < ggml_row_size(tr->vec_dot_type, D); o += 64)  // xq's rows are 64-aligned, CL_XQB apart
        _mm512_stream_si512((__m512i*) (xq + o), _mm512_load_si512(tmp + o));
    _mm_sfence();
}

int n_threads = 1, n_active = 1;  // the pool's size; the threads that take part in a phase
std::vector<std::thread>& workers = *new std::vector<std::thread>;  // never destroyed: cl_shutdown joins at exit
uint64_t phase_ns[4];  // the caller's time per step: hn + routing, gate|up, h, down
alignas(64) std::atomic<uint32_t> phase_gen{0};  // bumped per phase; the phase is in phase_id
alignas(64) std::atomic<int> phase_id{0};
alignas(64) std::atomic<int> phase_left{0};
alignas(64) std::atomic<bool> stop{false};
std::atomic<int> sleepers{0};
pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

#ifdef CL_TRACE
// per call: the caller's stamps (TR_*), and per thread and phase (0 gate|up, 1 down) when it saw the phase, when its
// rows began (after the in-phase quantization wait) and when it was done. Each thread writes only its own rows.
enum { TR_ENTRY, TR_ROUTED, TR_PUB1, TR_DONE1, TR_PUB2, TR_DONE2, TR_RET, TR_XQ, TR_N };
constexpr int TR_CALLS = 4096, TR_THREADS = 128;
uint64_t tr_caller[TR_CALLS][TR_N];
struct alignas(64) TrThread { uint64_t s[TR_CALLS][2][3]; };
TrThread* tr_thread = new TrThread[TR_THREADS]();
int tr_n = 0;               // calls recorded since the last cl_trace
alignas(64) int tr_call = 0;  // the call in flight (set before its first phase)
bool tr_entered = false;      // cl_exchange stamped the entry before its routing
inline uint64_t tsc() { return __rdtsc(); }
inline void tr_c(int i) { if (tr_n < TR_CALLS) tr_caller[tr_n][i] = tsc(); }
inline void tr_t(int t, int ph, int i) {
    if (t < TR_THREADS && tr_call < TR_CALLS) tr_thread[t].s[tr_call][ph][i] = tsc();
}
#define TR_C(i) tr_c(i)
#define TR_T(t, ph, i) tr_t(t, ph, i)
#else
#define TR_C(i) ((void) 0)
#define TR_T(t, ph, i) ((void) 0)
#endif

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

constexpr bool kClamp = CL_SWIGLU_LIM < std::numeric_limits<float>::infinity();

// Strata's iq512_gu_rows takes no clamp: where the model clamps (GLM's swiglu_clamp_exp) gate and up are two passes of
// its rows, then silu(min(gate, lim)) * clamp(up, -lim, lim) here, as Strata's own DS41 lane does (ds4_gg_cpu.cpp,
// gg_cpu_gate_up). nt <= 8.
void gu_rows_clamped(int type, const uint8_t* gw, size_t row, size_t up_off, const void* const* act, int nt,
                     float* const* ff, int r0, int r1) {
    static thread_local float up[8][FF];
    float* uo[8];
    for (int t = 0; t < nt; t++) uo[t] = up[t];
    strata::kernels::cpu::iq512_rows(type, gw, row, D, act, nt, ff, r0, r1);
    strata::kernels::cpu::iq512_rows(type, gw + up_off, row, D, act, nt, uo, r0, r1);
    constexpr float lim = CL_SWIGLU_LIM;
    for (int t = 0; t < nt; t++)
        for (int r = r0; r < r1; r++)
            ff[t][r] = silu(std::min(ff[t][r], lim)) * std::min(std::max(up[t][r], -lim), lim);
}

// The rows' split. Even contiguous shares leave each phase as long as its slowest thread, and which thread is slow
// changes from call to call (at 96 threads cold, down's median thread 34 us, the slowest 53), while the streams want
// to stay long and contiguous (handing out small chunks from a shared counter made every thread slower: 8-row chunks,
// gate|up's median thread 60 -> 82 us cold). So each thread starts on its even contiguous share, held as [lo, hi) in a
// word of its own, and takes it a step at a time from the front; a thread out of rows takes the back half of the
// largest share left in its group (GRP threads: a CCX, 8 cores sharing an L3, at the pool's CPU order) and goes on with
// that as its own. Stealing across groups (a scan of every thread's word) cost more than it saved; groups of 4, 16 or
// 32 did worse than 8. Every row is still computed whole by one thread, so the results do not depend on the split.
// CL_STATIC=1: the even shares alone.
constexpr int MAXT = 256, GRP = 8;
// a step: gate|up in rows of one expert and token (cl_moe_m's cost units: one decode counts gu_decode); down in cost
// units (a row of one expert, 1 + its uses); the least share worth stealing is two steps
constexpr int GU_STEP = 4, DOWN_M_STEP = 16;
struct alignas(64) Share {
    std::atomic<uint64_t> v{0};  // lo in the low 32 bits, hi in the high: empty (lo >= hi) at the end of every phase
};
Share shares[MAXT];
bool dynamic_split = true;
inline uint64_t pack(uint32_t lo, uint32_t hi) { return (uint64_t) hi << 32 | lo; }

// thread t's next step of at most `step` units from the front of its share, not crossing a multiple of `seg`
bool take(int t, int step, int seg, int& a, int& b) {
    std::atomic<uint64_t>& w = shares[t].v;
    uint64_t v = w.load(std::memory_order_relaxed);
    for (;;) {
        const uint32_t lo = (uint32_t) v, hi = (uint32_t) (v >> 32);
        if (lo >= hi) return false;
        const uint32_t end = std::min({lo + (uint32_t) step, hi, (lo / seg + 1) * (uint32_t) seg});
        if (w.compare_exchange_weak(v, pack(end, hi), std::memory_order_relaxed)) {
            a = (int) lo;
            b = (int) end;
            return true;
        }
    }
}

// thread t, out of rows: the back half of the largest share of at least `least` units left in its group, as its own
bool steal(int t, uint32_t least) {
    const int g0 = t / GRP * GRP, g1 = std::min(g0 + GRP, n_active);
    for (;;) {
        int best = -1;
        uint32_t most = least - 1;
        uint64_t bv = 0;
        for (int u = g0; u < g1; u++) {
            const uint64_t v = u == t ? 0 : shares[u].v.load(std::memory_order_relaxed);
            const uint32_t lo = (uint32_t) v, hi = (uint32_t) (v >> 32);
            if (hi > lo && hi - lo > most) {
                most = hi - lo;
                best = u;
                bv = v;
            }
        }
        if (best < 0) return false;
        const uint32_t lo = (uint32_t) bv, hi = (uint32_t) (bv >> 32), mid = lo + (hi - lo + 1) / 2;
        if (shares[best].v.compare_exchange_strong(bv, pack(lo, mid), std::memory_order_relaxed)) {
            shares[t].v.store(pack(mid, hi), std::memory_order_relaxed);
            return true;
        }
    }
}

// thread t's rows of a phase of `units` units: its even share, then what it steals; body(a, b) per step
template <class F>
void rows(int t, int units, int step, int seg, uint32_t least, F&& body) {
    const uint32_t lo = (int64_t) units * t / n_active, hi = (int64_t) units * (t + 1) / n_active;
    shares[t].v.store(pack(lo, hi), std::memory_order_relaxed);
    do {
        for (int a, b; take(t, step, seg, a, b);) body(a, b);
    } while (steal(t, least));
}

// h quantized for down by the thread that computes its last gate|up rows, so down starts without waiting for it (at
// the start of down, the first k threads' quantization and everyone's wait on it took 3.1-3.5 us at k = 10): each
// thread adds the rows it did per union expert, and the one that brings it to FF
// quantizes its h and resets the count
struct alignas(64) Count {
    std::atomic<int> n{0};
};
Count gu_done[MAXP];
constexpr int DB = 32;  // cl_moe_m's down: rows per block
Count down_done[D / DB];

template <class Q>
struct Finish {
    Q quantize;  // quantize(e): union expert e's h for down
    int e = -1, n = 0;
    void flush() {
        if (e >= 0 && gu_done[e].n.fetch_add(n, std::memory_order_acq_rel) + n == FF) {
            quantize(e);
            gu_done[e].n.store(0, std::memory_order_relaxed);
        }
        e = -1;
        n = 0;
    }
    void add(int route, int rows) {  // after computing them
        if (route != e) flush();
        e = route;
        n += rows;
    }
};
template <class Q>
Finish(Q) -> Finish<Q>;

// down's rows for several uses at once: one weight row (IQ4_NL or Q8_0, against q8_0 activations) decoded once per
// block for up to 8 uses, each use's sum the arithmetic of ggml-cpu's AVX2 vec_dot for that pair (x86 quants.c:
// ggml_vec_dot_iq4_nl_q8_0's two accumulators over even and odd blocks, ggml_vec_dot_q8_0_q8_0's one; the integer
// dots exact either way), so every value is vec_dot's to the bit. At 1,000 tokens a call ~20 uses share an expert
// row, and the 2,560 x 20 vec_dot calls an expert were most of down's time.

inline float hsum8(const __m256 x) {   // ggml's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline __m256i add_epi8_pairs(const __m256i x, const __m256i y) {   // ggml's mul_add_epi8
    return _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
}

inline __m256 i8_pairs_float(const __m256i ax, const __m256i sy) {  // ggml's mul_sum_us8_pairs_float
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy));
#else
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(ax, sy)));
#endif
}

inline float f16(ggml_fp16_t h) { return _cvtsh_ss(h); }

template <int NT>
void iq4nl_rows(const block_iq4_nl* x, int nb, const block_q8_0* const* y, float* const* out) {
    const __m128i values128 = _mm_loadu_si128((const __m128i*) kvalues_iq4nl), m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 a1[NT], a2[NT];
    for (int t = 0; t < NT; t++) a1[t] = a2[t] = _mm256_setzero_ps();
    for (int ib = 0; ib + 1 < nb; ib += 2) {
        const __m128i b1 = _mm_loadu_si128((const __m128i*) x[ib].qs), b2 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs);
        const __m256i q1 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b1, 4), m4b)),
                                            _mm_shuffle_epi8(values128, _mm_and_si128(b1, m4b)));
        const __m256i q2 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b2, 4), m4b)),
                                            _mm_shuffle_epi8(values128, _mm_and_si128(b2, m4b)));
        const float d1 = f16(x[ib].d), d2 = f16(x[ib + 1].d);
        for (int t = 0; t < NT; t++) {
            const __m256i p1 = _mm256_madd_epi16(add_epi8_pairs(q1, _mm256_loadu_si256((const __m256i*) y[t][ib].qs)), mone);
            const __m256i p2 = _mm256_madd_epi16(add_epi8_pairs(q2, _mm256_loadu_si256((const __m256i*) y[t][ib + 1].qs)),
                                                 mone);
            a1[t] = _mm256_fmadd_ps(_mm256_set1_ps(f16(y[t][ib].d) * d1), _mm256_cvtepi32_ps(p1), a1[t]);
            a2[t] = _mm256_fmadd_ps(_mm256_set1_ps(f16(y[t][ib + 1].d) * d2), _mm256_cvtepi32_ps(p2), a2[t]);
        }
    }
    for (int t = 0; t < NT; t++) *out[t] = hsum8(_mm256_add_ps(a1[t], a2[t]));
}

template <int NT>
void q8_rows(const block_q8_0* x, int nb, const block_q8_0* const* y, float* const* out) {
    __m256 acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ib++) {
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const float dx = f16(x[ib].d);
        for (int t = 0; t < NT; t++) {
            const __m256i sy = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*) y[t][ib].qs), qx);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * f16(y[t][ib].d)), i8_pairs_float(ax, sy), acc[t]);
        }
    }
    for (int t = 0; t < NT; t++) *out[t] = hsum8(acc[t]);
}

template <class X, void (*F1)(const X*, int, const block_q8_0* const*, float* const*),
          void (*F2)(const X*, int, const block_q8_0* const*, float* const*),
          void (*F3)(const X*, int, const block_q8_0* const*, float* const*),
          void (*F4)(const X*, int, const block_q8_0* const*, float* const*),
          void (*F8)(const X*, int, const block_q8_0* const*, float* const*)>
void rows_nt(int nt, const X* x, int nb, const block_q8_0* const* y, float* const* out) {
    switch (nt) {   // (the per-use arithmetic is the same at any NT: 5..7 take 4 + the rest)
        case 1: F1(x, nb, y, out); return;
        case 2: F2(x, nb, y, out); return;
        case 3: F3(x, nb, y, out); return;
        case 4: F4(x, nb, y, out); return;
        case 8: F8(x, nb, y, out); return;
        default: F4(x, nb, y, out); rows_nt<X, F1, F2, F3, F4, F8>(nt - 4, x, nb, y + 4, out + 4); return;
    }
}

// GLM's down formats (2026-11-13): IQ4_XS and Q5_K against q8_K activations, a weight row's super-blocks decoded
// once for up to 8 uses, each use's sum the arithmetic of ggml-cpu's AVX2 vec_dot for that pair (x86 quants.c:
// ggml_vec_dot_iq4_xs_q8_K, ggml_vec_dot_q5_K_q8_K; the integer parts exact in any order), so every value is
// vec_dot's to the bit. The decode's one-token calls go through vec_dot itself; a prefill's rows came here as 2,048 x
// uses vec_dot calls an expert (down ran at less than half gate|up's rate).
constexpr int KMAXB = 16;   // super-blocks a row (FF / 256)

template <int NT>
void iq4xs_rows(const block_iq4_xs* x, int nb, const block_q8_K* const* y, float* const* out) {
    const __m128i values128 = _mm_loadu_si128((const __m128i*) kvalues_iq4nl), m4b = _mm_set1_epi8(0x0f);
    __m256i q4[KMAXB][QK_K / 32];
    int16_t ls[KMAXB][QK_K / 32];
    float xd[KMAXB];
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t* qs = x[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m128i b1 = _mm_loadu_si128((const __m128i*) qs), b2 = _mm_loadu_si128((const __m128i*) (qs + 16));
            qs += 32;
            q4[ibl][ib] = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b1, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(b1, m4b)));
            q4[ibl][ib + 1] = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b2, 4), m4b)),
                                               _mm_shuffle_epi8(values128, _mm_and_si128(b2, m4b)));
            ls[ibl][ib] = (int16_t) (((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32);
            ls[ibl][ib + 1] = (int16_t) (((x[ibl].scales_l[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
        }
        xd[ibl] = f16(x[ibl].d);
    }
    for (int t = 0; t < NT; t++) {
        __m256 accum = _mm256_setzero_ps();
        for (int ibl = 0; ibl < nb; ++ibl) {
            const int8_t* q8 = y[t][ibl].qs;
            __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*) q8), q8b_2 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                q8 += 64;
                const __m256i p_1 = _mm256_madd_epi16(add_epi8_pairs(q4[ibl][ib], q8b_1), _mm256_set1_epi16(ls[ibl][ib]));
                const __m256i p_2 = _mm256_madd_epi16(add_epi8_pairs(q4[ibl][ib + 1], q8b_2),
                                                      _mm256_set1_epi16(ls[ibl][ib + 1]));
                sumi1 = _mm256_add_epi32(p_1, sumi1);
                sumi2 = _mm256_add_epi32(p_2, sumi2);
            }
            accum = _mm256_fmadd_ps(_mm256_set1_ps(xd[ibl] * y[t][ibl].d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
                                    accum);
        }
        *out[t] = hsum8(accum);
    }
}

template <int NT>
void q5k_rows(const block_q5_K* x, int nb, const block_q8_K* const* y, float* const* out) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF), mone = _mm256_set1_epi8(1);
    __m256i q5[KMAXB][QK_K / 32];
    int16_t sc[KMAXB][8], mn[KMAXB][8];
    float xd[KMAXB], xm[KMAXB];
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8_t* b = (const uint8_t*) utmp;   // 8 scales, then 8 mins (vec_dot's mins_and_scales)
        for (int k = 0; k < 8; k++) {
            sc[i][k] = b[k];
            mn[i][k] = b[8 + k];
        }
        const uint8_t* qp = x[i].qs;
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) qp);
            qp += 32;
            const __m256i h0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5[i][2 * j] = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), h0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i h1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5[i][2 * j + 1] = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), h1);
            hmask = _mm256_slli_epi16(hmask, 1);
        }
        ggml_fp16_t dd[2];   // d, dmin: the block's first 4 bytes (a union's struct, named apart in C++)
        std::memcpy(dd, &x[i], sizeof dd);
        xd[i] = f16(dd[0]);
        xm[i] = f16(dd[1]);
    }
    for (int t = 0; t < NT; t++) {
        __m256 acc = _mm256_setzero_ps();
        float summs = 0.f;
        for (int i = 0; i < nb; ++i) {
            const block_q8_K& yb = y[t][i];
            const float d = yb.d * xd[i], dmin = -yb.d * xm[i];
            int32_t ms = 0;   // vec_dot's madd + hadds of the mins against the bsums' pair sums: exact integers
            for (int k = 0; k < 8; k++) ms += (int32_t) mn[i][k] * (int32_t) (int16_t) (yb.bsums[2 * k] + yb.bsums[2 * k + 1]);
            summs += dmin * (float) ms;
            const int8_t* q8 = yb.qs;
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8), q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                q8 += 64;
                const __m256i p0 = _mm256_madd_epi16(_mm256_set1_epi16(sc[i][2 * j]), _mm256_maddubs_epi16(q5[i][2 * j], q8_0));
                const __m256i p1 = _mm256_madd_epi16(_mm256_set1_epi16(sc[i][2 * j + 1]),
                                                     _mm256_maddubs_epi16(q5[i][2 * j + 1], q8_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }
        *out[t] = hsum8(acc) + summs;
    }
}

// row r against uses p0 .. p1 - 1 (q8_K activations), 8 uses a call (the per-use arithmetic is the same at any NT)
template <class X>
void k_rows(const X* x, int nb, int p0, int p1, int r, void (*f8)(const X*, int, const block_q8_K* const*, float* const*),
            void (*f1)(const X*, int, const block_q8_K* const*, float* const*)) {
    const block_q8_K* y[8];
    float* out[8];
    for (int p = p0; p < p1; p += 8) {
        const int nt = std::min(8, p1 - p);
        for (int t = 0; t < nt; t++) {
            y[t] = (const block_q8_K*) jm.hq[p + t];
            out[t] = &jm.v[p + t][r];
        }
        if (nt == 8) {
            f8(x, nb, y, out);
        } else {
            for (int t = 0; t < nt; t++) f1(x, nb, y + t, out + t);
        }
    }
}

// row r's dot with every use p0 .. p1 - 1's hq, into v[p][r], 8 uses a pass; false: no batched rows for the type
bool down_rows(const Proj& P, const uint8_t* row, int p0, int p1, int r) {
    if (P.type == GGML_TYPE_IQ4_XS && FF % QK_K == 0 && FF / QK_K <= KMAXB) {
        k_rows((const block_iq4_xs*) row, FF / QK_K, p0, p1, r, iq4xs_rows<8>, iq4xs_rows<1>);
        return true;
    }
    if (P.type == GGML_TYPE_Q5_K && FF % QK_K == 0 && FF / QK_K <= KMAXB) {
        k_rows((const block_q5_K*) row, FF / QK_K, p0, p1, r, q5k_rows<8>, q5k_rows<1>);
        return true;
    }
    const int nb = FF / 32;
    const bool iq4 = P.type == GGML_TYPE_IQ4_NL;
    if ((!iq4 && P.type != GGML_TYPE_Q8_0) || nb % 2) return false;
    const block_q8_0* y[8];
    float* out[8];
    for (int p = p0; p < p1; p += 8) {
        const int nt = std::min(8, p1 - p);
        for (int t = 0; t < nt; t++) {
            y[t] = (const block_q8_0*) jm.hq[p + t];
            out[t] = &jm.v[p + t][r];
        }
        if (iq4)
            rows_nt<block_iq4_nl, iq4nl_rows<1>, iq4nl_rows<2>, iq4nl_rows<3>, iq4nl_rows<4>, iq4nl_rows<8>>(
                nt, (const block_iq4_nl*) row, nb, y, out);
        else
            rows_nt<block_q8_0, q8_rows<1>, q8_rows<2>, q8_rows<3>, q8_rows<4>, q8_rows<8>>(
                nt, (const block_q8_0*) row, nb, y, out);
    }
    return true;
}
// gate|up's IQ3_S rows for any number of uses: each row's blocks decoded once into L1 (Strata's Fmt<21>::decode: the
// grid values, the sign mask, the scales), then the uses 8 at a time through Strata's row_dot arithmetic per use
// (iq_avx512.cpp: an int32 sum per block, one fp32 fma a block), so h is iq512_gu_rows's to the bit; that kernel
// decodes a row once per 8 uses (at 1,000 tokens a call ~20 uses share an expert).
constexpr int TY_IQ3_S = 21, IQ3S_BYTES = 110, NBK = D / QK_K;   // 10 blocks a row
inline int gu_decode(const Layer& L) {
    return L.iq512 ? 16 : 1;
}

inline __m512i scales4(int s0, int s1, int s2, int s3) {   // Strata's
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

struct Row3s {               // a decoded row: per block its scale and 4 sub-blocks of 64 values
    __m512i g[NBK][4], sc[NBK][4];
    __mmask64 m[NBK][4];
    float dx[NBK];
};

void decode3s(const uint8_t* row, Row3s& o) {
    for (int i = 0; i < NBK; i++) {
        const uint8_t* b = row + (size_t) i * IQ3S_BYTES;
        for (int j = 0; j < 4; j++) {
            const uint8_t* q = b + 2 + 16 * j;
            const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
            o.g[i][j] = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                                         G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                                         G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                                         G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
            uint64_t mm;
            std::memcpy(&mm, b + 74 + 8 * j, 8);
            o.m[i][j] = _cvtu64_mask64(mm);
            const uint8_t sb = b[106 + j];
            const int sa = 2 * (sb & 15) + 1, sz = 2 * (sb >> 4) + 1;
            o.sc[i][j] = scales4(sa, sa, sz, sz);
        }
        uint16_t d;
        std::memcpy(&d, b, 2);
        o.dx[i] = _cvtsh_ss(d);   // (Fmt<21>::K = 1)
    }
}

template <int NT>
void dot3s(const Row3s& w, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < NBK; ++i) {
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i g = w.g[i][j], sc = w.sc[i][j];
            const __mmask64 m = w.m[i][j];
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(w.dx[i] * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

void dot3s_nt(int nt, const Row3s& w, const block_q8_K* const* y, float* res) {
    switch (nt) {   // (the per-use arithmetic is the same at any NT: 5..7 take 4 + the rest)
        case 1: dot3s<1>(w, y, res); return;
        case 2: dot3s<2>(w, y, res); return;
        case 3: dot3s<3>(w, y, res); return;
        case 4: dot3s<4>(w, y, res); return;
        case 8: dot3s<8>(w, y, res); return;
        default: dot3s<4>(w, y, res); dot3s_nt(nt - 4, w, y + 4, res + 4); return;
    }
}

// rows r0 .. r1 of an expert's gate|up (gate at blob, up at blob + up_off, gu_row bytes a row) for nu uses: ff[p][r] =
// silu(min(gate . act[p], lim)) * clamp(up . act[p], -lim, lim), as iq512_gu_rows (lim: CL_SWIGLU_LIM)
thread_local Row3s rg, ru;
void gu3s_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const void* const* act, int nu, float* const* ff, int r0,
               int r1) {
    float g[8], u[8];
    for (int r = r0; r < r1; ++r) {
        decode3s(blob + (size_t) r * gu_row, rg);
        decode3s(blob + up_off + (size_t) r * gu_row, ru);
        for (int q = 0; q < nu; q += 8) {
            const int nt = std::min(8, nu - q);
            const block_q8_K* y[8];
            for (int t = 0; t < nt; t++) y[t] = (const block_q8_K*) act[q + t];
            dot3s_nt(nt, rg, y, g);
            dot3s_nt(nt, ru, y, u);
            constexpr float lim = CL_SWIGLU_LIM;
            for (int t = 0; t < nt; ++t) {
                const float gc = std::min(g[t], lim), uc = std::min(std::max(u[t], -lim), lim);
                ff[q + t][r] = (gc / (1.f + std::exp(-gc))) * uc;
            }
        }
    }
}

// CL_DOWN_ROWS=0: the previous rows (Strata's gate|up 8 uses a decode, ggml's vec_dot a use, the sums row by row)
bool batched_down = getenv("CL_DOWN_ROWS") == nullptr || atoi(getenv("CL_DOWN_ROWS")) != 0;

alignas(64) std::atomic<int> xq_done{0};

// GLM's router on the lane's threads (cl_glm_route, phase 6): logits[e] = W_r[e] . x, the rows split over the active
// threads; W_r fp32 [NX][D] per layer (cl_router_load), the routing's bias beside it
std::vector<float> router_w[LAYERS], router_b[LAYERS];
struct JobR {
    const float* w;
    const float* x;
    float* logits;
} jr;

// one row's routing from its logits (cl_glm_route's and cl_glm_route_m's, the same bits)
void glm_pick(int layer, const float* logits, int k, float scale, int32_t* ids, float* w) {
    float p[NX], sel[NX];
    for (int e = 0; e < NX; e++) {
        p[e] = 1.f / (1.f + std::exp(-logits[e]));
        sel[e] = p[e] + router_b[layer][e];
    }
    float sum = 0;
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < NX; e++) {
            if (sel[e] != -INFINITY && (best < 0 || sel[e] > sel[best])) best = e;
        }
        ids[j] = best;
        sel[best] = -INFINITY;
        sum += p[best];
    }
    for (int j = 0; j < k; j++) w[j] = p[ids[j]] / sum * scale;
}

// one row on the calling thread: router_rows' sums (the same order per expert), then glm_pick
void glm_route_one(int layer, const float* x, int k, float scale, int32_t* ids, float* w, float* logits_out) {
    alignas(64) float logits[NX];
    const float* wr = router_w[layer].data();
    for (int e = 0; e < NX; e++) {
        const float* row = wr + (size_t) e * D;
        float acc[16] = {};
        for (int i = 0; i < D; i += 16) {
            for (int j = 0; j < 16; j++) acc[j] += row[i + j] * x[i + j];
        }
        float s = 0;
        for (int j = 0; j < 16; j++) s += acc[j];
        logits[e] = s;
    }
    glm_pick(layer, logits, k, scale, ids, w);
    if (logits_out) std::memcpy(logits_out, logits, sizeof(logits));
}

// cl_glm_route_m (a prefill chunk's rows): each thread routes whole rows, every row as cl_glm_route does
struct JobRM {
    int layer, m, k;
    float scale;
    const float* x;
    int32_t* ids;
    float* w;
} jrm;
void router_rows(int t) {
    for (int e = t; e < NX; e += n_active) {
        const float* row = jr.w + (size_t) e * D;
        float acc[16] = {};
        for (int i = 0; i < D; i += 16) {
            for (int j = 0; j < 16; j++) acc[j] += row[i + j] * jr.x[i + j];
        }
        float s = 0;
        for (int j = 0; j < 16; j++) s += acc[j];
        jr.logits[e] = s;
    }
}

// cl_moe_m's phases: 3 the M activations quantized (thread i row i), then gate|up (each union expert's h quantized by
// the thread that finishes it); 4 down
void run_phase_m(int phase, int t) {
    if (phase == 6) {
        router_rows(t);
        return;
    }
    if (phase == 7) {
        for (int i = t; i < jrm.m; i += n_active)
            glm_route_one(jrm.layer, jrm.x + (size_t) i * D, jrm.k, jrm.scale, jrm.ids + (size_t) i * jrm.k,
                          jrm.w + (size_t) i * jrm.k, nullptr);
        return;
    }
    const Layer& L = layers[jm.layer];
    const int n = n_active;
    const bool batch = batched_down && jm.m > 8;   // a prefill's call (the decode's passes keep their rows)
    if (phase == 3) {
        for (int i = t; i < jm.m; i += n) {
            quantize_row(L.gate.tr, jm.x + (size_t) i * D, jm.xq[i]);
            xq_done.fetch_add(1, std::memory_order_acq_rel);
        }
        while (xq_done.load(std::memory_order_acquire) < jm.m) _mm_pause();
        TR_T(t, 0, 1);
        const auto* dq = ggml_get_type_traits_cpu(L.down.tr->vec_dot_type);
        Finish f{[&](int j) {
            for (int p = jm.first[j]; p < jm.first[j + 1]; p++) dq->from_float(jm.h[p], jm.hq[p], FF);
        }};
        // the cost [a, b); union expert j's row r starts at cost[j] + r * c_j
        const size_t up_off = L.up.data - L.gate.data;
        auto gu = [&](int64_t a, int64_t b) {
            for (int j = 0; j < jm.u; j++) {
                if (jm.cost[j + 1] <= a) continue;
                if (jm.cost[j] >= b) break;
                const int nu = jm.first[j + 1] - jm.first[j], c = jm.decode + nu;
                const int r0 = (int) std::clamp<int64_t>((a - jm.cost[j] + c - 1) / c, 0, FF);
                const int r1 = (int) std::clamp<int64_t>((b - jm.cost[j] + c - 1) / c, 0, FF);
                if (r0 >= r1) continue;
                const size_t base = (size_t) jm.ex[j] * FF;
                if (L.iq512 && batch && nu > 8 && L.gate.type == TY_IQ3_S) {
                    gu3s_rows(L.gate.data + base * L.gate.row_bytes, L.gate.row_bytes, up_off, &jm.act[jm.first[j]],
                              nu, &jm.ff[jm.first[j]], r0, r1);
                } else if (L.iq512) {
                    // in groups of <= 8 uses: iq512_gu_rows computes only the first 8 of nt (its switch's default is
                    // gu_rows<TY, 8>), which only calls of more than 8 tokens reach. Per use the same bits
                    for (int q = 0; q < nu; q += 8) {
#ifdef __AVX512VBMI__
                        if (L.gate.type == TY_IQ3_S && iq3s_vbmi) {
                            cl3s::gu_rows_nt(std::min(8, nu - q), L.gate.data + base * L.gate.row_bytes,
                                             L.gate.row_bytes, up_off, D, &jm.act[jm.first[j] + q],
                                             &jm.ff[jm.first[j] + q], r0, r1, CL_SWIGLU_LIM);
                            continue;
                        }
#endif
                        if constexpr (kClamp) {
                            gu_rows_clamped(L.gate.type, L.gate.data + base * L.gate.row_bytes, L.gate.row_bytes,
                                            up_off, &jm.act[jm.first[j] + q], std::min(8, nu - q),
                                            &jm.ff[jm.first[j] + q], r0, r1);
                        } else {
                            strata::kernels::cpu::iq512_gu_rows(L.gate.type, L.gate.data + base * L.gate.row_bytes,
                                                                L.gate.row_bytes, up_off, D, &jm.act[jm.first[j] + q],
                                                                std::min(8, nu - q), &jm.ff[jm.first[j] + q], r0, r1);
                        }
                    }
                } else {
                    for (int r = r0; r < r1; r++) {
                        const uint8_t* gr = L.gate.data + (base + r) * L.gate.row_bytes;
                        const uint8_t* ur = L.up.data + (base + r) * L.up.row_bytes;
                        for (int p = jm.first[j]; p < jm.first[j + 1]; p++) {
                            float g, v;
                            L.gate.tr->vec_dot(D, &g, 0, gr, 0, jm.act[p], 0, 1);
                            L.up.tr->vec_dot(D, &v, 0, ur, 0, jm.act[p], 0, 1);
                            constexpr float lim = CL_SWIGLU_LIM;
                            jm.h[p][r] = silu(std::min(g, lim)) * std::min(std::max(v, -lim), lim);
                        }
                    }
                }
                f.add(j, r1 - r0);
            }
        };
        const int total = jm.cost[jm.u], step = GU_STEP * (jm.decode + 1);
        if (dynamic_split) {
            rows(t, total, step, total, 2 * step, gu);
        } else {
            gu((int64_t) total * t / n, (int64_t) total * (t + 1) / n);
        }
        f.flush();
        return;
    }
    TR_T(t, 1, 1);
    // down, a block of DB rows at a time, expert by expert within it: v[p][r] = down_j[r] . hq[p] for each use p of
    // union expert j, so a thread reads DB rows of one expert in a stream (row by row over all U experts, each row
    // came from U places at once: the phase ran ~25% under the RAM's rate at M = 4). The units are a block's rows of
    // each expert, weighted by 1 + its uses (rows of a shared expert cost more); the thread that completes a block's
    // rows sums them: each token's sum in its route order
    auto down = [&](int a, int b) {  // [a, b) within one block
        const int blk = a / jm.dcost[jm.u], a0 = a - blk * jm.dcost[jm.u], b0 = b - blk * jm.dcost[jm.u];
        int done = 0;
        for (int j = 0; j < jm.u; j++) {
            if (jm.dcost[j + 1] <= a0) continue;
            if (jm.dcost[j] >= b0) break;
            const int c = 1 + jm.first[j + 1] - jm.first[j];
            const int r0 = std::clamp((a0 - jm.dcost[j] + c - 1) / c, 0, DB) + blk * DB;
            const int r1 = std::clamp((b0 - jm.dcost[j] + c - 1) / c, 0, DB) + blk * DB;
            const uint8_t* row = L.down.data + ((size_t) jm.ex[j] * D + r0) * L.down.row_bytes;
            for (int r = r0; r < r1; r++, row += L.down.row_bytes)
                if (!batch || !down_rows(L.down, row, jm.first[j], jm.first[j + 1], r))
                    for (int p = jm.first[j]; p < jm.first[j + 1]; p++)
                        L.down.tr->vec_dot(FF, &jm.v[p][r], 0, row, 0, jm.hq[p], 0, 1);
            done += r1 - r0;
        }
        if (!done || down_done[blk].n.fetch_add(done, std::memory_order_acq_rel) + done != DB * jm.u) return;
        down_done[blk].n.store(0, std::memory_order_relaxed);
        if (batch) {
            // token by token, its routes' rows of the block in order: each value's fma chain as below (bit for bit),
            // the block's 32 rows of a use in two cache lines (row by row it touched m k lines a row)
            for (int i = 0; i < jm.m; i++) {
                float s[DB] = {};
                for (int e = 0; e < jm.kn[i]; e++) {
                    const float we = jm.w[i][e], *v = &jm.v[jm.use_of[i][e]][blk * DB];
                    for (int r = 0; r < DB; r++) s[r] = std::fma(we, v[r], s[r]);
                }
                std::memcpy(&jm.out[(size_t) i * D + blk * DB], s, sizeof s);
            }
            return;
        }
        for (int r = blk * DB; r < (blk + 1) * DB; r++)
            for (int i = 0; i < jm.m; i++) {
                float s = 0;
                // fused as cl_moe's s += w v is (fp-contract): left unfused, GCC vectorizes the products (k >= 8) and
                // the sums differ from cl_moe's in the last bit
                for (int e = 0; e < jm.kn[i]; e++) s = std::fma(jm.w[i][e], jm.v[jm.use_of[i][e]][r], s);
                jm.out[(size_t) i * D + r] = s;
            }
    };
    const int cb = jm.dcost[jm.u], total = D / DB * cb;
    if (dynamic_split) {
        rows(t, total, DOWN_M_STEP, cb, 2 * DOWN_M_STEP, down);
    } else {
        const int a = (int64_t) total * t / n, b = (int64_t) total * (t + 1) / n;
        for (int k = a / cb; k * cb < b; k++) down(std::max(a, k * cb), std::min(b, (k + 1) * cb));
    }
}

// The warm (2026-11-08): between a pass's exchanges the card computes the next layer (~370 us) while the lane's threads
// spin. cl_warm posts phase 5, not waited for: threads 1..n-1 read the given experts of a layer (the host loop's guess
// at the next exchange's routes) into their caches, split as cl_moe_m would split that union (gate|up rows in even
// shares in the list's order, down's blocks of DB rows expert by expert), so a right guess finds most of its rows in
// the core, or the CCX, that computes them. The next phase (or cl_warm) stops it and waits for the threads: reads
// only, so the results are the same bits. CPU-only bench on the census's 4-row unions (~31 experts): 346 us cold,
// 289 warmed by a 40-expert guess (58% of the union), 269 warmed by the union itself.
constexpr int MAXWARM = 64;
struct WarmJob {
    int layer = 0, n = 0;
    int ex[MAXWARM];
} wj;
alignas(64) std::atomic<bool> warm_stop{false};
bool warm_posted = false;  // the caller's: a warm may still be running

void run_warm(int t) {
    const Layer& L = layers[wj.layer];
    const int n = n_active - 1, i = t - 1;  // the workers: the caller is back on the card's socket
    __m512i acc = _mm512_setzero_si512();
    // [p, p + bytes) touched a cache line at a time (the line holding p first); false: stopped
    auto touch = [&](const uint8_t* p, size_t bytes) {
        const uint8_t* a = (const uint8_t*) ((uintptr_t) p & ~(uintptr_t) 63);
        for (const uint8_t* q = a; q < p + bytes; q += 64) {
            if (!(((uintptr_t) q) & 4095) && warm_stop.load(std::memory_order_relaxed)) return false;
            acc = _mm512_or_si512(acc, _mm512_load_si512(q));
        }
        return true;
    };
    const size_t up_off = L.up.data - L.gate.data;
    const int64_t gt = (int64_t) wj.n * FF;  // gate|up: the list's rows, an even share each
    for (int64_t r = gt * i / n, r1 = gt * (i + 1) / n; r < r1;) {
        const int j = (int) (r / FF), a = (int) (r % FF), b = (int) std::min<int64_t>(FF, a + (r1 - r));
        const uint8_t* g = L.gate.data + ((size_t) wj.ex[j] * FF + a) * L.gate.row_bytes;
        if (!touch(g, (size_t) (b - a) * L.gate.row_bytes) || !touch(g + up_off, (size_t) (b - a) * L.up.row_bytes))
            goto done;
        r += b - a;
    }
    {
        const int64_t dt = (int64_t) (D / DB) * wj.n;  // down: (block, expert) units, block-major, as cl_moe_m's
        for (int64_t u = dt * i / n, u1 = dt * (i + 1) / n; u < u1; u++) {
            const int blk = (int) (u / wj.n), j = (int) (u % wj.n);
            if (!touch(L.down.data + ((size_t) wj.ex[j] * D + blk * DB) * L.down.row_bytes, DB * L.down.row_bytes))
                goto done;
        }
    }
done:
    asm volatile("" : : "v"(acc));  // the loads are the point: kept (a store nothing reads let GCC drop them all)
}

// a posted warm stopped and its threads back (the caller's, before anything else uses the pool)
void warm_drain() {
    if (!warm_posted) return;
    warm_stop.store(true, std::memory_order_relaxed);
    while (phase_left.load(std::memory_order_acquire)) _mm_pause();
    warm_stop.store(false, std::memory_order_relaxed);
    warm_posted = false;
}

void pin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

void worker(int t, int cpu) {
    pin(cpu);
    uint32_t seen = phase_gen.load(std::memory_order_acquire);
    for (;;) {
        // spin for new work, then sleep
        auto t0 = std::chrono::steady_clock::now();
        uint32_t g;
        int spins = 0;
        while ((g = phase_gen.load(std::memory_order_acquire)) == seen && !stop.load(std::memory_order_relaxed)) {
            _mm_pause();
            if (++spins == 4096) {
                spins = 0;
                if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(CL_IDLE_MS)) {
                    pthread_mutex_lock(&mu);
                    sleepers++;
                    while (phase_gen.load(std::memory_order_acquire) == seen && !stop.load()) {
                        pthread_cond_wait(&cv, &mu);
                    }
                    sleepers--;
                    pthread_mutex_unlock(&mu);
                    t0 = std::chrono::steady_clock::now();
                }
            }
        }
        if (stop.load()) return;
        seen = g;
        if (t < n_active) {
            const int id = phase_id.load(std::memory_order_relaxed);
            if (id == 5) {
                run_warm(t);
                phase_left.fetch_sub(1, std::memory_order_acq_rel);
                continue;
            }
            TR_T(t, id & 1 ? 0 : 1, 0);
            run_phase_m(id, t);
            TR_T(t, id & 1 ? 0 : 1, 2);
            phase_left.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
}

void phase(int id) {
    warm_drain();
    TR_C(id & 1 ? TR_PUB1 : TR_PUB2);
    TR_T(0, id & 1 ? 0 : 1, 0);
    phase_id.store(id, std::memory_order_relaxed);
    phase_left.store(n_active - 1, std::memory_order_relaxed);
    phase_gen.fetch_add(1, std::memory_order_acq_rel);
    if (sleepers.load()) {
        pthread_mutex_lock(&mu);
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mu);
    }
    run_phase_m(id, 0);
    TR_T(0, id & 1 ? 0 : 1, 2);
    while (phase_left.load(std::memory_order_acquire)) _mm_pause();
    TR_C(id & 1 ? TR_DONE1 : TR_DONE2);
}

inline float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

extern "C" {

// Start the pool: n threads (the caller is thread 0), threads 1..n-1 pinned to cpus[1..n-1] (cpus may be null).
int cl_init(int n, const int* cpus) {
    ggml_cpu_init();
    if (!workers.empty()) return -1;
    n_threads = n_active = n < 1 ? 1 : n > MAXT ? MAXT : n;
    dynamic_split = getenv("CL_STATIC") == nullptr;
    if (cpus) pin(cpus[0]);
    for (int t = 1; t < n_threads; t++) workers.emplace_back(worker, t, cpus ? cpus[t] : -1);
    return n_threads;
}

// The caller's cumulative time per step (ns): quantize hn, gate|up, quantize h, down; reset after reading.
void cl_stats(uint64_t* out) {
    for (int i = 0; i < 4; i++) {
        out[i] = phase_ns[i];
        phase_ns[i] = 0;
    }
}

#ifdef CL_TRACE
// The stamps since the last call (TSC ticks; tsc_hz from cl_tsc_hz): caller [n][TR_N]; threads [n][nt][2][3] for
// threads 0..nt-1 (phase 0 gate|up, 1 down; seen, rows began, done). Returns n, and resets.
int cl_trace(uint64_t* caller, uint64_t* threads, int max, int nt) {
    const int n = std::min(tr_n, max);
    nt = std::min(nt, TR_THREADS);
    for (int c = 0; c < n; c++) {
        std::memcpy(caller + (size_t) c * TR_N, tr_caller[c], sizeof(tr_caller[c]));
        for (int t = 0; t < nt; t++) std::memcpy(threads + ((size_t) c * nt + t) * 6, tr_thread[t].s[c], 6 * 8);
    }
    tr_n = 0;
    std::memset(tr_caller, 0, sizeof(tr_caller));
    for (int t = 0; t < TR_THREADS; t++) std::memset(&tr_thread[t], 0, sizeof(TrThread));
    return n;
}

// The TSC's rate, measured against the steady clock over 50 ms.
double cl_tsc_hz() {
    const auto a = std::chrono::steady_clock::now();
    const uint64_t t0 = tsc();
    while (std::chrono::steady_clock::now() - a < std::chrono::milliseconds(50)) {}
    const uint64_t t1 = tsc();
    const auto b = std::chrono::steady_clock::now();
    return (t1 - t0) / std::chrono::duration<double>(b - a).count();
}
#endif

// The most tokens one cl_moe_m call takes.
int cl_max_m() { return MAXM; }
int cl_d() { return D; }
int cl_nx() { return NX; }

// How many of the pool's threads take part from now on (1..n); returns it.
int cl_threads(int n) {
    warm_drain();
    n_active = n < 1 ? 1 : n > n_threads ? n_threads : n;
    return n_active;
}

// Post a warm of layer's experts ids[0..n) (distinct, the likeliest first; at most MAXWARM) and return at once: the
// pool's other threads read them into cache until done or the next call (a running warm is stopped first).
// Returns 0, or -22.
int cl_warm(int layer, int n, const int32_t* ids) {
    warm_drain();
    if (layer < 0 || layer >= LAYERS || !layers[layer].down.data || !layers[layer].gate.data || n_active < 2) return -22;
    n = std::min(n, MAXWARM);
    for (int j = 0; j < n; j++)
        if (ids[j] < 0 || ids[j] >= NX) return -22;
    if (n < 1) return 0;
    wj.layer = layer;
    wj.n = n;
    std::copy(ids, ids + n, wj.ex);
    phase_id.store(5, std::memory_order_relaxed);
    phase_left.store(n_active - 1, std::memory_order_relaxed);
    phase_gen.fetch_add(1, std::memory_order_acq_rel);
    warm_posted = true;
    if (sleepers.load()) {
        pthread_mutex_lock(&mu);
        pthread_cond_broadcast(&cv);
        pthread_mutex_unlock(&mu);
    }
    return 0;
}

// The threads still in a posted warm (0: done or none).
int cl_warm_left() { return warm_posted ? phase_left.load(std::memory_order_acquire) : 0; }

__attribute__((destructor)) void cl_shutdown() {
    warm_drain();
    stop.store(true);
    pthread_mutex_lock(&mu);
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
    for (auto& w : workers) w.join();
    workers.clear();
    stop.store(false);
}

// Read one expert tensor (which: 0 gate, 1 up, 2 down) of a layer from a GGUF file into huge-page memory.
// Returns 0, or -errno.
int cl_load(int layer, int which, const char* path, uint64_t offset, uint64_t bytes, int type) {
    if (layer < 0 || layer >= LAYERS || which < 0 || which > 2) return -22;
    Layer& L = layers[layer];
    Proj& p = which == 0 ? L.gate : which == 1 ? L.up : L.down;
    if (p.data) return -17;
    const int64_t ne0 = which == 2 ? FF : D;
    const size_t row_bytes = ggml_row_size((ggml_type) type, ne0);
    if (row_bytes * (which == 2 ? D : FF) * NX != bytes) return -34;
    // the activation quantized for this tensor's vec_dot must fit its buffer row (xq for gate|up, hq for down)
    if (ggml_row_size(ggml_get_type_traits_cpu((ggml_type) type)->vec_dot_type, ne0) > (which == 2 ? CL_HQB : CL_XQB))
        return -75;
    uint8_t* dst;
    if (which == 2) {
        void* m = mmap(nullptr, huge(bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return -12;
        madvise(m, huge(bytes), MADV_HUGEPAGE);
        dst = (uint8_t*) m;
    } else {  // gate at gu, up at gu + huge(bytes): gate and up have the same size in every layer
        pthread_mutex_lock(&load_mu);
        if (!L.gu) {
            void* m = mmap(nullptr, 2 * huge(bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED) {
                pthread_mutex_unlock(&load_mu);
                return -12;
            }
            madvise(m, 2 * huge(bytes), MADV_HUGEPAGE);
            L.gu = (uint8_t*) m;
        }
        pthread_mutex_unlock(&load_mu);
        dst = L.gu + (which == 1 ? huge(bytes) : 0);
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return -2;
    size_t done = 0;
    while (done < bytes) {
        const ssize_t r = pread(fd, dst + done, bytes - done > (64u << 20) ? (64u << 20) : bytes - done, offset + done);
        if (r <= 0) {
            close(fd);
            return -5;
        }
        done += r;
    }
    posix_fadvise(fd, offset, bytes, POSIX_FADV_DONTNEED);
    close(fd);
    p.type = type;
    p.tr = ggml_get_type_traits_cpu((ggml_type) type);
    p.row_bytes = row_bytes;
    p.bytes = bytes;
    p.data = dst;
    if (which < 2) {
        pthread_mutex_lock(&load_mu);
        L.iq512 = L.gate.data && L.up.data && L.gate.type == L.up.type &&
                  strata::kernels::cpu::iq512_supported(L.gate.type) && getenv("CL_NO_IQ512") == nullptr;
        pthread_mutex_unlock(&load_mu);
    }
    return 0;
}

// The routing alone (for checks): softmax over NX logits, the top k, weights normalized.
void cl_route(const float* logits, int k, int32_t* ids, float* w) {
    float m = logits[0];
    for (int i = 1; i < NX; i++) m = logits[i] > m ? logits[i] : m;
    float p[NX], sum = 0;
    for (int i = 0; i < NX; i++) sum += (p[i] = std::exp(logits[i] - m));
    // the k largest p in one pass, kept in order with ties to the lower index: what k scans for the first maximum
    // among the unused pick (those took ~4 us of the exchange's routing)
    int top[NX], n = 0;
    for (int i = 0; i < NX; i++) {
        if (n == k && !(p[i] > p[top[k - 1]])) continue;
        int j = n < k ? n++ : k - 1;
        for (; j > 0 && p[i] > p[top[j - 1]]; j--) top[j] = top[j - 1];
        top[j] = i;
    }
    float ws = 0;
    for (int j = 0; j < k; j++) {
        ids[j] = top[j];
        w[j] = p[top[j]] / sum;
        ws += w[j];
    }
    for (int j = 0; j < k; j++) w[j] /= ws;
}

// cl_route for m rows of logits `stride` floats apart (a prefill chunk's): ids and w m x k, the decode's bits.
void cl_route_m(const float* logits, int m, int stride, int k, int32_t* ids, float* w) {
    for (int i = 0; i < m; i++) cl_route(logits + (size_t) i * stride, k, ids + (size_t) i * k, w + (size_t) i * k);
}

// The experts' sum for x (D floats) over the given experts and weights.
// The experts' sum for x (D floats) over the given experts and weights: cl_moe_m's for one token (its down in blocks
// was ~3 us faster than rows over all k experts at once, M = 1 at 96 threads; the same bits)
int cl_moe_m(int layer, int m, const float* x, int k, const int32_t* ids, const float* w, float* out);
int cl_moe(int layer, const float* x, int k, const int32_t* ids, const float* w, float* out) {
    const Layer& L = layers[layer];
    if (!L.gate.data || !L.up.data || !L.down.data || k < 0 || k > MAXK) return -22;
    if (k == 0) {  // every route on the card (host_loop's card exchanges)
        std::fill(out, out + D, 0.f);
        return 0;
    }
    return cl_moe_m(layer, 1, x, k, ids, w, out);
}

// The experts' sums for m <= MAXM tokens (x: m x D floats; ids, w: m x k; out: m x D), each distinct expert of the
// union streamed once. Each token's out is cl_moe's for that token alone, to the bit.
int cl_moe_m(int layer, int m, const float* x, int k, const int32_t* ids, const float* w, float* out) {
    const Layer& L = layers[layer];
    if (!L.gate.data || !L.up.data || !L.down.data || k < 1 || k > MAXK || m < 1 || m > MAXM) return -22;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ns = [](auto a, auto b) { return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
    const auto t0 = now();
#ifdef CL_TRACE
    if (!tr_entered) TR_C(TR_ENTRY);
    tr_entered = false;
    tr_call = tr_n;
#endif
    jm.layer = layer;
    jm.m = m;
    jm.k = k;
    jm.x = x;
    jm.out = out;
    // the union in first-seen order, then each expert's uses in token order
    int slot[NX], last[NX];
    std::fill(slot, slot + NX, -1);
    int nu[MAXP] = {};
    jm.u = 0;
    for (int i = 0; i < m * k; i++) {
        const int e = ids[i];
        if (e < 0) continue;                      // a skipped route (another engine's: the card's share)
        if (e >= NX || (slot[e] >= 0 && last[e] == i / k)) return -22;  // a token's routes are distinct
        last[e] = i / k;
        if (slot[e] < 0) {
            slot[e] = jm.u;
            jm.ex[jm.u++] = e;
        }
        nu[slot[e]]++;
    }
    jm.decode = gu_decode(L);
    jm.first[0] = jm.cost[0] = jm.dcost[0] = 0;
    for (int j = 0; j < jm.u; j++) {
        jm.first[j + 1] = jm.first[j] + nu[j];
        jm.cost[j + 1] = jm.cost[j] + FF * (jm.decode + nu[j]);
        jm.dcost[j + 1] = jm.dcost[j] + DB * (1 + nu[j]);
    }
    jm.p = jm.first[jm.u];
    int fill[MAXP];
    std::copy(jm.first, jm.first + jm.u, fill);
    for (int i = 0; i < m; i++) {
        int n = 0;
        for (int e = 0; e < k; e++) {
            if (ids[i * k + e] < 0) continue;
            const int p = fill[slot[ids[i * k + e]]]++;
            jm.use_of[i][n] = p;
            jm.w[i][n++] = w[i * k + e];
            jm.act[p] = jm.xq[i];
            jm.ff[p] = jm.h[p];
        }
        jm.kn[i] = n;
    }
    if (jm.p == 0) {                              // every route skipped
        std::fill(out, out + (size_t) m * D, 0.f);
        return 0;
    }
    xq_done.store(0, std::memory_order_relaxed);
    const auto t1 = now();
    TR_C(TR_ROUTED);
    TR_C(TR_XQ);  // the rows are quantized in gate|up's phase
    phase(3);
    const auto t2 = now();
    phase(4);
    const auto t3 = now();
    phase_ns[0] += ns(t0, t1);
    phase_ns[1] += ns(t1, t2);
    phase_ns[3] += ns(t2, t3);
#ifdef CL_TRACE
    TR_C(TR_RET);
    if (tr_n < TR_CALLS) tr_n++;
#endif
    return 0;
}

// One exchange: the card's page (hn_f bf16 [D], then the logits fp32 [NX]) -> reply (fp32 [D]). The routing is
// written to ids / w when they are not null.
int cl_exchange(int layer, const uint8_t* page, float* reply, int32_t* ids_out, float* w_out) {
#ifdef CL_TRACE
    TR_C(TR_ENTRY);
    tr_entered = true;
#endif
    alignas(64) float x[D];
    const uint16_t* hb = (const uint16_t*) page;
    for (int i = 0; i < D; i++) x[i] = bf16_to_f32(hb[i]);
    float logits[NX];
    std::memcpy(logits, page + 2 * D, sizeof(logits));
    int32_t ids[10];
    float w[10];
    cl_route(logits, 10, ids, w);
    if (ids_out) std::memcpy(ids_out, ids, sizeof(ids));
    if (w_out) std::memcpy(w_out, w, sizeof(w));
    return cl_moe(layer, x, 10, ids, w, reply);
}

// GLM's router weights for a layer: w fp32 [NX][D] (llama.cpp's ffn_gate_inp, rows = experts), bias fp32 [NX]
// (exp_probs_b)
int cl_router_load(int layer, const float* w, const float* bias) {
    if (layer < 0 || layer >= LAYERS) return -22;
    router_w[layer].assign(w, w + (size_t) NX * D);
    router_b[layer].assign(bias, bias + NX);
    return 0;
}

// GLM's routing of x (fp32 [D]) on the lane's threads: p = sigmoid(W_r x), the top k of p + bias, weights p over
// their sum times scale (expert_weights_norm, expert_weights_scale); logits_out (fp32 [NX]) when not null

int cl_glm_route(int layer, const float* x, int k, float scale, int32_t* ids, float* w, float* logits_out) {
    if (layer < 0 || layer >= LAYERS || router_w[layer].empty() || k < 1 || k > MAXK) return -22;
    alignas(64) float logits[NX];
    jr.w = router_w[layer].data();
    jr.x = x;
    jr.logits = logits;
    phase(6);
    glm_pick(layer, logits, k, scale, ids, w);
    if (logits_out) std::memcpy(logits_out, logits, sizeof(logits));
    return 0;
}

// m rows (x fp32 [m][D]) -> ids / w [m][k]: every row cl_glm_route's, to the bit; rows split over the lane's threads
int cl_glm_route_m(int layer, int m, const float* x, int k, float scale, int32_t* ids, float* w) {
    if (layer < 0 || layer >= LAYERS || router_w[layer].empty() || k < 1 || k > MAXK || m < 1) return -22;
    jrm = {layer, m, k, scale, x, ids, w};
    phase(7);
    return 0;
}

// GLM's exchange: hn (bf16 [D], the card's page) -> its routing (cl_glm_route, top 8, scale 2.5) -> the experts' sum
// (fp32 [D]); the routing to ids / w when they are not null
int cl_glm_exchange(int layer, const uint16_t* hn, float* reply, int32_t* ids_out, float* w_out) {
    alignas(64) float x[D];
    for (int i = 0; i < D; i++) x[i] = bf16_to_f32(hn[i]);
    int32_t ids[MAXK];
    float w[MAXK];
    const int rc = cl_glm_route(layer, x, 8, 2.5f, ids, w, nullptr);
    if (rc) return rc;
    if (ids_out) std::memcpy(ids_out, ids, sizeof(int32_t) * 8);
    if (w_out) std::memcpy(w_out, w, sizeof(float) * 8);
    return cl_moe(layer, x, 8, ids, w, reply);
}

// GLM's exchange for a verify pass's M <= 4 rows: hn in the M-row blocks of mp rows (bf16, mrows.h) -> each
// row's routing (as cl_glm_exchange) -> cl_moe_m over the union of the rows' experts -> the rows' sums as mp-row
// blocks (fp32; rows M..mp-1 copies of row M - 1); the routes to ids / w (M x 8) when they are not null
int cl_glm_exchange_m(int layer, int M, const uint16_t* hn, float* reply, int32_t* ids_out, float* w_out) {
    constexpr int MR = 4;   // mrows.h's layout holds 4 rows
    if (M < 1 || M > MR) return -22;
    const int mp = mp_of(M);
    static thread_local float x[MR][D], out[MR][D];
    int32_t ids[MR * 8];
    float w[MR * 8];
    for (int m = 0; m < M; m++) {
        mrow_bf16(hn, D, m, mp, x[m]);
        const int rc = cl_glm_route(layer, x[m], 8, 2.5f, ids + 8 * m, w + 8 * m, nullptr);
        if (rc) return rc;
    }
    if (ids_out) std::memcpy(ids_out, ids, sizeof(int32_t) * 8 * M);
    if (w_out) std::memcpy(w_out, w, sizeof(float) * 8 * M);
    const int rc = cl_moe_m(layer, M, &x[0][0], 8, ids, w, &out[0][0]);
    if (rc) return rc;
    for (int m = 0; m < mp; m++) {
        for (int i = 0; i < D; i++) reply[mrow_at(i, m, mp)] = out[m < M ? m : M - 1][i];
    }
    return 0;
}

}  // extern "C"
