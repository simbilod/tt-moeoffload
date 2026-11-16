// GLM's mHC parametrization on one data-movement RISC-V. Blackhole's baby RISC-Vs implement part of
// the F extension (fadd.s / fsub.s / fmul.s / fmadd.s, round to nearest even, denormals flushed; no fdiv.s, no
// fsqrt.s: tt-isa-documentation BlackholeA0/TensixTile/BabyRISCV/InstructionSet.md), so the coordinator can do a
// site's whole parametrization in software instead of ~120 dependent tile ops (mhc_split_sinkhorn: 55-57 us):
//   the cores' partials summed: y[j] = sum_c P[c][j] (j < 24), ss = sum_c P[c][32]
//   r = rsqrt(ss / (HC * D) + eps)                        (Newton from the bit trick; no fsqrt)
//   a = r * y * scale + base;  pre = sigmoid(a[0:4]) + eps;  post = 2 sigmoid(a[4:8])
//   comb = row softmax of a[8:24] as 4 x 4 [src][dst], + eps; a column normalization; then (iters - 1) times a row
//          normalization and a column normalization (llama.cpp's build_hc_pre, as G2's hc_ref)
// exp by range reduction (n = round(x log2 e), r = x - n ln 2 in two parts) and a degree-6 polynomial; 1/x by the bit
// trick and 3 Newton steps (fused/fpu.h: the F instructions by inline asm; -DSOFT_FLOAT: libgcc's soft floats).
// REPS times over the same input (timing), the cycle counter around it.
// In: one DRAM page of fp32: P [CORES][33] (32 partial mixes, then the partial sum of squares), then scale [3], base
// [24]. Out: one DRAM page: pre [4], post [4], comb [16] (row-major [src][dst]), then the cycles (uint32) per pass.
// Compile args: CORES, ITERS, REPS, the input's and output's accessor args. Runtime args: in address, out address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/fpu.h"

namespace {

constexpr uint32_t cb_in = 0, cb_out = 1;
constexpr uint32_t HC = 4, D = 4096;
constexpr float EPS_RMS = 1e-5f, EPS_HC = 1e-6f;

#ifdef SOFT_FLOAT  // libgcc's soft floats (the first measurement: 374 us a site)
inline float add(float a, float b) { return a + b; }
inline float sub(float a, float b) { return a - b; }
inline float mul(float a, float b) { return a * b; }
inline bool lt(float a, float b) { return a < b; }
#else
using fused::fpu::add;
using fused::fpu::lt;
using fused::fpu::mul;
using fused::fpu::sub;
#endif
using fused::fpu::exp;
using fused::fpu::recip;
using fused::fpu::rsqrt;
using fused::fpu::sigmoid;

// the cores' partials summed into y[0..23], y[24] = the sum of squares
template <uint32_t CORES>
void reduce(const volatile float* in, float* y) {
    for (uint32_t j = 0; j < 25; j++) y[j] = 0.0f;
    for (uint32_t c = 0; c < CORES; c++) {
        const volatile float* p = in + c * 33;
        for (uint32_t j = 0; j < 24; j++) y[j] = add(y[j], p[j]);
        y[24] = add(y[24], p[32]);
    }
}

template <uint32_t CORES, uint32_t ITERS>
void param(const volatile float* in, const float* y, volatile float* out) {
    const volatile float* scale = in + CORES * 33;
    const volatile float* base = scale + 3;
    const float ss = y[24];
    const float r = rsqrt(add(mul(ss, 1.0f / (HC * D)), EPS_RMS));
    float a[24];
    for (uint32_t j = 0; j < 24; j++) a[j] = add(mul(mul(r, y[j]), scale[j < 4 ? 0 : (j < 8 ? 1 : 2)]), base[j]);
    for (uint32_t j = 0; j < 4; j++) {
        out[j] = add(sigmoid(a[j]), EPS_HC);
        out[4 + j] = mul(2.0f, sigmoid(a[4 + j]));
    }
    float m[16];
    for (uint32_t i = 0; i < HC; i++) {  // row softmax (over dst), + eps
        float mx = a[8 + 4 * i];
        for (uint32_t j = 1; j < HC; j++) mx = lt(mx, a[8 + 4 * i + j]) ? a[8 + 4 * i + j] : mx;
        float s = 0.0f;
        for (uint32_t j = 0; j < HC; j++) {
            m[4 * i + j] = exp(sub(a[8 + 4 * i + j], mx));
            s = add(s, m[4 * i + j]);
        }
        const float inv = recip(s);
        for (uint32_t j = 0; j < HC; j++) m[4 * i + j] = add(mul(m[4 * i + j], inv), EPS_HC);
    }
    auto col = [&]() {
        for (uint32_t j = 0; j < HC; j++) {
            const float inv = recip(add(add(add(add(m[j], m[4 + j]), m[8 + j]), m[12 + j]), EPS_HC));
            for (uint32_t i = 0; i < HC; i++) m[4 * i + j] = mul(m[4 * i + j], inv);
        }
    };
    auto row = [&]() {
        for (uint32_t i = 0; i < HC; i++) {
            const float inv = recip(add(add(add(add(m[4 * i], m[4 * i + 1]), m[4 * i + 2]), m[4 * i + 3]), EPS_HC));
            for (uint32_t j = 0; j < HC; j++) m[4 * i + j] = mul(m[4 * i + j], inv);
        }
    };
    col();
    for (uint32_t it = 0; it + 1 < ITERS; it++) {
        row();
        col();
    }
    for (uint32_t k = 0; k < 16; k++) out[8 + k] = m[k];
}

inline uint32_t cycles() {
    uint32_t c;
    asm volatile("csrr %0, cycle" : "=r"(c));
    return c;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t CORES = get_compile_time_arg_val(0), ITERS = get_compile_time_arg_val(1);
    constexpr uint32_t REPS = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in_t = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const auto out_t = TensorAccessor(out_args, get_arg_val<uint32_t>(1));
    const uint32_t in_l1 = get_write_ptr(cb_in), out_l1 = get_write_ptr(cb_out);
    noc_async_read(in_t.get_noc_addr(0), in_l1, (CORES * 33 + 27) * 4);
    noc_async_read_barrier();
    const volatile float* in = reinterpret_cast<const volatile float*>(in_l1);
    volatile float* out = reinterpret_cast<volatile float*>(out_l1);
    float y[25];
    reduce<CORES>(in, y);  // once, untimed (instruction cache)
    param<CORES, ITERS>(in, y, out);
    const uint32_t t0 = cycles();
    for (uint32_t i = 0; i < REPS; i++) reduce<CORES>(in, y);
    const uint32_t t1 = cycles();
    for (uint32_t i = 0; i < REPS; i++) param<CORES, ITERS>(in, y, out);
    const uint32_t t2 = cycles();
    reinterpret_cast<volatile uint32_t*>(out_l1)[24] = (t2 - t0) / REPS;
    reinterpret_cast<volatile uint32_t*>(out_l1)[25] = (t1 - t0) / REPS;   // the reduction alone
    reinterpret_cast<volatile uint32_t*>(out_l1)[26] = (t2 - t1) / REPS;   // the parametrization alone
    noc_async_write(out_l1, out_t.get_noc_addr(0), 32 * 4);
    noc_async_write_barrier();
}
