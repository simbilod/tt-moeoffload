// Hardware single-precision floats on Blackhole's baby RISC-Vs, which tt-metal's kernel flags leave unused.
//
// The cores implement part of the F extension (fadd.s / fsub.s / fmul.s / flt.s / fcvt, round to nearest even,
// denormals flushed to zero; no fdiv.s / fsqrt.s: tt-isa-documentation BlackholeA0/TensixTile/BabyRISCV), but
// tt-metal compiles kernels for rv32im (-mcpu=tt-bh-tensix, the ilp32 soft-float ABI), so C++ float arithmetic
// becomes calls to libgcc's __mulsf3 / __addsf3 (~50-100 cycles each; glm_hc_param.cpp measured 374 us for a mHC
// parametrization that way). GCC 15 here refuses `#pragma GCC target("arch=+f")`, so each operation is an inline
// asm block under `.option arch, +f`: operands moved into ft0 / ft1, the instruction, the result moved back. Under the
// soft-float ABI a float already lives in an integer register, so these take and return plain floats. No fmadd: on
// this core it is neither fused nor separate (tt-isa-documentation Miscellaneous/FMA), and llama.cpp's references use
// a separate multiply and add.
// Division, sqrt, exp, sigmoid are built on top (Newton steps from the usual bit tricks, a polynomial exp).
#pragma once

#include <cstdint>

namespace fused::fpu {

#define FPU_OP2(name, insn)                                                                                    \
    inline float name(float a, float b) {                                                                      \
        float r;                                                                                               \
        asm(".option push\n.option arch, +f\nfmv.w.x ft0, %1\nfmv.w.x ft1, %2\n" insn                          \
            " ft0, ft0, ft1\nfmv.x.w %0, ft0\n.option pop"                                                     \
            : "=r"(r) : "r"(a), "r"(b) : "ft0", "ft1");                                                        \
        return r;                                                                                              \
    }
FPU_OP2(add, "fadd.s")
FPU_OP2(sub, "fsub.s")
FPU_OP2(mul, "fmul.s")
#undef FPU_OP2

inline bool lt(float a, float b) {  // a < b
    uint32_t r;
    asm(".option push\n.option arch, +f\nfmv.w.x ft0, %1\nfmv.w.x ft1, %2\nflt.s %0, ft0, ft1\n.option pop"
        : "=r"(r) : "r"(a), "r"(b) : "ft0", "ft1");
    return r != 0;
}

inline int32_t to_int(float a) {  // round to nearest even (the core ignores rounding-mode bits)
    int32_t r;
    asm(".option push\n.option arch, +f\nfmv.w.x ft0, %1\nfcvt.w.s %0, ft0\n.option pop" : "=r"(r) : "r"(a) : "ft0");
    return r;
}

inline float from_int(int32_t a) {
    float r;
    asm(".option push\n.option arch, +f\nfcvt.s.w ft0, %1\nfmv.x.w %0, ft0\n.option pop" : "=r"(r) : "r"(a) : "ft0");
    return r;
}

inline float bits_f(uint32_t u) {
    union { uint32_t u; float f; } v{u};
    return v.f;
}
inline uint32_t f_bits(float f) {
    union { float f; uint32_t u; } v{f};
    return v.u;
}

// 1/d for d > 0 (normal): the bit trick, then 3 Newton steps (about 1 ulp)
inline float recip(float d) {
    float y = bits_f(0x7EF311C7u - f_bits(d));
    for (int i = 0; i < 3; i++) y = mul(y, sub(2.0f, mul(d, y)));
    return y;
}

// 1/sqrt(x) for x > 0: the bit trick, then 3 Newton steps
inline float rsqrt(float x) {
    float y = bits_f(0x5F375A86u - (f_bits(x) >> 1));
    const float hx = mul(0.5f, x);
    for (int i = 0; i < 3; i++) y = mul(y, sub(1.5f, mul(hx, mul(y, y))));
    return y;
}

// e^x: n = round(x log2 e), r = x - n ln 2 (ln 2 in two parts), e^r by Taylor to degree 6, times 2^n (exponent bits)
inline float exp(float x) {
    if (lt(x, -87.0f)) return 0.0f;
    if (lt(88.0f, x)) x = 88.0f;
    const int32_t n = to_int(mul(x, 1.44269504088896341f));
    const float fn = from_int(n);
    const float r = sub(sub(x, mul(fn, 0.693145751953125f)), mul(fn, 1.428606765330187045e-06f));
    float p = 1.0f / 720.0f;
    p = add(mul(p, r), 1.0f / 120.0f);
    p = add(mul(p, r), 1.0f / 24.0f);
    p = add(mul(p, r), 1.0f / 6.0f);
    p = add(mul(p, r), 0.5f);
    p = add(mul(p, r), 1.0f);
    p = add(mul(p, r), 1.0f);
    return bits_f(f_bits(p) + (static_cast<uint32_t>(n) << 23));
}

inline float sigmoid(float x) { return recip(add(1.0f, exp(sub(0.0f, x)))); }

}  // namespace fused::fpu
