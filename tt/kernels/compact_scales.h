// The compact expert scales (g7_experts.compact_pages, repack.scale_factors / dm_bits): scale = d m, d the GGUF
// block's fp16 scale, m a small integer (int8) per 32 weights. Conversions for the data-movement cores (no FPU):
// fp16 -> fp32 bits exactly (f16_bits), and d m's fp32 bits exactly in integers (dm_bits: d = M 2^E, M <= 2047,
// P = M |m| < 2^18, normalized; m's leading bit from a table). pf_gexp_writer.cpp turns the compact pages into the
// m and d rows the compute multiplies in on the SFPU (step 3); dm_bits is what the host's repack.dm_bits checks.
#pragma once
#include <cstdint>

struct Dm {
    uint32_t sign, M, ebase, lead;            // d's sign bit, M, E + 127, M's leading bit
};

inline Dm split_d(uint32_t h) {
    const uint32_t e = (h >> 10) & 31, f = h & 1023;
    Dm d{(h >> 15) << 31, e ? (1024 | f) : f, uint32_t(e ? int32_t(e) - 25 + 127 : -24 + 127), 10};
    if (e == 0) {                              // subnormal d: M's leading bit below 10
        d.lead = 0;
        for (uint32_t t = d.M; t > 1; t >>= 1) {
            d.lead++;
        }
    }
    return d;
}

inline uint8_t lead_of[129];                 // the leading bit of 0 .. 128 (lead_table() fills it)

inline uint32_t dm_bits(const Dm& d, int32_t m) {
    const uint32_t s = uint32_t(m >> 31);    // 0, or all ones for m < 0
    const uint32_t am = (uint32_t(m) ^ s) - s, sign = d.sign ^ (s & 0x80000000u);
    const uint32_t P = d.M * am;
    if (P == 0) {
        return sign;
    }
    uint32_t b = d.lead + lead_of[am];       // P's leading bit: this or one more
    b += P >> (b + 1);
    return sign | ((d.ebase + b) << 23) | ((P << (23 - b)) & 0x7FFFFF);
}

inline uint32_t f16_bits(uint32_t h) {       // fp16 -> fp32 bits, exactly
    const uint32_t e = h & 0x7C00;
    if (e == 0) {                            // zero or subnormal (no Inf / NaN: the builder checks)
        return dm_bits(split_d(h), 1);
    }
    return ((h & 0x8000) << 16) | (((h & 0x7FFF) + 0x1C000) << 13);
}

inline void lead_table() {
    for (uint32_t v = 0; v <= 128; v++) {
        uint32_t b = 0;
        for (uint32_t t = v; t > 1; t >>= 1) {
            b++;
        }
        lead_of[v] = b;
    }
}
