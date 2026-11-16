// The sum of a K-split output projection's partials on this RISC-V's FPU (GLM's fmo, fsh and fkda readers; the
// catalog's P4). Output tiles n = s0 .. s0 + ns - 1 of a row-0 vector of nt tiles: part s of tile n is the fp32 row 0
// (32 values) at P + (s * nt + n) * 128, s < S. Each sum starts from base + n * 128 (base 0: from part 0) and adds the
// parts in order, so the result is the same to the bit as the loops it replaces; staged at st, then written to the
// row-major fp32 tensor out from tile s0.
#pragma once

#include <cstdint>

#include "fused/dm.h"
#include "fused/fpu.h"

namespace fused {

template <typename Accessor>
inline void ksplit_sum(const Accessor& out, uint32_t P, uint32_t S, uint32_t nt, uint32_t s0, uint32_t ns,
                       uint32_t base, uint32_t st) {
    if (!ns) return;
    invalidate_l1_cache();
    auto f = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); };
    for (uint32_t n = s0; n < s0 + ns; n++) {
        for (uint32_t c = 0; c < 32; c++) {
            uint32_t s = 0;
            float y;
            if (base) {
                y = f(base + n * 128)[c];
            } else {
                y = f(P + n * 128)[c];
                s = 1;
            }
            for (; s < S; s++) y = fpu::add(y, f(P + (s * nt + n) * 128)[c]);
            f(st + (n - s0) * 128)[c] = y;
        }
    }
    noc_async_write(st, out.get_noc_addr(0, s0 * 128), ns * 128);
    noc_async_write_barrier();
}

// the same at M rows (fused/hc_dm.h's blocks, blk = 128 mp bytes a tile) with the partials by owner: part s of this
// core's i-th output tile (tile s0 + i) at P + (s * ns + i) * blk; the sums to blocks s0 .. of out
// base (0: none): each sum starts from the block of its tile at base + (s0 + i) * blk (GLM's FFN: the lane's routed sum)
template <typename Accessor>
inline void ksplit_sum_m(const Accessor& out, uint32_t P, uint32_t S, uint32_t s0, uint32_t ns, uint32_t blk,
                         uint32_t st, uint32_t base = 0) {
    if (!ns) return;
    invalidate_l1_cache();
    auto f = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); };
    for (uint32_t i = 0; i < ns; i++) {
        for (uint32_t c = 0; c < blk / 4; c++) {
            float y = base ? fpu::add(f(base + (s0 + i) * blk)[c], f(P + i * blk)[c]) : f(P + i * blk)[c];
            for (uint32_t s = 1; s < S; s++) y = fpu::add(y, f(P + (s * ns + i) * blk)[c]);
            f(st + i * blk)[c] = y;
        }
    }
    noc_async_write(st, out.get_noc_addr(0, s0 * blk), ns * blk);
    noc_async_write_barrier();
}

}  // namespace fused
