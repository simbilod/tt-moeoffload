// The M-row layout on the host (a verify pass's rows; the card's fused/hc_dm.h blocks, tt/fused/mrows.py): a vector of
// n values is n / 32 tile blocks, each its face-0 pieces (16 values) of rows 0..mp-1, then its face-1 pieces. Shared
// by Flash-Next's host loop (host_loop.cpp) and GLM's (glm_loop.cpp, experts.cpp's cl_glm_exchange_m).
#pragma once

#include <cstdint>
#include <cstring>

// the layout's rows for M rows (1, 2, 4: M = 3 runs as 4 with row 3 a copy of row 2)
inline int mp_of(int m) { return m == 1 ? 1 : m == 2 ? 2 : 4; }

// element i of row m in the blocks
inline int mrow_at(int i, int m, int mp) { return ((i / 32) * 2 + (i % 32) / 16) * mp * 16 + m * 16 + i % 16; }

// row m of n bf16 values in the blocks, to fp32
inline void mrow_bf16(const uint16_t* blocks, int n, int m, int mp, float* x) {
    for (int i = 0; i < n; i++) {
        const uint32_t u = (uint32_t) blocks[mrow_at(i, m, mp)] << 16;
        std::memcpy(&x[i], &u, 4);
    }
}
