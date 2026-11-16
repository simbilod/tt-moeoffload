// GLM's KDA decode step, reader (the math in glm_kda_compute.cpp). One value head per core. In:
//   S       the state, fp32 tiles, head h's tile (j, c) at page 16 h + 4 j + c; read once (cb 8 exact), then copied in L1 (cb 9 operand)
//   rows    row-major fp32, a row of 128 a page: q at h, k at 64 + h, v at 128 + h, sigmoid(g2) at 192 + h, the
//           decays d at 256 + h
//   par     row-major fp32 rows of 128: row 0 the output norm's weight, row 1 beta [64]
// q and k go into zero tiles (row 0 used: k becomes a column by transpose); d's row 0 tiles into cb 25 (zeroed
// first: d * k must not meet garbage); Dbc_j (cb 26): d[32 j + r] across every column of row r, the RISC-V copying
// float bits (no float math), so the update's per-row decay is exact.
// Runtime args: S address, rows address, par address, head (0xffffffff: idle). Compile args: the accessor args of S,
// rows, par.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"

constexpr uint32_t cb_ones = 2, cb_sx = 8, cb_sop = 9, cb_nw = 10, cb_q = 11, cb_k = 12, cb_ex = 13, cb_kd = 25;
constexpr uint32_t cb_dbc = 26, cb_zero = 29, cb_stage = 31;
constexpr uint32_t TILE = 4096, ROW = 512, IDLE = 0xffffffff, NH = 64;

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }

void kernel_main() {
    constexpr auto S_args = TensorAccessorArgs<0>();
    constexpr auto r_args = TensorAccessorArgs<S_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto S = TensorAccessor(S_args, get_arg_val<uint32_t>(0), TILE);
    const auto rows = TensorAccessor(r_args, get_arg_val<uint32_t>(1), ROW);
    const auto par = TensorAccessor(p_args, get_arg_val<uint32_t>(2), ROW);
    const uint32_t h = get_arg_val<uint32_t>(3);
    if (h == IDLE) return;

    const uint32_t zero = get_write_ptr(cb_zero);
    fused::fill_words(zero, TILE / 4, 0);
    cb_reserve_back(cb_ones, 1);
    fused::fill_words(get_write_ptr(cb_ones), 2048 / 4, 0x3F803F80u);   // bf16 1.0 everywhere
    cb_push_back(cb_ones, 1);

    // the state once from card memory (cb 8, exact), its operand copy (cb 9) by a local L1 copy: one 4 MB read a
    // layer instead of fgdn's two
    cb_reserve_back(cb_sx, 16);
    cb_reserve_back(cb_sop, 16);
    const uint32_t sx = get_write_ptr(cb_sx), sop = get_write_ptr(cb_sop);
    for (uint32_t t = 0; t < 16; t++) noc_async_read(S.get_noc_addr(h * 16 + t), sx + t * TILE, TILE);
    noc_async_read_barrier();
    fused::l1_copy(sx, sop, 16 * TILE);
    noc_async_read_barrier();
    cb_push_back(cb_sx, 16);
    cb_push_back(cb_sop, 16);
    cb_reserve_back(cb_nw, 4);
    for (uint32_t c = 0; c < 4; c++) fused::row0_from_dram(par, 0, c * 128, get_write_ptr(cb_nw) + c * TILE);
    noc_async_read_barrier();
    cb_push_back(cb_nw, 4);

    for (uint32_t cb : {cb_q, cb_k, cb_kd}) {
        cb_reserve_back(cb, 4);
        for (uint32_t t = 0; t < 4; t++) fused::l1_copy(zero, get_write_ptr(cb) + t * TILE, TILE);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < 4; t++) {
        fused::row0_from_dram(rows, h, t * 128, get_write_ptr(cb_q) + t * TILE);
        fused::row0_from_dram(rows, NH + h, t * 128, get_write_ptr(cb_k) + t * TILE);
        fused::row0_from_dram(rows, 4 * NH + h, t * 128, get_write_ptr(cb_kd) + t * TILE);
    }
    cb_reserve_back(cb_ex, 10);
    const uint32_t e = get_write_ptr(cb_ex);
    for (uint32_t t = 0; t < 4; t++) {
        fused::row0_from_dram(rows, 2 * NH + h, t * 128, e + t * TILE);        // v
        fused::row0_from_dram(rows, 3 * NH + h, t * 128, e + (4 + t) * TILE);  // sigmoid(g2)
    }
    const uint32_t stage = get_write_ptr(cb_stage);
    noc_async_read(par.get_noc_addr(1), stage, ROW);                      // beta
    noc_async_read(rows.get_noc_addr(4 * NH + h), stage + ROW, ROW);      // d, for Dbc
    noc_async_read_barrier();
    cb_push_back(cb_q, 4);
    cb_push_back(cb_k, 4);
    cb_push_back(cb_kd, 4);
    fused::fill_row0(e + 8 * TILE, stage + h * 4);
    cb_push_back(cb_ex, 10);

    // Dbc_j (r, c) = d[32 j + r]: face (r / 16) * 2 + c / 16, element (r % 16) * 16 + c % 16
    cb_reserve_back(cb_dbc, 4);
    const uint32_t dbc = get_write_ptr(cb_dbc);
    for (uint32_t j = 0; j < 4; j++) {
        for (uint32_t r = 0; r < 32; r++) {
            const uint32_t bits = w32(stage + ROW)[32 * j + r];
            const uint32_t base = dbc + j * TILE + ((r >> 4) * 2 * 256 + (r & 15) * 16) * 4;
            for (uint32_t c = 0; c < 16; c++) {
                w32(base)[c] = bits;                // face (r / 16) * 2
                w32(base + 256 * 4)[c] = bits;      // face (r / 16) * 2 + 1
            }
        }
    }
    asm volatile("fence rw, rw" ::: "memory");
    cb_push_back(cb_dbc, 4);
}
