// GLM's fused MLA query side, writer (the phases in glm_fmq_reader.cpp):
//   A  per tile n (cb 16, fp32): row 0 to A's check tensor; q_a's tiles (n < 48) as bf16 to R1 + 64 n with their
//      sum of squares (this RISC-V's FPU) at R1 + 3072 + 16 n; the others as fp32 to R1 + 3840 + 128 (n - 48);
//      barrier 1
//   B  per tile n (cb 17, bf16): row 0 to R2 + 64 n and to B's check tensor; barrier 2
//   C  the absorb's 16 rows (cb 17, bf16) into row h % 32 of group h / 32's tiles (two 32-byte pieces a tile, each
//      staged at the alignment of its destination: a NoC transfer whose two ends differ in alignment corrupts it,
//      glm_fmo_reader.cpp); the scores (cb 16, fp32 row 0) to SC + 128 t; barrier 3
// Runtime args: grid.h's block, A's first tile and count, B's, head, the first pool tile and count, the A and B check
// tensors, the two query groups (bf16 tiles [1, 1, 32, 512]), the arena offsets of R1, R2, SC. Compile args: a DRAM
// accessor.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#include "fused/fpu.h"
#ifndef M_MAX
#define M_MAX 1
#endif
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif

constexpr uint32_t cb_out = 16, cb_ob = 17, cb_arena = 30, cb_stage = 31, NQA = 48, IDLE = 0xffffffff;
constexpr uint32_t SQ_OFF = NQA * 64, A2_OFF = SQ_OFF + NQA * 16;

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
inline volatile tt_l1_ptr float* f32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); }

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    const fused::Grid grid = fused::Grid::from_args(0);
    uint32_t i = fused::Grid::args(grid.nrects);
    const uint32_t a0 = get_arg_val<uint32_t>(i++), na = get_arg_val<uint32_t>(i++);
    const uint32_t b0 = get_arg_val<uint32_t>(i++), nb = get_arg_val<uint32_t>(i++);
    const uint32_t head = get_arg_val<uint32_t>(i++);
    const uint32_t p0 = get_arg_val<uint32_t>(i++), np = get_arg_val<uint32_t>(i++);
    const auto check_a = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 73 * 128);
    const auto check_b = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 640 * 64);
    const auto qg0 = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 2048);
    const auto qg1 = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 2048);
    const uint32_t R1 = get_arg_val<uint32_t>(i++), R2 = get_arg_val<uint32_t>(i++), SC = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
#if M_MAX > 1
    // M rows (glm_fmq_reader.cpp): runtime args after SC: the step tensor, the row page's bytes and its A2 offset
    const uint32_t step_addr = get_arg_val<uint32_t>(i++), PG = get_arg_val<uint32_t>(i++);
    const uint32_t PG_A2 = get_arg_val<uint32_t>(i++), PG_IQ = PG_A2 + (73 - NQA) * 128;
    const uint32_t IQT = get_arg_val<uint32_t>(i++);   // the pool cores' iq tiles (glm_fmq_reader.cpp), every core
    noc_async_read(TensorAccessor(d_args, step_addr, 64).get_noc_addr(0), stage, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t M = w32(stage)[1], mp = w32(stage)[4];
    constexpr uint32_t SQ_M = NQA * M_MAX * 64, A2_M = SQ_M + NQA * M_MAX * 16;
    const uint32_t row = stage + 3072;   // a row's two face pieces made contiguous
    // ---- A: blocks to R1 (q_a bf16 with its squares per row; the rest fp32), the A2 rows to the row pages
    for (uint32_t n = a0; n < a0 + na; n++) {
        fused::stage_rows_m(cb_out, 1, 64, stage, mp);
        noc_async_write(stage, check_a.get_noc_addr(0, n * mp * 128), mp * 128);
        invalidate_l1_cache();
        if (n < NQA) {
            volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + 1024);
            for (uint32_t m = 0; m < mp; m++) {
                float ss = 0.0f;
                for (uint32_t c = 0; c < 32; c++) {
                    const uint32_t a = stage + (c < 16 ? 0 : mp * 64) + m * 64 + (c & 15) * 4;
                    const float x = *reinterpret_cast<volatile tt_l1_ptr float*>(a);
                    ss = fused::fpu::add(ss, fused::fpu::mul(x, x));
                    const uint32_t b = w32(a)[0];
                    o[(c < 16 ? 0 : mp * 16) + m * 16 + (c & 15)] = static_cast<uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
                }
                f32(stage + 2048 + m * 16)[0] = ss;
            }
            grid.send(stage + 1024, arena + R1 + n * mp * 64, mp * 64);
            grid.send(stage + 2048, arena + R1 + SQ_M + n * M_MAX * 16, M_MAX * 16);
        } else {
            grid.send(stage, arena + R1 + A2_M + (n - NQA) * mp * 128, mp * 128);
            for (uint32_t m = 0; m < M; m++) {
                fused::l1_copy(stage + m * 64, row + m * 128, 64);
                fused::l1_copy(stage + mp * 64 + m * 64, row + m * 128 + 64, 64);
            }
            noc_async_read_barrier();
            for (uint32_t m = 0; m < M; m++) grid.send(row + m * 128, arena + SC + m * PG + PG_A2 + (n - NQA) * 128, 128);
        }
        noc_async_write_barrier();
    }
    grid.arrive(1);

    // ---- B: blocks to R2; iq's rows (tiles 512..) to the row pages
    for (uint32_t n = b0; n < b0 + nb; n++) {
        fused::stage_rows_m(cb_ob, 1, 32, stage, mp);
        grid.send(stage, arena + R2 + n * mp * 64, mp * 64);
        noc_async_write(stage, check_b.get_noc_addr(0, n * mp * 64), mp * 64);
        if (n >= 512) {
            for (uint32_t m = 0; m < M; m++) {
                fused::l1_copy(stage + m * 32, row + m * 64, 32);
                fused::l1_copy(stage + mp * 32 + m * 32, row + m * 64 + 32, 32);
            }
            noc_async_read_barrier();
            for (uint32_t m = 0; m < M; m++) grid.send(row + m * 64, arena + SC + m * PG + PG_IQ + (n - 512) * 64, 64);
            // row m's two face pieces of iq tile n = 512 + 4 h + d to row h of every core's IQT tile 4 m + d
            const uint32_t h = (n - 512) / 4, d = (n - 512) % 4;
            invalidate_l1_cache();
            for (uint32_t m = 0; m < M; m++) {
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t dst = (4 * m + d) * 2048 + (((h >> 4) * 2 + f) * 256 + (h & 15) * 16) * 2;
                    const uint32_t src = row + 1024 + (m * 2 + f) * 128 + (dst & 63);   // the destination's alignment
                    for (uint32_t e = 0; e < 8; e++) w32(src)[e] = w32(stage + f * mp * 32 + m * 32)[e];
                    grid.send(src, arena + IQT + dst, 32);   // the coordinator's: barrier 2 broadcasts it
                }
            }
        }
        noc_async_write_barrier();
    }
    grid.arrive(2);

    // ---- C: the absorb's rows m < M into row h % 32 of the groups' tile (m, ct); the scores per row
    if (head != IDLE) {
        const auto& qg = head < 32 ? qg0 : qg1;
        const uint32_t r = head % 32;
        for (uint32_t ct = 0; ct < 16; ct++) {
            fused::stage_rows_m(cb_ob, 1, 32, stage, mp);
            invalidate_l1_cache();
            for (uint32_t m = 0; m < M; m++) {
                for (uint32_t half = 0; half < 2; half++) {
                    const uint32_t off = (((r >> 4) * 2 + half) * 256 + (r & 15) * 16) * 2;
                    const uint32_t src = row + (m * 2 + half) * 128 + (off & 63);   // the destination's alignment
                    for (uint32_t e = 0; e < 8; e++) w32(src)[e] = w32(stage + half * mp * 32 + m * 32)[e];
                    noc_async_write(src, qg.get_noc_addr(m * 16 + ct, off), 32);
                }
            }
            noc_async_write_barrier();
        }
    }
    for (uint32_t t = p0; t < p0 + np; t++) {
        for (uint32_t m = 0; m < M_MAX; m++) {
            fused::stage_rows(cb_out, 1, 64, stage);
            if (m < M) grid.send(stage, arena + SC + m * PG + t * 128, 128);
            noc_async_write_barrier();
        }
    }
    grid.arrive(3);
    return;
#endif

    // ---- A
    for (uint32_t n = a0; n < a0 + na; n++) {
        fused::stage_rows(cb_out, 1, 64, stage);
        noc_async_write(stage, check_a.get_noc_addr(0, n * 128), 128);
        if (n < NQA) {
            invalidate_l1_cache();                        // the NoC wrote the stage: no stale lines
            float ss = 0.0f;
            for (uint32_t c = 0; c < 32; c++) ss = fused::fpu::add(ss, fused::fpu::mul(f32(stage)[c], f32(stage)[c]));
            volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + 128);
            for (uint32_t c = 0; c < 32; c++) {
                const uint32_t b = w32(stage)[c];
                o[c] = static_cast<uint16_t>((b + 0x7fff + ((b >> 16) & 1)) >> 16);
            }
            f32(stage + 192)[0] = ss;
            grid.send(stage + 128, arena + R1 + n * 64, 64);
            grid.send(stage + 192, arena + R1 + SQ_OFF + n * 16, 16);
        } else {
            grid.send(stage, arena + R1 + A2_OFF + (n - NQA) * 128, 128);
        }
        noc_async_write_barrier();
    }
    grid.arrive(1);

    // ---- B
    for (uint32_t n = b0; n < b0 + nb; n++) {
        fused::stage_rows(cb_ob, 1, 32, stage);
        grid.send(stage, arena + R2 + n * 64, 64);
        noc_async_write(stage, check_b.get_noc_addr(0, n * 64), 64);
        noc_async_write_barrier();
    }
    grid.arrive(2);

    // ---- C
    if (head != IDLE) {
        const auto& qg = head < 32 ? qg0 : qg1;
        const uint32_t r = head % 32;
        for (uint32_t ct = 0; ct < 16; ct++) {
            fused::stage_rows(cb_ob, 1, 32, stage);
            invalidate_l1_cache();
            for (uint32_t half = 0; half < 2; half++) {
                const uint32_t off = (((r >> 4) * 2 + half) * 256 + (r & 15) * 16) * 2;
                const uint32_t src = stage + 128 + half * 128 + (off & 63);   // the destination's alignment
                for (uint32_t e = 0; e < 8; e++) w32(src)[e] = w32(stage + half * 32)[e];
                noc_async_write(src, qg.get_noc_addr(ct, off), 32);
            }
            noc_async_write_barrier();
        }
    }
    for (uint32_t t = p0; t < p0 + np; t++) {
        fused::stage_rows(cb_out, 1, 64, stage);
        grid.send(stage, arena + SC + t * 128, 128);
        noc_async_write_barrier();
    }
    grid.arrive(3);
}
