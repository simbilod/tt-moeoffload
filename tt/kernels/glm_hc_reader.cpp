// GLM's mHC site on the card, reader (host side in tt/glm/g3_hc.py). The residual streams live as fp32
// "stream tiles": column tile t (32 of D = 4096) holds stream s in row s (rows 0-3). Core i of the first NC cores owns
// column tiles 2i and 2i + 1. Phases:
//   A  the 24 mixes' partial over this core's slices: fn's k-tiles s * 128 + t (bf16, page k = rows 32k..32k+31 of
//      fn^T [16384, 32]) against bf16 x tiles whose row 0 is stream row s of tile t (built here, rounded to nearest even);
//      the partial sum of squares of the same values (this core's FPU: fused/fpu.h) to the writer through cb_sq
//   1  the barrier: once all cores arrived, the coordinator sends the gathered partials (mixes [NC][32] fp32, then
//      sums of squares, 16 bytes a core) to the host over the D2H socket, takes the 24 parameters back over H2D (pre [4], post
//      [4], C [dst][src] [16], fp32) and multicasts them with the barrier
//   B  PRE (row 0, cols 0-3 = pre) @ X_t: row 0 = h's 32 values of tile t
//   C  C' (rows 0-3: cols 0-3 = C[dst][:], col 4 = post[dst]) @ X'_t, X'_t = X_t with row 4 = the sublayer's output
//      (`out`, read from DRAM here): rows 0-3 = the new streams
// The page to the host: mixes [NC][32] fp32, then one 16-byte slot a core whose first float is its sum of squares.
// Runtime args: the grid block, X address (stream tiles, fp32, page t), fn address, out address (row-major fp32 row
// of D), first column tile, count (0 off the first NC cores), phases (bit 0: A, the barrier and B, the pre-mix; bit 1:
// C, the post-mix, a separate launch once the sublayer's output exists), par address (the 24 parameters, fp32, one
// 128-byte page: the pre launch's coordinator writes them, a post-only launch reads them). Compile args: NC, then X's,
// fn's, out's and par's accessor args.
// Defines: H2D_CONFIG, D2H_CONFIG (the sockets on the coordinator).
//
// M rows (-DM_MAX = 2..4; M_MAX = 1 compiles the one-row source): row m's streams in rows 4 m + s of the
// stream tiles, the sublayer's output rows (fused/hc_dm.h's mp-row blocks, fp32) put in rows OUT_ROW + m; A's x tiles
// carry stream s of row m in row m (each row's mixes), the sums of squares per row (M_MAX 16-byte slots); the host's
// page is M_MAX row pages of ROW_PAGE bytes (each the one-row page: mixes [NC][32], then NC square slots), its reply
// M_MAX rows of 128 bytes (cl_hc_params per row: glm_loop.cpp); PRE row m = pre_m at cols 4 m + s, C' rows 4 m + d =
// C_m[d][s] at cols 4 m + s and post_m[d] at col OUT_ROW + m. The step tensor (mp) is a runtime arg after par.
#include <cstdint>
#ifndef M_MAX
#define M_MAX 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/fpu.h"
#include "fused/grid.h"
#include "fused/link.h"
#include "fused/dm.h"

constexpr uint32_t cb_w = 0, cb_x = 1, cb_orow = 3, cb_X = 4, cb_coef = 5, cb_sq = 6, cb_arena = 8;
constexpr uint32_t cb_ones_m = 10;   // M rows: the ones tile of the compute's row sums
constexpr uint32_t HC = 4, KT_STREAM = 128, TILE_F32 = 4096, FACE = 1024;

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
inline volatile tt_l1_ptr float* f32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); }
// fp32 tile element (r, c) (faces of 16 x 16, row-major inside a face)
inline uint32_t at(uint32_t r, uint32_t c) { return (((r >> 4) << 1 | (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)) * 4; }
// bf16 tile element (r, c)
inline uint32_t at16(uint32_t r, uint32_t c) { return (((r >> 4) << 1 | (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)) * 2; }
inline uint16_t bf16_rne(uint32_t b) { return static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16); }
inline void zero(uint32_t a, uint32_t bytes) {
    for (uint32_t i = 0; i < bytes / 4; i++) w32(a)[i] = 0;
}

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr auto X_args = TensorAccessorArgs<1>();
    constexpr auto fn_args = TensorAccessorArgs<X_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto par_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    // the arena's regions; a square takes a 16-byte slot (NoC writes into L1 land only at 16-byte-aligned addresses)
#if M_MAX > 1
    constexpr uint32_t ROW_PAGE = NC * 144, OUT_ROW = 4 * M_MAX;
    constexpr uint32_t MIX = 0, SQ = NC * 128, PARAM = M_MAX * ROW_PAGE;
#else
    constexpr uint32_t MIX = 0, SQ = NC * 128, PARAM = SQ + NC * 16;
#endif
    const fused::Grid g = fused::Grid::from_args(0);
    uint32_t a = fused::Grid::args(g.nrects);
    const auto Xd = TensorAccessor(X_args, get_arg_val<uint32_t>(a), TILE_F32);
    const auto fn = TensorAccessor(fn_args, get_arg_val<uint32_t>(a + 1), 2048);
    const auto outd = TensorAccessor(out_args, get_arg_val<uint32_t>(a + 2), M_MAX * 4 * 4096);
    const uint32_t t0 = get_arg_val<uint32_t>(a + 3), nt = get_arg_val<uint32_t>(a + 4);
    const uint32_t phases = get_arg_val<uint32_t>(a + 5);
    const auto par = TensorAccessor(par_args, get_arg_val<uint32_t>(a + 6), 128);
    const uint32_t arena = get_write_ptr(cb_arena);
#if M_MAX > 1
    const auto step = TensorAccessor(par_args, get_arg_val<uint32_t>(a + 7), 64);
    noc_async_read(step.get_noc_addr(0), arena + PARAM + M_MAX * 128, 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t M = w32(arena + PARAM + M_MAX * 128)[1], mp = w32(arena + PARAM + M_MAX * 128)[4];
#endif

    // this core's stream tiles, resident in cb_X; row 4 of each = the sublayer's output there (phase C's X')
    cb_reserve_back(cb_X, 2);
    const uint32_t X = get_write_ptr(cb_X);
    const uint32_t orow = get_write_ptr(cb_orow);
#if M_MAX > 1
    for (uint32_t j = 0; j < nt; j++) {
        noc_async_read(Xd.get_noc_addr(t0 + j), X + j * TILE_F32, TILE_F32);
        if (phases & 2) noc_async_read(outd.get_noc_addr(0, (t0 + j) * mp * 128), orow + j * mp * 128, mp * 128);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < nt && (phases & 2); j++) {
        for (uint32_t m = 0; m < mp; m++) {
            for (uint32_t c = 0; c < 32; c++) {
                w32(X + j * TILE_F32 + at(OUT_ROW + m, c))[0] =
                    w32(orow + j * mp * 128 + (c < 16 ? 0 : mp * 64) + m * 64)[c & 15];
            }
        }
    }
#else
    for (uint32_t j = 0; j < nt; j++) {
        noc_async_read(Xd.get_noc_addr(t0 + j), X + j * TILE_F32, TILE_F32);
        if (phases & 2) noc_async_read(outd.get_noc_addr(0, (t0 + j) * 128), orow + j * 128, 128);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < nt && (phases & 2); j++) {
        for (uint32_t c = 0; c < 32; c++) w32(X + j * TILE_F32 + at(4, c))[0] = w32(orow + j * 128)[c];
    }
#endif
    cb_push_back(cb_X, 2);

    // A: fn's k-tiles and the x tiles, HC * nt of each; the sum of squares meanwhile
    if (nt && (phases & 1)) {
        cb_reserve_back(cb_w, HC * nt);
        cb_reserve_back(cb_x, HC * nt);
        const uint32_t wl = get_write_ptr(cb_w), xl = get_write_ptr(cb_x);
#if M_MAX > 1
        // fp32 x tiles by the NoC (x row m = row m's stream s: its two 16-float face pieces), no conversion; the rows'
        // sums of squares are the compute's (glm_hc_compute.cpp)
        for (uint32_t s = 0, i = 0; s < HC; s++) {
            for (uint32_t j = 0; j < nt; j++, i++) {
                noc_async_read(fn.get_noc_addr(s * KT_STREAM + t0 + j), wl + i * 2048, 2048);
                const uint32_t src = X + j * TILE_F32, dst = xl + i * TILE_F32;
                for (uint32_t m = 0; m < mp; m++) {
                    fused::l1_copy(src + at(4 * m + s, 0), dst + at(m, 0), 64);
                    fused::l1_copy(src + at(4 * m + s, 16), dst + at(m, 16), 64);
                }
            }
        }
        cb_reserve_back(cb_ones_m, 1);
        fused::fill_words(get_write_ptr(cb_ones_m), 512, 0x3f803f80);   // bf16 1.0 everywhere
        noc_async_read_barrier();
        cb_push_back(cb_ones_m, 1);
        cb_push_back(cb_w, HC * nt);
        cb_push_back(cb_x, HC * nt);
#else
        float ss = 0.0f;
        for (uint32_t s = 0, i = 0; s < HC; s++) {
            for (uint32_t j = 0; j < nt; j++, i++) {
                noc_async_read(fn.get_noc_addr(s * KT_STREAM + t0 + j), wl + i * 2048, 2048);
                const uint32_t src = X + j * TILE_F32, dst = xl + i * 2048;
                for (uint32_t c = 0; c < 32; c++) {
                    const float v = f32(src + at(s, c))[0];
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + at16(0, c))[0] = bf16_rne(w32(src + at(s, c))[0]);
                    ss = fused::fpu::add(ss, fused::fpu::mul(v, v));
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, HC * nt);
        cb_push_back(cb_x, HC * nt);
        cb_reserve_back(cb_sq, 1);
        f32(get_write_ptr(cb_sq))[0] = ss;
        cb_push_back(cb_sq, 1);
#endif
    }

    // 1: the host's parameters, broadcast with the barrier (and kept in par for the post launch); a post-only launch
    // reads them from par
    constexpr uint32_t NPAR = 128 * M_MAX;   // the parameters' bytes (a row's 24 floats in 128)
    if (phases & 1) {
        g.wait(1, arena + PARAM, NPAR, [&] {
            fused::Link link;
            link.open(H2D_CONFIG, D2H_CONFIG, NPAR, PARAM);   // (M rows: M_MAX row pages; M pages a row: no faster)
            link.send(arena + MIX);
            link.recv(arena + PARAM);
            link.close();
            noc_async_write(arena + PARAM, par.get_noc_addr(0), NPAR);
            noc_async_write_barrier();
        });
    } else {
        noc_async_read(par.get_noc_addr(0), arena + PARAM, NPAR);
        noc_async_read_barrier();
    }
    const volatile tt_l1_ptr float* p = f32(arena + PARAM);
    if (!nt) return;

    // B: PRE; C: C' (a coefficient tile a phase, pushed as each is needed)
#if M_MAX > 1
    // the rows past M: row M - 1's parameters (their rows' results are not read)
    auto prow = [&](uint32_t m) { return p + 32 * (m < M ? m : M - 1); };
    if (phases & 1) {
        cb_reserve_back(cb_coef, 1);
        const uint32_t co = get_write_ptr(cb_coef);
        zero(co, TILE_F32);
        for (uint32_t m = 0; m < mp; m++) {
            for (uint32_t s = 0; s < HC; s++) f32(co + at(m, 4 * m + s))[0] = prow(m)[s];
        }
        cb_push_back(cb_coef, 1);
    }
    if (phases & 2) {
        cb_reserve_back(cb_coef, 1);
        const uint32_t co = get_write_ptr(cb_coef);
        zero(co, TILE_F32);
        for (uint32_t m = 0; m < mp; m++) {
            for (uint32_t d = 0; d < HC; d++) {
                for (uint32_t s = 0; s < HC; s++) f32(co + at(4 * m + d, 4 * m + s))[0] = prow(m)[8 + 4 * d + s];
                f32(co + at(4 * m + d, OUT_ROW + m))[0] = prow(m)[4 + d];
            }
        }
        cb_push_back(cb_coef, 1);
    }
#else
    if (phases & 1) {
        cb_reserve_back(cb_coef, 1);
        const uint32_t co = get_write_ptr(cb_coef);
        zero(co, TILE_F32);
        for (uint32_t s = 0; s < HC; s++) f32(co + at(0, s))[0] = p[s];
        cb_push_back(cb_coef, 1);
    }
    if (phases & 2) {
        cb_reserve_back(cb_coef, 1);
        const uint32_t co = get_write_ptr(cb_coef);
        zero(co, TILE_F32);
        for (uint32_t d = 0; d < HC; d++) {
            for (uint32_t s = 0; s < HC; s++) f32(co + at(d, s))[0] = p[8 + 4 * d + s];
            f32(co + at(d, 4))[0] = p[4 + d];
        }
        cb_push_back(cb_coef, 1);
    }
#endif
#ifdef HC_NORM
    // 2: the norm's partial sums of squares (glm_hc_writer.cpp), broadcast once every writer has sent its own: after
    // the coefficient tiles, which h needs first; only the coordinator's wait matters (the writers poll the release)
#if M_MAX > 1
    if ((phases & 1) && g.coordinator) g.wait(2, arena + PARAM + NPAR + 64, M_MAX * NC * 16);   // (after the step words)
#else
    if ((phases & 1) && g.coordinator) g.wait(2, arena + PARAM + 128, NC * 16);
#endif
#endif
}
