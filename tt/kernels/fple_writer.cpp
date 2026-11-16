// Fused PLE block writer (the phases are in fple_reader.cpp); token r's values are row r of every tile.
//   P1  key tile n: A and B (cb 6, the first 16 bytes of each row) to region PS's slots n and 320 + n of each token;
//       value tile n: the tile's rows (cb 7) to region PV, V (cb 6) to slot 640 + n - 320; arrive
//   P2  item (r, c): gate (cb 13) and g (cb 26), their first 16 bytes, to region PG at 128 r + 32 c and + 16; arrive
//   P3  per tile n: row r of n (cb 21) to the history ring's row (pos + r) % RING, and with m > 3 the tile's rows to
//       the reader (cb 27, the conv's rows from this pass); X's tile n (cb 24 fp32, cb 25 bf16) back into X's regions
//       and X's check; arrive
// Runtime args: grid.h's block; arena offsets X32, X16, EMB, PV, PS, PG; P1's first tile and count, P2's item, P3's
// first tile and count; the ring, the step tensor (word 0: pos), the checks PS ([m x 720 x 4]), PG ([m x 4 x 8]), X
// ([m x 4 x 2560]; the X fp32 input itself in the token loop), X's bf16 out (0 none); [GRID_PROFILE: the profile];
// m. Compile args: a DRAM accessor.
// -DX_BLOCKS=n (fple_reader.cpp): m and Mp from the step tensor; X's tile n goes back as its block of Mp rows (rows
// m..Mp-1 copies of row m - 1) to page n / 80 of X's check (in place: the X tensors) and of X's bf16 out, nothing to
// the arena.
#include <cstdint>

#include "fused/dm.h"

constexpr uint32_t cb_psum = 6, cb_pv = 7, cb_qgate = 13, cb_qg = 26, cb_rnw = 21, cb_rx32 = 24, cb_rx16 = 25;
constexpr uint32_t cb_shift = 27, cb_arena = 61, cb_stage = 62;
constexpr uint32_t D = 2560, CH = 10240, NKEY = 320, IDLE = 0xffffffff;
constexpr uint32_t RING = 17, XROW = 4 * D * 4, X16ROW = 4 * D * 2, PSROW = 720 * 16;
constexpr uint32_t T1 = XROW + X16ROW + D * 2, T2 = D * 4 + PSROW;  // a token's arena blocks: X32 X16 EMB, PV PS

// rows 0..m-1 of the front tile of cb (fp32: piece 64, bf16: piece 32) to stage, 2 piece bytes per row; pops it
inline void stage_m(uint32_t cb, uint32_t m, uint32_t piece, uint32_t stage) {
    cb_wait_front(cb, 1);
    const uint32_t t = get_read_ptr(cb);
    for (uint32_t r = 0; r < m; r++) {
        fused::l1_copy(t + r * piece, stage + r * 2 * piece, piece);
        fused::l1_copy(t + 16 * piece + r * piece, stage + r * 2 * piece + piece, piece);
    }
    noc_async_read_barrier();
    cb_pop_front(cb, 1);
}

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    auto arg = [&]() { return get_arg_val<uint32_t>(i++); };
    auto acc = [&](uint32_t page) { return TensorAccessor(d_args, arg(), page); };
    const fused::Grid grid = fused::Grid::from_args(0);
    i += fused::Grid::args(grid.nrects);
    const uint32_t arena = get_write_ptr(cb_arena), stage = get_write_ptr(cb_stage);
    const uint32_t r_x32 = arena + arg(), r_x16 = arena + arg();
    arg();  // EMB
    const uint32_t r_pv = arena + arg(), r_ps = arena + arg(), r_pg = arena + arg();
    const uint32_t p1_first = arg(), p1_count = arg(), item = arg(), p3_first = arg(), p3_count = arg();
    const auto ring = acc(CH * 4);
    const auto step = acc(64);
#ifdef X_BLOCKS
    const auto chk_ps = acc(PSROW), chk_pg = acc(128), chk_x = acc(X_BLOCKS * D * 4);  // X: a stream a page
    const uint32_t x16_out = arg();  // X's bf16 [4 x n 2560] to card memory too (0: none)
    const auto x16o = TensorAccessor(d_args, x16_out, X_BLOCKS * D * 2);
#else
    const auto chk_ps = acc(PSROW), chk_pg = acc(128), chk_x = acc(XROW);
    const uint32_t x16_out = arg();  // X's bf16 [m x 4 x 2560] to card memory too (0: none)
    const auto x16o = TensorAccessor(d_args, x16_out, X16ROW);
#endif
#ifdef GRID_PROFILE
    const uint32_t gp_addr = arg(), gp_slot = arg();  // grid.h's per-core profile
#endif
#ifdef X_BLOCKS
    arg();  // m: the step tensor's
#else
    const uint32_t m = arg();
#endif
    noc_async_read(step.get_noc_addr(0), stage, 64);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
#ifdef X_BLOCKS
    const uint32_t m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[1];
    const uint32_t mp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[4];
#endif

    auto send16 = [&](uint32_t cb, uint32_t slot) {  // the first 16 bytes of each row of cb's front tile to PS's slot
        cb_wait_front(cb, 1);
        for (uint32_t r = 0; r < m; r++) {
            fused::l1_copy(get_read_ptr(cb) + r * 64, stage + 1024 + r * 16, 16);
        }
        noc_async_read_barrier();
        cb_pop_front(cb, 1);
        for (uint32_t r = 0; r < m; r++) {
            grid.send(stage + 1024 + r * 16, r_ps + r * T2 + slot * 16, 16);
            noc_async_write(stage + 1024 + r * 16, chk_ps.get_noc_addr(r, slot * 16), 16);
        }
        noc_async_write_barrier();
    };
    uint32_t epoch = 1;
    grid.arrive(epoch);

    // ---- P1
    for (uint32_t n = p1_first; n < p1_first + p1_count; n++) {
        if (n < NKEY) {
            send16(cb_psum, n);
            send16(cb_psum, NKEY + n);
        } else {
            stage_m(cb_pv, m, 64, stage);
            for (uint32_t r = 0; r < m; r++) {
                grid.send(stage + r * 128, r_pv + r * T2 + (n - NKEY) * 128, 128);
            }
            noc_async_write_barrier();
            send16(cb_psum, 2 * NKEY + n - NKEY);
        }
    }
    grid.arrive(++epoch);

    // ---- P2
    if (item != IDLE) {
        const uint32_t r = item / 4, c = item % 4;
        for (uint32_t k = 0; k < 2; k++) {
            const uint32_t cb = k ? cb_qg : cb_qgate;
            cb_wait_front(cb, 1);
            fused::l1_copy(get_read_ptr(cb), stage, 16);
            noc_async_read_barrier();
            cb_pop_front(cb, 1);
            grid.send(stage, r_pg + r * 128 + c * 32 + k * 16, 16);
            noc_async_write(stage, chk_pg.get_noc_addr(r, c * 32 + k * 16), 16);
            noc_async_write_barrier();
        }
    }
    grid.arrive(++epoch);

    // ---- P3: stage rows n at 0, X fp32 at m 128, X bf16 at 2 m 128
    const uint32_t s32 = stage + m * 128, s16 = stage + 2 * m * 128;
    for (uint32_t n = p3_first; n < p3_first + p3_count; n++) {
        stage_m(cb_rnw, m, 64, stage);
        for (uint32_t r = 0; r < m; r++) {
            noc_async_write(stage + r * 128, ring.get_noc_addr((pos + r) % RING, n * 128), 128);
        }
        if (m > 3) {
            cb_reserve_back(cb_shift, 1);
            fused::l1_copy(stage, get_write_ptr(cb_shift), m * 128);
            noc_async_read_barrier();
            cb_push_back(cb_shift, 1);
        }
        noc_async_write_barrier();
#ifdef X_BLOCKS
        // the tile's block: per face rows 0..m-1, then copies of row m - 1 up to Mp
        const uint32_t sb32 = stage + 4 * 128, sb16 = sb32 + 4 * 128;  // past the rows staged above (m <= 4)
        for (uint32_t k = 0; k < 2; k++) {
            const uint32_t cb = k ? cb_rx16 : cb_rx32, piece = k ? 32 : 64, sb = k ? sb16 : sb32;
            cb_wait_front(cb, 1);
            const uint32_t t = get_read_ptr(cb);
            for (uint32_t f = 0; f < 2; f++) {
                fused::l1_copy(t + f * 16 * piece, sb + f * mp * piece, m * piece);
                for (uint32_t r = m; r < mp; r++) {
                    fused::l1_copy(t + f * 16 * piece + (m - 1) * piece, sb + f * mp * piece + r * piece, piece);
                }
            }
            noc_async_read_barrier();
            cb_pop_front(cb, 1);
        }
        noc_async_write(sb32, chk_x.get_noc_addr(n / 80, (n % 80) * mp * 128), mp * 128);
        if (x16_out) {
            noc_async_write(sb16, x16o.get_noc_addr(n / 80, (n % 80) * mp * 64), mp * 64);
        }
        noc_async_write_barrier();
        continue;
#endif
        stage_m(cb_rx32, m, 64, s32);
        stage_m(cb_rx16, m, 32, s16);
        for (uint32_t r = 0; r < m; r++) {
            grid.send(s32 + r * 128, r_x32 + r * T1 + n * 128, 128);
            grid.send(s16 + r * 64, r_x16 + r * T1 + n * 64, 64);
            noc_async_write(s32 + r * 128, chk_x.get_noc_addr(r, n * 128), 128);
            if (x16_out) {
                noc_async_write(s16 + r * 64, x16o.get_noc_addr(r, n * 64), 64);
            }
        }
        noc_async_write_barrier();
    }
    grid.arrive(++epoch);
#ifdef GRID_PROFILE
    fused::profile_dump(d_args, gp_addr, gp_slot, false);
#endif
}
