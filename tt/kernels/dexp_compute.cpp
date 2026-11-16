// Decode-time routed experts on the card (reader and phases in dexp_reader.cpp). One token, the experts
// of a route list, from the experts' native codes in card form (tt/formats/repack.py; the unit,
// tt/kernels/expert_unit.cpp, for the arithmetic). A unit is one 32-k-tile group (32 quant blocks) of one output tile:
//   decode   each code tile (8 k-tiles) on the SFPU in one Dest pass: codes in Dest 0, nibbles 0..6 into Dest 1..7,
//            nibble 7 over Dest 0; packed as bfp8_b F tiles (exact: integers up to 127)
//   matmul   P = sum_k X_k F_k over the group, X_k holding x's k-tile k in row k % 32 (zeros elsewhere), so row b of
//            P is block b's partial sum; then Y = A * P (A: the group's scale tile, row b = block b's scales), all in
//            one Dest acquisition, Y packed (fp32)
//   sum      out = ones Y: every row the column sums, row 0 the unit's 32 partial outputs
// The F buffer is filled to 32 tiles for every unit (padding pushes), so its indexed reads never wrap.
// Unpack-to-dest (the codes, the scales, E2's partials) waits for the math thread only, not for the packer: the first
// such unpack after a pack must not hit a Dest tile that pack reads. So the sums and h come out of Dest 7, which is
// never unpacked into, and a decode's next codes wait until the previous code tile's F tiles are pushed.
// E1 (gate|up, IQ3_S, linear decode): unit (expert j, 32-output tile p), 20 per expert: gate column p and up column p,
// three groups each (4, 4, 2 code tiles), the groups' Y summed by the packer's L1 accumulation, then h = w_j * silu(g)
// * u. E2 (down): unit (j, column 0..79), one group of 20 k-tiles from bfp8_b tiles of the decoded values (no SFPU).
// Units split evenly over the cores from the route count.
// Runtime args: this core's index, the core count.
#include <cstdint>
#include "fused/cmp.h"

#ifdef TRISC_MATH
#include "dexp_sfpu.h"
#endif

using namespace fused;

constexpr uint32_t cb_codes = 0, cb_x = 1, cb_ones = 2, cb_s = 3, cb_y = 5, cb_out = 6, cb_f = 7;
constexpr uint32_t cb_w2 = 9, cb_h16 = 10, cb_xd = 11, cb_meta = 12, cb_fd = 18;
constexpr uint32_t FT = 32, KD = 20;  // F tiles pushed per unit; down's k-tiles

inline uint32_t lo(uint32_t n, uint32_t k, uint32_t c) { return n * k / c; }

// decode nc code tiles from cb `codes` into cb_f (bfp8_b, k order), padded to FT tiles. HALF (half-sync Dest, 4 tiles
// per half): three passes per code tile (nibbles 0-2, 3-5, 6-7 into Dest 1..3), each copying the codes again; math
// decodes one half while the packer drains the other, and unpack-to-dest writes the math's half. Else (full sync,
// 8 tiles): one pass, nibbles 0..6 into Dest 1..7, nibble 7 over Dest 0.
template <bool IQ4>
inline void decode(uint32_t codes, uint32_t nc) {
    MATH((SFPU_UNARY_INIT(tanh)));  // generic SFPU address setup (a binary SFPU op ran since)
#ifdef TRISC_MATH
    if constexpr (IQ4) {
        ckernel::sfpu::dexp::iq4nl_init();
    }
#endif
    reconfig_data_format_srca(codes);
    copy_init(codes);
    pack_reconfig_data_format(cb_f);
    for (uint32_t c = 0; c < nc; c++) {
        cb_wait_front(codes, 1);
        cb_reserve_back(cb_f, 8);
#ifdef HALF
        for (uint32_t pass = 0; pass < 3; pass++) {
            tile_regs_acquire();
            copy_tile(codes, 0, 0);
#ifndef DEXP_NO_DECODE
            if (pass == 0) {
                SFPU_CALL(dexp::decode3_pass0);
            } else if (pass == 1) {
                SFPU_CALL(dexp::decode3_pass1);
            } else {
                SFPU_CALL(dexp::decode3_pass2);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 1; t <= (pass == 2 ? 2u : 3u); t++) {
                pack_tile(t, cb_f);
            }
            tile_regs_release();
        }
        cb_pop_front(codes, 1);
#else
        if (c) {
            cb_wait_front(cb_f, 8 * c);  // the previous code tile's pack (it read Dest 0) is done
        }
        tile_regs_acquire();
        copy_tile(codes, 0, 0);
#ifndef DEXP_NO_DECODE  // timing breakdown: codes copied, no SFPU decode
        if constexpr (IQ4) {
            SFPU_CALL(dexp::decode8_iq4nl);
        } else {
            SFPU_CALL(dexp::decode8_linear);
        }
#endif
        tile_regs_commit();
        cb_pop_front(codes, 1);
        tile_regs_wait();
        for (uint32_t t = 1; t < 8; t++) {
            pack_tile(t, cb_f);
        }
        pack_tile(0, cb_f);
        tile_regs_release();
#endif
        cb_push_back(cb_f, 8);
    }
    if (nc * 8 < FT) {
        cb_reserve_back(cb_f, FT - nc * 8);
        cb_push_back(cb_f, FT - nc * 8);
    }
}

#ifdef HALF
constexpr uint32_t D_G = 2, D_U = 3, D_W = 1, D_OUT = 2;  // E1's final tiles and the sums' tile in half-sync Dest
#else
constexpr uint32_t D_G = 6, D_U = 7, D_W = 5, D_OUT = 7;
#endif

// Y (+)= A * (sum over the group's nk k-tiles of X F) into slot `slot` of cb_y (the packer's L1 accumulation adds
// the groups of one output tile), x from k-tile kx of xcb, F from fcb
inline void group(uint32_t xcb, uint32_t fcb, uint32_t kx, uint32_t nk, uint32_t slot, bool first) {
    cb_wait_front(fcb, FT);
    cb_wait_front(cb_s, 1);
    tile_regs_acquire();
    matmul_start(xcb, fcb);
#ifndef DEXP_NO_MATMUL  // timing breakdown
    for (uint32_t t = 0; t < nk; t++) {
        matmul_tiles(xcb, fcb, kx + t, t, 0);
    }
#endif
    load(cb_s, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    cb_pop_front(fcb, FT);
    cb_pop_front(cb_s, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_y);
    pack_reconfig_l1_acc(first ? 0 : 1);
    pack_tile<true>(0, cb_y, slot);
    pack_reconfig_l1_acc(0);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0), cores = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_f, cb_y);
    cb_wait_front(cb_meta, 1);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    cb_wait_front(cb_ones, 1);

    // ---- E1: gate column p and up column p (three groups each, summed in L1), then h = w * silu(g) * u, bf16
    cb_wait_front(cb_x, 80);
    for (uint32_t v = lo(20 * n, me, cores); v < lo(20 * n, me + 1, cores); v++) {
        cb_reserve_back(cb_y, 2);
        for (uint32_t half = 0; half < 2; half++) {
            for (uint32_t g = 0; g < 3; g++) {
                const uint32_t nc = g == 2 ? 2 : 4;
                decode<false>(cb_codes, nc);
                group(cb_x, cb_f, 32 * g, 8 * nc, half, g == 0);
            }
        }
        cb_push_back(cb_y, 2);
        cb_wait_front(cb_y, 2);
        cb_wait_front(cb_w2, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_y);
        matmul_tiles(cb_ones, cb_y, 0, 0, D_G);  // every row: g's column sums
        matmul_tiles(cb_ones, cb_y, 0, 1, D_U);  // u's
        load(cb_w2, 0, D_W);
        silu_tile_init();
        silu_tile(D_G);
        mul_binary_tile_init();
        mul_binary_tile(D_G, D_U, D_G);
        mul_binary_tile(D_G, D_W, D_U);
        tile_regs_commit();
        cb_pop_front(cb_y, 2);
        cb_pop_front(cb_w2, 1);
        tile_regs_wait();
        pack_n(cb_h16, 1, D_U);
        tile_regs_release();
    }
    cb_pop_front(cb_x, 80);

    // ---- E2: down, one unit per (expert, output column), F from card memory; x = h_j from the barrier's broadcast
    uint32_t cur = 0xffffffff;
    for (uint32_t w = lo(80 * n, me, cores); w < lo(80 * n, me + 1, cores); w++) {
        const uint32_t j = w / 80;
        if (j != cur) {
            if (cur != 0xffffffff) {
                cb_pop_front(cb_xd, KD);
            }
            cb_wait_front(cb_xd, KD);
            cur = j;
        }
        cb_reserve_back(cb_y, 1);
        group(cb_xd, cb_fd, 0, KD, 0, true);
        cb_push_back(cb_y, 1);
        cb_wait_front(cb_y, 1);
        tile_regs_acquire();
        matmul_start(cb_ones, cb_y);
        matmul_tiles(cb_ones, cb_y, 0, 0, D_OUT);
        cb_pop_front(cb_y, 1);
        finish(cb_out, 1, D_OUT);
    }
    if (cur != 0xffffffff) {
        cb_pop_front(cb_xd, KD);
    }
}
