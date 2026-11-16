// GLM indexer score compute (prefill): score[t, p] = sum_h relu(iq_h[t] . pk[p]) * w[t, h], fp32, in the stock
// chain's order (g6_prefill MlaMixer.sparse: per head a matmul with relu, a multiply by the gate column, then
// score = score + s for h = 1 .. 31; each step rounded to fp32). Per item (query tile, pool tile), in Dest: 0 the
// score, heads in rounds of 6 (round 0: heads 0-5 in Dest 0-5; then 6 a round in Dest 1-6; every slot zeroed first), each
// accumulating its 4 k-tiles (pk transposed in the unpacker, Flash-Next's pf_qsa_score), relu, then per head its gate
// tile copied to Dest 7, Dest J *= Dest 7, Dest 0 += Dest J (two SFPU passes: no fused multiply-add). cb 0 iq (128 a
// query tile), cb 1 pk (4 a pool tile), cb 2 the gates (32 a query tile), cb 16 the score. Runtime args: first item,
// count. Compile args: PT, NHEAD (32), MUL (1).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// called with Dest 0 as the base: tile J at dst_reg offset J * 32
template <int J>
inline void is_mul() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat a = sfpi::dst_reg[J * 32], b = sfpi::dst_reg[7 * 32];
        sfpi::dst_reg[J * 32] = a * b;
        sfpi::dst_reg++;
    }
}

template <int J>
inline void is_add() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[J * 32];
        sfpi::dst_reg[0] = a + b;
        sfpi::dst_reg++;
    }
}

template <int J>
inline void is_zero() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::dst_reg[J * 32] = sfpi::vFloat(0.0f);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#define SFPU(F) MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, F, 0, VectorMode::RC)))
#else
#define SFPU(F)
#endif

#define BY_SLOT(OP, J)                       \
    switch (J) {                             \
        case 0: SFPU(OP<0>); break; \
        case 1: SFPU(OP<1>); break; \
        case 2: SFPU(OP<2>); break; \
        case 3: SFPU(OP<3>); break; \
        case 4: SFPU(OP<4>); break; \
        case 5: SFPU(OP<5>); break; \
        default: SFPU(OP<6>); break; \
    }

constexpr uint32_t cb_iq = 0, cb_pk = 1, cb_w = 2, cb_out = 16;

void kernel_main() {
    constexpr uint32_t PT = get_compile_time_arg_val(0);
    constexpr uint32_t NHEAD = get_compile_time_arg_val(1);   // 32 (fewer, MUL 0: the probe's stage checks)
    constexpr uint32_t MUL = get_compile_time_arg_val(2);
    const uint32_t first = get_arg_val<uint32_t>(0), count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_iq, cb_pk, cb_out);
    uint32_t q_prev = 0xffffffff;
    for (uint32_t it = first; it < first + count; it++) {
        const uint32_t q = it / PT;
        if (q != q_prev) {
            if (q_prev != 0xffffffff) {
                cb_pop_front(cb_iq, 128);
                cb_pop_front(cb_w, 32);
            }
            cb_wait_front(cb_iq, 128);
            cb_wait_front(cb_w, 32);
            q_prev = q;
        }
        cb_wait_front(cb_pk, 4);
        tile_regs_acquire();
        for (uint32_t r = 0; r < NHEAD; r += 6) {
            const uint32_t nh = r + 6 <= NHEAD ? 6 : NHEAD - r, base = r ? 1 : 0;
            for (uint32_t i = 0; i < nh; i++) {   // the matmul accumulates into its slot
                BY_SLOT(is_zero, base + i)
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_iq, cb_pk);
            matmul_init(cb_iq, cb_pk, 1);
            for (uint32_t i = 0; i < nh; i++) {
                for (uint32_t k = 0; k < 4; k++) {
                    matmul_tiles(cb_iq, cb_pk, (r + i) * 4 + k, k, base + i);
                }
            }
            relu_tile_init();
            for (uint32_t i = 0; i < nh; i++) {
                relu_tile(base + i);
            }
            reconfig_data_format_srca(cb_w);
            copy_init(cb_w);
            for (uint32_t i = 0; i < nh; i++) {
                copy_tile(cb_w, r + i, 7);
                if constexpr (MUL) {
                    BY_SLOT(is_mul, base + i)
                }
                if (r + i) {
                    BY_SLOT(is_add, base + i)
                }
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_pk, 4);
    }
    cb_pop_front(cb_iq, 128);
    cb_pop_front(cb_w, 32);
}
