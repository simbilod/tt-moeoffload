// Probe: which tile elements form one SFPU row (32 lanes), and what a COL broadcast puts there.
// Dest 0: cb 0's tile as is; Dest 1: cb 1's tile broadcast down its column 0 (unary_bcast<COL>);
// Dest 2: every lane of SFPU row d (d = 0..7 within a face) set to d + 8 * lane-group probe (see below).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
namespace ckernel::sfpu {
inline void probe_face() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::dst_reg[2 * 32] = sfpi::vFloat(float(d));
        // a lane id: the input tile holds element indices, so the SFPU sees each lane's own index
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[3 * 32] = v;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_v = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_a, cb_out);
    MATH((SFPU_UNARY_INIT(tanh)));
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_v, 1);
    tile_regs_acquire();
    copy_init(cb_a);
    copy_tile(cb_a, 0, 0);
    unary_bcast_init<BroadcastType::COL>(cb_v);
    unary_bcast<BroadcastType::COL>(cb_v, 0, 1);
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, probe_face, 0, VectorMode::RC)));
    tile_regs_commit();
    cb_reserve_back(cb_out, 4);
    tile_regs_wait();
    for (uint32_t t = 0; t < 4; t++) {
        pack_tile(t, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, 4);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_v, 1);
}
