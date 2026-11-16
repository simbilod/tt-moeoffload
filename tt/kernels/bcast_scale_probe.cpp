// Probe for the scales folded into the decode (step 3, tt/formats/bcast_scale_probe.py): F (cb 0, fp32 small
// integers, unpacked to Dest exactly) times an m row (cb 1, bf16, row 0 broadcast down the tile) times a d row (cb 2,
// fp32, row 0 broadcast), on the SFPU in fp32. Dest 4: F m d; Dest 1, 2: the broadcast tiles (to check the broadcast).
// REPS: the bcast + multiply sequence repeated, for its cost.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"

#ifndef REPS
#define REPS 1
#endif

void kernel_main() {
    constexpr uint32_t cb_f = 0, cb_m = 1, cb_d = 2, cb_out = 16;
    compute_kernel_hw_startup(cb_f, cb_out);
    cb_wait_front(cb_f, 1);
    cb_wait_front(cb_m, 1);
    cb_wait_front(cb_d, 1);
    for (uint32_t rep = 0; rep < REPS; rep++) {
        tile_regs_acquire();
        reconfig_data_format(cb_f, cb_f);
        copy_init(cb_f);
        copy_tile(cb_f, 0, 4);
        reconfig_data_format(cb_m, cb_m);     // the unpacker set up for each source's format (cb 1 bf16, cb 2 fp32)
        unary_bcast_init<BroadcastType::ROW>(cb_m);
        unary_bcast<BroadcastType::ROW>(cb_m, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(4, 1, 4);
        reconfig_data_format(cb_d, cb_d);
        unary_bcast_init<BroadcastType::ROW>(cb_d);
        unary_bcast<BroadcastType::ROW>(cb_d, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(4, 2, 4);
        tile_regs_commit();
        if (rep + 1 < REPS) {
            tile_regs_wait();
            tile_regs_release();
        }
    }
    cb_reserve_back(cb_out, 3);
    tile_regs_wait();
    pack_tile(4, cb_out);
    pack_tile(1, cb_out);
    pack_tile(2, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 3);
}
