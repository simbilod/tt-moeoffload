// Fused QSA attention layer, program 2 compute (the phases are described in fattn2_reader.cpp).
// Runtime args: AP's role, AC's role, oproj's tile, the H1 item, the H2 tile, the R1 item, R2's tile.
// Compile args: batch.
// M rows (-DM_MAX > 1, fattn2_reader.cpp): M, mp and the union's blocks from the reader's cb 2 tile 2; AP's partials
// (partial_m) with T column tiles of q^T and the mask as a ninth dims tile, AC's combine over P partials (combine_m),
// the mix at M rows (fused/hc_cmp.h compute_m); oproj and the FFN as at one row on M-row tiles. Preprocessed out at
// M_MAX = 1.
#include <cstdint>
#include "fused/cmp.h"
#include "fused/hc_cmp.h"
#include "fused/tail_cmp.h"
#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/fill.h"

#ifndef M_MAX
#define M_MAX 1
#endif

using namespace fused;

constexpr uint32_t cb_w = 0, cb_x = 1, cb_ones = 2;
constexpr uint32_t cb_aq = 16, cb_akrm = 17, cb_avrm = 18, cb_ak = 19, cb_av = 20, cb_amask = 22, cb_as = 23;
constexpr uint32_t cb_am = 24, cb_ap = 25, cb_al = 26, cb_aout = 27, cb_aqt = 21, cb_apt = 48, cb_aq16 = 58, cb_ap32 = 59;
constexpr uint32_t cb_cm = 28, cb_co = 29, cb_cl = 30, cb_cid = 31, cb_cg = 32, cb_cM = 33, cb_cdiag = 34;
constexpr uint32_t cb_cA = 35, cb_cAL = 36, cb_cW = 36, cb_crl = 37, cb_cout16 = 38, cb_cchk = 39;
constexpr uint32_t NB = 4, NPARTS = 17, IDLE = 0xffffffff;

void tilize8(uint32_t src, uint32_t dst) {
    cb_wait_front(src, 8);
    cb_reserve_back(dst, 8);
    reconfig_data_format_srca(src);
    pack_reconfig_data_format(dst);
    tilize_init(src, 8, dst);
    tilize_block(src, 8, dst);
    tilize_uninit(src, dst);
    cb_push_back(dst, 8);
    cb_pop_front(src, 8);
}

// the attention partial of one core, nb blocks of 32 cells, in the orientation of the validated decode kernel
// (dattn_compute.cpp): S^T_b = K_b q^T (cells x heads), m the column max, P^T_b = exp(S^T_b - m), l the column sum,
// P_b = transpose(P^T_b), O = sum_b P_b V_b (heads x dims). q arrives in rows (heads x dims) and is transposed here;
// m and l leave as row tiles (row 0, a value per head). Transposes run on bf16 tiles (q, P), as in dattn.
void partial(bool own) {
    const uint32_t nb = own ? 1 : NB;
    cb_wait_front(cb_aq, 8);
    tile_regs_acquire();  // q^T, through a bf16 copy (the transpose of fp32 tiles came out wrong)
    load_n(cb_aq, 8);
    cb_pop_front(cb_aq, 8);
    finish(cb_aq16, 8);
    cb_wait_front(cb_aq16, 8);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_aq16);
    transpose_init(cb_aq16);
    for (uint32_t d = 0; d < 8; d++) {
        transpose_tile(cb_aq16, d, d);
    }
    cb_pop_front(cb_aq16, 8);
    finish(cb_aqt, 8);
    cb_wait_front(cb_aqt, 8);
    for (uint32_t b = 0; b < nb; b++) {
        tilize8(cb_akrm, cb_ak);
        tilize8(cb_avrm, cb_av);
        cb_wait_front(cb_ak, (b + 1) * 8);
        tile_regs_acquire();
        cb_wait_front(cb_amask, 1);  // every block's mask (the cells not taken at -10000)
        load(cb_amask, 0, 0);
        cb_pop_front(cb_amask, 1);
        matmul_start(cb_ak, cb_aqt);
        for (uint32_t d = 0; d < 8; d++) {
            matmul_tiles(cb_ak, cb_aqt, b * 8 + d, d, 0);
        }
        finish(cb_as, 1);
    }
    cb_pop_front(cb_aqt, 8);
    cb_pop_front(cb_ak, nb * 8);
    cb_wait_front(cb_as, nb);

    tile_regs_acquire();  // m: the column max over the blocks (reduce_tile folds into Dest: start it at -1e30)
    fill_tile_init();
    fill_tile(0, -1e30f);
    reconfig_data_format(cb_as, cb_ones);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_as, cb_ones, cb_am);
    for (uint32_t b = 0; b < nb; b++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_as, cb_ones, b, 0, 0);
    }
    reduce_uninit();
    finish(cb_am, 1);
    cb_wait_front(cb_am, 1);

    for (uint32_t b = 0; b < nb; b++) {  // P^T_b = exp(S^T_b - m)
        tile_regs_acquire();
        reconfig_data_format(cb_as, cb_am);
        sub_bcast_rows_init(cb_as, cb_am);
        sub_tiles_bcast_rows(cb_as, cb_am, b, 0, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_ap, 1);    // bf16, for the transpose and P V
        pack_n(cb_ap32, 1);  // fp32, for l
        tile_regs_release();
    }
    cb_pop_front(cb_as, nb);
    cb_wait_front(cb_ap, nb);

    cb_wait_front(cb_ap32, nb);
    tile_regs_acquire();  // l: the column sum of P^T (from 0)
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_ap32, cb_ones);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_ap32, cb_ones, cb_al);
    for (uint32_t b = 0; b < nb; b++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_ap32, cb_ones, b, 0, 0);
    }
    reduce_uninit();
    cb_pop_front(cb_ap32, nb);
    finish(cb_al, 1);

    reconfig_data_format_srca(cb_ap);  // P = transpose(P^T)
    transpose_init(cb_ap);
    for (uint32_t b = 0; b < nb; b++) {
        tile_regs_acquire();
        transpose_tile(cb_ap, b, 0);
        finish(cb_apt, 1);
    }
    cb_pop_front(cb_ap, nb);
    cb_wait_front(cb_apt, nb);

    cb_wait_front(cb_av, nb * 8);  // O = sum_b P_b V_b
    tile_regs_acquire();
    matmul_start(cb_apt, cb_av);
    for (uint32_t d = 0; d < 8; d++) {
        for (uint32_t b = 0; b < nb; b++) {
            matmul_tiles(cb_apt, cb_av, b, b * 8 + d, d);
        }
    }
    cb_pop_front(cb_apt, nb);
    cb_pop_front(cb_av, nb * 8);
    finish(cb_aout, 8);
    cb_wait_front(cb_al, 1);  // then m and l (row 0: a value per head)
    tile_regs_acquire();
    load(cb_am, 0, 0);
    load(cb_al, 0, 1);
    cb_pop_front(cb_am, 1);
    cb_pop_front(cb_al, 1);
    finish(cb_aout, 2);
}

// the combine of one (KV head, dims tile), times the gate, as the standalone combine (dcomb_compute.cpp): the
// partials' m and l are the rows of one statistics tile each (heads along the columns; rows past the partials
// -1e30 / 0), so the softmax statistics are a few tile operations, not one per partial:
//   M = the column max of m, A = exp(m - M), 1 / L = 1 / the column sum of A l, W = A / L,
//   D_i = the identity times W's row i, O = sum_i D_i O_i (matmuls accumulating in Dest), times the gate
void combine_parts() {
    cb_wait_front(cb_cm, 1);
    cb_wait_front(cb_cl, 1);
    cb_wait_front(cb_cid, 1);
    tile_regs_acquire();  // M
    fill_tile_init();
    fill_tile(0, -1e30f);
    reconfig_data_format(cb_cm, cb_ones);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_cm, cb_ones, cb_cM);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_cm, cb_ones, 0, 0, 0);
    reduce_uninit();
    finish(cb_cM, 1);
    cb_wait_front(cb_cM, 1);
    tile_regs_acquire();  // A
    reconfig_data_format(cb_cm, cb_cM);
    sub_bcast_rows_init(cb_cm, cb_cM);
    sub_tiles_bcast_rows(cb_cm, cb_cM, 0, 0, 0);
    exp_tile_init();
    exp_tile(0);
    cb_pop_front(cb_cm, 1);
    cb_pop_front(cb_cM, 1);
    finish(cb_cA, 1);
    cb_wait_front(cb_cA, 1);
    tile_regs_acquire();  // A l
    reconfig_data_format(cb_cA, cb_cl);
    mul_init(cb_cA, cb_cl);
    mul_tiles(cb_cA, cb_cl, 0, 0, 0);
    cb_pop_front(cb_cl, 1);
    finish(cb_cAL, 1);
    cb_wait_front(cb_cAL, 1);
    tile_regs_acquire();  // 1 / L
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_cAL, cb_ones);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_cAL, cb_ones, cb_crl);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_cAL, cb_ones, 0, 0, 0);
    reduce_uninit();
    recip_tile_init();
    recip_tile(0);
    cb_pop_front(cb_cAL, 1);
    finish(cb_crl, 1);
    cb_wait_front(cb_crl, 1);
    tile_regs_acquire();  // W
    reconfig_data_format(cb_cA, cb_crl);
    mul_bcast_rows_init(cb_cA, cb_crl);
    mul_tiles_bcast_rows(cb_cA, cb_crl, 0, 0, 0);
    cb_pop_front(cb_cA, 1);
    cb_pop_front(cb_crl, 1);
    finish(cb_cW, 1);
    cb_wait_front(cb_cW, 1);
    reconfig_data_format(cb_cid, cb_cW);  // D_i
    mul_bcast_rows_init(cb_cid, cb_cW);
    for (uint32_t p = 0; p < NPARTS; p++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_cid, cb_cW, 0, 0, 0, p);
        finish(cb_cdiag, 1);
    }
    cb_pop_front(cb_cW, 1);
    cb_pop_front(cb_cid, 1);
    cb_wait_front(cb_cdiag, NPARTS);
    cb_wait_front(cb_co, NPARTS);
    cb_wait_front(cb_cg, 1);
    tile_regs_acquire();  // O, times the gate
    matmul_start(cb_cdiag, cb_co);
    for (uint32_t p = 0; p < NPARTS; p++) {
        matmul_tiles(cb_cdiag, cb_co, p, p, 0);
    }
    load(cb_cg, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    cb_pop_front(cb_cdiag, NPARTS);
    cb_pop_front(cb_co, NPARTS);
    cb_pop_front(cb_cg, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_n(cb_cout16, 1);
    pack_n(cb_cchk, 1);
    tile_regs_release();
}

#if M_MAX > 1
constexpr uint32_t NAP = 32, T_MAX = (12 * M_MAX + 31) / 32;  // the reader's
constexpr uint32_t cb_akx = 22, cb_cacc = 16;

// finish() a tile per push (the writer takes them one by one: cb 27 holds fewer than a partial's 10 T)
inline void finish_each(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < n; t++) {
        pack_n(cb, 1, t);
    }
    tile_regs_release();
}

// q^T for every partial of the core: the reader's 9 T bf16 row tiles (per column tile t its 8 dims tiles, then the
// mask's) transposed into cb_aqt, kept until the core's last partial
void q_transpose(uint32_t T) {
    cb_wait_front(cb_aq16, 9 * T);
    reconfig_data_format_srca(cb_aq16);
    transpose_init(cb_aq16);
    for (uint32_t j = 0; j < 9 * T; j++) {
        tile_regs_acquire();
        transpose_tile(cb_aq16, j, 0);
        finish(cb_aqt, 1);
    }
    cb_pop_front(cb_aq16, 9 * T);
    cb_wait_front(cb_aqt, 9 * T);
}

// one partial of nb blocks over T column tiles, partial()'s steps: S^T_{b,t} = Kx_b Qx_t + sum_d K_{b,d} q^T_{t,d} (the
// mask's product first: it is exact), m_t the column max over the blocks, P^T = exp(S^T - m), l_t the column sum,
// P = transpose(P^T), O_t = sum_b P_{b,t} V_b. Out: O_t's 8 tiles for each t, then the T m tiles, the T l tiles
void partial_m(uint32_t nb, uint32_t T) {
    for (uint32_t b = 0; b < nb; b++) {
        tilize8(cb_akrm, cb_ak);
        tilize8(cb_avrm, cb_av);
        cb_wait_front(cb_ak, (b + 1) * 8);
        cb_wait_front(cb_akx, 1);
        for (uint32_t t = 0; t < T; t++) {
            tile_regs_acquire();
            matmul_start(cb_akx, cb_aqt);
            matmul_tiles(cb_akx, cb_aqt, 0, 9 * t + 8, 0);
            matmul_start(cb_ak, cb_aqt);
            for (uint32_t d = 0; d < 8; d++) {
                matmul_tiles(cb_ak, cb_aqt, b * 8 + d, 9 * t + d, 0);
            }
            finish(cb_as, 1);
        }
        cb_pop_front(cb_akx, 1);
    }
    cb_pop_front(cb_ak, nb * 8);
    const uint32_t n = nb * T;  // S^T_{b,t}, P^T, P: tile b T + t
    cb_wait_front(cb_as, n);
    for (uint32_t t = 0; t < T; t++) {  // m_t (reduce_tile folds into Dest: start it at -1e30)
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, -1e30f);
        reconfig_data_format(cb_as, cb_ones);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_as, cb_ones, cb_am);
        for (uint32_t b = 0; b < nb; b++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_as, cb_ones, b * T + t, 0, 0);
        }
        reduce_uninit();
        finish(cb_am, 1);
    }
    cb_wait_front(cb_am, T);
    for (uint32_t j = 0; j < n; j++) {  // P^T = exp(S^T - m)
        tile_regs_acquire();
        reconfig_data_format(cb_as, cb_am);
        sub_bcast_rows_init(cb_as, cb_am);
        sub_tiles_bcast_rows(cb_as, cb_am, j, j % T, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_ap, 1);    // bf16, for the transpose and P V
        pack_n(cb_ap32, 1);  // fp32, for l
        tile_regs_release();
    }
    cb_pop_front(cb_as, n);
    cb_wait_front(cb_ap, n);
    cb_wait_front(cb_ap32, n);
    for (uint32_t t = 0; t < T; t++) {  // l_t: the column sum of P^T (from 0)
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        reconfig_data_format(cb_ap32, cb_ones);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_ap32, cb_ones, cb_al);
        for (uint32_t b = 0; b < nb; b++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_ap32, cb_ones, b * T + t, 0, 0);
        }
        reduce_uninit();
        finish(cb_al, 1);
    }
    cb_pop_front(cb_ap32, n);
    reconfig_data_format_srca(cb_ap);  // P = transpose(P^T)
    transpose_init(cb_ap);
    for (uint32_t j = 0; j < n; j++) {
        tile_regs_acquire();
        transpose_tile(cb_ap, j, 0);
        finish(cb_apt, 1);
    }
    cb_pop_front(cb_ap, n);
    cb_wait_front(cb_apt, n);
    cb_wait_front(cb_av, nb * 8);  // O_t = sum_b P_{b,t} V_b
    for (uint32_t t = 0; t < T; t++) {
        tile_regs_acquire();
        matmul_start(cb_apt, cb_av);
        for (uint32_t d = 0; d < 8; d++) {
            for (uint32_t b = 0; b < nb; b++) {
                matmul_tiles(cb_apt, cb_av, b * T + t, b * 8 + d, d);
            }
        }
        finish_each(cb_aout, 8);
    }
    cb_pop_front(cb_apt, n);
    cb_pop_front(cb_av, nb * 8);
    cb_wait_front(cb_al, T);  // then m and l (row 0: a value per q row)
    tile_regs_acquire();
    for (uint32_t t = 0; t < T; t++) {
        load(cb_am, t, t);
        load(cb_al, t, T + t);
    }
    cb_pop_front(cb_am, T);
    cb_pop_front(cb_al, T);
    finish_each(cb_aout, 2 * T);
}

// the combine of one (KV head, column tile, dims tile) over P partials, combine_parts()'s way: the statistics as the
// rows of ns = ceil(P / 32) tiles (partial i: tile i / 32, row i % 32), M = the column max over them,
// A_s = exp(m_s - M),
// 1 / L = 1 / the column sum of A_s l_s, W_s = A_s / L; then O = sum_i D_i O_i in groups of 16 partials (D_i the
// identity times W's row of partial i; the O tiles streamed), each group's sum added to the earlier ones' in fp32
// (cb_cacc), times the gate
void combine_m(uint32_t P) {
    const uint32_t ns = (P + 31) / 32, groups = (P + 15) / 16;
    cb_wait_front(cb_cm, ns);
    cb_wait_front(cb_cl, ns);
    cb_wait_front(cb_cid, 1);
    tile_regs_acquire();  // M
    fill_tile_init();
    fill_tile(0, -1e30f);
    reconfig_data_format(cb_cm, cb_ones);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_cm, cb_ones, cb_cM);
    for (uint32_t s = 0; s < ns; s++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_cm, cb_ones, s, 0, 0);
    }
    reduce_uninit();
    finish(cb_cM, 1);
    cb_wait_front(cb_cM, 1);
    for (uint32_t s = 0; s < ns; s++) {  // A_s
        tile_regs_acquire();
        reconfig_data_format(cb_cm, cb_cM);
        sub_bcast_rows_init(cb_cm, cb_cM);
        sub_tiles_bcast_rows(cb_cm, cb_cM, s, 0, 0);
        exp_tile_init();
        exp_tile(0);
        finish(cb_cA, 1);
    }
    cb_pop_front(cb_cm, ns);
    cb_pop_front(cb_cM, 1);
    cb_wait_front(cb_cA, ns);
    for (uint32_t s = 0; s < ns; s++) {  // A_s l_s
        tile_regs_acquire();
        reconfig_data_format(cb_cA, cb_cl);
        mul_init(cb_cA, cb_cl);
        mul_tiles(cb_cA, cb_cl, s, s, 0);
        finish(cb_cAL, 1);
    }
    cb_pop_front(cb_cl, ns);
    cb_wait_front(cb_cAL, ns);
    tile_regs_acquire();  // 1 / L
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_cAL, cb_ones);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_cAL, cb_ones, cb_crl);
    for (uint32_t s = 0; s < ns; s++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_cAL, cb_ones, s, 0, 0);
    }
    reduce_uninit();
    recip_tile_init();
    recip_tile(0);
    cb_pop_front(cb_cAL, ns);
    finish(cb_crl, 1);
    cb_wait_front(cb_crl, 1);
    for (uint32_t s = 0; s < ns; s++) {  // W_s
        tile_regs_acquire();
        reconfig_data_format(cb_cA, cb_crl);
        mul_bcast_rows_init(cb_cA, cb_crl);
        mul_tiles_bcast_rows(cb_cA, cb_crl, s, 0, 0);
        finish(cb_cW, 1);
    }
    cb_pop_front(cb_cA, ns);
    cb_pop_front(cb_crl, 1);
    cb_wait_front(cb_cW, ns);
    for (uint32_t q = 0; q < groups; q++) {
        const uint32_t n = P - 16 * q < 16 ? P - 16 * q : 16;
        reconfig_data_format(cb_cid, cb_cW);  // D_i
        mul_bcast_rows_init(cb_cid, cb_cW);
        for (uint32_t p = 16 * q; p < 16 * q + n; p++) {
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_cid, cb_cW, 0, p / 32, 0, p % 32);
            finish(cb_cdiag, 1);
        }
        cb_wait_front(cb_cdiag, n);
        tile_regs_acquire();  // the group's sum_i D_i O_i, the O tiles one at a time
        matmul_start(cb_cdiag, cb_co);
        for (uint32_t p = 0; p < n; p++) {
            cb_wait_front(cb_co, 1);
            matmul_tiles(cb_cdiag, cb_co, p, 0, 0);
            cb_pop_front(cb_co, 1);
        }
        cb_pop_front(cb_cdiag, n);
        if (q) {  // + the earlier groups'
            cb_wait_front(cb_cacc, 1);
            load(cb_cacc, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(cb_cacc, 1);
        }
        if (q + 1 < groups) {
            finish(cb_cacc, 1);
            continue;
        }
        cb_wait_front(cb_cg, 1);  // times the gate
        load(cb_cg, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        cb_pop_front(cb_cg, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_n(cb_cout16, 1);
        pack_n(cb_cchk, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_cW, ns);
    cb_pop_front(cb_cid, 1);
}
#endif

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    uint32_t i = 0;
    const uint32_t ap = get_arg_val<uint32_t>(i++), ac = get_arg_val<uint32_t>(i++);
    const uint32_t o_tile = get_arg_val<uint32_t>(i++);
    const uint32_t h_item = get_arg_val<uint32_t>(i++), h_tile = get_arg_val<uint32_t>(i++);
    const uint32_t r_item = get_arg_val<uint32_t>(i++), s_tile = get_arg_val<uint32_t>(i++);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_aout);
#if M_MAX > 1
    cb_wait_front(cb_ones, 3);  // M, the union's blocks, the layout's rows: the reader's words in cb 2, tile 2
    const uint32_t M = read_tile_value(cb_ones, 2, 0), nblk = read_tile_value(cb_ones, 2, 1);
    const uint32_t mp = read_tile_value(cb_ones, 2, 3), T = (12 * M + 31) / 32, P = (nblk + 3) / 4;
    Weights w{cb_w, batch}, rw{tail::cb_rw, batch};
    if (ap != IDLE) {
        q_transpose(T);
        for (uint32_t p = ap % NAP; p < P; p += NAP) {  // the core's partials, all of 4 blocks but the last one
            partial_m(nblk - 4 * p < 4 ? nblk - 4 * p : 4, T);
        }
        cb_pop_front(cb_aqt, 9 * T);
    }
    if (ac != IDLE && ac / 8 % T_MAX < T) {
        combine_m(P);
    }
    tail::oproj(o_tile, batch);
    hc::compute_m(h_item, h_tile, w, mp);
    tail::ffn(r_item, s_tile, w, rw);
#else
    cb_wait_front(cb_ones, 1);
    Weights w{cb_w, batch}, rw{tail::cb_rw, batch};
    if (ap != IDLE) {
        partial(ap % NPARTS == NPARTS - 1);
    }
    if (ac != IDLE) {
        combine_parts();
    }
    tail::oproj(o_tile, batch);
    hc::compute(h_item, h_tile, w);
    tail::ffn(r_item, s_tile, w, rw);
#endif
}
