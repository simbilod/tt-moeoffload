// GLM's fused FFN half, reader (compute glm_fsh_compute.cpp, writer glm_fsh_writer.cpp, host
// tt/glm/g3_fsh.py): hn -> the FFN's output. MoE layers (FSH_LANE): the routed experts on the CPU lane while the card
// runs the shared expert (FF 2048); the dense layers 0-2: their FFN (FF 12288) alone. Two phases on the whole grid
// (fused/grid.h):
//
// 0   (FSH_LANE) the coordinator sends hn (bf16 [4096], its D2H page) to the host, which routes (fp32, cpulane
//     cl_glm_route) and runs the lane while the card works
// R1  units j = u0 .. u0 + nu - 1: the gate and up column tiles j (W_gu's tiles 2 j, 2 j + 1, n-major pages 128 n + k),
//     h_j = silu(min(g, 10)) * clamp(u, -10, 10) (llama.cpp's swiglu_clamp_shexp, every build_ffn) as bf16 to region H
// R2  down (FF -> 4096, KD = FF / 32 tiles of K) split by K in S parts (glm_fmo_reader.cpp's E2: item i = (n = i / S,
//     part i % S), pages KD n + (KD / S) part ..), the partials to region P; at barrier 2 (FSH_LANE) the coordinator
//     takes the lane's sum (fp32 [4096], H2D) into region RT, after P, and the barrier broadcasts both
// SUM each core adds its output tiles' partials (and the routed sum) on this RISC-V's FPU into out (row-major fp32
//     [4096]: the mHC site's input)
//
// x travels as mgemv's ring; R2's first pages are read before barrier 1.
// PF  (FSH_LANE) after R2's weights, while the lane works: the next layer's mixer's first pages for this core (up to
//     4 page ranges of its phase A's weights, in its order) into this core's L1 prefetch buffer (glm_fkda_reader.cpp,
//     glm_fmq_reader.cpp read them from there)
// Runtime args: grid.h's block, hn, W_gu, W_d, out addresses, R1's first unit and count, R2's first item and count,
// the first output tile this core sums and their count, the arena offsets of X (hn), H, P (RT = P + S * 16 KB), then
// the prefetch: the buffer's L1 address, the source weights' address (0: none), the range count, (first page, count).
// Compile args: batch, depth, S, KD, then a DRAM accessor. Defines: FSH_LANE with H2D_CONFIG, D2H_CONFIG.
//
// M rows (-DM_MAX = 2..4; M_MAX = 1 compiles the one-row source): the runtime args after P's offset gain the
// step tensor and RT's offset; hn, H and out as fused/hc_dm.h's mp-row blocks, R2's partials sent by owner (the
// writer's; fused::ksplit_sum_m), the host's page hn's blocks (M_MAX rows of room) and its reply the routed rows'
// blocks (experts.cpp's cl_glm_exchange_m), broadcast at barrier 2 into RT, the sums' base.
#include <cstdint>
#ifndef M_MAX
#define M_MAX 1
#endif

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "fused/dm.h"
#include "fused/ksum.h"
#include "fused/fpu.h"
#include "fused/link.h"
#if M_MAX > 1
#include "fused/hc_dm.h"
#endif

constexpr uint32_t cb_w = 0, cb_x = 1, cb_arena = 30, cb_stage = 31;
constexpr uint32_t W_PAGE = 1088, KA = 128, NT = 128, D = 4096;

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
inline volatile tt_l1_ptr float* f32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr float*>(a); }

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t KD = get_compile_time_arg_val(3);
    constexpr uint32_t KP = KD / S;
    constexpr auto d_args = TensorAccessorArgs<4>();
    const fused::Grid grid = fused::Grid::from_args(0);
    uint32_t i = fused::Grid::args(grid.nrects);
    const auto hn = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), D * 2);
    const auto wgu = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto wd = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), W_PAGE);
    const auto out = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), NT * 128);
    const uint32_t u0 = get_arg_val<uint32_t>(i++), nu = get_arg_val<uint32_t>(i++);
    const uint32_t it0 = get_arg_val<uint32_t>(i++), n_it = get_arg_val<uint32_t>(i++);
    const uint32_t s0 = get_arg_val<uint32_t>(i++), n_s = get_arg_val<uint32_t>(i++);
    const uint32_t arena = get_write_ptr(cb_arena);
    const uint32_t X = arena + get_arg_val<uint32_t>(i++), H = arena + get_arg_val<uint32_t>(i++);
#if M_MAX > 1
    const uint32_t P = arena + get_arg_val<uint32_t>(i++);
    const auto step = TensorAccessor(d_args, get_arg_val<uint32_t>(i++), 64);
    const uint32_t RT = arena + get_arg_val<uint32_t>(i++);
    noc_async_read(step.get_noc_addr(0), get_write_ptr(cb_stage), 32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t mp = w32(get_write_ptr(cb_stage))[4];
    auto xr = [&](uint32_t row, uint32_t k0) { fused::x_ring_m_issue(cb_x, row, k0, batch, mp); };
    auto xpush = [&] { cb_push_back(cb_x, batch); };
#else
    const uint32_t P = arena + get_arg_val<uint32_t>(i++), RT = P + S * NT * 128;
    auto xr = [&](uint32_t row, uint32_t k0) { fused::x_ring(cb_x, row, k0, batch); };
    auto xpush = [] {};
#endif
    const uint32_t pf = get_arg_val<uint32_t>(i++), pf_src = get_arg_val<uint32_t>(i++);
    const uint32_t n_ranges = get_arg_val<uint32_t>(i++), ra = i;

    fused::Stream<decltype(wd)> ws{cb_w, batch, W_PAGE};   // the weight stream (the open batch persists)

    // ---- 0: hn, and to the host from the coordinator
#if M_MAX > 1
    noc_async_read(hn.get_noc_addr(0), X, mp * D * 2);
#else
    noc_async_read(hn.get_noc_addr(0), X, D * 2);
#endif
    noc_async_read_barrier();
#ifdef FSH_LANE
    fused::Link link;
    if (grid.coordinator) {
        link.open(H2D_CONFIG, D2H_CONFIG, M_MAX * D * 4, M_MAX * D * 2);
        link.send(X);
    }
#endif

    // ---- R1
    for (uint32_t u = u0; u < u0 + nu; u++) {
        for (uint32_t p = 0; p < 2 * KA; p += batch) {
            xr(X, p % KA);
            ws.read(wgu, u * 2 * KA + p, batch);
            xpush();
        }
    }
    const uint32_t pages = n_it * KP, page0 = it0 * KP;
    const uint32_t pre = pages < depth * batch ? pages : depth * batch;
    ws.read(wd, page0, pre);
#if M_MAX > 1
    grid.wait(1, H, KD * mp * 64);
#else
    grid.wait(1, H, KD * 64);
#endif

    // ---- R2
    for (uint32_t p = 0; p < pages; p += batch) {
        xr(H, (page0 + p) % KD);
        if (p >= pre) {
            ws.read(wd, page0 + p, batch);
        } else if (M_MAX > 1) {
            noc_async_read_barrier();
        }
        xpush();
    }

    // ---- PF: issued now, landed by the end of the program (the lane takes longer than R2 and the sum)
    if (pf_src) {
        const auto src = TensorAccessor(d_args, pf_src, W_PAGE);
        uint32_t dst = pf;
        for (uint32_t r = 0; r < n_ranges; r++) {
            const uint32_t first = get_arg_val<uint32_t>(ra + 2 * r), count = get_arg_val<uint32_t>(ra + 2 * r + 1);
            for (uint32_t q = first; q < first + count; q++, dst += W_PAGE) noc_async_read(src.get_noc_addr(q), dst, W_PAGE);
        }
    }
#if M_MAX > 1
#ifdef FSH_LANE
    grid.wait(2, RT, mp * D * 4, [&] {   // the routed rows, broadcast; the partials came by owner
        link.recv(RT);
        link.close();
    });
    fused::ksplit_sum_m(out, P, S, s0, n_s, mp * 128, get_write_ptr(cb_stage), RT);
#else
    grid.wait(2);
    fused::ksplit_sum_m(out, P, S, s0, n_s, mp * 128, get_write_ptr(cb_stage));
#endif
    noc_async_read_barrier();   // the prefetch landed
    return;
#endif
#ifdef FSH_LANE
    grid.wait(2, P, (S + 1) * NT * 128, [&] {
        link.recv(RT);
        link.close();
    });
#else
    grid.wait(2, P, S * NT * 128);
#endif

    // ---- SUM
#ifdef FSH_LANE
    fused::ksplit_sum(out, P, S, NT, s0, n_s, RT, get_write_ptr(cb_stage));   // from the routed sum
#else
    fused::ksplit_sum(out, P, S, NT, s0, n_s, 0, get_write_ptr(cb_stage));
#endif
    noc_async_read_barrier();   // the prefetch landed
}
