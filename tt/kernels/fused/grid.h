// Shared machinery for persistent, phase-sequenced programs on the whole worker grid (tt/fused/grid.py builds the
// arguments). A program is a run of phases; between two phases every core arrives at a barrier and waits for it.
//
// Barrier: the WRITER of each core flushes its writes and increments semaphore 0 on the
// coordinator core (arrive(n)); the coordinator's READER waits for every arrival, optionally multicasts a region of
// its arena to every core, then sets its semaphore 2 to the barrier's number and multicasts it into semaphore 1 of
// every core; every reader waits for semaphore 1 to reach the number (wait(n)). Numbers only grow (an epoch), so a
// program can cross any number of barriers and nothing is ever reset. The reader can prefetch the next phase's
// weights before it waits.
//
// Gather (send() + the barrier's broadcast): a phase's producers write their pieces into the coordinator's arena
// (unicast), and the coordinator multicasts the whole region once, so every core finds the phase's outputs at the
// same arena address. Multicasts are serialized on the NOC (path reservation): 110 cores each multicasting their
// piece cost ~1 us per multicast, hundreds of us per phase; one multicast of the whole vector costs ~1-2 us.
//
// Multicast rules found on this card (2026-10-23):
//   - NOC1 runs the other way: its rectangle is given from the far corner (x1, y1) to (x0, y0); rect_addr() picks
//     the order from noc_index (here the reader is on NOC0, the writer on NOC1). The wrong order hangs.
//   - Only the rectangle holding the sender takes the loopback form (the worker grid's NOC columns have a gap, so
//     the grid is two rectangles).
//   - A core with nothing to do in a phase must not arrive at later barriers early: arrive(n) first waits for
//     barrier n - 1 to be released.
//
// Grid arguments, a block of runtime args starting at index i (the same block for reader and writer):
//   is coordinator, coordinator NOC x, NOC y, arrivals expected per barrier, the rectangle holding this core,
//   rectangle count, then per rectangle NOC x0, y0, x1, y1, core count.
// Semaphores 0, 1, 2 must exist on every core, initial value 0.
//
// Profile (with -DGRID_PROFILE=<cb>; -DGRID_PROFILE_N=<N>, default 32, and -DGRID_PROFILE_OFF=<bytes into the CB>,
// default 0): each core's wall clock at every barrier n < N, in 16 N bytes of that CB: the reader's entry into wait(n)
// at word n and its release at word N + n, the writer's arrive(n) at word 2 N + n, and the reader's marks (mark(k),
// the kernels' own events: the program's start and end, the host link) at word 3 N + k; profile_dump() writes them to
// card memory, 16 N bytes per core. One core's clocks are comparable, not two cores'.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#ifdef GRID_PROFILE
#include "api/tensor/tensor_accessor.h"
#ifndef GRID_PROFILE_N
#define GRID_PROFILE_N 32
#endif
#ifndef GRID_PROFILE_OFF
#define GRID_PROFILE_OFF 0
#endif
#endif

namespace fused {

struct Grid {
    uint32_t coordinator, cx, cy, expected, own_rect, nrects, rect_arg;

    static Grid from_args(uint32_t i) {
        Grid g;
        g.coordinator = get_arg_val<uint32_t>(i);
        g.cx = get_arg_val<uint32_t>(i + 1);
        g.cy = get_arg_val<uint32_t>(i + 2);
        g.expected = get_arg_val<uint32_t>(i + 3);
        g.own_rect = get_arg_val<uint32_t>(i + 4);
        g.nrects = get_arg_val<uint32_t>(i + 5);
        g.rect_arg = i + 6;
        return g;
    }
    static constexpr uint32_t args(uint32_t nrects) { return 6 + 5 * nrects; }

    uint64_t rect_addr(uint32_t r, uint32_t l1) const {
        const uint32_t a = rect_arg + 5 * r;
        const uint32_t x0 = get_arg_val<uint32_t>(a), y0 = get_arg_val<uint32_t>(a + 1);
        const uint32_t x1 = get_arg_val<uint32_t>(a + 2), y1 = get_arg_val<uint32_t>(a + 3);
        return noc_index == 1 ? get_noc_multicast_addr(x1, y1, x0, y0, l1) : get_noc_multicast_addr(x0, y0, x1, y1, l1);
    }
    uint32_t rect_count(uint32_t r) const { return get_arg_val<uint32_t>(rect_arg + 5 * r + 4); }

    // bytes at local L1 src to L1 dst on every core (this one too); complete after noc_async_write_barrier()
    void mcast(uint32_t src, uint32_t dst, uint32_t bytes) const {
        for (uint32_t r = 0; r < nrects; r++) {
            if (r == own_rect) {
                noc_async_write_multicast_loopback_src(src, rect_addr(r, dst), bytes, rect_count(r));
            } else {
                noc_async_write_multicast(src, rect_addr(r, dst), bytes, rect_count(r));
            }
        }
    }

    // writer: bytes at local L1 src into the coordinator's L1 at dst (its arena); complete at arrive()
    void send(uint32_t src, uint32_t dst, uint32_t bytes) const { noc_async_write(src, get_noc_addr(cx, cy, dst), bytes); }

    // writer: this core's outputs for the phase before barrier n are written; count it in
    void arrive(uint32_t n) const {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), n - 1);
        noc_async_write_barrier();
        stamp(2, n);
        noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(0)), 1);
        noc_async_atomic_barrier();
    }

    // reader: wait for barrier n (1, 2, ...); the coordinator first broadcasts bytes of its L1 at region
    void wait(uint32_t n, uint32_t region = 0, uint32_t bytes = 0) const { wait(n, region, bytes, [] {}); }

    // the same, the coordinator running before() once every core has arrived and before it broadcasts (e.g. a round
    // trip to the host that fills region: GLM's mHC parameters, glm_hc_reader.cpp)
    template <typename F>
    void wait(uint32_t n, uint32_t region, uint32_t bytes, F before) const {
        stamp(0, n);
        if (coordinator) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), expected * n);
            before();
            if (bytes) {
                mcast(region, region, bytes);
                noc_async_write_barrier();
            }
            const uint32_t value = get_semaphore(2);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(value), n);
            for (uint32_t r = 0; r < nrects; r++) {
                if (r == own_rect) {
                    noc_semaphore_set_multicast_loopback_src(value, rect_addr(r, get_semaphore(1)), rect_count(r));
                } else {
                    noc_semaphore_set_multicast(value, rect_addr(r, get_semaphore(1)), rect_count(r));
                }
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), n);
        stamp(1, n);
    }

    // group 0 the reader's wait entries, 1 its releases, 2 the writer's arrivals, 3 the reader's marks
    static void stamp(uint32_t group, uint32_t n) {
#ifdef GRID_PROFILE
        if (n < GRID_PROFILE_N) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(GRID_PROFILE) + GRID_PROFILE_OFF)
                [group * GRID_PROFILE_N + n] = get_timestamp_32b();
        }
#endif
    }

    // reader: this core's clock at its own event k (the profile's word 3 N + k; nothing without GRID_PROFILE)
    static void mark(uint32_t k) { stamp(3, k); }
};

#ifdef GRID_PROFILE
// the reader's words (0..2 N - 1 and its marks 3 N..4 N - 1) or the writer's (2 N..3 N - 1) to slot `slot` (16 N
// bytes) of the row-major tensor at addr (128 slots in one page)
template <typename Args>
inline void profile_dump(const Args& args, uint32_t addr, uint32_t slot, bool reader) {
    constexpr uint32_t N = GRID_PROFILE_N, bytes = 16 * N;
    const auto t = TensorAccessor(args, addr, bytes * 128);
    const uint32_t src = get_write_ptr(GRID_PROFILE) + GRID_PROFILE_OFF, dst = slot * bytes;
    if (reader) {
        noc_async_write(src, t.get_noc_addr(0, dst), 8 * N);
        noc_async_write(src + 12 * N, t.get_noc_addr(0, dst + 12 * N), 4 * N);
    } else {
        noc_async_write(src + 8 * N, t.get_noc_addr(0, dst + 8 * N), 4 * N);
    }
    noc_async_write_barrier();
}
#endif

// a local L1 -> L1 copy through the NOC (the data movement engines copy far faster than the RISC-V's loads/stores)
inline void l1_copy(uint32_t src, uint32_t dst, uint32_t bytes) { noc_async_read(get_noc_addr(src), dst, bytes); }

}  // namespace fused
