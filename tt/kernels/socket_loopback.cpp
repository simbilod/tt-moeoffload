// The hand-off floor: a persistent loopback on one data-movement core. Each iteration waits for one page from
// the host's H2D socket, NOC-writes it into the host's D2H FIFO (pinned RAM, over PCIe) and notifies; nothing else.
// H2D_DEVICE_PULL: the page sits in pinned host RAM and the core reads it into its CB first (else HOST_PUSH: the host
// already wrote it into this core's L1 FIFO).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

// D2H: one page from L1 to host RAM in NOC_MAX_BURST_SIZE chunks. Derived from tt-metal's write_page_to_pcie
// (ttnn/.../send_async_d2h/device/kernels/d2h_sender_reader.cpp), Copyright 2026 Tenstorrent USA, Inc., Apache
// License 2.0 (LICENSES/tt-metal-Apache-2.0.txt); changed: renamed, the write state set up here.
inline void noc_write_page_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

void kernel_main() {
    constexpr uint32_t h2d_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t d2h_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(3);
    constexpr uint32_t cb_page = 0;

    SocketReceiverInterface in = create_receiver_socket_interface(h2d_config_addr);
    set_receiver_socket_page_size(in, page_size);
    SocketSenderInterface out = create_sender_socket_interface(d2h_config_addr);
    set_sender_socket_page_size(out, page_size);
    const uint64_t out_base = (static_cast<uint64_t>(out.d2h.data_addr_hi) << 32) | out.downstream_fifo_addr;
#ifdef H2D_DEVICE_PULL
    const uint64_t in_base = (static_cast<uint64_t>(in.h2d.data_addr_hi) << 32) | in.h2d.data_addr_lo;
    const uint32_t local = get_write_ptr(cb_page);
#endif

    for (uint32_t i = 0; i < num_iterations; i++) {
        socket_wait_for_pages(in, 1);
#ifdef H2D_DEVICE_PULL
        const uint64_t from = in_base + in.read_ptr - in.fifo_addr;
        for (uint32_t done = 0; done < page_size;) {   // as fused/link.h's recv: bursts of at most NOC_MAX_BURST_SIZE
            const uint32_t chunk = page_size - done > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : page_size - done;
            noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                NOC_INDEX, in.h2d.pcie_xy_enc, from + done, local + done, chunk);
            done += chunk;
        }
        noc_async_read_barrier();
        const uint32_t src = local;
#else
        const uint32_t src = in.read_ptr;
#endif
        socket_reserve_pages(out, 1);
        noc_write_page_chunked(out.d2h.pcie_xy_enc, src, out_base + out.write_ptr, page_size);
        socket_push_pages(out, 1);
        socket_notify_receiver(out);
        socket_pop_pages(in, 1);
        socket_notify_sender(in);
    }

    socket_barrier(out);
#ifdef H2D_DEVICE_PULL
    noc_async_read_clear_pcie_state(NOC_INDEX, read_cmd_buf);
#endif
    update_socket_config(in);
    update_socket_config(out);
    noc_async_write_barrier();
}
