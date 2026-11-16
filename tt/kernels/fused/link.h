// The host link of the fused layer programs, for the coordinator's reader (sockets, tt/kernels/socket_loopback.cpp):
// a D2H page out to pinned host RAM (the experts' input and the router's logits), then an H2D page back (DEVICE_PULL:
// the core reads it from pinned host RAM; the routed experts' sum), each one page per exchange. The sockets are
// created by the host on the coordinator core; their config buffers' L1 addresses come in as compile-time defines.
// Needs the card's IOMMU group in DMA-FQ (tools/iommu): with identity, the socket FIFOs cannot be pinned.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

namespace fused {

struct Link {
    SocketReceiverInterface in;
    SocketSenderInterface out;
    uint64_t in_base, out_base;
    uint32_t in_page, out_page;

    void open(uint32_t h2d_config, uint32_t d2h_config, uint32_t in_bytes, uint32_t out_bytes) {
        in = create_receiver_socket_interface(h2d_config);
        set_receiver_socket_page_size(in, in_bytes);
        out = create_sender_socket_interface(d2h_config);
        set_sender_socket_page_size(out, out_bytes);
        in_base = (static_cast<uint64_t>(in.h2d.data_addr_hi) << 32) | in.h2d.data_addr_lo;
        out_base = (static_cast<uint64_t>(out.d2h.data_addr_hi) << 32) | out.downstream_fifo_addr;
        in_page = in_bytes;
        out_page = out_bytes;
    }

    // out_page bytes at L1 src to the host, then the host's reply (in_page bytes) into L1 dst
    void exchange(uint32_t src, uint32_t dst) {
        send(src);
        recv(dst);
    }

    // the two halves of exchange(), for work between them (the host computes meanwhile)
    void send(uint32_t src) {
        socket_reserve_pages(out, 1);
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        uint64_t to = out_base + out.write_ptr;
        for (uint32_t done = 0; done < out_page;) {
            const uint32_t chunk = out_page - done > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : out_page - done;
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_INDEX, src + done, out.d2h.pcie_xy_enc, to + done, chunk, 1);
            done += chunk;
        }
        socket_push_pages(out, 1);
        socket_notify_receiver(out);
        // the PCIe routing bits are sticky per command buffer: clear them so later ordinary writes and reads go to
        // the NOC addresses they name
        while (!noc_cmd_buf_ready(NOC_INDEX, write_cmd_buf));
        NOC_CMD_BUF_WRITE_REG(NOC_INDEX, write_cmd_buf, NOC_RET_ADDR_MID, 0);
    }

    void recv(uint32_t dst) {
        socket_wait_for_pages(in, 1);
        const uint64_t from = in_base + in.read_ptr - in.fifo_addr;
        for (uint32_t done = 0; done < in_page;) {
            const uint32_t chunk = in_page - done > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : in_page - done;
            noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                NOC_INDEX, in.h2d.pcie_xy_enc, from + done, dst + done, chunk);
            done += chunk;
        }
        noc_async_read_barrier();
        noc_async_read_clear_pcie_state(NOC_INDEX, read_cmd_buf);
        socket_pop_pages(in, 1);
        socket_notify_sender(in);
    }

    void close() {
        socket_barrier(out);
        update_socket_config(in);
        update_socket_config(out);
        noc_async_write_barrier();
    }
};

}  // namespace fused
