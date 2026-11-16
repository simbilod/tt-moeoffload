// GLM's hand-off, the receive half: wait for the host's reply page on the H2D socket (DEVICE_PULL: the
// page sits in pinned host RAM) and read it into this core's L1 (fused/link.h's Link::recv as a program of its own).
// Compile args: the H2D socket's config address, the page's bytes. Runtime arg 0: the destination's L1 address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    constexpr uint32_t h2d_config = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    const uint32_t dst = get_arg_val<uint32_t>(0);

    SocketReceiverInterface in = create_receiver_socket_interface(h2d_config);
    set_receiver_socket_page_size(in, page);
    const uint64_t base = (static_cast<uint64_t>(in.h2d.data_addr_hi) << 32) | in.h2d.data_addr_lo;
    socket_wait_for_pages(in, 1);
    const uint64_t from = base + in.read_ptr - in.fifo_addr;
    for (uint32_t done = 0; done < page;) {
        const uint32_t chunk = page - done > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : page - done;
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, in.h2d.pcie_xy_enc, from + done, dst + done, chunk);
        done += chunk;
    }
    noc_async_read_barrier();
    noc_async_read_clear_pcie_state(NOC_INDEX, read_cmd_buf);
    socket_pop_pages(in, 1);
    socket_notify_sender(in);
    update_socket_config(in);     // the read pointer, for the next program on this socket
    noc_async_write_barrier();    // the ack (notify_sender) and the config written before the program ends
}
