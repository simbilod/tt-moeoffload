// GLM's hand-off, the send half: one page from this core's L1 to the host over the D2H socket
// (fused/link.h's Link::send as a program of its own, so the card's queue runs other work while the host computes,
// and glm_xrecv.cpp takes the reply later). Compile args: the D2H socket's config address, the page's bytes.
// Runtime arg 0: the page's L1 address on this core.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    constexpr uint32_t d2h_config = get_compile_time_arg_val(0);
    constexpr uint32_t page = get_compile_time_arg_val(1);
    const uint32_t src = get_arg_val<uint32_t>(0);

    SocketSenderInterface out = create_sender_socket_interface(d2h_config);
    set_sender_socket_page_size(out, page);
    const uint64_t base = (static_cast<uint64_t>(out.d2h.data_addr_hi) << 32) | out.downstream_fifo_addr;
    socket_reserve_pages(out, 1);
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    const uint64_t to = base + out.write_ptr;
    for (uint32_t done = 0; done < page;) {
        const uint32_t chunk = page - done > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : page - done;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src + done, out.d2h.pcie_xy_enc, to + done, chunk, 1);
        done += chunk;
    }
    socket_push_pages(out, 1);
    socket_notify_receiver(out);
    // the PCIe routing bits are sticky per command buffer: clear them for later ordinary writes (link.h)
    while (!noc_cmd_buf_ready(NOC_INDEX, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(NOC_INDEX, write_cmd_buf, NOC_RET_ADDR_MID, 0);
    socket_barrier(out);          // the host has taken the page: the next send may reuse this L1
    update_socket_config(out);    // the write pointer, for the next program on this socket
    noc_async_write_barrier();
}
