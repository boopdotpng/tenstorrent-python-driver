// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch kernel
//  - receives data in pages from prefetch kernel into the dispatch buffer ring buffer
//  - processes commands with embedded data from the dispatch buffer to write/sync/etc w/ destination
//  - sync w/ prefetcher is via 2 semaphores, page_ready, page_done
//  - page size must be a power of 2
//  - # blocks must evenly divide the dispatch buffer size
//  - dispatch buffer base must be page size aligned

#include "cq_fixed_config.hpp"
#include "api/dataflow/dataflow_api.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "cq_commands.hpp"
#include "cq_common.hpp"
#include "cq_relay.hpp"

// The command queue write interface controls writes to the completion region, host owns the completion region read
// interface Data requests from device and event states are written to the completion region

CQWriteInterface cq_write_interface;

// Keep only derived constants and encoded values.
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t dispatch_s_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_SUBORDINATE_NOC_X, DOWNSTREAM_SUBORDINATE_NOC_Y));
// BH: PCIe coords are absolute, not subject to NOC mirroring.
constexpr uint64_t pcie_noc_xy = uint64_t(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));
constexpr uint32_t dispatch_cb_page_size = 1 << DISPATCH_CB_LOG_PAGE_SIZE;

constexpr uint32_t completion_queue_end_addr = COMPLETION_QUEUE_BASE_ADDR + COMPLETION_QUEUE_SIZE;
constexpr uint32_t completion_queue_page_size = dispatch_cb_page_size;
constexpr uint32_t completion_queue_size_16B = COMPLETION_QUEUE_SIZE >> 4;
constexpr uint32_t completion_queue_page_size_16B = completion_queue_page_size >> 4;
constexpr uint32_t completion_queue_end_addr_16B = completion_queue_end_addr >> 4;
constexpr uint32_t completion_queue_base_addr_16B = COMPLETION_QUEUE_BASE_ADDR >> 4;
constexpr uint32_t dispatch_cb_size = dispatch_cb_page_size * DISPATCH_CB_PAGES;
constexpr uint32_t dispatch_cb_end = DISPATCH_CB_BASE + dispatch_cb_size;
constexpr uint32_t downstream_cb_end = DOWNSTREAM_CB_BASE + DOWNSTREAM_CB_SIZE;

// Break buffer into blocks, 1/n of the total (dividing equally)
// Do bookkeeping (release, etc) based on blocks
// Note: due to the current method of release pages, up to 1 block of pages
// may be unavailable to the prefetcher at any time
constexpr uint32_t dispatch_cb_pages_per_block = DISPATCH_CB_PAGES / DISPATCH_CB_BLOCKS;

static uint32_t cmd_ptr;   // walks through pages in cb cmd by cmd
static uint32_t downstream_cb_data_ptr = DOWNSTREAM_CB_BASE;

static uint32_t write_offset[CQ_DISPATCH_MAX_WRITE_OFFSETS];  // added to write address on non-host writes

using RelayClientType = CQRelayClient;

RelayClientType relay_client;

struct DispatchReleasePolicy {
    template <uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id>
    static FORCE_INLINE void release(uint32_t pages) {
        if constexpr (IS_H_VARIANT && !IS_D_VARIANT) {
            relay_client.template release_pages<noc_idx, noc_xy, sem_id>(pages);
        } else {
            uint32_t sem_addr = get_semaphore<fd_core_type>(sem_id);
            noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), pages, noc_idx);
        }
    }
};

using CBReaderType = CBReaderWithReleasePolicy<
    MY_DISPATCH_CB_SEM_ID,
    DISPATCH_CB_LOG_PAGE_SIZE,
    DISPATCH_CB_BLOCKS,
    UPSTREAM_NOC_INDEX,
    upstream_noc_xy,
    UPSTREAM_DISPATCH_CB_SEM_ID,
    dispatch_cb_pages_per_block,
    DISPATCH_CB_BASE,
    DispatchReleasePolicy>;

static CBReaderType dispatch_cb_reader;

constexpr uint32_t packed_write_max_multicast_sub_cmds =
    get_packed_write_max_multicast_sub_cmds(PACKED_WRITE_MAX_UNICAST_SUB_CMDS);
constexpr uint32_t max_write_packed_large_cmd =
    CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS * sizeof(CQDispatchWritePackedLargeSubCmd) / sizeof(uint32_t);
constexpr uint32_t max_write_packed_cmd =
    PACKED_WRITE_MAX_UNICAST_SUB_CMDS * sizeof(CQDispatchWritePackedUnicastSubCmd) / sizeof(uint32_t);
constexpr uint32_t l1_cache_elements =
    (max_write_packed_cmd > max_write_packed_large_cmd) ? max_write_packed_cmd : max_write_packed_large_cmd;
constexpr uint32_t l1_cache_elements_rounded =
    ((l1_cache_elements + l1_to_local_cache_copy_chunk - 1) / l1_to_local_cache_copy_chunk) *
    l1_to_local_cache_copy_chunk;

static uint32_t go_signal_noc_data[MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES];

FORCE_INLINE volatile uint32_t* get_cq_completion_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(DEV_COMPLETION_Q_RD_PTR);
}

FORCE_INLINE volatile uint32_t* get_cq_completion_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(DEV_COMPLETION_Q_WR_PTR);
}

FORCE_INLINE
void completion_queue_reserve_back(uint32_t num_pages) {
    // Transfer pages are aligned
    uint32_t data_size_16B = num_pages * completion_queue_page_size_16B;
    uint32_t completion_rd_ptr_and_toggle;
    uint32_t completion_rd_ptr;
    uint32_t completion_rd_toggle;
    uint32_t available_space;
    do {
        invalidate_l1_cache();
        completion_rd_ptr_and_toggle = *get_cq_completion_read_ptr();
        completion_rd_ptr = completion_rd_ptr_and_toggle & 0x7fffffff;
        completion_rd_toggle = completion_rd_ptr_and_toggle >> 31;
        // Toggles not equal means write ptr has wrapped but read ptr has not
        // so available space is distance from write ptr to read ptr
        // Toggles are equal means write ptr is ahead of read ptr
        // so available space is total space minus the distance from read to write ptr
        available_space =
            completion_rd_toggle != cq_write_interface.completion_fifo_wr_toggle
                ? completion_rd_ptr - cq_write_interface.completion_fifo_wr_ptr
                : (completion_queue_size_16B - (cq_write_interface.completion_fifo_wr_ptr - completion_rd_ptr));
    } while (data_size_16B > available_space);
}

// This fn expects NOC coords to be preprogrammed
// Note that this fn does not increment any counters
FORCE_INLINE
void notify_host_of_completion_queue_write_pointer() {
    uint32_t completion_queue_write_ptr_addr = COMMAND_QUEUE_BASE_ADDR + HOST_COMPLETION_Q_WR_PTR;
    uint32_t completion_wr_ptr_and_toggle =
        cq_write_interface.completion_fifo_wr_ptr | (cq_write_interface.completion_fifo_wr_toggle << 31);
    volatile tt_l1_ptr uint32_t* completion_wr_ptr_addr = get_cq_completion_write_ptr();
    completion_wr_ptr_addr[0] = completion_wr_ptr_and_toggle;
    cq_noc_async_write_with_state<CQ_NOC_SnDL>(DEV_COMPLETION_Q_WR_PTR, completion_queue_write_ptr_addr, 4);
}

FORCE_INLINE
void completion_queue_push_back(uint32_t num_pages) {
    // Transfer pages are aligned
    uint32_t push_size_16B = num_pages * completion_queue_page_size_16B;
    cq_write_interface.completion_fifo_wr_ptr += push_size_16B;

    if (cq_write_interface.completion_fifo_wr_ptr >= completion_queue_end_addr_16B) {
        cq_write_interface.completion_fifo_wr_ptr =
            cq_write_interface.completion_fifo_wr_ptr - completion_queue_end_addr_16B + completion_queue_base_addr_16B;
        // Flip the toggle
        cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
    }

    // Notify host of updated completion wr ptr
    notify_host_of_completion_queue_write_pointer();
}

void process_write_host_h() {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;

    uint32_t completion_write_ptr;
    // We will send the cmd back in the first X bytes, this makes the logic of reserving/pushing completion queue
    // pages much simpler since we are always sending writing full pages (except for last page)
    uint64_t wlength = cmd->write_linear_host.length;
    bool is_event = cmd->write_linear_host.is_event;
    uint32_t data_ptr = cmd_ptr;
    cq_noc_async_write_init_state<CQ_NOC_sNdl>(0, pcie_noc_xy, 0);
    constexpr uint32_t max_batch_size = ~(dispatch_cb_page_size - 1);
    while (wlength != 0) {
        uint32_t length = (wlength > max_batch_size) ? max_batch_size : static_cast<uint32_t>(wlength);
        wlength -= length;
        while (length != 0) {
            // Get a page if needed
            uint32_t available_data = dispatch_cb_reader.wait_for_available_data_and_release_old_pages(data_ptr);
            uint32_t xfer_size = (length > available_data) ? available_data : length;
            uint32_t npages = (xfer_size + completion_queue_page_size - 1) / completion_queue_page_size;
            completion_queue_reserve_back(npages);
            uint32_t completion_queue_write_addr = cq_write_interface.completion_fifo_wr_ptr << 4;
            // completion_queue_write_addr will never be equal to completion_queue_end_addr due to
            // completion_queue_push_back wrap logic so we don't need to handle this case explicitly to avoid 0 sized
            // transactions
            if (completion_queue_write_addr + xfer_size > completion_queue_end_addr) {
                uint32_t last_chunk_size = completion_queue_end_addr - completion_queue_write_addr;
                cq_noc_async_write_with_state_any_len(data_ptr, completion_queue_write_addr, last_chunk_size);
                uint32_t num_noc_packets_written = div_up(last_chunk_size, NOC_MAX_BURST_SIZE);
                noc_nonposted_writes_num_issued[noc_index] += num_noc_packets_written;
                noc_nonposted_writes_acked[noc_index] += num_noc_packets_written;
                completion_queue_write_addr = COMPLETION_QUEUE_BASE_ADDR;
                data_ptr += last_chunk_size;
                length -= last_chunk_size;
                xfer_size -= last_chunk_size;
            }
            cq_noc_async_write_with_state_any_len(data_ptr, completion_queue_write_addr, xfer_size);
            // completion_queue_push_back below will do a write to host, so we add 1 to the number of data packets
            // written
            uint32_t num_noc_packets_written = div_up(xfer_size, NOC_MAX_BURST_SIZE) + 1;
            noc_nonposted_writes_num_issued[noc_index] += num_noc_packets_written;
            noc_nonposted_writes_acked[noc_index] += num_noc_packets_written;

            // This will update the write ptr on device and host
            // We flush to ensure the ptr has been read out of l1 before we update it again
            completion_queue_push_back(npages);

            length -= xfer_size;
            data_ptr += xfer_size;
            noc_async_writes_flushed();
        }
    }
    cmd_ptr = data_ptr;
}

void process_exec_buf_end_h() {
    if constexpr (SPLIT_PREFETCH) {
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore<fd_core_type>(PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR));

        noc_semaphore_inc(
            get_noc_addr_helper(PREFETCH_H_NOC_XY, (uint32_t)sem_addr), PREFETCH_H_MAX_CREDITS, noc_index);
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

CBWriter<MY_DOWNSTREAM_CB_SEM_ID, 0, 0, 0> dispatch_h_cb_writer{};

// Relay, potentially through the mux/dmux/tunneller path
// Code below sends 1 page worth of data except at the end of a cmd
// This means the downstream buffers are always page aligned, simplifies wrap handling
template <uint32_t preamble_size>
void relay_to_next_cb(uint32_t data_ptr, uint64_t wlength) {
    static_assert(preamble_size == 0, "Dispatcher preamble size must be 0. This is not supported anymore with Fabric");

    // regular write, inline writes, and atomic writes use different cmd bufs, so we can init state for each
    // TODO: Add support for stateful atomics. We can preserve state once cb_acquire_pages is changed to a free running
    // counter so we would only need to inc atomics downstream
    relay_client.init_write_state_only<NOC_INDEX, NCRISC_WR_CMD_BUF>(get_noc_addr_helper(downstream_noc_xy, 0));
    relay_client.init_inline_write_state_only<NOC_INDEX>(get_noc_addr_helper(downstream_noc_xy, 0));

    constexpr uint32_t max_batch_size = ~(dispatch_cb_page_size - 1);
    while (wlength != 0) {
        uint32_t length = (wlength > max_batch_size) ? max_batch_size : static_cast<uint32_t>(wlength);
        wlength -= length;
        while (length > 0) {
            dispatch_h_cb_writer.acquire_pages(1);

            uint32_t xfer_size;
            bool not_end_of_cmd;
            if (length > dispatch_cb_page_size - preamble_size) {
                xfer_size = dispatch_cb_page_size - preamble_size;
                not_end_of_cmd = true;
            } else {
                xfer_size = length;
                not_end_of_cmd = false;
            }
            length -= xfer_size;

            if constexpr (preamble_size > 0) {
                uint32_t flag;
                relay_client.write_inline<NOC_INDEX>(
                    get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr),
                    xfer_size + preamble_size + not_end_of_cmd);
                downstream_cb_data_ptr += preamble_size;
            }
            // Get a page if needed
            if (xfer_size > dispatch_cb_reader.available_bytes(data_ptr)) {
                dispatch_cb_reader.get_cb_page_and_release_pages(data_ptr, [&](bool will_wrap) {
                    uint32_t orphan_size = dispatch_cb_reader.available_bytes(data_ptr);
                    if (orphan_size != 0) {
                        relay_client.write<NOC_INDEX, true, NCRISC_WR_CMD_BUF>(
                            data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr), orphan_size);
                        xfer_size -= orphan_size;
                        downstream_cb_data_ptr += orphan_size;
                        if (downstream_cb_data_ptr == downstream_cb_end) {
                            downstream_cb_data_ptr = DOWNSTREAM_CB_BASE;
                        }
                        if (!will_wrap) {
                            data_ptr += orphan_size;
                        }
                    }
                });
            }

            relay_client.write_atomic_inc_any_len<
                NOC_INDEX,
                downstream_noc_xy,
                DOWNSTREAM_CB_SEM_ID,
                true,
                NCRISC_WR_CMD_BUF>(
                data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_cb_data_ptr), xfer_size, 1);

            data_ptr += xfer_size;
            downstream_cb_data_ptr += xfer_size;
            if (downstream_cb_data_ptr == downstream_cb_end) {
                downstream_cb_data_ptr = DOWNSTREAM_CB_BASE;
            }
        }
    }

    // Move to next page
    downstream_cb_data_ptr = round_up_pow2(downstream_cb_data_ptr, dispatch_cb_page_size);
    if (downstream_cb_data_ptr == downstream_cb_end) {
        downstream_cb_data_ptr = DOWNSTREAM_CB_BASE;
    }

    cmd_ptr = data_ptr;
}

void process_write_host_d() {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;
    // Remember: host transfer command includes the command in the payload, don't add it here
    uint64_t length = cmd->write_linear_host.length;
    uint32_t data_ptr = cmd_ptr;

    relay_to_next_cb<SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE>(data_ptr, length);
}

void relay_write_h() {
    volatile tt_l1_ptr CQDispatchCmdLarge* cmd = (volatile tt_l1_ptr CQDispatchCmdLarge*)cmd_ptr;
    uint64_t length = sizeof(CQDispatchCmdLarge) + cmd->write_linear.length;
    uint32_t data_ptr = cmd_ptr;

    relay_to_next_cb<SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE>(data_ptr, length);
}

void process_exec_buf_end_d() { relay_to_next_cb<SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE>(cmd_ptr, sizeof(CQDispatchCmd)); }

// Note that for non-paged writes, the number of writes per page is always 1
// This means each noc_write frees up a page
void process_write_linear(uint32_t num_mcast_dests) {
    volatile tt_l1_ptr CQDispatchCmdLarge* cmd = (volatile tt_l1_ptr CQDispatchCmdLarge*)cmd_ptr;
    bool multicast = num_mcast_dests > 0;
    if (not multicast) {
        num_mcast_dests = 1;
    }

    uint32_t dst_noc = cmd->write_linear.noc_xy_addr;
    uint32_t write_offset_index = cmd->write_linear.write_offset_index;
    uint64_t dst_addr = cmd->write_linear.addr + write_offset[write_offset_index];
    uint64_t length = cmd->write_linear.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmdLarge);
    if (multicast) {
        cq_noc_async_wwrite_init_state<CQ_NOC_sNDl, true>(0, dst_noc, dst_addr);
    } else {
        cq_noc_async_wwrite_init_state<CQ_NOC_sNDl, false>(0, dst_noc, dst_addr);
    }

    while (length != 0) {
        // Transfer size is min(remaining_length, data_available_in_cb)
        uint32_t available_data = dispatch_cb_reader.wait_for_available_data_and_release_old_pages(data_ptr);
        uint32_t xfer_size = length > available_data ? available_data : length;

        cq_noc_async_write_with_state_any_len(data_ptr, dst_addr, xfer_size, num_mcast_dests);
        // Increment counters based on the number of packets that were written
        uint32_t num_noc_packets_written = div_up(xfer_size, NOC_MAX_BURST_SIZE);
        noc_nonposted_writes_num_issued[noc_index] += num_noc_packets_written;
        noc_nonposted_writes_acked[noc_index] += num_mcast_dests * num_noc_packets_written;
        length -= xfer_size;
        data_ptr += xfer_size;
        dst_addr += xfer_size;
    }

    cmd_ptr = data_ptr;
}

void process_write() {
    volatile tt_l1_ptr CQDispatchCmdLarge* cmd = (volatile tt_l1_ptr CQDispatchCmdLarge*)cmd_ptr;
    uint32_t num_mcast_dests = cmd->write_linear.num_mcast_dests;
    process_write_linear(num_mcast_dests);
}

template <bool is_dram>
void process_write_paged() {
    volatile tt_l1_ptr CQDispatchCmd* cmd = (volatile tt_l1_ptr CQDispatchCmd*)cmd_ptr;

    uint32_t page_id = cmd->write_paged.start_page;
    uint32_t base_addr = cmd->write_paged.base_addr;
    uint32_t page_size = cmd->write_paged.page_size;
    uint32_t pages = cmd->write_paged.pages;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    uint32_t write_length = pages * page_size;
    auto addr_gen = TensorAccessor(tensor_accessor::make_interleaved_dspec<is_dram>(), base_addr, page_size);
    uint32_t dst_addr_offset = 0;  // Offset into page.

    while (write_length != 0) {
        // Transfer size is min(remaining_length, data_available_in_cb)
        uint32_t available_data = dispatch_cb_reader.wait_for_available_data_and_release_old_pages(data_ptr);
        uint32_t remaining_page_size = page_size - dst_addr_offset;
        uint32_t xfer_size = remaining_page_size > available_data ? available_data : remaining_page_size;
        // Cap the transfer size to the NOC packet size - use of One Packet NOC API (better performance
        // than writing a generic amount of data)
        xfer_size = xfer_size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : xfer_size;
        uint64_t dst = addr_gen.get_noc_addr(page_id, dst_addr_offset);

        noc_async_write<NOC_MAX_BURST_SIZE>(data_ptr, dst, xfer_size);
        // If paged write is not completed for a page (dispatch_cb_page_size < page_size) then add offset, otherwise
        // incr page_id.
        if (xfer_size < remaining_page_size) {
            // The above evaluates to: dst_addr_offset + xfer_size < page_size, but this saves a redundant calculation.
            dst_addr_offset += xfer_size;
        } else {
            page_id++;
            dst_addr_offset = 0;
        }

        write_length -= xfer_size;
        data_ptr += xfer_size;
    }

    cmd_ptr = data_ptr;
}

// Packed write command
// Layout looks like:
//   - CQDispatchCmd struct
//   - count CQDispatchWritePackedSubCmd structs (max 1020)
//   - pad to L1 alignment
//   - count data packets of size size, each L1 aligned
//
// Note that there are multiple size restrictions on this cmd:
//  - all sub_cmds fit in one page
//  - size fits in one page
//
// Since all subcmds all appear in the first page and given the size restrictions
// this command can't be too many pages.  All pages are released at the end
template <bool mcast, typename WritePackedSubCmd>
void process_write_packed(uint32_t flags, uint32_t* l1_cache) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;

    uint32_t count = cmd->write_packed.count;
    constexpr uint32_t sub_cmd_size = sizeof(WritePackedSubCmd);
    // Copying in a burst is about a 30% net gain vs reading one value per loop below
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        (volatile uint32_t tt_l1_ptr*)(cmd_ptr + sizeof(CQDispatchCmd)),
        count * sub_cmd_size / sizeof(uint32_t),
        l1_cache);

    uint32_t xfer_size = cmd->write_packed.size;
    uint32_t write_offset_index = cmd->write_packed.write_offset_index;
    uint32_t dst_addr = cmd->write_packed.addr + write_offset[write_offset_index];

    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd) + count * sizeof(WritePackedSubCmd);
    data_ptr = round_up_pow2(data_ptr, L1_ALIGNMENT);
    uint32_t stride =
        (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE) ? 0 : round_up_pow2(xfer_size, L1_ALIGNMENT);

    volatile uint32_t tt_l1_ptr* l1_addr = (uint32_t*)(cmd_ptr + sizeof(CQDispatchCmd));
    cq_noc_async_write_init_state<CQ_NOC_snDL, mcast>(0, dst_addr, xfer_size);

    uint32_t writes = 0;
    uint32_t mcasts = 0;
    auto wait_for_barrier = [&]() {
        if (!mcast) {
            return;
        }
        noc_nonposted_writes_num_issued[noc_index] += writes;
        noc_nonposted_writes_acked[noc_index] += mcasts;
        writes = 0;
        mcasts = 0;
        // Workaround mcast path reservation hangs by always waiting for a write
        // barrier before doing an mcast that isn't linked to a previous mcast.
        noc_async_write_barrier();
    };
    WritePackedSubCmd* sub_cmd_ptr = (WritePackedSubCmd*)l1_cache;
    while (count != 0) {
        uint32_t dst_noc = sub_cmd_ptr->noc_xy_addr;
        uint32_t num_dests = mcast ? ((CQDispatchWritePackedMulticastSubCmd*)sub_cmd_ptr)->num_mcast_dests : 1;
        sub_cmd_ptr++;
        uint64_t dst = get_noc_addr_helper(dst_noc, dst_addr);
        // Get a page if needed
        if (xfer_size > dispatch_cb_reader.available_bytes(data_ptr)) {
            // Check for block completion and issue orphan writes for this block
            // before proceeding to next block
            uint32_t orphan_size = 0;
            dispatch_cb_reader.get_cb_page_and_release_pages(data_ptr, [&](bool will_wrap) {
                orphan_size = dispatch_cb_reader.available_bytes(data_ptr);
                if (orphan_size != 0) {
                    wait_for_barrier();
                    cq_noc_async_write_with_state<CQ_NOC_SNdL>(data_ptr, dst, orphan_size, num_dests);
                    writes++;
                    mcasts += num_dests;
                    if (!will_wrap) {
                        data_ptr += orphan_size;
                    }
                }
                noc_nonposted_writes_num_issued[noc_index] += writes;
                noc_nonposted_writes_acked[noc_index] += mcasts;
                writes = 0;
                mcasts = 0;
            });

            // Write the remainder of the transfer. All the remaining contents of the transfer is now available, since
            // the size of a single transfer is at most the CB page size. This write has a different destination address
            // than the default, so we restore the destination address to the start immediately afterwards to avoid the
            // overhead in the common case.
            if (orphan_size != 0) {
                uint32_t remainder_xfer_size = xfer_size - orphan_size;
                // Creating full NOC addr not needed as we are not programming the noc coords
                uint32_t remainder_dst_addr = dst_addr + orphan_size;
                wait_for_barrier();
                cq_noc_async_write_with_state<CQ_NOC_SnDL>(
                    data_ptr, remainder_dst_addr, remainder_xfer_size, num_dests);
                // Reset values expected below
                cq_noc_async_write_with_state<CQ_NOC_snDL, CQ_NOC_WAIT, CQ_NOC_send>(0, dst, xfer_size);
                writes++;
                mcasts += num_dests;

                count--;
                data_ptr += stride - orphan_size;

                continue;
            }
        }

        wait_for_barrier();
        cq_noc_async_write_with_state<CQ_NOC_SNdl>(data_ptr, dst, xfer_size, num_dests);
        writes++;
        mcasts += num_dests;

        count--;
        data_ptr += stride;
    }

    noc_nonposted_writes_num_issued[noc_index] += writes;
    noc_nonposted_writes_acked[noc_index] += mcasts;

    cmd_ptr = data_ptr;
}

// This routine below can be implemented to either prefetch sub_cmds into local memory or leave them in L1
// Prefetching into local memory limits the number of sub_cmds (used as kernel writes) in one cmd
// Leaving in L1 limits the number of bytes of data in one cmd (whole command must fit in CB)
//
// The code below prefetches sub_scmds into local cache because:
//  - it is likely faster (not measured yet, but base based on write_packed)
//  - allows pages to be released as they are processed (since prefetcher won't overwrite the sub-cmds)
//  - can presently handle 36 subcmds, or 7 5-processor kernels
// Without prefetching:
//  - cmd size is limited to CB size which is 128K and may go to 192K
//  - w/ 4K kernel binaries, 192K is 9 5-processor kernels, 128K is 6
//  - utilizing the full space creates a full prefetcher stall as all memory is tied up
//  - so a better practical full size is 3-4 full sets of 4K kernel binaries
// May eventually want a separate implementation for tensix vs eth dispatch
void process_write_packed_large(uint32_t* l1_cache) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;

    uint32_t count = cmd->write_packed_large.count;
    uint32_t alignment = cmd->write_packed_large.alignment;
    uint32_t write_offset_index = cmd->write_packed_large.write_offset_index;
    uint32_t local_write_offset = write_offset[write_offset_index];
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd) + count * sizeof(CQDispatchWritePackedLargeSubCmd);
    data_ptr = round_up_pow2(data_ptr, L1_ALIGNMENT);

    constexpr uint32_t sub_cmd_size = sizeof(CQDispatchWritePackedLargeSubCmd);
    careful_copy_from_l1_to_local_cache<l1_to_local_cache_copy_chunk, l1_cache_elements_rounded>(
        (volatile uint32_t tt_l1_ptr*)(cmd_ptr + sizeof(CQDispatchCmd)),
        count * sub_cmd_size / sizeof(uint32_t),
        l1_cache);

    uint32_t writes = 0;
    uint32_t mcasts = noc_nonposted_writes_acked[noc_index];
    CQDispatchWritePackedLargeSubCmd* sub_cmd_ptr = (CQDispatchWritePackedLargeSubCmd*)l1_cache;

    bool init_state = true;
    bool must_barrier = true;
    while (count != 0) {
        uint32_t dst_addr = sub_cmd_ptr->addr + local_write_offset;
        // CQDispatchWritePackedLargeSubCmd always stores length - 1, so add 1 to get the actual length
        // This avoids the need to handle the special case where 65536 bytes overflows to 0
        uint32_t length = sub_cmd_ptr->length_minus1 + 1;
        uint32_t num_dests = sub_cmd_ptr->num_mcast_dests;
        uint32_t pad_size = align_power_of_2(length, alignment) - length;
        uint32_t unlink = sub_cmd_ptr->flags & CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;
        auto wait_for_barrier = [&]() {
            if (!must_barrier) {
                return;
            }
            noc_nonposted_writes_num_issued[noc_index] += writes;

            mcasts += num_dests * writes;
            noc_nonposted_writes_acked[noc_index] = mcasts;
            writes = 0;
            // Workaround mcast path reservation hangs by always waiting for a write
            // barrier before doing an mcast that isn't linked to a previous mcast.
            noc_async_write_barrier();
        };

        // Only re-init state after we have unlinked the last transaction
        // Otherwise we assume NOC coord hasn't changed
        // TODO: If we are able to send 0 length txn to unset link, we don't need a flag and can compare dst_noc to prev
        // to determine linking
        if (init_state) {
            uint32_t dst_noc = sub_cmd_ptr->noc_xy_addr;
            cq_noc_async_write_init_state<CQ_NOC_sNdl, true, true>(0, get_noc_addr_helper(dst_noc, dst_addr));
            must_barrier = true;
        }

        sub_cmd_ptr++;

        while (length != 0) {
            // More data needs to be written, but we've exhausted the CB. Acquire more pages.
            if (dispatch_cb_reader.available_bytes(data_ptr) == 0) {
                dispatch_cb_reader.get_cb_page_and_release_pages(data_ptr, [&](bool /*will_wrap*/) {
                    // Block completion - account for all writes issued for this block before moving to next
                    noc_nonposted_writes_num_issued[noc_index] += writes;
                    mcasts += num_dests * writes;
                    writes = 0;
                });
            }
            // Transfer size is min(remaining_length, data_available_in_cb)
            uint32_t available_data = dispatch_cb_reader.available_bytes(data_ptr);
            uint32_t xfer_size;
            if (length > available_data) {
                xfer_size = available_data;
                wait_for_barrier();
                cq_noc_async_write_with_state_any_len(data_ptr, dst_addr, xfer_size, num_dests);
                must_barrier = false;
            } else {
                xfer_size = length;
                if (unlink) {
                    wait_for_barrier();
                    uint32_t rem_xfer_size =
                        cq_noc_async_write_with_state_any_len<false>(data_ptr, dst_addr, xfer_size, num_dests);
                    // Unset Link flag
                    cq_noc_async_write_init_state<CQ_NOC_sndl, true, false>(0, 0, 0);
                    uint32_t data_offset = xfer_size - rem_xfer_size;
                    cq_noc_async_write_with_state<CQ_NOC_SnDL, CQ_NOC_wait>(
                        data_ptr + data_offset, dst_addr + data_offset, rem_xfer_size, num_dests);
                    // Later writes must barrier, but the `must_barrier = true` in the `if (init_state)` block above
                    // will see to that.
                } else {
                    wait_for_barrier();
                    cq_noc_async_write_with_state_any_len(data_ptr, dst_addr, xfer_size, num_dests);
                    must_barrier = false;
                }
            }
            writes += div_up(xfer_size, NOC_MAX_BURST_SIZE);
            length -= xfer_size;
            data_ptr += xfer_size;
            dst_addr += xfer_size;
        }

        init_state = unlink;

        noc_nonposted_writes_num_issued[noc_index] += writes;
        mcasts += num_dests * writes;
        writes = 0;

        // Handle padded size and potential wrap
        if (pad_size > dispatch_cb_reader.available_bytes(data_ptr)) {
            dispatch_cb_reader.get_cb_page_and_release_pages(data_ptr, [&](bool will_wrap) {
                if (will_wrap) {
                    uint32_t orphan_size = dispatch_cb_reader.available_bytes(data_ptr);
                    pad_size -= orphan_size;
                }
            });
        }
        data_ptr += pad_size;

        count--;
    }
    noc_nonposted_writes_acked[noc_index] = mcasts;

    cmd_ptr = data_ptr;
}

FORCE_INLINE
uint32_t stream_wrap_ge(uint32_t a, uint32_t b) {
    constexpr uint32_t shift = 32 - MEM_WORD_ADDR_WIDTH;
    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return (diff << shift) >= 0;
}

static void process_wait() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    auto flags = cmd->wait.flags;

    uint32_t barrier = flags & CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER;
    uint32_t notify_prefetch = flags & CQ_DISPATCH_CMD_WAIT_FLAG_NOTIFY_PREFETCH;
    uint32_t clear_stream = flags & CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM;
    uint32_t wait_memory = flags & CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_MEMORY;
    uint32_t wait_stream = flags & CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM;
    uint32_t count = cmd->wait.count;
    uint32_t stream = cmd->wait.stream;

    if (barrier) {
        noc_async_write_barrier();
    }

    if (wait_memory) {
        uint32_t addr = cmd->wait.addr;
        volatile tt_l1_ptr uint32_t* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        do {
            invalidate_l1_cache();
        } while (!wrap_ge(*sem_addr, count));
    }
    if (wait_stream) {
        volatile uint32_t* sem_addr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        do {
        } while (!stream_wrap_ge(*sem_addr, count));
    }

    if (clear_stream) {
        volatile uint32_t* sem_addr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        uint32_t neg_sem_val = -(*sem_addr);
        NOC_STREAM_WRITE_REG(
            stream,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            neg_sem_val << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }
    if (notify_prefetch) {
        noc_semaphore_inc(
            get_noc_addr_helper(upstream_noc_xy, get_semaphore<fd_core_type>(UPSTREAM_SYNC_SEM)),
            1,
            UPSTREAM_NOC_INDEX);
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void process_go_signal_mcast_cmd() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t stream = cmd->mcast.wait_stream;
    // The location of the go signal embedded in the command does not meet NOC alignment requirements.
    // cmd_ptr is guaranteed to meet the alignment requirements, since it is written to by prefetcher over NOC.
    // Copy the go signal from an unaligned location to an aligned (cmd_ptr) location. This is safe as long as we
    // can guarantee that copying the go signal does not corrupt any other command fields, which is true (see
    // CQDispatchGoSignalMcastCmd).
    volatile uint32_t tt_l1_ptr* aligned_go_signal_storage = (volatile uint32_t tt_l1_ptr*)cmd_ptr;
    uint32_t go_signal_value = cmd->mcast.go_signal;
    uint8_t go_signal_noc_data_idx = cmd->mcast.noc_data_start_index;
    uint32_t multicast_go_offset = cmd->mcast.multicast_go_offset;
    uint32_t num_unicasts = cmd->mcast.num_unicast_txns;
    uint32_t wait_count = cmd->mcast.wait_count;
    if (multicast_go_offset != CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET) {
        // Setup registers before waiting for workers so only the NOC_CMD_CTRL register needs to be touched after.
        uint64_t dst_noc_addr_multicast =
            get_noc_addr_helper(WORKER_MCAST_GRID, MCAST_GO_SIGNAL_ADDR + sizeof(uint32_t) * multicast_go_offset);
        uint32_t num_dests = NUM_WORKER_CORES_TO_MCAST;
        // Ensure the offset with respect to L1_ALIGNMENT is the same for the source and destination.
        uint32_t storage_offset = multicast_go_offset % (L1_ALIGNMENT / sizeof(uint32_t));
        aligned_go_signal_storage[storage_offset] = go_signal_value;

        cq_noc_async_write_init_state<CQ_NOC_SNDL, true>(
            (uint32_t)&aligned_go_signal_storage[storage_offset], dst_noc_addr_multicast, sizeof(uint32_t));
        noc_nonposted_writes_acked[noc_index] += num_dests;

        while (!stream_wrap_ge(
            NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX), wait_count)) {
        }
        cq_noc_async_write_with_state<CQ_NOC_sndl, CQ_NOC_wait>(0, 0, 0);
        noc_nonposted_writes_num_issued[noc_index] += 1;
    } else {
        while (!stream_wrap_ge(
            NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX), wait_count)) {
        }
    }

    *aligned_go_signal_storage = go_signal_value;
    if constexpr (VIRTUALIZE_UNICAST_CORES) {
        // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
        // This chip is virtualizing cores the go signal is unicasted to
        // In this case, the number of unicasts specified in the command can exceed
        // the number of actual cores on this chip.
        if (cmd->mcast.num_unicast_txns > NUM_PHYSICAL_UNICAST_CORES) {
            // If this is the case, cap the number of unicasts to avoid invalid NOC txns
            num_unicasts = NUM_PHYSICAL_UNICAST_CORES;
            // Fake updates from non-existent workers here. The dispatcher expects an ack from
            // the number of cores specified inside cmd->mcast.num_unicast_txns. If this is
            // greater than the number of cores actually on the chip, we must account for acks
            // from non-existent cores here.
            NOC_STREAM_WRITE_REG(
                stream,
                STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
                (NUM_VIRTUAL_UNICAST_CORES - NUM_PHYSICAL_UNICAST_CORES) << REMOTE_DEST_BUF_WORDS_FREE_INC);
        }
    }

    for (uint32_t i = 0; i < num_unicasts; ++i) {
        uint64_t dst = get_noc_addr_helper(go_signal_noc_data[go_signal_noc_data_idx++], UNICAST_GO_SIGNAL_ADDR);
        noc_async_write_one_packet((uint32_t)(aligned_go_signal_storage), dst, sizeof(uint32_t));
    }

    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void process_notify_dispatch_s_go_signal_cmd() {
    // Update free running counter on dispatch_s, signalling that it's safe to send a go signal to workers
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t wait = cmd->notify_dispatch_s_go_signal.wait;
    // write barrier to wait before sending the go signal
    if (wait) {
        noc_async_write_barrier();
    }
    uint16_t index_bitmask = cmd->notify_dispatch_s_go_signal.index_bitmask;

    while (index_bitmask != 0) {
        uint32_t set_index = __builtin_ctz(index_bitmask);
        uint32_t dispatch_s_sync_sem_addr = DISPATCH_S_SYNC_SEM_BASE_ADDR + set_index * L1_ALIGNMENT;
        if constexpr (DISTRIBUTED_DISPATCHER) {
            static uint32_t num_go_signals_safe_to_send[MAX_NUM_WORKER_SEMS] = {0};
            uint64_t dispatch_s_notify_addr = get_noc_addr_helper(dispatch_s_noc_xy, dispatch_s_sync_sem_addr);
            num_go_signals_safe_to_send[set_index]++;
            noc_inline_dw_write(dispatch_s_notify_addr, num_go_signals_safe_to_send[set_index]);
        } else {
            tt_l1_ptr uint32_t* notify_ptr = (uint32_t tt_l1_ptr*)(dispatch_s_sync_sem_addr);
            *notify_ptr = (*notify_ptr) + 1;
        }
        // Unset the bit
        index_bitmask &= index_bitmask - 1;
    }
    cmd_ptr += sizeof(CQDispatchCmd);
}

FORCE_INLINE
void set_go_signal_noc_data() {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)cmd_ptr;
    uint32_t num_words = cmd->set_go_signal_noc_data.num_words;
    cmd_ptr = copy_words_to_l1_and_advance(
        cmd_ptr + sizeof(CQDispatchCmd), num_words, MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES, go_signal_noc_data);
}
FORCE_INLINE void process_set_write_offset_cmd(uint32_t& local_cmd_ptr) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)local_cmd_ptr;
    uint32_t offset_count = cmd->set_write_offset.offset_count;
    uint32_t* cmd_write_offset = (uint32_t*)(local_cmd_ptr + sizeof(CQDispatchCmd));

    for (uint32_t i = 0; i < offset_count; i++) {
        write_offset[i] = cmd_write_offset[i];
    }
    local_cmd_ptr += sizeof(CQDispatchCmd) + sizeof(uint32_t) * offset_count;
}

FORCE_INLINE void process_timestamp(uint32_t& local_cmd_ptr) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)local_cmd_ptr;
    uint32_t dst_noc_xy = cmd->timestamp.noc_xy_addr;
    uint32_t dst_addr = cmd->timestamp.addr;

    // Read wall clock (reading L latches H)
    uint32_t lo = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t hi = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);

    // Write timestamp to command buffer L1 (bypass data cache via tt_l1_ptr)
    volatile tt_l1_ptr uint32_t* scratch = (volatile tt_l1_ptr uint32_t*)local_cmd_ptr;
    scratch[0] = lo;
    scratch[1] = hi;

    // NOC unicast write 8 bytes to DRAM
    uint64_t dst = get_noc_addr_helper(dst_noc_xy, dst_addr);
    noc_async_write_one_packet(local_cmd_ptr, dst, 8);
    noc_async_write_barrier();

    local_cmd_ptr += sizeof(CQDispatchCmd);
}

template <bool d_variant>
static inline bool process_cmd(uint32_t& local_cmd_ptr, uint32_t* l1_cache) {
    volatile CQDispatchCmd tt_l1_ptr* cmd = (volatile CQDispatchCmd tt_l1_ptr*)local_cmd_ptr;

    switch (cmd->base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE_LINEAR:
            if constexpr (d_variant) {
                process_write();
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H:
            if constexpr (IS_H_VARIANT) {
                process_write();
            } else {
                relay_write_h();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
            if constexpr (IS_H_VARIANT) {
                process_write_host_h();
            } else {
                process_write_host_d();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PAGED:
            if constexpr (d_variant) {
                if (cmd->write_paged.is_dram) {
                    process_write_paged<true>();
                } else {
                    process_write_paged<false>();
                }
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PACKED:
            if constexpr (d_variant) {
                uint32_t flags = cmd->write_packed.flags;
                if (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST) {
                    process_write_packed<true, CQDispatchWritePackedMulticastSubCmd>(flags, l1_cache);
                } else {
                    process_write_packed<false, CQDispatchWritePackedUnicastSubCmd>(flags, l1_cache);
                }
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL:
            if constexpr (d_variant) {
                process_notify_dispatch_s_go_signal_cmd();
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:
            if constexpr (d_variant) {
                process_write_packed_large(l1_cache);
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_WAIT:
            if constexpr (d_variant) {
                process_wait();
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_EXEC_BUF_END:
            if constexpr (IS_H_VARIANT) {
                process_exec_buf_end_h();
            } else {
                process_exec_buf_end_d();
            }
            break;

        case CQ_DISPATCH_CMD_SEND_GO_SIGNAL:
            if constexpr (d_variant) {
                process_go_signal_mcast_cmd();
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_SET_NUM_WORKER_SEMS:
            // This command is only used by dispatch_s
            local_cmd_ptr += sizeof(CQDispatchCmd);
            break;

        case CQ_DISPATCH_SET_GO_SIGNAL_NOC_DATA:
            if constexpr (d_variant) {
                set_go_signal_noc_data();
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_TIMESTAMP:
            if constexpr (d_variant) {
                process_timestamp(local_cmd_ptr);
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_SET_WRITE_OFFSET:
            if constexpr (d_variant) {
                process_set_write_offset_cmd(local_cmd_ptr);
            } else {
                __builtin_unreachable();
            }
            break;

        case CQ_DISPATCH_CMD_TERMINATE:
            if constexpr (d_variant && !IS_H_VARIANT) {
                relay_to_next_cb<SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE>(local_cmd_ptr, sizeof(CQDispatchCmd));
            }
            local_cmd_ptr += sizeof(CQDispatchCmd);
            return true;

        default: __builtin_unreachable();
    }

    return false;
}

void kernel_main() {
    set_l1_data_cache<true>();

    // Initialize local state of any additional nocs used instead of the default
    static_assert(NOC_INDEX != UPSTREAM_NOC_INDEX);
    if constexpr (NOC_INDEX != UPSTREAM_NOC_INDEX) {
        noc_local_state_init(UPSTREAM_NOC_INDEX);
    }

    reset_worker_completion_stream_counts<FIRST_STREAM_USED, MAX_NUM_WORKER_SEMS>();

    static_assert(IS_D_VARIANT || SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE == 0);

    uint32_t l1_cache[l1_cache_elements_rounded];

    dispatch_cb_reader.init();
    cmd_ptr = DISPATCH_CB_BASE;
    write_offset[0] = write_offset[1] = write_offset[2] = 0;

    {
        uint32_t completion_queue_wr_ptr_and_toggle = *get_cq_completion_write_ptr();
        cq_write_interface.completion_fifo_wr_ptr = completion_queue_wr_ptr_and_toggle & 0x7fffffff;
        cq_write_interface.completion_fifo_wr_toggle = completion_queue_wr_ptr_and_toggle >> 31;
    }
    // Initialize the relay client for split dispatch
    if constexpr (!(IS_H_VARIANT && IS_D_VARIANT)) {
        relay_client.init<NOC_INDEX, NCRISC_WR_CMD_BUF>(get_noc_addr_helper(downstream_noc_xy, 0));
    }
    bool done = false;
    while (!done) {
        dispatch_cb_reader.wait_for_available_data_and_release_old_pages(cmd_ptr);

        done = process_cmd<IS_D_VARIANT>(cmd_ptr, l1_cache);

        // Move to next page
        cmd_ptr = round_up_pow2(cmd_ptr, dispatch_cb_page_size);
    }

    dispatch_cb_reader.release_all_pages(cmd_ptr);

    noc_async_write_barrier();

    // Confirm expected number of pages, spinning here is a leak
    dispatch_cb_reader.wait_all_pages();

    noc_async_full_barrier();

    if (IS_H_VARIANT && !IS_D_VARIANT) {
        relay_client.template teardown<UPSTREAM_NOC_INDEX, upstream_noc_xy, UPSTREAM_DISPATCH_CB_SEM_ID>();
    }
    set_l1_data_cache<false>();
}
