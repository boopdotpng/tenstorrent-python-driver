// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include <cstdint>
#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "hostdev/dev_msgs.h"
#include "stream_io_map.h"
#include "internal/firmware_common.h"
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"
#include "internal/circular_buffer_init.h"

#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"
// clang-format on

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
volatile tt_l1_ptr uint8_t* const ncrisc_run = mailboxes->subordinate_sync.map;

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

// These arrays are used to store the worker logical to virtual coordinate mapping
uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)] __attribute__((used));
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}  // namespace kernel_profiler
#endif

inline __attribute__((always_inline)) void wait_for_brisc_notification() {
    while (true) {
        uint8_t run_value = *ncrisc_run;
        if (run_value == RUN_SYNC_MSG_GO || run_value == RUN_SYNC_MSG_LOAD) {
            break;
        }
        invalidate_l1_cache();
    }
}

inline __attribute__((always_inline)) void signal_ncrisc_completion() { *ncrisc_run = RUN_SYNC_MSG_DONE; }

int main(int argc, char* argv[]) {
    configure_csr();
    WAYPOINT("I");

    do_crt1((uint32_t tt_l1_ptr*)MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
    noc_worker_logical_to_virtual_map_init(MEM_LOGICAL_TO_VIRTUAL_SCRATCH);

    risc_init();

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    signal_ncrisc_completion();

    DeviceProfilerInit();
    while (1) {
        WAYPOINT("W");
        wait_for_brisc_notification();
        DeviceZoneScopedMainN("NCRISC-FW");

        uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
        launch_msg_t* launch_msg = &(mailboxes->launch[launch_msg_rd_ptr]);

        uint32_t kernel_config_base = firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, PROCESSOR_INDEX);
        int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1);

        uint32_t kernel_lma = kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index];
        uint32_t tt_l1_ptr* cb_l1_base =
            (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.local_cb_offset);
        uint32_t local_cb_mask = launch_msg->kernel_config.local_cb_mask;
        setup_local_cb_read_write_interfaces<true, true, false>(cb_l1_base, 0, local_cb_mask);

        cb_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg->kernel_config.remote_cb_offset);
        uint32_t end_cb_index = launch_msg->kernel_config.min_remote_cb_start_index;
        // NOC argument is unused
        experimental::setup_remote_cb_interfaces<false>(cb_l1_base, end_cb_index, 0, 0, 0, 0);
        my_relative_x_ = my_logical_x_ - launch_msg->kernel_config.sub_device_origin_x;
        my_relative_y_ = my_logical_y_ - launch_msg->kernel_config.sub_device_origin_y;

        WAYPOINT("R");

        while (*ncrisc_run != RUN_SYNC_MSG_GO) {
            invalidate_l1_cache();
        }
        auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
        record_stack_usage(stack_free);
        WAYPOINT("D");

        signal_ncrisc_completion();
    }

    return 0;
}
