// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#undef PROFILE_NOC_EVENTS
#include "risc_common.h"
#include "tensix.h"
#include "tensix_types.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "hostdev/dev_msgs.h"
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"
#include "internal/circular_buffer_init.h"
#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

#include "internal/debug/watcher_common.h"
#include "api/debug/waypoint.h"
#include "api/debug/dprint.h"
#include "internal/debug/stack_usage.h"

uint8_t noc_index;

constexpr uint32_t RISCV_IC_BRISC_MASK = 0x1;
constexpr uint32_t RISCV_IC_NCRISC_MASK = 0x10;
constexpr uint32_t RISCV_IC_TRISC0_MASK = 0x2;
constexpr uint32_t RISCV_IC_TRISC1_MASK = 0x4;
constexpr uint32_t RISCV_IC_TRISC2_MASK = 0x8;
constexpr uint32_t RISCV_IC_TRISC_ALL_MASK = RISCV_IC_TRISC0_MASK | RISCV_IC_TRISC1_MASK | RISCV_IC_TRISC2_MASK;

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
tt_l1_ptr subordinate_map_t* const subordinate_sync = (subordinate_map_t*)mailboxes->subordinate_sync.map;

c_tensix_core core;

volatile tt_l1_ptr uint32_t* instrn_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* pc_buf[MAX_THREADS];
volatile tt_l1_ptr uint32_t* mailbox[MAX_THREADS];

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

uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));
uint8_t prev_noc_mode = DM_DEDICATED_NOC;

uint8_t worker_logical_col_to_virtual_col[round_up_to_mult_of_4(noc_size_x)] __attribute__((used));
uint8_t worker_logical_row_to_virtual_row[round_up_to_mult_of_4(noc_size_y)] __attribute__((used));

#if defined(PROFILE_KERNEL)
namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t stackSize __attribute__((used));
uint32_t sums[SUM_COUNT] __attribute__((used));
uint32_t sumIDs[SUM_COUNT] __attribute__((used));
uint32_t traceCount __attribute__((used));
}  // namespace kernel_profiler
#endif

void enable_power_management() {
    uint32_t pm_mask = 0xFFFF;
    uint32_t pm_hyst = 32;

    uint32_t hyst_val = pm_hyst;

    {
        uint32_t hyst0_reg_data = ((hyst_val) << 24) | ((hyst_val) << 16) | ((hyst_val) << 8) | hyst_val;
        uint32_t hyst1_reg_data = ((hyst_val) << 24) | ((hyst_val) << 16) | ((hyst_val) << 8) | hyst_val;
        uint32_t hyst2_reg_data = ((hyst_val) << 24) | ((hyst_val) << 16) | ((hyst_val) << 8) | hyst_val;

        WRITE_REG(RISCV_DEBUG_REG_CG_CTRL_HYST0, hyst0_reg_data);
        WRITE_REG(RISCV_DEBUG_REG_CG_CTRL_HYST1, hyst1_reg_data);
        WRITE_REG(RISCV_DEBUG_REG_CG_CTRL_HYST2, hyst2_reg_data);
    }

    /*FIXME: need to deal with srcb ctrl bit not fitting in 16 bits. For  */
    /*now just always turn it on */
    *((volatile uint32_t*)RISCV_DEBUG_REG_CG_CTRL_EN) = 0x10000 | (pm_mask);

    if (((pm_mask & 0x0100) >> 8) == 1) {
        uint32_t hyst_val = pm_hyst & 0x7f;

        core.write_stream_register(0, STREAM_PERF_CONFIG_REG_INDEX,
                                   pack_field(1, CLOCK_GATING_EN_WIDTH, CLOCK_GATING_EN) |
                                       pack_field(hyst_val, CLOCK_GATING_HYST_WIDTH, CLOCK_GATING_HYST) |

                                       pack_field(32, PARTIAL_SEND_WORDS_THR_WIDTH, PARTIAL_SEND_WORDS_THR));

        uint32_t oldval;
        oldval = NOC_READ_REG(NOC0_REGS_START_ADDR + 0x100);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC0_REGS_START_ADDR + 0x100, oldval);

        oldval = NOC_READ_REG(NOC0_REGS_START_ADDR + 0x104);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC0_REGS_START_ADDR + 0x104, oldval);

        oldval = NOC_READ_REG(NOC1_REGS_START_ADDR + 0x100);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC1_REGS_START_ADDR + 0x100, oldval);

        oldval = NOC_READ_REG(NOC1_REGS_START_ADDR + 0x104);
        oldval = (oldval & 0xFFFFFF00) | 1 | (hyst_val << 1);
        NOC_WRITE_REG(NOC1_REGS_START_ADDR + 0x104, oldval);
    }
}

void set_deassert_addresses() {
    WRITE_REG(RISCV_DEBUG_REG_NCRISC_RESET_PC, MEM_NCRISC_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC0_RESET_PC, MEM_TRISC0_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC1_RESET_PC, MEM_TRISC1_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC2_RESET_PC, MEM_TRISC2_FIRMWARE_BASE);
    WRITE_REG(RISCV_DEBUG_REG_TRISC_RESET_PC_OVERRIDE, 0b111);
    WRITE_REG(RISCV_DEBUG_REG_NCRISC_RESET_PC_OVERRIDE, 0x1);
}

void device_setup() {
    for (uint32_t i = 0; i < 3; ++i) {
        instrn_buf[i] = core.instrn_buf_base(i);
        pc_buf[i] = core.pc_buf_base(i);
    }

    volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);

    *((volatile uint32_t*)RISCV_DEBUG_REG_DEST_CG_CTRL) = 0;

    WRITE_REG(RISCV_TDMA_REG_CLK_GATE_EN, 0x3f);

    noc_set_active_instance(0);
    uint32_t niu_cfg0 = noc_get_cfg_reg(NIU_CFG_0);
    noc_set_cfg_reg(NIU_CFG_0, niu_cfg0 | 0x1);
    uint32_t router_cfg0 = noc_get_cfg_reg(ROUTER_CFG_0);
    noc_set_cfg_reg(ROUTER_CFG_0, router_cfg0 | 0x1);

    noc_set_active_instance(1);
    uint32_t niu_cfg1 = noc_get_cfg_reg(NIU_CFG_0);
    noc_set_cfg_reg(NIU_CFG_0, niu_cfg1 | 0x1);
    uint32_t router_cfg1 = noc_get_cfg_reg(ROUTER_CFG_0);
    noc_set_cfg_reg(ROUTER_CFG_0, router_cfg1 | 0x1);
    noc_set_active_instance(0);

    set_deassert_addresses();

    wzeromem(MEM_ZEROS_BASE, MEM_ZEROS_SIZE);

    cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] =
        RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

    core.ex_zeroacc(instrn_buf[0]);

    core.ex_encc(instrn_buf[0]);

    core.ex_load_const(instrn_buf[0]);

    core.ex_rmw_cfg(0, ECC_SCRUBBER_Enable_RMW, 1);
    core.ex_rmw_cfg(0, ECC_SCRUBBER_Scrub_On_Error_RMW, 1);
    core.ex_rmw_cfg(0, ECC_SCRUBBER_Delay_RMW, 0x100);

    core.initialize_tensix_semaphores(instrn_buf[0]);
}

inline void deassert_ncrisc_trisc() {
    subordinate_sync->all = RUN_SYNC_MSG_ALL_INIT;

    deassert_all_reset();
}

inline void run_triscs(uint32_t enables) {
    while (subordinate_sync->trisc0 != RUN_SYNC_MSG_DONE) {
        invalidate_l1_cache();
    }

    if (enables & (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::MATH0))) {
        subordinate_sync->trisc0 = RUN_SYNC_MSG_GO;
        subordinate_sync->trisc1 = RUN_SYNC_MSG_GO;
        subordinate_sync->trisc2 = RUN_SYNC_MSG_GO;
    }
}

inline void start_ncrisc_kernel_run_early(uint32_t enables) {
    if (enables & (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1))) {
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;
    }
}

inline void start_ncrisc_kernel_run([[maybe_unused]] uint32_t enables) {}

inline void wait_ncrisc_trisc() {
    WAYPOINT("NTW");
    while (subordinate_sync->all != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE) {
        invalidate_l1_cache();
    }
    WAYPOINT("NTD");
}

inline void trigger_sync_register_init() {
    subordinate_sync->trisc0 = RUN_SYNC_MSG_INIT_SYNC_REGISTERS;
}

inline void barrier_remote_cb_interface_setup(uint8_t noc_index, uint32_t end_cb_index) {
    if (end_cb_index != NUM_CIRCULAR_BUFFERS) {
        noc_async_atomic_barrier(noc_index);
    }
}

int main() {
    configure_csr();
    WAYPOINT("I");

    do_crt1((uint32_t*)MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_BANK_TO_NOC_SCRATCH);
    noc_worker_logical_to_virtual_map_init(MEM_LOGICAL_TO_VIRTUAL_SCRATCH);

    mailboxes->launch_msg_rd_ptr = 0;
    noc_index = 0;
    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    risc_init();
    device_setup();

    mailboxes->ncrisc_halt.resume_addr = 0;
    deassert_ncrisc_trisc();

    wait_ncrisc_trisc();
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;

    uint8_t noc_mode;
    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    noc_local_state_init(noc_index);
    trigger_sync_register_init();

    DeviceProfilerInit();
    while (1) {
        WAYPOINT("GW");
        uint8_t go_message_signal = RUN_MSG_DONE;

        while (
            ((go_message_signal = mailboxes->go_messages[mailboxes->go_message_index].signal) != RUN_MSG_GO) &&
            !(mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config.preload & DISPATCH_ENABLE_FLAG_PRELOAD)) {
            invalidate_l1_cache();

            if ((go_message_signal == RUN_MSG_RESET_READ_PTR) ||
                (go_message_signal == RUN_MSG_RESET_READ_PTR_FROM_HOST) ||
                (go_message_signal == RUN_MSG_REPLAY_TRACE)) {
                mailboxes->launch_msg_rd_ptr = 0;
                if (go_message_signal == RUN_MSG_RESET_READ_PTR || go_message_signal == RUN_MSG_REPLAY_TRACE) {
                    if (go_message_signal == RUN_MSG_REPLAY_TRACE) {
                        DeviceIncrementTraceCount();
                        DeviceTraceOnlyProfilerInit();
                    }
                    uint32_t go_message_index = mailboxes->go_message_index;

                    uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                    mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;

                    DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                    notify_dispatch_core_done(dispatch_addr, noc_index);
                }
            }
        }

        WAYPOINT("GD");

        {
            DeviceZoneScopedMainN("BRISC-FW");
            uint32_t launch_msg_rd_ptr = mailboxes->launch_msg_rd_ptr;
            launch_msg_t* launch_msg_address = &(mailboxes->launch[launch_msg_rd_ptr]);
            DeviceValidateProfiler(launch_msg_address->kernel_config.enables);
            DeviceZoneSetCounter(launch_msg_address->kernel_config.host_assigned_id);
            uint32_t enables = launch_msg_address->kernel_config.enables;

            if (enables &
                (1u << static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM1))) {
                subordinate_sync->dm1 = RUN_SYNC_MSG_LOAD;
            }
            uint32_t kernel_config_base =
                firmware_config_init(mailboxes, ProgrammableCoreType::TENSIX, PROCESSOR_INDEX);

            volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
            cfg_regs[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] =
                RISCV_IC_BRISC_MASK | RISCV_IC_TRISC_ALL_MASK | RISCV_IC_NCRISC_MASK;

            run_triscs(enables);

            noc_index = launch_msg_address->kernel_config.brisc_noc_id;
            noc_mode = launch_msg_address->kernel_config.brisc_noc_mode;
            my_relative_x_ = my_logical_x_ - launch_msg_address->kernel_config.sub_device_origin_x;
            my_relative_y_ = my_logical_y_ - launch_msg_address->kernel_config.sub_device_origin_y;

            uint8_t cmd_buf;
            if (noc_mode == DM_DEDICATED_NOC) {
                if (prev_noc_mode != noc_mode) {
                    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
                }

                noc_local_state_init(noc_index);
                cmd_buf = BRISC_AT_CMD_BUF;
            } else {
                if (prev_noc_mode != noc_mode) {
                    dynamic_noc_init();
                }
                dynamic_noc_local_state_init();
                cmd_buf = DYNAMIC_NOC_BRISC_AT_CMD_BUF;
            }
            prev_noc_mode = noc_mode;

            uint32_t tt_l1_ptr* cb_l1_base =
                (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.local_cb_offset);
            start_ncrisc_kernel_run_early(enables);

            WAYPOINT("R");
            int index = static_cast<std::underlying_type<TensixProcessorTypes>::type>(TensixProcessorTypes::DM0);
            if (enables & (1u << index)) {
                uint32_t local_cb_mask = launch_msg_address->kernel_config.local_cb_mask;
                setup_local_cb_read_write_interfaces<true, true, false>(cb_l1_base, 0, local_cb_mask);
                cb_l1_base =
                    (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.remote_cb_offset);
                uint32_t end_cb_index = launch_msg_address->kernel_config.min_remote_cb_start_index;
                experimental::setup_remote_cb_interfaces<true>(cb_l1_base, end_cb_index, noc_index, noc_mode, true,
                                                               cmd_buf);
                barrier_remote_cb_interface_setup(noc_index, end_cb_index);
                start_ncrisc_kernel_run(enables);
                uint32_t kernel_lma =
                    (kernel_config_base + launch_msg_address->kernel_config.kernel_text_offset[index]);
                auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
                record_stack_usage(stack_free);
            } else {
#if defined(PROFILE_KERNEL)

                if (noc_mode == DM_DEDICATED_NOC) {
                    noc_local_state_init(noc_index);
                }
#endif

                if (launch_msg_address->kernel_config.enables) {
                    cb_l1_base =
                        (uint32_t tt_l1_ptr*)(kernel_config_base + launch_msg_address->kernel_config.remote_cb_offset);
                    uint32_t end_cb_index = launch_msg_address->kernel_config.min_remote_cb_start_index;
                    experimental::setup_remote_cb_interfaces<true>(cb_l1_base, end_cb_index, noc_index, noc_mode, true,
                                                                   cmd_buf);
                    barrier_remote_cb_interface_setup(noc_index, end_cb_index);
                }
                start_ncrisc_kernel_run(enables);
                wait_for_go_message();
            }
            WAYPOINT("D");

            wait_ncrisc_trisc();

            trigger_sync_register_init();

            if constexpr (ASSERT_ENABLED) {
                if (noc_mode == DM_DYNAMIC_NOC) {
                    WAYPOINT("NKFW");

                    invalidate_l1_cache();
                    for (int noc = 0; noc < NUM_NOCS; noc++) {
                        ASSERT(ncrisc_dynamic_noc_reads_flushed(noc));
                        ASSERT(ncrisc_dynamic_noc_nonposted_writes_sent(noc));
                        ASSERT(ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
                        ASSERT(ncrisc_dynamic_noc_nonposted_atomics_flushed(noc));
                        ASSERT(ncrisc_dynamic_noc_posted_writes_sent(noc));
                    }
                    WAYPOINT("NKFD");
                }
            }

#if defined(PROFILE_KERNEL)
            if (noc_mode == DM_DYNAMIC_NOC) {
                noc_local_state_init(noc_index);
            }
#endif

            uint32_t go_message_index = mailboxes->go_message_index;
            mailboxes->go_messages[go_message_index].signal = RUN_MSG_DONE;

            if (launch_msg_address->kernel_config.mode == DISPATCH_MODE_DEV) {
                launch_msg_address->kernel_config.enables = 0;
                launch_msg_address->kernel_config.preload = 0;
                uint64_t dispatch_addr = calculate_dispatch_addr(&mailboxes->go_messages[go_message_index]);
                DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
                CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER();
                notify_dispatch_core_done(dispatch_addr, noc_index);
                mailboxes->launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1);
            }
        }
    }

    return 0;
}
