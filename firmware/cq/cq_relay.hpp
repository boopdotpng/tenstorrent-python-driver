// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "cq_common.hpp"
#include "internal/risc_attribs.h"
#include "api/debug/waypoint.h"
#include "noc/noc_parameters.h"

#if !defined(FD_CORE_TYPE)
#define FD_CORE_TYPE 0
#endif

class CQRelayClient {
private:
  constexpr static ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);

public:
  CQRelayClient() = default;

  template <uint8_t noc_index, uint8_t downstream_cmd_buf>
  FORCE_INLINE void init(uint64_t downstream_noc_addr) {
    WAYPOINT("FMCW");
    init_write_state_only<noc_index, downstream_cmd_buf>(downstream_noc_addr);
    WAYPOINT("FMCD");
  }

  template <uint8_t noc_index, uint8_t downstream_cmd_buf>
  FORCE_INLINE void init_write_state_only(uint64_t downstream_noc_addr) {
    cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, downstream_cmd_buf>(0, downstream_noc_addr, 0, noc_index);
  }

  template <uint8_t noc_index>
  FORCE_INLINE void init_inline_write_state_only(uint64_t downstream_noc_addr) {
    cq_noc_inline_dw_write_init_state<CQ_NOC_INLINE_Ndvb>(downstream_noc_addr);
  }

  template <uint8_t noc_index, uint64_t noc_xy, uint32_t sem_id>
  FORCE_INLINE void teardown() {
    constexpr uint32_t k_PacketQueueTeardownFlag = 0x80000000;
    noc_semaphore_inc(
      get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), k_PacketQueueTeardownFlag, noc_index);
  }

  template <uint8_t noc_idx, bool count = true>
  FORCE_INLINE void write_inline(uint64_t dst, uint32_t val) {
    cq_noc_inline_dw_write_with_state<CQ_NOC_INLINE_nDVB>(dst, val, 0xF, noc_idx);
    if constexpr (count) {
      noc_nonposted_writes_num_issued[noc_idx]++;
      noc_nonposted_writes_acked[noc_idx]++;
    }
  }

  template <uint8_t noc_idx, bool wait, uint8_t downstream_cmd_buf, bool count = true>
  FORCE_INLINE void write_any_len(uint32_t data_ptr, uint64_t dst_ptr, uint32_t length) {
    if constexpr (wait) {
      cq_noc_async_write_with_state_any_len<true, count, CQNocWait::CQ_NOC_WAIT, downstream_cmd_buf>(
        data_ptr, dst_ptr, length, 1, noc_idx);
    } else {
      cq_noc_async_write_with_state_any_len<true, count, CQNocWait::CQ_NOC_wait, downstream_cmd_buf>(
        data_ptr, dst_ptr, length, 1, noc_idx);
    }
  }

  template <uint8_t noc_idx, bool wait, uint8_t downstream_cmd_buf, bool count = true>
  FORCE_INLINE void write(uint32_t data_ptr, uint64_t dst_ptr, uint32_t length) {
    if constexpr (wait) {
      cq_noc_async_write_with_state<CQ_NOC_SnDL, CQNocWait::CQ_NOC_WAIT, CQNocSend::CQ_NOC_SEND, downstream_cmd_buf>(
        data_ptr, dst_ptr, length, 1, noc_idx);
    } else {
      cq_noc_async_write_with_state<CQ_NOC_SnDL, CQNocWait::CQ_NOC_wait, CQNocSend::CQ_NOC_SEND, downstream_cmd_buf>(
        data_ptr, dst_ptr, length, 1, noc_idx);
    }
    if (count) {
      noc_nonposted_writes_num_issued[noc_idx]++;
      noc_nonposted_writes_acked[noc_idx]++;
    }
  }

  template <uint8_t noc_idx, uint32_t dest_noc_xy, uint32_t dest_sem_id>
  FORCE_INLINE void release_pages(uint32_t n) {
    noc_semaphore_inc(get_noc_addr_helper(dest_noc_xy, get_semaphore<fd_core_type>(dest_sem_id)), n, noc_idx);
  }

  template <uint32_t downstream_noc_idx, uint32_t downstream_noc_xy, uint32_t downstream_sem_id, bool wait, uint8_t downstream_cmd_buf>
  FORCE_INLINE void write_atomic_inc_any_len(uint32_t data_ptr, uint64_t dst_ptr, uint32_t length, uint32_t n) {
    write_any_len<downstream_noc_idx, wait, downstream_cmd_buf>(data_ptr, dst_ptr, length);
    release_pages<downstream_noc_idx, downstream_noc_xy, downstream_sem_id>(n);
  }
};
