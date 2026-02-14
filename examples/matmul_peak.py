#!/usr/bin/env python3
"""Peak matmul benchmark with 2D multicast — 4-role dataflow split.

Roles follow TTNN-style partitioning with per-role kernels on disjoint core sets:
- NCRISC (NOC0): in0 sender, in0 receiver
- BRISC (NOC1): in1 sender+writer, in1 receiver+writer
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import time
import numpy as np
from codegen import Compiler, DataFormat, CkernelConfig, MathFidelity
from device import Device, Program, DataflowLaunch
from dram import DType

# Matrix dimensions tuned for 10x11 core grid
M, K, N = 5120, 4096, 5632
Mt, Kt, Nt = M // 32, K // 32, N // 32
TILE_BYTES = 32 * 32 * 2

NUM_ROWS, NUM_COLS = 10, 11
PER_CORE_M = Mt // NUM_ROWS
PER_CORE_N = Nt // NUM_COLS

IN0_BLOCK_W = 4
NUM_BLOCKS = Kt // IN0_BLOCK_W
OUT_SUBBLOCK_H, OUT_SUBBLOCK_W = 4, 2
OUT_SUBBLOCK_NUM_TILES = OUT_SUBBLOCK_H * OUT_SUBBLOCK_W
IN0_NUM_SUBBLOCKS = PER_CORE_M // OUT_SUBBLOCK_H
IN1_NUM_SUBBLOCKS = PER_CORE_N // OUT_SUBBLOCK_W
IN0_BLOCK_NUM_TILES = PER_CORE_M * IN0_BLOCK_W
IN0_SUBBLOCK_NUM_TILES = OUT_SUBBLOCK_H * IN0_BLOCK_W
IN1_BLOCK_NUM_TILES = PER_CORE_N * IN0_BLOCK_W
IN1_PER_CORE_W = PER_CORE_N

CB0_PAGES = 2 * IN0_BLOCK_NUM_TILES
CB1_PAGES = 2 * IN1_BLOCK_NUM_TILES
CB16_PAGES = PER_CORE_M * PER_CORE_N
CB24_PAGES = CB16_PAGES
OUT_BLOCK_NUM_TILES = PER_CORE_M * PER_CORE_N

IN0_SEND_SEM, IN0_RECV_SEM = 0, 1
IN1_SEND_SEM, IN1_RECV_SEM = 2, 3
NUM_SEMS = 4

WARMUP_ITERS = 2
TIMED_ITERS = 5

# -- Reader kernel (NCRISC/NOC0): in0 sender --
K_READER_SENDER = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t in0_addr              = get_arg_val<uint32_t>(0);
  uint32_t in0_start_tile_id     = get_arg_val<uint32_t>(1);
  uint32_t in0_stride_w          = get_arg_val<uint32_t>(2);
  uint32_t in0_stride_h          = get_arg_val<uint32_t>(3);
  uint32_t in0_next_block_stride = get_arg_val<uint32_t>(4);
  uint32_t in0_block_w           = get_arg_val<uint32_t>(5);
  uint32_t in0_block_h           = get_arg_val<uint32_t>(6);
  uint32_t in0_block_num_tiles   = get_arg_val<uint32_t>(7);
  uint32_t num_blocks            = get_arg_val<uint32_t>(8);

  uint32_t w_sx = get_arg_val<uint32_t>(9);
  uint32_t w_sy = get_arg_val<uint32_t>(10);
  uint32_t w_ex = get_arg_val<uint32_t>(11);
  uint32_t w_ey = get_arg_val<uint32_t>(12);
  uint32_t w_nd = get_arg_val<uint32_t>(13);

  uint32_t e_sx = get_arg_val<uint32_t>(14);
  uint32_t e_sy = get_arg_val<uint32_t>(15);
  uint32_t e_ex = get_arg_val<uint32_t>(16);
  uint32_t e_ey = get_arg_val<uint32_t>(17);
  uint32_t e_nd = get_arg_val<uint32_t>(18);

  uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(19);
  uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(20);

  uint32_t in0_sender_sem_addr = get_semaphore(get_arg_val<uint32_t>(21));
  uint32_t in0_recv_sem_addr   = get_semaphore(get_arg_val<uint32_t>(22));

  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  const uint32_t tile_bytes = get_tile_size(cb_in0);

  volatile tt_l1_ptr uint32_t* in0_sender_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_sem_addr);
  volatile tt_l1_ptr uint32_t* in0_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_recv_sem_addr);

  *(in0_recv_sem_ptr) = VALID;

  const InterleavedAddrGenFast<true> in0_gen = {{
    .bank_base_address = in0_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};

  uint32_t in0_current_block_start = in0_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {{
    cb_reserve_back(cb_in0, in0_block_num_tiles);

    uint32_t l1_addr = get_write_ptr(cb_in0);
    uint32_t in0_start_address = l1_addr;
    uint32_t in0_row_start = in0_current_block_start;
    uint32_t in0_block_size_bytes = 0;
    for (uint32_t h = 0; h < in0_block_h; h++) {{
      uint32_t tile_id = in0_row_start;
      for (uint32_t w = 0; w < in0_block_w; w++) {{
        noc_async_read_tile(tile_id, in0_gen, l1_addr);
        l1_addr += tile_bytes;
        tile_id += in0_stride_w;
        in0_block_size_bytes += tile_bytes;
      }}
      in0_row_start += in0_stride_h;
    }}
    in0_current_block_start += in0_next_block_stride;
    noc_async_read_barrier();

    noc_semaphore_wait(in0_sender_sem_ptr, w_nd + e_nd);
    noc_semaphore_set(in0_sender_sem_ptr, 0);

    if (w_nd > 0) {{
      uint64_t w_addr = get_noc_multicast_addr(w_sx, w_sy, w_ex, w_ey, in0_start_address);
      noc_async_write_multicast(in0_start_address, w_addr, in0_block_size_bytes, w_nd);
      noc_async_writes_flushed();
      uint64_t w_sem = get_noc_multicast_addr(w_sx, w_sy, w_ex, w_ey, in0_recv_sem_addr);
      noc_semaphore_set_multicast(in0_recv_sem_addr, w_sem, w_nd);
    }}

    if (e_nd > 0) {{
      uint64_t e_addr = get_noc_multicast_addr(e_sx, e_sy, e_ex, e_ey, in0_start_address);
      noc_async_write_multicast(in0_start_address, e_addr, in0_block_size_bytes, e_nd);
      noc_async_writes_flushed();
      uint64_t e_sem = get_noc_multicast_addr(e_sx, e_sy, e_ex, e_ey, in0_recv_sem_addr);
      noc_semaphore_set_multicast(in0_recv_sem_addr, e_sem, e_nd);
    }}

    cb_push_back(cb_in0, in0_block_num_tiles);
  }}
}}
"""

# -- Reader kernel (NCRISC/NOC0): in0 receiver --
K_READER_RECV = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t in0_block_num_tiles   = get_arg_val<uint32_t>(7);
  uint32_t num_blocks            = get_arg_val<uint32_t>(8);
  uint32_t in0_sender_noc_x      = get_arg_val<uint32_t>(19);
  uint32_t in0_sender_noc_y      = get_arg_val<uint32_t>(20);
  uint32_t in0_sender_sem_addr   = get_semaphore(get_arg_val<uint32_t>(21));
  uint32_t in0_recv_sem_addr     = get_semaphore(get_arg_val<uint32_t>(22));

  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  volatile tt_l1_ptr uint32_t* in0_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_recv_sem_addr);

  for (uint32_t block = 0; block < num_blocks; block++) {{
    cb_reserve_back(cb_in0, in0_block_num_tiles);
    noc_semaphore_set(in0_recv_sem_ptr, INVALID);
    uint64_t sender_sem_noc = get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_sem_addr);
    noc_semaphore_inc(sender_sem_noc, 1);
    noc_semaphore_wait(in0_recv_sem_ptr, VALID);
    cb_push_back(cb_in0, in0_block_num_tiles);
  }}
}}
"""

# -- Writer kernel (BRISC/NOC1): in1 sender + output writer --
K_WRITER_SENDER = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t in1_addr              = get_arg_val<uint32_t>(0);
  uint32_t in1_start_tile_id     = get_arg_val<uint32_t>(1);
  uint32_t in1_stride_w          = get_arg_val<uint32_t>(2);
  uint32_t in1_stride_h          = get_arg_val<uint32_t>(3);
  uint32_t in1_next_block_stride = get_arg_val<uint32_t>(4);
  uint32_t in1_block_w           = get_arg_val<uint32_t>(5);
  uint32_t in1_block_h           = get_arg_val<uint32_t>(6);
  uint32_t in1_block_num_tiles   = get_arg_val<uint32_t>(7);
  uint32_t num_blocks            = get_arg_val<uint32_t>(8);

  uint32_t i1_sx = get_arg_val<uint32_t>(9);
  uint32_t i1_sy = get_arg_val<uint32_t>(10);
  uint32_t i1_ex = get_arg_val<uint32_t>(11);
  uint32_t i1_ey = get_arg_val<uint32_t>(12);
  uint32_t i1_nd = get_arg_val<uint32_t>(13);

  uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(14);
  uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(15);

  uint32_t in1_sender_sem_addr = get_semaphore(get_arg_val<uint32_t>(16));
  uint32_t in1_recv_sem_addr   = get_semaphore(get_arg_val<uint32_t>(17));

  uint32_t out_addr            = get_arg_val<uint32_t>(18);
  uint32_t out_start_tile_id   = get_arg_val<uint32_t>(19);
  uint32_t out_stride_w        = get_arg_val<uint32_t>(20);
  uint32_t out_stride_h        = get_arg_val<uint32_t>(21);
  uint32_t out_next_sb_stride_w = get_arg_val<uint32_t>(22);
  uint32_t out_next_sb_stride_h = get_arg_val<uint32_t>(23);
  uint32_t out_subblock_w      = get_arg_val<uint32_t>(24);
  uint32_t out_subblock_h      = get_arg_val<uint32_t>(25);
  uint32_t out_sb_tile_count   = get_arg_val<uint32_t>(26);
  uint32_t out_num_sb_w        = get_arg_val<uint32_t>(27);
  uint32_t out_num_sb_h        = get_arg_val<uint32_t>(28);

  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_in1);

  volatile tt_l1_ptr uint32_t* in1_sender_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_sem_addr);
  volatile tt_l1_ptr uint32_t* in1_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_recv_sem_addr);

  *(in1_recv_sem_ptr) = VALID;

  const InterleavedAddrGenFast<true> in1_gen = {{
    .bank_base_address = in1_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};
  const InterleavedAddrGenFast<true> out_gen = {{
    .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};

  // === in1 block loop ===
  uint32_t in1_current_block_start = in1_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {{
    cb_reserve_back(cb_in1, in1_block_num_tiles);

    uint32_t l1_addr = get_write_ptr(cb_in1);
    uint32_t in1_start_address = l1_addr;
    uint32_t in1_row_start = in1_current_block_start;
    uint32_t in1_block_size_bytes = 0;
    for (uint32_t h = 0; h < in1_block_h; h++) {{
      uint32_t tile_id = in1_row_start;
      for (uint32_t w = 0; w < in1_block_w; w++) {{
        noc_async_read_tile(tile_id, in1_gen, l1_addr);
        l1_addr += tile_bytes;
        tile_id += in1_stride_w;
        in1_block_size_bytes += tile_bytes;
      }}
      in1_row_start += in1_stride_h;
    }}
    in1_current_block_start += in1_next_block_stride;
    noc_async_read_barrier();

    noc_semaphore_wait(in1_sender_sem_ptr, i1_nd);
    noc_semaphore_set(in1_sender_sem_ptr, 0);

    if (i1_nd > 0) {{
      uint64_t mcast_addr = get_noc_multicast_addr(i1_sx, i1_sy, i1_ex, i1_ey, in1_start_address);
      noc_async_write_multicast(in1_start_address, mcast_addr, in1_block_size_bytes, i1_nd);
      noc_async_writes_flushed();
      uint64_t sem_addr = get_noc_multicast_addr(i1_sx, i1_sy, i1_ex, i1_ey, in1_recv_sem_addr);
      noc_semaphore_set_multicast(in1_recv_sem_addr, sem_addr, i1_nd);
    }}

    cb_push_back(cb_in1, in1_block_num_tiles);
  }}

  // === output write ===
  uint32_t sbh_start = out_start_tile_id;
  for (uint32_t sbh = 0; sbh < out_num_sb_h; sbh++) {{
    uint32_t sbw_start = sbh_start;
    for (uint32_t sbw = 0; sbw < out_num_sb_w; sbw++) {{
      cb_wait_front(cb_out, out_sb_tile_count);
      uint32_t l1_addr = get_read_ptr(cb_out);
      uint32_t row_start = sbw_start;
      for (uint32_t h = 0; h < out_subblock_h; h++) {{
        uint32_t tile_id = row_start;
        for (uint32_t w = 0; w < out_subblock_w; w++) {{
          noc_async_write_tile(tile_id, out_gen, l1_addr);
          l1_addr += tile_bytes;
          tile_id += out_stride_w;
        }}
        row_start += out_stride_h;
      }}
      noc_async_write_barrier();
      cb_pop_front(cb_out, out_sb_tile_count);
      sbw_start += out_next_sb_stride_w;
    }}
    sbh_start += out_next_sb_stride_h;
  }}
}}
"""

# -- Writer kernel (BRISC/NOC1): in1 receiver + output writer --
K_WRITER_RECV = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t in1_block_num_tiles   = get_arg_val<uint32_t>(7);
  uint32_t num_blocks            = get_arg_val<uint32_t>(8);
  uint32_t in1_sender_noc_x      = get_arg_val<uint32_t>(14);
  uint32_t in1_sender_noc_y      = get_arg_val<uint32_t>(15);
  uint32_t in1_sender_sem_addr   = get_semaphore(get_arg_val<uint32_t>(16));
  uint32_t in1_recv_sem_addr     = get_semaphore(get_arg_val<uint32_t>(17));

  uint32_t out_addr              = get_arg_val<uint32_t>(18);
  uint32_t out_start_tile_id     = get_arg_val<uint32_t>(19);
  uint32_t out_stride_w          = get_arg_val<uint32_t>(20);
  uint32_t out_stride_h          = get_arg_val<uint32_t>(21);
  uint32_t out_next_sb_stride_w  = get_arg_val<uint32_t>(22);
  uint32_t out_next_sb_stride_h  = get_arg_val<uint32_t>(23);
  uint32_t out_subblock_w        = get_arg_val<uint32_t>(24);
  uint32_t out_subblock_h        = get_arg_val<uint32_t>(25);
  uint32_t out_sb_tile_count     = get_arg_val<uint32_t>(26);
  uint32_t out_num_sb_w          = get_arg_val<uint32_t>(27);
  uint32_t out_num_sb_h          = get_arg_val<uint32_t>(28);

  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_in1);
  const InterleavedAddrGenFast<true> out_gen = {{
    .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};

  volatile tt_l1_ptr uint32_t* in1_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_recv_sem_addr);

  for (uint32_t block = 0; block < num_blocks; block++) {{
    cb_reserve_back(cb_in1, in1_block_num_tiles);
    noc_semaphore_set(in1_recv_sem_ptr, INVALID);
    uint64_t sender_sem_noc = get_noc_addr(in1_sender_noc_x, in1_sender_noc_y, in1_sender_sem_addr);
    noc_semaphore_inc(sender_sem_noc, 1);
    noc_semaphore_wait(in1_recv_sem_ptr, VALID);
    cb_push_back(cb_in1, in1_block_num_tiles);
  }}

  uint32_t sbh_start = out_start_tile_id;
  for (uint32_t sbh = 0; sbh < out_num_sb_h; sbh++) {{
    uint32_t sbw_start = sbh_start;
    for (uint32_t sbw = 0; sbw < out_num_sb_w; sbw++) {{
      cb_wait_front(cb_out, out_sb_tile_count);
      uint32_t l1_addr = get_read_ptr(cb_out);
      uint32_t row_start = sbw_start;
      for (uint32_t h = 0; h < out_subblock_h; h++) {{
        uint32_t tile_id = row_start;
        for (uint32_t w = 0; w < out_subblock_w; w++) {{
          noc_async_write_tile(tile_id, out_gen, l1_addr);
          l1_addr += tile_bytes;
          tile_id += out_stride_w;
        }}
        row_start += out_stride_h;
      }}
      noc_async_write_barrier();
      cb_pop_front(cb_out, out_sb_tile_count);
      sbw_start += out_next_sb_stride_w;
    }}
    sbh_start += out_next_sb_stride_h;
  }}
}}
"""

K_COMPUTE = f"""
#include <cstdint>
#define PACKER_L1_ACC 1
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {{
void MAIN {{
  constexpr uint32_t in0_block_w = {IN0_BLOCK_W};

  constexpr uint32_t in0_num_subblocks = {IN0_NUM_SUBBLOCKS};
  constexpr uint32_t in0_block_num_tiles = {IN0_BLOCK_NUM_TILES};
  constexpr uint32_t in0_subblock_num_tiles = {IN0_SUBBLOCK_NUM_TILES};

  constexpr uint32_t in1_num_subblocks = {IN1_NUM_SUBBLOCKS};
  constexpr uint32_t in1_block_num_tiles = {IN1_BLOCK_NUM_TILES};
  constexpr uint32_t in1_per_core_w = {IN1_PER_CORE_W};

  constexpr uint32_t num_blocks = {NUM_BLOCKS};

  constexpr uint32_t out_subblock_h = {OUT_SUBBLOCK_H};
  constexpr uint32_t out_subblock_w = {OUT_SUBBLOCK_W};
  constexpr uint32_t out_subblock_num_tiles = {OUT_SUBBLOCK_NUM_TILES};
  constexpr uint32_t out_block_num_tiles = {OUT_BLOCK_NUM_TILES};

  constexpr uint32_t transpose = 0;

  mm_block_init(
    tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16,
    transpose, out_subblock_w, out_subblock_h, in0_block_w);

  bool enable_reload = false;
  uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

  for (uint32_t block = 0; block < num_blocks; block++) {{
    const bool last_out = (block == (num_blocks - 1));

    cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
    cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);

    int in0_index_subblock_offset = 0;
    for (uint32_t in0_sb = 0; in0_sb < in0_num_subblocks; in0_sb++) {{
      int in1_index_subblock_offset = 0;
      for (uint32_t in1_sb = 0; in1_sb < in1_num_subblocks; in1_sb++) {{
        tile_regs_acquire();

        if (enable_reload) {{
          copy_tile_to_dst_init_short(tt::CBIndex::c_24);
          cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
          #pragma GCC unroll 0
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{
            copy_tile(tt::CBIndex::c_24, i, i);
          }}
          cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);

          mm_block_init_short(
            tt::CBIndex::c_0, tt::CBIndex::c_1,
            transpose, out_subblock_w, out_subblock_h, in0_block_w);
        }}

        #pragma GCC unroll 0
        for (uint32_t inner = 0; inner < in0_block_w; inner++) {{
          const uint32_t in0_tile_index = (uint32_t)(in0_index_subblock_offset + (int)inner);
          const uint32_t in1_tile_index = (uint32_t)(in1_index_subblock_offset + (int)(inner * in1_per_core_w));

          matmul_block(
            tt::CBIndex::c_0, tt::CBIndex::c_1,
            in0_tile_index, in1_tile_index,
            0, transpose, out_subblock_w, out_subblock_h, in0_block_w);
        }}
        tile_regs_commit();
        if (last_out) {{
          cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
          tile_regs_wait();
          PACK((llk_pack_reconfig_l1_acc(0)));
          #pragma GCC unroll 0
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{
            pack_tile(i, tt::CBIndex::c_16);
          }}
          tile_regs_release();
          cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
        }} else {{
          if (block == 0) {{
            cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
            out_num_tiles_to_wait += out_subblock_num_tiles;
          }}
          cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
          tile_regs_wait();
          if (block == 0) {{
            PACK((llk_pack_reconfig_l1_acc(0)));
          }} else if (block == 1) {{
            PACK((llk_pack_reconfig_l1_acc(1)));
          }}
          #pragma GCC unroll 0
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{
            pack_tile(i, tt::CBIndex::c_24);
          }}
          tile_regs_release();
          cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
        }}

        in1_index_subblock_offset += out_subblock_w;
      }}
      in0_index_subblock_offset += in0_subblock_num_tiles;
    }}

    if (block < num_blocks - 2) {{
      cb_wait_front(tt::CBIndex::c_24, out_block_num_tiles);
      cb_pop_front(tt::CBIndex::c_24, out_block_num_tiles);
    }}
    if (block == num_blocks - 2) enable_reload = true;

    cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
    cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
  }}
}}
}} // namespace NAMESPACE
"""


def _build_grid(dispatchable_cores: list[tuple[int, int]]):
  xs = sorted({x for x, _ in dispatchable_cores})
  ys = sorted({y for _, y in dispatchable_cores})
  core_set = set(dispatchable_cores)
  cols = xs[:NUM_COLS]
  rows = ys[:NUM_ROWS]
  grid = []
  for y in rows:
    row = []
    for x in cols:
      assert (x, y) in core_set, f"core ({x},{y}) not in dispatchable_cores"
      row.append((x, y))
    grid.append(row)
  return grid, cols, rows


def _mcast_rect_args(x_list: list[int], y: int):
  if not x_list:
    return (0, 0, 0, 0, 0)
  return (min(x_list), y, max(x_list), y, len(x_list))

def _f16_to_bf16_bytes(x: np.ndarray) -> bytes:
  # Convert f16 values to f32 first, then truncate mantissa to bf16 encoding.
  u32 = np.ascontiguousarray(x, dtype=np.float16).astype(np.float32).view(np.uint32)
  return (u32 >> 16).astype(np.uint16).tobytes()

def _bf16_bytes_to_f32(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
  u16 = np.frombuffer(data, dtype=np.uint16)
  u32 = (u16.astype(np.uint32) << 16)
  return u32.view(np.float32).reshape(shape)

def _validate_matmul(a: np.ndarray, b: np.ndarray, c_bytes: bytes):
  c_ref = a @ b
  c_got = _bf16_bytes_to_f32(c_bytes, (M, N))
  ref_flat = c_ref.reshape(-1)
  got_flat = c_got.reshape(-1)
  finite_mask = np.isfinite(got_flat)
  if not np.all(finite_mask):
    bad = int(got_flat.size - np.count_nonzero(finite_mask))
    raise SystemExit(f"Validation failed: output has {bad} non-finite values")
  pcc = float(np.corrcoef(ref_flat, got_flat)[0, 1])
  rel_l2 = float(np.linalg.norm(got_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-12))
  max_abs = float(np.max(np.abs(got_flat - ref_flat)))
  print(f"Validation: PCC={pcc:.6f}, rel_l2={rel_l2:.6f}, max_abs={max_abs:.6f}")
  if pcc < 0.995 or rel_l2 > 0.08:
    raise SystemExit(f"Validation failed: PCC={pcc:.6f}, rel_l2={rel_l2:.6f}")


def main():
  print(f"Matmul Peak (4-role split): C[{M},{N}] = A[{M},{K}] @ B[{K},{N}] (bf16, HiFi2)")
  print(f"  Mt={Mt} Kt={Kt} Nt={Nt} grid={NUM_ROWS}x{NUM_COLS}")
  print(f"  per_core_M={PER_CORE_M} per_core_N={PER_CORE_N} in0_block_w={IN0_BLOCK_W} num_blocks={NUM_BLOCKS}")
  print(f"  subblock: {OUT_SUBBLOCK_H}h x {OUT_SUBBLOCK_W}w = {OUT_SUBBLOCK_NUM_TILES} tiles")

  cfg = CkernelConfig(
    input_format=DataFormat.Float16_b,
    output_format=DataFormat.Float16_b,
    math_fidelity=MathFidelity.HiFi2,
  )
  compiler = Compiler(cfg)
  reader_sender = compiler.compile_dataflow(K_READER_SENDER, processor="ncrisc")
  reader_recv = compiler.compile_dataflow(K_READER_RECV, processor="ncrisc")
  writer_sender = compiler.compile_dataflow(K_WRITER_SENDER, processor="brisc")
  writer_recv = compiler.compile_dataflow(K_WRITER_RECV, processor="brisc")
  compute = compiler.compile_compute(K_COMPUTE)

  device = Device()
  num_cores = len(device.dispatchable_cores)
  print(f"Device: {num_cores} dispatchable cores")
  assert num_cores >= NUM_ROWS * NUM_COLS, f"need {NUM_ROWS * NUM_COLS} cores, have {num_cores}"

  try:
    grid, cols, rows = _build_grid(device.dispatchable_cores)
    active_cores = [grid[r][c] for r in range(NUM_ROWS) for c in range(NUM_COLS)]
    print(f"Grid columns: {cols}")
    print(f"Grid rows: {rows}")

    west_cols = [x for x in cols if x < 8]
    east_cols = [x for x in cols if x >= 10]

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(123)
    a_src = rng_a.uniform(-0.5, 0.5, size=(M, K)).astype(np.float16)
    b_src = rng_b.uniform(-0.5, 0.5, size=(K, N)).astype(np.float16)
    a_bytes = _f16_to_bf16_bytes(a_src)
    b_bytes = _f16_to_bf16_bytes(b_src)
    a_ref = _bf16_bytes_to_f32(a_bytes, (M, K))
    b_ref = _bf16_bytes_to_f32(b_bytes, (K, N))

    a_buf = device.dram.alloc_write(
      a_bytes, name="A", page_size=TILE_BYTES, dtype=DType.bfloat16, shape=(M, K)
    )
    b_buf = device.dram.alloc_write(
      b_bytes, name="B", page_size=TILE_BYTES, dtype=DType.bfloat16, shape=(K, N)
    )
    c_buf = device.dram.alloc(
      TILE_BYTES * Mt * Nt, name="C", page_size=TILE_BYTES, dtype=DType.bfloat16, shape=(M, N)
    )

    core_to_rc = {grid[r][c]: (r, c) for r in range(NUM_ROWS) for c in range(NUM_COLS)}
    top_left = [grid[0][0]]
    top_row = [grid[0][c] for c in range(1, NUM_COLS)]
    left_col = [grid[r][0] for r in range(1, NUM_ROWS)]
    interior = [grid[r][c] for r in range(1, NUM_ROWS) for c in range(1, NUM_COLS)]

    def reader_args(role_idx: int, core_xy: tuple[int, int], num_cores: int) -> list[int]:
      row_idx, col_idx = core_to_rc[core_xy]
      y = core_xy[1]

      # in0 mcast: sender at col 0, west and east receiver rects
      sender_x = cols[0]
      recv_west = [c for c in west_cols if c != sender_x]
      recv_east = list(east_cols)
      w_sx, w_sy, w_ex, w_ey, w_nd = _mcast_rect_args(recv_west, y)
      e_sx, e_sy, e_ex, e_ey, e_nd = _mcast_rect_args(recv_east, y)
      # NOC0 mcast: start=min, end=max (no reversal needed)

      sender_xy = grid[row_idx][0]
      return [
        a_buf.addr, row_idx * PER_CORE_M * Kt, 1, Kt, IN0_BLOCK_W,
        IN0_BLOCK_W, PER_CORE_M, IN0_BLOCK_NUM_TILES, NUM_BLOCKS,
        w_sx, w_sy, w_ex, w_ey, w_nd,
        e_sx, e_sy, e_ex, e_ey, e_nd,
        sender_xy[0], sender_xy[1],
        IN0_SEND_SEM, IN0_RECV_SEM,
      ]

    def writer_args(role_idx: int, core_xy: tuple[int, int], num_cores: int) -> list[int]:
      row_idx, col_idx = core_to_rc[core_xy]
      x = core_xy[0]
      # in1 mcast: sender at row 0, receivers at rows 1..N-1
      recv_ys = rows[1:]
      if recv_ys:
        # NOC1 mcast: start=max, end=min (reversed for NOC1 scan direction)
        i1_sx, i1_sy = x, max(recv_ys)
        i1_ex, i1_ey = x, min(recv_ys)
        i1_nd = len(recv_ys)
      else:
        i1_sx = i1_sy = i1_ex = i1_ey = i1_nd = 0

      sender_xy = grid[0][col_idx]
      out_start = row_idx * PER_CORE_M * Nt + col_idx * PER_CORE_N
      return [
        b_buf.addr, col_idx * PER_CORE_N, 1, Nt, IN0_BLOCK_W * Nt,
        PER_CORE_N, IN0_BLOCK_W, IN1_BLOCK_NUM_TILES, NUM_BLOCKS,
        i1_sx, i1_sy, i1_ex, i1_ey, i1_nd,
        sender_xy[0], sender_xy[1],
        IN1_SEND_SEM, IN1_RECV_SEM,
        c_buf.addr, out_start, 1, Nt,
        OUT_SUBBLOCK_W, OUT_SUBBLOCK_H * Nt,
        OUT_SUBBLOCK_W, OUT_SUBBLOCK_H,
        OUT_SUBBLOCK_NUM_TILES,
        IN1_NUM_SUBBLOCKS, IN0_NUM_SUBBLOCKS,
      ]

    def compute_args(core_idx: int, core_xy: tuple[int, int], n_cores: int) -> list[int]:
      return []

    dataflow: list[DataflowLaunch] = []
    for cores, reader_k, writer_k in (
      (top_left, reader_sender, writer_sender),
      (top_row, reader_recv, writer_sender),
      (left_col, reader_sender, writer_recv),
      (interior, reader_recv, writer_recv),
    ):
      if not cores:
        continue
      dataflow.append(
        DataflowLaunch(
          cores=cores,
          reader=reader_k,
          writer=writer_k,
          reader_rt_args=reader_args,
          writer_rt_args=writer_args,
        )
      )

    program = Program(
      dataflow=dataflow,
      compute=compute,
      compute_rt_args=compute_args,
      cbs=[0, 1, 16, 24],
      tile_size=TILE_BYTES,
      num_pages=CB0_PAGES,
      cores=len(active_cores),
      num_sems=NUM_SEMS,
      cb_config={
        0:  (CB0_PAGES, TILE_BYTES),
        1:  (CB1_PAGES, TILE_BYTES),
        16: (CB16_PAGES, TILE_BYTES),
        24: (CB24_PAGES, TILE_BYTES),
      },
    )

    print(f"\nWarmup ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
      device.queue(program)
    device.run()

    print(f"Timing ({TIMED_ITERS} iters)...")
    for _ in range(TIMED_ITERS):
      device.queue(program)
    t0 = time.perf_counter()
    device.run()
    elapsed_batch = time.perf_counter() - t0
    elapsed_wall = elapsed_batch / TIMED_ITERS

    flops = 2.0 * M * N * K
    tflops_wall = flops / elapsed_wall / 1e12
    print(f"\nAvg latency (wall):    {elapsed_wall * 1e3:.3f} ms")
    print(f"Throughput (wall):     {tflops_wall:.2f} TFLOPS (HiFi2 bf16, {NUM_ROWS * NUM_COLS} cores)")
    c_rm = device.dram.read(c_buf)
    _validate_matmul(a_ref, b_ref, c_rm)

  finally:
    device.close()

if __name__ == "__main__":
  main()
