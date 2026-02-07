#!/usr/bin/env python3
"""Peak matmul benchmark with 2D multicast — port of tt-metal's matmul_peak_fast_dispatch_v2.

Uses semaphore-synchronized multicast: left column reads A and mcasts right,
top row reads B and mcasts down. Interior cores receive both via NOC mcast.
Target: >135 TFLOPS on P100A (110 active cores, LoFi bf16).
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import time
from codegen import Compiler, DataFormat, CkernelConfig, MathFidelity
from device import Device, Program

# Matrix dimensions (same as tt-metal v2)
M, K, N = 5120, 4096, 5632
Mt, Kt, Nt = M // 32, K // 32, N // 32
TILE_BYTES = 32 * 32 * 2  # bf16

# Grid: 10 rows × 11 cols = 110 cores
NUM_ROWS, NUM_COLS = 10, 11
PER_CORE_M = Mt // NUM_ROWS   # 16
PER_CORE_N = Nt // NUM_COLS   # 16

# Blocking params
IN0_BLOCK_W = 4
NUM_BLOCKS = Kt // IN0_BLOCK_W  # 64
OUT_SUBBLOCK_H, OUT_SUBBLOCK_W = 4, 2
OUT_SUBBLOCK_NUM_TILES = OUT_SUBBLOCK_H * OUT_SUBBLOCK_W  # 8
IN0_NUM_SUBBLOCKS = PER_CORE_M // OUT_SUBBLOCK_H   # 4
IN1_NUM_SUBBLOCKS = PER_CORE_N // OUT_SUBBLOCK_W   # 8
IN0_BLOCK_NUM_TILES = PER_CORE_M * IN0_BLOCK_W     # 32
IN0_SUBBLOCK_NUM_TILES = OUT_SUBBLOCK_H * IN0_BLOCK_W  # 8
IN1_BLOCK_NUM_TILES = PER_CORE_N * IN0_BLOCK_W     # 32
IN1_PER_CORE_W = PER_CORE_N                        # 16

# CB sizes (double-buffered)
CB0_PAGES = 2 * IN0_BLOCK_NUM_TILES   # 64  (128KB)
CB1_PAGES = 2 * IN1_BLOCK_NUM_TILES   # 64  (128KB)
CB16_PAGES = PER_CORE_M * PER_CORE_N  # 256 (512KB, shared with CB24)
CB24_PAGES = CB16_PAGES

# Semaphore IDs
IN0_SEND_SEM, IN0_RECV_SEM = 0, 1
IN1_SEND_SEM, IN1_RECV_SEM = 2, 3
NUM_SEMS = 4

WARMUP_ITERS = 2
TIMED_ITERS = 5

# -- Reader kernel: unified sender/receiver via rt_args flags --
K_READER = f"""
#include <cstdint>

void kernel_main() {{

  // in0 DRAM args
  uint32_t in0_addr              = get_arg_val<uint32_t>(0);
  uint32_t in0_start_tile_id     = get_arg_val<uint32_t>(1);
  uint32_t in0_stride_w          = get_arg_val<uint32_t>(2);
  uint32_t in0_stride_h          = get_arg_val<uint32_t>(3);
  uint32_t in0_next_block_stride = get_arg_val<uint32_t>(4);
  uint32_t in0_block_w           = get_arg_val<uint32_t>(5);
  uint32_t in0_block_h           = get_arg_val<uint32_t>(6);
  uint32_t in0_block_num_tiles   = get_arg_val<uint32_t>(7);

  // in1 DRAM args
  uint32_t in1_addr              = get_arg_val<uint32_t>(8);
  uint32_t in1_start_tile_id     = get_arg_val<uint32_t>(9);
  uint32_t in1_stride_w          = get_arg_val<uint32_t>(10);
  uint32_t in1_stride_h          = get_arg_val<uint32_t>(11) ;
  uint32_t in1_next_block_stride = get_arg_val<uint32_t>(12);
  uint32_t in1_block_w           = get_arg_val<uint32_t>(13);
  uint32_t in1_block_h           = get_arg_val<uint32_t>(14);
  uint32_t in1_block_num_tiles   = get_arg_val<uint32_t>(15);

  uint32_t num_blocks            = get_arg_val<uint32_t>(16);

  // in0 mcast args (west rect)
  uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(17);
  uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(18);
  uint32_t in0_mcast_dest_noc_end_x   = get_arg_val<uint32_t>(19);
  uint32_t in0_mcast_dest_noc_end_y   = get_arg_val<uint32_t>(20);
  uint32_t in0_mcast_num_dests        = get_arg_val<uint32_t>(21);
  uint32_t in0_mcast_sender_noc_x     = get_arg_val<uint32_t>(22);
  uint32_t in0_mcast_sender_noc_y     = get_arg_val<uint32_t>(23);

  // in0 mcast east rect (if gap)
  uint32_t in0_east_start_x           = get_arg_val<uint32_t>(24);
  uint32_t in0_east_start_y           = get_arg_val<uint32_t>(25);
  uint32_t in0_east_end_x             = get_arg_val<uint32_t>(26);
  uint32_t in0_east_end_y             = get_arg_val<uint32_t>(27);
  uint32_t in0_east_num_dests         = get_arg_val<uint32_t>(28);

  // in1 mcast args
  uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(29);
  uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(30);
  uint32_t in1_mcast_dest_noc_end_x   = get_arg_val<uint32_t>(31);
  uint32_t in1_mcast_dest_noc_end_y   = get_arg_val<uint32_t>(32);
  uint32_t in1_mcast_num_dests        = get_arg_val<uint32_t>(33);
  uint32_t in1_mcast_sender_noc_x     = get_arg_val<uint32_t>(34);
  uint32_t in1_mcast_sender_noc_y     = get_arg_val<uint32_t>(35);

  // Flags
  uint32_t is_in0_sender = get_arg_val<uint32_t>(36);
  uint32_t is_in1_sender = get_arg_val<uint32_t>(37);

  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(cb_in0);

  // Semaphore addresses
  uint32_t in0_sender_sem_addr  = get_semaphore({IN0_SEND_SEM});
  uint32_t in0_recv_sem_addr    = get_semaphore({IN0_RECV_SEM});
  uint32_t in1_sender_sem_addr  = get_semaphore({IN1_SEND_SEM});
  uint32_t in1_recv_sem_addr    = get_semaphore({IN1_RECV_SEM});

  volatile tt_l1_ptr uint32_t* in0_sender_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_sem_addr);
  volatile tt_l1_ptr uint32_t* in0_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_recv_sem_addr);
  volatile tt_l1_ptr uint32_t* in1_sender_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_sem_addr);
  volatile tt_l1_ptr uint32_t* in1_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_recv_sem_addr);

  // Sender: set local recv_sem to VALID (will be mcast to receivers)
  if (is_in0_sender) *(in0_recv_sem_ptr) = VALID;
  if (is_in1_sender) *(in1_recv_sem_ptr) = VALID;

  const InterleavedAddrGenFast<true> in0_gen = {{
    .bank_base_address = in0_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};
  const InterleavedAddrGenFast<true> in1_gen = {{
    .bank_base_address = in1_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};

  uint32_t in0_current_block_start = in0_start_tile_id;
  uint32_t in1_current_block_start = in1_start_tile_id;

  for (uint32_t block = 0; block < num_blocks; block++) {{
    // === in0 (A matrix) ===
    cb_reserve_back(cb_in0, in0_block_num_tiles);

    if (is_in0_sender) {{
      // Read in0 block from DRAM
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

      // Wait for all receivers to be ready
      noc_semaphore_wait(in0_sender_sem_ptr, in0_mcast_num_dests + in0_east_num_dests);
      noc_semaphore_set(in0_sender_sem_ptr, 0);

      // Multicast to west rect
      if (in0_mcast_num_dests > 0) {{
        uint64_t mcast_addr = get_noc_multicast_addr(
          in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y,
          in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y,
          in0_start_address);
        noc_async_write_multicast(in0_start_address, mcast_addr, in0_block_size_bytes, in0_mcast_num_dests);
        noc_async_writes_flushed();
        uint64_t sem_mcast_addr = get_noc_multicast_addr(
          in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y,
          in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y,
          in0_recv_sem_addr);
        noc_semaphore_set_multicast(in0_recv_sem_addr, sem_mcast_addr, in0_mcast_num_dests);
      }}

      // Multicast to east rect (if present)
      if (in0_east_num_dests > 0) {{
        uint64_t mcast_addr = get_noc_multicast_addr(
          in0_east_start_x, in0_east_start_y,
          in0_east_end_x, in0_east_end_y,
          in0_start_address);
        noc_async_write_multicast(in0_start_address, mcast_addr, in0_block_size_bytes, in0_east_num_dests);
        noc_async_writes_flushed();
        uint64_t sem_mcast_addr = get_noc_multicast_addr(
          in0_east_start_x, in0_east_start_y,
          in0_east_end_x, in0_east_end_y,
          in0_recv_sem_addr);
        noc_semaphore_set_multicast(in0_recv_sem_addr, sem_mcast_addr, in0_east_num_dests);
      }}
    }} else {{
      // Receiver: signal ready, wait for data
      noc_semaphore_set(in0_recv_sem_ptr, INVALID);
      uint64_t sender_sem_noc = get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_sender_sem_addr);
      noc_semaphore_inc(sender_sem_noc, 1);
      noc_semaphore_wait(in0_recv_sem_ptr, VALID);
    }}

    cb_push_back(cb_in0, in0_block_num_tiles);

    // === in1 (B matrix) ===
    cb_reserve_back(cb_in1, in1_block_num_tiles);

    if (is_in1_sender) {{
      // Read in1 block from DRAM
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

      // Wait for all receivers to be ready
      noc_semaphore_wait(in1_sender_sem_ptr, in1_mcast_num_dests);
      noc_semaphore_set(in1_sender_sem_ptr, 0);

      // Multicast to column (single rect, no gap in Y)
      uint64_t mcast_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x, in1_mcast_dest_noc_end_y,
        in1_start_address);
      noc_async_write_multicast(in1_start_address, mcast_addr, in1_block_size_bytes, in1_mcast_num_dests);
      noc_async_writes_flushed();
      uint64_t sem_mcast_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x, in1_mcast_dest_noc_end_y,
        in1_recv_sem_addr);
      noc_semaphore_set_multicast(in1_recv_sem_addr, sem_mcast_addr, in1_mcast_num_dests);
    }} else {{
      // Receiver: signal ready, wait for data
      noc_semaphore_set(in1_recv_sem_ptr, INVALID);
      uint64_t sender_sem_noc = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_sender_sem_addr);
      noc_semaphore_inc(sender_sem_noc, 1);
      noc_semaphore_wait(in1_recv_sem_ptr, VALID);
    }}

    cb_push_back(cb_in1, in1_block_num_tiles);
  }}
}}
"""

# -- Writer: 2D subblock output --
K_WRITER = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t out_addr            = get_arg_val<uint32_t>(0);
  uint32_t out_start_tile_id   = get_arg_val<uint32_t>(1);
  uint32_t out_stride_w        = get_arg_val<uint32_t>(2);
  uint32_t out_stride_h        = get_arg_val<uint32_t>(3);
  uint32_t out_next_sb_stride_w = get_arg_val<uint32_t>(4);
  uint32_t out_next_sb_stride_h = get_arg_val<uint32_t>(5);
  uint32_t out_subblock_w      = get_arg_val<uint32_t>(6);
  uint32_t out_subblock_h      = get_arg_val<uint32_t>(7);
  uint32_t out_sb_tile_count   = get_arg_val<uint32_t>(8);
  uint32_t out_num_sb_w        = get_arg_val<uint32_t>(9);
  uint32_t out_num_sb_h        = get_arg_val<uint32_t>(10);

  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_out);
  const InterleavedAddrGenFast<true> out_gen = {{
    .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};

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

# -- Compute: block matmul with subblocks and spill/reload --
K_COMPUTE = f"""
#include <cstdint>
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

  mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

  bool spill = num_blocks > 1;
  bool enable_reload = false;
  uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

  for (uint32_t block = 0; block < num_blocks; block++) {{
    bool last_out = block == (num_blocks - 1);

    cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
    cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);

    int in0_index_subblock_offset = 0;
    for (uint32_t in0_sb = 0; in0_sb < in0_num_subblocks; in0_sb++) {{
      int in1_index_subblock_offset = 0;
      for (uint32_t in1_sb = 0; in1_sb < in1_num_subblocks; in1_sb++) {{
        acquire_dst();

        if (enable_reload) {{
          copy_tile_to_dst_init_short(tt::CBIndex::c_24);
          cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            copy_tile(tt::CBIndex::c_24, i, i);
          cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
          mm_init_short(tt::CBIndex::c_0, tt::CBIndex::c_1);
        }}

        int dst_index = 0;
        int in0_index_h_offset = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {{
          for (uint32_t w = 0; w < out_subblock_w; w++) {{
            int in1_index_inner_dim_offset = 0;
            for (uint32_t inner = 0; inner < in0_block_w; inner++) {{
              int in0_idx = in0_index_subblock_offset + in0_index_h_offset + inner;
              int in1_idx = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
              matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, in0_idx, in1_idx, dst_index);
              in1_index_inner_dim_offset += in1_per_core_w;
            }}
            dst_index++;
          }}
          in0_index_h_offset += in0_block_w;
        }}

        if (last_out) {{
          cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            pack_tile(i, tt::CBIndex::c_16);
          cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
        }} else {{
          if (block == 0) {{
            cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
            out_num_tiles_to_wait += out_subblock_num_tiles;
          }}
          cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            pack_tile(i, tt::CBIndex::c_24);
          cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
        }}

        release_dst();
        in1_index_subblock_offset += out_subblock_w;
      }}
      in0_index_subblock_offset += in0_subblock_num_tiles;
    }}

    if (spill) enable_reload = true;
    cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
    cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
  }}
}}
}}  // namespace NAMESPACE
"""


def _build_grid(dispatchable_cores: list[tuple[int, int]]):
  """Pick NUM_ROWS × NUM_COLS cores from dispatchable_cores.
  Returns grid[row][col] = (x, y) with sorted physical coordinates."""
  xs = sorted({x for x, _ in dispatchable_cores})
  ys = sorted({y for _, y in dispatchable_cores})
  core_set = set(dispatchable_cores)
  # Pick first NUM_COLS columns and NUM_ROWS rows that have all cores present
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
  """Build mcast rect args for a horizontal mcast (same row).
  Returns (start_x, start_y, end_x, end_y, num_dests).
  get_noc_multicast_addr uses NOC 0 order: start=min, end=max."""
  if not x_list:
    return (0, 0, 0, 0, 0)
  return (min(x_list), y, max(x_list), y, len(x_list))


def main():
  print(f"Matmul Peak v2: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}] (bf16, LoFi)")
  print(f"  Mt={Mt} Kt={Kt} Nt={Nt} grid={NUM_ROWS}×{NUM_COLS}")
  print(f"  per_core_M={PER_CORE_M} per_core_N={PER_CORE_N} in0_block_w={IN0_BLOCK_W} num_blocks={NUM_BLOCKS}")
  print(f"  subblock: {OUT_SUBBLOCK_H}h×{OUT_SUBBLOCK_W}w = {OUT_SUBBLOCK_NUM_TILES} tiles")

  cfg = CkernelConfig(
    input_format=DataFormat.Float16_b,
    output_format=DataFormat.Float16_b,
    math_fidelity=MathFidelity.LoFi,
  )
  kernels = Compiler(cfg).compile(K_READER, K_WRITER, K_COMPUTE)

  device = Device()
  num_cores = len(device.dispatchable_cores)
  print(f"Device: {num_cores} dispatchable cores")
  assert num_cores >= NUM_ROWS * NUM_COLS, f"need {NUM_ROWS * NUM_COLS} cores, have {num_cores}"

  try:
    grid, cols, rows = _build_grid(device.dispatchable_cores)
    active_cores = [grid[r][c] for r in range(NUM_ROWS) for c in range(NUM_COLS)]
    print(f"Grid columns: {cols}")
    print(f"Grid rows: {rows}")

    # Split columns into west (x<8) and east (x>=10) groups
    west_cols = [x for x in cols if x < 8]
    east_cols = [x for x in cols if x >= 10]

    a_buf = device.dram.alloc(TILE_BYTES * Mt * Kt, name="A", page_size=TILE_BYTES)
    b_buf = device.dram.alloc(TILE_BYTES * Kt * Nt, name="B", page_size=TILE_BYTES)
    c_buf = device.dram.alloc(TILE_BYTES * Mt * Nt, name="C", page_size=TILE_BYTES)

    def reader_args(core_idx, core_xy, n_cores):
      col_idx = core_idx % NUM_COLS
      row_idx = core_idx // NUM_COLS
      x, y = core_xy

      is_in0_sender = 1 if col_idx == 0 else 0  # left column sends A right
      is_in1_sender = 1 if row_idx == 0 else 0  # top row sends B down

      # in0 DRAM read params (only used by sender, but set for all)
      in0_row = row_idx * PER_CORE_M  # which Mt row this core starts at
      in0_start_tile = in0_row * Kt  # tile index in A
      in0_stride_w = 1       # adjacent K tiles
      in0_stride_h = Kt      # next row of M
      in0_next_block_stride = IN0_BLOCK_W  # advance K by block width

      # in1 DRAM read params
      in1_col = col_idx * PER_CORE_N  # which Nt column this core starts at
      in1_start_tile = in1_col  # tile index in B (first K-row)
      in1_stride_w = 1         # adjacent N tiles
      in1_stride_h = Nt        # next row of K
      in1_next_block_stride = IN0_BLOCK_W * Nt  # advance K by block width

      # in0 horizontal mcast: sender at col 0, receivers at cols 1..NUM_COLS-1
      sender_x = cols[0]
      # Split receiver cols into west and east (excluding sender col)
      recv_west = [c for c in west_cols if c != sender_x]
      recv_east = list(east_cols)

      # West rect
      w_sx, w_sy, w_ex, w_ey, w_nd = _mcast_rect_args(recv_west, y)
      # East rect
      e_sx, e_sy, e_ex, e_ey, e_nd = _mcast_rect_args(recv_east, y)

      # in1 vertical mcast: sender at row 0, receivers at rows 1..NUM_ROWS-1
      recv_ys = rows[1:]  # y coords of receiver rows
      in1_sender_xy = grid[0][col_idx]
      if recv_ys:
        i1_sx, i1_sy = x, min(recv_ys)
        i1_ex, i1_ey = x, max(recv_ys)
        i1_nd = len(recv_ys)
      else:
        i1_sx = i1_sy = i1_ex = i1_ey = i1_nd = 0

      return [
        # [0-7] in0 DRAM
        a_buf.addr, in0_start_tile, in0_stride_w, in0_stride_h,
        in0_next_block_stride, IN0_BLOCK_W, PER_CORE_M, IN0_BLOCK_NUM_TILES,
        # [8-15] in1 DRAM
        b_buf.addr, in1_start_tile, in1_stride_w, in1_stride_h,
        in1_next_block_stride, PER_CORE_N, IN0_BLOCK_W, IN1_BLOCK_NUM_TILES,
        # [16] num_blocks
        NUM_BLOCKS,
        # [17-23] in0 mcast west rect + sender
        w_sx, w_sy, w_ex, w_ey, w_nd, sender_x, y,
        # [24-28] in0 mcast east rect
        e_sx, e_sy, e_ex, e_ey, e_nd,
        # [29-35] in1 mcast rect + sender
        i1_sx, i1_sy, i1_ex, i1_ey, i1_nd, in1_sender_xy[0], in1_sender_xy[1],
        # [36-37] flags
        is_in0_sender, is_in1_sender,
      ]

    def writer_args(core_idx, core_xy, n_cores):
      col_idx = core_idx % NUM_COLS
      row_idx = core_idx // NUM_COLS
      # Output tile start: row_idx * PER_CORE_M rows, col_idx * PER_CORE_N cols
      out_start = row_idx * PER_CORE_M * Nt + col_idx * PER_CORE_N
      return [
        c_buf.addr,
        out_start,
        1,                        # stride_w: adjacent tiles
        Nt,                       # stride_h: next row
        OUT_SUBBLOCK_W,           # next_subblock_stride_w
        OUT_SUBBLOCK_H * Nt,      # next_subblock_stride_h
        OUT_SUBBLOCK_W,
        OUT_SUBBLOCK_H,
        OUT_SUBBLOCK_NUM_TILES,
        IN1_NUM_SUBBLOCKS,        # num_subblocks_w
        IN0_NUM_SUBBLOCKS,        # num_subblocks_h
      ]

    def compute_args(core_idx, core_xy, n_cores):
      return []  # all params hardcoded in kernel source

    program = Program(
      reader=kernels.reader,
      writer=kernels.writer,
      compute=kernels.compute,
      reader_rt_args=reader_args,
      writer_rt_args=writer_args,
      compute_rt_args=compute_args,
      cbs=[0, 1, 16, 24],
      tile_size=TILE_BYTES,
      num_pages=CB0_PAGES,  # unused when cb_config is set
      cores=active_cores,
      num_sems=NUM_SEMS,
      cb_config={
        0:  (CB0_PAGES, TILE_BYTES),
        1:  (CB1_PAGES, TILE_BYTES),
        16: (CB16_PAGES, TILE_BYTES),
        24: (CB24_PAGES, TILE_BYTES),  # shares address with CB16
      },
    )

    print(f"\nWarmup ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
      device.run(program)

    print(f"Timing ({TIMED_ITERS} iters)...")
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
      device.run(program)
    elapsed = (time.perf_counter() - t0) / TIMED_ITERS

    flops = 2.0 * M * N * K
    tflops = flops / elapsed / 1e12
    print(f"\nAvg latency: {elapsed * 1e3:.3f} ms")
    print(f"Throughput:  {tflops:.2f} TFLOPS (LoFi bf16, {NUM_ROWS * NUM_COLS} cores)")

  finally:
    device.close()

if __name__ == "__main__":
  main()
