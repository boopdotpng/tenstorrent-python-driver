#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from compiler import CkernelConfig, Compiler, DataFormat, MathFidelity
from device import DataflowLaunch, Device, Program
from dram import DType

TILE_BYTES = 32 * 32 * 2
TILE_BYTES_F32 = 32 * 32 * 4
L1_DATA_BYTES_AVAILABLE = 0x180000 - 0x037000

IN0_SEND_SEM, IN0_RECV_SEM = 0, 1
IN1_SEND_SEM, IN1_RECV_SEM = 2, 3
NUM_SEMS = 4

WARMUP_ITERS = 2
TIMED_ITERS = 5

F32_ACC = os.environ.get("F32_ACC") == "1"
IO_MODE = "f16" if os.environ.get("F16") == "1" else "bf16"
IO_DATA_FORMAT = DataFormat.Float16 if IO_MODE == "f16" else DataFormat.Float16_b
IO_DTYPE = DType.float16 if IO_MODE == "f16" else DType.bfloat16
CB24_TILE_BYTES = TILE_BYTES_F32 if F32_ACC else TILE_BYTES
_MATH_FIDELITY_MAP = {
  "lofi": MathFidelity.LoFi,
  "hifi2": MathFidelity.HiFi2,
  "hifi3": MathFidelity.HiFi3,
  "hifi4": MathFidelity.HiFi4,
}
MATH_FIDELITY_NAME = os.environ.get("MATH_FIDELITY", "hifi2").lower()
if MATH_FIDELITY_NAME not in _MATH_FIDELITY_MAP:
  valid = ", ".join(sorted(_MATH_FIDELITY_MAP))
  raise SystemExit(f"Invalid MATH_FIDELITY={MATH_FIDELITY_NAME!r}. Expected one of: {valid}")
MATH_FIDELITY = _MATH_FIDELITY_MAP[MATH_FIDELITY_NAME]
BF16_SMALL_OUT_TILE_THRESHOLD = 16
BF16_BLOCK_W_SMALL_CAP = 32
BF16_BLOCK_W_LARGE_CAP = 64

Core = tuple[int, int]

def ceil32(x):
  return (x + 31) & ~31


def _divisors(n):
  divs = []
  for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
      divs.append(i)
      if i != n // i:
        divs.append(n // i)
  return sorted(divs)


@dataclass(frozen=True)
class CoreTopology:
  cores: tuple[Core, ...]
  core_set: frozenset[Core]
  xs: tuple[int, ...]
  ys: tuple[int, ...]


@dataclass(frozen=True)
class GridLayout:
  rows: tuple[int, ...]
  cols: tuple[int, ...]

  @property
  def num_rows(self) -> int:
    return len(self.rows)

  @property
  def num_cols(self) -> int:
    return len(self.cols)

  @property
  def active_cores(self) -> int:
    return self.num_rows * self.num_cols


@dataclass(frozen=True)
class PrecisionPolicy:
  f32_acc: bool
  bf16_small_tile_threshold: int = BF16_SMALL_OUT_TILE_THRESHOLD
  bf16_small_block_w_cap: int = BF16_BLOCK_W_SMALL_CAP
  bf16_large_block_w_cap: int = BF16_BLOCK_W_LARGE_CAP

  def max_block_w(self, out_tiles_per_core: int) -> int:
    if self.f32_acc:
      return 1 << 30
    if out_tiles_per_core <= self.bf16_small_tile_threshold:
      return self.bf16_small_block_w_cap
    return self.bf16_large_block_w_cap


@dataclass(frozen=True)
class PlannerCandidate:
  layout: GridLayout
  per_core_m: int
  per_core_n: int
  in0_block_w: int
  out_subblock_h: int
  out_subblock_w: int
  mt: int
  nt: int

  @property
  def out_tiles_per_core(self) -> int:
    return self.per_core_m * self.per_core_n


@dataclass(frozen=True)
class MatmulPlan:
  rows: tuple[int, ...]
  cols: tuple[int, ...]
  mt: int
  kt: int
  nt: int
  per_core_m: int
  per_core_n: int
  in0_block_w: int
  out_subblock_h: int
  out_subblock_w: int
  out_subblock_num_tiles: int
  num_blocks: int
  in0_num_subblocks: int
  in1_num_subblocks: int
  in0_block_num_tiles: int
  in0_subblock_num_tiles: int
  in1_block_num_tiles: int
  in1_per_core_w: int
  out_block_num_tiles: int
  cb0_pages: int
  cb1_pages: int
  cb16_pages: int
  cb24_pages: int

  @property
  def num_rows(self) -> int:
    return len(self.rows)

  @property
  def num_cols(self) -> int:
    return len(self.cols)

  @property
  def active_core_count(self) -> int:
    return self.num_rows * self.num_cols

  def grid(self) -> list[list[Core]]:
    return [[(x, y) for x in self.cols] for y in self.rows]

  def active_cores(self) -> list[Core]:
    return [core for row in self.grid() for core in row]

  def validate_against(self, dispatchable_cores: list[Core]):
    core_set = set(dispatchable_cores)
    missing = [core for core in self.active_cores() if core not in core_set]
    if missing:
      raise ValueError(f"planner produced non-dispatchable cores: {missing[:4]}")


def _build_topology(dispatchable_cores: list[Core]) -> CoreTopology:
  cores = tuple(sorted(set(dispatchable_cores), key=lambda xy: (xy[0], xy[1])))
  if not cores:
    raise SystemExit("No dispatchable cores")
  core_set = frozenset(cores)
  xs = tuple(sorted({x for x, _ in cores}))
  ys = tuple(sorted({y for _, y in cores}))
  return CoreTopology(cores=cores, core_set=core_set, xs=xs, ys=ys)


def _iter_layouts(topology: CoreTopology, max_cores: int):
  ys = topology.ys
  for start in range(len(ys)):
    for stop in range(start + 1, len(ys) + 1):
      rows = ys[start:stop]
      valid_cols = [x for x in topology.xs if all((x, y) in topology.core_set for y in rows)]
      if not valid_cols:
        continue
      for col_count in range(1, len(valid_cols) + 1):
        layout = GridLayout(rows=rows, cols=tuple(valid_cols[:col_count]))
        if layout.active_cores <= max_cores:
          yield layout


def _fits_l1(per_core_m: int, per_core_n: int, in0_block_w: int) -> bool:
  cb0 = 2 * per_core_m * in0_block_w * TILE_BYTES
  cb1 = 2 * per_core_n * in0_block_w * TILE_BYTES
  cb_out = per_core_m * per_core_n * max(TILE_BYTES, CB24_TILE_BYTES)
  return (cb0 + cb1 + cb_out) <= L1_DATA_BYTES_AVAILABLE


def _iter_candidates(mt_base: int, kt: int, nt_base: int, layout: GridLayout, policy: PrecisionPolicy):
  per_core_m = (mt_base + layout.num_rows - 1) // layout.num_rows
  per_core_n = (nt_base + layout.num_cols - 1) // layout.num_cols
  mt = layout.num_rows * per_core_m
  nt = layout.num_cols * per_core_n
  max_block_w = policy.max_block_w(per_core_m * per_core_n)
  kt_divs = _divisors(kt)

  for out_subblock_h in range(1, 9):
    for out_subblock_w in range(1, 9):
      if out_subblock_h * out_subblock_w > 8:
        continue
      if per_core_m % out_subblock_h != 0 or per_core_n % out_subblock_w != 0:
        continue
      for in0_block_w in kt_divs:
        if in0_block_w > max_block_w:
          continue
        if not _fits_l1(per_core_m, per_core_n, in0_block_w):
          continue
        yield PlannerCandidate(
          layout=layout,
          per_core_m=per_core_m,
          per_core_n=per_core_n,
          in0_block_w=in0_block_w,
          out_subblock_h=out_subblock_h,
          out_subblock_w=out_subblock_w,
          mt=mt,
          nt=nt,
        )


def _score_candidate(c: PlannerCandidate):
  # Throughput proxy:
  # - reward more cores and larger K blocks
  # - strongly penalize tiny per-core output tiles (sync/mcast overhead)
  # - lightly penalize extra Mt/Nt padding
  per_core_work_bias = min(c.out_tiles_per_core, 16)
  pad_work = c.mt * c.nt
  subblock_tiles = c.out_subblock_h * c.out_subblock_w
  return (
    c.layout.active_cores * c.in0_block_w * per_core_work_bias * per_core_work_bias,
    -pad_work,
    c.layout.active_cores * c.in0_block_w,
    subblock_tiles,
    c.layout.active_cores,
  )


def _candidate_to_plan(candidate: PlannerCandidate, kt: int) -> MatmulPlan:
  out_subblock_num_tiles = candidate.out_subblock_h * candidate.out_subblock_w
  num_blocks = kt // candidate.in0_block_w
  in0_num_subblocks = candidate.per_core_m // candidate.out_subblock_h
  in1_num_subblocks = candidate.per_core_n // candidate.out_subblock_w
  in0_block_num_tiles = candidate.per_core_m * candidate.in0_block_w
  in0_subblock_num_tiles = candidate.out_subblock_h * candidate.in0_block_w
  in1_block_num_tiles = candidate.per_core_n * candidate.in0_block_w
  in1_per_core_w = candidate.per_core_n
  out_block_num_tiles = candidate.per_core_m * candidate.per_core_n
  cb0_pages = 2 * in0_block_num_tiles
  cb1_pages = 2 * in1_block_num_tiles
  cb16_pages = out_block_num_tiles
  cb24_pages = cb16_pages
  return MatmulPlan(
    rows=candidate.layout.rows,
    cols=candidate.layout.cols,
    mt=candidate.mt,
    kt=kt,
    nt=candidate.nt,
    per_core_m=candidate.per_core_m,
    per_core_n=candidate.per_core_n,
    in0_block_w=candidate.in0_block_w,
    out_subblock_h=candidate.out_subblock_h,
    out_subblock_w=candidate.out_subblock_w,
    out_subblock_num_tiles=out_subblock_num_tiles,
    num_blocks=num_blocks,
    in0_num_subblocks=in0_num_subblocks,
    in1_num_subblocks=in1_num_subblocks,
    in0_block_num_tiles=in0_block_num_tiles,
    in0_subblock_num_tiles=in0_subblock_num_tiles,
    in1_block_num_tiles=in1_block_num_tiles,
    in1_per_core_w=in1_per_core_w,
    out_block_num_tiles=out_block_num_tiles,
    cb0_pages=cb0_pages,
    cb1_pages=cb1_pages,
    cb16_pages=cb16_pages,
    cb24_pages=cb24_pages,
  )


def _plan_matmul(mt_base: int, kt: int, nt_base: int, dispatchable_cores: list[Core], policy: PrecisionPolicy | None = None):
  topology = _build_topology(dispatchable_cores)
  policy = policy or PrecisionPolicy(f32_acc=F32_ACC)
  best = None
  best_score = None
  for layout in _iter_layouts(topology, max_cores=len(topology.cores)):
    for candidate in _iter_candidates(mt_base, kt, nt_base, layout, policy):
      score = _score_candidate(candidate)
      if best is None or score > best_score:
        best = candidate
        best_score = score

  if best is None:
    raise SystemExit(
      f"No valid plan for Mt={mt_base} Kt={kt} Nt={nt_base} "
      f"(dispatchable={len(topology.cores)}, xs={len(topology.xs)}, ys={len(topology.ys)})"
    )
  plan = _candidate_to_plan(best, kt)
  plan.validate_against(list(topology.cores))
  return plan


# -- Reader kernel (NCRISC/NOC0): in0 sender --
K_READER_SENDER = f"""
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

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
    .bank_base_address = in0_addr, .page_size = tile_bytes, .data_format = DataFormat::{IO_DATA_FORMAT.name},
  }};

  uint32_t in0_current_block_start = in0_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {{
    DeviceZoneScopedN("reader_in0_block");
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
K_READER_RECV = """
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

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
    DeviceZoneScopedN("reader_in0_recv_wait");
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
#include "tools/profiler/kernel_profiler.hpp"

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
    .bank_base_address = in1_addr, .page_size = tile_bytes, .data_format = DataFormat::{IO_DATA_FORMAT.name},
  }};
  const InterleavedAddrGenFast<true> out_gen = {{
    .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::{IO_DATA_FORMAT.name},
  }};

  // === in1 block loop ===
  uint32_t in1_current_block_start = in1_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {{
    DeviceZoneScopedN("writer_in1_block");
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
      {{
        DeviceZoneScopedN("writer_output_wait_front");
        cb_wait_front(cb_out, out_sb_tile_count);
      }}
      uint32_t l1_addr = get_read_ptr(cb_out);
      uint32_t row_start = sbw_start;
      DeviceZoneScopedN("writer_output_store");
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
#include "tools/profiler/kernel_profiler.hpp"

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
    .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::{IO_DATA_FORMAT.name},
  }};

  volatile tt_l1_ptr uint32_t* in1_recv_sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_recv_sem_addr);

  for (uint32_t block = 0; block < num_blocks; block++) {{
    DeviceZoneScopedN("writer_in1_recv_wait");
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
      {{
        DeviceZoneScopedN("writer_recv_output_wait_front");
        cb_wait_front(cb_out, out_sb_tile_count);
      }}
      uint32_t l1_addr = get_read_ptr(cb_out);
      uint32_t row_start = sbw_start;
      DeviceZoneScopedN("writer_recv_output_store");
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


def _compute_kernel(p):
  """Generate compute kernel source from blocking parameters."""
  f32_define = "#define FP32_DEST_ACC_EN 1\n" if F32_ACC else ""
  pack_include = '#include "compute_kernel_api/pack.h"\n' if F32_ACC else ""
  reload_init = (
    "copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_1, tt::CBIndex::c_24);"
    if F32_ACC
    else "copy_tile_to_dst_init_short(tt::CBIndex::c_24);"
  )
  mm_short = (
    """mm_block_init_short_with_dt(
            tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24,
            transpose, out_subblock_w, out_subblock_h, in0_block_w);"""
    if F32_ACC
    else """mm_block_init_short(
            tt::CBIndex::c_0, tt::CBIndex::c_1,
            transpose, out_subblock_w, out_subblock_h, in0_block_w);"""
  )
  pack_cb16 = "PACK((pack_reconfig_data_format(tt::CBIndex::c_16)));" if F32_ACC else ""
  pack_cb24 = "PACK((pack_reconfig_data_format(tt::CBIndex::c_24)));" if F32_ACC else ""

  return f"""
#include <cstdint>
{f32_define}#define PACKER_L1_ACC 1
#include "compute_kernel_api/matmul.h"
{pack_include}#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {{
void MAIN {{
  constexpr uint32_t in0_block_w = {p['in0_block_w']};

  constexpr uint32_t in0_num_subblocks = {p['in0_num_subblocks']};
  constexpr uint32_t in0_block_num_tiles = {p['in0_block_num_tiles']};
  constexpr uint32_t in0_subblock_num_tiles = {p['in0_subblock_num_tiles']};

  constexpr uint32_t in1_num_subblocks = {p['in1_num_subblocks']};
  constexpr uint32_t in1_block_num_tiles = {p['in1_block_num_tiles']};
  constexpr uint32_t in1_per_core_w = {p['in1_per_core_w']};

  constexpr uint32_t num_blocks = {p['num_blocks']};

  constexpr uint32_t out_subblock_h = {p['out_subblock_h']};
  constexpr uint32_t out_subblock_w = {p['out_subblock_w']};
  constexpr uint32_t out_subblock_num_tiles = {p['out_subblock_num_tiles']};
  constexpr uint32_t out_block_num_tiles = {p['out_block_num_tiles']};

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
          {reload_init}
          cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
          #pragma GCC unroll 0
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{
            copy_tile(tt::CBIndex::c_24, i, i);
          }}
          cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);

          {mm_short}
        }}

        #pragma GCC unroll 0
        for (uint32_t inner = 0; inner < in0_block_w; inner++) {{
          DeviceZoneScopedN("compute_matmul_inner");
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
          {pack_cb16}
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
            {pack_cb24}
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

    if (num_blocks > 2 && block < num_blocks - 2) {{
      cb_wait_front(tt::CBIndex::c_24, out_block_num_tiles);
      cb_pop_front(tt::CBIndex::c_24, out_block_num_tiles);
    }}
    if (num_blocks >= 2 && block == num_blocks - 2) enable_reload = true;

    cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
    cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
  }}
}}
}} // namespace NAMESPACE
"""


def _mcast_rect_args(x_list: list[int], y: int):
  if not x_list:
    return (0, 0, 0, 0, 0)
  return (min(x_list), y, max(x_list), y, len(x_list))


def _f16_to_device_bytes(x: np.ndarray) -> bytes:
  x16 = np.ascontiguousarray(x, dtype=np.float16)
  if IO_MODE == "f16":
    return x16.tobytes()
  u32 = x16.astype(np.float32).view(np.uint32)
  return (u32 >> 16).astype(np.uint16).tobytes()


def _device_bytes_to_f32(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
  if IO_MODE == "f16":
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
  u16 = np.frombuffer(data, dtype=np.uint16)
  u32 = (u16.astype(np.uint32) << 16)
  return u32.view(np.float32).reshape(shape)


def _validate_matmul(a: np.ndarray, b: np.ndarray, c_bytes: bytes, M: int, N: int, Mp: int, Np: int):
  c_ref = a @ b
  c_full = _device_bytes_to_f32(c_bytes, (Mp, Np))
  c_got = c_full[:M, :N]
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
  if len(sys.argv) == 4:
    M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
  elif len(sys.argv) == 1:
    M, K, N = 5120, 4096, 5632
  else:
    raise SystemExit("Usage: matmul_peak.py [M K N]")
  if M < 1 or K < 1 or N < 1:
    raise SystemExit("M, K, N must be positive integers")

  # Pad K/N to tile boundaries for I/O buffers.
  Kp, Np = ceil32(K), ceil32(N)
  if Np < 32:
    Np = 32
  Kt, Nt_base = Kp // 32, Np // 32

  device = Device()
  max_cores = len(device.dispatchable_cores)

  Mt_base = ceil32(M) // 32
  plan = _plan_matmul(Mt_base, Kt, Nt_base, device.dispatchable_cores)
  Mp = plan.mt * 32
  Np = plan.nt * 32

  K_COMPUTE = _compute_kernel({
    'in0_block_w': plan.in0_block_w,
    'in0_num_subblocks': plan.in0_num_subblocks,
    'in0_block_num_tiles': plan.in0_block_num_tiles,
    'in0_subblock_num_tiles': plan.in0_subblock_num_tiles,
    'in1_num_subblocks': plan.in1_num_subblocks,
    'in1_block_num_tiles': plan.in1_block_num_tiles,
    'in1_per_core_w': plan.in1_per_core_w,
    'num_blocks': plan.num_blocks,
    'out_subblock_h': plan.out_subblock_h,
    'out_subblock_w': plan.out_subblock_w,
    'out_subblock_num_tiles': plan.out_subblock_num_tiles,
    'out_block_num_tiles': plan.out_block_num_tiles,
  })

  padded = (M != Mp or K != Kp or N != Np)
  print(
    f"Matmul Peak (4-role split): C[{M},{N}] = A[{M},{K}] @ B[{K},{N}] "
    f"({IO_MODE} io, {'fp32' if F32_ACC else 'mixed'} accum, {MATH_FIDELITY.name})"
  )
  if padded:
    print(f"  padded: A[{Mp},{Kp}] @ B[{Kp},{Np}] -> C[{Mp},{Np}]")
  print(f"  Mt={plan.mt} Kt={Kt} Nt={plan.nt} grid={plan.num_rows}x{plan.num_cols}")
  print(
    f"  per_core_M={plan.per_core_m} per_core_N={plan.per_core_n} "
    f"in0_block_w={plan.in0_block_w} num_blocks={plan.num_blocks}"
  )
  print(
    f"  subblock: {plan.out_subblock_h}h x {plan.out_subblock_w}w = "
    f"{plan.out_subblock_num_tiles} tiles"
  )
  print(
    f"  env switches: F16={'1' if IO_MODE == 'f16' else '0'} "
    f"F32_ACC={'1' if F32_ACC else '0'} MATH_FIDELITY={MATH_FIDELITY_NAME}"
  )

  cfg_kwargs = {}
  if F32_ACC:
    cfg_kwargs = {
      "cb_data_formats": ((24, DataFormat.Float32),),
      "dst_accum_mode": True,
    }

  cfg = CkernelConfig(
    input_format=IO_DATA_FORMAT,
    output_format=IO_DATA_FORMAT,
    math_fidelity=MATH_FIDELITY,
    **cfg_kwargs,
  )
  compiler = Compiler(cfg)
  reader_sender = compiler.compile_dataflow(K_READER_SENDER, processor="ncrisc")
  reader_recv = compiler.compile_dataflow(K_READER_RECV, processor="ncrisc")
  writer_sender = compiler.compile_dataflow(K_WRITER_SENDER, processor="brisc")
  writer_recv = compiler.compile_dataflow(K_WRITER_RECV, processor="brisc")
  compute = compiler.compile_compute(K_COMPUTE)

  print(f"Device: {max_cores} dispatchable cores")
  assert max_cores >= plan.active_core_count, f"need {plan.active_core_count} cores, have {max_cores}"

  try:
    plan.validate_against(device.dispatchable_cores)
    grid = plan.grid()
    cols = list(plan.cols)
    rows = list(plan.rows)
    active_cores = plan.active_cores()
    print(f"Grid columns: {cols}")
    print(f"Grid rows: {rows}")

    west_cols = [x for x in cols if x < 8]
    east_cols = [x for x in cols if x >= 10]

    # Generate input data — pad to tile boundaries if needed
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(123)
    a_src = rng_a.uniform(-0.5, 0.5, size=(M, K)).astype(np.float16)
    b_src = rng_b.uniform(-0.5, 0.5, size=(K, N)).astype(np.float16)

    if padded:
      a_padded = np.zeros((Mp, Kp), dtype=np.float16)
      a_padded[:M, :K] = a_src
      b_padded = np.zeros((Kp, Np), dtype=np.float16)
      b_padded[:K, :N] = b_src
      a_bytes = _f16_to_device_bytes(a_padded)
      b_bytes = _f16_to_device_bytes(b_padded)
    else:
      a_bytes = _f16_to_device_bytes(a_src)
      b_bytes = _f16_to_device_bytes(b_src)

    # Reference uses quantized original (unpadded) data
    a_ref = _device_bytes_to_f32(_f16_to_device_bytes(a_src), (M, K))
    b_ref = _device_bytes_to_f32(_f16_to_device_bytes(b_src), (K, N))

    a_buf = device.dram.alloc_write(
      a_bytes, name="A", page_size=TILE_BYTES, dtype=IO_DTYPE, shape=(Mp, Kp)
    )
    b_buf = device.dram.alloc_write(
      b_bytes, name="B", page_size=TILE_BYTES, dtype=IO_DTYPE, shape=(Kp, Np)
    )
    c_buf = device.dram.alloc(
      TILE_BYTES * plan.mt * plan.nt, name="C", page_size=TILE_BYTES, dtype=IO_DTYPE, shape=(Mp, Np)
    )

    core_to_rc = {grid[r][c]: (r, c) for r in range(plan.num_rows) for c in range(plan.num_cols)}
    top_left = [grid[0][0]]
    top_row = [grid[0][c] for c in range(1, plan.num_cols)]
    left_col = [grid[r][0] for r in range(1, plan.num_rows)]
    interior = [grid[r][c] for r in range(1, plan.num_rows) for c in range(1, plan.num_cols)]

    def reader_args(role_idx: int, core_xy: tuple[int, int], num_cores: int) -> list[int]:
      row_idx, _ = core_to_rc[core_xy]
      y = core_xy[1]

      sender_x = cols[0]
      recv_west = [c for c in west_cols if c != sender_x]
      recv_east = list(east_cols)
      w_sx, w_sy, w_ex, w_ey, w_nd = _mcast_rect_args(recv_west, y)
      e_sx, e_sy, e_ex, e_ey, e_nd = _mcast_rect_args(recv_east, y)

      sender_xy = grid[row_idx][0]
      return [
        a_buf.addr, row_idx * plan.per_core_m * Kt, 1, Kt, plan.in0_block_w,
        plan.in0_block_w, plan.per_core_m, plan.in0_block_num_tiles, plan.num_blocks,
        w_sx, w_sy, w_ex, w_ey, w_nd,
        e_sx, e_sy, e_ex, e_ey, e_nd,
        sender_xy[0], sender_xy[1],
        IN0_SEND_SEM, IN0_RECV_SEM,
      ]

    def writer_args(role_idx: int, core_xy: tuple[int, int], num_cores: int) -> list[int]:
      row_idx, col_idx = core_to_rc[core_xy]
      x = core_xy[0]
      recv_ys = rows[1:]
      if recv_ys:
        i1_sx, i1_sy = x, max(recv_ys)
        i1_ex, i1_ey = x, min(recv_ys)
        i1_nd = len(recv_ys)
      else:
        i1_sx = i1_sy = i1_ex = i1_ey = i1_nd = 0

      sender_xy = grid[0][col_idx]
      out_start = row_idx * plan.per_core_m * plan.nt + col_idx * plan.per_core_n
      return [
        b_buf.addr, col_idx * plan.per_core_n, 1, plan.nt, plan.in0_block_w * plan.nt,
        plan.per_core_n, plan.in0_block_w, plan.in1_block_num_tiles, plan.num_blocks,
        i1_sx, i1_sy, i1_ex, i1_ey, i1_nd,
        sender_xy[0], sender_xy[1],
        IN1_SEND_SEM, IN1_RECV_SEM,
        c_buf.addr, out_start, 1, plan.nt,
        plan.out_subblock_w, plan.out_subblock_h * plan.nt,
        plan.out_subblock_w, plan.out_subblock_h,
        plan.out_subblock_num_tiles,
        plan.in1_num_subblocks, plan.in0_num_subblocks,
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
      num_pages=plan.cb0_pages,
      cores=active_cores,
      num_sems=NUM_SEMS,
      cb_config={
        0:  (plan.cb0_pages, TILE_BYTES),
        1:  (plan.cb1_pages, TILE_BYTES),
        16: (plan.cb16_pages, TILE_BYTES),
        24: (plan.cb24_pages, CB24_TILE_BYTES),
      },
      name="matmul_peak",
      sources={
        "reader_sender": K_READER_SENDER,
        "reader_recv": K_READER_RECV,
        "writer_sender": K_WRITER_SENDER,
        "writer_recv": K_WRITER_RECV,
        "compute": K_COMPUTE,
      },
    )

    print(f"\nWarmup ({WARMUP_ITERS} iters)...")
    program.profile = False
    for _ in range(WARMUP_ITERS):
      device.queue(program)
    device.run()

    print(f"Timing ({TIMED_ITERS} iters)...")
    program.profile = True
    for _ in range(TIMED_ITERS):
      device.queue(program)
    device.run()
    c_rm = device.dram.read(c_buf)
    _validate_matmul(a_ref, b_ref, c_rm, M, N, Mp, Np)

  finally:
    device.close()


if __name__ == "__main__":
  main()
