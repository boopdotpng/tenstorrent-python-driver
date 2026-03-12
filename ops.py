"""Operation templates for Blackhole — matmul, eltwise, reduction."""

from dataclasses import dataclass
from autogen import TensixL1
from device import DramBuffer
from dispatch import CBConfig, Core, Dtype, MathFidelity, Program

L1_DATA_BYTES = TensixL1.SIZE - TensixL1.DATA_BUFFER_SPACE_BASE
NUM_SEMS = 4  # IN0_SEND=0, IN0_RECV=1, IN1_SEND=2, IN1_RECV=3

def _divisors(n):
  divs = []
  for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
      divs.append(i)
      if i != n // i:
        divs.append(n // i)
  return sorted(divs)

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
    return [c for row in self.grid() for c in row]

def _ceil32(x):
  return (x + 31) & ~31

def plan_matmul(
  M: int,
  K: int,
  N: int,
  cores: list[Core],
  io_dtype: Dtype = Dtype.Float16_b,
  f32_acc: bool = False,
) -> MatmulPlan:
  mt_base, kt, nt_base = _ceil32(M) // 32, _ceil32(K) // 32, max(1, _ceil32(N) // 32)
  tile_bytes = io_dtype.tile_size
  cb24_tile_bytes = Dtype.Float32.tile_size if f32_acc else tile_bytes

  ordered = sorted(set(cores), key=lambda xy: (xy[0], xy[1]))
  if not ordered:
    raise SystemExit("No cores")
  core_set = frozenset(ordered)
  xs = tuple(sorted({x for x, _ in ordered}))
  ys = tuple(sorted({y for _, y in ordered}))

  def fits_l1(pcm, pcn, bw):
    return (
      2 * pcm * bw * tile_bytes
      + 2 * pcn * bw * tile_bytes
      + pcm * pcn * max(tile_bytes, cb24_tile_bytes)
    ) <= L1_DATA_BYTES

  max_bw_cap = (1 << 30) if f32_acc else None
  kt_divs = _divisors(kt)
  best, best_score = None, None

  for y_start in range(len(ys)):
    for y_stop in range(y_start + 1, len(ys) + 1):
      rows = ys[y_start:y_stop]
      valid_cols = [x for x in xs if all((x, y) in core_set for y in rows)]
      if not valid_cols:
        continue
      for nc in range(1, len(valid_cols) + 1):
        cols = tuple(valid_cols[:nc])
        nr = len(rows)
        if nr * nc > len(cores):
          continue
        pcm = (mt_base + nr - 1) // nr
        pcn = (nt_base + nc - 1) // nc
        mt, nt = nr * pcm, nc * pcn
        out_tiles = pcm * pcn
        bw_cap = max_bw_cap or (32 if out_tiles <= 16 else 64)
        for sbh in range(1, 9):
          for sbw in range(1, 9):
            if sbh * sbw > 8:
              continue
            if pcm % sbh != 0 or pcn % sbw != 0:
              continue
            for bw in kt_divs:
              if bw > bw_cap or not fits_l1(pcm, pcn, bw):
                continue
              bias = min(out_tiles, 16)
              score = (
                nr * nc * bw * bias * bias,
                -(mt * nt),
                nr * nc * bw,
                sbh * sbw,
                nr * nc,
              )
              if best_score is None or score > best_score:
                best = (rows, cols, mt, nt, pcm, pcn, bw, sbh, sbw)
                best_score = score

  if best is None:
    raise SystemExit(f"No valid matmul plan for Mt={mt_base} Kt={kt} Nt={nt_base}")
  rows, cols, mt, nt, pcm, pcn, bw, sbh, sbw = best
  return MatmulPlan(
    rows=rows,
    cols=cols,
    mt=mt,
    kt=kt,
    nt=nt,
    per_core_m=pcm,
    per_core_n=pcn,
    in0_block_w=bw,
    out_subblock_h=sbh,
    out_subblock_w=sbw,
    out_subblock_num_tiles=sbh * sbw,
    num_blocks=kt // bw,
    in0_num_subblocks=pcm // sbh,
    in1_num_subblocks=pcn // sbw,
    in0_block_num_tiles=pcm * bw,
    in0_subblock_num_tiles=sbh * bw,
    in1_block_num_tiles=pcn * bw,
    in1_per_core_w=pcn,
    out_block_num_tiles=pcm * pcn,
    cb0_pages=2 * pcm * bw,
    cb1_pages=2 * pcn * bw,
    cb16_pages=pcm * pcn,
    cb24_pages=pcm * pcn,
  )

_OUTPUT_WRITE_LOOP = """\
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
  }}"""

def _mcast_rect_args(x_list, y):
  if not x_list:
    return (0, 0, 0, 0, 0)
  return (min(x_list), y, max(x_list), y, len(x_list))

def _reader_sender_src(data_format):
  return f"""\
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
  .bank_base_address = in0_addr, .page_size = tile_bytes, .data_format = DataFormat::{data_format},
  }};
  uint32_t in0_current_block_start = in0_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {{
  cb_reserve_back(cb_in0, in0_block_num_tiles);
  uint32_t l1_addr = get_write_ptr(cb_in0);
  uint32_t in0_start_address = l1_addr;
  uint32_t in0_row_start = in0_current_block_start;
  uint32_t in0_block_size_bytes = 0;
  {{ DeviceZoneScopedN("DRAM_READ_IN0");
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
  }}
  noc_semaphore_wait(in0_sender_sem_ptr, w_nd + e_nd);
  noc_semaphore_set(in0_sender_sem_ptr, 0);
  {{ DeviceZoneScopedN("MCAST_IN0");
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
  }}
  cb_push_back(cb_in0, in0_block_num_tiles);
  }}
}}"""

_READER_RECV_SRC = """\
#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
  uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);
  uint32_t num_blocks          = get_arg_val<uint32_t>(8);
  uint32_t in0_sender_noc_x    = get_arg_val<uint32_t>(19);
  uint32_t in0_sender_noc_y    = get_arg_val<uint32_t>(20);
  uint32_t in0_sender_sem_addr = get_semaphore(get_arg_val<uint32_t>(21));
  uint32_t in0_recv_sem_addr   = get_semaphore(get_arg_val<uint32_t>(22));
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  volatile tt_l1_ptr uint32_t* in0_recv_sem_ptr =
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_recv_sem_addr);
  for (uint32_t block = 0; block < num_blocks; block++) {
  cb_reserve_back(cb_in0, in0_block_num_tiles);
  noc_semaphore_set(in0_recv_sem_ptr, INVALID);
  uint64_t sender_sem_noc = get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_sem_addr);
  noc_semaphore_inc(sender_sem_noc, 1);
  noc_semaphore_wait(in0_recv_sem_ptr, VALID);
  cb_push_back(cb_in0, in0_block_num_tiles);
  }
}"""

def _writer_sender_src(data_format):
  return f"""\
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
  .bank_base_address = in1_addr, .page_size = tile_bytes, .data_format = DataFormat::{data_format},
  }};
  const InterleavedAddrGenFast<true> out_gen = {{
  .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::{data_format},
  }};
  uint32_t in1_current_block_start = in1_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {{
  cb_reserve_back(cb_in1, in1_block_num_tiles);
  uint32_t l1_addr = get_write_ptr(cb_in1);
  uint32_t in1_start_address = l1_addr;
  uint32_t in1_row_start = in1_current_block_start;
  uint32_t in1_block_size_bytes = 0;
  {{ DeviceZoneScopedN("DRAM_READ_IN1");
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
  }}
  noc_semaphore_wait(in1_sender_sem_ptr, i1_nd);
  noc_semaphore_set(in1_sender_sem_ptr, 0);
  {{ DeviceZoneScopedN("MCAST_IN1");
  if (i1_nd > 0) {{
    uint64_t mcast_addr = get_noc_multicast_addr(i1_sx, i1_sy, i1_ex, i1_ey, in1_start_address);
    noc_async_write_multicast(in1_start_address, mcast_addr, in1_block_size_bytes, i1_nd);
    noc_async_writes_flushed();
    uint64_t sem_addr = get_noc_multicast_addr(i1_sx, i1_sy, i1_ex, i1_ey, in1_recv_sem_addr);
    noc_semaphore_set_multicast(in1_recv_sem_addr, sem_addr, i1_nd);
  }}
  }}
  cb_push_back(cb_in1, in1_block_num_tiles);
  }}
  {{ DeviceZoneScopedN("WRITE_OUT");
{_OUTPUT_WRITE_LOOP}
  }}
}}"""

def _writer_recv_src(data_format):
  return f"""\
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
  .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::{data_format},
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
{_OUTPUT_WRITE_LOOP}
}}"""

def _compute_src(plan: MatmulPlan, f32_acc: bool):
  f32_def = "#define FP32_DEST_ACC_EN 1\n" if f32_acc else ""
  pack_inc = '#include "compute_kernel_api/pack.h"\n' if f32_acc else ""
  reload_init = (
    "copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_1, tt::CBIndex::c_24);"
    if f32_acc
    else "copy_tile_to_dst_init_short(tt::CBIndex::c_24);"
  )
  mm_short = (
    """mm_block_init_short_with_dt(
      tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24,
      transpose, out_subblock_w, out_subblock_h, in0_block_w);"""
    if f32_acc
    else """mm_block_init_short(
      tt::CBIndex::c_0, tt::CBIndex::c_1,
      transpose, out_subblock_w, out_subblock_h, in0_block_w);"""
  )
  pack16 = "PACK((pack_reconfig_data_format(tt::CBIndex::c_16)));" if f32_acc else ""
  pack24 = "PACK((pack_reconfig_data_format(tt::CBIndex::c_24)));" if f32_acc else ""
  return f"""\
#include <cstdint>
{f32_def}#define PACKER_L1_ACC 1
#include "compute_kernel_api/matmul.h"
{pack_inc}#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {{
void MAIN {{
  constexpr uint32_t in0_block_w = {plan.in0_block_w};
  constexpr uint32_t in0_num_subblocks = {plan.in0_num_subblocks};
  constexpr uint32_t in0_block_num_tiles = {plan.in0_block_num_tiles};
  constexpr uint32_t in0_subblock_num_tiles = {plan.in0_subblock_num_tiles};
  constexpr uint32_t in1_num_subblocks = {plan.in1_num_subblocks};
  constexpr uint32_t in1_block_num_tiles = {plan.in1_block_num_tiles};
  constexpr uint32_t in1_per_core_w = {plan.in1_per_core_w};
  constexpr uint32_t num_blocks = {plan.num_blocks};
  constexpr uint32_t out_subblock_h = {plan.out_subblock_h};
  constexpr uint32_t out_subblock_w = {plan.out_subblock_w};
  constexpr uint32_t out_subblock_num_tiles = {plan.out_subblock_num_tiles};
  constexpr uint32_t out_block_num_tiles = {plan.out_block_num_tiles};
  constexpr uint32_t transpose = 0;
  mm_block_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16,
  transpose, out_subblock_w, out_subblock_h, in0_block_w);
  DeviceZoneScopedN("COMPUTE");
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
      for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{ copy_tile(tt::CBIndex::c_24, i, i); }}
      cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
      {mm_short}
    }}
    #pragma GCC unroll 0
    for (uint32_t inner = 0; inner < in0_block_w; inner++) {{
      const uint32_t in0_tile_index = (uint32_t)(in0_index_subblock_offset + (int)inner);
      const uint32_t in1_tile_index = (uint32_t)(in1_index_subblock_offset + (int)(inner * in1_per_core_w));
      matmul_block(tt::CBIndex::c_0, tt::CBIndex::c_1,
      in0_tile_index, in1_tile_index, 0, transpose, out_subblock_w, out_subblock_h, in0_block_w);
    }}
    tile_regs_commit();
    if (last_out) {{
      cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
      tile_regs_wait();
      {pack16}
      PACK((llk_pack_reconfig_l1_acc(0)));
      #pragma GCC unroll 0
      for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{ pack_tile(i, tt::CBIndex::c_16); }}
      tile_regs_release();
      cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
    }} else {{
      if (block == 0) {{
      cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
      out_num_tiles_to_wait += out_subblock_num_tiles;
      }}
      cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
      tile_regs_wait();
      if (block == 0) {{ {pack24} PACK((llk_pack_reconfig_l1_acc(0))); }}
      else if (block == 1) {{ PACK((llk_pack_reconfig_l1_acc(1))); }}
      #pragma GCC unroll 0
      for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {{ pack_tile(i, tt::CBIndex::c_24); }}
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
}} // namespace NAMESPACE"""

class MatmulProgram:
  def __init__(
    self,
    plan: MatmulPlan,
    a: DramBuffer,
    b: DramBuffer,
    c: DramBuffer,
    io_dtype: Dtype = Dtype.Float16_b,
    math_fidelity: MathFidelity = MathFidelity.HiFi2,
    f32_acc: bool = False,
  ):
    self.plan, self.a, self.b, self.c = plan, a, b, c
    self.io_dtype, self.math_fidelity, self.f32_acc = (
      io_dtype,
      math_fidelity,
      f32_acc,
    )
    cb24_dtype = Dtype.Float32 if f32_acc else io_dtype
    self.cbs = [
      CBConfig(index=0, dtype=io_dtype, tiles=plan.cb0_pages),
      CBConfig(index=1, dtype=io_dtype, tiles=plan.cb1_pages),
      CBConfig(index=16, dtype=io_dtype, tiles=plan.cb16_pages),
      CBConfig(index=24, dtype=cb24_dtype, tiles=plan.cb24_pages),
    ]

  def compile(self, compiler):
    plan = self.plan
    df = self.io_dtype.name
    M, K, N = plan.mt * 32, plan.kt * 32, plan.nt * 32

    reader_src = _reader_sender_src(df)
    writer_src = _writer_sender_src(df)
    compute_src = _compute_src(plan, self.f32_acc)

    r_send = compiler.compile_dataflow(reader_src, "ncrisc")
    r_recv = compiler.compile_dataflow(_READER_RECV_SRC, "ncrisc")
    w_send = compiler.compile_dataflow(writer_src, "brisc")
    w_recv = compiler.compile_dataflow(_writer_recv_src(df), "brisc")
    program = Program(
      cores="all",
      reader_kernel=reader_src,
      writer_kernel=writer_src,
      compute_kernel=compute_src,
      name=f"matmul_{M}x{K}x{N}",
      cbs=self.cbs,
      math_fidelity=self.math_fidelity,
      dst_accum_mode=self.f32_acc,
      semaphores=NUM_SEMS,
    )
    compute = compiler.compile_compute(compute_src, program)

    grid = plan.grid()
    rows, cols = plan.rows, plan.cols
    all_cores = sorted(plan.active_cores(), key=lambda core: (core[0], core[1]))
    core_to_rc = {
      grid[r][c]: (r, c) for r in range(len(rows)) for c in range(len(cols))
    }
    west_cols = [x for x in cols if x < 8]
    east_cols = [x for x in cols if x >= 10]

    roles = []
    top_left = [grid[0][0]]
    top_row = [grid[0][c] for c in range(1, len(cols))]
    left_col = [grid[r][0] for r in range(1, len(rows))]
    interior = [
      grid[r][c] for r in range(1, len(rows)) for c in range(1, len(cols))
    ]
    for cores, rk, wk in [
      (top_left, r_send, w_send),
      (top_row, r_recv, w_send),
      (left_col, r_send, w_recv),
      (interior, r_recv, w_recv),
    ]:
      if cores:
        roles.append((cores, rk, wk))

    def reader_args(core):
      ri, _ = core_to_rc[core]
      y = core[1]
      sender_x = cols[0]
      rw = [c for c in west_cols if c != sender_x]
      re = list(east_cols)
      w_sx, w_sy, w_ex, w_ey, w_nd = _mcast_rect_args(rw, y)
      e_sx, e_sy, e_ex, e_ey, e_nd = _mcast_rect_args(re, y)
      sender_xy = grid[ri][0]
      return [
        self.a.addr,
        ri * plan.per_core_m * plan.kt,
        1,
        plan.kt,
        plan.in0_block_w,
        plan.in0_block_w,
        plan.per_core_m,
        plan.in0_block_num_tiles,
        plan.num_blocks,
        w_sx,
        w_sy,
        w_ex,
        w_ey,
        w_nd,
        e_sx,
        e_sy,
        e_ex,
        e_ey,
        e_nd,
        sender_xy[0],
        sender_xy[1],
        0,
        1,
      ]

    def writer_args(core):
      ri, ci = core_to_rc[core]
      x = core[0]
      recv_ys = rows[1:]
      if recv_ys:
        i1_sx, i1_sy, i1_ex, i1_ey, i1_nd = (
          x,
          max(recv_ys),
          x,
          min(recv_ys),
          len(recv_ys),
        )
      else:
        i1_sx = i1_sy = i1_ex = i1_ey = i1_nd = 0
      sender_xy = grid[0][ci]
      out_start = ri * plan.per_core_m * plan.nt + ci * plan.per_core_n
      return [
        self.b.addr,
        ci * plan.per_core_n,
        1,
        plan.nt,
        plan.in0_block_w * plan.nt,
        plan.per_core_n,
        plan.in0_block_w,
        plan.in1_block_num_tiles,
        plan.num_blocks,
        i1_sx,
        i1_sy,
        i1_ex,
        i1_ey,
        i1_nd,
        sender_xy[0],
        sender_xy[1],
        2,
        3,
        self.c.addr,
        out_start,
        1,
        plan.nt,
        plan.out_subblock_w,
        plan.out_subblock_h * plan.nt,
        plan.out_subblock_w,
        plan.out_subblock_h,
        plan.out_subblock_num_tiles,
        plan.in1_num_subblocks,
        plan.in0_num_subblocks,
      ]

    per_core_args = [(writer_args(c), reader_args(c), []) for c in all_cores]
    self.compiled_program = program
    return program, roles, compute, all_cores, per_core_args
# class BinaryEltwiseProgram: ...  # ADD, MUL, SUB — no mcast, no sems, 1D tile striping
# class UnaryEltwiseProgram: ...   # EXP, LOG, SQRT, RECIP — single input CB, SFPU ops
# class ReductionProgram: ...      # SUM, MAX — accumulation loop, maybe 1D mcast
