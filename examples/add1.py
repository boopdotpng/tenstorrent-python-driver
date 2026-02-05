#!/usr/bin/env python3
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import random, struct
from codegen import Compiler, DataFormat
from device import Device, Program, TileGrid
from dram import tilize, untilize

NUM_CORES = len(TileGrid.TENSIX)
N_TILES = NUM_CORES * 4

K_READER = r"""
#include <cstdint>

void kernel_main() {
  uint32_t in0_addr = get_arg_val<uint32_t>(0);
  uint32_t tile_offset = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  const uint32_t tile_size_bytes = get_tile_size(cb_in0);
  const InterleavedAddrGenFast<true> in0 = {
    .bank_base_address = in0_addr,
    .page_size = tile_size_bytes,
    .data_format = DataFormat::Float16_b,
  };
  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_in0, 1);
    noc_async_read_tile(tile_offset + i, in0, get_write_ptr(cb_in0));
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);
  }
}
"""

K_WRITER = r"""
#include <cstdint>

void kernel_main() {
  uint32_t out_addr = get_arg_val<uint32_t>(0);
  uint32_t tile_offset = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);
  constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
  const uint32_t tile_size_bytes = get_tile_size(cb_out0);
  const InterleavedAddrGenFast<true> out0 = {
    .bank_base_address = out_addr,
    .page_size = tile_size_bytes,
    .data_format = DataFormat::Float16_b,
  };
  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_out0, 1);
    noc_async_write_tile(tile_offset + i, out0, get_read_ptr(cb_out0));
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
  }
}
"""

K_COMPUTE = r"""
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#ifdef TRISC_MATH
  #include "sfpi.h"
#endif

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    tile_regs_acquire();
    cb_wait_front(tt::CBIndex::c_0, 1);
    copy_tile(tt::CBIndex::c_0, 0, 0);
#ifdef TRISC_MATH
    const sfpi::vFloat scalar = 1.0f;
    for (uint32_t v = 0; v < 32; ++v)
      sfpi::dst_reg[v] = sfpi::dst_reg[v] + scalar;
#endif
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(tt::CBIndex::c_16, 1);
    pack_tile(0, tt::CBIndex::c_16);
    cb_pop_front(tt::CBIndex::c_0, 1);
    tile_regs_release();
    cb_push_back(tt::CBIndex::c_16, 1);
  }
}
}  // namespace NAMESPACE
"""

def _bf16_from_f32(x: float) -> int:
  return struct.unpack("<I", struct.pack("<f", x))[0] >> 16

def _f32_from_bf16(x: int) -> float:
  return struct.unpack("<f", struct.pack("<I", (x & 0xFFFF) << 16))[0]

def _make_bf16_buffer(n_tiles: int, *, seed: int = 0) -> bytes:
  r = random.Random(seed)
  out = bytearray(n_tiles * 32 * 32 * 2)
  for i in range(n_tiles * 32 * 32):
    out[i * 2 : (i + 1) * 2] = _bf16_from_f32(r.random()).to_bytes(2, "little")
  return bytes(out)

def main():
  kernels = Compiler().compile(K_READER, K_WRITER, K_COMPUTE)
  device = Device()
  try:
    tile_size_bytes = 32 * 32 * 2
    tiles_per_core = (N_TILES + NUM_CORES - 1) // NUM_CORES

    src_rm = _make_bf16_buffer(N_TILES)
    src = tilize(src_rm, 2)
    src_buf = device.dram.alloc_write(src, name="src", page_size=tile_size_bytes)
    dst_buf = device.dram.alloc(tile_size_bytes * N_TILES, name="dst", page_size=tile_size_bytes)

    def reader_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, N_TILES - start)
      return [src_buf.addr, start, max(count, 0)]

    def writer_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, N_TILES - start)
      return [dst_buf.addr, start, max(count, 0)]

    def compute_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, N_TILES - start)
      return [max(count, 0)]

    program = Program(
      reader=kernels.reader,
      writer=kernels.writer,
      compute=kernels.compute,
      reader_rt_args=reader_args,
      writer_rt_args=writer_args,
      compute_rt_args=compute_args,
      cbs=[0, 16],
      tile_size=tile_size_bytes,
      num_pages=2,
    )
    device.run(program)

    out_tiled = device.dram.read(dst_buf)
    out = untilize(out_tiled, 2)

    for i in range(0, len(out), 2):
      src_bf16 = int.from_bytes(src_rm[i : i + 2], "little")
      src_f32 = _f32_from_bf16(src_bf16)
      exp_bf16 = _bf16_from_f32(src_f32 + 1.0)
      got_bf16 = int.from_bytes(out[i : i + 2], "little")
      if exp_bf16 != got_bf16:
        raise SystemExit(
          f"mismatch at bf16[{i // 2}]: "
          f"src={src_f32} exp={_f32_from_bf16(exp_bf16)} got={_f32_from_bf16(got_bf16)} "
          f"(src_bf16=0x{src_bf16:04x} exp_bf16=0x{exp_bf16:04x} got_bf16=0x{got_bf16:04x})"
        )
    print("Test Passed")
  finally:
    device.close()

if __name__ == "__main__":
  main()
