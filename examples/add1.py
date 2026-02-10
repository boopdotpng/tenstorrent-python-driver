#!/usr/bin/env python3
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import random, struct
from codegen import Compiler
from device import Device, Program, DataflowLaunch
from dram import DType

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

def _make_bf16_buffer(n_tiles: int, seed: int = 0) -> bytes:
  r = random.Random(seed)
  out = bytearray(n_tiles * 32 * 32 * 2)
  for i in range(n_tiles * 32 * 32):
    out[i * 2 : (i + 1) * 2] = _bf16_from_f32(r.random()).to_bytes(2, "little")
  return bytes(out)

def main():
  compiler = Compiler()
  reader = compiler.compile_dataflow(K_READER, processor="ncrisc")
  writer = compiler.compile_dataflow(K_WRITER, processor="brisc")
  compute = compiler.compile_compute(K_COMPUTE)
  device = Device()
  try:
    num_cores = len(device.dispatchable_cores)
    n_tiles = num_cores * 4
    tile_size_bytes = 32 * 32 * 2
    tiles_per_core = (n_tiles + num_cores - 1) // num_cores

    src_rm = _make_bf16_buffer(n_tiles)
    tensor_shape = (n_tiles, 32, 32)
    src_buf = device.dram.alloc_write(
      src_rm, name="src", page_size=tile_size_bytes, dtype=DType.bfloat16, shape=tensor_shape
    )
    dst_buf = device.dram.alloc(
      tile_size_bytes * n_tiles, name="dst", page_size=tile_size_bytes, dtype=DType.bfloat16, shape=tensor_shape
    )

    def core_span(core_idx: int) -> tuple[int, int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, n_tiles - start)
      return start, max(count, 0)

    def reader_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start, count = core_span(core_idx)
      return [src_buf.addr, start, count]

    def writer_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start, count = core_span(core_idx)
      return [dst_buf.addr, start, count]

    def compute_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      _, count = core_span(core_idx)
      return [count]

    program = Program(
      dataflow=[DataflowLaunch(
        cores=device.dispatchable_cores,
        reader=reader,
        writer=writer,
        reader_rt_args=reader_args,
        writer_rt_args=writer_args,
      )],
      compute=compute,
      compute_rt_args=compute_args,
      cbs=[0, 16],
      tile_size=tile_size_bytes,
      num_pages=2,
    )
    device.run(program)

    out = device.dram.read(dst_buf)

    for i in range(0, len(out), 2):
      src_bf16 = int.from_bytes(src_rm[i : i + 2], "little")
      src_f32 = _f32_from_bf16(src_bf16)
      exp_bf16 = _bf16_from_f32(src_f32 + 1.0)
      got_bf16 = int.from_bytes(out[i : i + 2], "little")
      if exp_bf16 != got_bf16:
        raise SystemExit(f"mismatch at bf16[{i // 2}] src=0x{src_bf16:04x} exp=0x{exp_bf16:04x} got=0x{got_bf16:04x}")
    print("Test Passed")
  finally:
    device.close()

if __name__ == "__main__":
  main()
