#!/usr/bin/env python3
"""Simple eltwise add +1.0 example using auto-generated dataflow kernels."""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import random, struct
from codegen import Compiler, CBSpec, DataFormat
from device import Device, Program
from dram import tilize, untilize

N_TILES = 64

# User only writes the compute kernel - dataflow is auto-generated!
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
  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
  constexpr uint32_t n_tiles = 64;
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
  # Define CB specs - the dataflow kernels are auto-generated from these
  cb_in0 = CBSpec(cb_id=0, fmt=DataFormat.Float16_b, arg_index=0)
  cb_out0 = CBSpec(cb_id=16, fmt=DataFormat.Float16_b, arg_index=0)

  # Compile: user only provides compute kernel, dataflow is generated
  kernels = Compiler().compile_compute(K_COMPUTE, inputs=[cb_in0], output=cb_out0, n_tiles=N_TILES)
  device = Device()
  try:
    tile_size_bytes = 32 * 32 * 2
    src_rm = _make_bf16_buffer(N_TILES)
    src = tilize(src_rm, 2)  # 2 bytes per bf16 element
    src_buf = device.dram.alloc_write(src, name="src", page_size=tile_size_bytes)
    dst_buf = device.dram.alloc(tile_size_bytes * N_TILES, name="dst", page_size=tile_size_bytes)

    program = Program(
      reader=kernels.reader,
      writer=kernels.writer,
      compute=kernels.compute,
      reader_rt_args=[src_buf.addr],
      writer_rt_args=[dst_buf.addr],
      compute_rt_args=[],
      cbs=[0, 16],
      tile_size=tile_size_bytes,
      num_pages=2,
    )
    device.run(program)

    out_tiled = device.dram.read(dst_buf)
    out = untilize(out_tiled, 2)  # 2 bytes per bf16 element

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
