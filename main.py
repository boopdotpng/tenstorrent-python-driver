from __future__ import annotations

import argparse
import random
import struct

from codegen import Compiler, Processor
from device import Device

N_TILES = 64

K_READER = r"""
#include <cstdint>

void kernel_main() {
  uint32_t in0_addr = get_arg_val<uint32_t>(0);

  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  const uint32_t tile_size_bytes = get_tile_size(cb_in0);

  const InterleavedAddrGenFast<true> in0 = {
    .bank_base_address = in0_addr,
    .page_size = tile_size_bytes,
    .data_format = DataFormat::Float16_b,
  };

  constexpr uint32_t n_tiles = 64;
  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_in0, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);

    noc_async_read_tile(i, in0, cb_in0_addr);
    noc_async_read_barrier();

    cb_push_back(cb_in0, 1);
  }
}
"""

K_WRITER = r"""
#include <cstdint>

void kernel_main() {
  uint32_t out_addr = get_arg_val<uint32_t>(0);

  constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
  const uint32_t tile_size_bytes = get_tile_size(cb_out0);

  const InterleavedAddrGenFast<true> out0 = {
    .bank_base_address = out_addr,
    .page_size = tile_size_bytes,
    .data_format = DataFormat::Float16_b,
  };

  constexpr uint32_t n_tiles = 64;
  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);

    noc_async_write_tile(i, out0, cb_out0_addr);
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
  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);

  constexpr uint32_t n_tiles = 64;
  for (uint32_t i = 0; i < n_tiles; ++i) {
    tile_regs_acquire();
    cb_wait_front(tt::CBIndex::c_0, 1);
    copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef TRISC_MATH
    const sfpi::vFloat scalar = 1.0f;
    constexpr uint32_t vectors_per_tile = 32;

    for (uint32_t v = 0; v < vectors_per_tile; ++v) {
      sfpi::dst_reg[v] = sfpi::dst_reg[v] + scalar;
    }
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

def _fmt_float_noexp(x: float) -> str:
  if x != x: return "nan"
  if x == float("inf"): return "inf"
  if x == float("-inf"): return "-inf"
  ax = abs(x)
  decimals = 6 if ax >= 1 else 10 if ax >= 1e-4 else 14
  s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
  return s if s else "0"

def _make_bf16_buffer(n_tiles: int, *, seed: int = 0) -> bytes:
  r = random.Random(seed)
  out = bytearray(n_tiles * 32 * 32 * 2)
  for i in range(n_tiles * 32 * 32):
    out[i * 2:(i + 1) * 2] = _bf16_from_f32(r.random()).to_bytes(2, "little")
  return bytes(out)

def _compile_add1_sfpu():
  cc = Compiler(dm_wrapper="tt-metal")
  return {
    "ncrisc": cc.compile_kernel(K_READER, Processor.NCRISC, dispatch_message_addr=0),
    "brisc": cc.compile_kernel(K_WRITER, Processor.BRISC, dispatch_message_addr=0),
    "trisc0": cc.compile_kernel(K_COMPUTE, Processor.TRISC0, dispatch_message_addr=0),
    "trisc1": cc.compile_kernel(K_COMPUTE, Processor.TRISC1, dispatch_message_addr=0),
    "trisc2": cc.compile_kernel(K_COMPUTE, Processor.TRISC2, dispatch_message_addr=0),
  }

def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(prog="pure-py/main.py")
  ap.add_argument("--verify", action="store_true")
  ap.add_argument("--run", action="store_true", help="actually launch on device (compile-only by default)")
  ns = ap.parse_args(argv)

  kernels = _compile_add1_sfpu()
  if not ns.run: return

  device = Device()
  device.set_noc_translation_enabled(False)
  device.debug_noc_niu()
  try:
    core = (1, 2) if (1, 2) in device.tiles.tensix else device.tiles.tensix[0]

    tile_size_bytes = 32 * 32 * 2
    src = _make_bf16_buffer(N_TILES)
    src_buf = device.dram.alloc_write(src, name="src", page_size=tile_size_bytes)
    dst = bytes(tile_size_bytes * N_TILES)
    dst_buf = device.dram.alloc_write(dst, name="dst", page_size=tile_size_bytes)

    # Verify host can read back what it wrote
    readback = device.dram.read(src_buf)
    if readback != src:
      for i in range(min(len(src), len(readback))):
        if src[i] != readback[i]:
          print(f"DRAM readback mismatch at byte {i}: wrote 0x{src[i]:02x} read 0x{readback[i]:02x}")
          break
      else:
        print(f"DRAM readback length mismatch: wrote {len(src)} read {len(readback)}")
      raise SystemExit("DRAM readback failed - host write/read not working")
    print(f"DRAM readback OK: {len(readback)} bytes verified")

    rt_args = {
      "brisc": [dst_buf.addr],
      "ncrisc": [src_buf.addr],
      "trisc": [],
    }
    device.run(cores=[core], kernels=kernels, rt_args=rt_args)

    if ns.verify:
      out = device.dram.read(dst_buf)
      for i in range(0, len(out), 2):
        src_bf16 = int.from_bytes(src[i:i + 2], "little")
        src_f32 = _f32_from_bf16(src_bf16)
        exp_bf16 = _bf16_from_f32(src_f32 + 1.0)
        got_bf16 = int.from_bytes(out[i:i + 2], "little")
        if exp_bf16 != got_bf16:
          raise SystemExit(
            f"mismatch at bf16[{i // 2}]: "
            f"src={_fmt_float_noexp(src_f32)} exp={_fmt_float_noexp(_f32_from_bf16(exp_bf16))} got={_fmt_float_noexp(_f32_from_bf16(got_bf16))} "
            f"(src_bf16=0x{src_bf16:04x} exp_bf16=0x{exp_bf16:04x} got_bf16=0x{got_bf16:04x})"
          )
      print("Test Passed")
  finally:
    device.close()

if __name__ == "__main__":
  main()
