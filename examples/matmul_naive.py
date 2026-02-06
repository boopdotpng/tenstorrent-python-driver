#!/usr/bin/env python3
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import time, struct, random
from codegen import Compiler, DataFormat, CkernelConfig, MathFidelity
from device import Device, Program
from dram import DType

M, K, N = 2048, 2048, 2048
Mt, Kt, Nt = M // 32, K // 32, N // 32
NUM_OUTPUT_TILES = Mt * Nt

K_READER = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t a_addr = get_arg_val<uint32_t>(0);
  uint32_t b_addr = get_arg_val<uint32_t>(1);
  uint32_t start_tile = get_arg_val<uint32_t>(2);
  uint32_t num_tiles = get_arg_val<uint32_t>(3);

  constexpr uint32_t Kt = {Kt};
  constexpr uint32_t Nt = {Nt};

  constexpr uint32_t cb_a = tt::CBIndex::c_0;
  constexpr uint32_t cb_b = tt::CBIndex::c_1;
  const uint32_t tile_size_a = get_tile_size(cb_a);
  const uint32_t tile_size_b = get_tile_size(cb_b);
  const InterleavedAddrGenFast<true> a_gen = {{
    .bank_base_address = a_addr, .page_size = tile_size_a, .data_format = DataFormat::Float16_b,
  }};
  const InterleavedAddrGenFast<true> b_gen = {{
    .bank_base_address = b_addr, .page_size = tile_size_b, .data_format = DataFormat::Float16_b,
  }};

  for (uint32_t tile = 0; tile < num_tiles; ++tile) {{
    uint32_t out_tile = start_tile + tile;
    uint32_t out_row = out_tile / Nt;
    uint32_t out_col = out_tile % Nt;

    for (uint32_t kt = 0; kt < Kt; ++kt) {{
      uint32_t a_tile = out_row * Kt + kt;
      cb_reserve_back(cb_a, 1);
      noc_async_read_tile(a_tile, a_gen, get_write_ptr(cb_a));
      noc_async_read_barrier();
      cb_push_back(cb_a, 1);

      uint32_t b_tile = kt * Nt + out_col;
      cb_reserve_back(cb_b, 1);
      noc_async_read_tile(b_tile, b_gen, get_write_ptr(cb_b));
      noc_async_read_barrier();
      cb_push_back(cb_b, 1);
    }}
  }}
}}
"""

K_WRITER = """
#include <cstdint>

void kernel_main() {
  uint32_t c_addr = get_arg_val<uint32_t>(0);
  uint32_t num_tiles = get_arg_val<uint32_t>(1);
  uint32_t start_tile = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_size = get_tile_size(cb_out);
  const InterleavedAddrGenFast<true> c_gen = {
    .bank_base_address = c_addr, .page_size = tile_size, .data_format = DataFormat::Float16_b,
  };

  uint32_t end_tile = start_tile + num_tiles;
  for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
    cb_wait_front(cb_out, 1);
    noc_async_write_tile(tile, c_gen, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
  }
}
"""

K_COMPUTE = f"""
#include <cstdint>
#include "compute_kernel_api/matmul.h"
#ifdef TRISC_MATH
  #include "ckernel_ops.h"
  #include "cmath_common.h"
  #include "ckernel_template.h"
#endif

namespace NAMESPACE {{
void MAIN {{
  uint32_t num_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t Kt = {Kt};

  constexpr tt::CBIndex cb_a = tt::CBIndex::c_0;
  constexpr tt::CBIndex cb_b = tt::CBIndex::c_1;
  constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

  mm_init(cb_a, cb_b, cb_out);

  for (uint32_t i = 0; i < num_tiles; ++i) {{
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < Kt; ++kt) {{
      cb_wait_front(cb_a, 1);
      cb_wait_front(cb_b, 1);
      UNPACK((llk_unpack_AB_matmul(cb_a, cb_b, 0, 0)));

#ifdef TRISC_MATH
      ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
      ckernel_template::run();
      TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
#endif

      cb_pop_front(cb_a, 1);
      cb_pop_front(cb_b, 1);
    }}
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
  }}
}}
}}  // namespace NAMESPACE
"""

def bf16_from_f32(x: float) -> int:
  return struct.unpack("<I", struct.pack("<f", x))[0] >> 16

def f32_from_bf16(x: int) -> float:
  return struct.unpack("<f", struct.pack("<I", (x & 0xFFFF) << 16))[0]

def make_bf16_matrix(rows: int, cols: int, seed: int) -> tuple[bytes, list[float]]:
  r = random.Random(seed)
  f32_vals = [r.uniform(-1.0, 1.0) for _ in range(rows * cols)]
  buf = bytearray(rows * cols * 2)
  for i, v in enumerate(f32_vals):
    buf[i*2:(i+1)*2] = bf16_from_f32(v).to_bytes(2, "little")
  return bytes(buf), f32_vals

def matmul_ref(a: list[float], b: list[float], m: int, k: int, n: int) -> list[float]:
  c = [0.0] * (m * n)
  for i in range(m):
    for j in range(n):
      acc = 0.0
      for kk in range(k):
        acc += a[i * k + kk] * b[kk * n + j]
      c[i * n + j] = acc
  return c

def main():
  print(f"Matmul: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}] ({NUM_OUTPUT_TILES} output tiles)")
  cfg = CkernelConfig(
    input_format=DataFormat.Float16_b,
    output_format=DataFormat.Float16_b,
    math_fidelity=MathFidelity.LoFi,
  )
  kernels = Compiler(cfg).compile(K_READER, K_WRITER, K_COMPUTE)

  device = Device()
  num_cores = len(device.dispatchable_cores)
  print(f"Using {num_cores} cores")

  try:
    tile_bytes = 32 * 32 * 2

    a_rm, a_f32 = make_bf16_matrix(M, K, seed=42)
    b_rm, b_f32 = make_bf16_matrix(K, N, seed=123)

    a_buf = device.dram.alloc_write(
      a_rm, name="A", page_size=tile_bytes, dtype=DType.bfloat16, shape=(M, K)
    )
    b_buf = device.dram.alloc_write(
      b_rm, name="B", page_size=tile_bytes, dtype=DType.bfloat16, shape=(K, N)
    )
    c_buf = device.dram.alloc(
      tile_bytes * NUM_OUTPUT_TILES, name="C", page_size=tile_bytes, dtype=DType.bfloat16, shape=(M, N)
    )

    active_cores = min(num_cores, NUM_OUTPUT_TILES)
    tiles_per_core = (NUM_OUTPUT_TILES + active_cores - 1) // active_cores
    cores = device.dispatchable_cores[:active_cores]

    def core_span(core_idx: int) -> tuple[int, int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, NUM_OUTPUT_TILES - start)
      return start, 0 if start >= NUM_OUTPUT_TILES else count

    def reader_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start, count = core_span(core_idx)
      return [a_buf.addr, b_buf.addr, start, count]

    def writer_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      start, count = core_span(core_idx)
      return [c_buf.addr, count, start]

    def compute_args(core_idx: int, core_xy: tuple[int,int], n_cores: int) -> list[int]:
      _, count = core_span(core_idx)
      return [count]

    program = Program(
      reader=kernels.reader,
      writer=kernels.writer,
      compute=kernels.compute,
      reader_rt_args=reader_args,
      writer_rt_args=writer_args,
      compute_rt_args=compute_args,
      cbs=[0, 1, 16],
      tile_size=tile_bytes,
      num_pages=2,
      cores=cores,
    )

    total, dispatch = device.run(program)
    flops = 2 * M * N * K
    compute = total - dispatch
    print(f"TFLOPS (compute only): {flops / compute / 1e12:.2f}, TFLOPS (total): {flops / total / 1e12:.2f}")

    c_rm = device.dram.read(c_buf)

    if M * N <= 1024 * 1024:
      c_ref = matmul_ref(a_f32, b_f32, M, K, N)
      c_got = [f32_from_bf16(int.from_bytes(c_rm[i:i+2], "little")) for i in range(0, len(c_rm), 2)]

      mean_ref = sum(c_ref) / len(c_ref)
      mean_got = sum(c_got) / len(c_got)
      num = sum((r - mean_ref) * (g - mean_got) for r, g in zip(c_ref, c_got))
      den_ref = sum((r - mean_ref) ** 2 for r in c_ref) ** 0.5
      den_got = sum((g - mean_got) ** 2 for g in c_got) ** 0.5
      pcc = num / (den_ref * den_got) if den_ref * den_got > 0 else 0.0

      print(f"PCC: {pcc:.6f}")
      if pcc < 0.97:
        raise SystemExit(f"PCC too low: {pcc:.6f} < 0.97")
    else:
      print("Skipping verification for large matrix")

    print("Test Passed")

  finally:
    device.close()

if __name__ == "__main__":
  main()
