#!/usr/bin/env python3
"""Peak matmul benchmark — port of tt-metal's matmul_peak_fast_dispatch_v2.

Measures throughput in TFLOPS using all 118 P100A cores with K-blocked compute
and double-buffered circular buffers. Skips verification — purely a perf test.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import time
from codegen import Compiler, DataFormat, CkernelConfig, MathFidelity
from device import Device, Program

# Matrix dimensions (same as tt-metal v2: 5120x4096 @ 4096x5632)
M, K, N = 5120, 4096, 5632
Mt, Kt, Nt = M // 32, K // 32, N // 32
NUM_OUTPUT_TILES = Mt * Nt

IN0_BLOCK_W = 2                # K-dimension block width in tiles
NUM_BLOCKS = Kt // IN0_BLOCK_W # number of K-blocks per output tile

WARMUP_ITERS = 2
TIMED_ITERS = 5

# -- Reader: batched A+B DRAM reads per K-block --
K_READER = f"""
#include <cstdint>

void kernel_main() {{
  uint32_t a_addr     = get_arg_val<uint32_t>(0);
  uint32_t b_addr     = get_arg_val<uint32_t>(1);
  uint32_t start_tile = get_arg_val<uint32_t>(2);
  uint32_t num_tiles  = get_arg_val<uint32_t>(3);

  constexpr uint32_t Kt = {Kt};
  constexpr uint32_t Nt = {Nt};
  constexpr uint32_t in0_block_w = {IN0_BLOCK_W};
  constexpr uint32_t num_blocks = Kt / in0_block_w;

  constexpr uint32_t cb_a = tt::CBIndex::c_0;
  constexpr uint32_t cb_b = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(cb_a);
  const InterleavedAddrGenFast<true> a_gen = {{
    .bank_base_address = a_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};
  const InterleavedAddrGenFast<true> b_gen = {{
    .bank_base_address = b_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  }};

  for (uint32_t t = 0; t < num_tiles; ++t) {{
    uint32_t out_tile = start_tile + t;
    uint32_t row = out_tile / Nt;
    uint32_t col = out_tile % Nt;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {{
      uint32_t k0 = blk * in0_block_w;

      // Reserve space in both CBs, issue all reads, single barrier
      cb_reserve_back(cb_a, in0_block_w);
      cb_reserve_back(cb_b, in0_block_w);
      uint32_t l1_a = get_write_ptr(cb_a);
      uint32_t l1_b = get_write_ptr(cb_b);

      for (uint32_t i = 0; i < in0_block_w; ++i) {{
        noc_async_read_tile(row * Kt + k0 + i, a_gen, l1_a);
        l1_a += tile_bytes;
        noc_async_read_tile((k0 + i) * Nt + col, b_gen, l1_b);
        l1_b += tile_bytes;
      }}
      noc_async_read_barrier();
      cb_push_back(cb_a, in0_block_w);
      cb_push_back(cb_b, in0_block_w);
    }}
  }}
}}
"""

# -- Writer: unchanged from naive --
K_WRITER = """
#include <cstdint>

void kernel_main() {
  uint32_t c_addr     = get_arg_val<uint32_t>(0);
  uint32_t num_tiles  = get_arg_val<uint32_t>(1);
  uint32_t start_tile = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_out);
  const InterleavedAddrGenFast<true> c_gen = {
    .bank_base_address = c_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b,
  };

  for (uint32_t i = 0; i < num_tiles; ++i) {
    cb_wait_front(cb_out, 1);
    noc_async_write_tile(start_tile + i, c_gen, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
  }
}
"""

# -- Compute: K-blocked matmul with in0_block_w tiles per block --
K_COMPUTE = f"""
#include <cstdint>
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {{
void MAIN {{
  uint32_t num_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t in0_block_w = {IN0_BLOCK_W};
  constexpr uint32_t num_blocks = {NUM_BLOCKS};

  constexpr tt::CBIndex cb_a   = tt::CBIndex::c_0;
  constexpr tt::CBIndex cb_b   = tt::CBIndex::c_1;
  constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

  mm_init(cb_a, cb_b, cb_out);

  for (uint32_t t = 0; t < num_tiles; ++t) {{
    tile_regs_acquire();
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {{
      cb_wait_front(cb_a, in0_block_w);
      cb_wait_front(cb_b, in0_block_w);
      for (uint32_t i = 0; i < in0_block_w; ++i) {{
        matmul_tiles(cb_a, cb_b, i, i, 0);
      }}
      cb_pop_front(cb_a, in0_block_w);
      cb_pop_front(cb_b, in0_block_w);
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

def main():
  print(f"Matmul Peak: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}] (bf16, LoFi)")
  print(f"  Mt={Mt} Kt={Kt} Nt={Nt} in0_block_w={IN0_BLOCK_W} num_blocks={NUM_BLOCKS}")

  cfg = CkernelConfig(
    input_format=DataFormat.Float16_b,
    output_format=DataFormat.Float16_b,
    math_fidelity=MathFidelity.LoFi,
  )
  kernels = Compiler(cfg).compile(K_READER, K_WRITER, K_COMPUTE)

  device = Device()
  num_cores = len(device.dispatchable_cores)
  print(f"Device: {num_cores} cores")

  try:
    tile_bytes = 32 * 32 * 2

    # Allocate DRAM (no fill needed for perf-only benchmark)
    a_buf = device.dram.alloc(tile_bytes * Mt * Kt, name="A", page_size=tile_bytes)
    b_buf = device.dram.alloc(tile_bytes * Kt * Nt, name="B", page_size=tile_bytes)
    c_buf = device.dram.alloc(tile_bytes * NUM_OUTPUT_TILES, name="C", page_size=tile_bytes)

    active_cores = min(num_cores, NUM_OUTPUT_TILES)
    tiles_per_core = (NUM_OUTPUT_TILES + active_cores - 1) // active_cores

    def span(idx):
      start = idx * tiles_per_core
      return start, max(0, min(tiles_per_core, NUM_OUTPUT_TILES - start))

    program = Program(
      reader=kernels.reader,
      writer=kernels.writer,
      compute=kernels.compute,
      reader_rt_args=lambda i, xy, n: [a_buf.addr, b_buf.addr, span(i)[0], span(i)[1]],
      writer_rt_args=lambda i, xy, n: [c_buf.addr, span(i)[1], span(i)[0]],
      compute_rt_args=lambda i, xy, n: [span(i)[1]],
      cbs=[0, 1, 16],
      tile_size=tile_bytes,
      num_pages=IN0_BLOCK_W * 2,  # double-buffered K-blocks
      cores=device.dispatchable_cores[:active_cores],
    )

    print(f"Warmup ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
      device.run(program)

    print(f"Timing ({TIMED_ITERS} iters)...")
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
      device.run(program)
    elapsed = (time.perf_counter() - t0) / TIMED_ITERS

    flops = 2.0 * M * N * K
    print(f"\nAvg latency: {elapsed * 1e3:.3f} ms")
    print(f"Throughput:  {flops / elapsed / 1e12:.2f} TFLOPS (LoFi bf16, {active_cores} cores)")

  finally:
    device.close()

if __name__ == "__main__":
  main()
