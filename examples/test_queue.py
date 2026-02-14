#!/usr/bin/env python3
"""Test queuing multiple different programs with different kernels and RTAs.

Queues: add1(4 tiles) -> matmul_naive(256x256x256) -> add1(2 tiles, different RTAs)
Then drains all outputs and prints/validates them.
"""
from __future__ import annotations
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import random, struct
from codegen import Compiler, DataFormat, CkernelConfig, MathFidelity
from device import Device, Program, DataflowLaunch
from dram import DType

def bf16(x: float) -> int:
  return struct.unpack("<I", struct.pack("<f", x))[0] >> 16

def f32(x: int) -> float:
  return struct.unpack("<f", struct.pack("<I", (x & 0xFFFF) << 16))[0]

# -- add1 kernels --
ADD1_READER = r"""
#include <cstdint>
void kernel_main() {
  uint32_t addr = get_arg_val<uint32_t>(0);
  uint32_t off = get_arg_val<uint32_t>(1);
  uint32_t n = get_arg_val<uint32_t>(2);
  constexpr uint32_t cb = tt::CBIndex::c_0;
  const uint32_t ts = get_tile_size(cb);
  const InterleavedAddrGenFast<true> gen = {
    .bank_base_address = addr, .page_size = ts, .data_format = DataFormat::Float16_b,
  };
  for (uint32_t i = 0; i < n; ++i) {
    cb_reserve_back(cb, 1);
    noc_async_read_tile(off + i, gen, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
  }
}
"""

ADD1_WRITER = r"""
#include <cstdint>
void kernel_main() {
  uint32_t addr = get_arg_val<uint32_t>(0);
  uint32_t off = get_arg_val<uint32_t>(1);
  uint32_t n = get_arg_val<uint32_t>(2);
  constexpr uint32_t cb = tt::CBIndex::c_16;
  const uint32_t ts = get_tile_size(cb);
  const InterleavedAddrGenFast<true> gen = {
    .bank_base_address = addr, .page_size = ts, .data_format = DataFormat::Float16_b,
  };
  for (uint32_t i = 0; i < n; ++i) {
    cb_wait_front(cb, 1);
    noc_async_write_tile(off + i, gen, get_read_ptr(cb));
    noc_async_write_barrier();
    cb_pop_front(cb, 1);
  }
}
"""

ADD1_COMPUTE = r"""
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#ifdef TRISC_MATH
  #include "sfpi.h"
#endif

namespace NAMESPACE {
void MAIN {
  uint32_t n = get_arg_val<uint32_t>(0);
  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
  for (uint32_t i = 0; i < n; ++i) {
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

# -- matmul naive kernels (256x256x256) --
MM, MK, MN = 256, 256, 256
MMt, MKt, MNt = MM // 32, MK // 32, MN // 32
MM_OUT_TILES = MMt * MNt

MM_READER = f"""
#include <cstdint>
void kernel_main() {{
  uint32_t a_addr = get_arg_val<uint32_t>(0);
  uint32_t b_addr = get_arg_val<uint32_t>(1);
  uint32_t start_tile = get_arg_val<uint32_t>(2);
  uint32_t num_tiles = get_arg_val<uint32_t>(3);
  constexpr uint32_t Kt = {MKt}, Nt = {MNt};
  constexpr uint32_t cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1;
  const uint32_t tsa = get_tile_size(cb_a), tsb = get_tile_size(cb_b);
  const InterleavedAddrGenFast<true> a_gen = {{
    .bank_base_address = a_addr, .page_size = tsa, .data_format = DataFormat::Float16_b,
  }};
  const InterleavedAddrGenFast<true> b_gen = {{
    .bank_base_address = b_addr, .page_size = tsb, .data_format = DataFormat::Float16_b,
  }};
  for (uint32_t t = 0; t < num_tiles; ++t) {{
    uint32_t out = start_tile + t;
    for (uint32_t kt = 0; kt < Kt; ++kt) {{
      cb_reserve_back(cb_a, 1);
      noc_async_read_tile((out / Nt) * Kt + kt, a_gen, get_write_ptr(cb_a));
      noc_async_read_barrier();
      cb_push_back(cb_a, 1);
      cb_reserve_back(cb_b, 1);
      noc_async_read_tile(kt * Nt + (out % Nt), b_gen, get_write_ptr(cb_b));
      noc_async_read_barrier();
      cb_push_back(cb_b, 1);
    }}
  }}
}}
"""

MM_WRITER = """
#include <cstdint>
void kernel_main() {
  uint32_t c_addr = get_arg_val<uint32_t>(0);
  uint32_t num_tiles = get_arg_val<uint32_t>(1);
  uint32_t start_tile = get_arg_val<uint32_t>(2);
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t ts = get_tile_size(cb_out);
  const InterleavedAddrGenFast<true> c_gen = {
    .bank_base_address = c_addr, .page_size = ts, .data_format = DataFormat::Float16_b,
  };
  for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
    cb_wait_front(cb_out, 1);
    noc_async_write_tile(t, c_gen, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
  }
}
"""

MM_COMPUTE = f"""
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
  constexpr uint32_t Kt = {MKt};
  constexpr tt::CBIndex cb_a = tt::CBIndex::c_0, cb_b = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
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


def make_const_tile(val: float) -> bytes:
  """Make a single tile (32x32) of constant bf16 values."""
  b = bf16(val)
  return b.to_bytes(2, "little") * (32 * 32)

def make_seq_tile(start: float, step: float = 0.125) -> bytes:
  """Make a tile with sequential bf16 values (first 8 values matter for printing)."""
  buf = bytearray(32 * 32 * 2)
  for i in range(32 * 32):
    buf[i*2:(i+1)*2] = bf16(start + i * step).to_bytes(2, "little")
  return bytes(buf)

def print_tile_corner(label: str, data: bytes, n: int = 8):
  """Print first N values of a tile as floats."""
  vals = [f32(int.from_bytes(data[i*2:(i+1)*2], "little")) for i in range(n)]
  print(f"  {label}: {[f'{v:.4f}' for v in vals]}")


def make_bf16_matrix(rows, cols, seed):
  r = random.Random(seed)
  vals = [r.uniform(-1.0, 1.0) for _ in range(rows * cols)]
  buf = bytearray(rows * cols * 2)
  for i, v in enumerate(vals):
    buf[i*2:(i+1)*2] = bf16(v).to_bytes(2, "little")
  return bytes(buf), vals

def matmul_ref(a, b, m, k, n):
  c = [0.0] * (m * n)
  for i in range(m):
    for j in range(n):
      acc = 0.0
      for kk in range(k):
        acc += a[i * k + kk] * b[kk * n + j]
      c[i * n + j] = acc
  return c


def main():
  tile_bytes = 32 * 32 * 2

  print("Compiling kernels...")
  add1_compiler = Compiler()
  add1_reader = add1_compiler.compile_dataflow(ADD1_READER, processor="ncrisc")
  add1_writer = add1_compiler.compile_dataflow(ADD1_WRITER, processor="brisc")
  add1_compute = add1_compiler.compile_compute(ADD1_COMPUTE)

  mm_cfg = CkernelConfig(
    input_format=DataFormat.Float16_b,
    output_format=DataFormat.Float16_b,
    math_fidelity=MathFidelity.LoFi,
  )
  mm_compiler = Compiler(mm_cfg)
  mm_reader = mm_compiler.compile_dataflow(MM_READER, processor="ncrisc")
  mm_writer = mm_compiler.compile_dataflow(MM_WRITER, processor="brisc")
  mm_compute = mm_compiler.compile_compute(MM_COMPUTE)

  device = Device()
  num_cores = len(device.dispatchable_cores)
  print(f"Device: {num_cores} dispatchable cores\n")

  try:
    # ===== Program 1: add1 on 4 tiles (one core, const values) =====
    # Input: [1.0, 1.0, ...] -> Expected output: [2.0, 2.0, ...]
    add1a_n = 4
    add1a_src_data = b"".join(make_const_tile(1.0) for _ in range(add1a_n))
    add1a_src = device.dram.alloc_write(
      add1a_src_data, name="add1a_src", page_size=tile_bytes,
      dtype=DType.bfloat16, shape=(add1a_n, 32, 32),
    )
    add1a_dst = device.dram.alloc(
      tile_bytes * add1a_n, name="add1a_dst", page_size=tile_bytes,
      dtype=DType.bfloat16, shape=(add1a_n, 32, 32),
    )
    add1a_prog = Program(
      dataflow=[DataflowLaunch(
        cores=device.dispatchable_cores[:1],
        reader=add1_reader, writer=add1_writer,
        reader_rt_args=[add1a_src.addr, 0, add1a_n],
        writer_rt_args=[add1a_dst.addr, 0, add1a_n],
      )],
      compute=add1_compute,
      compute_rt_args=[add1a_n],
      cbs=[0, 16], tile_size=tile_bytes, num_pages=2, cores=1,
    )

    # ===== Program 2: matmul naive (256x256x256) =====
    active_mm = min(num_cores, MM_OUT_TILES)
    mm_tpc = (MM_OUT_TILES + active_mm - 1) // active_mm
    mm_cores = device.dispatchable_cores[:active_mm]
    a_rm, a_f32 = make_bf16_matrix(MM, MK, seed=100)
    b_rm, b_f32 = make_bf16_matrix(MK, MN, seed=200)
    a_buf = device.dram.alloc_write(a_rm, name="mm_A", page_size=tile_bytes, dtype=DType.bfloat16, shape=(MM, MK))
    b_buf = device.dram.alloc_write(b_rm, name="mm_B", page_size=tile_bytes, dtype=DType.bfloat16, shape=(MK, MN))
    c_buf = device.dram.alloc(tile_bytes * MM_OUT_TILES, name="mm_C", page_size=tile_bytes, dtype=DType.bfloat16, shape=(MM, MN))

    def mm_span(ci):
      s = ci * mm_tpc
      return s, 0 if s >= MM_OUT_TILES else min(mm_tpc, MM_OUT_TILES - s)

    mm_prog = Program(
      dataflow=[DataflowLaunch(
        cores=mm_cores,
        reader=mm_reader, writer=mm_writer,
        reader_rt_args=lambda ci, xy, nc: [a_buf.addr, b_buf.addr, *mm_span(ci)],
        writer_rt_args=lambda ci, xy, nc: [c_buf.addr, mm_span(ci)[1], mm_span(ci)[0]],
      )],
      compute=mm_compute,
      compute_rt_args=lambda ci, xy, nc: [mm_span(ci)[1]],
      cbs=[0, 1, 16], tile_size=tile_bytes, num_pages=2, cores=active_mm,
    )

    # ===== Program 3: add1 on 2 tiles (different RTAs, sequential values) =====
    # Input: [0.0, 0.125, 0.25, ...] -> Expected: [1.0, 1.125, 1.25, ...]
    add1b_n = 2
    add1b_src_data = b"".join(make_seq_tile(0.0, 0.125) for _ in range(add1b_n))
    add1b_src = device.dram.alloc_write(
      add1b_src_data, name="add1b_src", page_size=tile_bytes,
      dtype=DType.bfloat16, shape=(add1b_n, 32, 32),
    )
    add1b_dst = device.dram.alloc(
      tile_bytes * add1b_n, name="add1b_dst", page_size=tile_bytes,
      dtype=DType.bfloat16, shape=(add1b_n, 32, 32),
    )
    add1b_prog = Program(
      dataflow=[DataflowLaunch(
        cores=device.dispatchable_cores[:1],
        reader=add1_reader, writer=add1_writer,
        reader_rt_args=[add1b_src.addr, 0, add1b_n],
        writer_rt_args=[add1b_dst.addr, 0, add1b_n],
      )],
      compute=add1_compute,
      compute_rt_args=[add1b_n],
      cbs=[0, 16], tile_size=tile_bytes, num_pages=2, cores=1,
    )

    # ===== Queue all three, run once =====
    print("Queue: add1(4 tiles, const 1.0) -> matmul(256x256x256) -> add1(2 tiles, sequential)")
    device.queue(add1a_prog)
    device.queue(mm_prog)
    device.queue(add1b_prog)
    print("Running 3 programs in one batch...")
    device.run()
    print("Done.\n")

    # ===== Drain and validate =====
    print("--- add1a: 4 tiles of 1.0 + 1.0 = 2.0 ---")
    add1a_out = device.dram.read(add1a_dst)
    for t in range(add1a_n):
      off = t * tile_bytes
      print_tile_corner(f"tile {t}", add1a_out[off:off + tile_bytes])
    # verify every element
    for i in range(0, len(add1a_out), 2):
      got = f32(int.from_bytes(add1a_out[i:i+2], "little"))
      if abs(got - 2.0) > 0.01:
        raise SystemExit(f"add1a mismatch at [{i//2}]: expected 2.0, got {got}")
    print("  PASS\n")

    print(f"--- matmul: C[{MM},{MN}] = A[{MM},{MK}] @ B[{MK},{MN}] ---")
    c_rm_out = device.dram.read(c_buf)
    c_ref = matmul_ref(a_f32, b_f32, MM, MK, MN)
    c_got = [f32(int.from_bytes(c_rm_out[i:i+2], "little")) for i in range(0, len(c_rm_out), 2)]
    mean_ref, mean_got = sum(c_ref) / len(c_ref), sum(c_got) / len(c_got)
    num = sum((r - mean_ref) * (g - mean_got) for r, g in zip(c_ref, c_got))
    den = (sum((r - mean_ref)**2 for r in c_ref) * sum((g - mean_got)**2 for g in c_got)) ** 0.5
    pcc = num / den if den > 0 else 0.0
    print(f"  PCC: {pcc:.6f}")
    if pcc < 0.97:
      raise SystemExit(f"matmul PCC too low: {pcc:.6f}")
    print("  PASS\n")

    print("--- add1b: 2 tiles of sequential values + 1.0 ---")
    add1b_out = device.dram.read(add1b_dst)
    for t in range(add1b_n):
      off = t * tile_bytes
      src_vals = [f32(int.from_bytes(add1b_src_data[off + i*2 : off + (i+1)*2], "little")) for i in range(8)]
      out_vals = [f32(int.from_bytes(add1b_out[off + i*2 : off + (i+1)*2], "little")) for i in range(8)]
      print(f"  tile {t} src:  {[f'{v:.4f}' for v in src_vals]}")
      print(f"  tile {t} out:  {[f'{v:.4f}' for v in out_vals]}")
      print(f"  tile {t} diff: {[f'{o-s:.4f}' for s, o in zip(src_vals, out_vals)]}")
    for i in range(0, len(add1b_out), 2):
      src_bf16 = int.from_bytes(add1b_src_data[i:i+2], "little")
      exp_bf16 = bf16(f32(src_bf16) + 1.0)
      got_bf16 = int.from_bytes(add1b_out[i:i+2], "little")
      if exp_bf16 != got_bf16:
        raise SystemExit(f"add1b mismatch at [{i//2}] src=0x{src_bf16:04x} exp=0x{exp_bf16:04x} got=0x{got_bf16:04x}")
    print("  PASS\n")

    # ===== Test id() dedup: same program queued twice =====
    print("--- dedup test: same add1a queued twice ---")
    device.queue(add1a_prog)
    device.queue(add1a_prog)
    device.run()
    dedup_out = device.dram.read(add1a_dst)
    print_tile_corner("tile 0", dedup_out[:tile_bytes])
    for i in range(0, len(dedup_out), 2):
      got = f32(int.from_bytes(dedup_out[i:i+2], "little"))
      if abs(got - 2.0) > 0.01:
        raise SystemExit(f"dedup mismatch at [{i//2}]: expected 2.0, got {got}")
    print("  PASS\n")

    print("All queue tests passed!")

  finally:
    device.close()

if __name__ == "__main__":
  main()
