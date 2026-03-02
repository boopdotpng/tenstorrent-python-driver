#!/usr/bin/env python3
"""SFPU micro-benchmark for Blackhole P100A.

Measures SFPU instruction throughput for various operations by running each op
N_REPS times on a single tile per core, then reading kernel timing from the
device profiler.

The benchmark subtracts a load/store baseline to isolate pure SFPU compute cost.

Usage:
  TT_PROFILER=1 python3 examples/sfpu_bench.py
"""
from __future__ import annotations
import os, random, struct

os.environ.setdefault("TT_PROFILER", "1")

from compiler import Compiler
from device import Device, Program, DataflowLaunch
from dram import DType

N_REPS = 1000
TILE_ELEMS = 32 * 32

# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------

K_READER = r"""
#include <cstdint>
void kernel_main() {
  uint32_t addr = get_arg_val<uint32_t>(0);
  uint32_t off  = get_arg_val<uint32_t>(1);
  constexpr uint32_t cb = tt::CBIndex::c_0;
  const uint32_t tsz = get_tile_size(cb);
  const InterleavedAddrGenFast<true> src = {
    .bank_base_address = addr, .page_size = tsz,
    .data_format = DataFormat::Float16_b,
  };
  cb_reserve_back(cb, 1);
  noc_async_read_tile(off, src, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
}
"""

K_WRITER = r"""
#include <cstdint>
void kernel_main() {
  uint32_t addr = get_arg_val<uint32_t>(0);
  uint32_t off  = get_arg_val<uint32_t>(1);
  constexpr uint32_t cb = tt::CBIndex::c_16;
  const uint32_t tsz = get_tile_size(cb);
  const InterleavedAddrGenFast<true> dst = {
    .bank_base_address = addr, .page_size = tsz,
    .data_format = DataFormat::Float16_b,
  };
  cb_wait_front(cb, 1);
  noc_async_write_tile(off, dst, get_read_ptr(cb));
  noc_async_write_barrier();
  cb_pop_front(cb, 1);
}
"""

K_COMPUTE = r"""
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
  #include "sfpi.h"
  using namespace sfpi;
#endif

namespace NAMESPACE {
void MAIN {
  uint32_t n_reps = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_in  = tt::CBIndex::c_0;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  init_sfpu(cb_in, cb_out);

  tile_regs_acquire();
  cb_wait_front(cb_in, 1);
  copy_tile(cb_in, 0, 0);

#ifdef TRISC_MATH
  {OP_INIT}
  {
    DeviceZoneScopedN("sfpu_op");
    for (uint32_t rep = 0; rep < n_reps; ++rep) {
      for (uint32_t v = 0; v < 32; ++v) {
        {OP_BODY}
      }
    }
  }
#endif

  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(cb_out, 1);
  pack_tile(0, cb_out);
  cb_pop_front(cb_in, 1);
  tile_regs_release();
  cb_push_back(cb_out, 1);
}
}
"""

# ---------------------------------------------------------------------------
# SFPU operations to benchmark
# ---------------------------------------------------------------------------

OPS = dict([
  ("load_store", {
    "init": "",
    "body": "{ vFloat t = dst_reg[v]; dst_reg[v] = t; }",
    "desc": "Load+Store (baseline)",
    "flops": 0,
  }),
  ("add", {
    "init": "",
    "body": "dst_reg[v] = dst_reg[v] + vConst1;",
    "desc": "FP add",
    "flops": 1,
  }),
  ("mul", {
    "init": "vConstFloatPrgm0 = 0.99f;",
    "body": "dst_reg[v] = dst_reg[v] * vConstFloatPrgm0;",
    "desc": "FP multiply",
    "flops": 1,
  }),
  ("mad", {
    "init": "vConstFloatPrgm0 = 0.99f; vConstFloatPrgm1 = 0.01f;",
    "body": "dst_reg[v] = dst_reg[v] * vConstFloatPrgm0 + vConstFloatPrgm1;",
    "desc": "Fused multiply-add (SFPMAD)",
    "flops": 2,
  }),
  ("recip_hw", {
    "init": "",
    "body": "dst_reg[v] = approx_recip(dst_reg[v]);",
    "desc": "HW approx recip (SFPARECIP)",
    "flops": 1,
  }),
  ("recip_nr", {
    "init": "vConstFloatPrgm0 = 2.0f;",
    "body": "\n".join([
      "      {",
      "        vFloat x = dst_reg[v];",
      "        vFloat y = approx_recip(x);",
      "        vFloat t = x * y - vConstFloatPrgm0;",
      "        y = y * -t - vConst0;",
      "        t = x * y - vConstFloatPrgm0;",
      "        dst_reg[v] = y * -t - vConst0;",
      "      }",
    ]),
    "desc": "Recip + 2 NR iters",
    "flops": 9,
  }),
  ("poly3", {
    "init": "\n".join([
      "  vConstFloatPrgm0 = 0.1058f;",
      "  vConstFloatPrgm1 = -0.7166f;",
      "  vConstFloatPrgm2 = 2.0871f;",
    ]),
    "body": "\n".join([
      "      {",
      "        vFloat x = dst_reg[v];",
      "        vFloat r = x * vConstFloatPrgm0 + vConstFloatPrgm1;",
      "        r = r * x + vConstFloatPrgm2;",
      "        r = r * x + vConst1;",
      "        dst_reg[v] = r;",
      "      }",
    ]),
    "desc": "Deg-3 Horner (~ log body)",
    "flops": 6,
  }),
  ("poly7", {
    "init": "",
    "body": "\n".join([
      "      {",
      "        vFloat x = dst_reg[v];",
      "        vFloat r = x * 0.00014f + 0.00139f;",
      "        r = r * x + 0.00833f;",
      "        r = r * x + 0.04167f;",
      "        r = r * x + 0.16667f;",
      "        r = r * x + 0.5f;",
      "        r = r * x + vConst1;",
      "        r = r * x + vConst1;",
      "        dst_reg[v] = r;",
      "      }",
    ]),
    "desc": "Deg-7 Horner (~ exp body)",
    "flops": 14,
  }),
  ("sin7", {
    "init": "",
    "body": "\n".join([
      "      {",
      "        vFloat x = dst_reg[v];",
      "        vFloat x2 = x * x;",
      "        vFloat x3 = x2 * x;",
      "        vFloat x5 = x3 * x2;",
      "        vFloat x7 = x5 * x2;",
      "        dst_reg[v] = x - x3 * 0.16667f + x5 * 0.00833f - x7 * 0.000198f;",
      "      }",
    ]),
    "desc": "sin(x) Maclaurin 7th order",
    "flops": 13,
  }),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kernel(op):
  return K_COMPUTE.replace("{OP_INIT}", op["init"]).replace("{OP_BODY}", op["body"])

def bf16(x):
  return struct.unpack("<I", struct.pack("<f", x))[0] >> 16

def make_buf(n_tiles, seed=42):
  r = random.Random(seed)
  out = bytearray(n_tiles * TILE_ELEMS * 2)
  for i in range(n_tiles * TILE_ELEMS):
    val = r.uniform(0.1, 2.0)
    out[i*2:(i+1)*2] = bf16(val).to_bytes(2, "little")
  return bytes(out)

def extract_trisc1_kern_cycles(profile):
  """Return list of TRISC1 kernel cycle counts, one per core."""
  cycles = []
  for prog in profile.get("programs", []):
    for core_data in prog.get("profiles", {}).values():
      riscs = core_data.get("riscs", [])
      if len(riscs) <= 3:
        continue
      trisc1 = riscs[3]
      ks, ke = trisc1.get("kern_start"), trisc1.get("kern_end")
      if ks and ke and ke > ks:
        cycles.append(ke - ks)
  return cycles

def extract_sfpu_zone_cycles(profile):
  """Return list of 'sfpu_op' zone durations from TRISC1."""
  # Find the hash for "sfpu_op"
  target_hashes = set()
  for h_str, info in profile.get("zone_names", {}).items():
    if info.get("name") == "sfpu_op":
      target_hashes.add(int(h_str))
  if not target_hashes:
    return []
  durations = []
  for prog in profile.get("programs", []):
    for core_data in prog.get("profiles", {}).values():
      riscs = core_data.get("riscs", [])
      if len(riscs) <= 3:
        continue
      trisc1 = riscs[3]
      starts, ends = [], []
      for z in trisc1.get("custom", []):
        if z["hash"] in target_hashes:
          if z["type"] == 0:
            starts.append(z["ts"])
          elif z["type"] == 1:
            ends.append(z["ts"])
      for s, e in zip(starts, ends):
        if e > s:
          durations.append(e - s)
  return durations

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  compiler = Compiler()
  reader = compiler.compile_dataflow(K_READER, processor="ncrisc")
  writer = compiler.compile_dataflow(K_WRITER, processor="brisc")

  device = Device()
  try:
    cores = device.dispatchable_cores
    n_cores = len(cores)
    freq_mhz = device.profiler_freq_mhz()
    tile_size = TILE_ELEMS * 2

    src_data = make_buf(n_cores)
    src_buf = device.dram.alloc_write(
      src_data, name="bench_src", page_size=tile_size,
      dtype=DType.bfloat16, shape=(n_cores, 32, 32),
    )
    dst_buf = device.dram.alloc(
      tile_size * n_cores, name="bench_dst", page_size=tile_size,
      dtype=DType.bfloat16, shape=(n_cores, 32, 32),
    )

    results = {}
    for op_name, op in OPS.items():
      kern_src = make_kernel(op)
      compute = compiler.compile_compute(kern_src)
      program = Program(
        dataflow=[DataflowLaunch(
          cores=cores,
          reader=reader,
          writer=writer,
          reader_rt_args=lambda ci, _cxy, _nc: [src_buf.addr, ci, 1],
          writer_rt_args=lambda ci, _cxy, _nc: [dst_buf.addr, ci, 1],
        )],
        compute=compute,
        compute_rt_args=lambda ci, _cxy, _nc: [N_REPS],
        cbs=[0, 16],
        tile_size=tile_size,
        num_pages=2,
        sources={"compute": kern_src},
        name=f"sfpu_{op_name}",
      )
      device.queue(program)
      device.run()

      kern_cyc = extract_trisc1_kern_cycles(device.last_profile)
      zone_cyc = extract_sfpu_zone_cycles(device.last_profile)
      cyc = zone_cyc if zone_cyc else kern_cyc
      results[op_name] = {
        "avg": sum(cyc) / len(cyc) if cyc else 0,
        "min": min(cyc) if cyc else 0,
        "max": max(cyc) if cyc else 0,
        "kern_avg": sum(kern_cyc) / len(kern_cyc) if kern_cyc else 0,
        **op,
      }

    print_results(results, freq_mhz, n_cores)
  finally:
    device.close()


def print_results(results, freq_mhz, n_cores):
  baseline = results.get("load_store", {}).get("avg", 0)

  hdr = (
    f"\n{'='*90}\n"
    f"  SFPU Benchmark  ({n_cores} cores @ {freq_mhz} MHz, {N_REPS} reps/tile)\n"
    f"{'='*90}"
  )
  print(hdr)
  print(f"  {'Op':<14} {'Zone cyc':>9} {'SFPU cyc':>9} {'cyc/tile':>9} {'elem/cyc':>9} {'GFLOPS':>8}  {'Desc'}")
  print(f"  {'-'*86}")

  for name, r in results.items():
    avg = r["avg"]
    sfpu = max(0, avg - baseline)
    per_tile = sfpu / N_REPS if N_REPS else 0
    epc = TILE_ELEMS / per_tile if per_tile > 0 else float('inf')
    flops = r["flops"]
    total_flops = n_cores * TILE_ELEMS * N_REPS * flops
    seconds = avg / (freq_mhz * 1e6) if avg > 0 else 1
    gflops = total_flops / seconds / 1e9 if flops > 0 else 0
    print(
      f"  {name:<14} {avg:>9.0f} {sfpu:>9.0f} {per_tile:>9.1f}"
      f" {epc:>9.1f} {gflops:>8.1f}  {r['desc']}"
    )

  print(f"{'='*90}")

  # Per-core throughput summary
  print(f"\n  Per-core throughput (32 SIMD lanes, {freq_mhz} MHz):")
  for name, r in results.items():
    if name == "load_store":
      continue
    sfpu = max(0, r["avg"] - baseline)
    per_tile = sfpu / N_REPS if N_REPS else 0
    if per_tile > 0:
      epc = TILE_ELEMS / per_tile
      us = per_tile / freq_mhz
      print(f"    {name:<14} {per_tile:>6.1f} cyc/tile  {epc:>6.1f} elem/cyc  {us:>8.4f} us/tile")

  print()


if __name__ == "__main__":
  main()
