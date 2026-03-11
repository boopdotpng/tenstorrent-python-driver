#!/usr/bin/env python3
"""Matmul peak throughput benchmark using MatmulProgram template."""
from __future__ import annotations

import os, sys
from pathlib import Path
import numpy as np

from device import Device, Dtype, MathFidelity, untilize
from ops import plan_matmul, MatmulProgram

def ceil32(x): return (x + 31) & ~31

F32_ACC = os.environ.get("F32_ACC") == "1"
IO_MODE = "f16" if os.environ.get("F16") == "1" else "bf16"
IO_DTYPE = Dtype.Float16 if IO_MODE == "f16" else Dtype.Float16_b
_MATH_FIDELITY_MAP = {"lofi": MathFidelity.LoFi, "hifi2": MathFidelity.HiFi2,
                      "hifi3": MathFidelity.HiFi3, "hifi4": MathFidelity.HiFi4}
MATH_FIDELITY_NAME = os.environ.get("MATH_FIDELITY", "hifi2").lower()
MATH_FIDELITY = _MATH_FIDELITY_MAP.get(MATH_FIDELITY_NAME)
if MATH_FIDELITY is None:
  raise SystemExit(f"Invalid MATH_FIDELITY={MATH_FIDELITY_NAME!r}. Expected: {', '.join(_MATH_FIDELITY_MAP)}")

WARMUP_ITERS = 2
TIMED_ITERS = 5

def _to_device_bytes(x: np.ndarray) -> bytes:
  x16 = np.ascontiguousarray(x, dtype=np.float16)
  if IO_MODE == "f16": return x16.tobytes()
  u32 = x16.astype(np.float32).view(np.uint32)
  return (u32 >> 16).astype(np.uint16).tobytes()

def _from_device_bytes(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
  if IO_MODE == "f16":
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
  u16 = np.frombuffer(data, dtype=np.uint16)
  return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)

def _validate(a, b, c_bytes, M, N, Mp, Np):
  c_ref = a @ b
  c_full = _from_device_bytes(c_bytes, (Mp, Np))
  c_got = c_full[:M, :N]
  ref_flat, got_flat = c_ref.reshape(-1), c_got.reshape(-1)
  if not np.all(np.isfinite(got_flat)):
    bad = int(got_flat.size - np.count_nonzero(np.isfinite(got_flat)))
    raise SystemExit(f"Validation failed: {bad} non-finite values")
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

  Kp, Np = ceil32(K), max(32, ceil32(N))
  Mt_base, Kt, Nt_base = ceil32(M) // 32, Kp // 32, Np // 32

  device = Device()
  try:
    plan = plan_matmul(Mt_base, Kt, Nt_base, device.cores, io_dtype=IO_DTYPE, f32_acc=F32_ACC)
    Mp = plan.mt * 32
    Np = plan.nt * 32
    padded = (M != Mp or K != Kp or N != Np)

    print(
      f"Matmul Peak: C[{M},{N}] = A[{M},{K}] @ B[{K},{N}] "
      f"({IO_MODE} io, {'fp32' if F32_ACC else 'mixed'} accum, {MATH_FIDELITY.name})"
    )
    if padded: print(f"  padded: A[{Mp},{Kp}] @ B[{Kp},{Np}] -> C[{Mp},{Np}]")
    print(f"  Mt={plan.mt} Kt={Kt} Nt={plan.nt} grid={plan.num_rows}x{plan.num_cols}")
    print(f"  per_core_M={plan.per_core_m} per_core_N={plan.per_core_n} "
          f"in0_block_w={plan.in0_block_w} num_blocks={plan.num_blocks}")
    print(f"  subblock: {plan.out_subblock_h}h x {plan.out_subblock_w}w = {plan.out_subblock_num_tiles} tiles")
    print(f"  cores: {plan.active_core_count} ({len(device.cores)} available)")

    # Generate input data
    rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(123)
    a_src = rng_a.uniform(-0.5, 0.5, size=(M, K)).astype(np.float16)
    b_src = rng_b.uniform(-0.5, 0.5, size=(K, N)).astype(np.float16)

    if padded:
      a_padded = np.zeros((Mp, Kp), dtype=np.float16); a_padded[:M, :K] = a_src
      b_padded = np.zeros((Kp, Np), dtype=np.float16); b_padded[:K, :N] = b_src
      a_bytes, b_bytes = _to_device_bytes(a_padded), _to_device_bytes(b_padded)
    else:
      a_bytes, b_bytes = _to_device_bytes(a_src), _to_device_bytes(b_src)

    a_ref = _from_device_bytes(_to_device_bytes(a_src), (M, K))
    b_ref = _from_device_bytes(_to_device_bytes(b_src), (K, N))

    tile_bytes = IO_DTYPE.tile_size
    a_buf = device.alloc_write(a_bytes, dtype=IO_DTYPE, shape=(Mp, Kp), name="A")
    b_buf = device.alloc_write(b_bytes, dtype=IO_DTYPE, shape=(Kp, Np), name="B")
    c_buf = device.alloc(plan.mt * plan.nt, dtype=IO_DTYPE, name="C", shape=(Mp, Np))

    prog = MatmulProgram(plan, a_buf, b_buf, c_buf,
                         io_dtype=IO_DTYPE, math_fidelity=MATH_FIDELITY, f32_acc=F32_ACC)

    print(f"\nWarmup ({WARMUP_ITERS} iters)...")
    for _ in range(WARMUP_ITERS):
      device.queue(prog)
    device.run()
    if os.environ.get("TT_DEBUG_CQ_DUMP") == "1":
      return

    print(f"Timing ({TIMED_ITERS} iters)...")
    for _ in range(TIMED_ITERS):
      device.queue(prog)
    device.run()

    c_raw = device.dram_read(c_buf)
    _validate(a_ref, b_ref, c_raw, M, N, Mp, Np)

    if device.last_profile is not None:
      import json
      out = Path("profiler_data.json")
      out.write_text(json.dumps(device.last_profile))
      print(f"Profiler data written to {out.resolve()}")
      if os.environ.get("TT_PROFILER_SERVE") == "1":
        device.serve_profile()

  finally:
    device.close()

if __name__ == "__main__":
  main()
