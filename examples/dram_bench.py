#!/usr/bin/env python3
"""Benchmark DRAM<->host bandwidth with raw MMIO and/or kernel fill/drain routes."""
from __future__ import annotations

import os
import time

from device import Device
from defs import DRAM_ALIGNMENT
from helpers import USE_USB_DISPATCH


def _env_choice(name: str, default: str, valid: set[str]) -> str:
  raw = os.environ.get(name, default).strip().lower()
  if raw not in valid:
    raise SystemExit(f"{name} must be one of {sorted(valid)}, got {raw!r}")
  return raw


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name)
  if raw is None:
    return default
  try:
    value = int(raw)
  except ValueError as e:
    raise SystemExit(f"{name} must be an integer, got {raw!r}") from e
  if value <= 0:
    raise SystemExit(f"{name} must be > 0, got {value}")
  return value


def _bench(label: str, fn, size_bytes: int, iters: int) -> float:
  best_s = float("inf")
  for _ in range(iters):
    t0 = time.perf_counter()
    fn()
    best_s = min(best_s, time.perf_counter() - t0)
  bw_gbps = size_bytes / best_s / 1e9
  print(f"  {label}: {bw_gbps:.3f} GB/s  ({best_s * 1e3:.1f} ms, {size_bytes / 1e6:.1f} MB)")
  return bw_gbps


def _run_slow_mode(device: Device, size_bytes: int, iters: int, page_size: int) -> tuple[float, float]:
  print(f"\nSlow mode (raw MMIO), page_size={page_size // 1024} KB:")
  write_data = os.urandom(size_bytes)
  buf = device.dram.alloc(size_bytes, name=f"raw_mmio_{size_bytes // (1024 * 1024)}mb", page_size=page_size)
  h2d = _bench("host->dram", lambda: device.dram._write_slow(buf, write_data), size_bytes, iters)
  device.dram._write_slow(buf, write_data)
  d2h = _bench("dram->host", lambda: device.dram._read_slow(buf), size_bytes, iters)
  return h2d, d2h


def _run_kernel_mode(device: Device, size_bytes: int, iters: int, page_size: int, verify: bool) -> tuple[float, float] | None:
  if device.dram.sysmem is None:
    print("\nKernel mode unavailable: sysmem pinning is disabled in slow dispatch (`TT_USB=1`).")
    return None
  if page_size != 2048:
    raise SystemExit(
      "Kernel mode currently requires DRAM_BENCH_KERNEL_PAGE_KB=2; larger page sizes are known to corrupt transfers."
    )

  from dram import _drain_kernel, _fill_kernel
  fill = _fill_kernel()
  drain = _drain_kernel()

  print(f"\nKernel mode (fill/drain), page_size={page_size // 1024} KB:")
  write_data = os.urandom(size_bytes)
  buf = device.dram.alloc(size_bytes, name=f"kernel_xfer_{size_bytes // (1024 * 1024)}mb", page_size=page_size)

  if verify:
    device.dram.sysmem.buf[:size_bytes] = write_data
    device.dram._run_transfer_kernel(buf, fill, size_bytes)
    device.dram.sysmem.buf[:size_bytes] = b"\0" * size_bytes
    device.dram._run_transfer_kernel(buf, drain, size_bytes, extra_args=[0])
    out = bytes(device.dram.sysmem.buf[:size_bytes])
    if out != write_data:
      raise RuntimeError("kernel transfer verification failed: round-trip data mismatch")
    print("  verification: pass")

  # Warmup to avoid first-iteration outliers.
  device.dram.sysmem.buf[:size_bytes] = write_data
  device.dram._run_transfer_kernel(buf, fill, size_bytes)
  device.dram._run_transfer_kernel(buf, drain, size_bytes, extra_args=[0])

  # Measure transfer-only kernel bandwidth (exclude host memcpy/memset overhead).
  device.dram.sysmem.buf[:size_bytes] = write_data
  h2d = _bench("host->dram", lambda: device.dram._run_transfer_kernel(buf, fill, size_bytes), size_bytes, iters)
  d2h = _bench("dram->host", lambda: device.dram._run_transfer_kernel(buf, drain, size_bytes, extra_args=[0]), size_bytes, iters)
  return h2d, d2h


def main():
  size_mb = _env_int("DRAM_BENCH_MB", 64)
  iters = _env_int("DRAM_BENCH_ITERS", 3)
  mode = _env_choice("DRAM_BENCH_MODE", "both", {"slow", "kernel", "both"})
  slow_page_kb = _env_int("DRAM_BENCH_PAGE_KB", 1024)
  kernel_page_kb = _env_int("DRAM_BENCH_KERNEL_PAGE_KB", 2)
  verify = _env_int("DRAM_BENCH_VERIFY", 1) != 0

  slow_page_size = slow_page_kb * 1024
  kernel_page_size = kernel_page_kb * 1024
  if slow_page_size % DRAM_ALIGNMENT != 0:
    raise SystemExit(f"DRAM_BENCH_PAGE_KB must align to {DRAM_ALIGNMENT} bytes")
  if kernel_page_size % DRAM_ALIGNMENT != 0:
    raise SystemExit(f"DRAM_BENCH_KERNEL_PAGE_KB must align to {DRAM_ALIGNMENT} bytes")

  dispatch = "slow-dispatch" if USE_USB_DISPATCH else "fast-dispatch"
  print(f"DRAM benchmark ({dispatch})")
  print(f"  mode={mode}, size={size_mb} MB, iters={iters}, verify={int(verify)}")
  print(f"  slow_page={slow_page_kb} KB, kernel_page={kernel_page_kb} KB")

  size_bytes = size_mb * 1024 * 1024

  device = Device()
  try:
    print(f"  cores: {len(device.dispatchable_cores)}, banks: {len(device.dram.bank_tiles)}")
    results: dict[str, tuple[float, float]] = {}

    if mode in {"slow", "both"}:
      results["slow"] = _run_slow_mode(device, size_bytes, iters, slow_page_size)
    if mode in {"kernel", "both"}:
      kernel_res = _run_kernel_mode(device, size_bytes, iters, kernel_page_size, verify)
      if kernel_res is not None:
        results["kernel"] = kernel_res

    if "slow" in results and "kernel" in results:
      sh2d, sd2h = results["slow"]
      kh2d, kd2h = results["kernel"]
      print("\nComparison (kernel / slow):")
      print(f"  host->dram: {kh2d / sh2d:.2f}x")
      print(f"  dram->host: {kd2h / sd2h:.2f}x")
  finally:
    device.close()


if __name__ == "__main__":
  main()
