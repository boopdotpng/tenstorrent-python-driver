#!/usr/bin/env python3
"""Quick DRAM write/read round-trip test."""
import numpy as np
from device import Device, Dtype, tilize, untilize

dev = Device()
try:
  # Test 1: No tilize (raw bytes, no shape)
  print("=== Test 1: Raw round-trip (no tilize) ===")
  data = bytes(range(256)) * 128  # 32768 bytes = 16 tiles
  buf = dev.alloc(16, dtype=Dtype.Float16_b, name="raw")
  dev.dram.write(buf, data)
  result = dev.dram.read(buf)
  match = data == result
  print(f"Raw round-trip match: {match}")
  if not match:
    for i in range(len(data)):
      if data[i] != result[i]:
        print(f"  First mismatch at byte {i}: expected {data[i]:#04x}, got {result[i]:#04x}")
        break

  # Test 2: tilize then untilize (CPU only, no DRAM)
  print("\n=== Test 2: tilize/untilize CPU round-trip ===")
  shape = (128, 128)
  rng = np.random.default_rng(42)
  src = rng.uniform(-0.5, 0.5, size=shape).astype(np.float16)
  u32 = src.astype(np.float32).view(np.uint32)
  bf16 = (u32 >> 16).astype(np.uint16).tobytes()
  tilized = tilize(bf16, 2, shape)
  recovered = untilize(tilized, 2, shape)
  print(f"tilize/untilize match: {bf16 == recovered}")
  if bf16 != recovered:
    for i in range(len(bf16)):
      if bf16[i] != recovered[i]:
        print(f"  First mismatch at byte {i}")
        break

  # Test 3: Full round-trip with tilize
  print("\n=== Test 3: Full DRAM round-trip with tilize ===")
  buf2 = dev.alloc_write(bf16, dtype=Dtype.Float16_b, shape=shape, name="test")
  result2 = dev.dram_read(buf2)
  print(f"Full round-trip match: {bf16 == result2}")
  if bf16 != result2:
    u16_ref = np.frombuffer(bf16, dtype=np.uint16)
    u16_got = np.frombuffer(result2, dtype=np.uint16)
    ref_f = (u16_ref.astype(np.uint32) << 16).view(np.float32)
    got_f = (u16_got.astype(np.uint32) << 16).view(np.float32)
    pcc = float(np.corrcoef(ref_f, got_f)[0, 1])
    print(f"  PCC={pcc:.6f}")
    for i in range(len(bf16)):
      if bf16[i] != result2[i]:
        print(f"  First mismatch at byte {i}")
        break

finally:
  dev.close()
