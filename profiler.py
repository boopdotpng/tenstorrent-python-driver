"""Device kernel profiler — reads wall-clock markers from L1 after kernel execution."""
import struct
from defs import TensixL1, CoreList

RISC_NAMES = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")

# profiler_common.h: ControlBuffer enum indices
_DEVICE_BUF_END = 5   # DEVICE_BUFFER_END_INDEX_BR_ER .. +4
_NOC_X = 14
_NOC_Y = 15
_DROPPED = 18
_DONE = 19

# profiler_common.h: BufferIndex guaranteed marker positions (uint32 indices)
_GUARANTEED_1_H = 4   # FW scope start
_GUARANTEED_2_H = 6   # FW scope end
_GUARANTEED_3_H = 8   # Kernel child scope start
_GUARANTEED_4_H = 10  # Kernel child scope end
_CUSTOM_START = 12    # Optional markers begin here

def _parse_ts(w0, w1):
  """Extract 44-bit wall-clock timestamp from a profiler marker pair."""
  if not (w0 & 0x80000000): return None
  hi12 = w0 & 0xFFF
  return (hi12 << 32) | w1

def _parse_marker(w0, w1):
  """Parse a marker into (zone_hash, packet_type, timestamp) or None."""
  if not (w0 & 0x80000000): return None
  timer_id = (w0 >> 12) & 0x7FFFF
  hi12 = w0 & 0xFFF
  ts = (hi12 << 32) | w1
  zone_hash = timer_id & 0xFFFF
  ptype = (timer_id >> 16) & 0x7  # 0=START, 1=END, 2=TOTAL
  return zone_hash, ptype, ts

def read_core(win, core):
  """Read profiler data for one core via a TLB window. Returns dict with per-RISC timing."""
  ctrl_raw = bytes(win.uc[TensixL1.PROFILER_CONTROL : TensixL1.PROFILER_CONTROL + 128])
  ctrl = struct.unpack("<32I", ctrl_raw)

  result = {"core": core, "dropped": ctrl[_DROPPED], "done": ctrl[_DONE], "riscs": []}

  for risc in range(5):
    end_idx = ctrl[_DEVICE_BUF_END + risc]
    buf_addr = TensixL1.PROFILER_BUFFERS + risc * TensixL1.PROFILER_BUF_STRIDE
    buf_raw = bytes(win.uc[buf_addr : buf_addr + TensixL1.PROFILER_BUF_STRIDE])
    words = struct.unpack(f"<{len(buf_raw) // 4}I", buf_raw)

    # Guaranteed markers: FW scope (positions 4-7) and kernel child scope (positions 8-11)
    fw_start = _parse_ts(words[_GUARANTEED_1_H], words[_GUARANTEED_1_H + 1])
    fw_end = _parse_ts(words[_GUARANTEED_2_H], words[_GUARANTEED_2_H + 1])
    kern_start = _parse_ts(words[_GUARANTEED_3_H], words[_GUARANTEED_3_H + 1])
    kern_end = _parse_ts(words[_GUARANTEED_4_H], words[_GUARANTEED_4_H + 1])

    # Optional (custom) markers
    custom = []
    i = _CUSTOM_START
    while i + 1 < min(end_idx, len(words)):
      m = _parse_marker(words[i], words[i + 1])
      if m: custom.append(m)
      i += 2

    result["riscs"].append({
      "name": RISC_NAMES[risc],
      "fw_start": fw_start, "fw_end": fw_end,
      "kern_start": kern_start, "kern_end": kern_end,
      "custom": custom,
      "end_idx": end_idx,
    })
  return result

def read_and_report(device, cores: CoreList, freq_mhz: int = 1000, max_cores: int = 0):
  """Read profiler from all cores and print aggregate report."""
  from tlb import TLBConfig, TLBMode
  l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)

  profiles = []
  for core in cores:
    l1_cfg.start = l1_cfg.end = core
    device.win.configure(l1_cfg)
    profiles.append(read_core(device.win, core))

  _print_report(profiles, freq_mhz, max_cores)

def _cycles_us(cycles, freq_mhz):
  return cycles / freq_mhz if cycles else None

def _print_report(profiles, freq_mhz, max_cores):
  n = len(profiles)
  print(f"\n{'=' * 72}")
  print(f"  Device Profiler  ({n} cores @ {freq_mhz} MHz)")
  print(f"{'=' * 72}")

  # Aggregate per-RISC FW and kernel timing across all cores
  for risc_idx in range(5):
    fw_cycles, kern_cycles = [], []
    for p in profiles:
      r = p["riscs"][risc_idx]
      if r["fw_start"] and r["fw_end"]:
        fw_cycles.append(r["fw_end"] - r["fw_start"])
      if r["kern_start"] and r["kern_end"]:
        kern_cycles.append(r["kern_end"] - r["kern_start"])

    name = RISC_NAMES[risc_idx]
    if fw_cycles:
      lo, hi, avg = min(fw_cycles), max(fw_cycles), sum(fw_cycles) / len(fw_cycles)
      print(f"  {name:8s} FW      {_cycles_us(lo, freq_mhz):8.2f}us  ..  {_cycles_us(hi, freq_mhz):8.2f}us  (avg {_cycles_us(avg, freq_mhz):8.2f}us)")
    if kern_cycles:
      lo, hi, avg = min(kern_cycles), max(kern_cycles), sum(kern_cycles) / len(kern_cycles)
      print(f"  {name:8s} kernel  {_cycles_us(lo, freq_mhz):8.2f}us  ..  {_cycles_us(hi, freq_mhz):8.2f}us  (avg {_cycles_us(avg, freq_mhz):8.2f}us)")

  # Collect custom zone hashes across all cores/riscs
  zone_hashes = set()
  for p in profiles:
    for r in p["riscs"]:
      for zh, pt, ts in r["custom"]:
        zone_hashes.add(zh)

  if zone_hashes:
    print(f"\n  Custom zones ({len(zone_hashes)} unique):")
    for zh in sorted(zone_hashes):
      starts, ends = [], []
      for p in profiles:
        for r in p["riscs"]:
          for h, pt, ts in r["custom"]:
            if h == zh:
              if pt == 0: starts.append(ts)
              elif pt == 1: ends.append(ts)
      # Pair starts/ends in order for duration calculation
      if starts and ends and len(starts) == len(ends):
        durations = [e - s for s, e in zip(starts, ends) if e > s]
        if durations:
          total = sum(durations)
          avg = total / len(durations)
          print(f"    0x{zh:04x}  count={len(durations):4d}  total={_cycles_us(total, freq_mhz):10.2f}us  avg={_cycles_us(avg, freq_mhz):8.2f}us")

  # Per-core detail (limited)
  limit = max_cores if max_cores > 0 else min(5, n)
  if limit < n:
    print(f"\n  Per-core detail (first {limit} of {n}):")
  else:
    print(f"\n  Per-core detail:")
  for p in profiles[:limit]:
    x, y = p["core"]
    dropped = " DROPPED" if p["dropped"] else ""
    print(f"  core ({x:2d},{y:2d}):{dropped}")
    for r in p["riscs"]:
      parts = []
      if r["fw_start"] and r["fw_end"]:
        parts.append(f"fw={_cycles_us(r['fw_end'] - r['fw_start'], freq_mhz):.1f}us")
      if r["kern_start"] and r["kern_end"]:
        parts.append(f"kern={_cycles_us(r['kern_end'] - r['kern_start'], freq_mhz):.1f}us")
      if r["custom"]:
        parts.append(f"zones={len(r['custom']) // 2}")
      if parts:
        print(f"    {r['name']:8s} {', '.join(parts)}")

  print(f"{'=' * 72}\n")
