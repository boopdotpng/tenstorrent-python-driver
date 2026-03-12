import os, re, struct

from autogen import Profiler as P
from compiler import hash16

RISC_NAMES = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")

_ZONE_RE = re.compile(r'DeviceZoneScopedN\s*\(\s*"([^"]+)"')

def _parse_ts(w0, w1):
  if not (w0 & 0x80000000):
    return None
  return ((w0 & 0xFFF) << 32) | w1

def _parse_ctrl(raw_128_bytes):
  return struct.unpack("<32I", raw_128_bytes)

def _parse_run(words, start, end, program_ids):
  n = end - start
  if n < P.CUSTOM_START:
    return None
  prog_id = words[start + 3]
  if prog_id not in program_ids:
    return None

  # Validate guaranteed markers
  for off in (
    P.GUARANTEED_FW_START,
    P.GUARANTEED_FW_END,
    P.GUARANTEED_KERN_START,
    P.GUARANTEED_KERN_END,
  ):
    w0 = words[start + off]
    if not (w0 & 0x80000000):
      return None
    ptype = ((w0 >> 12) & 0x7FFFF) >> 16
    if ptype not in (P.ZONE_START, P.ZONE_END):
      return None

  fw_start = _parse_ts(
    words[start + P.GUARANTEED_FW_START], words[start + P.GUARANTEED_FW_START + 1]
  )
  fw_end = _parse_ts(
    words[start + P.GUARANTEED_FW_END], words[start + P.GUARANTEED_FW_END + 1]
  )
  kern_start = _parse_ts(
    words[start + P.GUARANTEED_KERN_START], words[start + P.GUARANTEED_KERN_START + 1]
  )
  kern_end = _parse_ts(
    words[start + P.GUARANTEED_KERN_END], words[start + P.GUARANTEED_KERN_END + 1]
  )

  fw = (
    (fw_end - fw_start)
    if (fw_start is not None and fw_end is not None and fw_end > fw_start)
    else 0
  )
  kern = (
    (kern_end - kern_start)
    if (kern_start is not None and kern_end is not None and kern_end > kern_start)
    else 0
  )

  # Parse custom zone markers
  custom = []  # list of (zone_hash, ptype, ts)
  i = start + P.CUSTOM_START
  while i + 1 < end:
    w0, w1 = words[i], words[i + 1]
    if w0 == 0 and w1 == 0:
      break
    if not (w0 & 0x80000000):
      i += 2
      continue
    timer_id = (w0 >> 12) & 0x7FFFF
    ptype = (timer_id >> 16) & 0x7
    if ptype in (P.ZONE_START, P.ZONE_END, P.ZONE_TOTAL):
      ts = w1 if ptype == P.ZONE_TOTAL else ((w0 & 0xFFF) << 32) | w1
      custom.append((timer_id & 0xFFFF, ptype, ts))
    if ptype == P.TS_DATA:
      i += 4
    elif ptype == P.TS_DATA_16B:
      i += 6
    else:
      i += 2

  return prog_id, fw, kern, kern_start, kern_end, custom

def _find_runs(words, program_ids):
  n = len(words)
  starts = []
  for i in range(0, n - 3, 2):
    if words[i] != 0 or words[i + 1] != 0:
      continue
    if i + P.CUSTOM_START > n:
      continue
    if words[i + 3] not in program_ids:
      continue
    # Check guaranteed markers are valid
    valid = True
    for off in (
      P.GUARANTEED_FW_START,
      P.GUARANTEED_FW_END,
      P.GUARANTEED_KERN_START,
      P.GUARANTEED_KERN_END,
    ):
      w0 = words[i + off]
      if not (w0 & 0x80000000):
        valid = False
        break
      ptype = ((w0 >> 12) & 0x7FFFF) >> 16
      if ptype not in (P.ZONE_START, P.ZONE_END):
        valid = False
        break
    if valid:
      starts.append(i)
  return starts

def _parse_risc(words, risc_id, program_ids):
  n = len(words)
  if n < P.CUSTOM_START:
    return {}
  starts = _find_runs(words, program_ids)
  if not starts:
    return {}

  out = {}
  for idx, start in enumerate(starts):
    end = starts[idx + 1] if idx + 1 < len(starts) else n
    parsed = _parse_run(words, start, end, program_ids)
    if parsed is None:
      continue
    prog_id, fw, kern, kern_start, kern_end, custom = parsed
    prev = out.get(prog_id)
    if prev is None or (prev[1] == 0 and kern > 0):
      out[prog_id] = (fw, kern, kern_start, kern_end, custom)
  return out

def _aggregate_zones(custom, zone_names):
  starts = {}  # hash -> [ts, ...]
  ends = {}  # hash -> [ts, ...]
  totals = {}  # hash -> accumulated cycles (for ZONE_TOTAL packets)
  for zone_hash, ptype, ts in custom:
    if ptype == P.ZONE_START:
      starts.setdefault(zone_hash, []).append(ts)
    elif ptype == P.ZONE_END:
      ends.setdefault(zone_hash, []).append(ts)
    elif ptype == P.ZONE_TOTAL:
      totals[zone_hash] = totals.get(zone_hash, 0) + ts

  zones = {}
  for zone_hash in set(starts) | set(totals):
    s_list = starts.get(zone_hash, [])
    e_list = ends.get(zone_hash, [])
    n = min(len(s_list), len(e_list))

    if n > 0:
      durations = [e - s for s, e in zip(s_list[:n], e_list[:n]) if e > s]
      if not durations:
        continue
      total = sum(durations)
      count = len(durations)
      mn, mx = min(durations), max(durations)
    elif zone_hash in totals:
      total = totals[zone_hash]
      count, mn, mx = 1, total, total
    else:
      continue

    name = zone_names.get(zone_hash, f"0x{zone_hash:04x}")
    zones[name] = {"total": total, "count": count, "min": mn, "max": mx}
  return zones

def _hash_msg(name, fpath, lineno):
  return hash16(f"{name},{fpath},{lineno},KERNEL_PROFILER")

def _resolve_zone_names(programs_info):
  from compiler import _zone_map

  names = {}  # int hash -> str name
  for info in programs_info:
    for label, src in info.get("sources", {}).items():
      if not src:
        continue
      for lineno, line in enumerate(src.splitlines(), start=1):
        m = _ZONE_RE.search(line)
        if not m:
          continue
        name = m.group(1)
        for fpath in ("./kernel_includes.hpp", "kernel_includes.hpp", label):
          names.setdefault(_hash_msg(name, fpath, lineno), name)
        names.setdefault(hash16(name), name)

  for h, (name, _, _) in _zone_map.items():
    names.setdefault(h, name)
  return names

def collect(
  programs_info,
  raw_dram,
  ctrl_regs,
  flat_ids,
  page_size,
  core_count_per_dram,
  harvested_dram_bank,
):
  program_ids = {info["index"] + 1 for info in programs_info}
  zone_names = _resolve_zone_names(programs_info)

  needed = set()
  for info in programs_info:
    needed.update(info["cores"])

  # Parse all RISC data from DRAM, keyed by (core, prog_id)
  # core_data[core][prog_id] = list of (risc_name, fw, kern, kern_start, kern_end, zones)
  core_data = {}
  debug = os.environ.get("TT_PROFILER_DEBUG") == "1"
  debug_remaining = 3

  for core in sorted(needed):
    flat_id = flat_ids[core]
    ctrl = _parse_ctrl(ctrl_regs[core])
    page_id = flat_id // core_count_per_dram
    slot = (flat_id % core_count_per_dram) * 5 * _HOST_BUF_BYTES_PER_RISC

    by_program = {}  # prog_id -> list of risc dicts
    for risc in range(5):
      host_end = min(ctrl[P.HOST_BUF_END + risc], _HOST_BUF_WORDS_PER_RISC)
      base = page_id * page_size + slot + risc * _HOST_BUF_BYTES_PER_RISC
      raw = raw_dram[base : base + host_end * 4]
      words = struct.unpack(f"<{len(raw) // 4}I", raw) if raw else ()

      if debug and debug_remaining > 0 and risc == 0:
        host_ends = [ctrl[P.HOST_BUF_END + i] for i in range(5)]
        dev_ends = [ctrl[P.DEVICE_BUF_END + i] for i in range(5)]
        print(
          f"[profdbg] core={core} done={ctrl[P.DONE]} dropped=0x{ctrl[P.DROPPED]:x} "
          f"host_ends={host_ends} dev_ends={dev_ends}"
        )
        debug_remaining -= 1

      for prog_id, (fw, kern, kern_start, kern_end, custom) in _parse_risc(
        words, risc, program_ids
      ).items():
        zones = _aggregate_zones(custom, zone_names)
        by_program.setdefault(prog_id, []).append(
          {
            "name": RISC_NAMES[risc],
            "fw": fw,
            "kern": kern,
            "kern_start": kern_start,
            "kern_end": kern_end,
            "zones": zones,
          }
        )
    core_data[core] = by_program

  # Build output programs
  programs = []
  for info in programs_info:
    prog_id = info["index"] + 1
    profiles = {}
    for core in info["cores"]:
      by_prog = core_data.get(core, {})
      riscs = by_prog.get(prog_id, [])
      # Fill missing RISCs
      present = {r["name"] for r in riscs}
      for name in RISC_NAMES:
        if name not in present:
          riscs.append(
            {
              "name": name,
              "fw": 0,
              "kern": 0,
              "kern_start": None,
              "kern_end": None,
              "zones": {},
            }
          )
      riscs.sort(key=lambda r: RISC_NAMES.index(r["name"]))
      # Compute total wall time across RISCs
      starts = [
        r["kern_start"]
        for r in riscs
        if r["kern_start"] is not None and r["kern_end"] is not None
      ]
      ends = [
        r["kern_end"]
        for r in riscs
        if r["kern_start"] is not None and r["kern_end"] is not None
      ]
      total = max(0, max(ends) - min(starts)) if starts and ends else 0
      # Strip internal fields from output
      out_riscs = [
        {
          "name": r["name"],
          "fw": r["fw"],
          "kern": r["kern"],
          "zones": r["zones"],
        }
        for r in riscs
      ]
      profiles[f"{core[0]},{core[1]}"] = {"total": total, "riscs": out_riscs}

    programs.append(
      {
        "index": info["index"],
        "name": info.get("name"),
        "cores": [list(c) for c in info["cores"]],
        "sources": info.get("sources", {}),
        "profiles": profiles,
      }
    )

  return {
    "dispatch_mode": "fast",
    "harvested_dram_bank": harvested_dram_bank,
    "programs": programs,
  }

def print_summary(data):
  programs = data.get("programs", [])
  print(f"Profiler collected {len(programs)} program(s)")
  for prog in programs:
    totals = [p["total"] for p in prog["profiles"].values() if p["total"] > 0]
    n = len(prog["profiles"])
    if not totals:
      print(
        f"  program {prog['index']} ({prog.get('name') or 'unnamed'}): {n} core(s), no cycles"
      )
      continue
    mn, mx = min(totals), max(totals)
    avg = sum(totals) / len(totals)
    print(
      f"  program {prog['index']} ({prog.get('name') or 'unnamed'}): {n} core(s), min/avg/max = {mn}/{avg:.1f}/{mx} cycles"
    )
