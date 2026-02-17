"""Serve dummy profiler data for local UI testing without hardware."""
import pathlib, sys, random

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import profiler_ui

# Zone hashes (arbitrary, just need to be consistent between custom markers and zone_names)
Z_TILE_LOOP = 1001
Z_NOC_READ  = 1002
Z_COMPUTE   = 1003
Z_NOC_WRITE = 1004

def make_risc(name, fw_start, fw_dur, kern_start, kern_dur, custom=None):
  return {
    "name": name, "end_idx": 20,
    "fw_start": fw_start, "fw_end": fw_start + fw_dur,
    "kern_start": kern_start, "kern_end": kern_start + kern_dur,
    "custom": custom or [],
  }

def make_zones(kern_start, kern_dur, jitter=0):
  """Generate zone markers within a kernel window. Each zone is a start/end pair."""
  j = lambda: random.randint(-jitter, jitter)
  # tile_loop wraps the whole kernel body
  tl_s = kern_start + 50 + j()
  tl_e = kern_start + kern_dur - 30 + j()
  return [
    {"hash": Z_TILE_LOOP, "type": 0, "ts": tl_s},
    {"hash": Z_TILE_LOOP, "type": 1, "ts": tl_e},
  ]

def make_read_zones(kern_start, kern_dur, jitter=0):
  j = lambda: random.randint(-jitter, jitter)
  s = kern_start + 80 + j()
  e = s + int(kern_dur * 0.6) + j()
  return [
    {"hash": Z_NOC_READ, "type": 0, "ts": s},
    {"hash": Z_NOC_READ, "type": 1, "ts": e},
  ]

def make_write_zones(kern_start, kern_dur, jitter=0):
  j = lambda: random.randint(-jitter, jitter)
  s = kern_start + 60 + j()
  e = s + int(kern_dur * 0.5) + j()
  return [
    {"hash": Z_NOC_WRITE, "type": 0, "ts": s},
    {"hash": Z_NOC_WRITE, "type": 1, "ts": e},
  ]

def make_compute_zones(kern_start, kern_dur, jitter=0):
  j = lambda: random.randint(-jitter, jitter)
  s = kern_start + 40 + j()
  e = s + int(kern_dur * 0.7) + j()
  return [
    {"hash": Z_COMPUTE, "type": 0, "ts": s},
    {"hash": Z_COMPUTE, "type": 1, "ts": e},
  ]

def make_core(x, y, base_cycles, jitter=200):
  """Generate a full core profile with per-RISC timings and zone markers."""
  random.seed(x * 100 + y)  # deterministic per-core
  j = lambda: random.randint(-jitter, jitter)
  fw_dur = 1600 + j()
  kern_dur = base_cycles + j()
  kern_start = fw_dur + 80
  return {
    "core": [x, y], "dropped": 0, "done": 1,
    "total_cycles": kern_dur,
    "riscs": [
      make_risc("BRISC",  80,  fw_dur,      kern_start, kern_dur, make_read_zones(kern_start, kern_dur, 50)),
      make_risc("NCRISC", 100, fw_dur - 200, kern_start, kern_dur + j()//2, make_write_zones(kern_start, kern_dur, 50)),
      make_risc("TRISC0", 100, 800 + j()//4, kern_start, kern_dur - 200 + j()//2, make_compute_zones(kern_start, kern_dur, 50)),
      make_risc("TRISC1", 95,  820 + j()//4, kern_start, kern_dur - 150 + j()//2, make_zones(kern_start, kern_dur, 50)),
      make_risc("TRISC2", 95,  810 + j()//4, kern_start, kern_dur - 180 + j()//2),
    ],
  }

def main():
  # Program 1: matmul — uses 8 cores in a 4x2 block with zones
  matmul_cores = [(x, y) for y in [2, 3] for x in [1, 2, 3, 4]]
  matmul_profiles = {}
  for x, y in matmul_cores:
    p = make_core(x, y, 4200, jitter=400)
    matmul_profiles[f"{x},{y}"] = p

  # Program 2: add1 — uses 4 cores, simpler
  add1_cores = [(x, 5) for x in [1, 2, 3, 4]]
  add1_profiles = {}
  for x, y in add1_cores:
    p = make_core(x, y, 2800, jitter=300)
    # fewer zones on this one — just tile_loop on TRISC0
    for r in p["riscs"]:
      if r["name"] not in ("TRISC0",):
        r["custom"] = []
    add1_profiles[f"{x},{y}"] = p

  # One dropped core for testing
  matmul_profiles["4,3"]["dropped"] = 1

  data = {
    "dispatch_mode": "fast",
    "dispatch_cores": [[14, 2], [14, 3]],
    "freq_mhz": 1000,
    "grid_x": [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14],
    "grid_y": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "zone_names": {
      str(Z_TILE_LOOP): {"name": "tile_loop", "file": "compute.cpp", "line": 8},
      str(Z_NOC_READ):  {"name": "noc_read",  "file": "reader.cpp",  "line": 10},
      str(Z_COMPUTE):   {"name": "sfpu_math",  "file": "compute.cpp", "line": 9},
      str(Z_NOC_WRITE): {"name": "noc_write", "file": "writer.cpp",  "line": 10},
    },
    "programs": [
      {
        "index": 0,
        "name": "matmul",
        "cores": matmul_cores,
        "profiles": matmul_profiles,
        "sources": {
          "reader": (
            '#include <cstdint>\n\n'
            'void kernel_main() {\n'
            '  uint32_t in0_addr = get_arg_val<uint32_t>(0);\n'
            '  uint32_t tile_offset = get_arg_val<uint32_t>(1);\n'
            '  uint32_t n_tiles = get_arg_val<uint32_t>(2);\n'
            '  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;\n'
            '  for (uint32_t i = 0; i < n_tiles; ++i) {\n'
            '    cb_reserve_back(cb_in0, 1);\n'
            '    DeviceZoneScopedN("noc_read");\n'
            '    noc_async_read_tile(tile_offset + i, in0, get_write_ptr(cb_in0));\n'
            '    noc_async_read_barrier();\n'
            '    cb_push_back(cb_in0, 1);\n'
            '  }\n'
            '}\n'
          ),
          "compute": (
            '#include <cstdint>\n'
            '#include "compute_kernel_api/common.h"\n\n'
            'void MAIN {\n'
            '  uint32_t n_tiles = get_arg_val<uint32_t>(0);\n'
            '  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);\n'
            '  for (uint32_t i = 0; i < n_tiles; ++i) {\n'
            '    DeviceZoneScopedN("tile_loop");\n'
            '    cb_wait_front(tt::CBIndex::c_0, 1);\n'
            '    DeviceZoneScopedN("sfpu_math");\n'
            '    copy_tile(tt::CBIndex::c_0, 0, 0);\n'
            '    pack_tile(0, tt::CBIndex::c_16);\n'
            '    cb_pop_front(tt::CBIndex::c_0, 1);\n'
            '    cb_push_back(tt::CBIndex::c_16, 1);\n'
            '  }\n'
            '}\n'
          ),
          "writer": (
            '#include <cstdint>\n\n'
            'void kernel_main() {\n'
            '  uint32_t out_addr = get_arg_val<uint32_t>(0);\n'
            '  uint32_t tile_offset = get_arg_val<uint32_t>(1);\n'
            '  uint32_t n_tiles = get_arg_val<uint32_t>(2);\n'
            '  constexpr uint32_t cb_out0 = tt::CBIndex::c_16;\n'
            '  for (uint32_t i = 0; i < n_tiles; ++i) {\n'
            '    cb_wait_front(cb_out0, 1);\n'
            '    DeviceZoneScopedN("noc_write");\n'
            '    noc_async_write_tile(tile_offset + i, out0, get_read_ptr(cb_out0));\n'
            '    noc_async_write_barrier();\n'
            '    cb_pop_front(cb_out0, 1);\n'
            '  }\n'
            '}\n'
          ),
        },
      },
      {
        "index": 1,
        "name": "add1",
        "cores": add1_cores,
        "profiles": add1_profiles,
        "sources": {
          "reader": (
            '#include <cstdint>\n\n'
            'void kernel_main() {\n'
            '  uint32_t in0_addr = get_arg_val<uint32_t>(0);\n'
            '  uint32_t tile_offset = get_arg_val<uint32_t>(1);\n'
            '  uint32_t n_tiles = get_arg_val<uint32_t>(2);\n'
            '  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;\n'
            '  for (uint32_t i = 0; i < n_tiles; ++i) {\n'
            '    cb_reserve_back(cb_in0, 1);\n'
            '    noc_async_read_tile(tile_offset + i, in0, get_write_ptr(cb_in0));\n'
            '    noc_async_read_barrier();\n'
            '    cb_push_back(cb_in0, 1);\n'
            '  }\n'
            '}\n'
          ),
          "compute": (
            '#include <cstdint>\n'
            '#include "compute_kernel_api/common.h"\n\n'
            'void MAIN {\n'
            '  uint32_t n_tiles = get_arg_val<uint32_t>(0);\n'
            '  init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);\n'
            '  for (uint32_t i = 0; i < n_tiles; ++i) {\n'
            '    DeviceZoneScopedN("sfpu_math");\n'
            '    cb_wait_front(tt::CBIndex::c_0, 1);\n'
            '    // add 1.0 to each element\n'
            '    sfpu_add_imm(tt::CBIndex::c_0, 0, 0x3F800000);\n'
            '    pack_tile(0, tt::CBIndex::c_16);\n'
            '    cb_pop_front(tt::CBIndex::c_0, 1);\n'
            '    cb_push_back(tt::CBIndex::c_16, 1);\n'
            '  }\n'
            '}\n'
          ),
          "writer": (
            '#include <cstdint>\n\n'
            'void kernel_main() {\n'
            '  uint32_t out_addr = get_arg_val<uint32_t>(0);\n'
            '  uint32_t tile_offset = get_arg_val<uint32_t>(1);\n'
            '  uint32_t n_tiles = get_arg_val<uint32_t>(2);\n'
            '  constexpr uint32_t cb_out0 = tt::CBIndex::c_16;\n'
            '  for (uint32_t i = 0; i < n_tiles; ++i) {\n'
            '    cb_wait_front(cb_out0, 1);\n'
            '    noc_async_write_tile(tile_offset + i, out0, get_read_ptr(cb_out0));\n'
            '    noc_async_write_barrier();\n'
            '    cb_pop_front(cb_out0, 1);\n'
            '  }\n'
            '}\n'
          ),
        },
      },
    ],
  }
  profiler_ui.serve(data, port=8884)


if __name__ == "__main__":
  main()
