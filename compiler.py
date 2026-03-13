import hashlib, os, pickle, re, shutil, struct, subprocess, tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hw import *
from dispatch import Dtype, MathFidelity, Program

PROFILER = os.environ.get("PROFILE") == "1"

_REPO = Path(__file__).resolve().parent
_DEPS = _REPO / "tt-metal-deps"
_SFPI = _DEPS / "sfpi-toolchain" / "bin"
_CACHE_DIR = Path.home() / "cache" / "tt-cache"

# zone hash→name mapping captured from DeviceZoneScopedN #pragma messages
_zone_map: dict[int, tuple[str, str, int]] = {}

def _cache_hash(*parts) -> str:
  h = hashlib.sha256()
  for p in parts:
    if isinstance(p, bytes):
      h.update(p)
    else:
      h.update(repr(p).encode())
    h.update(b"\x00")
  return h.hexdigest()

def _cache_load(key: str):
  try:
    with open(_CACHE_DIR / f"{key}.pkl", "rb") as f:
      return pickle.load(f)
  except (FileNotFoundError, pickle.UnpicklingError, EOFError, ValueError):
    return None

def _cache_store(key: str, data):
  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  tmp = _CACHE_DIR / f"{key}.tmp"
  with open(tmp, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
  tmp.rename(_CACHE_DIR / f"{key}.pkl")

def hash16(s: str) -> int:
  h = 0x811c9dc5
  for c in s.encode():
    h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
  return (h >> 16) ^ (h & 0xFFFF)

_INCLUDE_PATHS = [
  "tt_metal/hw/inc", "tt_metal/hostdevcommon/api", "tt_metal/api",
  "tt_metal/include", "tt_metal/hw/inc/internal/tt-1xx",
  "tt_metal/hw/inc/internal/tt-1xx/blackhole",
  "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc",
  "tt_metal/hw/ckernels/blackhole/metal/llk_io",
  "tt_metal/hw/ckernels/blackhole/metal/common",
  "tt_metal/hw/ckernels/blackhole/metal/llk_api",
  "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu",
  "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
  "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
  "runtime/sfpi/include",
]
_INC = _DEPS / "include"
_INCLUDES = [f"-I{_INC}", *(f"-I{_INC / p}" for p in _INCLUDE_PATHS)]

_CFLAGS = (
  "-std=c++17", "-flto=auto", "-ffast-math", "-fno-exceptions",
  "-fno-use-cxa-atexit",
)
_LFLAGS = ("-Wl,-z,max-page-size=16", "-Wl,-z,common-page-size=16", "-nostartfiles")

_DEVICE_DEFINES = [
  "-DNUM_DRAM_BANKS=7", "-DIS_NOT_POW2_NUM_DRAM_BANKS=1",
  "-DNUM_L1_BANKS=110", "-DIS_NOT_POW2_NUM_L1_BANKS=1",
  "-DPCIE_NOC_X=19", "-DPCIE_NOC_Y=24",
]
_KERNEL_DEFINES = [
  "-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE",
  "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", "-DKERNEL_BUILD", *_DEVICE_DEFINES,
]

_CQ_SRC_DIR = _REPO / "firmware" / "cq"
_CQ_INC = [str(_CQ_SRC_DIR), str(_CQ_SRC_DIR / "includes")]

_PROFILE_DEFINES = [
  "-DPROFILE_KERNEL=1",
  f"-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC={TensixL1.PROFILER_HOST_BUFFER_BYTES_PER_RISC}",
]

_FW_TARGETS = [
  ("brisc.cc", "brisc", ["-DCOMPILE_FOR_BRISC", "-DPROCESSOR_INDEX=0", "-DNOC_INDEX=1", "-DNOC_MODE=0"],
   ["-mcpu=tt-bh", "-fno-tree-loop-distribute-patterns"], "-Os", ["noc.o"]),
  ("ncrisc.cc", "ncrisc", ["-DCOMPILE_FOR_NCRISC", "-DPROCESSOR_INDEX=1", "-DNOC_INDEX=0", "-DNOC_MODE=0"],
   ["-mcpu=tt-bh", "-fno-tree-loop-distribute-patterns"], "-Os", []),
  ("trisc.cc", "trisc0", ["-DCOMPILE_FOR_TRISC=0", "-DPROCESSOR_INDEX=2", "-DUCK_CHLKC_UNPACK", "-DNAMESPACE=chlkc_unpack"],
   ["-mcpu=tt-bh-tensix"], "-O3", []),
  ("trisc.cc", "trisc1", ["-DCOMPILE_FOR_TRISC=1", "-DPROCESSOR_INDEX=3", "-DUCK_CHLKC_MATH", "-DNAMESPACE=chlkc_math"],
   ["-mcpu=tt-bh-tensix"], "-O3", []),
  ("trisc.cc", "trisc2", ["-DCOMPILE_FOR_TRISC=2", "-DPROCESSOR_INDEX=4", "-DUCK_CHLKC_PACK", "-DNAMESPACE=chlkc_pack"],
   ["-mcpu=tt-bh-tensix"], "-O3", []),
]

_DEFAULT_PROGRAM = Program(cores=1, reader_kernel="", writer_kernel="", compute_kernel="", cbs=[])

def _ckernel_headers(program: Program) -> dict[str, str]:
  formats = [Dtype.Float16_b.value] * 32
  for cb in program.cbs:
    formats[cb.index] = cb.dtype.value
  tile_sizes = [Dtype(f).tile_size for f in formats]
  a = lambda vs: ", ".join(str(v) for v in vs)
  a32 = lambda v: a([v] * 32)
  b = lambda v: "true" if v else "false"

  def data_fmt(prefix, ctype):
    return (f"#pragma once\n#include <cstdint>\n"
            f"constexpr {ctype} {prefix}_src_format[32] = {{{a(formats)}}};\n"
            f"constexpr {ctype} {prefix}_dst_format[32] = {{{a(formats)}}};\n")

  def tile_dims(prefix):
    return (f"#pragma once\n#include <cstdint>\n"
            f"constexpr uint8_t {prefix}_tile_num_faces[32] = {{{a32(4)}}};\n"
            f"constexpr uint8_t {prefix}_partial_face[32] = {{{a32(0)}}};\n"
            f"constexpr uint8_t {prefix}_tile_face_r_dim[32] = {{{a32(16)}}};\n"
            f"constexpr uint8_t {prefix}_narrow_tile[32] = {{{a32(0)}}};\n"
            f"constexpr uint8_t {prefix}_tile_r_dim[32] = {{{a32(32)}}};\n"
            f"constexpr uint8_t {prefix}_tile_c_dim[32] = {{{a32(32)}}};\n"
            f"constexpr uint16_t {prefix}_tile_size[32] = {{{a(tile_sizes)}}};\n")

  dst_sync = "DstSync::SyncFull" if program.dst_full_sync else "DstSync::SyncHalf"
  return {
    "chlkc_unpack_data_format.h": data_fmt("unpack", "std::int32_t"),
    "chlkc_pack_data_format.h": data_fmt("pack", "unsigned char"),
    "chlkc_unpack_tile_dims.h": tile_dims("unpack"),
    "chlkc_pack_tile_dims.h": tile_dims("pack"),
    "chlkc_dst_accum_mode.h": f"#pragma once\nconstexpr bool DST_ACCUM_MODE = {b(program.dst_accum_mode)};\n",
    "chlkc_dst_sync_mode.h": f"#pragma once\n#define DST_SYNC_MODE {dst_sync}\n",
    "chlkc_math_fidelity.h": f"#pragma once\n#include <cstdint>\nconstexpr std::int32_t MATH_FIDELITY = {program.math_fidelity.value};\n",
    "chlkc_math_approx_mode.h": f"#pragma once\nconstexpr bool APPROX = {b(program.approx)};\n",
  }

@dataclass(frozen=True)
class PTLoad:
  paddr: int
  data: bytes
  memsz: int
  flags: int = 0

def iter_pt_load(elf: bytes):
  e_phoff = struct.unpack_from("<I", elf, 28)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 42)
  if e_phentsize < 32: raise ValueError(f"bad e_phentsize: {e_phentsize}")
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, _ = struct.unpack_from("<IIIIIIII", elf, off)
    if p_type != 1: continue
    paddr = p_paddr or p_vaddr
    yield PTLoad(paddr=paddr, data=elf[p_offset:p_offset + p_filesz], memsz=p_memsz, flags=p_flags)

def _xipify_riscv32_elf(elf: bytes) -> bytes:
  data = bytearray(elf)
  e_phoff = struct.unpack_from("<I", data, 28)[0]
  e_shoff = struct.unpack_from("<I", data, 32)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", data, 42)
  e_shentsize, e_shnum = struct.unpack_from("<HH", data, 46)
  if e_phentsize < 32 or e_shentsize < 40: return elf

  R_RISCV_HI20, R_RISCV_LO12_I, R_RISCV_LO12_S = 26, 27, 28

  def shdr(i):
    return struct.unpack_from("<IIIIIIIIII", data, e_shoff + i * e_shentsize)

  def sym(symtab_idx, sym_idx):
    _, _, _, _, sh_offset, sh_size, _, _, _, sh_entsize = shdr(symtab_idx)
    soff = sh_offset + sym_idx * sh_entsize
    return struct.unpack_from("<IIIBBH", data, soff)

  # find text segment
  text_vaddr = text_memsz = None
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, _, p_vaddr, _, _, p_memsz, p_flags, _ = struct.unpack_from("<IIIIIIII", data, off)
    if p_type == 1 and (p_flags & 1):
      text_vaddr, text_memsz = p_vaddr, p_memsz
      break
  if text_vaddr is None: return elf
  text_end = text_vaddr + text_memsz
  is_text = lambda addr: text_vaddr <= addr <= text_end

  def rw32(sec_idx, addr):
    _, _, _, sh_addr, sh_offset, _, _, _, _, _ = shdr(sec_idx)
    return sh_offset + (addr - sh_addr)

  for rel_sec_idx in range(e_shnum):
    _, sh_type, _, _, sh_offset, sh_size, sh_link, sh_info, _, sh_entsize = shdr(rel_sec_idx)
    if sh_type != 4 or sh_entsize < 12 or sh_size == 0: continue
    _, tgt_type, tgt_flags, _, _, _, _, _, _, _ = shdr(sh_info)
    if not (tgt_flags & 0x2) or tgt_type == 8: continue

    hi_by_sym: dict[int, list[tuple[int, int]]] = {}
    lo_relocs: list[tuple[int, int, int]] = []
    for j in range(sh_size // sh_entsize):
      roff = sh_offset + j * sh_entsize
      r_offset, r_info, r_addend = struct.unpack_from("<IIi", data, roff)
      r_type, r_sym = r_info & 0xFF, r_info >> 8
      if r_type == R_RISCV_HI20:
        _, st_value, _, _, _, _ = sym(sh_link, r_sym)
        if is_text(st_value):
          hi_by_sym.setdefault(r_sym, []).append((r_offset, r_addend))
      elif r_type in (R_RISCV_LO12_I, R_RISCV_LO12_S):
        lo_relocs.append((r_offset, r_sym, r_type))

    for items in hi_by_sym.values():
      items.sort(key=lambda x: x[0])

    for lo_offset, lo_sym, lo_type in lo_relocs:
      hi_list = hi_by_sym.get(lo_sym)
      if not hi_list: continue
      hi_offset, hi_addend = hi_list[0]
      for cand_off, cand_add in hi_list:
        if cand_off < lo_offset: hi_offset, hi_addend = cand_off, cand_add
        else: break

      _, st_value, _, _, _, _ = sym(sh_link, lo_sym)
      value = (st_value + hi_addend - hi_offset) & 0xFFFFFFFF
      foff = rw32(sh_info, hi_offset)
      hi_insn = struct.unpack_from("<I", data, foff)[0]
      if (hi_insn & 0x7F) != 0x37: continue
      rd = (hi_insn >> 7) & 0x1F
      new_hi = (((value + 0x800) >> 12) & 0xFFFFF) << 12 | (rd << 7) | 0x17
      struct.pack_into("<I", data, foff, new_hi)

      lo_foff = rw32(sh_info, lo_offset)
      lo_insn = struct.unpack_from("<I", data, lo_foff)[0]
      lo12 = value & 0xFFF
      if lo_type == R_RISCV_LO12_I:
        new_lo = (lo_insn & 0x000FFFFF) | (lo12 << 20)
      else:
        new_lo = (lo_insn & ~((0x7F << 25) | (0x1F << 7))) | ((lo12 >> 5) << 25) | ((lo12 & 0x1F) << 7)
      struct.pack_into("<I", data, lo_foff, new_lo & 0xFFFFFFFF)

  return bytes(data)

def pack_xip_elf(elf: bytes, xip_relocate: bool = False) -> tuple[bytes, int]:
  if xip_relocate:
    elf = _xipify_riscv32_elf(elf)
  segs = list(iter_pt_load(elf))
  if not segs: raise ValueError("no PT_LOAD segments")
  l1 = sorted([s for s in segs if (s.memsz or s.data) and (0 <= s.paddr < TensixL1.SIZE)], key=lambda s: s.paddr)
  if not l1: raise ValueError("no L1 PT_LOAD segments")
  base = l1[0].paddr
  out = bytearray()
  for s in l1:
    start = s.paddr - base
    end = start + max(s.memsz, len(s.data))
    if len(out) < end: out.extend(b"\0" * (end - len(out)))
    out[start:start + len(s.data)] = s.data
  text = next((s for s in l1 if (s.flags & 1) and s.data), l1[0])
  return bytes(out), len(text.data)

@dataclass(frozen=True)
class CompiledKernel:
  xip: bytes
  xip_text_bytes: int

_INIT_SCRATCH_BASE = TensixL1.KERNEL_CONFIG_BASE
_INIT_SCRATCH = {
  "brisc":  _INIT_SCRATCH_BASE,
  "ncrisc": _INIT_SCRATCH_BASE + 0x2000,
  "trisc0": _INIT_SCRATCH_BASE + 0x4000,
  "trisc1": _INIT_SCRATCH_BASE + 0x5000,
  "trisc2": _INIT_SCRATCH_BASE + 0x6000,
}

@dataclass(frozen=True)
class CompiledFirmware:
  elf_bytes: bytes
  segments: list[PTLoad]
  scratch_base: int

  @property
  def text_base(self) -> int: return self.segments[0].paddr

def _run(exe: Path, args: list[str], cwd: Path):
  r = subprocess.run([str(exe), *args], cwd=cwd, capture_output=True)
  if r.returncode != 0:
    raise RuntimeError(f"{exe.name} failed:\n{r.stderr.decode()}")
  if PROFILER:
    pragma_re = re.compile(r"#pragma message:\s*['\"]?(.+?)['\"]?\s*$")
    for line in r.stderr.decode(errors="replace").splitlines():
      if "KERNEL_PROFILER" not in line or "#pragma message:" not in line: continue
      m = pragma_re.search(line)
      if not m: continue
      msg = m.group(1).strip()
      if not msg.endswith("KERNEL_PROFILER"): continue
      parts = msg.rsplit(",", 3)
      if len(parts) != 4: continue
      name, fpath, lineno_s, tag = (p.strip() for p in parts)
      if tag != "KERNEL_PROFILER": continue
      try: _zone_map[hash16(msg)] = (name, fpath, int(lineno_s))
      except ValueError: pass

def _compile_and_link(cc: Path, src: Path, compile_args: list[str], link_args: list[str] | Callable[[Path], list[str]],
                      tmp_prefix: str, prepare: Callable[[Path], None] | None = None) -> bytes:
  build = Path(tempfile.mkdtemp(prefix=tmp_prefix))
  try:
    if prepare is not None: prepare(build)
    _run(cc, [*compile_args, "-c", "-o", "out.o", str(src)], build)
    args = link_args(build) if callable(link_args) else link_args
    _run(cc, [*args, "-o", "out.elf"], build)
    return (build / "out.elf").read_bytes()
  finally:
    shutil.rmtree(build, ignore_errors=True)

def compile_firmware(profile: bool = PROFILER) -> dict[str, CompiledFirmware]:
  cc = _SFPI / "riscv-tt-elf-g++"
  assert cc.is_file(), f"missing compiler: {cc}"

  fw_src_dir = _REPO / "firmware"
  unique_srcs = sorted(set(s for s, *_ in _FW_TARGETS))
  key = _cache_hash("fw", profile, tuple((n, (fw_src_dir / n).read_bytes()) for n in unique_srcs))
  cached = _cache_load(key)
  if cached is not None:
    _zone_map.update(cached["zones"])
    return cached["result"]

  zones_before = dict(_zone_map)
  common_defines = [
    "-DTENSIX_FIRMWARE", "-DFW_BUILD", "-DARCH_BLACKHOLE",
    "-DLOCAL_MEM_EN=0", "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", *_DEVICE_DEFINES,
  ]
  if profile: common_defines += _PROFILE_DEFINES
  lib = _DEPS / "lib" / "blackhole"
  ld_dir = _DEPS / "toolchain" / "blackhole"

  result: dict[str, CompiledFirmware] = {}
  for src_name, target, target_defs, mcpu, opt, extra in _FW_TARGETS:
    ld = ld_dir / f"firmware_{target}.ld"
    src = fw_src_dir / src_name
    compile_args = [opt, *_CFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay", *common_defines, *target_defs, *_INCLUDES]
    link_objs = [str(lib / "tmu-crt0.o"), "out.o", *(str(lib / o) for o in extra), str(lib / "substitutes.o")]
    fw_link_args = [opt, *_CFLAGS, *_LFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay", f"-T{ld}", *link_objs]
    elf = _compile_and_link(cc=cc, src=src, compile_args=compile_args, link_args=fw_link_args, tmp_prefix=f"tt-fw-{target}-")
    segs = list(iter_pt_load(elf))
    result[target] = CompiledFirmware(elf_bytes=elf, segments=segs, scratch_base=_INIT_SCRATCH[target])

  new_zones = {k: v for k, v in _zone_map.items() if k not in zones_before}
  _cache_store(key, {"result": result, "zones": new_zones})
  return result

class Compiler:
  def __init__(self):
    self._cc = _SFPI / "riscv-tt-elf-g++"
    self._objcopy = _SFPI / "riscv-tt-elf-objcopy"
    self._includes = ["-I.", *_INCLUDES]
    assert self._cc.is_file(), f"missing compiler: {self._cc}\nDownload toolchain to {_DEPS / 'sfpi-toolchain'}"
    self._fw = compile_firmware(profile=PROFILER)

  def compile_dataflow(self, src: str, processor: str, noc_index: int | None = None) -> CompiledKernel:
    if processor not in ("brisc", "ncrisc"):
      raise ValueError(f"processor must be 'brisc' or 'ncrisc', got: {processor}")
    if noc_index is None:
      noc_index = 1 if processor == "brisc" else 0
    return self._compile_dataflow(src, processor, noc_index=noc_index)

  def compile_compute(self, src: str, program: Program) -> tuple[CompiledKernel, CompiledKernel, CompiledKernel]:
    return (self._compile_trisc(src, 0, program), self._compile_trisc(src, 1, program), self._compile_trisc(src, 2, program))

  def _compile_dataflow(self, src: str, target: str, noc_index: int, extra_defines: list[str] | None = None,
                        extra_includes: list[str] | None = None, xip_relocate: bool = False,
                        profiler: bool = True) -> CompiledKernel:
    defines = [
      *_KERNEL_DEFINES,
      f"-DCOMPILE_FOR_{target.upper()}", f"-DPROCESSOR_INDEX={0 if target == 'brisc' else 1}",
      f"-DNOC_INDEX={noc_index}", "-DNOC_MODE=0", *(extra_defines or []),
    ]
    if PROFILER and profiler: defines += _PROFILE_DEFINES
    extra_objs = [str(_DEPS / "lib/blackhole/noc.o")] if target == "brisc" else []
    return self._build(src, target, defines, extra_objs, opt="-O2", trisc=False,
                       extra_includes=extra_includes, xip_relocate=xip_relocate)

  def _compile_trisc(self, src: str, trisc_id: int, program: Program) -> CompiledKernel:
    stage = ("unpack", "math", "pack")[trisc_id]
    defines = [
      *_KERNEL_DEFINES, f"-DCOMPILE_FOR_TRISC={trisc_id}", f"-DPROCESSOR_INDEX={trisc_id + 2}",
      f"-DUCK_CHLKC_{stage.upper()}", f"-DNAMESPACE=chlkc_{stage}",
    ]
    if PROFILER: defines += _PROFILE_DEFINES
    return self._build(src, f"trisc{trisc_id}", defines, [], opt="-O3", trisc=True, program=program)

  def _build(self, kern: str, target: str, defines: list[str], extra_objs: list[str], opt: str, trisc: bool,
             extra_includes: list[str] | None = None, xip_relocate: bool = False,
             program: Program = _DEFAULT_PROGRAM) -> CompiledKernel:
    hdrs = _ckernel_headers(program)
    inc_content = b""
    if extra_includes:
      for d in sorted(extra_includes):
        for f in sorted(Path(d).rglob("*")):
          if f.is_file(): inc_content += f.read_bytes()
    key = _cache_hash("kern", kern, target, tuple(defines), opt, trisc,
                      xip_relocate, tuple(sorted(hdrs.items())), self._fw[target].elf_bytes, inc_content)
    cached = _cache_load(key)
    if cached is not None:
      _zone_map.update(cached["zones"])
      return cached["result"]

    zones_before = dict(_zone_map)
    mcpu = ["-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"] if trisc else \
           ["-mcpu=tt-bh", "-mno-tt-tensix-optimize-replay", "-fno-tree-loop-distribute-patterns"]
    fw_src = _DEPS / "firmware-src" / ("trisck.cc" if trisc else f"{target}k.cc")
    includes = [*self._includes, *(f"-I{p}" for p in (extra_includes or []))]
    compile_args = [opt, *_CFLAGS, "-MMD", *mcpu, *includes, *defines]

    fw_link_elf: Path | None = None
    def _prepare(build: Path):
      nonlocal fw_link_elf
      fw_link_elf = self._weaken_fw_symbols(build, self._fw[target].elf_bytes)
      (build / "kernel_includes.hpp").write_text(kern)
      for name, content in hdrs.items():
        (build / name).write_text(content)
      if trisc:
        (build / "defines_generated.h").write_text("")
        for stage, macro in [("unpack", "TRISC_UNPACK"), ("math", "TRISC_MATH"), ("pack", "TRISC_PACK")]:
          (build / f"chlkc_{stage}.cpp").write_text(f'#define {macro}\n#include "defines_generated.h"\n#include <kernel_includes.hpp>\n')

    def _kernel_link_args(_: Path) -> list[str]:
      ld = _DEPS / "toolchain" / "blackhole" / f"kernel_{target}.ld"
      objs = ["out.o", *extra_objs, str(_DEPS / "lib/blackhole/substitutes.o")]
      return [opt, *_CFLAGS, *_LFLAGS, *mcpu, f"-T{ld}", "-Wl,--emit-relocs", f"-Wl,--just-symbols={fw_link_elf}", *objs]

    elf = _compile_and_link(cc=self._cc, src=fw_src, compile_args=compile_args,
                            link_args=_kernel_link_args, tmp_prefix=f"tt-{target}-", prepare=_prepare)
    result = CompiledKernel(*pack_xip_elf(elf, xip_relocate=xip_relocate))
    new_zones = {k: v for k, v in _zone_map.items() if k not in zones_before}
    _cache_store(key, {"result": result, "zones": new_zones})
    return result

  def _weaken_fw_symbols(self, build: Path, fw: bytes) -> Path:
    out = build / "fw.elf"
    out.write_bytes(fw)
    _run(self._objcopy, ["--localize-symbol=_start", "--localize-symbol=main",
                         "--localize-symbol=exit", "--weaken", str(out)], build)
    return out

  def compile_cq_kernels(self) -> dict[str, CompiledKernel]:
    cq = lambda src, proc, noc: self._compile_dataflow(
      (_CQ_SRC_DIR / src).read_text(), proc, noc_index=noc, extra_includes=_CQ_INC, xip_relocate=True, profiler=False)
    return {
      "prefetch_brisc": cq("cq_prefetch.cpp", "brisc", 0),
      "dispatch_brisc": cq("cq_dispatch.cpp", "brisc", 1),
      "dispatch_s_ncrisc": cq("cq_dispatch_subordinate.cpp", "ncrisc", 1),
    }
