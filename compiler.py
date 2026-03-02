from helpers import pack_xip_elf, iter_pt_load, PTLoad, PROFILER, hash16
from defs import TensixL1
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle, re, shutil, subprocess, tempfile

_DEPS = Path(__file__).resolve().parent / "tt-metal-deps"
_REPO = Path(__file__).resolve().parent
_SFPI = _DEPS / "sfpi-toolchain" / "bin"
_CACHE_DIR = Path.home() / ".cache" / "tt-cache"

Strs = list[str]

# Zone hash→name mapping captured from DeviceZoneScopedN #pragma messages
_zone_map: dict[int, tuple[str, str, int]] = {}  # hash → (name, file, line)

def get_zone_map() -> dict[int, tuple[str, str, int]]:
  return dict(_zone_map)
LinkArgs = Strs | Callable[[Path], Strs]
PrepareFn = Callable[[Path], None] | None

class DataFormat(Enum):
  Float32 = 0
  Float16 = 1
  Float16_b = 5
  Int32 = 8
  UInt16 = 9
  Int8 = 14
  UInt32 = 24
  Fp8_e4m3 = 0x1A
  UInt8 = 30

class MathFidelity(Enum):
  LoFi = 0; HiFi2 = 2; HiFi3 = 3; HiFi4 = 4

@dataclass(frozen=True)
class CkernelConfig:
  input_format: DataFormat = DataFormat.Float16_b
  output_format: DataFormat = DataFormat.Float16_b
  cb_data_formats: tuple[tuple[int, DataFormat], ...] = ()
  math_fidelity: MathFidelity = MathFidelity.HiFi4
  approx: bool = False
  dst_accum_mode: bool = False
  # full sync between math/pack stages (vs double-buffered SyncHalf)
  dst_full_sync: bool = False

@dataclass(frozen=True)
class CompiledKernel:
  xip: bytes; xip_text_bytes: int

@dataclass(frozen=True)
class CompiledFirmware:
  elf_bytes: bytes           # full linked ELF bytes
  segments: list[PTLoad]     # PT_LOAD segments for upload
  text_base: int             # where .text starts in L1

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
_CQ_INC = [str(_CQ_SRC_DIR)]
_CQ_PREFETCH_SRC = (_CQ_SRC_DIR / "cq_prefetch.cpp").read_text()
_CQ_DISPATCH_SRC = (_CQ_SRC_DIR / "cq_dispatch.cpp").read_text()
_CQ_DISPATCH_S_SRC = (_CQ_SRC_DIR / "cq_dispatch_subordinate.cpp").read_text()

def _render_ckernel_headers(cfg: CkernelConfig) -> dict[str, str]:
  in_fmt, out_fmt = cfg.input_format.value, cfg.output_format.value
  formats = [in_fmt] * 16 + [out_fmt] * 16
  for cb_id, fmt in cfg.cb_data_formats:
    if not (0 <= cb_id < 32):
      raise ValueError(f"cb_data_formats has invalid cb_id {cb_id}; expected 0..31")
    formats[cb_id] = fmt.value

  tile_sizes = [_tile_size(fmt) for fmt in formats]
  def arr(vals): return ", ".join(str(v) for v in vals)
  def arr32(v): return arr([v] * 32)
  dst_sync_mode = "DstSync::SyncFull" if cfg.dst_full_sync else "DstSync::SyncHalf"
  def bool_str(v): return "true" if v else "false"

  return {
    "chlkc_unpack_data_format.h":
      "#pragma once\n"
      "#include <cstdint>\n"
      f"constexpr std::int32_t unpack_src_format[32] = {{{arr(formats)}}};\n"
      f"constexpr std::int32_t unpack_dst_format[32] = {{{arr(formats)}}};\n",
    "chlkc_pack_data_format.h":
      "#pragma once\n"
      f"constexpr unsigned char pack_src_format[32] = {{{arr(formats)}}};\n"
      f"constexpr unsigned char pack_dst_format[32] = {{{arr(formats)}}};\n",
    "chlkc_unpack_tile_dims.h":
      "#pragma once\n"
      "#include <cstdint>\n"
      f"constexpr uint8_t unpack_tile_num_faces[32] = {{{arr32(4)}}};\n"
      f"constexpr uint8_t unpack_partial_face[32] = {{{arr32(0)}}};\n"
      f"constexpr uint8_t unpack_tile_face_r_dim[32] = {{{arr32(16)}}};\n"
      f"constexpr uint8_t unpack_narrow_tile[32] = {{{arr32(0)}}};\n"
      f"constexpr uint8_t unpack_tile_r_dim[32] = {{{arr32(32)}}};\n"
      f"constexpr uint8_t unpack_tile_c_dim[32] = {{{arr32(32)}}};\n"
      f"constexpr uint16_t unpack_tile_size[32] = {{{arr(tile_sizes)}}};\n",
    "chlkc_pack_tile_dims.h":
      "#pragma once\n"
      "#include <cstdint>\n"
      f"constexpr uint8_t pack_tile_num_faces[32] = {{{arr32(4)}}};\n"
      f"constexpr uint8_t pack_partial_face[32] = {{{arr32(0)}}};\n"
      f"constexpr uint8_t pack_tile_face_r_dim[32] = {{{arr32(16)}}};\n"
      f"constexpr uint8_t pack_narrow_tile[32] = {{{arr32(0)}}};\n"
      f"constexpr uint8_t pack_tile_r_dim[32] = {{{arr32(32)}}};\n"
      f"constexpr uint8_t pack_tile_c_dim[32] = {{{arr32(32)}}};\n"
      f"constexpr uint16_t pack_tile_size[32] = {{{arr(tile_sizes)}}};\n",
    "chlkc_dst_accum_mode.h":
      "#pragma once\n"
      f"constexpr bool DST_ACCUM_MODE = {bool_str(cfg.dst_accum_mode)};\n",
    "chlkc_dst_sync_mode.h":
      "#pragma once\n"
      f"#define DST_SYNC_MODE {dst_sync_mode}\n",
    "chlkc_math_fidelity.h":
      "#pragma once\n"
      "#include <cstdint>\n"
      f"constexpr std::int32_t MATH_FIDELITY = {cfg.math_fidelity.value};\n",
    "chlkc_math_approx_mode.h":
      "#pragma once\n"
      f"constexpr bool APPROX = {bool_str(cfg.approx)};\n",
  }

_PROFILE_DEFINES = [
  "-DPROFILE_KERNEL=1",
  f"-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC={TensixL1.PROFILER_HOST_BUFFER_BYTES_PER_RISC}",
]

def _run(exe: Path, args: list[str], cwd: Path):
  r = subprocess.run([str(exe), *args], cwd=cwd, capture_output=True)
  if r.returncode != 0:
    raise RuntimeError(f"{exe.name} failed:\n{r.stderr.decode()}")
  if PROFILER:
    pragma_re = re.compile(r"#pragma message:\s*['\"]?(.+?)['\"]?\s*$")
    for line in r.stderr.decode(errors="replace").splitlines():
      if "KERNEL_PROFILER" not in line or "#pragma message:" not in line:
        continue
      m = pragma_re.search(line)
      if not m:
        continue
      # Format: "name,file,line,KERNEL_PROFILER". Name may contain commas, so parse from the right.
      msg = m.group(1).strip()
      if not msg.endswith("KERNEL_PROFILER"):
        continue
      parts = msg.rsplit(",", 3)
      if len(parts) != 4:
        continue
      name, fpath, lineno_s, tag = (p.strip() for p in parts)
      if tag != "KERNEL_PROFILER":
        continue
      try:
        lineno = int(lineno_s)
      except ValueError:
        continue
      _zone_map[hash16(msg)] = (name, fpath, lineno)

def _compile_and_link(cc: Path, src: Path, compile_args: Strs, link_args: LinkArgs, tmp_prefix: str, prepare: PrepareFn = None) -> bytes:
  build = Path(tempfile.mkdtemp(prefix=tmp_prefix))
  try:
    if prepare is not None:
      prepare(build)
    _run(cc, [*compile_args, "-c", "-o", "out.o", str(src)], build)
    args = link_args(build) if callable(link_args) else link_args
    _run(cc, [*args, "-o", "out.elf"], build)
    return (build / "out.elf").read_bytes()
  finally:
    shutil.rmtree(build, ignore_errors=True)

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

def compile_firmware(profile: bool = PROFILER) -> dict[str, CompiledFirmware]:
  cached = _CACHE_DIR / ("firmware_profiler.pkl" if profile else "firmware.pkl")
  if cached.is_file():
    return pickle.loads(cached.read_bytes())

  cc = _SFPI / "riscv-tt-elf-g++"
  assert cc.is_file(), f"missing compiler: {cc}"

  common_defines = [
    "-DTENSIX_FIRMWARE", "-DFW_BUILD", "-DARCH_BLACKHOLE",
    "-DLOCAL_MEM_EN=0", "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", *_DEVICE_DEFINES,
  ]
  if profile:
    common_defines += _PROFILE_DEFINES
  lib = _DEPS / "lib" / "blackhole"
  ld_dir = _DEPS / "toolchain" / "blackhole"
  fw_src_dir = _REPO / "firmware"

  result: dict[str, CompiledFirmware] = {}

  for src_name, target, target_defs, mcpu, opt, extra in _FW_TARGETS:
    ld = ld_dir / f"firmware_{target}.ld"
    assert ld.is_file(), f"missing linker script: {ld}"
    src = fw_src_dir / src_name
    compile_args = [
      opt, *_CFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay",
      *common_defines, *target_defs, *_INCLUDES,
    ]
    link_objs = [str(lib / "tmu-crt0.o"), "out.o", *(str(lib / o) for o in extra), str(lib / "substitutes.o")]
    fw_link_args = [opt, *_CFLAGS, *_LFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay", f"-T{ld}", *link_objs]
    elf = _compile_and_link(
      cc=cc,
      src=src,
      compile_args=compile_args,
      link_args=fw_link_args,
      tmp_prefix=f"tt-fw-{target}-",
    )

    segs = list(iter_pt_load(elf))
    text_base = segs[0].paddr
    result[target] = CompiledFirmware(elf_bytes=elf, segments=segs, text_base=text_base)

  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cached.write_bytes(pickle.dumps(result))
  return result

class Compiler:
  def __init__(self, ckernel: CkernelConfig = CkernelConfig()):
    self._cc = _SFPI / "riscv-tt-elf-g++"
    self._objcopy = _SFPI / "riscv-tt-elf-objcopy"
    self._ckernel = ckernel
    self._includes = ["-I.", *_INCLUDES]
    assert self._cc.is_file(), f"missing compiler: {self._cc}\nDownload toolchain to {_DEPS / 'sfpi-toolchain'}"
    self._fw = compile_firmware(profile=PROFILER)

  def compile_dataflow(self, src: str, processor: str, noc_index: int | None = None) -> CompiledKernel:
    if processor not in ("brisc", "ncrisc"):
      raise ValueError(f"processor must be 'brisc' or 'ncrisc', got: {processor}")
    if noc_index is None:
      noc_index = 1 if processor == "brisc" else 0
    return self._compile_dataflow(src, processor, noc_index=noc_index)

  def compile_compute(self, src: str) -> tuple[CompiledKernel, CompiledKernel, CompiledKernel]:
    return (
      self._compile_trisc(src, 0),
      self._compile_trisc(src, 1),
      self._compile_trisc(src, 2),
    )

  def _compile_dataflow(self, src: str, target: str, noc_index: int, extra_defines: Strs | None = None,
                        extra_includes: Strs | None = None,
                        xip_relocate: bool = False, profiler: bool = True) -> CompiledKernel:
    defines = [
      *_KERNEL_DEFINES,
      f"-DCOMPILE_FOR_{target.upper()}",
      f"-DPROCESSOR_INDEX={0 if target == 'brisc' else 1}",
      f"-DNOC_INDEX={noc_index}", "-DNOC_MODE=0",
      *(extra_defines or []),
    ]
    if PROFILER and profiler:
      defines += _PROFILE_DEFINES
    extra_objs = [str(_DEPS / "lib/blackhole/noc.o")] if target == "brisc" else []
    return self._build(
      src, target, defines, extra_objs, opt="-O2", trisc=False,
      extra_includes=extra_includes, xip_relocate=xip_relocate
    )

  def _compile_trisc(self, src: str, trisc_id: int, profiler: bool = True) -> CompiledKernel:
    stage = ("unpack", "math", "pack")[trisc_id]
    defines = [
      *_KERNEL_DEFINES,
      f"-DCOMPILE_FOR_TRISC={trisc_id}",
      f"-DPROCESSOR_INDEX={trisc_id + 2}",
      f"-DUCK_CHLKC_{stage.upper()}", f"-DNAMESPACE=chlkc_{stage}",
    ]
    if PROFILER and profiler:
      defines += _PROFILE_DEFINES
    return self._build(src, f"trisc{trisc_id}", defines, [], opt="-O3", trisc=True)

  def _build(self, kern: str, target: str, defines: Strs, extra_objs: Strs, opt: str, trisc: bool, extra_includes: Strs | None = None,
             xip_relocate: bool = False) -> CompiledKernel:
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
      self._write_ckernel_headers(build)
      if trisc:
        self._write_trisc_stubs(build)

    def _kernel_link_args(_: Path) -> Strs:
      assert fw_link_elf is not None
      ld = _DEPS / "toolchain" / "blackhole" / f"kernel_{target}.ld"
      objs = ["out.o", *extra_objs, str(_DEPS / "lib/blackhole/substitutes.o")]
      return [
        opt, *_CFLAGS, *_LFLAGS, *mcpu, f"-T{ld}",
        "-Wl,--emit-relocs", f"-Wl,--just-symbols={fw_link_elf}", *objs,
      ]

    elf = _compile_and_link(
      cc=self._cc,
      src=fw_src,
      compile_args=compile_args,
      link_args=_kernel_link_args,
      tmp_prefix=f"tt-{target}-",
      prepare=_prepare,
    )
    xip, text_size = pack_xip_elf(elf, xip_relocate=xip_relocate)
    return CompiledKernel(xip=xip, xip_text_bytes=text_size)

  def _weaken_fw_symbols(self, build: Path, fw: bytes) -> Path:
    out = build / "fw.elf"
    out.write_bytes(fw)
    _run(self._objcopy, [
      "--localize-symbol=_start",
      "--localize-symbol=main",
      "--localize-symbol=exit",
      "--weaken",
      str(out),
    ], build)
    return out

  def _write_trisc_stubs(self, build: Path):
    (build / "defines_generated.h").write_text("")
    for stage, macro in [("unpack", "TRISC_UNPACK"), ("math", "TRISC_MATH"), ("pack", "TRISC_PACK")]:
      (build / f"chlkc_{stage}.cpp").write_text(
        f'#define {macro}\n#include "defines_generated.h"\n#include <kernel_includes.hpp>\n')

  def _write_ckernel_headers(self, build: Path):
    for name, content in _render_ckernel_headers(self._ckernel).items():
      (build / name).write_text(content)


@dataclass(frozen=True)
class CompiledCQKernels:
  prefetch_brisc: CompiledKernel
  dispatch_brisc: CompiledKernel
  dispatch_s_ncrisc: CompiledKernel

def compile_cq_kernels() -> CompiledCQKernels:
  cached = _CACHE_DIR / "cq_kernels.pkl"
  if cached.is_file():
    return pickle.loads(cached.read_bytes())

  compiler = Compiler()
  result = CompiledCQKernels(
    prefetch_brisc=compiler._compile_dataflow(_CQ_PREFETCH_SRC, "brisc", noc_index=0,
      extra_includes=_CQ_INC, xip_relocate=True, profiler=False),
    dispatch_brisc=compiler._compile_dataflow(_CQ_DISPATCH_SRC, "brisc", noc_index=1,
      extra_includes=_CQ_INC, xip_relocate=True, profiler=False),
    dispatch_s_ncrisc=compiler._compile_dataflow(_CQ_DISPATCH_S_SRC, "ncrisc", noc_index=1,
      extra_includes=_CQ_INC, xip_relocate=True, profiler=False),
  )
  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cached.write_bytes(pickle.dumps(result))
  return result

def _tile_size(fmt: int) -> int:
  sizes = {
    0: 4096,   # Float32
    1: 2048,   # Float16
    5: 2048,   # Float16_b
    8: 4096,   # Int32
    9: 2048,   # UInt16
    14: 1024,  # Int8
    24: 4096,  # UInt32
    26: 1024,  # Fp8_e4m3
    30: 1024,  # UInt8
  }
  if fmt not in sizes:
    raise ValueError(f"unsupported format: {fmt}")
  return sizes[fmt]
