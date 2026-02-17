from helpers import pack_xip_elf, load_pt_load, PTLoad, PROFILER
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib, shutil, subprocess, tempfile

_DEPS = Path(__file__).resolve().parent / "tt-metal-deps"
_REPO = Path(__file__).resolve().parent
_SFPI = _DEPS / "sfpi-toolchain" / "bin"
_CACHE = _REPO / ".metal-cache"
_FW_CACHE_DIR = _CACHE / "firmware"
_KERNEL_CACHE_DIR = _CACHE / "kernels"
_CKERNEL_DEFAULTS = _REPO / "firmware" / "ckernel_defaults"

Strs = list[str]

# Zone hash→name mapping captured from DeviceZoneScopedN #pragma messages
_zone_map: dict[int, tuple[str, str, int]] = {}  # hash → (name, file, line)

def _hash16(s: str) -> int:
  """FNV1a-32 folded to 16 bits, matching Hash16_CT in kernel_profiler.hpp."""
  h = 0x811c9dc5
  for c in s.encode():
    h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
  return (h >> 16) ^ (h & 0xFFFF)

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
  elf_path: Path             # path to the ELF on disk (in cache dir)
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
  "-Wall", "-Werror", "-Wno-unknown-pragmas",
  "-Wno-deprecated-declarations", "-Wno-error=multistatement-macros",
  "-Wno-error=parentheses", "-Wno-error=unused-but-set-variable",
  "-Wno-unused-variable", "-Wno-unused-function",
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

def _hash_default_ckernel_headers() -> str:
  h = hashlib.sha256()
  for name in (
    "chlkc_unpack_data_format.h",
    "chlkc_pack_data_format.h",
    "chlkc_unpack_tile_dims.h",
    "chlkc_pack_tile_dims.h",
    "chlkc_dst_accum_mode.h",
    "chlkc_dst_sync_mode.h",
    "chlkc_math_fidelity.h",
    "chlkc_math_approx_mode.h",
  ):
    h.update((_CKERNEL_DEFAULTS / name).read_bytes())
  return h.hexdigest()

_CKERNEL_DEFAULTS_HASH = _hash_default_ckernel_headers()

def _run(exe: Path, args: list[str], cwd: Path):
  r = subprocess.run([str(exe), *args], cwd=cwd, capture_output=True)
  if r.returncode != 0:
    raise RuntimeError(f"{exe.name} failed:\n{r.stderr.decode()}")
  if PROFILER:
    for line in r.stderr.decode(errors="replace").splitlines():
      if "KERNEL_PROFILER" in line and "#pragma message:" in line:
        # Format: "...#pragma message: name,file,line,KERNEL_PROFILER"
        try:
          msg = line.split("#pragma message:", 1)[1].strip().strip('"')
          parts = msg.split(",")
          if len(parts) >= 4 and parts[-1].strip() == "KERNEL_PROFILER":
            name, fpath, lineno = parts[0].strip(), parts[1].strip(), int(parts[2].strip())
            _zone_map[_hash16(name)] = (name, fpath, lineno)
        except (ValueError, IndexError):
          pass

def _compile_and_link_cached(cc: Path, src: Path, cache_elf: Path, compile_args: Strs, link_args: LinkArgs, tmp_prefix: str,
                             prepare: PrepareFn = None):
  if cache_elf.is_file():
    return
  build = Path(tempfile.mkdtemp(prefix=tmp_prefix))
  try:
    if prepare is not None:
      prepare(build)
    _run(cc, [*compile_args, "-c", "-o", "out.o", str(src)], build)
    args = link_args(build) if callable(link_args) else link_args
    _run(cc, [*args, "-o", "out.elf"], build)
    shutil.copy2(build / "out.elf", cache_elf)
  finally:
    shutil.rmtree(build, ignore_errors=True)

def _source_target_cache_key(source: str, target: str, meta: dict[str, object] | None = None) -> str:
  payload = f"{target}\0{source}".encode() if not meta else repr((target, source, tuple(sorted(meta.items())))).encode()
  return hashlib.sha256(payload).hexdigest()

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

_fw_cache_by_mode: dict[bool, dict[str, CompiledFirmware]] = {}

def compile_firmware() -> dict[str, CompiledFirmware]:
  mode = PROFILER
  if mode in _fw_cache_by_mode:
    return _fw_cache_by_mode[mode]
  _FW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

  cc = _SFPI / "riscv-tt-elf-g++"
  assert cc.is_file(), f"missing compiler: {cc}"

  common_defines = [
    "-DTENSIX_FIRMWARE", "-DFW_BUILD", "-DARCH_BLACKHOLE",
    "-DLOCAL_MEM_EN=0", "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", *_DEVICE_DEFINES,
  ]
  if PROFILER:
    common_defines += ["-DPROFILE_KERNEL=1", "-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=65536"]
  lib = _DEPS / "lib" / "blackhole"
  ld_dir = _DEPS / "toolchain" / "blackhole"
  fw_src_dir = _REPO / "firmware"

  result: dict[str, CompiledFirmware] = {}

  for src_name, target, target_defs, mcpu, opt, extra in _FW_TARGETS:
    ld = ld_dir / f"firmware_{target}.ld"
    assert ld.is_file(), f"missing linker script: {ld}"
    src = fw_src_dir / src_name
    cache_target = f"{target}_firmware" + ("_prof" if PROFILER else "")
    key = _source_target_cache_key(src.read_text(), cache_target)
    elf = _FW_CACHE_DIR / f"{cache_target}-{key[:16]}.elf"
    compile_args = [
      opt, *_CFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay",
      *common_defines, *target_defs, *_INCLUDES,
    ]
    link_objs = [str(lib / "tmu-crt0.o"), "out.o", *(str(lib / o) for o in extra), str(lib / "substitutes.o")]
    fw_link_args = [opt, *_CFLAGS, *_LFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay", f"-T{ld}", *link_objs]
    _compile_and_link_cached(
      cc=cc,
      src=src,
      cache_elf=elf,
      compile_args=compile_args,
      link_args=fw_link_args,
      tmp_prefix=f"tt-fw-{target}-",
    )

    segs = load_pt_load(elf)
    text_base = segs[0].paddr
    result[target] = CompiledFirmware(elf_path=elf, segments=segs, text_base=text_base)

  _fw_cache_by_mode[mode] = result
  return result

class Compiler:
  def __init__(self, ckernel: CkernelConfig = CkernelConfig()):
    self._cc = _SFPI / "riscv-tt-elf-g++"
    self._objcopy = _SFPI / "riscv-tt-elf-objcopy"
    self._ckernel = ckernel
    self._includes = ["-I.", f"-I{_CKERNEL_DEFAULTS}", *_INCLUDES]
    assert self._cc.is_file(), f"missing compiler: {self._cc}\nDownload toolchain to {_DEPS / 'sfpi-toolchain'}"
    self._fw = compile_firmware()

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
                        xip_relocate: bool = False) -> CompiledKernel:
    defines = [
      *_KERNEL_DEFINES,
      f"-DCOMPILE_FOR_{target.upper()}",
      f"-DPROCESSOR_INDEX={0 if target == 'brisc' else 1}",
      f"-DNOC_INDEX={noc_index}", "-DNOC_MODE=0",
      *(extra_defines or []),
    ]
    if PROFILER:
      defines += ["-DPROFILE_KERNEL=1", "-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=65536"]
    extra_objs = [str(_DEPS / "lib/blackhole/noc.o")] if target == "brisc" else []
    return self._build(
      src, target, defines, extra_objs, opt="-O2", trisc=False,
      extra_includes=extra_includes, xip_relocate=xip_relocate
    )

  def _compile_trisc(self, src: str, trisc_id: int) -> CompiledKernel:
    stage = ("unpack", "math", "pack")[trisc_id]
    defines = [
      *_KERNEL_DEFINES,
      f"-DCOMPILE_FOR_TRISC={trisc_id}",
      f"-DPROCESSOR_INDEX={trisc_id + 2}",
      f"-DUCK_CHLKC_{stage.upper()}", f"-DNAMESPACE=chlkc_{stage}",
    ]
    if PROFILER:
      defines += ["-DPROFILE_KERNEL=1", "-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=65536"]
    return self._build(src, f"trisc{trisc_id}", defines, [], opt="-O3", trisc=True)

  def _build(self, kern: str, target: str, defines: Strs, extra_objs: Strs, opt: str, trisc: bool, extra_includes: Strs | None = None,
             xip_relocate: bool = False) -> CompiledKernel:
    _KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _source_target_cache_key(
      kern,
      target,
      meta={
        "defines": tuple(defines),
        "extra_objs": tuple(extra_objs),
        "opt": opt,
        "trisc": trisc,
        "extra_includes": tuple(extra_includes or []),
        "xip_relocate": xip_relocate,
        "ckernel_input_format": self._ckernel.input_format.value,
        "ckernel_output_format": self._ckernel.output_format.value,
        "ckernel_cb_data_formats": tuple((cb, fmt.value) for cb, fmt in self._ckernel.cb_data_formats),
        "ckernel_math_fidelity": self._ckernel.math_fidelity.value,
        "ckernel_approx": self._ckernel.approx,
        "ckernel_dst_accum_mode": self._ckernel.dst_accum_mode,
        "ckernel_dst_full_sync": self._ckernel.dst_full_sync,
        "ckernel_defaults_hash": _CKERNEL_DEFAULTS_HASH,
      },
    )
    cached_elf = _KERNEL_CACHE_DIR / f"{target}-{key[:16]}.elf"
    if cached_elf.is_file():
      xip, text_size = pack_xip_elf(cached_elf, xip_relocate=xip_relocate)
      return CompiledKernel(xip=xip, xip_text_bytes=text_size)
    mcpu = ["-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"] if trisc else \
           ["-mcpu=tt-bh", "-mno-tt-tensix-optimize-replay", "-fno-tree-loop-distribute-patterns"]
    fw_src = _DEPS / "firmware-src" / ("trisck.cc" if trisc else f"{target}k.cc")
    includes = [*self._includes, *(f"-I{p}" for p in (extra_includes or []))]
    compile_args = [opt, *_CFLAGS, "-MMD", *mcpu, *includes, *defines]

    fw_link_elf: Path | None = None
    def _prepare(build: Path):
      nonlocal fw_link_elf
      fw_link_elf = self._weaken_fw_symbols(build, self._fw[target].elf_path)
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

    _compile_and_link_cached(
      cc=self._cc,
      src=fw_src,
      cache_elf=cached_elf,
      compile_args=compile_args,
      link_args=_kernel_link_args,
      tmp_prefix=f"tt-{target}-",
      prepare=_prepare,
    )
    xip, text_size = pack_xip_elf(cached_elf, xip_relocate=xip_relocate)
    return CompiledKernel(xip=xip, xip_text_bytes=text_size)

  def _weaken_fw_symbols(self, build: Path, fw: Path) -> Path:
    out = build / "fw.elf"
    out.write_bytes(fw.read_bytes())
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
    cfg = self._ckernel
    if cfg.input_format != DataFormat.Float16_b or cfg.output_format != DataFormat.Float16_b or cfg.cb_data_formats:
      in_fmt, out_fmt = cfg.input_format.value, cfg.output_format.value
      formats = [in_fmt] * 16 + [out_fmt] * 16
      for cb_id, fmt in cfg.cb_data_formats:
        if not (0 <= cb_id < 32):
          raise ValueError(f"cb_data_formats has invalid cb_id {cb_id}; expected 0..31")
        formats[cb_id] = fmt.value
      tile_sizes = [_tile_size(f) for f in formats]
      arr = lambda vals: ", ".join(str(v) for v in vals)
      arr32 = lambda v: arr([v] * 32)
      (build / "chlkc_unpack_data_format.h").write_text(
        f"#pragma once\nconstexpr std::int32_t unpack_src_format[32] = {{{arr(formats)}}};\n"
        f"constexpr std::int32_t unpack_dst_format[32] = {{{arr(formats)}}};\n")
      (build / "chlkc_pack_data_format.h").write_text(
        f"#pragma once\nconstexpr unsigned char pack_src_format[32] = {{{arr(formats)}}};\n"
        f"constexpr unsigned char pack_dst_format[32] = {{{arr(formats)}}};\n")
      dims = lambda prefix: (
        f"constexpr uint8_t {prefix}_tile_num_faces[32] = {{{arr32(4)}}};\n"
        f"constexpr uint8_t {prefix}_partial_face[32] = {{{arr32(0)}}};\n"
        f"constexpr uint8_t {prefix}_tile_face_r_dim[32] = {{{arr32(16)}}};\n"
        f"constexpr uint8_t {prefix}_narrow_tile[32] = {{{arr32(0)}}};\n"
        f"constexpr uint8_t {prefix}_tile_r_dim[32] = {{{arr32(32)}}};\n"
        f"constexpr uint8_t {prefix}_tile_c_dim[32] = {{{arr32(32)}}};\n"
        f"constexpr uint16_t {prefix}_tile_size[32] = {{{arr(tile_sizes)}}};\n"
      )
      (build / "chlkc_unpack_tile_dims.h").write_text("#pragma once\n" + dims("unpack"))
      (build / "chlkc_pack_tile_dims.h").write_text("#pragma once\n" + dims("pack"))
    if cfg.dst_accum_mode:
      (build / "chlkc_dst_accum_mode.h").write_text("constexpr bool DST_ACCUM_MODE = true;\n")
    if cfg.dst_full_sync:
      (build / "chlkc_dst_sync_mode.h").write_text("#define DST_SYNC_MODE DstSync::SyncFull\n")
    if cfg.math_fidelity != MathFidelity.HiFi4:
      (build / "chlkc_math_fidelity.h").write_text(f"constexpr std::int32_t MATH_FIDELITY = {cfg.math_fidelity.value};\n")
    if cfg.approx:
      (build / "chlkc_math_approx_mode.h").write_text("constexpr bool APPROX = true;\n")


@dataclass(frozen=True)
class CompiledCQKernels:
  prefetch_brisc: CompiledKernel
  dispatch_brisc: CompiledKernel
  dispatch_s_ncrisc: CompiledKernel

def compile_cq_kernels() -> CompiledCQKernels:
  compiler = Compiler()
  cfg_hash = hashlib.sha256((_CQ_SRC_DIR / "cq_fixed_config.hpp").read_bytes()).hexdigest()[:16]
  source_tag = f"// cq_fixed_config={cfg_hash}\n"

  return CompiledCQKernels(
    prefetch_brisc=compiler._compile_dataflow(source_tag + _CQ_PREFETCH_SRC, "brisc", noc_index=0,
      extra_includes=_CQ_INC,
      xip_relocate=True),
    dispatch_brisc=compiler._compile_dataflow(source_tag + _CQ_DISPATCH_SRC, "brisc", noc_index=1,
      extra_includes=_CQ_INC,
      xip_relocate=True),
    dispatch_s_ncrisc=compiler._compile_dataflow(source_tag + _CQ_DISPATCH_S_SRC, "ncrisc", noc_index=1,
      extra_includes=_CQ_INC,
      xip_relocate=True),
  )

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
