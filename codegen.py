from helpers import pack_xip_elf, load_pt_load, PTLoad
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os, shutil, subprocess, tempfile, time

# Local bundled dependencies (headers, libs, linker scripts, firmware sources)
_DEPS = Path(__file__).resolve().parent / "tt-metal-deps"
_REPO = Path(__file__).resolve().parent
_SFPI = _DEPS / "sfpi-toolchain" / "bin"

class DataFormat(Enum):
  Float32, Float16, Bfp8, Bfp4, Tf32, Float16_b, Bfp8_b, Bfp4_b = 0, 1, 2, 3, 4, 5, 6, 7
  Int32, UInt16, Lf8, Bfp2, Int8, Bfp2_b, UInt32, Fp8_e4m3, UInt8 = 8, 9, 10, 11, 14, 15, 24, 0x1A, 30
  Invalid = 0xFF

  @property
  def cname(self) -> str: return self.name  # C++ DataFormat::* enum name

class MathFidelity(Enum):
  LoFi = 0   # Lowest precision, fastest
  HiFi2 = 2  # Medium precision
  HiFi3 = 3  # Higher precision
  HiFi4 = 4  # Full precision, slowest

@dataclass(frozen=True)
class CkernelConfig:
  input_format: DataFormat = DataFormat.Float16_b
  output_format: DataFormat = DataFormat.Float16_b
  math_fidelity: MathFidelity = MathFidelity.HiFi4
  # use faster approximations for SFPU ops (exp, log, sqrt, etc)
  approx: bool = False
  # keep dest register in FP32 for accumulation (e.g. FP16 matmul with FP32 accum)
  dst_accum_mode: bool = False
  # full sync between math/pack stages (vs double-buffered SyncHalf)
  dst_full_sync: bool = False

@dataclass(frozen=True)
class CompiledKernel:
  xip: bytes
  xip_text_bytes: int

@dataclass(frozen=True)
class CompiledKernels:
  reader: CompiledKernel
  writer: CompiledKernel
  compute: tuple[CompiledKernel, CompiledKernel, CompiledKernel]

@dataclass(frozen=True)
class CompiledFirmware:
  """A compiled firmware ELF ready for upload to a Tensix core."""
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

_CFLAGS = (
  "-std=c++17", "-flto=auto", "-ffast-math", "-fno-exceptions",
  "-fno-use-cxa-atexit", "-Wall", "-Werror", "-Wno-unknown-pragmas",
  "-Wno-deprecated-declarations", "-Wno-error=multistatement-macros",
  "-Wno-error=parentheses", "-Wno-error=unused-but-set-variable",
  "-Wno-unused-variable", "-Wno-unused-function",
)
_LFLAGS = ("-Wl,-z,max-page-size=16", "-Wl,-z,common-page-size=16", "-nostartfiles")

_DEVICE_DEFINES = [
  "-DNUM_DRAM_BANKS=7", "-DIS_NOT_POW2_NUM_DRAM_BANKS=1",
  "-DNUM_L1_BANKS=110", "-DIS_NOT_POW2_NUM_L1_BANKS=1",
  "-DPCIE_NOC_X=0", "-DPCIE_NOC_Y=3",
]

def _run(exe: Path, args: list[str], cwd: Path):
  r = subprocess.run([str(exe), *args], cwd=cwd, capture_output=True)
  if r.returncode != 0:
    raise RuntimeError(f"{exe.name} failed:\n{r.stderr.decode()}")

# === Firmware compiler ===

# Firmware build targets: (source, target_name, defines, mcpu_flags, opt, extra_objs)
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

_fw_cache: dict[str, CompiledFirmware] | None = None

def compile_firmware() -> dict[str, CompiledFirmware]:
  """Compile all 5 slow-dispatch firmware ELFs from source. Cached after first call."""
  global _fw_cache
  if _fw_cache is not None: return _fw_cache

  cc = _SFPI / "riscv-tt-elf-g++"
  assert cc.is_file(), f"missing compiler: {cc}"

  inc = _DEPS / "include"
  includes = [f"-I{inc}"] + [f"-I{inc / p}" for p in _INCLUDE_PATHS]
  common_defines = [
    "-DTENSIX_FIRMWARE", "-DFW_BUILD", "-DARCH_BLACKHOLE",
    "-DLOCAL_MEM_EN=0", "-DDISPATCH_MESSAGE_ADDR=0", *_DEVICE_DEFINES,
  ]
  lib = _DEPS / "lib" / "blackhole"
  ld_dir = _DEPS / "toolchain" / "blackhole"
  fw_src_dir = _REPO / "firmware"

  cache_dir = Path(tempfile.mkdtemp(prefix="tt-fw-"))
  result: dict[str, CompiledFirmware] = {}

  for src_name, target, target_defs, mcpu, opt, extra in _FW_TARGETS:
    obj = cache_dir / f"{target}.o"
    elf = cache_dir / f"{target}.elf"
    ld = ld_dir / f"firmware_{target}.ld"
    assert ld.is_file(), f"missing linker script: {ld}"

    # Compile
    _run(cc, [opt, *_CFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay",
              *common_defines, *target_defs, *includes,
              "-c", "-o", str(obj), str(fw_src_dir / src_name)], cache_dir)

    # Link: tmu-crt0.o + firmware.o + [noc.o] + substitutes.o
    link_objs = [str(lib / "tmu-crt0.o"), str(obj)]
    link_objs += [str(lib / o) for o in extra]
    link_objs.append(str(lib / "substitutes.o"))
    _run(cc, [opt, *_CFLAGS, *_LFLAGS, *mcpu, "-mno-tt-tensix-optimize-replay",
              f"-T{ld}", *link_objs, "-o", str(elf)], cache_dir)

    segs = load_pt_load(elf)
    # Text base comes from the first PT_LOAD segment (placed by the linker script)
    text_base = segs[0].paddr
    result[target] = CompiledFirmware(elf_path=elf, segments=segs, text_base=text_base)

  _fw_cache = result
  return result

# === Kernel compiler ===

class Compiler:
  def __init__(self, ckernel: CkernelConfig = CkernelConfig()):
    self._cc = _SFPI / "riscv-tt-elf-g++"
    self._objcopy = _SFPI / "riscv-tt-elf-objcopy"
    self._nm = _SFPI / "riscv-tt-elf-nm"
    self._ckernel = ckernel
    inc = _DEPS / "include"
    self._includes = ["-I.", f"-I{inc}"] + [f"-I{inc / p}" for p in _INCLUDE_PATHS]
    assert self._cc.is_file(), f"missing compiler: {self._cc}\nDownload toolchain to {_DEPS / 'sfpi-toolchain'}"
    # Ensure firmware is compiled (needed for --just-symbols linking)
    self._fw = compile_firmware()

  def compile(self, reader: str, writer: str, compute: str) -> CompiledKernels:
    return CompiledKernels(
      reader=self._compile_dataflow(reader, "ncrisc", noc_index=0),
      writer=self._compile_dataflow(writer, "brisc", noc_index=1),
      compute=(
        self._compile_trisc(compute, 0),
        self._compile_trisc(compute, 1),
        self._compile_trisc(compute, 2),
      ),
    )

  def _compile_dataflow(self, src: str, target: str, noc_index: int) -> CompiledKernel:
    defines = [
      "-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE",
      "-DDISPATCH_MESSAGE_ADDR=0", "-DKERNEL_BUILD", *_DEVICE_DEFINES,
      f"-DCOMPILE_FOR_{target.upper()}",
      f"-DPROCESSOR_INDEX={0 if target == 'brisc' else 1}",
      f"-DNOC_INDEX={noc_index}", "-DNOC_MODE=0",
    ]
    extra_objs = [str(_DEPS / "lib/blackhole/noc.o")] if target == "brisc" else []
    return self._build(src, target, defines, extra_objs, opt="-O2", trisc=False)

  def _compile_trisc(self, src: str, trisc_id: int) -> CompiledKernel:
    stage = ("unpack", "math", "pack")[trisc_id]
    defines = [
      "-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE",
      "-DDISPATCH_MESSAGE_ADDR=0", "-DKERNEL_BUILD", *_DEVICE_DEFINES,
      f"-DCOMPILE_FOR_TRISC={trisc_id}",
      f"-DPROCESSOR_INDEX={trisc_id + 2}",
      f"-DUCK_CHLKC_{stage.upper()}", f"-DNAMESPACE=chlkc_{stage}",
    ]
    return self._build(src, f"trisc{trisc_id}", defines, [], opt="-O3", trisc=True)

  def _build(self, kern: str, target: str, defines: list[str], extra_objs: list[str],
             opt: str, trisc: bool) -> CompiledKernel:
    build = Path(tempfile.mkdtemp(prefix=f"tt-{target}-"))
    try:
      # Get the compiled firmware ELF for this target (for --just-symbols)
      fw_elf = self._fw[target].elf_path
      fw = self._weaken_fw_symbols(build, fw_elf)
      (build / "kernel_includes.hpp").write_text(kern)
      self._write_ckernel_headers(build)
      if trisc: self._write_trisc_stubs(build)

      # Compile
      mcpu = ["-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"] if trisc else \
             ["-mcpu=tt-bh", "-mno-tt-tensix-optimize-replay", "-fno-tree-loop-distribute-patterns"]
      fw_src = _DEPS / "firmware-src" / ("trisck.cc" if trisc else f"{target}k.cc")
      _run(self._cc, [opt, *_CFLAGS, "-MMD", *mcpu, *self._includes, "-c", "-o", "out.o", str(fw_src), *defines], build)

      # Link
      ld = _DEPS / "toolchain" / "blackhole" / f"kernel_{target}.ld"
      objs = ["out.o", *extra_objs, str(_DEPS / "lib/blackhole/substitutes.o")]
      _run(self._cc, [opt, *_CFLAGS, *_LFLAGS, *mcpu, f"-T{ld}",
                "-Wl,--emit-relocs", f"-Wl,--just-symbols={fw}", *objs, "-o", "out.elf"], build)

      xip, text_size = pack_xip_elf(build / "out.elf")
      return CompiledKernel(xip=xip, xip_text_bytes=text_size)
    finally:
      shutil.rmtree(build, ignore_errors=True)

  def _weaken_fw_symbols(self, build: Path, fw: Path) -> Path:
    out = build / "fw.elf"
    out.write_bytes(fw.read_bytes())

    # Parse symbols
    result = subprocess.run([str(self._nm), "-a", str(out)], cwd=build, check=True, capture_output=True, text=True)
    localize, weaken = [], []
    for line in result.stdout.splitlines():
      parts = line.split()
      if len(parts) < 3: continue
      _, sym_type, name = parts[:3]
      if sym_type == "U" or not sym_type.isupper(): continue
      if name == "__global_pointer$" or name.startswith("__fw_export_"): continue
      # Data symbols get weakened, code symbols get localized
      if sym_type in "BDRSGV":
        weaken.append(name)
      else:
        localize.append(name)

    # Apply
    (build / "localize.txt").write_text("\n".join(sorted(set(localize))))
    (build / "weaken.txt").write_text("\n".join(sorted(set(weaken))))
    _run(self._objcopy, ["--localize-symbols=localize.txt", "--weaken-symbols=weaken.txt", str(out)], build)
    return out

  def _write_trisc_stubs(self, build: Path):
    (build / "defines_generated.h").write_text("")
    for stage, macro in [("unpack", "TRISC_UNPACK"), ("math", "TRISC_MATH"), ("pack", "TRISC_PACK")]:
      (build / f"chlkc_{stage}.cpp").write_text(
        f'#define {macro}\n#include "defines_generated.h"\n#include <kernel_includes.hpp>\n')

  def _write_ckernel_headers(self, build: Path):
    cfg = self._ckernel
    in_fmt, out_fmt = cfg.input_format.value, cfg.output_format.value
    formats = [in_fmt] * 16 + [out_fmt] * 16
    tile_sizes = [_tile_size(f) for f in formats]

    def arr(vals): return ", ".join(str(v) for v in vals)
    def arr32(v): return arr([v] * 32)

    # Data formats
    (build / "chlkc_unpack_data_format.h").write_text(
      f"#pragma once\nconstexpr std::int32_t unpack_src_format[32] = {{{arr(formats)}}};\n"
      f"constexpr std::int32_t unpack_dst_format[32] = {{{arr(formats)}}};\n")
    (build / "chlkc_pack_data_format.h").write_text(
      f"#pragma once\nconstexpr unsigned char pack_src_format[32] = {{{arr(formats)}}};\n"
      f"constexpr unsigned char pack_dst_format[32] = {{{arr(formats)}}};\n")

    # Tile dimensions (fixed 32x32 tiles, 4 faces of 16x16)
    def tile_dims(prefix: str) -> str:
      return (f"constexpr uint8_t {prefix}_tile_num_faces[32] = {{{arr32(4)}}};\n"
              f"constexpr uint8_t {prefix}_partial_face[32] = {{{arr32(0)}}};\n"
              f"constexpr uint8_t {prefix}_tile_face_r_dim[32] = {{{arr32(16)}}};\n"
              f"constexpr uint8_t {prefix}_narrow_tile[32] = {{{arr32(0)}}};\n"
              f"constexpr uint8_t {prefix}_tile_r_dim[32] = {{{arr32(32)}}};\n"
              f"constexpr uint8_t {prefix}_tile_c_dim[32] = {{{arr32(32)}}};\n"
              f"constexpr uint16_t {prefix}_tile_size[32] = {{{arr(tile_sizes)}}};\n")
    (build / "chlkc_unpack_tile_dims.h").write_text("#pragma once\n" + tile_dims("unpack"))
    (build / "chlkc_pack_tile_dims.h").write_text("#pragma once\n" + tile_dims("pack"))

    # Math settings
    (build / "chlkc_dst_accum_mode.h").write_text(f"constexpr bool DST_ACCUM_MODE = {str(cfg.dst_accum_mode).lower()};\n")
    (build / "chlkc_dst_sync_mode.h").write_text(f"#define DST_SYNC_MODE DstSync::{'SyncFull' if cfg.dst_full_sync else 'SyncHalf'}\n")
    (build / "chlkc_math_fidelity.h").write_text(f"constexpr std::int32_t MATH_FIDELITY = {cfg.math_fidelity.value};\n")
    (build / "chlkc_math_approx_mode.h").write_text(f"constexpr bool APPROX = {str(cfg.approx).lower()};\n")

DRAIN_KERNEL_SRC = r"""
#include <cstdint>

void kernel_main() {
  uint32_t dram_addr = get_arg_val<uint32_t>(0);
  uint32_t pcie_noc_xy = get_arg_val<uint32_t>(1);
  uint32_t sysmem_offset = get_arg_val<uint32_t>(2);
  uint32_t tile_offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
  uint32_t page_size = get_arg_val<uint32_t>(5);
  constexpr uint32_t cb_id = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = dram_addr,
    .page_size = page_size,
    .data_format = DataFormat::Float16_b,
  };
  uint64_t pcie_base = ((uint64_t)pcie_noc_xy << 36) | (1ULL << 60);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t tile_id = tile_offset + i;
    cb_reserve_back(cb_id, 1);
    uint32_t l1_addr = get_write_ptr(cb_id);
    noc_async_read_tile(tile_id, dram, l1_addr);
    noc_async_read_barrier();
    uint64_t dst = pcie_base + sysmem_offset + (uint64_t)tile_id * page_size;
    noc_async_write(l1_addr, dst, page_size);
    noc_async_write_barrier();
    cb_push_back(cb_id, 1);
    cb_wait_front(cb_id, 1);
    cb_pop_front(cb_id, 1);
  }
}
"""

_drain_kernel: CompiledKernel | None = None

def get_drain_kernel() -> CompiledKernel:
  global _drain_kernel
  if _drain_kernel is None:
    _drain_kernel = Compiler()._compile_dataflow(DRAIN_KERNEL_SRC, "ncrisc", noc_index=0)
  return _drain_kernel

def _tile_size(fmt: int) -> int:
  return {
    0: 4096,                    # Float32
    1: 2048, 5: 2048,           # Float16, Float16_b
    2: 1088, 6: 1088, 10: 1088, 26: 1088,  # Bfp8, Bfp8_b, Lf8, Fp8_e4m3
    3: 576, 7: 576,             # Bfp4, Bfp4_b
    11: 320, 15: 320,           # Bfp2, Bfp2_b
  }.get(fmt) or (_ for _ in ()).throw(ValueError(f"unsupported format: {fmt}"))
