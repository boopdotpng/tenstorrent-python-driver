from helpers import TT_HOME, pack_xip_elf
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil, subprocess, tempfile

class DataFormat(Enum):
  Float32, Float16, Bfp8, Bfp4, Tf32, Float16_b, Bfp8_b, Bfp4_b = 0, 1, 2, 3, 4, 5, 6, 7
  Int32, UInt16, Lf8, Bfp2, Int8, Bfp2_b, UInt32, Fp8_e4m3, UInt8 = 8, 9, 10, 11, 14, 15, 24, 0x1A, 30
  Invalid = 0xFF

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

_HW = TT_HOME / "runtime" / "hw"
_FW_SRC = TT_HOME / "tt_metal" / "hw" / "firmware" / "src" / "tt-1xx"
_SFPI = TT_HOME / "runtime" / "sfpi" / "compiler" / "bin"

_INCLUDES = [
  "tt_metal", "tt_metal/hw/inc", "tt_metal/hostdevcommon/api", "tt_metal/api",
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
  "-std=c++17", "-flto=auto", "-ffast-math", "-fno-exceptions", "-MMD",
  "-fno-use-cxa-atexit", "-Wall", "-Werror", "-Wno-unknown-pragmas",
  "-Wno-deprecated-declarations", "-Wno-error=multistatement-macros",
  "-Wno-error=parentheses", "-Wno-error=unused-but-set-variable",
  "-Wno-unused-variable", "-Wno-unused-function",
)
_LFLAGS = ("-Wl,-z,max-page-size=16", "-Wl,-z,common-page-size=16", "-nostartfiles")

# === Compiler ===

_DEVICE_DEFINES = [
  "-DNUM_DRAM_BANKS=7", "-DIS_NOT_POW2_NUM_DRAM_BANKS=1",
  "-DNUM_L1_BANKS=110", "-DIS_NOT_POW2_NUM_L1_BANKS=1",
  "-DPCIE_NOC_X=0", "-DPCIE_NOC_Y=3",
]

class Compiler:
  def __init__(self, ckernel: CkernelConfig = CkernelConfig()):
    self._cc = _SFPI / "riscv-tt-elf-g++"
    self._objcopy = _SFPI / "riscv-tt-elf-objcopy"
    self._nm = _SFPI / "riscv-tt-elf-nm"
    self._fw_dir = Path(__file__).resolve().parent / "riscv-firmware" / "p100a"
    self._ckernel = ckernel
    self._includes = ["-I."] + [f"-I{TT_HOME / p}" for p in _INCLUDES]
    assert self._cc.is_file(), f"missing compiler: {self._cc}"
    assert self._fw_dir.is_dir(), f"missing firmware: {self._fw_dir}"

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
    """Compile BRISC/NCRISC dataflow kernel."""
    defines = [
      "-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE",
      "-DDISPATCH_MESSAGE_ADDR=0", "-DKERNEL_BUILD", *_DEVICE_DEFINES,
      f"-DCOMPILE_FOR_{target.upper()}",
      f"-DPROCESSOR_INDEX={0 if target == 'brisc' else 1}",
      f"-DNOC_INDEX={noc_index}", "-DNOC_MODE=0",
    ]
    extra_objs = [str(_HW / "lib/blackhole/noc.o")] if target == "brisc" else []
    return self._build(src, target, defines, extra_objs, opt="-O2", trisc=False)

  def _compile_trisc(self, src: str, trisc_id: int) -> CompiledKernel:
    """Compile TRISC0/1/2 compute kernel."""
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
      # Setup build directory
      fw = self._weaken_fw_symbols(build, self._fw_dir / f"{target}.elf")
      (build / "kernel_includes.hpp").write_text(kern)
      self._write_ckernel_headers(build)
      if trisc: self._write_trisc_stubs(build)

      # Compile
      mcpu = ["-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"] if trisc else \
             ["-mcpu=tt-bh", "-mno-tt-tensix-optimize-replay", "-fno-tree-loop-distribute-patterns"]
      fw_src = _FW_SRC / ("trisck.cc" if trisc else f"{target}k.cc")
      self._run(self._cc, [opt, *_CFLAGS, *mcpu, *self._includes, "-c", "-o", "out.o", str(fw_src), *defines], build)

      # Link
      ld = _HW / "toolchain" / "blackhole" / f"kernel_{target}.ld"
      objs = ["out.o", *extra_objs, str(_HW / "lib/blackhole/substitutes.o")]
      self._run(self._cc, [opt, *_CFLAGS, *_LFLAGS, *mcpu, f"-T{ld}",
                "-Wl,--emit-relocs", f"-Wl,--just-symbols={fw}", *objs, "-o", "out.elf"], build)

      xip, text_size = pack_xip_elf(build / "out.elf")
      return CompiledKernel(xip=xip, xip_text_bytes=text_size)
    finally:
      shutil.rmtree(build, ignore_errors=True)

  def _run(self, exe: Path, args: list[str], cwd: Path):
    subprocess.run([str(exe), *args], cwd=cwd, check=True, capture_output=True)

  def _weaken_fw_symbols(self, build: Path, fw: Path) -> Path:
    """Copy firmware ELF and weaken/localize symbols to avoid collisions with kernel."""
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
    self._run(self._objcopy, [f"--localize-symbols=localize.txt", f"--weaken-symbols=weaken.txt", str(out)], build)
    return out

  def _write_trisc_stubs(self, build: Path):
    """Generate TRISC stage stub files."""
    (build / "defines_generated.h").write_text("")
    for stage, macro in [("unpack", "TRISC_UNPACK"), ("math", "TRISC_MATH"), ("pack", "TRISC_PACK")]:
      (build / f"chlkc_{stage}.cpp").write_text(
        f'#define {macro}\n#include "defines_generated.h"\n#include <kernel_includes.hpp>\n')

  def _write_ckernel_headers(self, build: Path):
    """Generate ckernel config headers (data formats, tile dims, math settings)."""
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

def _tile_size(fmt: int) -> int:
  """Tile size in bytes for 32x32 tile (1024 elements)."""
  return {
    0: 4096,                    # Float32
    1: 2048, 5: 2048,           # Float16, Float16_b
    2: 1088, 6: 1088, 10: 1088, 26: 1088,  # Bfp8, Bfp8_b, Lf8, Fp8_e4m3
    3: 576, 7: 576,             # Bfp4, Bfp4_b
    11: 320, 15: 320,           # Bfp2, Bfp2_b
  }.get(fmt) or (_ for _ in ()).throw(ValueError(f"unsupported format: {fmt}"))
