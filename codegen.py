from __future__ import annotations
from helpers import TT_HOME, pack_xip_elf
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import shutil
import subprocess
import tempfile

# Kernel compilation for Blackhole (p100a).
# Compiles C++ kernel sources for BRISC/NCRISC/TRISC0-2, linked against firmware symbols,
# with JIT-generated chlkc_* descriptor headers for tile formats/dims.

class Processor(Enum):
  NCRISC = auto()
  BRISC = auto()
  TRISC0 = auto()
  TRISC1 = auto()
  TRISC2 = auto()

@dataclass(frozen=True)
class CkernelConfig:
  # Generated chlkc_* headers for compile-time tile format/dim configuration.
  # Runtime CB config must match these or behavior is undefined.
  compile_time_args: tuple[int, ...] = (2,)  # TensorAccessorArgs<0> => IsDram
  data_format: int = 5  # DataFormat::Float16_b
  cb_data_formats: dict[int, int] | None = (
    None  # {cb_index: DataFormat}, others become 255
  )
  inactive_tile_size_bytes: int = 1088
  tile_r_dim: int = 32
  tile_c_dim: int = 32
  tile_num_faces: int = 4
  tile_face_r_dim: int = 16
  partial_face: int = 0
  narrow_tile: int = 0
  math_fidelity: int = 4  # MathFidelity::HiFi4
  approx: bool = False
  dst_accum_mode: bool = False
  dst_full_sync: bool = False

@dataclass(frozen=True)
class CompiledKernel:
  processor: Processor
  elf: bytes
  xip: bytes
  xip_text_bytes: int

_BASE_INCLUDES = [
  "tt_metal",
  "tt_metal/hw/inc",
  "tt_metal/hostdevcommon/api",
  "tt_metal/api",
  "tt_metal/include",
  "tt_metal/hw/inc/internal/tt-1xx",
  "tt_metal/hw/inc/internal/tt-1xx/blackhole",
  "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc",
  "tt_metal/hw/ckernels/blackhole/metal/llk_io",
]
_CKERNEL_INCLUDES = [
  "tt_metal/hw/ckernels/blackhole/metal/common",
  "tt_metal/hw/ckernels/blackhole/metal/llk_api",
  "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu",
  "tt_metal/hw/ckernels/blackhole/metal/llk_io",
  "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc",
  "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
]

# Per-processor compilation parameters
_PROC_CONFIG = {
  Processor.BRISC: {
    "target": "brisc",
    "proc_idx": 0,
    "opt": "-O2",
    "mcpu": (
      "-mcpu=tt-bh",
      "-mno-tt-tensix-optimize-replay",
      "-fno-tree-loop-distribute-patterns",
    ),
    "is_trisc": False,
  },
  Processor.NCRISC: {
    "target": "ncrisc",
    "proc_idx": 1,
    "opt": "-O2",
    "mcpu": (
      "-mcpu=tt-bh",
      "-mno-tt-tensix-optimize-replay",
      "-fno-tree-loop-distribute-patterns",
    ),
    "is_trisc": False,
  },
  Processor.TRISC0: {
    "target": "trisc0",
    "proc_idx": 2,
    "trisc_id": 0,
    "opt": "-O3",
    "mcpu": ("-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"),
    "stage_defines": ("-DUCK_CHLKC_UNPACK", "-DNAMESPACE=chlkc_unpack"),
    "is_trisc": True,
  },
  Processor.TRISC1: {
    "target": "trisc1",
    "proc_idx": 3,
    "trisc_id": 1,
    "opt": "-O3",
    "mcpu": ("-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"),
    "stage_defines": ("-DUCK_CHLKC_MATH", "-DNAMESPACE=chlkc_math"),
    "is_trisc": True,
  },
  Processor.TRISC2: {
    "target": "trisc2",
    "proc_idx": 4,
    "trisc_id": 2,
    "opt": "-O3",
    "mcpu": ("-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay"),
    "stage_defines": ("-DUCK_CHLKC_PACK", "-DNAMESPACE=chlkc_pack"),
    "is_trisc": True,
  },
}

_HW = TT_HOME / "runtime" / "hw"
_FW_SRC = TT_HOME / "tt_metal" / "hw" / "firmware" / "src" / "tt-1xx"
_SUBSTITUTES = _HW / "lib" / "blackhole" / "substitutes.o"

class Compiler:
  # Flags from TT-metal's JIT build. `-flto` required, `-nostartfiles` because linker scripts own entry.
  COMMON_CFLAGS = (
    "-std=c++17",
    "-flto=auto",
    "-ffast-math",
    "-fno-exceptions",
    "-MMD",
    "-fno-use-cxa-atexit",
    "-Wall",
    "-Werror",
    "-Wno-unknown-pragmas",
    "-Wno-deprecated-declarations",
    "-Wno-error=multistatement-macros",
    "-Wno-error=parentheses",
    "-Wno-error=unused-but-set-variable",
    "-Wno-unused-variable",
    "-Wno-unused-function",
  )
  COMMON_LFLAGS = (
    "-Wl,-z,max-page-size=16",
    "-Wl,-z,common-page-size=16",
    "-nostartfiles",
  )

  def __init__(
    self,
    *,
    arch: str = "p100a",
    firmware_dir: Path | None = None,
    ckernel: CkernelConfig = CkernelConfig(),
    debug_info: bool = False,
    device_defines: dict[str, int] | None = None,
  ):
    sfpi = TT_HOME / "runtime" / "sfpi" / "compiler" / "bin"
    self.compiler = sfpi / "riscv-tt-elf-g++"
    self.objcopy = sfpi / "riscv-tt-elf-objcopy"
    self.nm = sfpi / "riscv-tt-elf-nm"
    assert self.compiler.is_file(), f"missing SFPI compiler: {self.compiler}"
    assert self.objcopy.is_file(), f"missing SFPI objcopy: {self.objcopy}"
    self.firmware_dir = firmware_dir or self._default_firmware_dir(arch)
    assert self.firmware_dir.is_dir(), f"missing firmware dir: {self.firmware_dir}"
    self.ckernel = ckernel
    self.debug_info = debug_info
    inferred = self._infer_device_defines_from_firmware(self.firmware_dir / "brisc.elf")
    self.device_defines = {**inferred, **(device_defines or {})}

  def _infer_device_defines_from_firmware(self, fw: Path) -> dict[str, int]:
    if not (self.nm.is_file() and fw.is_file()):
      return {}
    try:
      proc = subprocess.run(
        [str(self.nm), "--print-size", str(fw)],
        check=True,
        text=True,
        capture_output=True,
      )
    except subprocess.CalledProcessError:
      return {}
    sizes: dict[str, int] = {}
    for line in proc.stdout.splitlines():
      parts = line.split()
      if len(parts) < 4:
        continue
      _, size_hex, _, name = parts[:4]
      if name in ("bank_to_dram_offset", "bank_to_l1_offset"):
        sizes[name] = int(size_hex, 16)
    defs: dict[str, int] = {}
    if sz := sizes.get("bank_to_dram_offset"):
      defs["NUM_DRAM_BANKS"] = sz // 4
    if sz := sizes.get("bank_to_l1_offset"):
      defs["NUM_L1_BANKS"] = sz // 4
    if defs:
      defs.setdefault("IS_NOT_POW2_NUM_DRAM_BANKS", 1)
      defs.setdefault("IS_NOT_POW2_NUM_L1_BANKS", 1)
    return defs

  def compile_kernel(
    self,
    kern: str,
    processor: Processor,
    *,
    dispatch_message_addr: int = 0,
    keep: bool | None = None,
    noc_index: int | None = None,
  ) -> CompiledKernel:
    cfg = _PROC_CONFIG[processor]
    target = cfg["target"]
    build_dir = Path(tempfile.mkdtemp(prefix=f"tt-kern-{target}-", dir="/tmp"))
    keep = keep or False
    try:
      fw = self._make_fw_symbols_elf(build_dir, self.firmware_dir / f"{target}.elf")
      (build_dir / "kernel_includes.hpp").write_text(kern)
      self._write_chlkc_descriptors(build_dir)

      obj, elf = f"{target}.o", build_dir / f"{target}.elf"
      debug = ["-g"] if self.debug_info else []
      opt = cfg["opt"]
      mcpu = list(cfg["mcpu"])

      if cfg["is_trisc"]:
        self._write_ckernel_trisc_genfiles(build_dir)
        src = _FW_SRC / "trisck.cc"
        defines = [
          *self._common_defines(dispatch_message_addr),
          f"-DCOMPILE_FOR_TRISC={cfg['trisc_id']}",
          f"-DPROCESSOR_INDEX={cfg['proc_idx']}",
          *cfg["stage_defines"],
          "-DKERNEL_BUILD",
        ]
        link_objs: list[str] = []
      else:
        src = _FW_SRC / f"{target}k.cc"
        ct_args = ",".join(str(x) for x in self.ckernel.compile_time_args)
        defines = [
          *self._common_defines(
            dispatch_message_addr, noc_index=noc_index or 0, noc_mode=0
          ),
          f"-DCOMPILE_FOR_{target.upper()}",
          f"-DPROCESSOR_INDEX={cfg['proc_idx']}",
          "-DKERNEL_BUILD",
          f"-DKERNEL_COMPILE_TIME_ARGS={ct_args}",
        ]
        link_objs = (
          [str(_HW / "lib" / "blackhole" / "noc.o")]
          if processor == Processor.BRISC
          else []
        )

      self._run_compile(
        build_dir,
        src,
        obj,
        opt=opt,
        cflags=[*self.COMMON_CFLAGS, *debug, *mcpu],
        defines=defines,
        includes=self._include_flags(include_ckernel=True),
      )

      ld = _HW / "toolchain" / "blackhole" / f"kernel_{target}.ld"
      self._run_link(
        build_dir,
        elf,
        opt=opt,
        lflags=[
          *self.COMMON_CFLAGS,
          *debug,
          *self.COMMON_LFLAGS,
          *mcpu,
          f"-T{ld}",
          "-Wl,--emit-relocs",
          f"-Wl,--just-symbols={fw}",
        ],
        objs=[obj, *link_objs, str(_SUBSTITUTES)],
      )
      xip, xip_text_bytes = pack_xip_elf(elf)
      return CompiledKernel(
        processor=processor,
        elf=elf.read_bytes(),
        xip=xip,
        xip_text_bytes=xip_text_bytes,
      )
    finally:
      if not keep:
        shutil.rmtree(build_dir, ignore_errors=True)

  def _common_defines(
    self,
    dispatch_message_addr: int,
    *,
    noc_index: int | None = None,
    noc_mode: int | None = None,
  ) -> list[str]:
    defaults: dict[str, int] = {
      "NUM_DRAM_BANKS": 7,
      "IS_NOT_POW2_NUM_DRAM_BANKS": 1,
      "NUM_L1_BANKS": 110,
      "IS_NOT_POW2_NUM_L1_BANKS": 1,
      "PCIE_NOC_X": 0,
      "PCIE_NOC_Y": 3,
    }
    defs = {**defaults, **self.device_defines}
    out = [
      "-DTENSIX_FIRMWARE",
      "-DLOCAL_MEM_EN=0",
      "-DARCH_BLACKHOLE",
      f"-DDISPATCH_MESSAGE_ADDR={dispatch_message_addr}",
    ]
    out += [f"-D{k}={v}" for k, v in defs.items()]
    if noc_index is not None:
      out.append(f"-DNOC_INDEX={noc_index}")
    if noc_mode is not None:
      out.append(f"-DNOC_MODE={noc_mode}")
    return out

  def _include_flags(self, *, include_ckernel: bool) -> list[str]:
    rel = (
      _BASE_INCLUDES
      + (_CKERNEL_INCLUDES if include_ckernel else [])
      + ["runtime/sfpi/include"]
    )
    return ["-I."] + [f"-I{TT_HOME / p}" for p in rel]

  def _run_compile(
    self,
    cwd: Path,
    src: Path,
    obj: str,
    *,
    opt: str,
    cflags: list[str],
    defines: list[str],
    includes: list[str],
  ):
    cmd = [
      str(self.compiler),
      opt,
      *cflags,
      *includes,
      "-c",
      "-o",
      obj,
      str(src),
      *defines,
    ]
    subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)

  def _run_link(
    self, cwd: Path, out: Path, *, opt: str, lflags: list[str], objs: list[str]
  ):
    cmd = [str(self.compiler), opt, *lflags, *objs, "-o", str(out)]
    subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)

  def _make_fw_symbols_elf(self, build_dir: Path, fw: Path) -> Path:
    # Weaken data symbols + localize code symbols so kernels don't collide with firmware.
    out = build_dir / (fw.stem + ".weakened.elf")
    out.write_bytes(fw.read_bytes())
    proc = subprocess.run(
      [str(self.nm), "-a", str(out)],
      cwd=build_dir,
      check=True,
      text=True,
      capture_output=True,
    )
    data_types = {"B", "D", "R", "S", "G", "V"}
    localize, weaken = [], []
    for line in proc.stdout.splitlines():
      parts = line.split()
      if len(parts) < 3:
        continue
      _, sym_type, name = parts[:3]
      if sym_type == "U" or not sym_type.isupper():
        continue
      if name == "__global_pointer$" or name.startswith("__fw_export_"):
        continue
      if name in ("_start", "main", "exit") or sym_type not in data_types:
        localize.append(name)
      else:
        weaken.append(name)
    for path, syms in [
      (build_dir / "fw.localize.txt", localize),
      (build_dir / "fw.weaken.txt", weaken),
    ]:
      path.write_text("\n".join(sorted(set(syms))) + "\n")
    subprocess.run(
      [
        str(self.objcopy),
        f"--localize-symbols={build_dir / 'fw.localize.txt'}",
        f"--weaken-symbols={build_dir / 'fw.weaken.txt'}",
        str(out),
      ],
      cwd=build_dir,
      check=True,
      text=True,
      capture_output=True,
    )
    return out

  def _default_firmware_dir(self, arch: str) -> Path:
    base = Path(__file__).resolve().parent / "riscv-firmware"
    for candidate in [arch, "p100a"]:
      if (base / candidate).is_dir():
        return base / candidate
    return base / arch

  def _write_ckernel_trisc_genfiles(self, build_dir: Path):
    (build_dir / "defines_generated.h").write_text("")
    for stage, define in [
      ("unpack", "TRISC_UNPACK"),
      ("math", "TRISC_MATH"),
      ("pack", "TRISC_PACK"),
    ]:
      (build_dir / f"chlkc_{stage}.cpp").write_text(
        f'#define {define}\n#include "defines_generated.h"\n#include <kernel_includes.hpp>\n'
      )

  def _write_chlkc_descriptors(self, build_dir: Path):
    cfg = self.ckernel
    arr32 = lambda v: [v] * 32
    formats = arr32(cfg.data_format) if cfg.cb_data_formats is None else [255] * 32
    if cfg.cb_data_formats:
      for cb, fmt in cfg.cb_data_formats.items():
        formats[cb] = fmt
    tile_sizes = [
      self._tile_size_bytes(f, r=cfg.tile_r_dim, c=cfg.tile_c_dim)
      if f != 255
      else cfg.inactive_tile_size_bytes
      for f in formats
    ]
    fmt32 = ", ".join(str(x) for x in formats) + ","
    tile_sz32 = ", ".join(str(x) for x in tile_sizes) + ","
    dims = {
      "tile_num_faces": ", ".join(str(x) for x in arr32(cfg.tile_num_faces)) + ",",
      "partial_face": ", ".join(str(x) for x in arr32(cfg.partial_face)) + ",",
      "tile_face_r_dim": ", ".join(str(x) for x in arr32(cfg.tile_face_r_dim)) + ",",
      "narrow_tile": ", ".join(str(x) for x in arr32(cfg.narrow_tile)) + ",",
      "tile_r_dim": ", ".join(str(x) for x in arr32(cfg.tile_r_dim)) + ",",
      "tile_c_dim": ", ".join(str(x) for x in arr32(cfg.tile_c_dim)) + ",",
    }
    (build_dir / "chlkc_unpack_data_format.h").write_text(
      "#pragma once\n\n"
      f"constexpr std::int32_t unpack_src_format[32] = {{{fmt32}}};\n"
      f"constexpr std::int32_t unpack_dst_format[32] = {{{fmt32}}};\n"
    )
    (build_dir / "chlkc_pack_data_format.h").write_text(
      "#pragma once\n\n"
      f"constexpr unsigned char pack_src_format[32] = {{{fmt32}}};\n"
      f"constexpr unsigned char pack_dst_format[32] = {{{fmt32}}};\n"
    )
    for prefix in ("unpack", "pack"):
      (build_dir / f"chlkc_{prefix}_tile_dims.h").write_text(
        "#pragma once\n\n"
        + "".join(
          f"constexpr uint8_t {prefix}_{k}[32] = {{\n    {v}\n}};\n"
          for k, v in dims.items()
        )
        + f"constexpr uint16_t {prefix}_tile_size[32] = {{\n    {tile_sz32}\n}};\n"
      )
    (build_dir / "chlkc_dst_accum_mode.h").write_text(
      f"constexpr bool DST_ACCUM_MODE = {str(cfg.dst_accum_mode).lower()};\n"
    )
    (build_dir / "chlkc_dst_sync_mode.h").write_text(
      f"#define DST_SYNC_MODE DstSync::{'SyncFull' if cfg.dst_full_sync else 'SyncHalf'}\n"
    )
    (build_dir / "chlkc_math_fidelity.h").write_text(
      f"constexpr std::int32_t MATH_FIDELITY = {cfg.math_fidelity};\n"
    )
    (build_dir / "chlkc_math_approx_mode.h").write_text(
      f"constexpr bool APPROX = {str(cfg.approx).lower()};\n"
    )

  def _tile_size_bytes(self, data_format: int, *, r: int, c: int) -> int:
    match data_format:
      case 0:
        return r * c * 4  # Float32
      case 1 | 5:
        return r * c * 2  # Float16 / Float16_b
      case 2 | 6 | 10 | 26:
        return r * c + 64  # Bfp8 / Bfp8_b / Lf8 / Fp8_e4m3
      case 3 | 7:
        return (r * c // 2) + 64  # Bfp4 / Bfp4_b
      case 11 | 15:
        return (r * c // 4) + 64  # Bfp2 / Bfp2_b
      case _:
        raise ValueError(f"unsupported data format: {data_format}")
