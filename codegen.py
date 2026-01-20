from __future__ import annotations
from helpers import TT_HOME, DEBUG, pack_xip_elf
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import shutil
import subprocess
import tempfile

"""
`pure-py` kernel compilation overview (Blackhole)

Goal: take a C++ kernel source string (like TT-metal examples embed in host code), compile it for a specific
processor (BRISC/NCRISC/TRISC0/1/2), and return an ELF + packed bytes for upload.

The key TT-metal concept: kernels are not freestanding programs.
They are *linked against firmware symbols* and often expect a set of *JIT-generated headers* that describe
tile formats/dims and other compile-time properties.

This file implements the smallest subset needed to:
- compile dataflow kernels (BRISC/NCRISC) that use `dataflow_api.h` (CB ops, noc_async_*, TensorAccessor, etc.)
- compile compute kernels (TRISC0/1/2) that use `compute_kernel_api/*` / `ckernel` orchestration

It does NOT implement TT-metal's full JIT (no per-CB specialization, no multi-kernel bundles, etc.).
Instead, it generates a small fixed set of "chlkc_*" descriptor headers and a minimal TRISC wrapper layout.
"""

class Processor(Enum):
  NCRISC = auto()
  BRISC = auto()
  TRISC0 = auto()
  TRISC1 = auto()
  TRISC2 = auto()

@dataclass(frozen=True)
class CkernelConfig:
  # TT-metal's compute stack (ckernel/LLK + compute_kernel_api) expects a pile of generated headers:
  #   chlkc_*_data_format.h, chlkc_*_tile_dims.h, chlkc_math_fidelity.h, etc.
  #
  # In TT-metal, those are derived from the host-side CB config + compute config, and are *compile-time*.
  # In pure-py we currently emit a single fixed descriptor set (configurable via this dataclass).
  #
  # IMPORTANT: your runtime CB config must match these descriptors or behavior is undefined.
  # Example: `get_tile_size(cb)` comes from `unpack_tile_size[cb]` generated here, not from runtime args.
  compile_time_args: tuple[int, ...] = (2,)  # TensorAccessorArgs<0> => IsDram
  data_format: int = 5  # DataFormat::Float16_b
  cb_data_formats: dict[int, int] | None = None  # {cb_index: DataFormat}, others become invalid (255)
  inactive_tile_size_bytes: int = 1088  # TT-metal default for unused CBs in generated headers
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

# Paths relative to TT_HOME for include flags
_BASE_INCLUDES = [
  "tt_metal", "tt_metal/hw/inc", "tt_metal/hostdevcommon/api", "tt_metal/api", "tt_metal/include",
  "tt_metal/hw/inc/internal/tt-1xx", "tt_metal/hw/inc/internal/tt-1xx/blackhole",
  "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc", "tt_metal/hw/ckernels/blackhole/metal/llk_io",
]
_CKERNEL_INCLUDES = [
  "tt_metal/hw/ckernels/blackhole/metal/common", "tt_metal/hw/ckernels/blackhole/metal/llk_api",
  "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu", "tt_metal/hw/ckernels/blackhole/metal/llk_io",
  "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc", "tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
]

class Compiler:
  # These flags are copied from TT-metal's JIT build (Blackhole). Keep them boring:
  # - `-flto` is important because TT-metal compiles kernels with LTO enabled.
  # - `-nostartfiles` because our entry point / layout is controlled by linker scripts.
  COMMON_CFLAGS = (
    "-std=c++17", "-flto=auto", "-ffast-math", "-fno-exceptions",
    "-MMD", "-fno-use-cxa-atexit", "-Wall", "-Werror", "-Wno-unknown-pragmas",
    "-Wno-deprecated-declarations", "-Wno-error=multistatement-macros", "-Wno-error=parentheses",
    "-Wno-error=unused-but-set-variable", "-Wno-unused-variable", "-Wno-unused-function",
  )
  COMMON_LFLAGS = ("-Wl,-z,max-page-size=16", "-Wl,-z,common-page-size=16", "-nostartfiles")

  def __init__(
    self,
    *,
    arch: str = "p100a",
    firmware_dir: Path | None = None,
    ckernel: CkernelConfig = CkernelConfig(),
    dm_wrapper: str = "minimal",
    debug_info: bool = False,
    device_defines: dict[str, int] | None = None,
  ):
    self.compiler = TT_HOME/"runtime"/"sfpi"/"compiler"/"bin"/"riscv-tt-elf-g++"
    self.objcopy = TT_HOME/"runtime"/"sfpi"/"compiler"/"bin"/"riscv-tt-elf-objcopy"
    self.nm = TT_HOME/"runtime"/"sfpi"/"compiler"/"bin"/"riscv-tt-elf-nm"
    assert self.compiler.is_file(), f"missing SFPI compiler: {self.compiler}"
    assert self.objcopy.is_file(), f"missing SFPI objcopy: {self.objcopy}"
    self.firmware_dir = firmware_dir or self._default_firmware_dir(arch)
    assert self.firmware_dir.is_dir(), f"missing firmware dir: {self.firmware_dir}"
    self.ckernel = ckernel
    self.dm_wrapper = dm_wrapper
    self.debug_info = debug_info
    inferred = self._infer_device_defines_from_firmware(self.firmware_dir/"brisc.elf")
    self.device_defines = {**inferred, **(device_defines or {})}

  def _infer_device_defines_from_firmware(self, fw: Path) -> dict[str, int]:
    if not (self.nm.is_file() and fw.is_file()): return {}
    try:
      proc = subprocess.run([str(self.nm), "--print-size", str(fw)], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
      return {}
    sizes: dict[str, int] = {}
    for line in proc.stdout.splitlines():
      # <addr> <size> <type> <name>
      parts = line.split()
      if len(parts) < 4: continue
      _, size_hex, _, name = parts[:4]
      if name in ("bank_to_dram_offset", "bank_to_l1_offset"):
        sizes[name] = int(size_hex, 16)
    defs: dict[str, int] = {}
    if (sz := sizes.get("bank_to_dram_offset")): defs["NUM_DRAM_BANKS"] = sz // 4
    if (sz := sizes.get("bank_to_l1_offset")): defs["NUM_L1_BANKS"] = sz // 4
    if defs:
      defs.setdefault("IS_NOT_POW2_NUM_DRAM_BANKS", 1)
      defs.setdefault("IS_NOT_POW2_NUM_L1_BANKS", 1)
    return defs

  def compile_kernel(self, kern: str, processor: Processor, *, dispatch_message_addr: int = 0, keep: bool | None = None) -> CompiledKernel:
    # Each processor is compiled separately into its own ELF:
    # - BRISC: typically writer-side dataflow
    # - NCRISC: typically reader-side dataflow
    # - TRISC0/1/2: compute pipeline stages (unpack/math/pack)
    #
    # "dispatch_message_addr" is part of the mailbox protocol; pure-py uses slow-dispatch and currently
    # doesn't rely on device-side dispatch notifications, but the define still exists in TT-metal builds.
    if processor in (Processor.BRISC, Processor.NCRISC):
      return self._compile_dm(kern, processor, dispatch_message_addr=dispatch_message_addr, keep=keep)
    elif processor in (Processor.TRISC0, Processor.TRISC1, Processor.TRISC2):
      return self._compile_trisc(kern, processor, dispatch_message_addr=dispatch_message_addr, keep=keep)
    raise ValueError(f"unsupported processor: {processor}")

  def _compile_dm(self, kern: str, processor: Processor, *, dispatch_message_addr: int, keep: bool | None) -> CompiledKernel:
    # Data-movement RISCs (BRISC/NCRISC) run C++ kernels that call into `dataflow_api.h` (CB ops, noc ops, etc).
    # We compile a tiny entry translation unit that:
    # - waits for RUN_MSG_GO via the mailbox
    # - calls `kernel_main()`
    #
    # NOTE: it does *not* write RUN_MSG_DONE. BRISC firmware owns completion.
    is_brisc = processor == Processor.BRISC
    target = "brisc" if is_brisc else "ncrisc"
    mcpu = ("-mcpu=tt-bh", "-mno-tt-tensix-optimize-replay", "-fno-tree-loop-distribute-patterns")
    link_objs = [TT_HOME/"runtime"/"hw"/"lib"/"blackhole"/"noc.o"] if is_brisc else []

    build_dir = Path(tempfile.mkdtemp(prefix=f"tt-kern-{target}-", dir="/tmp"))
    keep = DEBUG >= 3 if keep is None else keep
    try:
      fw = self._make_fw_symbols_elf(build_dir, self.firmware_dir/f"{target}.elf")
      (build_dir/"kernel_includes.hpp").write_text(kern)
      self._write_chlkc_descriptors(build_dir)
      if self.dm_wrapper == "minimal":
        (build_dir/"dm_entry.cc").write_text(
          '#include <cstdint>\n'
          '#include "internal/risc_attribs.h"\n'
          '#include "internal/tt-1xx/risc_common.h"\n'
          '#include "hostdevcommon/kernel_structs.h"\n'
          '#include "api/dataflow/dataflow_api.h"\n'
          '#include <kernel_includes.hpp>\n\n'
          'extern "C" [[gnu::section(".start")]] uint32_t _start() {\n'
          '  asm("0: .reloc 0b, R_RISCV_NONE, __global_pointer$");\n'
          '  volatile tt_l1_ptr uint32_t* const go_message_index_ptr = GET_MAILBOX_ADDRESS_DEV(go_message_index);\n'
          '  volatile tt_l1_ptr go_msg_t* const go_messages = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);\n'
          '  const uint32_t go_message_index = *go_message_index_ptr;\n'
          '  while (go_messages[go_message_index].signal != RUN_MSG_GO) invalidate_l1_cache();\n'
          '  kernel_main();\n'
          '  return 0;\n'
          '}\n'
        )
      obj, elf = f"{target}.o", build_dir/f"{target}.elf"
      ct_args = ",".join(str(x) for x in self.ckernel.compile_time_args)
      # `KERNEL_BUILD` matters: some TT-metal headers change ABI/vars between KERNEL_BUILD and FW_BUILD.
      # In particular `dataflow_api_common.h` defines `noc_index` as a constexpr when KERNEL_BUILD is set,
      # which avoids undefined symbol link failures.
      #
      # `KERNEL_COMPILE_TIME_ARGS` seeds the compile-time args array; TensorAccessorArgs reads bitflags from it.
      debug = ["-g"] if self.debug_info else []
      src = build_dir/"dm_entry.cc" if self.dm_wrapper == "minimal" else TT_HOME/"tt_metal"/"hw"/"firmware"/"src"/"tt-1xx"/f"{target}k.cc"
      opt = "-O2" if self.dm_wrapper == "tt-metal" else "-Os"
      noc_index = 0 if is_brisc else 1
      self._run_compile(
        build_dir, src, obj, opt=opt,
        cflags=[*self.COMMON_CFLAGS, *debug, *mcpu],
        defines=[*self._common_defines(dispatch_message_addr, noc_index=noc_index, noc_mode=0), f"-DCOMPILE_FOR_{target.upper()}",
                 f"-DPROCESSOR_INDEX={0 if is_brisc else 1}", "-DKERNEL_BUILD", f"-DKERNEL_COMPILE_TIME_ARGS={ct_args}"],
        includes=self._include_flags(include_ckernel=self.dm_wrapper != "minimal"),
      )
      # `--just-symbols=<firmware>` is the critical trick:
      # the kernel references firmware-defined globals (mailboxes, rta_l1_base, coords, etc.)
      # and we want those references to resolve to the addresses used by the firmware we already uploaded.
      ld = TT_HOME/"runtime"/"hw"/"toolchain"/"blackhole"/f"kernel_{target}.ld"
      self._run_link(
        build_dir, elf, opt=opt,
        lflags=[*self.COMMON_CFLAGS, *debug, *self.COMMON_LFLAGS, *mcpu, f"-T{ld}", "-Wl,--emit-relocs", f"-Wl,--just-symbols={fw}"],
        objs=[obj, *(str(p) for p in [*link_objs, TT_HOME/"runtime"/"hw"/"lib"/"blackhole"/"substitutes.o"])],
      )
      xip, xip_text_bytes = pack_xip_elf(elf)
      return CompiledKernel(processor=processor, elf=elf.read_bytes(), xip=xip, xip_text_bytes=xip_text_bytes)
    finally:
      if not keep: shutil.rmtree(build_dir, ignore_errors=True)

  def _compile_trisc(self, kern: str, processor: Processor, *, dispatch_message_addr: int, keep: bool | None) -> CompiledKernel:
    # TRISC compute kernels are built through TT-metal's `trisck.cc` wrapper (not firmware).
    # That wrapper includes `chlkc_list.h`, which expects:
    # - generated `chlkc_unpack.cpp` / `chlkc_math.cpp` / `chlkc_pack.cpp`
    # - generated descriptor headers (data formats, tile dims, etc.)
    #
    # TT-metal normally generates those from host-side kernel config. We generate a minimal set here.
    trisc_id = {Processor.TRISC0: 0, Processor.TRISC1: 1, Processor.TRISC2: 2}[processor]
    mcpu = ("-mcpu=tt-bh-tensix", "-mno-tt-tensix-optimize-replay")

    build_dir = Path(tempfile.mkdtemp(prefix=f"tt-kern-trisc{trisc_id}-", dir="/tmp"))
    keep = DEBUG >= 3 if keep is None else keep
    try:
      fw = self._make_fw_symbols_elf(build_dir, self.firmware_dir/f"trisc{trisc_id}.elf")
      (build_dir/"kernel_includes.hpp").write_text(kern)
      self._write_chlkc_descriptors(build_dir)
      self._write_ckernel_trisc_genfiles(build_dir)

      obj, elf = f"trisc{trisc_id}.o", build_dir/f"trisc{trisc_id}.elf"
      # These defines select which pipeline stage gets built into this ELF:
      # - TRISC0: unpack
      # - TRISC1: math
      # - TRISC2: pack
      stage_defines = [("-DUCK_CHLKC_UNPACK", "-DNAMESPACE=chlkc_unpack"),
                       ("-DUCK_CHLKC_MATH", "-DNAMESPACE=chlkc_math"),
                       ("-DUCK_CHLKC_PACK", "-DNAMESPACE=chlkc_pack")][trisc_id]
      debug = ["-g"] if self.debug_info else []
      self._run_compile(
        build_dir, TT_HOME/"tt_metal"/"hw"/"firmware"/"src"/"tt-1xx"/"trisck.cc", obj, opt="-O3",
        cflags=[*self.COMMON_CFLAGS, *debug, *mcpu],
        defines=[*self._common_defines(dispatch_message_addr), f"-DCOMPILE_FOR_TRISC={trisc_id}",
                 f"-DPROCESSOR_INDEX={trisc_id + 2}", *stage_defines, "-DKERNEL_BUILD"],
        includes=self._include_flags(include_ckernel=True),
      )
      ld = TT_HOME/"runtime"/"hw"/"toolchain"/"blackhole"/f"kernel_trisc{trisc_id}.ld"
      self._run_link(
        build_dir, elf, opt="-O3",
        lflags=[*self.COMMON_CFLAGS, *debug, *self.COMMON_LFLAGS, *mcpu, f"-T{ld}", "-Wl,--emit-relocs", f"-Wl,--just-symbols={fw}"],
        objs=[obj, str(TT_HOME/"runtime"/"hw"/"lib"/"blackhole"/"substitutes.o")],
      )
      xip, xip_text_bytes = pack_xip_elf(elf)
      return CompiledKernel(processor=processor, elf=elf.read_bytes(), xip=xip, xip_text_bytes=xip_text_bytes)
    finally:
      if not keep: shutil.rmtree(build_dir, ignore_errors=True)

  def _common_defines(self, dispatch_message_addr: int, *, noc_index: int | None = None, noc_mode: int | None = None) -> list[str]:
    defaults: dict[str, int] = {
      # TT-metal provides these as device-specific defines derived from harvesting/topology.
      # Keep them overrideable; sane defaults should compile even if not accurate for your board.
      "NUM_DRAM_BANKS": 7,  # p100a: exactly 1 DRAM bank harvested
      "IS_NOT_POW2_NUM_DRAM_BANKS": 1,  # TT-metal sets this; LOG_BASE_2_OF_NUM_DRAM_BANKS isn't always defined.
      "NUM_L1_BANKS": 110,  # p100a: device-provided TT-metal define
      "IS_NOT_POW2_NUM_L1_BANKS": 1,
      "PCIE_NOC_X": 0, "PCIE_NOC_Y": 3,
    }
    defs = {**defaults, **self.device_defines}
    out = ["-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE", f"-DDISPATCH_MESSAGE_ADDR={dispatch_message_addr}"]
    out += [f"-D{k}={v}" for k, v in defs.items()]
    if noc_index is not None: out.append(f"-DNOC_INDEX={noc_index}")
    if noc_mode is not None: out.append(f"-DNOC_MODE={noc_mode}")
    return out

  def _include_flags(self, *, include_ckernel: bool) -> list[str]:
    # These include paths match how TT-metal's build sees its own headers.
    # When `include_ckernel=True` we also add the LLK/ckernel include roots needed by compute kernels.
    rel = _BASE_INCLUDES + (_CKERNEL_INCLUDES if include_ckernel else []) + ["runtime/sfpi/include"]
    return ["-I."] + [f"-I{TT_HOME/p}" for p in rel]

  def _run_compile(self, cwd: Path, src: Path, obj: str, *, opt: str, cflags: list[str], defines: list[str], includes: list[str]):
    cmd = [str(self.compiler), opt, *cflags, *includes, "-c", "-o", obj, str(src), *defines]
    if DEBUG >= 2: print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=DEBUG < 2)

  def _run_link(self, cwd: Path, out: Path, *, opt: str, lflags: list[str], objs: list[str]):
    cmd = [str(self.compiler), opt, *lflags, *objs, "-o", str(out)]
    if DEBUG >= 2: print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=DEBUG < 2)

  def _make_fw_symbols_elf(self, build_dir: Path, fw: Path) -> Path:
    # Match TT-metal: link kernels against a "weakened" firmware ELF.
    #
    # TT-metal weakens global data symbols so kernels can define their own storage without
    # accidentally binding to firmware globals (notably the NCRISC NOC counter arrays).
    # It also localizes non-data globals to reduce collisions, while keeping `__fw_export_*`
    # symbols strong so kernels can import fixed-address exports.
    out = build_dir/(fw.stem + ".weakened.elf")
    out.write_bytes(fw.read_bytes())

    proc = subprocess.run([str(self.nm), "-a", str(out)], cwd=build_dir, check=True, text=True, capture_output=True)
    data_types = {"B", "D", "R", "S", "G", "V"}
    localize, weaken = [], []
    for line in proc.stdout.splitlines():
      parts = line.split()
      if len(parts) < 3: continue
      _, sym_type, name = parts[:3]
      if sym_type == "U" or (not sym_type.isupper()): continue
      if name == "__global_pointer$" or name.startswith("__fw_export_"): continue
      if name in ("_start", "main", "exit"):
        localize.append(name)
      elif sym_type in data_types:
        weaken.append(name)
      else:
        localize.append(name)

    localize_path, weaken_path = build_dir/"fw.localize.txt", build_dir/"fw.weaken.txt"
    localize_path.write_text("\n".join(sorted(set(localize))) + "\n")
    weaken_path.write_text("\n".join(sorted(set(weaken))) + "\n")

    subprocess.run(
      [str(self.objcopy), f"--localize-symbols={localize_path}", f"--weaken-symbols={weaken_path}", str(out)],
      cwd=build_dir, check=True, text=True, capture_output=DEBUG < 2,
    )
    return out

  def _default_firmware_dir(self, arch: str) -> Path:
    base = Path(__file__).resolve().parent/"riscv-firmware"
    for candidate in [arch, "p100a"]:
      if (base/candidate).is_dir(): return base/candidate
    return base/arch

  def _write_ckernel_trisc_genfiles(self, build_dir: Path):
    # TT-metal JIT generates `chlkc_{unpack,math,pack}.cpp` which include the same user kernel source,
    # but with different stage defines so `compute_kernel_api/common_globals.h` maps MAIN to the right symbol.
    #
    # `trisck.cc` then calls `run_kernel()`, which dispatches to `{unpack,math,pack}_main()` based on stage.
    (build_dir/"defines_generated.h").write_text("")
    for stage, define in [("unpack", "TRISC_UNPACK"), ("math", "TRISC_MATH"), ("pack", "TRISC_PACK")]:
      (build_dir/f"chlkc_{stage}.cpp").write_text(f'#define {define}\n#include "defines_generated.h"\n#include <kernel_includes.hpp>\n')

  def _write_chlkc_descriptors(self, build_dir: Path):
    # These headers satisfy the expectations of:
    # - `dataflow_api.h` (for `get_tile_size()` / data formats)
    # - `compute_kernel_api/*` + LLK (for pack/unpack configuration)
    #
    cfg = self.ckernel
    arr32 = lambda v: [v] * 32
    formats = arr32(cfg.data_format) if cfg.cb_data_formats is None else [255] * 32
    if cfg.cb_data_formats:
      for cb, fmt in cfg.cb_data_formats.items(): formats[cb] = fmt

    def tile_size_bytes(fmt: int) -> int:
      return self._tile_size_bytes(fmt, r=cfg.tile_r_dim, c=cfg.tile_c_dim)

    tile_sizes = [tile_size_bytes(f) if f != 255 else cfg.inactive_tile_size_bytes for f in formats]

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

    (build_dir/"chlkc_unpack_data_format.h").write_text(
      "#pragma once\n\n"
      f"constexpr std::int32_t unpack_src_format[32] = {{{fmt32}}};\n"
      f"constexpr std::int32_t unpack_dst_format[32] = {{{fmt32}}};\n"
    )
    (build_dir/"chlkc_pack_data_format.h").write_text(
      "#pragma once\n\n"
      f"constexpr unsigned char pack_src_format[32] = {{{fmt32}}};\n"
      f"constexpr unsigned char pack_dst_format[32] = {{{fmt32}}};\n"
    )
    for prefix in ("unpack", "pack"):
      (build_dir/f"chlkc_{prefix}_tile_dims.h").write_text(
        "#pragma once\n\n"
        + "".join(f"constexpr uint8_t {prefix}_{k}[32] = {{\n    {v}\n}};\n" for k, v in dims.items())
        + f"constexpr uint16_t {prefix}_tile_size[32] = {{\n    {tile_sz32}\n}};\n"
      )
    (build_dir/"chlkc_dst_accum_mode.h").write_text(f"constexpr bool DST_ACCUM_MODE = {str(cfg.dst_accum_mode).lower()};\n")
    (build_dir/"chlkc_dst_sync_mode.h").write_text(f"#define DST_SYNC_MODE DstSync::{'SyncFull' if cfg.dst_full_sync else 'SyncHalf'}\n")
    (build_dir/"chlkc_math_fidelity.h").write_text(f"constexpr std::int32_t MATH_FIDELITY = {cfg.math_fidelity};\n")
    (build_dir/"chlkc_math_approx_mode.h").write_text(f"constexpr bool APPROX = {str(cfg.approx).lower()};\n")

  def _tile_size_bytes(self, data_format: int, *, r: int, c: int) -> int:
    # Tile payload size in bytes (matches TT-metal's generated headers and `get_tile_size()` usage).
    match data_format:
      case 0: return r * c * 4  # Float32
      case 1 | 5: return r * c * 2  # Float16 / Float16_b
      case 2 | 6 | 10 | 26: return r * c + 64  # Bfp8 / Bfp8_b / Lf8 / Fp8_e4m3
      case 3 | 7: return (r * c // 2) + 64  # Bfp4 / Bfp4_b
      case 11 | 15: return (r * c // 4) + 64  # Bfp2 / Bfp2_b
      case _: raise ValueError(f"unsupported data format: {data_format}")
