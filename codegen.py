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
  "-fno-use-cxa-atexit",
  "-fno-jump-tables",  # XIP kernels: jump tables land in .rodata at wrong addresses
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
    "-DLOCAL_MEM_EN=0", "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", *_DEVICE_DEFINES,
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

  def _compile_dataflow(self, src: str, target: str, noc_index: int,
                        extra_defines: list[str] | None = None,
                        extra_includes: list[str] | None = None) -> CompiledKernel:
    defines = [
      "-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE",
      "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", "-DKERNEL_BUILD", *_DEVICE_DEFINES,
      f"-DCOMPILE_FOR_{target.upper()}",
      f"-DPROCESSOR_INDEX={0 if target == 'brisc' else 1}",
      f"-DNOC_INDEX={noc_index}", "-DNOC_MODE=0",
      *(extra_defines or []),
    ]
    extra_objs = [str(_DEPS / "lib/blackhole/noc.o")] if target == "brisc" else []
    return self._build(src, target, defines, extra_objs, opt="-O2", trisc=False,
                       extra_includes=extra_includes)

  def _compile_trisc(self, src: str, trisc_id: int) -> CompiledKernel:
    stage = ("unpack", "math", "pack")[trisc_id]
    defines = [
      "-DTENSIX_FIRMWARE", "-DLOCAL_MEM_EN=0", "-DARCH_BLACKHOLE",
      "-DDISPATCH_MESSAGE_ADDR=0xFFB70438", "-DKERNEL_BUILD", *_DEVICE_DEFINES,
      f"-DCOMPILE_FOR_TRISC={trisc_id}",
      f"-DPROCESSOR_INDEX={trisc_id + 2}",
      f"-DUCK_CHLKC_{stage.upper()}", f"-DNAMESPACE=chlkc_{stage}",
    ]
    return self._build(src, f"trisc{trisc_id}", defines, [], opt="-O3", trisc=True)

  def _build(self, kern: str, target: str, defines: list[str], extra_objs: list[str],
             opt: str, trisc: bool, extra_includes: list[str] | None = None) -> CompiledKernel:
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
      includes = [*self._includes, *(f"-I{p}" for p in (extra_includes or []))]
      _run(self._cc, [opt, *_CFLAGS, "-MMD", *mcpu, *includes, "-c", "-o", "out.o", str(fw_src), *defines], build)

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

# === CQ (command queue) kernel compiler ===

@dataclass(frozen=True)
class CQConfig:
  """Runtime parameters needed to compile CQ kernels. All addresses are L1 or NOC."""
  # Core coordinates (NOC physical)
  prefetch_xy: tuple[int, int]
  dispatch_xy: tuple[int, int]
  # PCIe hugepage layout
  pcie_base: int          # offset into hugepage where issue queue starts
  pcie_size: int          # size of issue queue region
  completion_base: int    # NOC address of completion queue region
  completion_size: int    # size of completion queue region
  # Device L1 layout (on prefetch core)
  prefetch_q_base: int
  prefetch_q_size: int
  prefetch_q_rd_ptr_addr: int
  prefetch_q_pcie_rd_ptr_addr: int
  cmddat_q_base: int
  cmddat_q_size: int
  scratch_db_base: int
  scratch_db_size: int
  # Dispatch circular buffer (on dispatch core)
  dispatch_cb_base: int
  dispatch_cb_pages: int
  dispatch_cb_log_page_size: int = 12  # 4KB pages
  dispatch_cb_blocks: int = 4
  # Dispatch subordinate buffer (on dispatch core)
  dispatch_s_buffer_base: int = 0
  dispatch_s_buffer_size: int = 32 * 1024
  dispatch_s_cb_log_page_size: int = 8  # 256B pages
  # Completion queue device pointers (on dispatch core)
  completion_q_wr_ptr_addr: int = 0
  completion_q_rd_ptr_addr: int = 0
  dispatch_s_sync_sem_addr: int = 0
  # PCIe host base address (NOC local offset of hugepage)
  command_queue_base: int = 0
  # Worker grid for go-signal multicast
  worker_grid_start: tuple[int, int] = (0, 0)
  worker_grid_end: tuple[int, int] = (0, 0)
  num_worker_cores: int = 0

@dataclass(frozen=True)
class CompiledCQKernels:
  prefetch_brisc: CompiledKernel     # cq_prefetch on prefetch core BRISC
  dispatch_brisc: CompiledKernel     # cq_dispatch on dispatch core BRISC
  dispatch_s_ncrisc: CompiledKernel  # cq_dispatch_subordinate on dispatch core NCRISC

def _noc_mcast_encoding(x0: int, y0: int, x1: int, y1: int) -> int:
  """Encode a multicast rectangle as a 32-bit NOC address. BH NOC encoding: (y << 6) | x."""
  start = (y0 << 6) | x0
  end = (y1 << 6) | x1
  return (start << 16) | end

def _noc_xy(x: int, y: int) -> int:
  return (y << 6) | x

def compile_cq_kernels(cfg: CQConfig) -> CompiledCQKernels:
  """Compile the 3 CQ kernels (prefetch, dispatch, dispatch_subordinate) with baked-in config."""
  compiler = Compiler()
  px, py = cfg.prefetch_xy
  dx, dy = cfg.dispatch_xy
  go_msg_addr = 0x370  # TensixL1.GO_MSG (MAILBOX_BASE + 0x310)

  # NOC multicast grid for worker go signals
  gx0, gy0 = cfg.worker_grid_start
  gx1, gy1 = cfg.worker_grid_end
  mcast_grid = _noc_mcast_encoding(gx0, gy0, gx1, gy1)

  # Shared fabric defines (unused in HD mode, but consumed unconditionally as constexpr)
  fabric_zeros = {
    "FABRIC_HEADER_RB_BASE": 0, "FABRIC_HEADER_RB_ENTRIES": 0,
    "MY_FABRIC_SYNC_STATUS_ADDR": 0, "FABRIC_MUX_X": 0, "FABRIC_MUX_Y": 0,
    "FABRIC_MUX_NUM_BUFFERS_PER_CHANNEL": 1, "FABRIC_MUX_CHANNEL_BUFFER_SIZE_BYTES": 1,
    "FABRIC_MUX_CHANNEL_BASE_ADDRESS": 0, "FABRIC_MUX_CONNECTION_INFO_ADDRESS": 0,
    "FABRIC_MUX_CONNECTION_HANDSHAKE_ADDRESS": 0, "FABRIC_MUX_FLOW_CONTROL_ADDRESS": 0,
    "FABRIC_MUX_BUFFER_INDEX_ADDRESS": 0, "FABRIC_MUX_STATUS_ADDRESS": 0,
    "FABRIC_MUX_TERMINATION_SIGNAL_ADDRESS": 0, "WORKER_CREDITS_STREAM_ID": 0,
    "FABRIC_WORKER_FLOW_CONTROL_SEM": 0, "FABRIC_WORKER_TEARDOWN_SEM": 0,
    "FABRIC_WORKER_BUFFER_INDEX_SEM": 0, "NUM_HOPS": 0, "EW_DIM": 0,
    "TO_MESH_ID": 0, "FABRIC_2D": 0,
  }

  # Semaphore IDs: prefetch sem 0 = downstream credits, sem 1 = dispatch_s credits, sem 2 = sync
  # Dispatch sem 0 = upstream CB pages available
  prefetch_defs = {
    "IS_H_VARIANT": 1, "IS_D_VARIANT": 1, "DISPATCH_KERNEL": 1, "FD_CORE_TYPE": 0,
    "MY_NOC_X": px, "MY_NOC_Y": py,
    "UPSTREAM_NOC_X": px, "UPSTREAM_NOC_Y": py,  # HD: no upstream, point to self
    "DOWNSTREAM_NOC_X": dx, "DOWNSTREAM_NOC_Y": dy,
    "DOWNSTREAM_SUBORDINATE_NOC_X": dx, "DOWNSTREAM_SUBORDINATE_NOC_Y": dy,
    "UPSTREAM_NOC_INDEX": 0,
    # Prefetch -> Dispatch CB flow control
    "DOWNSTREAM_CB_BASE": cfg.dispatch_cb_base,
    "DOWNSTREAM_CB_LOG_PAGE_SIZE": cfg.dispatch_cb_log_page_size,
    "DOWNSTREAM_CB_PAGES": cfg.dispatch_cb_pages,
    "MY_DOWNSTREAM_CB_SEM_ID": 0,
    "DOWNSTREAM_CB_SEM_ID": 0,
    # PCIe
    "PCIE_BASE": cfg.pcie_base, "PCIE_SIZE": cfg.pcie_size,
    # Prefetch queue
    "PREFETCH_Q_BASE": cfg.prefetch_q_base, "PREFETCH_Q_SIZE": cfg.prefetch_q_size,
    "PREFETCH_Q_RD_PTR_ADDR": cfg.prefetch_q_rd_ptr_addr,
    "PREFETCH_Q_PCIE_RD_PTR_ADDR": cfg.prefetch_q_pcie_rd_ptr_addr,
    # Command/data queue
    "CMDDAT_Q_BASE": cfg.cmddat_q_base, "CMDDAT_Q_SIZE": cfg.cmddat_q_size,
    "SCRATCH_DB_BASE": cfg.scratch_db_base, "SCRATCH_DB_SIZE": cfg.scratch_db_size,
    "DOWNSTREAM_SYNC_SEM_ID": 2,
    # D-variant prefetch_d buffer (unused in HD, but constexpr needs values)
    "CMDDAT_Q_PAGES": cfg.cmddat_q_size >> 12, "MY_UPSTREAM_CB_SEM_ID": 3,
    "UPSTREAM_CB_SEM_ID": 3, "CMDDAT_Q_LOG_PAGE_SIZE": 12, "CMDDAT_Q_BLOCKS": 4,
    # Dispatch subordinate buffer
    "DISPATCH_S_BUFFER_BASE": cfg.dispatch_s_buffer_base,
    "MY_DISPATCH_S_CB_SEM_ID": 1, "DOWNSTREAM_DISPATCH_S_CB_SEM_ID": 0,
    "DISPATCH_S_BUFFER_SIZE": cfg.dispatch_s_buffer_size,
    "DISPATCH_S_CB_LOG_PAGE_SIZE": cfg.dispatch_s_cb_log_page_size,
    # Ringbuffer (used by exec_buf, set to 0 for basic usage)
    "RINGBUFFER_SIZE": 0,
    # Runtime arg offsets
    "OFFSETOF_MY_DEV_ID": 0, "OFFSETOF_TO_DEV_ID": 1, "OFFSETOF_ROUTER_DIRECTION": 2,
    **fabric_zeros,
  }

  dispatch_defs = {
    "IS_H_VARIANT": 1, "IS_D_VARIANT": 1, "DISPATCH_KERNEL": 1, "FD_CORE_TYPE": 0,
    "MY_NOC_X": dx, "MY_NOC_Y": dy,
    "UPSTREAM_NOC_X": px, "UPSTREAM_NOC_Y": py,
    "DOWNSTREAM_NOC_X": dx, "DOWNSTREAM_NOC_Y": dy,  # HD: no downstream dispatch_h
    "DOWNSTREAM_SUBORDINATE_NOC_X": dx, "DOWNSTREAM_SUBORDINATE_NOC_Y": dy,
    "UPSTREAM_NOC_INDEX": 0,
    # Dispatch CB
    "DISPATCH_CB_BASE": cfg.dispatch_cb_base,
    "DISPATCH_CB_LOG_PAGE_SIZE": cfg.dispatch_cb_log_page_size,
    "DISPATCH_CB_PAGES": cfg.dispatch_cb_pages,
    "DISPATCH_CB_BLOCKS": cfg.dispatch_cb_blocks,
    "MY_DISPATCH_CB_SEM_ID": 0, "UPSTREAM_DISPATCH_CB_SEM_ID": 0,
    "UPSTREAM_SYNC_SEM": 2,
    # Completion queue
    "COMMAND_QUEUE_BASE_ADDR": cfg.command_queue_base,
    "COMPLETION_QUEUE_BASE_ADDR": cfg.completion_base,
    "COMPLETION_QUEUE_SIZE": cfg.completion_size,
    "HOST_COMPLETION_Q_WR_PTR": 2 * 64,  # HOST_COMPLETION_Q_WR_OFF
    "DEV_COMPLETION_Q_WR_PTR": cfg.completion_q_wr_ptr_addr,
    "DEV_COMPLETION_Q_RD_PTR": cfg.completion_q_rd_ptr_addr,
    # Downstream (unused in HD, but constexpr needs values)
    "DOWNSTREAM_CB_BASE": 0, "DOWNSTREAM_CB_SIZE": 0,
    "MY_DOWNSTREAM_CB_SEM_ID": 1, "DOWNSTREAM_CB_SEM_ID": 1,
    # Split dispatch (disabled)
    "SPLIT_DISPATCH_PAGE_PREAMBLE_SIZE": 0, "SPLIT_PREFETCH": 0,
    "PREFETCH_H_NOC_XY": 0, "PREFETCH_H_LOCAL_DOWNSTREAM_SEM_ADDR": 0,
    "PREFETCH_H_MAX_CREDITS": 0,
    # Worker dispatch
    "PACKED_WRITE_MAX_UNICAST_SUB_CMDS": cfg.num_worker_cores,
    "DISPATCH_S_SYNC_SEM_BASE_ADDR": cfg.dispatch_s_sync_sem_addr,
    "MAX_NUM_WORKER_SEMS": 8, "MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES": 256,
    "MCAST_GO_SIGNAL_ADDR": go_msg_addr, "UNICAST_GO_SIGNAL_ADDR": go_msg_addr,
    "DISTRIBUTED_DISPATCHER": 0, "FIRST_STREAM_USED": 48,
    "VIRTUALIZE_UNICAST_CORES": 0, "NUM_VIRTUAL_UNICAST_CORES": 0,
    "NUM_PHYSICAL_UNICAST_CORES": 0,
    "WORKER_MCAST_GRID": mcast_grid, "NUM_WORKER_CORES_TO_MCAST": cfg.num_worker_cores,
    # Runtime arg offsets
    "OFFSETOF_MY_DEV_ID": 0, "OFFSETOF_TO_DEV_ID": 1, "OFFSETOF_ROUTER_DIRECTION": 2,
    **fabric_zeros,
  }

  dispatch_s_defs = {
    "DISPATCH_KERNEL": 1, "FD_CORE_TYPE": 0,
    "MY_NOC_X": dx, "MY_NOC_Y": dy,
    "UPSTREAM_NOC_X": px, "UPSTREAM_NOC_Y": py,
    "DOWNSTREAM_NOC_X": dx, "DOWNSTREAM_NOC_Y": dy,
    "CB_BASE": cfg.dispatch_s_buffer_base,
    "CB_LOG_PAGE_SIZE": cfg.dispatch_s_cb_log_page_size,
    "CB_SIZE": cfg.dispatch_s_buffer_size,
    "MY_DISPATCH_CB_SEM_ID": 0, "UPSTREAM_DISPATCH_CB_SEM_ID": 1,
    "DISPATCH_S_SYNC_SEM_BASE_ADDR": cfg.dispatch_s_sync_sem_addr,
    "MCAST_GO_SIGNAL_ADDR": go_msg_addr, "UNICAST_GO_SIGNAL_ADDR": go_msg_addr,
    "DISTRIBUTED_DISPATCHER": 0, "FIRST_STREAM_USED": 48,
    "MAX_NUM_WORKER_SEMS": 8, "MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES": 256,
    "VIRTUALIZE_UNICAST_CORES": 0, "NUM_VIRTUAL_UNICAST_CORES": 0,
    "NUM_PHYSICAL_UNICAST_CORES": 0,
    "WORKER_MCAST_GRID": mcast_grid, "NUM_WORKER_CORES_TO_MCAST": cfg.num_worker_cores,
  }

  def _to_defs(d: dict) -> list[str]:
    return [f"-D{k}={v}" for k, v in d.items()]

  cq_src = _REPO / "firmware" / "cq"
  cq_inc = [str(cq_src)]
  prefetch_src = (cq_src / "cq_prefetch.cpp").read_text()
  dispatch_src = (cq_src / "cq_dispatch.cpp").read_text()
  dispatch_s_src = (cq_src / "cq_dispatch_subordinate.cpp").read_text()

  return CompiledCQKernels(
    prefetch_brisc=compiler._compile_dataflow(prefetch_src, "brisc", noc_index=0,
      extra_defines=_to_defs(prefetch_defs), extra_includes=cq_inc),
    dispatch_brisc=compiler._compile_dataflow(dispatch_src, "brisc", noc_index=1,
      extra_defines=_to_defs(dispatch_defs), extra_includes=cq_inc),
    dispatch_s_ncrisc=compiler._compile_dataflow(dispatch_s_src, "ncrisc", noc_index=1,
      extra_defines=_to_defs(dispatch_s_defs), extra_includes=cq_inc),
  )

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
