import ctypes, struct, time
from ctypes import c_uint8 as u8, c_uint16 as u16, c_uint32 as u32
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal

from hw import *

class DevMsgs:
  RUN_MSG_INIT = 0x40
  RUN_MSG_GO = 0x80
  RUN_MSG_RESET_READ_PTR_FROM_HOST = 0xE0
  RUN_MSG_DONE = 0x00
  DISPATCH_MODE_DEV = 0
  DISPATCH_MODE_HOST = 1
  ProgrammableCoreType_COUNT = 3
  MaxProcessorsPerCoreType = 5

class _RtaOffset(S):
  _pack_ = 1
  _fields_ = [("rta_offset", u16), ("crta_offset", u16)]

class _KernelConfigMsg(S):
  _pack_ = 1
  _fields_ = [
    ("kernel_config_base", u32 * DevMsgs.ProgrammableCoreType_COUNT),
    ("sem_offset", u16 * DevMsgs.ProgrammableCoreType_COUNT),
    ("local_cb_offset", u16),
    ("remote_cb_offset", u16),
    ("rta_offset", _RtaOffset * DevMsgs.MaxProcessorsPerCoreType),
    ("mode", u8),
    ("pad2", u8),
    ("kernel_text_offset", u32 * DevMsgs.MaxProcessorsPerCoreType),
    ("local_cb_mask", u32),
    ("brisc_noc_id", u8),
    ("brisc_noc_mode", u8),
    ("min_remote_cb_start_index", u8),
    ("exit_erisc_kernel", u8),
    ("host_assigned_id", u32),
    ("enables", u32),
    ("watcher_kernel_ids", u16 * DevMsgs.MaxProcessorsPerCoreType),
    ("ncrisc_kernel_size16", u16),
    ("sub_device_origin_x", u8),
    ("sub_device_origin_y", u8),
    ("pad3", u8 * 1),
    ("preload", u8),
  ]

class LaunchMsg(S):
  _pack_ = 1
  _fields_ = [("kernel_config", _KernelConfigMsg)]

class _GoMsgBits(S):
  _pack_ = 1
  _fields_ = [
    ("dispatch_message_offset", u8),
    ("master_x", u8),
    ("master_y", u8),
    ("signal", u8),
  ]

class GoMsg(ctypes.Union):
  _pack_ = 1
  _fields_ = [("all", u32), ("bits", _GoMsgBits)]

Rect = tuple[int, int, int, int]  # (x0, x1, y0, y1)
Args = list[int]
RtArgs = Args | Callable[[int, Core, int], Args]
CoreArgs = tuple[Args, Args, Args]  # (writer, reader, compute) per core

FAST_CQ_NUM_CIRCULAR_BUFFERS = 32

class Dtype(Enum):
  Float32 = 0
  Float16 = 1
  Float16_b = 5
  Int32 = 8
  UInt16 = 9
  Int8 = 14
  UInt32 = 24
  UInt8 = 30

  @property
  def bpe(self) -> int:
    return {0: 4, 1: 2, 5: 2, 8: 4, 9: 2, 14: 1, 24: 4, 30: 1}[self.value]

  @property
  def tile_size(self) -> int:
    return 32 * 32 * self.bpe

class MathFidelity(Enum):
  LoFi = 0
  HiFi2 = 2
  HiFi3 = 3
  HiFi4 = 4

@dataclass
class CBConfig:
  index: int
  dtype: Dtype
  tiles: int = 2

@dataclass
class Program:
  cores: int | Literal["all"]
  reader_kernel: str
  compute_kernel: str
  writer_kernel: str
  cbs: list[CBConfig]
  name: str = ""
  reader_args: RtArgs = field(default_factory=list)
  writer_args: RtArgs = field(default_factory=list)
  compute_args: RtArgs = field(default_factory=list)
  semaphores: int = 0
  math_fidelity: MathFidelity = MathFidelity.HiFi4
  approx: bool = False
  dst_accum_mode: bool = False
  dst_full_sync: bool = False
  reader_recv_kernel: str = ""
  writer_recv_kernel: str = ""
  grid: tuple[tuple[int, ...], tuple[int, ...]] | None = None
  profile: bool = True

def noc_mcast_xy(rect: Rect) -> tuple[int, int]:
  x0, x1, y0, y1 = rect
  return (y1 << 18) | (x1 << 12) | (y0 << 6) | x0, (x1 - x0 + 1) * (y1 - y0 + 1)

def mcast_rects(cores: list[Core]) -> list[Rect]:
  if not cores:
    return []
  remaining = set(cores)
  rects = []
  while remaining:
    x0, y0 = min(remaining, key=lambda c: (c[1], c[0]))
    x1 = x0
    while (x1 + 1, y0) in remaining:
      x1 += 1
    y1 = y0
    while all((x, y1 + 1) in remaining for x in range(x0, x1 + 1)):
      y1 += 1
    for x in range(x0, x1 + 1):
      for y in range(y0, y1 + 1):
        remaining.discard((x, y))
    rects.append((x0, x1, y0, y1))
  return rects

@dataclass
class Write:
  cores: list[Core]
  addr: int
  data: bytes | list[bytes]

@dataclass
class Launch:
  cores: list[Core]

IRCommand = Write | Launch

def resolve_args(args: RtArgs, core_idx: int, core_xy: Core, num_cores: int) -> Args:
  return args if isinstance(args, list) else args(core_idx, core_xy, num_cores)

def pack_rta(writer_args: Args, reader_args: Args, compute_args: Args, num_sems: int, sem_off: int) -> bytes:
  pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
  rta = pack(writer_args) + pack(reader_args) + pack(compute_args)
  if num_sems > 0:
    if sem_off > len(rta):
      rta = rta.ljust(sem_off, b"\0")
    rta += b"\0" * (num_sems * 16)
  return rta

def build_cb_blob(program: Program) -> tuple[int, bytes]:
  if not program.cbs:
    return 0, b""
  mask = 0
  for cb in program.cbs:
    mask |= 1 << cb.index
  end = mask.bit_length()
  arr = bytearray(end * 16)
  addr = TensixL1.DATA_BUFFER_SPACE_BASE
  shared_addr: dict[int, int] = {}
  for cb in program.cbs:
    page_size = cb.dtype.tile_size
    size = page_size * cb.tiles
    share_with = {16: 24, 24: 16}.get(cb.index)
    if share_with is not None and share_with in shared_addr:
      cb_addr = shared_addr[share_with]
    else:
      cb_addr = addr
      addr += size
    shared_addr[cb.index] = cb_addr
    struct.pack_into("<IIII", arr, cb.index * 16, cb_addr, size, cb.tiles, page_size)
  return mask, bytes(arr)

Role = tuple[list[Core], object, object]  # (cores, reader_kernel, writer_kernel)

def build_payload(
  program: Program, reader, writer, compute: tuple | None,
  rta_sizes: tuple[int, int, int], dispatch_mode: int,
  sem_off: int | None = None, host_assigned_id: int = 0,
) -> tuple[int, bytes, bytes]:
  rta_offsets = [0, rta_sizes[0], rta_sizes[0] + rta_sizes[1]]
  rta_total = align_up(rta_offsets[2] + rta_sizes[2], L1_ALIGN)
  if sem_off is None:
    sem_off = rta_total
  local_cb_off = align_up(sem_off + program.semaphores * 16, L1_ALIGN)
  cb_mask, cb_blob = build_cb_blob(program)
  kernel_off = align_up(local_cb_off + len(cb_blob), L1_ALIGN)
  proc = []
  if writer is not None:
    proc.append(("brisc", writer, 0))
  if reader is not None:
    proc.append(("ncrisc", reader, 1))
  if compute is not None:
    for i, kernel in enumerate(compute):
      proc.append((f"trisc{i}", kernel, i + 2))
  enables = 0
  kernel_text_off = [0] * 5
  off = kernel_off
  for _, kernel, idx in proc:
    kernel_text_off[idx] = off
    off = align_up(off + len(kernel.xip), L1_ALIGN)
    enables |= 1 << idx
  shared = bytearray(off - local_cb_off)
  shared[0:len(cb_blob)] = cb_blob
  for _, kernel, idx in proc:
    dst = kernel_text_off[idx] - local_cb_off
    shared[dst:dst + len(kernel.xip)] = kernel.xip
  shared_addr = TensixL1.KERNEL_CONFIG_BASE + local_cb_off

  launch = LaunchMsg()
  cfg = launch.kernel_config
  for i in range(3):
    cfg.kernel_config_base[i] = TensixL1.KERNEL_CONFIG_BASE
    cfg.sem_offset[i] = sem_off
  cfg.local_cb_offset = local_cb_off
  cfg.remote_cb_offset = local_cb_off + len(cb_blob)
  cfg.local_cb_mask = cb_mask
  cfg.min_remote_cb_start_index = FAST_CQ_NUM_CIRCULAR_BUFFERS
  cfg.enables = enables
  cfg.brisc_noc_id = 1
  cfg.brisc_noc_mode = 0
  cfg.mode = dispatch_mode
  cfg.rta_offset[0].rta_offset, cfg.rta_offset[0].crta_offset = rta_offsets[0], local_cb_off
  cfg.rta_offset[1].rta_offset, cfg.rta_offset[1].crta_offset = rta_offsets[1], local_cb_off
  for i in (2, 3, 4):
    cfg.rta_offset[i].rta_offset, cfg.rta_offset[i].crta_offset = rta_offsets[2], local_cb_off
  for i, value in enumerate(kernel_text_off):
    cfg.kernel_text_offset[i] = value
  cfg.host_assigned_id = host_assigned_id
  return shared_addr, bytes(shared), as_bytes(launch)

def build_ir(
  program: Program, roles: list[Role], compute: tuple | None,
  all_cores: list[Core], per_core_args: list[CoreArgs],
  dispatch_mode: int, host_assigned_id: int = 0,
) -> list[IRCommand]:
  max_w = max((len(a[0]) for a in per_core_args), default=0) * 4
  max_r = max((len(a[1]) for a in per_core_args), default=0) * 4
  max_c = max((len(a[2]) for a in per_core_args), default=0) * 4
  rta_sizes = (max_w, max_r, max_c)
  sem_off = align_up(max_w + max_r + max_c, L1_ALIGN)
  rta_blobs = [pack_rta(w, r, c, program.semaphores, sem_off) for w, r, c in per_core_args]
  reset_blob = struct.pack("<BBBB", 0, 0, 0, DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST)

  commands: list[IRCommand] = [
    Write(all_cores, TensixL1.GO_MSG, reset_blob),
    Write(all_cores, TensixL1.GO_MSG_INDEX, b"\0\0\0\0"),
  ]

  # RTAs: broadcast if uniform across cores, per-core unicast otherwise
  if rta_blobs and all(b == rta_blobs[0] for b in rta_blobs):
    commands.append(Write(all_cores, TensixL1.KERNEL_CONFIG_BASE, rta_blobs[0]))
  elif rta_blobs:
    commands.append(Write(all_cores, TensixL1.KERNEL_CONFIG_BASE, rta_blobs))

  # Per-role: launch message + shared payload (CB config + kernel text)
  for role_cores, reader, writer in roles:
    shared_addr, shared_blob, launch_blob = build_payload(
      program, reader, writer, compute, rta_sizes, dispatch_mode,
      sem_off=sem_off, host_assigned_id=host_assigned_id,
    )
    commands.append(Write(role_cores, TensixL1.LAUNCH, launch_blob))
    commands.append(Write(role_cores, shared_addr, shared_blob))

  commands.append(Launch(all_cores))
  return commands

def slow_dispatch(win, commands: list[IRCommand]):
  for cmd in commands:
    match cmd:
      case Write(cores=cores, addr=addr, data=data) if isinstance(data, list):
        for core, d in zip(cores, data):
          win.target(core)
          win.write(addr, d)
      case Write(cores=cores, addr=addr, data=data):
        for x0, x1, y0, y1 in mcast_rects(cores):
          win.target((x0, y0), (x1, y1))
          win.write(addr, data)
      case Launch(cores=cores):
        go = GoMsg()
        go.bits.signal = DevMsgs.RUN_MSG_GO
        go_blob = struct.pack("<I", go.all)
        for x0, x1, y0, y1 in mcast_rects(cores):
          win.target((x0, y0), (x1, y1))
          win.uc[TensixL1.GO_MSG:TensixL1.GO_MSG + 4] = go_blob
        for x, y in cores:
          win.target((x, y))
          deadline = time.perf_counter() + 10.0
          while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
            if time.perf_counter() > deadline:
              raise TimeoutError(f"timeout waiting for core ({x}, {y}) -- try tt-smi -r")
