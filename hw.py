import ctypes, fcntl, functools, mmap, os, struct, time
from ctypes import c_uint8 as u8, c_uint16 as u16, c_uint32 as u32, c_uint64 as u64
from enum import Enum

Core = tuple[int, int]
PAGE_SIZE = 4096
L1_ALIGN = 16
PCIE_ALIGN = 64

def align_up(value: int, align: int) -> int:
  return (value + align - 1) // align * align

def align_down(value: int, alignment: int) -> tuple[int, int]:
  base = value & ~(alignment - 1)
  return base, value - base

def as_bytes(obj) -> bytes:
  return ctypes.string_at(ctypes.addressof(obj), ctypes.sizeof(obj))

def noc_xy(x: int, y: int) -> int:
  return ((y << 6) | x) & 0xFFFF

class S(ctypes.LittleEndianStructure):
  def __init__(self, **kw):
    super().__init__()
    for k, v in kw.items(): setattr(self, k, v)

class TensixL1:
  SIZE = 0x180000
  LAUNCH = 0x000070                        # mailbox_base(0x60) + 0x10
  GO_MSG = 0x000370                        # mailbox_base + 0x310
  GO_MSG_INDEX = 0x0003A0                  # mailbox_base + 0x340
  KERNEL_CONFIG_BASE = 0x0082B0
  BRISC_FIRMWARE_BASE = 0x003840
  DATA_BUFFER_SPACE_BASE = 0x037000
  PROFILER_CONTROL = 0x0009C0              # 32 x u32 = 128 bytes
  PROFILER_HOST_BUFFER_BYTES_PER_RISC = 65536
  MEM_BANK_TO_NOC_SCRATCH = 0x0112B0

class TensixMMIO:
  LOCAL_RAM_START = 0xFFB00000
  LOCAL_RAM_END = 0xFFB01FFF
  RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0
  RISCV_DEBUG_REG_TRISC0_RESET_PC = 0xFFB12228
  RISCV_DEBUG_REG_TRISC1_RESET_PC = 0xFFB1222C
  RISCV_DEBUG_REG_TRISC2_RESET_PC = 0xFFB12230
  RISCV_DEBUG_REG_NCRISC_RESET_PC = 0xFFB12238
  SOFT_RESET_ALL = 0x47800                 # all 5 RISC-V cores
  SOFT_RESET_BRISC_ONLY_RUN = 0x47000      # keep TRISC/NCRISC in reset, release BRISC

class Arc:
  NOC_BASE = 0x80000000
  RESET_UNIT_OFFSET = 0x30000
  SCRATCH_RAM_13 = RESET_UNIT_OFFSET + 0x434
  TAG_GDDR_ENABLED = 36
  DEFAULT_GDDR_ENABLED = 0xFF

class Dram:
  BANK_COUNT = 8
  TILES_PER_BANK = 3
  WRITE_OFFSET = 0x40
  BARRIER_BASE = 0x0
  ALIGNMENT = 64
  BARRIER_FLAGS = (0xAA, 0xBB)
  BANK_TILE_YS = {
    0: (0, 1, 11), 1: (2, 3, 10), 2: (4, 8, 9), 3: (5, 6, 7),
    4: (0, 1, 11), 5: (2, 3, 10), 6: (4, 8, 9), 7: (5, 6, 7),
  }
  BANK_X = {b: 0 if b < 4 else 9 for b in range(8)}

def _tt_ioctl(nr, in_t, out_t=None):
  cmd = (0xFA << 8) | nr
  @functools.wraps(_tt_ioctl)
  def call(fd, **kwargs):
    buf = bytearray(ctypes.sizeof(in_t) + (ctypes.sizeof(out_t) if out_t else 0))
    inp = in_t.from_buffer(buf)
    if out_t and hasattr(in_t, 'output_size_bytes'):
      inp.output_size_bytes = ctypes.sizeof(out_t)
    if hasattr(in_t, 'argsz'):
      inp.argsz = ctypes.sizeof(in_t)
    for k, v in kwargs.items():
      setattr(inp, k, v)
    fcntl.ioctl(fd, cmd, buf, True)
    return out_t.from_buffer_copy(buf, ctypes.sizeof(in_t)) if out_t else None
  return call

class _AllocIn(S):
  _pack_ = 1
  _fields_ = [("size", u64), ("reserved", u64)]

class _AllocOut(S):
  _pack_ = 1
  _fields_ = [("id", u32), ("reserved0", u32), ("mmap_offset_uc", u64), ("mmap_offset_wc", u64), ("reserved1", u64)]

class _FreeIn(S):
  _pack_ = 1
  _fields_ = [("id", u32)]

class _NocTlbConfig(S):
  _pack_ = 1
  _fields_ = [
    ("addr", u64),
    ("x_end", u16), ("y_end", u16), ("x_start", u16), ("y_start", u16),
    ("noc", u8), ("mcast", u8), ("ordering", u8), ("linked", u8),
    ("static_vc", u8), ("reserved0", u8 * 3),
    ("reserved1", u32 * 2),
  ]

class _ConfigIn(S):
  _pack_ = 1
  _fields_ = [("id", u32), ("reserved", u32), ("config", _NocTlbConfig)]

class _PinIn(S):
  _pack_ = 1
  _fields_ = [("output_size_bytes", u32), ("flags", u32), ("virtual_address", u64), ("size", u64)]

class _PinOut(S):
  _pack_ = 1
  _fields_ = [("physical_address", u64), ("noc_address", u64)]

class _UnpinIn(S):
  _pack_ = 1
  _fields_ = [("virtual_address", u64), ("size", u64), ("reserved", u64)]

class _PowerStateIn(S):
  _pack_ = 1
  _fields_ = [
    ("argsz", u32), ("flags", u32),
    ("reserved0", u8), ("validity", u8),
    ("power_flags", u16), ("power_settings", u16 * 14),
  ]

_ioctl_alloc_tlb = _tt_ioctl(11, _AllocIn, _AllocOut)
_ioctl_config_tlb = _tt_ioctl(13, _ConfigIn)
_ioctl_free_tlb = _tt_ioctl(12, _FreeIn)
_ioctl_pin_pages = _tt_ioctl(7, _PinIn, _PinOut)
_ioctl_unpin_pages = _tt_ioctl(10, _UnpinIn)
_ioctl_set_power_state = _tt_ioctl(15, _PowerStateIn)
_PIN_NOC_DMA = 2

class NocOrdering(Enum):
  RELAXED = 0
  STRICT = 1
  POSTED = 2

class TLBWindow:
  SIZE_2M = 1 << 21
  SIZE_4G = 1 << 32

  def __init__(self, fd: int, start: Core, end: Core | None = None, addr: int = 0,
               mode: NocOrdering = NocOrdering.STRICT, size: int = SIZE_2M):
    self.fd, self.size = fd, size
    out = _ioctl_alloc_tlb(fd, size=size)
    self._id = out.id
    rw = mmap.PROT_READ | mmap.PROT_WRITE
    self.uc = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=rw, offset=out.mmap_offset_uc)
    self.wc = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=rw, offset=out.mmap_offset_wc)
    self.target(start, end, addr=addr, mode=mode)

  def target(self, start: Core, end: Core | None = None, addr: int = 0, mode: NocOrdering = NocOrdering.STRICT):
    end = end or start
    _ioctl_config_tlb(self.fd, id=self._id, config=_NocTlbConfig(
      addr=addr, x_start=start[0], y_start=start[1], x_end=end[0], y_end=end[1],
      mcast=int(end != start), ordering=mode.value))

  def read32(self, offset: int) -> int:
    return int.from_bytes(self.uc[offset : offset + 4], "little")

  def write32(self, offset: int, value: int):
    self.uc[offset : offset + 4] = value.to_bytes(4, "little")

  def write(self, addr: int, data: bytes, wc: bool = False):
    view = self.wc if wc else self.uc
    view[addr : addr + len(data)] = data

  def close(self):
    self.uc.close()
    self.wc.close()
    _ioctl_free_tlb(self.fd, id=self._id)

  def __enter__(self): return self
  def __exit__(self, *_): self.close()

class Sysmem:
  PCIE_NOC_XY = (24 << 6) | 19

  def __init__(self, fd: int, size: int = 1 << 30):
    self.fd = fd
    page_size = os.sysconf("SC_PAGE_SIZE")
    self.size = (size + page_size - 1) & ~(page_size - 1)
    self.buf = mmap.mmap(-1, self.size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                         prot=mmap.PROT_READ | mmap.PROT_WRITE)
    self._va = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
    out = _ioctl_pin_pages(self.fd, flags=_PIN_NOC_DMA, virtual_address=self._va, size=self.size)
    self.noc_addr = out.noc_address

  def close(self):
    _ioctl_unpin_pages(self.fd, virtual_address=self._va, size=self.size)
    self.buf.close()

class TileGrid:
  ARC = (8, 0)
  TENSIX_X = (*range(1, 8), *range(10, 15))
  WORKER_CORES = [(x, y) for x in TENSIX_X for y in range(2, 12)]

USE_USB_DISPATCH = os.environ.get("TT_USB") == "1"

def build_bank_noc_table(harvested_dram: int, worker_cores: list[Core]) -> bytes:
  NOCS, DRAM_BANKS, L1_BANKS, PORTS = 2, 7, 110, 3
  # per-bank NOC port selection: bank -> [noc0_port, noc1_port]
  BANK_PORT = [[2,1],[0,1],[0,1],[0,1],[2,1],[2,1],[2,1],[2,1]]

  # 7 logical DRAM banks mapped to translated coords on x=17/18, 3 ports each, y from 12
  # harvested bank's mirror gets pushed to the last slot on its column
  h, half = harvested_dram, 4
  mirror = h + half - 1 if h < half else h - half
  if h < half:
    right = list(range(half - 1))
    left = [b for b in range(half - 1, DRAM_BANKS) if b != mirror] + [mirror]
  else:
    left = [b for b in range(half) if b != mirror] + [mirror]
    right = list(range(half, DRAM_BANKS))
  bank_xy = {}
  for i, b in enumerate(right):
    bank_xy[b] = (18, 12 + i * PORTS)
  for i, b in enumerate(left):
    bank_xy[b] = (17, 12 + i * PORTS)

  # DRAM section: noc_xy per (noc, bank), selecting the right port
  dram = []
  for noc in range(NOCS):
    for b in range(DRAM_BANKS):
      x, y0 = bank_xy[b]
      dram.append(noc_xy(x, y0 + BANK_PORT[b][noc]))

  # L1 section: worker cores in column-major order, same for both NOCs
  cols = sorted({x for x, _ in worker_cores})
  l1 = []
  for _ in range(NOCS):
    for i in range(L1_BANKS):
      l1.append(noc_xy(cols[i % len(cols)], 2 + (i // len(cols)) % 10))

  return struct.pack(f"<{len(dram)}H{len(l1)}H{DRAM_BANKS + L1_BANKS}i", *dram, *l1, *([0] * (DRAM_BANKS + L1_BANKS)))

