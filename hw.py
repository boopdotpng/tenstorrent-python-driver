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
  BRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x0082B0
  NCRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x00A2B0
  TRISC0_INIT_LOCAL_L1_BASE_SCRATCH = 0x00C2B0
  TRISC1_INIT_LOCAL_L1_BASE_SCRATCH = 0x00D2B0
  TRISC2_INIT_LOCAL_L1_BASE_SCRATCH = 0x00E2B0
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
  SCRATCH_RAM_11 = RESET_UNIT_OFFSET + 0x42C
  SCRATCH_RAM_13 = RESET_UNIT_OFFSET + 0x434
  MSG_AICLK_GO_BUSY = 0x52
  MSG_AICLK_GO_LONG_IDLE = 0x54
  TAG_AICLK = 14
  TAG_GDDR_ENABLED = 36
  DEFAULT_AICLK = 800
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

_ioctl_alloc_tlb = _tt_ioctl(11, _AllocIn, _AllocOut)
_ioctl_config_tlb = _tt_ioctl(13, _ConfigIn)
_ioctl_free_tlb = _tt_ioctl(12, _FreeIn)
_ioctl_pin_pages = _tt_ioctl(7, _PinIn, _PinOut)
_ioctl_unpin_pages = _tt_ioctl(10, _UnpinIn)
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


_CQ_L1 = 0x196C0

CQ_PREFETCH_Q_RD_PTR    = _CQ_L1 + 0x00
CQ_PREFETCH_Q_PCIE_RD   = _CQ_L1 + 0x04
CQ_COMPLETION_WR_PTR    = _CQ_L1 + 0x10
CQ_COMPLETION_RD_PTR    = _CQ_L1 + 0x20
CQ_COMPLETION_Q0_EVENT  = _CQ_L1 + 0x30
CQ_COMPLETION_Q1_EVENT  = _CQ_L1 + 0x40
CQ_DISPATCH_SYNC_SEM    = _CQ_L1 + 0x50
CQ_PREFETCH_Q_BASE      = _CQ_L1 + 0x180
CQ_PREFETCH_Q_ENTRIES   = 1534
CQ_PREFETCH_Q_ENTRY_SZ  = 2
CQ_PREFETCH_Q_SIZE      = CQ_PREFETCH_Q_ENTRIES * CQ_PREFETCH_Q_ENTRY_SZ
CQ_DISPATCH_CB_PAGES    = (512 * 1024) >> 12


_PCIE_NOC_BASE = 1 << 60

# Host sysmem layout
_HOST_ISSUE_BASE       = 4 * PCIE_ALIGN
_HOST_ISSUE_SIZE       = align_up(64 << 20, PCIE_ALIGN)
_HOST_COMPLETION_BASE  = _HOST_ISSUE_BASE + _HOST_ISSUE_SIZE
_HOST_COMPLETION_SIZE  = align_up(32 << 20, PCIE_ALIGN)
_HOST_SYSMEM_SIZE      = align_up(128 << 20, PAGE_SIZE)
_HOST_CQ_WR_OFF        = 2 * PCIE_ALIGN
_HOST_CQ_RD_OFF        = 3 * PCIE_ALIGN

assert _HOST_COMPLETION_BASE + _HOST_COMPLETION_SIZE <= _HOST_SYSMEM_SIZE


class CQSysmem:
  def __init__(self, fd: int, prefetch_win: TLBWindow, dispatch_win: TLBWindow):
    self.fd = fd
    self._prefetch_win = prefetch_win
    self._dispatch_win = dispatch_win
    flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS
    if hasattr(mmap, "MAP_POPULATE"):
      flags |= mmap.MAP_POPULATE
    self.sysmem = mmap.mmap(-1, _HOST_SYSMEM_SIZE, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    self._sysmem_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.sysmem))
    if (self._sysmem_addr % PAGE_SIZE) != 0 or (_HOST_SYSMEM_SIZE % PAGE_SIZE) != 0:
      raise RuntimeError("CQ sysmem must be page-aligned and page-sized")
    out = _ioctl_pin_pages(self.fd, flags=_PIN_NOC_DMA, virtual_address=self._sysmem_addr, size=_HOST_SYSMEM_SIZE)
    self.noc_addr = out.noc_address
    if (self.noc_addr & _PCIE_NOC_BASE) != _PCIE_NOC_BASE:
      raise RuntimeError(f"bad NOC sysmem address: 0x{self.noc_addr:x}")
    self.noc_local = self.noc_addr - _PCIE_NOC_BASE
    if self.noc_local > 0xFFFF_FFFF:
      raise RuntimeError(f"CQ sysmem NOC offset too large: 0x{self.noc_local:x}")
    self._issue_wr = 0
    self._prefetch_q_wr_idx = 0
    self._event_id = 0
    self._completion_base_16b = ((self.noc_local + _HOST_COMPLETION_BASE) >> 4) & 0x7FFF_FFFF
    self._completion_page_16b = PAGE_SIZE >> 4
    self._completion_end_16b = self._completion_base_16b + (_HOST_COMPLETION_SIZE >> 4)
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0

  def _sysmem_read32(self, off):
    return struct.unpack("<I", self.sysmem[off : off + 4])[0]

  def _sysmem_write32(self, off, val):
    self.sysmem[off : off + 4] = struct.pack("<I", val)

  def _wait_prefetch_slot_free(self, idx: int, timeout_s: float = 1.0):
    off = CQ_PREFETCH_Q_BASE + idx * CQ_PREFETCH_Q_ENTRY_SZ
    deadline = time.perf_counter() + timeout_s
    while struct.unpack("<H", self._prefetch_win.uc[off : off + 2])[0] != 0:
      if time.perf_counter() > deadline:
        raise TimeoutError("timeout waiting for prefetch queue slot")

  def _issue_write(self, record: bytes):
    self._issue_wr = align_up(self._issue_wr, PCIE_ALIGN)
    if self._issue_wr + len(record) > _HOST_ISSUE_SIZE:
      self._issue_wr = 0
    base = _HOST_ISSUE_BASE + self._issue_wr
    self.sysmem[base : base + len(record)] = record
    self._issue_wr += len(record)
    idx = self._prefetch_q_wr_idx
    self._wait_prefetch_slot_free(idx)
    off = CQ_PREFETCH_Q_BASE + idx * CQ_PREFETCH_Q_ENTRY_SZ
    self._prefetch_win.uc[off : off + 2] = struct.pack("<H", len(record) >> 4)
    self._prefetch_q_wr_idx = (idx + 1) % CQ_PREFETCH_Q_ENTRIES

  def flush(self, cq):
    offset = 0
    for size_16b in cq.sizes:
      size = size_16b << 4
      self._issue_write(cq.stream[offset : offset + size])
      offset += size
    cq.clear()

  def wait_completion(self, event_id: int, timeout_s: float = 10.0):
    deadline = time.perf_counter() + timeout_s
    while True:
      wr_raw = self._sysmem_read32(_HOST_CQ_WR_OFF)
      wr_16b, wr_toggle = wr_raw & 0x7FFF_FFFF, (wr_raw >> 31) & 1
      if wr_16b != self._completion_rd_16b or wr_toggle != self._completion_rd_toggle:
        off = (self._completion_rd_16b << 4) - self.noc_local
        got = self._sysmem_read32(off + 16)  # skip 16-byte dispatch command header
        self._completion_rd_16b += self._completion_page_16b
        if self._completion_rd_16b >= self._completion_end_16b:
          self._completion_rd_16b = self._completion_base_16b
          self._completion_rd_toggle ^= 1
        raw = (self._completion_rd_16b & 0x7FFF_FFFF) | (self._completion_rd_toggle << 31)
        self._dispatch_win.write32(CQ_COMPLETION_RD_PTR, raw)
        self._sysmem_write32(_HOST_CQ_RD_OFF, raw)
        if got != (event_id & 0xFFFFFFFF):
          raise RuntimeError(f"completion event mismatch: got {got}, expected {event_id}")
        return
      if time.perf_counter() > deadline:
        raise TimeoutError(f"timeout waiting for completion event {event_id} -- try tt-smi -r")
      time.sleep(0.0002)

  def reset_run_state(self):
    self._issue_wr = 0
    self._prefetch_q_wr_idx = 0
    self._prefetch_win.write32(CQ_PREFETCH_Q_RD_PTR, CQ_PREFETCH_Q_BASE + CQ_PREFETCH_Q_SIZE)
    self._prefetch_win.write32(CQ_PREFETCH_Q_PCIE_RD, (self.noc_local + _HOST_ISSUE_BASE) & 0xFFFFFFFF)
    for i in range(CQ_PREFETCH_Q_ENTRIES):
      off = CQ_PREFETCH_Q_BASE + i * CQ_PREFETCH_Q_ENTRY_SZ
      self._prefetch_win.uc[off : off + 2] = b"\0\0"
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    self._dispatch_win.write32(CQ_COMPLETION_WR_PTR, self._completion_base_16b)
    self._dispatch_win.write32(CQ_COMPLETION_RD_PTR, self._completion_base_16b)
    self._sysmem_write32(_HOST_CQ_WR_OFF, self._completion_base_16b)
    self._sysmem_write32(_HOST_CQ_RD_OFF, self._completion_base_16b)

  def close(self):
    self._prefetch_win.close()
    self._dispatch_win.close()
    _ioctl_unpin_pages(self.fd, virtual_address=self._sysmem_addr, size=_HOST_SYSMEM_SIZE)
    self.sysmem.close()


class TileGrid:
  ARC = (8, 0)
  TENSIX_X = (*range(1, 8), *range(10, 15))
  WORKER_CORES = [(x, y) for x in TENSIX_X for y in range(2, 12)]


USE_USB_DISPATCH = os.environ.get("TT_USB") == "1"


def build_bank_noc_table(harvested_dram: int, worker_cores: list[Core]) -> bytes:
  NUM_NOCS, NUM_DRAM_BANKS, NUM_L1_BANKS = 2, 7, 110
  WORKER_EP_LOGICAL = {
    0: [2, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1],
    4: [2, 1], 5: [2, 1], 6: [2, 1], 7: [2, 1],
  }

  def dram_translated_map(harvested_bank: int) -> dict[tuple[int, int], Core]:
    START_X, START_Y, PORTS, TOTAL_BANKS = 17, 12, 3, 8
    m: dict[tuple[int, int], Core] = {}

    def map_banks(start, end, x, y0=START_Y):
      y = y0
      for bank in range(start, end):
        for port in range(PORTS):
          m[(bank, port)] = (x, y)
          y += 1

    half = TOTAL_BANKS // 2
    if harvested_bank < half:
      mirror = harvested_bank + half - 1
      map_banks(0, half - 1, START_X + 1)
      map_banks(half - 1, mirror, START_X)
      map_banks(mirror + 1, TOTAL_BANKS - 1, START_X, START_Y + (mirror - (half - 1)) * PORTS)
      map_banks(mirror, mirror + 1, START_X, START_Y + (half - 1) * PORTS)
    else:
      mirror = harvested_bank - half
      map_banks(0, mirror, START_X)
      map_banks(mirror + 1, half, START_X, START_Y + mirror * PORTS)
      map_banks(mirror, mirror + 1, START_X, START_Y + (half - 1) * PORTS)
      map_banks(half, TOTAL_BANKS - 1, START_X + 1)
    return m

  dram_translated = dram_translated_map(harvested_dram)
  dram_xy = []
  for noc in range(NUM_NOCS):
    for bank in range(NUM_DRAM_BANKS):
      port = WORKER_EP_LOGICAL[bank][noc]
      x, y = dram_translated[(bank, port)]
      dram_xy.append(noc_xy(x, y))

  tensix_cols = sorted({x for x, _ in worker_cores})
  l1_xy = []
  for _ in range(NUM_NOCS):
    for bank_id in range(NUM_L1_BANKS):
      col_idx = bank_id % len(tensix_cols)
      row_idx = bank_id // len(tensix_cols)
      x = tensix_cols[col_idx]
      y = 2 + (row_idx % 10)
      l1_xy.append(noc_xy(x, y))

  blob = struct.pack(f"<{len(dram_xy)}H", *dram_xy)
  blob += struct.pack(f"<{len(l1_xy)}H", *l1_xy)
  blob += struct.pack(f"<{NUM_DRAM_BANKS}i", *([0] * NUM_DRAM_BANKS))
  blob += struct.pack(f"<{NUM_L1_BANKS}i", *([0] * NUM_L1_BANKS))
  return blob
