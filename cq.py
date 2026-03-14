import ctypes, mmap, os, struct, time
from dataclasses import dataclass

from hw import *
from hw import _ioctl_pin_pages, _ioctl_unpin_pages, _PIN_NOC_DMA
from dispatch import Write, Launch, IRCommand, Rect, noc_mcast_xy, mcast_rects

# CQ L1 address map
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

# CQ command type IDs
_RELAY_INLINE       = 5
_WRITE_PACKED       = 5
_WRITE_PACKED_LARGE = 6
_WRITE_LINEAR_HOST  = 3
_WAIT               = 7
_GO_SIGNAL          = 14
_SET_GO_NOC_DATA    = 17
_TIMESTAMP          = 18

CQ_CMD_SIZE = 16
DONE_STREAM = 48

def _cq_hdr(fmt, *args):
  return struct.pack(fmt, *args).ljust(CQ_CMD_SIZE, b"\0")

@dataclass
class CQWritePackedLarge:
  rects: list[Rect]
  addr: int
  data: bytes

  def to_bytes(self) -> list[bytes]:
    padded = bytes(self.data).ljust(align_up(len(self.data), L1_ALIGN), b"\0")
    records = []
    for i in range(0, len(self.rects), 35):  # max 35 sub-commands per batch
      batch = self.rects[i:i + 35]
      hdr = _cq_hdr("<BBHHH", _WRITE_PACKED_LARGE, 2, len(batch), L1_ALIGN, 0)
      subs = b"".join(
        struct.pack("<IIHBB", xy, self.addr, len(self.data) - 1, cnt, 0x01)
        for r in batch for xy, cnt in [noc_mcast_xy(r)]
      ).ljust(align_up(len(batch) * 12, L1_ALIGN), b"\0")
      records.append(hdr + subs + padded * len(batch))
    return records

@dataclass
class CQWritePacked:
  cores: list[Core]
  addr: int
  data: bytes | list[bytes]

  def to_bytes(self) -> list[bytes]:
    uniform = isinstance(self.data, bytes)
    size = len(self.data) if uniform else len(self.data[0])
    flags = 0x02 if uniform else 0  # NO_STRIDE flag
    hdr = _cq_hdr("<BBHHHI", _WRITE_PACKED, flags, len(self.cores), 0, size, self.addr)
    nocs = b"".join(struct.pack("<I", noc_xy(x, y)) for x, y in self.cores)
    nocs = nocs.ljust(align_up(len(self.cores) * 4, L1_ALIGN), b"\0")
    if uniform:
      body = bytes(self.data).ljust(align_up(size, L1_ALIGN), b"\0")
    else:
      stride = align_up(size, L1_ALIGN)
      body = b"".join(d.ljust(stride, b"\0") for d in self.data)
    return [hdr + nocs + body]

@dataclass
class CQSetGoSignalNocData:
  cores: list[Core]

  def to_bytes(self) -> list[bytes]:
    hdr = _cq_hdr("<BBHI", _SET_GO_NOC_DATA, 0, 0, len(self.cores))
    return [hdr + b"".join(struct.pack("<I", noc_xy(x, y)) for x, y in self.cores)]

@dataclass
class CQSendGoSignal:
  go_word: int
  stream: int
  count: int
  num_unicast: int

  def to_bytes(self) -> list[bytes]:
    return [_cq_hdr("<BIBBBII", _GO_SIGNAL, self.go_word, 0xFF, self.num_unicast, 0, self.count, self.stream)]

@dataclass
class CQWaitStream:
  stream: int
  count: int
  clear: bool = True

  def to_bytes(self) -> list[bytes]:
    flags = 0x08 | (0x10 if self.clear else 0)  # WAIT_STREAM | CLEAR_STREAM
    return [_cq_hdr("<BBHII", _WAIT, flags, self.stream, 0, self.count)]

@dataclass
class CQHostEvent:
  event_id: int

  def to_bytes(self) -> list[bytes]:
    payload = struct.pack("<I", self.event_id & 0xFFFFFFFF).ljust(L1_ALIGN, b"\0")
    return [_cq_hdr("<BBHIQ", _WRITE_LINEAR_HOST, 1, 0, 0, CQ_CMD_SIZE + len(payload)) + payload]

@dataclass
class CQTimestamp:
  noc_xy_addr: int
  addr: int

  def to_bytes(self) -> list[bytes]:
    return [_cq_hdr("<BxHII", _TIMESTAMP, 0, self.noc_xy_addr, self.addr)]

CQCommand = CQWritePackedLarge | CQWritePacked | CQSetGoSignalNocData | CQSendGoSignal | CQWaitStream | CQHostEvent | CQTimestamp

def _relay_inline(payload: bytes) -> bytes:
  stride = align_up(CQ_CMD_SIZE + len(payload), PCIE_ALIGN)
  hdr = _cq_hdr("<BBHII", _RELAY_INLINE, 0, 0, len(payload), stride)
  return hdr + payload.ljust(stride - CQ_CMD_SIZE, b"\0")

class CommandQueue:
  def __init__(self):
    self.stream = bytearray()
    self.sizes: list[int] = []

  def clear(self):
    self.stream.clear()
    self.sizes.clear()

  def append(self, cmd: CQCommand):
    for payload in cmd.to_bytes():
      record = _relay_inline(payload)
      self.stream.extend(record)
      self.sizes.append(len(record) >> 4)

  def extend(self, cmds: list[CQCommand]):
    for cmd in cmds:
      self.append(cmd)

def _lower_ir(commands: list[IRCommand], go_word: int) -> list[CQCommand]:
  result: list[CQCommand] = []
  for cmd in commands:
    match cmd:
      case Write(cores=cores, addr=addr, data=data) if isinstance(data, list):
        result.append(CQWritePacked(cores, addr, data))
      case Write(cores=cores, addr=addr, data=data):
        result.append(CQWritePackedLarge(mcast_rects(cores), addr, data))
      case Launch(cores=cores):
        result.append(CQSetGoSignalNocData(cores))
        result.append(CQWaitStream(DONE_STREAM, 0))
        result.append(CQSendGoSignal(go_word, DONE_STREAM, 0, len(cores)))
        result.append(CQWaitStream(DONE_STREAM, len(cores)))
  return result

def lower_fast(
  programs: list[tuple[list[IRCommand], bool]],
  go_word: int, cores: list[Core],
  timestamps: list[tuple[int, int]] | None = None,
  profiler_flat_ids: dict | None = None,
  profiler_dram_addr: int = 0, profiler_core_count_per_dram: int = 0,
) -> list[CQCommand]:
  profiling = os.environ.get("PROFILE") == "1" and profiler_flat_ids is not None
  result: list[CQCommand] = []
  if profiling:
    rects = mcast_rects(cores)
    prof_cores = sorted(profiler_flat_ids, key=lambda xy: (xy[0], xy[1]))
    blobs = []
    for core in prof_cores:
      x, y = core
      ctrl = [0] * 32
      ctrl[12] = profiler_dram_addr
      ctrl[14], ctrl[15] = x, y
      ctrl[16] = profiler_flat_ids[core]
      ctrl[17] = profiler_core_count_per_dram
      blobs.append(struct.pack("<32I", *ctrl))
    result.append(CQWritePacked(prof_cores, TensixL1.PROFILER_CONTROL, blobs))
    result.append(CQWritePackedLarge(rects, TensixL1.PROFILER_CONTROL, b"\0" * (5 * 4)))
  for i, (ir, profiled) in enumerate(programs):
    if profiling and profiled:
      base = TensixL1.PROFILER_CONTROL
      result.append(CQWritePackedLarge(rects, base + 5 * 4, b"\0" * (5 * 4)))
      result.append(CQWritePackedLarge(rects, base + 19 * 4, b"\0" * 4))
    ts = 2 * i
    if timestamps and ts + 1 < len(timestamps):
      result.append(CQTimestamp(*timestamps[ts]))
    result.extend(_lower_ir(ir, go_word))
    if timestamps and ts + 1 < len(timestamps):
      result.append(CQTimestamp(*timestamps[ts + 1]))
  return result

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
    # init prefetch core L1 pointers and zero the prefetch queue
    self._prefetch_win.write32(CQ_PREFETCH_Q_RD_PTR, CQ_PREFETCH_Q_BASE + CQ_PREFETCH_Q_SIZE)
    self._prefetch_win.write32(CQ_PREFETCH_Q_PCIE_RD, (self.noc_local + _HOST_ISSUE_BASE) & 0xFFFFFFFF)
    self._prefetch_win.uc[CQ_PREFETCH_Q_BASE : CQ_PREFETCH_Q_BASE + CQ_PREFETCH_Q_SIZE] = bytes(CQ_PREFETCH_Q_SIZE)
    # init host sysmem completion doorbells
    self._sysmem_write32(_HOST_CQ_WR_OFF, self._completion_base_16b)
    self._sysmem_write32(_HOST_CQ_RD_OFF, self._completion_base_16b)

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

  def close(self):
    self._prefetch_win.close()
    self._dispatch_win.close()
    _ioctl_unpin_pages(self.fd, virtual_address=self._sysmem_addr, size=_HOST_SYSMEM_SIZE)
    self.sysmem.close()
