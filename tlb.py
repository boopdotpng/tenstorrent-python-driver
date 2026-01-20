from dataclasses import dataclass
from enum import Enum
from abi import *
from configs import TLBSize
from helpers import _IO, noc1
import fcntl, mmap

class TLBMode(Enum):
  # Values are (ordering, static_vc): ordering 0=relaxed, 1=ordered, 2=posted
  # NOTE: On Blackhole, UMD programs `static_vc = 0` (disabled). Setting it on BH can wedge the chip (ARC hangs).
  STRICT = (1, 0)        # register access: full ordering, writes land in order
  BULK = (0, 0)          # L1/DRAM data: max parallelism, no ordering guarantees
  POSTED = (2, 0)        # fire-and-forget writes: fastest, weakest ordering
  ORDERED_BULK = (1, 0)  # high throughput but packets stay in order through NoC

Coord = tuple[int, int]

@dataclass
class TLBConfig:
  addr: int  # offset into target tile's local address space
  start: Coord | None = None  # start NoC coordinates (x, y)
  end: Coord | None = None    # end NoC coordinates (x, y); same as start for unicast
  noc: int = 0  # 0 or 1
  mcast: bool = False
  mode: TLBMode = TLBMode.BULK

  def to_struct(self) -> NocTlbConfig:
    if self.start is None or self.end is None: raise ValueError("tlb start/end must be set before configure")
    ordering, static_vc = self.mode.value
    start = noc1(*self.start) if self.noc == 1 else self.start
    end = noc1(*self.end) if self.noc == 1 else self.end
    cfg = NocTlbConfig()
    cfg.addr = self.addr #! you must align this with the size of the TLB window. 2MB or 4GB
    cfg.x_start, cfg.y_start = start
    cfg.x_end, cfg.y_end = end
    cfg.noc = self.noc
    cfg.mcast = int(self.mcast)
    cfg.ordering = ordering
    cfg.static_vc = static_vc
    cfg.linked = 0  # never modify this
    return cfg 

class TLBWindow:
  def __init__(self, fd: int, size: TLBSize, config: TLBConfig | None = None):
    self.fd = fd
    self.size = size.value
    self.config: TLBConfig | None = None
    self._allocate(size)
    self._mmap()
    if config is not None: self.configure(config)

  def _allocate(self, size: TLBSize):
    buf = bytearray(sizeof(AllocateTlbIn) + sizeof(AllocateTlbOut))
    cfg = AllocateTlbIn.from_buffer(buf)
    cfg.size = size.value
    fcntl.ioctl(self.fd, _IO(IOCTL_ALLOCATE_TLB), buf, True)
    out = AllocateTlbOut.from_buffer(buf, sizeof(AllocateTlbIn))
    self.tlb_id = out.tlb_id
    self._mmap_offset_uc = out.mmap_offset_uc
    self._mmap_offset_wc = out.mmap_offset_wc

  def _mmap(self):
    # UC (uncached) for register access, WC (write-combining) for bulk data
    self.uc = mmap.mmap(self.fd, self.size, flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=self._mmap_offset_uc)
    self.wc = mmap.mmap(self.fd, self.size, flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=self._mmap_offset_wc)

  def configure(self, config: TLBConfig):
    assert (config.addr & (self.size - 1)) == 0, f"tlb addr must be {self.size}-aligned"
    buf = bytearray(sizeof(ConfigureTlbIn) + sizeof(NocTlbConfig))
    cfg = ConfigureTlbIn.from_buffer(buf)
    cfg.tlb_id = self.tlb_id
    cfg.config = config.to_struct()
    fcntl.ioctl(self.fd, _IO(IOCTL_CONFIGURE_TLB), buf, False)
    self.config = config

  def write(self, addr: int, data: bytes, use_uc: bool = False, restore: bool = True):
    if not data: return
    if self.config is None: raise RuntimeError("tlb window has no active config")
    config = self.config
    prev = config.addr
    try:
      while data:
        base = addr & ~(self.size - 1)
        off = addr - base
        config.addr = base
        self.configure(config)
        n = min(len(data), self.size - off)
        view = self.uc if use_uc else self.wc
        view[off:off + n] = data[:n]
        addr, data = addr + n, data[n:]
    finally:
      if restore:
        config.addr = prev
        self.configure(config)

  def readi32(self, offset: int) -> int:
    return int.from_bytes(self.uc[offset:offset+4], 'little')

  def writei32(self, offset: int, value: int):
    self.uc[offset:offset+4] = value.to_bytes(4, 'little')

  def free(self):
    self.uc.close()
    self.wc.close()
    buf = bytearray(sizeof(FreeTlbIn))
    cfg = FreeTlbIn.from_buffer(buf)
    cfg.tlb_id = self.tlb_id
    fcntl.ioctl(self.fd, _IO(IOCTL_FREE_TLB), buf, False)

  def __enter__(self): return self
  def __exit__(self, *_): self.free()
