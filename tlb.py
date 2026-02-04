from dataclasses import dataclass
from enum import Enum
from defs import *
from helpers import _IO, noc1
import fcntl, mmap

class TLBMode(Enum):
  RELAXED = 0  # bulk transfers, may reorder
  STRICT = 1   # full ordering, slow dispatch
  POSTED = 2   # fire-and-forget, RAW hazard possible

@dataclass
class TLBConfig:
  addr: int
  start: tuple[int, int] | None = None
  end: tuple[int, int] | None = None
  noc: int = 0
  mcast: bool = False
  mode: TLBMode = TLBMode.RELAXED

  def to_struct(self) -> NocTlbConfig:
    if self.start is None or self.end is None:
      raise ValueError("tlb start/end must be set before configure")
    ordering = self.mode.value
    start = noc1(*self.start) if self.noc == 1 else self.start
    end = noc1(*self.end) if self.noc == 1 else self.end
    cfg = NocTlbConfig()
    cfg.addr = self.addr
    cfg.x_start, cfg.y_start = start
    cfg.x_end, cfg.y_end = end
    cfg.noc = self.noc
    cfg.mcast = int(self.mcast)
    cfg.ordering = ordering
    cfg.static_vc = 0
    cfg.linked = 0
    return cfg

class TLBWindow:
  def __init__(self, fd: int, size: TLBSize, config: TLBConfig | None = None):
    self.fd = fd
    self.size = size.value
    self.config: TLBConfig | None = None
    self._allocate(size)
    self._mmap()
    if config is not None:
      self.configure(config)

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
    self.uc = mmap.mmap(
      self.fd,
      self.size,
      flags=mmap.MAP_SHARED,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
      offset=self._mmap_offset_uc,
    )
    self.wc = mmap.mmap(
      self.fd,
      self.size,
      flags=mmap.MAP_SHARED,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
      offset=self._mmap_offset_wc,
    )

  def configure(self, config: TLBConfig):
    assert (config.addr & (self.size - 1)) == 0, f"tlb addr must be {self.size}-aligned"
    buf = bytearray(sizeof(ConfigureTlbIn) + sizeof(NocTlbConfig))
    cfg = ConfigureTlbIn.from_buffer(buf)
    cfg.tlb_id = self.tlb_id
    cfg.config = config.to_struct()
    fcntl.ioctl(self.fd, _IO(IOCTL_CONFIGURE_TLB), buf, False)
    self.config = config

  def write(self, addr: int, data: bytes, use_uc: bool = False, restore: bool = True):
    if self.config is None:
      raise RuntimeError("tlb window has no active config")
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
        view[off : off + n] = data[:n]
        addr, data = addr + n, data[n:]
    finally:
      if restore:
        config.addr = prev
        self.configure(config)

  def read32(self, offset: int) -> int:
    return int.from_bytes(self.uc[offset : offset + 4], "little")

  def write32(self, offset: int, value: int):
    self.uc[offset : offset + 4] = value.to_bytes(4, "little")

  def free(self):
    self.uc.close()
    self.wc.close()
    buf = bytearray(sizeof(FreeTlbIn))
    cfg = FreeTlbIn.from_buffer(buf)
    cfg.tlb_id = self.tlb_id
    fcntl.ioctl(self.fd, _IO(IOCTL_FREE_TLB), buf, False)

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.free()
