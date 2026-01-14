# the slow dispatch path involves mapping TLB windows to tiles and writing to offsets in mmapped memory 
# this is slow because of the # of PCIe transactions needed to launch one kernel 
# you can multicast, but it's still slower than fast dispatch (coming soon)
from dataclasses import dataclass
from typing import Tuple, Literal
from enum import Enum
from autogen import *
import fcntl, mmap, ctypes

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr

class TLBMode(Enum):
  """Common TLB config presets combining ordering + static_vc."""
  # (ordering, static_vc)
  STRICT = (1, 1)   # register access: full ordering, writes land in order
  BULK = (0, 0)     # L1/DRAM data: max parallelism, no ordering guarantees
  POSTED = (2, 0)   # fire-and-forget writes: fastest, weakest ordering
  ORDERED_BULK = (0, 1)  # high throughput but packets stay in order through NoC

@dataclass
class TLBConfig:
  addr: int # 64-bit offset into the target tile's local address space (l1 for tensix, bank offset for dram, for example)
  start: Tuple[int, int] # start NoC coordinates
  end: Tuple[int, int] # end coordinates in NoC grid
  noc: Literal[0, 1] = 0
  mcast: bool = False
  mode: TLBMode = TLBMode.BULK

  def to_struct(self) -> NocTlbConfig:
    if (self.start == self.end) and self.mcast: print("warning: cannot multicast to one tile")
    ordering, static_vc = self.mode.value
    cfg = NocTlbConfig()
    cfg.addr = self.addr #! you must align this with the size of the TLB window. 2MB or 4GB
    cfg.x_start, cfg.y_start = self.start
    cfg.x_end, cfg.y_end = self.end
    cfg.noc = self.noc
    cfg.mcast = int(self.mcast)
    cfg.ordering = ordering
    cfg.static_vc = static_vc
    cfg.linked = 0  # never modify this
    return cfg 

class TLBSize(Enum):
  MiB_2 = 1 << 21  # BAR 0: 201 available, for L1/registers
  GiB_4 = 1 << 32  # BAR 4: 8 available, for GDDR6 banks

class TLBWindow:
  def __init__(self, fd: int, size: TLBSize, config: TLBConfig):
    self.fd = fd
    self.size = size.value
    self._allocate(size)
    self._mmap()
    self.configure(config)

  def _allocate(self, size: TLBSize):
    buf = bytearray(ctypes.sizeof(AllocateTlbIn) + ctypes.sizeof(AllocateTlbOut))
    cfg = AllocateTlbIn.from_buffer(buf)
    cfg.size = size.value
    fcntl.ioctl(self.fd, _IO(IOCTL_ALLOCATE_TLB), buf, True)
    out = AllocateTlbOut.from_buffer(buf, ctypes.sizeof(AllocateTlbIn))
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
    buf = bytearray(ctypes.sizeof(ConfigureTlbIn) + ctypes.sizeof(NocTlbConfig))
    cfg = ConfigureTlbIn.from_buffer(buf)
    cfg.tlb_id = self.tlb_id
    cfg.config = config.to_struct()
    fcntl.ioctl(self.fd, _IO(IOCTL_CONFIGURE_TLB), buf, False)

  def readi32(self, offset: int) -> int:
    """
    i32 read from the UC mmap
    
    :param self: Description
    :param offset: Description
    :type offset: int
    :return: Description
    :rtype: int
    """
    return int.from_bytes(self.uc[offset:offset+4], 'little')

  def writei32(self, offset: int, value: int):
    """
    i32 write to the UC mmap
    
    :param self: Description
    :param offset: Description
    :type offset: int
    :param value: Description
    :type value: int
    """
    self.uc[offset:offset+4] = value.to_bytes(4, 'little')

  def free(self):
    self.uc.close()
    self.wc.close()
    buf = bytearray(ctypes.sizeof(FreeTlbIn))
    cfg = FreeTlbIn.from_buffer(buf)
    cfg.tlb_id = self.tlb_id
    fcntl.ioctl(self.fd, _IO(IOCTL_FREE_TLB), buf, False)

  def __enter__(self): return self
  def __exit__(self, *_): self.free()
