from dataclasses import dataclass
from typing import Callable
import time
from configs import DRAM_ALIGNMENT, DRAM_BARRIER_BASE, Dram
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import dbg, DEBUG

def _align(x: int, a: int = DRAM_ALIGNMENT) -> int:
  return (x + a - 1) & ~(a - 1)

@dataclass(frozen=True)
class DramBuffer:
  name: str | None
  addr: int        # bank-local base address (same base in every bank)
  size: int        # total bytes
  page_size: int   # bytes per page (interleave granule)

class DramAllocator:
  def __init__(self, fd: int, dram_tiles: list[tuple[int, int, int]]):
    self.fd = fd
    self.bank_tiles = dram_tiles[::Dram.TILES_PER_BANK]  # one tile per bank (all tiles expose same 4GB)
    self.next = Dram.DRAM_WRITE_OFFSET
    self.max_page_size = 2 * 1024 * 1024
    self.win = TLBWindow(self.fd, TLBSize.GiB_4)

  def alloc(self, size: int, name: str | None = None) -> DramBuffer:
    num_banks = len(self.bank_tiles)
    page_size = min(self.max_page_size, _align((size + num_banks - 1) // num_banks))
    addr = self.next
    pages = (size + page_size - 1) // page_size
    pages_per_bank = (pages + num_banks - 1) // num_banks
    self.next = _align(self.next + pages_per_bank * page_size)
    dbg(2, "dram", f"alloc name={name!r} addr={addr:#x} size={size} pages={pages} page_size={page_size}")
    return DramBuffer(name=name, addr=addr, size=size, page_size=page_size)

  def alloc_write(self, data: bytes, name: str | None = None) -> DramBuffer:
    buf = self.alloc(len(data), name=name)
    self.write(buf, data)
    return buf

  def _for_each_page(
      self, buf: DramBuffer, size: int, mode: TLBMode,
      fn: Callable[[int, int], None]
    ) -> list[tuple[int, int, int]]:
    num_banks = len(self.bank_tiles)
    pages = (size + buf.page_size - 1) // buf.page_size
    touched = []

    for bank_idx, (bank_id, x, y) in enumerate(self.bank_tiles):
      if bank_idx >= pages: break
      touched.append((bank_id, x, y))
      self.win.configure(TLBConfig(addr=0, start=(x, y), end=(x, y), noc=0, mcast=False, mode=mode))
      local_page = 0
      for page_idx in range(bank_idx, pages, num_banks):
        addr = buf.addr + local_page * buf.page_size
        off = page_idx * buf.page_size
        fn(addr, off)
        local_page += 1

    return touched

  def barrier(self, tiles: list[tuple[int, int, int]]):
    for flag in Dram.BARRIER_FLAGS:
      for _, x, y in tiles:
        self.win.configure(TLBConfig(addr=0, start=(x, y), end=(x, y), noc=0, mcast=False, mode=TLBMode.STRICT))
        self.win.writei32(DRAM_BARRIER_BASE, flag)
        while self.win.readi32(DRAM_BARRIER_BASE) != flag: pass

  def write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    assert buf.page_size >= DRAM_ALIGNMENT and (buf.page_size & (DRAM_ALIGNMENT - 1)) == 0
    view = memoryview(data)
    t0 = time.perf_counter() if DEBUG >= 2 else 0

    def do_write(addr: int, off: int):
      page = view[off:off + buf.page_size]
      self.win.wc[addr:addr + len(page)] = page

    touched = self._for_each_page(buf, len(data), TLBMode.POSTED, do_write)
    self.barrier(touched)

    if DEBUG >= 2:
      elapsed_ms = (time.perf_counter() - t0) * 1000
      dbg(2, "dram", f"write {len(data)} bytes in {elapsed_ms:.1f}ms ({len(data) / elapsed_ms / 1e6:.2f} GB/s)")

  def read(self, buf: DramBuffer) -> bytes:
    result = bytearray(buf.size)

    def do_read(addr: int, off: int):
      remaining = buf.size - off
      n = min(buf.page_size, remaining)
      result[off:off + n] = self.win.wc[addr:addr + n]

    self._for_each_page(buf, buf.size, TLBMode.BULK, do_read)
    return bytes(result)

  def close(self):
    self.win.free()
