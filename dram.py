from dataclasses import dataclass
from enum import Enum
from typing import Callable
from defs import DRAM_ALIGNMENT, DRAM_BARRIER_BASE, Dram, TLBSize
from tlb import TLBConfig, TLBWindow, TLBMode

TILE_R, TILE_C = 32, 32
FACE_R, FACE_C = 16, 16
TILE_ELEMS = TILE_R * TILE_C

class DType(Enum):
  float32 = 4
  int32 = 4
  uint32 = 4
  float16 = 2
  bfloat16 = 2
  uint16 = 2
  int8 = 1
  uint8 = 1

def _align(x: int, a: int = DRAM_ALIGNMENT) -> int:
  return (x + a - 1) & ~(a - 1)

def _face_transform(data: bytes, bpe: int, forward: bool) -> bytes:
  """Convert between row-major and TILED_NFACES (4 faces of 16x16 per 32x32 tile)."""
  n_tiles = len(data) // (TILE_ELEMS * bpe)
  out = bytearray(len(data))
  for t in range(n_tiles):
    toff = t * TILE_ELEMS * bpe
    for face_r in range(2):
      for face_c in range(2):
        face_idx = face_r * 2 + face_c
        for r in range(FACE_R):
          rm_off = toff + ((face_r * FACE_R + r) * TILE_C + face_c * FACE_C) * bpe
          tf_off = toff + (face_idx * FACE_R * FACE_C + r * FACE_C) * bpe
          src, dst = (rm_off, tf_off) if forward else (tf_off, rm_off)
          out[dst : dst + FACE_C * bpe] = data[src : src + FACE_C * bpe]
  return bytes(out)

def _grid_transform(data: bytes, bpe: int, rows: int, cols: int, forward: bool) -> bytes:
  """Reorder between 2D row-major matrix and sequential tiles (row-major within each tile)."""
  assert rows % TILE_R == 0 and cols % TILE_C == 0, "Dimensions must be tile-aligned"
  tile_rows, tile_cols = rows // TILE_R, cols // TILE_C
  out = bytearray(len(data))
  for tr in range(tile_rows):
    for tc in range(tile_cols):
      tile_idx = tr * tile_cols + tc
      for r in range(TILE_R):
        grid_off = ((tr * TILE_R + r) * cols + tc * TILE_C) * bpe
        tile_off = (tile_idx * TILE_ELEMS + r * TILE_C) * bpe
        src, dst = (grid_off, tile_off) if forward else (tile_off, grid_off)
        out[dst : dst + TILE_C * bpe] = data[src : src + TILE_C * bpe]
  return bytes(out)

def tilize(data: bytes, bpe: int, rows: int | None = None, cols: int | None = None) -> bytes:
  """Convert row-major data to tiled format (with face reordering)."""
  if rows is not None and cols is not None:
    data = _grid_transform(data, bpe, rows, cols, forward=True)
  return _face_transform(data, bpe, forward=True)

def untilize(data: bytes, bpe: int, rows: int | None = None, cols: int | None = None) -> bytes:
  """Convert tiled data back to row-major format."""
  data = _face_transform(data, bpe, forward=False)
  if rows is not None and cols is not None:
    data = _grid_transform(data, bpe, rows, cols, forward=False)
  return data

@dataclass(frozen=True)
class DramBuffer:
  name: str | None
  addr: int
  size: int
  page_size: int
  dtype: DType | None = None

class DramAllocator:
  def __init__(self, fd: int, dram_tiles: list[tuple[int, int, int]]):
    self.fd = fd
    self.bank_tiles = dram_tiles[:: Dram.TILES_PER_BANK]
    self.next = Dram.DRAM_WRITE_OFFSET
    self.max_page_size = 2 * 1024 * 1024
    self.win = TLBWindow(self.fd, TLBSize.GiB_4)

  def alloc(
    self, size: int, name: str | None = None, *, page_size: int | None = None, dtype: DType | None = None
  ) -> DramBuffer:
    num_banks = len(self.bank_tiles)
    page_size = min(
      self.max_page_size, page_size or _align((size + num_banks - 1) // num_banks)
    )
    page_size = _align(page_size)
    addr = self.next
    pages = (size + page_size - 1) // page_size
    pages_per_bank = (pages + num_banks - 1) // num_banks
    self.next = _align(self.next + pages_per_bank * page_size)
    return DramBuffer(name=name, addr=addr, size=size, page_size=page_size, dtype=dtype)

  def alloc_write(
    self, data: bytes, name: str | None = None, *, page_size: int | None = None, dtype: DType | None = None
  ) -> DramBuffer:
    buf = self.alloc(len(data), name=name, page_size=page_size, dtype=dtype)
    self.write(buf, data)
    return buf

  def _for_each_page(
    self, buf: DramBuffer, size: int, mode: TLBMode, fn: Callable[[int, int], None]
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
        self.win.write32(DRAM_BARRIER_BASE, flag)
        while self.win.read32(DRAM_BARRIER_BASE) != flag:
          pass

  def write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    assert buf.page_size >= DRAM_ALIGNMENT and (buf.page_size & (DRAM_ALIGNMENT - 1)) == 0
    if buf.dtype is not None: data = tilize(data, buf.dtype.value)
    view = memoryview(data)

    def do_write(addr: int, off: int):
      page = view[off : off + buf.page_size]
      self.win.wc[addr : addr + len(page)] = page

    touched = self._for_each_page(buf, len(data), TLBMode.POSTED, do_write)
    self.barrier(touched)

  def read(self, buf: DramBuffer) -> bytes:
    result = bytearray(buf.size)

    def do_read(addr: int, off: int):
      remaining = buf.size - off
      n = min(buf.page_size, remaining)
      result[off : off + n] = self.win.wc[addr : addr + n]

    self._for_each_page(buf, buf.size, TLBMode.RELAXED, do_read)
    if buf.dtype is not None: return untilize(bytes(result), buf.dtype.value)
    return bytes(result)

  def close(self):
    self.win.free()
