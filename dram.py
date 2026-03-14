from dataclasses import dataclass
import struct

import numpy as np

from hw import *
from dispatch import *
from kernels import tilize_reader, TILIZE_COMPUTE, tilize_writer, untilize_reader, UNTILIZE_COMPUTE, untilize_writer

Shape = tuple[int, ...]
TILE_R, TILE_C, FACE_R, FACE_C = 32, 32, 16, 16

def _np_dtype(bpe: int) -> np.dtype:
  return {2: np.dtype('uint16'), 4: np.dtype('uint32')}[bpe]

def tilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  rows, cols = shape[-2], shape[-1]
  assert rows % TILE_R == 0 and cols % TILE_C == 0
  batch = 1
  for d in shape[:-2]: batch *= d
  dt = _np_dtype(bpe)
  a = np.frombuffer(data, dtype=dt).reshape(batch, rows // TILE_R, TILE_R, cols // TILE_C, TILE_C)
  a = a.transpose(0, 1, 3, 2, 4)                                    # grid: (b, tr, tc, 32, 32)
  a = a.reshape(-1, 2, FACE_R, 2, FACE_C).transpose(0, 1, 3, 2, 4) # face: (n_tiles, 2, 2, 16, 16)
  return a.tobytes()

def untilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  rows, cols = shape[-2], shape[-1]
  assert rows % TILE_R == 0 and cols % TILE_C == 0
  batch = 1
  for d in shape[:-2]: batch *= d
  tr, tc = rows // TILE_R, cols // TILE_C
  dt = _np_dtype(bpe)
  a = np.frombuffer(data, dtype=dt).reshape(-1, 2, 2, FACE_R, FACE_C)
  a = a.transpose(0, 1, 3, 2, 4).reshape(batch, tr, tc, TILE_R, TILE_C)
  a = a.transpose(0, 1, 3, 2, 4).reshape(batch, rows, cols)
  return a.tobytes()

@dataclass
class DramBuffer:
  name: str
  addr: int
  num_tiles: int
  dtype: Dtype
  shape: Shape | None = None

  @property
  def page_size(self) -> int:
    return self.dtype.tile_size

  @property
  def size(self) -> int:
    return self.num_tiles * self.page_size

def build_transfer_program(
  buf: DramBuffer, direction: str, n_cores: int, sysmem_noc_addr: int,
) -> tuple[Program, int]:
  assert buf.shape is not None
  rows, cols = buf.shape[-2], buf.shape[-1]
  assert rows % TILE_R == 0 and cols % TILE_C == 0
  batch = 1
  for d in buf.shape[:-2]:
    batch *= d
  logical_bytes = batch * rows * cols * buf.dtype.bpe
  assert logical_bytes == buf.size, f"shape {buf.shape} does not match buffer size {buf.size}"
  tile_cols = cols // TILE_C
  total_tiles = batch * (rows // TILE_R) * tile_cols
  n = min(n_cores, total_tiles)
  tpc = (total_tiles + n - 1) // n

  pcie_base = (Sysmem.PCIE_NOC_XY << 36) | (1 << 60) | (sysmem_noc_addr & ((1 << 36) - 1))
  tile_row_bytes = TILE_C * buf.dtype.bpe
  row_bytes = cols * buf.dtype.bpe

  def tile_args(ci, _xy, _n):
    start = ci * tpc
    return [start, min(tpc, total_tiles - start) if start < total_tiles else 0]
  def compute_args(ci, _xy, _n):
    start = ci * tpc
    return [min(tpc, total_tiles - start) if start < total_tiles else 0]

  if direction == "tilize":
    rk = tilize_reader(pcie_base, tile_row_bytes, tile_cols, row_bytes)
    wk = tilize_writer(buf.addr)
    ck, name = TILIZE_COMPUTE, "dram_fill_tilize"
  else:
    rk = untilize_reader(buf.addr)
    wk = untilize_writer(pcie_base, tile_row_bytes, tile_cols, row_bytes)
    ck, name = UNTILIZE_COMPUTE, "dram_drain_untilize"

  return Program(
    cores=n, name=name, reader_kernel=rk, compute_kernel=ck, writer_kernel=wk,
    cbs=[CBConfig(index=0, dtype=buf.dtype, tiles=1), CBConfig(index=16, dtype=buf.dtype, tiles=1)],
    reader_args=tile_args, writer_args=tile_args, compute_args=compute_args, profile=False,
  ), logical_bytes

class Allocator:
  def __init__(self, fd: int, bank_tiles: list):
    self.bank_tiles = bank_tiles[:: Dram.TILES_PER_BANK]
    self.win = TLBWindow(fd, start=self.bank_tiles[0][1:], size=TLBWindow.SIZE_4G)
    self.next = Dram.WRITE_OFFSET

  def alloc(self, num_tiles: int, dtype: Dtype, name: str = "", shape: Shape | None = None) -> DramBuffer:
    num_banks = len(self.bank_tiles)
    pages_per_bank = (num_tiles + num_banks - 1) // num_banks
    addr = self.next
    self.next = align_up(addr + pages_per_bank * dtype.tile_size, Dram.ALIGNMENT)
    return DramBuffer(name=name, addr=addr, num_tiles=num_tiles, dtype=dtype, shape=shape)

  def alloc_write(self, data: bytes, dtype: Dtype, shape: Shape, name: str = "") -> DramBuffer:
    num_tiles = len(data) // dtype.tile_size
    buf = self.alloc(num_tiles, dtype, name=name, shape=shape)
    self.write(buf, data)
    return buf

  def barrier(self):
    for flag in Dram.BARRIER_FLAGS:
      for _, x, y in self.bank_tiles:
        self.win.target((x, y))
        self.win.write32(Dram.BARRIER_BASE, flag)
        while self.win.read32(Dram.BARRIER_BASE) != flag:
          pass

  def write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    view, ps, nb = memoryview(data), buf.page_size, len(self.bank_tiles)
    n_pages = (len(data) + ps - 1) // ps
    for bi, (_, x, y) in enumerate(self.bank_tiles):
      bank_data = b''.join(bytes(view[p * ps : p * ps + ps]) for p in range(bi, n_pages, nb))
      if not bank_data: continue
      self.win.target((x, y), mode=NocOrdering.POSTED)
      self.win.wc[buf.addr : buf.addr + len(bank_data)] = bank_data
    self.barrier()

  def read(self, buf: DramBuffer) -> bytes:
    result, ps, nb = bytearray(buf.size), buf.page_size, len(self.bank_tiles)
    n_pages = (buf.size + ps - 1) // ps
    for bi, (_, x, y) in enumerate(self.bank_tiles):
      bank_pages = list(range(bi, n_pages, nb))
      if not bank_pages: continue
      self.win.target((x, y), mode=NocOrdering.RELAXED)
      bank_data = self.win.wc[buf.addr : buf.addr + len(bank_pages) * ps]
      for i, p in enumerate(bank_pages):
        n = min(ps, buf.size - p * ps)
        result[p * ps : p * ps + n] = bank_data[i * ps : i * ps + n]
    return bytes(result)

  def read_raw_bank_pages(self, addr: int, page_size: int) -> bytes:
    result = bytearray(page_size * len(self.bank_tiles))
    for bank_idx, (_, x, y) in enumerate(self.bank_tiles):
      self.win.target((x, y), mode=NocOrdering.RELAXED)
      off = bank_idx * page_size
      result[off : off + page_size] = self.win.wc[addr : addr + page_size]
    return bytes(result)

  def close(self):
    self.win.close()
