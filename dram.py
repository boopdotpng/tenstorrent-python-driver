import os, mmap, fcntl, ctypes
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Callable
from defs import DRAM_ALIGNMENT, DRAM_BARRIER_BASE, Dram, TLBSize, PinPagesIn, PinPagesOutExtended, UnpinPagesIn, PIN_PAGES_NOC_DMA, IOCTL_PIN_PAGES, IOCTL_UNPIN_PAGES
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import _IO, align_up

TILE_R, TILE_C = 32, 32
FACE_R, FACE_C = 16, 16
TILE_ELEMS = TILE_R * TILE_C
Core = tuple[int, int]
DramTile = tuple[int, int, int]
Shape = tuple[int, ...]

class DType(Enum):
  float32 = 4
  float16 = 2
  int32 = 4
  uint32 = 4
  uint16 = 2
  int8 = 1
  fp8_e4m3 = 1
  uint8 = 1
  bfloat16 = uint16  # compatibility alias

def _face_transform(data: bytes, bpe: int, forward: bool) -> bytes:
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

def _grid_transform(data: bytes, bpe: int, shape: Shape, forward: bool) -> bytes:
  assert len(shape) >= 2, "Expected at least 2 dimensions"
  rows, cols = shape[-2], shape[-1]
  assert rows % TILE_R == 0 and cols % TILE_C == 0, "Last two dimensions must be tile-aligned"
  batch = 1
  for dim in shape[:-2]:
    batch *= dim
  slice_bytes = rows * cols * bpe
  assert len(data) == batch * slice_bytes, "Data size does not match shape"
  tile_rows, tile_cols = rows // TILE_R, cols // TILE_C
  out = bytearray(len(data))
  for b in range(batch):
    base = b * slice_bytes
    for tr in range(tile_rows):
      for tc in range(tile_cols):
        tile_idx = tr * tile_cols + tc
        for r in range(TILE_R):
          grid_off = base + ((tr * TILE_R + r) * cols + tc * TILE_C) * bpe
          tile_off = base + (tile_idx * TILE_ELEMS + r * TILE_C) * bpe
          src, dst = (grid_off, tile_off) if forward else (tile_off, grid_off)
          out[dst : dst + TILE_C * bpe] = data[src : src + TILE_C * bpe]
  return bytes(out)

def tilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  data = _grid_transform(data, bpe, shape, forward=True)
  return _face_transform(data, bpe, forward=True)

def untilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  data = _face_transform(data, bpe, forward=False)
  return _grid_transform(data, bpe, shape, forward=False)

class Sysmem:
  PCIE_NOC_XY = (24 << 6) | 19

  def __init__(self, fd: int, size: int = 1024 * 1024 * 1024):
    self.fd = fd
    page_size = os.sysconf("SC_PAGE_SIZE")
    self.size = (size + page_size - 1) & ~(page_size - 1)
    self.buf = mmap.mmap(
      -1,
      self.size,
      flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    self.buf[:] = b'\x00' * self.size  # fault in all pages
    buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
    pin_buf = bytearray(ctypes.sizeof(PinPagesIn) + ctypes.sizeof(PinPagesOutExtended))
    pin_in = PinPagesIn.from_buffer(pin_buf)
    pin_in.output_size_bytes = ctypes.sizeof(PinPagesOutExtended)
    pin_in.flags = PIN_PAGES_NOC_DMA
    pin_in.virtual_address = buf_addr
    pin_in.size = self.size
    fcntl.ioctl(self.fd, _IO(IOCTL_PIN_PAGES), pin_buf, True)
    pin_out = PinPagesOutExtended.from_buffer(pin_buf, ctypes.sizeof(PinPagesIn))
    self.noc_addr = pin_out.noc_address
    self._va = buf_addr

  def close(self):
    unpin_buf = bytearray(ctypes.sizeof(UnpinPagesIn))
    unpin_in = UnpinPagesIn.from_buffer(unpin_buf)
    unpin_in.virtual_address = self._va
    unpin_in.size = self.size
    fcntl.ioctl(self.fd, _IO(IOCTL_UNPIN_PAGES), unpin_buf, False)
    self.buf.close()

_DRAIN_KERNEL_SRC = r"""
#include <cstdint>

void kernel_main() {
  uint32_t dram_addr = get_arg_val<uint32_t>(0);
  uint32_t pcie_noc_xy = get_arg_val<uint32_t>(1);
  uint32_t sysmem_offset = get_arg_val<uint32_t>(2);
  uint32_t tile_offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
  uint32_t page_size = get_arg_val<uint32_t>(5);
  uint32_t sysmem_local_offset = get_arg_val<uint32_t>(6);
  constexpr uint32_t cb_id = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = dram_addr,
    .page_size = page_size,
    .data_format = DataFormat::Float16_b,
  };
  uint64_t pcie_base = ((uint64_t)pcie_noc_xy << 36) | (1ULL << 60);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t tile_id = tile_offset + i;
    cb_reserve_back(cb_id, 1);
    uint32_t l1_addr = get_write_ptr(cb_id);
    noc_async_read_tile(tile_id, dram, l1_addr);
    noc_async_read_barrier();
    uint64_t dst = pcie_base + sysmem_offset + sysmem_local_offset + (uint64_t)tile_id * page_size;
    noc_async_write(l1_addr, dst, page_size);
    noc_async_write_barrier();
    cb_push_back(cb_id, 1);
    cb_wait_front(cb_id, 1);
    cb_pop_front(cb_id, 1);
  }
}
"""

_FILL_KERNEL_SRC = r"""
#include <cstdint>

void kernel_main() {
  uint32_t dram_addr = get_arg_val<uint32_t>(0);
  uint32_t pcie_noc_xy = get_arg_val<uint32_t>(1);
  uint32_t sysmem_offset = get_arg_val<uint32_t>(2);
  uint32_t tile_offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
  uint32_t page_size = get_arg_val<uint32_t>(5);
  constexpr uint32_t cb_id = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = dram_addr,
    .page_size = page_size,
    .data_format = DataFormat::Float16_b,
  };
  uint64_t pcie_base = ((uint64_t)pcie_noc_xy << 36) | (1ULL << 60);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t tile_id = tile_offset + i;
    cb_reserve_back(cb_id, 1);
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint64_t src = pcie_base + sysmem_offset + (uint64_t)tile_id * page_size;
    noc_async_read(src, l1_addr, page_size);
    noc_async_read_barrier();
    noc_async_write_tile(tile_id, dram, l1_addr);
    noc_async_write_barrier();
    cb_push_back(cb_id, 1);
    cb_wait_front(cb_id, 1);
    cb_pop_front(cb_id, 1);
  }
}
"""

@lru_cache(maxsize=1)
def _drain_kernel():
  from codegen import Compiler
  return Compiler()._compile_dataflow(_DRAIN_KERNEL_SRC, "ncrisc", noc_index=0)

@lru_cache(maxsize=1)
def _fill_kernel():
  from codegen import Compiler
  return Compiler()._compile_dataflow(_FILL_KERNEL_SRC, "ncrisc", noc_index=0)

@dataclass(frozen=True)
class DramBuffer:
  name: str | None
  addr: int
  size: int
  page_size: int
  dtype: DType | None = None
  shape: Shape | None = None

class DramAllocator:
  def __init__(self, fd: int, dram_tiles: list[DramTile], run_fn: Callable | None = None, sync_fn: Callable | None = None,
               enable_sysmem: bool = False):
    self.fd = fd
    self.bank_tiles = dram_tiles[:: Dram.TILES_PER_BANK]
    self.next = Dram.DRAM_WRITE_OFFSET
    self.max_page_size = 2 * 1024 * 1024
    self.win = TLBWindow(self.fd, TLBSize.GiB_4)
    self._run_fn = run_fn
    self._sync_fn = sync_fn
    self.sysmem = Sysmem(self.fd) if enable_sysmem and run_fn is not None else None
    self._pending_reads: list[tuple[DramBuffer, bool]] = []
    self._inflight_fast_reads: list[tuple[DramBuffer, int]] = []
    self._inflight_manual_reads: list[DramBuffer] = []
    self._ready_reads: dict[int, list[bytes]] = {}

  def alloc(self, size: int, name: str | None = None, page_size: int | None = None, dtype: DType | None = None,
            shape: Shape | None = None) -> DramBuffer:
    num_banks = len(self.bank_tiles)
    page_size = min(
      self.max_page_size, page_size or align_up((size + num_banks - 1) // num_banks, DRAM_ALIGNMENT)
    )
    page_size = align_up(page_size, DRAM_ALIGNMENT)
    addr = self.next
    pages = (size + page_size - 1) // page_size
    pages_per_bank = (pages + num_banks - 1) // num_banks
    self.next = align_up(self.next + pages_per_bank * page_size, DRAM_ALIGNMENT)
    return DramBuffer(name=name, addr=addr, size=size, page_size=page_size, dtype=dtype, shape=shape)

  def alloc_write(self, data: bytes, name: str | None = None, page_size: int | None = None, dtype: DType | None = None,
                  shape: Shape | None = None) -> DramBuffer:
    buf = self.alloc(len(data), name=name, page_size=page_size, dtype=dtype, shape=shape)
    self.write(buf, data)
    return buf

  def _for_each_page(self, buf: DramBuffer, size: int, mode: TLBMode, fn: Callable[[int, int], None]) -> list[DramTile]:
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

  def barrier(self, tiles: list[DramTile]):
    for flag in Dram.BARRIER_FLAGS:
      for _, x, y in tiles:
        self.win.configure(TLBConfig(addr=0, start=(x, y), end=(x, y), noc=0, mcast=False, mode=TLBMode.STRICT))
        self.win.write32(DRAM_BARRIER_BASE, flag)
        while self.win.read32(DRAM_BARRIER_BASE) != flag:
          pass

  def write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    assert buf.page_size >= DRAM_ALIGNMENT and (buf.page_size & (DRAM_ALIGNMENT - 1)) == 0
    if buf.dtype is not None:
      assert buf.shape is not None, "Shape is required when dtype is set"
      data = tilize(data, buf.dtype.value, buf.shape)
    if self.sysmem is not None and len(data) <= self.sysmem.size:
      self._write_fast(buf, data)
    else:
      self._write_slow(buf, data)

  def _run_transfer_kernel(self, buf: DramBuffer, kernel, size: int, extra_args: list[int] = []):
    assert self.sysmem is not None and self._run_fn is not None
    from device_runtime import Program, DataflowLaunch, TileGrid
    n_tiles = (size + buf.page_size - 1) // buf.page_size
    sysmem_offset = self.sysmem.noc_addr & ((1 << 36) - 1)
    dev = getattr(self._run_fn, "__self__", None)
    all_cores = list(dev.dispatchable_cores) if dev and hasattr(dev, "dispatchable_cores") else list(TileGrid.TENSIX)
    tiles_per_core = (n_tiles + len(all_cores) - 1) // len(all_cores)
    cores = all_cores[:min(len(all_cores), n_tiles)]
    base_args = [buf.addr, Sysmem.PCIE_NOC_XY, sysmem_offset]

    def reader_args(core_idx: int, core_xy: Core, n_cores: int) -> list[int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, n_tiles - start) if start < n_tiles else 0
      return base_args + [start, count, buf.page_size] + extra_args

    self._run_fn(Program(
      dataflow=[DataflowLaunch(cores=cores, reader=kernel, reader_rt_args=reader_args, writer_rt_args=[])],
      compute=None, compute_rt_args=[], cbs=[0],
      tile_size=buf.page_size, num_pages=2, cores=len(cores),
    ), timing=False, wait=False)

  def _write_fast(self, buf: DramBuffer, data: bytes):
    self.sysmem.buf[:len(data)] = data
    self._run_transfer_kernel(buf, _fill_kernel(), len(data))

  def _write_slow(self, buf: DramBuffer, data: bytes):
    # Slow TLB writes bypass CQ ordering; drain any queued device work first.
    if self._sync_fn is not None:
      self._sync_fn()
    view = memoryview(data)

    def do_write(addr: int, off: int):
      page = view[off : off + buf.page_size]
      self.win.wc[addr : addr + len(page)] = page

    touched = self._for_each_page(buf, len(data), TLBMode.POSTED, do_write)
    self.barrier(touched)

  def read(self, buf: DramBuffer) -> bytes:
    self.enqueue_read(buf)
    if self._sync_fn is None:
      raise RuntimeError("read requires a sync_fn to finalize queued DRAM transfers")
    self._sync_fn()
    key = id(buf)
    if key not in self._ready_reads or not self._ready_reads[key]:
      raise RuntimeError(f"readback for buffer '{buf.name or hex(buf.addr)}' was not finalized by sync")
    out = self._ready_reads[key].pop(0)
    if not self._ready_reads[key]:
      del self._ready_reads[key]
    return out

  def enqueue_read(self, buf: DramBuffer):
    use_fast = self.sysmem is not None and buf.size <= self.sysmem.size
    self._pending_reads.append((buf, use_fast))

  def prepare_sync(self):
    if not self._pending_reads:
      return
    next_off = 0
    for buf, use_fast in self._pending_reads:
      if use_fast:
        assert self.sysmem is not None
        n_tiles = (buf.size + buf.page_size - 1) // buf.page_size
        span = n_tiles * buf.page_size
        off = align_up(next_off, buf.page_size)
        if off + span <= self.sysmem.size:
          self._enqueue_read_fast(buf, sysmem_local_offset=off)
          self._inflight_fast_reads.append((buf, off))
          next_off = off + span
          continue
      self._inflight_manual_reads.append(buf)
    self._pending_reads.clear()

  def finish_sync(self):
    for buf, off in self._inflight_fast_reads:
      self._store_ready_read(buf, self._read_fast_result(buf, off))
    self._inflight_fast_reads.clear()
    for buf in self._inflight_manual_reads:
      self._store_ready_read(buf, self._read_slow_now(buf))
    self._inflight_manual_reads.clear()

  def _store_ready_read(self, buf: DramBuffer, data: bytes):
    self._ready_reads.setdefault(id(buf), []).append(data)

  def _enqueue_read_fast(self, buf: DramBuffer, sysmem_local_offset: int):
    self._run_transfer_kernel(buf, _drain_kernel(), buf.size, extra_args=[sysmem_local_offset])

  def _read_fast_result(self, buf: DramBuffer, off: int) -> bytes:
    assert self.sysmem is not None
    result = bytes(self.sysmem.buf[off : off + buf.size])
    if buf.dtype is not None:
      assert buf.shape is not None, "Shape is required when dtype is set"
      return untilize(result, buf.dtype.value, buf.shape)
    return result

  def _read_slow_now(self, buf: DramBuffer) -> bytes:
    result = bytearray(buf.size)

    def do_read(addr: int, off: int):
      remaining = buf.size - off
      n = min(buf.page_size, remaining)
      result[off : off + n] = self.win.wc[addr : addr + n]

    self._for_each_page(buf, buf.size, TLBMode.RELAXED, do_read)
    if buf.dtype is not None:
      assert buf.shape is not None, "Shape is required when dtype is set"
      return untilize(bytes(result), buf.dtype.value, buf.shape)
    return bytes(result)

  def close(self):
    if self.sysmem is not None: self.sysmem.close()
    self.win.free()
