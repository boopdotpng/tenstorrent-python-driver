from __future__ import annotations

import ctypes, time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal
import fcntl, mmap, os, struct

from autogen import *
from autogen import _IO
from dispatch import CBConfig, Core, CoreArgs, Dtype, MathFidelity, Program, Rect, FAST_CQ_NUM_CIRCULAR_BUFFERS
from dispatch import Go, McastWrite, SetGoSignalNocData, UnicastWrite, Wait
from dispatch import build_commands, mcast_rects, noc_mcast_xy, resolve_args

def noc_xy(x: int, y: int) -> int: return ((y << 6) | x) & 0xFFFF

class NocOrdering(Enum):
  RELAXED = 0
  STRICT = 1
  POSTED = 2

class TLBWindow:
  SIZE_2M = 1 << 21
  SIZE_4G = 1 << 32

  def __init__(self, fd: int, start: Core | None, end: Core | None = None, addr: int = 0,
               mode: NocOrdering = NocOrdering.STRICT, size: int = SIZE_2M):
    self.fd, self.size = fd, size

    # AllocateTlbIn -> AllocateTlbOut
    buf = bytearray(48)
    struct.pack_into("<Q", buf, 0, size)
    fcntl.ioctl(fd, _IO(IOCTL_ALLOCATE_TLB), buf, True)
    self._id, _, uc_off, wc_off, _ = struct.unpack_from("<IIQQQ", buf, 16)

    rw = mmap.PROT_READ | mmap.PROT_WRITE
    self.uc = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=rw, offset=uc_off)
    self.wc = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=rw, offset=wc_off)

    self.target(start, end, addr=addr, mode=mode)

  def target(self, start: Core, end: Core | None = None, addr: int = 0,
             mode: NocOrdering = NocOrdering.STRICT) -> None:
    end = end or start
    mcast = end != start
    # NocTlbConfig(u64 addr, 4×u16 end/start, 8×u8 flags, 2×u32 res)
    buf = struct.pack(
      "<IIQ4H8B2I",
      self._id, 0, addr, end[0], end[1], start[0], start[1], 0, int(mcast), mode.value,
      0, 0, 0, 0, 0, 0, 0,
    )
    fcntl.ioctl(self.fd, _IO(IOCTL_CONFIGURE_TLB), bytearray(buf), False)

  def read32(self, offset: int) -> int:
    return int.from_bytes(self.uc[offset:offset + 4], "little")

  def write32(self, offset: int, value: int) -> None:
    self.uc[offset:offset + 4] = value.to_bytes(4, "little")

  def write(self, addr: int, data: bytes, wc: bool = False) -> None:
    view = self.wc if wc else self.uc
    view[addr:addr + len(data)] = data

  def close(self) -> None:
    self.uc.close()
    self.wc.close()
    fcntl.ioctl(self.fd, _IO(IOCTL_FREE_TLB), bytearray(struct.pack("<I", self._id)), False)

  def __enter__(self) -> TLBWindow: return self
  def __exit__(self, *_): self.close()

USE_USB_DISPATCH = os.environ.get("TT_USB") == "1"

class Sysmem:
  PCIE_NOC_XY = (24 << 6) | 19

  def __init__(self, fd: int, size: int = 1 << 30, shared: bool = False, populate: bool = False):
    self.fd = fd
    page_size = os.sysconf("SC_PAGE_SIZE")
    self.size = (size + page_size - 1) & ~(page_size - 1)
    flags = (mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE) | mmap.MAP_ANONYMOUS
    if populate and hasattr(mmap, "MAP_POPULATE"):
      flags |= mmap.MAP_POPULATE
    self.buf = mmap.mmap(
      -1, self.size,
      flags=flags,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    self._va = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
    # PinPagesIn(output_size u32, flags u32, va u64, size u64) -> PinPagesOutExtended(phys u64, noc u64)
    buf = bytearray(40)
    struct.pack_into("<IIQQ", buf, 0, 16, PIN_PAGES_NOC_DMA, self._va, self.size)
    fcntl.ioctl(self.fd, _IO(IOCTL_PIN_PAGES), buf, True)
    _, self.noc_addr = struct.unpack_from("<QQ", buf, 24)

  def close(self):
    # UnpinPagesIn(va u64, size u64, reserved u64)
    buf = bytearray(24)
    struct.pack_into("<QQQ", buf, 0, self._va, self.size, 0)
    fcntl.ioctl(self.fd, _IO(IOCTL_UNPIN_PAGES), buf, False)
    self.buf.close()

Shape = tuple[int, ...]
TILE_R, TILE_C, FACE_R, FACE_C = 32, 32, 16, 16

def _face_transform(data: bytes, bpe: int, forward: bool) -> bytes:
  n_tiles = len(data) // (TILE_R * TILE_C * bpe)
  out = bytearray(len(data))
  for t in range(n_tiles):
    toff = t * TILE_R * TILE_C * bpe
    for fr in range(2):
      for fc in range(2):
        fi = fr * 2 + fc
        for r in range(FACE_R):
          rm = toff + ((fr * FACE_R + r) * TILE_C + fc * FACE_C) * bpe
          tf = toff + (fi * FACE_R * FACE_C + r * FACE_C) * bpe
          src, dst = (rm, tf) if forward else (tf, rm)
          out[dst:dst + FACE_C * bpe] = data[src:src + FACE_C * bpe]
  return bytes(out)

def _grid_transform(data: bytes, bpe: int, shape: Shape, forward: bool) -> bytes:
  rows, cols = shape[-2], shape[-1]
  assert rows % TILE_R == 0 and cols % TILE_C == 0
  batch = 1
  for d in shape[:-2]: batch *= d
  slice_bytes = rows * cols * bpe
  assert len(data) == batch * slice_bytes
  tr, tc = rows // TILE_R, cols // TILE_C
  out = bytearray(len(data))
  for b in range(batch):
    base = b * slice_bytes
    for ri in range(tr):
      for ci in range(tc):
        ti = ri * tc + ci
        for r in range(TILE_R):
          grid_off = base + ((ri * TILE_R + r) * cols + ci * TILE_C) * bpe
          tile_off = base + (ti * TILE_R * TILE_C + r * TILE_C) * bpe
          src, dst = (grid_off, tile_off) if forward else (tile_off, grid_off)
          out[dst:dst + TILE_C * bpe] = data[src:src + TILE_C * bpe]
  return bytes(out)

def tilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  return _face_transform(_grid_transform(data, bpe, shape, True), bpe, True)

def untilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  return _grid_transform(_face_transform(data, bpe, False), bpe, shape, False)

@dataclass
class DramBuffer:
  name: str
  addr: int
  num_tiles: int
  dtype: Dtype
  shape: Shape | None = None

  @property
  def page_size(self) -> int: return self.dtype.tile_size

  @property
  def size(self) -> int: return self.num_tiles * self.page_size

class Allocator:
  def __init__(self, fd: int, bank_tiles: list):
    self.bank_tiles = bank_tiles[::Dram.TILES_PER_BANK]
    self.win = TLBWindow(fd, start=self.bank_tiles[0][1:], size=TLBWindow.SIZE_4G)
    self.next = Dram.DRAM_WRITE_OFFSET

  def alloc(self, num_tiles: int, dtype: Dtype, name: str = "", shape: Shape | None = None) -> DramBuffer:
    num_banks = len(self.bank_tiles)
    pages_per_bank = (num_tiles + num_banks - 1) // num_banks
    addr = self.next
    self.next = align_up(addr + pages_per_bank * dtype.tile_size, DRAM_ALIGNMENT)
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
        self.win.write32(DRAM_BARRIER_BASE, flag)
        while self.win.read32(DRAM_BARRIER_BASE) != flag: pass

  def write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    view = memoryview(data)
    num_banks = len(self.bank_tiles)
    pages = (len(data) + buf.page_size - 1) // buf.page_size
    for bank_idx, (_, x, y) in enumerate(self.bank_tiles):
      if bank_idx >= pages: break
      self.win.target((x, y), mode=NocOrdering.POSTED)
      local_page = 0
      for page_idx in range(bank_idx, pages, num_banks):
        addr = buf.addr + local_page * buf.page_size
        off = page_idx * buf.page_size
        page = view[off:off + buf.page_size]
        self.win.wc[addr:addr + len(page)] = page
        local_page += 1
    self.barrier()

  def read(self, buf: DramBuffer) -> bytes:
    result = bytearray(buf.size)
    num_banks = len(self.bank_tiles)
    pages = (buf.size + buf.page_size - 1) // buf.page_size
    for bank_idx, (_, x, y) in enumerate(self.bank_tiles):
      if bank_idx >= pages: break
      self.win.target((x, y), mode=NocOrdering.RELAXED)
      local_page = 0
      for page_idx in range(bank_idx, pages, num_banks):
        addr = buf.addr + local_page * buf.page_size
        off = page_idx * buf.page_size
        n = min(buf.page_size, buf.size - off)
        result[off:off + n] = self.win.wc[addr:addr + n]
        local_page += 1
    return bytes(result)

  def close(self):
    self.win.close()

class TileGrid:
  ARC = (8, 0)
  # See tt-isa-documentation. Cols 8,9 are l2cpu & dram.
  TENSIX_X = (*range(1, 8), *range(10, 15))
  WORKER_CORES = [(x, y) for x in TENSIX_X for y in range(2, 12)]

CQ_CMD_SIZE = 16  # all prefetch/dispatch commands are 16 bytes


# ── slow dispatch ──────────────────────────────────────────────────────────────

def _slow_dispatch(win: TLBWindow, commands: list) -> None:
  """Execute dispatch commands via direct TLB writes."""
  for cmd in commands:
    match cmd:
      case McastWrite(rects=rects, addr=addr, data=data):
        for x0, x1, y0, y1 in rects:
          win.target((x0, y0), (x1, y1))
          win.write(addr, data)
      case UnicastWrite(cores=cores, addr=addr, data=data):
        for core, d in zip(cores, data):
          win.target(core)
          win.write(addr, d)
      case SetGoSignalNocData():
        pass
      case Go(cores=cores):
        go = GoMsg()
        go.bits.signal = DevMsgs.RUN_MSG_GO
        go_blob = struct.pack("<I", go.all)
        for x0, x1, y0, y1 in mcast_rects(cores):
          win.target((x0, y0), (x1, y1))
          win.uc[TensixL1.GO_MSG:TensixL1.GO_MSG + 4] = go_blob
      case Wait(cores=cores):
        for x, y in cores:
          win.target((x, y))
          deadline = time.perf_counter() + 10.0
          while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
            if time.perf_counter() > deadline:
              raise TimeoutError(f"timeout waiting for core ({x}, {y}) — try tt-smi -r")

# ── CQ (fast dispatch command queue) ──────────────────────────────────────────

class CQ:
  def __init__(self, fd: int):
    self.fd = fd
    flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS
    if hasattr(mmap, "MAP_POPULATE"):
      flags |= mmap.MAP_POPULATE
    self.sysmem = mmap.mmap(-1, HOST_SYSMEM_SIZE, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    self._sysmem_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.sysmem))
    if (self._sysmem_addr % PAGE_SIZE) != 0 or (HOST_SYSMEM_SIZE % PAGE_SIZE) != 0:
      raise RuntimeError("CQ sysmem must be page-aligned and page-sized")

    # PinPagesIn -> PinPagesOutExtended
    buf = bytearray(40)
    struct.pack_into("<IIQQ", buf, 0, 16, PIN_PAGES_NOC_DMA, self._sysmem_addr, HOST_SYSMEM_SIZE)
    fcntl.ioctl(self.fd, _IO(IOCTL_PIN_PAGES), buf, True)
    _, self.noc_addr = struct.unpack_from("<QQ", buf, 24)
    if (self.noc_addr & FastDispatch.PCIE_NOC_BASE) != FastDispatch.PCIE_NOC_BASE:
      raise RuntimeError(f"bad NOC sysmem address: 0x{self.noc_addr:x}")
    self.noc_local = self.noc_addr - FastDispatch.PCIE_NOC_BASE
    if self.noc_local > 0xFFFF_FFFF:
      raise RuntimeError(f"CQ sysmem NOC offset too large: 0x{self.noc_local:x}")
    self._stream = bytearray()
    self._sizes: list[int] = []
    self._issue_wr = 0
    self._prefetch_q_wr_idx = 0
    self._event_id = 0
    self._completion_base_16b = ((self.noc_local + HOST_COMPLETION_BASE) >> 4) & 0x7FFF_FFFF
    self._completion_page_16b = PAGE_SIZE >> 4
    self._completion_end_16b = self._completion_base_16b + (HOST_COMPLETION_SIZE >> 4)
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    # TLB windows for prefetch queue and dispatch core (set up by Device)
    self._prefetch_win: TLBWindow | None = None
    self._dispatch_win: TLBWindow | None = None

  def _relay_inline(self, inner: bytes) -> None:
    stride = align_up(CQ_CMD_SIZE + len(inner), FastDispatch.PCIE_ALIGNMENT)
    # prefetch cmd: cmd_id(1) + dispatcher_type(1) + pad(2) + length(4) + stride(4) + pad(4) = 16
    hdr = struct.pack("<BBHII", CQ_PREFETCH_CMD_RELAY_INLINE, 0, 0, len(inner), stride)
    hdr = hdr.ljust(CQ_CMD_SIZE, b"\0")
    record = hdr + inner + b"\0" * (stride - CQ_CMD_SIZE - len(inner))
    self._stream.extend(record)
    self._sizes.append(len(record) >> 4)

  def _read_prefetch_entry(self, idx: int) -> int:
    off = DEV_PREFETCH_Q_BASE + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    return struct.unpack("<H", self._prefetch_win.uc[off:off + 2])[0]

  def _wait_prefetch_slot_free(self, idx: int, timeout_s: float = 1.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while self._read_prefetch_entry(idx) != 0:
      if time.perf_counter() > deadline:
        raise TimeoutError("timeout waiting for prefetch queue slot")

  def _issue_write(self, record: bytes) -> None:
    self._issue_wr = align_up(self._issue_wr, FastDispatch.PCIE_ALIGNMENT)
    if self._issue_wr + len(record) > HOST_ISSUE_SIZE:
      self._issue_wr = 0
    base = HOST_ISSUE_BASE + self._issue_wr
    self.sysmem[base:base + len(record)] = record
    self._issue_wr += len(record)
    idx = self._prefetch_q_wr_idx
    self._wait_prefetch_slot_free(idx)
    off = DEV_PREFETCH_Q_BASE + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self._prefetch_win.uc[off:off + 2] = struct.pack("<H", len(record) >> 4)
    entries = DEV_PREFETCH_Q_SIZE // FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self._prefetch_q_wr_idx = (idx + 1) % entries

  def write_packed(self, cores: list[Core], addr: int, data: bytes | list[bytes]) -> None:
    uniform = isinstance(data, bytes)
    count = len(cores)
    payload_size = len(data) if uniform else len(data[0])
    flags = CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE if uniform else 0
    # cmd_id(1) + flags(1) + count(2) + write_offset_index(2) + size(2) + addr(4) + pad(4) = 16
    hdr = struct.pack("<BBHHHI", CQ_DISPATCH_CMD_WRITE_PACKED, flags, count, 0, payload_size, addr)
    hdr = hdr.ljust(CQ_CMD_SIZE, b"\0")
    sub = b"".join(struct.pack("<I", noc_xy(x, y)) for x, y in cores)
    sub = sub.ljust(align_up(len(sub), FastDispatch.L1_ALIGNMENT), b"\0")
    if uniform:
      data_section = bytes(data).ljust(align_up(payload_size, FastDispatch.L1_ALIGNMENT), b"\0")
    else:
      stride = align_up(payload_size, FastDispatch.L1_ALIGNMENT)
      data_section = b"".join(d.ljust(stride, b"\0") for d in data)
    self._relay_inline(hdr + sub + data_section)

  def write_packed_large(self, rects: list[Rect], addr: int, data: bytes) -> None:
    data_padded = bytes(data).ljust(align_up(len(data), FastDispatch.L1_ALIGNMENT), b"\0")
    max_batch = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS
    for batch_start in range(0, len(rects), max_batch):
      batch = rects[batch_start:batch_start + max_batch]
      # cmd_id(1) + type(1) + count(2) + alignment(2) + write_offset_index(2) + pad(8) = 16
      hdr = struct.pack("<BBHHH", CQ_DISPATCH_CMD_WRITE_PACKED_LARGE, 2, len(batch), FastDispatch.L1_ALIGNMENT, 0)
      hdr = hdr.ljust(CQ_CMD_SIZE, b"\0")
      sub = b"".join(
        # noc_xy_addr(4) + addr(4) + length_minus1(2) + num_mcast_dests(1) + flags(1) = 12
        struct.pack("<IIHBB", noc_mcast_xy(rect)[0], addr, len(data) - 1,
                    noc_mcast_xy(rect)[1], CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK)
        for rect in batch
      )
      sub = sub.ljust(align_up(len(sub), FastDispatch.L1_ALIGNMENT), b"\0")
      self._relay_inline(hdr + sub + data_padded * len(batch))

  def set_go_signal_noc_data(self, cores: list[Core]) -> None:
    # cmd_id(1) + pad(3) + num_words(4) + pad(8) = 16
    hdr = struct.pack("<BBHI", CQ_DISPATCH_CMD_SET_GO_SIGNAL_NOC_DATA, 0, 0, len(cores))
    hdr = hdr.ljust(CQ_CMD_SIZE, b"\0")
    payload = b"".join(struct.pack("<I", noc_xy(x, y)) for x, y in cores)
    self._relay_inline(hdr + payload)

  def send_go_signal(self, go_word: int, stream: int, count: int, num_unicast: int) -> None:
    # cmd_id(1) + go_signal(4) + mcast_offset(1) + num_unicast(1) + noc_start_idx(1) + wait_count(4) + wait_stream(4) = 16
    hdr = struct.pack("<BIBBBII", CQ_DISPATCH_CMD_SEND_GO_SIGNAL, go_word & 0xFFFFFFFF,
                      CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET, num_unicast & 0xFF, 0, count & 0xFFFFFFFF, stream & 0xFFFFFFFF)
    self._relay_inline(hdr)

  def wait_stream(self, stream: int, count: int, clear: bool = True) -> None:
    flags = CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM
    if clear: flags |= CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM
    # cmd_id(1) + flags(1) + stream(2) + addr(4) + count(4) + pad(4) = 16
    hdr = struct.pack("<BBHII", CQ_DISPATCH_CMD_WAIT, flags, stream, 0, count)
    hdr = hdr.ljust(CQ_CMD_SIZE, b"\0")
    self._relay_inline(hdr)

  def host_event(self, event_id: int) -> None:
    payload = struct.pack("<I", event_id & 0xFFFFFFFF).ljust(FastDispatch.L1_ALIGNMENT, b"\0")
    # cmd_id(1) + is_event(1) + pad(6) + length(8) = 16
    hdr = struct.pack("<BBHIQ", CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST, 1, 0, 0, CQ_CMD_SIZE + len(payload))
    self._relay_inline(hdr + payload)

  def timestamp(self, noc_xy_addr: int, addr: int) -> None:
    # cmd_id(1) + pad(3) + noc_xy_addr(4) + addr(4) + pad(4) = 16
    hdr = struct.pack("<BBHII", CQ_DISPATCH_CMD_TIMESTAMP, 0, 0, noc_xy_addr, addr)
    hdr = hdr.ljust(CQ_CMD_SIZE, b"\0")
    self._relay_inline(hdr)

  def flush(self) -> None:
    offset = 0
    for size_16b in self._sizes:
      size = size_16b << 4
      self._issue_write(self._stream[offset:offset + size])
      offset += size
    self._stream.clear()
    self._sizes.clear()

  def wait_completion(self, event_id: int, timeout_s: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while True:
      wr_raw = struct.unpack("<I", self.sysmem[FastDispatch.HOST_COMPLETION_Q_WR_OFF:FastDispatch.HOST_COMPLETION_Q_WR_OFF + 4])[0]
      wr_16b = wr_raw & 0x7FFF_FFFF
      wr_toggle = (wr_raw >> 31) & 0x1
      if (wr_16b != self._completion_rd_16b) or (wr_toggle != self._completion_rd_toggle):
        off = (self._completion_rd_16b << 4) - self.noc_local
        got = struct.unpack("<I", self.sysmem[off + CQ_CMD_SIZE:off + CQ_CMD_SIZE + 4])[0]
        self._completion_rd_16b += self._completion_page_16b
        if self._completion_rd_16b >= self._completion_end_16b:
          self._completion_rd_16b = self._completion_base_16b
          self._completion_rd_toggle ^= 1
        raw = (self._completion_rd_16b & 0x7FFF_FFFF) | (self._completion_rd_toggle << 31)
        self._dispatch_win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, raw)
        self.sysmem[FastDispatch.HOST_COMPLETION_Q_RD_OFF:FastDispatch.HOST_COMPLETION_Q_RD_OFF + 4] = struct.pack("<I", raw)
        if got != (event_id & 0xFFFFFFFF):
          raise RuntimeError(f"completion event mismatch: got {got}, expected {event_id}")
        return
      if time.perf_counter() > deadline:
        raise TimeoutError(f"timeout waiting for completion event {event_id} — try tt-smi -r")
      time.sleep(0.0002)

  def reset_run_state(self) -> None:
    self._issue_wr = 0
    self._prefetch_q_wr_idx = 0
    self._stream.clear()
    self._sizes.clear()
    self._prefetch_win.write32(DEV_PREFETCH_Q_RD_PTR_ADDR, DEV_PREFETCH_Q_BASE + DEV_PREFETCH_Q_SIZE)
    self._prefetch_win.write32(DEV_PREFETCH_Q_PCIE_RD_PTR_ADDR, (self.noc_local + HOST_ISSUE_BASE) & 0xFFFFFFFF)
    for i in range(FastDispatch.PREFETCH_Q_ENTRIES_WORKER_DEFAULT):
      off = DEV_PREFETCH_Q_BASE + i * FastDispatch.PREFETCH_Q_ENTRY_BYTES
      self._prefetch_win.uc[off:off + 2] = b"\0\0"
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    base_val = self._completion_base_16b
    self._dispatch_win.write32(DEV_COMPLETION_Q_WR_PTR_ADDR, base_val)
    self._dispatch_win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, base_val)
    self.sysmem[FastDispatch.HOST_COMPLETION_Q_WR_OFF:FastDispatch.HOST_COMPLETION_Q_WR_OFF + 4] = struct.pack("<I", base_val)
    self.sysmem[FastDispatch.HOST_COMPLETION_Q_RD_OFF:FastDispatch.HOST_COMPLETION_Q_RD_OFF + 4] = struct.pack("<I", base_val)

  def close(self):
    if self._prefetch_win: self._prefetch_win.close()
    if self._dispatch_win: self._dispatch_win.close()
    buf = bytearray(24)
    struct.pack_into("<QQQ", buf, 0, self._sysmem_addr, HOST_SYSMEM_SIZE, 0)
    fcntl.ioctl(self.fd, _IO(IOCTL_UNPIN_PAGES), buf, False)
    self.sysmem.close()

# ── fast dispatch ──────────────────────────────────────────────────────────────

def _fast_enqueue(cq: CQ, commands: list, go_word: int) -> None:
  """Translate dispatch commands into CQ command stream."""
  for cmd in commands:
    match cmd:
      case McastWrite(rects=rects, addr=addr, data=data):
        cq.write_packed_large(rects, addr, data)
      case UnicastWrite(cores=cores, addr=addr, data=data):
        cq.write_packed(cores, addr, data)
      case SetGoSignalNocData(cores=cores):
        cq.set_go_signal_noc_data(cores)
      case Go(cores=cores):
        cq.wait_stream(48, 0)
        cq.send_go_signal(go_word, stream=48, count=0, num_unicast=len(cores))
        cq.wait_stream(48, len(cores))
      case Wait():
        pass  # handled by Go's wait_stream in fast dispatch

# ── Device ─────────────────────────────────────────────────────────────────────

_INIT_SCRATCH = {
  "brisc": TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH,
  "ncrisc": TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH,
  "trisc0": TensixL1.TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
  "trisc1": TensixL1.TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
  "trisc2": TensixL1.TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
}

class Device:
  _CQ_PREFETCH_CORE = (14, 2)
  _CQ_DISPATCH_CORE = (14, 3)

  def _select_dispatch_core_pair(self) -> tuple[Core, Core]:
    pair = (self._CQ_PREFETCH_CORE, self._CQ_DISPATCH_CORE)
    missing = [core for core in pair if core not in self.cores]
    if missing:
      raise RuntimeError(f"fixed CQ cores unavailable: {missing}")
    return pair

  def __init__(self, device: int = 0):
    self.device = device
    self.path = f"/dev/tenstorrent/{device}"
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    card_type = Path(f"/sys/class/tenstorrent/tenstorrent!{device}/tt_card_type")
    self.arch = card_type.read_text().strip()
    if self.arch != "p100a":
      os.close(self.fd)
      raise SystemExit(f"unsupported blackhole device {self.arch}; p100a only for now")

    self.harvested_dram = self._get_harvested_dram_bank()
    self._set_active_dram_tiles()
    self.dram = Allocator(self.fd, self.dram_tiles)
    self._init_timestamp_dram()
    self._dispatch_mode = DevMsgs.DISPATCH_MODE_HOST if USE_USB_DISPATCH else DevMsgs.DISPATCH_MODE_DEV
    self._use_fast_dispatch = not USE_USB_DISPATCH

    # Reserve dispatch cores only when the runtime is actually using fast dispatch.
    self.cores = list(TileGrid.WORKER_CORES)
    if self._use_fast_dispatch:
      self._prefetch_core, self._dispatch_core = self._select_dispatch_core_pair()
      self.cores = [c for c in self.cores if c not in {self._prefetch_core, self._dispatch_core}]

    # Compiler and firmware
    from compiler import Compiler
    self.compiler = Compiler()
    self._upload_firmware()

    # Fast dispatch expects the CQ host buffer to land in the second PCIe aperture.
    # Reserve the first pinned host window up front, matching the source-of-truth
    # runtime's dram allocator initialization order.
    self._dram_sysmem: Sysmem | None = Sysmem(self.fd) if self._use_fast_dispatch else None

    # CQ for device-side fast dispatch
    self.cq: CQ | None = None
    if self._use_fast_dispatch:
      self.cq = CQ(self.fd)
      self.cq._prefetch_win = TLBWindow(self.fd, start=self._prefetch_core)
      self.cq._dispatch_win = TLBWindow(self.fd, start=self._dispatch_core)
      self._start_dispatch_cores()

    self._programs: list = []
    self.last_profile: dict | None = None
    self._profiler_initialized = False

    # Profiler DRAM buffer (fast dispatch only)
    from compiler import PROFILER
    self._profiler = PROFILER
    self._profiler_dram_buf: DramBuffer | None = None
    self._profiler_flat_ids: dict[Core, int] = {}
    self._profiler_core_count_per_dram = 0
    if self._profiler and self._use_fast_dispatch:
      self._init_profiler_dram()

  # ── DRAM bank detection ────────────────────────────────────────────────────

  def _set_active_dram_tiles(self) -> None:
    dram_bank_ys = Dram.BANK_TILE_YS
    self.dram_tiles = [
      (bank, Dram.BANK_X[bank], y)
      for bank in range(Dram.BANK_COUNT) if bank != self.harvested_dram
      for y in dram_bank_ys[bank]
    ]

  def _init_timestamp_dram(self) -> None:
    self._ts_bank_tiles = list(self.dram.bank_tiles)
    self._ts_bank_count = len(self._ts_bank_tiles)
    self._ts_slots_per_page = TIMESTAMP_PAGE_SIZE // TIMESTAMP_STRIDE
    self._ts_active_bank_indices = [i for i, (_, _, y) in enumerate(self._ts_bank_tiles) if y != 0]
    if not self._ts_active_bank_indices:
      self._ts_active_bank_indices = list(range(self._ts_bank_count))
    ts_pages = (TIMESTAMP_MAX_SLOTS + self._ts_slots_per_page - 1) // self._ts_slots_per_page
    ts_local_pages = (ts_pages + len(self._ts_active_bank_indices) - 1) // len(self._ts_active_bank_indices)
    ts_addr = self.dram.next
    self.dram.next = align_up(ts_addr + ts_local_pages * TIMESTAMP_PAGE_SIZE, DRAM_ALIGNMENT)
    self._ts_addr = ts_addr

  def _ts_noc_dest(self, slot: int) -> tuple[int, int]:
    bank_idx, local_page, within_page = self._ts_slot_layout(slot)
    _, x, y = self._ts_bank_tiles[bank_idx]
    return noc_xy(x, y), self._ts_addr + local_page * TIMESTAMP_PAGE_SIZE + within_page

  def _ts_slot_layout(self, slot: int) -> tuple[int, int, int]:
    page = slot // self._ts_slots_per_page
    within_page = (slot % self._ts_slots_per_page) * TIMESTAMP_STRIDE
    active_bank_pos = page % len(self._ts_active_bank_indices)
    bank_idx = self._ts_active_bank_indices[active_bank_pos]
    local_page = page // len(self._ts_active_bank_indices)
    return bank_idx, local_page, within_page

  def _read_ts_slot(self, slot: int) -> int:
    bank_idx, local_page, within_page = self._ts_slot_layout(slot)
    _, x, y = self._ts_bank_tiles[bank_idx]
    self.dram.win.target((x, y), mode=NocOrdering.RELAXED)
    addr = self._ts_addr + local_page * TIMESTAMP_PAGE_SIZE + within_page
    lo = self.dram.win.read32(addr)
    hi = self.dram.win.read32(addr + 4)
    return (hi << 32) | lo

  # ── profiler DRAM ───────────────────────────────────────────────────────────

  def _init_profiler_dram(self) -> None:
    cores = sorted(self.cores, key=lambda xy: (xy[0], xy[1]))
    self._profiler_flat_ids = {core: i for i, core in enumerate(cores)}
    bank_count = len(self.dram.bank_tiles)
    self._profiler_core_count_per_dram = max(1, (len(cores) + bank_count - 1) // bank_count)
    bytes_per_risc = TensixL1.PROFILER_HOST_BUFFER_BYTES_PER_RISC
    page_size = bytes_per_risc * 5 * self._profiler_core_count_per_dram
    # Reserve raw DRAM space (one page per bank, interleaved)
    addr = self.dram.next
    self.dram.next = align_up(addr + page_size, DRAM_ALIGNMENT)
    self._profiler_dram_addr = addr
    self._profiler_page_size = page_size

  def _profiler_control_blob(self, core: Core) -> bytes:
    x, y = core
    ctrl = [0] * 32
    ctrl[12] = self._profiler_dram_addr  # DRAM_PROFILER_ADDRESS_DEFAULT
    ctrl[14] = x   # NOC_X
    ctrl[15] = y   # NOC_Y
    ctrl[16] = self._profiler_flat_ids[core]  # FLAT_ID
    ctrl[17] = self._profiler_core_count_per_dram  # CORE_COUNT_PER_DRAM
    return struct.pack("<32I", *ctrl)

  def _enqueue_profiler_init(self) -> None:
    cores = sorted(self._profiler_flat_ids, key=lambda xy: (xy[0], xy[1]))
    self.cq.write_packed(cores, TensixL1.PROFILER_CONTROL,
                         [self._profiler_control_blob(c) for c in cores])

  def _enqueue_profiler_reset(self, rects: list[Rect]) -> None:
    base = TensixL1.PROFILER_CONTROL
    self.cq.write_packed_large(rects, base + 5 * 4, b"\0" * (5 * 4))   # DEVICE_BUFFER_END × 5
    self.cq.write_packed_large(rects, base + 19 * 4, b"\0" * 4)         # PROFILER_DONE

  def _read_arc_tag(self, tag: int, default: int) -> int:
    arc = TLBWindow(self.fd, start=TileGrid.ARC, addr=Arc.NOC_BASE)
    try:
      telem_ptr = arc.read32(Arc.SCRATCH_RAM_13)
      csm_base, csm_off = align_down(telem_ptr, TLBWindow.SIZE_2M)
      arc.target(TileGrid.ARC, addr=csm_base)
      entry_count = arc.read32(csm_off + 4)
      tags_base = csm_off + 8
      data_base = tags_base + entry_count * 4
      tag_to_offset = {}
      for i in range(entry_count):
        tag_offset = arc.read32(tags_base + i * 4)
        tag_to_offset[tag_offset & 0xFFFF] = (tag_offset >> 16) & 0xFFFF
      off = tag_to_offset.get(tag)
      return default if off is None else arc.read32(data_base + off * 4)
    finally:
      arc.close()

  def _get_harvested_dram_bank(self) -> int:
    gddr_enabled = self._read_arc_tag(Arc.TAG_GDDR_ENABLED, Arc.DEFAULT_GDDR_ENABLED)
    dram_off = [bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0]
    assert len(dram_off) == 1, f"expected 1 harvested dram bank, got {dram_off}"
    return dram_off[0]

  def arc_msg(self, msg: int, arg0: int = 0, arg1: int = 0, queue: int = 0, timeout_ms: int = 1000) -> list[int]:
    MSG_QUEUE_SIZE, REQUEST_MSG_LEN, RESPONSE_MSG_LEN = 4, 8, 8
    MSG_QUEUE_POINTER_WRAP = 2 * MSG_QUEUE_SIZE
    HEADER_BYTES = 8 * 4
    REQUEST_BYTES, RESPONSE_BYTES = REQUEST_MSG_LEN * 4, RESPONSE_MSG_LEN * 4
    QUEUE_STRIDE = HEADER_BYTES + MSG_QUEUE_SIZE * REQUEST_BYTES + MSG_QUEUE_SIZE * RESPONSE_BYTES
    ARC_MISC_CNTL = Arc.RESET_UNIT_OFFSET + 0x100
    IRQ0_TRIG = 1 << 16

    arc = TLBWindow(self.fd, start=TileGrid.ARC, addr=Arc.NOC_BASE)
    try:
      info_ptr = arc.read32(Arc.SCRATCH_RAM_11)
      if info_ptr == 0: raise RuntimeError("msgqueue not initialized (SCRATCH_RAM_11 == 0) — try tt-smi -r")
      info_base, info_off = align_down(info_ptr, TLBWindow.SIZE_2M)
      arc.target(TileGrid.ARC, addr=info_base)
      queues_ptr = arc.read32(info_off)
      q_base, q_off = align_down(queues_ptr, TLBWindow.SIZE_2M)
      arc.target(TileGrid.ARC, addr=q_base)
      q = q_off + queue * QUEUE_STRIDE

      # Enqueue request
      wptr = arc.read32(q)
      req = q + HEADER_BYTES + (wptr % MSG_QUEUE_SIZE) * REQUEST_BYTES
      words = [msg & 0xFF, arg0 & 0xFFFFFFFF, arg1 & 0xFFFFFFFF] + [0] * (REQUEST_MSG_LEN - 3)
      for i, w in enumerate(words): arc.write32(req + i * 4, w)
      arc.write32(q, (wptr + 1) % MSG_QUEUE_POINTER_WRAP)

      # Trigger IRQ
      arc.target(TileGrid.ARC, addr=Arc.NOC_BASE)
      arc.write32(ARC_MISC_CNTL, arc.read32(ARC_MISC_CNTL) | IRQ0_TRIG)

      # Poll for response
      arc.target(TileGrid.ARC, addr=q_base)
      rptr = arc.read32(q + 4)
      deadline = time.monotonic() + timeout_ms / 1000
      while time.monotonic() < deadline:
        if arc.read32(q + 20) != rptr:
          resp = q + HEADER_BYTES + MSG_QUEUE_SIZE * REQUEST_BYTES + (rptr % MSG_QUEUE_SIZE) * RESPONSE_BYTES
          out = [arc.read32(resp + i * 4) for i in range(RESPONSE_MSG_LEN)]
          arc.write32(q + 4, (rptr + 1) % MSG_QUEUE_POINTER_WRAP)
          return out
        time.sleep(0.001)
      raise TimeoutError(f"arc_msg timeout ({timeout_ms} ms) — try tt-smi -r")
    finally:
      arc.close()

  def _set_power_busy(self, timeout_s: float = 2.0):
    self.arc_msg(Arc.MSG_AICLK_GO_BUSY, 0, 0, timeout_ms=1000)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
      aiclk = self._read_arc_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
      if aiclk > Arc.DEFAULT_AICLK: return
      time.sleep(0.001)
    raise RuntimeError(f"AICLK failed to reach busy state (last={aiclk} MHz)")

  def _set_power_idle(self):
    try:
      self.arc_msg(Arc.MSG_AICLK_GO_LONG_IDLE, 0, 0, timeout_ms=1000)
    except (TimeoutError, RuntimeError):
      pass  # best-effort on shutdown

  # ── firmware upload ────────────────────────────────────────────────────────

  def _upload_firmware(self) -> None:
    fw = self.compiler._fw
    mmio_base, mmio_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBWindow.SIZE_2M)
    reset_off = TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0 - mmio_base

    # Resolve local-RAM segments to L1 scratch addresses
    staged: dict[str, list[tuple[int, bytes]]] = {}
    for name, cfw in fw.items():
      scratch = _INIT_SCRATCH[name]
      spans = []
      for s in cfw.segments:
        if not s.data and s.memsz == 0: continue
        data = s.data if s.memsz <= len(s.data) else s.data + b"\0" * (s.memsz - len(s.data))
        addr = s.paddr
        if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
          addr = scratch + (addr - TensixMMIO.LOCAL_RAM_START)
        assert 0 <= addr < TensixL1.SIZE, f"{name}: bad paddr 0x{s.paddr:x} -> 0x{addr:x}"
        spans.append((addr, data))
      staged[name] = spans

    brisc_base = TensixL1.BRISC_FIRMWARE_BASE
    jal = ((brisc_base & 0xFF000) | ((brisc_base & 0x800) << 9) | ((brisc_base & 0x7FE) << 20) | 0x6F).to_bytes(4, "little")
    go_init = struct.pack("<BBBB", 0, 0, 0, DevMsgs.RUN_MSG_INIT)
    bank_table = self._build_bank_noc_table()

    all_cores = list(self.cores)
    if self._use_fast_dispatch:
      all_cores += [self._prefetch_core, self._dispatch_core]

    with TLBWindow(self.fd, start=all_cores[0]) as win:
      for core in all_cores:
        # Put all RISCs in reset
        win.target(core, addr=mmio_base)
        win.write32(reset_off, TensixMMIO.SOFT_RESET_ALL)
        # Write firmware segments to L1
        win.target(core, mode=NocOrdering.RELAXED)
        for spans in staged.values():
          for addr, data in spans:
            win.write(addr, data)
        # JAL at address 0 → BRISC firmware entry
        win.write(0, jal)
        # GO_MSG = RUN_MSG_INIT
        win.write(TensixL1.GO_MSG, go_init)
        # Bank-to-NOC tables
        win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, bank_table)
        # Reset PCs for NCRISC/TRISC
        win.target(core, addr=mmio_base)
        for reg, text_base in [
          (TensixMMIO.RISCV_DEBUG_REG_NCRISC_RESET_PC, fw["ncrisc"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC0_RESET_PC, fw["trisc0"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC1_RESET_PC, fw["trisc1"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC2_RESET_PC, fw["trisc2"].text_base),
        ]:
          win.write32(reg - mmio_base, text_base)

      # Release BRISC on all cores (keep NCRISC/TRISC in reset)
      for core in all_cores:
        win.target(core, addr=mmio_base)
        win.read32(reset_off)  # fence
        win.write32(reset_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)

      # Wait for firmware ready on one core
      probe = (1, 2) if (1, 2) in all_cores else all_cores[0]
      win.target(probe)
      deadline = time.perf_counter() + 2.0
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"firmware not ready on {probe} — try tt-smi -r")
        time.sleep(0.001)

  def _build_bank_noc_table(self) -> bytes:
    NUM_NOCS, NUM_DRAM_BANKS, NUM_L1_BANKS = 2, 7, 110
    WORKER_EP_LOGICAL = {0: [2, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [2, 1], 5: [2, 1], 6: [2, 1], 7: [2, 1]}

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

    dram_translated = dram_translated_map(self.harvested_dram)
    dram_xy = []
    for noc in range(NUM_NOCS):
      for bank in range(NUM_DRAM_BANKS):
        port = WORKER_EP_LOGICAL[bank][noc]
        x, y = dram_translated[(bank, port)]
        dram_xy.append(noc_xy(x, y))

    tensix_cols = sorted({x for x, _ in self.cores})
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

  # ── CQ dispatch core setup ────────────────────────────────────────────────

  def _wait_core_done(self, core: Core, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    with TLBWindow(self.fd, start=core) as win:
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          go = win.uc[TensixL1.GO_MSG + 3]
          raise TimeoutError(f"core {core} firmware init timeout (GO_MSG signal=0x{go:02x}) — try tt-smi -r")
        time.sleep(0.001)

  def _start_dispatch_cores(self) -> None:
    from compiler import compile_cq_kernels
    cq_kernels = compile_cq_kernels()
    l1a = FastDispatch.L1_ALIGNMENT

    self._wait_core_done(self._prefetch_core)
    self._wait_core_done(self._dispatch_core)

    # Kernel config layout: [rt_args(16B)] [sem0(16B)] [sem1(16B)] [kernel XIP...]
    kernel_off = l1a + 2 * l1a  # = 48

    # Prefetch core: BRISC only, sem[0] = DEV_DISPATCH_CB_PAGES
    pref_rt = b"\0" * l1a
    pref_sems = struct.pack("<I", DEV_DISPATCH_CB_PAGES).ljust(l1a, b"\0") + b"\0" * l1a
    pref_img = pref_rt + pref_sems
    pref_launch = self._build_cq_launch(kernel_off, 0, sem_off=l1a)

    # Dispatch core: BRISC + NCRISC
    disp_rt = b"\0" * l1a
    disp_sems = b"\0" * l1a + b"\0" * l1a
    disp_img = disp_rt + disp_sems
    ncrisc_off = align_up(kernel_off + len(cq_kernels["dispatch_brisc"].xip), l1a)
    disp_launch = self._build_cq_launch(kernel_off, ncrisc_off, sem_off=l1a)

    self.cq.reset_run_state()
    self._upload_cq_core(
      self._prefetch_core,
      pref_img,
      pref_launch,
      [(kernel_off, cq_kernels["prefetch_brisc"].xip)],
    )
    self._upload_cq_core(
      self._dispatch_core,
      disp_img,
      disp_launch,
      [(kernel_off, cq_kernels["dispatch_brisc"].xip),
       (ncrisc_off, cq_kernels["dispatch_s_ncrisc"].xip)],
      init=self._init_dispatch_core_state,
    )

  @staticmethod
  def _build_cq_launch(brisc_text_off: int, ncrisc_text_off: int = 0, sem_off: int = 16) -> LaunchMsg:
    launch = LaunchMsg()
    kc = launch.kernel_config
    for i in range(3):
      kc.kernel_config_base[i] = TensixL1.KERNEL_CONFIG_BASE
    kc.sem_offset[0] = sem_off
    kc.rta_offset[0].rta_offset = 0
    kc.rta_offset[0].crta_offset = FastDispatch.L1_ALIGNMENT
    kc.kernel_text_offset[0] = brisc_text_off
    kc.kernel_text_offset[1] = ncrisc_text_off
    kc.enables = 1 | (2 if ncrisc_text_off else 0)
    kc.mode = DevMsgs.DISPATCH_MODE_HOST
    kc.local_cb_mask = 0
    kc.min_remote_cb_start_index = FAST_CQ_NUM_CIRCULAR_BUFFERS
    return launch

  def _init_dispatch_core_state(self, win: TLBWindow) -> None:
    l1a = FastDispatch.L1_ALIGNMENT
    base_16b = self.cq._completion_base_16b
    win.write32(DEV_COMPLETION_Q_WR_PTR_ADDR, base_16b)
    win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, base_16b)
    win.write32(DEV_COMPLETION_Q0_LAST_EVENT_PTR_ADDR, 0)
    win.write32(DEV_COMPLETION_Q1_LAST_EVENT_PTR_ADDR, 0)
    win.uc[DEV_DISPATCH_S_SYNC_SEM_ADDR : DEV_DISPATCH_S_SYNC_SEM_ADDR + 8 * l1a] = b"\0" * (8 * l1a)

  def _upload_cq_core(self, core: Core, img: bytes, launch: LaunchMsg,
                      kernels: list[tuple[int, bytes]], init: Callable[[TLBWindow], None] | None = None) -> None:
    win = self.cq._prefetch_win if core == self._prefetch_core else self.cq._dispatch_win
    win.target(core)
    if init is not None:
      init(win)
    win.write(TensixL1.KERNEL_CONFIG_BASE, img)
    for off, xip in kernels:
      win.write(TensixL1.KERNEL_CONFIG_BASE + off, xip)
    win.write(TensixL1.LAUNCH, as_bytes(launch))
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    win.write(TensixL1.GO_MSG, struct.pack("<I", go.all))

  # ── program dispatch ───────────────────────────────────────────────────────

  def _resolve_cores(self, spec: int | Literal["all"]) -> list[Core]:
    if spec == "all":
      return list(self.cores)
    return self.cores[:spec]

  def _go_word(self) -> int:
    """Build the go signal word for fast dispatch (includes dispatch core XY)."""
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go.bits.master_x, go.bits.master_y = self._dispatch_core
    go.bits.dispatch_message_offset = 0
    return go.all

  def alloc(self, num_tiles: int, dtype: Dtype, name: str = "", shape: Shape | None = None) -> DramBuffer:
    return self.dram.alloc(num_tiles, dtype, name, shape)

  def alloc_write(self, data: bytes, dtype: Dtype, shape: Shape, name: str = "") -> DramBuffer:
    buf = self.dram.alloc(len(data) // dtype.tile_size, dtype, name, shape)
    self.dram_write(buf, data)
    return buf

  def _ensure_dram_sysmem(self):
    if self._dram_sysmem is None:
      self._dram_sysmem = Sysmem(self.fd, size=128 * 1024 * 1024)

  def _compile_dram_kernel(self, name: str):
    src = (Path(__file__).parent / "firmware" / name).read_text()
    return self.compiler.compile_dataflow(src, "ncrisc")

  def _run_dram_transfer(self, buf: DramBuffer, kernel, extra_args: list[int] | None = None):
    """Run a fill/drain kernel across worker cores to transfer between DRAM and sysmem."""
    self._ensure_dram_sysmem()
    assert not self._programs, "queue must be empty for DRAM transfers"
    n_tiles = buf.num_tiles
    all_cores = self.cores
    tiles_per_core = (n_tiles + len(all_cores) - 1) // len(all_cores)
    n_cores = min(len(all_cores), n_tiles)
    sysmem_offset = self._dram_sysmem.noc_addr & ((1 << 36) - 1)
    base_args = [buf.addr, Sysmem.PCIE_NOC_XY, sysmem_offset]
    ea = extra_args or []

    def reader_args(core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
      start = core_idx * tiles_per_core
      count = min(tiles_per_core, n_tiles - start) if start < n_tiles else 0
      return base_args + [start, count, buf.page_size] + ea

    self.queue(Program(
      cores=n_cores, reader_kernel="", writer_kernel="", compute_kernel="",
      cbs=[CBConfig(index=0, dtype=buf.dtype, tiles=2)],
      reader_args=reader_args,
    ))
    prog = self._programs[0]
    self._set_power_busy()
    dispatch_mode = self._dispatch_mode
    cores = self._resolve_cores(prog.cores)
    rects = mcast_rects(cores)
    commands = build_commands(prog, kernel, None, None, cores, rects, dispatch_mode)
    if not self._use_fast_dispatch:
      with TLBWindow(self.fd, start=cores[0]) as win:
        _slow_dispatch(win, commands)
    else:
      go_word = self._go_word()
      _fast_enqueue(self.cq, commands, go_word)
      self.cq._event_id += 1
      self.cq.host_event(self.cq._event_id)
      self.cq.flush()
      self.cq.wait_completion(self.cq._event_id)
    self._programs.clear()
    self._set_power_idle()

  def dram_write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    if buf.shape is not None:
      data = tilize(data, buf.dtype.bpe, buf.shape)
    self.dram.write(buf, data)

  def dram_read(self, buf: DramBuffer) -> bytes:
    result = self.dram.read(buf)
    if buf.shape is not None:
      return untilize(result, buf.dtype.bpe, buf.shape)
    return result

  def queue(self, program):
    self._programs.append(program)

  def _compile_commands(self, program, dispatch_mode, host_assigned_id: int = 0) -> list:
    if hasattr(program, "compile"):
      prog, roles, compute, all_cores, per_core_args = program.compile(self.compiler)
      return build_commands(prog, roles, compute, all_cores, per_core_args, dispatch_mode,
                            host_assigned_id=host_assigned_id)
    writer = self.compiler.compile_dataflow(program.writer_kernel, "brisc") if program.writer_kernel else None
    reader = self.compiler.compile_dataflow(program.reader_kernel, "ncrisc") if program.reader_kernel else None
    compute = self.compiler.compile_compute(program.compute_kernel, program) if program.compute_kernel else None
    cores = self._resolve_cores(program.cores)
    n = len(cores)
    per_core_args = [
      (resolve_args(program.writer_args, i, c, n), resolve_args(program.reader_args, i, c, n),
       resolve_args(program.compute_args, i, c, n))
      for i, c in enumerate(cores)
    ]
    return build_commands(program, [(cores, reader, writer)], compute, cores, per_core_args, dispatch_mode,
                          host_assigned_id=host_assigned_id)

  def _program_cores(self, program) -> list[Core]:
    if hasattr(program, "plan"):
      return sorted(program.plan.active_cores(), key=lambda c: (c[0], c[1]))
    return self._resolve_cores(program.cores)

  def _programs_info(self) -> list[dict]:
    """Build profiler metadata for queued programs."""
    info = []
    for i, program in enumerate(self._programs):
      cores = self._program_cores(program)
      sources = {}
      if hasattr(program, "plan"):
        # MatmulProgram — kernel sources are generated, not stored
        pass
      else:
        if program.reader_kernel: sources["reader"] = program.reader_kernel
        if program.writer_kernel: sources["writer"] = program.writer_kernel
        if program.compute_kernel: sources["compute"] = program.compute_kernel
      info.append({"index": i, "name": getattr(program, "name", None), "cores": cores, "sources": sources})
    return info

  def run(self) -> list[dict] | None:
    if not self._programs: return None
    timing = os.environ.get("TIMING") == "1"
    profiling = self._profiler
    self._set_power_busy()
    try:
      if self._use_fast_dispatch:
        n = len(self._programs)
        if profiling:
          programs_info = self._programs_info()
          all_rects = mcast_rects(self.cores)
          if not self._profiler_initialized:
            self._enqueue_profiler_init()
            self._profiler_initialized = True
          base = TensixL1.PROFILER_CONTROL
          self.cq.write_packed_large(all_rects, base, b"\0" * (5 * 4))  # reset HOST_BUFFER_END × 5
        for i, program in enumerate(self._programs):
          if profiling:
            self._enqueue_profiler_reset(all_rects)
          ts_slot = 2 * i
          if timing and ts_slot + 1 < TIMESTAMP_MAX_SLOTS:
            self.cq.timestamp(*self._ts_noc_dest(ts_slot))
          prof_id = (i + 1) if profiling else 0
          commands = self._compile_commands(program, self._dispatch_mode, host_assigned_id=prof_id)
          _fast_enqueue(self.cq, commands, self._go_word())
          if timing and ts_slot + 1 < TIMESTAMP_MAX_SLOTS:
            self.cq.timestamp(*self._ts_noc_dest(ts_slot + 1))
        self.cq._event_id += 1
        self.cq.host_event(self.cq._event_id)
        self.cq.flush()
        self.cq.wait_completion(self.cq._event_id)

        # Profiler collection (fast dispatch)
        if profiling:
          self.dram.barrier()
          import profiler
          freq_mhz = self._read_arc_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
          self.last_profile = profiler.collect_fast_dram(
            self, programs_info, core_flat_ids=self._profiler_flat_ids,
            dram_addr=self._profiler_dram_addr, page_size=self._profiler_page_size,
            core_count_per_dram=self._profiler_core_count_per_dram,
            freq_mhz=freq_mhz, dispatch_cores=[self._prefetch_core, self._dispatch_core])
          profiler.print_data_summary(self.last_profile)

        if not timing:
          return None
        self.dram.barrier()
        freq_mhz = self._read_arc_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
        timings = []
        for i in range(n):
          ts_slot = 2 * i
          if ts_slot + 1 >= TIMESTAMP_MAX_SLOTS: break
          t0 = self._read_ts_slot(ts_slot)
          t1 = self._read_ts_slot(ts_slot + 1)
          cycles = t1 - t0
          timings.append({"cycles": cycles, "us": cycles / freq_mhz, "freq_mhz": freq_mhz})
        for i, t in enumerate(timings):
          print(f"  [{i}] {t['us']:.1f} us ({t['cycles']} cycles)")
        self.last_device_timing = timings
        return timings
      else:
        t0 = time.perf_counter() if timing else 0
        with TLBWindow(self.fd, start=self.cores[0]) as win:
          for program in self._programs:
            commands = self._compile_commands(program, self._dispatch_mode)
            _slow_dispatch(win, commands)
        # Profiler collection (slow dispatch)
        if profiling:
          import profiler
          programs_info = self._programs_info()
          freq_mhz = self._read_arc_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
          self.last_profile = profiler.collect(self, programs_info, freq_mhz=freq_mhz)
          profiler.print_data_summary(self.last_profile)
        if timing:
          elapsed_us = (time.perf_counter() - t0) * 1e6
          print(f"  slow dispatch: {elapsed_us:.1f} us (host wall-clock)")
        return None
    finally:
      self._programs.clear()
      self._set_power_idle()

  def serve_profile(self, port: int = 8000):
    if self.last_profile is None:
      print("No profiler data to serve")
      return
    from profiler import ui as profiler_ui
    profiler_ui.serve(self.last_profile, port=port)

  def close(self):
    try:
      self._set_power_idle()
    finally:
      if self._dram_sysmem is not None:
        self._dram_sysmem.close()
      if self.cq is not None:
        self.cq.close()
      self.dram.close()
      os.close(self.fd)
