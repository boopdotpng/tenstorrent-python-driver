import ctypes, time, mmap, os, struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal
import numpy as np
from autogen import *
from dispatch import CBConfig, CQ, CQ_CMD_SIZE, Core, CoreArgs, Dtype, MathFidelity, Program, Rect, FAST_CQ_NUM_CIRCULAR_BUFFERS
from dispatch import Go, McastWrite, SetGoSignalNocData, UnicastWrite, Wait
from dispatch import build_commands, fast_enqueue, mcast_rects, noc_mcast_xy, noc_xy, resolve_args, slow_dispatch
from kernels import TILIZE_READER, TILIZE_COMPUTE, TILIZE_WRITER, UNTILIZE_READER, UNTILIZE_COMPUTE, UNTILIZE_WRITER

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
    out = IOCTL_ALLOCATE_TLB(fd, size=size)
    self._id = out.id
    rw = mmap.PROT_READ | mmap.PROT_WRITE
    self.uc = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=rw, offset=out.mmap_offset_uc)
    self.wc = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=rw, offset=out.mmap_offset_wc)
    self.target(start, end, addr=addr, mode=mode)

  def target(self, start: Core, end: Core | None = None, addr: int = 0, mode: NocOrdering = NocOrdering.STRICT):
    end = end or start
    IOCTL_CONFIGURE_TLB(self.fd, id=self._id, config=NocTlbConfig(
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
    IOCTL_FREE_TLB(self.fd, id=self._id)

  def __enter__(self): return self
  def __exit__(self, *_): self.close()

USE_USB_DISPATCH = os.environ.get("TT_USB") == "1"

class Sysmem:
  PCIE_NOC_XY = (24 << 6) | 19

  def __init__(self, fd: int, size: int = 1 << 30):
    self.fd = fd
    page_size = os.sysconf("SC_PAGE_SIZE")
    self.size = (size + page_size - 1) & ~(page_size - 1)
    self.buf = mmap.mmap(-1, self.size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                         prot=mmap.PROT_READ | mmap.PROT_WRITE)
    self._va = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
    out = IOCTL_PIN_PAGES(self.fd, flags=PIN_PAGES_NOC_DMA, virtual_address=self._va, size=self.size)
    self.noc_addr = out.noc_address

  def close(self):
    IOCTL_UNPIN_PAGES(self.fd, virtual_address=self._va, size=self.size)
    self.buf.close()

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
  a = a.transpose(0, 1, 3, 2, 4)                                   # grid transform: (b, tr, tc, 32, 32)
  a = a.reshape(-1, 2, FACE_R, 2, FACE_C).transpose(0, 1, 3, 2, 4) # face transform: (n_tiles, 2, 2, 16, 16)
  return a.tobytes()

def untilize(data: bytes, bpe: int, shape: Shape) -> bytes:
  rows, cols = shape[-2], shape[-1]
  assert rows % TILE_R == 0 and cols % TILE_C == 0
  batch = 1
  for d in shape[:-2]: batch *= d
  tr, tc = rows // TILE_R, cols // TILE_C
  dt = _np_dtype(bpe)
  a = np.frombuffer(data, dtype=dt).reshape(-1, 2, 2, FACE_R, FACE_C)
  a = a.transpose(0, 1, 3, 2, 4).reshape(batch, tr, tc, TILE_R, TILE_C) # undo face, then undo grid
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

@dataclass(frozen=True)
class LayoutTransferPlan:
  shape: Shape
  logical_bytes: int
  tile_cols: int
  total_tiles: int
  row_bytes: int
  n_cores: int
  tiles_per_core: int

class Allocator:
  def __init__(self, fd: int, bank_tiles: list):
    self.bank_tiles = bank_tiles[:: Dram.TILES_PER_BANK]
    self.win = TLBWindow(fd, start=self.bank_tiles[0][1:], size=TLBWindow.SIZE_4G)
    self.next = Dram.DRAM_WRITE_OFFSET

  def alloc(
    self, num_tiles: int, dtype: Dtype, name: str = "", shape: Shape | None = None
  ) -> DramBuffer:
    num_banks = len(self.bank_tiles)
    pages_per_bank = (num_tiles + num_banks - 1) // num_banks
    addr = self.next
    self.next = align_up(addr + pages_per_bank * dtype.tile_size, DRAM_ALIGNMENT)
    return DramBuffer(
      name=name, addr=addr, num_tiles=num_tiles, dtype=dtype, shape=shape
    )

  def alloc_write(
    self, data: bytes, dtype: Dtype, shape: Shape, name: str = ""
  ) -> DramBuffer:
    num_tiles = len(data) // dtype.tile_size
    buf = self.alloc(num_tiles, dtype, name=name, shape=shape)
    self.write(buf, data)
    return buf

  def barrier(self):
    for flag in Dram.BARRIER_FLAGS:
      for _, x, y in self.bank_tiles:
        self.win.target((x, y))
        self.win.write32(DRAM_BARRIER_BASE, flag)
        while self.win.read32(DRAM_BARRIER_BASE) != flag:
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

class TileGrid:
  ARC = (8, 0)
  TENSIX_X = (*range(1, 8), *range(10, 15))
  WORKER_CORES = [(x, y) for x in TENSIX_X for y in range(2, 12)]

class CQSysmem:
  """Host-side sysmem + TLB windows for fast dispatch. Handles flush, completion, and device communication."""

  def __init__(self, fd: int, prefetch_win: TLBWindow, dispatch_win: TLBWindow):
    self.fd = fd
    self._prefetch_win = prefetch_win
    self._dispatch_win = dispatch_win
    flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS
    if hasattr(mmap, "MAP_POPULATE"):
      flags |= mmap.MAP_POPULATE
    self.sysmem = mmap.mmap(-1, HOST_SYSMEM_SIZE, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    self._sysmem_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.sysmem))
    if (self._sysmem_addr % PAGE_SIZE) != 0 or (HOST_SYSMEM_SIZE % PAGE_SIZE) != 0:
      raise RuntimeError("CQ sysmem must be page-aligned and page-sized")
    out = IOCTL_PIN_PAGES(self.fd, flags=PIN_PAGES_NOC_DMA, virtual_address=self._sysmem_addr, size=HOST_SYSMEM_SIZE)
    self.noc_addr = out.noc_address
    if (self.noc_addr & FastDispatch.PCIE_NOC_BASE) != FastDispatch.PCIE_NOC_BASE:
      raise RuntimeError(f"bad NOC sysmem address: 0x{self.noc_addr:x}")
    self.noc_local = self.noc_addr - FastDispatch.PCIE_NOC_BASE
    if self.noc_local > 0xFFFF_FFFF:
      raise RuntimeError(f"CQ sysmem NOC offset too large: 0x{self.noc_local:x}")
    self._issue_wr = 0
    self._prefetch_q_wr_idx = 0
    self._event_id = 0
    self._completion_base_16b = ((self.noc_local + HOST_COMPLETION_BASE) >> 4) & 0x7FFF_FFFF
    self._completion_page_16b = PAGE_SIZE >> 4
    self._completion_end_16b = self._completion_base_16b + (HOST_COMPLETION_SIZE >> 4)
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0

  def _sysmem_read32(self, off):
    return struct.unpack("<I", self.sysmem[off : off + 4])[0]

  def _sysmem_write32(self, off, val):
    self.sysmem[off : off + 4] = struct.pack("<I", val)

  def _wait_prefetch_slot_free(self, idx: int, timeout_s: float = 1.0):
    off = DEV_PREFETCH_Q_BASE + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    deadline = time.perf_counter() + timeout_s
    while struct.unpack("<H", self._prefetch_win.uc[off : off + 2])[0] != 0:
      if time.perf_counter() > deadline:
        raise TimeoutError("timeout waiting for prefetch queue slot")

  def _issue_write(self, record: bytes):
    self._issue_wr = align_up(self._issue_wr, FastDispatch.PCIE_ALIGNMENT)
    if self._issue_wr + len(record) > HOST_ISSUE_SIZE:
      self._issue_wr = 0
    base = HOST_ISSUE_BASE + self._issue_wr
    self.sysmem[base : base + len(record)] = record
    self._issue_wr += len(record)
    idx = self._prefetch_q_wr_idx
    self._wait_prefetch_slot_free(idx)
    off = DEV_PREFETCH_Q_BASE + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self._prefetch_win.uc[off : off + 2] = struct.pack("<H", len(record) >> 4)
    self._prefetch_q_wr_idx = (idx + 1) % (DEV_PREFETCH_Q_SIZE // FastDispatch.PREFETCH_Q_ENTRY_BYTES)

  def flush(self, cq: 'CQ'):
    offset = 0
    for size_16b in cq.sizes:
      size = size_16b << 4
      self._issue_write(cq.stream[offset : offset + size])
      offset += size
    cq.clear()

  def wait_completion(self, event_id: int, timeout_s: float = 10.0):
    deadline = time.perf_counter() + timeout_s
    while True:
      wr_raw = self._sysmem_read32(FastDispatch.HOST_COMPLETION_Q_WR_OFF)
      wr_16b, wr_toggle = wr_raw & 0x7FFF_FFFF, (wr_raw >> 31) & 1
      if wr_16b != self._completion_rd_16b or wr_toggle != self._completion_rd_toggle:
        off = (self._completion_rd_16b << 4) - self.noc_local
        got = self._sysmem_read32(off + CQ_CMD_SIZE)
        self._completion_rd_16b += self._completion_page_16b
        if self._completion_rd_16b >= self._completion_end_16b:
          self._completion_rd_16b = self._completion_base_16b
          self._completion_rd_toggle ^= 1
        raw = (self._completion_rd_16b & 0x7FFF_FFFF) | (self._completion_rd_toggle << 31)
        self._dispatch_win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, raw)
        self._sysmem_write32(FastDispatch.HOST_COMPLETION_Q_RD_OFF, raw)
        if got != (event_id & 0xFFFFFFFF):
          raise RuntimeError(f"completion event mismatch: got {got}, expected {event_id}")
        return
      if time.perf_counter() > deadline:
        raise TimeoutError(f"timeout waiting for completion event {event_id} — try tt-smi -r")
      time.sleep(0.0002)

  def reset_run_state(self):
    self._issue_wr = 0
    self._prefetch_q_wr_idx = 0
    self._prefetch_win.write32(DEV_PREFETCH_Q_RD_PTR_ADDR, DEV_PREFETCH_Q_BASE + DEV_PREFETCH_Q_SIZE)
    self._prefetch_win.write32(DEV_PREFETCH_Q_PCIE_RD_PTR_ADDR, (self.noc_local + HOST_ISSUE_BASE) & 0xFFFFFFFF)
    for i in range(FastDispatch.PREFETCH_Q_ENTRIES_WORKER_DEFAULT):
      off = DEV_PREFETCH_Q_BASE + i * FastDispatch.PREFETCH_Q_ENTRY_BYTES
      self._prefetch_win.uc[off : off + 2] = b"\0\0"
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    self._dispatch_win.write32(DEV_COMPLETION_Q_WR_PTR_ADDR, self._completion_base_16b)
    self._dispatch_win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, self._completion_base_16b)
    self._sysmem_write32(FastDispatch.HOST_COMPLETION_Q_WR_OFF, self._completion_base_16b)
    self._sysmem_write32(FastDispatch.HOST_COMPLETION_Q_RD_OFF, self._completion_base_16b)

  def close(self):
    self._prefetch_win.close()
    self._dispatch_win.close()
    IOCTL_UNPIN_PAGES(self.fd, virtual_address=self._sysmem_addr, size=HOST_SYSMEM_SIZE)
    self.sysmem.close()

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
      raise SystemExit(
        f"unsupported blackhole device {self.arch}; p100a only for now"
      )

    self.harvested_dram = self._get_harvested_dram_bank()
    self._set_active_dram_tiles()
    self.dram = Allocator(self.fd, self.dram_tiles)
    self._init_timestamp_dram()
    self._dispatch_mode = DevMsgs.DISPATCH_MODE_HOST if USE_USB_DISPATCH else DevMsgs.DISPATCH_MODE_DEV
    self._use_fast_dispatch = not USE_USB_DISPATCH
    self.cores = list(TileGrid.WORKER_CORES)
    if self._use_fast_dispatch:
      self._prefetch_core, self._dispatch_core = self._select_dispatch_core_pair()
      self.cores = [
        c
        for c in self.cores
        if c not in {self._prefetch_core, self._dispatch_core}
      ]

    from compiler import Compiler
    self.compiler = Compiler()
    self._upload_firmware()

    self._dram_sysmem = Sysmem(self.fd) if self._use_fast_dispatch else None
    self.cq = CQ()
    self._cq_hw = None
    if self._use_fast_dispatch:
      self._cq_hw = CQSysmem(
        self.fd,
        prefetch_win=TLBWindow(self.fd, start=self._prefetch_core),
        dispatch_win=TLBWindow(self.fd, start=self._dispatch_core),
      )
      self._start_dispatch_cores()

    self._programs = []
    self.last_profile = None
    self._profiler_initialized = False

    from compiler import PROFILER
    self._profiler = PROFILER and self._use_fast_dispatch
    self._profiler_flat_ids = {}
    self._profiler_core_count_per_dram = 0
    if self._profiler:
      self._init_profiler_dram()

  def _set_active_dram_tiles(self):
    dram_bank_ys = Dram.BANK_TILE_YS
    self.dram_tiles = [
      (bank, Dram.BANK_X[bank], y)
      for bank in range(Dram.BANK_COUNT)
      if bank != self.harvested_dram
      for y in dram_bank_ys[bank]
    ]

  def _init_timestamp_dram(self):
    self._ts_bank_tiles = list(self.dram.bank_tiles)
    self._ts_bank_count = len(self._ts_bank_tiles)
    self._ts_slots_per_page = TIMESTAMP_PAGE_SIZE // TIMESTAMP_STRIDE
    self._ts_active_bank_indices = [
      i for i, (_, _, y) in enumerate(self._ts_bank_tiles) if y != 0
    ]
    if not self._ts_active_bank_indices:
      self._ts_active_bank_indices = list(range(self._ts_bank_count))
    ts_pages = (
      TIMESTAMP_MAX_SLOTS + self._ts_slots_per_page - 1
    ) // self._ts_slots_per_page
    ts_local_pages = (ts_pages + len(self._ts_active_bank_indices) - 1) // len(
      self._ts_active_bank_indices
    )
    ts_addr = self.dram.next
    self.dram.next = align_up(
      ts_addr + ts_local_pages * TIMESTAMP_PAGE_SIZE, DRAM_ALIGNMENT
    )
    self._ts_addr = ts_addr

  def _ts_noc_dest(self, slot: int) -> tuple[int, int]:
    bank_idx, local_page, within_page = self._ts_slot_layout(slot)
    _, x, y = self._ts_bank_tiles[bank_idx]
    return noc_xy(
      x, y
    ), self._ts_addr + local_page * TIMESTAMP_PAGE_SIZE + within_page

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

  def _init_profiler_dram(self):
    cores = sorted(self.cores, key=lambda xy: (xy[0], xy[1]))
    self._profiler_flat_ids = {core: i for i, core in enumerate(cores)}
    bank_count = len(self.dram.bank_tiles)
    self._profiler_core_count_per_dram = max(
      1, (len(cores) + bank_count - 1) // bank_count
    )
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
    ctrl[14] = x  # NOC_X
    ctrl[15] = y  # NOC_Y
    ctrl[16] = self._profiler_flat_ids[core]  # FLAT_ID
    ctrl[17] = self._profiler_core_count_per_dram  # CORE_COUNT_PER_DRAM
    return struct.pack("<32I", *ctrl)

  def _enqueue_profiler_init(self):
    cores = sorted(self._profiler_flat_ids, key=lambda xy: (xy[0], xy[1]))
    self.cq.write_packed(
      cores,
      TensixL1.PROFILER_CONTROL,
      [self._profiler_control_blob(c) for c in cores],
    )

  def _enqueue_profiler_reset(self, rects: list[Rect]):
    base = TensixL1.PROFILER_CONTROL
    self.cq.write_packed_large(
      rects, base + 5 * 4, b"\0" * (5 * 4)
    )  # DEVICE_BUFFER_END × 5
    self.cq.write_packed_large(rects, base + 19 * 4, b"\0" * 4)  # PROFILER_DONE

  def _read_profiler_dram(self) -> bytes:
    return self.dram.read_raw_bank_pages(self._profiler_dram_addr, self._profiler_page_size)

  def _read_profiler_ctrl(self, cores: list[Core]) -> dict[Core, bytes]:
    ctrl = {}
    for core in cores:
      with TLBWindow(self.fd, start=core) as win:
        ctrl[core] = bytes(
          win.uc[TensixL1.PROFILER_CONTROL : TensixL1.PROFILER_CONTROL + 128]
        )
    return ctrl

  def _read_arc_tag(self, tag: int, default: int) -> int:
    with TLBWindow(self.fd, start=TileGrid.ARC, addr=Arc.NOC_BASE) as arc:
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

  def _get_harvested_dram_bank(self) -> int:
    gddr_enabled = self._read_arc_tag(
      Arc.TAG_GDDR_ENABLED, Arc.DEFAULT_GDDR_ENABLED
    )
    dram_off = [
      bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0
    ]
    assert len(dram_off) == 1, f"expected 1 harvested dram bank, got {dram_off}"
    return dram_off[0]

  def arc_msg(
    self,
    msg: int,
    arg0: int = 0,
    arg1: int = 0,
    queue: int = 0,
    timeout_ms: int = 1000,
  ) -> list[int]:
    MSG_QUEUE_SIZE, REQUEST_MSG_LEN, RESPONSE_MSG_LEN = 4, 8, 8
    MSG_QUEUE_POINTER_WRAP = 2 * MSG_QUEUE_SIZE
    HEADER_BYTES = 8 * 4
    REQUEST_BYTES, RESPONSE_BYTES = REQUEST_MSG_LEN * 4, RESPONSE_MSG_LEN * 4
    QUEUE_STRIDE = (
      HEADER_BYTES
      + MSG_QUEUE_SIZE * REQUEST_BYTES
      + MSG_QUEUE_SIZE * RESPONSE_BYTES
    )
    ARC_MISC_CNTL = Arc.RESET_UNIT_OFFSET + 0x100
    IRQ0_TRIG = 1 << 16

    with TLBWindow(self.fd, start=TileGrid.ARC, addr=Arc.NOC_BASE) as arc:
      info_ptr = arc.read32(Arc.SCRATCH_RAM_11)
      if info_ptr == 0:
        raise RuntimeError("msgqueue not initialized (SCRATCH_RAM_11 == 0) — try tt-smi -r")
      info_base, info_off = align_down(info_ptr, TLBWindow.SIZE_2M)
      arc.target(TileGrid.ARC, addr=info_base)
      queues_ptr = arc.read32(info_off)
      q_base, q_off = align_down(queues_ptr, TLBWindow.SIZE_2M)
      arc.target(TileGrid.ARC, addr=q_base)
      q = q_off + queue * QUEUE_STRIDE

      wptr = arc.read32(q)
      req = q + HEADER_BYTES + (wptr % MSG_QUEUE_SIZE) * REQUEST_BYTES
      words = [msg & 0xFF, arg0 & 0xFFFFFFFF, arg1 & 0xFFFFFFFF] + [0] * (REQUEST_MSG_LEN - 3)
      for i, w in enumerate(words):
        arc.write32(req + i * 4, w)
      arc.write32(q, (wptr + 1) % MSG_QUEUE_POINTER_WRAP)

      arc.target(TileGrid.ARC, addr=Arc.NOC_BASE)
      arc.write32(ARC_MISC_CNTL, arc.read32(ARC_MISC_CNTL) | IRQ0_TRIG)

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

  def _set_power_busy(self, timeout_s: float = 2.0):
    self.arc_msg(Arc.MSG_AICLK_GO_BUSY, 0, 0, timeout_ms=1000)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
      aiclk = self._read_arc_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
      if aiclk > Arc.DEFAULT_AICLK:
        return
      time.sleep(0.001)
    raise RuntimeError(f"AICLK failed to reach busy state (last={aiclk} MHz)")

  def _set_power_idle(self):
    try:
      self.arc_msg(Arc.MSG_AICLK_GO_LONG_IDLE, 0, 0, timeout_ms=1000)
    except (TimeoutError, RuntimeError):
      pass  # best-effort on shutdown

  def _upload_firmware(self):
    fw = self.compiler._fw
    mmio_base, mmio_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBWindow.SIZE_2M)
    reset_off = TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0 - mmio_base
    staged = {}
    for name, cfw in fw.items():
      scratch = _INIT_SCRATCH[name]
      spans = []
      for s in cfw.segments:
        if not s.data and s.memsz == 0:
          continue
        data = s.data if s.memsz <= len(s.data) else s.data + b"\0" * (s.memsz - len(s.data))
        addr = s.paddr
        if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
          addr = scratch + (addr - TensixMMIO.LOCAL_RAM_START)
        assert 0 <= addr < TensixL1.SIZE, (
          f"{name}: bad paddr 0x{s.paddr:x} -> 0x{addr:x}"
        )
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
        win.target(core, addr=mmio_base)
        win.write32(reset_off, TensixMMIO.SOFT_RESET_ALL)
        win.target(core, mode=NocOrdering.RELAXED)
        for spans in staged.values():
          for addr, data in spans:
            win.write(addr, data)
        win.write(0, jal)
        win.write(TensixL1.GO_MSG, go_init)
        win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, bank_table)
        win.target(core, addr=mmio_base)
        for reg, text_base in [
          (TensixMMIO.RISCV_DEBUG_REG_NCRISC_RESET_PC, fw["ncrisc"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC0_RESET_PC, fw["trisc0"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC1_RESET_PC, fw["trisc1"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC2_RESET_PC, fw["trisc2"].text_base),
        ]:
          win.write32(reg - mmio_base, text_base)

      for core in all_cores:
        win.target(core, addr=mmio_base)
        win.read32(reset_off)  # fence
        win.write32(reset_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)

      probe = (1, 2) if (1, 2) in all_cores else all_cores[0]
      win.target(probe)
      deadline = time.perf_counter() + 2.0
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"firmware not ready on {probe} — try tt-smi -r")
        time.sleep(0.001)

  def _build_bank_noc_table(self) -> bytes:
    NUM_NOCS, NUM_DRAM_BANKS, NUM_L1_BANKS = 2, 7, 110
    WORKER_EP_LOGICAL = {
      0: [2, 1],
      1: [0, 1],
      2: [0, 1],
      3: [0, 1],
      4: [2, 1],
      5: [2, 1],
      6: [2, 1],
      7: [2, 1],
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
        map_banks(
          mirror + 1,
          TOTAL_BANKS - 1,
          START_X,
          START_Y + (mirror - (half - 1)) * PORTS,
        )
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

  def _wait_core_done(self, core: Core, timeout_s: float = 2.0):
    deadline = time.perf_counter() + timeout_s
    with TLBWindow(self.fd, start=core) as win:
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          go = win.uc[TensixL1.GO_MSG + 3]
          raise TimeoutError(
            f"core {core} firmware init timeout (GO_MSG signal=0x{go:02x}) — try tt-smi -r"
          )
        time.sleep(0.001)

  def _start_dispatch_cores(self):
    cq_kernels = self.compiler.compile_cq_kernels()
    l1a = FastDispatch.L1_ALIGNMENT

    self._wait_core_done(self._prefetch_core)
    self._wait_core_done(self._dispatch_core)

    kernel_off = l1a + 2 * l1a
    pref_rt = b"\0" * l1a
    pref_sems = struct.pack("<I", DEV_DISPATCH_CB_PAGES).ljust(l1a, b"\0") + b"\0" * l1a
    pref_img = pref_rt + pref_sems
    pref_launch = self._build_cq_launch(kernel_off, 0, sem_off=l1a)

    disp_rt = b"\0" * l1a
    disp_sems = b"\0" * l1a + b"\0" * l1a
    disp_img = disp_rt + disp_sems
    ncrisc_off = align_up(kernel_off + len(cq_kernels["dispatch_brisc"].xip), l1a)
    disp_launch = self._build_cq_launch(kernel_off, ncrisc_off, sem_off=l1a)

    self._cq_hw.reset_run_state()
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
      [
        (kernel_off, cq_kernels["dispatch_brisc"].xip),
        (ncrisc_off, cq_kernels["dispatch_s_ncrisc"].xip),
      ],
      init=self._init_dispatch_core_state,
    )

  @staticmethod
  def _build_cq_launch(
    brisc_text_off: int, ncrisc_text_off: int = 0, sem_off: int = 16
  ) -> LaunchMsg:
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

  def _init_dispatch_core_state(self, win: TLBWindow):
    l1a = FastDispatch.L1_ALIGNMENT
    base_16b = self._cq_hw._completion_base_16b
    win.write32(DEV_COMPLETION_Q_WR_PTR_ADDR, base_16b)
    win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, base_16b)
    win.write32(DEV_COMPLETION_Q0_LAST_EVENT_PTR_ADDR, 0)
    win.write32(DEV_COMPLETION_Q1_LAST_EVENT_PTR_ADDR, 0)
    win.uc[
      DEV_DISPATCH_S_SYNC_SEM_ADDR : DEV_DISPATCH_S_SYNC_SEM_ADDR + 8 * l1a
    ] = b"\0" * (8 * l1a)

  def _upload_cq_core(
    self,
    core: Core,
    img: bytes,
    launch: LaunchMsg,
    kernels: list[tuple[int, bytes]],
    init: Callable[[TLBWindow], None] | None = None,
  ):
    win = (
      self._cq_hw._prefetch_win
      if core == self._prefetch_core
      else self._cq_hw._dispatch_win
    )
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

  def _resolve_cores(self, spec: int | Literal["all"]) -> list[Core]:
    if spec == "all":
      return list(self.cores)
    return self.cores[:spec]

  def _go_word(self) -> int:
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go.bits.master_x, go.bits.master_y = self._dispatch_core
    go.bits.dispatch_message_offset = 0
    return go.all

  def alloc(
    self, num_tiles: int, dtype: Dtype, name: str = "", shape: Shape | None = None
  ) -> DramBuffer:
    return self.dram.alloc(num_tiles, dtype, name, shape)

  def alloc_write(
    self, data: bytes, dtype: Dtype, shape: Shape, name: str = ""
  ) -> DramBuffer:
    buf = self.dram.alloc(len(data) // dtype.tile_size, dtype, name, shape)
    self.dram_write(buf, data)
    return buf

  def _ensure_dram_sysmem(self, size: int = 128 * 1024 * 1024):
    need = align_up(size, os.sysconf("SC_PAGE_SIZE"))
    if self._dram_sysmem is not None and self._dram_sysmem.size >= need:
      return
    if self._dram_sysmem is not None:
      self._dram_sysmem.close()
    self._dram_sysmem = Sysmem(self.fd, size=max(128 * 1024 * 1024, need))

  @staticmethod
  def _firmware_src(name: str) -> str:
    return (Path(__file__).parent / "firmware" / name).read_text()

  def _layout_transfer_plan(self, buf: DramBuffer) -> LayoutTransferPlan:
    assert buf.shape is not None
    shape = buf.shape
    assert len(shape) >= 2
    rows, cols = shape[-2], shape[-1]
    assert rows % TILE_R == 0 and cols % TILE_C == 0
    batch = 1
    for d in shape[:-2]:
      batch *= d
    logical_bytes = batch * rows * cols * buf.dtype.bpe
    assert logical_bytes == buf.size, f"shape {shape} does not match tiled buffer size {buf.size}"
    tile_rows = batch * (rows // TILE_R)
    tile_cols = cols // TILE_C
    total_tiles = tile_rows * tile_cols
    n_cores = min(len(self.cores), total_tiles)
    tiles_per_core = (total_tiles + n_cores - 1) // n_cores
    return LayoutTransferPlan(
      shape=shape,
      logical_bytes=logical_bytes,
      tile_cols=tile_cols,
      total_tiles=total_tiles,
      row_bytes=cols * buf.dtype.bpe,
      n_cores=n_cores,
      tiles_per_core=tiles_per_core,
    )

  def _layout_core_range(self, plan: LayoutTransferPlan, core_idx: int) -> tuple[int, int]:
    start = core_idx * plan.tiles_per_core
    count = min(plan.tiles_per_core, plan.total_tiles - start) if start < plan.total_tiles else 0
    return start, count

  def _sysmem_defines(self, buf: DramBuffer, plan: LayoutTransferPlan) -> str:
    pcie_base = (Sysmem.PCIE_NOC_XY << 36) | (1 << 60) | (self._dram_sysmem.noc_addr & ((1 << 36) - 1))
    return (
      f"#define PCIE_BASE 0x{pcie_base:x}ULL\n"
      f"#define TILE_ROW_BYTES {TILE_C * buf.dtype.bpe}\n"
      f"#define TILE_COLS {plan.tile_cols}\n"
      f"#define ROW_BYTES {plan.row_bytes}\n"
    )

  def _build_tilize_transfer_program(self, buf: DramBuffer, plan: LayoutTransferPlan) -> Program:
    sysmem_defs = self._sysmem_defines(buf, plan)
    dram_defs = f"#define DRAM_ADDR {buf.addr}\n"

    def tile_args(core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
      start_tile, num_tiles = self._layout_core_range(plan, core_idx)
      return [start_tile, num_tiles]

    def compute_args(core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
      _, num_tiles = self._layout_core_range(plan, core_idx)
      return [num_tiles]

    return Program(
      cores=plan.n_cores,
      name="dram_fill_tilize",
      reader_kernel=sysmem_defs + TILIZE_READER,
      compute_kernel=TILIZE_COMPUTE,
      writer_kernel=dram_defs + TILIZE_WRITER,
      cbs=[CBConfig(index=0, dtype=buf.dtype, tiles=1), CBConfig(index=16, dtype=buf.dtype, tiles=1)],
      reader_args=tile_args,
      writer_args=tile_args,
      compute_args=compute_args,
      profile=False,
    )

  def _build_untilize_transfer_program(self, buf: DramBuffer, plan: LayoutTransferPlan) -> Program:
    sysmem_defs = self._sysmem_defines(buf, plan)
    dram_defs = f"#define DRAM_ADDR {buf.addr}\n"

    def tile_args(core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
      start_tile, num_tiles = self._layout_core_range(plan, core_idx)
      return [start_tile, num_tiles]

    def compute_args(core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
      _, num_tiles = self._layout_core_range(plan, core_idx)
      return [num_tiles]

    return Program(
      cores=plan.n_cores,
      name="dram_drain_untilize",
      reader_kernel=dram_defs + UNTILIZE_READER,
      compute_kernel=UNTILIZE_COMPUTE,
      writer_kernel=sysmem_defs + UNTILIZE_WRITER,
      cbs=[CBConfig(index=0, dtype=buf.dtype, tiles=1), CBConfig(index=16, dtype=buf.dtype, tiles=1)],
      reader_args=tile_args,
      writer_args=tile_args,
      compute_args=compute_args,
      profile=False,
    )

  def _run_transfer_program(self, program: Program):
    assert not self._programs, "queue must be empty for DRAM transfers"
    self.queue(program)
    self.run()

  def dram_write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    if self._use_fast_dispatch and buf.shape is not None:
      plan = self._layout_transfer_plan(buf)
      assert len(data) == plan.logical_bytes
      self._ensure_dram_sysmem(plan.logical_bytes)
      self._dram_sysmem.buf[:len(data)] = data
      self._run_transfer_program(self._build_tilize_transfer_program(buf, plan))
      return
    if buf.shape is not None:
      data = tilize(data, buf.dtype.bpe, buf.shape)
    self.dram.write(buf, data)

  def dram_read(self, buf: DramBuffer) -> bytes:
    if self._use_fast_dispatch and buf.shape is not None:
      plan = self._layout_transfer_plan(buf)
      self._ensure_dram_sysmem(plan.logical_bytes)
      self._run_transfer_program(self._build_untilize_transfer_program(buf, plan))
      return bytes(self._dram_sysmem.buf[:plan.logical_bytes])
    result = self.dram.read(buf)
    if buf.shape is not None:
      return untilize(result, buf.dtype.bpe, buf.shape)
    return result

  def queue(self, program: Program):
    self._programs.append(program)

  def _compile_commands(self, program: Program, dispatch_mode, host_assigned_id: int = 0) -> list:
    writer = self.compiler.compile_dataflow(program.writer_kernel, "brisc") if program.writer_kernel else None
    reader = self.compiler.compile_dataflow(program.reader_kernel, "ncrisc") if program.reader_kernel else None
    compute = self.compiler.compile_compute(program.compute_kernel, program) if program.compute_kernel else None

    if program.grid is not None:
      rows, cols = program.grid
      grid = [[(x, y) for x in cols] for y in rows]
      all_cores = sorted([c for row in grid for c in row], key=lambda c: (c[0], c[1]))
      n = len(all_cores)
      per_core_args = [
        (resolve_args(program.writer_args, i, c, n), resolve_args(program.reader_args, i, c, n),
         resolve_args(program.compute_args, i, c, n))
        for i, c in enumerate(all_cores)
      ]
      r_recv = self.compiler.compile_dataflow(program.reader_recv_kernel, "ncrisc") if program.reader_recv_kernel else reader
      w_recv = self.compiler.compile_dataflow(program.writer_recv_kernel, "brisc") if program.writer_recv_kernel else writer
      top_left = [grid[0][0]]
      top_row = [grid[0][c] for c in range(1, len(cols))]
      left_col = [grid[r][0] for r in range(1, len(rows))]
      interior = [grid[r][c] for r in range(1, len(rows)) for c in range(1, len(cols))]
      roles = [(cs, rk, wk) for cs, rk, wk in [
        (top_left, reader, writer), (top_row, r_recv, writer), (left_col, reader, w_recv), (interior, r_recv, w_recv),
      ] if cs]
    else:
      cores = self._resolve_cores(program.cores)
      all_cores = cores
      n = len(cores)
      per_core_args = [
        (resolve_args(program.writer_args, i, c, n), resolve_args(program.reader_args, i, c, n),
         resolve_args(program.compute_args, i, c, n))
        for i, c in enumerate(cores)
      ]
      roles = [(cores, reader, writer)]

    return build_commands(program, roles, compute, all_cores, per_core_args, dispatch_mode, host_assigned_id=host_assigned_id)

  def _program_cores(self, program: Program) -> list[Core]:
    if program.grid is not None:
      rows, cols = program.grid
      return sorted([(x, y) for x in cols for y in rows], key=lambda c: (c[0], c[1]))
    return self._resolve_cores(program.cores)

  def _programs_info(self) -> list[dict]:
    info = []
    for i, prog in enumerate(self._programs):
      if not prog.profile:
        continue
      cores = self._program_cores(prog)
      sources = {}
      if prog.reader_kernel: sources["reader"] = prog.reader_kernel
      if prog.writer_kernel: sources["writer"] = prog.writer_kernel
      if prog.compute_kernel: sources["compute"] = prog.compute_kernel
      info.append({"index": i, "name": prog.name or None, "cores": cores, "sources": sources})
    return info

  def run(self) -> list[dict] | None:
    if not self._programs:
      return None
    timing = os.environ.get("TIMING") == "1"
    profiling = self._profiler
    self._set_power_busy()
    try:
      if self._use_fast_dispatch:
        n = len(self._programs)
        if profiling:
          all_rects = mcast_rects(self.cores)
          if not self._profiler_initialized:
            self._enqueue_profiler_init()
            self._profiler_initialized = True
          base = TensixL1.PROFILER_CONTROL
          self.cq.write_packed_large(
            all_rects, base, b"\0" * (5 * 4)
          )  # reset HOST_BUFFER_END × 5
        for i, program in enumerate(self._programs):
          prof_this = profiling and program.profile
          if prof_this:
            self._enqueue_profiler_reset(all_rects)
          ts_slot = 2 * i
          if timing and ts_slot + 1 < TIMESTAMP_MAX_SLOTS:
            self.cq.timestamp(*self._ts_noc_dest(ts_slot))
          prof_id = (i + 1) if prof_this else 0
          commands = self._compile_commands(
            program, self._dispatch_mode, host_assigned_id=prof_id
          )
          fast_enqueue(self.cq, commands, self._go_word())
          if timing and ts_slot + 1 < TIMESTAMP_MAX_SLOTS:
            self.cq.timestamp(*self._ts_noc_dest(ts_slot + 1))
        self._cq_hw._event_id += 1
        self.cq.host_event(self._cq_hw._event_id)
        self._cq_hw.flush(self.cq)
        self._cq_hw.wait_completion(self._cq_hw._event_id)

        if profiling:
          programs_info = self._programs_info()
          if programs_info:
            self.dram.barrier()
            import profiler

            needed = set()
            for info in programs_info:
              needed.update(info["cores"])
            raw_dram = self._read_profiler_dram()
            ctrl_regs = self._read_profiler_ctrl(sorted(needed))
            self.last_profile = profiler.collect(
              programs_info,
              raw_dram,
              ctrl_regs,
              flat_ids=self._profiler_flat_ids,
              page_size=self._profiler_page_size,
              core_count_per_dram=self._profiler_core_count_per_dram,
              harvested_dram_bank=self.harvested_dram,
            )
            profiler.print_summary(self.last_profile)

        if not timing:
          return None
        self.dram.barrier()
        freq_mhz = self._read_arc_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
        timings = []
        for i in range(n):
          ts_slot = 2 * i
          if ts_slot + 1 >= TIMESTAMP_MAX_SLOTS:
            break
          t0 = self._read_ts_slot(ts_slot)
          t1 = self._read_ts_slot(ts_slot + 1)
          cycles = t1 - t0
          timings.append(
            {
              "cycles": cycles,
              "us": cycles / freq_mhz,
              "freq_mhz": freq_mhz,
            }
          )
        for i, t in enumerate(timings):
          print(f"  [{i}] {t['us']:.1f} us ({t['cycles']} cycles)")
        self.last_device_timing = timings
        return timings
      else:
        t0 = time.perf_counter() if timing else 0
        with TLBWindow(self.fd, start=self.cores[0]) as win:
          for program in self._programs:
            commands = self._compile_commands(program, self._dispatch_mode)
            slow_dispatch(win, commands)
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
      if self._cq_hw is not None:
        self._cq_hw.close()
      self.dram.close()
      os.close(self.fd)
