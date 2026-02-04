from __future__ import annotations

import ctypes
import fcntl
import mmap
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from abi import (
  CQDispatchCmdId,
  CQDispatchCmdLarge,
  CQPrefetchCmd,
  CQPrefetchCmdId,
  DevMsgs,
  FastDispatch,
  GoMsg,
  IOCTL_PIN_PAGES,
  IOCTL_UNPIN_PAGES,
  KernelConfigMsg,
  LaunchMsg,
  PinPagesIn,
  PinPagesOutExtended,
  TENSTORRENT_PIN_PAGES_NOC_DMA,
  UnpinPagesIn,
  as_bytes,
)
from configs import TensixL1, TensixMMIO
from device import Device
from helpers import _IO, ioctl, iter_pt_load
from tlb import TLBConfig, TLBMode, TLBSize, TLBWindow

PAGE_SIZE = 4096

def _align_up(n: int, a: int) -> int:
  return (n + a - 1) & ~(a - 1)

def _pack_noc_xy(x: int, y: int) -> int:
  return ((y << 6) | x) & 0xFFFF

@dataclass(frozen=True)
class _HostCQ:
  sysmem_size: int
  issue_base: int
  issue_size: int
  completion_base: int
  completion_size: int

@dataclass(frozen=True)
class _DeviceCQ:
  l1_base: int
  prefetch_q_rd_ptr_addr: int
  prefetch_q_pcie_rd_ptr_addr: int
  completion_q_wr_ptr_addr: int
  completion_q_rd_ptr_addr: int
  completion_q0_last_event_ptr_addr: int
  completion_q1_last_event_ptr_addr: int
  dispatch_s_sync_sem_addr: int
  prefetch_q_base: int
  prefetch_q_size: int
  cmddat_q_base: int
  scratch_db_base: int
  dispatch_cb_base: int
  dispatch_cb_log_page_size: int
  dispatch_cb_pages: int
  dispatch_cb_blocks: int

def _device_cq_layout(*, prefetch_q_entries: int) -> _DeviceCQ:
  l1_base = FastDispatch.BH_TENSIX_DEFAULT_UNRESERVED
  prefetch_q_rd_ptr_addr = l1_base + FastDispatch.BH_PREFETCH_Q_RD_PTR_OFF
  prefetch_q_pcie_rd_ptr_addr = l1_base + FastDispatch.BH_PREFETCH_Q_PCIE_RD_PTR_OFF
  completion_q_wr_ptr_addr = l1_base + FastDispatch.BH_COMPLETION_Q_WR_PTR_OFF
  completion_q_rd_ptr_addr = l1_base + FastDispatch.BH_COMPLETION_Q_RD_PTR_OFF
  completion_q0_last_event_ptr_addr = l1_base + FastDispatch.BH_COMPLETION_Q0_LAST_EVENT_PTR_OFF
  completion_q1_last_event_ptr_addr = l1_base + FastDispatch.BH_COMPLETION_Q1_LAST_EVENT_PTR_OFF
  dispatch_s_sync_sem_addr = l1_base + FastDispatch.BH_DISPATCH_S_SYNC_SEM_OFF
  prefetch_q_base = l1_base + FastDispatch.BH_UNRESERVED_OFF
  prefetch_q_size = prefetch_q_entries * 2
  cmddat_q_base = _align_up(prefetch_q_base + prefetch_q_size, FastDispatch.PCIE_ALIGNMENT)
  scratch_db_base = _align_up(cmddat_q_base + FastDispatch.PREFETCH_CMDDAT_Q_SIZE, FastDispatch.PCIE_ALIGNMENT)
  dispatch_cb_base = _align_up(l1_base + FastDispatch.BH_UNRESERVED_OFF, 1 << 12)
  dispatch_cb_log_page_size = 12
  dispatch_cb_pages = (512 * 1024) >> dispatch_cb_log_page_size
  dispatch_cb_blocks = 4
  return _DeviceCQ(
    l1_base=l1_base,
    prefetch_q_rd_ptr_addr=prefetch_q_rd_ptr_addr,
    prefetch_q_pcie_rd_ptr_addr=prefetch_q_pcie_rd_ptr_addr,
    completion_q_wr_ptr_addr=completion_q_wr_ptr_addr,
    completion_q_rd_ptr_addr=completion_q_rd_ptr_addr,
    completion_q0_last_event_ptr_addr=completion_q0_last_event_ptr_addr,
    completion_q1_last_event_ptr_addr=completion_q1_last_event_ptr_addr,
    dispatch_s_sync_sem_addr=dispatch_s_sync_sem_addr,
    prefetch_q_base=prefetch_q_base,
    prefetch_q_size=prefetch_q_size,
    cmddat_q_base=cmddat_q_base,
    scratch_db_base=scratch_db_base,
    dispatch_cb_base=dispatch_cb_base,
    dispatch_cb_log_page_size=dispatch_cb_log_page_size,
    dispatch_cb_pages=dispatch_cb_pages,
    dispatch_cb_blocks=dispatch_cb_blocks,
  )

def _host_cq_layout(*, sysmem_size: int, issue_size: int, completion_size: int) -> _HostCQ:
  sysmem_size = _align_up(sysmem_size, PAGE_SIZE)
  issue_base = FastDispatch.HOST_UNRESERVED_OFF
  issue_size = _align_up(issue_size, FastDispatch.PCIE_ALIGNMENT)
  completion_base = issue_base + issue_size
  completion_size = _align_up(completion_size, FastDispatch.PCIE_ALIGNMENT)
  need = completion_base + completion_size
  if need > sysmem_size:
    raise ValueError(f"sysmem_size too small: need {need}, have {sysmem_size}")
  return _HostCQ(
    sysmem_size=sysmem_size,
    issue_base=issue_base,
    issue_size=issue_size,
    completion_base=completion_base,
    completion_size=completion_size,
  )

class _FastCQ:
  def __init__(
    self,
    device: Device,
    *,
    prefetch_core: tuple[int, int],
    dispatch_core: tuple[int, int],
    sysmem_size: int,
    issue_size: int,
    completion_size: int,
    prefetch_q_entries: int,
  ):
    self.device = device
    self.prefetch_core = prefetch_core
    self.dispatch_core = dispatch_core
    self.host = _host_cq_layout(sysmem_size=sysmem_size, issue_size=issue_size, completion_size=completion_size)
    self.dev = _device_cq_layout(prefetch_q_entries=prefetch_q_entries)

    self.sysmem = mmap.mmap(
      -1,
      self.host.sysmem_size,
      flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    self.sysmem_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.sysmem))
    if (self.sysmem_addr % PAGE_SIZE) != 0 or (self.host.sysmem_size % PAGE_SIZE) != 0:
      raise RuntimeError("sysmem must be page-aligned and page-sized")

    pin = ioctl(
      self.device.fd,
      IOCTL_PIN_PAGES,
      PinPagesIn,
      PinPagesOutExtended,
      output_size_bytes=ctypes.sizeof(PinPagesOutExtended),
      flags=TENSTORRENT_PIN_PAGES_NOC_DMA,
      virtual_address=self.sysmem_addr,
      size=self.host.sysmem_size,
    )
    self.noc_addr = int(pin.noc_address)
    if (self.noc_addr & FastDispatch.PCIE_NOC_BASE) != FastDispatch.PCIE_NOC_BASE:
      raise RuntimeError(f"bad NOC sysmem address: 0x{self.noc_addr:x}")
    self.noc_local = self.noc_addr - FastDispatch.PCIE_NOC_BASE
    if self.noc_local > 0xFFFF_FFFF:
      raise RuntimeError(f"sysmem NOC offset too large for CQ: 0x{self.noc_local:x}")

    self.issue_wr = 0
    self.prefetch_q_wr_idx = 0

    cfg = TLBConfig(addr=0, start=prefetch_core, end=prefetch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    self.prefetch_win = TLBWindow(self.device.fd, TLBSize.MiB_2, cfg)

  def close(self):
    try:
      if hasattr(self, "prefetch_win"):
        self.prefetch_win.free()
    finally:
      try:
        if hasattr(self, "sysmem"):
          buf = bytearray(ctypes.sizeof(UnpinPagesIn))
          view = UnpinPagesIn.from_buffer(buf)
          view.virtual_address = self.sysmem_addr
          view.size = self.host.sysmem_size
          view.reserved = 0
          fcntl.ioctl(self.device.fd, _IO(IOCTL_UNPIN_PAGES), buf, False)
          self.sysmem.close()
      except Exception:
        pass

  def _write_prefetch_q_entry(self, size_16b: int):
    if not (0 < size_16b <= 0x7FFF):
      raise ValueError(f"prefetch entry out of range: {size_16b}")
    idx = self.prefetch_q_wr_idx
    off = self.dev.prefetch_q_base + (idx * 2)
    self.prefetch_win.uc[off:off + 2] = struct.pack("<H", size_16b)
    self.prefetch_q_wr_idx = (idx + 1) % (self.dev.prefetch_q_size // 2)

  def _issue_write(self, record: bytes):
    record = bytes(record)
    if len(record) % FastDispatch.PCIE_ALIGNMENT != 0:
      raise ValueError("record must be 64B-aligned")
    wr = _align_up(self.issue_wr, FastDispatch.PCIE_ALIGNMENT)
    end = wr + len(record)
    if end > self.host.issue_size:
      raise RuntimeError("issue queue overflow (no wrap support yet)")
    base = self.host.issue_base + wr
    self.sysmem[base:base + len(record)] = record
    self.issue_wr = end
    self._write_prefetch_q_entry(len(record) >> 4)

  def enqueue_write_linear(self, *, tile: tuple[int, int], addr: int, data: bytes):
    x, y = tile
    dispatch = CQDispatchCmdLarge()
    dispatch.cmd_id = CQDispatchCmdId.WRITE_LINEAR
    dispatch.payload.write_linear.num_mcast_dests = 0
    dispatch.payload.write_linear.write_offset_index = 0
    dispatch.payload.write_linear.pad1 = 0
    dispatch.payload.write_linear.noc_xy_addr = _pack_noc_xy(x, y)
    dispatch.payload.write_linear.addr = addr
    dispatch.payload.write_linear.length = len(data)
    payload = as_bytes(dispatch) + data

    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0  # DispatcherSelect::DISPATCH_MASTER
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(payload)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(payload), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(payload))
    self._issue_write(as_bytes(prefetch) + payload + pad)

  def init_prefetch_l1(self):
    end_ptr = self.dev.prefetch_q_base + self.dev.prefetch_q_size
    self.prefetch_win.uc[self.dev.prefetch_q_rd_ptr_addr:self.dev.prefetch_q_rd_ptr_addr + 4] = struct.pack("<I", end_ptr)
    pcie_base = self.noc_local + self.host.issue_base
    self.prefetch_win.uc[self.dev.prefetch_q_pcie_rd_ptr_addr:self.dev.prefetch_q_pcie_rd_ptr_addr + 4] = struct.pack("<I", pcie_base)
    self.prefetch_win.uc[self.dev.prefetch_q_base:self.dev.prefetch_q_base + self.dev.prefetch_q_size] = b"\0" * self.dev.prefetch_q_size

def _load_dispatch_elf(path: Path) -> tuple[list, int]:
  """Parse dispatch ELF, return (PT_LOAD segments, entry point)."""
  elf = path.read_bytes()
  entry = struct.unpack_from("<I", elf, 24)[0]  # e_entry in ELF32
  return list(iter_pt_load(elf)), entry

def _write_firmware_segs(win: TLBWindow, segs: list):
  """Write ELF PT_LOAD segments directly to their linked L1 addresses."""
  for seg in segs:
    if not seg.data and seg.memsz == 0: continue
    data = seg.data
    if seg.memsz > len(data): data += b"\0" * (seg.memsz - len(data))
    addr = seg.paddr
    if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
      addr = TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH + (addr - TensixMMIO.LOCAL_RAM_START)
    if 0 <= addr < TensixL1.SIZE:
      win.write(addr, data, use_uc=True, restore=False)

def _build_dispatch_launch(
  *, rt_args: list[int], sem_values: list[int], entry: int, brisc_noc_id: int = 0,
) -> tuple[bytes, KernelConfigMsg]:
  """Build minimal kernel config image and KernelConfigMsg for dispatch cores."""
  L1A = FastDispatch.L1_ALIGNMENT  # 16
  # RT args padded to L1_ALIGNMENT
  rt_blob = b"".join((a & 0xFFFFFFFF).to_bytes(4, "little") for a in rt_args)
  rt_blob = rt_blob.ljust(L1A, b"\0")
  # Semaphores, each padded to L1_ALIGNMENT
  sem_blob = b"".join(
    (v & 0xFFFFFFFF).to_bytes(4, "little").ljust(L1A, b"\0") for v in sem_values
  )
  img = rt_blob + sem_blob

  kc = KernelConfigMsg()
  kc.kernel_config_base[0] = TensixL1.KERNEL_CONFIG_BASE
  kc.kernel_config_base[1] = TensixL1.KERNEL_CONFIG_BASE
  kc.kernel_config_base[2] = TensixL1.KERNEL_CONFIG_BASE
  kc.sem_offset[0] = L1A  # semaphores start after rt_args
  kc.rta_offset[0].rta_offset = 0
  kc.rta_offset[0].crta_offset = len(rt_blob)
  # u32 wrapping: KERNEL_CONFIG_BASE + offset = entry on 32-bit RISC-V
  kc.kernel_text_offset[0] = (entry - TensixL1.KERNEL_CONFIG_BASE) & 0xFFFFFFFF
  kc.enables = 1  # BRISC only
  kc.brisc_noc_id = brisc_noc_id
  kc.mode = DevMsgs.DISPATCH_MODE_HOST
  kc.local_cb_mask = 0
  kc.min_remote_cb_start_index = 32
  return bytes(img), kc

class FastDevice(Device):
  def __init__(
    self,
    path: str = "/dev/tenstorrent/0",
    *,
    upload_firmware: bool = True,
    noc_translation_enabled: bool | dict[int, bool] | None = None,
    sysmem_size: int = 16 * 1024 * 1024,
    issue_size: int = 8 * 1024 * 1024,
    completion_size: int = 4 * 1024 * 1024,
  ):
    super().__init__(path, upload_firmware=upload_firmware, noc_translation_enabled=noc_translation_enabled)
    dispatch_x = max(x1 for x0, x1 in self.tiles.tensix_mcast)
    y0, _ = self.tiles.TENSIX_Y
    self.prefetch_core = (dispatch_x, y0)
    self.dispatch_core = (dispatch_x, y0 + 1)
    self._cq = _FastCQ(
      self,
      prefetch_core=self.prefetch_core,
      dispatch_core=self.dispatch_core,
      sysmem_size=sysmem_size,
      issue_size=issue_size,
      completion_size=completion_size,
      prefetch_q_entries=FastDispatch.PREFETCH_Q_ENTRIES_WORKER_DEFAULT,
    )
    self._start_dispatch_cores()

  def close(self):
    if hasattr(self, "_cq"):
      self._cq.close()
    super().close()

  def _write_l1_i32(self, win: TLBWindow, addr: int, v: int):
    win.uc[addr:addr + 4] = struct.pack("<I", v & 0xFFFF_FFFF)

  def _start_dispatch_cores(self):
    cq = self._cq
    dev = cq.dev

    completion_base = cq.noc_local + cq.host.completion_base
    if cq.noc_local != 0:
      raise RuntimeError(f"expected pinned sysmem noc_local=0, got 0x{cq.noc_local:x} (run tt-smi -r and retry)")

    fw_dir = Path(__file__).parent / "riscv-firmware" / self.arch
    prefetch_segs, prefetch_entry = _load_dispatch_elf(fw_dir / "cq_prefetch_brisc.elf")
    dispatch_segs, dispatch_entry = _load_dispatch_elf(fw_dir / "cq_dispatch_brisc.elf")

    # --- Prefetch core: load firmware directly to linked L1 addresses ---
    cfg0 = TLBConfig(addr=0, start=self.prefetch_core, end=self.prefetch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg0) as win:
      _write_firmware_segs(win, prefetch_segs)
      img, kc = _build_dispatch_launch(
        rt_args=[0, 0, 0],
        sem_values=[dev.dispatch_cb_pages, 0],  # sem0=downstream credits, sem1=sync
        entry=prefetch_entry,
      )
      cq.init_prefetch_l1()
      win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
      reset = GoMsg(); reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
      launch = LaunchMsg(); launch.kernel_config = kc
      go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO
      win.write(TensixL1.GO_MSG, as_bytes(reset), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little"), use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(launch), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG, as_bytes(go), use_uc=True, restore=False)

    # --- Dispatch core: load firmware directly to linked L1 addresses ---
    cfg1 = TLBConfig(addr=0, start=self.dispatch_core, end=self.dispatch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg1) as win:
      _write_firmware_segs(win, dispatch_segs)
      img, kc = _build_dispatch_launch(
        rt_args=[0, 0, 0],
        sem_values=[0, 0],  # sem0=pages ready (starts empty)
        entry=dispatch_entry,
      )
      win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
      base_16b = (completion_base >> 4) & 0x7FFF_FFFF
      self._write_l1_i32(win, dev.completion_q_wr_ptr_addr, base_16b)
      self._write_l1_i32(win, dev.completion_q_rd_ptr_addr, base_16b)
      self._write_l1_i32(win, dev.completion_q0_last_event_ptr_addr, 0)
      self._write_l1_i32(win, dev.completion_q1_last_event_ptr_addr, 0)
      win.uc[dev.dispatch_s_sync_sem_addr:dev.dispatch_s_sync_sem_addr + (8 * FastDispatch.L1_ALIGNMENT)] = b"\0" * (8 * FastDispatch.L1_ALIGNMENT)
      reset = GoMsg(); reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
      launch = LaunchMsg(); launch.kernel_config = kc
      go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO
      win.write(TensixL1.GO_MSG, as_bytes(reset), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little"), use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(launch), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG, as_bytes(go), use_uc=True, restore=False)

    time.sleep(0.01)

  def run(
    self,
    *,
    cores: list[tuple[int, int]],
    kernels: dict[str, object],
    rt_args: dict[str, list[int]],
    brisc_noc_id: int = 0,
  ):
    if len(cores) != 1:
      raise ValueError("fast dispatch currently supports exactly one core")
    if brisc_noc_id not in (0, 1):
      raise ValueError("brisc_noc_id must be 0 or 1")

    core = cores[0]

    # Reset CQ state so the issue buffer can be reused from the beginning.
    # Safe because the previous run() polled to completion before returning.
    cq = self._cq
    cq.issue_wr = 0
    cq.prefetch_q_wr_idx = 0
    pcie_base = cq.noc_local + cq.host.issue_base
    cq.prefetch_win.uc[cq.dev.prefetch_q_pcie_rd_ptr_addr:cq.dev.prefetch_q_pcie_rd_ptr_addr + 4] = struct.pack("<I", pcie_base)

    img, kc = self._pack_kernel_config(kernels, rt_args, brisc_noc_id=brisc_noc_id)

    reset = GoMsg(); reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    launch = LaunchMsg(); launch.kernel_config = kc
    go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO

    self._cq.enqueue_write_linear(tile=core, addr=TensixL1.KERNEL_CONFIG_BASE, data=img)
    self._cq.enqueue_write_linear(tile=core, addr=TensixL1.GO_MSG, data=as_bytes(reset))
    self._cq.enqueue_write_linear(tile=core, addr=TensixL1.GO_MSG_INDEX, data=(0).to_bytes(4, "little"))
    self._cq.enqueue_write_linear(tile=core, addr=TensixL1.LAUNCH, data=as_bytes(launch))
    self._cq.enqueue_write_linear(tile=core, addr=TensixL1.GO_MSG, data=as_bytes(go))

    cfg = TLBConfig(addr=0, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      deadline = time.perf_counter() + 10.0
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"timeout waiting for core {core}")
        time.sleep(0.001)
