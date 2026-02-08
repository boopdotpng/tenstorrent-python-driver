import ctypes, fcntl, mmap, struct, time
from dataclasses import dataclass
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import _IO, align_down
from codegen import CQConfig, compile_cq_kernels
from dram import DramAllocator
from device_runtime import CommonDevice, Program, ArgGen

PAGE_SIZE = 4096

def _align_up(n: int, a: int) -> int:
  return (n + a - 1) & ~(a - 1)

def _pack_noc_xy(x: int, y: int) -> int:
  return ((y << 6) | x) & 0xFFFF

def _pack_noc_mcast_xy(x: int, y: int) -> int:
  """Encode a 1x1 multicast rectangle (same start/end) for NOC dispatch operations.
  BH format: [y_start:18][x_start:12][y_end:6][x_end:0], 6 bits each."""
  return (y << 18) | (x << 12) | (y << 6) | x

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
  dispatch_cb_pages: int

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
  prefetch_q_size = prefetch_q_entries * FastDispatch.PREFETCH_Q_ENTRY_BYTES
  cmddat_q_base = _align_up(prefetch_q_base + prefetch_q_size, FastDispatch.PCIE_ALIGNMENT)
  scratch_db_base = _align_up(cmddat_q_base + FastDispatch.PREFETCH_CMDDAT_Q_SIZE, FastDispatch.PCIE_ALIGNMENT)
  dispatch_cb_pages = (512 * 1024) >> 12
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
    dispatch_cb_pages=dispatch_cb_pages,
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
    fd: int,
    *,
    prefetch_core: tuple[int, int],
    dispatch_core: tuple[int, int],
    sysmem_size: int,
    issue_size: int,
    completion_size: int,
    prefetch_q_entries: int,
  ):
    self.fd = fd
    self.host = _host_cq_layout(sysmem_size=sysmem_size, issue_size=issue_size, completion_size=completion_size)
    self.dev = _device_cq_layout(prefetch_q_entries=prefetch_q_entries)

    self.sysmem = mmap.mmap(
      -1,
      self.host.sysmem_size,
      flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    self.sysmem_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.sysmem))
    if (self.sysmem_addr % PAGE_SIZE) != 0 or (self.host.sysmem_size % PAGE_SIZE) != 0:
      raise RuntimeError("sysmem must be page-aligned and page-sized")

    buf = bytearray(ctypes.sizeof(PinPagesIn) + ctypes.sizeof(PinPagesOutExtended))
    pin = PinPagesIn.from_buffer(buf)
    pin.output_size_bytes = ctypes.sizeof(PinPagesOutExtended)
    pin.flags = PIN_PAGES_NOC_DMA
    pin.virtual_address = self.sysmem_addr
    pin.size = self.host.sysmem_size
    fcntl.ioctl(self.fd, _IO(IOCTL_PIN_PAGES), buf, True)
    out = PinPagesOutExtended.from_buffer(buf, ctypes.sizeof(PinPagesIn))
    self.noc_addr = int(out.noc_address)
    if (self.noc_addr & FastDispatch.PCIE_NOC_BASE) != FastDispatch.PCIE_NOC_BASE:
      raise RuntimeError(f"bad NOC sysmem address: 0x{self.noc_addr:x}")
    self.noc_local = self.noc_addr - FastDispatch.PCIE_NOC_BASE
    if self.noc_local > 0xFFFF_FFFF:
      raise RuntimeError(f"sysmem NOC offset too large for CQ: 0x{self.noc_local:x}")

    self.issue_wr = 0
    self.prefetch_q_wr_idx = 0

    cfg = TLBConfig(addr=0, start=prefetch_core, end=prefetch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    self.prefetch_win = TLBWindow(self.fd, TLBSize.MiB_2, cfg)
    dcfg = TLBConfig(addr=0, start=dispatch_core, end=dispatch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    self.dispatch_win = TLBWindow(self.fd, TLBSize.MiB_2, dcfg)
    self._completion_page_16b = 4096 >> 4
    self._completion_base_16b = (self.noc_local + self.host.completion_base) >> 4
    self._completion_end_16b = (self.noc_local + self.host.completion_base + self.host.completion_size) >> 4
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    self._init_host_completion_ctrl()
    self._write_completion_rd_ptr()
    self.profile_reset()

  def close(self):
    try:
      if hasattr(self, "prefetch_win"): self.prefetch_win.free()
      if hasattr(self, "dispatch_win"): self.dispatch_win.free()
    finally:
      if hasattr(self, "sysmem"):
        unpin = UnpinPagesIn(virtual_address=self.sysmem_addr, size=self.host.sysmem_size, reserved=0)
        fcntl.ioctl(self.fd, _IO(IOCTL_UNPIN_PAGES), bytearray(as_bytes(unpin)), False)
        self.sysmem.close()

  def _read_prefetch_entry(self, idx: int) -> int:
    off = self.dev.prefetch_q_base + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    return struct.unpack("<H", self.prefetch_win.uc[off : off + 2])[0]

  def _wait_prefetch_slot_free(self, idx: int, timeout_s: float = 1.0) -> float:
    t0 = time.perf_counter()
    deadline = time.perf_counter() + timeout_s
    while self._read_prefetch_entry(idx) != 0:
      if time.perf_counter() > deadline:
        raise TimeoutError("timeout waiting for prefetch queue slot")
    return time.perf_counter() - t0

  def _write_prefetch_q_entry(self, size_16b: int):
    if not (0 < size_16b <= 0x7FFF):
      raise ValueError(f"prefetch entry out of range: {size_16b}")
    idx = self.prefetch_q_wr_idx
    wait_dt = self._wait_prefetch_slot_free(idx)
    self._profile_wait_s += wait_dt
    if wait_dt > 1e-9:
      self._profile_wait_count += 1
    off = self.dev.prefetch_q_base + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self.prefetch_win.uc[off : off + 2] = struct.pack("<H", size_16b)
    entries = self.dev.prefetch_q_size // FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self.prefetch_q_wr_idx = (idx + 1) % entries

  def _issue_write(self, record: bytes):
    if len(record) % FastDispatch.PCIE_ALIGNMENT != 0:
      raise ValueError("record must be 64B-aligned")
    t0 = time.perf_counter()
    wr = _align_up(self.issue_wr, FastDispatch.PCIE_ALIGNMENT)
    if wr + len(record) > self.host.issue_size: wr = 0
    base = self.host.issue_base + wr
    self.sysmem[base : base + len(record)] = record
    self.issue_wr = wr + len(record)
    self._write_prefetch_q_entry(len(record) >> 4)
    self._profile_entries += 1
    self._profile_bytes += len(record)
    self._profile_issue_s += time.perf_counter() - t0

  def profile_reset(self):
    self._profile_entries = 0
    self._profile_bytes = 0
    self._profile_wait_count = 0
    self._profile_wait_s = 0.0
    self._profile_issue_s = 0.0

  def profile_snapshot(self) -> dict[str, float]:
    return {
      "entries": float(self._profile_entries),
      "bytes": float(self._profile_bytes),
      "wait_count": float(self._profile_wait_count),
      "wait_ms": self._profile_wait_s * 1e3,
      "issue_ms": self._profile_issue_s * 1e3,
    }

  def enqueue_write_packed(self, *, cores: list[tuple[int, int]], addr: int, data: bytes | list[bytes]):
    count = len(cores)
    uniform = isinstance(data, bytes)
    payload_size = len(data) if uniform else len(data[0])

    # Build dispatch cmd (16 bytes)
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQDispatchCmdId.WRITE_PACKED
    dispatch.payload.write_packed.flags = CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE if uniform else 0
    dispatch.payload.write_packed.count = count
    dispatch.payload.write_packed.write_offset_index = 0
    dispatch.payload.write_packed.size = payload_size
    dispatch.payload.write_packed.addr = addr

    # Build sub-cmd array (count × 4 bytes), padded to L1_ALIGNMENT
    sub_cmds = b"".join(
      as_bytes(CQDispatchWritePackedUnicastSubCmd(noc_xy_addr=_pack_noc_xy(x, y)))
      for x, y in cores
    )
    sub_cmds_padded = sub_cmds.ljust(_align_up(len(sub_cmds), FastDispatch.L1_ALIGNMENT), b"\0")

    # Build data section
    if uniform:
      data_section = bytes(data).ljust(_align_up(payload_size, FastDispatch.L1_ALIGNMENT), b"\0")
    else:
      stride = _align_up(payload_size, FastDispatch.L1_ALIGNMENT)
      data_section = b"".join(bytes(d).ljust(stride, b"\0") for d in data)

    inner = as_bytes(dispatch) + sub_cmds_padded + data_section

    # Wrap in prefetch relay inline cmd
    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(inner)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(inner), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(inner))
    self._issue_write(as_bytes(prefetch) + inner + pad)

  def enqueue_write_packed_large(self, *, dests: list[tuple[int, int]], addr: int, data: bytes):
    """Packed large write: sends identical data to mcast destinations.
    dests: list of (noc_xy_mcast, num_mcast_dests) — each entry is one sub-cmd."""
    alignment = FastDispatch.L1_ALIGNMENT
    data_padded = bytes(data).ljust(_align_up(len(data), alignment), b"\0")
    max_batch = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS

    for batch_start in range(0, len(dests), max_batch):
      batch = dests[batch_start : batch_start + max_batch]
      count = len(batch)

      dispatch = CQDispatchCmd()
      dispatch.cmd_id = CQDispatchCmdId.WRITE_PACKED_LARGE
      dispatch.payload.write_packed_large.type = 2  # PROGRAM_BINARIES
      dispatch.payload.write_packed_large.count = count
      dispatch.payload.write_packed_large.alignment = alignment
      dispatch.payload.write_packed_large.write_offset_index = 0

      sub_cmds = b"".join(
        as_bytes(CQDispatchWritePackedLargeSubCmd(
          noc_xy_addr=noc_xy, addr=addr, length_minus1=len(data) - 1,
          num_mcast_dests=n_dests,
          flags=CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK,
        )) for noc_xy, n_dests in batch
      )
      sub_cmds_padded = sub_cmds.ljust(_align_up(len(sub_cmds), alignment), b"\0")
      data_section = data_padded * count

      inner = as_bytes(dispatch) + sub_cmds_padded + data_section
      prefetch = CQPrefetchCmd()
      prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
      prefetch.payload.relay_inline.dispatcher_type = 0
      prefetch.payload.relay_inline.pad = 0
      prefetch.payload.relay_inline.length = len(inner)
      stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(inner), FastDispatch.PCIE_ALIGNMENT)
      prefetch.payload.relay_inline.stride = stride
      pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(inner))
      self._issue_write(as_bytes(prefetch) + inner + pad)

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
    payload = as_bytes(dispatch) + bytes(data)

    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(payload)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(payload), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(payload))
    self._issue_write(as_bytes(prefetch) + payload + pad)

  def enqueue_set_go_signal_noc_data(self, *, noc_words: list[int]):
    payload_words = b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in noc_words)
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQDispatchCmdId.SET_GO_SIGNAL_NOC_DATA
    dispatch.payload.set_go_signal_noc_data.num_words = len(noc_words)
    payload = as_bytes(dispatch) + payload_words

    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(payload)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(payload), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(payload))
    self._issue_write(as_bytes(prefetch) + payload + pad)

  def enqueue_send_go_signal(
    self,
    *,
    go_signal: int,
    wait_stream: int,
    wait_count: int,
    num_unicast_txns: int,
    noc_data_start_index: int = 0,
    multicast_go_offset: int = CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET,
  ):
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQDispatchCmdId.SEND_GO_SIGNAL
    dispatch.payload.mcast.go_signal = go_signal & 0xFFFFFFFF
    dispatch.payload.mcast.multicast_go_offset = multicast_go_offset & 0xFF
    dispatch.payload.mcast.num_unicast_txns = num_unicast_txns & 0xFF
    dispatch.payload.mcast.noc_data_start_index = noc_data_start_index & 0xFF
    dispatch.payload.mcast.wait_count = wait_count & 0xFFFFFFFF
    dispatch.payload.mcast.wait_stream = wait_stream & 0xFFFFFFFF
    payload = as_bytes(dispatch)

    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(payload)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(payload), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(payload))
    self._issue_write(as_bytes(prefetch) + payload + pad)

  def enqueue_wait_stream(self, *, stream: int, count: int, clear_stream: bool = True):
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQDispatchCmdId.WAIT
    flags = CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM
    if clear_stream:
      flags |= CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM
    dispatch.payload.wait.flags = flags
    dispatch.payload.wait.stream = stream
    dispatch.payload.wait.addr = 0
    dispatch.payload.wait.count = count
    payload = as_bytes(dispatch)

    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(payload)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(payload), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(payload))
    self._issue_write(as_bytes(prefetch) + payload + pad)

  def enqueue_host_event(self, *, event_id: int, pad1: int = 0):
    payload_data = struct.pack("<I", event_id & 0xFFFFFFFF).ljust(FastDispatch.L1_ALIGNMENT, b"\0")
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQDispatchCmdId.WRITE_LINEAR_H_HOST
    dispatch.payload.write_linear_host.is_event = 1
    dispatch.payload.write_linear_host.pad1 = pad1 & 0xFFFF
    dispatch.payload.write_linear_host.pad2 = 0
    dispatch.payload.write_linear_host.length = ctypes.sizeof(CQDispatchCmd) + len(payload_data)
    payload = as_bytes(dispatch) + payload_data

    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQPrefetchCmdId.RELAY_INLINE
    prefetch.payload.relay_inline.dispatcher_type = 0
    prefetch.payload.relay_inline.pad = 0
    prefetch.payload.relay_inline.length = len(payload)
    stride = _align_up(ctypes.sizeof(CQPrefetchCmd) + len(payload), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(payload))
    self._issue_write(as_bytes(prefetch) + payload + pad)

  def _read_completion_wr_ptr(self) -> tuple[int, int]:
    raw = struct.unpack("<I", self.sysmem[FastDispatch.HOST_COMPLETION_Q_WR_OFF : FastDispatch.HOST_COMPLETION_Q_WR_OFF + 4])[0]
    return raw & 0x7FFF_FFFF, (raw >> 31) & 0x1

  def _read_completion_event_id(self) -> int:
    off = (self._completion_rd_16b << 4) - self.noc_local
    return struct.unpack("<I", self.sysmem[off + ctypes.sizeof(CQDispatchCmd) : off + ctypes.sizeof(CQDispatchCmd) + 4])[0]

  def _write_completion_rd_ptr(self):
    raw = (self._completion_rd_16b & 0x7FFF_FFFF) | (self._completion_rd_toggle << 31)
    self.dispatch_win.write32(self.dev.completion_q_rd_ptr_addr, raw)
    self.sysmem[FastDispatch.HOST_COMPLETION_Q_RD_OFF : FastDispatch.HOST_COMPLETION_Q_RD_OFF + 4] = struct.pack("<I", raw)

  def _init_host_completion_ctrl(self):
    raw = self._completion_base_16b & 0x7FFF_FFFF
    self.sysmem[FastDispatch.HOST_COMPLETION_Q_WR_OFF : FastDispatch.HOST_COMPLETION_Q_WR_OFF + 4] = struct.pack("<I", raw)
    self.sysmem[FastDispatch.HOST_COMPLETION_Q_RD_OFF : FastDispatch.HOST_COMPLETION_Q_RD_OFF + 4] = struct.pack("<I", raw)

  def _pop_completion_page(self):
    self._completion_rd_16b += self._completion_page_16b
    if self._completion_rd_16b >= self._completion_end_16b:
      self._completion_rd_16b = self._completion_base_16b + (self._completion_rd_16b - self._completion_end_16b)
      self._completion_rd_toggle ^= 1
    self._write_completion_rd_ptr()

  def wait_host_event(self, *, event_id: int, timeout_s: float):
    deadline = time.perf_counter() + timeout_s
    while True:
      wr_16b, wr_toggle = self._read_completion_wr_ptr()
      if (wr_16b != self._completion_rd_16b) or (wr_toggle != self._completion_rd_toggle):
        got = self._read_completion_event_id()
        self._pop_completion_page()
        if got != (event_id & 0xFFFFFFFF):
          raise RuntimeError(f"completion event mismatch: got {got}, expected {event_id}")
        return
      if time.perf_counter() > deadline:
        wr_16b, wr_toggle = self._read_completion_wr_ptr()
        print(f"  completion wr=0x{wr_16b:x} toggle={wr_toggle} rd=0x{self._completion_rd_16b:x} rd_toggle={self._completion_rd_toggle}")
        raise TimeoutError("timeout waiting for CQ host completion event")
      time.sleep(0.0002)

  def init_prefetch_l1(self):
    end_ptr = self.dev.prefetch_q_base + self.dev.prefetch_q_size
    self.prefetch_win.uc[self.dev.prefetch_q_rd_ptr_addr : self.dev.prefetch_q_rd_ptr_addr + 4] = struct.pack("<I", end_ptr)
    pcie_base = self.noc_local + self.host.issue_base
    self.prefetch_win.uc[self.dev.prefetch_q_pcie_rd_ptr_addr : self.dev.prefetch_q_pcie_rd_ptr_addr + 4] = struct.pack("<I", pcie_base)
    self.prefetch_win.uc[self.dev.prefetch_q_base : self.dev.prefetch_q_base + self.dev.prefetch_q_size] = b"\0" * self.dev.prefetch_q_size

  def reset_run_state(self):
    self.issue_wr = 0
    self.prefetch_q_wr_idx = 0
    end_ptr = self.dev.prefetch_q_base + self.dev.prefetch_q_size
    self.prefetch_win.uc[self.dev.prefetch_q_rd_ptr_addr : self.dev.prefetch_q_rd_ptr_addr + 4] = struct.pack("<I", end_ptr)
    pcie_base = self.noc_local + self.host.issue_base
    self.prefetch_win.uc[self.dev.prefetch_q_pcie_rd_ptr_addr : self.dev.prefetch_q_pcie_rd_ptr_addr + 4] = struct.pack("<I", pcie_base)
    self.prefetch_win.uc[self.dev.prefetch_q_base : self.dev.prefetch_q_base + self.dev.prefetch_q_size] = b"\0" * self.dev.prefetch_q_size
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    self._init_host_completion_ctrl()
    self._write_completion_rd_ptr()

class SlowDevice(CommonDevice):
  def __init__(self, device: int = 0):
    super().__init__(device=device)
    self.win = TLBWindow(self.fd, TLBSize.MiB_2)
    self.dram = DramAllocator(fd=self.fd, dram_tiles=self.tiles.dram, run_fn=self.run)

  @staticmethod
  def _tile_ready(win: TLBWindow) -> bool:
    go = win.uc[TensixL1.GO_MSG + 3]
    sync = win.read32(TensixL1.MAILBOX_BASE + 8)
    dm1, tr0, tr1, tr2 = sync & 0xFF, (sync >> 8) & 0xFF, (sync >> 16) & 0xFF, (sync >> 24) & 0xFF
    return go == DevMsgs.RUN_MSG_DONE and dm1 == 0 and tr1 == 0 and tr2 == 0 and tr0 in (0, 3)

  def _build_local_cb_blob(self, program: Program) -> tuple[int, bytes]:
    mask = 0
    for cb in program.cbs:
      mask |= 1 << cb
    end = mask.bit_length()
    arr = (LocalCBConfig * end)()
    addr = TensixL1.DATA_BUFFER_SPACE_BASE
    shared_addr: dict[int, int] = {}  # cb_id -> addr for shared CBs
    if program.cb_config:
      for cb in program.cbs:
        num_pages, page_size = program.cb_config[cb]
        # Check if another CB already allocated at same (num_pages, page_size) with share intent
        # CB_16 and CB_24 share address when both present
        share_with = {16: 24, 24: 16}.get(cb)
        if share_with is not None and share_with in shared_addr:
          cb_addr = shared_addr[share_with]
        else:
          cb_addr = addr
          size = page_size * num_pages
          addr += size
        shared_addr[cb] = cb_addr
        arr[cb] = LocalCBConfig(
          addr_bytes=cb_addr, size_bytes=page_size * num_pages,
          num_pages=num_pages, page_size_bytes=page_size,
        )
    else:
      for cb in program.cbs:
        size = program.tile_size * program.num_pages
        arr[cb] = LocalCBConfig(
          addr_bytes=addr, size_bytes=size,
          num_pages=program.num_pages, page_size_bytes=program.tile_size,
        )
        addr += size
    return mask, as_bytes(arr)

  def _pack_kernel_shared(self, program: Program, rta_sizes: tuple[int, int, int], dispatch_mode: int = DevMsgs.DISPATCH_MODE_HOST):
    align16 = lambda n: (n + 15) & ~15

    rta_offsets = [0, rta_sizes[0], rta_sizes[0] + rta_sizes[1]]
    rta_total = align16(rta_offsets[2] + rta_sizes[2])

    # Semaphore space: num_sems * 16 bytes between per-core args and shared data
    sem_size = program.num_sems * 16
    sem_off = rta_total
    shared_off_start = align16(rta_total + sem_size)
    crta_off = shared_off_start

    local_cb_mask, local_cb_blob = self._build_local_cb_blob(program)
    local_cb_off = shared_off_start
    kernel_off = align16(local_cb_off + len(local_cb_blob))

    kernels = {"brisc": program.writer, "ncrisc": program.reader}
    if program.compute:
      for i, k in enumerate(program.compute):
        kernels[f"trisc{i}"] = k
    proc = [("brisc", 0), ("ncrisc", 1), ("trisc0", 2), ("trisc1", 3), ("trisc2", 4)]
    kernel_text_off = [0] * len(proc)
    enables = 0
    off = kernel_off
    for name, idx in proc:
      if (k := kernels.get(name)) is None: continue
      kernel_text_off[idx] = off
      off = align16(off + len(k.xip))
      enables |= 1 << idx

    # Shared image: from local_cb_off to end (CB config + kernel binaries)
    shared = bytearray(off - local_cb_off)
    shared[0 : len(local_cb_blob)] = local_cb_blob
    for name, idx in proc:
      if (k := kernels.get(name)) is None: continue
      dst = kernel_text_off[idx] - local_cb_off
      shared[dst : dst + len(k.xip)] = k.xip

    cfg = LaunchMsg().kernel_config
    cfg.kernel_config_base[0] = TensixL1.KERNEL_CONFIG_BASE
    cfg.kernel_config_base[1] = TensixL1.KERNEL_CONFIG_BASE
    cfg.kernel_config_base[2] = TensixL1.KERNEL_CONFIG_BASE
    for i in range(3):
      cfg.sem_offset[i] = sem_off
    cfg.local_cb_offset = local_cb_off
    cfg.remote_cb_offset = local_cb_off + len(local_cb_blob)
    cfg.local_cb_mask = local_cb_mask
    cfg.min_remote_cb_start_index = CB.NUM_CIRCULAR_BUFFERS
    cfg.enables = enables
    cfg.brisc_noc_id = 1
    cfg.brisc_noc_mode = 0
    cfg.mode = dispatch_mode
    cfg.rta_offset[0].rta_offset, cfg.rta_offset[0].crta_offset = rta_offsets[0], crta_off
    cfg.rta_offset[1].rta_offset, cfg.rta_offset[1].crta_offset = rta_offsets[1], crta_off
    for i in (2, 3, 4):
      cfg.rta_offset[i].rta_offset, cfg.rta_offset[i].crta_offset = rta_offsets[2], crta_off
    for i, v in enumerate(kernel_text_off):
      cfg.kernel_text_offset[i] = v

    launch = LaunchMsg()
    launch.kernel_config = cfg
    return local_cb_off, bytes(shared), as_bytes(launch), rta_offsets

  @staticmethod
  def _pack_rta(reader_args: list[int], writer_args: list[int], compute_args: list[int],
                num_sems: int = 0) -> bytes:
    pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
    rta = pack(writer_args) + pack(reader_args) + pack(compute_args)
    if num_sems > 0:
      rta += b"\0" * (num_sems * 16)
    return rta

  def _resolve_args(self, args: list[int] | ArgGen, core_idx: int, core_xy: tuple[int, int], num_cores: int) -> list[int]:
    return args if isinstance(args, list) else args(core_idx, core_xy, num_cores)

  @staticmethod
  def _mcast_rects(cores: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    west = [(x, y) for x, y in cores if x < 8]
    east = [(x, y) for x, y in cores if x >= 10]
    rects = []
    for group in (west, east):
      if group:
        xs = [x for x, _ in group]
        ys = [y for _, y in group]
        rects.append((min(xs), max(xs), min(ys), max(ys)))
    return rects

  @staticmethod
  def _mcast_write_rects(
    win: TLBWindow,
    cfg: TLBConfig,
    rects: list[tuple[int, int, int, int]],
    writes: list[tuple[int, bytes]],
  ):
    for x0, x1, y0, y1 in rects:
      cfg.start, cfg.end = (x0, y0), (x1, y1)
      win.configure(cfg)
      for addr, data in writes:
        win.write(addr, data, use_uc=True, restore=False)

  def run(self, program: Program, *, timing: bool = False) -> tuple[float, float]:
    cores = program.cores if program.cores is not None else self.dispatchable_cores
    num_cores = len(cores)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go_blob = as_bytes(go)

    # Compute rta sizes from core 0 to determine shared layout
    first_r = self._resolve_args(program.reader_rt_args, 0, cores[0], num_cores)
    first_w = self._resolve_args(program.writer_rt_args, 0, cores[0], num_cores)
    first_c = self._resolve_args(program.compute_rt_args, 0, cores[0], num_cores)
    rta_sizes = (len(first_w) * 4, len(first_r) * 4, len(first_c) * 4)
    shared_off, shared_img, launch_blob, _ = self._pack_kernel_shared(program, rta_sizes)

    mcast_cfg = TLBConfig(addr=0, noc=0, mcast=True, mode=TLBMode.STRICT)
    l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    rects = self._mcast_rects(cores)
    win = self.win
    # Mcast shared data: reset GO_MSG, GO_MSG_INDEX, CB config + kernel binaries, LAUNCH
    self._mcast_write_rects(
      win,
      mcast_cfg,
      rects,
      [
        (TensixL1.GO_MSG, reset_blob),
        (TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little")),
        (TensixL1.KERNEL_CONFIG_BASE + shared_off, shared_img),
        (TensixL1.LAUNCH, launch_blob),
      ],
    )

    ns = program.num_sems
    shared_rta = None
    if (
      isinstance(program.reader_rt_args, list)
      and isinstance(program.writer_rt_args, list)
      and isinstance(program.compute_rt_args, list)
    ):
      shared_rta = self._pack_rta(program.reader_rt_args, program.writer_rt_args, program.compute_rt_args, ns)

    # Unicast per-core runtime args
    for core_idx, (x, y) in enumerate(cores):
      if shared_rta is None:
        reader_args = self._resolve_args(program.reader_rt_args, core_idx, (x, y), num_cores)
        writer_args = self._resolve_args(program.writer_rt_args, core_idx, (x, y), num_cores)
        compute_args = self._resolve_args(program.compute_rt_args, core_idx, (x, y), num_cores)
        rta = self._pack_rta(reader_args, writer_args, compute_args, ns)
      else:
        rta = shared_rta
      l1_cfg.start = l1_cfg.end = (x, y)
      win.configure(l1_cfg)
      win.write(TensixL1.KERNEL_CONFIG_BASE, rta, use_uc=True, restore=False)

    # Dispatch: multicast GO_MSG to all cores
    t_dispatch_start = time.perf_counter()
    self._mcast_write_rects(win, mcast_cfg, rects, [(TensixL1.GO_MSG, go_blob)])
    t_compute_start = time.perf_counter()

    # Wait for all cores to complete (unicast poll)
    l1_cfg.mcast = False
    for x, y in cores:
      l1_cfg.start = l1_cfg.end = (x, y)
      win.configure(l1_cfg)
      deadline = time.perf_counter() + 10.0
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"timeout waiting for core ({x}, {y})")

    t_end = time.perf_counter()
    dispatch = t_compute_start - t_dispatch_start
    compute = t_end - t_compute_start
    if timing:
      print(f"compute: {compute * 1e3:.3f} ms")
    return dispatch + compute, dispatch

class FastDevice(SlowDevice):
  DISPATCH_STREAM_INDEX = 48
  DISPATCH_MSG_OFFSET = 0

  def _core_exists(self, core: tuple[int, int]) -> bool:
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)
    cfg = TLBConfig(addr=reg_base, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      return win.read32(reg_off) != 0xFFFF_FFFF

  def _select_dispatch_core_pair(self) -> tuple[tuple[int, int], tuple[int, int]]:
    # Prefer dedicated dispatch column so dispatch stays off compute columns.
    # Choose the topmost vertical pair from the rightmost valid column.
    ys = sorted({y for _, y in self.worker_cores})
    xs = sorted({x for x, _ in self.worker_cores}, reverse=True)
    if len(ys) < 2 or not xs:
      raise RuntimeError("not enough worker cores to choose dispatch pair")
    for x in xs:
      col_ys = [y for y in ys if self._core_exists((x, y))]
      for i in range(0, len(col_ys) - 1, 2):
        return (x, col_ys[i]), (x, col_ys[i + 1])
    raise RuntimeError("could not find a valid dispatch pair on a single column")

  def _firmware_skip_cores(self) -> set[tuple[int, int]]:
    # Don't skip dispatch cores — they need base firmware so CQ kernels can launch on top
    self._dispatch_core_pair = self._select_dispatch_core_pair()
    return set()

  def __init__(
    self,
    device: int = 0,
    *,
    sysmem_size: int = 128 * 1024 * 1024,
    issue_size: int = 64 * 1024 * 1024,
    completion_size: int = 32 * 1024 * 1024,
  ):
    super().__init__(device=device)
    self.prefetch_core, self.dispatch_core = getattr(self, "_dispatch_core_pair", self._select_dispatch_core_pair())
    self.dispatchable_cores = [c for c in self.worker_cores if c not in {self.prefetch_core, self.dispatch_core}]
    self._cq = _FastCQ(
      self.fd,
      prefetch_core=self.prefetch_core,
      dispatch_core=self.dispatch_core,
      sysmem_size=sysmem_size,
      issue_size=issue_size,
      completion_size=completion_size,
      prefetch_q_entries=FastDispatch.PREFETCH_Q_ENTRIES_WORKER_DEFAULT,
    )
    self._event_id = 1
    self._start_dispatch_cores()
    time.sleep(0.3)  # give dispatch firmware time to init

  def close(self):
    if hasattr(self, "_cq"):
      self._cq.close()
    super().close()

  def _wait_core_done(self, core: tuple[int, int], timeout_s: float = 2.0):
    """Wait for a core's firmware init to complete (GO_MSG signal = RUN_MSG_DONE)."""
    cfg = TLBConfig(addr=0, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    deadline = time.perf_counter() + timeout_s
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          go = win.uc[TensixL1.GO_MSG + 3]
          raise TimeoutError(f"core {core} firmware init timeout (GO_MSG signal=0x{go:02x})")
        time.sleep(0.001)

  def _build_cq_config(self) -> CQConfig:
    """Build CQ compile config from the current device state and CQ layout."""
    dev = self._cq.dev
    host = self._cq.host
    px, py = self.prefetch_core
    dx, dy = self.dispatch_core
    workers = self.dispatchable_cores
    xs = [x for x, _ in workers]
    ys = [y for _, y in workers]
    dispatch_cb_base = (dev.l1_base + FastDispatch.BH_UNRESERVED_OFF + 4095) & ~4095
    dispatch_s_base = dispatch_cb_base + (dev.dispatch_cb_pages << 12)
    return CQConfig(
      prefetch_xy=(px, py), dispatch_xy=(dx, dy),
      pcie_base=self._cq.noc_local + host.issue_base,
      pcie_size=host.issue_size,
      completion_base=self._cq.noc_local + host.completion_base,
      completion_size=host.completion_size,
      prefetch_q_base=dev.prefetch_q_base, prefetch_q_size=dev.prefetch_q_size,
      prefetch_q_rd_ptr_addr=dev.prefetch_q_rd_ptr_addr,
      prefetch_q_pcie_rd_ptr_addr=dev.prefetch_q_pcie_rd_ptr_addr,
      cmddat_q_base=dev.cmddat_q_base, cmddat_q_size=FastDispatch.PREFETCH_CMDDAT_Q_SIZE,
      scratch_db_base=dev.scratch_db_base, scratch_db_size=FastDispatch.PREFETCH_SCRATCH_DB_SIZE,
      dispatch_cb_base=dispatch_cb_base, dispatch_cb_pages=dev.dispatch_cb_pages,
      dispatch_s_buffer_base=dispatch_s_base, dispatch_s_buffer_size=32 * 1024,
      completion_q_wr_ptr_addr=dev.completion_q_wr_ptr_addr,
      completion_q_rd_ptr_addr=dev.completion_q_rd_ptr_addr,
      dispatch_s_sync_sem_addr=dev.dispatch_s_sync_sem_addr,
      command_queue_base=self._cq.noc_local,
      worker_grid_start=(min(xs), min(ys)), worker_grid_end=(max(xs), max(ys)),
      num_worker_cores=len(workers),
    )

  @staticmethod
  def _build_cq_launch(*, rt_args: list[int], sem_values: list[int],
                        kernel_text_off: int, ncrisc_text_off: int = 0) -> tuple[bytes, LaunchMsg]:
    """Build kernel config image (rt_args + semaphores) and LaunchMsg for a CQ kernel."""
    l1a = FastDispatch.L1_ALIGNMENT
    rt_blob = b"".join((a & 0xFFFFFFFF).to_bytes(4, "little") for a in rt_args).ljust(l1a, b"\0")
    sem_blob = b"".join((v & 0xFFFFFFFF).to_bytes(4, "little").ljust(l1a, b"\0") for v in sem_values)
    img = rt_blob + sem_blob

    kc = KernelConfigMsg()
    kc.kernel_config_base[0] = TensixL1.KERNEL_CONFIG_BASE
    kc.kernel_config_base[1] = TensixL1.KERNEL_CONFIG_BASE
    kc.kernel_config_base[2] = TensixL1.KERNEL_CONFIG_BASE
    kc.sem_offset[0] = l1a
    kc.rta_offset[0].rta_offset = 0
    kc.rta_offset[0].crta_offset = len(rt_blob)
    kc.kernel_text_offset[0] = kernel_text_off  # BRISC
    kc.kernel_text_offset[1] = ncrisc_text_off   # NCRISC
    kc.enables = 1 | (2 if ncrisc_text_off else 0)  # bit 0 = BRISC, bit 1 = NCRISC
    kc.mode = DevMsgs.DISPATCH_MODE_HOST
    kc.local_cb_mask = 0
    kc.min_remote_cb_start_index = CB.NUM_CIRCULAR_BUFFERS
    launch = LaunchMsg()
    launch.kernel_config = kc
    return img, launch

  def _start_dispatch_cores(self):
    # Wait for firmware init to complete on dispatch cores before launching CQ kernels
    self._wait_core_done(self.prefetch_core)
    self._wait_core_done(self.dispatch_core)

    cq_cfg = self._build_cq_config()
    cq = compile_cq_kernels(cq_cfg)

    # Kernel config layout: [rt_args(16B)] [sem0(16B)] [sem1(16B)] ... then kernel XIP blobs
    # Offsets are relative to KERNEL_CONFIG_BASE
    l1a = FastDispatch.L1_ALIGNMENT
    # rt_args = [my_dev_id=0, to_dev_id=0, router_direction=0] -> 12 bytes, padded to 16
    n_rt = l1a  # 16 bytes for 3 rt args
    # Kernel text starts after rt_args + semaphores

    # -- Prefetch core: BRISC only, 2 semaphores --
    pref_sem = [self._cq.dev.dispatch_cb_pages, 0]  # sem0=downstream credits, sem1=dispatch_s credits
    pref_kernel_off = n_rt + len(pref_sem) * l1a
    pref_img, pref_launch = self._build_cq_launch(
      rt_args=[0, 0, 0], sem_values=pref_sem, kernel_text_off=pref_kernel_off)

    # -- Dispatch core: BRISC + NCRISC, 2 semaphores --
    disp_sem = [0, 0]  # sem0=upstream CB (starts at 0), sem1=dispatch_s upstream
    disp_brisc_off = n_rt + len(disp_sem) * l1a
    disp_ncrisc_off = _align_up(disp_brisc_off + len(cq.dispatch_brisc.xip), l1a)
    disp_img, disp_launch = self._build_cq_launch(
      rt_args=[0, 0, 0], sem_values=disp_sem,
      kernel_text_off=disp_brisc_off, ncrisc_text_off=disp_ncrisc_off)

    # Upload to prefetch core
    cfg = TLBConfig(addr=0, start=self.prefetch_core, end=self.prefetch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      self._cq.init_prefetch_l1()
      win.write(TensixL1.KERNEL_CONFIG_BASE, pref_img, use_uc=True, restore=False)
      win.write(TensixL1.KERNEL_CONFIG_BASE + pref_kernel_off, cq.prefetch_brisc.xip, use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(pref_launch), use_uc=True, restore=False)
      go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO
      win.write(TensixL1.GO_MSG, as_bytes(go), use_uc=True, restore=False)

    # Upload to dispatch core
    cfg.start = cfg.end = self.dispatch_core
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      # Init completion queue pointers
      base_16b = ((self._cq.noc_local + self._cq.host.completion_base) >> 4) & 0x7FFF_FFFF
      win.write32(self._cq.dev.completion_q_wr_ptr_addr, base_16b)
      win.write32(self._cq.dev.completion_q_rd_ptr_addr, base_16b)
      win.write32(self._cq.dev.completion_q0_last_event_ptr_addr, 0)
      win.write32(self._cq.dev.completion_q1_last_event_ptr_addr, 0)
      win.uc[self._cq.dev.dispatch_s_sync_sem_addr : self._cq.dev.dispatch_s_sync_sem_addr + 8 * l1a] = b"\0" * (8 * l1a)
      # Write kernel config + kernel binaries
      win.write(TensixL1.KERNEL_CONFIG_BASE, disp_img, use_uc=True, restore=False)
      win.write(TensixL1.KERNEL_CONFIG_BASE + disp_brisc_off, cq.dispatch_brisc.xip, use_uc=True, restore=False)
      win.write(TensixL1.KERNEL_CONFIG_BASE + disp_ncrisc_off, cq.dispatch_s_ncrisc.xip, use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(disp_launch), use_uc=True, restore=False)
      go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO
      win.write(TensixL1.GO_MSG, as_bytes(go), use_uc=True, restore=False)

    time.sleep(0.3)

  def _mcast_dests(self, cores: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Compute NOC mcast destinations for packed_large, avoiding dispatch cores.
    Returns list of (noc_xy_mcast, num_dests) tuples. NOC 1 encoding (start=max, end=min)."""
    dispatch_col = self.prefetch_core[0]
    dispatch_set = {self.prefetch_core, self.dispatch_core}
    west = [(x, y) for x, y in cores if x < 8]
    east_safe = [(x, y) for x, y in cores if x >= 10 and x != dispatch_col]
    col_cores = sorted([(x, y) for x, y in cores if x == dispatch_col and (x, y) not in dispatch_set])
    dests = []
    for group in (west, east_safe, col_cores):
      if group:
        xs = [x for x, _ in group]
        ys = [y for _, y in group]
        noc_xy = (max(ys) << 18) | (max(xs) << 12) | (min(ys) << 6) | min(xs)
        dests.append((noc_xy, len(group)))
    return dests

  def run(self, program: Program, *, timing: bool = False) -> tuple[float, float]:
    cores = program.cores if program.cores is not None else self.dispatchable_cores
    num_cores = len(cores)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    go = GoMsg()
    # Always set dispatch core coordinates when using DEV mode so workers
    # can compute the correct dispatch_addr for notify_dispatch_core_done.
    go.bits.dispatch_message_offset = FastDevice.DISPATCH_MSG_OFFSET
    go.bits.master_x = self.dispatch_core[0]
    go.bits.master_y = self.dispatch_core[1]
    go.bits.signal = DevMsgs.RUN_MSG_GO

    first_r = self._resolve_args(program.reader_rt_args, 0, cores[0], num_cores)
    first_w = self._resolve_args(program.writer_rt_args, 0, cores[0], num_cores)
    first_c = self._resolve_args(program.compute_rt_args, 0, cores[0], num_cores)
    rta_sizes = (len(first_w) * 4, len(first_r) * 4, len(first_c) * 4)
    dispatch_mode = DevMsgs.DISPATCH_MODE_DEV  # always DEV for testing
    shared_off, shared_img, launch_blob, _ = self._pack_kernel_shared(program, rta_sizes, dispatch_mode=dispatch_mode)

    go_msg_index_zero = (0).to_bytes(4, "little")

    # Build per-core RTA payloads
    ns = program.num_sems
    shared_rta = None
    if (
      isinstance(program.reader_rt_args, list)
      and isinstance(program.writer_rt_args, list)
      and isinstance(program.compute_rt_args, list)
    ):
      shared_rta = self._pack_rta(program.reader_rt_args, program.writer_rt_args, program.compute_rt_args, ns)

    if shared_rta is not None:
      rta_payloads = shared_rta  # uniform: single bytes, NO_STRIDE
    else:
      rta_payloads = [
        self._pack_rta(
          self._resolve_args(program.reader_rt_args, i, cores[i], num_cores),
          self._resolve_args(program.writer_rt_args, i, cores[i], num_cores),
          self._resolve_args(program.compute_rt_args, i, cores[i], num_cores),
          ns,
        ) for i in range(num_cores)
      ]

    # Dispatch via CQ packed writes (RTA + launch + GO as packed, shared_img as packed_large)
    mcast_dests = self._mcast_dests(cores)
    self._cq.enqueue_write_packed_large(dests=mcast_dests, addr=TensixL1.GO_MSG, data=reset_blob)
    self._cq.enqueue_write_packed_large(dests=mcast_dests, addr=TensixL1.GO_MSG_INDEX, data=go_msg_index_zero)
    self._cq.enqueue_write_packed(cores=cores, addr=TensixL1.KERNEL_CONFIG_BASE, data=rta_payloads)
    self._cq.enqueue_write_packed_large(dests=mcast_dests, addr=TensixL1.KERNEL_CONFIG_BASE + shared_off, data=shared_img)
    self._cq.enqueue_write_packed(cores=cores, addr=TensixL1.LAUNCH, data=launch_blob)
    t_dispatch_start = time.perf_counter()
    # CQ wait path: dispatch sends GO + waits for worker completion + notifies host
    noc_words = [(y << 6) | x for x, y in cores]
    self._cq.enqueue_wait_stream(stream=48, count=0, clear_stream=True)  # clear stale count
    self._cq.enqueue_set_go_signal_noc_data(noc_words=noc_words)
    self._cq.enqueue_send_go_signal(
      go_signal=go.all, wait_stream=48, wait_count=0,
      num_unicast_txns=num_cores, noc_data_start_index=0)
    self._cq.enqueue_wait_stream(stream=48, count=num_cores, clear_stream=True)
    event_id = self._event_id; self._event_id += 1
    self._cq.enqueue_host_event(event_id=event_id)
    t_compute_start = time.perf_counter()

    # Poll completion queue event marker emitted after dispatch-side WAIT(stream==num_cores)
    try:
      self._cq.wait_host_event(event_id=event_id, timeout_s=10.0)
    except TimeoutError:
      win = self.win
      # Debug: check worker GO_MSG state and launch mode
      l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
      sample = cores[:3] + cores[-1:]
      for x, y in sample:
        l1_cfg.start = l1_cfg.end = (x, y)
        win.configure(l1_cfg)
        go_val = win.read32(TensixL1.GO_MSG)
        # Read mode byte from launch msg kernel config (offset 42 in LaunchMsg)
        mode_val = win.uc[TensixL1.LAUNCH + 42]
        # Read debug area at 0x100
        dbg = [win.read32(0x100 + i*4) for i in range(8)]
        print(f"  core ({x},{y}): GO_MSG=0x{go_val:08x} signal=0x{(go_val>>24)&0xff:02x} mode={mode_val}")
        if dbg[0] == 0xDEAD0001:
          disp_lo, disp_hi, nidx = dbg[1], dbg[2], dbg[3]
          disp_addr = (disp_hi << 32) | disp_lo
          done = "yes" if dbg[5] in (0xDEAD0002, 0xDEAD0003) else f"no (0x{dbg[5]:08x})"
          hw_posted = dbg[6]
          sw_posted = dbg[7]
          print(f"    notify_done={done} dispatch_addr=0x{disp_addr:016x} noc_idx={nidx}")
          print(f"    NOC_HW_POSTED_WR_SENT={hw_posted} SW_POSTED_WR_ISSUED={sw_posted}")
        else:
          print(f"    DEBUG: no DEV epilog marker (dbg[0]=0x{dbg[0]:08x})")
      # Read stream 48 registers on dispatch core
      dx, dy = self.dispatch_core
      s48_base = 0xFFB70000  # NOC_OVERLAY_START + 48 * 0x1000
      s48_avail_off = 297 * 4  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE
      s48_update_off = 270 * 4  # STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE
      tlb_base = s48_base & ~((1 << 21) - 1)  # 2MB-aligned
      dcfg = TLBConfig(addr=tlb_base, start=(dx, dy), end=(dx, dy), noc=0, mcast=False, mode=TLBMode.STRICT)
      self._cq.dispatch_win.configure(dcfg)
      avail = self._cq.dispatch_win.read32(s48_base - tlb_base + s48_avail_off)
      update = self._cq.dispatch_win.read32(s48_base - tlb_base + s48_update_off)
      print(f"  dispatch ({dx},{dy}): stream48 AVAILABLE=0x{avail:08x} ({avail}) UPDATE_REG=0x{update:08x}")
      # Read L1 addr 0 on dispatch core (check if workers wrote there instead of stream reg)
      dcfg_l1 = TLBConfig(addr=0, start=(dx, dy), end=(dx, dy), noc=0, mcast=False, mode=TLBMode.STRICT)
      self._cq.dispatch_win.configure(dcfg_l1)
      l1_0 = self._cq.dispatch_win.read32(0)
      l1_4 = self._cq.dispatch_win.read32(4)
      print(f"  dispatch L1[0x0]=0x{l1_0:08x} L1[0x4]=0x{l1_4:08x}")
      # Read prefetch RD ptr to check how far CQ got
      pf_rd = self._cq.dispatch_win.read32(self._cq.dev.prefetch_q_rd_ptr_addr)
      pf_pcie_rd = self._cq.dispatch_win.read32(self._cq.dev.prefetch_q_pcie_rd_ptr_addr)
      cq_wr = self._cq.dispatch_win.read32(self._cq.dev.completion_q_wr_ptr_addr)
      print(f"  dispatch prefetch_rd=0x{pf_rd:08x} pcie_rd=0x{pf_pcie_rd:08x} cq_wr=0x{cq_wr:08x}")
      # Restore dispatch_win config
      dcfg2 = TLBConfig(addr=0, start=(dx, dy), end=(dx, dy), noc=0, mcast=False, mode=TLBMode.STRICT)
      self._cq.dispatch_win.configure(dcfg2)
      # Check prefetch queue entry consumption
      for i in range(min(8, self._cq.dev.prefetch_q_size // 2)):
        v = self._cq._read_prefetch_entry(i)
        if v != 0: print(f"  prefetch_q[{i}] = 0x{v:04x} (not consumed)")
      raise
    t_end = time.perf_counter()
    dispatch = t_compute_start - t_dispatch_start
    compute = t_end - t_compute_start
    if timing:
      print(f"compute: {compute * 1e3:.3f} ms")
    return dispatch + compute, dispatch
