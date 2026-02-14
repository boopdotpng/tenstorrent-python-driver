import ctypes, fcntl, mmap, struct, time
from typing import Callable
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import _IO, TIMING, align_up, noc_xy
from codegen import compile_cq_kernels
from device import Program
from device_slow import (
  AddrPayload,
  CorePair,
  FAST_CQ_NUM_CIRCULAR_BUFFERS,
  McastDest,
  SlowDevice,
)

class _FastCQ:
  NUM_CIRCULAR_BUFFERS = 32

  def __init__(self, fd: int, prefetch_core: Core, dispatch_core: Core):
    self.fd = fd

    self.sysmem = mmap.mmap(
      -1,
      HOST_SYSMEM_SIZE,
      flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    self.sysmem_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.sysmem))
    if (self.sysmem_addr % PAGE_SIZE) != 0 or (HOST_SYSMEM_SIZE % PAGE_SIZE) != 0:
      raise RuntimeError("sysmem must be page-aligned and page-sized")

    buf = bytearray(ctypes.sizeof(PinPagesIn) + ctypes.sizeof(PinPagesOutExtended))
    pin = PinPagesIn.from_buffer(buf)
    pin.output_size_bytes = ctypes.sizeof(PinPagesOutExtended)
    pin.flags = PIN_PAGES_NOC_DMA
    pin.virtual_address = self.sysmem_addr
    pin.size = HOST_SYSMEM_SIZE
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
    self._completion_base_16b = (self.noc_local + HOST_COMPLETION_BASE) >> 4
    self._completion_end_16b = (self.noc_local + HOST_COMPLETION_BASE + HOST_COMPLETION_SIZE) >> 4
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    self._init_host_completion_ctrl()
    self._write_completion_rd_ptr()
    self._recording = False
    self._recorded: list[bytes] = []

  def close(self):
    try:
      if hasattr(self, "prefetch_win"): self.prefetch_win.free()
      if hasattr(self, "dispatch_win"): self.dispatch_win.free()
    finally:
      if hasattr(self, "sysmem"):
        unpin = UnpinPagesIn(virtual_address=self.sysmem_addr, size=HOST_SYSMEM_SIZE, reserved=0)
        fcntl.ioctl(self.fd, _IO(IOCTL_UNPIN_PAGES), bytearray(as_bytes(unpin)), False)
        self.sysmem.close()

  def _read_prefetch_entry(self, idx: int) -> int:
    off = DEV_PREFETCH_Q_BASE + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
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
    self._wait_prefetch_slot_free(idx)
    off = DEV_PREFETCH_Q_BASE + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self.prefetch_win.uc[off : off + 2] = struct.pack("<H", size_16b)
    entries = DEV_PREFETCH_Q_SIZE // FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self.prefetch_q_wr_idx = (idx + 1) % entries

  def _issue_write(self, record: bytes):
    if len(record) % FastDispatch.PCIE_ALIGNMENT != 0:
      raise ValueError("record must be 64B-aligned")
    if self._recording:
      self._recorded.append(record)
      return
    wr = align_up(self.issue_wr, FastDispatch.PCIE_ALIGNMENT)
    if wr + len(record) > HOST_ISSUE_SIZE: wr = 0
    base = HOST_ISSUE_BASE + wr
    self.sysmem[base : base + len(record)] = record
    self.issue_wr = wr + len(record)
    self._write_prefetch_q_entry(len(record) >> 4)

  def start_recording(self):
    self._recording = True
    self._recorded = []

  def stop_recording(self) -> tuple[bytearray, list[int], int]:
    self._recording = False
    records = self._recorded
    self._recorded = []
    stream = bytearray(b"".join(records))
    sizes = [len(r) >> 4 for r in records]
    last_start = len(stream) - len(records[-1])
    event_offset = last_start + ctypes.sizeof(CQPrefetchCmd) + ctypes.sizeof(CQDispatchCmd)
    return stream, sizes, event_offset

  def submit(self, stream: bytearray, entry_sizes: list[int]):
    off = 0
    for size_16b in entry_sizes:
      size = size_16b << 4
      self._issue_write(stream[off : off + size])
      off += size
    if off != len(stream):
      raise RuntimeError(f"CQ submit size mismatch: consumed {off} bytes from {len(stream)}-byte stream")

  def _relay_inline(self, inner: bytes):
    prefetch = CQPrefetchCmd()
    prefetch.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE
    prefetch.payload.relay_inline.length = len(inner)
    stride = align_up(ctypes.sizeof(CQPrefetchCmd) + len(inner), FastDispatch.PCIE_ALIGNMENT)
    prefetch.payload.relay_inline.stride = stride
    pad = b"\0" * (stride - ctypes.sizeof(CQPrefetchCmd) - len(inner))
    self._issue_write(as_bytes(prefetch) + inner + pad)

  def enqueue_write_packed(self, cores: CoreList, addr: int, data: bytes | list[bytes]):
    count = len(cores)
    uniform = isinstance(data, bytes)
    payload_size = len(data) if uniform else len(data[0])

    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED
    dispatch.payload.write_packed.flags = CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE if uniform else 0
    dispatch.payload.write_packed.count = count
    dispatch.payload.write_packed.write_offset_index = 0
    dispatch.payload.write_packed.size = payload_size
    dispatch.payload.write_packed.addr = addr

    sub_cmds = b"".join(
      as_bytes(CQDispatchWritePackedUnicastSubCmd(noc_xy_addr=noc_xy(x, y)))
      for x, y in cores
    )
    sub_cmds_padded = sub_cmds.ljust(align_up(len(sub_cmds), FastDispatch.L1_ALIGNMENT), b"\0")

    if uniform:
      data_section = bytes(data).ljust(align_up(payload_size, FastDispatch.L1_ALIGNMENT), b"\0")
    else:
      stride = align_up(payload_size, FastDispatch.L1_ALIGNMENT)
      data_section = b"".join(bytes(d).ljust(stride, b"\0") for d in data)

    self._relay_inline(as_bytes(dispatch) + sub_cmds_padded + data_section)

  def enqueue_write_packed_large(self, dests: list[McastDest], addr: int, data: bytes):
    alignment = FastDispatch.L1_ALIGNMENT
    data_padded = bytes(data).ljust(align_up(len(data), alignment), b"\0")
    max_batch = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS

    for batch_start in range(0, len(dests), max_batch):
      batch = dests[batch_start : batch_start + max_batch]
      count = len(batch)

      dispatch = CQDispatchCmd()
      dispatch.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED_LARGE
      dispatch.payload.write_packed_large.type = 2  # PROGRAM_BINARIES
      dispatch.payload.write_packed_large.count = count
      dispatch.payload.write_packed_large.alignment = alignment
      dispatch.payload.write_packed_large.write_offset_index = 0

      sub_cmds = b"".join(
        as_bytes(CQDispatchWritePackedLargeSubCmd(
          noc_xy_addr=mcast_xy, addr=addr, length_minus1=len(data) - 1,
          num_mcast_dests=n_dests,
          flags=CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK,
        )) for mcast_xy, n_dests in batch
      )
      sub_cmds_padded = sub_cmds.ljust(align_up(len(sub_cmds), alignment), b"\0")
      self._relay_inline(as_bytes(dispatch) + sub_cmds_padded + data_padded * count)

  def enqueue_write_linear(self, tile: Core, addr: int, data: bytes):
    x, y = tile
    dispatch = CQDispatchCmdLarge()
    dispatch.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR
    dispatch.payload.write_linear.num_mcast_dests = 0
    dispatch.payload.write_linear.write_offset_index = 0
    dispatch.payload.write_linear.pad1 = 0
    dispatch.payload.write_linear.noc_xy_addr = noc_xy(x, y)
    dispatch.payload.write_linear.addr = addr
    dispatch.payload.write_linear.length = len(data)
    self._relay_inline(as_bytes(dispatch) + bytes(data))

  def enqueue_set_go_signal_noc_data(self, noc_words: list[int]):
    payload_words = b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in noc_words)
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQ_DISPATCH_CMD_SET_GO_SIGNAL_NOC_DATA
    dispatch.payload.set_go_signal_noc_data.num_words = len(noc_words)
    self._relay_inline(as_bytes(dispatch) + payload_words)

  def enqueue_send_go_signal(self, go_signal: int, wait_stream: int, wait_count: int, num_unicast_txns: int, noc_data_start_index: int = 0,
                             multicast_go_offset: int = CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET):
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQ_DISPATCH_CMD_SEND_GO_SIGNAL
    dispatch.payload.mcast.go_signal = go_signal & 0xFFFFFFFF
    dispatch.payload.mcast.multicast_go_offset = multicast_go_offset & 0xFF
    dispatch.payload.mcast.num_unicast_txns = num_unicast_txns & 0xFF
    dispatch.payload.mcast.noc_data_start_index = noc_data_start_index & 0xFF
    dispatch.payload.mcast.wait_count = wait_count & 0xFFFFFFFF
    dispatch.payload.mcast.wait_stream = wait_stream & 0xFFFFFFFF
    self._relay_inline(as_bytes(dispatch))

  def enqueue_wait_stream(self, stream: int, count: int, clear_stream: bool = True):
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQ_DISPATCH_CMD_WAIT
    flags = CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM
    if clear_stream:
      flags |= CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM
    dispatch.payload.wait.flags = flags
    dispatch.payload.wait.stream = stream
    dispatch.payload.wait.addr = 0
    dispatch.payload.wait.count = count
    self._relay_inline(as_bytes(dispatch))

  def enqueue_host_event(self, event_id: int, pad1: int = 0):
    payload_data = struct.pack("<I", event_id & 0xFFFFFFFF).ljust(FastDispatch.L1_ALIGNMENT, b"\0")
    dispatch = CQDispatchCmd()
    dispatch.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST
    dispatch.payload.write_linear_host.is_event = 1
    dispatch.payload.write_linear_host.pad1 = pad1 & 0xFFFF
    dispatch.payload.write_linear_host.pad2 = 0
    dispatch.payload.write_linear_host.length = ctypes.sizeof(CQDispatchCmd) + len(payload_data)
    self._relay_inline(as_bytes(dispatch) + payload_data)

  def _read_completion_wr_ptr(self) -> tuple[int, int]:
    raw = struct.unpack("<I", self.sysmem[FastDispatch.HOST_COMPLETION_Q_WR_OFF : FastDispatch.HOST_COMPLETION_Q_WR_OFF + 4])[0]
    return raw & 0x7FFF_FFFF, (raw >> 31) & 0x1

  def _read_completion_event_id(self) -> int:
    off = (self._completion_rd_16b << 4) - self.noc_local
    return struct.unpack("<I", self.sysmem[off + ctypes.sizeof(CQDispatchCmd) : off + ctypes.sizeof(CQDispatchCmd) + 4])[0]

  def _write_completion_rd_ptr(self):
    raw = (self._completion_rd_16b & 0x7FFF_FFFF) | (self._completion_rd_toggle << 31)
    self.dispatch_win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, raw)
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

  def wait_host_event(self, event_id: int, timeout_s: float):
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

  def reset_run_state(self):
    self.issue_wr = 0
    self.prefetch_q_wr_idx = 0
    end_ptr = DEV_PREFETCH_Q_BASE + DEV_PREFETCH_Q_SIZE
    self.prefetch_win.uc[DEV_PREFETCH_Q_RD_PTR_ADDR : DEV_PREFETCH_Q_RD_PTR_ADDR + 4] = struct.pack("<I", end_ptr)
    pcie_base = self.noc_local + HOST_ISSUE_BASE
    self.prefetch_win.uc[DEV_PREFETCH_Q_PCIE_RD_PTR_ADDR : DEV_PREFETCH_Q_PCIE_RD_PTR_ADDR + 4] = struct.pack("<I", pcie_base)
    self.prefetch_win.uc[DEV_PREFETCH_Q_BASE : DEV_PREFETCH_Q_BASE + DEV_PREFETCH_Q_SIZE] = b"\0" * DEV_PREFETCH_Q_SIZE
    self._completion_rd_16b = self._completion_base_16b
    self._completion_rd_toggle = 0
    self._init_host_completion_ctrl()
    self._write_completion_rd_ptr()

class FastDevice(SlowDevice):
  DISPATCH_STREAM_INDEX = 48
  DISPATCH_MSG_OFFSET = 0

  def _select_dispatch_core_pair(self) -> CorePair:
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

  def _firmware_skip_cores(self) -> set[Core]:
    # Don't skip dispatch cores — they need base firmware so CQ kernels can launch on top
    self._dispatch_core_pair = self._select_dispatch_core_pair()
    return set()

  def __init__(self, device: int = 0):
    super().__init__(device=device, enable_sysmem=True, init_core_plans=False)
    self.prefetch_core, self.dispatch_core = getattr(self, "_dispatch_core_pair", self._select_dispatch_core_pair())
    self.dispatchable_cores = [c for c in self.worker_cores if c not in {self.prefetch_core, self.dispatch_core}]
    self._core_plans = self._build_core_plans()
    self._cq = _FastCQ(
      self.fd,
      prefetch_core=self.prefetch_core,
      dispatch_core=self.dispatch_core,
    )
    self._event_id = 1
    self._start_dispatch_cores()

  def close(self):
    if hasattr(self, "_cq"):
      self._cq.close()
    super().close()

  def _wait_core_done(self, core: Core, timeout_s: float = 2.0):
    cfg = TLBConfig(addr=0, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    deadline = time.perf_counter() + timeout_s
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          go = win.uc[TensixL1.GO_MSG + 3]
          raise TimeoutError(f"core {core} firmware init timeout (GO_MSG signal=0x{go:02x})")
        time.sleep(0.001)

  @staticmethod
  def _build_cq_launch(rt_args: list[int], sem_values: list[int], kernel_text_off: int, ncrisc_text_off: int = 0) -> tuple[bytes, LaunchMsg]:
    l1a = FastDispatch.L1_ALIGNMENT
    rt_blob = b"".join((a & 0xFFFFFFFF).to_bytes(4, "little") for a in rt_args).ljust(l1a, b"\0")
    sem_blob = b"".join((v & 0xFFFFFFFF).to_bytes(4, "little").ljust(l1a, b"\0") for v in sem_values)
    img = rt_blob + sem_blob

    kc = KernelConfigMsg()
    for i in range(3):
      kc.kernel_config_base[i] = TensixL1.KERNEL_CONFIG_BASE
    kc.sem_offset[0] = l1a
    kc.rta_offset[0].rta_offset = 0
    kc.rta_offset[0].crta_offset = len(rt_blob)
    kc.kernel_text_offset[0], kc.kernel_text_offset[1] = kernel_text_off, ncrisc_text_off
    kc.enables = 1 | (2 if ncrisc_text_off else 0)
    kc.mode = DevMsgs.DISPATCH_MODE_HOST
    kc.local_cb_mask = 0
    kc.min_remote_cb_start_index = FAST_CQ_NUM_CIRCULAR_BUFFERS
    launch = LaunchMsg()
    launch.kernel_config = kc
    return img, launch

  @staticmethod
  def _go_signal(dispatch_message_offset: int = 0, master_xy: Core | None = None) -> int:
    go = GoMsg()
    if master_xy is not None:
      go.bits.dispatch_message_offset = dispatch_message_offset
      go.bits.master_x, go.bits.master_y = master_xy
    go.bits.signal = DevMsgs.RUN_MSG_GO
    return go.all

  def _upload_cq_core(self, core: Core, kernel_cfg_image: bytes, launch: LaunchMsg, kernels: list[AddrPayload],
                      init: Callable[[TLBWindow], None] | None = None):
    cfg = TLBConfig(addr=0, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      if init is not None:
        init(win)
      win.write(TensixL1.KERNEL_CONFIG_BASE, kernel_cfg_image, use_uc=True, restore=False)
      for off, xip in kernels:
        win.write(TensixL1.KERNEL_CONFIG_BASE + off, xip, use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(launch), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG, struct.pack("<I", self._go_signal()), use_uc=True, restore=False)

  def _init_dispatch_core_state(self, win: TLBWindow):
    l1a = FastDispatch.L1_ALIGNMENT
    base_16b = ((self._cq.noc_local + HOST_COMPLETION_BASE) >> 4) & 0x7FFF_FFFF
    win.write32(DEV_COMPLETION_Q_WR_PTR_ADDR, base_16b)
    win.write32(DEV_COMPLETION_Q_RD_PTR_ADDR, base_16b)
    win.write32(DEV_COMPLETION_Q0_LAST_EVENT_PTR_ADDR, 0)
    win.write32(DEV_COMPLETION_Q1_LAST_EVENT_PTR_ADDR, 0)
    win.uc[DEV_DISPATCH_S_SYNC_SEM_ADDR : DEV_DISPATCH_S_SYNC_SEM_ADDR + 8 * l1a] = b"\0" * (8 * l1a)

  def _start_dispatch_cores(self):
    self._wait_core_done(self.prefetch_core)
    self._wait_core_done(self.dispatch_core)

    cq = compile_cq_kernels()
    l1a = FastDispatch.L1_ALIGNMENT
    kernel_off = l1a + 2 * l1a
    pref_img, pref_launch = self._build_cq_launch(
      rt_args=[0, 0, 0], sem_values=[DEV_DISPATCH_CB_PAGES, 0], kernel_text_off=kernel_off)
    disp_ncrisc_off = align_up(kernel_off + len(cq.dispatch_brisc.xip), l1a)
    disp_img, disp_launch = self._build_cq_launch(
      rt_args=[0, 0, 0], sem_values=[0, 0], kernel_text_off=kernel_off,
      ncrisc_text_off=disp_ncrisc_off)
    self._cq.reset_run_state()
    self._upload_cq_core(
      self.prefetch_core,
      kernel_cfg_image=pref_img,
      launch=pref_launch,
      kernels=[(kernel_off, cq.prefetch_brisc.xip)],
    )
    self._upload_cq_core(
      self.dispatch_core,
      kernel_cfg_image=disp_img,
      launch=disp_launch,
      kernels=[(kernel_off, cq.dispatch_brisc.xip), (disp_ncrisc_off, cq.dispatch_s_ncrisc.xip)],
      init=self._init_dispatch_core_state,
    )

  def _packed_large_dests(self, cores: CoreList) -> list[McastDest]:
    return self._mcast_dests_for_cores(cores)

  def _enqueue_write_packed_groups(self, payload_by_core: list[tuple[Core, bytes]], addr: int):
    by_size: dict[int, list[tuple[Core, bytes]]] = {}
    for core, payload in payload_by_core:
      by_size.setdefault(len(payload), []).append((core, payload))
    for entries in by_size.values():
      self._cq.enqueue_write_packed(cores=[c for c, _ in entries], addr=addr, data=[p for _, p in entries])

  def _enqueue_shared_payloads(self, shared_by_core: list[tuple[Core, int, bytes]]):
    for (addr, payload), cores in self._group_shared_payloads(shared_by_core).items():
      self._cq.enqueue_write_packed_large(dests=self._packed_large_dests(cores), addr=addr, data=payload)

  def run(self, programs: list[Program], timing: bool = False) -> tuple[float, float]:
    if not programs:
      return 0.0, 0.0

    debug_cores: CoreList | None = None
    prev_kernel_payloads: tuple[tuple[int, bytes], ...] | None = None
    prev_rta_payloads: tuple[bytes, ...] | None = None
    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    go_signal = self._go_signal(
      dispatch_message_offset=FastDevice.DISPATCH_MSG_OFFSET,
      master_xy=self.dispatch_core,
    )
    dispatch_mode = DevMsgs.DISPATCH_MODE_DEV
    self._cq.start_recording()
    for p in programs:
      plan = self._resolve_core_plan(p.cores)
      debug_cores = plan.cores
      cores = plan.cores
      num_cores = len(cores)
      launch_roles = self._core_launches(p, cores)
      payloads = self._prepare_core_payloads(p, cores, launch_roles, dispatch_mode)
      rta_by_core = [(entry.core, entry.rta) for entry in payloads]
      launch_by_core_blob = [(entry.core, entry.launch_blob) for entry in payloads]
      shared_by_core = [(entry.core, entry.shared_addr, entry.shared_blob) for entry in payloads]
      kernel_payloads = tuple((entry.shared_addr, entry.shared_blob) for entry in payloads)
      rta_payloads = tuple(entry.rta for entry in payloads)
      skip_kernels = prev_kernel_payloads == kernel_payloads
      skip_rtas = prev_rta_payloads == rta_payloads

      self._cq.enqueue_set_go_signal_noc_data(noc_words=[(y << 6) | x for x, y in cores])
      self._cq.enqueue_write_packed_large(dests=plan.mcast_dests, addr=TensixL1.GO_MSG, data=reset_blob)
      self._cq.enqueue_write_packed_large(dests=plan.mcast_dests, addr=TensixL1.GO_MSG_INDEX, data=(0).to_bytes(4, "little"))
      if not skip_rtas:
        self._enqueue_write_packed_groups(rta_by_core, TensixL1.KERNEL_CONFIG_BASE)
      self._enqueue_write_packed_groups(launch_by_core_blob, TensixL1.LAUNCH)
      if not skip_kernels:
        self._enqueue_shared_payloads(shared_by_core)
      self._cq.enqueue_wait_stream(stream=48, count=0, clear_stream=True)
      self._cq.enqueue_send_go_signal(
        go_signal=go_signal, wait_stream=48, wait_count=0,
        num_unicast_txns=num_cores, noc_data_start_index=0)
      self._cq.enqueue_wait_stream(stream=48, count=num_cores, clear_stream=True)
      prev_kernel_payloads = kernel_payloads
      prev_rta_payloads = rta_payloads

    stream, sizes, _ = self._cq.stop_recording()
    t0 = time.perf_counter()
    self._cq.submit(stream, sizes)
    dispatch = time.perf_counter() - t0
    self._wait_event(timeout_s=10.0, debug_cores=debug_cores)
    elapsed = time.perf_counter() - t0
    n = len(programs)
    if programs is self._queue:
      self._queue.clear()
    if timing or TIMING:
      print(f"  batch ({n} programs): {elapsed * 1e3:.3f} ms")
    return elapsed, dispatch

  def _wait_event(self, timeout_s: float = 10.0, debug_cores: CoreList | None = None):
    event_id = self._event_id
    self._event_id += 1
    self._cq.enqueue_host_event(event_id=event_id)
    try:
      self._cq.wait_host_event(event_id=event_id, timeout_s=timeout_s)
    except TimeoutError:
      cores = debug_cores if debug_cores is not None else self.dispatchable_cores
      win = self.win
      l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
      for x, y in (cores[:3] + cores[-1:]):
        l1_cfg.start = l1_cfg.end = (x, y)
        win.configure(l1_cfg)
        go_val = win.read32(TensixL1.GO_MSG)
        mode_val = win.uc[TensixL1.LAUNCH + 42]
        print(f"  core ({x},{y}): GO_MSG=0x{go_val:08x} signal=0x{(go_val>>24)&0xff:02x} mode={mode_val}")
      dx, dy = self.dispatch_core
      s48_base = 0xFFB70000
      tlb_base = s48_base & ~((1 << 21) - 1)
      dcfg = TLBConfig(addr=tlb_base, start=(dx, dy), end=(dx, dy), noc=0, mcast=False, mode=TLBMode.STRICT)
      self._cq.dispatch_win.configure(dcfg)
      avail = self._cq.dispatch_win.read32(s48_base - tlb_base + 297 * 4)
      print(f"  dispatch ({dx},{dy}): stream48 AVAILABLE=0x{avail:08x}")
      TLBWindow(self.fd, TLBSize.MiB_2, TLBConfig(addr=0, start=(dx, dy), end=(dx, dy), noc=0, mcast=False, mode=TLBMode.STRICT))
      raise
