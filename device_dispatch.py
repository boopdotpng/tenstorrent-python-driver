import ctypes, fcntl, mmap, struct, time
from dataclasses import dataclass
from pathlib import Path
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import tlog, _IO, align_down, iter_pt_load, generate_jal_instruction
from dram import DramAllocator
from device_runtime import CommonDevice, Program, TileGrid, ArgGen

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
      flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE,
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

  def close(self):
    try:
      if hasattr(self, "prefetch_win"): self.prefetch_win.free()
    finally:
      if hasattr(self, "sysmem"):
        unpin = UnpinPagesIn(virtual_address=self.sysmem_addr, size=self.host.sysmem_size, reserved=0)
        fcntl.ioctl(self.fd, _IO(IOCTL_UNPIN_PAGES), bytearray(as_bytes(unpin)), False)
        self.sysmem.close()

  def _read_prefetch_entry(self, idx: int) -> int:
    off = self.dev.prefetch_q_base + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    return struct.unpack("<H", self.prefetch_win.uc[off : off + 2])[0]

  def _wait_prefetch_slot_free(self, idx: int, timeout_s: float = 1.0):
    deadline = time.perf_counter() + timeout_s
    while self._read_prefetch_entry(idx) != 0:
      if time.perf_counter() > deadline:
        raise TimeoutError("timeout waiting for prefetch queue slot")

  def _write_prefetch_q_entry(self, size_16b: int):
    if not (0 < size_16b <= 0x7FFF):
      raise ValueError(f"prefetch entry out of range: {size_16b}")
    idx = self.prefetch_q_wr_idx
    self._wait_prefetch_slot_free(idx)
    off = self.dev.prefetch_q_base + idx * FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self.prefetch_win.uc[off : off + 2] = struct.pack("<H", size_16b)
    entries = self.dev.prefetch_q_size // FastDispatch.PREFETCH_Q_ENTRY_BYTES
    self.prefetch_q_wr_idx = (idx + 1) % entries

  def _issue_write(self, record: bytes):
    if len(record) % FastDispatch.PCIE_ALIGNMENT != 0:
      raise ValueError("record must be 64B-aligned")
    wr = _align_up(self.issue_wr, FastDispatch.PCIE_ALIGNMENT)
    if wr + len(record) > self.host.issue_size: wr = 0
    base = self.host.issue_base + wr
    self.sysmem[base : base + len(record)] = record
    self.issue_wr = wr + len(record)
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
    for cb in program.cbs:
      size = program.tile_size * program.num_pages
      arr[cb] = LocalCBConfig(
        addr_bytes=addr,
        size_bytes=size,
        num_pages=program.num_pages,
        page_size_bytes=program.tile_size,
      )
      addr += size
    return mask, as_bytes(arr)

  def _pack_kernel_shared(self, program: Program, rta_sizes: tuple[int, int, int]):
    align16 = lambda n: (n + 15) & ~15

    rta_offsets = [0, rta_sizes[0], rta_sizes[0] + rta_sizes[1]]
    rta_total = align16(rta_offsets[2] + rta_sizes[2])
    crta_off = rta_total

    local_cb_mask, local_cb_blob = self._build_local_cb_blob(program)
    local_cb_off = rta_total
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
    cfg.local_cb_offset = local_cb_off
    cfg.remote_cb_offset = local_cb_off + len(local_cb_blob)
    cfg.local_cb_mask = local_cb_mask
    cfg.min_remote_cb_start_index = CB.NUM_CIRCULAR_BUFFERS
    cfg.enables = enables
    cfg.brisc_noc_id = 1
    cfg.brisc_noc_mode = 0
    cfg.mode = DevMsgs.DISPATCH_MODE_HOST
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
  def _pack_rta(reader_args: list[int], writer_args: list[int], compute_args: list[int]) -> bytes:
    pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
    return pack(writer_args) + pack(reader_args) + pack(compute_args)

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

  def run(self, program: Program) -> tuple[float, float]:
    cores = program.cores if program.cores is not None else self.worker_cores
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
    shared_off, shared_img, launch_blob, rta_offsets = self._pack_kernel_shared(program, rta_sizes)

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

    shared_rta = None
    if (
      isinstance(program.reader_rt_args, list)
      and isinstance(program.writer_rt_args, list)
      and isinstance(program.compute_rt_args, list)
    ):
      shared_rta = self._pack_rta(program.reader_rt_args, program.writer_rt_args, program.compute_rt_args)

    # Unicast per-core runtime args
    for core_idx, (x, y) in enumerate(cores):
      if shared_rta is None:
        reader_args = self._resolve_args(program.reader_rt_args, core_idx, (x, y), num_cores)
        writer_args = self._resolve_args(program.writer_rt_args, core_idx, (x, y), num_cores)
        compute_args = self._resolve_args(program.compute_rt_args, core_idx, (x, y), num_cores)
        rta = self._pack_rta(reader_args, writer_args, compute_args)
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
    tlog("dispatch", dispatch)
    compute = t_end - t_compute_start
    tlog("compute", compute)
    total = dispatch + compute
    return total, dispatch

class FastDevice(SlowDevice):
  def _core_exists(self, core: tuple[int, int]) -> bool:
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)
    cfg = TLBConfig(addr=reg_base, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      return win.read32(reg_off) != 0xFFFF_FFFF

  def _select_dispatch_cores(self) -> tuple[tuple[int, int], tuple[int, int]]:
    y0, _ = self.tiles.TENSIX_Y
    y1 = y0 + 1
    worker_x = sorted({x for x, _ in self.worker_cores}, reverse=True)
    candidates = [
      x for x in worker_x
      if self._core_exists((x, y0)) and self._core_exists((x, y1))
    ]
    if not candidates:
      raise RuntimeError("could not find a valid worker dispatch core column")
    x = candidates[0]
    return (x, y0), (x, y1)

  def _firmware_skip_cores(self) -> set[tuple[int, int]]:
    self._dispatch_core_pair = self._select_dispatch_cores()
    return set(self._dispatch_core_pair)

  def __init__(
    self,
    device: int = 0,
    *,
    sysmem_size: int = 16 * 1024 * 1024,
    issue_size: int = 8 * 1024 * 1024,
    completion_size: int = 4 * 1024 * 1024,
  ):
    super().__init__(device=device)
    self.prefetch_core, self.dispatch_core = getattr(self, "_dispatch_core_pair", self._select_dispatch_cores())
    self._cq = _FastCQ(
      self.fd,
      prefetch_core=self.prefetch_core,
      sysmem_size=sysmem_size,
      issue_size=issue_size,
      completion_size=completion_size,
      prefetch_q_entries=FastDispatch.PREFETCH_Q_ENTRIES_WORKER_DEFAULT,
    )
    self._start_dispatch_cores()

  def close(self):
    if hasattr(self, "_cq"): self._cq.close()
    super().close()

  @staticmethod
  def _load_dispatch_elf(path: Path) -> tuple[list, int]:
    elf = path.read_bytes()
    entry = struct.unpack_from("<I", elf, 24)[0]
    return list(iter_pt_load(elf)), entry

  @staticmethod
  def _write_dispatch_segs(win: TLBWindow, segs: list, local_init_base: int):
    for seg in segs:
      if not seg.data and seg.memsz == 0: continue
      data = seg.data if seg.memsz <= len(seg.data) else seg.data + b"\0" * (seg.memsz - len(seg.data))
      addr = seg.paddr
      if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
        addr = local_init_base + (addr - TensixMMIO.LOCAL_RAM_START)
      if 0 <= addr < TensixL1.SIZE:
        win.write(addr, data, use_uc=True, restore=False)

  @staticmethod
  def _build_dispatch_launch(*, rt_args: list[int], sem_values: list[int], entry: int) -> tuple[bytes, KernelConfigMsg]:
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
    kc.kernel_text_offset[0] = (entry - TensixL1.KERNEL_CONFIG_BASE) & 0xFFFFFFFF
    kc.enables = 1
    kc.mode = DevMsgs.DISPATCH_MODE_HOST
    kc.local_cb_mask = 0
    kc.min_remote_cb_start_index = CB.NUM_CIRCULAR_BUFFERS
    return img, kc

  def _set_soft_reset(self, core: tuple[int, int], value: int):
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)
    cfg = TLBConfig(addr=reg_base, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      win.read32(reg_off)
      win.write32(reg_off, value)

  def _write_mmio32(self, core: tuple[int, int], addr: int, value: int):
    reg_base, reg_off = align_down(addr, TLBSize.MiB_2)
    cfg = TLBConfig(addr=reg_base, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      win.write32(reg_off, value)

  def _wait_dispatch_core_ready(self, core: tuple[int, int], timeout_s: float = 0.5):
    cfg = TLBConfig(addr=0, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    deadline = time.perf_counter() + timeout_s
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      while True:
        go = win.uc[TensixL1.GO_MSG + 3]
        if go != 0xFF:
          return
        if time.perf_counter() > deadline:
          raise TimeoutError(f"dispatch core {core} did not become readable (GO_MSG=0xFF)")
        time.sleep(0.001)

  def _start_dispatch_cores(self):
    fw_dir = Path(__file__).parent / "riscv-firmware" / self.arch
    prefetch_path = fw_dir / "cq_prefetch_brisc.elf"
    dispatch_path = fw_dir / "cq_dispatch_brisc.elf"
    sub_path = fw_dir / "cq_dispatch_subordinate_ncrisc.elf"
    missing = [str(p) for p in (prefetch_path, dispatch_path, sub_path) if not p.exists()]
    if missing:
      raise RuntimeError("missing fast-dispatch firmware: " + ", ".join(missing))

    prefetch_segs, prefetch_entry = self._load_dispatch_elf(prefetch_path)
    dispatch_segs, dispatch_entry = self._load_dispatch_elf(dispatch_path)
    subordinate_segs, subordinate_entry = self._load_dispatch_elf(sub_path)

    self._set_soft_reset(self.prefetch_core, TensixMMIO.SOFT_RESET_ALL)
    self._set_soft_reset(self.dispatch_core, TensixMMIO.SOFT_RESET_ALL)

    cfg_pref = TLBConfig(addr=0, start=self.prefetch_core, end=self.prefetch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg_pref) as win:
      self._write_dispatch_segs(win, prefetch_segs, TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH)
      win.write(0x0, generate_jal_instruction(prefetch_entry).to_bytes(4, "little"), use_uc=True, restore=False)
      win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, self._build_bank_noc_tables(), use_uc=True, restore=False)
      img, kc = self._build_dispatch_launch(rt_args=[0, 0, 0], sem_values=[self._cq.dev.dispatch_cb_pages, 0], entry=prefetch_entry)
      self._cq.init_prefetch_l1()
      win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
      reset = GoMsg(); reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
      launch = LaunchMsg(); launch.kernel_config = kc
      go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO
      win.write(TensixL1.GO_MSG, as_bytes(reset), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little"), use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(launch), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG, as_bytes(go), use_uc=True, restore=False)

    cfg_disp = TLBConfig(addr=0, start=self.dispatch_core, end=self.dispatch_core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg_disp) as win:
      self._write_dispatch_segs(win, dispatch_segs, TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH)
      self._write_dispatch_segs(win, subordinate_segs, TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH)
      win.write(0x0, generate_jal_instruction(dispatch_entry).to_bytes(4, "little"), use_uc=True, restore=False)
      win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, self._build_bank_noc_tables(), use_uc=True, restore=False)
      img, kc = self._build_dispatch_launch(rt_args=[0, 0, 0], sem_values=[0, 0], entry=dispatch_entry)
      win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
      base_16b = ((self._cq.noc_local + self._cq.host.completion_base) >> 4) & 0x7FFF_FFFF
      win.write32(self._cq.dev.completion_q_wr_ptr_addr, base_16b)
      win.write32(self._cq.dev.completion_q_rd_ptr_addr, base_16b)
      win.write32(self._cq.dev.completion_q0_last_event_ptr_addr, 0)
      win.write32(self._cq.dev.completion_q1_last_event_ptr_addr, 0)
      win.uc[self._cq.dev.dispatch_s_sync_sem_addr : self._cq.dev.dispatch_s_sync_sem_addr + (8 * FastDispatch.L1_ALIGNMENT)] = b"\0" * (8 * FastDispatch.L1_ALIGNMENT)
      reset = GoMsg(); reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
      launch = LaunchMsg(); launch.kernel_config = kc
      go = GoMsg(); go.bits.signal = DevMsgs.RUN_MSG_GO
      win.write(TensixL1.GO_MSG, as_bytes(reset), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little"), use_uc=True, restore=False)
      win.write(TensixL1.LAUNCH, as_bytes(launch), use_uc=True, restore=False)
      win.write(TensixL1.GO_MSG, as_bytes(go), use_uc=True, restore=False)

    self._write_mmio32(self.dispatch_core, TensixMMIO.RISCV_DEBUG_REG_NCRISC_RESET_PC, subordinate_entry)
    self._set_soft_reset(self.prefetch_core, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)
    self._set_soft_reset(self.dispatch_core, TensixMMIO.SOFT_RESET_BRISC_NCRISC_RUN)
    self._wait_dispatch_core_ready(self.prefetch_core)
    self._wait_dispatch_core_ready(self.dispatch_core)

  def run(self, program: Program) -> tuple[float, float]:
    if program.cores is not None:
      cores = program.cores
    else:
      reserved = {self.prefetch_core, self.dispatch_core}
      cores = [core for core in self.worker_cores if core not in reserved]
    num_cores = len(cores)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go_blob = as_bytes(go)

    first_r = self._resolve_args(program.reader_rt_args, 0, cores[0], num_cores)
    first_w = self._resolve_args(program.writer_rt_args, 0, cores[0], num_cores)
    first_c = self._resolve_args(program.compute_rt_args, 0, cores[0], num_cores)
    rta_sizes = (len(first_w) * 4, len(first_r) * 4, len(first_c) * 4)
    shared_off, shared_img, launch_blob, _ = self._pack_kernel_shared(program, rta_sizes)

    mcast_cfg = TLBConfig(addr=0, noc=0, mcast=True, mode=TLBMode.STRICT)
    l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    rects = self._mcast_rects(cores)
    win = self.win
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

    shared_rta = None
    if (
      isinstance(program.reader_rt_args, list)
      and isinstance(program.writer_rt_args, list)
      and isinstance(program.compute_rt_args, list)
    ):
      shared_rta = self._pack_rta(program.reader_rt_args, program.writer_rt_args, program.compute_rt_args)

    for core_idx, (x, y) in enumerate(cores):
      if shared_rta is None:
        reader_args = self._resolve_args(program.reader_rt_args, core_idx, (x, y), num_cores)
        writer_args = self._resolve_args(program.writer_rt_args, core_idx, (x, y), num_cores)
        compute_args = self._resolve_args(program.compute_rt_args, core_idx, (x, y), num_cores)
        rta = self._pack_rta(reader_args, writer_args, compute_args)
      else:
        rta = shared_rta
      l1_cfg.start = l1_cfg.end = (x, y)
      win.configure(l1_cfg)
      win.write(TensixL1.KERNEL_CONFIG_BASE, rta, use_uc=True, restore=False)

    self._cq.reset_run_state()
    t_dispatch_start = time.perf_counter()
    for core in cores:
      self._cq.enqueue_write_linear(tile=core, addr=TensixL1.GO_MSG, data=go_blob)
    t_compute_start = time.perf_counter()

    l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    for x, y in cores:
      l1_cfg.start = l1_cfg.end = (x, y)
      self.win.configure(l1_cfg)
      deadline = time.perf_counter() + 10.0
      while self.win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"timeout waiting for core ({x}, {y})")

    t_end = time.perf_counter()
    dispatch = t_compute_start - t_dispatch_start
    tlog("dispatch", dispatch)
    compute = t_end - t_compute_start
    tlog("compute", compute)
    total = dispatch + compute
    return total, dispatch
