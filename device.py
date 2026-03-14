import functools, os, struct, time
from pathlib import Path
from typing import Callable, Literal

from hw import *
from hw import _ioctl_set_power_state
from dispatch import *
from dram import DramBuffer, Allocator, Shape, tilize, untilize, build_transfer_program

_TS_STRIDE = 16
_TS_MAX_SLOTS = 512

class Device:
  _PREFETCH_CORE: Core = (14, 2)
  _DISPATCH_CORE: Core = (14, 3)
  _CQ_CORES = {_PREFETCH_CORE, _DISPATCH_CORE}

  @functools.cached_property
  def cores(self) -> list[Core]:
    if self._use_fast_dispatch:
      return [c for c in TileGrid.WORKER_CORES if c not in self._CQ_CORES]
    return list(TileGrid.WORKER_CORES)

  def __init__(self, device: int = 0):
    self.device = device
    self.path = f"/dev/tenstorrent/{device}"
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    card_type = Path(f"/sys/class/tenstorrent/tenstorrent!{device}/tt_card_type")
    self.arch = card_type.read_text().strip()
    if self.arch != "p100a":
      os.close(self.fd)
      raise SystemExit(f"unsupported blackhole device {self.arch}; p100a only for now")

    self._init_dram_tiles()
    self.dram = Allocator(self.fd, self.dram_tiles)
    self._init_timestamp_dram()
    self._dispatch_mode = DevMsgs.DISPATCH_MODE_HOST if USE_USB_DISPATCH else DevMsgs.DISPATCH_MODE_DEV
    self._use_fast_dispatch = not USE_USB_DISPATCH

    from compiler import Compiler
    self.compiler = Compiler()
    self._upload_firmware()

    self._dram_sysmem = Sysmem(self.fd) if self._use_fast_dispatch else None
    self.cq = CommandQueue()
    self._cq_hw = None
    if self._use_fast_dispatch:
      self._cq_hw = CQSysmem(
        self.fd,
        prefetch_win=TLBWindow(self.fd, start=self._PREFETCH_CORE),
        dispatch_win=TLBWindow(self.fd, start=self._DISPATCH_CORE),
      )
      self._start_dispatch_cores()

    self._programs = []
    self.last_profile = None

    from compiler import PROFILER
    self._profiler = PROFILER and self._use_fast_dispatch
    self._profiler_flat_ids = {}
    self._profiler_core_count_per_dram = 0
    self._profiler_dram_addr = 0
    if self._profiler:
      self._init_profiler_dram()

  def _init_dram_tiles(self):
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
      off = tag_to_offset.get(Arc.TAG_GDDR_ENABLED)
      gddr_enabled = Arc.DEFAULT_GDDR_ENABLED if off is None else arc.read32(data_base + off * 4)
    harvested = [bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0]
    self.harvested_dram = harvested[0]
    self.dram_tiles = [
      (bank, Dram.BANK_X[bank], y)
      for bank in range(Dram.BANK_COUNT)
      if bank != self.harvested_dram
      for y in Dram.BANK_TILE_YS[bank]
    ]

  def _set_power(self, busy: bool):
    _ioctl_set_power_state(self.fd, validity=1, power_flags=int(busy))

  def _upload_firmware(self):
    fw = self.compiler._fw
    mmio_base, mmio_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBWindow.SIZE_2M)
    reset_off = TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0 - mmio_base
    staged = {}
    for name, cfw in fw.items():
      scratch = cfw.scratch_base
      spans = []
      for s in cfw.segments:
        if not s.data and s.memsz == 0:
          continue
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
    all_cores = list(TileGrid.WORKER_CORES)
    bank_table = build_bank_noc_table(self.harvested_dram, all_cores)

    rects = mcast_rects(all_cores)
    with TLBWindow(self.fd, start=all_cores[0]) as win:
      # assert soft reset on all cores via multicast
      for x0, x1, y0, y1 in rects:
        win.target((x0, y0), (x1, y1), addr=mmio_base)
        win.write32(reset_off, TensixMMIO.SOFT_RESET_ALL)

      # upload firmware segments + bootstrap to all cores via multicast
      for x0, x1, y0, y1 in rects:
        win.target((x0, y0), (x1, y1))
        for spans in staged.values():
          for addr, data in spans:
            win.write(addr, data)
        win.write(0, jal)
        win.write(TensixL1.GO_MSG, go_init)
        win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, bank_table)

      # set subordinate reset PCs on all cores via multicast
      for x0, x1, y0, y1 in rects:
        win.target((x0, y0), (x1, y1), addr=mmio_base)
        for reg, text_base in [
          (TensixMMIO.RISCV_DEBUG_REG_NCRISC_RESET_PC, fw["ncrisc"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC0_RESET_PC, fw["trisc0"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC1_RESET_PC, fw["trisc1"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC2_RESET_PC, fw["trisc2"].text_base),
        ]:
          win.write32(reg - mmio_base, text_base)

      # release BRISC from reset on all cores via multicast
      # all prior writes used STRICT ordering, so data has landed on every core
      for x0, x1, y0, y1 in rects:
        win.target((x0, y0), (x1, y1), addr=mmio_base)
        win.write32(reset_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)

      probe = (1, 2) if (1, 2) in all_cores else all_cores[0]
      win.target(probe)
      deadline = time.perf_counter() + 2.0
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"firmware not ready on {probe} -- try tt-smi -r")
        time.sleep(0.001)

  def _start_dispatch_cores(self):
    cq_kernels = self.compiler.compile_cq_kernels()

    kernel_off = L1_ALIGN + 2 * L1_ALIGN
    pref_img = b"\0" * L1_ALIGN + struct.pack("<I", CQ_DISPATCH_CB_PAGES).ljust(L1_ALIGN, b"\0") + b"\0" * L1_ALIGN
    pref_launch = self._build_cq_launch(kernel_off, 0, sem_off=L1_ALIGN)

    disp_img = b"\0" * L1_ALIGN + b"\0" * L1_ALIGN + b"\0" * L1_ALIGN
    ncrisc_off = align_up(kernel_off + len(cq_kernels["dispatch_brisc"].xip), L1_ALIGN)
    disp_launch = self._build_cq_launch(kernel_off, ncrisc_off, sem_off=L1_ALIGN)

    self._cq_hw.reset_run_state()
    self._upload_cq_core(
      self._PREFETCH_CORE, pref_img, pref_launch,
      [(kernel_off, cq_kernels["prefetch_brisc"].xip)],
    )
    self._upload_cq_core(
      self._DISPATCH_CORE, disp_img, disp_launch,
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
    kc.rta_offset[0].crta_offset = L1_ALIGN
    kc.kernel_text_offset[0] = brisc_text_off
    kc.kernel_text_offset[1] = ncrisc_text_off
    kc.enables = 1 | (2 if ncrisc_text_off else 0)
    kc.mode = DevMsgs.DISPATCH_MODE_HOST
    kc.local_cb_mask = 0
    kc.min_remote_cb_start_index = FAST_CQ_NUM_CIRCULAR_BUFFERS
    return launch

  def _init_dispatch_core_state(self, win: TLBWindow):
    base_16b = self._cq_hw._completion_base_16b
    win.write32(CQ_COMPLETION_WR_PTR, base_16b)
    win.write32(CQ_COMPLETION_RD_PTR, base_16b)
    win.write32(CQ_COMPLETION_Q0_EVENT, 0)
    win.write32(CQ_COMPLETION_Q1_EVENT, 0)
    win.uc[CQ_DISPATCH_SYNC_SEM : CQ_DISPATCH_SYNC_SEM + 8 * L1_ALIGN] = b"\0" * (8 * L1_ALIGN)

  def _upload_cq_core(self, core: Core, img: bytes, launch: LaunchMsg,
                      kernels: list[tuple[int, bytes]], init: Callable[[TLBWindow], None] | None = None):
    win = self._cq_hw._prefetch_win if core == self._PREFETCH_CORE else self._cq_hw._dispatch_win
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

  def _go_word(self) -> int:
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go.bits.master_x, go.bits.master_y = self._DISPATCH_CORE
    go.bits.dispatch_message_offset = 0
    return go.all

  def _init_timestamp_dram(self):
    _, x, y = next((b, x, y) for b, x, y in self.dram.bank_tiles if y != 0)
    self._ts_noc_xy = noc_xy(x, y)
    self._ts_bank_tile = (x, y)
    self._ts_addr = self.dram.next
    self.dram.next = align_up(self._ts_addr + _TS_MAX_SLOTS * _TS_STRIDE, Dram.ALIGNMENT)

  def _ts_noc_dest(self, slot: int) -> tuple[int, int]:
    return self._ts_noc_xy, self._ts_addr + slot * _TS_STRIDE

  def _read_ts_slot(self, slot: int) -> int:
    self.dram.win.target(self._ts_bank_tile, mode=NocOrdering.RELAXED)
    addr = self._ts_addr + slot * _TS_STRIDE
    lo = self.dram.win.read32(addr)
    hi = self.dram.win.read32(addr + 4)
    return (hi << 32) | lo

  def _init_profiler_dram(self):
    cores = sorted(self.cores, key=lambda xy: (xy[0], xy[1]))
    self._profiler_flat_ids = {core: i for i, core in enumerate(cores)}
    bank_count = len(self.dram.bank_tiles)
    self._profiler_core_count_per_dram = max(1, (len(cores) + bank_count - 1) // bank_count)
    bytes_per_risc = TensixL1.PROFILER_HOST_BUFFER_BYTES_PER_RISC
    page_size = bytes_per_risc * 5 * self._profiler_core_count_per_dram
    addr = self.dram.next
    self.dram.next = align_up(addr + page_size, Dram.ALIGNMENT)
    self._profiler_dram_addr = addr
    self._profiler_page_size = page_size

  def _collect_profiler_data(self):
    programs_info = []
    for i, prog in enumerate(self._programs):
      if not prog.profile: continue
      sources = {}
      if prog.reader_kernel: sources["reader"] = prog.reader_kernel
      if prog.writer_kernel: sources["writer"] = prog.writer_kernel
      if prog.compute_kernel: sources["compute"] = prog.compute_kernel
      if prog.grid is not None:
        rows, cols = prog.grid
        cores = sorted([(x, y) for x in cols for y in rows], key=lambda c: (c[0], c[1]))
      else:
        cores = self.cores if prog.cores == "all" else self.cores[:prog.cores]
      programs_info.append({"index": i, "name": prog.name or None, "cores": cores, "sources": sources})
    if not programs_info: return
    self.dram.barrier()
    needed = sorted({c for info in programs_info for c in info["cores"]})
    ctrl_regs = {}
    for core in needed:
      with TLBWindow(self.fd, start=core) as win:
        ctrl_regs[core] = bytes(win.uc[TensixL1.PROFILER_CONTROL : TensixL1.PROFILER_CONTROL + 128])
    import profiler
    self.last_profile = profiler.collect(
      programs_info,
      self.dram.read_raw_bank_pages(self._profiler_dram_addr, self._profiler_page_size),
      ctrl_regs,
      flat_ids=self._profiler_flat_ids,
      page_size=self._profiler_page_size,
      core_count_per_dram=self._profiler_core_count_per_dram,
      harvested_dram_bank=self.harvested_dram,
    )
    profiler.print_summary(self.last_profile)

  def alloc_write(self, data: bytes, dtype: Dtype, shape: Shape, name: str = "") -> DramBuffer:
    buf = self.dram.alloc(len(data) // dtype.tile_size, dtype, name, shape)
    self.dram_write(buf, data)
    return buf

  def _run_transfer(self, buf: DramBuffer, direction: str):
    prog, _ = build_transfer_program(buf, direction, len(self.cores), self._dram_sysmem.noc_addr)
    assert not self._programs, "queue must be empty for DRAM transfers"
    self.queue(prog)
    self.run()

  def dram_write(self, buf: DramBuffer, data: bytes):
    assert len(data) <= buf.size
    if self._use_fast_dispatch and buf.shape is not None:
      assert buf.size <= self._dram_sysmem.size, f"transfer {buf.size} exceeds sysmem {self._dram_sysmem.size}"
      self._dram_sysmem.buf[:len(data)] = data
      self._run_transfer(buf, "tilize")
      return
    if buf.shape is not None:
      data = tilize(data, buf.dtype.bpe, buf.shape)
    self.dram.write(buf, data)

  def dram_read(self, buf: DramBuffer) -> bytes:
    if self._use_fast_dispatch and buf.shape is not None:
      assert buf.size <= self._dram_sysmem.size, f"transfer {buf.size} exceeds sysmem {self._dram_sysmem.size}"
      self._run_transfer(buf, "untilize")
      return bytes(self._dram_sysmem.buf[:buf.size])
    result = self.dram.read(buf)
    if buf.shape is not None: return untilize(result, buf.dtype.bpe, buf.shape)
    return result

  def queue(self, program: Program): self._programs.append(program)

  def _compile_ir(self, program: Program, dispatch_mode, host_assigned_id: int = 0) -> list:
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
      roles: list[Role] = [(cs, rk, wk) for cs, rk, wk in [
        (top_left, reader, writer), (top_row, r_recv, writer), (left_col, reader, w_recv), (interior, r_recv, w_recv),
      ] if cs]
    else:
      cores = self.cores if program.cores == "all" else self.cores[:program.cores]
      all_cores = cores
      n = len(cores)
      per_core_args = [
        (resolve_args(program.writer_args, i, c, n), resolve_args(program.reader_args, i, c, n),
         resolve_args(program.compute_args, i, c, n))
        for i, c in enumerate(cores)
      ]
      roles = [(cores, reader, writer)]

    return build_ir(program, roles, compute, all_cores, per_core_args, dispatch_mode, host_assigned_id=host_assigned_id)

  def run(self) -> list[dict] | None:
    self._set_power(True)
    try:
      if self._use_fast_dispatch: return self._run_fast_dispatch()
      else: return self._run_slow_dispatch()
    finally:
      self._programs.clear()
      self._set_power(False)

  def _run_fast_dispatch(self) -> list[dict] | None:
    timing = os.environ.get("TIMING") == "1"
    n = len(self._programs)

    programs = []
    for i, program in enumerate(self._programs):
      profiled = program.profile
      prof_id = (i + 1) if self._profiler and profiled else 0
      ir = self._compile_ir(program, self._dispatch_mode, host_assigned_id=prof_id)
      programs.append((ir, profiled))

    timestamps = None
    if timing:
      timestamps = [self._ts_noc_dest(s) for s in range(min(2 * n, _TS_MAX_SLOTS))]

    self.cq.extend(lower_fast(
      programs, self._go_word(), self.cores,
      timestamps=timestamps,
      profiler_flat_ids=self._profiler_flat_ids or None,
      profiler_dram_addr=self._profiler_dram_addr,
      profiler_core_count_per_dram=self._profiler_core_count_per_dram,
    ))
    self._cq_hw._event_id += 1
    self.cq.append(CQHostEvent(self._cq_hw._event_id))
    self._cq_hw.flush(self.cq)
    self._cq_hw.wait_completion(self._cq_hw._event_id)

    if self._profiler:
      self._collect_profiler_data()
    if not timing:
      return None
    return self._collect_timing_data(n)

  def _run_slow_dispatch(self) -> list[dict] | None:
    timing = os.environ.get("TIMING") == "1"
    t0 = time.perf_counter() if timing else 0
    with TLBWindow(self.fd, start=self.cores[0]) as win:
      for program in self._programs:
        ir = self._compile_ir(program, self._dispatch_mode)
        slow_dispatch(win, ir)
    if timing:
      elapsed_us = (time.perf_counter() - t0) * 1e6
      print(f"  slow dispatch: {elapsed_us:.1f} us (host wall-clock)")
    return None

  def _collect_timing_data(self, n: int) -> list[dict]:
    self.dram.barrier()
    freq_mhz = 1350
    timings = []
    for i in range(n):
      ts_slot = 2 * i
      if ts_slot + 1 >= _TS_MAX_SLOTS:
        break
      t0 = self._read_ts_slot(ts_slot)
      t1 = self._read_ts_slot(ts_slot + 1)
      cycles = t1 - t0
      timings.append({"cycles": cycles, "us": cycles / freq_mhz, "freq_mhz": freq_mhz})
    for i, t in enumerate(timings):
      print(f"  [{i}] {t['us']:.1f} us ({t['cycles']} cycles)")
    self.last_device_timing = timings
    return timings

  def serve_profile(self, port: int = int(os.environ.get("PORT", 8000))):
    from profiler import ui as profiler_ui
    profiler_ui.serve(self.last_profile, port=port)

  def close(self):
    self._set_power(False)
    if self._dram_sysmem is not None:
      self._dram_sysmem.close()
    if self._cq_hw is not None:
      self._cq_hw.close()
    self.dram.close()
    os.close(self.fd)
