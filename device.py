import os, struct, time
from dataclasses import dataclass, field
from typing import ClassVar, Callable, Literal
from pathlib import Path
from defs import *
from codegen import CompiledKernel, compile_firmware
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import USE_USB_DISPATCH, align_down, align_up, generate_jal_instruction, noc_xy
from dram import DramAllocator

CoreSpec = int | Literal["all"]
BankPort = tuple[int, int]
ArgGen = Callable[[int, Core, int], list[int]]
Rect = tuple[int, int, int, int]
McastDest = tuple[int, int]
CorePair = tuple[Core, Core]
AddrPayload = tuple[int, bytes]
RtaSizes = tuple[int, int, int]
FAST_CQ_NUM_CIRCULAR_BUFFERS = 32

@dataclass
class DataflowLaunch:
  cores: CoreList
  reader_rt_args: list[int] | ArgGen
  writer_rt_args: list[int] | ArgGen
  reader: CompiledKernel | None = None
  writer: CompiledKernel | None = None

@dataclass
class Program:
  dataflow: list[DataflowLaunch]
  compute: tuple[CompiledKernel, CompiledKernel, CompiledKernel] | None
  compute_rt_args: list[int] | ArgGen
  cbs: list[int]
  tile_size: int
  num_pages: int
  cores: CoreSpec = "all"
  num_sems: int = 0
  cb_config: dict[int, tuple[int, int]] | None = None  # {cb_id: (num_pages, page_size)}

@dataclass(frozen=True)
class _LaunchRole:
  core: Core
  launch: DataflowLaunch
  role_idx: int
  role_count: int

@dataclass(frozen=True)
class _CorePayload:
  core: Core
  rta: bytes
  launch_blob: bytes
  shared_addr: int
  shared_blob: bytes

@dataclass(frozen=True)
class _CorePlan:
  cores: CoreList
  rects: list[Rect]
  mcast_dests: list[McastDest]

@dataclass
class TileGrid:
  ARC: ClassVar[Core] = (8, 0)
  TENSIX_X: ClassVar[tuple[int, ...]] = (*range(1, 8), *range(10, 15))
  TENSIX_Y: ClassVar[tuple[int, int]] = (2, 11)
  TENSIX: ClassVar[CoreList] = [(x, y) for x in TENSIX_X for y in range(2, 12)]
  TENSIX_MCAST: ClassVar[CoreList] = [(1, 7), (10, 14)]

  harvested_dram_bank: int
  dram: DramTileList = field(init=False)

  def __post_init__(self):
    self.dram = [(bank, Dram.BANK_X[bank], y) for bank in range(Dram.BANK_COUNT)
    if bank != self.harvested_dram_bank for y in Dram.BANK_TILE_YS[bank]]

class CommonDevice:
  def __init__(self, device: int = 0):
    if device < 0:
      raise ValueError("device must be >= 0")
    self.device = device
    self.path = f"/dev/tenstorrent/{device}"
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    ordinal = str(device)
    self.arch = Path(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type").read_text().strip()
    if self.arch != "p100a":
      os.close(self.fd)
      raise SystemExit(f"unsupported blackhole device {self.arch}; p100a only for now")
    self._assert_arc_booted()
    self._set_power_state_busy()
    self.harvested_dram = self.get_harvested_dram_bank()
    self.tiles = TileGrid(self.harvested_dram)
    self.worker_cores = [core for core in TileGrid.TENSIX if self._core_exists(core)]
    if not self.worker_cores:
      raise RuntimeError("no active Tensix worker cores detected")
    self.dispatchable_cores = list(self.worker_cores)
    self.upload_firmware()

  def _core_exists(self, core: Core) -> bool:
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)
    cfg = TLBConfig(addr=reg_base, start=core, end=core, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      return win.read32(reg_off) != 0xFFFF_FFFF

  @staticmethod
  def _tile_ready(win: TLBWindow) -> bool:
    go = win.uc[TensixL1.GO_MSG + 3]
    sync = win.read32(TensixL1.MAILBOX_BASE + 8)
    dm1, tr0, tr1, tr2 = sync & 0xFF, (sync >> 8) & 0xFF, (sync >> 16) & 0xFF, (sync >> 24) & 0xFF
    return go == DevMsgs.RUN_MSG_DONE and dm1 == 0 and tr1 == 0 and tr2 == 0 and tr0 in (0, 3)

  # Map each firmware target to its init scratch area in L1 (for local-mem data relocation)
  _INIT_SCRATCH = {
    "brisc":  TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH,
    "ncrisc": TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH,
    "trisc0": TensixL1.TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
    "trisc1": TensixL1.TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
    "trisc2": TensixL1.TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
  }

  def _cores_needing_firmware(self, cores: list[Core]) -> list[Core]:
    """Probe cores and return those that need firmware upload.
    A core is 'ready' if BRISC is out of reset and the tile is idle (RUN_MSG_DONE).
    Cores running stale CQ firmware or held in reset are returned for re-upload."""
    if not cores: return cores
    # Quick check: if the first core needs firmware, all likely do (cold boot)
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)
    cfg = TLBConfig(addr=reg_base, start=cores[0], end=cores[0], noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      def _core_ready(core: Core) -> bool:
        cfg.start = cfg.end = core
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        if win.read32(reg_off) & TensixMMIO.SOFT_RESET_BRISC: return False
        cfg.addr, cfg.mode = 0, TLBMode.STRICT
        win.configure(cfg)
        return self._tile_ready(win)
      if not _core_ready(cores[0]): return cores
      return [c for c in cores[1:] if not _core_ready(c)]

  def upload_firmware(self):
    skip = self._firmware_skip_cores()
    all_cores = [core for core in self.worker_cores if core not in skip]
    cores = self._cores_needing_firmware(all_cores)
    if not cores: return

    fw = compile_firmware()
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)

    # Prepare upload spans: resolve local-mem segments to L1 scratch addresses
    staged: dict[str, list[tuple[int, bytes]]] = {}
    for name, cfw in fw.items():
      init_scratch = self._INIT_SCRATCH[name]
      spans = []
      for s in cfw.segments:
        if not s.data and s.memsz == 0: continue
        data = s.data if s.memsz <= len(s.data) else s.data + b"\0" * (s.memsz - len(s.data))
        addr = s.paddr
        if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
          addr = init_scratch + (addr - TensixMMIO.LOCAL_RAM_START)
        assert 0 <= addr < TensixL1.SIZE, f"{name}: bad paddr 0x{s.paddr:x} -> 0x{addr:x}"
        spans.append((addr, data))
      staged[name] = spans

    cfg = TLBConfig(addr=reg_base, noc=0, mcast=True, mode=TLBMode.STRICT)
    # BRISC has no configurable reset PC — it always boots from address 0.
    # We place a JAL instruction there to jump to the firmware entry point.
    jal_insn = generate_jal_instruction(TensixL1.BRISC_FIRMWARE_BASE)
    go_msg = struct.pack("<BBBB", 0, 0, 0, DevMsgs.RUN_MSG_INIT)

    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      def _reset_and_stage():
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.write32(reg_off, TensixMMIO.SOFT_RESET_ALL)

        cfg.mode = TLBMode.RELAXED
        for spans in staged.values():
          for addr, data in spans:
            win.write(addr, data, use_uc=True, restore=False)

        win.write(0x0, jal_insn.to_bytes(4, "little"), use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, go_msg, use_uc=True, restore=False)

        # Set reset PCs for NCRISC and TRISCs (BRISC uses the JAL at addr 0)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        for reg, base in [
          (TensixMMIO.RISCV_DEBUG_REG_NCRISC_RESET_PC, fw["ncrisc"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC0_RESET_PC, fw["trisc0"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC1_RESET_PC, fw["trisc1"].text_base),
          (TensixMMIO.RISCV_DEBUG_REG_TRISC2_RESET_PC, fw["trisc2"].text_base),
        ]:
          win.write32(reg - reg_base, base)

      cfg.mcast = False
      for core in cores:
        cfg.start = cfg.end = core
        _reset_and_stage()

      test_tile = cores[0]
      cfg.start, cfg.end = test_tile, test_tile
      cfg.addr, cfg.mode = 0, TLBMode.STRICT
      cfg.mcast = False
      win.configure(cfg)
      cfg.mcast = True

      bank_tables = self._build_bank_noc_tables()
      def _write_bank_tables():
        cfg.addr, cfg.mode = 0, TLBMode.RELAXED
        win.configure(cfg)
        win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, bank_tables, use_uc=True, restore=False)
      for core in cores:
        cfg.start = cfg.end = core
        _write_bank_tables()

      def _release_brisc():
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.read32(reg_off)
        win.write32(reg_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)
      for core in cores:
        cfg.start = cfg.end = core
        _release_brisc()

    self._wait_firmware_ready()

  def _firmware_skip_cores(self) -> set[Core]:
    return set()

  def _wait_firmware_ready(self):
    skip = self._firmware_skip_cores()
    candidates = [core for core in self.worker_cores if core not in skip]
    tile = (1, 2) if (1, 2) in candidates else candidates[0]
    cfg = TLBConfig(addr=0, start=tile, end=tile, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      deadline = time.perf_counter() + 2.0
      while True:
        if self._tile_ready(win): return
        if time.perf_counter() > deadline:
          go = win.uc[TensixL1.GO_MSG + 3]
          sync = win.read32(TensixL1.MAILBOX_BASE + 8)
          raise TimeoutError(f"firmware not ready on tile {tile}: go={go:#x} sync={sync:#x}")
        time.sleep(0.001)

  def _build_bank_noc_tables(self) -> bytes:
    NUM_NOCS, NUM_DRAM_BANKS, NUM_L1_BANKS = 2, 7, 110
    WORKER_EP_LOGICAL = {0: [2, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [2, 1], 5: [2, 1], 6: [2, 1], 7: [2, 1]}

    def dram_translated_map(harvested_bank: int | None) -> dict[BankPort, Core]:
      START_X, START_Y, PORTS, TOTAL_BANKS = 17, 12, 3, 8
      m: dict[BankPort, Core] = {}

      def map_banks(start: int, end: int, x: int, y0: int = START_Y):
        y = y0
        for bank in range(start, end):
          for port in range(PORTS):
            m[(bank, port)] = (x, y)
            y += 1

      if harvested_bank is None:
        map_banks(0, TOTAL_BANKS // 2, START_X)
        map_banks(TOTAL_BANKS // 2, TOTAL_BANKS, START_X + 1)
        return m
      if not (0 <= harvested_bank < TOTAL_BANKS):
        raise ValueError("harvested_dram_bank out of range")
      half = TOTAL_BANKS // 2
      if harvested_bank < half:
        mirror_east_bank = harvested_bank + half - 1
        map_banks(0, half - 1, START_X + 1)
        map_banks(half - 1, mirror_east_bank, START_X)
        map_banks(mirror_east_bank + 1, TOTAL_BANKS - 1, START_X, START_Y + (mirror_east_bank - (half - 1)) * PORTS)
        map_banks(mirror_east_bank, mirror_east_bank + 1, START_X, START_Y + (half - 1) * PORTS)
      else:
        mirror_west_bank = harvested_bank - half
        map_banks(0, mirror_west_bank, START_X)
        map_banks(mirror_west_bank + 1, half, START_X, START_Y + mirror_west_bank * PORTS)
        map_banks(mirror_west_bank, mirror_west_bank + 1, START_X, START_Y + (half - 1) * PORTS)
        map_banks(half, TOTAL_BANKS - 1, START_X + 1)
      return m

    dram_translated = dram_translated_map(self.harvested_dram)
    dram_xy = []
    for noc in range(NUM_NOCS):
      for logical_bank in range(NUM_DRAM_BANKS):
        port = WORKER_EP_LOGICAL[logical_bank][noc]
        x, y = dram_translated[(logical_bank, port)]
        dram_xy.append(noc_xy(x, y))

    tensix_cols = sorted({x for x, _ in self.worker_cores})
    l1_xy = []
    for _ in range(NUM_NOCS):
      for bank_id in range(NUM_L1_BANKS):
        col_idx = bank_id % len(tensix_cols)
        row_idx = bank_id // len(tensix_cols)
        x = tensix_cols[col_idx]
        y = 2 + (row_idx % 10)
        l1_xy.append(noc_xy(x, y))

    dram_offsets = [0] * NUM_DRAM_BANKS
    l1_offsets = [0] * NUM_L1_BANKS
    blob = struct.pack(f"<{len(dram_xy)}H", *dram_xy)
    blob += struct.pack(f"<{len(l1_xy)}H", *l1_xy)
    blob += struct.pack(f"<{len(dram_offsets)}i", *dram_offsets)
    blob += struct.pack(f"<{len(l1_offsets)}i", *l1_offsets)
    return blob

  def _read_arc_boot_status(self) -> int:
    cfg = TLBConfig(addr=Arc.NOC_BASE, start=TileGrid.ARC, end=TileGrid.ARC, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as arc:
      return arc.read32(Arc.SCRATCH_RAM_2)

  def _assert_arc_booted(self, timeout_s: float = 5.0):
    deadline = time.perf_counter() + timeout_s
    status = self._read_arc_boot_status()
    while (status & 0x7) != 0x5 and time.perf_counter() < deadline:
      time.sleep(0.001)
      status = self._read_arc_boot_status()
    if (status & 0x7) != 0x5:
      self._close()
      raise RuntimeError(f"ARC not booted (SCRATCH_RAM_2=0x{status:08x}, expected (status&0x7)==0x5)")

  def get_harvested_dram_bank(self) -> int:
    gddr_enabled = self._read_arc_telem_tag(Arc.TAG_GDDR_ENABLED, Arc.DEFAULT_GDDR_ENABLED)
    dram_off = [bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0]
    assert len(dram_off) == 1, f"expected 1 harvested dram bank, got {dram_off}"
    return dram_off[0]

  def _read_arc_telem_tag(self, tag: int, default: int) -> int:
    tlb_config = TLBConfig(addr=Arc.NOC_BASE, start=TileGrid.ARC, end=TileGrid.ARC, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, tlb_config) as arc:
      telem_struct_addr = arc.read32(Arc.SCRATCH_RAM_13)
      if telem_struct_addr == 0 or not (Arc.CSM_START <= telem_struct_addr <= Arc.CSM_END):
        raise RuntimeError("device probably not working, try tt-smi -r")
      csm_base, csm_offset = align_down(telem_struct_addr, TLBSize.MiB_2)
      tlb_config.addr = csm_base
      arc.configure(tlb_config)
      entry_count = arc.read32(csm_offset + 4)
      tags_base = csm_offset + 8
      data_base = tags_base + entry_count * 4
      tag_to_offset = {}
      for i in range(entry_count):
        tag_offset = arc.read32(tags_base + i * 4)
        tag_to_offset[tag_offset & 0xFFFF] = (tag_offset >> 16) & 0xFFFF

      off = tag_to_offset.get(tag)
      return default if off is None else arc.read32(data_base + off * 4)

  def _set_power_state_busy(self, timeout_s: float = 2.0):
    self.arc_msg(Arc.MSG_AICLK_GO_BUSY, 0, 0, timeout_ms=1000)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
      aiclk = self._read_arc_telem_tag(Arc.TAG_AICLK, Arc.DEFAULT_AICLK)
      if aiclk > Arc.DEFAULT_AICLK: return
      time.sleep(0.001)
    raise RuntimeError(f"AICLK failed to enter busy state (last={aiclk} MHz)")

  def _close(self):
    if hasattr(self, "dram"): self.dram.close()
    if hasattr(self, "win"): self.win.free()
    os.close(self.fd)

  def arc_msg(self, msg: int, arg0: int = 0, arg1: int = 0, queue: int = 0, timeout_ms: int = 1000) -> list[int]:
    MSG_QUEUE_SIZE, REQUEST_MSG_LEN, RESPONSE_MSG_LEN = 4, 8, 8
    MSG_QUEUE_POINTER_WRAP = 2 * MSG_QUEUE_SIZE
    HEADER_BYTES, REQUEST_BYTES, RESPONSE_BYTES = 8 * 4, REQUEST_MSG_LEN * 4, RESPONSE_MSG_LEN * 4
    QUEUE_STRIDE = HEADER_BYTES + (MSG_QUEUE_SIZE * REQUEST_BYTES) + (MSG_QUEUE_SIZE * RESPONSE_BYTES)
    RESET_UNIT_ARC_MISC_CNTL = Arc.RESET_UNIT_OFFSET + 0x100
    IRQ0_TRIG_BIT = 1 << 16
    if queue < 0 or queue >= 4: raise ValueError("queue must be 0..3")
    cfg = TLBConfig(addr=Arc.NOC_BASE, start=TileGrid.ARC, end=TileGrid.ARC, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as arc:
      info_ptr = arc.read32(Arc.SCRATCH_RAM_11)
      if info_ptr == 0: raise RuntimeError("msgqueue not initialized (SCRATCH_RAM_11 == 0)")

      info_base, info_off = align_down(info_ptr, TLBSize.MiB_2)
      cfg.addr = info_base
      arc.configure(cfg)
      queues_ptr = arc.read32(info_off)

      q_base, q_off = align_down(queues_ptr, TLBSize.MiB_2)
      cfg.addr = q_base
      arc.configure(cfg)
      q = q_off + queue * QUEUE_STRIDE

      wptr = arc.read32(q + 0)
      req = q + HEADER_BYTES + (wptr % MSG_QUEUE_SIZE) * REQUEST_BYTES
      words = [msg & 0xFF, arg0 & 0xFFFFFFFF, arg1 & 0xFFFFFFFF] + [0] * (REQUEST_MSG_LEN - 3)
      for i, word in enumerate(words): arc.write32(req + i * 4, word)
      arc.write32(q + 0, (wptr + 1) % MSG_QUEUE_POINTER_WRAP)

      cfg.addr = Arc.NOC_BASE
      arc.configure(cfg)
      arc.write32(RESET_UNIT_ARC_MISC_CNTL, arc.read32(RESET_UNIT_ARC_MISC_CNTL) | IRQ0_TRIG_BIT)

      cfg.addr = q_base
      arc.configure(cfg)
      rptr = arc.read32(q + 4)
      deadline = time.monotonic() + (timeout_ms / 1000.0)
      while time.monotonic() < deadline:
        resp_wptr = arc.read32(q + 20)
        if resp_wptr != rptr:
          resp = q + HEADER_BYTES + (MSG_QUEUE_SIZE * REQUEST_BYTES) + (rptr % MSG_QUEUE_SIZE) * RESPONSE_BYTES
          out = [arc.read32(resp + i * 4) for i in range(RESPONSE_MSG_LEN)]
          arc.write32(q + 4, (rptr + 1) % MSG_QUEUE_POINTER_WRAP)
          return out
        time.sleep(0.001)
    raise TimeoutError(f"arc_msg timeout ({timeout_ms} ms)")

  def close(self):
    if hasattr(self, "fd"):
      self.arc_msg(Arc.MSG_AICLK_GO_LONG_IDLE, 0, 0, timeout_ms=1000)
    self._close()


class DeviceBase(CommonDevice):
  """Shared infrastructure for both dispatch modes: core planning, kernel packing, payload prep."""

  def __init__(self, device: int = 0, enable_sysmem: bool = False, init_core_plans: bool = True):
    super().__init__(device=device)
    if init_core_plans:
      self._core_plans = self._build_core_plans()
    self._exec_list: list[Program] = []
    self.win = TLBWindow(self.fd, TLBSize.MiB_2)
    self.dram = DramAllocator(fd=self.fd, dram_tiles=self.tiles.dram, device=self, enable_sysmem=enable_sysmem)

  def queue(self, program: Program):
    self._exec_list.append(program)

  def resolve_cores(self, cores: CoreSpec = "all") -> CoreList:
    return list(self._resolve_core_plan(cores).cores)

  def resolve_mcast_rects(self, cores: CoreSpec = "all") -> list[Rect]:
    return list(self._resolve_core_plan(cores).rects)

  # -- Core planning --

  def _build_core_plans(self) -> dict[int | str, _CorePlan]:
    ordered = sorted(self.dispatchable_cores, key=lambda xy: (xy[0], xy[1]))
    if not ordered:
      raise RuntimeError("no dispatchable cores available")
    plans: dict[int | str, _CorePlan] = {}
    for n in range(1, len(ordered) + 1):
      selected = ordered[:n]
      rects = self._prefix_rects(selected)
      plans[n] = _CorePlan(
        cores=selected,
        rects=rects,
        mcast_dests=[self._rect_to_noc_mcast(rect) for rect in rects],
      )
    plans["all"] = plans[len(ordered)]
    return plans

  def _resolve_core_plan(self, spec: CoreSpec) -> _CorePlan:
    if spec == "all":
      return self._core_plans["all"]
    if not isinstance(spec, int):
      raise TypeError("program.cores must be int or 'all'")
    if spec < 1:
      raise ValueError("program.cores must be >= 1")
    if spec > len(self.dispatchable_cores):
      raise ValueError(f"program.cores={spec} exceeds dispatchable cores ({len(self.dispatchable_cores)})")
    return self._core_plans[spec]

  @staticmethod
  def _rect_to_noc_mcast(rect: Rect) -> McastDest:
    x0, x1, y0, y1 = rect
    noc_mcast_xy = (y1 << 18) | (x1 << 12) | (y0 << 6) | x0
    num_dests = (x1 - x0 + 1) * (y1 - y0 + 1)
    return noc_mcast_xy, num_dests

  @staticmethod
  def _core_rects(cores: CoreList) -> list[Rect]:
    remaining = set(cores)
    rects: list[Rect] = []
    while remaining:
      x0, y0 = min(remaining, key=lambda xy: (xy[1], xy[0]))
      x1 = x0
      while (x1 + 1, y0) in remaining:
        x1 += 1
      y1 = y0
      while all((x, y1 + 1) in remaining for x in range(x0, x1 + 1)):
        y1 += 1
      for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
          remaining.remove((x, y))
      rects.append((x0, x1, y0, y1))
    return rects

  @staticmethod
  def _prefix_rects(cores: CoreList) -> list[Rect]:
    if not cores:
      return []
    by_x: dict[int, list[int]] = {}
    for x, y in cores:
      by_x.setdefault(x, []).append(y)
    y0 = min(y for _, y in cores)
    cols = sorted(by_x)
    col_tops: CoreList = []
    for x in cols:
      ys = sorted(by_x[x])
      top = y0 + len(ys) - 1
      if ys != list(range(y0, top + 1)):
        return DeviceBase._core_rects(cores)
      col_tops.append((x, top))
    rects: list[Rect] = []
    i = 0
    while i < len(col_tops):
      x0, y1 = col_tops[i]
      j = i
      while j + 1 < len(col_tops) and col_tops[j + 1][0] == col_tops[j][0] + 1 and col_tops[j + 1][1] == y1:
        j += 1
      rects.append((x0, col_tops[j][0], y0, y1))
      i = j + 1
    return rects

  def _mcast_dests_for_cores(self, cores: CoreList) -> list[McastDest]:
    return [self._rect_to_noc_mcast(rect) for rect in self._core_rects(cores)]

  # -- Kernel packing --

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

  def _pack_kernel_shared(self, program: Program, reader: CompiledKernel, writer: CompiledKernel, rta_sizes: RtaSizes,
                          dispatch_mode: int = DevMsgs.DISPATCH_MODE_HOST, sem_off: int | None = None):
    rta_offsets = [0, rta_sizes[0], rta_sizes[0] + rta_sizes[1]]
    rta_total = align_up(rta_offsets[2] + rta_sizes[2], 16)

    sem_size = program.num_sems * 16
    if sem_off is None:
      sem_off = rta_total
    shared_off_start = align_up(sem_off + sem_size, 16)
    crta_off = shared_off_start

    local_cb_mask, local_cb_blob = self._build_local_cb_blob(program)
    local_cb_off = shared_off_start
    kernel_off = align_up(local_cb_off + len(local_cb_blob), 16)

    kernels = {"brisc": writer, "ncrisc": reader}
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
      off = align_up(off + len(k.xip), 16)
      enables |= 1 << idx

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
    cfg.min_remote_cb_start_index = FAST_CQ_NUM_CIRCULAR_BUFFERS
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
  def _pack_rta(reader_args: list[int], writer_args: list[int], compute_args: list[int], num_sems: int = 0,
                sem_off: int | None = None) -> bytes:
    pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
    rta = pack(writer_args) + pack(reader_args) + pack(compute_args)
    if num_sems > 0:
      if sem_off is not None and sem_off > len(rta):
        rta = rta.ljust(sem_off, b"\0")
      rta += b"\0" * (num_sems * 16)
    return rta

  # -- RTA / payload prep --

  def _resolve_args(self, args: list[int] | ArgGen, core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
    return args if isinstance(args, list) else args(core_idx, core_xy, num_cores)

  def _build_rta(self, program: Program, launch: DataflowLaunch, core_idx: int, core_xy: Core, num_cores: int, role_idx: int,
                 role_cores: int, sem_off: int | None = None) -> tuple[RtaSizes, bytes]:
    reader_args = self._resolve_args(launch.reader_rt_args, role_idx, core_xy, role_cores)
    writer_args = self._resolve_args(launch.writer_rt_args, role_idx, core_xy, role_cores)
    compute_args = self._resolve_args(program.compute_rt_args, core_idx, core_xy, num_cores)
    rta_sizes = (len(writer_args) * 4, len(reader_args) * 4, len(compute_args) * 4)
    rta = self._pack_rta(reader_args, writer_args, compute_args, program.num_sems, sem_off=sem_off)
    return rta_sizes, rta

  def _uniform_sem_off(self, program: Program, launch_roles: list[_LaunchRole]) -> int:
    max_rta_total = 0
    seen = set()
    for role in launch_roles:
      lid = id(role.launch)
      if lid in seen:
        continue
      seen.add(lid)
      rta_sizes, _ = self._build_rta(
        program, role.launch, 0, role.core, len(launch_roles), role.role_idx, role.role_count)
      rta_total = align_up(sum(rta_sizes), 16)
      max_rta_total = max(max_rta_total, rta_total)
    return max_rta_total

  def _core_launches(self, program: Program, cores: CoreList) -> list[_LaunchRole]:
    core_set = set(cores)
    assigned: dict[Core, tuple[DataflowLaunch, int, int]] = {}
    for launch in program.dataflow:
      role_cores = [c for c in launch.cores if c in core_set]
      role_n = len(role_cores)
      for role_i, core in enumerate(role_cores):
        if core in assigned:
          raise ValueError(f"core {core} appears in multiple dataflow launches")
        assigned[core] = (launch, role_i, role_n)
    missing = [c for c in cores if c not in assigned]
    if missing:
      raise ValueError(f"every program core must have a dataflow launch; missing {len(missing)} cores")
    return [
      _LaunchRole(core=core, launch=launch, role_idx=role_i, role_count=role_n)
      for core in cores
      for launch, role_i, role_n in [assigned[core]]
    ]

  def _prepare_core_payloads(self, program: Program, cores: CoreList, launch_roles: list[_LaunchRole],
                             dispatch_mode: int) -> list[_CorePayload]:
    num_cores = len(cores)
    sem_off = self._uniform_sem_off(program, launch_roles) if program.num_sems > 0 else None
    payloads: list[_CorePayload] = []
    shared_cache: dict[tuple, tuple[int, bytes, bytes]] = {}
    for core_idx, role in enumerate(launch_roles):
      rta_sizes, rta = self._build_rta(
        program, role.launch, core_idx, role.core, num_cores, role.role_idx, role.role_count, sem_off=sem_off)
      key = (role.launch.reader, role.launch.writer, *rta_sizes, dispatch_mode)
      if key not in shared_cache:
        shared_cache[key] = self._pack_kernel_shared(
          program, reader=role.launch.reader, writer=role.launch.writer,
          rta_sizes=rta_sizes, dispatch_mode=dispatch_mode, sem_off=sem_off,
        )[:3]
      shared_off, shared_img, launch_blob = shared_cache[key]
      payloads.append(_CorePayload(
        core=role.core,
        rta=rta,
        launch_blob=launch_blob,
        shared_addr=TensixL1.KERNEL_CONFIG_BASE + shared_off,
        shared_blob=shared_img,
      ))
    return payloads

  # -- Grouping --

  @staticmethod
  def _group_by(items: list[tuple]) -> dict:
    groups: dict = {}
    for item in items:
      core, *rest = item
      key = rest[0] if len(rest) == 1 else tuple(rest)
      groups.setdefault(key, []).append(core)
    return groups


from device_slow import SlowDevice
from device_fast import FastDevice

__all__ = ["Program", "DataflowLaunch", "TileGrid", "ArgGen", "SlowDevice", "FastDevice", "Device"]

class Device:
  def __new__(cls, device: int = 0):
    if USE_USB_DISPATCH:
      return SlowDevice(device=device)
    try:
      return FastDevice(device=device)
    except RuntimeError as exc:
      if "missing fast-dispatch firmware" not in str(exc): raise
      print(f"[blackhole-py] {exc}; falling back to slow dispatch")
      return SlowDevice(device=device)
