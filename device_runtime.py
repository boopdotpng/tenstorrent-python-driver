import os, struct, time
from dataclasses import dataclass, field
from typing import ClassVar, Callable, Literal
from pathlib import Path
from defs import *
from codegen import CompiledKernel, compile_firmware
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import align_down, generate_jal_instruction, noc_xy

CoreSpec = int | Literal["all"]
BankPort = tuple[int, int]
ArgGen = Callable[[int, Core, int], list[int]]

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

  def synchronize(self, timeout_s: float = 10.0):
    self.sync(timeout_s=timeout_s)

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
