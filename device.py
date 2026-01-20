from __future__ import annotations
import ctypes, fcntl, mmap, os, struct, time
from dataclasses import dataclass
from typing import ClassVar
from abi import *
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import _IO, align_down, find_dev_by_bdf, format_bdf, generate_jal_instruction, ioctl, load_pt_load
from configs import Arc, Dram, NocNIU, TensixL1, TensixMMIO
from pathlib import Path
from dram import DramAllocator

@dataclass
class TileGrid:
  # these are all NoC 0 coordinates by default
  # assuming origin is top left
  # on NoC 1, origin is bottom right.
  ARC: ClassVar[tuple[int, int]] = (8, 0) # ARC tile (same location on both boards)
  TENSIX_Y: ClassVar[tuple[int, int]] = (2, 11)
  TENSIX_X_P100A: ClassVar[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14)  # BH w/ 2 harvested cols
  tensix: list[tuple[int, int]] # all valid tensix (x, y) for unicast (noc0)
  tensix_mcast: list[tuple[int, int]] # multicast x-ranges: [(x0, x1), ...] (y is always 2-11)
  dram: list[tuple[int, int, int]] # (bank_id, x, y)

  @classmethod
  def p100a(cls, harvested_dram_bank: int) -> TileGrid:
    tensix_cols = list(cls.TENSIX_X_P100A)
    y0, y1 = cls.TENSIX_Y
    tensix = [(x, y) for x in tensix_cols for y in range(y0, y1 + 1)]
    tensix_mcast = [(1, 7), (10, 14)]

    # dram tiles in bank order (skip the single harvested bank)
    dram = []
    for bank in range(Dram.BANK_COUNT):
      if bank == harvested_dram_bank: continue
      dram.extend((bank, Dram.BANK_X[bank], y) for y in Dram.BANK_TILE_YS[bank])

    return cls(tensix=tensix, tensix_mcast=tensix_mcast, dram=dram)

class Device:
  def __init__(
    self,
    path: str = "/dev/tenstorrent/0",
    *,
    upload_firmware: bool = True,
    noc_translation_enabled: bool | dict[int, bool] | None = None,
  ):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self._setup()
    self._assert_arc_booted()
    self.harvested_dram = self.get_harvested_dram_bank()
    self.tiles = TileGrid.p100a(self.harvested_dram)
    ref_tile = (1, 2) if (1, 2) in self.tiles.tensix else self.tiles.tensix[0]
    detected = self.get_tile_noc_translation_enabled(ref_tile)
    desired = detected.copy()
    if isinstance(noc_translation_enabled, bool):
      desired = {0: noc_translation_enabled, 1: noc_translation_enabled}
    elif isinstance(noc_translation_enabled, dict):
      for noc in (0, 1):
        if noc in noc_translation_enabled:
          desired[noc] = bool(noc_translation_enabled[noc])
    self.noc_translation_enabled = desired

    if upload_firmware: self.upload_firmware()

    self.dram = DramAllocator(fd=self.fd, dram_tiles=self.tiles.dram)

  def _build_local_cb_blob(self, configs: dict[int, LocalCBConfig], mask: int) -> bytes:
    end = mask.bit_length()
    arr = (LocalCBConfig * end)()
    for i, cfg in configs.items(): arr[i] = cfg
    return as_bytes(arr)

  def _build_remote_cb_blob(self, configs: dict[int, RemoteCBConfig], start: int) -> bytes:
    count = CB.NUM_CIRCULAR_BUFFERS - start
    if count <= 0: return b""
    arr = (RemoteCBConfig * count)()
    for i, cb_id in enumerate(range(CB.NUM_CIRCULAR_BUFFERS - 1, start - 1, -1)):
      arr[i] = configs.get(cb_id, RemoteCBConfig())
    return as_bytes(arr)

  def _cb_blobs(self) -> tuple[int, bytes, int, bytes]:
    cb0_addr = TensixL1.DATA_BUFFER_SPACE_BASE + 0x0
    cb16_addr = TensixL1.DATA_BUFFER_SPACE_BASE + 0x2000

    local_mask = (1 << 0) | (1 << 16)
    local_cfg = {
      0: LocalCBConfig(addr_bytes=cb0_addr, size_bytes=4096, num_pages=2, page_size_bytes=2048),
      16: LocalCBConfig(addr_bytes=cb16_addr, size_bytes=4096, num_pages=2, page_size_bytes=2048),
    }
    local_blob = self._build_local_cb_blob(local_cfg, local_mask)

    min_remote_start = 32
    remote_blob = self._build_remote_cb_blob({}, min_remote_start)
    return local_mask, local_blob, min_remote_start, remote_blob

  def _pack_kernel_config(
    self,
    kernels: dict[str, CompiledKernel],
    rt_args: dict[str, list[int]],
    *,
    brisc_noc_id: int,
  ):
    pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
    align16 = lambda n: (n + 15) & ~15

    brisc_rta, ncrisc_rta, trisc_rta = pack(rt_args.get("brisc", [])), pack(rt_args.get("ncrisc", [])), pack(rt_args.get("trisc", []))
    rta_offsets = [0, len(brisc_rta), len(brisc_rta) + len(ncrisc_rta)]
    rta_total = align16(rta_offsets[2] + len(trisc_rta))
    crta_off = rta_total

    local_cb_mask, local_cb_blob, min_remote_start, remote_cb_blob = self._cb_blobs()
    local_cb_off = rta_total
    remote_cb_off = local_cb_off + len(local_cb_blob)
    kernel_off = align16(remote_cb_off + len(remote_cb_blob))

    proc = [("brisc", 0), ("ncrisc", 1), ("trisc0", 2), ("trisc1", 3), ("trisc2", 4)]
    kernel_text_off = [0] * len(proc)
    enables = 0
    off = kernel_off
    for name, idx in proc:
      if (k := kernels.get(name)) is None: continue
      kernel_text_off[idx] = off
      off = align16(off + len(k.xip))
      enables |= 1 << idx

    img = bytearray(off)
    img[0:len(brisc_rta)] = brisc_rta
    img[rta_offsets[1]:rta_offsets[1] + len(ncrisc_rta)] = ncrisc_rta
    img[rta_offsets[2]:rta_offsets[2] + len(trisc_rta)] = trisc_rta
    img[local_cb_off:local_cb_off + len(local_cb_blob)] = local_cb_blob
    if remote_cb_blob: img[remote_cb_off:remote_cb_off + len(remote_cb_blob)] = remote_cb_blob
    for name, idx in proc:
      if (k := kernels.get(name)) is None: continue
      dst = kernel_text_off[idx]
      img[dst:dst + len(k.xip)] = k.xip

    cfg = LaunchMsg().kernel_config
    cfg.kernel_config_base[0] = TensixL1.KERNEL_CONFIG_BASE
    cfg.kernel_config_base[1] = TensixL1.KERNEL_CONFIG_BASE
    cfg.kernel_config_base[2] = TensixL1.KERNEL_CONFIG_BASE
    cfg.local_cb_offset = local_cb_off
    cfg.remote_cb_offset = remote_cb_off
    cfg.local_cb_mask = local_cb_mask
    cfg.min_remote_cb_start_index = min_remote_start
    cfg.enables = enables
    cfg.brisc_noc_id = brisc_noc_id
    cfg.brisc_noc_mode = 0
    cfg.mode = DevMsgs.DISPATCH_MODE_HOST

    cfg.rta_offset[0].rta_offset, cfg.rta_offset[0].crta_offset = rta_offsets[0], crta_off
    cfg.rta_offset[1].rta_offset, cfg.rta_offset[1].crta_offset = rta_offsets[1], crta_off
    for i in (2, 3, 4):
      cfg.rta_offset[i].rta_offset, cfg.rta_offset[i].crta_offset = rta_offsets[2], crta_off
    for i, v in enumerate(kernel_text_off): cfg.kernel_text_offset[i] = v

    return bytes(img), cfg

  def run(
    self,
    *,
    cores: list[tuple[int, int]],
    kernels: dict[str, CompiledKernel],
    rt_args: dict[str, list[int]],
    brisc_noc_id: int = 0,
  ):
    if brisc_noc_id not in (0, 1): raise ValueError("brisc_noc_id must be 0 or 1")
    img, kc = self._pack_kernel_config(kernels, rt_args, brisc_noc_id=brisc_noc_id)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    launch = LaunchMsg()
    launch.kernel_config = kc
    launch_blob = as_bytes(launch)
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go_blob = as_bytes(go)

    l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    mmio_base, _ = align_down(TensixMMIO.LOCAL_RAM_START, TLBSize.MiB_2)
    mmio_cfg = TLBConfig(addr=mmio_base, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      for x, y in cores:
        self._set_tile_noc_translation_enabled(win, mmio_cfg, (x, y), 1, self.noc_translation_enabled[1])
        l1_cfg.start = l1_cfg.end = (x, y)
        win.configure(l1_cfg)
        win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, reset_blob, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little"), use_uc=True, restore=False)
        win.write(TensixL1.LAUNCH, launch_blob, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, go_blob, use_uc=True, restore=False)

      for x, y in cores:
        l1_cfg.start = l1_cfg.end = (x, y)
        win.configure(l1_cfg)
        deadline = time.perf_counter() + 10.0
        while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
          if time.perf_counter() > deadline:
            tile = (x, y)
            go_u32 = win.readi32(TensixL1.GO_MSG)
            sync = win.readi32(TensixL1.MAILBOX_BASE + 8)
            dbg_words = [win.readi32(TensixL1.LLK_DEBUG_BASE + i * 4) for i in range(8)]
            self._set_tile_noc_translation_enabled(win, mmio_cfg, tile, 1, self.noc_translation_enabled[1])
            base, _ = align_down(TensixMMIO.LOCAL_RAM_START, TLBSize.MiB_2)
            mmio_cfg.start = mmio_cfg.end = tile
            win.configure(mmio_cfg)
            noc0_cfg0 = win.readi32((TensixMMIO.NOC0_NIU_START + 0x100) - base)
            noc1_cfg0 = win.readi32((TensixMMIO.NOC1_NIU_START + 0x100) - base)
            dm1, tr0, tr1, tr2 = sync & 0xFF, (sync >> 8) & 0xFF, (sync >> 16) & 0xFF, (sync >> 24) & 0xFF
            def status(noc_base: int, reg: int) -> int: return win.readi32((noc_base + 0x200 + reg * 4) - base)
            n0 = {
              "rd_sent": status(TensixMMIO.NOC0_NIU_START, 0x5),
              "rd_resp": status(TensixMMIO.NOC0_NIU_START, 0x2),
              "np_wr_sent": status(TensixMMIO.NOC0_NIU_START, 0xA),
              "wr_ack": status(TensixMMIO.NOC0_NIU_START, 0x1),
              "p_wr_sent": status(TensixMMIO.NOC0_NIU_START, 0xB),
            }
            n1 = {
              "rd_sent": status(TensixMMIO.NOC1_NIU_START, 0x5),
              "rd_resp": status(TensixMMIO.NOC1_NIU_START, 0x2),
              "np_wr_sent": status(TensixMMIO.NOC1_NIU_START, 0xA),
              "wr_ack": status(TensixMMIO.NOC1_NIU_START, 0x1),
              "p_wr_sent": status(TensixMMIO.NOC1_NIU_START, 0xB),
            }
            def stream_reg(stream_id: int, reg_id: int) -> int:
              return win.readi32((0xFFB40000 + stream_id * 0x1000 + reg_id * 4) - base)
            # CB sync uses stream scratch regs:
            # - tiles_received: STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX (10)
            # - tiles_acked: STREAM_REMOTE_DEST_BUF_START_REG_INDEX (8)
            # Operand->stream_id: 8 + operand. So operand0->8, operand16->24.
            cb_regs = {
              "cb0": {"received": stream_reg(8, 10), "acked": stream_reg(8, 8)},
              "cb16": {"received": stream_reg(24, 10), "acked": stream_reg(24, 8)},
            }
            raise TimeoutError(
              f"timeout waiting for core {tile}: "
              f"go=0x{go_u32:08x} sync=0x{sync:08x} "
              f"sync_bytes=[dm1=0x{dm1:02x},tr0=0x{tr0:02x},tr1=0x{tr1:02x},tr2=0x{tr2:02x}] "
              f"dbg={[f'0x{v:08x}' for v in dbg_words]} "
              f"NIU_CFG_0[noc0]=0x{noc0_cfg0:08x} NIU_CFG_0[noc1]=0x{noc1_cfg0:08x} "
              f"status0={n0} status1={n1} cb={cb_regs}"
            )
          time.sleep(0.001)

  # upload firmware to risc-v cores inside tensix tiles (required every fresh boot)
  # if tt-metal runs, it will erase the firmware
  def upload_firmware(self):
    fw_dir = Path(__file__).parent / "riscv-firmware" / self.arch
    names = ("brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf")
    paths = [fw_dir / n for n in names]
    fws = [(p.name, load_pt_load(p)) for p in paths]

    # Tile-local MMIO soft-reset register (not in L1 SRAM). We map a TLB window to it to
    # hold all RISCs in reset while we write firmware into L1.
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)

    fw_map = {
      "brisc.elf":  (TensixL1.BRISC_FIRMWARE_BASE,  TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH),
      "ncrisc.elf": (TensixL1.NCRISC_FIRMWARE_BASE, TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc0.elf": (TensixL1.TRISC0_BASE,          TensixL1.TRISC0_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc1.elf": (TensixL1.TRISC1_BASE,          TensixL1.TRISC1_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc2.elf": (TensixL1.TRISC2_BASE,          TensixL1.TRISC2_INIT_LOCAL_L1_BASE_SCRATCH),
    }

    def stage_spans(name: str, segs) -> list[tuple[int, bytes]]:
      base, init = fw_map[name]
      assert any(s.paddr == base for s in segs), f"{name}: missing text base 0x{base:x}"

      spans = []
      for s in segs:
        if not s.data and s.memsz == 0: continue
        data = s.data if s.memsz <= len(s.data) else s.data + b"\0" * (s.memsz - len(s.data))
        # PT_LOADs in 0x0... are normal L1 SRAM writes.
        # PT_LOADs in 0xFFB0.... are RISCV local-mem initializers; match tt-metal:
        # stage them into per-core L1 init scratch so firmware/loader can copy later.
        addr = s.paddr
        if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
          addr = init + (addr - TensixMMIO.LOCAL_RAM_START)
          assert 0 <= addr < TensixL1.SIZE
        else:
          assert 0 <= addr < TensixL1.SIZE, f"{name}: unexpected paddr 0x{addr:x}"
        spans.append((addr, data))
      return spans

    staged = {name: stage_spans(name, segs) for name, segs in fws}

    cfg = TLBConfig(addr=reg_base, noc=0, mcast=True, mode=TLBMode.STRICT)
    y0, y1 = self.tiles.TENSIX_Y
    jal_insn = generate_jal_instruction(TensixL1.BRISC_FIRMWARE_BASE)
    go_msg = struct.pack("<BBBB", 0, 0, 0, DevMsgs.RUN_MSG_INIT)

    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      # Phase 1: Write firmware to ALL tiles (hold all in reset)
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_ALL)  # hold cores in reset

        cfg.mode = TLBMode.ORDERED_BULK
        for name, spans in staged.items():
          for addr, data in spans:
            win.write(addr, data, use_uc=True, restore=False)

        # Write JAL instruction at address 0 for BRISC bootstrap
        win.write(0x0, jal_insn.to_bytes(4, "little"), use_uc=True, restore=False)

        # Initialize go_msg with signal = RUN_MSG_INIT
        win.write(TensixL1.GO_MSG, go_msg, use_uc=True, restore=False)

        # Write reset PC registers for TRISC and NCRISC
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        trisc0_pc_off = TensixMMIO.RISCV_DEBUG_REG_TRISC0_RESET_PC - reg_base
        trisc1_pc_off = TensixMMIO.RISCV_DEBUG_REG_TRISC1_RESET_PC - reg_base
        trisc2_pc_off = TensixMMIO.RISCV_DEBUG_REG_TRISC2_RESET_PC - reg_base
        ncrisc_pc_off = TensixMMIO.RISCV_DEBUG_REG_NCRISC_RESET_PC - reg_base
        win.writei32(trisc0_pc_off, TensixL1.TRISC0_BASE)
        win.writei32(trisc1_pc_off, TensixL1.TRISC1_BASE)
        win.writei32(trisc2_pc_off, TensixL1.TRISC2_BASE)
        win.writei32(ncrisc_pc_off, TensixL1.NCRISC_FIRMWARE_BASE)

      # Verify firmware was written to first tile before releasing
      test_tile = self.tiles.tensix[0]
      cfg.start, cfg.end = test_tile, test_tile
      cfg.addr, cfg.mode = 0, TLBMode.STRICT
      cfg.mcast = False
      win.configure(cfg)
      cfg.mcast = True

      # Write bank-to-NoC tables to scratch area (firmware reads these during init)
      bank_tables = self._build_bank_noc_tables()
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = 0, TLBMode.ORDERED_BULK
        win.configure(cfg)
        win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, bank_tables, use_uc=True, restore=False)

      cfg.mcast = True

      # Phase 2: Release BRISC on ALL tiles (after all firmware is written)
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.readi32(reg_off)  # flush posted writes
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)

    self._wait_firmware_ready()

  def _wait_firmware_ready(self):
    tile = (1, 2) if (1, 2) in self.tiles.tensix else self.tiles.tensix[0]
    cfg = TLBConfig(addr=0, start=tile, end=tile, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      deadline = time.perf_counter() + 2.0
      while True:
        go = win.uc[TensixL1.GO_MSG + 3]
        sync = win.readi32(TensixL1.MAILBOX_BASE + 8)
        dm1, tr0, tr1, tr2 = sync & 0xFF, (sync >> 8) & 0xFF, (sync >> 16) & 0xFF, (sync >> 24) & 0xFF
        if go == DevMsgs.RUN_MSG_DONE and dm1 == 0 and tr1 == 0 and tr2 == 0 and tr0 in (0, 3): return
        if time.perf_counter() > deadline: raise TimeoutError(f"firmware not ready on tile {tile}: go={go:#x} sync={sync:#x}")
        time.sleep(0.001)

  def _build_bank_noc_tables(self) -> bytes:
    """Build the bank-to-NoC mapping tables that firmware copies to local memory.

    Layout in MEM_BANK_TO_NOC_SCRATCH:
      dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS]  (uint16_t)
      l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS]      (uint16_t)
      bank_to_dram_offset[NUM_DRAM_BANKS]            (int32_t)
      bank_to_l1_offset[NUM_L1_BANKS]                (int32_t)
    """
    NUM_NOCS, NUM_DRAM_BANKS, NUM_L1_BANKS = 2, 7, 110
    GRID_X, GRID_Y = 17, 12

    # DRAM core locations in physical NOC0 coordinates.
    # This matches tt-umd `blackhole::DRAM_CORES_NOC0` and `blackhole_140_arch.yaml`.
    DRAM_PHYS_NOC0 = {
      0: [(0, 0), (0, 1), (0, 11)],
      1: [(0, 2), (0, 10), (0, 3)],
      2: [(0, 9), (0, 4), (0, 8)],
      3: [(0, 5), (0, 7), (0, 6)],
      4: [(9, 0), (9, 1), (9, 11)],
      5: [(9, 2), (9, 10), (9, 3)],
      6: [(9, 9), (9, 4), (9, 8)],
      7: [(9, 5), (9, 7), (9, 6)],
    }
    # DRAM view routing endpoints from `blackhole_140_arch.yaml` (dram_views[].worker_endpoint).
    #
    # Important: this table is indexed by the *logical* DRAM bank id (0..7), not the physical
    # DRAM channel id. With one harvested bank, logical bank ids are compacted and map onto the
    # remaining physical channels in order.
    WORKER_EP_LOGICAL = {
      0: [2, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1],
      4: [2, 1], 5: [2, 1], 6: [2, 1], 7: [2, 1],
    }

    def dram_translated_map_for_harvested_bank(harvested_bank: int | None) -> dict[tuple[int, int], tuple[int, int]]:
      """Return (logical_bank, port) -> translated (x, y) for DRAM cores.

      This matches tt-umd's `BlackholeCoordinateManager::fill_dram_noc0_translated_mapping` using:
      - blackhole::dram_translated_coordinate_start_x = 17
      - blackhole::dram_translated_coordinate_start_y = 12
      - blackhole::NUM_NOC_PORTS_PER_DRAM_BANK = 3
      - blackhole::NUM_DRAM_BANKS = 8
      """
      START_X, START_Y, PORTS, TOTAL_BANKS = 17, 12, 3, 8
      m: dict[tuple[int, int], tuple[int, int]] = {}
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

      if not (0 <= harvested_bank < TOTAL_BANKS): raise ValueError("harvested_dram_bank out of range")
      half = TOTAL_BANKS // 2
      # When a bank is harvested, tt-umd exposes 7 logical banks (0..6) and reserves the last virtual slot.
      # The mapping below lays out those 7 banks into translated DRAM space.
      if harvested_bank < half:
        mirror_east_bank = harvested_bank + half - 1
        map_banks(0, half - 1, START_X + 1)
        map_banks(half - 1, mirror_east_bank, START_X)
        map_banks(
          mirror_east_bank + 1,
          TOTAL_BANKS - 1,
          START_X,
          START_Y + (mirror_east_bank - (half - 1)) * PORTS,
        )
        map_banks(mirror_east_bank, mirror_east_bank + 1, START_X, START_Y + (half - 1) * PORTS)
      else:
        mirror_west_bank = harvested_bank - half
        map_banks(0, mirror_west_bank, START_X)
        map_banks(
          mirror_west_bank + 1,
          half,
          START_X,
          START_Y + mirror_west_bank * PORTS,
        )
        map_banks(mirror_west_bank, mirror_west_bank + 1, START_X, START_Y + (half - 1) * PORTS)
        map_banks(half, TOTAL_BANKS - 1, START_X + 1)
      return m

    def noc_coord(noc: int, grid_size: int, coord: int) -> int:
      if noc == 0 or self.noc_translation_enabled.get(noc, True): return coord
      return grid_size - 1 - coord

    def pack_xy(x: int, y: int) -> int:
      """Pack (x, y) into 16-bit NoC XY encoding: (y << 6) | x"""
      return ((y << 6) | x) & 0xFFFF

    # Map logical DRAM bank ids (0..6) onto physical channels (0..7, skipping the harvested one).
    physical_channels = [c for c in range(8) if c != self.harvested_dram]
    assert len(physical_channels) == NUM_DRAM_BANKS, f"expected {NUM_DRAM_BANKS} banks, got {len(physical_channels)}"
    dram_translated = dram_translated_map_for_harvested_bank(self.harvested_dram)

    # dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] - stored as [noc0_banks..., noc1_banks...]
    dram_xy = []
    for noc in range(NUM_NOCS):
      for logical_bank, phys_ch in enumerate(physical_channels):
        port = WORKER_EP_LOGICAL[logical_bank][noc]
        if self.noc_translation_enabled.get(noc, True):
          x, y = dram_translated[(logical_bank, port)]
        else:
          raw_x, raw_y = DRAM_PHYS_NOC0[phys_ch][port]
          x = noc_coord(noc, GRID_X, raw_x)
          y = noc_coord(noc, GRID_Y, raw_y)
        dram_xy.append(pack_xy(x, y))

    # l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS]
    # Simplified mapping: distribute banks across tensix cores in row-major order
    # Apply NOC1 mirroring for physical coordinates
    tensix_cols = list(TileGrid.TENSIX_X_P100A)
    l1_xy = []
    for noc in range(NUM_NOCS):
      for bank_id in range(NUM_L1_BANKS):
        col_idx = bank_id % len(tensix_cols)
        row_idx = bank_id // len(tensix_cols)
        raw_x = tensix_cols[col_idx]
        raw_y = 2 + (row_idx % 10)  # tensix rows 2-11
        x = noc_coord(noc, GRID_X, raw_x)
        y = noc_coord(noc, GRID_Y, raw_y)
        l1_xy.append(pack_xy(x, y))

    # bank_to_dram_offset[NUM_DRAM_BANKS] - all zeros for simple interleaving
    dram_offsets = [0] * NUM_DRAM_BANKS

    # bank_to_l1_offset[NUM_L1_BANKS] - all zeros (no per-bank offset needed)
    l1_offsets = [0] * NUM_L1_BANKS

    # Pack into bytes (little-endian)
    blob = struct.pack(f"<{len(dram_xy)}H", *dram_xy)
    blob += struct.pack(f"<{len(l1_xy)}H", *l1_xy)
    blob += struct.pack(f"<{len(dram_offsets)}i", *dram_offsets)
    blob += struct.pack(f"<{len(l1_offsets)}i", *l1_offsets)

    return blob

  def _read_arc_boot_status(self) -> int:
    cfg = TLBConfig(addr=Arc.NOC_BASE, start=TileGrid.ARC, end=TileGrid.ARC, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as arc:
      return arc.readi32(Arc.SCRATCH_RAM_2)

  def _assert_arc_booted(self, timeout_s: float = 5.0):
    deadline = time.perf_counter() + timeout_s
    status = self._read_arc_boot_status()
    while (status & 0x7) != 0x5 and time.perf_counter() < deadline:
      time.sleep(0.001)
      status = self._read_arc_boot_status()
    if (status & 0x7) != 0x5:
      self._close()
      raise RuntimeError(
        f"ARC not booted (SCRATCH_RAM_2=0x{status:08x}, expected (status&0x7)==0x5); "
        "tt-metal will time out waiting for ARC too"
      )

  def get_harvested_dram_bank(self) -> int:
    gddr_enabled = None
    tlb_config = TLBConfig(
      addr = Arc.NOC_BASE,
      start = TileGrid.ARC,
      end = TileGrid.ARC,
      noc = 0,
      mcast = False,
      mode = TLBMode.STRICT
    )

    with TLBWindow(self.fd, TLBSize.MiB_2, tlb_config) as arc:
      if gddr_enabled is None:
        telem_struct_addr = arc.readi32(Arc.SCRATCH_RAM_13)
        if telem_struct_addr == 0 or (not (Arc.CSM_START <= telem_struct_addr <= Arc.CSM_END)):
          raise RuntimeError("device probably not working, try tt-smi -r")

        csm_base, csm_offset = align_down(telem_struct_addr, TLBSize.MiB_2)

        tlb_config.addr = csm_base
        arc.configure(tlb_config)

        entry_count = arc.readi32(csm_offset + 4)
        tags_base = csm_offset + 8
        data_base = tags_base + entry_count * 4

        tag_to_offset = {}
        for i in range(entry_count):
          tag_offset = arc.readi32(tags_base + i * 4)
          tag_to_offset[tag_offset & 0xFFFF] = (tag_offset >> 16) & 0xFFFF

        def read_tag(tag: int, default: int) -> int:
          off = tag_to_offset.get(tag)
          return default if off is None else arc.readi32(data_base + off * 4)

        gddr_enabled = read_tag(Arc.TAG_GDDR_ENABLED, Arc.DEFAULT_GDDR_ENABLED)

      dram_off = [bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0]
      assert len(dram_off) == 1, f"expected 1 harvested dram bank, got {dram_off}"
      return dram_off[0]

  def _setup(self, retried: bool = False):
    self.arch = self._get_arch()
    if self.arch != "p100a":
      os.close(self.fd)
      raise SystemExit(f"unsupported blackhole device {self.arch}; p100a only for now")

    self._map_bars()

  def _close(self):
    if hasattr(self, 'mm0'): self.mm0.close()
    if hasattr(self, 'mm1'): self.mm1.close()
    if hasattr(self, 'dram'): self.dram.close()
    os.close(self.fd)

  def get_bdf(self) -> str:
    info = ioctl(self.fd, IOCTL_GET_DEVICE_INFO, TenstorrentGetDeviceInfoIn,
                 TenstorrentGetDeviceInfoOut, output_size_bytes=ctypes.sizeof(TenstorrentGetDeviceInfoOut))
    return format_bdf(info.pci_domain, info.bus_dev_fn)

  def _map_bars(self):
    in_sz = ctypes.sizeof(QueryMappingsIn)
    out_sz = ctypes.sizeof(TenstorrentMapping)
    buf = bytearray(in_sz + 6 * out_sz)
    QueryMappingsIn.from_buffer(buf).output_mapping_count = 6
    fcntl.ioctl(self.fd, _IO(IOCTL_QUERY_MAPPINGS), buf, True)
    bars = list((TenstorrentMapping * 6).from_buffer(buf, in_sz))

    # UC bars for bar0 and bar1 are 0,2 (the others are WC which is bad for reading/writing registers)
    # we don't need to mmap global vram (4+5), that is done through the dram tiles and the NoC
    self.mm0 = mmap.mmap(self.fd, bars[0].mapping_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bars[0].mapping_base)
    self.mm1 = mmap.mmap(self.fd, bars[2].mapping_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bars[2].mapping_base)

  def bar0_read32(self, addr: int) -> int:
    return struct.unpack_from("<I", self.mm0, addr)[0]

  def bar0_write32(self, addr: int, val: int):
    struct.pack_into("<I", self.mm0, addr, val)

  def get_tile_noc_translation_enabled(self, tile: tuple[int, int]) -> dict[int, bool]:
    """Read NIU_CFG_0 on a Tensix tile for both NOCs."""
    base, _ = align_down(TensixMMIO.LOCAL_RAM_START, TLBSize.MiB_2)
    cfg = TLBConfig(addr=base, start=tile, end=tile, noc=0, mcast=False, mode=TLBMode.STRICT)
    bit = 1 << NocNIU.NIU_CFG_0_NOC_ID_TRANSLATE_EN
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      v0 = win.readi32((TensixMMIO.NOC0_NIU_START + 0x100) - base)
      v1 = win.readi32((TensixMMIO.NOC1_NIU_START + 0x100) - base)
    return {0: (v0 & bit) != 0, 1: (v1 & bit) != 0}

  def _set_tile_noc_translation_enabled(
    self,
    win: TLBWindow,
    cfg: TLBConfig,
    tile: tuple[int, int],
    noc: int,
    enable: bool,
  ):
    if noc not in (0, 1): raise ValueError("noc must be 0 or 1")
    base, _ = align_down(TensixMMIO.LOCAL_RAM_START, TLBSize.MiB_2)
    cfg.addr, cfg.start, cfg.end, cfg.mcast, cfg.mode = base, tile, tile, False, TLBMode.STRICT
    win.configure(cfg)
    reg = (TensixMMIO.NOC0_NIU_START if noc == 0 else TensixMMIO.NOC1_NIU_START) + 0x100
    off = reg - base
    bit = 1 << NocNIU.NIU_CFG_0_NOC_ID_TRANSLATE_EN
    prev = win.readi32(off)
    nextv = (prev | bit) if enable else (prev & ~bit)
    if nextv != prev:
      win.writei32(off, nextv)
      win.readi32(off)  # flush posted write

  def reset(self, dmc_reset: bool = False) -> int:
    bdf = self.get_bdf()
    in_sz, out_sz = ctypes.sizeof(ResetDeviceIn), ctypes.sizeof(ResetDeviceOut)

    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes = out_sz
    view.flags = TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET if dmc_reset else TENSTORRENT_RESET_DEVICE_ASIC_RESET
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    self._close()

    # poll for device to come back (up to 10s)
    for _ in range(50):
      time.sleep(0.2)
      if (path := find_dev_by_bdf(bdf)):
        self.path = path
        break
    else:
      raise RuntimeError(f"device {bdf} didn't come back after reset")

    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    # POST_RESET reinits hardware
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_POST_RESET
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    result = ResetDeviceOut.from_buffer(buf, in_sz).result

    self._setup(retried=True)
    return result

  def _get_arch(self):
    ordinal = self.path.split('/')[-1]
    return Path(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type").read_text().strip()

  def close(self): self._close()
