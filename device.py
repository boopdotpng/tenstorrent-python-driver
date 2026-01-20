from __future__ import annotations
import ctypes, fcntl, mmap, os, struct, time
from dataclasses import dataclass
from typing import ClassVar
from abi import *
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import _IO, align_down, dbg, find_dev_by_bdf, format_bdf, generate_jal_instruction, ioctl, load_pt_load, trace_ioctl
from configs import Arc, Dram, TensixL1, TensixMMIO
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
  def __init__(self, path: str = "/dev/tenstorrent/0", *, upload_firmware: bool = True):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self._setup()
    self._assert_arc_booted()
    self.harvested_dram = self.get_harvested_dram_bank()
    dbg(2, "dev", f"harvested_dram={self.harvested_dram}")
    self.tiles = TileGrid.p100a(self.harvested_dram)
    dbg(2, "tiles", f"tensix={len(self.tiles.tensix)} dram={len(self.tiles.dram)} mcast={self.tiles.tensix_mcast}")

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

  def _pack_kernel_config(self, kernels: dict[str, CompiledKernel], rt_args: dict[str, list[int]]):
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
    cfg.brisc_noc_id = 0
    cfg.brisc_noc_mode = 0
    cfg.mode = DevMsgs.DISPATCH_MODE_HOST

    cfg.rta_offset[0].rta_offset, cfg.rta_offset[0].crta_offset = rta_offsets[0], crta_off
    cfg.rta_offset[1].rta_offset, cfg.rta_offset[1].crta_offset = rta_offsets[1], crta_off
    for i in (2, 3, 4):
      cfg.rta_offset[i].rta_offset, cfg.rta_offset[i].crta_offset = rta_offsets[2], crta_off
    for i, v in enumerate(kernel_text_off): cfg.kernel_text_offset[i] = v

    return bytes(img), cfg

  def run(self, *, cores: list[tuple[int, int]], kernels: dict[str, CompiledKernel], rt_args: dict[str, list[int]]):
    img, kc = self._pack_kernel_config(kernels, rt_args)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    launch = LaunchMsg()
    launch.kernel_config = kc
    launch_blob = as_bytes(launch)
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go_blob = as_bytes(go)

    cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      for x, y in cores:
        cfg.start = cfg.end = (x, y)
        win.configure(cfg)
        win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, reset_blob, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little"), use_uc=True, restore=False)
        win.write(TensixL1.LAUNCH, launch_blob, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, go_blob, use_uc=True, restore=False)

      for x, y in cores:
        cfg.start = cfg.end = (x, y)
        win.configure(cfg)
        deadline = time.perf_counter() + 10.0
        while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
          if time.perf_counter() > deadline: raise TimeoutError(f"timeout waiting for core {(x, y)}")
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

    dbg(1, "fw", f"upload tiles={len(self.tiles.tensix)} mcast_ranges={len(self.tiles.tensix_mcast)} cores={len(fw_map)}")

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

      dbg(2, "fw", f"core={name.removesuffix('.elf')} base=0x{base:x} init=0x{init:x} spans={len(spans)} bytes={sum(len(d) for _, d in spans)}")
      return spans

    staged = {name: stage_spans(name, segs) for name, segs in fws}

    cfg = TLBConfig(addr=reg_base, noc=0, mcast=True, mode=TLBMode.STRICT)
    y0, y1 = self.tiles.TENSIX_Y
    jal_insn = generate_jal_instruction(TensixL1.BRISC_FIRMWARE_BASE)
    go_msg = struct.pack("<BBBB", 0, 0, 0, DevMsgs.RUN_MSG_INIT)

    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      # Phase 1: Write firmware to ALL tiles (hold all in reset)
      for x0, x1 in self.tiles.tensix_mcast:
        dbg(2, "fw", f"mcast x=[{x0},{x1}] y=[{y0},{y1}]")
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_ALL)  # hold cores in reset

        cfg.mode = TLBMode.ORDERED_BULK
        for name, spans in staged.items():
          for addr, data in spans:
            dbg(3, "fw", f"seg core={name.removesuffix('.elf')} addr=0x{addr:x} bytes={len(data)}")
            win.write(addr, data, use_uc=True, restore=False)

        # Write JAL instruction at address 0 for BRISC bootstrap
        dbg(2, "fw", f"JAL at 0x0 -> 0x{TensixL1.BRISC_FIRMWARE_BASE:x} (insn=0x{jal_insn:08x})")
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
        dbg(2, "fw", f"reset PCs written")

      # Verify firmware was written to first tile before releasing
      test_tile = self.tiles.tensix[0]
      cfg.start, cfg.end = test_tile, test_tile
      cfg.addr, cfg.mode = 0, TLBMode.STRICT
      cfg.mcast = False
      win.configure(cfg)
      v_jal = win.readi32(0x0)
      v_go = win.readi32(TensixL1.GO_MSG)
      v_fw = win.readi32(TensixL1.BRISC_FIRMWARE_BASE)
      dbg(2, "fw", f"verify tile {test_tile}: JAL=0x{v_jal:08x} GO=0x{v_go:08x} FW=0x{v_fw:08x}")
      cfg.mcast = True

      # Write bank-to-NoC tables to scratch area (firmware reads these during init)
      bank_tables = self._build_bank_noc_tables()
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = 0, TLBMode.ORDERED_BULK
        win.configure(cfg)
        win.write(TensixL1.MEM_BANK_TO_NOC_SCRATCH, bank_tables, use_uc=True, restore=False)
      dbg(2, "fw", f"bank tables written to 0x{TensixL1.MEM_BANK_TO_NOC_SCRATCH:x}")

      # Phase 2: Release BRISC on ALL tiles (after all firmware is written)
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.readi32(reg_off)  # flush posted writes
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)
        dbg(2, "fw", f"released BRISC x=[{x0},{x1}]")

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

    # DRAM coordinates: use subchannel 0 for all banks (matches DramAllocator)
    # All subchannels access the same 4GB DRAM, subchannel 0 is first in BANK_TILE_YS
    # Bank 0-3 at x=0, Bank 4-7 at x=9
    DRAM_COORDS = {
      0: (0, 0), 1: (0, 2), 2: (0, 4), 3: (0, 5),
      4: (9, 0), 5: (9, 2), 6: (9, 4), 7: (9, 5),
    }  # Matches Dram.BANK_TILE_YS[bank][0] and Dram.BANK_X[bank]

    def noc_coord(noc: int, grid_size: int, coord: int) -> int:
      """NoC coordinate transformation.

      On Blackhole with COORDINATE_VIRTUALIZATION_ENABLED, the NOC hardware
      handles coordinate translation. Since pure-py doesn't set up the full
      translation tables like tt-metal does, use raw physical coordinates
      for both NoCs.
      """
      # TODO: If this doesn't work, may need to set up NOC translation tables
      # or match tt-metal's virtualization logic more closely
      return coord  # Use raw coordinates for both NoCs

    def pack_xy(x: int, y: int) -> int:
      """Pack (x, y) into 16-bit NoC XY encoding: (y << 6) | x"""
      return ((y << 6) | x) & 0xFFFF

    # Build list of unharvested channels (bank_id -> channel mapping)
    channels = [c for c in range(8) if c != self.harvested_dram]
    assert len(channels) == NUM_DRAM_BANKS, f"expected {NUM_DRAM_BANKS} banks, got {len(channels)}"

    # dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] - stored as [noc0_banks..., noc1_banks...]
    dram_xy = []
    for noc in range(NUM_NOCS):
      for ch in channels:
        raw_x, raw_y = DRAM_COORDS[ch]
        x = noc_coord(noc, GRID_X, raw_x)
        y = noc_coord(noc, GRID_Y, raw_y)
        dram_xy.append(pack_xy(x, y))

    # l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS]
    # Simplified mapping: distribute banks across tensix cores in row-major order
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

    dbg(2, "noc", f"bank tables: dram_xy={len(dram_xy)*2}B l1_xy={len(l1_xy)*2}B "
                  f"dram_off={len(dram_offsets)*4}B l1_off={len(l1_offsets)*4}B total={len(blob)}B")
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

    dbg(1, "dev", f"open arch={self.arch} bdf={self.get_bdf()} path={self.path}")
    self._map_bars()

  def _close(self):
    if hasattr(self, 'mm0'): self.mm0.close()
    if hasattr(self, 'mm1'): self.mm1.close()
    if hasattr(self, 'dram'): self.dram.close()
    os.close(self.fd)

  def get_bdf(self) -> str:
    trace_ioctl(IOCTL_GET_DEVICE_INFO)
    info = ioctl(self.fd, IOCTL_GET_DEVICE_INFO, TenstorrentGetDeviceInfoIn,
                 TenstorrentGetDeviceInfoOut, output_size_bytes=ctypes.sizeof(TenstorrentGetDeviceInfoOut))
    return format_bdf(info.pci_domain, info.bus_dev_fn)

  def _map_bars(self):
    in_sz = ctypes.sizeof(QueryMappingsIn)
    out_sz = ctypes.sizeof(TenstorrentMapping)
    buf = bytearray(in_sz + 6 * out_sz)
    QueryMappingsIn.from_buffer(buf).output_mapping_count = 6
    trace_ioctl(IOCTL_QUERY_MAPPINGS)
    fcntl.ioctl(self.fd, _IO(IOCTL_QUERY_MAPPINGS), buf, True)
    bars = list((TenstorrentMapping * 6).from_buffer(buf, in_sz))

    # UC bars for bar0 and bar1 are 0,2 (the others are WC which is bad for reading/writing registers)
    # we don't need to mmap global vram (4+5), that is done through the dram tiles and the NoC
    self.mm0 = mmap.mmap(self.fd, bars[0].mapping_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bars[0].mapping_base)
    self.mm1 = mmap.mmap(self.fd, bars[2].mapping_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bars[2].mapping_base)
    dbg(3, "mmap",
        f"bar0 base={bars[0].mapping_base:#x} size={bars[0].mapping_size:#x} "
        f"bar2 base={bars[2].mapping_base:#x} size={bars[2].mapping_size:#x}")

  def reset(self, dmc_reset: bool = False) -> int:
    bdf = self.get_bdf()
    dbg(1, "dev", f"reset bdf={bdf} flags={'ASIC_DMC_RESET' if dmc_reset else 'ASIC_RESET'}")
    in_sz, out_sz = ctypes.sizeof(ResetDeviceIn), ctypes.sizeof(ResetDeviceOut)

    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes = out_sz
    view.flags = TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET if dmc_reset else TENSTORRENT_RESET_DEVICE_ASIC_RESET
    trace_ioctl(IOCTL_RESET_DEVICE, f"flags={'ASIC_DMC_RESET' if dmc_reset else 'ASIC_RESET'}")
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    self._close()

    # poll for device to come back (up to 10s)
    dbg(2, "dev", "reset waiting for device...")
    for _ in range(50):
      time.sleep(0.2)
      if (path := find_dev_by_bdf(bdf)):
        self.path = path
        break
    else:
      raise RuntimeError(f"device {bdf} didn't come back after reset")

    dbg(2, "dev", f"reset device back path={self.path}")
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    # POST_RESET reinits hardware
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_POST_RESET
    trace_ioctl(IOCTL_RESET_DEVICE, "flags=POST_RESET")
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    result = ResetDeviceOut.from_buffer(buf, in_sz).result
    dbg(1, "dev", f"reset complete result={result}")

    self._setup(retried=True)
    return result

  def _get_arch(self):
    ordinal = self.path.split('/')[-1]
    return Path(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type").read_text().strip()

  def close(self): self._close()
