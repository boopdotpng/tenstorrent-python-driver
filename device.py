from __future__ import annotations
import ctypes, fcntl, mmap, os, struct, time
from dataclasses import dataclass, field
from typing import ClassVar
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import (
  _IO,
  align_down,
  find_dev_by_bdf,
  format_bdf,
  generate_jal_instruction,
  ioctl,
  load_pt_load,
)
from pathlib import Path
from dram import DramAllocator
from codegen import CompiledKernel

@dataclass
class Program:
  reader: CompiledKernel | None = None
  writer: CompiledKernel | None = None
  compute: list[CompiledKernel] = field(
    default_factory=list
  )  # [trisc0, trisc1, trisc2]
  reader_rt_args: list[int] = field(default_factory=list)
  writer_rt_args: list[int] = field(default_factory=list)
  compute_rt_args: list[int] = field(default_factory=list)
  cbs: list[int] = field(default_factory=lambda: [0, 16])
  tile_size: int = 2048
  num_pages: int = 2

@dataclass
class TileGrid:
  ARC: ClassVar[tuple[int, int]] = (8, 0)
  TENSIX_Y: ClassVar[tuple[int, int]] = (2, 11)
  TENSIX_X_P100A: ClassVar[tuple[int, ...]] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    10,
    11,
    12,
    13,
    14,
  )
  tensix: list[tuple[int, int]]
  tensix_mcast: list[tuple[int, int]]
  dram: list[tuple[int, int, int]]

  @classmethod
  def p100a(cls, harvested_dram_bank: int) -> TileGrid:
    tensix_cols = list(cls.TENSIX_X_P100A)
    y0, y1 = cls.TENSIX_Y
    tensix = [(x, y) for x in tensix_cols for y in range(y0, y1 + 1)]
    tensix_mcast = [(1, 7), (10, 14)]
    dram = []
    for bank in range(Dram.BANK_COUNT):
      if bank == harvested_dram_bank:
        continue
      dram.extend((bank, Dram.BANK_X[bank], y) for y in Dram.BANK_TILE_YS[bank])
    return cls(tensix=tensix, tensix_mcast=tensix_mcast, dram=dram)

class Device:
  def __init__(
    self, path: str = "/dev/tenstorrent/0", *, upload_firmware: bool = True
  ):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self._setup()
    self._assert_arc_booted()
    self.harvested_dram = self.get_harvested_dram_bank()
    self.tiles = TileGrid.p100a(self.harvested_dram)
    ref_tile = (1, 2) if (1, 2) in self.tiles.tensix else self.tiles.tensix[0]
    self.noc_translation_enabled = self.get_tile_noc_translation_enabled(ref_tile)
    if upload_firmware:
      self.upload_firmware()
    self.dram = DramAllocator(fd=self.fd, dram_tiles=self.tiles.dram)

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

  def _pack_kernel_config(self, program: Program):
    pack = lambda xs: b"".join(
      int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs
    )
    align16 = lambda n: (n + 15) & ~15

    brisc_rta = pack(program.writer_rt_args)
    ncrisc_rta = pack(program.reader_rt_args)
    trisc_rta = pack(program.compute_rt_args)
    rta_offsets = [0, len(brisc_rta), len(brisc_rta) + len(ncrisc_rta)]
    rta_total = align16(rta_offsets[2] + len(trisc_rta))
    crta_off = rta_total

    local_cb_mask, local_cb_blob = self._build_local_cb_blob(program)
    local_cb_off = rta_total
    kernel_off = align16(local_cb_off + len(local_cb_blob))

    kernels = {"brisc": program.writer, "ncrisc": program.reader}
    for i, k in enumerate(program.compute):
      kernels[f"trisc{i}"] = k
    proc = [
      ("brisc", 0),
      ("ncrisc", 1),
      ("trisc0", 2),
      ("trisc1", 3),
      ("trisc2", 4),
    ]
    kernel_text_off = [0] * len(proc)
    enables = 0
    off = kernel_off
    for name, idx in proc:
      if (k := kernels.get(name)) is None:
        continue
      kernel_text_off[idx] = off
      off = align16(off + len(k.xip))
      enables |= 1 << idx

    img = bytearray(off)
    img[0 : len(brisc_rta)] = brisc_rta
    img[rta_offsets[1] : rta_offsets[1] + len(ncrisc_rta)] = ncrisc_rta
    img[rta_offsets[2] : rta_offsets[2] + len(trisc_rta)] = trisc_rta
    img[local_cb_off : local_cb_off + len(local_cb_blob)] = local_cb_blob
    for name, idx in proc:
      if (k := kernels.get(name)) is None:
        continue
      dst = kernel_text_off[idx]
      img[dst : dst + len(k.xip)] = k.xip

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

    cfg.rta_offset[0].rta_offset, cfg.rta_offset[0].crta_offset = (
      rta_offsets[0],
      crta_off,
    )
    cfg.rta_offset[1].rta_offset, cfg.rta_offset[1].crta_offset = (
      rta_offsets[1],
      crta_off,
    )
    for i in (2, 3, 4):
      cfg.rta_offset[i].rta_offset, cfg.rta_offset[i].crta_offset = (
        rta_offsets[2],
        crta_off,
      )
    for i, v in enumerate(kernel_text_off):
      cfg.kernel_text_offset[i] = v

    return bytes(img), cfg

  def run(self, program: Program):
    img, kc = self._pack_kernel_config(program)
    cores = self.tiles.tensix

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
        self._set_tile_noc_translation_enabled(
          win, mmio_cfg, (x, y), 1, self.noc_translation_enabled[1]
        )
        l1_cfg.start = l1_cfg.end = (x, y)
        win.configure(l1_cfg)
        win.write(TensixL1.KERNEL_CONFIG_BASE, img, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, reset_blob, use_uc=True, restore=False)
        win.write(
          TensixL1.GO_MSG_INDEX,
          (0).to_bytes(4, "little"),
          use_uc=True,
          restore=False,
        )
        win.write(TensixL1.LAUNCH, launch_blob, use_uc=True, restore=False)
        win.write(TensixL1.GO_MSG, go_blob, use_uc=True, restore=False)

      for x, y in cores:
        l1_cfg.start = l1_cfg.end = (x, y)
        win.configure(l1_cfg)
        deadline = time.perf_counter() + 10.0
        while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
          if time.perf_counter() > deadline:
            raise TimeoutError(f"timeout waiting for core ({x}, {y})")
          time.sleep(0.001)

  def upload_firmware(self):
    fw_dir = Path(__file__).parent / "riscv-firmware" / self.arch
    names = ("brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf")
    paths = [fw_dir / n for n in names]
    fws = [(p.name, load_pt_load(p)) for p in paths]

    reg_base, reg_off = align_down(
      TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2
    )

    fw_map = {
      "brisc.elf": (
        TensixL1.BRISC_FIRMWARE_BASE,
        TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH,
      ),
      "ncrisc.elf": (
        TensixL1.NCRISC_FIRMWARE_BASE,
        TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH,
      ),
      "trisc0.elf": (
        TensixL1.TRISC0_BASE,
        TensixL1.TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
      ),
      "trisc1.elf": (
        TensixL1.TRISC1_BASE,
        TensixL1.TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
      ),
      "trisc2.elf": (
        TensixL1.TRISC2_BASE,
        TensixL1.TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
      ),
    }

    def stage_spans(name: str, segs) -> list[tuple[int, bytes]]:
      base, init = fw_map[name]
      assert any(s.paddr == base for s in segs), (
        f"{name}: missing text base 0x{base:x}"
      )
      spans = []
      for s in segs:
        if not s.data and s.memsz == 0:
          continue
        data = (
          s.data
          if s.memsz <= len(s.data)
          else s.data + b"\0" * (s.memsz - len(s.data))
        )
        addr = s.paddr
        if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
          addr = init + (addr - TensixMMIO.LOCAL_RAM_START)
          assert 0 <= addr < TensixL1.SIZE
        else:
          assert 0 <= addr < TensixL1.SIZE, (
            f"{name}: unexpected paddr 0x{addr:x}"
          )
        spans.append((addr, data))
      return spans

    staged = {name: stage_spans(name, segs) for name, segs in fws}

    cfg = TLBConfig(addr=reg_base, noc=0, mcast=True, mode=TLBMode.STRICT)
    y0, y1 = self.tiles.TENSIX_Y
    jal_insn = generate_jal_instruction(TensixL1.BRISC_FIRMWARE_BASE)
    go_msg = struct.pack("<BBBB", 0, 0, 0, DevMsgs.RUN_MSG_INIT)

    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_ALL)

        cfg.mode = TLBMode.ORDERED_BULK
        for name, spans in staged.items():
          for addr, data in spans:
            win.write(addr, data, use_uc=True, restore=False)

        win.write(
          0x0, jal_insn.to_bytes(4, "little"), use_uc=True, restore=False
        )
        win.write(TensixL1.GO_MSG, go_msg, use_uc=True, restore=False)

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

      test_tile = self.tiles.tensix[0]
      cfg.start, cfg.end = test_tile, test_tile
      cfg.addr, cfg.mode = 0, TLBMode.STRICT
      cfg.mcast = False
      win.configure(cfg)
      cfg.mcast = True

      bank_tables = self._build_bank_noc_tables()
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = 0, TLBMode.ORDERED_BULK
        win.configure(cfg)
        win.write(
          TensixL1.MEM_BANK_TO_NOC_SCRATCH,
          bank_tables,
          use_uc=True,
          restore=False,
        )

      cfg.mcast = True
      for x0, x1 in self.tiles.tensix_mcast:
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.readi32(reg_off)
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_BRISC_ONLY_RUN)

    self._wait_firmware_ready()

  def _wait_firmware_ready(self):
    tile = (1, 2) if (1, 2) in self.tiles.tensix else self.tiles.tensix[0]
    cfg = TLBConfig(
      addr=0, start=tile, end=tile, noc=0, mcast=False, mode=TLBMode.STRICT
    )
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as win:
      deadline = time.perf_counter() + 2.0
      while True:
        go = win.uc[TensixL1.GO_MSG + 3]
        sync = win.readi32(TensixL1.MAILBOX_BASE + 8)
        dm1, tr0, tr1, tr2 = (
          sync & 0xFF,
          (sync >> 8) & 0xFF,
          (sync >> 16) & 0xFF,
          (sync >> 24) & 0xFF,
        )
        if (
          go == DevMsgs.RUN_MSG_DONE
          and dm1 == 0
          and tr1 == 0
          and tr2 == 0
          and tr0 in (0, 3)
        ):
          return
        if time.perf_counter() > deadline:
          raise TimeoutError(
            f"firmware not ready on tile {tile}: go={go:#x} sync={sync:#x}"
          )
        time.sleep(0.001)

  def _build_bank_noc_tables(self) -> bytes:
    NUM_NOCS, NUM_DRAM_BANKS, NUM_L1_BANKS = 2, 7, 110
    GRID_X, GRID_Y = 17, 12

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

    def dram_translated_map(
      harvested_bank: int | None,
    ) -> dict[tuple[int, int], tuple[int, int]]:
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
      if not (0 <= harvested_bank < TOTAL_BANKS):
        raise ValueError("harvested_dram_bank out of range")
      half = TOTAL_BANKS // 2
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
        map_banks(
          mirror_east_bank,
          mirror_east_bank + 1,
          START_X,
          START_Y + (half - 1) * PORTS,
        )
      else:
        mirror_west_bank = harvested_bank - half
        map_banks(0, mirror_west_bank, START_X)
        map_banks(
          mirror_west_bank + 1,
          half,
          START_X,
          START_Y + mirror_west_bank * PORTS,
        )
        map_banks(
          mirror_west_bank,
          mirror_west_bank + 1,
          START_X,
          START_Y + (half - 1) * PORTS,
        )
        map_banks(half, TOTAL_BANKS - 1, START_X + 1)
      return m

    def noc_coord(noc: int, grid_size: int, coord: int) -> int:
      if noc == 0 or self.noc_translation_enabled.get(noc, True):
        return coord
      return grid_size - 1 - coord

    def pack_xy(x: int, y: int) -> int:
      return ((y << 6) | x) & 0xFFFF

    physical_channels = [c for c in range(8) if c != self.harvested_dram]
    assert len(physical_channels) == NUM_DRAM_BANKS
    dram_translated = dram_translated_map(self.harvested_dram)

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

    tensix_cols = list(TileGrid.TENSIX_X_P100A)
    l1_xy = []
    for noc in range(NUM_NOCS):
      for bank_id in range(NUM_L1_BANKS):
        col_idx = bank_id % len(tensix_cols)
        row_idx = bank_id // len(tensix_cols)
        raw_x = tensix_cols[col_idx]
        raw_y = 2 + (row_idx % 10)
        x = noc_coord(noc, GRID_X, raw_x)
        y = noc_coord(noc, GRID_Y, raw_y)
        l1_xy.append(pack_xy(x, y))

    dram_offsets = [0] * NUM_DRAM_BANKS
    l1_offsets = [0] * NUM_L1_BANKS

    blob = struct.pack(f"<{len(dram_xy)}H", *dram_xy)
    blob += struct.pack(f"<{len(l1_xy)}H", *l1_xy)
    blob += struct.pack(f"<{len(dram_offsets)}i", *dram_offsets)
    blob += struct.pack(f"<{len(l1_offsets)}i", *l1_offsets)
    return blob

  def _read_arc_boot_status(self) -> int:
    cfg = TLBConfig(
      addr=Arc.NOC_BASE,
      start=TileGrid.ARC,
      end=TileGrid.ARC,
      noc=0,
      mcast=False,
      mode=TLBMode.STRICT,
    )
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
        f"ARC not booted (SCRATCH_RAM_2=0x{status:08x}, expected (status&0x7)==0x5)"
      )

  def get_harvested_dram_bank(self) -> int:
    tlb_config = TLBConfig(
      addr=Arc.NOC_BASE,
      start=TileGrid.ARC,
      end=TileGrid.ARC,
      noc=0,
      mcast=False,
      mode=TLBMode.STRICT,
    )
    with TLBWindow(self.fd, TLBSize.MiB_2, tlb_config) as arc:
      telem_struct_addr = arc.readi32(Arc.SCRATCH_RAM_13)
      if telem_struct_addr == 0 or not (
        Arc.CSM_START <= telem_struct_addr <= Arc.CSM_END
      ):
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
    dram_off = [
      bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0
    ]
    assert len(dram_off) == 1, f"expected 1 harvested dram bank, got {dram_off}"
    return dram_off[0]

  def _setup(self, retried: bool = False):
    self.arch = self._get_arch()
    if self.arch != "p100a":
      os.close(self.fd)
      raise SystemExit(
        f"unsupported blackhole device {self.arch}; p100a only for now"
      )
    self._map_bars()

  def _close(self):
    if hasattr(self, "mm0"):
      self.mm0.close()
    if hasattr(self, "mm1"):
      self.mm1.close()
    if hasattr(self, "dram"):
      self.dram.close()
    os.close(self.fd)

  def get_bdf(self) -> str:
    info = ioctl(
      self.fd,
      IOCTL_GET_DEVICE_INFO,
      TenstorrentGetDeviceInfoIn,
      TenstorrentGetDeviceInfoOut,
      output_size_bytes=ctypes.sizeof(TenstorrentGetDeviceInfoOut),
    )
    return format_bdf(info.pci_domain, info.bus_dev_fn)

  def _map_bars(self):
    in_sz = ctypes.sizeof(QueryMappingsIn)
    out_sz = ctypes.sizeof(TenstorrentMapping)
    buf = bytearray(in_sz + 6 * out_sz)
    QueryMappingsIn.from_buffer(buf).output_mapping_count = 6
    fcntl.ioctl(self.fd, _IO(IOCTL_QUERY_MAPPINGS), buf, True)
    bars = list((TenstorrentMapping * 6).from_buffer(buf, in_sz))
    self.mm0 = mmap.mmap(
      self.fd,
      bars[0].mapping_size,
      flags=mmap.MAP_SHARED,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
      offset=bars[0].mapping_base,
    )
    self.mm1 = mmap.mmap(
      self.fd,
      bars[2].mapping_size,
      flags=mmap.MAP_SHARED,
      prot=mmap.PROT_READ | mmap.PROT_WRITE,
      offset=bars[2].mapping_base,
    )

  def arc_msg(
    self,
    msg: int,
    arg0: int = 0,
    arg1: int = 0,
    *,
    queue: int = 0,
    timeout_ms: int = 1000,
  ) -> list[int]:
    MSG_QUEUE_SIZE = 4
    MSG_QUEUE_POINTER_WRAP = 2 * MSG_QUEUE_SIZE
    REQUEST_MSG_LEN = 8
    RESPONSE_MSG_LEN = 8
    HEADER_BYTES = 8 * 4
    REQUEST_BYTES = REQUEST_MSG_LEN * 4
    RESPONSE_BYTES = RESPONSE_MSG_LEN * 4
    QUEUE_STRIDE = (
      HEADER_BYTES
      + (MSG_QUEUE_SIZE * REQUEST_BYTES)
      + (MSG_QUEUE_SIZE * RESPONSE_BYTES)
    )
    RESET_UNIT_ARC_MISC_CNTL = Arc.RESET_UNIT_OFFSET + 0x100
    IRQ0_TRIG_BIT = 1 << 16
    if queue < 0 or queue >= 4:
      raise ValueError("queue must be 0..3")

    cfg = TLBConfig(
      addr=Arc.NOC_BASE,
      start=TileGrid.ARC,
      end=TileGrid.ARC,
      noc=0,
      mcast=False,
      mode=TLBMode.STRICT,
    )
    with TLBWindow(self.fd, TLBSize.MiB_2, cfg) as arc:
      info_ptr = arc.readi32(Arc.SCRATCH_RAM_11)
      if info_ptr == 0:
        raise RuntimeError("msgqueue not initialized (SCRATCH_RAM_11 == 0)")

      info_base, info_off = align_down(info_ptr, TLBSize.MiB_2)
      cfg.addr = info_base
      arc.configure(cfg)
      queues_ptr = arc.readi32(info_off)

      q_base, q_off = align_down(queues_ptr, TLBSize.MiB_2)
      cfg.addr = q_base
      arc.configure(cfg)
      q = q_off + queue * QUEUE_STRIDE

      wptr = arc.readi32(q + 0)
      req = q + HEADER_BYTES + (wptr % MSG_QUEUE_SIZE) * REQUEST_BYTES
      words = [msg & 0xFF, arg0 & 0xFFFFFFFF, arg1 & 0xFFFFFFFF] + [0] * (
        REQUEST_MSG_LEN - 3
      )
      for i, word in enumerate(words):
        arc.writei32(req + i * 4, word)
      arc.writei32(q + 0, (wptr + 1) % MSG_QUEUE_POINTER_WRAP)

      cfg.addr = Arc.NOC_BASE
      arc.configure(cfg)
      arc.writei32(
        RESET_UNIT_ARC_MISC_CNTL,
        arc.readi32(RESET_UNIT_ARC_MISC_CNTL) | IRQ0_TRIG_BIT,
      )

      cfg.addr = q_base
      arc.configure(cfg)
      rptr = arc.readi32(q + 4)
      deadline = time.monotonic() + (timeout_ms / 1000.0)
      while time.monotonic() < deadline:
        resp_wptr = arc.readi32(q + 20)
        if resp_wptr != rptr:
          resp = (
            q
            + HEADER_BYTES
            + (MSG_QUEUE_SIZE * REQUEST_BYTES)
            + (rptr % MSG_QUEUE_SIZE) * RESPONSE_BYTES
          )
          out = [arc.readi32(resp + i * 4) for i in range(RESPONSE_MSG_LEN)]
          arc.writei32(q + 4, (rptr + 1) % MSG_QUEUE_POINTER_WRAP)
          return out
        time.sleep(0.001)
    raise TimeoutError(f"arc_msg timeout ({timeout_ms} ms)")

  def get_tile_noc_translation_enabled(
    self, tile: tuple[int, int]
  ) -> dict[int, bool]:
    base, _ = align_down(TensixMMIO.LOCAL_RAM_START, TLBSize.MiB_2)
    cfg = TLBConfig(
      addr=base, start=tile, end=tile, noc=0, mcast=False, mode=TLBMode.STRICT
    )
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
    if noc not in (0, 1):
      raise ValueError("noc must be 0 or 1")
    base, _ = align_down(TensixMMIO.LOCAL_RAM_START, TLBSize.MiB_2)
    cfg.addr, cfg.start, cfg.end, cfg.mcast, cfg.mode = (
      base,
      tile,
      tile,
      False,
      TLBMode.STRICT,
    )
    win.configure(cfg)
    reg = (
      TensixMMIO.NOC0_NIU_START if noc == 0 else TensixMMIO.NOC1_NIU_START
    ) + 0x100
    off = reg - base
    bit = 1 << NocNIU.NIU_CFG_0_NOC_ID_TRANSLATE_EN
    prev = win.readi32(off)
    nextv = (prev | bit) if enable else (prev & ~bit)
    if nextv != prev:
      win.writei32(off, nextv)
      win.readi32(off)

  def reset(self, dmc_reset: bool = False) -> int:
    bdf = self.get_bdf()
    in_sz, out_sz = ctypes.sizeof(ResetDeviceIn), ctypes.sizeof(ResetDeviceOut)
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes = out_sz
    view.flags = (
      TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET
      if dmc_reset
      else TENSTORRENT_RESET_DEVICE_ASIC_RESET
    )
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    self._close()
    for _ in range(50):
      time.sleep(0.2)
      if path := find_dev_by_bdf(bdf):
        self.path = path
        break
    else:
      raise RuntimeError(f"device {bdf} didn't come back after reset")
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_POST_RESET
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    result = ResetDeviceOut.from_buffer(buf, in_sz).result
    self._setup(retried=True)
    return result

  def _get_arch(self):
    ordinal = self.path.split("/")[-1]
    return (
      Path(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type")
      .read_text()
      .strip()
    )

  def close(self):
    self._close()
