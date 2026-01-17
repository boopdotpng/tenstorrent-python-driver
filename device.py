from __future__ import annotations
import ctypes, fcntl, mmap, os, time
from dataclasses import dataclass
from typing import ClassVar
from abi import *
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import _IO, align_down, contiguous_ranges, dbg, find_dev_by_bdf, format_bdf, ioctl, load_pt_load, trace_ioctl
from configs import Arc, Dram, HARVESTING_NOC_LOCATIONS, TensixL1, TensixMMIO
from pathlib import Path
from dram import DramAllocator

@dataclass
class Harvesting:
  tensix_cols: tuple[int, int]  # exactly 2 disabled tensix columns
  dram_bank: int  # single disabled DRAM bank
  all_eth_disabled: bool  # true if all ethernet cores are disabled (p100a)

  def __repr__(self) -> str:
    return f"harvesting(tensix={self.tensix_cols}, dram={self.dram_bank}, eth={'disabled' if self.all_eth_disabled else 'enabled'})"

@dataclass
class TileGrid:
  # these are all NoC 0 coordinates by default
  # assuming origin is top left
  # on NoC 1, origin is bottom right.
  ARC: ClassVar[tuple[int, int]] = (8, 0) # ARC tile (same location on both boards)
  TENSIX_Y: ClassVar[tuple[int, int]] = (2, 11)
  MAX_X: ClassVar[int] = 16
  MAX_Y: ClassVar[int] = 11
  tensix: list[tuple[int, int]] # all valid tensix (x, y) for unicast (noc0)
  tensix_mcast: list[tuple[int, int]] # multicast x-ranges: [(x0, x1), ...] (y is always 2-11)
  dram: list[tuple[int, int, int]] # (bank_id, x, y)

  @classmethod
  def from_harvesting(cls, harvesting: Harvesting) -> TileGrid:
    dram_cols, l2cpu_col = (0, 9), 8
    disabled = set(harvesting.tensix_cols) | set(dram_cols) | {l2cpu_col}
    tensix_cols = [x for x in range(cls.MAX_X + 1) if x not in disabled]

    y0, y1 = cls.TENSIX_Y
    tensix = [(x, y) for x in tensix_cols for y in range(y0, y1 + 1)]
    tensix_mcast = contiguous_ranges(tensix_cols)

    # dram tiles in bank order (skip the single harvested bank)
    dram = []
    for bank in range(Dram.BANK_COUNT):
      if bank == harvesting.dram_bank: continue
      dram.extend((bank, Dram.BANK_X[bank], y) for y in Dram.BANK_TILE_YS[bank])

    return cls(tensix=tensix, tensix_mcast=tensix_mcast, dram=dram)

class Device:
  def __init__(self, path: str = "/dev/tenstorrent/0"):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self._setup()
    self.harvesting = self.get_harvesting()
    dbg(2, "dev", f"harvesting {self.harvesting}")
    self.tiles = TileGrid.from_harvesting(self.harvesting)
    dbg(2, "tiles", f"tensix={len(self.tiles.tensix)} dram={len(self.tiles.dram)} mcast={self.tiles.tensix_mcast}")

    self.upload_firmware()

    self.dram = DramAllocator(fd=self.fd, dram_tiles=self.tiles.dram)
  
  def upload_kernels(self):
    # Load elfs for all the cores 
    # perform the XIP transformation (execute in place) 
    # write kernels to all the tiles requested by the kernel
    # setup CB config 
    kern_dir = Path(__file__).parent / "kernels" / self.arch
    for name in os.listdir(kern_dir):
      pass
  
  def run(self):
    # send launch_msg_t to start the compute 
    # read go_msg_t.signal until it becomes done
    pass
  
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
        if not s.data: continue
        # PT_LOADs in 0x0... are normal L1 SRAM writes.
        # PT_LOADs in 0xFFB0.... are RISCV local-mem initializers; match tt-metal:
        # stage them into per-core L1 init scratch so firmware/loader can copy later.
        addr = s.paddr
        if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
          addr = init + (addr - TensixMMIO.LOCAL_RAM_START)
          assert 0 <= addr < TensixL1.SIZE
        else:
          assert 0 <= addr < TensixL1.SIZE, f"{name}: unexpected paddr 0x{addr:x}"
        spans.append((addr, s.data))

      dbg(2, "fw", f"core={name.removesuffix('.elf')} base=0x{base:x} init=0x{init:x} spans={len(spans)} bytes={sum(len(d) for _, d in spans)}")
      return spans

    staged = {name: stage_spans(name, segs) for name, segs in fws}

    cfg = TLBConfig(addr=reg_base, noc=0, mcast=True, mode=TLBMode.STRICT)
    y0, y1 = self.tiles.TENSIX_Y
    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      for x0, x1 in self.tiles.tensix_mcast:
        # Multicast to a rectangle: all tensix cols in [x0,x1] and rows y=2..11
        dbg(2, "fw", f"mcast x=[{x0},{x1}] y=[{y0},{y1}]")
        cfg.start, cfg.end = (x0, y0), (x1, y1)
        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_ALL)  # hold cores in reset

        cfg.mode = TLBMode.ORDERED_BULK
        for name, spans in staged.items():
          for addr, data in spans:
            dbg(3, "fw", f"seg core={name.removesuffix('.elf')} addr=0x{addr:x} bytes={len(data)}")
            win.write(addr, data, restore=False)

        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, 0x0)  # release reset, cores start executing

  def get_harvesting(self) -> Harvesting:
    tlb_config = TLBConfig(
      addr = Arc.NOC_BASE,
      start = TileGrid.ARC,
      end = TileGrid.ARC,
      noc = 0,
      mcast = False,
      mode = TLBMode.STRICT
    )

    with TLBWindow(self.fd, TLBSize.MiB_2, tlb_config) as arc:
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

      tensix_enabled = read_tag(Arc.TAG_TENSIX_ENABLED, Arc.DEFAULT_TENSIX_ENABLED)
      eth_enabled = read_tag(Arc.TAG_ETH_ENABLED, Arc.DEFAULT_ETH_ENABLED)
      gddr_enabled = read_tag(Arc.TAG_GDDR_ENABLED, Arc.DEFAULT_GDDR_ENABLED)

      # HARVESTING_NOC_LOCATIONS maps bit positions to column IDs (alternating left/right)
      # A 0-bit means that column is harvested (disabled)
      tensix_off = sorted(loc for pos, loc in enumerate(HARVESTING_NOC_LOCATIONS) if ((tensix_enabled >> pos) & 1) == 0)
      dram_off = [bank for bank in range(Dram.BANK_COUNT) if ((gddr_enabled >> bank) & 1) == 0]

      assert len(tensix_off) == 2, f"expected 2 harvested tensix cols, got {tensix_off}"
      assert len(dram_off) == 1, f"expected 1 harvested dram bank, got {dram_off}"

    return Harvesting(
      tensix_cols=(tensix_off[0], tensix_off[1]),
      dram_bank=dram_off[0],
      all_eth_disabled=(eth_enabled & Arc.DEFAULT_ETH_ENABLED) == 0,
    )

  def _setup(self, retried: bool = False):
    self.arch = self._get_arch()
    if self.arch not in ("p100a", "p150b"):
      if retried:
        os.close(self.fd)
        raise SystemExit("device still in bad state after reset")
      confirm = input("only blackhole is supported. alternatively, the card might be in a bad state. reset y/n: ")
      if confirm.lower() == 'y':
        self.reset(dmc_reset=True)
        return
      os.close(self.fd)
      raise SystemExit("exiting")

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
