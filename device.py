from __future__ import annotations
import ctypes, fcntl, mmap, os, time
from dataclasses import dataclass, field
from typing import ClassVar
from autogen import *
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import DEBUG, _IO, align_down, find_dev_by_bdf, format_bdf, load_pt_load, trace_ioctl
from configs import Arc, Dram, HARVESTING_NOC_LOCATIONS, TensixL1, TensixMMIO
from pathlib import Path

@dataclass
class Harvesting:
  tensix_cols: tuple[int, ...]   # disabled tensix column x-coords
  dram_banks: tuple[int, ...]    # disabled DRAM bank indices (0-7)
  all_eth_disabled: bool         # True if all ethernet cores are disabled (p100a)
  pcie: tuple[int, ...]          # disabled PCIe instances (0 or 1)

  def __repr__(self) -> str:
    return (
      f"Harvesting(tensix_cols={self.tensix_cols}, dram_banks={self.dram_banks}, "
      f"eth={'disabled' if self.all_eth_disabled else 'enabled'}, pcie={self.pcie})"
    )

@dataclass
class TileGrid:
  ARC: ClassVar[tuple[int, int]] = (8, 0)     # ARC tile on both boards
  TENSIX_Y: ClassVar[tuple[int, int]] = (2, 11)
  tensix: list[tuple[int, int]]               # all valid tensix (x, y) for unicast
  tensix_mcast: list[tuple[int, int]]         # multicast x-ranges: [(x0, x1), ...] (y is always TENSIX_Y)
  dram: list[tuple[int, int, int]]            # (bank_id, x, y) in bank order
  _dram_by_bank: dict[int, list[tuple[int, int]]] = field(default_factory=dict, repr=False)

  @property
  def dram_by_bank(self) -> dict[int, list[tuple[int, int]]]:
    if not self._dram_by_bank:
      for bank, x, y in self.dram:
        self._dram_by_bank.setdefault(bank, []).append((x, y))
    return self._dram_by_bank

  @classmethod
  def from_harvesting(cls, harvesting: Harvesting) -> TileGrid:
    dram_cols, l2cpu_col = (0, 9), 8
    max_x = 16  # blackhole physical X is always 0..16

    # valid tensix columns (exclude dram, l2cpu, and harvested)
    disabled = set(harvesting.tensix_cols)
    tensix_cols = [x for x in range(max_x + 1) if x not in dram_cols and x != l2cpu_col and x not in disabled]

    # unicast: all valid (x, y) pairs
    y0, y1 = cls.TENSIX_Y
    tensix = [(x, y) for x in tensix_cols for y in range(y0, y1 + 1)]

    # multicast: contiguous x-ranges
    tensix_mcast = []
    if tensix_cols:
      start = prev = tensix_cols[0]
      for x in tensix_cols[1:]:
        if x != prev + 1:
          tensix_mcast.append((start, prev))
          start = x
        prev = x
      tensix_mcast.append((start, prev))

    # dram tiles in bank order
    dram = []
    for bank in range(Dram.BANK_COUNT):
      if bank in harvesting.dram_banks: continue
      col = Dram.BANK_X[bank]
      dram.extend((bank, col, y) for y in Dram.BANK_TILE_YS[bank])

    return cls(tensix=tensix, tensix_mcast=tensix_mcast, dram=dram)

class Device:
  def __init__(self, path: str = "/dev/tenstorrent/0"):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self._setup()
    self.harvesting = self.get_harvesting()
    if DEBUG >= 2: print(self.harvesting)
    self.tiles = TileGrid.from_harvesting(self.harvesting)
    if DEBUG >= 2: print(f"tensix tiles: {len(self.tiles.tensix)}")

    self.upload_firmware()
  
  def _load_firmware_elfs(self):
    fw_dir = Path(__file__).parent / "riscv-firmware" / self.arch
    names = ("brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf")
    paths = [fw_dir / n for n in names]
    for p in paths: assert p.is_file(), f"missing firmware ELF: {p}"
    return [(p.name, load_pt_load(p)) for p in paths]

  # upload firmware to risc-v cores inside tensix tiles (required every fresh boot)
  def upload_firmware(self):
    fw = self._load_firmware_elfs()
    reg_base, reg_off = align_down(TensixMMIO.RISCV_DEBUG_REG_SOFT_RESET_0, TLBSize.MiB_2)

    # (expected_base, local_init_scratch) for each core
    fw_map = {
      "brisc.elf":  (TensixL1.BRISC_FIRMWARE_BASE,  TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH),
      "ncrisc.elf": (TensixL1.NCRISC_FIRMWARE_BASE, TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc0.elf": (TensixL1.TRISC0_BASE,          TensixL1.TRISC0_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc1.elf": (TensixL1.TRISC1_BASE,          TensixL1.TRISC1_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc2.elf": (TensixL1.TRISC2_BASE,          TensixL1.TRISC2_INIT_LOCAL_L1_BASE_SCRATCH),
    }

    for name, segs in fw:
      exp_base, _ = fw_map[name]
      assert any(s.paddr == exp_base for s in segs), f"{name}: missing expected pt_load base"

    if DEBUG >= 1: print(f"writing firmware to {len(self.tiles.tensix)} tensix tiles")
    for name, (base, init) in fw_map.items():
      if DEBUG >= 2: print(f"{name}: base=0x{base:x} init=0x{init:x}")

    def remap_addr(addr: int, init_base: int) -> int:
      if TensixMMIO.LOCAL_RAM_START <= addr <= TensixMMIO.LOCAL_RAM_END:
        addr = (addr - TensixMMIO.LOCAL_RAM_START) + init_base
        assert 0 <= addr < TensixL1.SIZE
      return addr

    cfg = TLBConfig(addr=reg_base, noc=0, mcast=True, mode=TLBMode.STRICT)
    y0, y1 = self.tiles.TENSIX_Y
    with TLBWindow(self.fd, TLBSize.MiB_2) as win:
      for x0, x1 in self.tiles.tensix_mcast:
        if DEBUG >= 2: print(f"mcast x=[{x0},{x1}] y=[{y0},{y1}]")
        cfg.start, cfg.end, cfg.addr, cfg.mode = (x0, y0), (x1, y1), reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, TensixMMIO.SOFT_RESET_ALL)

        cfg.mode = TLBMode.ORDERED_BULK
        for name, segs in fw:
          _, init_base = fw_map[name]
          for seg in segs:
            if not seg.data: continue
            addr = remap_addr(seg.paddr, init_base)
            if DEBUG >= 3: print(f"{name}: 0x{seg.paddr:x} -> 0x{addr:x} ({len(seg.data)} bytes)")
            win.write(addr, seg.data, restore=False)

        cfg.addr, cfg.mode = reg_base, TLBMode.STRICT
        win.configure(cfg)
        win.writei32(reg_off, 0x0)

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

      if telem_struct_addr == 0:
        raise RuntimeError("telemetry struct address is 0 (ARC not ready)")
      if not (Arc.CSM_START <= telem_struct_addr <= Arc.CSM_END):
        raise RuntimeError(f"invalid telemetry struct address: 0x{telem_struct_addr:08x}")

      csm_base, csm_offset = align_down(telem_struct_addr, TLBSize.MiB_2)
      
      # change base address so we can read the where the pointer points
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
      pcie_usage = read_tag(Arc.TAG_PCIE_USAGE, Arc.DEFAULT_PCIE_USAGE)

      pcie_disabled = tuple(
        i for i in (0, 1)
        if ((pcie_usage >> (i * 2)) & 0x3) != 1
      )

      tensix_cols = tuple(sorted(
        loc for pos, loc in enumerate(HARVESTING_NOC_LOCATIONS)
        if ((tensix_enabled >> pos) & 1) == 0
      ))

      dram_banks = tuple(
        bank for bank in range(Dram.BANK_COUNT)
        if ((gddr_enabled >> bank) & 1) == 0
      )

      all_eth_disabled = (eth_enabled & Arc.DEFAULT_ETH_ENABLED) == 0

    return Harvesting(
      tensix_cols=tensix_cols,
      dram_banks=dram_banks,
      all_eth_disabled=all_eth_disabled,
      pcie=pcie_disabled,
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


    if DEBUG >= 1: print(f"opened blackhole {self.arch} at {self.get_bdf()}")
    self._map_bars()

  def _close(self):
    if hasattr(self, 'mm0'): self.mm0.close()
    if hasattr(self, 'mm1'): self.mm1.close()
    os.close(self.fd)

  def get_bdf(self) -> str:
    in_sz = ctypes.sizeof(TenstorrentGetDeviceInfoIn)
    out_sz = ctypes.sizeof(TenstorrentGetDeviceInfoOut)
    buf = bytearray(in_sz + out_sz)
    TenstorrentGetDeviceInfoIn.from_buffer(buf).output_size_bytes = out_sz
    trace_ioctl(IOCTL_GET_DEVICE_INFO)
    fcntl.ioctl(self.fd, _IO(IOCTL_GET_DEVICE_INFO), buf, True)
    info = TenstorrentGetDeviceInfoOut.from_buffer(buf, in_sz)
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
    if DEBUG >= 3: print(f"mmap bar0: {bars[0].mapping_size:#x} bytes, bar2: {bars[2].mapping_size:#x} bytes")

  def reset(self, dmc_reset: bool = False) -> int:
    bdf = self.get_bdf()
    if DEBUG >= 1: print(f"resetting device {bdf}")
    in_sz, out_sz = ctypes.sizeof(ResetDeviceIn), ctypes.sizeof(ResetDeviceOut)

    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes = out_sz
    view.flags = TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET if dmc_reset else TENSTORRENT_RESET_DEVICE_ASIC_RESET
    trace_ioctl(IOCTL_RESET_DEVICE, "ASIC_DMC_RESET" if dmc_reset else "ASIC_RESET")
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    self._close()

    # poll for device to come back (up to 10s)
    if DEBUG >= 2: print("waiting for device to come back...")
    for _ in range(50):
      time.sleep(0.2)
      if (path := find_dev_by_bdf(bdf)):
        self.path = path
        break
    else:
      raise RuntimeError(f"device {bdf} didn't come back after reset")

    if DEBUG >= 2: print(f"device back at {self.path}")
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    # POST_RESET reinits hardware
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_POST_RESET
    trace_ioctl(IOCTL_RESET_DEVICE, "POST_RESET")
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    result = ResetDeviceOut.from_buffer(buf, in_sz).result
    if DEBUG >= 1: print(f"reset complete, result={result}")

    self._setup(retried=True)
    return result

  def _get_arch(self):
    ordinal = self.path.split('/')[-1]
    with open(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type", "r") as f: return f.read().strip()

  def close(self): self._close()
