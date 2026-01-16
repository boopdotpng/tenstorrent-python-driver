from __future__ import annotations
import ctypes, fcntl, mmap, os, time
from dataclasses import dataclass, field
from typing import ClassVar
from autogen import *
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import DEBUG, _IO, align_down, find_dev_by_bdf
from configs import Arc, Dram, HARVESTING_NOC_LOCATIONS

@dataclass
class Harvesting:
  """
  determine harvesting, very important
  the specific Tensix tiles (columns) and DRAM banks turned off vary per card. 
  noc writes and reads have to avoid these columns 
  certain dram banks also cannot be used
  pcie is less important, but we might need it for something later
  """
  tensix_tile_cols: tuple[int, ...]
  dram_banks: tuple[int, ...]
  eth_cores: bool
  pcie: tuple[int, ...]

  def __repr__(self) -> str:
    return (
      "the following items are harvested (disabled):\n"
      f"tensix_tile_cols={self.tensix_tile_cols} "
      f"dram_banks={self.dram_banks} "
      # on p100a, there is no ethernet interface
      f"eth_cores={"none" if self.eth_cores else "all"} "
      f"pcie={self.pcie}"
    )

@dataclass
class TileGrid:
  ARC: ClassVar[tuple[int, int]] = (8, 0)     # ARC tile on both boards
  tensix: list[tuple[int, int]]               # all valid tensix (x, y) for unicast
  tensix_mcast: list[tuple[int, int]]         # contiguous x-ranges for multicast: [(start_x, end_x), ...]
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
    p100a = not harvesting.eth_cores
    max_x = 14 if p100a else 16
    dram_cols, l2cpu_col = (0, 9), 8

    # valid tensix columns (exclude dram, l2cpu, and harvested)
    disabled = set(harvesting.tensix_tile_cols)
    tensix_cols = [x for x in range(max_x + 1) if x not in dram_cols and x != l2cpu_col and x not in disabled]

    # unicast: all valid (x, y) pairs, y=2..11 for tensix
    tensix = [(x, y) for x in tensix_cols for y in range(2, 12)]

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

# we have to upload firmware to the risc v cores inside the tensix tile every fresh boot,
# since l1 is volatile 

class Device:
  def __init__(self, path: str = "/dev/tenstorrent/0"):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self._setup()
    self.harvesting = self.get_harvesting()
    self.tiles = TileGrid.from_harvesting(self.harvesting)

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

      csm_base, csm_offset = align_down(telem_struct_addr, TLBSize.MiB_2.value)
      
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

      tensix_enabled = read_tag(Arc.TAG_TENSIX_ENABLED, 0x3FFF)
      eth_enabled = read_tag(Arc.TAG_ETH_ENABLED, 0x3FFF)
      gddr_enabled = read_tag(Arc.TAG_GDDR_ENABLED, 0xFF)
      pcie_usage = read_tag(Arc.TAG_PCIE_USAGE, 0x5)

      pcie_disabled = tuple(
        i for i in (0, 1)
        if ((pcie_usage >> (i * 2)) & 0x3) != 1
      )

      tensix_tile_cols = tuple(sorted(
        loc for pos, loc in enumerate(HARVESTING_NOC_LOCATIONS)
        if ((tensix_enabled >> pos) & 1) == 0
      ))

      dram_banks = tuple(
        bank for bank in range(Dram.BANK_COUNT)
        if ((gddr_enabled >> bank) & 1) == 0
      )

      # blackhole p100a has all eth cores disabled, p150a has all on.
      # check this info
      eth_mask = (eth_enabled & ((1<<14) - 1)) != 0

    return Harvesting(
      # disabled tensix columns in sorted noc x order.
      tensix_tile_cols = tuple(tensix_tile_cols),
      # disabled dram banks (0-7).
      dram_banks = tuple(dram_banks),
      # disabled ethernet cores (0-13).
      eth_cores = eth_mask,
      # disabled PCIe instances: 0 or 1.
      pcie = tuple(pcie_disabled),
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


    print(f"opened blackhole {self.arch} at {self.get_bdf()}")
    self._map_bars()

  def _close(self):
    if hasattr(self, 'mm0'): self.mm0.close()
    if hasattr(self, 'mm1'): self.mm1.close()
    os.close(self.fd)

  def get_bdf(self):
    in_sz = ctypes.sizeof(TenstorrentGetDeviceInfoIn)
    out_sz = ctypes.sizeof(TenstorrentGetDeviceInfoOut)
    buf = bytearray(in_sz + out_sz)
    TenstorrentGetDeviceInfoIn.from_buffer(buf).output_size_bytes = out_sz

    fcntl.ioctl(self.fd, _IO(IOCTL_GET_DEVICE_INFO), buf, True)
    # we only really care about the bdf
    bdf = TenstorrentGetDeviceInfoOut.from_buffer(buf, in_sz).bus_dev_fn
    pci_domain = TenstorrentGetDeviceInfoOut.from_buffer(buf, in_sz).pci_domain

    return f"{pci_domain:04x}:{(bdf >> 8) & 0xFF:02x}:{(bdf >> 3) & 0x1F:02x}.{bdf & 0x7}"

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

  def reset(self, dmc_reset:bool=False) -> int:
    # mirrors reset logic in tt-kmd/tools/reset.c
    bdf = self.get_bdf()
    print(f"resetting device {bdf}")
    in_sz, out_sz = ctypes.sizeof(ResetDeviceIn), ctypes.sizeof(ResetDeviceOut)

    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, (TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET if dmc_reset else TENSTORRENT_RESET_DEVICE_ASIC_RESET)
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    self._close()

    # poll for device to come back by bus, device, function from get_device_info (up to 10s)
    print("waiting for device to come back...")
    for _ in range(50):
      time.sleep(0.2)
      if (path := find_dev_by_bdf(bdf)):
        self.path = path
        break
    else:
      raise RuntimeError(f"device {bdf} didn't come back after reset")

    print(f"device back at {self.path}")

    # open fd first, then POST_RESET to reinit hardware, then finish setup
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    # post reset: without this, the device doesn't init again
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_POST_RESET
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    result = ResetDeviceOut.from_buffer(buf, in_sz).result
    print(f"reset complete, result={result}")

    self._setup(retried=True)
    return result

  def _get_arch(self):
    ordinal = self.path.split('/')[-1]
    with open(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type", "r") as f: return f.read().strip()

  def close(self): self._close()
