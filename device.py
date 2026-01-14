import ctypes, fcntl, mmap, os, time
from dataclasses import dataclass
from autogen import *
from tlb import TLBConfig, TLBWindow, TLBMode, TLBSize
from helpers import DEBUG, _IO, align_down

# device constants, shared between p100 and p150

## ARC tile constants
ARC_CORE = (8, 0) # from NoC 0
ARC_NOC_BASE = 0x80000000 # arc noc xbar base (from tt-umd)
SCRATCH_RAM_13 = 0x30434 # scratch_ram_13. u32 pointer to the telemetry struct table in csm
# (core/control system memory) used by firmware 

# memory bounds used by firmware on blackhole
CSM_START = 0x10000000 
CSM_END = 0x1007FFFF

## harvesting
# in firmware, the cores are stored out of order, left edge, right edge, left edge, etc
# the bitmask stored in firmware (0x1020 for blackhole) is applied to this order
HARVESTING_NOC_LOCATIONS = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10]

@dataclass
class Harvesting:
  """
  determine harvesting, very important
  the specific Tensix tiles (columns) and DRAM banks turned off vary per card. 
  noc writes and reads have to avoid these columns 
  certain dram banks also cannot be used
  """
  tensix_tile_cols: tuple[int, ...]
  dram_banks: tuple[int, ...]
  eth_cores: bool
  pcie: tuple[int, ...]

  def __repr__(self) -> str:
    return (
      "the following items are harvested (disabled): "
      f"tensix_tile_cols={self.tensix_tile_cols} "
      f"dram_banks={self.dram_banks} "
      # on p100a, there is no ethernet interface
      f"eth_cores={"none" if self.eth_cores else "all"} "
      f"pcie={self.pcie}"
    )

def _get_bdf_for_path(path: str) -> str | None:
  try:
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    in_sz = ctypes.sizeof(TenstorrentGetDeviceInfoIn)
    out_sz = ctypes.sizeof(TenstorrentGetDeviceInfoOut)
    buf = bytearray(in_sz + out_sz)
    TenstorrentGetDeviceInfoIn.from_buffer(buf).output_size_bytes = out_sz
    fcntl.ioctl(fd, _IO(IOCTL_GET_DEVICE_INFO), buf, True)
    if DEBUG: print(f"ioctl get device info: output_size_bytes: {out_sz}")
    os.close(fd)
    info = TenstorrentGetDeviceInfoOut.from_buffer(buf, in_sz)
    bdf = info.bus_dev_fn
    return f"{info.pci_domain:04x}:{(bdf >> 8) & 0xFF:02x}:{(bdf >> 3) & 0x1F:02x}.{bdf & 0x7}"
  except: return None

def find_dev_by_bdf(target_bdf: str) -> str | None:
  for entry in os.listdir("/dev/tenstorrent"):
    if not entry.isdigit(): continue
    path = f"/dev/tenstorrent/{entry}"
    if _get_bdf_for_path(path) == target_bdf: return path
  return None

class Device:
  def __init__(self, path: str = "/dev/tenstorrent/0"):
    self.path = path
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    if DEBUG: print(f"opened {path}, file descriptor {self.fd}")
    self._setup()
    self.harvesting = self.get_harvesting()
    print(self.harvesting)

  def get_harvesting(self) -> Harvesting:
    tlb_config = TLBConfig(
      addr = ARC_NOC_BASE,
      start = ARC_CORE,
      end = ARC_CORE,
      noc = 0,
      mcast = False,
      mode = TLBMode.STRICT
    )

    with TLBWindow(self.fd, TLBSize.MiB_2, tlb_config) as arc:
      telem_struct_addr = arc.readi32(SCRATCH_RAM_13)

      if telem_struct_addr == 0:
        raise RuntimeError("telemetry struct address is 0 (ARC not ready)")
      if not (CSM_START <= telem_struct_addr <= CSM_END):
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

      tensix_enabled = read_tag(34, 0x3FFF)
      eth_enabled = read_tag(35, 0x3FFF)
      gddr_enabled = read_tag(36, 0xFF)
      pcie_usage = read_tag(38, 0x5)

      pcie_disabled = tuple(
        i for i in (0, 1)
        if ((pcie_usage >> (i * 2)) & 0x3) != 1
      )

      tensix_tile_cols = tuple(sorted(
        loc for pos, loc in enumerate(HARVESTING_NOC_LOCATIONS)
        if ((tensix_enabled >> pos) & 1) == 0
      ))

      dram_banks = tuple(
        bank for bank in range(8)
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
    if DEBUG: print("device closed")

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

    if DEBUG:
      print(f"mapped bar 0 (0x{bars[0].mapping_size:x} bytes) at address 0x{bars[0].mapping_base:x}")
      print(f"mapped bar 1 (0x{bars[2].mapping_size:x} bytes) at address 0x{bars[2].mapping_base:x}")

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
