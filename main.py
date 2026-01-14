import ctypes, fcntl, mmap, os, time
from autogen import *

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr

def _get_bdf_for_path(path: str) -> str | None:
  """Get BDF for a device path, or None if it fails."""
  try:
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    in_sz = ctypes.sizeof(TenstorrentGetDeviceInfoIn)
    out_sz = ctypes.sizeof(TenstorrentGetDeviceInfoOut)
    buf = bytearray(in_sz + out_sz)
    TenstorrentGetDeviceInfoIn.from_buffer(buf).output_size_bytes = out_sz
    fcntl.ioctl(fd, _IO(IOCTL_GET_DEVICE_INFO), buf, True)
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
    self._open()
  
  def _open(self):
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)
    self.arch = self._get_arch()

    assert self.arch in ("p100a", "p150b"), "only blackhole is supported"
    print(f"opened blackhole {self.arch} at {self.get_bdf()}")

    # mmap bar0 and bar1 // not sure if these are necessary
    self._map_bars()

  def _close(self):
    self.mm0.close() 
    self.mm1.close() 
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

    # UC bars for bar0 and bar1 are 1,3 (the others are WC which is bad for reading/writing registers)
    # we don't need to mmap global vram (4+5), that is done through the dram tiles and the NoC
    self.mm0 = mmap.mmap(self.fd, bars[0].mapping_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bars[0].mapping_base)
    self.mm1 = mmap.mmap(self.fd, bars[2].mapping_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=bars[2].mapping_base)

  def reset(self) -> int:
    bdf = self.get_bdf()
    print(f"resetting device {bdf}")
    in_sz, out_sz = ctypes.sizeof(ResetDeviceIn), ctypes.sizeof(ResetDeviceOut)

    # trigger reset
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_ASIC_RESET
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    self._close()

    # poll for device to come back by BDF (up to 10s)
    print("waiting for device to come back...")
    for _ in range(50):
      time.sleep(0.2)
      if (path := find_dev_by_bdf(bdf)):
        self.path = path
        break
    else:
      raise RuntimeError(f"Device {bdf} didn't come back after reset")

    print(f"device back at {self.path}")

    # open fd first, then POST_RESET to reinit hardware, then finish setup
    self.fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC)

    # POST_RESET: confirms reset completed (checks reset marker) and tells driver to reinit hardware
    buf = bytearray(in_sz + out_sz)
    view = ResetDeviceIn.from_buffer(buf)
    view.output_size_bytes, view.flags = out_sz, TENSTORRENT_RESET_DEVICE_POST_RESET
    fcntl.ioctl(self.fd, _IO(IOCTL_RESET_DEVICE), buf, True)
    result = ResetDeviceOut.from_buffer(buf, in_sz).result
    print(f"reset complete, result={result}")

    # now hardware is reinitialized, do arch check and bar mapping
    self.arch = self._get_arch()
    assert self.arch in ("p100a", "p150b"), "only blackhole is supported"
    self._map_bars()
    print(f"reopened blackhole {self.arch}")
    return result
  
  def _get_arch(self):
    ordinal = self.path.split('/')[-1]
    with open(f"/sys/class/tenstorrent/tenstorrent!{ordinal}/tt_card_type", "r", encoding="ascii") as f: return f.read().strip()
  
  def close(self): self._close()

def main():
  device = Device()
  device.reset()
t 
if __name__ == "__main__":
  main()
