import os, ctypes, fcntl
from autogen import TENSTORRENT_IOCTL_MAGIC, TenstorrentGetDeviceInfoIn
from autogen import TenstorrentGetDeviceInfoOut, IOCTL_GET_DEVICE_INFO
from pathlib import Path

# UT3G cannot support fast dispatch because of the 1g iommu map requirement
# will test this later
# used by default
# SLOW_DISPATCH = int(os.environ.get("TT_SLOW_DISPATCH", 0)) == 1
DEBUG = int(os.environ.get("DEBUG", 0)) > 1
TT_HOME = Path(os.environ.get("TT_HOME", ""))

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr
def align_down(value: int, alignment: int) -> tuple[int, int]:
  base = value & ~(alignment - 1)
  return base, value - base

def _get_bdf_for_path(path: str) -> str | None:
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

