import os, sys, ctypes, fcntl, struct
from autogen import TENSTORRENT_IOCTL_MAGIC, TenstorrentGetDeviceInfoIn
from autogen import TenstorrentGetDeviceInfoOut, IOCTL_GET_DEVICE_INFO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from configs import TLBSize

# UT3G cannot support fast dispatch because of the 1g iommu map requirement
# will test this later
# used by default
# SLOW_DISPATCH = int(os.environ.get("TT_SLOW_DISPATCH", 0)) == 1
DEBUG = int(os.environ.get("DEBUG", 0))
TT_HOME = Path(os.environ.get("TT_HOME", ""))

# DEBUG levels:
#   0: errors/prompts only
#   1: progress (device opened, reset complete, major ops)
#   2: details (harvesting, tile counts, fw map, mcast ranges)
#   3: data (segment writes, TLB alloc/free)
#   4: trace (all ioctls, all TLB configures, memory writes)

IOCTL_NAMES = {
  0: "GET_DEVICE_INFO", 2: "QUERY_MAPPINGS", 6: "RESET_DEVICE",
  7: "PIN_PAGES", 10: "UNPIN_PAGES", 11: "ALLOCATE_TLB",
  12: "FREE_TLB", 13: "CONFIGURE_TLB",
}

def trace_ioctl(nr: int, extra: str = ""):
  if DEBUG >= 4: print(f"ioctl: {IOCTL_NAMES.get(nr, nr)}{' ' + extra if extra else ''}")

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr

def align_down(value: int, alignment: TLBSize) -> tuple[int, int]:
  base = value & ~(alignment.value - 1)
  return base, value - base

def format_bdf(pci_domain: int, bus_dev_fn: int) -> str:
  """Format PCI bus:device.function address."""
  return f"{pci_domain:04x}:{(bus_dev_fn >> 8) & 0xFF:02x}:{(bus_dev_fn >> 3) & 0x1F:02x}.{bus_dev_fn & 0x7}"

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
    return format_bdf(info.pci_domain, info.bus_dev_fn)
  except OSError: return None

def find_dev_by_bdf(target_bdf: str) -> str | None:
  for entry in os.listdir("/dev/tenstorrent"):
    if not entry.isdigit(): continue
    path = f"/dev/tenstorrent/{entry}"
    if _get_bdf_for_path(path) == target_bdf: return path
  return None

@dataclass(frozen=True)
class PTLoad:
  paddr: int
  data: bytes
  memsz: int

def iter_pt_load(elf: bytes) -> Iterable[PTLoad]:
  if elf[:4] != b"\x7fELF": raise ValueError("not an ELF")
  if elf[4] != 1: raise ValueError("expected ELF32")
  if elf[5] != 1: raise ValueError("expected little-endian")

  e_phoff = struct.unpack_from("<I", elf, 28)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 42)
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_offset, _, p_paddr, p_filesz, p_memsz, _, _ = struct.unpack_from("<IIIIIIII", elf, off)
    if p_type != 1: continue  # PT_LOAD
    if p_offset + p_filesz > len(elf): raise ValueError("ELF truncated")
    yield PTLoad(paddr=p_paddr, data=elf[p_offset:p_offset + p_filesz], memsz=p_memsz)


def load_pt_load(elf_path: str | os.PathLike[str]) -> list[PTLoad]:
  with open(os.fspath(elf_path), "rb") as f:
    return list(iter_pt_load(f.read()))
