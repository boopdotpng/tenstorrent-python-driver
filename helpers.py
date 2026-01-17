import os, sys,  fcntl, struct
from ctypes import sizeof
from abi import TENSTORRENT_IOCTL_MAGIC, TenstorrentGetDeviceInfoIn
from abi import TenstorrentGetDeviceInfoOut, IOCTL_GET_DEVICE_INFO
from abi import IOCTL_ALLOCATE_TLB, IOCTL_FREE_TLB, IOCTL_CONFIGURE_TLB
from dataclasses import dataclass
from pathlib import Path
from configs import TLBSize

DEBUG = int(os.environ.get("DEBUG", 0))
TT_HOME = Path(os.environ.get("TT_HOME", ""))

# ANSI colors
class C:
  RESET = "\033[0m"
  BOLD = "\033[1m"
  DIM = "\033[2m"
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"

TAG_COLORS = {
  "dev": C.GREEN, "fw": C.CYAN, "tlb": C.BLUE,
  "dram": C.MAGENTA, "mmap": C.YELLOW, "ioctl": C.DIM,
}

def dbg(level: int, tag: str, msg: str):
  if DEBUG < level: return
  color = TAG_COLORS.get(tag, C.RESET)
  print(f"{color}{C.BOLD}{tag}{C.RESET}{color}:{C.RESET} {msg}")

def warn(tag: str, msg: str):
  print(f"{C.YELLOW}{C.BOLD}{tag}{C.RESET}{C.YELLOW}:{C.RESET} {msg}", file=sys.stderr)

IOCTL_NAMES = {
  0: "GET_DEVICE_INFO", 2: "QUERY_MAPPINGS", 6: "RESET_DEVICE",
  7: "PIN_PAGES", 10: "UNPIN_PAGES", 11: "ALLOCATE_TLB",
  12: "FREE_TLB", 13: "CONFIGURE_TLB",
}

_TLB_IOCTLS = {IOCTL_ALLOCATE_TLB, IOCTL_FREE_TLB, IOCTL_CONFIGURE_TLB}

def trace_ioctl(nr: int, extra: str = ""):
  if DEBUG < 4 or nr in _TLB_IOCTLS: return
  name = IOCTL_NAMES.get(nr, str(nr))
  dbg(4, "ioctl", f"{name}{' ' + extra if extra else ''}")

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr

def ioctl[T](fd: int, nr: int, in_cls, out_cls: type[T], **fields) -> T:
  in_sz, out_sz = sizeof(in_cls), sizeof(out_cls) # type: ignore
  buf = bytearray(in_sz + out_sz)
  view = in_cls.from_buffer(buf)
  for k, v in fields.items(): setattr(view, k, v)
  fcntl.ioctl(fd, _IO(nr), buf, True)
  return out_cls.from_buffer(buf, in_sz) # type: ignore

def contiguous_ranges(xs: list[int]) -> list[tuple[int, int]]:
  if not xs: return []
  ranges, start = [], xs[0]
  for i, x in enumerate(xs[1:], 1):
    if x != xs[i-1] + 1:
      ranges.append((start, xs[i-1]))
      start = x
  return ranges + [(start, xs[-1])]

def align_down(value: int, alignment: TLBSize) -> tuple[int, int]:
  base = value & ~(alignment.value - 1)
  return base, value - base

# NoC 1 has its origin at bottom-right instead of top-left
def noc1(x: int, y: int) -> tuple[int, int]:
  return (16 - x, 11 - y)  # MAX_X=16, MAX_Y=11 for blackhole

def format_bdf(pci_domain: int, bus_dev_fn: int) -> str:
  return f"{pci_domain:04x}:{(bus_dev_fn >> 8) & 0xFF:02x}:{(bus_dev_fn >> 3) & 0x1F:02x}.{bus_dev_fn & 0x7}"

def _get_bdf_for_path(path: str) -> str | None:
  try:
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    info = ioctl(fd, IOCTL_GET_DEVICE_INFO, TenstorrentGetDeviceInfoIn,
                 TenstorrentGetDeviceInfoOut, output_size_bytes=sizeof(TenstorrentGetDeviceInfoOut))
    os.close(fd)
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

def load_pt_load(path: str | os.PathLike[str]) -> list[PTLoad]:
  with open(os.fspath(path), "rb") as f: elf = f.read()
  e_phoff = struct.unpack_from("<I", elf, 28)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 42)
  segs = []
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    # ELF32 Phdr: type, offset, vaddr, paddr, filesz, memsz, flags, align
    p_type, p_offset, _, p_paddr, p_filesz, p_memsz, _, _ = struct.unpack_from("<IIIIIIII", elf, off)
    if p_type != 1: continue  # PT_LOAD
    if p_offset + p_filesz > len(elf): raise ValueError("ELF truncated")
    segs.append(PTLoad(paddr=p_paddr, data=elf[p_offset:p_offset + p_filesz], memsz=p_memsz))
  return segs

def pack_xip_elf(path: str | os.PathLike[str]) -> tuple[bytes, int]:
  segs = load_pt_load(path)
  if not segs: raise ValueError("no PT_LOAD segments")
  pad4 = lambda b: b + b"\0" * (-len(b) & 3)
  return b"".join(pad4(s.data) for s in segs if s.data), len(pad4(segs[0].data))
