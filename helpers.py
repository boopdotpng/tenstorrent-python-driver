import os, fcntl, struct
from ctypes import sizeof
from abi import TENSTORRENT_IOCTL_MAGIC, TenstorrentGetDeviceInfoIn
from abi import TenstorrentGetDeviceInfoOut, IOCTL_GET_DEVICE_INFO
from dataclasses import dataclass
from pathlib import Path
from configs import TLBSize, TensixL1

_tt_home_env = os.environ.get("TT_HOME")
if _tt_home_env:
  TT_HOME = Path(_tt_home_env)
else:
  TT_HOME = Path(__file__).resolve().parents[1]/"tt-metal"

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
  flags: int = 0

def iter_pt_load(elf: bytes):
  if elf[:4] != b"\x7fELF": raise ValueError("not an ELF")
  if elf[4] != 1: raise ValueError("expected ELF32")
  if elf[5] != 1: raise ValueError("expected little-endian")

  e_phoff = struct.unpack_from("<I", elf, 28)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 42)
  if e_phentsize < 32: raise ValueError(f"bad e_phentsize: {e_phentsize}")
  if e_phoff + e_phentsize * e_phnum > len(elf): raise ValueError("ELF truncated")

  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, _ = struct.unpack_from("<IIIIIIII", elf, off)
    if p_type != 1: continue  # PT_LOAD
    if p_offset + p_filesz > len(elf): raise ValueError("ELF truncated")
    paddr = p_paddr or p_vaddr
    yield PTLoad(paddr=paddr, data=elf[p_offset:p_offset + p_filesz], memsz=p_memsz, flags=p_flags)

def load_pt_load(path: str | os.PathLike[str]) -> list[PTLoad]:
  with open(os.fspath(path), "rb") as f:
    return list(iter_pt_load(f.read()))

def generate_jal_instruction(target_addr: int) -> int:
  """Generate RISC-V JAL x0, offset instruction for BRISC bootstrap.

  BRISC has hardcoded reset PC of 0, so we place a JAL at address 0
  that jumps to the actual firmware base address.
  """
  assert target_addr < 0x80000, f"target too far for JAL: {target_addr:#x}"
  opcode = 0x6f  # JAL opcode
  # RISC-V JAL immediate encoding: imm[20|10:1|11|19:12] rd opcode
  # rd = x0 (zero register, so this is an unconditional jump)
  imm_10_1 = (target_addr & 0x7fe) << 20
  imm_11 = (target_addr & 0x800) << 9
  imm_19_12 = target_addr & 0xff000
  return imm_19_12 | imm_11 | imm_10_1 | opcode

def pack_xip_elf(path: str | os.PathLike[str]) -> tuple[bytes, int]:
  segs = load_pt_load(path)
  if not segs: raise ValueError("no PT_LOAD segments")

  l1 = [s for s in segs if (s.memsz or s.data) and (0 <= s.paddr < TensixL1.SIZE)]
  if not l1: raise ValueError("no L1 PT_LOAD segments")

  l1.sort(key=lambda s: s.paddr)
  base = l1[0].paddr
  out = bytearray()
  for s in l1:
    start = s.paddr - base
    size = max(s.memsz, len(s.data))
    end = start + size
    if len(out) < end: out.extend(b"\0" * (end - len(out)))
    out[start:start + len(s.data)] = s.data

  text = next((s for s in l1 if (s.flags & 1) and s.data), l1[0])
  return bytes(out), len(text.data)
