import os, struct
from defs import TLBSize, TensixL1, TENSTORRENT_IOCTL_MAGIC
from dataclasses import dataclass
from pathlib import Path

TT_HOME = Path(os.environ["TT_HOME"])

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr

def align_down(value: int, alignment: TLBSize) -> tuple[int, int]:
  base = value & ~(alignment.value - 1)
  return base, value - base

def noc1(x: int, y: int) -> tuple[int, int]: return (16 - x, 11 - y)

@dataclass(frozen=True)
class PTLoad:
  paddr: int
  data: bytes
  memsz: int
  flags: int = 0

def iter_pt_load(elf: bytes):
  e_phoff = struct.unpack_from("<I", elf, 28)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", elf, 42)
  if e_phentsize < 32:
    raise ValueError(f"bad e_phentsize: {e_phentsize}")
  if e_phoff + e_phentsize * e_phnum > len(elf):
    raise ValueError("ELF truncated")
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, _ = (
      struct.unpack_from("<IIIIIIII", elf, off)
    )
    if p_type != 1:
      continue  # PT_LOAD
    if p_offset + p_filesz > len(elf):
      raise ValueError("ELF truncated")
    paddr = p_paddr or p_vaddr
    yield PTLoad(
      paddr=paddr,
      data=elf[p_offset : p_offset + p_filesz],
      memsz=p_memsz,
      flags=p_flags,
    )

def load_pt_load(path: str | os.PathLike[str]) -> list[PTLoad]:
  with open(os.fspath(path), "rb") as f:
    return list(iter_pt_load(f.read()))

def generate_jal_instruction(target_addr: int) -> int:
  assert target_addr < 0x80000, f"target too far for JAL: {target_addr:#x}"
  opcode = 0x6F
  imm_10_1 = (target_addr & 0x7FE) << 20
  imm_11 = (target_addr & 0x800) << 9
  imm_19_12 = target_addr & 0xFF000
  return imm_19_12 | imm_11 | imm_10_1 | opcode

def pack_xip_elf(path: str | os.PathLike[str]) -> tuple[bytes, int]:
  segs = load_pt_load(path)
  if not segs:
    raise ValueError("no PT_LOAD segments")
  l1 = [s for s in segs if (s.memsz or s.data) and (0 <= s.paddr < TensixL1.SIZE)]
  if not l1:
    raise ValueError("no L1 PT_LOAD segments")
  l1.sort(key=lambda s: s.paddr)
  base = l1[0].paddr
  out = bytearray()
  for s in l1:
    start = s.paddr - base
    size = max(s.memsz, len(s.data))
    end = start + size
    if len(out) < end:
      out.extend(b"\0" * (end - len(out)))
    out[start : start + len(s.data)] = s.data
  text = next((s for s in l1 if (s.flags & 1) and s.data), l1[0])
  return bytes(out), len(text.data)
