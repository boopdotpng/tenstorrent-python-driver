import os, struct
from defs import TLBSize, TensixL1, TENSTORRENT_IOCTL_MAGIC
from dataclasses import dataclass

USE_USB_DISPATCH = os.environ.get("TT_USB") == "1"
PROFILER = os.environ.get("TT_PROFILER") == "1"
TIMING = bool(int(os.getenv("TIMING", "0")))
HiReloc = tuple[int, int]
LoReloc = tuple[int, int, int]

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr

def align_up(n: int, a: int) -> int: return (n + a - 1) & ~(a - 1)

def align_down(value: int, alignment: TLBSize) -> tuple[int, int]:
  base = value & ~(alignment.value - 1)
  return base, value - base

def noc_xy(x: int, y: int) -> int: return ((y << 6) | x) & 0xFFFF

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
  if e_phentsize < 32: raise ValueError(f"bad e_phentsize: {e_phentsize}")
  if e_phoff + e_phentsize * e_phnum > len(elf): raise ValueError("ELF truncated")
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, _ = struct.unpack_from("<IIIIIIII", elf, off)
    if p_type != 1: continue  # PT_LOAD
    if p_offset + p_filesz > len(elf): raise ValueError("ELF truncated")
    paddr = p_paddr or p_vaddr
    yield PTLoad(paddr=paddr, data=elf[p_offset : p_offset + p_filesz], memsz=p_memsz, flags=p_flags)

def load_pt_load(path: str | os.PathLike[str]) -> list[PTLoad]:
  with open(os.fspath(path), "rb") as f:
    return list(iter_pt_load(f.read()))

def _xipify_riscv32_elf(elf: bytes) -> bytes:
  # Minimal XIP relocation transform for kernel ELFs:
  # convert ABS HI20/LO12 references to text symbols into PCREL-style encodings.
  data = bytearray(elf)

  e_phoff = struct.unpack_from("<I", data, 28)[0]
  e_shoff = struct.unpack_from("<I", data, 32)[0]
  e_phentsize, e_phnum = struct.unpack_from("<HH", data, 42)
  e_shentsize, e_shnum = struct.unpack_from("<HH", data, 46)

  if e_phentsize < 32 or e_shentsize < 40:
    return elf
  if e_phoff + e_phentsize * e_phnum > len(data) or e_shoff + e_shentsize * e_shnum > len(data):
    return elf

  R_RISCV_HI20 = 26
  R_RISCV_LO12_I = 27
  R_RISCV_LO12_S = 28

  def shdr(i: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    off = e_shoff + i * e_shentsize
    return struct.unpack_from("<IIIIIIIIII", data, off)

  def sym(symtab_idx: int, sym_idx: int) -> tuple[int, int, int, int, int, int]:
    _, _, _, _, sh_offset, sh_size, sh_link, _, _, sh_entsize = shdr(symtab_idx)
    if sh_entsize < 16:
      raise ValueError("bad sym entsize")
    soff = sh_offset + sym_idx * sh_entsize
    if soff + 16 > sh_offset + sh_size:
      raise ValueError("bad sym index")
    st_name, st_value, st_size, st_info, st_other, st_shndx = struct.unpack_from("<IIIBBH", data, soff)
    return st_name, st_value, st_size, st_info, st_other, st_shndx

  text_vaddr, text_memsz = None, None
  for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, _, p_vaddr, _, _, p_memsz, p_flags, _ = struct.unpack_from("<IIIIIIII", data, off)
    if p_type == 1 and (p_flags & 1):  # PT_LOAD + executable
      text_vaddr, text_memsz = p_vaddr, p_memsz
      break
  if text_vaddr is None or text_memsz is None:
    return elf
  text_end = text_vaddr + text_memsz

  def is_text_addr(addr: int) -> bool:
    return text_vaddr <= addr <= text_end

  def read_u32_at_vaddr(sec_idx: int, addr: int) -> int:
    _, _, _, sh_addr, sh_offset, sh_size, _, _, _, _ = shdr(sec_idx)
    rel = addr - sh_addr
    if rel < 0 or rel + 4 > sh_size:
      raise ValueError("reloc offset out of section")
    return struct.unpack_from("<I", data, sh_offset + rel)[0]

  def write_u32_at_vaddr(sec_idx: int, addr: int, value: int):
    _, _, _, sh_addr, sh_offset, sh_size, _, _, _, _ = shdr(sec_idx)
    rel = addr - sh_addr
    if rel < 0 or rel + 4 > sh_size:
      raise ValueError("reloc offset out of section")
    struct.pack_into("<I", data, sh_offset + rel, value & 0xFFFFFFFF)

  for rel_sec_idx in range(e_shnum):
    sh_name, sh_type, _, _, sh_offset, sh_size, sh_link, sh_info, _, sh_entsize = shdr(rel_sec_idx)
    if sh_type != 4:  # SHT_RELA
      continue
    if sh_entsize < 12 or sh_size == 0:
      continue

    _, tgt_type, tgt_flags, _, _, _, _, _, _, _ = shdr(sh_info)
    if not (tgt_flags & 0x2) or tgt_type == 8:  # SHF_ALLOC and not SHT_NOBITS
      continue

    rela_count = sh_size // sh_entsize
    hi_by_sym: dict[int, list[HiReloc]] = {}
    lo_relocs: list[LoReloc] = []

    for j in range(rela_count):
      roff = sh_offset + j * sh_entsize
      r_offset, r_info, r_addend = struct.unpack_from("<IIi", data, roff)
      r_type = r_info & 0xFF
      r_sym = r_info >> 8
      if r_type == R_RISCV_HI20:
        _, st_value, _, _, _, _ = sym(sh_link, r_sym)
        if is_text_addr(st_value):
          hi_by_sym.setdefault(r_sym, []).append((r_offset, r_addend))
      elif r_type in (R_RISCV_LO12_I, R_RISCV_LO12_S):
        lo_relocs.append((r_offset, r_sym, r_type))

    for sym_idx, items in hi_by_sym.items():
      items.sort(key=lambda x: x[0])

    for lo_offset, lo_sym, lo_type in lo_relocs:
      hi_list = hi_by_sym.get(lo_sym)
      if not hi_list:
        continue
      hi_offset, hi_addend = hi_list[0]
      for cand_off, cand_add in hi_list:
        if cand_off < lo_offset:
          hi_offset, hi_addend = cand_off, cand_add
        else:
          break

      _, st_value, _, _, _, _ = sym(sh_link, lo_sym)
      value = (st_value + hi_addend - hi_offset) & 0xFFFFFFFF

      hi_insn = read_u32_at_vaddr(sh_info, hi_offset)
      if (hi_insn & 0x7F) != 0x37:
        continue
      rd = (hi_insn >> 7) & 0x1F
      new_hi_imm = ((value + 0x800) >> 12) & 0xFFFFF
      new_hi = (new_hi_imm << 12) | (rd << 7) | 0x17
      write_u32_at_vaddr(sh_info, hi_offset, new_hi)

      lo_insn = read_u32_at_vaddr(sh_info, lo_offset)
      lo12 = value & 0xFFF
      if lo_type == R_RISCV_LO12_I:
        new_lo = (lo_insn & 0x000FFFFF) | (lo12 << 20)
      else:
        imm_4_0 = lo12 & 0x1F
        imm_11_5 = (lo12 >> 5) & 0x7F
        new_lo = (lo_insn & ~((0x7F << 25) | (0x1F << 7))) | (imm_11_5 << 25) | (imm_4_0 << 7)
      write_u32_at_vaddr(sh_info, lo_offset, new_lo)

  return bytes(data)

def generate_jal_instruction(target_addr: int) -> int:
  assert target_addr < 0x80000, f"target too far for JAL: {target_addr:#x}"
  opcode = 0x6F
  imm_10_1 = (target_addr & 0x7FE) << 20
  imm_11 = (target_addr & 0x800) << 9
  imm_19_12 = target_addr & 0xFF000
  return imm_19_12 | imm_11 | imm_10_1 | opcode

def pack_xip_elf(path: str | os.PathLike[str], xip_relocate: bool = False) -> tuple[bytes, int]:
  with open(os.fspath(path), "rb") as f:
    elf = f.read()
    if xip_relocate:
      elf = _xipify_riscv32_elf(elf)
  segs = list(iter_pt_load(elf))
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
    out[start : start + len(s.data)] = s.data
  text = next((s for s in l1 if (s.flags & 1) and s.data), l1[0])
  return bytes(out), len(text.data)
