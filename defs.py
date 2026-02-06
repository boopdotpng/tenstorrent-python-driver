# Hardware constants + kernel driver ioctl structs for Blackhole (p100a)
import ctypes
from ctypes import (
  LittleEndianStructure as S,
  c_uint8 as u8,
  c_uint16 as u16,
  c_uint32 as u32,
  c_uint64 as u64,
)
from enum import Enum

# -- TLB sizes ----------------------------------------------------------------

class TLBSize(Enum):
  MiB_2 = 1 << 21  # BAR 0: 201 available, for L1/registers
  GiB_4 = 1 << 32  # BAR 4: 8 available, for GDDR6 banks

# -- DRAM constants ------------------------------------------------------------

DRAM_BARRIER_BASE = 0x0
DRAM_ALIGNMENT = 64

# -- Tensix L1 memory map -----------------------------------------------------

class TensixL1:
  SIZE = 0x180000
  MAILBOX_BASE = 0x000060
  LAUNCH = MAILBOX_BASE + 0x000010
  GO_MSG = MAILBOX_BASE + 0x000310
  GO_MSG_INDEX = MAILBOX_BASE + 0x000340
  KERNEL_CONFIG_BASE = 0x0082B0
  BRISC_FIRMWARE_BASE = 0x003840
  NCRISC_FIRMWARE_BASE = 0x005440
  TRISC0_BASE = 0x005A40
  TRISC1_BASE = 0x006040
  TRISC2_BASE = 0x006640
  DATA_BUFFER_SPACE_BASE = 0x037000

  # Init scratch for local-mem relocation
  BRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x0082B0
  NCRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x00A2B0
  TRISC0_INIT_LOCAL_L1_BASE_SCRATCH = 0x00C2B0
  TRISC1_INIT_LOCAL_L1_BASE_SCRATCH = 0x00D2B0
  TRISC2_INIT_LOCAL_L1_BASE_SCRATCH = 0x00E2B0

  # Bank-to-NoC mapping tables (written by host, read by firmware)
  MEM_BANK_TO_NOC_SCRATCH = 0x0112B0

# -- Tensix MMIO registers ----------------------------------------------------

class TensixMMIO:
  LOCAL_RAM_START = 0xFFB00000
  LOCAL_RAM_END = 0xFFB01FFF
  RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0
  RISCV_DEBUG_REG_TRISC0_RESET_PC = 0xFFB12228
  RISCV_DEBUG_REG_TRISC1_RESET_PC = 0xFFB1222C
  RISCV_DEBUG_REG_TRISC2_RESET_PC = 0xFFB12230
  RISCV_DEBUG_REG_NCRISC_RESET_PC = 0xFFB12238
  SOFT_RESET_BRISC = 1 << 11
  SOFT_RESET_TRISC0 = 1 << 12
  SOFT_RESET_TRISC1 = 1 << 13
  SOFT_RESET_TRISC2 = 1 << 14
  SOFT_RESET_NCRISC = 1 << 18
  SOFT_RESET_ALL = 0x47800  # all 5 RISC-V cores
  SOFT_RESET_BRISC_ONLY_RUN = 0x47000  # keep TRISC/NCRISC in reset, release BRISC
  SOFT_RESET_BRISC_NCRISC_RUN = 0x7000  # keep TRISC in reset, release BRISC/NCRISC

# -- ARC -----------------------------------------------------------------------

class Arc:
  NOC_BASE = 0x80000000
  CSM_START = 0x10000000
  CSM_END = 0x1007FFFF
  RESET_UNIT_OFFSET = 0x30000
  SCRATCH_RAM_2 = RESET_UNIT_OFFSET + 0x408
  SCRATCH_RAM_11 = RESET_UNIT_OFFSET + 0x42C
  SCRATCH_RAM_13 = RESET_UNIT_OFFSET + 0x434
  TAG_GDDR_ENABLED = 36
  DEFAULT_GDDR_ENABLED = 0xFF

# -- DRAM banks ----------------------------------------------------------------

class Dram:
  BANK_COUNT = 8
  TILES_PER_BANK = 3
  DRAM_WRITE_OFFSET = 0x40
  BARRIER_FLAGS = (0xAA, 0xBB)
  BANK_TILE_YS = {
    0: (0, 1, 11),
    1: (2, 3, 10),
    2: (4, 8, 9),
    3: (5, 6, 7),
    4: (0, 1, 11),
    5: (2, 3, 10),
    6: (4, 8, 9),
    7: (5, 6, 7),
  }
  BANK_X = {b: 0 if b < 4 else 9 for b in range(8)}

# -- Kernel driver ioctls ------------------------------------------------------

TENSTORRENT_IOCTL_MAGIC = 0xFA
IOCTL_PIN_PAGES = 7
IOCTL_UNPIN_PAGES = 10
IOCTL_ALLOCATE_TLB = 11
IOCTL_FREE_TLB = 12
IOCTL_CONFIGURE_TLB = 13

PIN_PAGES_CONTIGUOUS = 1
PIN_PAGES_NOC_DMA = 2

# -- Ioctl structs -------------------------------------------------------------

class PinPagesIn(S):
  _fields_ = [("output_size_bytes", u32), ("flags", u32), ("virtual_address", u64), ("size", u64)]

class PinPagesOutExtended(S):
  _fields_ = [("physical_address", u64), ("noc_address", u64)]

class UnpinPagesIn(S):
  _fields_ = [("virtual_address", u64), ("size", u64), ("reserved", u64)]

class AllocateTlbIn(S):
  _fields_ = [("size", u64), ("reserved", u64)]

class AllocateTlbOut(S):
  _fields_ = [
    ("tlb_id", u32),
    ("reserved0", u32),
    ("mmap_offset_uc", u64),
    ("mmap_offset_wc", u64),
    ("reserved1", u64),
  ]

class FreeTlbIn(S):
  _fields_ = [("tlb_id", u32)]

class NocTlbConfig(S):
  _fields_ = [
    ("addr", u64),
    ("x_end", u16),
    ("y_end", u16),
    ("x_start", u16),
    ("y_start", u16),
    ("noc", u8),
    ("mcast", u8),
    ("ordering", u8),
    ("linked", u8),
    ("static_vc", u8),
    ("reserved0_0", u8),
    ("reserved0_1", u8),
    ("reserved0_2", u8),
    ("reserved1_0", u32),
    ("reserved1_1", u32),
  ]

class ConfigureTlbIn(S):
  _fields_ = [("tlb_id", u32), ("reserved", u32), ("config", NocTlbConfig)]

# -- Firmware dispatch structs -------------------------------------------------

def as_bytes(obj) -> bytes:
  return ctypes.string_at(ctypes.addressof(obj), ctypes.sizeof(obj))

class CB:
  NUM_CIRCULAR_BUFFERS = 32

class LocalCBConfig(S):
  _pack_ = 1
  _fields_ = [
    ("addr_bytes", u32),
    ("size_bytes", u32),
    ("num_pages", u32),
    ("page_size_bytes", u32),
  ]

class DevMsgs:
  RUN_MSG_INIT = 0x40
  RUN_MSG_GO = 0x80
  RUN_MSG_RESET_READ_PTR_FROM_HOST = 0xE0
  RUN_MSG_DONE = 0x00
  DISPATCH_MODE_HOST = 1
  ProgrammableCoreType_COUNT = 3
  MaxProcessorsPerCoreType = 5

class RtaOffset(S):
  _pack_ = 1
  _fields_ = [("rta_offset", u16), ("crta_offset", u16)]

class KernelConfigMsg(S):
  _pack_ = 1
  _fields_ = [
    ("kernel_config_base", u32 * DevMsgs.ProgrammableCoreType_COUNT),
    ("sem_offset", u16 * DevMsgs.ProgrammableCoreType_COUNT),
    ("local_cb_offset", u16),
    ("remote_cb_offset", u16),
    ("rta_offset", RtaOffset * DevMsgs.MaxProcessorsPerCoreType),
    ("mode", u8),
    ("pad2", u8),
    ("kernel_text_offset", u32 * DevMsgs.MaxProcessorsPerCoreType),
    ("local_cb_mask", u32),
    ("brisc_noc_id", u8),
    ("brisc_noc_mode", u8),
    ("min_remote_cb_start_index", u8),
    ("exit_erisc_kernel", u8),
    ("host_assigned_id", u32),
    ("enables", u32),
    ("watcher_kernel_ids", u16 * DevMsgs.MaxProcessorsPerCoreType),
    ("ncrisc_kernel_size16", u16),
    ("sub_device_origin_x", u8),
    ("sub_device_origin_y", u8),
    ("pad3", u8 * 1),
    ("preload", u8),
  ]

class LaunchMsg(S):
  _pack_ = 1
  _fields_ = [("kernel_config", KernelConfigMsg)]

class GoMsgBits(S):
  _pack_ = 1
  _fields_ = [
    ("dispatch_message_offset", u8),
    ("master_x", u8),
    ("master_y", u8),
    ("signal", u8),
  ]

class GoMsg(ctypes.Union):
  _pack_ = 1
  _fields_ = [("all", u32), ("bits", GoMsgBits)]

class FastDispatch:
  CQ_CMD_ALIGN = 16
  L1_ALIGNMENT = 16
  PCIE_ALIGNMENT = 64
  PCIE_NOC_BASE = 1 << 60
  MAX_HUGEPAGE_SIZE = 1 << 30
  MAX_DEV_CHANNEL_SIZE = 1 << 28
  DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE // MAX_DEV_CHANNEL_SIZE

  BH_TENSIX_DEFAULT_UNRESERVED = 0x196C0
  BH_PREFETCH_Q_RD_PTR_OFF = 0x00
  BH_PREFETCH_Q_PCIE_RD_PTR_OFF = 0x04
  BH_COMPLETION_Q_WR_PTR_OFF = 0x10
  BH_COMPLETION_Q_RD_PTR_OFF = 0x20
  BH_COMPLETION_Q0_LAST_EVENT_PTR_OFF = 0x30
  BH_COMPLETION_Q1_LAST_EVENT_PTR_OFF = 0x40
  BH_DISPATCH_S_SYNC_SEM_OFF = 0x50
  BH_FABRIC_HEADER_RB_OFF = 0xD0
  BH_FABRIC_SYNC_STATUS_OFF = 0x150
  BH_UNRESERVED_OFF = 0x180

  HOST_ISSUE_Q_RD_OFF = 0 * PCIE_ALIGNMENT
  HOST_ISSUE_Q_WR_OFF = 1 * PCIE_ALIGNMENT
  HOST_COMPLETION_Q_WR_OFF = 2 * PCIE_ALIGNMENT
  HOST_COMPLETION_Q_RD_OFF = 3 * PCIE_ALIGNMENT
  HOST_UNRESERVED_OFF = 4 * PCIE_ALIGNMENT

  PREFETCH_Q_ENTRY_BYTES = 2
  PREFETCH_Q_ENTRIES_WORKER_DEFAULT = 1534
  PREFETCH_CMDDAT_Q_SIZE = 256 * 1024
  PREFETCH_SCRATCH_DB_SIZE = 128 * 1024

class CQPrefetchCmdId:
  RELAY_INLINE = 5

class CQDispatchCmdId:
  WRITE_LINEAR = 1

class CQPrefetchRelayInlineCmd(S):
  _pack_ = 1
  _fields_ = [("dispatcher_type", u8), ("pad", u16), ("length", u32), ("stride", u32)]

class CQPrefetchCmdPayload(ctypes.Union):
  _pack_ = 1
  _fields_ = [("relay_inline", CQPrefetchRelayInlineCmd), ("raw", u8 * 15)]

class CQPrefetchCmd(S):
  _pack_ = 1
  _fields_ = [("cmd_id", u8), ("payload", CQPrefetchCmdPayload)]

class CQDispatchWriteCmd(S):
  _pack_ = 1
  _fields_ = [
    ("num_mcast_dests", u8),
    ("write_offset_index", u8),
    ("pad1", u8),
    ("noc_xy_addr", u32),
    ("addr", u64),
    ("length", u64),
  ]

class CQDispatchCmdLargePayload(ctypes.Union):
  _pack_ = 1
  _fields_ = [("write_linear", CQDispatchWriteCmd), ("raw", u8 * 31)]

class CQDispatchCmdLarge(S):
  _pack_ = 1
  _fields_ = [("cmd_id", u8), ("payload", CQDispatchCmdLargePayload)]
