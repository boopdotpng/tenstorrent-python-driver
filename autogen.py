import ctypes
from ctypes import (
  LittleEndianStructure as _S,
  c_uint8 as _u8,
  c_uint16 as _u16,
  c_uint32 as _u32,
  c_uint64 as _u64,
)
S, u8, u16, u32, u64 = _S, _u8, _u16, _u32, _u64

DRAM_BARRIER_BASE = 0x0
DRAM_ALIGNMENT = 64

Core = tuple[int, int]
DramTile = tuple[int, int, int]
CoreList = list[Core]
DramTileList = list[DramTile]

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

  # Profiler region (inside mailboxes_t)
  # Derived from tt-metal dev_msgs/dev_mem_map for p100a (blackhole):
  #   MEM_MAILBOX_BASE = 96 (0x60)
  #   offsetof(mailboxes_t, profiler.control_vector) = 2400 (0x960)
  #   offsetof(mailboxes_t, profiler.buffer) = 2528 (0x9e0)
  # Effective addresses:
  #   control = 0x60 + 0x960 = 0x9c0
  #   buffer  = 0x60 + 0x9e0 = 0xa40
  PROFILER_CONTROL = 0x0009C0       # 32 × u32 = 128 bytes
  PROFILER_BUFFERS = 0x000A40       # 5 × 512 × u32 = 10240 bytes (one per RISC)
  PROFILER_BUF_STRIDE = 2048        # bytes per RISC data buffer
  PROFILER_HOST_BUFFER_BYTES_PER_RISC = 65536

  BRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x0082B0
  NCRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x00A2B0
  TRISC0_INIT_LOCAL_L1_BASE_SCRATCH = 0x00C2B0
  TRISC1_INIT_LOCAL_L1_BASE_SCRATCH = 0x00D2B0
  TRISC2_INIT_LOCAL_L1_BASE_SCRATCH = 0x00E2B0

  MEM_BANK_TO_NOC_SCRATCH = 0x0112B0

class TensixMMIO:
  LOCAL_RAM_START = 0xFFB00000
  LOCAL_RAM_END = 0xFFB01FFF
  RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0
  RISCV_DEBUG_REG_TRISC0_RESET_PC = 0xFFB12228
  RISCV_DEBUG_REG_TRISC1_RESET_PC = 0xFFB1222C
  RISCV_DEBUG_REG_TRISC2_RESET_PC = 0xFFB12230
  RISCV_DEBUG_REG_NCRISC_RESET_PC = 0xFFB12238
  SOFT_RESET_ALL = 0x47800  # all 5 RISC-V cores
  SOFT_RESET_BRISC_ONLY_RUN = 0x47000  # keep TRISC/NCRISC in reset, release BRISC

class Arc:
  NOC_BASE = 0x80000000
  RESET_UNIT_OFFSET = 0x30000
  SCRATCH_RAM_2 = RESET_UNIT_OFFSET + 0x408
  SCRATCH_RAM_11 = RESET_UNIT_OFFSET + 0x42C
  SCRATCH_RAM_13 = RESET_UNIT_OFFSET + 0x434
  MSG_AICLK_GO_BUSY = 0x52
  MSG_AICLK_GO_LONG_IDLE = 0x54
  TAG_AICLK = 14
  TAG_GDDR_ENABLED = 36
  DEFAULT_AICLK = 800
  DEFAULT_GDDR_ENABLED = 0xFF

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

class Profiler:
  # ControlBuffer enum indices (into 32×u32 control block, from profiler_common.h)
  HOST_BUF_END = 0       # per RISC (indices 0-4)
  DEVICE_BUF_END = 5     # per RISC (indices 5-9)
  DROPPED = 18
  DONE = 19
  # BufferIndex guaranteed marker positions (u32 indices within a run)
  GUARANTEED_FW_START = 4
  GUARANTEED_FW_END = 6
  GUARANTEED_KERN_START = 8
  GUARANTEED_KERN_END = 10
  CUSTOM_START = 12
  # Marker packet types
  ZONE_START = 0
  ZONE_END = 1
  ZONE_TOTAL = 2
  TS_DATA = 3
  TS_DATA_16B = 5
  # Derived
  HOST_BUF_BYTES_PER_RISC = TensixL1.PROFILER_HOST_BUFFER_BYTES_PER_RISC
  HOST_BUF_WORDS_PER_RISC = HOST_BUF_BYTES_PER_RISC // 4

TENSTORRENT_IOCTL_MAGIC = 0xFA
def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr
IOCTL_PIN_PAGES = _IO(7)
IOCTL_UNPIN_PAGES = _IO(10)
IOCTL_ALLOCATE_TLB = _IO(11)
IOCTL_FREE_TLB = _IO(12)
IOCTL_CONFIGURE_TLB = _IO(13)

# --- ioctl structs (match kernel header: /usr/src/tenstorrent-*/ioctl.h) ---

class AllocateTlbIn(S):
  _pack_ = 1
  _fields_ = [("size", u64), ("reserved", u64)]

class AllocateTlbOut(S):
  _pack_ = 1
  _fields_ = [("id", u32), ("reserved0", u32), ("mmap_offset_uc", u64), ("mmap_offset_wc", u64), ("reserved1", u64)]

class AllocateTlb(S):
  _pack_ = 1
  _fields_ = [("input", AllocateTlbIn), ("output", AllocateTlbOut)]

class FreeTlbIn(S):
  _pack_ = 1
  _fields_ = [("id", u32)]

class NocTlbConfig(S):
  _pack_ = 1
  _fields_ = [
    ("addr", u64),
    ("x_end", u16), ("y_end", u16), ("x_start", u16), ("y_start", u16),
    ("noc", u8), ("mcast", u8), ("ordering", u8), ("linked", u8),
    ("static_vc", u8), ("reserved0", u8 * 3),
    ("reserved1", u32 * 2),
  ]

class ConfigureTlbIn(S):
  _pack_ = 1
  _fields_ = [("id", u32), ("reserved", u32), ("config", NocTlbConfig)]

PIN_PAGES_NOC_DMA = 2

class PinPagesIn(S):
  _pack_ = 1
  _fields_ = [("output_size_bytes", u32), ("flags", u32), ("virtual_address", u64), ("size", u64)]

class PinPagesOut(S):
  _pack_ = 1
  _fields_ = [("physical_address", u64), ("noc_address", u64)]

class PinPages(S):
  _pack_ = 1
  _fields_ = [("input", PinPagesIn), ("output", PinPagesOut)]

class UnpinPagesIn(S):
  _pack_ = 1
  _fields_ = [("virtual_address", u64), ("size", u64), ("reserved", u64)]

def as_bytes(obj) -> bytes:
  return ctypes.string_at(ctypes.addressof(obj), ctypes.sizeof(obj))

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
  DISPATCH_MODE_DEV = 0
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

  BH_TENSIX_DEFAULT_UNRESERVED = 0x196C0
  BH_PREFETCH_Q_RD_PTR_OFF = 0x00
  BH_PREFETCH_Q_PCIE_RD_PTR_OFF = 0x04
  BH_COMPLETION_Q_WR_PTR_OFF = 0x10
  BH_COMPLETION_Q_RD_PTR_OFF = 0x20
  BH_COMPLETION_Q0_LAST_EVENT_PTR_OFF = 0x30
  BH_COMPLETION_Q1_LAST_EVENT_PTR_OFF = 0x40
  BH_DISPATCH_S_SYNC_SEM_OFF = 0x50
  BH_UNRESERVED_OFF = 0x180

  HOST_COMPLETION_Q_WR_OFF = 2 * PCIE_ALIGNMENT
  HOST_COMPLETION_Q_RD_OFF = 3 * PCIE_ALIGNMENT
  HOST_UNRESERVED_OFF = 4 * PCIE_ALIGNMENT

  PREFETCH_Q_ENTRY_BYTES = 2
  PREFETCH_Q_ENTRIES_WORKER_DEFAULT = 1534

def align_up(value: int, align: int) -> int:
  return (value + align - 1) // align * align

def align_down(value: int, alignment: int) -> tuple[int, int]:
  base = value & ~(alignment - 1)
  return base, value - base

PAGE_SIZE = 4096

HOST_SYSMEM_SIZE = align_up(128 * 1024 * 1024, PAGE_SIZE)
HOST_ISSUE_BASE = FastDispatch.HOST_UNRESERVED_OFF
HOST_ISSUE_SIZE = align_up(64 * 1024 * 1024, FastDispatch.PCIE_ALIGNMENT)
HOST_COMPLETION_BASE = HOST_ISSUE_BASE + HOST_ISSUE_SIZE
HOST_COMPLETION_SIZE = align_up(32 * 1024 * 1024, FastDispatch.PCIE_ALIGNMENT)
if HOST_COMPLETION_BASE + HOST_COMPLETION_SIZE > HOST_SYSMEM_SIZE:
  raise ValueError(
    f"sysmem_size too small: need {HOST_COMPLETION_BASE + HOST_COMPLETION_SIZE}, have {HOST_SYSMEM_SIZE}"
  )

DEV_L1_BASE = FastDispatch.BH_TENSIX_DEFAULT_UNRESERVED
DEV_PREFETCH_Q_BASE = DEV_L1_BASE + FastDispatch.BH_UNRESERVED_OFF
DEV_PREFETCH_Q_SIZE = FastDispatch.PREFETCH_Q_ENTRIES_WORKER_DEFAULT * FastDispatch.PREFETCH_Q_ENTRY_BYTES
DEV_PREFETCH_Q_RD_PTR_ADDR = DEV_L1_BASE + FastDispatch.BH_PREFETCH_Q_RD_PTR_OFF
DEV_PREFETCH_Q_PCIE_RD_PTR_ADDR = DEV_L1_BASE + FastDispatch.BH_PREFETCH_Q_PCIE_RD_PTR_OFF
DEV_COMPLETION_Q_WR_PTR_ADDR = DEV_L1_BASE + FastDispatch.BH_COMPLETION_Q_WR_PTR_OFF
DEV_COMPLETION_Q_RD_PTR_ADDR = DEV_L1_BASE + FastDispatch.BH_COMPLETION_Q_RD_PTR_OFF
DEV_COMPLETION_Q0_LAST_EVENT_PTR_ADDR = DEV_L1_BASE + FastDispatch.BH_COMPLETION_Q0_LAST_EVENT_PTR_OFF
DEV_COMPLETION_Q1_LAST_EVENT_PTR_ADDR = DEV_L1_BASE + FastDispatch.BH_COMPLETION_Q1_LAST_EVENT_PTR_OFF
DEV_DISPATCH_S_SYNC_SEM_ADDR = DEV_L1_BASE + FastDispatch.BH_DISPATCH_S_SYNC_SEM_OFF
DEV_DISPATCH_CB_PAGES = (512 * 1024) >> 12

CQ_PREFETCH_CMD_RELAY_INLINE = 5
CQ_DISPATCH_CMD_WRITE_LINEAR = 1
CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST = 3
CQ_DISPATCH_CMD_WRITE_PACKED = 5
CQ_DISPATCH_CMD_WRITE_PACKED_LARGE = 6
CQ_DISPATCH_CMD_WAIT = 7
CQ_DISPATCH_CMD_SEND_GO_SIGNAL = 14
CQ_DISPATCH_CMD_SET_GO_SIGNAL_NOC_DATA = 17
CQ_DISPATCH_CMD_TIMESTAMP = 18

TIMESTAMP_PAGE_SIZE = 64   # DRAM_ALIGNMENT, 4 timestamp slots per page at TIMESTAMP_STRIDE=16
TIMESTAMP_STRIDE = 16      # keep 8-byte timestamp payload on 16-byte aligned destinations
TIMESTAMP_MAX_SLOTS = 512  # max timestamps per batch (256 programs)

CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE = 0x02
CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK = 0x01
CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS = 35
CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM = 0x08
CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM = 0x10
CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET = 0xFF

__all__ = [name for name in globals() if not name.startswith("_") and name not in ("S", "u8", "u16", "u32", "u64")]
