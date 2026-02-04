# Hardware constants + kernel driver ioctl structs for Blackhole (p100a)
import ctypes
from ctypes import (
    LittleEndianStructure as S,
    c_uint8 as u8,
    c_uint16 as u16,
    c_uint32 as u32,
    c_uint64 as u64,
    sizeof,
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
    NOC0_NIU_START = 0xFFB20000
    NOC1_NIU_START = 0xFFB30000
    RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0
    RISCV_DEBUG_REG_TRISC0_RESET_PC = 0xFFB12228
    RISCV_DEBUG_REG_TRISC1_RESET_PC = 0xFFB1222C
    RISCV_DEBUG_REG_TRISC2_RESET_PC = 0xFFB12230
    RISCV_DEBUG_REG_NCRISC_RESET_PC = 0xFFB12238
    SOFT_RESET_ALL = 0x47800  # all 5 RISC-V cores
    SOFT_RESET_BRISC_ONLY_RUN = 0x47000  # keep TRISC/NCRISC in reset, release BRISC

# -- NoC NIU -------------------------------------------------------------------

class NocNIU:
    NIU_CFG_0_NOC_ID_TRANSLATE_EN = 14

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
IOCTL_GET_DEVICE_INFO = 0
IOCTL_QUERY_MAPPINGS = 2
IOCTL_RESET_DEVICE = 6
IOCTL_ALLOCATE_TLB = 11
IOCTL_FREE_TLB = 12
IOCTL_CONFIGURE_TLB = 13

TENSTORRENT_RESET_DEVICE_ASIC_RESET = 4
TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET = 5
TENSTORRENT_RESET_DEVICE_POST_RESET = 6

# -- Ioctl structs -------------------------------------------------------------

class QueryMappingsIn(S):
    _fields_ = [("output_mapping_count", u32), ("reserved", u32)]

class TenstorrentMapping(S):
    _fields_ = [
        ("mapping_id", u32),
        ("reserved", u32),
        ("mapping_base", u64),
        ("mapping_size", u64),
    ]

class ResetDeviceIn(S):
    _fields_ = [("output_size_bytes", u32), ("flags", u32)]

class ResetDeviceOut(S):
    _fields_ = [("output_size_bytes", u32), ("result", u32)]

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

class TenstorrentGetDeviceInfoIn(S):
    _fields_ = [("output_size_bytes", u32)]

class TenstorrentGetDeviceInfoOut(S):
    _fields_ = [
        ("output_size_bytes", u32),
        ("vendor_id", u16),
        ("device_id", u16),
        ("subsystem_vendor_id", u16),
        ("subsystem_id", u16),
        ("bus_dev_fn", u16),
        ("max_dma_buf_size_log2", u16),
        ("pci_domain", u16),
        ("reserved", u16),
    ]

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
