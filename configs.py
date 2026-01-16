# Blackhole tile memory maps and constants
# All addresses are offsets within the tile's local address space

from enum import Enum

class TLBSize(Enum):
  MiB_2 = 1 << 21  # BAR 0: 201 available, for L1/registers
  GiB_4 = 1 << 32  # BAR 4: 8 available, for GDDR6 banks

class TensixL1:
  """Tensix tile L1 memory map (1536 KiB)"""
  SIZE = 0x180000

  # Firmware + system-reserved regions (Blackhole, from tt-metal dev_mem_map.h)
  MAILBOX_BASE = 0x000060
  MAILBOX_SIZE = 0x0031e0  # 12768
  ZEROS_BASE = 0x003240
  ZEROS_SIZE = 0x000200
  LLK_DEBUG_BASE = 0x003440
  LLK_DEBUG_SIZE = 0x000400

  BRISC_FIRMWARE_BASE = 0x003840
  NCRISC_FIRMWARE_BASE = 0x005440
  NCRISC_L1_CODE_BASE = 0x009000
  NCRISC_LOCAL_MEM_BASE = 0x00c000
  TRISC0_BASE = 0x005a40
  TRISC0_LOCAL_MEM_BASE = 0x011000
  TRISC1_BASE = 0x006040
  TRISC1_LOCAL_MEM_BASE = 0x015000
  TRISC2_BASE = 0x006640
  TRISC2_LOCAL_MEM_BASE = 0x01a000

  # Runtime config
  EPOCH_RUNTIME_CONFIG_BASE = 0x023000
  OVERLAY_BLOB_BASE = 0x023080

  # NCRISC runtime
  NCRISC_L1_RUNTIME_SECTION_BASE = 0x033000
  NCRISC_L1_SCRATCH_BASE = 0x033200
  NCRISC_L1_CONTEXT_BASE = 0x033020
  NCRISC_L1_DRAM_POLLING_CTRL_BASE = 0x033040
  NCRISC_PERF_QUEUE_HEADER_ADDR = 0x034000
  NCRISC_L1_PERF_BUF_BASE = 0x034040
  NCRISC_L1_EPOCH_Q_BASE = 0x035000

  # Data/CB space
  DATA_BUFFER_SPACE_BASE = 0x037000
  L1_BARRIER_BASE = 0x16dfc0

  # Init scratch used for local-mem relocation (tt-metal dev_mem_map.h)
  INIT_LOCAL_MAP_END = 0x0082b0
  BRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x0082b0
  NCRISC_INIT_LOCAL_L1_BASE_SCRATCH = 0x00a2b0
  TRISC0_INIT_LOCAL_L1_BASE_SCRATCH = 0x00c2b0
  TRISC1_INIT_LOCAL_L1_BASE_SCRATCH = 0x00d2b0
  TRISC2_INIT_LOCAL_L1_BASE_SCRATCH = 0x00e2b0

class TensixMMIO:
  """Tensix tile-local MMIO / register space (tile-local view)"""
  LOCAL_RAM_START = 0xFFB00000
  LOCAL_RAM_END = 0xFFB01FFF
  NOC0_NIU_START = 0xFFB20000
  NOC1_NIU_START = 0xFFB30000
  RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0
  SOFT_RESET_ALL = 0x47800  # reset all 5 RISC-V cores (brisc, ncrisc, trisc0-2)


class Arc:
  """ARC tile constants"""
  NOC_BASE = 0x80000000  # ARC NoC xbar base

  # Control system memory bounds
  CSM_START = 0x10000000
  CSM_END = 0x1007FFFF

  # Scratch registers (offsets from ARC reset unit)
  RESET_UNIT_OFFSET = 0x30000
  SCRATCH_RAM_2 = RESET_UNIT_OFFSET + 0x408   # boot status
  SCRATCH_RAM_11 = RESET_UNIT_OFFSET + 0x42C  # msg queue ctrl block ptr
  SCRATCH_RAM_12 = RESET_UNIT_OFFSET + 0x430  # telemetry table ptr
  SCRATCH_RAM_13 = RESET_UNIT_OFFSET + 0x434  # telemetry data ptr

  # Telemetry tags
  TAG_TENSIX_ENABLED = 34
  TAG_ETH_ENABLED = 35
  TAG_GDDR_ENABLED = 36
  TAG_PCIE_USAGE = 38

  # Default telemetry values (all enabled)
  DEFAULT_TENSIX_ENABLED = 0x3FFF  # 14 tensix columns
  DEFAULT_ETH_ENABLED = 0x3FFF    # 14 ethernet cores
  DEFAULT_GDDR_ENABLED = 0xFF     # 8 DRAM banks
  DEFAULT_PCIE_USAGE = 0x5        # both PCIe instances enabled


class Dram:
  """DRAM tile constants"""
  BANK_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB per bank
  BANK_COUNT = 8
  TILES_PER_BANK = 3

  # Y-coordinates for each bank's 3 tiles (all 3 expose same 4 GiB)
  BANK_TILE_YS = {
    0: (0, 1, 11), 1: (2, 3, 10), 2: (4, 8, 9), 3: (5, 6, 7),
    4: (0, 1, 11), 5: (2, 3, 10), 6: (4, 8, 9), 7: (5, 6, 7),
  }
  # Banks 0-3 at x=0, banks 4-7 at x=9
  BANK_X = {b: 0 if b < 4 else 9 for b in range(8)}


class EthL1:
  """Ethernet tile L1 memory map"""
  FIRMWARE_BASE = 0x009040
  L1_EPOCH_Q_BASE = 0x009000
  L1_DRAM_POLLING_CTRL_BASE = 0x009020
  COMMAND_Q_BASE = 0x011000
  DATA_BUFFER_BASE = 0x012000
  TILE_HEADER_BUFFER_BASE = 0x018000
  EPOCH_RUNTIME_CONFIG_BASE = 0x020000
  OVERLAY_BLOB_BASE = 0x020080
  DATA_BUFFER_SPACE_BASE = 0x028000
  ERISC_BARRIER_BASE = 0x011fe0


# Harvesting: firmware stores tensix columns in this order (left/right alternating)
# The bitmask from telemetry is applied to this ordering
HARVESTING_NOC_LOCATIONS = (1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10)
