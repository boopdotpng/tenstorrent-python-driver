import ctypes
from ctypes import LittleEndianStructure as S, c_uint8 as u8, c_uint16 as u16, c_uint32 as u32, c_uint64 as u64, sizeof

TENSTORRENT_IOCTL_MAGIC = 0xFA
IOCTL_GET_DEVICE_INFO = 0
IOCTL_QUERY_MAPPINGS = 2
IOCTL_RESET_DEVICE = 6
IOCTL_PIN_PAGES = 7
IOCTL_UNPIN_PAGES = 10
IOCTL_ALLOCATE_TLB = 11
IOCTL_FREE_TLB = 12
IOCTL_CONFIGURE_TLB = 13

TENSTORRENT_RESET_DEVICE_ASIC_RESET = 4
TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET = 5
TENSTORRENT_RESET_DEVICE_POST_RESET = 6

TENSTORRENT_PIN_PAGES_CONTIGUOUS = 1
TENSTORRENT_PIN_PAGES_NOC_DMA = 2
TENSTORRENT_PIN_PAGES_NOC_TOP_DOWN = 4

class QueryMappingsIn(S):
  _fields_ = [("output_mapping_count", u32), ("reserved", u32)]

class TenstorrentMapping(S):
  _fields_ = [("mapping_id", u32), ("reserved", u32), ("mapping_base", u64), ("mapping_size", u64)]

class ResetDeviceIn(S):
  _fields_ = [("output_size_bytes", u32), ("flags", u32)]

class ResetDeviceOut(S):
  _fields_ = [("output_size_bytes", u32), ("result", u32)]

class PinPagesIn(S):
  _fields_ = [("output_size_bytes", u32), ("flags", u32), ("virtual_address", u64), ("size", u64)]

class PinPagesOut(S):
  _fields_ = [("physical_address", u64)]

class PinPagesOutExtended(S):
  _fields_ = [("physical_address", u64), ("noc_address", u64)]

class UnpinPagesIn(S):
  _fields_ = [("virtual_address", u64), ("size", u64), ("reserved", u64)]

class AllocateTlbIn(S):
  _fields_ = [("size", u64), ("reserved", u64)]

class AllocateTlbOut(S):
  _fields_ = [("tlb_id", u32), ("reserved0", u32), ("mmap_offset_uc", u64), ("mmap_offset_wc", u64), ("reserved1", u64)]

class FreeTlbIn(S):
  _fields_ = [("tlb_id", u32)]

class NocTlbConfig(S):
  _fields_ = [
    ("addr", u64), ("x_end", u16), ("y_end", u16), ("x_start", u16), ("y_start", u16),
    ("noc", u8), ("mcast", u8), ("ordering", u8), ("linked", u8), ("static_vc", u8),
    ("reserved0_0", u8), ("reserved0_1", u8), ("reserved0_2", u8), ("reserved1_0", u32), ("reserved1_1", u32),
  ]

class ConfigureTlbIn(S):
  _fields_ = [("tlb_id", u32), ("reserved", u32), ("config", NocTlbConfig)]

class TenstorrentGetDeviceInfoIn(S):
  _fields_ = [("output_size_bytes", u32)]

class TenstorrentGetDeviceInfoOut(S):
  _fields_ = [
    ("output_size_bytes", u32), ("vendor_id", u16), ("device_id", u16), ("subsystem_vendor_id", u16),
    ("subsystem_id", u16), ("bus_dev_fn", u16), ("max_dma_buf_size_log2", u16), ("pci_domain", u16), ("reserved", u16),
  ]

def as_bytes(obj: ctypes.Structure | ctypes.Union) -> bytes:
  return ctypes.string_at(ctypes.addressof(obj), ctypes.sizeof(obj))

class CB:
  NUM_CIRCULAR_BUFFERS = 32
  UINT32_WORDS_PER_LOCAL_CONFIG = 4
  UINT32_WORDS_PER_REMOTE_CONFIG = 2
  COMPUTE_ADDR_SHIFT = 4  # device shifts byte addrs by 4 (16B words)

class LocalCBConfig(S):
  _pack_ = 1
  _fields_ = [
    ("addr_bytes", u32),
    ("size_bytes", u32),
    ("num_pages", u32),
    ("page_size_bytes", u32),
  ]

class RemoteCBConfig(S):
  _pack_ = 1
  _fields_ = [
    ("config_address", u32),
    ("page_size_bytes", u32),
  ]

class DevMsgs:
  RUN_MSG_INIT = 0x40
  RUN_MSG_GO = 0x80
  RUN_MSG_RESET_READ_PTR = 0xC0
  RUN_MSG_RESET_READ_PTR_FROM_HOST = 0xE0
  RUN_MSG_DONE = 0x00

  DISPATCH_MODE_DEV = 0
  DISPATCH_MODE_HOST = 1

  ProgrammableCoreType_COUNT = 3
  MaxProcessorsPerCoreType = 5

class RtaOffset(S):
  _pack_ = 1
  _fields_ = [
    ("rta_offset", u16),
    ("crta_offset", u16),
  ]

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
  _fields_ = [
    ("all", u32),
    ("bits", GoMsgBits),
  ]

# --- Fast Dispatch ---

class FastDispatch:
  CQ_CMD_ALIGN = 16
  L1_ALIGNMENT = 16
  PCIE_ALIGNMENT = 64
  DRAM_ALIGNMENT = 64

  NOC_ADDR_NODE_ID_BITS = 6
  NOC_XY_ENCODING_SHIFT = NOC_ADDR_NODE_ID_BITS
  PCIE_NOC_BASE = 1 << 60

  MAX_HUGEPAGE_SIZE = 1 << 30
  MAX_DEV_CHANNEL_SIZE = 1 << 28
  DEVICES_PER_UMD_CHANNEL = MAX_HUGEPAGE_SIZE // MAX_DEV_CHANNEL_SIZE

  BH_TENSIX_DEFAULT_UNRESERVED = 0x196C0

  # Device CQ L1 layout, relative to DEFAULT_UNRESERVED
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

  # Host CQ sysmem layout (CommandQueueHostAddrType * PCIE_ALIGNMENT)
  HOST_ISSUE_Q_RD_OFF = 0 * PCIE_ALIGNMENT
  HOST_ISSUE_Q_WR_OFF = 1 * PCIE_ALIGNMENT
  HOST_COMPLETION_Q_WR_OFF = 2 * PCIE_ALIGNMENT
  HOST_COMPLETION_Q_RD_OFF = 3 * PCIE_ALIGNMENT
  HOST_UNRESERVED_OFF = 4 * PCIE_ALIGNMENT

  # Default worker settings
  PREFETCH_Q_ENTRY_BYTES = 2
  PREFETCH_Q_ENTRIES_WORKER_DEFAULT = 1534
  PREFETCH_MAX_CMD_SIZE = 128 * 1024
  PREFETCH_CMDDAT_Q_SIZE = 256 * 1024
  PREFETCH_SCRATCH_DB_SIZE = 128 * 1024
  PREFETCH_RINGBUFFER_SIZE = 1024 * 1024

class CQPrefetchCmdId:
  ILLEGAL = 0
  RELAY_LINEAR = 1
  RELAY_LINEAR_H = 2
  RELAY_PAGED = 3
  RELAY_PAGED_PACKED = 4
  RELAY_INLINE = 5
  RELAY_INLINE_NOFLUSH = 6
  EXEC_BUF = 7
  EXEC_BUF_END = 8
  STALL = 9
  DEBUG = 10
  TERMINATE = 11
  PAGED_TO_RINGBUFFER = 12
  SET_RINGBUFFER_OFFSET = 13
  RELAY_RINGBUFFER = 14

class CQDispatchCmdId:
  ILLEGAL = 0
  WRITE_LINEAR = 1
  WRITE_LINEAR_H = 2
  WRITE_LINEAR_H_HOST = 3
  WRITE_PAGED = 4
  WRITE_PACKED = 5
  WRITE_PACKED_LARGE = 6
  WAIT = 7
  SINK = 8
  DEBUG = 9
  DELAY = 10
  EXEC_BUF_END = 11
  SET_WRITE_OFFSET = 12
  TERMINATE = 13
  SEND_GO_SIGNAL = 14
  NOTIFY_SUBORDINATE_GO_SIGNAL = 15
  SET_NUM_WORKER_SEMS = 16
  SET_GO_SIGNAL_NOC_DATA = 17

# --- Prefetch command structures ---

class CQGenericDebugCmd(S):
  _pack_ = 1
  _fields_ = [("pad", u8), ("key", u16), ("size", u32), ("stride", u32)]

class CQPrefetchRelayInlineCmd(S):
  _pack_ = 1
  _fields_ = [("dispatcher_type", u8), ("pad", u16), ("length", u32), ("stride", u32)]

class CQPrefetchRelayLinearCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u16), ("length", u64), ("noc_xy_addr", u32), ("addr", u64)]

class CQPrefetchRelayLinearHCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("pad2", u16), ("noc_xy_addr", u32), ("addr", u64), ("length", u32)]

class CQPrefetchRelayPagedCmd(S):
  _pack_ = 1
  _fields_ = [("start_page", u8), ("is_dram_and_length_adjust", u16), ("base_addr", u32), ("page_size", u32), ("pages", u32)]

class CQPrefetchRelayPagedPackedCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("count", u16), ("total_length", u32), ("stride", u32)]

class CQPrefetchRelayPagedPackedSubCmd(S):
  _pack_ = 1
  _fields_ = [("start_page", u16), ("log_page_size", u16), ("base_addr", u32), ("length", u32)]

class CQPrefetchExecBufCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("pad2", u16), ("base_addr", u32), ("log_page_size", u32), ("pages", u32)]

class CQPrefetchPagedToRingbufferCmd(S):
  _pack_ = 1
  _fields_ = [("flags", u8), ("log2_page_size", u8), ("start_page", u8), ("wp_offset_update", u32), ("base_addr", u32), ("length", u32)]

class CQPrefetchSetRingbufferOffsetCmd(S):
  _pack_ = 1
  _fields_ = [("offset", u32), ("pad1", u16), ("pad2", u8), ("update_wp", u8)]

class CQPrefetchRelayRingbufferCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("count", u16), ("stride", u32)]

class CQPrefetchRelayRingbufferSubCmd(S):
  _pack_ = 1
  _fields_ = [("start", u32), ("length", u32)]

class CQPrefetchCmdPayload(ctypes.Union):
  _pack_ = 1
  _fields_ = [
    ("relay_paged", CQPrefetchRelayPagedCmd),
    ("relay_paged_packed", CQPrefetchRelayPagedPackedCmd),
    ("relay_inline", CQPrefetchRelayInlineCmd),
    ("exec_buf", CQPrefetchExecBufCmd),
    ("debug", CQGenericDebugCmd),
    ("paged_to_ringbuffer", CQPrefetchPagedToRingbufferCmd),
    ("set_ringbuffer_offset", CQPrefetchSetRingbufferOffsetCmd),
    ("relay_ringbuffer", CQPrefetchRelayRingbufferCmd),
    ("raw", u8 * 15),
  ]

class CQPrefetchCmd(S):
  _pack_ = 1
  _fields_ = [("cmd_id", u8), ("payload", CQPrefetchCmdPayload)]

class CQPrefetchCmdLargePayload(ctypes.Union):
  _pack_ = 1
  _fields_ = [
    ("relay_linear_h", CQPrefetchRelayLinearHCmd),
    ("relay_linear", CQPrefetchRelayLinearCmd),
    ("raw", u8 * 31),
  ]

class CQPrefetchCmdLarge(S):
  _pack_ = 1
  _fields_ = [("cmd_id", u8), ("payload", CQPrefetchCmdLargePayload)]

# --- Dispatch command structures ---

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

class CQDispatchWriteHostCmd(S):
  _pack_ = 1
  _fields_ = [("is_event", u8), ("pad1", u16), ("pad2", u32), ("length", u64)]

class CQDispatchWritePagedCmd(S):
  _pack_ = 1
  _fields_ = [("is_dram", u8), ("start_page", u16), ("base_addr", u32), ("page_size", u32), ("pages", u32)]

class CQDispatchWritePackedCmd(S):
  _pack_ = 1
  _fields_ = [("flags", u8), ("count", u16), ("write_offset_index", u16), ("size", u16), ("addr", u32)]

class CQDispatchWaitCmd(S):
  _pack_ = 1
  _fields_ = [("flags", u8), ("stream", u16), ("addr", u32), ("count", u32)]

class CQDispatchDelayCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("pad2", u16), ("delay", u32)]

class CQDispatchSetWriteOffsetCmd(S):
  _pack_ = 1
  _fields_ = [("offset_count", u8), ("program_host_id", u16)]

class CQDispatchGoSignalMcastCmd(S):
  _pack_ = 1
  _fields_ = [
    ("go_signal", u32),
    ("multicast_go_offset", u8),
    ("num_unicast_txns", u8),
    ("noc_data_start_index", u8),
    ("wait_count", u32),
    ("wait_stream", u32),
  ]

class CQDispatchNotifySubordinateGoSignalCmd(S):
  _pack_ = 1
  _fields_ = [("wait", u8), ("index_bitmask", u16), ("pad3", u32)]

class CQDispatchSetNumWorkerSemsCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("pad2", u16), ("num_worker_sems", u32)]

class CQDispatchSetGoSignalNocDataCmd(S):
  _pack_ = 1
  _fields_ = [("pad1", u8), ("pad2", u16), ("num_words", u32)]

class CQDispatchCmdPayload(ctypes.Union):
  _pack_ = 1
  _fields_ = [
    ("write_linear_host", CQDispatchWriteHostCmd),
    ("write_paged", CQDispatchWritePagedCmd),
    ("write_packed", CQDispatchWritePackedCmd),
    ("wait", CQDispatchWaitCmd),
    ("debug", CQGenericDebugCmd),
    ("delay", CQDispatchDelayCmd),
    ("set_write_offset", CQDispatchSetWriteOffsetCmd),
    ("mcast", CQDispatchGoSignalMcastCmd),
    ("notify_dispatch_s_go_signal", CQDispatchNotifySubordinateGoSignalCmd),
    ("set_num_worker_sems", CQDispatchSetNumWorkerSemsCmd),
    ("set_go_signal_noc_data", CQDispatchSetGoSignalNocDataCmd),
    ("raw", u8 * 15),
  ]

class CQDispatchCmd(S):
  _pack_ = 1
  _fields_ = [("cmd_id", u8), ("payload", CQDispatchCmdPayload)]

class CQDispatchCmdLargePayload(ctypes.Union):
  _pack_ = 1
  _fields_ = [("write_linear", CQDispatchWriteCmd), ("raw", u8 * 31)]

class CQDispatchCmdLarge(S):
  _pack_ = 1
  _fields_ = [("cmd_id", u8), ("payload", CQDispatchCmdLargePayload)]
