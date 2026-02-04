import ctypes
from ctypes import LittleEndianStructure as S, c_uint8 as u8, c_uint16 as u16, c_uint32 as u32, c_uint64 as u64, sizeof

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

class QueryMappingsIn(S):
  _fields_ = [("output_mapping_count", u32), ("reserved", u32)]

class TenstorrentMapping(S):
  _fields_ = [("mapping_id", u32), ("reserved", u32), ("mapping_base", u64), ("mapping_size", u64)]

class ResetDeviceIn(S):
  _fields_ = [("output_size_bytes", u32), ("flags", u32)]

class ResetDeviceOut(S):
  _fields_ = [("output_size_bytes", u32), ("result", u32)]

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
