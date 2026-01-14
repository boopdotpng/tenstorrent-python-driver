import ctypes

TENSTORRENT_IOCTL_MAGIC = 0xFA
IOCTL_GET_DEVICE_INFO = 0
IOCTL_QUERY_MAPPINGS = 2
IOCTL_RESET_DEVICE = 6
IOCTL_PIN_PAGES = 7
IOCTL_UNPIN_PAGES = 10
IOCTL_ALLOCATE_TLB = 11
IOCTL_FREE_TLB = 12
IOCTL_CONFIGURE_TLB = 13

class QueryMappingsIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("output_mapping_count", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32),
  ]

class TenstorrentMapping(ctypes.LittleEndianStructure):
  _fields_ = [
    ("mapping_id", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32),
    ("mapping_base", ctypes.c_uint64),
    ("mapping_size", ctypes.c_uint64),
  ]

# reset device consts from tt-kmd
TENSTORRENT_RESET_DEVICE_ASIC_RESET = 4
TENSTORRENT_RESET_DEVICE_ASIC_DMC_RESET = 5 # use this when the NoC is hung, more full reset
TENSTORRENT_RESET_DEVICE_POST_RESET = 6
class ResetDeviceIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("output_size_bytes", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
  ]

class ResetDeviceOut(ctypes.LittleEndianStructure):
  _fields_ = [
    ("output_size_bytes", ctypes.c_uint32),
    ("result", ctypes.c_uint32),
  ]

class PinPagesIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("output_size_bytes", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
    ("virtual_address", ctypes.c_uint64),
    ("size", ctypes.c_uint64),
  ]

class PinPagesOut(ctypes.LittleEndianStructure):
  _fields_ = [("physical_address", ctypes.c_uint64)]

class PinPagesOutExtended(ctypes.LittleEndianStructure):
  _fields_ = [
    ("physical_address", ctypes.c_uint64),
    ("noc_address", ctypes.c_uint64),
  ]

class UnpinPagesIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("virtual_address", ctypes.c_uint64),
    ("size", ctypes.c_uint64),
    ("reserved", ctypes.c_uint64),
  ]

class AllocateTlbIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("size", ctypes.c_uint64),
    ("reserved", ctypes.c_uint64),
  ]

class AllocateTlbOut(ctypes.LittleEndianStructure):
  _fields_ = [
    ("tlb_id", ctypes.c_uint32),
    ("reserved0", ctypes.c_uint32),
    ("mmap_offset_uc", ctypes.c_uint64),
    ("mmap_offset_wc", ctypes.c_uint64),
    ("reserved1", ctypes.c_uint64),
  ]

class FreeTlbIn(ctypes.LittleEndianStructure):
  _fields_ = [("tlb_id", ctypes.c_uint32)]

class NocTlbConfig(ctypes.LittleEndianStructure):
  _fields_ = [
    ("addr", ctypes.c_uint64),
    ("x_end", ctypes.c_uint16),
    ("y_end", ctypes.c_uint16),
    ("x_start", ctypes.c_uint16),
    ("y_start", ctypes.c_uint16),
    ("noc", ctypes.c_uint8),
    ("mcast", ctypes.c_uint8),
    ("ordering", ctypes.c_uint8),
    ("linked", ctypes.c_uint8),
    ("static_vc", ctypes.c_uint8),
    ("reserved0_0", ctypes.c_uint8),
    ("reserved0_1", ctypes.c_uint8),
    ("reserved0_2", ctypes.c_uint8),
    ("reserved1_0", ctypes.c_uint32),
    ("reserved1_1", ctypes.c_uint32),
  ]

class ConfigureTlbIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("tlb_id", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32),
    ("config", NocTlbConfig),
  ]

class TenstorrentGetDeviceInfoIn(ctypes.LittleEndianStructure):
  _fields_ = [
    ("output_size_bytes", ctypes.c_uint32),
  ]

class TenstorrentGetDeviceInfoOut(ctypes.LittleEndianStructure):
  _fields_ = [
    ("output_size_bytes", ctypes.c_uint32),
    ("vendor_id", ctypes.c_uint16),
    ("device_id", ctypes.c_uint16),
    ("subsystem_vendor_id", ctypes.c_uint16),
    ("subsystem_id", ctypes.c_uint16),
    ("bus_dev_fn", ctypes.c_uint16),  # [0:2] function, [3:7] device, [8:15] bus
    ("max_dma_buf_size_log2", ctypes.c_uint16),  
    ("pci_domain", ctypes.c_uint16),  
    ("reserved", ctypes.c_uint16),
  ]