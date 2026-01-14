import os
from autogen import TENSTORRENT_IOCTL_MAGIC

# UT3G cannot support fast dispatch because of the 1g iommu map requirement
# will test this later
# used by default
# SLOW_DISPATCH = int(os.environ.get("TT_SLOW_DISPATCH", 0)) == 1
DEBUG = int(os.environ.get("DEBUG", 0)) > 1

def _IO(nr: int) -> int: return (TENSTORRENT_IOCTL_MAGIC << 8) | nr
def align_down(value: int, alignment: int) -> tuple[int, int]:
  base = value & ~(alignment - 1)
  return base, value - base
