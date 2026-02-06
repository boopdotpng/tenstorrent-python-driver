from device_runtime import Program, TileGrid, ArgGen
from device_dispatch import SlowDevice, FastDevice
from helpers import USE_SLOW_DISPATCH

class Device:
  def __init__(self, path: str = "/dev/tenstorrent/0", *, upload_firmware: bool = True):
    impl = SlowDevice if USE_SLOW_DISPATCH else FastDevice
    self._impl = impl(path=path, upload_firmware=upload_firmware)

  def __getattr__(self, name: str):
    return getattr(self._impl, name)

  def close(self):
    self._impl.close()

__all__ = ["ArgGen", "Program", "TileGrid", "SlowDevice", "FastDevice", "Device"]
