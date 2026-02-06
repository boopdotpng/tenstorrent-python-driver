from device_runtime import Program, TileGrid, ArgGen
from device_dispatch import SlowDevice, FastDevice
from helpers import USE_SLOW_DISPATCH

class Device:
  def __init__(self, device: int = 0):
    impl = SlowDevice if USE_SLOW_DISPATCH else FastDevice
    self._impl = impl(device=device)

  def __getattr__(self, name: str):
    return getattr(self._impl, name)

  def close(self):
    self._impl.close()

__all__ = ["ArgGen", "Program", "TileGrid", "SlowDevice", "FastDevice", "Device"]
