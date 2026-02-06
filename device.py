from device_runtime import Program, TileGrid, ArgGen
from device_dispatch import SlowDevice, FastDevice
from helpers import USE_SLOW_DISPATCH

class Device:
  def __init__(self, device: int = 0):
    if USE_SLOW_DISPATCH:
      self._impl = SlowDevice(device=device)
      return
    try:
      self._impl = FastDevice(device=device)
    except RuntimeError as exc:
      if "missing fast-dispatch firmware" not in str(exc): raise
      print(f"[blackhole-py] {exc}; falling back to slow dispatch")
      self._impl = SlowDevice(device=device)

  def __getattr__(self, name: str):
    return getattr(self._impl, name)

  def close(self):
    self._impl.close()

__all__ = ["ArgGen", "Program", "TileGrid", "SlowDevice", "FastDevice", "Device"]
