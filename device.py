from device_runtime import Program, DataflowLaunch, TileGrid, ArgGen
from device_dispatch import SlowDevice, FastDevice
from helpers import USE_USB_DISPATCH

__all__ = ["Program", "DataflowLaunch", "TileGrid", "ArgGen", "SlowDevice", "FastDevice", "Device"]

class Device:
  def __new__(cls, device: int = 0):
    if USE_USB_DISPATCH:
      return SlowDevice(device=device)
    try:
      return FastDevice(device=device)
    except RuntimeError as exc:
      if "missing fast-dispatch firmware" not in str(exc): raise
      print(f"[blackhole-py] {exc}; falling back to slow dispatch")
      return SlowDevice(device=device)
