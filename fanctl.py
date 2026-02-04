#!/usr/bin/env python3
import argparse
from device import Device

FAN_MSG_FORCE_SPEED = 0xAC

def main():
  ap = argparse.ArgumentParser(
    description="Tenstorrent Blackhole fan control (direct ARC msgqueue)"
  )
  ap.add_argument("--dev", default="/dev/tenstorrent/0")
  ap.add_argument("--timeout-ms", type=int, default=1000)
  mx = ap.add_mutually_exclusive_group(required=True)
  mx.add_argument("--set", type=int, metavar="PCT", help="force fan speed (0..100)")
  mx.add_argument("--reset", action="store_true", help="reset to firmware curve")
  args = ap.parse_args()
  pct = None if args.reset else int(args.set)
  if pct is not None and not (0 <= pct <= 100):
    raise SystemExit("--set must be 0..100")
  device = Device(args.dev, upload_firmware=False)
  try:
    raw = 0xFFFFFFFF if pct is None else pct
    resp = device.arc_msg(FAN_MSG_FORCE_SPEED, raw, 0, timeout_ms=args.timeout_ms)
    print("resp:", " ".join(f"{x:08x}" for x in resp))
  finally:
    device.close()

if __name__ == "__main__":
  main()
