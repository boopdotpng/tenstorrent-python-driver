import argparse
from pathlib import Path
from device import Device

FAN_MSG_FORCE_SPEED = 0xAC

def parse_devices(specs: list[str]) -> list[int]:
  tokens = []
  for spec in specs:
    tokens.extend(part.strip() for part in spec.split(",") if part.strip())
  if not tokens:
    raise SystemExit("--dev requires at least one device")
  if any(tok.lower() == "all" for tok in tokens):
    root = Path("/dev/tenstorrent")
    if not root.exists():
      raise SystemExit("/dev/tenstorrent does not exist")
    devs = sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())
    if not devs:
      raise SystemExit("no /dev/tenstorrent/<n> devices found")
    return devs
  devs = []
  for tok in tokens:
    try:
      dev = int(tok)
    except ValueError as e:
      raise SystemExit(f"invalid --dev '{tok}' (expected int or 'all')") from e
    if dev < 0:
      raise SystemExit("device must be >= 0")
    if dev not in devs:
      devs.append(dev)
  return devs

def main():
  ap = argparse.ArgumentParser(
    description="Tenstorrent Blackhole fan control (direct ARC msgqueue)"
  )
  ap.add_argument("--dev", action="append", default=["0"],
    help="device selector: int, comma-list, repeated flag, or 'all'")
  ap.add_argument("--timeout-ms", type=int, default=1000)
  mx = ap.add_mutually_exclusive_group(required=True)
  mx.add_argument("--set", type=int, metavar="PCT", help="force fan speed (0..100)")
  mx.add_argument("--reset", action="store_true", help="reset to firmware curve")
  args = ap.parse_args()
  devs = parse_devices(args.dev)
  pct = None if args.reset else int(args.set)
  if pct is not None and not (0 <= pct <= 100):
    raise SystemExit("--set must be 0..100")
  raw = 0xFFFFFFFF if pct is None else pct
  failed = []
  for dev in devs:
    device = Device(dev)
    try:
      resp = device.arc_msg(FAN_MSG_FORCE_SPEED, raw, 0, timeout_ms=args.timeout_ms)
      print(f"dev {dev} resp:", " ".join(f"{x:08x}" for x in resp))
    except Exception as e:
      failed.append((dev, e))
      print(f"dev {dev} error: {e}")
    finally:
      device.close()
  if failed:
    names = ", ".join(str(dev) for dev, _ in failed)
    raise SystemExit(f"fanctl failed on device(s): {names}")

if __name__ == "__main__":
  main()
