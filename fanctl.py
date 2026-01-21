#!/usr/bin/env python3
import argparse, os, time
from tlb import TLBWindow, TLBConfig, TLBMode, TLBSize
from configs import Arc
from device import TileGrid
from helpers import align_down

FAN_MSG_FORCE_SPEED = 0xAC

def arc_msg(fd: int, msg: int, arg0: int = 0, arg1: int = 0, *, queue: int = 0, timeout_ms: int = 1000) -> list[int]:
  MSG_QUEUE_SIZE = 4
  MSG_QUEUE_POINTER_WRAP = 2 * MSG_QUEUE_SIZE
  REQUEST_MSG_LEN = 8
  RESPONSE_MSG_LEN = 8
  HEADER_BYTES = 8 * 4
  REQUEST_BYTES = REQUEST_MSG_LEN * 4
  RESPONSE_BYTES = RESPONSE_MSG_LEN * 4
  QUEUE_STRIDE = HEADER_BYTES + (MSG_QUEUE_SIZE * REQUEST_BYTES) + (MSG_QUEUE_SIZE * RESPONSE_BYTES)
  RESET_UNIT_ARC_MISC_CNTL = Arc.RESET_UNIT_OFFSET + 0x100
  IRQ0_TRIG_BIT = 1 << 16

  if queue < 0 or queue >= 4: raise ValueError("queue must be 0..3")

  cfg = TLBConfig(addr=Arc.NOC_BASE, start=TileGrid.ARC, end=TileGrid.ARC, noc=0, mcast=False, mode=TLBMode.STRICT)
  with TLBWindow(fd, TLBSize.MiB_2, cfg) as arc:
    info_ptr = arc.readi32(Arc.SCRATCH_RAM_11)
    if info_ptr == 0: raise RuntimeError("msgqueue not initialized (SCRATCH_RAM_11 == 0)")

    info_base, info_off = align_down(info_ptr, TLBSize.MiB_2)
    cfg.addr = info_base
    arc.configure(cfg)
    queues_ptr = arc.readi32(info_off)

    q_base, q_off = align_down(queues_ptr, TLBSize.MiB_2)
    cfg.addr = q_base
    arc.configure(cfg)
    q = q_off + queue * QUEUE_STRIDE

    wptr = arc.readi32(q + 0)
    req = q + HEADER_BYTES + (wptr % MSG_QUEUE_SIZE) * REQUEST_BYTES
    words = [msg & 0xFF, arg0 & 0xFFFFFFFF, arg1 & 0xFFFFFFFF] + [0] * (REQUEST_MSG_LEN - 3)
    for i, word in enumerate(words): arc.writei32(req + i * 4, word)
    arc.writei32(q + 0, (wptr + 1) % MSG_QUEUE_POINTER_WRAP)

    cfg.addr = Arc.NOC_BASE
    arc.configure(cfg)
    arc.writei32(RESET_UNIT_ARC_MISC_CNTL, arc.readi32(RESET_UNIT_ARC_MISC_CNTL) | IRQ0_TRIG_BIT)

    cfg.addr = q_base
    arc.configure(cfg)
    rptr = arc.readi32(q + 4)
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
      resp_wptr = arc.readi32(q + 20)
      if resp_wptr != rptr:
        resp = q + HEADER_BYTES + (MSG_QUEUE_SIZE * REQUEST_BYTES) + (rptr % MSG_QUEUE_SIZE) * RESPONSE_BYTES
        out = [arc.readi32(resp + i * 4) for i in range(RESPONSE_MSG_LEN)]
        arc.writei32(q + 4, (rptr + 1) % MSG_QUEUE_POINTER_WRAP)
        return out
      time.sleep(0.001)
  raise TimeoutError(f"arc_msg timeout ({timeout_ms} ms)")

def main():
  ap = argparse.ArgumentParser(description="Tenstorrent Blackhole fan control (direct ARC msgqueue)")
  ap.add_argument("--dev", default="/dev/tenstorrent/0")
  ap.add_argument("--timeout-ms", type=int, default=1000)
  mx = ap.add_mutually_exclusive_group(required=True)
  mx.add_argument("--set", type=int, metavar="PCT", help="force fan speed (0..100)")
  mx.add_argument("--reset", action="store_true", help="reset to firmware curve")
  args = ap.parse_args()

  pct = None if args.reset else int(args.set)
  if pct is not None and not (0 <= pct <= 100): raise SystemExit("--set must be 0..100")

  fd = os.open(args.dev, os.O_RDWR | os.O_CLOEXEC)
  try:
    raw = 0xFFFFFFFF if pct is None else pct
    resp = arc_msg(fd, FAN_MSG_FORCE_SPEED, raw, 0, timeout_ms=args.timeout_ms)
    print("resp:", " ".join(f"{x:08x}" for x in resp))
  finally:
    os.close(fd)

if __name__ == "__main__":
  main()
