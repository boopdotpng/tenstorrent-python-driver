## style
- use 2 space indents everywhere 
- max 150 characters per line
- refer to the zen of python

## repo-specific

This is a low level driver for blackhole p100a (i don't have a p150a so i can't test that). After you make a big change, run an example to verify correctness (use a short timeout — kernels should finish in <10 seconds). There is a .venv at ~/tenstorrent that has `tt-smi` installed so you can reset the device (`tt-smi -r`). If that errors, then the card is broken beyond repair and a reboot is required. In this case, stop and inform the user that they may need to reboot.
Do not run multiple device-using commands in parallel. Only one process can own the device at a time, so run hardware tests strictly sequentially.

**Definition of done:** A feature is not complete until it passes both dispatch modes:
1. Fast dispatch (default): `python3 examples/matmul_peak.py`
2. Slow dispatch: `TT_SLOW_DISPATCH=1 python3 examples/matmul_peak.py`

**Device locking:** Multiple agents may be working in this repo concurrently. Always wrap device-accessing commands with `flock` to prevent collisions that brick the device:
```bash
flock /tmp/tt-device.lock python3 examples/matmul_peak.py
flock /tmp/tt-device.lock TT_SLOW_DISPATCH=1 python3 examples/matmul_peak.py
flock /tmp/tt-device.lock tt-smi -r
```
This blocks until the lock is free — no polling or sleep needed.

**Workspace setup (rule):** After creating a new workspace with `jj workspace add`, run `./setup-deps.sh` from that workspace
before running examples/tests. Do not symlink `tt-metal-deps/` from another checkout.

**Kernel/FW cache note:** `~/.cache/tt-metal-cache` can reuse previously built firmware/kernel artifacts for this repo across
workspaces. After firmware/dispatch changes, stale cache entries can mask real behavior (something can look broken or fixed when
it isn't). If results look inconsistent, clear the relevant cache entries and re-run once before concluding.

When you write kernels, refer to tt-metal for syntax, and note that our kernels will always use every available core (minus 2 if using fast dispatch), and we will write compute kernels exclusively in SFPI/SFPU/FPU ops, not high level tt-llk functions.
