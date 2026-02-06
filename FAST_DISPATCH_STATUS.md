# Fast Dispatch Status

## Overview

Fast dispatch uses a command queue (CQ) system: a **prefetch kernel** on a device core reads commands from host memory over PCIe and relays them to a **dispatch kernel**, which writes GO signals to worker cores.

The `add1.py` example is the test case. It writes tiles to DRAM, dispatches compute kernels via CQ, then reads results back via a drain kernel also dispatched through CQ.

## Architecture

- **Prefetch core:** Physical (14, 2), BRISC, NOC_INDEX=0
- **Dispatch core:** Physical (14, 3), BRISC NOC_INDEX=1, NCRISC NOC_INDEX=1
- **PCIe translated coords:** (19, 24) — physical coords are (11, 0) but Blackhole boots with NOC translation enabled
- Tensix physical coords = translated coords (identity mapping on unharvested chips)
- Dispatch CB: base=0x1A000, 128 pages, 4KB page size

## CQ Firmware Files

There are 3 CQ kernel source files, all compiled as XIP (execute-in-place) kernels:

| Source | Compiled as | NOC | Role |
|--------|------------|-----|------|
| `firmware/cq/cq_prefetch.cpp` | BRISC, noc_index=0 | NOC 0 | Reads commands from host sysmem over PCIe, relays to dispatch |
| `firmware/cq/cq_dispatch.cpp` | BRISC, noc_index=1 | NOC 1 | Processes commands, writes GO signals to workers |
| `firmware/cq/cq_dispatch_subordinate.cpp` | NCRISC, noc_index=1 | NOC 1 | Handles subordinate dispatch (go-signal multicast) |

Compilation flow: `codegen.py:compile_cq_kernels()` reads each `.cpp` as text, writes it into a temp dir as `kernel_includes.hpp`, then compiles `brisck.cc`/`ncrisck.cc` (which `#include <kernel_includes.hpp>`). The resulting ELF is packed into an XIP blob via `pack_xip_elf()`. The CRT init (`do_crt1`) in `brisck.cc` properly initializes `.data`/`.bss` before calling `kernel_main()`.

## Bugs Fixed

### Bug #1: Static globals not initialized in XIP kernels

**Status:** Fixed (but turned out to be unnecessary — `do_crt1` handles this)

Static globals like `prefetch_q_rd_ptr` and `pending_read_size` were explicitly re-initialized at the top of `kernel_main()` as a safety measure. The `do_crt1()` call in `brisck.cc` already handles `.data`/`.bss` initialization from the LMA, so this is belt-and-suspenders.

**File:** `firmware/cq/cq_prefetch.cpp`, `kernel_main()`

### Bug #2: Wrong PCIe NOC coordinates

**Status:** Fixed

The compile defines had `PCIE_NOC_X=11, PCIE_NOC_Y=0` (physical coordinates), but Blackhole boots with NOC coordinate translation enabled. The NOC hardware expects translated coordinates `(19, 24)`.

**File:** `codegen.py`, line ~82 in `_DEVICE_DEFINES`

### Bug #3: Jump tables broken in XIP kernels

**Status:** Fixed

The GCC RISC-V backend generates jump tables in `.rodata` for `switch` statements (and also reconverts large if-else chains into jump tables at `-O2`). In XIP kernels, `.rodata` is at a different address than the code expects — the table entries contain absolute addresses from link time, but the kernel loads at an offset relative to `KERNEL_CONFIG_BASE`. The computed jump lands in garbage and the core hangs.

**Symptom:** The prefetch kernel entered `process_cmd`, read `cmd_id=5` (RELAY_INLINE), reached the switch statement, and hung — no case was reached, not even the default. Debug markers before the switch were written but markers in any case body were not.

**Root cause:** The switch's jump table in `.rodata` had wrong absolute addresses for the XIP load context.

**Fix (two layers):**
1. `codegen.py`: Added `-fno-jump-tables` to `_CFLAGS` — prevents the compiler from generating jump tables
2. All 3 CQ firmware files: Converted `switch` statements to `if-else` chains as defense-in-depth

**Files:**
- `codegen.py` — `-fno-jump-tables` flag
- `firmware/cq/cq_prefetch.cpp` — `process_cmd()` switch→if-else
- `firmware/cq/cq_dispatch.cpp` — `process_cmd_d()` and `process_cmd_h()` switch→if-else
- `firmware/cq/cq_dispatch_subordinate.cpp` — main loop switch→if-else

## Current State

The first `device.run(program)` in `add1.py` **succeeds** — the CQ pipeline dispatches compute kernels to all worker cores and they complete. The hang is now in the **second** `device.run()` call, which dispatches the DRAM drain kernel to read results back.

```
add1.py line 149: device.run(program)       # WORKS - compute kernels dispatched and complete
add1.py line 151: out = device.dram.read()   # HANGS - drain kernel dispatch via CQ
```

The drain kernel dispatch uses the same CQ pipeline (prefetch→dispatch→workers). The pipeline processes the first batch of commands fine but fails on the second. Likely a CQ state issue — command queue pointers, semaphores, or the prefetch kernel's fetch loop not properly cycling for a second batch of commands.

## Known Future Issue

The dispatch kernel (`cq_dispatch.cpp`) compiles with `noc_index=1` (NOC 1) and computes `pcie_noc_xy` using `NOC_X_PHYS_COORD(PCIE_NOC_X)` which on NOC 1 = `17-1-19 = -3` (wraps to garbage). This is only used for completion queue writes to host memory, which aren't exercised by `add1.py`. Will need fixing when completion queue support is needed.
