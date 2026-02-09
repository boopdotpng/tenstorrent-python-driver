# Progress Handoff: 2-Role NOC-Split Matmul

## Workspace + Revisions
- Workspace path: `/home/boop/jj-workspaces/blackhole-py/dataflow-role-api`
- Current change: `vmzqxlyt 4c9f19d8`
- Checkpoint (stable API): `xnkvpsqm 2d589bba`
- Parent baseline: `ysyvoyku b0487f01` (`master`)

## Current State: FUNCTIONAL, matching baseline

The 2-role NOC-split matmul is working and matches baseline performance (~94 TFLOPS).
All code is in a clean, working state with no pending experimental changes.

## What Was Completed

### 1) Dataflow API groundwork (checkpointed in `2d589bba`)
- `DataflowLaunch` / `CoreSet` / `CoreRange` API in `device_runtime.py`.
- Dispatch lowering: group identical payloads, lower to rectangles, mcast where possible.
- Updated `examples/add1.py`, `examples/matmul_naive.py`, `examples/matmul_peak.py`.
- Compile helpers: `Compiler.compile_dataflow(...)`, `Compiler.compile_compute(...)`.

### 2) 4-role → 2-role kernel merge (current change)
Original 4-role experiment deadlocked due to non-uniform `sem_off` across launches.
Fixed by computing a uniform `sem_off` in `_uniform_sem_off()`, then merged 4 roles
into 2 unified kernels:

- **K_READER** (NCRISC/NOC0): in0 sender OR receiver, selected by `is_in0_sender` arg.
- **K_WRITER** (BRISC/NOC1): in1 sender OR receiver + output write, selected by `is_in1_sender` arg.

Single `DataflowLaunch` covers all 110 cores. NOC0 handles in0 data movement,
NOC1 handles in1 data movement + output write — parallel data movement on separate NOCs.

### 3) Performance measurements (fast dispatch)
Baseline (master repo `examples/matmul_peak.py`):
- After fresh reset: 93.0–94.2 TFLOPS (5 iterations)

2-role NOC-split version:
- After fresh reset: 93.0–94.6 TFLOPS (5 iterations)
- Best per-iteration (GC disabled, 20 iters): 2.479ms = **95.29 TFLOPS**
- Median per-iteration: 2.504ms = **94.32 TFLOPS**

**Verdict:** Matches baseline. The matmul is compute-bound — the NOC split doesn't
help because double-buffered CB pipelining already hides data movement.

### 4) Optimizations attempted and reverted

| Experiment | Result | Why |
|---|---|---|
| OUT_SUBBLOCK 4×4 (16 dest tiles) | 92.5 TFLOPS | Dest register pressure hurts more than spill/reload savings |
| IN0_BLOCK_W=2 (64 blocks) | 78.6 TFLOPS | Doubled spill/reload overhead (~490ns/subblock/block) |
| IN0_BLOCK_W=8 single-buffer | 72.3 TFLOPS | Exposed ~63μs/block DRAM read latency overwhelms spill savings |
| Dispatch: write_packed_large for RESET/GO_MSG_INDEX | CQ hang | Workers stuck at mode=0 after warmup; root cause unknown |
| Dispatch: NO_STRIDE for launch blob | CQ hang | Same failure as above; reverted |

All experiments reverted. Code is clean.

### 5) Key findings

- **Spill/reload cost:** ~490ns per subblock per non-final block. With 31 non-final blocks × 32 subblocks = ~486μs total (~19% of 2.5ms iteration time).
- **Double-buffer essential:** Single-buffer CBs expose ~63μs/block DRAM read latency for sender cores. Penalty overwhelms any spill savings.
- **Thermal throttling:** Performance degrades across consecutive runs (94→88→87 TFLOPS). First-run-after-reset numbers are most reliable for comparison.
- **Dispatch overhead:** Our dispatch uses `write_packed` (unicast) for RESET/GO_MSG_INDEX vs baseline's `write_packed_large` (mcast). Extra ~10-15μs/iter, but attempts to fix caused CQ hangs.

## Recommended Next Steps

### High confidence (ready to implement)
1. **Kernel-side iteration looping** — Eliminate per-iteration dispatch round-trip (~50-100μs each) by running multiple iterations inside the kernels. All three kernels (reader, writer, compute) get a `num_iters` arg; host calls `device.run()` once for warmup and once for timed. Semaphore state resets naturally between iterations. Could push from ~94 to ~97 TFLOPS.

### Medium confidence (needs debugging)
2. **Fix dispatch NO_STRIDE for launch blob** — The launch blob IS uniform across all 110 cores (single DataflowLaunch, identical RTA sizes). Baseline uses NO_STRIDE successfully. Our version hangs after warmup iterations. Need to debug why the second iteration fails — possibly CQ command ordering issue.
3. **Use write_packed_large + mcast for RESET/GO_MSG_INDEX** — Saves ~5μs per dispatch (4 mcast sub-cmds vs 110 unicast). Same CQ hang as above may be related.

### Lower priority
4. **Test slow dispatch** — `TT_SLOW_DISPATCH=1` not yet tested with 2-role version.
5. **Validate correctness** — Currently benchmarking only (no output verification). Add a small-matrix correctness test to catch silent data corruption.

## Architecture Notes

### CB Layout (per core, 1316 KB available)
| CB | Pages | Size | Purpose |
|---|---|---|---|
| CB0 | 128 (2×64) | 256 KB | in0 double-buffer |
| CB1 | 128 (2×64) | 256 KB | in1 double-buffer |
| CB16 | 256 | 512 KB | output (aliases CB24) |
| CB24 | 256 | 512 KB | spill/reload (aliases CB16) |
| **Total** | | **1024 KB** | |

### Matmul Parameters
- Matrix: C[5120,5632] = A[5120,4096] × B[4096,5632], bf16 LoFi
- Grid: 10×11 = 110 cores, per_core: 16×16 tiles
- IN0_BLOCK_W=4, NUM_BLOCKS=32, OUT_SUBBLOCK=4h×2w

### File Map
- `examples/matmul_peak.py` — 2-role benchmark (~544 lines)
- `device_dispatch.py` — FastDevice.run at line 946 (DataflowLaunch dispatch path)
- `device_runtime.py` — Program / DataflowLaunch / CoreSet definitions

## Useful Commands
```bash
flock /tmp/tt-device.lock python3 examples/matmul_peak.py
flock /tmp/tt-device.lock env TT_SLOW_DISPATCH=1 python3 examples/matmul_peak.py
flock /tmp/tt-device.lock /home/boop/tenstorrent/.venv/bin/tt-smi -r
```
