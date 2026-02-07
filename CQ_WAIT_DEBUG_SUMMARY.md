# Fast Dispatch CQ Wait Debug Handoff

## Goal
Switch fast dispatch completion polling from per-worker `GO_MSG` mailbox polling to dispatcher/CQ-based completion (`SEND_GO_SIGNAL` + `WAIT` + host completion event).

## Current State
- Repo is in **single-lane fast dispatch only** mode (multi-lane support removed intentionally).
- Default fast path is stable and still uses worker `GO_MSG` polling.
- Experimental CQ wait path exists behind env flag:
  - `TT_FAST_CQ_WAIT=1`
- Experimental path currently **fails**: times out waiting for CQ host completion event.

## Key Files Modified
- `device_dispatch.py`
- `defs.py`
- `codegen.py`
- `AGENTS.md` (note added: do not run multiple device commands in parallel)

## What Was Implemented

### 1) Single-lane fast dispatch (intentional simplification)
- `FastDevice` now uses one dispatch pair and one `_FastCQ` instance.
- `TT_FAST_DISPATCH_LANES` parsing and multi-lane round-robin were removed.

### 2) CQ command definitions added (`defs.py`)
Added support for these dispatch command IDs/structs:
- `WRITE_LINEAR_H_HOST` (id 3)
- `WAIT` (id 7)
- `SEND_GO_SIGNAL` (id 14)
- `SET_GO_SIGNAL_NOC_DATA` (id 17)

Plus structs/flags:
- `CQDispatchWriteHostCmd`
- `CQDispatchWaitCmd`
- `CQDispatchGoSignalMcastCmd`
- `CQDispatchSetGoSignalNocDataCmd`
- `CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM`
- `CQ_DISPATCH_CMD_WAIT_FLAG_CLEAR_STREAM`
- `CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET`

### 3) CQ kernel config tweak (`codegen.py`)
- Increased:
  - `MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES`: `64 -> 256`
- This was needed because the runtime can target >64 workers and we now try to preload unicast target data via `SET_GO_SIGNAL_NOC_DATA`.

### 4) `_FastCQ` helper methods added (`device_dispatch.py`)
Added enqueue/poll methods for CQ wait path:
- `enqueue_set_go_signal_noc_data(...)`
- `enqueue_send_go_signal(...)`
- `enqueue_wait_stream(...)`
- `enqueue_host_event(...)`
- Completion queue host polling helpers:
  - `_read_completion_wr_ptr`, `_read_completion_event_id`, `_write_completion_rd_ptr`, `_init_host_completion_ctrl`, `_pop_completion_page`, `wait_host_event`

### 5) Fast `run()` branch
- Default (`TT_FAST_CQ_WAIT` unset or `0`):
  - unchanged behavior, uses `WRITE_PACKED` GO + per-core `GO_MSG` polling.
- Experimental (`TT_FAST_CQ_WAIT=1`):
  - emits command sequence:
    1. `WAIT(stream=48,count=0,clear)`
    2. `SET_GO_SIGNAL_NOC_DATA(unicast core noc_xy list)`
    3. `SEND_GO_SIGNAL(go_signal with dispatch master xy + msg offset 0, unicast only)`
    4. `WAIT(stream=48,count=num_cores,clear)`
    5. `WRITE_LINEAR_H_HOST(is_event=1,payload=event_id)`
  - host polls CQ completion pointer and reads event record.

## Failure Mode (important)
With `TT_FAST_CQ_WAIT=1`:
- `python3 examples/add1.py` fails with:
  - `TimeoutError: timeout waiting for CQ host completion event`
- Observed diagnostics:
  - host completion WR ptr remains equal to RD ptr (`0x180010` in one run)
  - workers appear done (`GO_MSG == RUN_MSG_DONE` for sampled cores)
- Interpretation:
  - workers complete, but dispatcher-side stream wait/event path is not advancing to host completion write.

## Likely Root Causes
Most likely one or more of:
1. `SEND_GO_SIGNAL` fields still not fully aligned with firmware expectations for this runtime.
2. Stream index or stream counter semantics mismatch in current command sequence.
3. Missing prerequisite command(s) that TT-Metal issues around GO/wait setup.
4. `GO` message format details (master coords/message offset) still not equivalent to TT-Metal path for our exact dispatch firmware/runtime.

## Known-good / sanity checks
- Default fast path works:
  - `python3 examples/add1.py` -> `Test Passed`
- Slow path works:
  - `TT_SLOW_DISPATCH=1 python3 examples/add1.py` -> `Test Passed`
- Experimental path fails:
  - `TT_FAST_CQ_WAIT=1 python3 examples/add1.py` -> timeout waiting CQ event

## Debug Suggestions for Next Agent
1. Compare generated command bytes against TT-Metal sequence for one launch:
   - Verify exact `SEND_GO_SIGNAL` and `WAIT` field values.
2. Instrument dispatch firmware-visible stream progress:
   - Read stream register (`STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE`) for stream 48 during run.
3. Validate GO message encoding exactly:
   - `dispatch_message_offset`
   - `master_x/master_y`
   - signal byte
4. Confirm required preconditions for `SEND_GO_SIGNAL` path:
   - any required zeroing/reset waits before first use.
5. Keep `TT_FAST_CQ_WAIT=1` gated until parity is proven.

## Operational Note
- Device commands must be run **sequentially** (one process at a time). This is now documented in `AGENTS.md`.
