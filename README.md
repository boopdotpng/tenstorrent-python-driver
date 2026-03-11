## blackhole-py

A minimal Python driver for the Tenstorrent Blackhole accelerator. Compiles and dispatches RISC-V kernels directly from Python — no TT-Metal runtime required.

> **~7k lines of code** — the entire driver, compiler, firmware, and dispatch stack. TT-Metal's equivalent is ~290k.

### Matmul

```python
from device import Device, Dtype, MathFidelity
from ops import plan_matmul, MatmulProgram

device = Device()
M, K, N = 4096, 4096, 4096

plan = plan_matmul(M, K, N, device.cores)
a_buf = device.alloc_write(a_bytes, Dtype.Float16_b, (M, K), "A")
b_buf = device.alloc_write(b_bytes, Dtype.Float16_b, (K, N), "B")
c_buf = device.alloc(plan.mt * plan.nt, Dtype.Float16_b, "C", (M, N))

device.queue(MatmulProgram(plan, a_buf, b_buf, c_buf, Dtype.Float16_b, MathFidelity.HiFi2))
device.run()
```

```sh
PYTHONPATH=. uv run examples/matmul_peak.py
```

### Requirements

- **Hardware**: Blackhole P100A only (P150A is untested)
- **Kernel driver**: tt-kmd >= 2.6.0
- **Firmware**: any version < 15
- **Python**: 3.10+, numpy

### Setup

Run `./setup-deps.sh` to download the SFPI compiler toolchain and TT-Metal headers.

### Dispatch modes

```sh
PYTHONPATH=. uv run examples/matmul_peak.py              # fast dispatch (on-device CQ)
PYTHONPATH=. TT_USB=1 uv run examples/matmul_peak.py     # slow dispatch (over UT3G USB adapter)
```

Fast dispatch uses on-device command queues. Slow dispatch (`TT_USB=1`) drives the chip over the UT3G USB-C adapter via host TLB writes.
