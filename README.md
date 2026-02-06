# blackhole-py
A full replacement of **tt-metal** and **tt-umd**, written in Python. 

## requirements 
- **tt-kmd**
- Firmware version 19.4.2. I don't really see a need to update past this, and in 19.5 they disable a bunch of your Tensix cores (on p150a). 

## instructions 
Run `setup-deps.sh` to install the tarball containing the risc-v compiler, include headers, and libraries. Previously, you had to build a whole copy of `tt-metal` to get these headers, but the tarball contains the minimal set of files required to run all tt-metal kernels. 


Currently, I have add1 and a very naive, slow matmul working. 

## dispatch mode
- Fast dispatch is selected by default.
- Set `TT_SLOW_DISPATCH=1` to force slow dispatch.
- Fast DRAM read path is only enabled in fast-dispatch mode.

### fast-dispatch firmware
Fast dispatch requires three extra firmware ELFs in `riscv-firmware/p100a/`:
- `cq_prefetch_brisc.elf`
- `cq_dispatch_brisc.elf`
- `cq_dispatch_subordinate_ncrisc.elf`

If they are missing, blackhole-py prints a warning and falls back to slow dispatch.
