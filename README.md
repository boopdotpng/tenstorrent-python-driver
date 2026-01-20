# blackhole-py
A full replacement of **tt-metal** and **tt-umd**, written in Python. 

## requirements 
- A build of tt-metal. You need the compiler and some of the header files, notably `sfpi.h` and some `ckernel` headers (low level control of the tensix coprocessor). At some point I'll extract the headers I need into this project and bundle the compiler so that it works without installing anything, but for now set `TT_HOME` to wherever you have `tt-metal` installed. This is only required for kernel compilation. 
- **tt-kmd**, their kernel driver. 
- Firmware for each baby risc-v core (this is already in the repo)

