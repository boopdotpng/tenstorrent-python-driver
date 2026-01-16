# tenstorrent-python-driver


## todos
- [x] set up mmaps, TLB reads, autogen, ioctls, and other low level stuff
- [x] get device up and recognized
- [x] verify device is working with telemetry
- [x] get harvesting data on the ARC tile using a NoC read 
- [ ] good NoC abstraction for tensix tiles (can't address harvested columns) 
- [ ] load firmware for Tensix tiles on device (not present after fresh boot), check if firmware is present
- [ ] figure out address space for dram tiles / read and write data in vram 
- [ ] set up compiler to work with dataflow and compute (SFPI) kernels. extract raw elf, figure out what to upload
- [ ] figure out address space for tensix tiles 
  - [ ] where to upload brisc, ncrisc, and trisc0,1,2 code 
  - [ ] do we have to upload firmware for those cores too? 
  - [ ] how to set up the CB
  - [ ] how to kick off the work 
  - [ ] how to poll for completion 
- [ ] fast dispatch using the PIN_PAGES ioctl and a command queue on the host. apparently there are some prefetcher cores that aren't documented anywhere that are used for this 

## bring-up steps 
1. open device, mmap bars, set up TLB windows 
2. check if firmware is loaded (not sure how) and load the firmware for the RISC cores inside each tensix tile

## tt-metal interop
`TT_METAL_SKIP_LOADING_FW=1` skips tt-metal firmware writes, but tt-metal/UMD still puts the device into `LONG_IDLE` and asserts RISC resets on close, so firmware loaded by this repo may not persist between tt-metal runs. Use:
- `python main.py` (upload firmware), then run your tt-metal/ttnn command
- `python with_fw.py -- <cmd ...>` to upload firmware and exec a command in one step
