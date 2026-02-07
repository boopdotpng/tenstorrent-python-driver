# agents
Write excellent, concise code that solves problems elegantly.

## philosophy
**Every line must earn its keep.** Prefer readability over cleverness. If carefully designed, 10 lines can have the impact of 1000.

## style
- use 2 space indents everywhere 
- max 150 characters per line
- do not run a formatter after you write code 

## repo-specific

This is a low level driver for blackhole p100a (i don't have a p150a so i can't test that). After you make a big change, its generally good practice to run `python3 examples/add1.py` or another example to verify that the code still works (do this with a relatively short timeout -- the kernels should take less than 10 seconds) so that you can recover faster in case something broke. Test both dispatch modes: run once normally (fast dispatch) and once with `TT_SLOW_DISPATCH=1 python3 examples/add1.py` (slow dispatch). There is a .venv at ~/tenstorrent that has `tt-smi` installed so you can reset the device (`tt-smi -r`). If that errors, then the card is broken beyond repair and a reboot is required. In this case, stop and inform the user that they may need to reboot. 
Do not run multiple device-using commands in parallel. Only one process can own the device at a time, so run hardware tests strictly sequentially.

When you write kernels, refer to tt-metal for syntax, and note that our kernels will always use every available core, and we will write compute kernels exclusively in SFPI/SFPU ops, not high level tt-llk functions. 
