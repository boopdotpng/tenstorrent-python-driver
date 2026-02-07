## style
- use 2 space indents everywhere 
- max 150 characters per line
- do not run a formatter after you write code 

## repo-specific

This is a low level driver for blackhole p100a (i don't have a p150a so i can't test that). After you make a big change, its generally good practice to run `python3 examples/add1.py` or another example to verify that the code still works (do this with a relatively short timeout -- the kernels should take less than 10 seconds) so that you can recover faster in case something broke. Test both dispatch modes: run once normally (fast dispatch) and once with `TT_SLOW_DISPATCH=1 python3 examples/add1.py` (slow dispatch). There is a .venv at ~/tenstorrent that has `tt-smi` installed so you can reset the device (`tt-smi -r`). If that errors, then the card is broken beyond repair and a reboot is required. In this case, stop and inform the user that they may need to reboot. 
Do not run multiple device-using commands in parallel. Only one process can own the device at a time, so run hardware tests strictly sequentially.

When you write kernels, refer to tt-metal for syntax, and note that our kernels will always use every available core, and we will write compute kernels exclusively in SFPI/SFPU ops, not high level tt-llk functions.

## version control
This repo uses **jj** (Jujutsu) colocated with git. Use jj commands, not raw git.

**Agent workflow:** For each new user-requested change, create and use a fresh jj workspace. Do NOT merge into trunk or push — the user will handle merging.

**How to use changes:**
- **Debugging/exploring:** Stay in one change while trying ideas. If an experiment fails, use `jj restore <path>` (or `jj restore .`) to discard edits quickly.
- **Building/implementing:** Use a stack of small logical changes. After each meaningful step, run `jj describe -m "what you did"` then `jj new` to start the next step. This keeps each step independently revertible.
- Before long tests or handoff, ensure every change is named (no anonymous `no description set` changes).
- **Reverting/cleanup:** Prefer jj history operations over manual cleanup edits.
- If a change added lots of debug prints, temporary env vars, or other throwaway code, revert that change instead of deleting lines by hand.
- `jj restore` discards current uncommitted file edits.
- `jj undo` reverses the last jj operation.
- `jj backout <change>` creates a new change that reverses an earlier change.

## workspace lifecycle
- For every new requested change, create a new workspace (for example: `jj workspace add /tmp/blackhole-py-<task> -r @`).
- Do not modify the main/default workspace unless the user explicitly asks.
- Do not reuse old workspaces for unrelated tasks.
- Agents should not delete workspaces on their own.
- The user decides when a task is done and when a workspace should be removed.
- Before handoff or cleanup, ensure work is captured in named changes (`jj describe`) so history is preserved.

## workspace setup
`tt-metal-deps/` is gitignored (extracted from a tarball, not committed). When creating a new workspace, symlink it from the main workspace:
```
ln -s ~/tenstorrent/blackhole-py/tt-metal-deps tt-metal-deps
```
