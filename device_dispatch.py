import time
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import tlog
from dram import DramAllocator
from device_runtime import CommonDevice, Program, TileGrid, ArgGen

class SlowDevice(CommonDevice):
  def __init__(self, path: str = "/dev/tenstorrent/0", *, upload_firmware: bool = True):
    super().__init__(path=path, upload_firmware=upload_firmware)
    self.win = TLBWindow(self.fd, TLBSize.MiB_2)
    self.dram = DramAllocator(fd=self.fd, dram_tiles=self.tiles.dram, run_fn=self.run)

  @staticmethod
  def _firmware_layout() -> dict[str, tuple[int, int]]:
    return {
      "brisc.elf": (TensixL1.BRISC_FIRMWARE_BASE, TensixL1.BRISC_INIT_LOCAL_L1_BASE_SCRATCH),
      "ncrisc.elf": (TensixL1.NCRISC_FIRMWARE_BASE, TensixL1.NCRISC_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc0.elf": (TensixL1.TRISC0_BASE, TensixL1.TRISC0_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc1.elf": (TensixL1.TRISC1_BASE, TensixL1.TRISC1_INIT_LOCAL_L1_BASE_SCRATCH),
      "trisc2.elf": (TensixL1.TRISC2_BASE, TensixL1.TRISC2_INIT_LOCAL_L1_BASE_SCRATCH),
    }

  @staticmethod
  def _tile_ready(win: TLBWindow) -> bool:
    go = win.uc[TensixL1.GO_MSG + 3]
    sync = win.read32(TensixL1.MAILBOX_BASE + 8)
    dm1, tr0, tr1, tr2 = sync & 0xFF, (sync >> 8) & 0xFF, (sync >> 16) & 0xFF, (sync >> 24) & 0xFF
    return go == DevMsgs.RUN_MSG_DONE and dm1 == 0 and tr1 == 0 and tr2 == 0 and tr0 in (0, 3)

  def _build_local_cb_blob(self, program: Program) -> tuple[int, bytes]:
    mask = 0
    for cb in program.cbs:
      mask |= 1 << cb
    end = mask.bit_length()
    arr = (LocalCBConfig * end)()
    addr = TensixL1.DATA_BUFFER_SPACE_BASE
    for cb in program.cbs:
      size = program.tile_size * program.num_pages
      arr[cb] = LocalCBConfig(
        addr_bytes=addr,
        size_bytes=size,
        num_pages=program.num_pages,
        page_size_bytes=program.tile_size,
      )
      addr += size
    return mask, as_bytes(arr)

  def _pack_kernel_shared(self, program: Program, rta_sizes: tuple[int, int, int]):
    align16 = lambda n: (n + 15) & ~15

    rta_offsets = [0, rta_sizes[0], rta_sizes[0] + rta_sizes[1]]
    rta_total = align16(rta_offsets[2] + rta_sizes[2])
    crta_off = rta_total

    local_cb_mask, local_cb_blob = self._build_local_cb_blob(program)
    local_cb_off = rta_total
    kernel_off = align16(local_cb_off + len(local_cb_blob))

    kernels = {"brisc": program.writer, "ncrisc": program.reader}
    if program.compute:
      for i, k in enumerate(program.compute):
        kernels[f"trisc{i}"] = k
    proc = [("brisc", 0), ("ncrisc", 1), ("trisc0", 2), ("trisc1", 3), ("trisc2", 4)]
    kernel_text_off = [0] * len(proc)
    enables = 0
    off = kernel_off
    for name, idx in proc:
      if (k := kernels.get(name)) is None: continue
      kernel_text_off[idx] = off
      off = align16(off + len(k.xip))
      enables |= 1 << idx

    # Shared image: from local_cb_off to end (CB config + kernel binaries)
    shared = bytearray(off - local_cb_off)
    shared[0 : len(local_cb_blob)] = local_cb_blob
    for name, idx in proc:
      if (k := kernels.get(name)) is None: continue
      dst = kernel_text_off[idx] - local_cb_off
      shared[dst : dst + len(k.xip)] = k.xip

    cfg = LaunchMsg().kernel_config
    cfg.kernel_config_base[0] = TensixL1.KERNEL_CONFIG_BASE
    cfg.kernel_config_base[1] = TensixL1.KERNEL_CONFIG_BASE
    cfg.kernel_config_base[2] = TensixL1.KERNEL_CONFIG_BASE
    cfg.local_cb_offset = local_cb_off
    cfg.remote_cb_offset = local_cb_off + len(local_cb_blob)
    cfg.local_cb_mask = local_cb_mask
    cfg.min_remote_cb_start_index = CB.NUM_CIRCULAR_BUFFERS
    cfg.enables = enables
    cfg.brisc_noc_id = 1
    cfg.brisc_noc_mode = 0
    cfg.mode = DevMsgs.DISPATCH_MODE_HOST
    cfg.rta_offset[0].rta_offset, cfg.rta_offset[0].crta_offset = rta_offsets[0], crta_off
    cfg.rta_offset[1].rta_offset, cfg.rta_offset[1].crta_offset = rta_offsets[1], crta_off
    for i in (2, 3, 4):
      cfg.rta_offset[i].rta_offset, cfg.rta_offset[i].crta_offset = rta_offsets[2], crta_off
    for i, v in enumerate(kernel_text_off):
      cfg.kernel_text_offset[i] = v

    launch = LaunchMsg()
    launch.kernel_config = cfg
    return local_cb_off, bytes(shared), as_bytes(launch), rta_offsets

  @staticmethod
  def _pack_rta(reader_args: list[int], writer_args: list[int], compute_args: list[int]) -> bytes:
    pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
    return pack(writer_args) + pack(reader_args) + pack(compute_args)

  def _resolve_args(self, args: list[int] | ArgGen, core_idx: int, core_xy: tuple[int, int], num_cores: int) -> list[int]:
    return args if isinstance(args, list) else args(core_idx, core_xy, num_cores)

  @staticmethod
  def _mcast_rects(cores: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    west = [(x, y) for x, y in cores if x < 8]
    east = [(x, y) for x, y in cores if x >= 10]
    rects = []
    for group in (west, east):
      if group:
        xs = [x for x, _ in group]
        ys = [y for _, y in group]
        rects.append((min(xs), max(xs), min(ys), max(ys)))
    return rects

  @staticmethod
  def _mcast_write_rects(
    win: TLBWindow,
    cfg: TLBConfig,
    rects: list[tuple[int, int, int, int]],
    writes: list[tuple[int, bytes]],
  ):
    for x0, x1, y0, y1 in rects:
      cfg.start, cfg.end = (x0, y0), (x1, y1)
      win.configure(cfg)
      for addr, data in writes:
        win.write(addr, data, use_uc=True, restore=False)

  def run(self, program: Program) -> tuple[float, float]:
    cores = program.cores if program.cores is not None else TileGrid.TENSIX
    num_cores = len(cores)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset)
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go_blob = as_bytes(go)

    # Compute rta sizes from core 0 to determine shared layout
    first_r = self._resolve_args(program.reader_rt_args, 0, cores[0], num_cores)
    first_w = self._resolve_args(program.writer_rt_args, 0, cores[0], num_cores)
    first_c = self._resolve_args(program.compute_rt_args, 0, cores[0], num_cores)
    rta_sizes = (len(first_w) * 4, len(first_r) * 4, len(first_c) * 4)
    shared_off, shared_img, launch_blob, rta_offsets = self._pack_kernel_shared(program, rta_sizes)

    mcast_cfg = TLBConfig(addr=0, noc=0, mcast=True, mode=TLBMode.STRICT)
    l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    rects = self._mcast_rects(cores)
    win = self.win
    # Mcast shared data: reset GO_MSG, GO_MSG_INDEX, CB config + kernel binaries, LAUNCH
    self._mcast_write_rects(
      win,
      mcast_cfg,
      rects,
      [
        (TensixL1.GO_MSG, reset_blob),
        (TensixL1.GO_MSG_INDEX, (0).to_bytes(4, "little")),
        (TensixL1.KERNEL_CONFIG_BASE + shared_off, shared_img),
        (TensixL1.LAUNCH, launch_blob),
      ],
    )

    shared_rta = None
    if (
      isinstance(program.reader_rt_args, list)
      and isinstance(program.writer_rt_args, list)
      and isinstance(program.compute_rt_args, list)
    ):
      shared_rta = self._pack_rta(program.reader_rt_args, program.writer_rt_args, program.compute_rt_args)

    # Unicast per-core runtime args
    for core_idx, (x, y) in enumerate(cores):
      if shared_rta is None:
        reader_args = self._resolve_args(program.reader_rt_args, core_idx, (x, y), num_cores)
        writer_args = self._resolve_args(program.writer_rt_args, core_idx, (x, y), num_cores)
        compute_args = self._resolve_args(program.compute_rt_args, core_idx, (x, y), num_cores)
        rta = self._pack_rta(reader_args, writer_args, compute_args)
      else:
        rta = shared_rta
      l1_cfg.start = l1_cfg.end = (x, y)
      win.configure(l1_cfg)
      win.write(TensixL1.KERNEL_CONFIG_BASE, rta, use_uc=True, restore=False)

    # Dispatch: multicast GO_MSG to all cores
    t_dispatch_start = time.perf_counter()
    self._mcast_write_rects(win, mcast_cfg, rects, [(TensixL1.GO_MSG, go_blob)])
    t_compute_start = time.perf_counter()

    # Wait for all cores to complete (unicast poll)
    l1_cfg.mcast = False
    for x, y in cores:
      l1_cfg.start = l1_cfg.end = (x, y)
      win.configure(l1_cfg)
      deadline = time.perf_counter() + 10.0
      while win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"timeout waiting for core ({x}, {y})")

    t_end = time.perf_counter()
    dispatch = t_compute_start - t_dispatch_start
    tlog("dispatch", dispatch)
    compute = t_end - t_compute_start
    tlog("compute", compute)
    total = dispatch + compute
    return total, dispatch

class FastDevice(SlowDevice):
  # Stub: CQ/dispatch-core path will replace SlowDevice.run.
  pass
