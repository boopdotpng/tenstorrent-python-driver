import time
from dataclasses import dataclass
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from helpers import TIMING, align_up
from codegen import CompiledKernel
from dram import DramAllocator
from device import CommonDevice, Program, DataflowLaunch, ArgGen, CoreSpec

Rect = tuple[int, int, int, int]
McastDest = tuple[int, int]
CorePair = tuple[Core, Core]
AddrPayload = tuple[int, bytes]
RtaSizes = tuple[int, int, int]
FAST_CQ_NUM_CIRCULAR_BUFFERS = 32

@dataclass(frozen=True)
class _LaunchRole:
  core: Core
  launch: DataflowLaunch
  role_idx: int
  role_count: int

@dataclass(frozen=True)
class _CorePayload:
  core: Core
  rta: bytes
  launch_blob: bytes
  shared_addr: int
  shared_blob: bytes

@dataclass(frozen=True)
class _CorePlan:
  cores: CoreList
  rects: list[Rect]
  mcast_dests: list[McastDest]

class SlowDevice(CommonDevice):
  def __init__(self, device: int = 0, enable_sysmem: bool = False, init_core_plans: bool = True):
    super().__init__(device=device)
    if init_core_plans:
      self._core_plans = self._build_core_plans()
    self._queue: list[Program] = []
    self.win = TLBWindow(self.fd, TLBSize.MiB_2)
    self.dram = DramAllocator(
      fd=self.fd,
      dram_tiles=self.tiles.dram,
      run_fn=self._run_single,
      enable_sysmem=enable_sysmem,
    )

  def resolve_cores(self, cores: CoreSpec = "all") -> CoreList:
    return list(self._resolve_core_plan(cores).cores)

  def resolve_mcast_rects(self, cores: CoreSpec = "all") -> list[Rect]:
    return list(self._resolve_core_plan(cores).rects)

  def _build_local_cb_blob(self, program: Program) -> tuple[int, bytes]:
    mask = 0
    for cb in program.cbs:
      mask |= 1 << cb
    end = mask.bit_length()
    arr = (LocalCBConfig * end)()
    addr = TensixL1.DATA_BUFFER_SPACE_BASE
    shared_addr: dict[int, int] = {}  # cb_id -> addr for shared CBs
    if program.cb_config:
      for cb in program.cbs:
        num_pages, page_size = program.cb_config[cb]
        # Check if another CB already allocated at same (num_pages, page_size) with share intent
        # CB_16 and CB_24 share address when both present
        share_with = {16: 24, 24: 16}.get(cb)
        if share_with is not None and share_with in shared_addr:
          cb_addr = shared_addr[share_with]
        else:
          cb_addr = addr
          size = page_size * num_pages
          addr += size
        shared_addr[cb] = cb_addr
        arr[cb] = LocalCBConfig(
          addr_bytes=cb_addr, size_bytes=page_size * num_pages,
          num_pages=num_pages, page_size_bytes=page_size,
        )
    else:
      for cb in program.cbs:
        size = program.tile_size * program.num_pages
        arr[cb] = LocalCBConfig(
          addr_bytes=addr, size_bytes=size,
          num_pages=program.num_pages, page_size_bytes=program.tile_size,
        )
        addr += size
    return mask, as_bytes(arr)

  def _pack_kernel_shared(self, program: Program, reader: CompiledKernel, writer: CompiledKernel, rta_sizes: RtaSizes,
                          dispatch_mode: int = DevMsgs.DISPATCH_MODE_HOST, sem_off: int | None = None):
    rta_offsets = [0, rta_sizes[0], rta_sizes[0] + rta_sizes[1]]
    rta_total = align_up(rta_offsets[2] + rta_sizes[2], 16)

    # Semaphore space: num_sems * 16 bytes between per-core args and shared data
    sem_size = program.num_sems * 16
    if sem_off is None:
      sem_off = rta_total
    shared_off_start = align_up(sem_off + sem_size, 16)
    crta_off = shared_off_start

    local_cb_mask, local_cb_blob = self._build_local_cb_blob(program)
    local_cb_off = shared_off_start
    kernel_off = align_up(local_cb_off + len(local_cb_blob), 16)

    kernels = {"brisc": writer, "ncrisc": reader}
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
      off = align_up(off + len(k.xip), 16)
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
    for i in range(3):
      cfg.sem_offset[i] = sem_off
    cfg.local_cb_offset = local_cb_off
    cfg.remote_cb_offset = local_cb_off + len(local_cb_blob)
    cfg.local_cb_mask = local_cb_mask
    cfg.min_remote_cb_start_index = FAST_CQ_NUM_CIRCULAR_BUFFERS
    cfg.enables = enables
    cfg.brisc_noc_id = 1
    cfg.brisc_noc_mode = 0
    cfg.mode = dispatch_mode
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
  def _pack_rta(reader_args: list[int], writer_args: list[int], compute_args: list[int], num_sems: int = 0,
                sem_off: int | None = None) -> bytes:
    pack = lambda xs: b"".join(int(x & 0xFFFFFFFF).to_bytes(4, "little") for x in xs)
    rta = pack(writer_args) + pack(reader_args) + pack(compute_args)
    if num_sems > 0:
      if sem_off is not None and sem_off > len(rta):
        rta = rta.ljust(sem_off, b"\0")
      rta += b"\0" * (num_sems * 16)
    return rta

  def _resolve_args(self, args: list[int] | ArgGen, core_idx: int, core_xy: Core, num_cores: int) -> list[int]:
    return args if isinstance(args, list) else args(core_idx, core_xy, num_cores)

  @staticmethod
  def _rect_to_noc_mcast(rect: Rect) -> McastDest:
    x0, x1, y0, y1 = rect
    noc_mcast_xy = (y1 << 18) | (x1 << 12) | (y0 << 6) | x0
    num_dests = (x1 - x0 + 1) * (y1 - y0 + 1)
    return noc_mcast_xy, num_dests

  @staticmethod
  def _core_rects(cores: CoreList) -> list[Rect]:
    remaining = set(cores)
    rects: list[Rect] = []
    while remaining:
      x0, y0 = min(remaining, key=lambda xy: (xy[1], xy[0]))
      x1 = x0
      while (x1 + 1, y0) in remaining:
        x1 += 1
      y1 = y0
      while all((x, y1 + 1) in remaining for x in range(x0, x1 + 1)):
        y1 += 1
      for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
          remaining.remove((x, y))
      rects.append((x0, x1, y0, y1))
    return rects

  @staticmethod
  def _prefix_rects(cores: CoreList) -> list[Rect]:
    if not cores:
      return []
    by_x: dict[int, list[int]] = {}
    for x, y in cores:
      by_x.setdefault(x, []).append(y)
    y0 = min(y for _, y in cores)
    cols = sorted(by_x)
    col_tops: CoreList = []
    for x in cols:
      ys = sorted(by_x[x])
      top = y0 + len(ys) - 1
      if ys != list(range(y0, top + 1)):
        return SlowDevice._core_rects(cores)
      col_tops.append((x, top))
    rects: list[Rect] = []
    i = 0
    while i < len(col_tops):
      x0, y1 = col_tops[i]
      j = i
      while j + 1 < len(col_tops) and col_tops[j + 1][0] == col_tops[j][0] + 1 and col_tops[j + 1][1] == y1:
        j += 1
      rects.append((x0, col_tops[j][0], y0, y1))
      i = j + 1
    return rects

  def _build_core_plans(self) -> dict[int | str, _CorePlan]:
    ordered = sorted(self.dispatchable_cores, key=lambda xy: (xy[0], xy[1]))
    if not ordered:
      raise RuntimeError("no dispatchable cores available")
    plans: dict[int | str, _CorePlan] = {}
    for n in range(1, len(ordered) + 1):
      selected = ordered[:n]
      rects = self._prefix_rects(selected)
      plans[n] = _CorePlan(
        cores=selected,
        rects=rects,
        mcast_dests=[self._rect_to_noc_mcast(rect) for rect in rects],
      )
    plans["all"] = plans[len(ordered)]
    return plans

  def _resolve_core_plan(self, spec: CoreSpec) -> _CorePlan:
    if spec == "all":
      return self._core_plans["all"]
    if not isinstance(spec, int):
      raise TypeError("program.cores must be int or 'all'")
    if spec < 1:
      raise ValueError("program.cores must be >= 1")
    if spec > len(self.dispatchable_cores):
      raise ValueError(f"program.cores={spec} exceeds dispatchable cores ({len(self.dispatchable_cores)})")
    return self._core_plans[spec]

  def _core_launches(self, program: Program, cores: CoreList) -> list[_LaunchRole]:
    core_set = set(cores)
    assigned: dict[Core, tuple[DataflowLaunch, int, int]] = {}
    for launch in program.dataflow:
      role_cores = [c for c in launch.cores if c in core_set]
      role_n = len(role_cores)
      for role_i, core in enumerate(role_cores):
        if core in assigned:
          raise ValueError(f"core {core} appears in multiple dataflow launches")
        assigned[core] = (launch, role_i, role_n)
    missing = [c for c in cores if c not in assigned]
    if missing:
      raise ValueError(f"every program core must have a dataflow launch; missing {len(missing)} cores")
    return [
      _LaunchRole(core=core, launch=launch, role_idx=role_i, role_count=role_n)
      for core in cores
      for launch, role_i, role_n in [assigned[core]]
    ]

  def _build_rta(self, program: Program, launch: DataflowLaunch, core_idx: int, core_xy: Core, num_cores: int, role_idx: int,
                 role_cores: int, sem_off: int | None = None) -> tuple[RtaSizes, bytes]:
    reader_args = self._resolve_args(launch.reader_rt_args, role_idx, core_xy, role_cores)
    writer_args = self._resolve_args(launch.writer_rt_args, role_idx, core_xy, role_cores)
    compute_args = self._resolve_args(program.compute_rt_args, core_idx, core_xy, num_cores)
    rta_sizes = (len(writer_args) * 4, len(reader_args) * 4, len(compute_args) * 4)
    rta = self._pack_rta(reader_args, writer_args, compute_args, program.num_sems, sem_off=sem_off)
    return rta_sizes, rta

  def _uniform_sem_off(self, program: Program, launch_roles: list[_LaunchRole]) -> int:
    max_rta_total = 0
    seen = set()
    for role in launch_roles:
      lid = id(role.launch)
      if lid in seen:
        continue
      seen.add(lid)
      rta_sizes, _ = self._build_rta(
        program, role.launch, 0, role.core, len(launch_roles), role.role_idx, role.role_count)
      rta_total = align_up(sum(rta_sizes), 16)
      max_rta_total = max(max_rta_total, rta_total)
    return max_rta_total

  def _prepare_core_payloads(self, program: Program, cores: CoreList, launch_roles: list[_LaunchRole],
                             dispatch_mode: int) -> list[_CorePayload]:
    num_cores = len(cores)
    sem_off = self._uniform_sem_off(program, launch_roles) if program.num_sems > 0 else None
    payloads: list[_CorePayload] = []
    shared_cache: dict[tuple, tuple[int, bytes, bytes]] = {}
    for core_idx, role in enumerate(launch_roles):
      rta_sizes, rta = self._build_rta(
        program, role.launch, core_idx, role.core, num_cores, role.role_idx, role.role_count, sem_off=sem_off)
      key = (role.launch.reader, role.launch.writer, *rta_sizes, dispatch_mode)
      if key not in shared_cache:
        shared_cache[key] = self._pack_kernel_shared(
          program, reader=role.launch.reader, writer=role.launch.writer,
          rta_sizes=rta_sizes, dispatch_mode=dispatch_mode, sem_off=sem_off,
        )[:3]
      shared_off, shared_img, launch_blob = shared_cache[key]
      payloads.append(_CorePayload(
        core=role.core,
        rta=rta,
        launch_blob=launch_blob,
        shared_addr=TensixL1.KERNEL_CONFIG_BASE + shared_off,
        shared_blob=shared_img,
      ))
    return payloads

  @staticmethod
  def _mcast_write_rects(win: TLBWindow, cfg: TLBConfig, rects: list[Rect], writes: list[AddrPayload]):
    for x0, x1, y0, y1 in rects:
      cfg.start, cfg.end = (x0, y0), (x1, y1)
      win.configure(cfg)
      for addr, data in writes:
        win.write(addr, data, use_uc=True, restore=False)

  @staticmethod
  def _group_payloads(payload_by_core: list[tuple[Core, bytes]]) -> dict[bytes, CoreList]:
    groups: dict[bytes, CoreList] = {}
    for core, payload in payload_by_core:
      groups.setdefault(payload, []).append(core)
    return groups

  @staticmethod
  def _group_shared_payloads(shared_by_core: list[tuple[Core, int, bytes]]) -> dict[tuple[int, bytes], CoreList]:
    groups: dict[tuple[int, bytes], CoreList] = {}
    for core, addr, payload in shared_by_core:
      groups.setdefault((addr, payload), []).append(core)
    return groups

  def _mcast_dests_for_cores(self, cores: CoreList) -> list[McastDest]:
    return [self._rect_to_noc_mcast(rect) for rect in self._core_rects(cores)]

  def _wait_cores_done(self, cores: CoreList, timeout_s: float = 10.0):
    l1_cfg = TLBConfig(addr=0, noc=0, mcast=False, mode=TLBMode.STRICT)
    deadline = time.perf_counter() + timeout_s
    for x, y in cores:
      l1_cfg.start = l1_cfg.end = (x, y)
      self.win.configure(l1_cfg)
      while self.win.uc[TensixL1.GO_MSG + 3] != DevMsgs.RUN_MSG_DONE:
        if time.perf_counter() > deadline:
          raise TimeoutError(f"timeout waiting for core ({x}, {y})")
        time.sleep(0.0002)

  def _run_single(self, program: Program, timing: bool = False) -> float:
    plan = self._resolve_core_plan(program.cores)
    cores = plan.cores
    if not cores:
      raise ValueError("program has no cores")
    launch_roles = self._core_launches(program, cores)

    reset = GoMsg()
    reset.bits.signal = DevMsgs.RUN_MSG_RESET_READ_PTR_FROM_HOST
    reset_blob = as_bytes(reset) + (0).to_bytes(4, "little")
    go = GoMsg()
    go.bits.signal = DevMsgs.RUN_MSG_GO
    go_blob = as_bytes(go)

    mcast_cfg = TLBConfig(addr=0, noc=0, mcast=True, mode=TLBMode.STRICT)
    win = self.win
    all_rects = plan.rects
    self._mcast_write_rects(win, mcast_cfg, all_rects, [(TensixL1.GO_MSG, reset_blob)])

    payloads = self._prepare_core_payloads(program, cores, launch_roles, DevMsgs.DISPATCH_MODE_HOST)

    def grouped_writes(payload_by_core: list[tuple[Core, bytes]], addr: int):
      for payload, group_cores in self._group_payloads(payload_by_core).items():
        self._mcast_write_rects(win, mcast_cfg, self._core_rects(group_cores), [(addr, payload)])

    grouped_writes([(p.core, p.rta) for p in payloads], TensixL1.KERNEL_CONFIG_BASE)
    grouped_writes([(p.core, p.launch_blob) for p in payloads], TensixL1.LAUNCH)

    shared = [(p.core, p.shared_addr, p.shared_blob) for p in payloads]
    for (addr, payload), group_cores in self._group_shared_payloads(shared).items():
      self._mcast_write_rects(win, mcast_cfg, self._core_rects(group_cores), [(addr, payload)])

    t_dispatch_start = time.perf_counter()
    self._mcast_write_rects(win, mcast_cfg, all_rects, [(TensixL1.GO_MSG, go_blob)])

    self._wait_cores_done(cores, timeout_s=10.0)

    dispatch = time.perf_counter() - t_dispatch_start
    if timing:
      print(f"run: {dispatch * 1e3:.3f} ms")
    return dispatch

  def queue(self, program: Program):
    self._queue.append(program)
    return self._queue

  @property
  def programs(self) -> list[Program]:
    return self._queue

  def run(self, programs: list[Program], timing: bool = False) -> tuple[float, float]:
    t0 = time.perf_counter()
    do_timing = timing or TIMING
    times: list[tuple[Program, float]] = []
    for p in programs:
      dt = self._run_single(p, timing=False)
      if do_timing:
        times.append((p, dt))
    elapsed = time.perf_counter() - t0
    n = len(programs)
    if programs is self._queue:
      self._queue.clear()
    if do_timing:
      for i, (p, dt) in enumerate(times):
        tflops = f"  ({p.flops / dt / 1e12:.1f} TFLOPS)" if p.flops else ""
        print(f"  kernel {i}: {dt * 1e3:.3f} ms{tflops}")
      if n > 1:
        print(f"  batch ({n} programs): {elapsed * 1e3:.3f} ms")
    return elapsed, elapsed
