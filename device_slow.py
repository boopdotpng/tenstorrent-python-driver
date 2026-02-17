import time
from defs import *
from tlb import TLBConfig, TLBWindow, TLBMode
from device import DeviceBase, Program, AddrPayload, Rect
from helpers import PROFILER

class SlowDevice(DeviceBase):
  def __init__(self, device: int = 0, enable_sysmem: bool = False, init_core_plans: bool = True):
    super().__init__(device=device, enable_sysmem=enable_sysmem, init_core_plans=init_core_plans)

  @staticmethod
  def _mcast_write_rects(win: TLBWindow, cfg: TLBConfig, rects: list[Rect], writes: list[AddrPayload]):
    for x0, x1, y0, y1 in rects:
      cfg.start, cfg.end = (x0, y0), (x1, y1)
      win.configure(cfg)
      for addr, data in writes:
        win.write(addr, data, use_uc=True, restore=False)

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

  def _run_single(self, program: Program):
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
    self._mcast_write_rects(win, mcast_cfg, plan.rects, [(TensixL1.GO_MSG, reset_blob)])
    if PROFILER:
      self._mcast_write_rects(win, mcast_cfg, plan.rects, [(TensixL1.PROFILER_CONTROL, b"\0" * 128)])

    payloads = self._prepare_core_payloads(program, cores, launch_roles, DevMsgs.DISPATCH_MODE_HOST)

    for payload, group_cores in self._group_by([(p.core, p.rta) for p in payloads]).items():
      self._mcast_write_rects(win, mcast_cfg, self._core_rects(group_cores), [(TensixL1.KERNEL_CONFIG_BASE, payload)])
    for payload, group_cores in self._group_by([(p.core, p.launch_blob) for p in payloads]).items():
      self._mcast_write_rects(win, mcast_cfg, self._core_rects(group_cores), [(TensixL1.LAUNCH, payload)])
    for (addr, payload), group_cores in self._group_by([(p.core, p.shared_addr, p.shared_blob) for p in payloads]).items():
      self._mcast_write_rects(win, mcast_cfg, self._core_rects(group_cores), [(addr, payload)])

    self._mcast_write_rects(win, mcast_cfg, plan.rects, [(TensixL1.GO_MSG, go_blob)])
    self._wait_cores_done(cores, timeout_s=10.0)

  def run(self):
    programs_info = []
    for i, p in enumerate(self._exec_list):
      self._run_single(p)
      if PROFILER:
        plan = self._resolve_core_plan(p.cores)
        programs_info.append({
          "cores": plan.cores,
          "sources": getattr(p, "sources", {}),
          "name": getattr(p, "name", None),
          "index": i,
        })
    if PROFILER and programs_info:
      import profiler, profiler_ui
      data = profiler.collect(self, programs_info, dispatch_mode="slow")
      profiler_ui.serve(data)
    self._exec_list.clear()
