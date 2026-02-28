#!/usr/bin/env python3
from __future__ import annotations

import unittest

from examples.matmul_peak import (
  BF16_BLOCK_W_SMALL_CAP,
  PrecisionPolicy,
  _iter_layouts,
  _build_topology,
  _plan_matmul,
  ceil32,
)


def _p100a_fast_dispatch_cores():
  xs = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
  ys = list(range(2, 12))
  cores = [(x, y) for x in xs for y in ys]
  cores.remove((14, 2))
  cores.remove((14, 3))
  return cores


class TestMatmulPlanner(unittest.TestCase):
  def test_layout_enumerator_never_emits_missing_cores(self):
    cores = _p100a_fast_dispatch_cores()
    topology = _build_topology(cores)
    core_set = set(cores)
    for layout in _iter_layouts(topology, max_cores=len(cores)):
      for y in layout.rows:
        for x in layout.cols:
          self.assertIn((x, y), core_set)

  def test_4608_square_plan_handles_topology_holes(self):
    cores = _p100a_fast_dispatch_cores()
    mt_base = ceil32(4608) // 32
    kt = ceil32(4608) // 32
    nt_base = ceil32(4608) // 32
    plan = _plan_matmul(mt_base, kt, nt_base, cores, policy=PrecisionPolicy(f32_acc=False))
    plan.validate_against(cores)
    self.assertLessEqual(plan.active_core_count, len(cores))
    if plan.num_cols == 12:
      self.assertGreaterEqual(min(plan.rows), 4)

  def test_bf16_policy_caps_block_w_for_small_output_tiles(self):
    cores = _p100a_fast_dispatch_cores()
    mt_base = ceil32(256) // 32
    kt = ceil32(5120) // 32
    nt_base = ceil32(256) // 32
    plan = _plan_matmul(mt_base, kt, nt_base, cores, policy=PrecisionPolicy(f32_acc=False))
    self.assertLessEqual(plan.in0_block_w, BF16_BLOCK_W_SMALL_CAP)
    self.assertGreaterEqual(plan.num_blocks, kt // BF16_BLOCK_W_SMALL_CAP)

  def test_fp32_policy_can_choose_larger_block_w(self):
    cores = _p100a_fast_dispatch_cores()
    mt_base = ceil32(256) // 32
    kt = ceil32(5120) // 32
    nt_base = ceil32(256) // 32
    plan = _plan_matmul(mt_base, kt, nt_base, cores, policy=PrecisionPolicy(f32_acc=True))
    self.assertGreaterEqual(plan.in0_block_w, BF16_BLOCK_W_SMALL_CAP)


if __name__ == "__main__":
  unittest.main()
