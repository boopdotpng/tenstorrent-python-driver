# Kernel sources for tilize/untilize data transfer programs.
# Each function returns a complete C++ source string with all defines baked in.

_A = '#define A(n) get_arg_val<uint32_t>(n)\n'

def _sysmem_defs(pcie_base: int, tile_row_bytes: int, tile_cols: int, row_bytes: int) -> str:
  return (
    f"#define PCIE_BASE 0x{pcie_base:x}ULL\n"
    f"#define TILE_ROW_BYTES {tile_row_bytes}\n"
    f"#define TILE_COLS {tile_cols}\n"
    f"#define ROW_BYTES {row_bytes}\n"
  )

def _dram_defs(dram_addr: int) -> str:
  return f"#define DRAM_ADDR {dram_addr}\n"

def tilize_reader(pcie_base: int, tile_row_bytes: int, tile_cols: int, row_bytes: int) -> str:
  return _sysmem_defs(pcie_base, tile_row_bytes, tile_cols, row_bytes) + _A + """\
#include <cstdint>

void kernel_main() {
  for (uint32_t t = 0; t < A(1); ++t) {
    uint32_t id = A(0) + t;
    uint32_t pixel_row = (id / TILE_COLS) * 32;
    uint32_t pixel_col_bytes = (id % TILE_COLS) * TILE_ROW_BYTES;

    cb_reserve_back(tt::CBIndex::c_0, 1);
    uint32_t l1 = get_write_ptr(tt::CBIndex::c_0);
    for (uint32_t r = 0; r < 32; ++r) {
      noc_async_read(PCIE_BASE + (uint64_t)(pixel_row + r) * ROW_BYTES + pixel_col_bytes, l1, TILE_ROW_BYTES);
      l1 += TILE_ROW_BYTES;
    }
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_0, 1);
  }
}
"""

TILIZE_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {
  const uint32_t num_tiles = get_arg_val<uint32_t>(0);
  compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
  tilize_init(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);
  for (uint32_t t = 0; t < num_tiles; ++t) {
    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_reserve_back(tt::CBIndex::c_16, 1);
    tilize_block(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);
    cb_push_back(tt::CBIndex::c_16, 1);
    cb_pop_front(tt::CBIndex::c_0, 1);
  }
}
}  // namespace NAMESPACE
"""

def tilize_writer(dram_addr: int) -> str:
  return _dram_defs(dram_addr) + _A + """\
#include <cstdint>

void kernel_main() {
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = DRAM_ADDR,
    .page_size = get_tile_size(tt::CBIndex::c_16),
    .data_format = get_dataformat(tt::CBIndex::c_16),
  };
  for (uint32_t t = 0; t < A(1); ++t) {
    cb_wait_front(tt::CBIndex::c_16, 1);
    noc_async_write_tile(A(0) + t, dram, get_read_ptr(tt::CBIndex::c_16));
    noc_async_write_barrier();
    cb_pop_front(tt::CBIndex::c_16, 1);
  }
}
"""

def untilize_reader(dram_addr: int) -> str:
  return _dram_defs(dram_addr) + _A + """\
#include <cstdint>

void kernel_main() {
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = DRAM_ADDR,
    .page_size = get_tile_size(tt::CBIndex::c_0),
    .data_format = get_dataformat(tt::CBIndex::c_0),
  };
  for (uint32_t t = 0; t < A(1); ++t) {
    cb_reserve_back(tt::CBIndex::c_0, 1);
    noc_async_read_tile(A(0) + t, dram, get_write_ptr(tt::CBIndex::c_0));
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_0, 1);
  }
}
"""

UNTILIZE_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
  const uint32_t num_tiles = get_arg_val<uint32_t>(0);
  compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
  pack_untilize_init<1, 1>(tt::CBIndex::c_0, tt::CBIndex::c_16);
  for (uint32_t t = 0; t < num_tiles; ++t) {
    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_reserve_back(tt::CBIndex::c_16, 1);
    pack_untilize_block<1, 1>(tt::CBIndex::c_0, 1, tt::CBIndex::c_16, 0);
    cb_push_back(tt::CBIndex::c_16, 1);
    cb_pop_front(tt::CBIndex::c_0, 1);
  }
  pack_untilize_uninit(tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
"""

def untilize_writer(pcie_base: int, tile_row_bytes: int, tile_cols: int, row_bytes: int) -> str:
  return _sysmem_defs(pcie_base, tile_row_bytes, tile_cols, row_bytes) + _A + """\
#include <cstdint>

void kernel_main() {
  for (uint32_t t = 0; t < A(1); ++t) {
    uint32_t id = A(0) + t;
    uint32_t pixel_row = (id / TILE_COLS) * 32;
    uint32_t pixel_col_bytes = (id % TILE_COLS) * TILE_ROW_BYTES;

    cb_wait_front(tt::CBIndex::c_16, 1);
    uint32_t l1 = get_read_ptr(tt::CBIndex::c_16);
    for (uint32_t r = 0; r < 32; ++r) {
      noc_async_write(l1, PCIE_BASE + (uint64_t)(pixel_row + r) * ROW_BYTES + pixel_col_bytes, TILE_ROW_BYTES);
      l1 += TILE_ROW_BYTES;
    }
    noc_async_write_barrier();
    cb_pop_front(tt::CBIndex::c_16, 1);
  }
}
"""
