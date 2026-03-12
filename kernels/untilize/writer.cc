// Writes row-major data from CB c_16 to host sysmem (PCIe), one tile at a time.
// Defines: PCIE_BASE, TILE_ROW_BYTES, TILE_COLS, ROW_BYTES
#include <cstdint>
#define A(n) get_arg_val<uint32_t>(n)

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
