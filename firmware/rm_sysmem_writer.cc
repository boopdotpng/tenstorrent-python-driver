#include <cstdint>

void kernel_main() {
  const uint32_t pcie_noc_xy = get_arg_val<uint32_t>(0);
  const uint32_t sysmem_offset = get_arg_val<uint32_t>(1);
  const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
  const uint32_t num_tile_rows = get_arg_val<uint32_t>(3);
  const uint32_t row_bytes = get_arg_val<uint32_t>(4);
  const uint32_t block_width_bytes = get_arg_val<uint32_t>(5);
  const uint32_t blocks_per_row = get_arg_val<uint32_t>(6);
  const uint32_t block_tile_cols = get_arg_val<uint32_t>(7);
  constexpr uint32_t cb_id = tt::CBIndex::c_16;
  constexpr uint32_t tile_height = 32;
  const uint64_t pcie_base = ((uint64_t)pcie_noc_xy << 36) | (1ULL << 60);

  uint64_t base_dst_noc_addr[tile_height];
  for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
    const uint32_t row_start = (start_tile_row + tile_row) * tile_height;
    for (uint32_t r = 0; r < tile_height; ++r) {
      base_dst_noc_addr[r] = pcie_base + sysmem_offset + (uint64_t)(row_start + r) * row_bytes;
    }

    for (uint32_t block = 0; block < blocks_per_row; ++block) {
      cb_wait_front(cb_id, block_tile_cols);
      uint32_t l1_read_addr = get_read_ptr(cb_id);
      for (uint32_t r = 0; r < tile_height; ++r) {
        noc_async_write(l1_read_addr, base_dst_noc_addr[r], block_width_bytes);
        l1_read_addr += block_width_bytes;
        base_dst_noc_addr[r] += block_width_bytes;
      }
      noc_async_write_barrier();
      cb_pop_front(cb_id, block_tile_cols);
    }
  }
}
