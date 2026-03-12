#include <cstdint>

void kernel_main() {
  const uint32_t dram_addr = get_arg_val<uint32_t>(0);
  const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
  const uint32_t num_blocks = get_arg_val<uint32_t>(2);
  const uint32_t block_tile_cols = get_arg_val<uint32_t>(3);
  constexpr uint32_t cb_id = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_id);
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = dram_addr,
    .page_size = tile_bytes,
    .data_format = get_dataformat(cb_id),
  };

  uint32_t tile_id = start_tile_id;
  for (uint32_t block = 0; block < num_blocks; ++block) {
    cb_wait_front(cb_id, block_tile_cols);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < block_tile_cols; ++i) {
      noc_async_write_tile(tile_id++, dram, l1_read_addr);
      l1_read_addr += tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, block_tile_cols);
  }
}
