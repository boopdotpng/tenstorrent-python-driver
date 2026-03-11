#include <cstdint>

void kernel_main() {
  uint32_t dram_addr = get_arg_val<uint32_t>(0);
  uint32_t pcie_noc_xy = get_arg_val<uint32_t>(1);
  uint32_t sysmem_offset = get_arg_val<uint32_t>(2);
  uint32_t tile_offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);
  uint32_t page_size = get_arg_val<uint32_t>(5);
  uint32_t sysmem_local_offset = get_arg_val<uint32_t>(6);
  constexpr uint32_t cb_id = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> dram = {
    .bank_base_address = dram_addr,
    .page_size = page_size,
    .data_format = DataFormat::Float16_b,
  };
  uint64_t pcie_base = ((uint64_t)pcie_noc_xy << 36) | (1ULL << 60);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t tile_id = tile_offset + i;
    cb_reserve_back(cb_id, 1);
    uint32_t l1_addr = get_write_ptr(cb_id);
    noc_async_read_tile(tile_id, dram, l1_addr);
    noc_async_read_barrier();
    uint64_t dst = pcie_base + sysmem_offset + sysmem_local_offset + (uint64_t)tile_id * page_size;
    noc_async_write(l1_addr, dst, page_size);
    noc_async_write_barrier();
    cb_push_back(cb_id, 1);
    cb_wait_front(cb_id, 1);
    cb_pop_front(cb_id, 1);
  }
}