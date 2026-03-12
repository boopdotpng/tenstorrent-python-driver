// Reads tiled data from interleaved DRAM into CB c_0, one tile at a time.
// Defines: DRAM_ADDR
#include <cstdint>
#define A(n) get_arg_val<uint32_t>(n)

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
