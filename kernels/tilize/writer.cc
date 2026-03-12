// Writes tiled data from CB c_16 to interleaved DRAM, one tile at a time.
// Defines: DRAM_ADDR
#include <cstdint>
#define A(n) get_arg_val<uint32_t>(n)

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
