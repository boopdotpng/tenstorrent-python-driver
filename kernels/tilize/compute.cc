// Tilize compute kernel: row-major → hardware tile format, one tile at a time.
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
