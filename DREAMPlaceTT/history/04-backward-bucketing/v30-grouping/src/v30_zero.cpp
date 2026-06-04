// SPDX-License-Identifier: Apache-2.0
// V30 backward P0: each core zeros its own grad slab (cell-owner partition).
// CB c_28 holds grad_fixed[cpc_cell][2] (gx,gy int32). Declared identically (and
// first) in P0/P1/P2 so it lands at the same L1 address across programs.
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
void kernel_main() {
    const uint32_t slab_ints = get_arg_val<uint32_t>(0);   // cpc_cell*2
    constexpr auto CB_G = tt::CBIndex::c_28;
    cb_reserve_back(CB_G,1);
    uint32_t* g = reinterpret_cast<uint32_t*>(get_write_ptr(CB_G));
    for (uint32_t i=0;i<slab_ints;++i) g[i]=0u;
}
