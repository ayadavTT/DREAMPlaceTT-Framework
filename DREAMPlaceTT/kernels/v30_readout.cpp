// SPDX-License-Identifier: Apache-2.0
// V30 backward P2: each core writes its grad slab (CB c_28) to DRAM grad_out[core].
// Runs after P1's atomics have settled (host Finish between programs = barrier).
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t slab_ints = get_arg_val<uint32_t>(1);   // cpc_cell*2
    const uint32_t out_base  = get_arg_val<uint32_t>(2);
    const uint32_t out_pg    = get_arg_val<uint32_t>(3);
    constexpr auto CB_G = tt::CBIndex::c_28;
    const InterleavedAddrGen<true> og={.bank_base_address=out_base,.page_size=out_pg};
    noc_async_write(get_write_ptr(CB_G), og.get_noc_addr(my_core), slab_ints*4u);
    noc_async_write_barrier();
}
