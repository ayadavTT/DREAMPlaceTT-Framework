// SPDX-License-Identifier: Apache-2.0
//
// V35 on-chip grouping — PASS 1 (count). Each core owns a contiguous slice
// [my_first, my_first+my_n) of the forward geometry stash (128 B records, global
// order). It histograms its cells by TILE = bxl_global / W and writes the local
// histogram counts[my_core][0..ntiles) to DRAM. No cross-core traffic, no atomics.
// Stash record (128 B = 32 u32): [1] = bxl_global (the only field needed here).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t my_first   = get_arg_val<uint32_t>(1);
    const uint32_t my_n       = get_arg_val<uint32_t>(2);
    const uint32_t stash_base = get_arg_val<uint32_t>(3);
    const uint32_t stash_pg   = get_arg_val<uint32_t>(4);   // whole-buffer page (ncells*128)
    const uint32_t ntiles     = get_arg_val<uint32_t>(5);
    const uint32_t W          = get_arg_val<uint32_t>(6);
    const uint32_t counts_base= get_arg_val<uint32_t>(7);
    const uint32_t cnt_pg     = get_arg_val<uint32_t>(8);    // per-core counts page bytes
    const uint32_t chunk      = get_arg_val<uint32_t>(9);

    constexpr auto CB_IN=tt::CBIndex::c_0, CB_CNT=tt::CBIndex::c_1;
    const InterleavedAddrGen<true> ing  = {.bank_base_address=stash_base, .page_size=stash_pg};
    const InterleavedAddrGen<true> cntg = {.bank_base_address=counts_base, .page_size=cnt_pg};

    cb_reserve_back(CB_CNT,1);
    uint32_t* cnt = reinterpret_cast<uint32_t*>(get_write_ptr(CB_CNT));
    for (uint32_t t=0;t<ntiles;++t) cnt[t]=0;
    const uint32_t in_l1 = get_write_ptr(CB_IN);

    { DeviceZoneScopedN("V35-COUNT");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*128u, in_l1, this_n*128u);
          noc_async_read_barrier();
          const uint32_t* rec = reinterpret_cast<const uint32_t*>(in_l1);
          for (uint32_t i=0;i<this_n;++i) {
              uint32_t bxl_g = rec[i*32u + 1u];
              uint32_t tile = bxl_g / W; if (tile>=ntiles) tile=ntiles-1u;
              cnt[tile]++;
          }
          done += this_n;
      }
    }
    noc_async_write((uint32_t)cnt, cntg.get_noc_addr(my_core), ntiles*4u);
    noc_async_write_barrier();
}
