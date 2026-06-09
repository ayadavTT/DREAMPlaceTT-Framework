// SPDX-License-Identifier: Apache-2.0
//
// V35 count — 64 B SOURCE + FOOTPRINT-BUCKET variant (BRISC, CBs c_0/c_1). Like
// v35_count64.cpp but histograms by group = 2*tile + bucket (ngroups = 2*ntiles), where
// bucket = large iff any px[k]!=0 for k in [K0,max_k) OR any py[h]!=0 for h in [H0,max_h).
// The 64 B source has px at [4+k], py at [12+h]. Pairs with v35_place_s64_bucket (places
// small cells first within each tile) + the per-batch-footprint gather. Args 10..13 = K0,H0,max_k,max_h.

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
    const uint32_t stash_pg   = get_arg_val<uint32_t>(4);
    const uint32_t ntiles     = get_arg_val<uint32_t>(5);
    const uint32_t W          = get_arg_val<uint32_t>(6);
    const uint32_t counts_base= get_arg_val<uint32_t>(7);
    const uint32_t cnt_pg     = get_arg_val<uint32_t>(8);
    const uint32_t chunk      = get_arg_val<uint32_t>(9);
    const uint32_t K0         = get_arg_val<uint32_t>(10);
    const uint32_t H0         = get_arg_val<uint32_t>(11);
    const uint32_t max_k      = get_arg_val<uint32_t>(12);
    const uint32_t max_h      = get_arg_val<uint32_t>(13);

    constexpr uint32_t REC = 16u;   // 64 B source record
    const uint32_t ngroups = 2u*ntiles;
    constexpr auto CB_IN=tt::CBIndex::c_0, CB_CNT=tt::CBIndex::c_1;
    const InterleavedAddrGen<true> ing  = {.bank_base_address=stash_base, .page_size=stash_pg};
    const InterleavedAddrGen<true> cntg = {.bank_base_address=counts_base, .page_size=cnt_pg};

    cb_reserve_back(CB_CNT,1);
    uint32_t* cnt = reinterpret_cast<uint32_t*>(get_write_ptr(CB_CNT));
    for (uint32_t g=0;g<ngroups;++g) cnt[g]=0;
    const uint32_t in_l1 = get_write_ptr(CB_IN);

    { DeviceZoneScopedN("V35-COUNT");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*64u, in_l1, this_n*64u);
          noc_async_read_barrier();
          const uint32_t* rec = reinterpret_cast<const uint32_t*>(in_l1);
          for (uint32_t i=0;i<this_n;++i) {
              const uint32_t* s = rec + i*REC;
              uint32_t tile = s[1] / W; if (tile>=ntiles) tile=ntiles-1u;
              uint32_t big=0u;
              for (uint32_t k=K0;k<max_k;++k) big |= s[4u+k];
              for (uint32_t h=H0;h<max_h;++h) big |= s[12u+h];
              cnt[2u*tile + (big?1u:0u)]++;
          }
          done += this_n;
      }
    }
    noc_async_write((uint32_t)cnt, cntg.get_noc_addr(my_core), ngroups*4u);
    noc_async_write_barrier();
}
