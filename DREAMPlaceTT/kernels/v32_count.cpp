// SPDX-License-Identifier: Apache-2.0
// COUNT-PREFIX REGROUP — pass 1 (count). Each core streams its M records
// (16B: owner, cell, bin, area_bits) and builds a PURELY LOCAL histogram
// cnt[owner] in L1 (no atomics, no contention), then writes its row to the
// count matrix cntm[my_core][0..NOWN). This is the reliable half of the atomic
// (counting) — we never use the flaky fetch-add return.
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t recs_base = get_arg_val<uint32_t>(1);
    const uint32_t recs_pg   = get_arg_val<uint32_t>(2);   // M*16
    const uint32_t M         = get_arg_val<uint32_t>(3);
    const uint32_t NOWN      = get_arg_val<uint32_t>(4);
    const uint32_t cntm_base = get_arg_val<uint32_t>(5);
    const uint32_t cntm_pg   = get_arg_val<uint32_t>(6);   // NOWN*4
    const uint32_t chunk     = get_arg_val<uint32_t>(7);   // records per read

    constexpr auto CB_REC=tt::CBIndex::c_0, CB_HIST=tt::CBIndex::c_1;
    const InterleavedAddrGen<true> rg ={.bank_base_address=recs_base,.page_size=recs_pg};
    const InterleavedAddrGen<true> cg ={.bank_base_address=cntm_base,.page_size=cntm_pg};
    const uint32_t stride = cntm_pg>>2;                    // padded row pitch (64-B aligned page)
    const uint32_t rl = get_write_ptr(CB_REC);
    uint32_t* hist = (uint32_t*)get_write_ptr(CB_HIST);
    for (uint32_t o=0;o<stride;++o) hist[o]=0u;             // zero full padded row

    { DeviceZoneScopedN("V32-COUNT");
      const uint64_t base = rg.get_noc_addr(my_core);
      uint32_t done=0;
      while(done<M){ uint32_t n=(M-done<chunk)?(M-done):chunk;
          noc_async_read(base+(uint64_t)done*16u, rl, n*16u); noc_async_read_barrier();
          const uint32_t* ri=(const uint32_t*)rl;
          for(uint32_t i=0;i<n;++i){ uint32_t o=ri[i*4u]; if(o<NOWN) hist[o]++; }
          done+=n; }
    }
    noc_async_write((uint32_t)hist, cg.get_noc_addr(my_core), cntm_pg);
    noc_async_write_barrier();
}
