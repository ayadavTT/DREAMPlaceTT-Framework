// SPDX-License-Identifier: Apache-2.0
// COUNT-PREFIX REGROUP — pass 2 (place). Each core reads its base row
// base[my_core][0..NOWN), re-streams its M records, and writes each record to
// route[ base[o]++ ] where o = record.owner. The per-owner cursor is a LOCAL L1
// counter (no atomics, no cross-core contention) — distinctness is guaranteed by
// the prefix layout, not by any hardware fetch-add. Output route[] is the input
// records regrouped contiguously by owner.
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
    const uint32_t base_base = get_arg_val<uint32_t>(5);
    const uint32_t base_pg   = get_arg_val<uint32_t>(6);   // NOWN*4
    const uint32_t route_base= get_arg_val<uint32_t>(7);
    const uint32_t route_pg  = get_arg_val<uint32_t>(8);   // 16
    const uint32_t chunk     = get_arg_val<uint32_t>(9);

    constexpr auto CB_REC=tt::CBIndex::c_0, CB_CUR=tt::CBIndex::c_1;
    const InterleavedAddrGen<true> rg ={.bank_base_address=recs_base,.page_size=recs_pg};
    const InterleavedAddrGen<true> bg ={.bank_base_address=base_base,.page_size=base_pg};
    const InterleavedAddrGen<true> og ={.bank_base_address=route_base,.page_size=route_pg};
    const uint32_t rl = get_write_ptr(CB_REC);
    uint32_t* cur=(uint32_t*)get_write_ptr(CB_CUR);
    noc_async_read(bg.get_noc_addr(my_core),(uint32_t)cur,base_pg); noc_async_read_barrier();

    { DeviceZoneScopedN("V32-PLACE");
      const uint64_t base=rg.get_noc_addr(my_core);
      uint32_t done=0;
      while(done<M){ uint32_t n=(M-done<chunk)?(M-done):chunk;
          noc_async_read(base+(uint64_t)done*16u, rl, n*16u); noc_async_read_barrier();
          const uint32_t* ri=(const uint32_t*)rl;
          for(uint32_t i=0;i<n;++i){ uint32_t o=ri[i*4u]; if(o>=NOWN)continue;
              uint32_t pos=cur[o]++; noc_async_write(rl+i*16u, og.get_noc_addr(pos), 16u); }
          noc_async_write_barrier();   // chunk writes done before we overwrite rl
          done+=n; }
    }
}
