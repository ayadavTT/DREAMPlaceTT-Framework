// SPDX-License-Identifier: Apache-2.0
// COUNT-PREFIX REGROUP — prefix (runs on ONE core). Reads the whole count
// matrix cnt[NC][NOWN] and computes each core's write base within each owner's
// contiguous region — entirely on chip (no host in the loop):
//     total[o]      = Σ_c cnt[c][o]
//     owner_start[o]= Σ_{o'<o} total[o']                  (owners packed in order)
//     base[c][o]    = owner_start[o] + Σ_{c'<c} cnt[c'][o] (core c's slice in owner o)
// Writes the base matrix back. NC*NOWN ints (~48KB) fit easily in one core's L1.
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t NC        = get_arg_val<uint32_t>(0);
    const uint32_t NOWN      = get_arg_val<uint32_t>(1);
    const uint32_t cntm_base = get_arg_val<uint32_t>(2);
    const uint32_t cntm_pg   = get_arg_val<uint32_t>(3);   // NOWN*4
    const uint32_t base_base = get_arg_val<uint32_t>(4);
    const uint32_t base_pg   = get_arg_val<uint32_t>(5);   // NOWN*4
    const uint32_t page_mode = get_arg_val<uint32_t>(6);   // 1 = per-owner pages (base[c][o] starts at 0 per owner)

    constexpr auto CB_CNT=tt::CBIndex::c_0, CB_BASE=tt::CBIndex::c_1, CB_OS=tt::CBIndex::c_2;
    const InterleavedAddrGen<true> cg={.bank_base_address=cntm_base,.page_size=cntm_pg};
    const InterleavedAddrGen<true> bg={.bank_base_address=base_base,.page_size=base_pg};
    uint32_t* cnt =(uint32_t*)get_write_ptr(CB_CNT);   // NC*NOWN
    uint32_t* base=(uint32_t*)get_write_ptr(CB_BASE);  // NC*NOWN
    uint32_t* os  =(uint32_t*)get_write_ptr(CB_OS);    // NOWN (owner_start)

    const uint32_t stride = cntm_pg>>2;                    // padded row pitch (64-B aligned page)
    { DeviceZoneScopedN("V32-PREFIX");
      for(uint32_t c=0;c<NC;++c){ noc_async_read(cg.get_noc_addr(c),(uint32_t)(cnt+c*stride),cntm_pg); }
      noc_async_read_barrier();
      // owner_start[o] = prefix over owners of total[o]; total[o] = column sum
      uint32_t acc=0;
      for(uint32_t o=0;o<NOWN;++o){ os[o]=page_mode?0u:acc; uint32_t tot=0; for(uint32_t c=0;c<NC;++c) tot+=cnt[c*stride+o]; acc+=tot; }
      // base[c][o] = owner_start[o] + prefix over cores (c'<c) of cnt[c'][o]
      for(uint32_t o=0;o<NOWN;++o){ uint32_t run=os[o]; for(uint32_t c=0;c<NC;++c){ base[c*stride+o]=run; run+=cnt[c*stride+o]; } }
      for(uint32_t c=0;c<NC;++c){ noc_async_write((uint32_t)(base+c*stride),bg.get_noc_addr(c),base_pg); }
      noc_async_write_barrier();
    }
}
