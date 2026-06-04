// SPDX-License-Identifier: Apache-2.0
// V29 PREP-REUSE bucket-only (BRISC). Replaces v29_prepbucket's soft-float overlap
// recompute: the forward (v4_compute SFPU) already produced px/py, so the per-cell
// prep record is PRE-COMPUTED and stashed. This kernel just BUCKETS it: read record
// {orig_idx, bin_xl_global, byl, kc, hc, ratio, px[8], py[8]} (128 B), route by band
// = bin_xl_global/cpc_x into route[my_core][band] (per-core page, no cross-core atomic).
// Keeps field[1] = bin_xl GLOBAL (the gather subtracts its col0) so a worker can cover
// a contiguous band RANGE (cold-merge). Pure integer + copy — no float overlap math.
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t my_first  = get_arg_val<uint32_t>(1);
    const uint32_t my_n      = get_arg_val<uint32_t>(2);
    const uint32_t in_base   = get_arg_val<uint32_t>(3);
    const uint32_t in_pg     = get_arg_val<uint32_t>(4);   // whole stash page bytes (ncells*128)
    const uint32_t route_base= get_arg_val<uint32_t>(5);
    const uint32_t cap       = get_arg_val<uint32_t>(6);
    const uint32_t nc        = get_arg_val<uint32_t>(7);
    const uint32_t cpc_x     = get_arg_val<uint32_t>(8);
    const uint32_t counts_base = get_arg_val<uint32_t>(9);
    const uint32_t chunk     = get_arg_val<uint32_t>(10);
    const uint32_t cnt_pg    = get_arg_val<uint32_t>(11);

    constexpr auto CB_IN=tt::CBIndex::c_0, CB_CNT=tt::CBIndex::c_1, CB_STAGE=tt::CBIndex::c_2;
    const uint32_t route_pg = cap*128u;
    const InterleavedAddrGen<true> ing = {.bank_base_address=in_base, .page_size=in_pg};
    const InterleavedAddrGen<true> rg  = {.bank_base_address=route_base, .page_size=route_pg};
    const InterleavedAddrGen<true> cntg= {.bank_base_address=counts_base, .page_size=cnt_pg};
    cb_reserve_back(CB_CNT,1);
    uint32_t* cnt = reinterpret_cast<uint32_t*>(get_write_ptr(CB_CNT));
    for (uint32_t b=0;b<nc;++b) cnt[b]=0;
    const uint32_t in_l1 = get_write_ptr(CB_IN);

    { DeviceZoneScopedN("V29-BUCKET-ONLY");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*128u, in_l1, this_n*128u);
          noc_async_read_barrier();
          uint32_t* rec = reinterpret_cast<uint32_t*>(in_l1);
          for (uint32_t i=0;i<this_n;++i) {
              uint32_t* r = rec + i*32u;                  // 128 B = 32 u32
              uint32_t bxl_g = r[1];
              uint32_t band = bxl_g/cpc_x; if (band>=nc) band=nc-1;
              // Keep r[1] = bxl_GLOBAL (do NOT rewrite to band-local): a cold-merge
              // gather worker covers a contiguous band RANGE and computes the field
              // column as bxl_global - col0 (col0 = lo_band*cpc_x). For a single-band
              // worker col0 = band*cpc_x, so this reduces to the old local index.
              uint32_t slot=cnt[band];
              if (slot<cap){ cnt[band]=slot+1;
                  uint32_t page=my_core*nc+band;
                  noc_async_write((uint32_t)r, rg.get_noc_addr(page)+(uint64_t)slot*128u, 128u); }
          }
          noc_async_write_barrier();
          done += this_n;
      }
    }
    noc_async_write((uint32_t)cnt, cntg.get_noc_addr(my_core), nc*4u);
    noc_async_write_barrier();
}
