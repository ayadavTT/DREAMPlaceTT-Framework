// SPDX-License-Identifier: Apache-2.0
//
// V27 cell-bucketing scatter — groups cells by field row-band for the V26
// backward, WITHOUT touching the V19 forward scatter and WITHOUT cross-core
// atomics. Each source core processes its slice of cells and writes each cell's
// record to its OWN page route[my_core][band] (locally-tracked slot — no atomic
// contention, since only this src writes that page). The backward reads
// route[*][my_band]. Cell-level (1 write/cell), ~1/4 the density scatter.
//
// Input record (24B,6u32):  {byl_global, bxl, w_nw,w_ne,w_sw,w_se}
// Output record (24B,6u32): {bxl, byl_local, w_nw,w_ne,w_sw,w_se}  (backward fmt)
// route page = (src*nc + band), page_size = cap*24. counts[band] → counts buf.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core   = get_arg_val<uint32_t>(0);
    const uint32_t my_first  = get_arg_val<uint32_t>(1);   // first input cell index
    const uint32_t my_n      = get_arg_val<uint32_t>(2);   // cells this core scatters
    const uint32_t in_base   = get_arg_val<uint32_t>(3);   // input cells DRAM (one page)
    const uint32_t in_pg     = get_arg_val<uint32_t>(4);   // input page bytes (all cells)
    const uint32_t route_base= get_arg_val<uint32_t>(5);
    const uint32_t cap       = get_arg_val<uint32_t>(6);   // max records per (src,band) page
    const uint32_t nc        = get_arg_val<uint32_t>(7);
    const uint32_t rpc       = get_arg_val<uint32_t>(8);   // field rows per core (band size)
    const uint32_t counts_base = get_arg_val<uint32_t>(9); // per-(src,band) counts out
    const uint32_t chunk      = get_arg_val<uint32_t>(10); // cells per L1 chunk

    constexpr auto CB_IN=tt::CBIndex::c_0, CB_CNT=tt::CBIndex::c_1;
    const uint32_t route_pg = cap*24u;
    const InterleavedAddrGen<true> ing = {.bank_base_address=in_base, .page_size=in_pg};
    const InterleavedAddrGen<true> rg  = {.bank_base_address=route_base, .page_size=route_pg};
    const InterleavedAddrGen<true> cntg= {.bank_base_address=counts_base, .page_size=nc*4u};

    // local per-band slot counters in L1
    cb_reserve_back(CB_CNT,1);
    uint32_t* cnt = reinterpret_cast<uint32_t*>(get_write_ptr(CB_CNT));
    for (uint32_t b=0;b<nc;++b) cnt[b]=0;

    const uint32_t in_l1 = get_write_ptr(CB_IN);
    const uint32_t chunk_bytes = chunk*24u;

    { DeviceZoneScopedN("V27-BUCKET");
      uint32_t done=0;
      while (done < my_n) {
          uint32_t this_n = (my_n-done < chunk) ? (my_n-done) : chunk;
          // load chunk of this core's input cells. Start offset is 64-aligned
          // (cpc & chunk are multiples of 8 → ×24 is a multiple of 192=3·64).
          // Round the read SIZE up to 64 too (Blackhole DRAM-read align=64;
          // extra bytes are within the padded buffer and never processed).
          const uint32_t rd_bytes = (this_n*24u + 63u) & ~63u;
          noc_async_read(ing.get_noc_addr(0) + (uint64_t)(my_first+done)*24u, in_l1, rd_bytes);
          noc_async_read_barrier();
          uint32_t* rec = reinterpret_cast<uint32_t*>(in_l1);
          for (uint32_t i=0;i<this_n;++i) {
              const uint32_t* r = rec + i*6u;
              uint32_t byl=r[0], bxl=r[1];
              uint32_t band = byl/rpc; if (band>=nc) band=nc-1;
              uint32_t byl_local = byl - band*rpc;
              uint32_t slot = cnt[band];
              if (slot < cap) {
                  cnt[band] = slot+1;
                  // build output record in a tiny L1 staging (reuse end of chunk buf)
                  uint32_t* o = reinterpret_cast<uint32_t*>(in_l1 + chunk_bytes); // staging
                  o[0]=bxl; o[1]=byl_local; o[2]=r[2]; o[3]=r[3]; o[4]=r[4]; o[5]=r[5];
                  uint32_t page = my_core*nc + band;
                  noc_async_write((uint32_t)o, rg.get_noc_addr(page) + (uint64_t)slot*24u, 24u);
              }
          }
          noc_async_write_barrier();
          done += this_n;
      }
    }
    // write this core's counts (counts[band] for src=my_core) to counts_base page my_core
    noc_async_write((uint32_t)cnt, cntg.get_noc_addr(my_core), nc*4u);
    noc_async_write_barrier();
}
