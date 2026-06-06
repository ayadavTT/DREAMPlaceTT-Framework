// SPDX-License-Identifier: Apache-2.0
// SLOT-CAPTURE probe for the bin-owner forward atomic-slot stash.
// Each core does N fetch-adds to a counter (own L1 = no contention, or a shared
// core's L1 = contention) and CAPTURES the returned prior value (= the unique
// slot index the record would be placed at). Uses the BLESSED static-VC path
// noc_fast_atomic_increment<noc_mode, program_ret_addr=true> so each in-flight
// atomic returns to a DISTINCT L1 addr → stays pipelined (MAXO outstanding),
// then one atomic barrier drains the batch and we read the MAXO returned slots.
// The raw noc_atomic_read_and_increment (dynamic VC) deadlocks under contention;
// this path does not. Captured slots are written to DRAM so the host can verify
// they form a dense permutation [0, N) (own mode → no drops, no dups).
#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t N        = get_arg_val<uint32_t>(0);
    const uint32_t tgt_x    = get_arg_val<uint32_t>(1);
    const uint32_t tgt_y    = get_arg_val<uint32_t>(2);
    const uint32_t coff     = get_arg_val<uint32_t>(5);   // counter L1 offset
    const uint32_t out_base = get_arg_val<uint32_t>(6);   // DRAM: per-core captured slots
    const uint32_t out_pg   = get_arg_val<uint32_t>(7);   // = N*4
    const uint32_t my_core  = get_arg_val<uint32_t>(8);
    const uint32_t do_zero  = get_arg_val<uint32_t>(9);   // clean verify: zero-pass (N=0) then probe-pass (do_zero=0)

    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(coff);
    if (do_zero) c[0] = 0u;                               // owner zeros; skipped on probe-pass to avoid mid-stream re-zero race
    const uint32_t rbase = coff + 64u;                    // MAXO return slots (16B apart)
    const uint32_t sbase = coff + 0x8000u;                // captured slots array (N*4)
    uint32_t* slots = reinterpret_cast<uint32_t*>(sbase);
    const uint64_t tgt = get_noc_addr(tgt_x, tgt_y, coff);

    { DeviceZoneScopedN("V31-SLOT-CAPTURE");
      const uint32_t MAXO = get_arg_val<uint32_t>(10);    // outstanding depth (1 = fully serialized)
      // Gate reads on noc_atomic_read_updates_completed() — the header guarantees this
      // counts value-returning atomics whose responses are COMMITTED TO LOCAL MEMORY.
      // (noc_async_atomic_barrier waits on the atomic ack, which under contention can
      // precede the returned-value L1 write → stale captures.)
      const uint32_t base = noc_atomic_read_updates_completed();
      uint32_t issued = 0, i = 0;
      while (i < N) {
          uint32_t batch = (N - i < MAXO) ? (N - i) : MAXO;
          for (uint32_t k = 0; k < batch; ++k) {
              noc_fast_atomic_increment<noc_mode, /*program_ret_addr=*/true>(
                  noc_index, write_at_cmd_buf, tgt, NOC_UNICAST_WRITE_VC,
                  1u /*incr*/, 31u /*wrap*/, false /*linked*/, false /*posted*/,
                  rbase + k * 16u /*atomic_ret_val: distinct L1 dst*/);
          }
          issued += batch;
          while ((noc_atomic_read_updates_completed() - base) < issued) { /* values committed */ }
          invalidate_l1_cache();                          // NoC wrote returns directly to L1; drop stale cached lines
          for (uint32_t k = 0; k < batch; ++k) {
              slots[i + k] = *reinterpret_cast<volatile uint32_t*>(rbase + k * 16u);
          }
          i += batch;
      }
    }
    if (N) {
        const InterleavedAddrGen<true> og = {.bank_base_address=out_base, .page_size=out_pg};
        noc_async_write(sbase, og.get_noc_addr(my_core), N * 4u);
        noc_async_write_barrier();
    }
}
