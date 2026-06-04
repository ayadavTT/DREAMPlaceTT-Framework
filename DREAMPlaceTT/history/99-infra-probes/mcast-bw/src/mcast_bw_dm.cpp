// SPDX-License-Identifier: Apache-2.0
//
// FCCS field-cast bandwidth microbench — canonical multicast handshake.
// Follows tt_metal/programming_examples/contributed/multicast (coordinator/inbound):
//   * multicast runs on NOC 0 (RISCV_0); rect coords are worker_core_from_logical_core
//     (NOC-0 coords) — running mcast on NOC 1 with these coords targets wrong cores.
//   * RECEIVERS signal readiness FIRST (inc sender's ready-sem), the producer waits for
//     all of them, THEN multicasts, then mcasts a VALID sem to release them.
// Producer (core 0) streams a DRAM field in tiles, multicasting each tile to all cores.
//
// Args: 0 my_core_id 1 nc_all 2 is_producer 3 tile_bytes 4 n_tiles 5 repeats
//       6 field_dram_addr 7 dst_l1_off 8 srcbuf_l1_off 9 core0_noc_x 10 core0_noc_y
//       11 ready_sid 12 valid_sid 13 n_rects  14.. per-rect quint (xs,ys,xe,ye,num_dests)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t SEM_VALID = 1u;

void kernel_main() {
    const uint32_t my_core_id      = get_arg_val<uint32_t>(0);
    const uint32_t nc_all          = get_arg_val<uint32_t>(1);
    const uint32_t is_producer     = get_arg_val<uint32_t>(2);
    const uint32_t core0_noc_x     = get_arg_val<uint32_t>(9);
    const uint32_t core0_noc_y     = get_arg_val<uint32_t>(10);
    const uint32_t ready_sid       = get_arg_val<uint32_t>(11);
    const uint32_t valid_sid       = get_arg_val<uint32_t>(12);
    const uint32_t n_rects         = get_arg_val<uint32_t>(13);

    volatile tt_l1_ptr uint32_t* valid_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(valid_sid));

    if (!is_producer) {
        // Receiver: signal ready to producer, then wait for the VALID release.
        noc_semaphore_set(valid_sem, 0u);
        uint64_t ready_addr = get_noc_addr(core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(ready_sid));
        noc_semaphore_inc(ready_addr, 1u);
        noc_semaphore_wait(valid_sem, SEM_VALID);
        return;
    }

    // ── Producer (core 0) ──
    const uint32_t tile_bytes      = get_arg_val<uint32_t>(3);
    const uint32_t n_tiles         = get_arg_val<uint32_t>(4);
    const uint32_t repeats         = get_arg_val<uint32_t>(5);
    const uint32_t field_dram_addr = get_arg_val<uint32_t>(6);
    const uint32_t dst_l1_off      = get_arg_val<uint32_t>(7);
    const uint32_t srcbuf_l1_off   = get_arg_val<uint32_t>(8);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    const uint32_t dst_l1_abs = (uint32_t)(base + dst_l1_off);
    const uint32_t srcbuf     = (uint32_t)(base + srcbuf_l1_off);

    // Wait for all receivers (nc_all - 1, self excluded) to signal ready.
    volatile tt_l1_ptr uint32_t* ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sid));
    noc_semaphore_wait_min(ready_sem, nc_all - 1u);
    noc_semaphore_set(ready_sem, 0u);

    const InterleavedAddrGen<true> fg = {.bank_base_address=field_dram_addr, .page_size=tile_bytes};
    { DeviceZoneScopedN("FCCS-FIELDCAST");
      for (uint32_t rep=0; rep<repeats; ++rep) {
          for (uint32_t t=0; t<n_tiles; ++t) {
              noc_async_read(fg.get_noc_addr(t), srcbuf, tile_bytes);
              noc_async_read_barrier();
              for (uint32_t r=0;r<n_rects;++r){ const uint32_t b=14u+r*5u;
                  uint32_t xs=get_arg_val<uint32_t>(b),ys=get_arg_val<uint32_t>(b+1),
                           xe=get_arg_val<uint32_t>(b+2),ye=get_arg_val<uint32_t>(b+3),nd=get_arg_val<uint32_t>(b+4);
                  if (!nd) { continue; }
                  uint64_t md=get_noc_multicast_addr(xs,ys,xe,ye,dst_l1_abs);
                  noc_async_write_multicast(srcbuf, md, tile_bytes, nd);
              }
              noc_async_write_barrier();
          }
      }
    }

    // Release receivers: local VALID + multicast VALID to all rects.
    noc_semaphore_set(valid_sem, SEM_VALID);
    uint32_t valid_l1 = (uint32_t)get_semaphore(valid_sid);
    for (uint32_t r=0;r<n_rects;++r){ const uint32_t b=14u+r*5u;
        uint32_t xs=get_arg_val<uint32_t>(b),ys=get_arg_val<uint32_t>(b+1),
                 xe=get_arg_val<uint32_t>(b+2),ye=get_arg_val<uint32_t>(b+3),nd=get_arg_val<uint32_t>(b+4);
        if (!nd) { continue; }
        uint64_t ma=get_noc_multicast_addr(xs,ys,xe,ye,valid_l1);
        noc_semaphore_set_multicast(valid_l1, ma, nd);
    }
    noc_async_write_barrier();
}
