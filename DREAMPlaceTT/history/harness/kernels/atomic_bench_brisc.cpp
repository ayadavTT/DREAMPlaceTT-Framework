// SPDX-License-Identifier: Apache-2.0
//
// Atomic increment microbenchmark — measures noc_semaphore_inc latency.
//
// Each call: noc_semaphore_inc(target_noc_addr, incr_value)
//   - target_noc_addr is the NOC address (x, y, L1_offset) of a uint32 counter
//     that we placed inside CB_SCRATCH at COUNTER_OFF.
//   - incr_value is any uint32 (NOT just +1). Different cores can pass
//     different values to test arbitrary-value summing.
//
// Runtime args:
//   0: n_iters         number of atomic_inc calls in the timed loop
//   1: counter_off     L1 byte offset inside CB_SCRATCH where the counter lives
//                      (if dram_mode != 0, this is the DRAM bank-page byte
//                       offset and target_x/y are ignored)
//   2: target_x        target NOC x  (L1 mode only)
//   3: target_y        target NOC y  (L1 mode only)
//   4: cyc_lo_off      L1 byte offset to write start mcycle (this core)
//   5: cyc_hi_off      L1 byte offset to write end mcycle (this core)
//   6: posted_flag     0 = non-posted, 1 = posted
//   7: incr_value      uint32 added per atomic call (each core can be unique)
//   8: dram_mode       0 = target is L1, 1 = target is DRAM bank 0
//   9: dram_base       DRAM buffer base address (only used if dram_mode=1)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

static inline uint32_t rdcycle32() {
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

void kernel_main() {
    const uint32_t n_iters     = get_arg_val<uint32_t>(0);
    const uint32_t counter_off = get_arg_val<uint32_t>(1);
    const uint32_t target_x    = get_arg_val<uint32_t>(2);
    const uint32_t target_y    = get_arg_val<uint32_t>(3);
    const uint32_t cyc_lo_off  = get_arg_val<uint32_t>(4);
    const uint32_t cyc_hi_off  = get_arg_val<uint32_t>(5);
    const uint32_t posted_flag = get_arg_val<uint32_t>(6);
    const uint32_t incr_value  = get_arg_val<uint32_t>(7);
    const uint32_t dram_mode   = get_arg_val<uint32_t>(8);
    const uint32_t dram_base   = get_arg_val<uint32_t>(9);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    volatile uint32_t* cyc_lo = reinterpret_cast<volatile uint32_t*>(base + cyc_lo_off);
    volatile uint32_t* cyc_hi = reinterpret_cast<volatile uint32_t*>(base + cyc_hi_off);

    // Resolve target NOC address (L1 of target core, or DRAM bank 0 page 0).
    uint64_t target_noc_addr;
    if (dram_mode) {
        // Single 32B DRAM page, bank 0. counter_off must be 0 here.
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = dram_base,
            .page_size = 32u,
        };
        target_noc_addr = dgen.get_noc_addr(0) + counter_off;
    } else {
        uint32_t my_l1_base = (uint32_t)base;
        target_noc_addr = get_noc_addr(target_x, target_y, my_l1_base + counter_off);
    }

    // (Counter is pre-zeroed by host via WriteToDeviceL1 — see host code.)
    *cyc_lo = rdcycle32();
    {
        DeviceZoneScopedN("ATOMIC-INC");
        if (posted_flag) {
            for (uint32_t i = 0; i < n_iters; ++i) {
                noc_semaphore_inc<true>(target_noc_addr, incr_value);
            }
        } else {
            for (uint32_t i = 0; i < n_iters; ++i) {
                noc_semaphore_inc<false>(target_noc_addr, incr_value);
            }
            noc_async_atomic_barrier();
        }
    }
    *cyc_hi = rdcycle32();
}
