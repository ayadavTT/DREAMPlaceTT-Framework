// SPDX-License-Identifier: Apache-2.0
//
// V16 Phase B.0 — BRISC consumer paired with v16_sfpu_bench_compute.cpp.
//
// The TRISC compute kernel must pack out a tile per batch (otherwise the
// PACK thread deadlocks waiting for pack_tile). BRISC drains the output
// CB so it never fills up. No DRAM write — pure flow-control consumer.
//
// Runtime args:
//   0: n_batches    must match the compute kernel's n_batches

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_OUT = tt::CBIndex::c_16;

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);
    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        cb_wait_front(CB_OUT, 1);
        cb_pop_front(CB_OUT, 1);
    }
}
