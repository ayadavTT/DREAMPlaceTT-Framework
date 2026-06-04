// SPDX-License-Identifier: Apache-2.0
//
// V19 microbench — density-slab zero-init kernel. Runs ONCE before the
// scatter program. By living in its own program (separated by Finish(cq)
// from scatter), we guarantee all 110 cores have a zeroed slab BEFORE any
// core starts emitting atomics — eliminating the race where one core's
// in-kernel INIT loop overwrites incoming atomics that arrived early from
// a faster-starting core.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

void kernel_main() {
    const uint32_t density_l1_off    = get_arg_val<uint32_t>(0);
    const uint32_t density_slab_bins = get_arg_val<uint32_t>(1);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);

    DeviceZoneScopedN("V19MB-ZERO");
    for (uint32_t i = 0; i < density_slab_bins; ++i) density_l1[i] = 0u;
}
