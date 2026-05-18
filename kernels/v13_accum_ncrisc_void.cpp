// SPDX-License-Identifier: Apache-2.0
//
// Void NCRISC paired with v13_accum_brisc_mt.cpp. BRISC does all the
// per-owned-tile route_buf reading + OX/OY packing + density DRAM write.
// NCRISC is idle (canonical TT-Metal programs pair both RISCs).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    // intentional no-op
}
