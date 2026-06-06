// SPDX-License-Identifier: Apache-2.0
//
// V13 smoke — minimal BRISC kernel that does nothing but exit.
// Paired with v13_mcast_smoke_dm.cpp on NCRISC so every core has both RISCs
// running (matches tt-metal canonical pattern; lone-NCRISC programs appear to
// hang the dispatch on BH-38).

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    // intentional no-op
}
