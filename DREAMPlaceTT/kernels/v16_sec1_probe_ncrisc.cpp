// SPDX-License-Identifier: Apache-2.0
//
// V16 Step 1 — NCRISC no-op for the SEC1 Haloize silicon validation probe.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    // intentional no-op
}
