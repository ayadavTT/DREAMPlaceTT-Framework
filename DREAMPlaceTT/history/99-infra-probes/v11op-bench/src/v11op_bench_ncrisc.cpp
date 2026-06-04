// SPDX-License-Identifier: Apache-2.0
//
// V11 Stage B' — NCRISC no-op for the SFPU outer-product microbench.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    // intentional no-op
}
