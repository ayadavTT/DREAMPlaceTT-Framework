// SPDX-License-Identifier: Apache-2.0
//
// V13 Phase 0b — void NCRISC kernel paired with v13_pack_bench_brisc.cpp.
// Pair both RISCs to match the canonical TT-Metal program structure that
// hello_world/matmul examples use; lone-RISC programs have empirically
// hung the dispatcher on BH-38.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    // intentional no-op
}
