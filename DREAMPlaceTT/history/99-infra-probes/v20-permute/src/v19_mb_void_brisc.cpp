// SPDX-License-Identifier: Apache-2.0
// V19 microbench — BRISC void pair for the scatter program.
// (NCRISC does all the work; BRISC just needs to be present.)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {}
