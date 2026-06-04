// SPDX-License-Identifier: Apache-2.0
//
// V20 permute microbench — NCRISC sibling.
// Handles cells [cpc/2, cpc) on NoC 1. See v19_mb_permute_common.h.

#define V20_CB_SCRATCH (tt::CBIndex::c_24)
#include "v19_mb_permute_common.h"

void kernel_main() { v20_permute_main(); }
