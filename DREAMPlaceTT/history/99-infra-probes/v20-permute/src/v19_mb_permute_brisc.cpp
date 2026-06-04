// SPDX-License-Identifier: Apache-2.0
//
// V20 permute microbench — BRISC sibling.
// Handles cells [0, cpc/2) on NoC 0. See v19_mb_permute_common.h.

#define V20_CB_SCRATCH (tt::CBIndex::c_25)
#include "v19_mb_permute_common.h"

void kernel_main() { v20_permute_main(); }
