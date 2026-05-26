// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force microbench — NCRISC sibling.
// Handles cells [brisc_end, cells_end) on NoC 1. See v21_ef_common.h.

#define V21_CB_SCRATCH (tt::CBIndex::c_25)
#include "v21_ef_common.h"

void kernel_main() { v21_ef_main(); }
