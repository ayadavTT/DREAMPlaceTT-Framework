// SPDX-License-Identifier: Apache-2.0
//
// V21a electric_force microbench — BRISC sibling.
// Handles cells [cells_start, brisc_end) on NoC 0. See v21_ef_common.h.

#define V21_CB_SCRATCH (tt::CBIndex::c_24)
#include "v21_ef_common.h"

void kernel_main() { v21_ef_main(); }
