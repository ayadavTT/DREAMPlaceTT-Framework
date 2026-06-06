// SPDX-License-Identifier: Apache-2.0
//
// V11 Stage B' — BRISC producer/consumer paired with v11op_bench_compute.cpp.
//
// Per batch:
//   - Stages two synthetic input tiles into c_0 and c_1 (the "ox" and "oy"
//     surrogates the compute kernel reads).
//   - Drains the 64 outer-product output tiles from c_16.
//
// L1 staging holds two pre-filled bf16 (= fp32 via reinterp) tiles that get
// reused each batch. No DRAM I/O — this is a pure on-chip throughput test.
//
// Runtime args:
//   0: n_batches    must match the compute kernel's n_batches

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_A   = tt::CBIndex::c_0;
constexpr uint32_t CB_B   = tt::CBIndex::c_1;
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;

constexpr uint32_t TILE_FP32_BYTES = 32u * 32u * 4u;  // 4 KB per fp32 tile

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);

    // Initialize the surrogate tiles ONCE with a synthetic pattern. We avoid
    // re-staging per batch to isolate the SFPU compute cost from BRISC's
    // tile-prep cost.
    {
        cb_reserve_back(CB_A, 1);
        cb_reserve_back(CB_B, 1);
        volatile float* a_ptr = reinterpret_cast<volatile float*>(get_write_ptr(CB_A));
        volatile float* b_ptr = reinterpret_cast<volatile float*>(get_write_ptr(CB_B));
        for (uint32_t i = 0; i < 1024; ++i) {
            // Realistic ox/oy magnitudes: ~[0.01, 0.5]
            a_ptr[i] = 0.1f + ((float)(i & 31) * 0.01f);
            b_ptr[i] = 0.1f + ((float)((i >> 5) & 31) * 0.01f);
        }
        cb_push_back(CB_A, 1);
        cb_push_back(CB_B, 1);
    }

    // For each batch: re-stage the same inputs (compute kernel will pop them)
    // and drain the 64 output tiles.
    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        // After the first batch, re-stage by re-pushing without re-filling
        // (data was preserved in the CB slot which the compute popped).
        // Need to re-fill though because cb_pop_front frees the slot.
        if (batch > 0) {
            cb_reserve_back(CB_A, 1);
            cb_reserve_back(CB_B, 1);
            // Trust the data is still in the slot (CB is single-tile so the
            // pointer just rotates back). Fill again to be safe.
            volatile float* a_ptr = reinterpret_cast<volatile float*>(get_write_ptr(CB_A));
            volatile float* b_ptr = reinterpret_cast<volatile float*>(get_write_ptr(CB_B));
            // Light refresh — write just one value per tile to keep BRISC busy
            // but not dominate batch time.
            a_ptr[0] = 0.1f;
            b_ptr[0] = 0.1f;
            cb_push_back(CB_A, 1);
            cb_push_back(CB_B, 1);
        }

        // Drain 64 output tiles per batch (the compute kernel pushes 64).
        for (uint32_t i = 0; i < 64; ++i) {
            cb_wait_front(CB_OUT, 1);
            cb_pop_front(CB_OUT, 1);
        }
    }
}
