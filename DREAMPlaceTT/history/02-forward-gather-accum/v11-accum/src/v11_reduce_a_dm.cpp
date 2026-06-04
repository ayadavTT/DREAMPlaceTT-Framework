// SPDX-License-Identifier: Apache-2.0
//
// V11 Phase 3 Reduce-A Kernel (RISCV_0, NOC_0) — shard write phase.
//
// Runs after v11_accum_dm on every Tensix core, before a host Finish()
// barrier that ensures all shard writes reach DRAM before reduce_bc reads.
//
// Purpose: cores that are SHARD OWNERS for a hot tile write their partial
// dense data into shard_reduce_buf[hot_tile_seq * MAX_K + shard_idx].
// Primary owners and cold-tile cores do nothing.
//
// Dense is still live in CB_SCRATCH at offset `dense_offset_bytes`
// (same CB index c_24 as accum, same L1 layout). Data persists in L1
// between sequential program launches.
//
// shard_reduce_buf page layout: one 4096-byte page per (hot_tile_seq, shard_idx),
//   page index = hot_tile_seq * max_K + shard_idx.
//   Shard index 0 is reserved for the primary owner (written by reduce_bc).
//   Alt owners use shard_idx = 1..K-1.
//
// Runtime args:
//   0: dense_offset_bytes      offset of dense[0] from CB_SCRATCH base
//   1: shard_reduce_dram_addr  DRAM base of shard_reduce_buf
//   2: shard_reduce_pgsz       bytes per page (= TILE_BYTES = 4096)
//   3: max_K                   shard count cap (= 8)
//   4: n_primary               number of primary tile slots (dense[0..n_primary-1])
//   5: n_shard                 number of shard slots at this core
//   // For each shard i = 0..n_shard-1 (2 args each):
//   6 + 2*i:   hot_tile_seq    compact index into shard_reduce_buf
//   7 + 2*i:   shard_idx       1..K-1 (which shard slot for this tile)
//   // Shard i occupies dense[n_primary + i] (local_idx = n_primary + i)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH  = tt::CBIndex::c_24;
constexpr uint32_t TILE_FLOATS = 32u * 32u;
constexpr uint32_t TILE_BYTES  = TILE_FLOATS * sizeof(float);

void kernel_main() {
    const uint32_t dense_offset   = get_arg_val<uint32_t>(0);
    const uint32_t srb_dram       = get_arg_val<uint32_t>(1);
    const uint32_t srb_pgsz       = get_arg_val<uint32_t>(2);
    const uint32_t max_K          = get_arg_val<uint32_t>(3);
    const uint32_t n_primary      = get_arg_val<uint32_t>(4);
    const uint32_t n_shard        = get_arg_val<uint32_t>(5);

    if (n_shard == 0u) return;  // cold core — nothing to do

    uint8_t* base  = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    float*   dense = reinterpret_cast<float*>(base + dense_offset);

    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = srb_dram,
        .page_size         = srb_pgsz,
    };

    for (uint32_t i = 0; i < n_shard; ++i) {
        uint32_t hot_seq   = get_arg_val<uint32_t>(6u + 2u * i);
        uint32_t shard_idx = get_arg_val<uint32_t>(7u + 2u * i);
        uint32_t local_idx = n_primary + i;  // shard slots follow primary slots in dense
        uint32_t page      = hot_seq * max_K + shard_idx;

        uint32_t src_l1 = reinterpret_cast<uint32_t>(dense + local_idx * TILE_FLOATS);
        noc_async_write(src_l1, rgen.get_noc_addr(page), TILE_BYTES);
    }
    noc_async_write_barrier();
}
