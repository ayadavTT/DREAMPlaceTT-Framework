// SPDX-License-Identifier: Apache-2.0
//
// V19 writeout — runs after scatter Finish(), guaranteeing all atomics
// settled.  Reads this core's L1 density slab and writes to DRAM staging.
//
// Runtime args:
//   0: density_l1_off    L1 byte offset of density slab in CB_SCRATCH
//   1: density_slab_bins number of uint32 slots
//   2: density_dram_base DRAM staging base addr
//   3: density_dram_pgsz per-core page size in DRAM
//   4: my_core_idx       this core's index for DRAM page assignment

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

void kernel_main() {
    const uint32_t density_l1_off    = get_arg_val<uint32_t>(0);
    const uint32_t density_slab_bins = get_arg_val<uint32_t>(1);
    const uint32_t density_dram_base = get_arg_val<uint32_t>(2);
    const uint32_t density_dram_pgsz = get_arg_val<uint32_t>(3);
    const uint32_t my_core_idx       = get_arg_val<uint32_t>(4);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);

    DeviceZoneScopedN("V19-WRITEOUT");
    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = density_dram_base,
        .page_size         = density_dram_pgsz,
    };
    // Chunked writeout: noc_async_write > ~96 KB silently corrupts on
    // Blackhole (same gotcha as V15 spill). Clamp each transfer to 64 KB.
    constexpr uint32_t MAX_CHUNK_BYTES = 64u * 1024u;
    const uint32_t total_bytes = density_slab_bins * 4u;
    uint64_t page_base = dgen.get_noc_addr(my_core_idx);
    uint32_t l1_addr   = reinterpret_cast<uint32_t>(density_l1);
    for (uint32_t off = 0; off < total_bytes; off += MAX_CHUNK_BYTES) {
        uint32_t chunk = total_bytes - off;
        if (chunk > MAX_CHUNK_BYTES) chunk = MAX_CHUNK_BYTES;
        noc_async_write(l1_addr + off, page_base + (uint64_t)off, chunk);
    }
    noc_async_write_barrier();
}
