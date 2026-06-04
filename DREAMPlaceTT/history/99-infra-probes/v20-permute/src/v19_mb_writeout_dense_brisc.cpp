// SPDX-License-Identifier: Apache-2.0
//
// V19 microbench — Variant B writeout: per-bin direct write from L1 to a
// row-major dense DRAM density map. Eliminates host-side de-stride.
//
// Strided ownership: bin_global = local * nc_all + my_core_idx.
// For each local in [0, density_slab_bins): write density_l1[local] (4 B)
// directly to dense_dram[bin_global * 4].
//
// Tracy zone: V19MB-WRITEOUT-DENSE  (single zone covering all writes + barrier)

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
    const uint32_t dense_dram_base   = get_arg_val<uint32_t>(2);
    const uint32_t dense_dram_pgsz   = get_arg_val<uint32_t>(3);
    const uint32_t my_core_idx       = get_arg_val<uint32_t>(4);
    const uint32_t nc_all            = get_arg_val<uint32_t>(5);
    const uint32_t total_bins        = get_arg_val<uint32_t>(6);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* density_l1 = reinterpret_cast<uint32_t*>(base + density_l1_off);

    const InterleavedAddrGen<true> dgen = {
        .bank_base_address = dense_dram_base,
        .page_size         = dense_dram_pgsz,
    };
    const uint32_t bins_per_page = dense_dram_pgsz / 4u;

    // Blackhole quirk: noc_async_write with 4-byte payload returns wrong data
    // unless the L1 source is 16-byte aligned. Stage each value into a slot
    // of a 16-aligned scratch ring in CB_SCRATCH (placed after the density
    // slab — host bumps cb24_size to fit it). Each slot is 16 bytes (one
    // entry of payload + 12 bytes of padding so the next slot stays
    // 16-aligned). SLOT_COUNT = number of outstanding writes; we flush with
    // a barrier each time we cycle the ring so we never overwrite an
    // in-flight slot.
    constexpr uint32_t SLOT_COUNT = 8u;
    constexpr uint32_t SLOT_BYTES = 32u;  // 32-byte spacing per slot
    const uint32_t scratch_off = ((density_l1_off + density_slab_bins * 4u) + 31u) & ~31u;
    uint8_t* scratch_base = base + scratch_off;

    DeviceZoneScopedN("V19MB-WRITEOUT-DENSE");
    uint32_t emitted = 0;
    for (uint32_t local = 0; local < density_slab_bins; ++local) {
        uint32_t bin_global = local * nc_all + my_core_idx;
        if (bin_global >= total_bins) break;

        uint32_t slot = emitted & (SLOT_COUNT - 1u);
        if (slot == 0u && emitted > 0u) {
            // Use the "sent" barrier (not the "acked" one): we only need
            // writes to have departed L1 before we overwrite their source
            // slot. The "acked" variant pays return-trip latency we don't
            // care about here.
            noc_async_writes_flushed();
        }
        uint32_t* slot_ptr = reinterpret_cast<uint32_t*>(scratch_base + slot * SLOT_BYTES);
        slot_ptr[0] = density_l1[local];

        uint32_t page_id     = bin_global / bins_per_page;
        uint32_t page_offset = (bin_global % bins_per_page) * 4u;
        uint64_t dst_noc     = dgen.get_noc_addr(page_id, page_offset);
        noc_async_write(reinterpret_cast<uint32_t>(slot_ptr), dst_noc, 4u);
        ++emitted;
    }
    noc_async_write_barrier();
}
