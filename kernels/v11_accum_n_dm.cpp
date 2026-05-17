// SPDX-License-Identifier: Apache-2.0
//
// V11 NCRISC Accumulator (RISCV_1, NOC_1) — accumulation-only twin of
// v11_accum_dm.cpp. Reads route_buf pages from writers [nc_split..nc_all),
// accumulates into a SEPARATE dense_n buffer in L1, then signals the BRISC
// twin via a per-core L1 semaphore. BRISC then merges dense_n into dense_b
// and runs Phase A/B/C alone.
//
// Splitting writers across the two RISC cores roughly halves the V11A-ACC
// time on the slowest receiver, since the per-tuple inner loop is L1-latency
// bound and benefits from a second independent pipeline.
//
// L1 layout (must match v11_accum_dm.cpp's offsets):
//   [owned_lookup]                                  (M_tiles*N_tiles uint16)
//   [inbound_hdrs_b][inbound_hdrs_n]                (BRISC + NCRISC headers)
//   [inbound_buf_b][inbound_buf_n]                  (BRISC + NCRISC tuples)
//   [128B safety gap]
//   [dense_b][dense_n]                              (BRISC + NCRISC accumulators)
//   [tmp_shard]                                     (BRISC's Phase B scratch)
//
// Runtime args (subset of BRISC's; same indices 0..14, 16..18, plus extras):
//   0:  owned_lookup_dram_addr
//   1:  owned_lookup_pgsz
//   2:  my_core_id
//   3:  nc_all
//   4:  M_tiles
//   5:  N_tiles
//   6:  nbx
//   7:  nby
//   8:  route_dram_addr
//   9:  route_pgsz
//  10:  max_per_writer
//   ...11..13 unused on NCRISC (density/scale-related)
//  14:  n_primary
//  15:  n_shard
//   ...16..19 unused on NCRISC (srb / max_K / hot count)
//  20:  nc_split             writer split point: NCRISC reads [nc_split..nc_all)
//  21:  merge_sem_id         semaphore on BRISC side; NCRISC inc's it once at end
//  22:  prim_noc_x           THIS core's NOC X (so NCRISC can NOC-write the sem inc home)
//  23:  prim_noc_y           THIS core's NOC Y

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t SRC_CHUNK  = 8u;  // see v11_accum_dm.cpp; halved to free L1 for max_per_writer=4096
constexpr uint32_t TILE_W     = 32u;
constexpr uint32_t TILE_H     = 32u;
constexpr uint32_t TILE_FLOATS= TILE_W * TILE_H;
constexpr uint32_t TILE_BYTES = TILE_FLOATS * sizeof(float);
constexpr uint32_t V11_PAGE_HDR_BYTES = 64u;

struct V11Contrib { uint16_t bx; uint16_t by; float area; };
static_assert(sizeof(V11Contrib) == 8, "V11Contrib must be 8 bytes");

void kernel_main() {
    const uint32_t owned_lookup_dram = get_arg_val<uint32_t>(0);
    const uint32_t owned_lookup_pgsz = get_arg_val<uint32_t>(1);
    const uint32_t my_core_id        = get_arg_val<uint32_t>(2);
    const uint32_t nc_all            = get_arg_val<uint32_t>(3);
    const uint32_t M_tiles           = get_arg_val<uint32_t>(4);
    const uint32_t N_tiles           = get_arg_val<uint32_t>(5);
    const uint32_t nbx               = get_arg_val<uint32_t>(6);
    const uint32_t nby               = get_arg_val<uint32_t>(7);
    (void)nbx; (void)nby;
    const uint32_t route_dram        = get_arg_val<uint32_t>(8);
    const uint32_t route_pgsz        = get_arg_val<uint32_t>(9);
    const uint32_t max_per_writer    = get_arg_val<uint32_t>(10);
    const uint32_t n_primary         = get_arg_val<uint32_t>(14);
    const uint32_t n_shard           = get_arg_val<uint32_t>(15);
    const uint32_t nc_split          = get_arg_val<uint32_t>(20);
    const uint32_t merge_sem_id      = get_arg_val<uint32_t>(21);
    const uint32_t my_noc_x          = get_arg_val<uint32_t>(22);
    const uint32_t my_noc_y          = get_arg_val<uint32_t>(23);
    // G-PMERGE: wait for BRISC's ACC done before merging dense_n into the
    // lower half of dense_b; signal when our half is complete.
    const uint32_t brisc_acc_done_sem_id        = get_arg_val<uint32_t>(24);
    const uint32_t ncrisc_half_merge_done_sem_id = get_arg_val<uint32_t>(25);

    const uint32_t n_total = n_primary + n_shard;

    // ── Mirror BRISC's L1 layout exactly ──────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0;

    uint16_t* owned_lookup = reinterpret_cast<uint16_t*>(base + off);
    const uint32_t total_tiles = M_tiles * N_tiles;
    off += total_tiles * sizeof(uint16_t);
    off = (off + 7u) & ~7u;
    off = (off + 63u) & ~63u;

    // BRISC's inbound_hdrs first
    off += nc_all * V11_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;
    // Then NCRISC's inbound_hdrs (we own this region)
    uint32_t* inbound_hdrs_n = reinterpret_cast<uint32_t*>(base + off);
    off += nc_all * V11_PAGE_HDR_BYTES;
    off = (off + 63u) & ~63u;

    // BRISC's inbound_buf first
    off += SRC_CHUNK * max_per_writer * sizeof(V11Contrib);
    off = (off + 63u) & ~63u;
    // Then NCRISC's inbound_buf
    V11Contrib* inbound_buf_n = reinterpret_cast<V11Contrib*>(base + off);
    off += SRC_CHUNK * max_per_writer * sizeof(V11Contrib);
    off = (off + 63u) & ~63u;
    off += 128u;  // safety gap

    // dense_b is at `off`; dense_n is right after it.
    float* dense_b = reinterpret_cast<float*>(base + off);
    float* dense_n = dense_b + n_total * TILE_FLOATS;

    // ── Step 0: load owned_lookup (also done by BRISC; cheap) ─────────────
    {
        DeviceZoneScopedN("V11N-LOOKUP-LOAD");
        const InterleavedAddrGen<true> lgen = {
            .bank_base_address = owned_lookup_dram,
            .page_size         = owned_lookup_pgsz,
        };
        noc_async_read(lgen.get_noc_addr(my_core_id),
                       reinterpret_cast<uint32_t>(owned_lookup),
                       total_tiles * sizeof(uint16_t));
        noc_async_read_barrier();
    }

    // ── Step 1: zero dense_n ──────────────────────────────────────────────
    {
        DeviceZoneScopedN("V11N-ZERO");
        const uint32_t total_floats = n_total * TILE_FLOATS;
        for (uint32_t i = 0; i < total_floats; ++i) dense_n[i] = 0.0f;
    }

    // Total writer count = 2 * nc_split (with Step-5 parallel scatter, route_buf
    // has 2*nc_all writer rows; nc_split=nc_all so this RISC reads [nc_split, 2*nc_split)).
    const uint32_t nc_writers = 2u * nc_split;

    if (nc_split >= nc_writers) {
        // Nothing to do; just signal BRISC.
        uint32_t sem_off = (uint32_t)get_semaphore(merge_sem_id);
        noc_semaphore_inc(get_noc_addr(my_noc_x, my_noc_y, sem_off), 1u);
        return;
    }

    const InterleavedAddrGen<true> rgen = {
        .bank_base_address = route_dram,
        .page_size         = route_pgsz,
    };

    // ── Step 2a: bulk-read headers for [nc_split..nc_writers) ─────────────
    {
        DeviceZoneScopedN("V11N-HDR-READ");
        for (uint32_t s = nc_split; s < nc_writers; ++s) {
            uint64_t page_base = rgen.get_noc_addr(s * nc_all + my_core_id);
            uint32_t dst_l1 = reinterpret_cast<uint32_t>(inbound_hdrs_n)
                            + (s - nc_split) * V11_PAGE_HDR_BYTES;
            noc_async_read(page_base, dst_l1, V11_PAGE_HDR_BYTES);
        }
        noc_async_read_barrier();
    }

    // ── Step 2b: per-chunk tuple read + accumulate ────────────────────────
    // Each writer page has two halves: NCRISC tuples in [0..max/2),
    // BRISC tuples in [max/2..max). cnt_n at hdr[0], cnt_b at hdr[8].
    const uint32_t HALF_CAP = max_per_writer / 2u;
    for (uint32_t cs = nc_split; cs < nc_writers; cs += SRC_CHUNK) {
        uint32_t cs_end = cs + SRC_CHUNK;
        if (cs_end > nc_writers) cs_end = nc_writers;
        const uint32_t batch = cs_end - cs;

        {
            DeviceZoneScopedN("V11N-DATA-READ");
            for (uint32_t s = cs; s < cs_end; ++s) {
                uint32_t cnt_n = inbound_hdrs_n[(s - nc_split) * (V11_PAGE_HDR_BYTES / 4) + 0];
                uint32_t cnt_b = inbound_hdrs_n[(s - nc_split) * (V11_PAGE_HDR_BYTES / 4) + 8];
                if (cnt_n == 0u && cnt_b == 0u) continue;
                uint64_t page_base = rgen.get_noc_addr(s * nc_all + my_core_id);
                uint64_t src_addr  = page_base + (uint64_t)V11_PAGE_HDR_BYTES;
                uint32_t dst_l1    = reinterpret_cast<uint32_t>(
                                        &inbound_buf_n[(s - cs) * max_per_writer]);
                noc_async_read(src_addr, dst_l1, max_per_writer * sizeof(V11Contrib));
            }
            noc_async_read_barrier();
        }

        {
            DeviceZoneScopedN("V11N-ACC");
            for (uint32_t bs = 0; bs < batch; ++bs) {
                uint32_t s = cs + bs;
                uint32_t cnt_n = inbound_hdrs_n[(s - nc_split) * (V11_PAGE_HDR_BYTES / 4) + 0];
                uint32_t cnt_b = inbound_hdrs_n[(s - nc_split) * (V11_PAGE_HDR_BYTES / 4) + 8];
                if (cnt_n > HALF_CAP) cnt_n = HALF_CAP;
                if (cnt_b > HALF_CAP) cnt_b = HALF_CAP;
                if (cnt_n == 0u && cnt_b == 0u) continue;
                const V11Contrib* base = &inbound_buf_n[bs * max_per_writer];
                const V11Contrib* halves[2] = { base, base + HALF_CAP };
                uint32_t halves_cnt[2] = { cnt_n, cnt_b };
                for (uint32_t half = 0; half < 2u; ++half) {
                    uint32_t cnt = halves_cnt[half];
                    if (cnt == 0u) continue;
                    const V11Contrib* src = halves[half];
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint32_t bx = src[i].bx;
                        uint32_t by = src[i].by;
                        float    a  = src[i].area;
                        if (a == 0.0f) continue;
                        uint32_t map_idx = (bx >> 5) * N_tiles + (by >> 5);
                        uint32_t local = (uint32_t)owned_lookup[map_idx];
                        if (local >= n_total) continue;
                        uint32_t bxw = bx & 31u;
                        uint32_t byw = by & 31u;
                        dense_n[local * TILE_FLOATS + bxw * TILE_W + byw] += a;
                    }
                }
            }
        }
    }

    // ── Signal BRISC that V11N-ACC is done (dense_n is final) ────────────
    {
        DeviceZoneScopedN("V11N-SIGNAL");
        uint32_t sem_off = (uint32_t)get_semaphore(merge_sem_id);
        noc_semaphore_inc(get_noc_addr(my_noc_x, my_noc_y, sem_off), 1u);
    }

    // ── G-PMERGE: wait for BRISC's V11A-ACC done, then merge LOWER HALF of
    // dense_n into dense_b. BRISC will merge the upper half in parallel.
    {
        DeviceZoneScopedN("V11N-MERGE-HALF");
        volatile tt_l1_ptr uint32_t* bptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_acc_done_sem_id));
        noc_semaphore_wait(bptr, 1u);
        noc_semaphore_set(bptr, 0u);
        const uint32_t total_floats = n_total * TILE_FLOATS;
        const uint32_t H = total_floats >> 1;
        for (uint32_t i = 0; i < H; ++i) dense_b[i] += dense_n[i];
    }

    // ── Signal BRISC that NCRISC's half of the merge is complete. ─────────
    {
        DeviceZoneScopedN("V11N-MERGE-SIGNAL");
        uint32_t sem_off = (uint32_t)get_semaphore(ncrisc_half_merge_done_sem_id);
        noc_semaphore_inc(get_noc_addr(my_noc_x, my_noc_y, sem_off), 1u);
    }
}
