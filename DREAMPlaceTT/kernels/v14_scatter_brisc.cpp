// SPDX-License-Identifier: Apache-2.0
//
// V14 Architecture-A — BRISC scatter + FPU driver kernel.
//
// Two phases on every core:
//
// Phase 1 (SFPU driver):
//   Push (px, py, sx, sy) tiles to input CBs, wait for SFPU outputs
//   (bxl, byl, overlap_x[0..7], overlap_y[0..7]) on CBs c_4..c_21.
//   For all 1024 cells per cell-tile: compute tile_id from (bxl, byl),
//   build a V14Record, and bucket it by tile_id in L1. BRISC handles
//   all 1024 cells (no NCRISC split).
//
// Phase 2 (FPU feed):
//   For each non-empty tile_id bucket: pack OX/OY tiles from records,
//   push to CBs c_22/c_23 in N_INFLIGHT batches for TRISC FPU matmul.
//   Push sentinel at end of each tile group. Wait for CB_PARTIAL (c_25),
//   convert face→row-major, and NOC-write 2 KB bf16 partial density tile
//   to partial_buf_v14[my_writer_id × total_tiles + tile_id] in DRAM.
//
// Runtime args:
//   0:  addr_px             DRAM addr of px SoA tiles
//   1:  addr_py
//   2:  addr_sx
//   3:  addr_sy
//   4:  tile_pgsz           bytes per fp32 tile page
//   5:  first_tile          first cell-tile index for this core
//   6:  n_tiles             number of cell-tiles this core processes
//   7:  tile_map_bytes      bytes of tile_to_core table
//   8:  (unused)            placeholder for core_id arg compatibility
//   9:  nc_all
//  10:  M_tiles
//  11:  N_tiles
//  12:  nbx
//  13:  nby
//  14:  total_tiles         M_tiles × N_tiles
//  15:  partial_dram        DRAM base address of partial_buf_v14
//  16:  partial_pgsz        page size = TILE_BF16_BYTES = 2048
//  17:  my_writer_id
//  18:  tables_ready_sem_id

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t V14_MAX_OVERLAP   = 8u;
constexpr uint32_t TILE_ELEMS        = 1024u;
constexpr uint32_t TILE_BF16_BYTES   = 32u * 32u * 2u;  // 2048
constexpr uint32_t K_BATCH           = 32u;
constexpr uint32_t N_INFLIGHT        = 4u;
#ifndef V14_BUCKET_CAP
#define V14_BUCKET_CAP 64
#endif
constexpr uint32_t BUCKET_CAP = V14_BUCKET_CAP;  // set by host JIT define per grid
constexpr uint32_t SENTINEL_U32      = 0xFFFFFFFFu;
constexpr uint32_t ALL_DONE_U32      = 0xFFFFFFFEu;

constexpr uint32_t CB_PX = 0u, CB_PY = 1u, CB_SX = 2u, CB_SY = 3u;
constexpr uint32_t CB_BXL     = 4u;
constexpr uint32_t CB_BYL     = 5u;
constexpr uint32_t CB_OX_BASE = 6u;   // c_6..c_13
constexpr uint32_t CB_OY_BASE = 14u;  // c_14..c_21
constexpr uint32_t CB_OX_FPU  = 22u;  // c_22
constexpr uint32_t CB_OY_FPU  = 23u;  // c_23
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t CB_PARTIAL = 25u;  // c_25

struct V14Record {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t tile_id;
    uint16_t pad;
    uint16_t overlap_x[8];
    uint16_t overlap_y[8];
};
static_assert(sizeof(V14Record) == 40, "V14Record must be 40 bytes");

constexpr uint32_t V14_RECORD_BYTES = 40u;

// fp32 → bf16 with round-to-nearest-even.
static inline uint16_t fp32_to_bf16_rne(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

// Tile/face element offset (in uint16_t units) for 32×32 bf16 tile.
static inline uint32_t face_idx(int row, int col) {
    uint32_t face = ((uint32_t)(row >> 4) * 2u) + (uint32_t)(col >> 4);
    return face * 256u + ((uint32_t)row & 15u) * 16u + ((uint32_t)col & 15u);
}

static inline void pack_ox_oy(
    const V14Record* batch, uint32_t k_used,
    uint16_t* ox, uint16_t* oy)
{
    uint32_t* ox32 = reinterpret_cast<uint32_t*>(ox);
    uint32_t* oy32 = reinterpret_cast<uint32_t*>(oy);
    for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) {
        ox32[i] = 0;
        oy32[i] = 0;
    }
    for (uint32_t k = 0; k < k_used; ++k) {
        const V14Record& r = batch[k];
        int bxl = r.bxl_local;
        int byl = r.byl_local;
        for (uint32_t j = 0; j < 8; ++j) {
            int row = bxl + (int)j;
            if (row >= 0 && row < 32) {
                ox[face_idx(row, (int)k)] = r.overlap_x[j];
            }
        }
        for (uint32_t j = 0; j < 8; ++j) {
            int col = byl + (int)j;
            if (col >= 0 && col < 32) {
                oy[face_idx((int)k, col)] = r.overlap_y[j];
            }
        }
    }
}

void kernel_main() {
    const uint32_t addr_px             = get_arg_val<uint32_t>(0);
    const uint32_t addr_py             = get_arg_val<uint32_t>(1);
    const uint32_t addr_sx             = get_arg_val<uint32_t>(2);
    const uint32_t addr_sy             = get_arg_val<uint32_t>(3);
    const uint32_t tile_pgsz           = get_arg_val<uint32_t>(4);
    const uint32_t first_tile          = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles             = get_arg_val<uint32_t>(6);
    const uint32_t tile_map_bytes      = get_arg_val<uint32_t>(7);
    (void)get_arg_val<uint32_t>(8);     // unused
    const uint32_t nc_all              = get_arg_val<uint32_t>(9);
    const uint32_t M_tiles             = get_arg_val<uint32_t>(10);
    const uint32_t N_tiles             = get_arg_val<uint32_t>(11);
    const uint32_t nbx                 = get_arg_val<uint32_t>(12);
    const uint32_t nby                 = get_arg_val<uint32_t>(13);
    const uint32_t total_tiles         = get_arg_val<uint32_t>(14);
    const uint32_t partial_dram        = get_arg_val<uint32_t>(15);
    const uint32_t partial_pgsz        = get_arg_val<uint32_t>(16);
    const uint32_t my_writer_id        = get_arg_val<uint32_t>(17);
    const uint32_t tables_ready_sem_id = get_arg_val<uint32_t>(18);
    const uint32_t drop_dram           = get_arg_val<uint32_t>(19);
    uint32_t total_drops = 0u;

    (void)nbx; (void)nby; (void)M_tiles;

    // ── L1 layout in CB_SCRATCH ─────────────────────────────────────────
    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t off = 0u;

    // tile_to_core[] — loaded by NCRISC at offset 0.
    const uint16_t* tile_to_core = reinterpret_cast<const uint16_t*>(base + off);
    off += tile_map_bytes;
    off = (off + 63u) & ~63u;

    // Per-tile-id bucket: up to BUCKET_CAP records per tile.
    V14Record* tile_buckets = reinterpret_cast<V14Record*>(base + off);
    off += total_tiles * BUCKET_CAP * V14_RECORD_BYTES;
    off = (off + 63u) & ~63u;

    // Per-tile-id record count.
    uint32_t* tile_bucket_count = reinterpret_cast<uint32_t*>(base + off);
    off += total_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;

    // Row-major staging for face→rowmaj conversion of partial density tile.
    uint16_t* dense_row_major = reinterpret_cast<uint16_t*>(base + off);
    off += TILE_BF16_BYTES;
    off = (off + 63u) & ~63u;

    // ── Wait for NCRISC to finish loading tile_to_core[] ────────────────
    {
        volatile tt_l1_ptr uint32_t* tr_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(tables_ready_sem_id));
        noc_semaphore_wait(tr_sem, 1u);
    }

    // Zero tile bucket counts.
    for (uint32_t i = 0; i < total_tiles; ++i) {
        tile_bucket_count[i] = 0u;
    }

    // Track which tile_ids have records (for Phase 2 iteration order).
    // We store tile_ids with records in a compact list as we discover them.
    // Max possible unique tiles = total_tiles.
    uint32_t* active_tile_list = reinterpret_cast<uint32_t*>(base + off);
    off += total_tiles * sizeof(uint32_t);
    off = (off + 63u) & ~63u;
    uint32_t n_active_tiles = 0u;

    // ── Address generators ──────────────────────────────────────────────
    const InterleavedAddrGen<true> gen_px = {.bank_base_address = addr_px, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_py = {.bank_base_address = addr_py, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sx = {.bank_base_address = addr_sx, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> gen_sy = {.bank_base_address = addr_sy, .page_size = tile_pgsz};
    const InterleavedAddrGen<true> pgen   = {.bank_base_address = partial_dram, .page_size = partial_pgsz};

    // ═══════════════════════════════════════════════════════════════════
    // Phase 1: SFPU driver + cell bucketing
    // ═══════════════════════════════════════════════════════════════════
    for (uint32_t t = 0; t < n_tiles; ++t) {
        DeviceZoneScopedN("V14B-SFPU-TILE");

        // Push input tile to CBs for TRISC SFPU.
        uint32_t page_id = first_tile + t;
        cb_reserve_back(CB_PX, 1);
        cb_reserve_back(CB_PY, 1);
        cb_reserve_back(CB_SX, 1);
        cb_reserve_back(CB_SY, 1);
        uint32_t l1_px = get_write_ptr(CB_PX);
        uint32_t l1_py = get_write_ptr(CB_PY);
        uint32_t l1_sx = get_write_ptr(CB_SX);
        uint32_t l1_sy = get_write_ptr(CB_SY);
        noc_async_read_page(page_id, gen_px, l1_px);
        noc_async_read_page(page_id, gen_py, l1_py);
        noc_async_read_page(page_id, gen_sx, l1_sx);
        noc_async_read_page(page_id, gen_sy, l1_sy);
        noc_async_read_barrier();
        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);
        cb_push_back(CB_SX, 1);
        cb_push_back(CB_SY, 1);

        // Wait for SFPU outputs.
        cb_wait_front(CB_BXL, 1);
        cb_wait_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V14_MAX_OVERLAP; ++j) {
            cb_wait_front(CB_OX_BASE + j, 1);
            cb_wait_front(CB_OY_BASE + j, 1);
        }

        // Read SFPU output pointers.
        const float* bxl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BXL));
        const float* byl_data = reinterpret_cast<const float*>(get_read_ptr(CB_BYL));
        const float* ox_data[V14_MAX_OVERLAP];
        const float* oy_data[V14_MAX_OVERLAP];
        for (uint32_t j = 0; j < V14_MAX_OVERLAP; ++j) {
            ox_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OX_BASE + j));
            oy_data[j] = reinterpret_cast<const float*>(get_read_ptr(CB_OY_BASE + j));
        }

        // Process all 1024 cells in this tile.
        {
        DeviceZoneScopedN("V14B-BUCKET");
        for (uint32_t ci = 0; ci < TILE_ELEMS; ++ci) {
            int bxl = (int)bxl_data[ci];
            int byl = (int)byl_data[ci];

            int tx_lo = bxl >> 5;
            int tx_hi = (bxl + 7) >> 5;
            int ty_lo = byl >> 5;
            int ty_hi = (byl + 7) >> 5;
            if (tx_hi < 0 || tx_lo >= (int)M_tiles) continue;
            if (ty_hi < 0 || ty_lo >= (int)N_tiles) continue;
            if (tx_lo < 0) tx_lo = 0;
            if (ty_lo < 0) ty_lo = 0;
            if (tx_hi >= (int)M_tiles) tx_hi = (int)M_tiles - 1;
            if (ty_hi >= (int)N_tiles) ty_hi = (int)N_tiles - 1;

            uint16_t ox_bf[8], oy_bf[8];
            for (uint32_t j = 0; j < 8; ++j) {
                ox_bf[j] = fp32_to_bf16_rne(ox_data[j][ci]);
                oy_bf[j] = fp32_to_bf16_rne(oy_data[j][ci]);
            }

            for (int tx = tx_lo; tx <= tx_hi; ++tx) {
                uint32_t tx_row = (uint32_t)tx * N_tiles;
                for (int ty = ty_lo; ty <= ty_hi; ++ty) {
                    uint32_t tile_id = tx_row + (uint32_t)ty;
                    if (tile_id >= total_tiles) continue;

                    uint32_t cnt = tile_bucket_count[tile_id];
                    if (cnt >= BUCKET_CAP) { ++total_drops; continue; }  // overflow guard

                    if (cnt == 0u) {
                        active_tile_list[n_active_tiles++] = tile_id;
                    }

                    V14Record& rec = tile_buckets[tile_id * BUCKET_CAP + cnt];
                    rec.bxl_local = (int16_t)(bxl - tx * 32);
                    rec.byl_local = (int16_t)(byl - ty * 32);
                    rec.tile_id   = (uint16_t)tile_id;
                    rec.pad       = 0;
                    for (uint32_t j = 0; j < 8; ++j) rec.overlap_x[j] = ox_bf[j];
                    for (uint32_t j = 0; j < 8; ++j) rec.overlap_y[j] = oy_bf[j];

                    tile_bucket_count[tile_id] = cnt + 1u;
                }
            }
        }
        }  // end V14B-BUCKET

        // Pop SFPU output CBs so TRISC can reuse them for next tile.
        cb_pop_front(CB_BXL, 1);
        cb_pop_front(CB_BYL, 1);
        for (uint32_t j = 0; j < V14_MAX_OVERLAP; ++j) {
            cb_pop_front(CB_OX_BASE + j, 1);
            cb_pop_front(CB_OY_BASE + j, 1);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: FPU feed — pack OX/OY tiles, push to CBs, write partial
    //          density tiles to DRAM.
    // ═══════════════════════════════════════════════════════════════════
    for (uint32_t at = 0; at < n_active_tiles; ++at) {
        DeviceZoneScopedN("V14B-FPU-TILE");
        uint32_t tile_id = active_tile_list[at];
        uint32_t cnt = tile_bucket_count[tile_id];
        V14Record* bucket = &tile_buckets[tile_id * BUCKET_CAP];

        // Pack and push in N_INFLIGHT-tile batches.
        uint32_t pushed = 0u;
        while (pushed < cnt) {
            // Pack up to N_INFLIGHT × K_BATCH records.
            uint32_t remaining = cnt - pushed;
            uint32_t n_full_batches = remaining / K_BATCH;
            if (n_full_batches > N_INFLIGHT) n_full_batches = N_INFLIGHT;
            uint32_t n_in_batch = n_full_batches;
            uint32_t leftover = remaining - n_full_batches * K_BATCH;

            // If we have leftover records and room in the batch, add one partial.
            if (leftover > 0 && n_in_batch < N_INFLIGHT) {
                n_in_batch++;
            }

            // Pad to N_INFLIGHT tiles (zero tiles contribute nothing).
            for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
                cb_reserve_back(CB_OX_FPU, 1);
                cb_reserve_back(CB_OY_FPU, 1);

                uint32_t k_used = 0;
                if (q < n_full_batches) {
                    k_used = K_BATCH;
                } else if (q == n_full_batches && leftover > 0 && q < n_in_batch) {
                    k_used = leftover;
                }

                pack_ox_oy(&bucket[pushed],
                           k_used,
                           reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX_FPU)),
                           reinterpret_cast<uint16_t*>(get_write_ptr(CB_OY_FPU)));

                cb_push_back(CB_OX_FPU, 1);
                cb_push_back(CB_OY_FPU, 1);

                if (q < n_in_batch) {
                    pushed += k_used;
                }
            }
        }

        // If cnt==0 (shouldn't happen for active tiles, but defensive), still
        // push N_INFLIGHT zero tiles so the sentinel protocol works.
        if (cnt == 0u) {
            for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
                cb_reserve_back(CB_OX_FPU, 1);
                cb_reserve_back(CB_OY_FPU, 1);
                pack_ox_oy(bucket, 0,
                           reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX_FPU)),
                           reinterpret_cast<uint16_t*>(get_write_ptr(CB_OY_FPU)));
                cb_push_back(CB_OX_FPU, 1);
                cb_push_back(CB_OY_FPU, 1);
            }
        }

        // Push sentinel batch (N_INFLIGHT sentinel OX tiles).
        for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
            cb_reserve_back(CB_OX_FPU, 1);
            uint32_t* sent = reinterpret_cast<uint32_t*>(get_write_ptr(CB_OX_FPU));
            for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) sent[i] = SENTINEL_U32;
            cb_push_back(CB_OX_FPU, 1);
        }

        // Wait for TRISC to produce the partial density tile.
        cb_wait_front(CB_PARTIAL, 1);

        // Write tile-format (face-layout) data directly to DRAM.
        // The reduce kernel's FPU unpack engine consumes tile format natively.
        {
            DeviceZoneScopedN("V14B-PARTIAL-WRITE");
            uint32_t tile_l1 = get_read_ptr(CB_PARTIAL);
            uint64_t dst_addr = pgen.get_noc_addr(tile_id * nc_all + my_writer_id);
            noc_async_write(tile_l1, dst_addr, TILE_BF16_BYTES);
            noc_async_write_barrier();
        }

        cb_pop_front(CB_PARTIAL, 1);
    }

    // Push ALL_DONE sentinel — tells TRISC Phase 2 that no more tile groups remain.
    for (uint32_t q = 0; q < N_INFLIGHT; ++q) {
        cb_reserve_back(CB_OX_FPU, 1);
        uint32_t* ad = reinterpret_cast<uint32_t*>(get_write_ptr(CB_OX_FPU));
        for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) ad[i] = ALL_DONE_U32;
        cb_push_back(CB_OX_FPU, 1);
    }

    // DROP-INSTRUMENTATION: write total_drops to drop_dram[my_writer_id].
    {
        const InterleavedAddrGen<true> dgen = {
            .bank_base_address = drop_dram,
            .page_size         = 32u,  // entire small buf is one page
        };
        noc_inline_dw_write(dgen.get_noc_addr(0) + (uint64_t)my_writer_id * 4u, total_drops);
    }
}
