// SPDX-License-Identifier: Apache-2.0
//
// V13 Phase 0b — OX/OY packing microbench (BRISC).
//
// Measures how long BRISC takes to pack K=32 synthetic cell records into a
// 32x32 bf16 OX tile + 32x32 bf16 OY tile, in TT-Metal tile/face layout.
// This is the inner loop of V13_accum_brisc.cpp.
//
// Per cell record (40 bytes, matches V13's plan):
//   int16_t bxl_local;  // signed bin offset in receiver tile (-7..31)
//   int16_t byl_local;
//   uint16_t pad[2];
//   bfloat16 overlap_x[8];
//   bfloat16 overlap_y[8];
//
// Tile/face layout of a 32x32 tile (4 faces of 16x16, row-major within face,
// face order TL, TR, BL, BR):
//   offset(row, col) = face_id * 256 + (row & 15) * 16 + (col & 15)
//   face_id          = (row >> 4) * 2 + (col >> 4)
//
// Runtime args:
//   0: n_batches             how many K=32 batches to pack
//   1: l1_records_off        L1 byte offset where 32 input records live
//   2: l1_ox_off             L1 byte offset where the OX tile is built
//   3: l1_oy_off             L1 byte offset where the OY tile is built
//   4: sink_dram_addr        DRAM address to dump final tile (prevents DCE)
//   5: cycle_lo_l1_off       (out) L1 offset where we write start cycle
//   6: cycle_hi_l1_off       (out) L1 offset where we write end cycle

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t TILE_BF16_BYTES = 32 * 32 * 2;  // 2048

struct CellRecord {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t pad[2];
    uint16_t overlap_x[8];  // bf16 stored as raw 16-bit
    uint16_t overlap_y[8];
};
static_assert(sizeof(CellRecord) == 40, "CellRecord must be 40 bytes");

// Tile/face element offset: returns the index (in uint16_t units) into a
// 32x32 bf16 tile stored in TT-Metal face layout (4 × 16x16, row-major
// within each face, face order TL, TR, BL, BR).
static inline uint32_t face_idx(int row, int col) {
    uint32_t face = ((uint32_t)(row >> 4) * 2u) + (uint32_t)(col >> 4);
    return face * 256u + ((uint32_t)row & 15u) * 16u + ((uint32_t)col & 15u);
}

static inline uint32_t rdcycle32() {
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

void kernel_main() {
    const uint32_t n_batches      = get_arg_val<uint32_t>(0);
    const uint32_t l1_records_off = get_arg_val<uint32_t>(1);
    const uint32_t l1_ox_off      = get_arg_val<uint32_t>(2);
    const uint32_t l1_oy_off      = get_arg_val<uint32_t>(3);
    const uint32_t sink_dram_addr = get_arg_val<uint32_t>(4);
    const uint32_t cycle_lo_off   = get_arg_val<uint32_t>(5);
    const uint32_t cycle_hi_off   = get_arg_val<uint32_t>(6);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    const CellRecord* records = reinterpret_cast<const CellRecord*>(base + l1_records_off);
    uint16_t* ox = reinterpret_cast<uint16_t*>(base + l1_ox_off);
    uint16_t* oy = reinterpret_cast<uint16_t*>(base + l1_oy_off);
    volatile uint32_t* cycle_lo = reinterpret_cast<volatile uint32_t*>(base + cycle_lo_off);
    volatile uint32_t* cycle_hi = reinterpret_cast<volatile uint32_t*>(base + cycle_hi_off);

    // Pre-fill records with deterministic synthetic data so the loop has
    // realistic divergence (cells with different bxl/byl offsets cause
    // different write patterns). We do this once outside the timing loop.
    CellRecord* recs_w = const_cast<CellRecord*>(records);
    for (uint32_t k = 0; k < 32; ++k) {
        // bxl_local distributed across [0, 24] so 8-wide overlap fits in tile
        // (skip the edge cases for now — the worst-case edge cells are
        // <5% of all cells in practice).
        recs_w[k].bxl_local = (int16_t)((k * 7) & 31);  // 0..24-ish
        recs_w[k].byl_local = (int16_t)((k * 11) & 31);
        for (uint32_t j = 0; j < 8; ++j) {
            // bf16 of 0.5 = 0x3F00; vary slightly per k, j.
            recs_w[k].overlap_x[j] = (uint16_t)(0x3F00 + k + j);
            recs_w[k].overlap_y[j] = (uint16_t)(0x3F00 + k * 2 + j);
        }
    }

    // ── Timed loop ────────────────────────────────────────────────────────
    *cycle_lo = rdcycle32();

    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        // Zero OX and OY (1024 × 2 B = 2048 B per tile).
        // BRISC has no memset; loop 32-bit word stores for throughput.
        uint32_t* ox32 = reinterpret_cast<uint32_t*>(ox);
        uint32_t* oy32 = reinterpret_cast<uint32_t*>(oy);
        for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) {
            ox32[i] = 0;
            oy32[i] = 0;
        }

        // Pack K=32 cells. Each cell contributes 8 bf16 to OX (one tile row)
        // and 8 bf16 to OY (one tile column) at offsets (bxl_local, byl_local).
        for (uint32_t k = 0; k < 32; ++k) {
            const CellRecord& r = records[k];
            int bxl = r.bxl_local;
            int byl = r.byl_local;

            // OX: row k, cols bxl..bxl+7
            for (uint32_t j = 0; j < 8; ++j) {
                int col = bxl + (int)j;
                if (col < 0 || col >= 32) continue;
                ox[face_idx((int)k, col)] = r.overlap_x[j];
            }
            // OY: rows byl..byl+7, col k
            for (uint32_t j = 0; j < 8; ++j) {
                int row = byl + (int)j;
                if (row < 0 || row >= 32) continue;
                oy[face_idx(row, (int)k)] = r.overlap_y[j];
            }
        }
    }

    *cycle_hi = rdcycle32();

    // Sink final OX (+ one byte of OY for completeness) to DRAM so the
    // compiler can't dead-code-eliminate the inner loop.
    const InterleavedAddrGen<true> gen = {
        .bank_base_address = sink_dram_addr,
        .page_size         = TILE_BF16_BYTES,
    };
    noc_async_write((uint32_t)ox, gen.get_noc_addr(0), TILE_BF16_BYTES);
    noc_async_write_barrier();
}
