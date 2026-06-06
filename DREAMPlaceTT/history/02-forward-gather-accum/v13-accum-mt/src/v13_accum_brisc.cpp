// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Phase 1.1 — BRISC reader + OX/OY packer.
//
// For the standalone accumulate smoke, this kernel:
//   1. Reads n_batches * K=32 cell records from a DRAM input page.
//   2. For each batch, builds a 32x32 bf16 OX tile and a 32x32 bf16 OY tile
//      in face/tile layout, then pushes them to CB_OX / CB_OY for TRISC.
//
// Packing convention (matches v13_accum_compute's matmul_tiles(OX, OY, ...)):
//   density[bxw, byw] += sum_k OX[bxw, k] * OY[k, byw]
//   → OX[bxw, k] = cell k's overlap_x at relative bin bxw (else 0)
//   → OY[k, byw] = cell k's overlap_y at relative bin byw (else 0)
//
// Cell record layout (40 bytes):
//   int16  bxl_local       signed offset of overlap_x[0] in receiver tile (-7..31)
//   int16  byl_local       same for y
//   uint16 pad[2]
//   uint16 overlap_x[8]    bf16 stored as raw 16-bit
//   uint16 overlap_y[8]
//
// CB layout:
//   c_0   CB_OX     bf16  2 tiles
//   c_1   CB_OY     bf16  2 tiles
//   c_16  CB_DENSE  bf16  1 tile  (TRISC produces, NCRISC consumes)
//
// L1 scratch (CB_SCRATCH = c_24): holds K=32 records per batch (1280 B/batch).
//
// Runtime args:
//   0: records_dram_addr   DRAM base of cell records (n_batches * K * 40 bytes)
//   1: records_pgsz        bytes per "page" — set to one batch (K * 40 = 1280)
//   2: n_batches

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_OX      = tt::CBIndex::c_0;
constexpr uint32_t CB_OY      = tt::CBIndex::c_1;
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t TILE_BF16_BYTES = 32 * 32 * 2;  // 2048
constexpr uint32_t K_BATCH = 32;

struct CellRecord {
    int16_t  bxl_local;
    int16_t  byl_local;
    uint16_t pad[2];
    uint16_t overlap_x[8];  // bf16 raw
    uint16_t overlap_y[8];
};
static_assert(sizeof(CellRecord) == 40, "CellRecord must be 40 bytes");

// Tile/face element offset (in uint16_t units) for 32x32 bf16 tile.
// 4 faces of 16x16, row-major within face, face order TL, TR, BL, BR.
static inline uint32_t face_idx(int row, int col) {
    uint32_t face = ((uint32_t)(row >> 4) * 2u) + (uint32_t)(col >> 4);
    return face * 256u + ((uint32_t)row & 15u) * 16u + ((uint32_t)col & 15u);
}

void kernel_main() {
    const uint32_t records_dram = get_arg_val<uint32_t>(0);
    const uint32_t records_pgsz = get_arg_val<uint32_t>(1);
    const uint32_t n_batches    = get_arg_val<uint32_t>(2);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    CellRecord* records = reinterpret_cast<CellRecord*>(base);

    const InterleavedAddrGen<true> gen = {
        .bank_base_address = records_dram,
        .page_size         = records_pgsz,
    };

    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        // Read one batch (K=32 records = 1280 B) into L1 staging.
        noc_async_read(gen.get_noc_addr(batch),
                       reinterpret_cast<uint32_t>(records),
                       records_pgsz);
        noc_async_read_barrier();

        // Reserve OX and OY tile slots in their CBs.
        cb_reserve_back(CB_OX, 1);
        cb_reserve_back(CB_OY, 1);

        uint16_t* ox = reinterpret_cast<uint16_t*>(get_write_ptr(CB_OX));
        uint16_t* oy = reinterpret_cast<uint16_t*>(get_write_ptr(CB_OY));

        // Zero both tiles (2 KB each, 32-bit stores for throughput).
        uint32_t* ox32 = reinterpret_cast<uint32_t*>(ox);
        uint32_t* oy32 = reinterpret_cast<uint32_t*>(oy);
        for (uint32_t i = 0; i < TILE_BF16_BYTES / 4; ++i) {
            ox32[i] = 0;
            oy32[i] = 0;
        }

        // Pack K=32 cells.
        //  - OX: column k contains cell k's overlap_x[0..7] at rows bxl..bxl+7
        //  - OY: row    k contains cell k's overlap_y[0..7] at cols byl..byl+7
        for (uint32_t k = 0; k < K_BATCH; ++k) {
            const CellRecord& r = records[k];
            int bxl = (int)r.bxl_local;
            int byl = (int)r.byl_local;

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

        cb_push_back(CB_OX, 1);
        cb_push_back(CB_OY, 1);
    }
}
