// SPDX-License-Identifier: Apache-2.0
//
// V16 Phase B.0 — SFPU outer-product microbenchmark (TRISC compute).
//
// Measures the per-batch cost of replacing V13/V15's BRISC scalar pack_ox_oy
// (11.6 µs/batch measured) with an SFPU-side equivalent that accumulates 32
// records' 8×8 outer products directly into a DST tile.
//
// Reference: V16_PLAN.md §6.5 Phase B.0 gate (≤ 4 µs/batch → green light;
//            ≤ 7 µs/batch → caution; > 7 → pivot to FPU+SFPU-pack).
//
// Per batch:
//   1. SFPU zero-fill DST[0]   (32 vFloat stores = full 32×32 tile)
//   2. SFPU outer-product loop: 32 "records" each contributing one face's
//      worth of multiply-add at runtime-varying offsets
//
// Tracy zones (read via tools/v13_profile_summary.py):
//   V16B0-BATCH   — whole batch wall time (the headline number)
//   V16B0-ZERO    — just the zero-fill phase
//   V16B0-OPROD   — just the outer-product phase
//
// Runtime args:
//   0: n_batches    how many batches to time (default 10000)

#include <cstdint>

#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api.h"
#endif

#include "tools/profiler/kernel_profiler.hpp"

// ── SFPU body (compiled only on the MATH TRISC) ────────────────────────────
#ifdef TRISC_MATH

// 32×32 tile in DST = 32 dst_reg slots (each is a 32-lane vFloat covering
// one row of 16 columns × 2 col-halves per face). Faces are: TL (idx 0..7),
// TR (8..15), BL (16..23), BR (24..31).
constexpr uint32_t N_VEC_PER_TILE = 32u;

// Zero-fill one face (8 vFloat stores = 16×16 elements).
// dst_idx is the tile index in DST (0..15 typically).
inline void sfpu_zero_face(uint32_t base_idx) {
    for (uint32_t i = 0; i < 8u; ++i) {
        sfpi::dst_reg[base_idx + i] = sfpi::vFloat(0.0f);
    }
}

// Zero-fill an entire 32×32 tile in DST (4 faces × 8 vFloat = 32 stores).
inline void sfpu_zero_tile(uint32_t dst_idx) {
    const uint32_t base = dst_idx * N_VEC_PER_TILE;
    sfpu_zero_face(base + 0);    // TL face
    sfpu_zero_face(base + 8);    // TR face
    sfpu_zero_face(base + 16);   // BL face
    sfpu_zero_face(base + 24);   // BR face
}

// One "record"'s worth of outer-product contribution.
//
// Writes 8 vFloat multiply-add stores covering ONE face (16×16 elements).
// Each vFloat is 32 lanes, so 8 stores × 32 lanes = 256 elements = full face.
// This is the granularity needed to do an 8×8 outer product since the face
// has 16 columns and we update 8 rows × 8 cols at runtime-varying offsets
// — but here we just measure the SFPU throughput, not the scatter logic.
//
// dst_idx  — destination tile index in DST (0..15)
// face_off — face index × 8 (0, 8, 16, or 24)
// ox_b     — broadcast scalar (overlap_x[r][j]-like value)
// oy_v     — vector of 8 oy values (treated as broadcast across all 32 lanes)
inline void sfpu_record_outer_product(
    uint32_t dst_idx,
    uint32_t face_off,
    sfpi::vFloat ox_b,
    sfpi::vFloat oy_v)
{
    const uint32_t base = dst_idx * N_VEC_PER_TILE + face_off;
    // 8 multiply-add stores covering one face (8 vFloat slots × 32 lanes
    // = 256 elements = full 16×16 face). Stays in DST bounds (base+0..7
    // is within [face_off, face_off+7] which is within [0, 31]).
    sfpi::vFloat prod = ox_b * oy_v;
    sfpi::dst_reg[base + 0] = sfpi::dst_reg[base + 0] + prod;
    sfpi::dst_reg[base + 1] = sfpi::dst_reg[base + 1] + prod;
    sfpi::dst_reg[base + 2] = sfpi::dst_reg[base + 2] + prod;
    sfpi::dst_reg[base + 3] = sfpi::dst_reg[base + 3] + prod;
    sfpi::dst_reg[base + 4] = sfpi::dst_reg[base + 4] + prod;
    sfpi::dst_reg[base + 5] = sfpi::dst_reg[base + 5] + prod;
    sfpi::dst_reg[base + 6] = sfpi::dst_reg[base + 6] + prod;
    sfpi::dst_reg[base + 7] = sfpi::dst_reg[base + 7] + prod;
}

// 32 records × 8 SFPU multiply-add stores = 256 vFloat ops per batch.
// Simulates the per-batch outer-product accumulate phase of V16. Records
// rotate through the 4 faces of the destination tile (r & 3 → face 0..3).
inline void sfpu_batch_outer_product(uint32_t dst_idx) {
    for (uint32_t r = 0u; r < 32u; ++r) {
        const uint32_t face_off = (r & 3u) * 8u;       // 0/8/16/24
        sfpi::vFloat ox_b = sfpi::vFloat(0.01f + (float)r * 0.001f);
        sfpi::vFloat oy_v = sfpi::vFloat(0.02f + (float)r * 0.001f);
        sfpu_record_outer_product(dst_idx, face_off, ox_b, oy_v);
    }
}

// One record's worth of pure-store work — same store count as the outer-
// product version but no read-modify-write. This is what V16-HYBRID pack
// would do: write the OX/OY tile contents to DST, then FPU kaccum consumes
// them. Eliminates the 3-instruction-per-vFloat penalty of accumulate.
inline void sfpu_record_pure_stores(
    uint32_t dst_idx,
    uint32_t face_off,
    sfpi::vFloat val)
{
    const uint32_t base = dst_idx * N_VEC_PER_TILE + face_off;
    sfpi::dst_reg[base + 0] = val;
    sfpi::dst_reg[base + 1] = val;
    sfpi::dst_reg[base + 2] = val;
    sfpi::dst_reg[base + 3] = val;
    sfpi::dst_reg[base + 4] = val;
    sfpi::dst_reg[base + 5] = val;
    sfpi::dst_reg[base + 6] = val;
    sfpi::dst_reg[base + 7] = val;
}

// 32 records × 8 pure SFPU stores = 256 vFloat stores per batch.
// Per the V16-HYBRID design, this is the SFPU pack work (writing OX/OY
// tiles in DST for FPU kaccum to consume). Compare to sfpu_batch_outer_product
// to isolate the read-modify-write overhead.
inline void sfpu_batch_pure_stores(uint32_t dst_idx) {
    for (uint32_t r = 0u; r < 32u; ++r) {
        const uint32_t face_off = (r & 3u) * 8u;
        sfpi::vFloat val = sfpi::vFloat(0.01f + (float)r * 0.001f);
        sfpu_record_pure_stores(dst_idx, face_off, val);
    }
}

#endif  // TRISC_MATH

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);

    constexpr auto cb_out = tt::CBIndex::c_16;
    init_sfpu(cb_out, cb_out);

    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        DeviceZoneScopedN("V16B0-BATCH");

        tile_regs_acquire();

        // Phase 1: SFPU zero-fill 32×32 tile
        {
            DeviceZoneScopedN("V16B0-ZERO");
            MATH(( sfpu_zero_tile(0u) ));
        }

        // Phase 2: SFPU outer-product loop (READ-MODIFY-WRITE — slow variant)
        // Measured ~7.73 µs/batch in previous bench.
        {
            DeviceZoneScopedN("V16B0-OPROD");
            MATH(( sfpu_batch_outer_product(0u) ));
        }

        // Phase 3: SFPU pure stores (WRITE-ONLY — V16-HYBRID pack variant).
        // Same store count as OPROD (256 vFloat stores) but no read-modify-
        // write. Hypothesis: each store collapses from 3 SFPU instructions
        // to 1, so this should be ~3× faster than OPROD (~2.6 µs/batch).
        {
            DeviceZoneScopedN("V16B0-STORE");
            MATH(( sfpu_batch_pure_stores(0u) ));
        }

        tile_regs_commit();

        // Pack out so TRISC PACK thread completes its end-of-iter handshake.
        // Production examples (custom_sfpi_add) always pack after commit;
        // without this the PACK thread waits forever for a pack_tile call,
        // deadlocking the kernel on the first iteration. BRISC consumes
        // (cb_wait_front + cb_pop_front) so the CB never fills up.
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
