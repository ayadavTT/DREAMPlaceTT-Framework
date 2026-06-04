// SPDX-License-Identifier: Apache-2.0
//
// V11 Stage B' — SFPU outer-product microbench (TRISC compute).
//
// Measures the per-batch wall-clock cost of computing 64 (j, k) outer-product
// values per cell on the SFPU, mirroring what V11 Stage B' would do if the
// 8x8 outer product were moved from BRISC/NCRISC scalar to TRISC SFPU.
//
// Per batch of 1024 cells (one SFPU tile of cells), the kernel:
//   1. Pre-loads two synthetic input tiles (used as ox[*] and oy[*] surrogates)
//      from c_0 and c_1. These are read once per batch and reused 64 times so
//      the SFPU operand source isn't constant-folded by the compiler.
//   2. Runs 64 SFPU passes, each computing outer[J, K] = ox[J] * oy[K] for all
//      1024 cells and packing the result to one output tile.
//
// The 64 SFPU passes use the v4_compute SFPU_PASS macro pattern verbatim so
// the per-pass overhead (acquire / unpack / face_fn / pack / push) matches
// what V11 would pay in a real Stage B' integration.
//
// Tracy zones:
//   V11OP-BATCH         — whole-batch wall time (the headline number)
//   V11OP-INPUT-LOAD    — initial unpack of ox/oy synthetic data
//   V11OP-PASS-JK       — one (J, K) outer-product pass
//
// Runtime args:
//   0: n_batches    how many batches to time (default 10000)

#include <cstdint>

#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

// ── SFPU face function — outer product (ox * oy elementwise) ──────────────
#ifdef TRISC_MATH

static constexpr uint32_t N = 32;

// face_outer_product: dst_out = dst_a * dst_b, elementwise across all 32 lanes
// of 8 vFloat slots per face (= one 16x16 face per call; full tile after the
// face wrapper invokes for all 4 faces).
//
// d_a holds the "ox" surrogate tile (32 vFloat slots, 1024 cells worth).
// d_b holds the "oy" surrogate tile.
// d_out receives the elementwise product.
//
// This is the per-(J, K) SFPU work that V11 Stage B' would do. The compute
// itself is one vFloat multiply per slot per face = 8 multiplies per face × 4
// faces = 32 SFPU instructions per pass.
static void face_outer_product(uint32_t d_a, uint32_t d_b, uint32_t, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat a = dst_reg[d_a * N + i];
        vFloat b = dst_reg[d_b * N + i];
        dst_reg[d_out * N + i] = a * b;
    }
}

#endif  // TRISC_MATH

// ── SFPU_PASS macro — one complete acquire→copy→compute→pack→release cycle ──
// Same pattern as v4_compute.cpp's SFPU_PASS, scoped per (J, K) so each pass
// shows up as its own Tracy zone.
#define V11OP_PASS(cb_a, cb_b, cb_out, face_fn, zone_label)  \
    {                                                         \
        DeviceZoneScopedN(zone_label);                        \
        tile_regs_acquire();                                  \
        copy_tile_init(cb_a);  copy_tile(cb_a, 0, 0);         \
        copy_tile_init(cb_b);  copy_tile(cb_b, 0, 1);         \
        MATH(_llk_math_eltwise_ternary_sfpu_params_(          \
            face_fn, 0, 1, 0, 0,                              \
            static_cast<int>(VectorMode::RC)));               \
        tile_regs_commit();                                   \
        cb_reserve_back(cb_out, 1);                           \
        tile_regs_wait();                                     \
        pack_reconfig_data_format(cb_out);                    \
        pack_tile(0, cb_out);                                 \
        tile_regs_release();                                  \
        cb_push_back(cb_out, 1);                              \
    }

void kernel_main() {
    const uint32_t n_batches = get_arg_val<uint32_t>(0);

    // Two input CBs hold synthetic ox / oy surrogate data. BRISC stages the
    // same tile into both each iteration — they get reused 64 times across
    // (J, K) pairs to amortize the loads.
    constexpr auto cb_a   = tt::CBIndex::c_0;     // surrogate ox[*]
    constexpr auto cb_b   = tt::CBIndex::c_1;     // surrogate oy[*]
    constexpr auto cb_out = tt::CBIndex::c_16;    // 64 output tiles per batch

    init_sfpu(cb_a, cb_out);

    for (uint32_t batch = 0; batch < n_batches; ++batch) {
        DeviceZoneScopedN("V11OP-BATCH");

        // Wait once per batch for the synthetic ox/oy tiles. BRISC keeps both
        // CBs primed; we read them many times during the 64 passes.
        {
            DeviceZoneScopedN("V11OP-INPUT-WAIT");
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);
        }

        // 64 outer-product passes. In a real V11 Stage B' integration, each
        // (J, K) would read ox[J] from one CB and oy[K] from another (16 input
        // CBs total). Here we reuse the same 2 surrogate tiles to keep the
        // microbench self-contained — the SFPU compute cost is identical
        // because each pass still loads from L1, multiplies, and packs out.
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-00")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-01")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-02")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-03")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-04")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-05")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-06")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-07")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-10")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-11")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-12")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-13")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-14")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-15")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-16")
        V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-17")
        // Remaining 48 passes don't get individual Tracy labels (would blow
        // out the profiler buffer); they get the same V11OP-BATCH wrapper.
        for (uint32_t pass = 16; pass < 64; ++pass) {
            V11OP_PASS(cb_a, cb_b, cb_out, face_outer_product, "V11OP-PASS-XX")
        }

        // Pop the synthetic inputs so BRISC can stage the next batch.
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
