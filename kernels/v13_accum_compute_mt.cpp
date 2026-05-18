// SPDX-License-Identifier: Apache-2.0
//
// V13_fpu Strategy-B — multi-tile TRISC compute kernel using custom
// K-accumulation no-MOP LLK to bypass matmul_tiles' ~24 µs per-call SW
// overhead.
//
// Per owned tile, accumulate K-cell matmul batches into fp32 DST, then pack
// to bf16 CB_DENSE. End-of-tile signaled by BRISC via N_INFLIGHT sentinel
// OX tiles (element[0] = 0xFFFFFFFF in bf16).
//
// Strategy B changes vs Strategy A (which used matmul_tiles):
//   * Init via `mm_no_mop_init_short` (experimental API in tt-metal) +
//     our v13::_llk_math_matmul_kaccum_init_ which programs the standard
//     16-MVMUL replay buffer once.
//   * Per batch: UNPACK side calls llk_unpack_AB_matmul N_INFLIGHT times
//     (one per tile index in CB front window). MATH side calls our custom
//     v13::_llk_math_matmul_kaccum_(0, N_INFLIGHT) which issues N_INFLIGHT
//     lltt::replay() calls into the SAME dst tile. Per-replay cost is
//     ~1 cycle issue + actual FPU MVMUL execution, vs ~24 µs for the full
//     matmul_tiles wrapper.
//
// Compute config: fp32_dest_acc_en = true, MathFidelity::HiFi4 (set on host).
//
// Runtime args:
//   0: n_owned_tiles

#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/matmul_custom.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/experimental/matmul_custom.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"
#include "v13_llk_math_kaccum.h"

constexpr uint32_t CB_OX    = tt::CBIndex::c_0;
constexpr uint32_t CB_OY    = tt::CBIndex::c_1;
constexpr uint32_t CB_DENSE = tt::CBIndex::c_16;

// Sentinel: bf16 elements 0 and 1 both 0xFFFF → packed uint32 = 0xFFFFFFFF.
// 0xFFFF in bf16 is NaN (since exponent=0xFF, mantissa nonzero); never a
// legitimate overlap product, so this is a safe marker.
constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

// Batch size — matches BRISC's PUSH_BATCH = N_INFLIGHT × K_BATCH.
constexpr uint32_t N_INFLIGHT = 4u;

void kernel_main() {
    const uint32_t n_owned_tiles = get_arg_val<uint32_t>(0);
    if (n_owned_tiles == 0u) return;

    // Full init via mm_init — sets up UNPACK, MATH AND PACK side for CB_DENSE
    // (mm_no_mop_init_short doesn't init pack, which we still need for pack_tile).
    // Then re-program math state for our no-MOP kaccum dispatch.
    //
    // NB: We deliberately do NOT use llk_unpack_AB_matmul_init with ct_dim=N
    // here. The standard matmul ct_dim batching has "reuse" semantics — one
    // operand (SrcA or SrcB) is loaded once and shared across all N matmul
    // outputs while the other is loaded N times. V13's "N records each with a
    // unique (OX[k], OY[k]) pair" does not fit this contract — neither operand
    // is shared. Attempting ct_dim=N deadlocks at warmup because kaccum's
    // CLR_AB after each replay clears SrcB DVALID, but the batched unpack only
    // loads SrcB once per batch.
    //
    // To genuinely HW-batch the unpack we would need a custom unpack LLK whose
    // MOP template loads BOTH SrcA and SrcB per inner iteration — i.e., write
    // our own llk_unpack_AB_matmul_mop_config that issues 2× TTI_UNPACR
    // (one to SrcA, one to SrcB) plus address increments. That is the next
    // strategy but out of scope for this iteration.
    mm_init(CB_OX, CB_OY, CB_DENSE);
    MATH(( v13::_llk_math_matmul_kaccum_init_<MathFidelity::HiFi4>() ));

    for (uint32_t t = 0u; t < n_owned_tiles; ++t) {
        DeviceZoneScopedN("V13C-TILE");
        tile_regs_acquire();  // zeroes DST

        {
            DeviceZoneScopedN("V13C-MATMUL-LOOP");
            while (true) {
                cb_wait_front(CB_OX, N_INFLIGHT);
                // Sentinel detection — BRISC pushed N_INFLIGHT sentinel OX tiles
                // at end-of-owned-tile. Read element[0] of front tile; if it's the
                // sentinel, pop the full sentinel batch and break.
                uint32_t v = read_tile_value(CB_OX, 0u, 0u);
                if (v == SENTINEL) {
                    cb_pop_front(CB_OX, N_INFLIGHT);
                    break;
                }
                cb_wait_front(CB_OY, N_INFLIGHT);

                // UNPACK side: push N_INFLIGHT distinct tile pairs into srcA/srcB.
                // The DVALID handshake auto-synchronises with MATH side replays.
                UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 0u, 0u, 1u, 1u, 1u) ));
                UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 1u, 1u, 1u, 1u, 1u) ));
                UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 2u, 2u, 1u, 1u, 1u) ));
                UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 3u, 3u, 1u, 1u, 1u) ));

                // MATH side: one set_dst_write_addr + N_INFLIGHT replays, all
                // accumulating into DST[0].
                MATH(( v13::_llk_math_matmul_kaccum_<MathFidelity::HiFi4>(/*dst*/ 0u, /*k_batch*/ N_INFLIGHT) ));

                cb_pop_front(CB_OX, N_INFLIGHT);
                cb_pop_front(CB_OY, N_INFLIGHT);
            }
        }

        tile_regs_commit();
        {
            DeviceZoneScopedN("V13C-PACK");
            tile_regs_wait();
            cb_reserve_back(CB_DENSE, 1);
            pack_tile(0, CB_DENSE);
            cb_push_back(CB_DENSE, 1);
            tile_regs_release();
        }
    }
}
