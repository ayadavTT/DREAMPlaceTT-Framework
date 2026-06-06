// SPDX-License-Identifier: Apache-2.0
//
// V13 Strategy-B custom math LLK — K-direction matmul accumulation.
//
// Bypasses `matmul_tiles` / `matmul_block` by calling `lltt::replay` directly
// on the standard 16-MVMUL replay buffer that the experimental no-MOP matmul
// API already programs at math::replay_buf_offset.
//
// Per V13 contract: each batch is N K-stripes (32×32 bf16 × 32×32 bf16 →
// 32×32 fp32) that all accumulate into ONE dst tile. Standard matmul
// addrmods (specifically ADDR_MOD_5 at the end of the 16-MVMUL replay)
// reset the dest *pointer* (clr=1) without clearing dest values, so
// consecutive replays naturally accumulate into the same 32×32 dst tile.
//
// Per K-step we also clear srcA and srcB DVALID so the unpacker can push
// the next K-stripe's operand pair.
//
// Caller responsibilities:
//   * Unpack must be initialised via `mm_no_mop_init_short(...)`
//     (api/compute/experimental/matmul_custom.h) BEFORE the first dispatch.
//   * For each batch the unpack side must push N tile pairs to CB_OX/CB_OY
//     (one `llk_unpack_AB_matmul` call per index 0..N-1) before calling
//     `_llk_math_matmul_kaccum_(dst_index, N)`.
//   * `tile_regs_acquire()` must have been called so dst[dst_index] starts
//     at 0 (acquire zeroes dst).

#pragma once

#ifdef TRISC_MATH
#include "experimental/llk_math_matmul_custom_no_mop.h"

namespace v13 {

// Programs ADDR_MOD_0..ADDR_MOD_5 and loads the standard 16-MVMUL replay
// sequence into math's replay slot (offset 16, length 16). Must be called
// AFTER `mm_no_mop_init_short` on the unpack side (the latter also calls
// llk_math_matmul_init_no_mop which writes the same slots; whichever is
// called second wins — both write equivalent contents for ct=rt=1).
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_kaccum_init_()
{
    static_assert(THROTTLE_LEVEL == 0,
                  "V13 kaccum supports only THROTTLE_LEVEL=0 right now");
    matmul_validate_no_mop_contract();
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(/*transpose*/ false);
    matmul_configure_mop_custom<math_fidelity>(/*ct_dim*/ 1, /*rt_dim*/ 1);
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
}

// Run k_batch consecutive 32×32 matmul-tile operations, all accumulating
// into dst[dst_index]. Mirrors the high-fidelity branch of
// `_llk_math_matmul_no_mop_` (line 491+ in llk_math_matmul_custom_no_mop.h)
// but pins dst_index across the inner loop instead of advancing it.
template <MathFidelity math_fidelity>
inline void _llk_math_matmul_kaccum_(std::uint32_t dst_index, std::uint32_t k_batch)
{
    constexpr int  num_fidelity_phases = get_math_num_fidelity_phases(math_fidelity);
    constexpr bool high_fidelity       = math_fidelity != MathFidelity::LoFi;
    constexpr std::uint32_t replay_buf_len = 16;

    // Set dst write address ONCE for the whole batch — all replays accumulate
    // here. Without this, the standard matmul_tiles would re-set it per call
    // and pay setup cost each time.
    ckernel::math::set_dst_write_addr<
        ckernel::DstTileShape::Tile32x32,
        ckernel::UnpackDestination::SrcRegs>(dst_index);

    for (std::uint32_t k = 0; k < k_batch; ++k) {
        if constexpr (high_fidelity) {
            // HiFi runs N fidelity phases per matmul-tile. ADDR_MOD_5 (at the
            // end of each replay) increments the fidelity counter; after
            // num_fidelity_phases replays it has wrapped, so we're ready for
            // the next K-stripe.
            for (int p = 0; p < num_fidelity_phases; ++p) {
                lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
            }
        } else {
            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
        }
        // Free srcA + srcB DVALID so unpack can push the next K-stripe's
        // operand pair. (Without this the unpacker would block waiting for
        // src registers to become available.)
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
}

}  // namespace v13

#endif  // TRISC_MATH
