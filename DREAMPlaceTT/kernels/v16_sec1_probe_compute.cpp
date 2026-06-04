// SPDX-License-Identifier: Apache-2.0
//
// V16 Step 1 — SEC1 Haloize silicon validation probe (TRISC compute).
//
// Runs V13's paired-unpack + custom kaccum LLKs UNCHANGED, optionally setting
// the THCON_SEC1_REG2_Haloize_mode bit (= SrcB within-face transpose) before
// the matmul. Used by host to compare baseline (Haloize=0) vs candidate
// (Haloize=1) outputs and answer "does Blackhole silicon honor SEC1 Haloize
// for matmul-style unpacks?".
//
// Test pattern (set up by host):
//   OX[r, c] = c     (col-index broadcast across all rows)
//   OY        = I   (identity matrix)
//   Expected D = OX × OY = OX
//     With Haloize OFF: D[r, c] = c (matches OX).
//     With Haloize ON, math addr_mods unchanged (V13 kaccum keeps
//       transpose=false in its init): SrcB gets within-face transposed,
//       face order stays TL/TR/BL/BR. So D[r, c] = (r & 15) + (c >> 4) * 16
//       — TL face: D = r within-face-row (constant per row, increases with r);
//       TR face: same pattern shifted by +16.
//
// Runtime args:
//   0: sec0_haloize  (0 or 1) — SrcA within-face transpose
//   1: sec1_haloize  (0 or 1) — SrcB within-face transpose

#include <cstdint>

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

#include "v13_llk_math_kaccum.h"
#include "v13_llk_unpack_paired.h"

constexpr uint32_t CB_OX    = tt::CBIndex::c_0;
constexpr uint32_t CB_OY    = tt::CBIndex::c_1;
constexpr uint32_t CB_DENSE = tt::CBIndex::c_16;

#ifdef TRISC_UNPACK
namespace v16_probe {
// Set THCON_SEC{0,1}_REG2_Haloize_mode = v. v ∈ {0, 1}. Uses the same
// cfg_reg_rmw_tensix helper that llk_unpack_AB_matmul.h:208 uses for SrcA
// (THCON_SEC0). Register addresses are defined in
// blackhole/cfg_defines.h:4216 (SEC0) and 4231 (SEC1).
inline void set_sec0_haloize(uint32_t v) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(v);
}
inline void set_sec1_haloize(uint32_t v) {
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Haloize_mode_RMW>(v);
}
}  // namespace v16_probe
#endif  // TRISC_UNPACK

void kernel_main() {
    const uint32_t sec0_haloize = get_arg_val<uint32_t>(0);
    const uint32_t sec1_haloize = get_arg_val<uint32_t>(1);

    // Standard V13 init sequence — sets up SEC0 Haloize=0 (no SrcA transpose)
    // via mm_init's default; SEC1 Haloize is left untouched at this point.
    mm_init(CB_OX, CB_OY, CB_DENSE);
    UNPACK(( v13::_v13_unpack_paired_init_() ));
    MATH(( v13::_llk_math_matmul_kaccum_init_<MathFidelity::HiFi4>() ));

    // Apply (or explicitly clear) BOTH Haloize bits AFTER all the standard
    // inits. We set unconditionally to avoid relying on any implicit
    // default from previous kernel launches in the same session.
    UNPACK(( v16_probe::set_sec0_haloize(sec0_haloize) ));
    UNPACK(( v16_probe::set_sec1_haloize(sec1_haloize) ));

    cb_wait_front(CB_OX, 1);
    cb_wait_front(CB_OY, 1);

    tile_regs_acquire();
    UNPACK(( v13::_v13_unpack_paired_run_(CB_OX, CB_OY, 1u) ));
    MATH(( v13::_llk_math_matmul_kaccum_<MathFidelity::HiFi4>(0u, 1u) ));
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(CB_DENSE, 1);
    pack_tile(0, CB_DENSE);
    cb_push_back(CB_DENSE, 1);
    tile_regs_release();

    cb_pop_front(CB_OX, 1);
    cb_pop_front(CB_OY, 1);
}
