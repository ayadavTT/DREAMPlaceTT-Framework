// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bin-Index Compute Kernel — pure SFPU (vector engine), FP32.
//
// Per-cell computation (matches DREAMPlace computeTriangleDensityMapLauncher):
//
//   sx_clamped   = max(sx, BIN_SIZE_X)
//   sy_clamped   = max(sy, BIN_SIZE_Y)
//   node_x       = pos_x + (sx_clamped - sx) / 2      (= x + offset_x)
//   node_y       = pos_y + (sy_clamped - sy) / 2
//   bxl          = max(0,          floor((node_x              - XL) * INV_BIN_X))
//   bxh          = min(NUM_BINS_X, floor((node_x + sx_clamped - XL) * INV_BIN_X) + 1)
//   byl          = max(0,          floor((node_y              - YL) * INV_BIN_Y))
//   byh          = min(NUM_BINS_Y, floor((node_y + sy_clamped - YL) * INV_BIN_Y) + 1)
//
// floor() is implemented via the round-then-adjust trick (safe for x in [0, 2^22]):
//   r = (x + 2^23) - 2^23;   // round-nearest-even
//   if (r > x) r -= 1.0f;    // pull back if rounded up
//
// Inputs  (DRAM → CB):  c_0=pos_x   c_1=pos_y   c_2=sx   c_3=sy
// Outputs (CB → DRAM):  c_16=bxl    c_17=bxh    c_18=byl  c_19=byh
//
// Dst slot layout (fp32_dest_acc_en=true → 8 tiles max, indices 0–7):
//   0=pos_x  1=pos_y  2=sx  3=sy   (inputs loaded with copy_tile)
//   4=bxl    5=bxh    6=byl  7=byh (outputs written by SFPU face fn)
//
// All geometry constants (BIN_SIZE_{X,Y}_F, INV_BIN_{X,Y}_F, {X,Y}L_F,
// NUM_BINS_{X,Y}) are supplied as JIT #defines by the host.
//
// Uses a single _llk_math_eltwise_ternary_sfpu_params_ invocation so the
// SETRWC write-pointer never straddles multiple LLK calls, avoiding the
// multi-pass addressing issue documented in overlap_compute_geometry.cpp.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif

// ── Compile-time geometry constants (injected by host as JIT #defines) ──────
// Fallback defaults match the small regression test (256-bin, 1000-unit domain).
#ifndef BIS_BIN_SIZE_X_F
#define BIS_BIN_SIZE_X_F 3.90625f
#endif
#ifndef BIS_BIN_SIZE_Y_F
#define BIS_BIN_SIZE_Y_F 3.90625f
#endif
#ifndef BIS_INV_BIN_X_F
#define BIS_INV_BIN_X_F 0.256f
#endif
#ifndef BIS_INV_BIN_Y_F
#define BIS_INV_BIN_Y_F 0.256f
#endif
#ifndef BIS_XL_F
#define BIS_XL_F 0.0f
#endif
#ifndef BIS_YL_F
#define BIS_YL_F 0.0f
#endif
#ifndef BIS_NUM_BINS_X
#define BIS_NUM_BINS_X 256
#endif
#ifndef BIS_NUM_BINS_Y
#define BIS_NUM_BINS_Y 256
#endif

static constexpr float BSZ_X    = BIS_BIN_SIZE_X_F;
static constexpr float BSZ_Y    = BIS_BIN_SIZE_Y_F;
static constexpr float INV_X    = BIS_INV_BIN_X_F;
static constexpr float INV_Y    = BIS_INV_BIN_Y_F;
static constexpr float XL_C     = BIS_XL_F;
static constexpr float YL_C     = BIS_YL_F;
static constexpr float NB_X     = static_cast<float>(BIS_NUM_BINS_X);
static constexpr float NB_Y     = static_cast<float>(BIS_NUM_BINS_Y);
// 2^23: adding then subtracting this value to a float rounds it to the
// nearest integer (nearest-even).  Safe for |x| < 2^23 ≈ 8.4 M.
static constexpr float BIG      = 8388608.0f;

// ── N = dst_reg stride per tile (32 SIMD-lane groups per tile) ───────────────
static constexpr uint32_t N = 32;

// ── Helper: floor via round-then-adjust, safe for x in [0, 2^22) ─────────────
#ifdef TRISC_MATH
inline vFloat sfpu_floor(vFloat x) {
    vFloat r = (x + BIG) - BIG;
    v_if (r > x) { r -= 1.0f; } v_endif;
    return r;
}

// ── SFPU face functions — one per output ─────────────────────────────────────
//
// REGISTER PRESSURE ROOT CAUSE (empirically confirmed):
//   The RISC-V/SFPI compiler allocates LREGs for ALL vFloat variables in a
//   function at once (function-level, not block-level).  A single function that
//   computes bxl + bxh + byl + byh holds ~8–10 LREGs simultaneously, forcing
//   compiler-generated spills to dst_reg.  Those spill writes alias the input
//   tiles for faces 2 and 3 (SETRWC offset 16/24), corrupting the reads and
//   producing a systematic -1 error.
//
//   Fix: four separate face functions, each computing ONE output.  Per-function
//   peak is ≤ 5 LREGs (px, sx, sxc, nx/rx, val + 1–2 floor temps), well within
//   the usable LREG budget.  Each function is passed independently to its own
//   _llk_math_eltwise_ternary_sfpu_params_ call so the compiler sees each one
//   in isolation.
//
//   The ternary LLK signature is (fn, a, b, c, d, mode).  We pass:
//     bxl/bxh:  a=d_px, b=d_sx, c=unused(0), d=output_slot
//     byl/byh:  a=d_py, b=d_sy(3), c=unused(0), d=output_slot
//   The fourth argument (c) is unused; we hard-code 0 to satisfy the signature.

// Diagnostic A: tile d_sx (tile 2) → tile d_bxl.
#ifdef BIS_DIAG_TILE2_READ
static void face_bxl(uint32_t, uint32_t d_sx, uint32_t, uint32_t d_bxl) {
    for (uint32_t i = 0; i < 8; ++i)
        dst_reg[d_bxl * N + i] = dst_reg[d_sx * N + i];
}
// Diagnostic B: tile d_px → tile d_bxl (isolates write to high tile index).
#elif defined(BIS_DIAG_TILE0_TO_TILE4)
static void face_bxl(uint32_t d_px, uint32_t, uint32_t, uint32_t d_bxl) {
    for (uint32_t i = 0; i < 8; ++i)
        dst_reg[d_bxl * N + i] = dst_reg[d_px * N + i];
}
// Diagnostic C: bxl but WITHOUT reading tile 1 (sx hardcoded to 0).
// If this gives 0 mismatches, reading tile 1 is the root cause.
#elif defined(BIS_DIAG_NO_TILE1)
static void face_bxl(uint32_t d_px, uint32_t, uint32_t, uint32_t d_bxl) {
    const vFloat half_bsz = vFloat(0.5f * BSZ_X);
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat nx  = dst_reg[d_px * N + i] + half_bsz;
        vFloat val = sfpu_floor((nx - XL_C) * INV_X);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        dst_reg[d_bxl * N + i] = val;
    }
}
#else
static void face_bxl(uint32_t d_px, uint32_t d_sx, uint32_t, uint32_t d_bxl) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat px  = dst_reg[d_px * N + i];
        vFloat sx  = dst_reg[d_sx * N + i];
        vFloat sxc = sx; v_if (sx < BSZ_X) { sxc = vFloat(BSZ_X); } v_endif;
        vFloat nx  = px + (sxc - sx) * 0.5f;
        vFloat val = sfpu_floor((nx - XL_C) * INV_X);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        dst_reg[d_bxl * N + i] = val;
    }
}
#endif

static void face_bxh(uint32_t d_px, uint32_t d_sx, uint32_t, uint32_t d_bxh) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat px  = dst_reg[d_px * N + i];
        vFloat sx  = dst_reg[d_sx * N + i];
        vFloat sxc = sx; v_if (sx < BSZ_X) { sxc = vFloat(BSZ_X); } v_endif;
        vFloat rx  = px + (sxc - sx) * 0.5f + sxc;  // = nx + sxc
        vFloat val = sfpu_floor((rx - XL_C) * INV_X) + 1.0f;
        v_if (val > vFloat(NB_X)) { val = vFloat(NB_X); } v_endif;
        dst_reg[d_bxh * N + i] = val;
    }
}

static void face_byl(uint32_t d_py, uint32_t d_sy, uint32_t, uint32_t d_byl) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat py  = dst_reg[d_py * N + i];
        vFloat sy  = dst_reg[d_sy * N + i];
        vFloat syc = sy; v_if (sy < BSZ_Y) { syc = vFloat(BSZ_Y); } v_endif;
        vFloat ny  = py + (syc - sy) * 0.5f;
        vFloat val = sfpu_floor((ny - YL_C) * INV_Y);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; } v_endif;
        dst_reg[d_byl * N + i] = val;
    }
}

static void face_byh(uint32_t d_py, uint32_t d_sy, uint32_t, uint32_t d_byh) {
    for (uint32_t i = 0; i < 8; ++i) {
        vFloat py  = dst_reg[d_py * N + i];
        vFloat sy  = dst_reg[d_sy * N + i];
        vFloat syc = sy; v_if (sy < BSZ_Y) { syc = vFloat(BSZ_Y); } v_endif;
        vFloat ry  = py + (syc - sy) * 0.5f + syc;  // = ny + syc
        vFloat val = sfpu_floor((ry - YL_C) * INV_Y) + 1.0f;
        v_if (val > vFloat(NB_Y)) { val = vFloat(NB_Y); } v_endif;
        dst_reg[d_byh * N + i] = val;
    }
}
#endif  // TRISC_MATH

// ── Kernel entry point ────────────────────────────────────────────────────────
//
// ROOT CAUSE: fp32_dest_acc_en=true limits the DST register file to 4 tiles
// (tile indices 0–3).  The prior approach used tiles 0–3 for inputs and 4–7
// for outputs.  Tiles 4–7 are out-of-bounds and alias garbage memory, causing
// the systematic output corruption.
//
// FIX: use exactly 2 DST tiles per pass.  Tile 0 holds the primary input
// (pos_x or pos_y) AND the output (written in-place by the SFPU).  Tile 1
// holds the secondary input (sx or sy).  Four separate acquire→copy→SFPU→pack
// cycles produce the four outputs sequentially; each cycle uses only DST[0:1].
//
// The SFPU reads tile 0 (primary) and tile 1 (secondary), computes the result,
// and stores it back to tile 0 (d_out=0 == d_px/d_py).  Within one face-
// function call, each SIMD group i reads dst_reg[d*N+i] before writing to the
// same slot, so in-place is data-race-free.
void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_px  = tt::CBIndex::c_0;
    constexpr auto cb_py  = tt::CBIndex::c_1;
    constexpr auto cb_sx  = tt::CBIndex::c_2;
    constexpr auto cb_sy  = tt::CBIndex::c_3;

    constexpr auto cb_bxl = tt::CBIndex::c_16;
    constexpr auto cb_bxh = tt::CBIndex::c_17;
    constexpr auto cb_byl = tt::CBIndex::c_18;
    constexpr auto cb_byh = tt::CBIndex::c_19;

    init_sfpu(cb_px, cb_bxl);

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        // All four input tiles must be ready before any pass starts.
        cb_wait_front(cb_px, 1);
        cb_wait_front(cb_py, 1);
        cb_wait_front(cb_sx, 1);
        cb_wait_front(cb_sy, 1);

        // ── Pass 1: bxl ────────────────────────────────────────────────────────
        // DST[0]=pos_x, DST[1]=sx → SFPU → DST[0]=bxl → pack → cb_bxl
        tile_regs_acquire();
        copy_tile_init(cb_px);  copy_tile(cb_px, 0, 0);
        copy_tile_init(cb_sx);  copy_tile(cb_sx, 0, 1);
        MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
            face_bxl, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(cb_bxl, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_bxl);
        pack_tile(0, cb_bxl);
        tile_regs_release();
        cb_push_back(cb_bxl, 1);

        // ── Pass 2: bxh ────────────────────────────────────────────────────────
        tile_regs_acquire();
        copy_tile_init(cb_px);  copy_tile(cb_px, 0, 0);
        copy_tile_init(cb_sx);  copy_tile(cb_sx, 0, 1);
        MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
            face_bxh, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(cb_bxh, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_bxh);
        pack_tile(0, cb_bxh);
        tile_regs_release();
        cb_push_back(cb_bxh, 1);

        // ── Pass 3: byl ────────────────────────────────────────────────────────
        tile_regs_acquire();
        copy_tile_init(cb_py);  copy_tile(cb_py, 0, 0);
        copy_tile_init(cb_sy);  copy_tile(cb_sy, 0, 1);
        MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
            face_byl, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(cb_byl, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_byl);
        pack_tile(0, cb_byl);
        tile_regs_release();
        cb_push_back(cb_byl, 1);

        // ── Pass 4: byh ────────────────────────────────────────────────────────
        tile_regs_acquire();
        copy_tile_init(cb_py);  copy_tile(cb_py, 0, 0);
        copy_tile_init(cb_sy);  copy_tile(cb_sy, 0, 1);
        MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
            face_byh, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(cb_byh, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_byh);
        pack_tile(0, cb_byh);
        tile_regs_release();
        cb_push_back(cb_byh, 1);

        // Retire input tiles.
        cb_pop_front(cb_px, 1);
        cb_pop_front(cb_py, 1);
        cb_pop_front(cb_sx, 1);
        cb_pop_front(cb_sy, 1);
    }
}
