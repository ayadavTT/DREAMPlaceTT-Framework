# TT Metal SFPU Programming Guide
## How We Used the SFPU — Verified Working Patterns

This folder contains the two SFPU compute kernels that were implemented, debugged, and
verified to produce correct results, along with every lesson learned during that process.

---

## The Two Kernels in This Folder

### 1. `bin_index_compute.cpp` — Production SFPU kernel (VERIFIED CORRECT)

**What it does:** Computes 4 bin-index values (bxl, bxh, byl, byh) per cell using the
DREAMPlace density-scatter algorithm entirely in FP32 on the SFPU.

**Verified:** 0 mismatches across 500,000 cells against the exact DREAMPlace CPU reference.
Device kernel time: 0.165 ms on 56 Tensix cores (8×7 grid, Wormhole B0).

**Algorithm executed on SFPU:**
```
sx_clamped = max(sx, BIN_SIZE_X)
node_x     = pos_x + (sx_clamped - sx) / 2
bxl        = max(0,         floor((node_x           - XL) * INV_BIN_X))
bxh        = min(NUM_BINS_X, floor((node_x + sx_clamped - XL) * INV_BIN_X) + 1)
(y identically)
```

---

### 2. `passthrough_compute.cpp` — Diagnostic SFPU kernel (VERIFIED CORRECT)

**What it does:** Copies 4 input tiles verbatim to 4 output tiles. No arithmetic.
With `#define PASSTHROUGH_FLOOR`, applies the floor trick to channel 0 only.

**Verified:** 0 mismatches in passthrough mode, 0 mismatches in floor mode.
Used to isolate data movement precision from computation bugs.

---

## How to Write an SFPU Kernel — The Exact Pattern We Used

### Step 1 — Includes

```cpp
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif
```

`TRISC_MATH` is defined only for the SFPU (math) processor. All SFPI code (`vFloat`,
`dst_reg`, `v_if`, `v_endif`) MUST be inside `#ifdef TRISC_MATH` guards.

---

### Step 2 — Write SFPU face functions

The SFPU processes one tile face (16×16 = 256 elements) at a time, in 8 SIMD groups
of 32 lanes. The LLK calls your function once per face.

```cpp
#ifdef TRISC_MATH

// N = 32 = number of SIMD lanes per group
static constexpr uint32_t N = 32;

// BIG = 2^23, used for the floor trick
static constexpr float BIG = 8388608.0f;

// floor() via round-then-adjust (safe for |x| < 2^22)
inline vFloat sfpu_floor(vFloat x) {
    vFloat r = (x + BIG) - BIG;   // rounds to nearest-even integer
    v_if (r > x) { r -= 1.0f; }   // pull back if rounded up
    v_endif;
    return r;
}

// Face function — called once per tile face (8 SIMD groups of 32 lanes)
// d_px, d_sx, d_out are DST tile indices (0, 1, 2, ...)
static void face_bxl(uint32_t d_px, uint32_t d_sx, uint32_t /*unused*/, uint32_t d_out) {
    for (uint32_t i = 0; i < 8; ++i) {              // i = SIMD group index
        vFloat px  = dst_reg[d_px * N + i];          // read from DST tile d_px, group i
        vFloat sx  = dst_reg[d_sx * N + i];
        vFloat sxc = sx;
        v_if (sx < BSZ_X) { sxc = vFloat(BSZ_X); }  // conditional: max(sx, BSZ_X)
        v_endif;
        vFloat nx  = px + (sxc - sx) * 0.5f;
        vFloat val = sfpu_floor((nx - XL_C) * INV_X);
        v_if (val < sfpi::vConst0) { val = sfpi::vConst0; }
        v_endif;
        dst_reg[d_out * N + i] = val;                // write result to DST tile d_out
    }
}

#endif  // TRISC_MATH
```

**Critical rules for face functions:**
- Each function may use at most 4-5 `vFloat` variables simultaneously. Exceeding the LREG
  (local register) budget causes the compiler to spill into `dst_reg`, corrupting input data
  in faces 2 and 3 (high SETRWC offsets). The symptom is a systematic -1 error in faces 2/3.
- One function per output quantity. Do NOT compute multiple outputs in one function.
- `dst_reg[tile * 32 + group]` is how you address the DST register file.
  `tile` is the DST tile index (0, 1, ...); `group` is the SIMD group within that face (0-7).

---

### Step 3 — Call the LLK entry point

```cpp
// This is the ONLY way to invoke your SFPU face function:
MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
    face_bxl,               // function pointer
    0,                      // arg a = DST tile index for input 1 (pos_x)
    1,                      // arg b = DST tile index for input 2 (sx)
    0,                      // arg c = unused third tile (hardcode 0)
    0,                      // arg d = DST tile index for output (written in-place)
    static_cast<int>(VectorMode::RC)));  // process full 32×32 tile
```

The LLK will call `face_bxl` exactly 4 times (once per 16×16 face), each time with the
correct SETRWC offset. The function arguments (0, 1, 0, 0) are DST tile indices, not
physical addresses — the LLK handles the SETRWC pointer adjustment internally.

---

### Step 4 — kernel_main: acquire → copy → SFPU → pack (4-pass pattern)

```cpp
void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_px  = tt::CBIndex::c_0;   // input circular buffers
    constexpr auto cb_sx  = tt::CBIndex::c_2;
    constexpr auto cb_bxl = tt::CBIndex::c_16;  // output circular buffer

    init_sfpu(cb_px, cb_bxl);  // initialise SFPU pipeline once

    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        cb_wait_front(cb_px, 1);   // wait for reader to fill input CB
        cb_wait_front(cb_sx, 1);

        // ── Acquire DST, copy inputs, compute, pack output ──
        tile_regs_acquire();
        copy_tile_init(cb_px);  copy_tile(cb_px, 0, 0);  // CB c_0, tile-in-CB 0 → DST[0]
        copy_tile_init(cb_sx);  copy_tile(cb_sx, 0, 1);  // CB c_2, tile-in-CB 0 → DST[1]

        MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
            face_bxl, 0, 1, 0, 0, static_cast<int>(VectorMode::RC)));
        // SFPU reads DST[0] and DST[1], writes result back to DST[0]

        tile_regs_commit();

        cb_reserve_back(cb_bxl, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_bxl);
        pack_tile(0, cb_bxl);   // pack DST[0] → output CB c_16
        tile_regs_release();
        cb_push_back(cb_bxl, 1);

        cb_pop_front(cb_px, 1);
        cb_pop_front(cb_sx, 1);
    }
}
```

**Why 4 separate passes (one per output) instead of one big pass:**
The DST register file with `fp32_dest_acc_en=true` holds at most 4 tiles (0–3).
A design that loads 4 inputs into DST[0–3] and writes 4 outputs to DST[4–7] puts
the outputs out of range. The in-place 2-tile pattern (load 2 inputs, overwrite DST[0]
with the output) stays within DST[0–1] and is safe.

---

## Host-Side Configuration — The Critical Fix

**The single most important thing for FP32 SFPU kernels:**

```cpp
// WRONG — this silently downcasts Float32 → TF32 (10-bit mantissa) in the UNPACKERB
ComputeConfig{ .fp32_dest_acc_en = true, ... }

// CORRECT — forces the UNPACKERB to preserve all 23 mantissa bits
std::vector<UnpackToDestMode> unpack_modes(32, UnpackToDestMode::Default);
for (int i = 0; i < 4; ++i)  // indices 0-3 = input CBs (c_0, c_1, c_2, c_3)
    unpack_modes[i] = UnpackToDestMode::UnpackToDestFp32;

ComputeConfig{
    .fp32_dest_acc_en    = true,
    .unpack_to_dest_mode = unpack_modes,   // ← without this, FP32 is silently TF32
    .math_approx_mode    = false,
}
```

**Why this matters:** `UnpackToDestMode::Default` is documented as "enables all dataformats
EXCEPT Float32 to be unpacked into Dest." Even if you declare the CB as `Float32`, the
UNPACKERB hardware converts the data to TF32 (10 mantissa bits) when loading into the
DST register file. For values ≥ 1024, TF32 precision = 1.0 — only integers are
representable, destroying all fractional information. This caused a systematic -1 error
in bin index calculations for faces 2 and 3 (which process larger cell position values).

The fix: `UnpackToDestFp32` bypasses the SrcA/SrcB intermediate registers and copies
Float32 data directly into DST, preserving all 23 mantissa bits.

---

## JIT #defines — How to Pass Constants from Host to SFPU Kernel

Geometry constants (bin sizes, grid dimensions) are injected at JIT compile time:

```cpp
// Host side (in bin_index_scatter.cpp):
auto to_float_literal = [](float v) -> std::string {
    std::string s = fmt::format("{:.9g}", v);
    if (s.find('.') == std::string::npos &&
        s.find('e') == std::string::npos &&
        s.find('n') == std::string::npos) s += ".0";
    return s + "f";
};

ComputeConfig{
    .defines = {
        {"BIS_BIN_SIZE_X_F", to_float_literal(BIN_SIZE_X)},   // e.g. "1.46484375f"
        {"BIS_INV_BIN_X_F",  to_float_literal(INV_BIN_X)},
        {"BIS_NUM_BINS_X",   std::to_string(NUM_BINS_X)},
    }
}

// Kernel side:
#ifndef BIS_BIN_SIZE_X_F
#define BIS_BIN_SIZE_X_F 3.90625f   // fallback default
#endif
static constexpr float BSZ_X = BIS_BIN_SIZE_X_F;
```

**Critical:** Always use `to_float_literal` to format floats. Without it, a value like
`3.90625` (no `f` suffix) is a `double` literal. When used in a `vFloat(...)` constructor
this causes: `error: unable to find numeric literal operator 'operator""f'`.

**Critical:** JIT defines are only seen by the SFPU kernel at runtime. CMake compile
definitions on the host binary target do NOT reach the SFPU kernel. Pass all
kernel-side defines through `ComputeConfig::defines`.

---

## Circular Buffer Setup for FP32

```cpp
constexpr uint32_t TILE_ELEMS = 32 * 32;            // 1024 elements per tile
constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4;     // 4096 bytes (Float32)
constexpr uint32_t TILES_PER_CB = 2;                // double-buffer

auto make_fp32_cb = [&](tt::CBIndex idx) {
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(TILES_PER_CB * TILE_BYTES, {{idx, tt::DataFormat::Float32}})
            .set_page_size(idx, TILE_BYTES));
};

make_fp32_cb(tt::CBIndex::c_0);   // pos_x input
make_fp32_cb(tt::CBIndex::c_1);   // pos_y input
make_fp32_cb(tt::CBIndex::c_2);   // sx input
make_fp32_cb(tt::CBIndex::c_3);   // sy input
make_fp32_cb(tt::CBIndex::c_16);  // bxl output
make_fp32_cb(tt::CBIndex::c_17);  // bxh output
make_fp32_cb(tt::CBIndex::c_18);  // byl output
make_fp32_cb(tt::CBIndex::c_19);  // byh output
```

---

## SFPI Language Quick Reference (what works in SFPU kernels)

| Construct | What it does |
|---|---|
| `vFloat x = dst_reg[tile * 32 + i]` | Read float from DST tile `tile`, SIMD group `i` |
| `dst_reg[tile * 32 + i] = x` | Write float to DST tile `tile`, SIMD group `i` |
| `vFloat y = x + 1.0f` | FP32 add |
| `vFloat y = x * 0.5f` | FP32 multiply |
| `vFloat y = x - z` | FP32 subtract |
| `v_if (x < threshold) { ... } v_endif` | Predicated (masked) execution — all 32 lanes |
| `sfpi::vConst0` | Hardware constant: 0.0f |
| `vFloat(literal)` | Load scalar constant into all 32 lanes |
| `for (uint32_t i = 0; i < 8; ++i)` | Iterate over 8 SIMD groups of 32 lanes in a face |

**What does NOT exist in SFPI:**
- No division (`/`) — use multiply by reciprocal
- No `sqrt`, `exp`, `log` in raw SFPI — use LLK math ops
- No integer types (`int`, `uint32_t`) as SIMD values — only `vFloat`, `vInt`, `vUInt`

---

## Verified Facts (empirically confirmed, not hypotheses)

1. **Tile element ordering is linear.** Physical element j in a Float32 tile maps to
   logical cell j. No odd/even swap. No face permutation. Strict linear order confirmed
   by passthrough test with pos_x[i] = i * BSZ_X — every output matched exactly.

2. **The `sfpu_floor` trick is correct.** `(x + 2^23) - 2^23; if (r > x) r -= 1`
   produces exact `floor(x)` for all values tested (0 to 2048).

3. **LREG budget is ~5 vFloat variables per face function.** Confirmed by compiler
   behaviour: 8+ vFloat variables in one function causes spills into dst_reg, producing
   corrupted values in faces 2 and 3 specifically (not faces 0 and 1, because spills only
   alias the high SETRWC offsets).

4. **UnpackToDestMode::Default silently downcasts Float32 → TF32.** Confirmed by
   passthrough test with large values (pos_x[i] = i * 1465): output was always the integer
   floor of the input. Fixed by UnpackToDestFp32.

5. **JIT kernel cache reuses stale builds.** TT Metal hashes source + defines. Changing
   source without changing the hash (or defines) reuses the cached .elf. Force
   recompilation with TT_METAL_CLEAR_KERNEL_CACHE=1.

6. **CMake defines do NOT reach JIT-compiled kernels.** Only `ComputeConfig::defines`
   (the kernel_defines map) reaches the SFPU kernel at JIT compile time.

---

## Full Verified Test Results

```
=== Accuracy: SFPU bin indices vs exact DREAMPlace CPU reference ===
  Cells compared  : 500000
  Total mismatches: 0  (0.0000%)
  Per-field : bxl=0  bxh=0  byl=0  byh=0

  Cell 0: pos=(404.839,1493.248) sz=(9.879,4.414)
          bxl=276/276  bxh=284/284  byl=1019/1019  byh=1023/1023
  Cell 1: pos=(2786.625,315.391) sz=(5.456,7.476)
          bxl=1902/1902  bxh=1907/1907  byl=215/215  byh=221/221

Device kernel time: 0.165 ms  (56 cores, 489 tiles, 500K cells)
```

---

## File Index

| File | Description |
|---|---|
| `bin_index_compute.cpp` | Production SFPU kernel — DREAMPlace bin-index formula, FP32, verified correct |
| `passthrough_compute.cpp` | Diagnostic SFPU kernel — verbatim tile copy (+ optional floor on ch0) |
| `SFPU_GUIDE.md` | This file — how to use the SFPU, the exact patterns, all critical gotchas |
| `SESSION_NOTES.md` | Full session log — every bug found, every fix applied, every observation verified |
