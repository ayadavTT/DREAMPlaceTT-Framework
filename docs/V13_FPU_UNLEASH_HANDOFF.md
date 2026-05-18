# V13 FPU-Throughput Handoff — Custom Paired Unpack MOP

**Status as of 2026-05-17 (end of session).** Owner of next iteration: TBA.

**Goal of next iteration:** **bring V13 gather from 25.83 ms median → ≤ 5 ms median** on adaptec1_512 by writing a custom paired-unpack LLK MOP that batches N HW unpacks (one srcA + one srcB pair per iteration) in a single MOP dispatch. **Target: gather faster than CPU 16T density (4.74 ms) and within 2× of V11 (2.82 ms).** This is the canonical "unleash the FPU" path — the FPU's per-call wrapper cost is ~99 % software overhead, and the wrapper sits on the UNPACK TRISC, not the MATH TRISC.

This document is self-contained for that one goal. Read it before touching LLK code.

---

## 0. Where the V13 work stands today

V13 (gather_mode `v13_fpu` in the server) is fully wired into the DREAMPlace integration and converges correctly:

| metric (adaptec1_512)              | V13 today | V11 baseline | CPU 16T density |
|------------------------------------|----------:|-------------:|----------------:|
| HPWL (M)                            | 70.35     | 70.33        | 70.25           |
| overflow                            | 0.069     | 0.069        | 0.069           |
| n_ep_calls                          | 630       | 628          | 632             |
| scatter median (ms)                 | 1.38      | 1.95         | —               |
| **gather median (ms)**              | **25.83** | **2.82**     | —               |
| sc+ga total (ms)                    | 27.21     | 4.77         | 4.74            |
| wall-time (s)                       | 42.0      | 30.3         | ~8              |

V13 is **~9× slower than V11** at grid 512 because of per-call FPU-wrapper overhead. This handoff describes the fix.

### Files that V13 currently uses (read these before changing anything)

| Path | Role |
|---|---|
| `kernels/v13_scatter_brisc.cpp` / `v13_scatter_ncrisc.cpp` | V13 scatter — emits 40-byte tile-records (one per touched tile, ≤4 per cell) |
| `kernels/v13_accum_brisc_mt.cpp` | BRISC accum — reads records, filters by tile_id, packs OX/OY into CB tiles. Already has OPT-1 staging (PUSH_BATCH=128, sentinel padding) |
| `kernels/v13_accum_compute_mt.cpp` | TRISC compute kernel — currently uses our custom kaccum LLK (Strategy B, see §3) |
| `kernels/v13_accum_ncrisc_void.cpp` | NCRISC stub — currently idle, available to use |
| **`kernels/v13_llk_math_kaccum.h`** | **NEW (this session)** — custom math LLK that bypasses matmul_tiles math wrapper via direct `lltt::replay()`. See §3 |
| `host/density_scatter_ttnn_server_host.cpp` | Server — V13 dispatch, init_density fold, CB config (CB_OXY_DEPTH=8, N_INFLIGHT=4) |

---

## 1. The bottleneck — proved with Tracy data

**Where the time goes** (Tracy zone times from `results/v13_stratB_final/profile_log_device.csv`, per owned tile, mean):

| zone | RISC | mean per tile (µs) |
|---|---|---:|
| V13A-TILE (BRISC body) | BRISC | 825 |
| **V13C-TILE (TRISC body, all 3 TRISCs)** | **T0 / T1 / T2** | **1220** |
| V13C-MATMUL-LOOP (inner matmul loop, math side) | T1 | 1220 |
| V13C-PACK (DST→bf16) | T2 | 0.08 |
| V13A-DENSITY-WRITE (BRISC write to DRAM) | BRISC | 16 |

**Critical observation:** all three TRISCs (T0=Unpack, T1=Math, T2=Pack) report identical 1220 µs per tile — they're locked together via the CB-mediated handshakes. The system is bound by the **slowest TRISC** path. With ~50 matmul calls per owned tile in the steady state, that's **~24 µs per matmul wall-clock**.

### What the math side actually does per call

`matmul_tiles(CB_OX, CB_OY, 0, 0, 0)` on TRISC T1 is just **2 RISC-V instructions**:

```cpp
// llk_math_matmul -> _llk_math_matmul_(dst_index=0, ct_dim=1, rt_dim=1)
math::set_dst_write_addr<DstTileShape::Tile32x32, ...>(0);  // 1 TT_SETC16 instruction
ckernel_template::run();                                      // 1 TTI_MOP(1, 0, 0) instruction
```

The actual FPU work (16 TTI_MVMUL HW instructions) runs in ~256 cycles ≈ **190 ns**. So math busy-time per call is **<0.5 µs**. The other **~23.5 µs is math waiting on srcA/srcB DVALID from the unpacker**.

### What the unpack side actually does per call

`llk_unpack_AB_matmul(CB_OX, CB_OY, idx_a, idx_b, 1, 1, 1)` on TRISC T0 does, per call (see `tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:251-346`):

1. `wait_for_next_context(2)` — HW handshake for unpack context machine
2. `_llk_unpack_configure_addresses_(addr_a, addr_b, cfg)` — **WRCFG x2** to update SrcA/SrcB L1 base addresses for current CB tile
3. `semaphore_post(UNPACK_SYNC)` — semaphore inc
4. `TTI_STALLWAIT(STALL_UNPACK, TRISC_CFG)` — stall the unpacker until previous CFG writes have drained on the threaded coprocessor (THCON)
5. `TTI_UNPACR(SrcB, ..., 1 Set Dvalid)` — issue srcB unpack
6. `TT_MOP(0, ct_dim-1, ...)` — inner MOP that loads srcA (per-iteration: TTI_UNPACR(SrcA) + RDCFG/ADDDMAREG/STALLWAIT/WRCFG to advance address + NOP)
7. `t6_semaphore_get(UNPACK_SYNC)` — semaphore get
8. `switch_config_context()` — context machine bookkeeping

The dominant per-call costs are the **STALLWAIT cycles** (waiting for the threaded coprocessor to drain WRCFGs) and the **THCON write latency** (each WRCFG is 2 cycles + has to commit through THCON). Each STALLWAIT can take 50-200 cycles depending on the state.

**Per call total ≈ 24 µs.** With 50 matmul calls per owned tile, that's 1200 µs — matches measured V13C-TILE = 1220 µs.

### Why "make math faster" didn't help

I implemented Strategy B (custom math LLK at `kernels/v13_llk_math_kaccum.h`) that uses `lltt::replay()` directly, bypassing `matmul_tiles`' math wrapper. Convergence preserved (HPWL=70.35M). **Speedup: 0×.** Profile confirms math was already <0.5 µs/call; bypassing a 0.5 µs wrapper saves nothing measurable.

---

## 2. Why ct_dim=N batched unpack via the standard API can't work

The intuitive next step — batch the unpacks via `llk_unpack_AB_matmul(... ct_dim=N)` so that one call drives N HW unpacks via the unpack MOP — **deadlocks at warmup**. I tried this twice (reset device between attempts) — both times the server hung after the "JIT compiling kernels..." log line.

### Root cause

`llk_unpack_AB_matmul` has **reuse semantics**: when `ct_dim ≥ rt_dim` (`reuse_a=true`), the unpack MOP issues:
- **1 explicit `TTI_UNPACR(SrcB)` outside the MOP** (one SrcB load per batch, shared)
- **N MOP iterations**, each loading a new SrcA tile

Net: 1 SrcB + N SrcA loads per batched call. SrcB is held constant; SrcA cycles.

This matches the standard matmul semantic: `DST[i] = SrcB × SrcA[i]` for i=0..N-1 — **same SrcB, different SrcA**.

V13's data pattern is **fundamentally incompatible**: each K-stripe record k has its **own unique** OX[k] AND its **own unique** OY[k]. We have N independent (in0, in1) matmul-pair operations to accumulate into one DST tile.

The deadlock mechanism:
1. UNPACK loads SrcB once (= OX[0]) + N SrcA loads (= OY[0..N-1])
2. MATH kaccum issues replay 1 → consumes srcA[0] + srcB[0] → writes DST[0]
3. Kaccum's `TTI_SETRWC(CLR_AB)` clears SrcB DVALID
4. Kaccum issues replay 2 → blocks waiting for SrcB DVALID
5. UNPACK never reloads SrcB (loaded only 1 in the batch)
6. **Deadlock.** Server times out at the 606s wall-clock limit; container often dies after.

If we change kaccum to clear only A (matching reuse_a semantics), no deadlock — but math uses OX[0] for **all** N matmuls instead of OX[0..N-1]. Wrong density → HPWL diverges within ~5 iterations.

### The standard matmul API can NOT batch our pattern

`matmul_block(ct_dim, rt_dim, kt_dim)` has the same constraint:
- `ct_dim` outputs: each shares one operand
- `rt_dim` outputs: each shares the other operand
- `kt_dim`: only used for unpack address stride (not HW-batched at all — production matmul iterates K in software)

**To genuinely batch N independent (srcA, srcB) pair loads in HW, we MUST write a custom unpack MOP.**

---

## 3. Strategy B infrastructure (already in place — reusable building block)

In this session I implemented Strategy B's math side. It's in the working tree, validated, ready to compose with a custom unpack.

### `kernels/v13_llk_math_kaccum.h` (~90 LOC)

```cpp
#ifdef TRISC_MATH
#include "experimental/llk_math_matmul_custom_no_mop.h"

namespace v13 {

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_kaccum_init_() {
    matmul_validate_no_mop_contract();
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(/*transpose*/ false);
    matmul_configure_mop_custom<math_fidelity>(/*ct_dim*/ 1, /*rt_dim*/ 1);  // loads 16-MVMUL replay
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
}

template <MathFidelity math_fidelity>
inline void _llk_math_matmul_kaccum_(std::uint32_t dst_index, std::uint32_t k_batch) {
    constexpr int num_phases = get_math_num_fidelity_phases(math_fidelity);  // 3 for HiFi4
    constexpr bool high_fi   = math_fidelity != MathFidelity::LoFi;

    // Set dst write address ONCE — all k_batch replays accumulate here.
    // ADDR_MOD_5 at the end of each replay resets dest *pointer* (clr=1)
    // but not values, so consecutive replays sum into DST[dst_index].
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    for (std::uint32_t k = 0; k < k_batch; ++k) {
        if constexpr (high_fi) {
            for (int p = 0; p < num_phases; ++p)
                lltt::replay(ckernel::math::replay_buf_offset, /*len*/ 16);
        } else {
            lltt::replay(ckernel::math::replay_buf_offset, 16);
        }
        // Free srcA + srcB DVALID so unpack can push the next K-stripe.
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
}

}  // namespace v13
#endif
```

### `kernels/v13_accum_compute_mt.cpp` (current state)

Uses kaccum on the math side, **per-tile unpacks** on the unpack side:

```cpp
mm_init(CB_OX, CB_OY, CB_DENSE);
MATH(( v13::_llk_math_matmul_kaccum_init_<MathFidelity::HiFi4>() ));

for (uint32_t t = 0; t < n_owned_tiles; ++t) {
    tile_regs_acquire();
    while (true) {
        cb_wait_front(CB_OX, N_INFLIGHT);
        if (read_tile_value(CB_OX, 0, 0) == SENTINEL) { cb_pop_front(CB_OX, N_INFLIGHT); break; }
        cb_wait_front(CB_OY, N_INFLIGHT);

        // N_INFLIGHT individual unpack calls — each ~24 µs (this is what to replace)
        UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 0u, 0u, 1u, 1u, 1u) ));
        UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 1u, 1u, 1u, 1u, 1u) ));
        UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 2u, 2u, 1u, 1u, 1u) ));
        UNPACK(( llk_unpack_AB_matmul(CB_OX, CB_OY, 3u, 3u, 1u, 1u, 1u) ));

        MATH(( v13::_llk_math_matmul_kaccum_<MathFidelity::HiFi4>(0u, N_INFLIGHT) ));

        cb_pop_front(CB_OX, N_INFLIGHT);
        cb_pop_front(CB_OY, N_INFLIGHT);
    }
    tile_regs_commit();
    // ... pack to CB_DENSE ...
}
```

**The 4 separate UNPACK calls are what we need to replace.** Math side stays untouched.

---

## 4. The actual fix — custom paired unpack LLK design

### Approach

Write a custom unpack LLK whose **MOP template loads BOTH srcA AND srcB per iteration**. One `lltt::replay()` (or one `TT_MOP`) trigger from RISC-V dispatches N pair-loads in HW without RISC-V intervention.

### Reference: deepseek_v3_b1 custom MM unpack

**THE template to copy from**:
`tt-metal/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_custom_mm.h`

This file is the production-tested example of a custom unpack MOP. It's parameterized for `in0` shape `[{1,2,4,8}, 32]` (partial-face) so the exact iter_insns body isn't a drop-in for our 32×32 case, but the **MOP-programming pattern** is exactly what we need. Read carefully:

- **Lines 29-68** (`_llk_unpack_AB_custom_mm_iter_insns`) — the per-iter HW instruction sequence. For our V13 case we need the equivalent for full 32×32 bf16 tiles: one TTI_UNPACR for srcA + one TTI_UNPACR for srcB + address advances for both.
- **Lines 70-171** (`_llk_unpack_AB_custom_mm_mop_config_`) — programs `ckernel_unpack_template` with the iter sequence as a replay segment.
- **Lines 173-195** (`_llk_unpack_AB_custom_mm_init_`) — full init flow.
- **Lines 197-230** (`_llk_unpack_AB_custom_mm_run_`) — per-batch run; this is the function the kernel calls.

The standard `_llk_unpack_AB_matmul_mop_config_` at `tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:22-186` is also a reference — it shows the MOP template for the standard `reuse_a` case (1 SrcB + N SrcA per iter). We're modifying it to do 1 SrcA + 1 SrcB per iter.

### Detailed design — `kernels/v13_llk_unpack_paired.h` (NEW)

```cpp
// kernels/v13_llk_unpack_paired.h
// V13 paired-unpack LLK — loads N (srcA, srcB) tile pairs in one HW MOP
// dispatch. Per iteration loads BOTH operands; consecutive iterations
// advance both L1 addresses by tile_size_{a,b}.

#pragma once
#ifdef TRISC_UNPACK

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace v13 {

inline void _v13_unpack_paired_iter_insns() {
    // Per MOP iter: load 1 SrcA tile (=in1=OY) + 1 SrcB tile (=in0=OX) +
    // advance both L1 addresses by one tile.
    //
    // Bit-mask 0b00000000 = all four faces in the tile; the third arg `1`
    // = "last unpack, set DVALID" (so math can consume).
    //
    // The standard reuse_a path uses these exact UNPACR forms (see
    // llk_unpack_AB_matmul.h:318 for SrcB and the MOP body lines 39-100 for
    // SrcA). We simply combine them.

    // Load srcA (one full 32×32 face-sequenced tile) + set DVALID.
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/,
               p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    // Load srcB.
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/,
               p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    // Advance srcA L1 address: read current, add tile_size_a (in TMP_LO),
    // stall for THCON to drain, write back.
    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);

    // Advance srcB L1 address.
    TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_B);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_address_ADDR32);

    TTI_NOP;  // timing — last WRCFG takes 2 cycles
}

inline void _v13_unpack_paired_mop_config_() {
    constexpr std::uint32_t replay_buf_len = 11;  // matches iter_insns instruction count

    load_replay_buf(0, replay_buf_len, [] {
        _v13_unpack_paired_iter_insns();
    });

    const std::uint32_t pair_insn = lltt::replay_insn(0, replay_buf_len);

    // Use unpackB mode (single A0 slot + B slot). Since we're encoding
    // BOTH unpacks in the same replay (A0 = full pair-load), we don't
    // actually need B slot — set B to also use A0 sequence for safety.
    ckernel_unpack_template tmp = ckernel_unpack_template(
        /*unpackB*/  true,    // template includes B slot
        /*halo*/     false,
        /*A0*/ pair_insn,
        /*A1*/ 0, /*A2*/ 0, /*A3*/ 0,
        /*skipA*/ 0,
        /*B*/ pair_insn,
        /*skipB*/ 0
    );
    tmp.program();
    TTI_MOP_CFG(0);  // commit MOP template
}

inline void _v13_unpack_paired_init_(
    uint32_t in0_cb_id, uint32_t in1_cb_id) {
    // Pre-condition: standard mm_init has already configured the unpacker's
    // hw state (formats, face dims, partial_face=false). We just need to
    // program our MOP template and set tile-size GPRs.

    const uint32_t op0_id = get_operand_id(in0_cb_id);  // OX in CB → SrcB
    const uint32_t op1_id = get_operand_id(in1_cb_id);  // OY in CB → SrcA
    const uint32_t ts_a = get_local_cb_interface(op1_id).fifo_page_size;
    const uint32_t ts_b = get_local_cb_interface(op0_id).fifo_page_size;

    // Pre-program TILE_SIZE GPRs (used in iter_insns for address advance).
    TT_SETDMAREG(0, LOWER_HALFWORD(ts_a), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    TT_SETDMAREG(0, LOWER_HALFWORD(ts_b), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));

    _v13_unpack_paired_mop_config_();

    // Reset ADC counters so subsequent unpacks start from face 0.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

inline void _v13_unpack_paired_run_(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t n_pairs) {
    // Per-batch entry point. Sets initial CFG addresses to CB front
    // positions for both srcA and srcB, then fires the MOP for n_pairs
    // iterations.

    const uint32_t op0_id = get_operand_id(in0_cb_id);
    const uint32_t op1_id = get_operand_id(in1_cb_id);
    const uint32_t addr_a = (get_local_cb_interface(op1_id).fifo_rd_ptr - 1);
    const uint32_t addr_b = (get_local_cb_interface(op0_id).fifo_rd_ptr - 1);

    volatile uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = addr_a;
    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_b;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Single MOP dispatch issues n_pairs paired-unpack iterations in HW.
    TT_MOP(0, n_pairs - 1, 0);

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);

    // Reset counters at end of batch.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

}  // namespace v13
#endif
```

### Per-batch kernel-side usage

In `kernels/v13_accum_compute_mt.cpp`, replace the 4-call unpack block:

```cpp
// Init (once at start of kernel_main):
mm_init(CB_OX, CB_OY, CB_DENSE);
UNPACK(( v13::_v13_unpack_paired_init_(CB_OX, CB_OY) ));
MATH(( v13::_llk_math_matmul_kaccum_init_<MathFidelity::HiFi4>() ));

// Per batch (replace the 4 UNPACK lines):
UNPACK(( v13::_v13_unpack_paired_run_(CB_OX, CB_OY, N_INFLIGHT) ));
MATH(( v13::_llk_math_matmul_kaccum_<MathFidelity::HiFi4>(0u, N_INFLIGHT) ));
```

---

## 5. Expected speedup math

Per-batch wall-clock breakdown:

| component | today (4× standard unpack + kaccum) | after paired-MOP |
|---|---:|---:|
| RISC-V issue (unpack) | 4 × ~30 instructions = ~90 ns | ~10 instructions + 1 TT_MOP = ~10 ns |
| WRCFG + STALLWAIT (per call) | 4 × (~10 cycles WRCFG + ~50-200 cycles STALLWAIT) ≈ 400-800 ns × 4 = ~1.6-3.2 µs | once per batch = ~400-800 ns |
| Semaphore acquire/release | 4 × ~30 cycles = ~90 ns | once per batch = ~22 ns |
| HW UNPACR (actual data movement, ~256 cycles per tile) | 4 × 256 cycles = ~750 ns × 2 (A+B) = ~1.5 µs | same (HW-bound) |
| Math wait on DVALID (current dominant cost) | ~96 µs (~24 µs per matmul × 4) | should drop to **<10 µs** because new unpack issues all 4 pairs back-to-back without per-call sync stalls |
| **Total per batch** | **~96 µs** | **~10-15 µs** |

**Per owned tile** (with ~13 batches when records cluster): 96 × 13 = 1248 µs (matches measured 1220 µs/tile) → 13 × 12 = ~155 µs (~8× faster).

**Per gather iter** (max of ~7 cores on critical path): ~155 × 8 / 5 = ~25 ms → **~3-5 ms** (target).

**Realistic target: 5-8 ms gather median.** If we hit 5 ms, that's:
- **5.2× faster than V13 today** (25.83 → 5 ms)
- **Beats CPU 16T density** (4.74 ms)
- **Approaches V11** (2.82 ms) — within 2×

Stretch goal: combine paired-unpack with the **NCRISC reader** optimization (NCRISC is idle in V13; use it to pre-stage DRAM reads while BRISC packs) → potential additional 1.5-2× → **gather 3-4 ms, beats V11**.

---

## 6. Implementation roadmap

### Step 1 — Write `kernels/v13_llk_unpack_paired.h` (~150 LOC)

Use the design in §4 above. Cross-check with:
- `tt-metal/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_custom_mm.h` — production custom MOP example
- `tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:22-186` — standard MOP body for reference of TTI_UNPACR parameters, ADC setup, CFGSHIFTMASK semantics

### Step 2 — Modify `kernels/v13_accum_compute_mt.cpp`

Replace the 4 individual `llk_unpack_AB_matmul` calls per batch with one `v13::_v13_unpack_paired_run_` call. Add `v13::_v13_unpack_paired_init_` to the init flow. Math side already uses `v13::_llk_math_matmul_kaccum_` and stays unchanged.

### Step 3 — Validate via the smoke test FIRST (don't go straight to DREAMPlace)

`host/build/v13_full_smoke_host` driven by `tools/run_v13_iter100.py` exercises V13 with a controlled iter-100 snapshot. Run it first:

```bash
docker exec -w "$(pwd)" bh-38-special-ayadav-for-reservation-75063 \
    bash -c '/opt/venv/bin/python3 tools/run_v13_iter100.py results/v13_dump/pos.iter100.bin 10'
```

Numerically compare the density output to the pre-Strategy-B baseline at `results/v13_dump/density.bin`. **Pass criterion**: rel L2 ≤ 0.5 % (bf16 truncation tolerance). If the smoke fails, the kernel is wrong — debug there, NOT in DREAMPlace (faster iteration).

### Step 4 — Full DREAMPlace run

```bash
CONTAINER=bh-38-special-ayadav-for-reservation-75063 \
CPU_DCT=1 GATHER_MODE=v13_fpu \
RESULTS_DIR=results/v13_paired_unpack_a1_512 \
DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn --container bh-38-special-ayadav-for-reservation-75063 \
    --benchmark benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir results/v13_paired_unpack_a1_512
```

**Pass criterion**: HPWL = 70.35 M ± 0.05, overflow = 0.069 ± 0.001, n_ep_calls = 630.
**Performance target**: gather median ≤ 8 ms (3× faster than 25.83 ms baseline).

### Step 5 — Tracy profile + zone analysis

Re-run with `TT_METAL_DEVICE_PROFILER=1`, process with `tools/v13_profile_summary.py`:

```bash
python3 tools/v13_profile_summary.py results/v13_paired_unpack_a1_512/profile_log_device.csv
```

**Pass criterion**:
- V13C-MATMUL-LOOP drops from 1220 µs/tile to ≤ 300 µs/tile
- V13A-TILE (BRISC) becomes the new bottleneck (~825 µs/tile)

If BRISC is now the bottleneck → follow-up task is to optimize BRISC (move some work to NCRISC, vectorize pack_ox_oy zero-fill, bucket scatter by tile_id).

---

## 7. Known pitfalls — read these before debugging

### Pitfall 1: ADC counter state between batches

`TTI_SETADCZW` and `TTI_SETADCXY` reset the unpacker's ADC counters (face position within tile). Between batches, the CB front advances; the next call must start from face 0 of the new tile. The `_v13_unpack_paired_run_` function resets ADC at the end — verify this matches your iter_insns address-advance pattern.

If you see srcA/srcB getting wrong face data (e.g., DST values are subtly off), the ADC counter is wrong. Debug: insert `TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010)` between each iter_insns to forcibly reset. If it fixes the data, the standard reset path has a gap.

### Pitfall 2: STALLWAIT placement

The standard unpack has `TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)` to wait for CFG writes to commit before issuing UNPACR. If you skip this and the WRCFG hasn't drained, UNPACR reads the OLD address → unpacks wrong data → density wrong.

In `_v13_unpack_paired_run_`, the STALLWAIT after the initial cfg writes is essential. Inside `_v13_unpack_paired_iter_insns`, the `TTI_STALLWAIT(STALL_CFG, THCON)` between RDCFG/WRCFG pairs is essential.

### Pitfall 3: Replay buffer offset collision

Math uses replay slots [16, 31] (ckernel::math::replay_buf_offset = 16). Unpack/SFPU uses [0, 15]. Our paired-unpack MOP must stay in [0, 15] (i.e., `load_replay_buf(0, len, ...)` with `len ≤ 16`).

If iter_insns has more than 16 instructions, you'll silently overwrite into math's slot → next math replay sees garbage → density wrong → HPWL diverges.

My draft has 11 instructions per iter — fits comfortably. If you add more (e.g., extra NOPs for timing), watch the count.

### Pitfall 4: TT_MOP loop_count encoding

`TT_MOP(mop_type, loop_count, zmask)` — the second arg is **count - 1** on Blackhole (verify with the deepseek code: line 219 uses `(kt_dim / 2) - 1`). Our `n_pairs - 1` matches that convention. If you pass `n_pairs` directly, you get one extra iteration → reads off the end of CB → garbage.

### Pitfall 5: ckernel_unpack_template B-slot vs halo mode

The MOP template has two execution modes (see `ckernel_template.h:73-130` comment block):
- **unpackB mode** (`unpackB=true, halo=false`): runs A0 then B per iter. We use this; both A0 and B are our paired-unpack replay segment.
- **halo mode** (`halo=true`): runs A0, A1, A2, A3 per iter.

If you accidentally enable halo mode, MOP runs A0-A3 each iteration → 4× too many unpacks → wrong density.

### Pitfall 6: Container exits / hangs

Custom MOP bugs frequently cause hardware deadlocks. When `density_scatter_ttnn_server` doesn't reach the first "done h2d" line within ~30 s of "JIT compiling kernels...", **the kernel has deadlocked**.

Recovery:
```bash
ps aux | grep density_scatter_ttnn_server | grep -v grep | awk '{print $2}' | xargs -r kill -9
docker exec bh-38-special-ayadav-for-reservation-75063 bash -c "pkill -9 -f density_scatter_ttnn_server"
docker exec bh-38-special-ayadav-for-reservation-75063 tt-smi -r    # device reset, AUTHORIZED for this work
sleep 10
docker exec bh-38-special-ayadav-for-reservation-75063 echo "alive"
```

After reset, clear JIT cache so the kernel re-compiles:
```bash
docker exec bh-38-special-ayadav-for-reservation-75063 bash -c "rm -rf /root/.cache/tt-metal-cache 2>/dev/null"
```

---

## 8. Critical reference files (in priority order to read)

### Must read before writing custom unpack
1. **`kernels/v13_llk_math_kaccum.h`** — Strategy B math LLK (already in tree). Mirror its style for the unpack header.
2. **`tt-metal/models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_custom_mm.h`** — production custom unpack MOP example. Section 4's design is heavily based on this.
3. **`tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`** lines 22-186 — standard MOP body, shows the TTI_UNPACR / RDCFG / WRCFG patterns.
4. **`tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h`** lines 73-130 — `ckernel_unpack_template` class docs.

### Reference HW instructions
5. `tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h` — TTI_UNPACR, TT_OP_MOP, TT_SETC16 macros.
6. `tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_addrmod.h` — ADDR_MOD struct.
7. `tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:29` — `replay_buf_offset` (math=16).
8. `tt-metal/runtime/sfpi/include/lltt.h` — `lltt::record/replay/replay_insn`.

### Build / run / debug infrastructure
9. `scripts/build_server.sh` — rebuild inside container.
10. `tools/run_v13_iter100.py` — kernel-only smoke driver.
11. `tools/v13_profile_summary.py` — process Tracy CSV into zone aggregates.

---

## 9. State of code in the working tree

```
M REPRODUCE.md                                   (modified earlier, not relevant)
M kernels/v13_accum_brisc_mt.cpp                 (OPT-1 staging — keep)
M kernels/v13_accum_compute_mt.cpp               (Strategy B kaccum — keep)
M host/density_scatter_ttnn_server_host.cpp      (V13 dispatch + init_density fix — keep)
?? kernels/v13_llk_math_kaccum.h                 (NEW — Strategy B math LLK — keep)
?? docs/V13_FPU_UNLEASH_HANDOFF.md               (this document)
?? kernels/v13_llk_unpack_paired.h               (TODO — to be created in next session)
```

All listed changes are needed for V13 to converge correctly. Don't revert any of them.

---

## 10. Backup plans if custom paired-unpack is too risky

### Plan B — SFPU outer-product kernel (Strategy C)
Replace FPU matmul with SFPU (T1 vector unit) computing 8×8 outer products directly. Simpler programming model — see `tt-metal/tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp` for `vFloat` SIMD usage. Each record's 8×8 outer product = ~2 SFPU instructions = ~50 ns. Expected speedup similar to paired unpack (5-10×) but doesn't use the FPU matmul.

### Plan C — Reduce matmul call count by record-density packing
Currently K_BATCH=32 records per matmul → ~50 matmuls per owned tile. Pack 2 records per OX column (different bxl_local offsets, non-overlapping rows) → K_BATCH=64 effective → 25 matmuls per tile → halves the unpack overhead total. Doesn't touch LLK. ~2-3 hours of careful BRISC and TRISC pack/unpack changes.

### Plan D — Activate NCRISC as parallel DRAM reader
`kernels/v13_accum_ncrisc_void.cpp` is currently a no-op. Have NCRISC pre-fetch route_buf records for tile T+1 while BRISC processes tile T. Cuts DRAM-read latency from BRISC critical path. ~2-3 hours, doesn't touch FPU/unpack at all.

The user's stated goal is "use FPU at all cost", so Plans B/C/D are fallbacks if the paired-unpack approach hits insurmountable LLK issues.

---

## 11. Quick reference — reproducing the current baseline

```bash
cd /localdev/ayadav/tt-work/TTPort/DREAMPlaceTT-Framework

# Ensure container is up
docker start bh-38-special-ayadav-for-reservation-75063 2>&1 | head -1

# Build (must happen inside container)
docker exec -w "$(pwd)" bh-38-special-ayadav-for-reservation-75063 \
    bash -c 'TT_METAL_HOME=$(pwd)/tt-metal bash scripts/build_server.sh -j 16'

# Single config — measures current V13 Strategy B perf (~25.88 ms median gather)
rm -rf results/v13_repro && mkdir -p results/v13_repro
CONTAINER=bh-38-special-ayadav-for-reservation-75063 \
CPU_DCT=1 GATHER_MODE=v13_fpu \
RESULTS_DIR=results/v13_repro \
timeout 300 DREAMPlace/dp_env/bin/python3 integration/run_dreamplace.py \
    --device scatter_ttnn \
    --container bh-38-special-ayadav-for-reservation-75063 \
    --benchmark benchmarks/configs/sweep_adaptec1_512.json \
    --results-dir results/v13_repro

# Check result
python3 -c "import json; m=json.load(open('results/v13_repro/sweep_adaptec1_512_scatter_ttnn_metrics.json'));
print(f\"hpwl={m['hpwl']} gather_med={m['scatter_ttnn_gather_ms_median']:.2f}\")"

# Expected: hpwl=70349880.0  gather_med=25.88
```

---

## 12. The 30-second TL;DR for the next person

1. V13 today: 25.83 ms gather, ~9× slower than V11. Each `matmul_tiles` call costs ~24 µs wall-clock — almost entirely **math waiting on srcA/srcB DVALID from the unpack TRISC**.
2. Strategy B (math-side bypass via `lltt::replay`) is in tree at `kernels/v13_llk_math_kaccum.h`. Correct but invisible: math wasn't the bottleneck.
3. `ct_dim=N` batched unpack via the standard API deadlocks — `reuse_a` semantics force one operand to be shared, but V13 needs both srcA and srcB to change every K-stripe.
4. **The fix**: write `kernels/v13_llk_unpack_paired.h` — a custom unpack LLK whose MOP template loads BOTH srcA and srcB per iteration. Drop-in design in §4. Reference: deepseek's `llk_unpack_AB_custom_mm.h` (§8.2). Replaces 4 separate unpack calls with 1 `TT_MOP` that runs 4 paired-unpack iterations in HW.
5. **Expected result**: gather 25.83 → ~5 ms median (5× speedup). Beats CPU 16T density. Approaches V11.
6. Test via the kernel-only smoke first (`tools/run_v13_iter100.py`), not DREAMPlace (faster iteration). Reset device aggressively when kernel deadlocks (custom MOP bugs are catastrophic — see §7.6).
7. Math side (`kaccum`) stays — compose unchanged with the new unpack.

Good luck. The FPU is fast. The wrapper is slow. Fix the wrapper.
