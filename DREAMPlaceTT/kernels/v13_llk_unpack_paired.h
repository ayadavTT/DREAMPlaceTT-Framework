// SPDX-License-Identifier: Apache-2.0
//
// V13 paired-unpack custom unpack LLK
//
// Each MOP iteration loads BOTH SrcA AND SrcB (one full 32×32 bf16 tile each)
// in hardware, then advances both L1 base addresses by one tile_size. One
// TT_MOP dispatch covers N (SrcA, SrcB) pair-loads with NO per-pair RISC-V
// STALLWAIT/WRCFG handshake — that handshake is what dominates the 24 µs/call
// wall-clock the standard llk_unpack_AB_matmul pays.
//
// V13's data pattern (each record has a UNIQUE ox + UNIQUE oy) does not fit
// the standard matmul ct_dim batching, which has reuse semantics (one operand
// is loaded once and shared across N outputs). Our MOP body loads both
// operands per iter.
//
// Pre-conditions (caller responsibility):
//   * mm_init(CB_OX, CB_OY, CB_DENSE) must have run on this kernel. mm_init
//     programs the unpacker for full-32×32 bf16 tiles (face dims, ADC X-end,
//     TILE_SIZE_A/B GPRs, KT_DIM GPR). We re-program ONLY the MOP template
//     into replay slot 0; everything else stays.
//   * Math side must have run v13::_llk_math_matmul_kaccum_init_<> which
//     programs the standard 16-MVMUL replay segment at math slot
//     ckernel::math::replay_buf_offset (= 16). Our paired-unpack uses
//     slots [0..10]; no collision.
//   * Before _v13_unpack_paired_run_(n_pairs), the kernel must
//     cb_wait_front(in0_cb, n_pairs) and cb_wait_front(in1_cb, n_pairs).
//
// Operand mapping (matches the standard llk_unpack_AB_matmul wrapper):
//   in0_cb (CB_OX) → SrcB → THCON_SEC1
//   in1_cb (CB_OY) → SrcA → THCON_SEC0

#pragma once

#ifdef TRISC_UNPACK

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace v13 {

// 6 HW instructions per iteration. CFGSHIFTMASK-based atomic address
// advance — pattern from tt-llk/.../llk_unpack_tilize.h:282-294. No
// RDCFG/ADDD/STALLWAIT/WRCFG round-trip — instead one CFGSHIFTMASK opcode
// reads `SCRATCH_SECn_val` and ADDs it to the target THCON register
// atomically in 2 cycles. Caller must pre-load SCRATCH_SEC0_val=TILE_SIZE_A
// and SCRATCH_SEC1_val=TILE_SIZE_B once in _init_.
inline void _v13_unpack_paired_iter_insns() {
    // (1) Unpack SrcA, full 32×32 bf16 tile, SetDvalid=1 so MATH can consume.
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*SetDvalid*/,
               p_unpacr::RAREFYB_DISABLE, 0, 0 /*ContextIdInc*/, 0, 0, 1);

    // (2) Unpack SrcB. Same shape.
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*SetDvalid*/,
               p_unpacr::RAREFYB_DISABLE, 0, 0 /*ContextIdInc*/, 0, 0, 1);

    // (3) Atomic: THCON_SEC0_REG3_Base_address += SCRATCH_SEC0_val (= TILE_SIZE_A).
    // Args: disable_mask=1, operation=0b011 (ADD), mask_width=31 (full 32-bit),
    //       right_cshift=0, scratch_sel=0b11 (use scratch matching target's SEC).
    TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);

    // (4) NOP — CFGSHIFTMASK is 2-cycle; let it commit before next instr.
    TTI_NOP;

    // (5) Atomic: THCON_SEC1_REG3_Base_address += SCRATCH_SEC1_val (= TILE_SIZE_B).
    TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC1_REG3_Base_address_ADDR32);

    // (6) NOP for the second CFGSHIFTMASK commit before next iter's UNPACR.
    TTI_NOP;
}

inline void _v13_unpack_paired_mop_config_() {
    constexpr std::uint32_t replay_buf_len = 6;
    static_assert(replay_buf_len <= 16,
                  "unpack replay buf slots [0..15] only; math owns [16..31]");

    // Record the 11-instruction iter sequence into replay slot 0.
    load_replay_buf(0, replay_buf_len, [] {
        _v13_unpack_paired_iter_insns();
    });

    const std::uint32_t pair_insn = lltt::replay_insn(0, replay_buf_len);

    // unpackB=false → no B-slot dispatch per MOP iter.
    // halo=false    → no A1/A2/A3 dispatch.
    // Per iter: run A0 once = one paired-unpack body. TT_MOP(0, n-1, 0)
    // gives n total executions.
    ckernel_unpack_template tmp = ckernel_unpack_template(
        /*unpackB*/    false,
        /*unpackHalo*/ false,
        /*A0_instr*/   pair_insn,
        /*A1_instr*/   0,
        /*A2_instr*/   0,
        /*A3_instr*/   0,
        /*skipA*/      0,
        /*B_instr*/    0,
        /*skipB*/      0);

    tmp.program();
    TTI_MOP_CFG(0);  // Commit template (deepseek pattern; redundant for some
                     // paths but cheap and safe).
}

// Call ONCE per kernel, after mm_init(CB_OX, CB_OY, CB_DENSE).
inline void _v13_unpack_paired_init_() {
    // mm_init already configured: face_r_dim, partial_face=false, ADC X-end
    // for full 32×32 tiles, TILE_SIZE_A/B GPRs, KT_DIM. It also programmed
    // the STANDARD matmul MOP template into the unpack replay slots — we
    // overwrite that with our paired-unpack template here.

    // Pre-load SCRATCH_SEC0_val ← TILE_SIZE_A, SCRATCH_SEC1_val ← TILE_SIZE_B
    // so CFGSHIFTMASK in the MOP body can atomically ADD these to the THCON
    // base address registers without paying STALLWAIT/WRCFG per iter.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TILE_SIZE_A, 0, SCRATCH_SEC0_val_ADDR32);
    TTI_WRCFG(p_gpr_unpack::TILE_SIZE_B, 0, SCRATCH_SEC1_val_ADDR32);
    TTI_NOP;

    _v13_unpack_paired_mop_config_();

    // Reset ADC Z/W and X/Y to face 0 so the first paired-unpack starts at
    // the new base address's first face.
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

// Per-batch entry. Programs the THCON base addresses for SrcA (= CB_OY rd_ptr)
// and SrcB (= CB_OX rd_ptr), then fires the MOP for n_pairs paired-unpacks.
inline void _v13_unpack_paired_run_(
    const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id,
    const std::uint32_t n_pairs) {
    const std::uint32_t op0_id = get_operand_id(in0_cb_id);
    const std::uint32_t op1_id = get_operand_id(in1_cb_id);

    // -1 matches the standard wrapper pattern (deepseek custom_mm:339-340).
    const std::uint32_t addr_a = get_local_cb_interface(op1_id).fifo_rd_ptr - 1;  // SrcA = in1 = CB_OY
    const std::uint32_t addr_b = get_local_cb_interface(op0_id).fifo_rd_ptr - 1;  // SrcB = in0 = CB_OX

    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    // Single-context model (mirrors deepseek custom_mm:222-225). The MOP
    // body advances SEC0/SEC1's context-0 base address registers — must NOT
    // switch_config_context between batches or those advances target dead
    // registers while the unpacker reads stale addresses.
    wait_for_next_context(1);

    // Program SEC0 = SrcA (addr_a) and SEC1 = SrcB (addr_b).
    _llk_unpack_configure_addresses_(addr_a, addr_b, cfg);

    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpack frontend until preceding CFG writes have drained.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Fire n_pairs paired-unpack iterations in HW. loop_count = n_pairs - 1.
    TT_MOP(0, n_pairs - 1, 0);

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Reset config context to 0 so the next call's _llk_unpack_configure_addresses_
    // writes the same SEC0/SEC1 registers our MOP body advances.
    reset_config_context();

    // Reset ADC for the next batch (counters may have wrapped during MOP).
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

}  // namespace v13

#endif  // TRISC_UNPACK
