// SPDX-License-Identifier: Apache-2.0
//
// V16 Phase 1.0 — DST face-layout sanity probe (TRISC compute).
//
// Goal: empirically determine the mapping between an SFPU `dst_reg[i]` slot
// and a (face, row_pair_within_face) position in the packed L1 tile.
//
// Method: write a unique value to each of the 32 dst_reg slots — slot i gets
// the constant (i + 1.0f). Pack out to CB_OX (bf16). BRISC then dumps the
// raw L1 tile to DRAM where the host can read it and decode the layout by
// checking which face_idx() positions hold which value.
//
// Runtime args: none.
//
// Output (via the paired BRISC): one 32x32 bf16 tile to DRAM, in the native
// face layout that pack_tile produces.

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

#ifdef TRISC_MATH

// Write each slot a unique constant. Slot i gets bf16(i + 1.0f) once the
// fp32 DST value (broadcast across 32 lanes) is downcast by pack_tile.
inline void sfpu_probe_fill_uniform() {
    // Slot 0 deliberately gets 0.0f so an absent / unwritten lane is also
    // visible as 0 in the output; downstream we compare against (i+1.0f).
    sfpi::dst_reg[0]  = sfpi::vFloat(1.0f);
    sfpi::dst_reg[1]  = sfpi::vFloat(2.0f);
    sfpi::dst_reg[2]  = sfpi::vFloat(3.0f);
    sfpi::dst_reg[3]  = sfpi::vFloat(4.0f);
    sfpi::dst_reg[4]  = sfpi::vFloat(5.0f);
    sfpi::dst_reg[5]  = sfpi::vFloat(6.0f);
    sfpi::dst_reg[6]  = sfpi::vFloat(7.0f);
    sfpi::dst_reg[7]  = sfpi::vFloat(8.0f);
    sfpi::dst_reg[8]  = sfpi::vFloat(9.0f);
    sfpi::dst_reg[9]  = sfpi::vFloat(10.0f);
    sfpi::dst_reg[10] = sfpi::vFloat(11.0f);
    sfpi::dst_reg[11] = sfpi::vFloat(12.0f);
    sfpi::dst_reg[12] = sfpi::vFloat(13.0f);
    sfpi::dst_reg[13] = sfpi::vFloat(14.0f);
    sfpi::dst_reg[14] = sfpi::vFloat(15.0f);
    sfpi::dst_reg[15] = sfpi::vFloat(16.0f);
    sfpi::dst_reg[16] = sfpi::vFloat(17.0f);
    sfpi::dst_reg[17] = sfpi::vFloat(18.0f);
    sfpi::dst_reg[18] = sfpi::vFloat(19.0f);
    sfpi::dst_reg[19] = sfpi::vFloat(20.0f);
    sfpi::dst_reg[20] = sfpi::vFloat(21.0f);
    sfpi::dst_reg[21] = sfpi::vFloat(22.0f);
    sfpi::dst_reg[22] = sfpi::vFloat(23.0f);
    sfpi::dst_reg[23] = sfpi::vFloat(24.0f);
    sfpi::dst_reg[24] = sfpi::vFloat(25.0f);
    sfpi::dst_reg[25] = sfpi::vFloat(26.0f);
    sfpi::dst_reg[26] = sfpi::vFloat(27.0f);
    sfpi::dst_reg[27] = sfpi::vFloat(28.0f);
    sfpi::dst_reg[28] = sfpi::vFloat(29.0f);
    sfpi::dst_reg[29] = sfpi::vFloat(30.0f);
    sfpi::dst_reg[30] = sfpi::vFloat(31.0f);
    sfpi::dst_reg[31] = sfpi::vFloat(32.0f);
}

// Probe 2: write each lane its own (slot*32 + lane_id) value. Reveals
// within-slot lane ordering.
//
// Uses vConstTileId which exposes [0, 2, 4, ..., 62] across the 32 lanes.
// Dividing by 2 gives lane_id (0..31). We multiply by a per-slot offset and
// store. Result: dst_reg[i][lane] = i*100 + lane_id (visible in the output).
inline void sfpu_probe_fill_lane_aware() {
    // vConstTileId is [0, 2, 4, ..., 62]. Convert to vFloat via the standard
    // int32_to_float helper (RoundMode::None to avoid stochastic rounding
    // artifacts since we're producing small exactly-representable values).
    sfpi::vInt   tid_v   = sfpi::vConstTileId;
    sfpi::vFloat tid_f   = sfpi::int32_to_float(tid_v, sfpi::RoundMode::NearestEven);
    sfpi::vFloat lane_f  = tid_f * 0.5f;  // [0, 1, 2, ..., 31]
    // dst[i] takes values in [i*64, i*64+31]. Safe under bf16 truncation:
    // 64 ≤ 2^7 ≤ ulp(slot_base) at i ≥ 4 → lane_f's 5 bits survive in bf16
    // mantissa for i*64 < 256. For i ≥ 4, lane_f's lower bits get truncated
    // — readable but coarsened. Keep the test focused on slot 0 first.
    for (uint32_t i = 0; i < 32u; ++i) {
        sfpi::dst_reg[i] = sfpi::vFloat((float)(i * 64)) + lane_f;
    }
}

#endif  // TRISC_MATH

void kernel_main() {
    constexpr auto cb_out = tt::CBIndex::c_16;
    init_sfpu(cb_out, cb_out);

    const uint32_t probe_mode = get_arg_val<uint32_t>(0);

    tile_regs_acquire();
    if (probe_mode == 0u) {
        MATH(( sfpu_probe_fill_uniform() ));
    } else {
        MATH(( sfpu_probe_fill_lane_aware() ));
    }
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
