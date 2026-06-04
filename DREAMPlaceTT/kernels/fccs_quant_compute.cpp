// SPDX-License-Identifier: Apache-2.0
//
// FCCS SFPU field-quantize — COMPUTE (TRISC). Per tile: dst = (int32)(field · 2^FS)
// using the SFPU (hardware float): copy float tile → mul_unary by 2^FS → typecast
// Float32→Int32 → pack int32. This replaces the single-core soft-float quantize
// (49 ms @512 / ~9.8 ms @2048) with hardware-float across all cores (~0.2 ms).
//
// CBs: c_0 Float32 in, c_16 Int32 out.
// Args: 0 my_core 1 tpc(tiles/core) 2 total_tiles 3 fscale_bits (fp32 2^FS as uint32)

#include <cstdint>
#if __has_include("api/compute/common.h")
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/compute_kernel_api.h"
#else
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"
#include "compute_kernel_api.h"
#endif

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t tpc     = get_arg_val<uint32_t>(1);
    const uint32_t total   = get_arg_val<uint32_t>(2);
    const uint32_t fscale  = get_arg_val<uint32_t>(3);   // fp32 bits of 2^FS
    constexpr auto CB_IN = tt::CBIndex::c_0, CB_OUT = tt::CBIndex::c_16;
    constexpr uint32_t F32 = (uint32_t)DataFormat::Float32;
    constexpr uint32_t I32 = (uint32_t)DataFormat::Int32;

    if (my_core * tpc >= total) return;
    uint32_t lo = my_core * tpc;
    uint32_t hi = lo + tpc; if (hi > total) hi = total;
    uint32_t n = hi - lo;

    init_sfpu(CB_IN, CB_OUT);
    binop_with_scalar_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(CB_IN, 1);
        tile_regs_acquire();
        copy_tile_init(CB_IN);
        copy_tile(CB_IN, 0, 0);
        mul_unary_tile(0, fscale);                  // dst[0] *= 2^FS
        typecast_tile_init<F32, I32>();
        typecast_tile<F32, I32>(0);                 // float → int32
        tile_regs_commit();
        cb_reserve_back(CB_OUT, 1);
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_IN, 1);
    }
}
