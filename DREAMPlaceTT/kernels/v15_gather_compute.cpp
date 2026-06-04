// SPDX-License-Identifier: Apache-2.0
//
// V15 — Single-CB-pair TRISC compute (kaccum). Identical structure to V13's
// v13_accum_compute_mt.cpp — BRISC pushes a stream of N_INFLIGHT-aligned
// OX/OY tiles per owned tile, terminated by a SENTINEL batch. TRISC pops in
// N_INFLIGHT batches and FPU-kaccums into DST[0]; pack to CB_DENSE; release.
//
// V15-specific note: BRISC drains BOTH bucket_b[t] and bucket_n[t] (from the
// NCRISC bucketing pass) into the same CB pair before the sentinel, so a
// single sentinel-bounded run per owned tile covers ALL writers.

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
#include "v13_llk_unpack_paired.h"

constexpr uint32_t CB_OX    = tt::CBIndex::c_0;
constexpr uint32_t CB_OY    = tt::CBIndex::c_1;
constexpr uint32_t CB_DENSE = tt::CBIndex::c_16;

constexpr uint32_t SENTINEL  = 0xFFFFFFFFu;
constexpr uint32_t N_INFLIGHT = 4u;

void kernel_main() {
    const uint32_t n_owned_tiles = get_arg_val<uint32_t>(0);
    if (n_owned_tiles == 0u) return;

    mm_init(CB_OX, CB_OY, CB_DENSE);
    UNPACK(( v13::_v13_unpack_paired_init_() ));
    MATH(( v13::_llk_math_matmul_kaccum_init_<MathFidelity::HiFi4>() ));

    for (uint32_t t = 0u; t < n_owned_tiles; ++t) {
        DeviceZoneScopedN("V15C-TILE");
        tile_regs_acquire();

        {
            DeviceZoneScopedN("V15C-MATMUL-LOOP");
            while (true) {
                {
                    DeviceZoneScopedN("V15C-WAIT-OX");
                    cb_wait_front(CB_OX, N_INFLIGHT);
                }
                uint32_t v = read_tile_value(CB_OX, 0u, 0u);
                if (v == SENTINEL) {
                    cb_pop_front(CB_OX, N_INFLIGHT);
                    break;
                }
                {
                    DeviceZoneScopedN("V15C-WAIT-OY");
                    cb_wait_front(CB_OY, N_INFLIGHT);
                }
                {
                    DeviceZoneScopedN("V15C-UNPACK");
                    UNPACK(( v13::_v13_unpack_paired_run_(CB_OX, CB_OY, N_INFLIGHT) ));
                }
                {
                    DeviceZoneScopedN("V15C-MATH");
                    MATH(( v13::_llk_math_matmul_kaccum_<MathFidelity::HiFi4>(0u, N_INFLIGHT) ));
                }
                {
                    DeviceZoneScopedN("V15C-POP");
                    cb_pop_front(CB_OX, N_INFLIGHT);
                    cb_pop_front(CB_OY, N_INFLIGHT);
                }
            }
        }

        tile_regs_commit();
        {
            DeviceZoneScopedN("V15C-PACK");
            tile_regs_wait();
            cb_reserve_back(CB_DENSE, 1);
            pack_tile(0, CB_DENSE);
            cb_push_back(CB_DENSE, 1);
            tile_regs_release();
        }
    }
}
