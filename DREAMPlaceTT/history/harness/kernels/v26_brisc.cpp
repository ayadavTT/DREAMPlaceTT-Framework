// SPDX-License-Identifier: Apache-2.0
//
// V26 EF backward — BRISC direct-L1 corner gather (fp32, NO math). Loads this
// core's field band into L1 once, then per 1024-cell batch fills 8 field tiles
// (fx/fy × 4 corners) + 4 weight tiles by direct L1 loads from the resident
// band. The SFPU (v26_compute) does the fp32 weighted sum.
//
// Cell record (24 B, 6 f32): {bxl, byl_local, w_nw, w_ne, w_sw, w_se}.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t my_core    = get_arg_val<uint32_t>(0);
    const uint32_t ncells     = get_arg_val<uint32_t>(1);
    const uint32_t band_bytes = get_arg_val<uint32_t>(2);
    const uint32_t W          = get_arg_val<uint32_t>(3);
    const uint32_t field_base = get_arg_val<uint32_t>(4);
    const uint32_t cells_base = get_arg_val<uint32_t>(5);
    const uint32_t cells_pg   = get_arg_val<uint32_t>(6);  // whole core's cell page bytes
    const uint32_t n_batches  = get_arg_val<uint32_t>(7);
    const uint32_t chunk_batches = get_arg_val<uint32_t>(8);  // batches per L1 cell-chunk (L1 safety)

    constexpr uint32_t B = 1024;  // cells per tile
    constexpr auto FXNW=tt::CBIndex::c_0, FXNE=tt::CBIndex::c_1, FXSW=tt::CBIndex::c_2, FXSE=tt::CBIndex::c_3;
    constexpr auto FYNW=tt::CBIndex::c_4, FYNE=tt::CBIndex::c_5, FYSW=tt::CBIndex::c_6, FYSE=tt::CBIndex::c_7;
    constexpr auto WNW=tt::CBIndex::c_8, WNE=tt::CBIndex::c_9, WSW=tt::CBIndex::c_10, WSE=tt::CBIndex::c_11;
    constexpr auto CB_FIELD=tt::CBIndex::c_24, CB_CELLS=tt::CBIndex::c_25;

    const InterleavedAddrGen<true> fg = {.bank_base_address=field_base, .page_size=band_bytes};
    const InterleavedAddrGen<true> cg = {.bank_base_address=cells_base, .page_size=cells_pg};
    const uint32_t chunk_bytes = chunk_batches * B * 6u * 4u;  // bytes per cell-chunk

    // ── Field band: resident in L1 (bounded by grid, ≤~442KB @2048) ──
    float* field;
    { DeviceZoneScopedN("V26-LOADBAND");
      cb_reserve_back(CB_FIELD,1); uint32_t l1=get_write_ptr(CB_FIELD);
      noc_async_read(fg.get_noc_addr(my_core), l1, band_bytes); noc_async_read_barrier();
      field=reinterpret_cast<float*>(l1);
    }
    const uint32_t cells_l1 = get_write_ptr(CB_CELLS);  // CB_CELLS sized for ONE chunk only

    // chunk_base byte offset within this core's cell page, advanced per chunk.
    uint32_t chunk_base = 0;
    uint32_t loaded_batches = 0;                 // batches already loaded into the current chunk
    const uint32_t* cellsu = reinterpret_cast<const uint32_t*>(cells_l1);

    for (uint32_t b = 0; b < n_batches; ++b) {
        // (Re)load a cell chunk into L1 when the current one is exhausted —
        // bounds L1 use to chunk_bytes regardless of total cell count.
        if (b % chunk_batches == 0) {
            DeviceZoneScopedN("V26-LOADCHUNK");
            uint32_t this_chunk = chunk_bytes;
            // last chunk may be shorter — clamp to remaining page bytes
            if (chunk_base + this_chunk > cells_pg) this_chunk = cells_pg - chunk_base;
            noc_async_read(cg.get_noc_addr(my_core) + chunk_base, cells_l1, this_chunk);
            noc_async_read_barrier();
            chunk_base += this_chunk;
        }
        const float* cells = reinterpret_cast<const float*>(cells_l1);
        DeviceZoneScopedN("V26-GATHER");
        cb_reserve_back(FXNW,1);cb_reserve_back(FXNE,1);cb_reserve_back(FXSW,1);cb_reserve_back(FXSE,1);
        cb_reserve_back(FYNW,1);cb_reserve_back(FYNE,1);cb_reserve_back(FYSW,1);cb_reserve_back(FYSE,1);
        cb_reserve_back(WNW,1);cb_reserve_back(WNE,1);cb_reserve_back(WSW,1);cb_reserve_back(WSE,1);
        float* fxnw=(float*)get_write_ptr(FXNW); float* fxne=(float*)get_write_ptr(FXNE);
        float* fxsw=(float*)get_write_ptr(FXSW); float* fxse=(float*)get_write_ptr(FXSE);
        float* fynw=(float*)get_write_ptr(FYNW); float* fyne=(float*)get_write_ptr(FYNE);
        float* fysw=(float*)get_write_ptr(FYSW); float* fyse=(float*)get_write_ptr(FYSE);
        float* wnw=(float*)get_write_ptr(WNW); float* wne=(float*)get_write_ptr(WNE);
        float* wsw=(float*)get_write_ptr(WSW); float* wse=(float*)get_write_ptr(WSE);

        const uint32_t g0 = b * B;                       // global cell index of batch start
        const uint32_t lb = b % chunk_batches;           // batch index within the resident chunk
        for (uint32_t j = 0; j < B; ++j) {
            const uint32_t c = g0 + j;                   // global index (for bounds)
            if (c < ncells) {
                const uint32_t base = (lb * B + j) * 6u; // chunk-LOCAL record offset
                const uint32_t bxl = cellsu[base+0], byl = cellsu[base+1];
                const uint32_t r0 = byl*W + bxl;
                const uint32_t inw=r0*2u, ine=(r0+1u)*2u, isw=(r0+W)*2u, ise=(r0+W+1u)*2u;
                fxnw[j]=field[inw];   fynw[j]=field[inw+1];
                fxne[j]=field[ine];   fyne[j]=field[ine+1];
                fxsw[j]=field[isw];   fysw[j]=field[isw+1];
                fxse[j]=field[ise];   fyse[j]=field[ise+1];
                wnw[j]=cells[base+2]; wne[j]=cells[base+3]; wsw[j]=cells[base+4]; wse[j]=cells[base+5];
            } else {
                fxnw[j]=0;fxne[j]=0;fxsw[j]=0;fxse[j]=0;fynw[j]=0;fyne[j]=0;fysw[j]=0;fyse[j]=0;
                wnw[j]=0;wne[j]=0;wsw[j]=0;wse[j]=0;
            }
        }
        cb_push_back(FXNW,1);cb_push_back(FXNE,1);cb_push_back(FXSW,1);cb_push_back(FXSE,1);
        cb_push_back(FYNW,1);cb_push_back(FYNE,1);cb_push_back(FYSW,1);cb_push_back(FYSE,1);
        cb_push_back(WNW,1);cb_push_back(WNE,1);cb_push_back(WSW,1);cb_push_back(WSE,1);
    }
}
