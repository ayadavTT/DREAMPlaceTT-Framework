// SPDX-License-Identifier: Apache-2.0
//
// V29 EF backward stage 2 — writer (NCRISC). Drains the SFPU gx/gy tiles and
// writes them CONTIGUOUSLY (bucketed order) to this band's page in grad_buf:
// per cell [gx, gy] (8 B). grad_buf is interleaved (page per band) → fast d2h.
// The host un-sorts to cell order using oidx_buf (written by the gather). This
// replaces the old single-page scatter-by-orig_idx, whose single-bank d2h was
// ~77 ms; contiguous interleaved write+read is ~1 ms.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t n_batches  = get_arg_val<uint32_t>(0);
    const uint32_t grad_base  = get_arg_val<uint32_t>(1);
    const uint32_t grad_pg    = get_arg_val<uint32_t>(2);   // mbatch*1024*8 bytes (per band)
    const uint32_t my_band    = get_arg_val<uint32_t>(3);

    constexpr uint32_t B=1024;
    constexpr auto GX=tt::CBIndex::c_16, GY=tt::CBIndex::c_17, STAGE=tt::CBIndex::c_28;
    const InterleavedAddrGen<true> gg = {.bank_base_address=grad_base, .page_size=grad_pg};
    const uint64_t gpage = gg.get_noc_addr(my_band);
    float* st = reinterpret_cast<float*>(get_write_ptr(STAGE));   // B*2 floats (8 KB)

    for (uint32_t b=0;b<n_batches;++b){
        cb_wait_front(GX,1); cb_wait_front(GY,1);
        const float* gx=reinterpret_cast<const float*>(get_read_ptr(GX));
        const float* gy=reinterpret_cast<const float*>(get_read_ptr(GY));
        for (uint32_t j=0;j<B;++j){ st[j*2+0]=gx[j]; st[j*2+1]=gy[j]; }
        noc_async_write((uint32_t)st, gpage + (uint64_t)b*B*8u, B*8u);
        noc_async_write_barrier();
        cb_pop_front(GX,1); cb_pop_front(GY,1);
    }
}
