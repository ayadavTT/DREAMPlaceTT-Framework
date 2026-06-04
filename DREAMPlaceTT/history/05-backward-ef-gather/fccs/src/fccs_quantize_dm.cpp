// SPDX-License-Identifier: Apache-2.0
//
// FCCS field-quantize pre-pass — PARALLEL across all cores.
//
// The chip-DCT field arrives as FLOAT; the FCCS backward wants int32 fixed-point.
// Doing that conversion on the single multicast producer serialized 512²×2≈524K
// soft-float ops on one FPU-less core = ~49 ms. This pre-pass splits the SAME work
// across all `nc` cores: core c converts its contiguous chunk [c*cpc_e,(c+1)*cpc_e)
// of EACH plane (fx, fy) IN PLACE, float → (int32)(f·2^FS). 1/nc of the work per core
// (~2.4K elems ≈ 0.2 ms), well below the multicast floor, so quantize is no longer
// the bottleneck. Field buffers are PADDED to nc*cpc_e (cpc_e a multiple of 16) so
// every chunk read/write is 64-B aligned and in-bounds (Blackhole DRAM-read align).
//
// Args: 0 my_core 1 fx_base 2 fy_base 3 cpc_e (elems/core, mult of 16) 4 FS 5 scratch_l1_off
//       6 field_pg (whole padded buffer bytes, = the buffer's single page size)

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t fx_base = get_arg_val<uint32_t>(1);
    const uint32_t fy_base = get_arg_val<uint32_t>(2);
    const uint32_t cpc_e   = get_arg_val<uint32_t>(3);   // elems per core (multiple of 16)
    const uint32_t FS      = get_arg_val<uint32_t>(4);
    const uint32_t scratch_off = get_arg_val<uint32_t>(5);
    const uint32_t field_pg = get_arg_val<uint32_t>(6);

    const float FSCALE = (float)(1u << FS);
    const uint32_t lo = my_core * cpc_e;                  // my chunk start (elems)
    const uint32_t bytes = cpc_e * 4u;                    // 64-B aligned (cpc_e mult of 16)
    const uint64_t off = (uint64_t)lo * 4u;               // 64-B aligned

    // page = whole padded buffer (single bank-0 page); read by byte offset.
    const InterleavedAddrGen<true> fxg = {.bank_base_address=fx_base, .page_size=field_pg};
    const InterleavedAddrGen<true> fyg = {.bank_base_address=fy_base, .page_size=field_pg};

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t l1 = (uint32_t)(base + scratch_off);
    float*   lf = reinterpret_cast<float*>(l1);
    int32_t* li = reinterpret_cast<int32_t*>(l1);

    // fx chunk: read float → quantize in place → write int32 back
    noc_async_read(fxg.get_noc_addr(0) + off, l1, bytes);
    noc_async_read_barrier();
    for (uint32_t i=0;i<cpc_e;++i) { float v=lf[i]; li[i]=(int32_t)(v*FSCALE); }
    noc_async_write(l1, fxg.get_noc_addr(0) + off, bytes);
    noc_async_write_barrier();

    // fy chunk (reuse scratch)
    noc_async_read(fyg.get_noc_addr(0) + off, l1, bytes);
    noc_async_read_barrier();
    for (uint32_t i=0;i<cpc_e;++i) { float v=lf[i]; li[i]=(int32_t)(v*FSCALE); }
    noc_async_write(l1, fyg.get_noc_addr(0) + off, bytes);
    noc_async_write_barrier();
}
