// SPDX-License-Identifier: Apache-2.0
//
// On-chip unsort for the V35 backward: scatter the gather output (per-slot gx,gy in the
// engine's page-per-core layout) into V22's INTERLEAVED density_grad buffer b_dg
// ([gx0,gy0,gx1,gy1,...], 2*num_nodes elems, tile-interleaved DRAM), entirely on device —
// replacing the host d2h + grad[sel[oidx]] unsort. Validated mechanism: onchip-unsort-64b-reliable.
//
// node = sel[oidx] is resolved ON DEVICE (no host): read the gather's oidx (page-per-core) and
// gather node from a resident sel buffer (page=64B, so sel[oi] = page(oi>>4)[oi&15] is a 64B-
// aligned read — per-slot 4B random reads would violate NoC 64B align). Reads are issued in
// CHUNKs to overlap latency. Sharded by NODE: owner = node/epc (epc mult of 512 → 2*epc tile-
// aligned). Band collisions (k>max_k wide cells) need step-3 accumulate (separate); the bulk
// (1 slot/node via the field halo) is correct with a single overwriting 16B write per slot.
//
// Three passes (same program, shard CB persists across launches; Finish between = barrier):
//   phase 2 ZERO    : each core zeroes its OWN shard (fixed/inactive nodes -> 0 in b_dg).
//   phase 0 SCATTER : each core reads its g_n slots (gxb/gyb/node_slot page c), writes each
//                     active slot's [gx,gy] 16B record into the owner core's L1 shard.
//   phase 1 FLUSH   : each core compacts its shard (16B/node -> 8B/node interleaved) + bulk-
//                     writes its whole tiles to b_dg.

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

void kernel_main() {
    const uint32_t phase     = get_arg_val<uint32_t>(0);
    const uint32_t my_core   = get_arg_val<uint32_t>(1);
    const uint32_t g_n       = get_arg_val<uint32_t>(2);    // valid slots this core (phase 0)
    const uint32_t gx_base   = get_arg_val<uint32_t>(3);    // page-per-core: page c = core c's slots (mc floats)
    const uint32_t gy_base   = get_arg_val<uint32_t>(4);
    const uint32_t oi_base   = get_arg_val<uint32_t>(5);    // gather oidx (u32 per slot, page-per-core); 0xffffffff = skip
    const uint32_t out_pg    = get_arg_val<uint32_t>(6);    // per-core page bytes (= mc*4) for gx/gy/oidx
    const uint32_t coord_base= get_arg_val<uint32_t>(7);    // nc u32 (noc_x<<16)|noc_y
    const uint32_t coord_pg  = get_arg_val<uint32_t>(8);
    const uint32_t nc        = get_arg_val<uint32_t>(9);
    const uint32_t epc       = get_arg_val<uint32_t>(10);   // nodes per core shard (multiple of 512)
    const uint32_t two_nn    = get_arg_val<uint32_t>(11);   // 2*num_nodes (valid interleaved range)
    const uint32_t my_tile0  = get_arg_val<uint32_t>(12);   // phase 1: first b_dg tile this core owns
    const uint32_t my_ntiles = get_arg_val<uint32_t>(13);
    const uint32_t bdg_base  = get_arg_val<uint32_t>(14);
    const uint32_t bdg_pg    = get_arg_val<uint32_t>(15);   // 4096 (1 tile)
    const uint32_t na        = get_arg_val<uint32_t>(16);   // num active nodes (oi validity bound)
    const uint32_t sel_base  = get_arg_val<uint32_t>(17);   // resident sel[]: packed (sub<<24)|node, page=64B
    const uint32_t maxsub    = get_arg_val<uint32_t>(18);   // sub-slots/node (max node multiplicity in sel)

    constexpr auto CB_SHARD=tt::CBIndex::c_0, CB_COORD=tt::CBIndex::c_1,
                   CB_GX=tt::CBIndex::c_2, CB_GY=tt::CBIndex::c_3, CB_OI=tt::CBIndex::c_4,
                   CB_REC=tt::CBIndex::c_5, CB_PACK=tt::CBIndex::c_6, CB_SEL=tt::CBIndex::c_7;
    // sel is non-injective: a big cell = multiple oi sharing a node. Each oi carries a sub-index
    // (packed in sel's top byte). A node gets maxsub sub-slots (16B each); oi's partial goes to
    // sub-slot sub[oi] (distinct -> no race); the flush SUMS the maxsub sub-slots.
    constexpr uint32_t S=16u, CHUNK=256u;
    const uint32_t shard_l1 = get_write_ptr(CB_SHARD);     // same L1 addr on every core; persists across launches

    if (phase==2u) {                                        // ZERO this core's shard (maxsub sub-slots/node)
        volatile uint32_t* sh=(volatile uint32_t*)shard_l1; const uint32_t nw=epc*maxsub*4u;  // 4 u32 per 16B sub-slot
        for (uint32_t i=0;i<nw;++i) sh[i]=0u;
        return;
    }

    if (phase==0u) {                                        // SCATTER
        const InterleavedAddrGen<true> gxg = {.bank_base_address=gx_base,   .page_size=out_pg};
        const InterleavedAddrGen<true> gyg = {.bank_base_address=gy_base,   .page_size=out_pg};
        const InterleavedAddrGen<true> oig = {.bank_base_address=oi_base,   .page_size=out_pg};
        const InterleavedAddrGen<true> cg  = {.bank_base_address=coord_base,.page_size=coord_pg};
        const InterleavedAddrGen<true> selg= {.bank_base_address=sel_base,  .page_size=64u};   // 16 entries/page
        const uint32_t coord_l1=get_write_ptr(CB_COORD), gx_l1=get_write_ptr(CB_GX),
                       gy_l1=get_write_ptr(CB_GY), oi_l1=get_write_ptr(CB_OI),
                       rec_l1=get_write_ptr(CB_REC), sel_l1=get_write_ptr(CB_SEL);
        noc_async_read(cg.get_noc_addr(0), coord_l1, nc*4u);
        if (g_n) {
            noc_async_read(gxg.get_noc_addr(my_core), gx_l1, g_n*4u);
            noc_async_read(gyg.get_noc_addr(my_core), gy_l1, g_n*4u);
            noc_async_read(oig.get_noc_addr(my_core), oi_l1, g_n*4u);
        }
        noc_async_read_barrier();
        const uint32_t* coord=(const uint32_t*)coord_l1; const uint32_t* oidx=(const uint32_t*)oi_l1;
        const uint32_t* gx=(const uint32_t*)gx_l1; const uint32_t* gy=(const uint32_t*)gy_l1;
        const uint32_t* selb=(const uint32_t*)sel_l1; volatile uint32_t* rec=(volatile uint32_t*)rec_l1;
        // CHUNK slots at a time: issue 64B sel-page reads (one per valid slot) -> barrier -> scatter
        for (uint32_t base=0; base<g_n; base+=CHUNK) {
            uint32_t cnt = (g_n-base<CHUNK)?(g_n-base):CHUNK;
            for (uint32_t i=0;i<cnt;++i){ uint32_t oi=oidx[base+i]; if (oi==0xffffffffu||oi>=na) continue;
                noc_async_read(selg.get_noc_addr(oi>>4), sel_l1 + i*64u, 64u); }
            noc_async_read_barrier();
            for (uint32_t i=0;i<cnt;++i){ uint32_t oi=oidx[base+i]; if (oi==0xffffffffu||oi>=na) continue;
                uint32_t v=selb[i*16u + (oi&15u)]; uint32_t nd=v&0xffffffu, sub=v>>24;   // unpack node + sub-index
                uint32_t owner=nd/epc, loff=((nd-owner*epc)*maxsub + sub)*S, pk=coord[owner];
                rec[0]=gx[base+i]; rec[1]=gy[base+i]; rec[2]=0u; rec[3]=0u;
                noc_async_write(rec_l1, get_noc_addr(pk>>16, pk&0xffffu, shard_l1+loff), 16u);
            }
        }
        noc_async_write_barrier();
        return;
    }

    // phase 1: FLUSH — sum the maxsub sub-slots/node -> 8B/node interleaved, write whole tiles to b_dg
    const InterleavedAddrGenFast<true> bdgg={.bank_base_address=bdg_base,.page_size=bdg_pg,.data_format=DataFormat::Float32};
    const uint32_t pack_l1=get_write_ptr(CB_PACK);
    volatile uint32_t* sh=(volatile uint32_t*)shard_l1; uint32_t* pk=(uint32_t*)pack_l1;
    const uint32_t nelem=my_ntiles*1024u, base=my_tile0*1024u;
    union { uint32_t u; float f; } acc, t;
    for (uint32_t e=0;e<nelem;++e){ uint32_t k=e>>1, comp=e&1u;
        if (base+e<two_nn){ acc.f=0.0f;
            for (uint32_t s=0;s<maxsub;++s){ t.u=sh[(k*maxsub+s)*4u+comp]; acc.f+=t.f; }   // sum node's sub-slots
            pk[e]=acc.u; }
        else pk[e]=0u; }
    for (uint32_t tt=0;tt<my_ntiles;++tt) noc_async_write_tile(my_tile0+tt, bdgg, pack_l1+tt*4096u);
    noc_async_write_barrier();
}
