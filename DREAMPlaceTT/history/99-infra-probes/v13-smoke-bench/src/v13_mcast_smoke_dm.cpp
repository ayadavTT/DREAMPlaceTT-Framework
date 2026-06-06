// SPDX-License-Identifier: Apache-2.0
//
// V13 Phase 0a — multicast smoke kernel with canonical handshake.
//
// Pattern (modeled on tt_metal/programming_examples/contributed/multicast/):
//   1. BOOTSTRAP barrier — all cores rendezvous via core-0 fan-in.
//      Every core unicast-incs `ready_count` on core 0; core 0 waits for
//      ready_count == nc_all then mcasts `release_sem` to all peers (and
//      locally sets its own). All cores wait on `release_sem == VALID`.
//      After this barrier, all 110 cores are in a known state — they all
//      just exited a sem-wait and have NOC primed.
//   2. DATA mcast — each sender stamps its outbound, does a local-memcpy
//      self-delivery, then mcasts the payload to data_rects (with the
//      gap-aware rectangle layout for BH-38's non-contiguous worker grid).
//   3. COMPLETION barrier — senders unicast-inc `done_count` on core 0;
//      core 0 waits for done_count == n_active_senders, mcasts
//      `done_release`. All cores wait then exit.
//
// L1 layout per core (within CB_SCRATCH = c_24):
//   [outbound_off : outbound_off + payload_bytes]
//   [inbound_off  : inbound_off  + nc_all * payload_bytes]
//
// Runtime args:
//   0:  my_core_id
//   1:  nc_all
//   2:  payload_bytes
//   3:  outbound_l1_off
//   4:  inbound_l1_off
//   5:  is_sender
//   6:  n_active_senders
//   7:  my_noc_x
//   8:  my_noc_y
//   9:  core0_noc_x
//  10:  core0_noc_y
//  11:  ready_count_sem_id    (counter on core 0; all cores inc it)
//  12:  ready_release_sem_id  (per-core; core 0 mcasts to set)
//  13:  done_count_sem_id     (counter on core 0; senders inc it)
//  14:  done_release_sem_id   (per-core; core 0 mcasts to set)
//  15:  n_full_rects          (= 2 on BH-38: covers all worker cores)
//  16..16+5*n_full_rects-1:   per-full-rect quint (xs, ys, xe, ye, num_dests)
//                             num_dests excludes core 0 (sender of release mcast)
//  16+5*n_full_rects:         n_data_rects
//  17+5*n_full_rects..:       per-data-rect quint (xs, ys, xe, ye, num_dests)
//                             num_dests excludes this core if it's in the rect

#if __has_include("api/dataflow/dataflow_api.h")
#include "api/dataflow/dataflow_api.h"
#else
#include "dataflow_api.h"
#endif

constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_24;
constexpr uint32_t SEM_VALID = 1u;

void kernel_main() {
    const uint32_t my_core_id        = get_arg_val<uint32_t>(0);
    const uint32_t nc_all            = get_arg_val<uint32_t>(1);
    const uint32_t payload_bytes     = get_arg_val<uint32_t>(2);
    const uint32_t outbound_off      = get_arg_val<uint32_t>(3);
    const uint32_t inbound_off       = get_arg_val<uint32_t>(4);
    const uint32_t is_sender         = get_arg_val<uint32_t>(5);
    const uint32_t n_active_senders  = get_arg_val<uint32_t>(6);
    const uint32_t my_noc_x          = get_arg_val<uint32_t>(7);
    const uint32_t my_noc_y          = get_arg_val<uint32_t>(8);
    const uint32_t core0_noc_x       = get_arg_val<uint32_t>(9);
    const uint32_t core0_noc_y       = get_arg_val<uint32_t>(10);
    const uint32_t ready_count_sid   = get_arg_val<uint32_t>(11);
    const uint32_t ready_release_sid = get_arg_val<uint32_t>(12);
    const uint32_t done_count_sid    = get_arg_val<uint32_t>(13);
    const uint32_t done_release_sid  = get_arg_val<uint32_t>(14);
    const uint32_t n_full_rects      = get_arg_val<uint32_t>(15);

    uint8_t* base = reinterpret_cast<uint8_t*>(get_write_ptr(CB_SCRATCH));
    uint32_t* outbound = reinterpret_cast<uint32_t*>(base + outbound_off);
    uint32_t* inbound  = reinterpret_cast<uint32_t*>(base + inbound_off);
    const uint32_t n_words = payload_bytes / 4u;

    volatile tt_l1_ptr uint32_t* ready_release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_release_sid));
    volatile tt_l1_ptr uint32_t* done_release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_release_sid));

    const bool i_am_core0 = (my_core_id == 0u);

    // ── Step A: pre-clear inbound stamps ──────────────────────────────────
    for (uint32_t s = 0; s < nc_all; ++s) {
        inbound[s * n_words] = 0xFFFFFFFFu;
    }

    // ── Step B: stamp outbound (sender only) ──────────────────────────────
    if (is_sender) {
        outbound[0] = my_core_id;
        outbound[1] = payload_bytes;
        for (uint32_t i = 2; i < n_words - 1u; ++i) {
            outbound[i] = (i * 0x12345u) ^ my_core_id;
        }
        outbound[n_words - 1u] = my_core_id;
    }

    // ── Step C: BOOTSTRAP barrier (core-0 fan-in + mcast release) ─────────
    {
        // All cores (including core 0): unicast inc ready_count on core 0.
        // Core 0's inc lands on itself (loopback unicast is fine).
        uint64_t ready_count_addr = get_noc_addr(
            core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(ready_count_sid));
        noc_semaphore_inc(ready_count_addr, 1u);

        if (i_am_core0) {
            // Wait for everyone (including self) to have inc'd.
            volatile tt_l1_ptr uint32_t* ready_count =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_count_sid));
            noc_semaphore_wait_min(ready_count, nc_all);
            noc_semaphore_set(ready_count, 0);  // reset for any future use

            // Mcast release to all peers across all full rects (gap-aware).
            uint32_t release_l1 = (uint32_t)get_semaphore(ready_release_sid);
            // Set local release first (mcast won't write to self).
            noc_semaphore_set(ready_release, SEM_VALID);
            for (uint32_t r = 0; r < n_full_rects; ++r) {
                const uint32_t b = 16u + r * 5u;
                const uint32_t xs = get_arg_val<uint32_t>(b + 0u);
                const uint32_t ys = get_arg_val<uint32_t>(b + 1u);
                const uint32_t xe = get_arg_val<uint32_t>(b + 2u);
                const uint32_t ye = get_arg_val<uint32_t>(b + 3u);
                const uint32_t nd = get_arg_val<uint32_t>(b + 4u);
                if (nd == 0u) continue;
                uint64_t mcast_addr = get_noc_multicast_addr(xs, ys, xe, ye, release_l1);
                noc_semaphore_set_multicast(release_l1, mcast_addr, nd);
            }
            // Barrier on the mcast writes.
            noc_async_atomic_barrier();
        }

        // All cores wait for local release to be VALID.
        noc_semaphore_wait(ready_release, SEM_VALID);
    }

    // ── Step D: data multicast (senders only) ─────────────────────────────
    if (is_sender) {
        // Self-delivery via local L1 copy (plain mcast doesn't write to self).
        uint32_t* self_slot = inbound + my_core_id * n_words;
        for (uint32_t i = 0; i < n_words; ++i) {
            self_slot[i] = outbound[i];
        }

        // Data multicasts.
        const uint32_t dst_local_addr =
            (uint32_t)(base + inbound_off + my_core_id * payload_bytes);

        // n_data_rects + rects start at (16 + 5*n_full_rects)
        const uint32_t data_base = 16u + 5u * n_full_rects;
        const uint32_t n_data_rects = get_arg_val<uint32_t>(data_base);
        for (uint32_t r = 0; r < n_data_rects; ++r) {
            const uint32_t b = data_base + 1u + r * 5u;
            const uint32_t xs = get_arg_val<uint32_t>(b + 0u);
            const uint32_t ys = get_arg_val<uint32_t>(b + 1u);
            const uint32_t xe = get_arg_val<uint32_t>(b + 2u);
            const uint32_t ye = get_arg_val<uint32_t>(b + 3u);
            const uint32_t nd = get_arg_val<uint32_t>(b + 4u);
            if (nd == 0u) continue;
            uint64_t mcast_dst = get_noc_multicast_addr(xs, ys, xe, ye, dst_local_addr);
            noc_async_write_multicast(
                (uint32_t)(base + outbound_off),
                mcast_dst,
                payload_bytes,
                nd);
        }
        noc_async_write_barrier();
    }

    // ── Step E: COMPLETION barrier (core-0 fan-in + mcast release) ────────
    {
        if (is_sender) {
            // All senders unicast-inc done_count on core 0.
            uint64_t done_count_addr = get_noc_addr(
                core0_noc_x, core0_noc_y, (uint32_t)get_semaphore(done_count_sid));
            noc_semaphore_inc(done_count_addr, 1u);
        }

        if (i_am_core0) {
            volatile tt_l1_ptr uint32_t* done_count =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_count_sid));
            noc_semaphore_wait_min(done_count, n_active_senders);
            noc_semaphore_set(done_count, 0);

            uint32_t done_l1 = (uint32_t)get_semaphore(done_release_sid);
            noc_semaphore_set(done_release, SEM_VALID);
            for (uint32_t r = 0; r < n_full_rects; ++r) {
                const uint32_t b = 16u + r * 5u;
                const uint32_t xs = get_arg_val<uint32_t>(b + 0u);
                const uint32_t ys = get_arg_val<uint32_t>(b + 1u);
                const uint32_t xe = get_arg_val<uint32_t>(b + 2u);
                const uint32_t ye = get_arg_val<uint32_t>(b + 3u);
                const uint32_t nd = get_arg_val<uint32_t>(b + 4u);
                if (nd == 0u) continue;
                uint64_t mcast_addr = get_noc_multicast_addr(xs, ys, xe, ye, done_l1);
                noc_semaphore_set_multicast(done_l1, mcast_addr, nd);
            }
            noc_async_atomic_barrier();
        }

        noc_semaphore_wait(done_release, SEM_VALID);
    }
}
