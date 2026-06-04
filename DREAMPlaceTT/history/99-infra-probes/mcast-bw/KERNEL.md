# mcast-bw — NoC multicast bandwidth probe (INFRA)

> **TL;DR:** Measures NoC multicast bandwidth (field-cast to all cores) — the
> **make-or-break** for the FCCS backward. Result: ~13.5 GB/s (0.16/0.62/2.45 ms
> @ 2/8/32 MB for 512/1024/2048 fields) — fast enough → FCCS is viable. Also
> nailed the multicast handshake that was hanging.

---

## 1. Identity
- **Stage:** infra — multicast bandwidth probe
- **Status:** 🔧 INFRA (unblocked FCCS).
- **Source files:** `src/mcast_bw_dm.cpp` (94).
- **Activated by:** `host/mcast_bw_host.cpp` (target `mcast_bw`).

## 2. Problem & contract
Measure time to multicast a field-sized buffer to all 110 cores' L1. Inputs: buffer size. Output: GB/s + per-grid ms.

## 3. Mechanism / findings
- **Bandwidth:** ~13.5 GB/s single-producer → field-cast costs 0.16/0.62/2.45 ms for the 2/8/32 MB fields at 512/1024/2048. That's cheap enough that **FCCS's "broadcast the field to every core" is viable** (the make-or-break passed).
- **Multicast handshake (the unblock):** the hang was mcast on **NOC1 with NOC0 coords** + wrong handshake. **FIX = NOC0 / RISCV_0 + receivers-ready-first handshake** (canonical contributed/multicast example). This rule is reused by FCCS.

## 4. Performance (measured — TT)
- ~13.5 GB/s; 0.16/0.62/2.45 ms @ 2/8/32 MB. (Re-run on a clean device — the 2026-06-03 batch hit a wedged device.)

## 5–7. Correctness / gotchas / when
- **Multicast MUST use NOC0/RISCV_0 + receivers-ready-first handshake** — else it hangs (and wedges the device). Used by FCCS's field-cast.

## 8. Provenance
- **Memory:** [[multicast_hangs_bh38_fw1980]], [[fccs_nextgen_backward_design]]
- **Host:** `host/mcast_bw_host.cpp`
