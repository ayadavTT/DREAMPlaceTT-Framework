# atomic-bench — NoC atomic contention/correctness study (INFRA)

> **TL;DR:** A microbench that characterizes NoC atomics under cross-core
> contention. **The lesson it produced is load-bearing for the whole design:**
> use `noc_semaphore_inc` (static VC), never raw `noc_atomic_*`; **blind atomic
> COUNT is always exact at any contention**, but the fetch-add **return value**
> (unique slot) double-reads under pipelined contention — correct only at
> MAXO≤2. ⇒ use count-prefix regroup (blind count + scan + place), not
> atomic-slot.

---

## 1. Identity
- **Stage:** infra — atomics probe (not a pipeline kernel)
- **Status:** 🔧 INFRA (diagnostic; produced a foundational design rule).
- **Source files:** `src/atomic_bench_brisc.cpp` (86), `src/atomic_bench_ncrisc.cpp` (11). Related probe (archived alongside): `src/v31_atomic_probe.cpp` — a V31-era single-kernel atomic-increment probe used to re-confirm the same blind-count-vs-fetch-add behaviour against the V31 stash buffers.
- **Activated by:** `host/atomic_bench_host.cpp` (target `atomic_bench_host`).

## 2. Problem & contract
Measure correctness + cost of cross-core NoC atomics at varying contention (MAXO = max outstanding). Inputs: contention config. Output: correctness (exact-count check) + ns/op.

## 3. Mechanism / findings
- **Blind COUNT** (`noc_semaphore_inc` of a fixed delta) is **exact at any contention** (~40 ns at MAXO=8) — this is what V19's atomic-add density and V30's atomic grad rely on.
- **Fetch-add return slot** (read-the-old-value-to-get-a-unique-index) **double-reads** when pipelined under cross-core contention — correct only at MAXO≤2 (98 ns/slot @2-way).
- **The "deadlock"** seen earlier was a wrong-API bug: use `noc_semaphore_inc` (static VC), never raw `noc_atomic_*`.
- **⇒ Design rule:** for grouping/bucketing, use **count-prefix** (blind count + on-chip scan + local-cursor place), NOT atomic fetch-add slots. This rule shaped V30/V32/V19.

## 4. Performance (measured — TT)
- ~40 ns/op blind count @MAXO=8; 98 ns/slot fetch-add @MAXO=2. (Standalone runs on a clean device; the 2026-06-03 batch found the device already wedged → re-run after reset.)

## 5–7. Correctness / gotchas / when
- Count: exact. Fetch-add slot: only ≤2-way. **Never** use raw `noc_atomic_*`; always `noc_semaphore_inc`. This is the single most reused atomics lesson in the codebase.

## 8. Provenance
- **Memory:** [[atomic_fetchadd_contention_deadlock]], [[v32_count_prefix_regroup]], [[v19_l1_atomic_converges]]
- **Host:** `host/atomic_bench_host.cpp`
