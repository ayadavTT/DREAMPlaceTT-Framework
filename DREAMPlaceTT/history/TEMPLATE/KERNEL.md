# <kernel name> — <one-line role>

> **TL;DR:** <2–3 sentences: what it computes, the one core trick, and its status.>

---

## 1. Identity
- **Stage:** forward-scatter | forward-gather | forward-geom-stash | backward-bucketing | backward-ef-gather | backward-grad-writeout | optimizer | dct-solve | infra
- **Status:** SHIPPED | VALIDATED | SUPERSEDED | REFUTED | INFRA
- **Lineage:** <predecessor> → THIS → <successor>  (what it improved; what replaced it)
- **Source files:** `src/<file>` (lines, RISC), …
- **Activated by:** <env flag / GATHER_MODE / host entry file:func>

## 2. Problem & contract
- **Computation** (DREAMPlace terms): inputs → outputs, the math.
- **Inputs:** buffers (DRAM/L1 layout, page size), CB indices, runtime args, semaphores, common args.
- **Output:** what + where + numeric format (fp32 / bf16 / uint32 fixed-point / int32).
- **Invariants / preconditions:** alignment (e.g. 64 B DRAM read on Blackhole), padding, caps, ownership scheme.

## 3. Mechanism (how it works)
- **Data residency:** L1 vs DRAM; CB allocation order + slot counts; scratch layout.
- **RISC division of labor:** BRISC / NCRISC / TRISC(0–2) roles + handshakes (semaphores).
- **Algorithm:** step-by-step.
- **Key data structures:** route bufs / owner maps / stash / shared-state structs.
- **THE TRICK:** the one (or two) ideas that make it correct/fast.

## 4. Performance (measured — TT)
- **Profiled config:** design / grid / #cells / AICLK.
- **Headline:** ms/iter or ns/cell | vs CPU 16T: <x> | vs predecessor: <x>.
- **Critical path:** which RISC/zone dominates, and why.
- **Load imbalance:** max/median ratio (balanced <1.3, skewed >3).
- **Scaling:** 512 / 1024 / 2048 behavior.
- → full per-zone table in `profile/zones.txt`; interpretation + headline schema in `profile/PROFILE.md`.

## 5. Correctness
- **Method:** bit-exact (N/M) | rel_l2 <tol> | HPWL convergence vs CPU.
- **Result:** numbers.

## 6. Gotchas / pitfalls
- <alignment, overflow caps, contention, CB-order, page-size, deadlock causes…>

## 7. When to use / avoid (the lesson)
- **Wins when:** … | **Avoid when:** … | **What killed earlier variants:** …

## 8. Provenance
- **Memory:** [[slug]], …
- **Handoff docs:** `docs/<…>.md`
- **Host entry / profile run:** <file:func> ; <exact profiler command + date>
