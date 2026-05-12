# Tenstorrent Blackhole p150b — Hardware Specification

This document captures the hardware specifications of the Tenstorrent Blackhole chip
installed on this server (`bh-38`), gathered from `tt-smi`, `ttnn` device queries,
and the tt-metal UMD (User Mode Driver) source descriptors.

---

## Board & System Info

| Item              | Value                          |
|-------------------|--------------------------------|
| Hostname          | `bh-38`                        |
| Board type        | `p150b`                        |
| Architecture      | Blackhole                      |
| PCIe bus ID       | `0000:01:00.0`                 |
| PCIe interface    | Gen 4 ×16                      |
| Board ID          | `0000041100000000`             |
| Host OS           | Ubuntu 20.04.6 LTS (x86_64)   |
| Host RAM          | 503.80 GB                      |
| Kernel driver     | TT-KMD 2.7.0                  |
| Firmware bundle   | 19.6.0.0                       |
| BM app firmware   | 0.22.0.0                       |
| tt-smi version    | 4.0.0                          |
| tt-umd version    | 0.9.5.dev260424                |

---

## NOC Grid Layout

The Blackhole NOC (Network-on-Chip) is a **17 × 12** mesh of nodes. Each node is one
of several functional types. Column 8 is the vertical spine reserved for L2CPU,
ARC management, security, and routing nodes. Column 9 is the DRAM spine.

```
NOC grid: 17 cols (x=0..16) × 12 rows (y=0..11)

  Col:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
y=0   [RT] [RT] [PC] [RT] [RT] [RT] [RT] [RT] [ARC][DM] [RT] [PC] [RT] [RT] [RT] [RT] [RT]
y=1         [E]  [E]  [E]  [E]  [E]  [E]  [E] [RT]      [E]  [E]  [E]  [E]  [E]  [E]  [E]
y=2         [T]  [T]  [T]  [T]  [T]  [T]  [T] [SC]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=3         [T]  [T]  [T]  [T]  [T]  [T]  [T] [L2]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=4         [T]  [T]  [T]  [T]  [T]  [T]  [T] [RT]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=5         [T]  [T]  [T]  [T]  [T]  [T]  [T] [L2]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=6         [T]  [T]  [T]  [T]  [T]  [T]  [T] [RT]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=7         [T]  [T]  [T]  [T]  [T]  [T]  [T] [L2]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=8         [T]  [T]  [T]  [T]  [T]  [T]  [T] [RT]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=9         [T]  [T]  [T]  [T]  [T]  [T]  [T] [L2]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=10        [T]  [T]  [T]  [T]  [T]  [T]  [T] [RT]      [T]  [T]  [T]  [T]  [T]  [T]  [T]
y=11        [T]  [T]  [T]  [T]  [T]  [T]  [T]      [DM] [T]  [T]  [T]  [T]  [T]  [T]  [T]

Legend: [T]=Tensix  [E]=Ethernet  [DM]=DRAM  [L2]=L2CPU  [ARC]=ARC mgmt
        [RT]=Router-only  [PC]=PCIe  [SC]=Security
```

---

## Tensix Compute Cores

Tensix is Tenstorrent's custom compute tile. Each Tensix core contains a **Matrix Engine**
(for GEMM/matmul), an **SFPU** (vector engine for element-wise ops), a **1.5 MB L1 SRAM**,
and **5 RISC-V processors** that orchestrate data movement and computation.

### Core Counts

| Item                                   | Count                     |
|----------------------------------------|---------------------------|
| Physical Tensix cores (full chip)      | 140 (14 cols × 10 rows)   |
| Usable cores (after harvesting)        | **110** (11 cols × 10 rows) |
| Harvested/disabled cores               | 30 (3 cols removed)       |
| Compute grid reported by ttnn          | 11 × 10                   |

> Harvesting state from UMD: `Tensix: 0xc0` — rows with defects are disabled
> during manufacturing test to improve yield.

### RISC-V Processors per Tensix Core

Each Tensix core has **5 independent RISC-V processors**:

| Processor | Role                                                          |
|-----------|---------------------------------------------------------------|
| **BRISC** | Data movement — sends/receives data over the NOC             |
| **TRISC0** | Compute — unpacks input tiles from L1 for the Matrix Engine  |
| **TRISC1** | Compute — drives the Matrix Engine (matmul, convolution)     |
| **TRISC2** | Compute — packs output tiles back to L1 after SFPU           |
| **NCRISC** | Data movement — manages DMA transfers from/to DRAM via NOC   |

Source: `blackhole_implementation.cpp` — soft reset enum confirms BRISC, TRISC0–2, NCRISC;
NEO-style RISC-V (a future architecture) explicitly throws on Blackhole.

### Tensix RISC-V Totals

| Item                        | Value     |
|-----------------------------|-----------|
| RISC-V per Tensix core      | 5         |
| Usable Tensix cores         | 110       |
| **Total Tensix RISC-V**     | **550**   |

### Tensix Memory

| Item                        | Value     |
|-----------------------------|-----------|
| L1 SRAM per core            | **1.5 MB** (1,572,864 bytes) |
| Total Tensix L1 SRAM        | **165 MB** (110 × 1.5 MB)   |

---

## Ethernet Cores

Blackhole has dedicated Ethernet NOC tiles for inter-chip and inter-rack communication.
Each ETH core runs **2 RISC-V processors** (ERISC0, ERISC1) and has its own 256 KB L1 SRAM.

| Item                        | Value     |
|-----------------------------|-----------|
| Physical ETH cores          | 14        |
| Enabled ETH cores           | **12** (2 harvested: mask `0x120`) |
| RISC-V per ETH core         | 2 (ERISC0, ERISC1)          |
| **Total ETH RISC-V**        | **24**    |
| L1 SRAM per ETH core        | 256 KB    |
| Total ETH L1 SRAM           | 3 MB      |

ETH cores are located at NOC row y=1, at columns 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16.

---

## L2CPU — Embedded RISC-V Linux CPUs

Blackhole contains **4 full Linux-capable RISC-V CPU cores** embedded within the chip
die. These are located along the column-8 NOC spine and can run standard Linux
workloads directly on the accelerator without host CPU involvement.

| Item                        | Value                              |
|-----------------------------|------------------------------------|
| L2CPU cores enabled         | **4** (all active, mask `0xf`)     |
| NOC positions               | (8,3), (8,5), (8,7), (8,9)        |
| Type                        | Full Linux-capable RISC-V CPUs     |
| Harvesting                  | None (L2CPU mask: `0x0`)           |

---

## Total RISC-V Processor Count

| Subsystem         | Count  |
|-------------------|--------|
| Tensix (5× each)  | 550    |
| Ethernet (2× each)| 24     |
| L2CPU             | 4      |
| **Grand Total**   | **578** |

> Note: The ARC management processor (at NOC position 8,0) is an additional
> embedded controller for telemetry and power management but is not user-programmable.

---

## GDDR6 Memory

| Item                        | Value               |
|-----------------------------|---------------------|
| GDDR channels               | **8**               |
| Total capacity              | **24 GB**           |
| Speed per pin               | 16 Gbps (GDDR6)     |
| DRAM NOC nodes              | 24 (8 channels × 3 NOC nodes each) |
| DRAM status                 | All channels active |
| Memory columns              | NOC column 0 (channels 0–3) and column 9 (channels 4–7) |

GDDR6 channel temperatures (from tt-smi at time of measurement):

| Channel pair | Temperature |
|--------------|-------------|
| GDDR 0–1     | 38°C / 38°C |
| GDDR 2–3     | 38°C / 40°C |
| GDDR 4–5     | 40°C / 40°C |
| GDDR 6–7     | 36°C / 38°C |

---

## Clock Frequencies

| Clock     | Current  | Max      |
|-----------|----------|----------|
| AICLK     | 800 MHz  | 1350 MHz |
| AXICLK    | 960 MHz  | —        |
| ARCCLK    | 800 MHz  | —        |
| L2CPUCLK  | 800 MHz  | —        |

---

## Power & Thermal

| Item                         | Value      |
|------------------------------|------------|
| Current power draw           | 44 W       |
| TDP (board limit)            | 150 W      |
| Max TDP (chip)               | 300 W      |
| Current (Icc)                | 60.0 A     |
| Core voltage (Vcore)         | 0.74 V     |
| Vcore range                  | 0.70–0.90 V |
| ASIC temperature             | 48.3°C     |
| Thermal throttle limit       | 90°C       |
| Thermal shutdown limit       | 110°C      |
| Fan speed                    | 38% (~1882 RPM) |
| Thermal trip count           | 0          |

---

## How to Query Hardware Live

```bash
# Activate the tt-metal Python environment
source /localdev/ayadav/tt-work/TTPort/tt-metal/python_env/bin/activate

# Dump full JSON snapshot (non-interactive)
tt-smi -s

# List detected boards
tt-smi -ls

# Query device specs via ttnn (run inside Docker container)
docker exec bh-38-special-ayadav-for-reservation-72646 bash -c "
  cd /localdev/ayadav/tt-work/TTPort/tt-metal
  source python_env/bin/activate
  PYTHONPATH=ttnn LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH TT_METAL_HOME=. \
  python -c \"
import ttnn
device = ttnn.open_device(device_id=0)
print('Arch:', device.arch())
print('Compute grid:', device.compute_with_storage_grid_size())
print('DRAM grid:', device.dram_grid_size())
ttnn.close_device(device)
\"
"
```

---

## Sources

| Source | Path |
|--------|------|
| SOC descriptor (full physical layout) | `tt_metal/third_party/umd/tests/soc_descs/blackhole_140_arch.yaml` |
| RISC-V type definitions | `tt_metal/third_party/umd/device/api/umd/device/types/risc_type.hpp` |
| Blackhole UMD implementation | `tt_metal/third_party/umd/device/arch/blackhole_implementation.cpp` |
| Live telemetry | `tt-smi -s` (run on `bh-38`, Apr 24 2026) |
| ttnn device query | `ttnn.open_device(device_id=0)` via Docker container |
