cmake_minimum_required(VERSION 3.16)
project(dreamplace_tt_server CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ───────────────────────────────────────────────────────────────────────────
# Locate TT-Metal SDK (TT_METAL_HOME).
#
# Either set the TT_METAL_HOME env var or pass -DTT_METAL_HOME=<path>.
# Default falls back to ../tt-metal/ relative to the framework root.
# ───────────────────────────────────────────────────────────────────────────
if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    else()
        get_filename_component(_FW_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
        set(TT_METAL_HOME "${_FW_ROOT}/tt-metal")
    endif()
endif()
message(STATUS "TT_METAL_HOME: ${TT_METAL_HOME}")

# Prefer the Release build (which has the installed CMake package), then dev build.
set(TT_BUILD_RELEASE "${TT_METAL_HOME}/build_Release")
set(TT_BUILD_DEV     "${TT_METAL_HOME}/build")

if(EXISTS "${TT_BUILD_RELEASE}/lib/cmake/tt-metalium/tt-metalium-config.cmake")
    set(TT_CMAKE_PREFIX "${TT_BUILD_RELEASE}/lib/cmake/tt-metalium")
    set(TT_BUILD_USED   "${TT_BUILD_RELEASE}")
elseif(EXISTS "${TT_BUILD_DEV}/lib/cmake/tt-metalium/tt-metalium-config.cmake")
    set(TT_CMAKE_PREFIX "${TT_BUILD_DEV}/lib/cmake/tt-metalium")
    set(TT_BUILD_USED   "${TT_BUILD_DEV}")
else()
    message(FATAL_ERROR
        "Could not find tt-metalium-config.cmake under ${TT_BUILD_RELEASE} or ${TT_BUILD_DEV}.\n"
        "Build tt-metal first inside the Docker container:\n"
        "  cd ${TT_METAL_HOME} && bash build_metal.sh --build-programming-examples")
endif()
message(STATUS "Using TT-Metal build: ${TT_BUILD_USED}")

list(PREPEND CMAKE_PREFIX_PATH "${TT_CMAKE_PREFIX}")
get_filename_component(TT_BUILD_PREFIX "${TT_CMAKE_PREFIX}/../../.." ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH "${TT_BUILD_PREFIX}")

find_package(TT-Metalium REQUIRED PATHS "${TT_CMAKE_PREFIX}" NO_DEFAULT_PATH)

# ───────────────────────────────────────────────────────────────────────────
# DREAMPlace-TT V11 server: scatter + accum + TTNN C++ DCT.
# Requires clang-20 (TTNN headers use C++23 reflection).
# ───────────────────────────────────────────────────────────────────────────
if(NOT EXISTS "${TT_BUILD_USED}/lib/_ttnncpp.so")
    message(FATAL_ERROR
        "Could not find _ttnncpp.so under ${TT_BUILD_USED}/lib/. "
        "Re-build tt-metal with the TTNN C++ target enabled.")
endif()

set(TT_TTNN_INCLUDES
    "${TT_BUILD_USED}/include"
    "${TT_BUILD_USED}/include/ttnn"
    "${TT_BUILD_USED}/include/metalium-thirdparty"
)

add_executable(density_scatter_ttnn_server
    density_scatter_ttnn_server_host.cpp
)

target_link_libraries(density_scatter_ttnn_server PUBLIC TT::Metalium)
target_include_directories(density_scatter_ttnn_server PRIVATE ${TT_TTNN_INCLUDES})
target_link_libraries(density_scatter_ttnn_server PRIVATE
    "${TT_BUILD_USED}/lib/_ttnncpp.so"
    "${TT_BUILD_USED}/lib/libtt_stl.so"
)
target_compile_options(density_scatter_ttnn_server PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
    -Wno-deprecated-declarations
    -DSPDLOG_FMT_EXTERNAL
)

# The server JIT-compiles kernels at runtime by reading their source files.
# DENSITY_KERNEL_DIR points to the framework's kernels/ directory.
get_filename_component(_FW_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
target_compile_definitions(density_scatter_ttnn_server PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)

# Honor host's CXX choice (e.g. clang-20). Don't let CCACHE/ld wrappers override.
set_target_properties(density_scatter_ttnn_server PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured density_scatter_ttnn_server")
message(STATUS "  Kernels dir : ${_FW_ROOT}/kernels/")

# ───────────────────────────────────────────────────────────────────────────
# V24 grid_sample speed probe — measures ttnn::grid_sample (FPU reduce-engine
# weighted pool) at EF dimensions vs the CPU electric_force baseline. Links
# ttnn C++ like density_scatter_ttnn_server.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v24_grid_sample_probe v24_grid_sample_probe.cpp)
target_link_libraries(v24_grid_sample_probe PUBLIC TT::Metalium)
target_include_directories(v24_grid_sample_probe PRIVATE
    ${TT_TTNN_INCLUDES}
    "${TT_METAL_HOME}/ttnn/cpp"   # grid_sample.hpp is source-only (not installed)
    "${TT_METAL_HOME}/ttnn"
)
target_link_libraries(v24_grid_sample_probe PRIVATE
    "${TT_BUILD_USED}/lib/_ttnncpp.so"
    "${TT_BUILD_USED}/lib/libtt_stl.so"
)
target_compile_options(v24_grid_sample_probe PRIVATE
    -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment
    -Wno-deprecated-declarations -DSPDLOG_FMT_EXTERNAL
)
set_target_properties(v24_grid_sample_probe PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v24_grid_sample_probe")

# ───────────────────────────────────────────────────────────────────────────
# V13 Phase 0a — multicast smoke harness (standalone tt-metal binary).
# Pure tt-metal; does not link against TTNN.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v13_mcast_smoke_host
    v13_mcast_smoke_host.cpp
)

target_link_libraries(v13_mcast_smoke_host PUBLIC TT::Metalium)
target_compile_options(v13_mcast_smoke_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v13_mcast_smoke_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v13_mcast_smoke_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v13_mcast_smoke_host")

# ───────────────────────────────────────────────────────────────────────────
# V13_fpu Phase 0a — FPU matmul smoke harness (single tile, single core).
# Verifies bf16xbf16 → fp32 DST → bf16 pack with LoFi math fidelity.
# Pure tt-metal; no TTNN dependency.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v13_matmul_smoke_host
    v13_matmul_smoke_host.cpp
)

target_link_libraries(v13_matmul_smoke_host PUBLIC TT::Metalium)
target_compile_options(v13_matmul_smoke_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v13_matmul_smoke_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v13_matmul_smoke_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v13_matmul_smoke_host")

# ───────────────────────────────────────────────────────────────────────────
# V13_fpu Phase 0b — OX/OY packing microbench host.
# Measures BRISC L1-write throughput at the V13 access pattern.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v13_pack_bench_host
    v13_pack_bench_host.cpp
)

target_link_libraries(v13_pack_bench_host PUBLIC TT::Metalium)
target_compile_options(v13_pack_bench_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v13_pack_bench_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v13_pack_bench_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v13_pack_bench_host")

# ───────────────────────────────────────────────────────────────────────────
# V16 Phase B.0 — SFPU outer-product microbench host.
# Validates the cost-model assumption that SFPU can do per-batch outer-product
# at ≤ 4 µs/batch (vs BRISC's measured 11.6 µs/batch for pack_ox_oy).
# ───────────────────────────────────────────────────────────────────────────
add_executable(v16_sfpu_bench_host
    v16_sfpu_bench_host.cpp
)

target_link_libraries(v16_sfpu_bench_host PUBLIC TT::Metalium)
target_compile_options(v16_sfpu_bench_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v16_sfpu_bench_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v16_sfpu_bench_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v16_sfpu_bench_host")

# ───────────────────────────────────────────────────────────────────────────
# V16 Phase 1.0 — DST face-layout sanity probe.
# Writes unique values into dst_reg slots [0..31] via SFPU on one core and
# dumps the packed L1 tile to DRAM so the host can decode the slot →
# (face, row_pair) mapping. Used to validate face_idx_to_dst_reg() before
# writing the real Phase 1 SFPU pack body.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v16_dst_probe_host
    v16_dst_probe_host.cpp
)

target_link_libraries(v16_dst_probe_host PUBLIC TT::Metalium)
target_compile_options(v16_dst_probe_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v16_dst_probe_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v16_dst_probe_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v16_dst_probe_host")

# ───────────────────────────────────────────────────────────────────────────
# V16 Step 1 — SEC1 Haloize silicon validation probe.
# Runs V13's paired-unpack + custom kaccum LLKs with optional toggle of
# THCON_SEC1_REG2_Haloize_mode (= SrcB within-face transpose). Answers the
# empirical question "does Blackhole silicon honor SEC1 Haloize for matmul-
# style unpacks?", which gates the V16 Option A LLK adapter approach.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v16_sec1_probe_host
    v16_sec1_probe_host.cpp
)

target_link_libraries(v16_sec1_probe_host PUBLIC TT::Metalium)
target_compile_options(v16_sec1_probe_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v16_sec1_probe_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v16_sec1_probe_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v16_sec1_probe_host")

# ───────────────────────────────────────────────────────────────────────────
# V11 Stage B' — SFPU outer-product microbench.
# Measures the per-batch wall time of computing 64 (j, k) outer-product
# values per cell on the SFPU. Used to evaluate whether moving the V11
# scatter-side ox*oy multiplies from BRISC/NCRISC scalar to TRISC SFPU is
# worth the V11 integration cost.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v11op_bench_host
    v11op_bench_host.cpp
)

target_link_libraries(v11op_bench_host PUBLIC TT::Metalium)
target_compile_options(v11op_bench_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v11op_bench_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v11op_bench_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v11op_bench_host")

# ───────────────────────────────────────────────────────────────────────────
# probe_devices — standalone: open each TT device by id, report grid size.
# Used to verify multi-card visibility.
# ───────────────────────────────────────────────────────────────────────────
add_executable(probe_devices probe_devices.cpp)
target_link_libraries(probe_devices PUBLIC TT::Metalium)
target_compile_options(probe_devices PRIVATE -O2 -Wall)
set_target_properties(probe_devices PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured probe_devices")

# ───────────────────────────────────────────────────────────────────────────
# V13_fpu Phase 1.1 — standalone accumulate smoke (single core, K=32 batched).
# ───────────────────────────────────────────────────────────────────────────
add_executable(v13_accum_smoke_host
    v13_accum_smoke_host.cpp
)

target_link_libraries(v13_accum_smoke_host PUBLIC TT::Metalium)
target_compile_options(v13_accum_smoke_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v13_accum_smoke_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v13_accum_smoke_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v13_accum_smoke_host")

# ───────────────────────────────────────────────────────────────────────────
# V13_fpu Phase 1.2 — scatter smoke harness (full SFPU+scatter on 110 cores).
# ───────────────────────────────────────────────────────────────────────────
add_executable(v13_scatter_smoke_host
    v13_scatter_smoke_host.cpp
)

target_link_libraries(v13_scatter_smoke_host PUBLIC TT::Metalium)
target_compile_options(v13_scatter_smoke_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
    -Wno-deprecated-declarations
)
target_compile_definitions(v13_scatter_smoke_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v13_scatter_smoke_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v13_scatter_smoke_host")

# ───────────────────────────────────────────────────────────────────────────
# V13_fpu Step 1.3.1 — end-to-end scatter + accumulate smoke.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v13_full_smoke_host
    v13_full_smoke_host.cpp
)

target_link_libraries(v13_full_smoke_host PUBLIC TT::Metalium)
target_compile_options(v13_full_smoke_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
    -Wno-deprecated-declarations
)
target_compile_definitions(v13_full_smoke_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v13_full_smoke_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)

message(STATUS "Configured v13_full_smoke_host")

# ───────────────────────────────────────────────────────────────────────────
# Atomic-increment microbench — measure noc_semaphore_inc latency in
# loopback / cross-core / contended patterns.  Used to validate atomic-
# based work-stealing for the gather load-balance design.
# ───────────────────────────────────────────────────────────────────────────
add_executable(atomic_bench_host
    atomic_bench_host.cpp
)
target_link_libraries(atomic_bench_host PUBLIC TT::Metalium)
target_compile_options(atomic_bench_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(atomic_bench_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(atomic_bench_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured atomic_bench_host")

# ───────────────────────────────────────────────────────────────────────────
# V19 microbench — isolate V19's atomic-scatter + writeout from v4_compute
# and IPC layers. Sweeps grid × cell-count × distribution. Fine-grained
# Tracy zones (V19MB-INIT, V19MB-CELLS-READ, V19MB-EMIT-BATCH, V19MB-BARRIER,
# V19-WRITEOUT) surface back-pressure / contention. Used to characterize why
# V19 hangs at 2048 grid in the production server.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v19_microbench_host
    v19_microbench_host.cpp
)
target_link_libraries(v19_microbench_host PUBLIC TT::Metalium)
target_compile_options(v19_microbench_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v19_microbench_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v19_microbench_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v19_microbench_host")

# ───────────────────────────────────────────────────────────────────────────
# V20 permute microbench — characterizes the proposed chip-side permute pass
# that would replace the host fused loop. See docs/V20_CHIP_PERMUTE_HANDOFF.md.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v20_mb_permute_host
    v20_mb_permute_host.cpp
)
target_link_libraries(v20_mb_permute_host PUBLIC TT::Metalium)
target_compile_options(v20_mb_permute_host PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v20_mb_permute_host PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v20_mb_permute_host PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v20_mb_permute_host")

# ───────────────────────────────────────────────────────────────────────────
# V21a electric_force (density-backward) microbench. Standalone TT-side half
# of the two-process microbench; the Python orchestrator at
# integration/v21_ef_microbench.py drives it and runs the CPU baseline.
# See docs/V21_DREAMPLACE_ON_CHIP_RESEARCH.md + the V21a plan.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v21_ef_microbench
    v21_electric_force_microbench_host.cpp
)
target_link_libraries(v21_ef_microbench PUBLIC TT::Metalium)
target_compile_options(v21_ef_microbench PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v21_ef_microbench PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v21_ef_microbench PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v21_ef_microbench")

# ───────────────────────────────────────────────────────────────────────────
# V22 K1 — Nesterov optimizer-update microbench. Standalone TT-side half of the
# two-process microbench; integration/v22_nesterov_microbench.py drives it and
# runs the torch Nesterov reference. See docs/V22_CHIP_RESIDENT_POS_HANDOFF.md §2.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v22_nesterov_microbench
    v22_nesterov_microbench_host.cpp
)
target_link_libraries(v22_nesterov_microbench PUBLIC TT::Metalium)
target_compile_options(v22_nesterov_microbench PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v22_nesterov_microbench PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v22_nesterov_microbench PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v22_nesterov_microbench")

# ───────────────────────────────────────────────────────────────────────────
# V22 K2/K3/K4 — generic element-wise op microbench (clamp / combine / precond).
# integration/v22_elt_microbench.py drives it. See docs/V22_CHIP_RESIDENT_POS_HANDOFF.md §2.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v22_elt_microbench
    v22_elt_microbench_host.cpp
)
target_link_libraries(v22_elt_microbench PUBLIC TT::Metalium)
target_compile_options(v22_elt_microbench PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v22_elt_microbench PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v22_elt_microbench PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v22_elt_microbench")

# ───────────────────────────────────────────────────────────────────────────
# V22 K5 — Barzilai-Borwein step-size reduction microbench (the only
# non-elementwise optimizer kernel: 3 global dot-product reductions → step size).
# integration/v22_stepsize_microbench.py drives it.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v22_stepsize_microbench
    v22_stepsize_microbench_host.cpp
)
target_link_libraries(v22_stepsize_microbench PUBLIC TT::Metalium)
target_compile_options(v22_stepsize_microbench PRIVATE
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
)
target_compile_definitions(v22_stepsize_microbench PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v22_stepsize_microbench PROPERTIES
    CXX_COMPILER_LAUNCHER ""
)
message(STATUS "Configured v22_stepsize_microbench")

# ───────────────────────────────────────────────────────────────────────────
# V22 Phase-B trajectory microbench — chip-resident pos/u optimizer chain over
# N steps vs a numpy Nesterov reference. The convergence gate before the
# DREAMPlace live-loop wiring. integration/v22_traj_microbench.py drives it.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v22_traj_microbench
    v22_traj_microbench_host.cpp
    v22_engine.cpp
)
target_link_libraries(v22_traj_microbench PUBLIC TT::Metalium)
target_compile_options(v22_traj_microbench PRIVATE
    -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment
)
target_compile_definitions(v22_traj_microbench PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v22_traj_microbench PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v22_traj_microbench")

# ───────────────────────────────────────────────────────────────────────────
# V22 optimizer-on-chip engine + full-step microbench. The engine (v22_engine.cpp)
# chains combine→precond→nesterov→clamp through persistent DRAM buffers; the
# microbench drives one full step and integration/v22_step_microbench.py diffs
# it against torch. See docs/V22_CHIP_RESIDENT_POS_HANDOFF.md §5 Phase A.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v22_step_microbench
    v22_step_microbench_host.cpp
    v22_engine.cpp
)
target_link_libraries(v22_step_microbench PUBLIC TT::Metalium)
target_compile_options(v22_step_microbench PRIVATE
    -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment
)
target_compile_definitions(v22_step_microbench PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(v22_step_microbench PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v22_step_microbench")

# ───────────────────────────────────────────────────────────────────────────
# V19 Rung 3 — in-process pybind11 extension.
#
# Builds `v19_engine` as a Python module that DREAMPlace can import directly,
# bypassing the docker-exec/shm IPC path used by `density_scatter_ttnn_server`.
# Handles GATHER_MODE in {v11, v11outer, v18, v18outer, v19} — the modes that
# share the V11 setup. V13/V14 stay on the IPC server.
#
# Build (inside the tt-metal container, where clang-20 lives):
#   cmake -Dpybind11_DIR=<dp_env>/lib/python3.8/site-packages/pybind11/share/cmake/pybind11
#         -DPython3_EXECUTABLE=<dp_env>/bin/python3 ..
#   cmake --build . --target v19_engine -j16
# ───────────────────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(v19_engine MODULE
        v19_engine.cpp
        v21_ef_engine.cpp
        v22_engine.cpp
        v29_ef_engine.cpp
        v30_ef_engine.cpp
        fccs_ef_engine.cpp
        v35_ef_engine.cpp
        v19_engine_pybind.cpp
    )
    target_link_libraries(v19_engine PUBLIC TT::Metalium)
    target_include_directories(v19_engine PRIVATE ${TT_TTNN_INCLUDES})
    target_link_libraries(v19_engine PRIVATE
        "${TT_BUILD_USED}/lib/_ttnncpp.so"
        "${TT_BUILD_USED}/lib/libtt_stl.so"
    )
    target_compile_options(v19_engine PRIVATE
        -O2 -Wall
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-comment
        -Wno-deprecated-declarations
        -DSPDLOG_FMT_EXTERNAL
        -fopenmp=libgomp
    )
    target_link_libraries(v19_engine PRIVATE gomp)
    target_compile_definitions(v19_engine PRIVATE
        "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
    )
    set_target_properties(v19_engine PROPERTIES
        CXX_COMPILER_LAUNCHER ""
    )
    message(STATUS "Configured v19_engine pybind11 module")
else()
    message(STATUS "Skipping v19_engine pybind11 module: pybind11 not found "
                   "(pass -Dpybind11_DIR=<dp_env>/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)")
endif()

# ───────────────────────────────────────────────────────────────────────────
# V25 EF backward — L1-resident field, direct-L1-load corner gather microbench.
# Pure tt-metal. Tests whether direct-L1 gather beats CPU's cache-resident EF.
# ───────────────────────────────────────────────────────────────────────────
add_executable(v25_ef_l1gather v25_ef_l1gather_host.cpp)
target_link_libraries(v25_ef_l1gather PUBLIC TT::Metalium)
target_compile_options(v25_ef_l1gather PRIVATE
    -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v25_ef_l1gather PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v25_ef_l1gather PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v25_ef_l1gather")

# V25 Path A — end-to-end EF backward per-iter (fresh grouping + V25 gather).
add_executable(v25_pathA v25_pathA_host.cpp)
target_link_libraries(v25_pathA PUBLIC TT::Metalium)
target_compile_options(v25_pathA PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v25_pathA PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v25_pathA PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v25_pathA")

# V26 EF backward — SFPU fp32 gather variant (BRISC gather + SFPU weighted sum).
add_executable(v26 v26_host.cpp)
target_link_libraries(v26 PUBLIC TT::Metalium)
target_compile_options(v26 PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v26 PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v26 PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v26")

# V27 on-chip cell-bucketing scatter (groups cells by field band for the backward).
add_executable(v27 v27_host.cpp)
target_link_libraries(v27 PUBLIC TT::Metalium)
target_compile_options(v27 PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v27 PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v27 PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v27")

# V28 multi-bin L1-resident EF gather (correct at all grids; generalizes V26).
add_executable(v28 v28_host.cpp)
target_link_libraries(v28 PUBLIC TT::Metalium)
target_compile_options(v28 PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v28 PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v28 PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v28")

# V29 full on-chip EF backward pipeline (prep+bucket → gather+compute → scatter-grad).
add_executable(v29 v29_host.cpp)
target_link_libraries(v29 PUBLIC TT::Metalium)
target_compile_options(v29 PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v29 PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v29 PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v29")

# V30 field-stationary EF backward (transpose of V19 forward: local field slab + per-cell contribution).
add_executable(v30 v30_host.cpp)
target_link_libraries(v30 PUBLIC TT::Metalium)
target_compile_options(v30 PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v30 PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v30 PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v30")

# FCCS field-cast multicast bandwidth microbench (make-or-break for next-gen backward).
add_executable(mcast_bw mcast_bw_host.cpp)
target_link_libraries(mcast_bw PUBLIC TT::Metalium)
target_compile_options(mcast_bw PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(mcast_bw PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(mcast_bw PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured mcast_bw")

# FCCS (Field-Cast, Cell-Stationary) density backward microbench.
add_executable(fccs fccs_host.cpp)
target_link_libraries(fccs PUBLIC TT::Metalium)
target_compile_options(fccs PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(fccs PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(fccs PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured fccs")

# V31 backward microbench (reads forward-stash route → grad, no prep).
add_executable(v31_backward v31_backward_host.cpp)
target_link_libraries(v31_backward PUBLIC TT::Metalium)
target_compile_options(v31_backward PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v31_backward PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v31_backward PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v31_backward")

add_executable(v31_atomic_probe v31_atomic_probe_host.cpp)
target_link_libraries(v31_atomic_probe PUBLIC TT::Metalium)
target_compile_options(v31_atomic_probe PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v31_atomic_probe PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v31_atomic_probe PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v31_atomic_probe")

add_executable(v32_regroup v32_regroup_host.cpp)
target_link_libraries(v32_regroup PUBLIC TT::Metalium)
target_compile_options(v32_regroup PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v32_regroup PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v32_regroup PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v32_regroup")

add_executable(v32_e2e v32_e2e_host.cpp)
target_link_libraries(v32_e2e PUBLIC TT::Metalium)
target_compile_options(v32_e2e PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v32_e2e PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v32_e2e PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v32_e2e")

add_executable(v32d_e2e v32d_e2e_host.cpp)
target_link_libraries(v32d_e2e PUBLIC TT::Metalium)
target_compile_options(v32d_e2e PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v32d_e2e PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v32d_e2e PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v32d_e2e")

add_executable(v33_cellgather v33_cellgather_host.cpp)
target_link_libraries(v33_cellgather PUBLIC TT::Metalium)
target_compile_options(v33_cellgather PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v33_cellgather PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v33_cellgather PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v33_cellgather")

add_executable(v34_reuse v34_reuse_host.cpp)
target_link_libraries(v34_reuse PUBLIC TT::Metalium)
target_compile_options(v34_reuse PRIVATE -O2 -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comment)
target_compile_definitions(v34_reuse PRIVATE "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\"")
set_target_properties(v34_reuse PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured v34_reuse")
