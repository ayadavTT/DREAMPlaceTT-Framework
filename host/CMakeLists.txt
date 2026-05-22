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
