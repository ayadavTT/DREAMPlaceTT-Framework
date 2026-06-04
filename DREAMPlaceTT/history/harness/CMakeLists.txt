cmake_minimum_required(VERSION 3.16)
project(dreamplacett_history_harnesses CXX)
# Builds the isolated per-kernel test harnesses for the history library.
# Self-contained: uses this folder's vendored tt-metal + kernels.
# Configure with TT_METAL_HOME = <DREAMPlaceTT>/tt-metal (see build_harnesses.sh).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    else()
        get_filename_component(TT_METAL_HOME "${CMAKE_SOURCE_DIR}/../../tt-metal" ABSOLUTE)
    endif()
endif()
set(TT_BUILD "${TT_METAL_HOME}/build_Release")
list(PREPEND CMAKE_PREFIX_PATH "${TT_BUILD}/lib/cmake/tt-metalium")
list(APPEND CMAKE_PREFIX_PATH "${TT_BUILD}")   # so transitive deps (fmt/nlohmann_json/... under lib|share/cmake) resolve
find_package(TT-Metalium REQUIRED PATHS "${TT_BUILD}/lib/cmake/tt-metalium" NO_DEFAULT_PATH)

get_filename_component(_DPTT "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)   # DREAMPlaceTT
set(HOST "${_DPTT}/host")          # harness .cpp live here
set(KDIR "${_DPTT}/kernels/")      # JIT kernel source dir

# Each harness = one standalone binary from one host .cpp. Pure TT-Metal
# (no TTNN). DENSITY_KERNEL_DIR points at the in-folder kernels/.
# NOTE: v31_backward is intentionally omitted from the default-run list in
# run_harnesses.sh — it wedges the device standalone (needs the forward stash).
set(HARNESSES
    v25_ef_l1gather:v25_ef_l1gather_host.cpp
    v26:v26_host.cpp
    v27:v27_host.cpp
    v28:v28_host.cpp
    v35:v35_host.cpp
    v35live:v35live_host.cpp
    v29:v29_host.cpp
    v30:v30_host.cpp
    fccs:fccs_host.cpp
    v32_regroup:v32_regroup_host.cpp
    v31_backward:v31_backward_host.cpp
    atomic_bench_host:atomic_bench_host.cpp
    mcast_bw:mcast_bw_host.cpp
    v11op_bench_host:v11op_bench_host.cpp
    v19_microbench_host:v19_microbench_host.cpp
)
foreach(pair ${HARNESSES})
    string(REPLACE ":" ";" parts "${pair}")
    list(GET parts 0 tgt)
    list(GET parts 1 srcf)
    if(EXISTS "${HOST}/${srcf}")
        add_executable(${tgt} "${HOST}/${srcf}")
        target_link_libraries(${tgt} PUBLIC TT::Metalium)
        target_compile_options(${tgt} PRIVATE -O2 -Wall -Wno-unused-variable
            -Wno-unused-but-set-variable -Wno-comment)
        target_compile_definitions(${tgt} PRIVATE "DENSITY_KERNEL_DIR=\"${KDIR}\"")
        set_target_properties(${tgt} PROPERTIES CXX_COMPILER_LAUNCHER "")
    endif()
endforeach()
message(STATUS "history harnesses: ${HARNESSES}")
