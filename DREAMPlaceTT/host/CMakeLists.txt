cmake_minimum_required(VERSION 3.16)
project(dreamplacett_forward CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════════════════════════════════════
# DREAMPlaceTT — forward-only build.
#
# This is the cleaned-up, forward-only cut of the DREAMPlace-TT framework. It
# builds exactly two artifacts, both running the *forward* density-map
# construction (scatter) + DCT path on Blackhole:
#
#   1. density_scatter_ttnn_server  — IPC server (docker-exec / shm path,
#                                      used by --device scatter_ttnn[_direct]).
#   2. v19_engine (pybind module)   — in-process engine (used by
#                                      --device scatter_ttnn_inprocess), the
#                                      production fast path.
#
# All electric-force / backward / optimizer / FCCS code has been removed; the
# pybind exposes only the forward surface (scatter, configure_cells,
# scatter_from_pos, profiler + field-addr helpers).
#
# GATHER_MODE=v19 is the only supported forward mode. Clamped vs non-clamped
# cell sizing is selected at runtime by the caller (see DREAMPlaceTT/README.md).
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Locate TT-Metal SDK (TT_METAL_HOME).
# Either set the TT_METAL_HOME env var or pass -DTT_METAL_HOME=<path>.
# Default falls back to ../tt-metal/ relative to this folder (the symlink).
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

# The server/engine JIT-compile kernels at runtime by reading their source
# files. DENSITY_KERNEL_DIR points to this folder's kernels/ directory.
get_filename_component(_FW_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

set(TT_COMMON_OPTS
    -O2 -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-comment
    -Wno-deprecated-declarations
    -DSPDLOG_FMT_EXTERNAL
)

# ───────────────────────────────────────────────────────────────────────────
# 1. density_scatter_ttnn_server — forward IPC server (scatter + accum + DCT).
#    Requires clang-20 (TTNN headers use C++23 reflection).
# ───────────────────────────────────────────────────────────────────────────
add_executable(density_scatter_ttnn_server
    density_scatter_ttnn_server_host.cpp
)
target_link_libraries(density_scatter_ttnn_server PUBLIC TT::Metalium)
target_include_directories(density_scatter_ttnn_server PRIVATE ${TT_TTNN_INCLUDES})
target_link_libraries(density_scatter_ttnn_server PRIVATE
    "${TT_BUILD_USED}/lib/_ttnncpp.so"
    "${TT_BUILD_USED}/lib/libtt_stl.so"
)
target_compile_options(density_scatter_ttnn_server PRIVATE ${TT_COMMON_OPTS})
target_compile_definitions(density_scatter_ttnn_server PRIVATE
    "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
)
set_target_properties(density_scatter_ttnn_server PROPERTIES CXX_COMPILER_LAUNCHER "")
message(STATUS "Configured density_scatter_ttnn_server")
message(STATUS "  Kernels dir : ${_FW_ROOT}/kernels/")

# ───────────────────────────────────────────────────────────────────────────
# 2. v19_engine — in-process pybind11 module (forward-only).
#    Built only when pybind11 + python3.8 are discoverable (DREAMPlace dp_env).
#
#    Build (inside the tt-metal container, where clang-20 lives):
#      cmake -Dpybind11_DIR=<dp_env>/lib/python3.8/site-packages/pybind11/share/cmake/pybind11
#            -DPython3_EXECUTABLE=<dp_env>/bin/python3 ..
#      cmake --build . --target v19_engine -j16
# ───────────────────────────────────────────────────────────────────────────
# Force pybind11 to use modern FindPython (honors -DPython_EXECUTABLE) instead of
# the legacy FindPythonLibs, which can grab a stray system/uv python (e.g. a
# cpython-3.10) and build a .so the dp_env python3.8 can't import.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(v19_engine MODULE
        v19_engine.cpp
        v19_engine_pybind.cpp
    )
    target_link_libraries(v19_engine PUBLIC TT::Metalium)
    target_include_directories(v19_engine PRIVATE ${TT_TTNN_INCLUDES})
    target_link_libraries(v19_engine PRIVATE
        "${TT_BUILD_USED}/lib/_ttnncpp.so"
        "${TT_BUILD_USED}/lib/libtt_stl.so"
    )
    target_compile_options(v19_engine PRIVATE ${TT_COMMON_OPTS} -fopenmp=libgomp)
    target_link_libraries(v19_engine PRIVATE gomp)
    target_compile_definitions(v19_engine PRIVATE
        "DENSITY_KERNEL_DIR=\"${_FW_ROOT}/kernels/\""
    )
    set_target_properties(v19_engine PROPERTIES CXX_COMPILER_LAUNCHER "")
    message(STATUS "Configured v19_engine pybind11 module (forward-only)")
else()
    message(STATUS "Skipping v19_engine pybind11 module: pybind11 not found "
                   "(pass -Dpybind11_DIR=<dp_env>/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)")
endif()
