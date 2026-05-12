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
