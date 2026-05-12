#!/usr/bin/env python3
"""
scatter_ttnn_worker.py — Docker-side wrapper for density_scatter_ttnn_server.

Sets up TT-Metal environment and exec's the C++ binary for Path 1:
V4 scatter + gather-density-only + TTNN C++ DCT.

Called by scatter_ttnn_client.py via:
    docker exec <container> python3 scatter_ttnn_worker.py \
        --server M N NC_max ipc_dir [xl yl xh yh]
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT     = os.path.dirname(_THIS_DIR)
# DREAMPLACE_TT_SERVER_BINARY env var overrides the default.
_BINARY = os.environ.get(
    "DREAMPLACE_TT_SERVER_BINARY",
    os.path.join(_ROOT, "host", "build", "density_scatter_ttnn_server"),
)
BINARY = os.path.realpath(_BINARY)

if not os.path.exists(BINARY):
    sys.exit(
        f"[scatter_ttnn_worker] Binary not found: {BINARY}\n"
        "Build it with the framework's build script:\n"
        "  bash scripts/build_server.sh\n"
        "or manually:\n"
        "  cd host && mkdir -p build && cd build\n"
        "  cmake .. -DCMAKE_CXX_COMPILER=clang++-20 -DTT_METAL_HOME=$TT_METAL_HOME\n"
        "  make -j$(nproc)"
    )

args = sys.argv[1:]
if not args or args[0] != "--server":
    sys.exit(f"Usage: {sys.argv[0]} --server M N NC_max ipc_dir [xl yl xh yh]")
args = args[1:]  # strip --server

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.join(_ROOT, "tt-metal"))

env = {
    **os.environ,
    "TT_METAL_HOME":         TT_METAL_HOME,
    "TT_METAL_RUNTIME_ROOT": TT_METAL_HOME,
    "ARCH_NAME": os.environ.get("ARCH_NAME", "blackhole"),
    "LD_LIBRARY_PATH": (
        f"{TT_METAL_HOME}/build_Release/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    ),
    "TT_METAL_LOGGER_LEVEL": os.environ.get("TT_METAL_LOGGER_LEVEL", "WARNING"),
    "TT_METAL_DEVICE_PROFILER": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
}

print(f"[scatter_ttnn_worker] Exec: {BINARY} {' '.join(args)}", flush=True)
os.execve(BINARY, [BINARY] + args, env)
