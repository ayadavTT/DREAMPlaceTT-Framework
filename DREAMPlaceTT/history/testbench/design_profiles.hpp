// SPDX-License-Identifier: Apache-2.0
//
// design_profiles.hpp — realistic problem-size profiles for the synthetic
// kernel testbenches. Cell counts are the REAL ISPD2005 movable-node counts
// (from benchmarks/ispd2005/<d>/<d>.nodes headers). Cell-size model is a
// representative bin-space distribution: the vast majority of standard cells
// are sub-bin, a small tail of macros span several bins (drives multi-bin
// overlap / atomic-contention behavior).
//
// Shared by every per-kernel testbench under history/. NOT a runner — a library.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tb {

struct DesignProfile {
    const char* name;
    uint32_t    n_cells;     // real ISPD2005 NumNodes (movable+std cells)
};

// Real per-design cell counts (ISPD2005).
inline const std::vector<DesignProfile>& designs() {
    static const std::vector<DesignProfile> d = {
        {"adaptec1",  211447},
        {"adaptec2",  255023},
        {"adaptec3",  451650},
        {"bigblue1",  278164},
        {"bigblue2",  557866},
        {"bigblue3", 1096812},
    };
    return d;
}

inline const DesignProfile* find_design(const std::string& name) {
    for (auto& p : designs()) if (name == p.name) return &p;
    return nullptr;
}

// Standard grids (num_bins per axis).
inline const std::vector<uint32_t>& grids() {
    static const std::vector<uint32_t> g = {512u, 1024u, 2048u};
    return g;
}

// Spatial cell distributions to exercise.
enum class Dist { Uniform, Cluster, Clouds };

inline const char* dist_name(Dist d) {
    switch (d) {
        case Dist::Uniform: return "uniform";
        case Dist::Cluster: return "cluster";   // one large dense blob
        case Dist::Clouds:  return "clouds";     // several blobs ("clouds")
    }
    return "uniform";
}
inline bool parse_dist(const std::string& s, Dist& out) {
    if (s == "uniform") { out = Dist::Uniform; return true; }
    if (s == "cluster") { out = Dist::Cluster; return true; }
    if (s == "clouds")  { out = Dist::Clouds;  return true; }
    return false;
}

}  // namespace tb
