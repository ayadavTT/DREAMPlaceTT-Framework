// SPDX-License-Identifier: Apache-2.0
//
// report.hpp — shared correctness+performance report accumulator for the
// per-kernel testbenches. Each testbench appends one Row per (design,grid,dist)
// config, then writes results.csv. The per-zone Tracy table (zones.txt) is
// produced separately by the run script (TT_METAL_DEVICE_PROFILER=1 +
// tools/profile_v11.py) and referenced from REPORT.md.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace tb {

struct Row {
    std::string design;
    uint32_t    grid = 0;
    std::string dist;
    uint32_t    n_cells = 0;
    double      rel_l2 = 0.0;     // vs CPU fp64 reference
    bool        pass = false;     // rel_l2 < tol
    double      kernel_ms = 0.0;  // measured device/kernel time (median)
    double      host_ref_ms = 0.0;
    std::string note;
};

class Report {
public:
    explicit Report(std::string kernel) : kernel_(std::move(kernel)) {}
    void add(const Row& r) { rows_.push_back(r); }

    void write_csv(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "report: cannot open %s\n", path.c_str()); return; }
        std::fprintf(f, "kernel,design,grid,dist,n_cells,rel_l2,pass,kernel_ms,host_ref_ms,note\n");
        for (auto& r : rows_)
            std::fprintf(f, "%s,%s,%u,%s,%u,%.3e,%d,%.4f,%.4f,%s\n",
                         kernel_.c_str(), r.design.c_str(), r.grid, r.dist.c_str(),
                         r.n_cells, r.rel_l2, r.pass ? 1 : 0, r.kernel_ms,
                         r.host_ref_ms, r.note.c_str());
        std::fclose(f);
        std::printf("[report] wrote %s (%zu configs)\n", path.c_str(), rows_.size());
    }

    void print_summary() const {
        int npass = 0; for (auto& r : rows_) npass += r.pass ? 1 : 0;
        std::printf("\n=== %s : %d/%zu configs PASS ===\n", kernel_.c_str(), npass, rows_.size());
        std::printf("%-10s %5s %-8s %9s %10s %8s  %s\n",
                    "design","grid","dist","n_cells","rel_l2","ms","pass");
        for (auto& r : rows_)
            std::printf("%-10s %5u %-8s %9u %10.2e %8.3f  %s\n",
                        r.design.c_str(), r.grid, r.dist.c_str(), r.n_cells,
                        r.rel_l2, r.kernel_ms, r.pass ? "OK" : "FAIL");
    }

    const std::vector<Row>& rows() const { return rows_; }
private:
    std::string kernel_;
    std::vector<Row> rows_;
};

}  // namespace tb
