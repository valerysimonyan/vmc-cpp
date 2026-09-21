#pragma once



#include "gpu_util.h"
#include "../constants.h"
#include "../precision.h"

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// If CUDA >= 11 can include time ranges
#ifdef VMC_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

struct ProfRow {
    std::string path;
    double      gpu_ms   = 0.0;   // cudaEvent elapsed, summed over calls
    double      host_ms  = 0.0;   // steady_clock elapsed, summed over calls
    long long   launches = 0;     // kernel launches issued inside the range
    long long   calls    = 0;
    int         depth    = 0;     // number of '/' in path, for readability
    bool        device   = true;  // false for host-only (wall-clock) ranges
};

class GpuProf {
public:
    static GpuProf& get();

    void push(const char* name, cudaStream_t stream);
    void push_host(const char* name);          // host wall only: no events
    void pop();                                // stream is remembered from push

    // Close an iteration: resolve every event pair recorded since the last call
    // and recycle the event pool. ms_iter is the loop's own wall time.
    void iteration_end(double ms_iter);

    // Print to stdout and append to `file` (created if absent).
    void report(const char* file, int B, int records, std::size_t P, const char* tag) const;

    void reset();                              // drop accumulators (warm-up)
    int  iterations() const { return iters_; }

    ~GpuProf();

private:
    using clock = std::chrono::steady_clock;

    struct Open {
        std::string  path;
        int          row = -1;                 // -1: range skipped (too deep)
        cudaStream_t stream = nullptr;
        cudaEvent_t  ev = nullptr;
        clock::time_point t0;
        long long    launch0 = 0;
    };
    struct Pending { int row; cudaEvent_t e0, e1; };

    int         row_index(const std::string& path, bool device);
    std::string make_path(const char* name) const;
    cudaEvent_t take_event();

    std::vector<Open>        stack_;
    std::vector<Pending>     pending_;
    std::vector<cudaEvent_t> events_;
    std::size_t              ev_used_ = 0;

    std::vector<ProfRow>                 rows_;
    std::unordered_map<std::string, int> index_;

    double iter_ms_ = 0.0;
    int    iters_   = 0;
};

struct ProfScope {
    ProfScope(const char* name, cudaStream_t stream) {
        if constexpr (prof_enabled) GpuProf::get().push(name, stream);
    }
    ~ProfScope() {
        if constexpr (prof_enabled) GpuProf::get().pop();
    }
    ProfScope(const ProfScope&) = delete;
    ProfScope& operator=(const ProfScope&) = delete;
};

struct ProfScopeHost {
    explicit ProfScopeHost(const char* name) {
        if constexpr (prof_enabled) GpuProf::get().push_host(name);
    }
    ~ProfScopeHost() {
        if constexpr (prof_enabled) GpuProf::get().pop();
    }
    ProfScopeHost(const ProfScopeHost&) = delete;
    ProfScopeHost& operator=(const ProfScopeHost&) = delete;
};

inline void prof_iteration_end(double ms_iter) {
    if constexpr (prof_enabled) GpuProf::get().iteration_end(ms_iter);
}
inline void prof_report(int B, int records, std::size_t P, const char* tag) {
    if constexpr (prof_enabled) GpuProf::get().report(prof_report_file, B, records, P, tag);
}
inline void prof_reset() {
    if constexpr (prof_enabled) GpuProf::get().reset();
}

// Merge the two strings
#define VMC_PROF_CAT2(a, b) a##b
#define VMC_PROF_CAT(a, b)  VMC_PROF_CAT2(a, b)

// VMC_PROF("name", stream) opens a device-timed range until the end of the
// enclosing scope; VMC_PROF_HOST("name") opens a host wall-clock one.
#define VMC_PROF(name, stream) ProfScope     VMC_PROF_CAT(vmc_prof_,  __LINE__)(name, stream)
#define VMC_PROF_HOST(name)    ProfScopeHost VMC_PROF_CAT(vmc_profh_, __LINE__)(name)