#include "prof.h"

#include <cstdio>
#include <cstring>
#include <ctime>

#ifndef VMC_GIT_REV
#define VMC_GIT_REV "unknown"
#endif

// Getter for timings
GpuProf& GpuProf::get() { static GpuProf p; return p; }

// Destructor
GpuProf::~GpuProf() {
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
}

// Makes a row for the new range of times
std::string GpuProf::make_path(const char* name) const {
    if (name[0] == '/') return std::string(name + 1);             
    if (stack_.empty()) return std::string(name);
    return stack_.back().path + "/" + name;
}

// Finds the index for the row of times 
int GpuProf::row_index(const std::string& path, bool device) {
    auto it = index_.find(path);
    if (it != index_.end()) return it->second;
    ProfRow r;
    r.path   = path;
    r.device = device;
    for (char c : path) if (c == '/') r.depth++;
    rows_.push_back(r);
    const int idx = (int)rows_.size() - 1;
    index_.emplace(path, idx);
    return idx;
}

// Takes from allocated CUDA memory for events and recycles it 
cudaEvent_t GpuProf::take_event() {
    if (ev_used_ == events_.size()) {
        cudaEvent_t e = nullptr;
        CUDA_CHECK(cudaEventCreate(&e));
        events_.push_back(e);
    }
    return events_[ev_used_++];
}

// Start a GPU time range
void GpuProf::push(const char* name, cudaStream_t stream) {
    Open o;
    o.path    = make_path(name);
    o.stream  = stream;
    o.t0      = clock::now();
    o.launch0 = launch_counter().load(std::memory_order_relaxed);

    // Depth cap: a range nested deeper than this is opened and closed but not measured, so instrumentation inside a leaf helper costs nothing when that helper is called from deep inside the sweep.
    if ((int)stack_.size() < prof_max_depth) {
        o.row = row_index(o.path, true);
        o.ev  = take_event();
        CUDA_CHECK(cudaEventRecord(o.ev, stream));
#ifdef VMC_NVTX
        nvtxRangePushA(o.path.c_str());
#endif
    }
    stack_.push_back(std::move(o));
}

// Start a CPU time range
void GpuProf::push_host(const char* name) {
    Open o;
    o.path    = make_path(name);
    o.stream  = nullptr;
    o.t0      = clock::now();
    o.launch0 = launch_counter().load(std::memory_order_relaxed);
    if ((int)stack_.size() < prof_max_depth) {
        o.row = row_index(o.path, false);
#ifdef VMC_NVTX
        nvtxRangePushA(o.path.c_str());
#endif
    }
    stack_.push_back(std::move(o));
}

// Kill the innermost range
void GpuProf::pop() {
    if (stack_.empty()) return;
    Open o = std::move(stack_.back());
    stack_.pop_back();
    if (o.row < 0) return;

#ifdef VMC_NVTX
    nvtxRangePop();
#endif
    ProfRow& r = rows_[o.row];
    r.calls++;
    r.launches += launch_counter().load(std::memory_order_relaxed) - o.launch0;
    r.host_ms  += std::chrono::duration<double, std::milli>(clock::now() - o.t0).count();

    if (o.ev) {
        cudaEvent_t e1 = take_event();
        CUDA_CHECK(cudaEventRecord(e1, o.stream));
        pending_.push_back({o.row, o.ev, e1});
    }
}

// Collect all GPU times
void GpuProf::iteration_end(double ms_iter) {
    if (!pending_.empty()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        for (const Pending& p : pending_) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, p.e0, p.e1));
            rows_[p.row].gpu_ms += (double)ms;
        }
        pending_.clear();
    }
    ev_used_ = 0;            
    iter_ms_ += ms_iter;
    iters_++;
}

// Clears all accumulated numbers
void GpuProf::reset() {
    pending_.clear();
    ev_used_ = 0;
    rows_.clear();
    index_.clear();
    iter_ms_ = 0.0;
    iters_   = 0;
}

// Find specs on ratio of device processing of fp64 and fp32 operations
struct SmCores { int fp64, fp32; };
static SmCores cores_per_sm(int major, int minor) {
    switch (major * 10 + minor) {
        case 60:                     return {32, 64};    // GP100
        case 61: case 62:            return { 4, 128};   // GP10x consumer
        case 70: case 72:            return {32, 64};    // Volta
        case 75:                     return { 2, 64};    // Turing
        case 80:                     return {32, 64};    // A100
        case 86: case 87: case 89:   return { 2, 128};   // GA10x / Ada consumer
        case 90:                     return {64, 128};   // H100
        default:                     return { 0, 0};
    }
}

// Write the time table
void GpuProf::report(const char* file, int B, int records, std::size_t P, const char* tag) const {
    if (iters_ == 0) return;

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const SmCores cores = cores_per_sm(prop.major, prop.minor);
    const double  ghz   = prop.clockRate / 1.0e6;
    const double  tf64  = 2.0 * cores.fp64 * prop.multiProcessorCount * ghz / 1000.0;
    const double  tf32  = 2.0 * cores.fp32 * prop.multiProcessorCount * ghz / 1000.0;

    char when[64];
    std::time_t t = std::time(nullptr);
    std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", std::localtime(&t));

    const double ms_iter = iter_ms_ / (double)iters_;

    std::string out;
    char line[512];
    auto add = [&](const char* s) { out += s; };

    std::snprintf(line, sizeof(line), "\n### prof %s | rev %s | %s\n", when, VMC_GIT_REV, tag ? tag : "");
    add(line);
    if (cores.fp64 > 0) {
        std::snprintf(line, sizeof(line),
                      "card: %s, sm_%d%d, %d SM, %.2f GHz, FP64 peak ~%.3f TF (est: %d FP64/SM), FP64:FP32 = 1:%.0f\n",
                      prop.name, prop.major, prop.minor, prop.multiProcessorCount, ghz, tf64, cores.fp64, tf32 / tf64);
    } else {
        std::snprintf(line, sizeof(line), "card: %s, sm_%d%d, %d SM, %.2f GHz, FP64 peak unknown for this cc\n",
                      prop.name, prop.major, prop.minor, prop.multiProcessorCount, ghz);
    }
    add(line);
    std::snprintf(line, sizeof(line),
                  "config: B=%d records=%d sweeps/iter=%d N=%d K=%d m_feat=%d P=%zu jet_chunk=%d real=%s\n",
                  B, records, therm_re_sweep + records * sweeps_between_records, N, K, m_feat, P, jet_chunk,
                  sizeof(real) == 8 ? "fp64" : "fp32");
    add(line);
    std::snprintf(line, sizeof(line), "iterations profiled: %d, mean %.2f ms/iter\n", iters_, ms_iter);
    add(line);
    add("rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the\n");
    add("profiled stream; host_ms is the wall time the host spent inside the range. host_ms much\n");
    add("larger than gpu_ms means the host is not keeping the device fed (launch latency, or a\n");
    add("blocking copy); host_ms much smaller means the range only enqueued work.\n\n");
    add("| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |\n");
    add("|---|---|---:|---:|---:|---:|---:|---:|\n");

    for (const ProfRow& r : rows_) {
        const double tot = r.device ? r.gpu_ms : r.host_ms;
        std::snprintf(line, sizeof(line), "| %s | %s | %.3f | %.3f | %.2f | %.1f | %.1f | %.2f |\n",
                      r.path.c_str(), r.device ? "gpu" : "host",
                      r.device ? r.gpu_ms / iters_ : 0.0, r.host_ms / iters_,
                      100.0 * tot / (iter_ms_ > 0.0 ? iter_ms_ : 1.0), tot,
                      (double)r.launches / iters_, (double)r.calls / iters_);
        add(line);
    }

    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
    if (file && file[0]) {
        if (std::FILE* f = std::fopen(file, "a")) {
            std::fputs(out.c_str(), f);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "prof: could not append to %s\n", file);
        }
    }
}
