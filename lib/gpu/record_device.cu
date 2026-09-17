#include "record_device.h"
#include "gpu_sampler.h"
#include "sampler_kernels.h"
#include "local_e.h"
#include "exchange_kernels.h"
#include "backprop.h"

#include <chrono>
#include <cstring>

// Record_one_walker's accumulation <r^2>
__device__ __forceinline__ double dev_walker_r2(const real* xw) {
    double R_cm[dim] = {};
    for (int p = 0; p < N; p++) for (int d = 0; d < dim; d++) R_cm[d] += (double)xw[p*dim + d];
    for (int d = 0; d < dim; d++) R_cm[d] /= N;
    double r2 = 0.0;
    for (int p = 0; p < N; p++) {
        for (int d = 0; d < dim; d++) {
            double diff = (double)xw[p*dim + d] - R_cm[d];
            r2 += diff * diff;
        }
    }
    return r2 / N;
}

// Accumulate stats
__global__ void walker_stats_kernel(const real* __restrict__ E_loc, const real* __restrict__ l2, const unsigned char* __restrict__ valid, const real* __restrict__ x,
                                    double* __restrict__ Ew, double* __restrict__ E2w, double* __restrict__ l2w, double* __restrict__ r2w, int* __restrict__ nw, int B) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B || !valid[w]) return;
    const double E = (double)E_loc[w];
    Ew[w]  += E;
    E2w[w] += E * E;
    l2w[w] += (double)l2[w];
    r2w[w] += dev_walker_r2(x + (std::size_t)w * D);
    nw[w]++;
}

// Log observables
void record_batch_device(DeviceState& ds, cublasHandle_t handle, const Ansatz& a, Workspace& ws, int B, double step, int records, bool with_O, RecordTimes* times, cudaStream_t stream) {
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); };
    RecordTimes tm;

    ds.Ew_d.zero(); ds.E2w_d.zero(); ds.l2w_d.zero(); ds.r2w_d.zero(); ds.nw_d.zero();
    const int threads = 128;

    for (int r = 0; r < records; r++) {
        auto t0 = clk::now();
        for (int sweep = 0; sweep < sweeps_between_records; sweep++) {
            sweep_device(ds, handle, B, step, stream);
            recenter_device(ds, B, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        tm.sweep_ms += ms_since(t0);

        t0 = clk::now();
        // The cached eval: with_O stashes this pass's activations, and nothing
        // between here and assemble_O_batch below moves a walker.
        tm.n_fallback += eval_local_E_device(ds, handle, a, ws, B, stream, with_O);
        pool_write_row(ds.E_loc.d, ds.valid_loc.d, ds.E_pool.d, ds.valid_pool.d, r, B, stream);
        walker_stats_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(ds.E_loc.d, ds.l2_out.d, ds.valid_loc.d, ds.x.d, ds.Ew_d.d, ds.E2w_d.d, ds.l2w_d.d, ds.r2w_d.d, ds.nw_d.d, B);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        tm.localE_ms += ms_since(t0);

        if (with_O) {
            t0 = clk::now();
            assemble_O_batch(ds, handle, r, B, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            tm.o_ms += ms_since(t0);
        }
    }
    if (times) *times = tm;
}

// Accumulate statistics
void download_iteration(DeviceState& ds, int B, int records, PinnedArray& staging, IterStatsHost& out) {
    const std::size_t Bs = (std::size_t)B, Ns = Bs * (std::size_t)records;
    struct Seg { const void* src; std::size_t bytes; };
    const Seg segs[] = {
        { ds.E_pool.d,     Ns * sizeof(double) },
        { ds.valid_pool.d, Ns * sizeof(unsigned char) },
        { ds.Ew_d.d,       Bs * sizeof(double) },
        { ds.E2w_d.d,      Bs * sizeof(double) },
        { ds.l2w_d.d,      Bs * sizeof(double) },
        { ds.r2w_d.d,      Bs * sizeof(double) },
        { ds.nw_d.d,       Bs * sizeof(int) },
        { ds.acc.d,        Bs * sizeof(long long) },
        { ds.sp_acc.d,     Bs * sizeof(long long) },
        { ds.tau_acc.d,    Bs * sizeof(long long) },
    };
    std::size_t total = 0;
    for (const Seg& s : segs) total += s.bytes;
    if (total > ds.pack_d.n) throw std::runtime_error("download_iteration: pack buffer smaller than this iteration's payload");

    // Stage on device, then ONE copy across the bus.
    std::size_t off = 0;
    for (const Seg& s : segs) {
        CUDA_CHECK(cudaMemcpy(ds.pack_d.d + off, s.src, s.bytes, cudaMemcpyDeviceToDevice));
        off += s.bytes;
    }
    staging.ensure(total);
    unsigned char* h = staging.as<unsigned char>();
    ds.pack_d.down(h, total);

    off = 0;
    auto take = [&](void* dst, std::size_t bytes) { std::memcpy(dst, h + off, bytes); off += bytes; };
    out.E_pool.resize(Ns);     take(out.E_pool.data(), Ns * sizeof(double));
    out.valid_pool.resize(Ns); take(out.valid_pool.data(), Ns);
    BatchStats& bs = out.bs;
    bs.Ew_sum.resize(Bs); bs.E2w_sum.resize(Bs); bs.l2w_sum.resize(Bs); bs.r2w_sum.resize(Bs); bs.nw.resize(Bs);
    take(bs.Ew_sum.data(),  Bs * sizeof(double));
    take(bs.E2w_sum.data(), Bs * sizeof(double));
    take(bs.l2w_sum.data(), Bs * sizeof(double));
    take(bs.r2w_sum.data(), Bs * sizeof(double));
    take(bs.nw.data(),      Bs * sizeof(int));
    std::vector<long long> c(Bs);
    out.acc = out.sp_acc = out.tau_acc = 0;
    take(c.data(), Bs * sizeof(long long)); for (long long v : c) out.acc += v;
    take(c.data(), Bs * sizeof(long long)); for (long long v : c) out.sp_acc += v;
    take(c.data(), Bs * sizeof(long long)); for (long long v : c) out.tau_acc += v;

    bs.E_sum = bs.E2_sum = bs.l2_sum = bs.r2_sum = 0.0;
    bs.n_valid = 0;
    for (int w = 0; w < B; w++) {
        bs.E_sum   += bs.Ew_sum[w];
        bs.E2_sum  += bs.E2w_sum[w];
        bs.l2_sum  += bs.l2w_sum[w];
        bs.r2_sum  += bs.r2w_sum[w];
        bs.n_valid += bs.nw[w];
    }
    bs.n_invalid = (long long)records * (long long)B - bs.n_valid;
}
