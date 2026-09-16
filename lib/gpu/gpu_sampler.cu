#include "gpu_sampler.h"
#include "sampler_kernels.h"
#include "eval.h"
#include "det_kernels.h"
#include "local_e.h"
#include "exchange_kernels.h"

#include <algorithm>
#include <chrono>

// Upload walkers and reset acceptance rates
void upload_and_reset(DeviceState& ds, const WalkerBatch& wb, PinnedArray& staging) {
    ds.upload_walkers(wb, staging);
    ds.acc.zero();
    ds.sp_acc.zero();
    ds.tau_acc.zero();
}

static void sweep_device(DeviceState& ds, cublasHandle_t handle, int B, double step, cudaStream_t stream) {
    // Propose new coordinates
    for (int j = 0; j < draws; j++) {
        propose_coord(ds, B, step, stream);
        eval_logp_batch_prop(ds, handle, B, ds.x_prop.d, ds.S_prop.d, ds.logp_prop.d, stream);
        accept_coord(ds, B, stream);
    }

    // Propose discrete steps
    const bool do_spin = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0);
    const bool do_tau  = (tau_mode  == TauMode::Sampled  && N_p > 0 && N_n > 0);
    if (!(do_spin || do_tau)) return;

    build_st_table_batch(ds, handle, B, stream);
    S_from_table_batch(ds, handle, B, ds.s.d, ds.t.d, ds.S_cur.d, stream);

    if (do_spin) {
        for (int j = 0; j < spin_draws; j++) {
            propose_discrete(ds, B, true, stream);
            S_from_table_batch(ds, handle, B, ds.s_prop.d, ds.t.d, ds.S_prop.d, stream);
            accept_discrete(ds, B, true, stream);
        }
    }
    if (do_tau) {
        for (int j = 0; j < tau_draws; j++) {
            propose_discrete(ds, B, false, stream);
            S_from_table_batch(ds, handle, B, ds.s.d, ds.t_prop.d, ds.S_prop.d, stream);
            accept_discrete(ds, B, false, stream);
        }
    }
}

// Thermalization sweeps
double therm_batch_device(DeviceState& ds, cublasHandle_t handle, int B, double step, int n_sweeps, cudaStream_t stream) {
    ds.acc.zero();
    ds.sp_acc.zero();
    ds.tau_acc.zero();

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        sweep_device(ds, handle, B, step, stream);
        recenter_device(ds, B, stream);
    }

    long long acc = 0, sp = 0, tau = 0;
    download_acceptance(ds, B, acc, sp, tau);
    if (n_sweeps <= 0) return 0.0;
    return (double)acc / ((double)B * (double)n_sweeps * (double)draws);
}

// Measure per step
void record_batch_hybrid(DeviceState& ds, cublasHandle_t handle, WalkerBatch& wb, const Ansatz& a, double step, int records, ThreadPool* pool, std::vector<Workspace>& wss, PinnedArray& staging, std::vector<double>& E_pool, std::vector<double>& O_pool, std::vector<uint8_t>& valid_pool, BatchStats& bs, HybridTimes* times, cudaStream_t stream) {
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); };

    const int B = wb.B;
    const int n_workers = (int)wss.size();
    const std::size_t P = a.n_params();

    bs.Ew_sum.assign(B, 0.0);
    bs.E2w_sum.assign(B, 0.0);
    bs.l2w_sum.assign(B, 0.0);
    bs.r2w_sum.assign(B, 0.0);
    bs.nw.assign(B, 0);

    HybridTimes tm;
    std::vector<real> E_h((std::size_t)B), l2_h((std::size_t)B);
    std::vector<uint8_t> v_h((std::size_t)B);

    for (int r = 0; r < records; r++) {
        auto t0 = clk::now();
        for (int sweep = 0; sweep < sweeps_between_records; sweep++) {
            sweep_device(ds, handle, B, step, stream);
            recenter_device(ds, B, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        tm.sweep_ms += ms_since(t0);

        t0 = clk::now();
        tm.n_fallback += eval_local_E_device(ds, handle, a, wss[0], B, stream);
        pool_write_row(ds.E_loc.d, ds.valid_loc.d, ds.E_pool.d, ds.valid_pool.d, r, B, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        tm.localE_ms += ms_since(t0);

        t0 = clk::now();
        ds.download_walkers(wb, staging);
        ds.E_loc.down(E_h.data(), (std::size_t)B);
        ds.l2_out.down(l2_h.data(), (std::size_t)B);
        ds.valid_loc.down(v_h.data(), (std::size_t)B);
        tm.download_ms += ms_since(t0);

        // Stats and host pool rows, in walker order, with record_one_walker's own
        // accumulation expressions -- the Prompt 2.0 bit-determinism convention.
        for (int w = 0; w < B; w++) {
            const std::size_t idx = (std::size_t)r * B + w;
            valid_pool[idx] = v_h[w];
            if (!v_h[w]) continue;
            const double E_loc = (double)E_h[w];
            E_pool[idx] = E_loc;
            bs.Ew_sum[w]  += E_loc;
            bs.E2w_sum[w] += E_loc * E_loc;
            bs.l2w_sum[w] += (double)l2_h[w];
            bs.r2w_sum[w] += walker_r2_pub(&wb.x[(std::size_t)w * D]);
            bs.nw[w]++;
        }

        // The only per-sample host work left: O, for valid samples only.
        t0 = clk::now();
        pool->run([&](int th) {
            int w0, w1;
            chunk_range_pub(B, n_workers, th, w0, w1);
            Workspace& ws = wss[th];
            std::vector<double> O(P);
            for (int w = w0; w < w1; w++) {
                if (!v_h[w]) continue;
                assemble_O(&wb.x[(std::size_t)w*D], &wb.s[(std::size_t)w*N], &wb.t[(std::size_t)w*N], a, ws, O);
                const std::size_t idx = (std::size_t)r * B + w;
                for (std::size_t k = 0; k < P; k++) O_pool[idx*P + k] = O[k];
            }
        });
        tm.o_ms += ms_since(t0);
    }

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
    if (times) *times = tm;
}
