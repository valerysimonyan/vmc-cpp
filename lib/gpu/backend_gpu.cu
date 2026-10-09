#include "../train.h"
#include "../constants.h"
#include "../seed.h"
#include "arena.h"
#include "eval.h"
#include "gpu_sampler.h"
#include "sampler_kernels.h"
#include "record_device.h"
#include "sr_device.h"
#include "planner.h"
#include "prof.h"
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Tune acceptance rate
static void tune_step(double acceptance, double& step) {
    if (acceptance < 0.45) step *= 0.9;
    else if (acceptance > 0.55) step *= 1.1;
}

// Stores per device information
struct Replica {
    int dev = 0;                                     // Device ID
    int B = 0;                                       // Walker count
    unsigned long long salt = 0;                     // Seed
    WalkerBatch wb;                                  // Batch of walkers
    std::vector<Workspace> wss;                      // Contains workspace for CPU
    std::unique_ptr<DeviceState> ds;                 // Contains workspace for GPU
    PinnedArray staging;                             // Staging memory to push to GPU and back
    cublasHandle_t h = nullptr;                      // Handle information
    IterStatsHost it;                                // Contains pool information
    double step = step0, acc = 0.0, therm_ms = 0.0;  // step size, acceptance rate, and thermalization time
    RecordTimes rt;                                  // struct containing times
    long long bytes_up = 0, bytes_dn = 0;            // Transfer bytes in this round
};

// Thermalization per devicewalkers
static double therm_init_tuned_dev(Replica& R) {
    const int B = R.B;
    int per_block = therm_steps_init / therm_init_blocks, remainder = therm_steps_init % therm_init_blocks;
    double acc = 0.0;
    int done = 0;
    for (int b = 0; b < therm_init_blocks; b++) {
        int sweeps = per_block + (b < remainder ? 1 : 0);
        if (sweeps <= 0) continue;
        acc = therm_batch_device(*R.ds, R.h, B, R.step, sweeps);
        tune_step(acc, R.step);
        done += sweeps;
        std::cout << "  therm[gpu" << R.dev << "] " << done << "/" << therm_steps_init << " sweeps: acceptance " << acc << ", step " << R.step << std::endl;
    }
    return acc;
}

// Run all functions across GPUs
template <typename F>
void run_all(std::vector<Replica>& Rs, F&& f) {
    std::vector<std::exception_ptr> err(Rs.size());  // Make a slot per replica for an error
    // Place per thread a different device, seed it once per thread, then run the job, catch errors
    std::vector<std::thread> th;      
    for (std::size_t k = 1; k < Rs.size(); k++) {
        th.emplace_back([&, k]() {
            try {
                CUDA_CHECK(cudaSetDevice(Rs[k].dev));
                vmc_seed_salt = Rs[k].salt;
                f(Rs[k]);
            } catch (...) { 
                err[k] = std::current_exception(); 
            }
        });
    }
    // 0th device runs on the calling thread
    try {
        CUDA_CHECK(cudaSetDevice(Rs[0].dev));
        vmc_seed_salt = Rs[0].salt;
        f(Rs[0]);
    } catch (...) { 
        err[0] = std::current_exception(); 
    }
    // Synchronize accrose threads, go back to 0'th GPU 
    for (std::thread& t : th) t.join();
    CUDA_CHECK(cudaSetDevice(Rs[0].dev));
    vmc_seed_salt = Rs[0].salt;
    // Throw the first error found
    for (std::exception_ptr& e : err) if (e) std::rethrow_exception(e);
}

// Merge statistics acoross devices, do everything on GPU 0
void merge_stats(const std::vector<Replica>& Rs, BatchStats& bs, std::vector<double>& E, std::vector<unsigned char>& valid) {
    bs = Rs[0].it.bs;
    E = Rs[0].it.E_pool;
    valid = Rs[0].it.valid_pool;
    for (std::size_t k = 1; k < Rs.size(); k++) {
        const IterStatsHost& b = Rs[k].it;
        bs.E_sum += b.bs.E_sum; bs.E2_sum += b.bs.E2_sum; bs.l2_sum += b.bs.l2_sum; bs.r2_sum += b.bs.r2_sum;
        bs.n_valid += b.bs.n_valid; bs.n_invalid += b.bs.n_invalid;
        bs.Ew_sum.insert(bs.Ew_sum.end(), b.bs.Ew_sum.begin(), b.bs.Ew_sum.end());
        bs.E2w_sum.insert(bs.E2w_sum.end(), b.bs.E2w_sum.begin(), b.bs.E2w_sum.end());
        bs.l2w_sum.insert(bs.l2w_sum.end(), b.bs.l2w_sum.begin(), b.bs.l2w_sum.end());
        bs.r2w_sum.insert(bs.r2w_sum.end(), b.bs.r2w_sum.begin(), b.bs.r2w_sum.end());
        bs.nw.insert(bs.nw.end(), b.bs.nw.begin(), b.bs.nw.end());
        E.insert(E.end(), b.E_pool.begin(), b.E_pool.end());
        valid.insert(valid.end(), b.valid_pool.begin(), b.valid_pool.end());
    }
}


class GpuBackend : public Backend {
public:
    GpuBackend(const Ansatz& a, bool training) : P_(a.n_params()), pool_(n_thread) {
        const bool same_seed = std::getenv("VMC_MG_SAME_SEED") && std::string(std::getenv("VMC_MG_SAME_SEED")) == "1";
        // Which GPUs, with how many walkers each:
        //   VMC_GPUS=auto  every usable GPU, walkers from its free memory (planner)
        //   VMC_GPUS=one   the single GPU on which the planner fits the most walkers
        //   VMC_NGPU=2     GPUs 0 and 1 with n_walkers each
        //   (default)      one GPU (VMC_CUDA_DEVICE, else the one with the most free memory) with n_walkers
        const char* gp = std::getenv("VMC_GPUS");
        const char* ng = std::getenv("VMC_NGPU");
        std::vector<GpuPlan> plan;
        if (gp && std::string(gp) == "auto") {
            plan = plan_gpus(a);
        } else if (gp && std::string(gp) == "one") {
            const std::vector<GpuPlan> all = plan_gpus(a);
            plan.push_back(*std::max_element(all.begin(), all.end(), [](const GpuPlan& x, const GpuPlan& y) { return x.B < y.B; }));
        } else if (ng && std::string(ng) == "2") {
            for (int d = 0; d < 2; d++) {
                GpuPlan g;
                g.dev = d;
                g.B = (std::size_t)n_walkers;
                plan.push_back(g);
            }
        } else {
            GpuPlan g;
            g.dev = gpu_select_device(true);
            g.B = (std::size_t)n_walkers;
            plan.push_back(g);
        }
        if (std::getenv("VMC_PLAN_ONLY")) {
            std::printf("plan only: arena for n_walkers=%d would be %.3f GiB\n", n_walkers, (double)arena_bytes_for(a, (std::size_t)n_walkers) / (1024.0*1024*1024));
            std::exit(0);
        }

        Rs_.resize(plan.size());
        for (std::size_t k = 0; k < plan.size(); k++) {
            Rs_[k].dev = plan[k].dev;
            Rs_[k].B = (int)plan[k].B;
            Rs_[k].salt = (k == 0 || same_seed) ? 0ULL : splitmix64(0x6d67ULL + k);
            B_tot_ += Rs_[k].B;
        }
        std::cout << "GPU backend: " << Rs_.size() << " GPU(s), walkers";
        for (const Replica& R : Rs_) std::cout << " GPU" << R.dev << ":" << R.B;
        std::cout << " (total " << B_tot_ << "), P = " << P_ << (same_seed ? ", SAME SEED (test mode)" : "") << std::endl;

        for (Replica& R : Rs_) {
            vmc_seed_salt = R.salt;
            R.wb.init(R.B);
            R.wss.resize(n_thread);
            init_batch(R.wb, a, &pool_, R.wss);
            CUDA_CHECK(cudaSetDevice(R.dev));
            R.ds = std::make_unique<DeviceState>(a, true, (std::size_t)R.B);
            R.ds->allocate(a, training);
            if (cublasCreate(&R.h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("GPU backend: cublasCreate failed");
        }
        vmc_seed_salt = 0ULL;
    }

    // Free GPU
    ~GpuBackend() override {
        for (Replica& R : Rs_) {
            if (!R.h) continue;
            cudaSetDevice(R.dev);
            cublasDestroy(R.h);
        }
        if (!Rs_.empty()) cudaSetDevice(Rs_[0].dev);
    }

    // Return walker count
    std::size_t walkers() const override { 
        return (std::size_t)B_tot_; 
    }

    // Thermalize initially
    void initial_thermalization(const Ansatz& a) override {
        run_all(Rs_, [&](Replica& R) {
            R.ds->upload_params(a, R.staging);
            upload_and_reset(*R.ds, R.wb, R.staging);
            eval_logp_batch(*R.ds, R.h, R.B);
            R.acc = therm_init_tuned_dev(R);
        });
        double acc = 0.0;
        for (const Replica& R : Rs_) acc += R.acc * R.B;
        std::cout << "Initial thermalization: acceptance " << acc / B_tot_ << ", steps";
        for (const Replica& R : Rs_) std::cout << " " << R.step;
        std::cout << "\n";
        if constexpr (prof_enabled) prof_reset();
    }

    // GPU sampler samples
    void sample(const Ansatz& a, int records, bool frozen, SampleStats& s) override {
        const int total_sweeps = therm_re_sweep + records * sweeps_between_records;
        const auto t0 = std::chrono::steady_clock::now();

        run_all(Rs_, [&](Replica& R) {
            xfer_stats().reset();
            // If not frozen re-evaluate psi with new parameters
            if (!frozen) { 
                R.ds->upload_params(a, R.staging);
                eval_logp_batch(*R.ds, R.h, R.B);
            }
            const auto ta = std::chrono::steady_clock::now();
            R.acc = therm_batch_device(*R.ds, R.h, R.B, R.step, therm_re_sweep);
            R.therm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
            if (frozen) tune_step(R.acc, R.step);
            R.rt = RecordTimes{};
            record_batch_device(*R.ds, R.h, a, R.wss[0], R.B, R.step, records, /*with_O=*/!frozen, &R.rt);
            download_iteration(*R.ds, R.B, records, R.staging, R.it);
            if (!frozen) tune_step(R.acc, R.step);
            R.bytes_up = xfer_stats().bytes_up;
            R.bytes_dn = xfer_stats().bytes_dn;
        });
        const auto t1 = std::chrono::steady_clock::now();

        merge_stats(Rs_, s.bs, s.E_pool, s.valid_pool);
        long long sp = 0, tau = 0;
        double acc = 0.0;
        s.metro_ms = s.local_E_ms = s.o_ms = 0.0;
        s.bytes_up = s.bytes_dn = 0;
        for (const Replica& R : Rs_) {
            sp += R.it.sp_acc;
            tau += R.it.tau_acc;
            acc += R.acc * R.B;
            // When measuring time always measure slowest replica
            s.metro_ms = std::max(s.metro_ms, R.therm_ms);        
            s.local_E_ms = std::max(s.local_E_ms, R.rt.localE_ms);
            s.o_ms = std::max(s.o_ms, R.rt.o_ms);
            s.bytes_up += R.bytes_up;
            s.bytes_dn += R.bytes_dn;
        }
        s.acceptance = acc / B_tot_;
        const double B2 = (double)B_tot_;
        s.spin_acceptance = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0) ? (double)sp / (B2 * total_sweeps * spin_draws) : 0.0;
        s.tau_acceptance = (tau_mode == TauMode::Sampled && N_p > 0 && N_n > 0) ? (double)tau / (B2 * total_sweeps * tau_draws) : 0.0;
        s.gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  
        records_ = records;
    }

    // Evaluate gradients
    void gradient(std::size_t, long long n_valid, const ClipStats& clip) override {
        refs_.clear();
        for (Replica& R : Rs_) refs_.push_back(ReplicaRef{R.ds.get(), R.h, R.dev, (std::size_t)R.B * (std::size_t)records_});
        O_exp_replicas(refs_, P_, n_valid);
        grad_replicas(refs_, P_, n_valid, clip);
    }

    // Download from device to host
    void grad_to_host(std::size_t first, std::size_t count, double* out) override {
        CUDA_CHECK(cudaSetDevice(Rs_[0].dev));
        CUDA_CHECK(cudaMemcpy(out, Rs_[0].ds->grad_d.d + first, count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // SR step
    SRStepLog sr_step(Ansatz& a, int iter, std::size_t /*n_samples*/, long long n_valid, std::vector<double>& delta, double& rms_damp_mean, long long& n_scalar_dl) override {
        Replica& R0 = Rs_[0];
        CUDA_CHECK(cudaSetDevice(R0.dev));
        rms_damp_mean = rms_update_device(R0.h, R0.ds->grad_d.d, R0.ds->v_rms_d.d, R0.ds->d_rms_d.d, P_);
        return SR_step_device(refs_, a, iter, n_valid, delta, &n_scalar_dl);
    }

    // Reset timers at end of iteration
    void iteration_end(double ms, int i, int records, std::size_t P, const char* tag) override {
        prof_iteration_end(ms);
        if (i == 0) prof_reset();
        else if (std::string(tag) == "descent" && (i + 1) % prof_report_every == 0) prof_report((int)B_tot_, records, P, tag);
    }

    // Log info about run
    void finish(int records, std::size_t P, const char* tag) override {
        prof_report((int)B_tot_, records, P, tag);
    }
// Initialize statistics locally
private:
    std::size_t P_;
    ThreadPool pool_;
    std::vector<Replica> Rs_;
    std::vector<ReplicaRef> refs_;
    int B_tot_ = 0;
    int records_ = 0;
};

// Caller for GPU backend, point to actual GPU Backend
std::unique_ptr<Backend> make_gpu_backend(const Ansatz& a, bool training) {
    return std::make_unique<GpuBackend>(a, training);
}
