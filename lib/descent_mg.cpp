#ifdef VMC_CUDA

#include "physics.h"
#include "constants.h"
#include "util.h"
#include "descent.h"
#include "envelope.h"
#include "checkpoint.h"
#include "seed.h"
#include "walkers.h"
#include "pool.h"

#include "gpu/arena.h"
#include "gpu/eval.h"
#include "gpu/gpu_sampler.h"
#include "gpu/sampler_kernels.h"
#include "gpu/record_device.h"
#include "gpu/sr_device.h"
#include "gpu/planner.h"
#include <cublas_v2.h>

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

// ADAM descent step
void ADAM(const std::vector<double>& grad, std::vector<double>& m, std::vector<double>& v, int i, Ansatz& a);

namespace {
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
};

// Thermalization per devicewalkers
double therm_init_tuned_dev(Replica& R) {
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

}

// Multi-GPU descent function
DescentResult descent_mg(Ansatz& a) {
    const std::size_t n_params = a.n_params();
    DescentResult r{};
    ThreadPool pool(n_thread);

    const bool same_seed = std::getenv("VMC_MG_SAME_SEED") && std::string(std::getenv("VMC_MG_SAME_SEED")) == "1";  // True if we set VMC_MG_SAME_SEED to 1, otherwise crash
    // VMC_GPUS = auto for filling up memory
    const char* gp = std::getenv("VMC_GPUS"); 
    std::vector<GpuPlan> plan;
    if (gp && std::string(gp) == "auto") {
        plan = plan_gpus(a);
    } else {  
        // Otherwise both devices got filled  
        for (int d = 0; d < 2; d++) { 
            GpuPlan g; 
            g.dev = d; 
            g.B = (std::size_t)n_walkers; 
            plan.push_back(g); 
        }
    }
    // If in plan mode print allocations
    if (std::getenv("VMC_PLAN_ONLY")) {   
        std::printf("plan only: arena for n_walkers=%d would be %.3f GiB\n", n_walkers, (double)arena_bytes_for(a, (std::size_t)n_walkers) / (1024.0*1024*1024));
        std::exit(0);
    }
    // Make a replica per planbn element, initialize the replicas, print the info
    std::vector<Replica> Rs(plan.size());
    int B_tot = 0;
    for (std::size_t k = 0; k < plan.size(); k++) {
        Rs[k].dev = plan[k].dev;
        Rs[k].B = (int)plan[k].B;
        Rs[k].salt = (k == 0 || same_seed) ? 0ULL : splitmix64(0x6d67ULL + k);   
        B_tot += Rs[k].B;
    }
    std::cout << "descent_mg: " << Rs.size() << " GPU(s), walkers";
    for (const Replica& R : Rs) std::cout << " GPU" << R.dev << ":" << R.B;
    std::cout << " (total " << B_tot << "), P = " << n_params << (same_seed ? ", SAME SEED (test mode)" : "") << std::endl;

    // Iterate through replacas and allocate memory
    for (Replica& Rr : Rs) {  
        Replica* R = &Rr;                     // Set replica
        vmc_seed_salt = R->salt;              // Set seed
        R->wb.init(R->B);                     // Initialize workbench
        R->wss.resize(n_thread);              // Initialize workspace
        init_batch(R->wb, a, &pool, R->wss);  // Initialize batches, evaluate log|ψ|
        CUDA_CHECK(cudaSetDevice(R->dev));    // Set device matching replica
        // Allocate memory
        R->ds = std::make_unique<DeviceState>(a, true, (std::size_t)R->B);
        R->ds->allocate(a, true);
        // Assign handles
        if (cublasCreate(&R->h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("descent_mg: cublasCreate failed");
    }
    // Set seed to default for main thread
    vmc_seed_salt = 0ULL;

    // Run all thermalization steps
    run_all(Rs, [&](Replica& R) {
        R.ds->upload_params(a, R.staging);
        upload_and_reset(*R.ds, R.wb, R.staging);
        eval_logp_batch(*R.ds, R.h, R.B);
        R.acc = therm_init_tuned_dev(R);
    });
    Replica& R0 = Rs[0]; // Set reference to 0'th Replica
    // Combine acceptances across replicas
    {
        double acc = 0.0;
        for (const Replica& R : Rs) acc += R.acc * R.B;
        std::cout << "Initial thermalization: acceptance " << acc / B_tot << ", steps";
        for (const Replica& R : Rs) std::cout << " " << R.step;
        std::cout << "\n";
    }

    // Initialize statistics and update parameters
    double best_E_ucb = std::numeric_limits<double>::infinity();
    const std::string best_ckpt_path = "best_checkpoint.txt";
    std::vector<double> m(n_params, 0.), v(n_params, 0.);
    std::vector<double> env_m(envelope::n_params_env, 0.), env_v(envelope::n_params_env, 0.);
    std::vector<double> delta(n_params, 0.0), grad(n_params, 0.0);
    std::vector<double> E_all;
    std::vector<unsigned char> valid_all;
    BatchStats bs;
    // Initialize file
    std::ofstream csv("training.csv");
    csv << "step,E_exp,E_err,var,acceptance,spin_acceptance,tau_acceptance,r_rms,lambda,cg_iters,cg_residual,delta_norm,sq_metric_norm,norm_capped,node_hits,n_valid,n_invalid,alpha,grad_alpha,delta_alpha,metro_ms,local_E_ms,o_ms,sr_ms,gpu_ms,ms_iter,L2,rms_damp_mean,xfer_ms,bytes_up,bytes_dn,n_scalar_dl\n";
    // Iterate through descent 
    for (int i = 0; i < N_descent; i++) {
        auto t0 = std::chrono::steady_clock::now();
        const int records_now = (i >= grow_at_iter) ? records_per_iter_max : records_per_iter;
        const int total_sweeps = therm_re_sweep + records_now * sweeps_between_records;
        // Run across devices accept-reject
        run_all(Rs, [&](Replica& R) {
            R.ds->upload_params(a, R.staging);
            eval_logp_batch(*R.ds, R.h, R.B);
            auto ta = std::chrono::steady_clock::now();
            R.acc = therm_batch_device(*R.ds, R.h, R.B, R.step, therm_re_sweep);
            R.therm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
            R.rt = RecordTimes{};
            record_batch_device(*R.ds, R.h, a, R.wss[0], R.B, R.step, records_now, /*with_O=*/true, &R.rt);
            download_iteration(*R.ds, R.B, records_now, R.staging, R.it);
            if (R.acc < 0.45) R.step *= 0.9;
            else if (R.acc > 0.55) R.step *= 1.1;
        });
        auto tS = std::chrono::steady_clock::now();
        // Merge statistics
        merge_stats(Rs, bs, E_all, valid_all);
        {
            long long sp = 0, tau = 0;
            double acc = 0.0;
            for (const Replica& R : Rs) { 
                sp += R.it.sp_acc; 
                tau += R.it.tau_acc; 
                acc += R.acc * R.B; 
            }
            r.acceptance = acc / B_tot;
            const double B2 = (double)B_tot;
            r.spin_acceptance = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0) ? (double)sp / (B2 * total_sweeps * spin_draws) : 0.0;
            r.tau_acceptance = (tau_mode == TauMode::Sampled && N_p > 0 && N_n > 0) ? (double)tau / (B2 * total_sweeps * tau_draws) : 0.0;
        }
        r.El_exp = bs.E_sum / (double)bs.n_valid;
        r.var = std::max(0.0, bs.E2_sum / (double)bs.n_valid - r.El_exp * r.El_exp);
        r.total_node_hits = (double)bs.n_invalid;
        r.L2 = bs.l2_sum / (double)bs.n_valid;
        r.r_rms = std::sqrt(bs.r2_sum / (double)bs.n_valid);
        // Compute gradient relevant parameters
        std::vector<MgRep> M;
        for (Replica& R : Rs) M.push_back(MgRep{R.ds.get(), R.h, R.dev, (std::size_t)R.B * (std::size_t)records_now});
        O_exp_mg(M, n_params, bs.n_valid);
        const ClipStats clip = clip_stats_host(E_all, valid_all, E_all.size(), bs.n_valid);
        grad_mg(M, n_params, bs.n_valid, clip);
        r.El_err = batch_error(bs);
        // Checkpointing for best energy
        const double E_ucb = r.El_exp + r.El_err;
        if (i >= N_gd && E_ucb < best_E_ucb && r.El_exp < diss_threshold) {
            best_E_ucb = E_ucb;
            save_checkpoint(best_ckpt_path, a);
        }
        if ((i + 1) % ckpt_every == 0) save_checkpoint("periodic_checkpoint.txt", a);
        // Take logs and have if-else for phase of gradient descent
        SRStepLog log{};
        double sr_ms = 0.0, rms_damp_mean = 0.0;
        long long n_scalar_dl = 0;
        CUDA_CHECK(cudaSetDevice(R0.dev));
        if (i < N_gd) {
            R0.ds->grad_d.down(grad.data(), n_params);
            ADAM(grad, m, v, i, a);
        } else {
            auto t_sr0 = std::chrono::steady_clock::now();
            rms_damp_mean = rms_update_device(R0.h, R0.ds->grad_d.d, R0.ds->v_rms_d.d, R0.ds->d_rms_d.d, n_params);
            log = SR_step_device_mg(M, a, i - N_gd, bs.n_valid, delta, &n_scalar_dl);
            if (env_adam_lr > 0.0) {
                const std::size_t e0 = n_params - envelope::n_params_env;
                double ge[envelope::n_params_env];
                CUDA_CHECK(cudaMemcpy(ge, R0.ds->grad_d.d + e0, sizeof(ge), cudaMemcpyDeviceToHost));
                const int t = i - N_gd + 1;
                const double lr_t = env_adam_lr / (1.0 + (double)(t - 1) / env_adam_decay_it);
                for (int q = env_adam_alpha ? 0 : 1; q < envelope::n_params_env; q++) {
                    env_m[q] = beta1 * env_m[q] + (1.0 - beta1) * ge[q];
                    env_v[q] = beta2 * env_v[q] + (1.0 - beta2) * ge[q] * ge[q];
                    const double mh = env_m[q] / (1.0 - std::pow(beta1, t)), vh = env_v[q] / (1.0 - std::pow(beta2, t));
                    a.add_to_param(e0 + q, -lr_t * mh / (std::sqrt(vh) + 1e-8));
                }
            }
            sr_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sr0).count();
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // Take gradient with respect to envelope parameter
        const std::size_t alpha_idx = n_params - envelope::n_params_env;
        double grad_alpha = 0.0;
        CUDA_CHECK(cudaMemcpy(&grad_alpha, R0.ds->grad_d.d + alpha_idx, sizeof(double), cudaMemcpyDeviceToHost));
        // Log slowest times across replicas
        double metro_ms = 0.0, local_E_ms = 0.0, o_ms = 0.0;
        for (const Replica& R : Rs) { 
            metro_ms = std::max(metro_ms, R.therm_ms); 
            local_E_ms = std::max(local_E_ms, R.rt.localE_ms); 
            o_ms = std::max(o_ms, R.rt.o_ms); 
        }
        const double gpu_ms = std::chrono::duration<double, std::milli>(tS - t0).count();   // parallel sampling phase, wall
        
        // Load data into csv file
        csv << i << "," << r.El_exp << "," << r.El_err << "," << r.var << ","
            << r.acceptance << "," << r.spin_acceptance << "," << r.tau_acceptance << "," << r.r_rms << ","
            << log.lambda << "," << log.cg_iters << "," << log.cg_residual << ","
            << log.delta_norm << "," << log.sq_metric_norm << "," << log.norm_capped << "," << r.total_node_hits << ","
            << bs.n_valid << "," << bs.n_invalid << ","
            << a.get_param(alpha_idx) << "," << grad_alpha << "," << delta[alpha_idx] << ","
            << metro_ms << "," << local_E_ms << "," << o_ms << "," << sr_ms << "," << gpu_ms << "," << ms << "," << r.L2 << ","
            << rms_damp_mean << "," << 0.0 << "," << 0 << "," << 0 << "," << n_scalar_dl << "\n";
        csv.flush();

        // Print data
        std::cout << (i < N_gd ? "ADAM|" : "SR|") << "Step: " << i << ": E_exp: " << r.El_exp << ", E_err: " << r.El_err
                  << ", var: " << r.var << ", acceptance: " << r.acceptance << ", r_rms: " << r.r_rms
                  << ", node_hits: " << r.total_node_hits << ", ms/iter: " << ms << ", L2: " << r.L2
                  << "\n               lam: " << log.lambda << ", cg: " << log.cg_iters << ", sampling_ms: " << gpu_ms
                  << ", sr_ms: " << sr_ms << ", n_valid: " << bs.n_valid << std::endl << std::endl;
    }
    // Save checkpoint, destroy handles, set devices
    save_checkpoint("final_checkpoint.txt", a);
    for (Replica& R : Rs) { 
        CUDA_CHECK(cudaSetDevice(R.dev)); 
        cublasDestroy(R.h); 
    }
    CUDA_CHECK(cudaSetDevice(R0.dev));
    return r;
}

#endif