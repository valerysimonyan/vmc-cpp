
#include "monte_carlo.h"
#include "physics.h"
#include "constants.h"
#include "util.h"
#include "sr.h"
#include "cg.h"
#include "descent.h"
#include "checkpoint.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef VMC_CUDA
#include "gpu/arena.h"
#include "gpu/eval.h"
#include "gpu/gpu_sampler.h"
#include "gpu/sampler_kernels.h"
#include "gpu/record_device.h"
#include "gpu/sr_device.h"
#include <cublas_v2.h>

inline constexpr bool debug_walker_download_at_checkpoint = false;
#endif

// ADAM Descent
void ADAM(const std::vector<double>& grad, std::vector<double>& m, std::vector<double>& v, int i, Ansatz& a) {
    double bias_corr1 = 1.0 - std::pow(beta1, i + 1);
    double bias_corr2 = 1.0 - std::pow(beta2, i + 1);
    for (std::size_t j = 0; j < m.size(); j++) {
        m[j] = beta1 * m[j] + (1 - beta1) * grad[j];
        v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        double m_hat = m[j] / bias_corr1;
        double v_hat = v[j] / bias_corr2;
        a.add_to_param(j, -lr * m_hat / (std::sqrt(v_hat) + eps));
    }
}

SRStepLog SR_step(const std::vector<double>& grad, const std::vector<double>& O_pool, const std::vector<double>& O_exp, Ansatz& a, SROp& sr_op, std::vector<double>& delta, int iter, std::size_t n_params, ThreadPool* pool, const double* d_rms, std::size_t n_samples, const uint8_t* valid_pool, std::size_t n_valid, std::vector<double>& M_inv_diag, std::vector<double>& S_delta) {
    // Exponentially decaying lambda until floor hit, same with learning rate
    double lambda_t = std::max(sr_lambda0 * std::pow(sr_rho, iter), sr_lambda_min);
    double sr_lr = std::max(sr_eta * std::pow(0.999, iter), 0.001);//  * std::cos(iter * PI / (2 * N_sr)); // 
    
    // Intialize the SR matrix
    sr_op.init(O_pool, O_exp, n_samples, n_params, lambda_t, sr_eps, pool, d_rms, valid_pool, n_valid);

    for (std::size_t j = 0; j < n_params; j++) {
        double diag_damp = sr_eps;
        if constexpr (sr_rms_damp) {
            diag_damp += sr_rms_eps * d_rms[j];
        }
        M_inv_diag[j] = 1.0 / (sr_op.S_diag[j] * (1.0 + lambda_t) + diag_damp);
    }

    // Build function S_ij v_j
    auto matvec = [&sr_op](const std::vector<double>& vv, std::vector<double>& Av) {
        sr_op.apply(vv, Av);
    };

    // Solve for v and record result
    CGResult cg = cg_solve(matvec, grad, delta, M_inv_diag, sr_cg_tol, sr_cg_maxit);

    // Check if raw (no lambda and eps) norm of (delta S delta) is within regulation
    sr_op.apply(delta, S_delta, true);
    // Record step size, if step too large, clip
    double q = 0.0;
    for (std::size_t j = 0; j < n_params; j++) {
        q += delta[j] * S_delta[j];
    }
    if (q > sr_trust_r2) {
        double scale = std::sqrt(sr_trust_r2 / q);
        for (std::size_t j = 0; j < n_params; j++) {
            delta[j] *= scale;
        }
    }
    
    double delta_norm_raw = norm(delta);
    bool norm_capped = delta_norm_raw > sr_delta_max; //Set to false for no cap on norm;
    if (norm_capped) {
        double scale = sr_delta_max / delta_norm_raw;
        for (std::size_t j = 0; j < n_params; j++) {
            delta[j] *= scale;
        }
        std::cerr << "SR_step: norm cap triggered -- raw ||delta||=" << delta_norm_raw
                   << " > sr_delta_max=" << sr_delta_max << ", rescaled.\n";
    }

    // Update parameters
    for (std::size_t j = 0; j < n_params; j++) {
        a.add_to_param(j,- sr_lr * delta[j]);
    }

    return {lambda_t, cg.iters, cg.rel_residual, norm(delta), q, norm_capped};
}

// Compute tau and spin acceptance rates here
static void batch_spin_tau_acceptance(const WalkerBatch& wb, int total_sweeps, double& spin_acc, double& tau_acc) {
    long long total_sp = 0, total_tau = 0;
    for (int w = 0; w < wb.B; w ++) {
        total_sp += wb.sp_acc[w];
        total_tau += wb.tau_acc[w];
    }
    spin_acc = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0) ? (double)total_sp / ((double)wb.B * total_sweeps * spin_draws) : 0.0;
    tau_acc  = (tau_mode == TauMode::Sampled && N_p > 0 && N_n > 0) ? (double)total_tau / ((double)wb.B * total_sweeps * tau_draws) : 0.0;
}

// Tune step
static void tune_step(double acceptance, double& step) {
    if (acceptance < 0.45) step *= 0.9;
    else if (acceptance > 0.55) step *= 1.1;
}


#ifdef VMC_CUDA
static double therm_init_tuned_device(DeviceState& ds, cublasHandle_t handle, int B, double& step) {
    int per_block = therm_steps_init / therm_init_blocks;
    int remainder = therm_steps_init % therm_init_blocks;
    double acc = 0.0;
    int done = 0;
    for (int b = 0; b < therm_init_blocks; b++) {
        int sweeps = per_block + (b < remainder ? 1 : 0);
        if (sweeps <= 0) continue;
        acc = therm_batch_device(ds, handle, B, step, sweeps);
        tune_step(acc, step);
        done += sweeps;
        std::cout << "  therm[gpu] " << done << "/" << therm_steps_init
                  << " sweeps: acceptance " << acc << ", step " << step << std::endl;
    }
    return acc;
}

static void spin_tau_from_counts(long long sp, long long tau, int B, int total_sweeps, double& spin_acc, double& tau_acc) {
    spin_acc = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0) ? (double)sp  / ((double)B * total_sweeps * spin_draws) : 0.0;
    tau_acc = (tau_mode  == TauMode::Sampled  && N_p > 0 && N_n > 0) ? (double)tau / ((double)B * total_sweeps * tau_draws)  : 0.0;
}
#endif

// Split thermalization into blocks, tune step size between blocks
static double therm_init_tuned(WalkerBatch& wb, const Ansatz& a, double& step, ThreadPool* pool, std::vector<Workspace>& wss) {
    int per_block = therm_steps_init / therm_init_blocks;
    int remainder = therm_steps_init % therm_init_blocks;
    double acc = 0.0;
    int done = 0;
    for (int b = 0; b < therm_init_blocks; b++) {
        int sweeps = per_block + (b < remainder ? 1 : 0);
        if (sweeps <= 0) continue;
        acc = therm_batch(wb, a, step, sweeps, pool, wss);
        tune_step(acc, step);
        done += sweeps;
        std::cout << "  therm " << done << "/" << therm_steps_init
                  << " sweeps: acceptance " << acc << ", step " << step << std::endl;        
    }
    return acc;
}

// Compute observables here
void compute_obs(const BatchStats& bs, DescentResult& r, std::size_t P, const std::vector<double>& E_pool, const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, ThreadPool* pool, std::vector<double>& O_exp, std::vector<double>& grad) {
    r.El_exp = bs.E_sum / (double)bs.n_valid;
    r.var = std::max(0.0, bs.E2_sum / (double)bs.n_valid - r.El_exp * r.El_exp);
    r.total_node_hits = (double)bs.n_invalid;
    r.L2 = bs.l2_sum / (double)bs.n_valid;
    r.r_rms = std::sqrt(bs.r2_sum / (double)bs.n_valid);

    masked_O_exp(O_pool, valid_pool, n_samples, P, pool, O_exp);
    
    // Clip energy for gradient, nth_element places the nth element, in this case the median, where it would be if the array were sorted but nothing else is sorted
    static std::vector<double> E_valid;
    E_valid.clear();
    E_valid.reserve(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) {
        if (valid_pool[i]) E_valid.push_back(E_pool[i]);
    }

    std::nth_element(E_valid.begin(), E_valid.begin() + bs.n_valid/2, E_valid.end());
    double E_med = E_valid[bs.n_valid/2];
    
    // Caculate average deviation from the median
    double MAD = 0.0;
    for (double e : E_valid) MAD += std::fabs(e - E_med);
    MAD /= (double)bs.n_valid;

    // Bounds on valid energies for gradient calculation
    double clip_lo = E_med - clip_mad * MAD;
    double clip_hi = E_med + clip_mad * MAD;

    int n_workers = pool -> n_workers();
    static std::vector<double> grad_partials;
    static std::vector<double> E_clip_sum_partials;
    if (grad_partials.size() != (std::size_t)n_workers*P) grad_partials.assign((std::size_t)n_workers*P, 0.0);
    if (E_clip_sum_partials.size() != (std::size_t)n_workers) E_clip_sum_partials.assign(n_workers, 0.0);
    
    std::size_t chunk = n_samples / n_workers;
    pool -> run([&](int th) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_workers-1) ? n_samples : start+chunk;
        for (std::size_t k = 0; k < P; k++) grad_partials[(std::size_t)th*P + k] = 0.0;
        double e_clip_sum = 0.0;
        for (std::size_t i = start; i < end; i++) {
            if (!valid_pool[i]) continue;
            double e_clip = std::min(std::max(E_pool[i], clip_lo), clip_hi);
            e_clip_sum += e_clip;
            for (std::size_t k = 0; k < P; k++) grad_partials[(std::size_t)th*P + k] += e_clip * O_pool[i*P + k];
        }
        E_clip_sum_partials[th] = e_clip_sum;
    });

    grad.assign(P, 0.0);
    double E_clip_sum = 0.0;
    for (int th = 0; th < n_workers; th++) {
        E_clip_sum += E_clip_sum_partials[th];
        for (std::size_t k = 0; k < P; k++) grad[k] += grad_partials[(std::size_t)th*P + k];
    }
    double E_clip_mean = E_clip_sum / (double)bs.n_valid;
    for (std::size_t k = 0; k < P; k++) grad[k] = 2.0 * (grad[k]/(double)bs.n_valid - E_clip_mean * O_exp[k]);
}

// Run descent across threads
DescentResult descent(Ansatz& a) {
    std::size_t n_params = a.n_params();
    DescentResult r{};
    
    WalkerBatch wb; 
    wb.init(n_walkers);
    ThreadPool pool(n_thread);
    std::vector<Workspace> wss(n_thread);
    
    // Initialize and thermalize, modify step size on the fly
    double step = step0;
    init_batch(wb, a, &pool, wss);

#ifdef VMC_CUDA
    gpu_select_device(true);
    DeviceState ds(a);
    ds.grow_phase3(a);
    ds.grow_phase33();
    ds.grow_phase4();
    ds.grow_phase42();
    ds.grow_phase43();
    ds.grow_phase5(a);
    ds.grow_phase52();
    ds.grow_phase53();
    PinnedArray staging;
    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("descent: cublasCreate failed");

    ds.upload_params(a, staging);
    upload_and_reset(ds, wb, staging);
    eval_logp_batch(ds, cublas, n_walkers);

    double therm_acc = therm_init_tuned_device(ds, cublas, n_walkers, step);
#else
    double therm_acc = therm_init_tuned(wb, a, step, &pool, wss);
#endif
    std::cout << "Initial thermalization: acceptance " << therm_acc
              << ", tuned step " << step << " (from step0=" << step0 << ")\n";

    // Initial best upper energy bound
    double best_E_ucb = std::numeric_limits<double>::infinity();
    const std::string best_ckpt_path = "best_checkpoint.txt";
    std::vector<double> r_rms_hist(diss_watch_window, 0.0);
    int hist_idx = 0;

    // Store ADAM update vectors
    std::vector<double> m(n_params, 0.), v(n_params, 0.);
    
    // Initialize SR matrix and initial trial vector
    std::vector<double> delta(n_params, 0.0);
    std::vector<double> M_inv_diag(n_params, 0.0);
    std::vector<double> S_delta;
    std::vector<double> v_rms(n_params, 0.0);  // RMSProp-style running average of grad_k^2, persists across SR iterations like delta's CG warm start
    std::vector<double> d_rms(n_params, 0.0);  // sqrt(v_rms)+1e-8, refreshed before each SR solve
    SROp sr_op; 
    
    // Vectors for statistics, declare once for memory use, allocate maximum amount now 
    std::size_t n_samples_max = (std::size_t)n_walkers * (std::size_t)records_per_iter_max;
#ifdef VMC_CUDA
    // No host O_pool: it lives on the device and never crosses the bus. E_pool,
    // valid_pool and the per-walker stats arrive in `it` once per iteration.
    (void)n_samples_max;
    IterStatsHost it;
#else
    std::vector<double> E_pool(n_samples_max);
    std::vector<double> O_pool(n_samples_max * n_params);
    std::vector<uint8_t> valid_pool(n_samples_max);
#endif
    std::vector<double> grad(n_params);
    std::vector<double> O_exp(n_params);    
    BatchStats bs;


    std::ofstream csv("training.csv");
    csv << "step,E_exp,E_err,var,acceptance,spin_acceptance,tau_acceptance,r_rms,lambda,cg_iters,cg_residual,delta_norm,sq_metric_norm,norm_capped,node_hits,n_valid,n_invalid,alpha,grad_alpha,delta_alpha,metro_ms,local_E_ms,o_ms,sr_ms,gpu_ms,ms_iter,L2,rms_damp_mean,xfer_ms,bytes_up,bytes_dn,n_scalar_dl\n";
        
    for (int i = 0; i < N_descent; i++) {
        auto t0 = std::chrono::steady_clock::now();

        // Raise number of records after certain time step
        int records_now = (i >= grow_at_iter) ? records_per_iter_max : records_per_iter;
        std::size_t n_samples = (std::size_t)n_walkers * (std::size_t)records_now;
        
        // Compute all acceptance rates, dynamically adjust step size
        int total_sweeps = therm_re_sweep + records_now * sweeps_between_records;
        double gpu_ms = 0.0;
        double local_E_dev_ms = -1.0;  // CUDA: device energy time; CPU build keeps the old record-window figure
        double o_ms = 0.0; 
        double xfer_ms = 0.0;          // CUDA: the batched per-iteration download

#ifdef VMC_CUDA
        xfer_stats().reset();
        ds.upload_params(a, staging);                                   // UP: params (P)
        eval_logp_batch(ds, cublas, n_walkers);

        // Thermalize after the parameter update, then the record rounds: sweeps,
        // device local_E, O assembly -- nothing crosses the bus until the end.
        auto tA = std::chrono::steady_clock::now();
        r.acceptance = therm_batch_device(ds, cublas, n_walkers, step, therm_re_sweep);
        auto tB = std::chrono::steady_clock::now();
        RecordTimes rtimes;
        record_batch_device(ds, cublas, a, wss[0], n_walkers, step, records_now, /*with_O=*/true, &rtimes);
        auto tC = std::chrono::steady_clock::now();

        // DOWN: E_pool, valid_pool, per-walker stats, acceptance counters -- one copy.
        download_iteration(ds, n_walkers, records_now, staging, it);
        bs = it.bs;
        xfer_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tC).count();

        gpu_ms = std::chrono::duration<double, std::milli>(tB - tA).count() + rtimes.sweep_ms + rtimes.localE_ms + rtimes.o_ms;
        local_E_dev_ms = rtimes.localE_ms;
        o_ms = rtimes.o_ms;
        spin_tau_from_counts(it.sp_acc, it.tau_acc, n_walkers, total_sweeps, r.spin_acceptance, r.tau_acceptance);
#else
        // Evaluate log|Ψ|
        refresh_logp(wb, a, &pool, wss);

        // Thermalize batch after each parameter update, then record samples
        auto tA = std::chrono::steady_clock::now();
        r.acceptance = therm_batch(wb, a, step, therm_re_sweep, &pool, wss);
        auto tB = std::chrono::steady_clock::now();
        record_batch(wb, a, step, records_now, &pool, wss, E_pool, O_pool, valid_pool, bs);
        auto tC = std::chrono::steady_clock::now();

        batch_spin_tau_acceptance(wb, total_sweeps, r.spin_acceptance, r.tau_acceptance);
#endif

        if (r.acceptance < 0.45) step *= 0.9;
        else if (r.acceptance > 0.55) step *= 1.1;
    
        // Compute observables from samples
#ifdef VMC_CUDA
        // compute_obs's scalar head, unchanged ...
        r.El_exp = bs.E_sum / (double)bs.n_valid;
        r.var = std::max(0.0, bs.E2_sum / (double)bs.n_valid - r.El_exp * r.El_exp);
        r.total_node_hits = (double)bs.n_invalid;
        r.L2 = bs.l2_sum / (double)bs.n_valid;
        r.r_rms = std::sqrt(bs.r2_sum / (double)bs.n_valid);
        // ... and its O_exp / clipped gradient on device. The clip bounds are the
        // host's median/MAD over the downloaded E_pool; they go UP as kernel args.
        build_mask(ds.valid_pool.d, ds.mask_d.d, n_samples);
        O_exp_device(cublas, ds.O_pool.d, ds.mask_d.d, n_samples, n_params, bs.n_valid, ds.O_exp_d.d);
        const ClipStats clip = clip_stats_host(it.E_pool, it.valid_pool, n_samples, bs.n_valid);
        grad_device(cublas, ds.O_pool.d, ds.E_pool.d, ds.valid_pool.d, ds.O_exp_d.d, n_samples, n_params, bs.n_valid,
                    clip, ds.E_clip_d.d, ds.grad_d.d);
#else
        compute_obs(bs, r, n_params, E_pool, O_pool, valid_pool, n_samples, &pool, O_exp, grad);
#endif
        r.El_err = batch_error(bs);

        // Checkpoint the lowest energy state based on upper bound of energy
        double E_ucb = r.El_exp + r.El_err;
        if (i >= N_gd && E_ucb < best_E_ucb && r.El_exp < diss_threshold) {
            best_E_ucb = E_ucb;
            save_checkpoint(best_ckpt_path, a);
#ifdef VMC_CUDA
            if constexpr (debug_walker_download_at_checkpoint) ds.download_walkers(wb, staging);
#endif
        }
        if ((i + 1) % ckpt_every == 0) save_checkpoint("periodic_checkpoint.txt", a);
        if (i >= diss_watch_window) {
            double r_rms_past = r_rms_hist[hist_idx];
            if (r.El_exp > diss_threshold && r.r_rms > r_rms_past * diss_growth_factor) {
                std::cerr << "descent: step " << i << " -- possible dissociation: E_exp="
                          << r.El_exp << " > " << diss_threshold << ", r_rms " << r_rms_past
                          << " -> " << r.r_rms << " over " << diss_watch_window << " steps.\n";
            }
        }
        r_rms_hist[hist_idx] = r.r_rms;
        hist_idx = (hist_idx + 1) % diss_watch_window;

        // Parameter update
        SRStepLog log{};
        double sr_ms = 0.0;
        double rms_damp_mean = 0.0;
        long long n_scalar_dl = 0;
        if (i < N_gd) {
#ifdef VMC_CUDA
            // ADAM warmup stays on the host: one P-vector down per iteration, a
            // price paid only for the N_gd warmup steps rather than porting ADAM.
            ds.grad_d.down(grad.data(), n_params);
#endif
            ADAM(grad, m, v, i, a);
        } else {
#ifdef VMC_CUDA
            auto t_sr0 = std::chrono::steady_clock::now();
            rms_damp_mean = rms_update_device(cublas, ds.grad_d.d, ds.v_rms_d.d, ds.d_rms_d.d, n_params);
            // SR init + CG + trust caps on device; delta (P) comes DOWN inside, and
            // the logged norms are the device-computed ones.
            log = SR_step_device(ds, cublas, a, i - N_gd, n_samples, bs.n_valid, delta, &n_scalar_dl);
            auto t_sr1 = std::chrono::steady_clock::now();
            sr_ms = std::chrono::duration<double, std::milli>(t_sr1 - t_sr0).count();
#else
            for (std::size_t k = 0; k < n_params; k++) {
                v_rms[k] = sr_rms_beta * v_rms[k] + (1.0 - sr_rms_beta) * grad[k] * grad[k];
                d_rms[k] = std::sqrt(v_rms[k]) + 1e-8;
                rms_damp_mean += sr_rms_eps * d_rms[k];
            }
            rms_damp_mean /= n_params;

            auto t_sr0 = std::chrono::steady_clock::now();
            log = SR_step(grad, O_pool, O_exp, a, sr_op, delta, i-N_gd, n_params, &pool, d_rms.data(), n_samples, valid_pool.data(), bs.n_valid, M_inv_diag, S_delta);
            auto t_sr1 = std::chrono::steady_clock::now();
            sr_ms = std::chrono::duration<double, std::milli>(t_sr1 - t_sr0).count();
#endif
        }

        // Evaluate times
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double metro_ms = std::chrono::duration<double, std::milli>(tB - tA).count();
        double local_E_ms = (local_E_dev_ms >= 0.0) ? local_E_dev_ms : std::chrono::duration<double, std::milli>(tC - tB).count();

        std::size_t alpha_idx = n_params - 1;
        double grad_alpha = grad[alpha_idx];
        long long bytes_up = 0, bytes_dn = 0;
#ifdef VMC_CUDA
        if (i >= N_gd) {   // SR phase: grad stays on device; one scalar down for the log
            CUDA_CHECK(cudaMemcpy(&grad_alpha, ds.grad_d.d + alpha_idx, sizeof(double), cudaMemcpyDeviceToHost));
            xfer_note_dn(sizeof(double));
        }
        bytes_up = xfer_stats().bytes_up;
        bytes_dn = xfer_stats().bytes_dn;
#endif
        csv << i << "," << r.El_exp << "," << r.El_err << "," << r.var << ","
            << r.acceptance << "," << r.spin_acceptance << "," << r.tau_acceptance << "," << r.r_rms << ","
            << log.lambda << "," << log.cg_iters << "," << log.cg_residual << ","
            << log.delta_norm << "," << log.sq_metric_norm << "," << log.norm_capped << "," << r.total_node_hits << ","
            << bs.n_valid << "," << bs.n_invalid << ","
            << a.get_param(alpha_idx) << "," << grad_alpha << "," << delta[alpha_idx] << ","
            << metro_ms << "," << local_E_ms << "," << o_ms << "," << sr_ms << "," <<  gpu_ms << "," << ms << "," << r.L2 << "," << rms_damp_mean << "," << xfer_ms << "," << bytes_up << "," << bytes_dn << "," << n_scalar_dl << "\n";
        csv.flush();

        if (i < N_gd) std::cout << "ADAM|"; else std::cout << "SR|";
        std::cout << "Step: " << i << ": E_exp: " << r.El_exp << ", E_err: " << r.El_err
            << ", var: " << r.var << ", acceptance: " << r.acceptance
            << ", spin_acceptance: " << r.spin_acceptance
            << ", tau_acceptance: " << r.tau_acceptance
            << ", r_rms: " << r.r_rms
            << ", node_hits: " << r.total_node_hits << ", ms/iter: " << ms
            << ", alpha: " << a.get_param(alpha_idx) << ", grad_alpha: " << grad_alpha
            << ", L2: " << r.L2 << std::endl;
        if (i >= N_gd) std::cout << "               lam: " << log.lambda << ", cg: " << log.cg_iters
                  << ", delta_alpha: " << delta[alpha_idx]
                  << ", metro_ms: " << metro_ms
                  << ", local_E_ms: " << local_E_ms
                  << ", sr_ms: " << sr_ms
                  << ", gpu_ms: " << gpu_ms;
        std::cout << std::endl << std::endl;
    }
    save_checkpoint("final_checkpoint.txt", a);
#ifdef VMC_CUDA
    ds.download_walkers(wb, staging);     // run end: the one sanctioned x/s/t download
    cublasDestroy(cublas);
#endif
    return r;
}



DescentResult evaluate_frozen(Ansatz& a) {
    DescentResult r{};
#ifndef VMC_CUDA
    std::size_t n_params = a.n_params();
#endif
    WalkerBatch wb;
    wb.init(n_walkers);
    ThreadPool pool(n_thread);
    std::vector<Workspace> wss(n_thread);
    init_batch(wb, a, &pool, wss);

    double step = step0;
#ifdef VMC_CUDA
    gpu_select_device(true);
    DeviceState ds(a);
    ds.grow_phase3(a);
    ds.grow_phase33();
    ds.grow_phase4();
    ds.grow_phase42();
    ds.grow_phase43();
    ds.grow_phase53();
    PinnedArray staging;
    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("evaluate_frozen: cublasCreate failed");
    ds.upload_params(a, staging);
    upload_and_reset(ds, wb, staging);
    eval_logp_batch(ds, cublas, n_walkers);
    double therm_acc = therm_init_tuned_device(ds, cublas, n_walkers, step);
    IterStatsHost it;
#else
    double therm_acc = therm_init_tuned(wb, a, step, &pool, wss);
#endif
    std::cout << "Frozen-eval thermalization: acceptance " << therm_acc
              << ", tuned step " << step << " (from step0=" << step0 << ")\n";

    std::size_t n_samples_max = (std::size_t)n_walkers * (std::size_t)records_per_iter_max;
#ifndef VMC_CUDA
    refresh_logp(wb, a, &pool, wss);

    std::vector<double> E_pool(n_samples_max);
    std::vector<double> O_pool(n_samples_max * n_params);
    std::vector<uint8_t> valid_pool(n_samples_max);
#endif

    BatchStats bs_all;
    bs_all.Ew_sum.assign(n_walkers, 0.0);
    bs_all.nw.assign(n_walkers, 0);
    bs_all.E_sum = bs_all.E2_sum = bs_all.l2_sum = bs_all.r2_sum = 0.0;
    bs_all.n_valid = bs_all.n_invalid = 0;

    double acc_sum = 0.0, spin_acc_sum = 0.0, tau_acc_sum = 0.0;

    for (int i = 0; i < eval_iters; i++) {
        BatchStats bs;
        int total_sweeps = therm_re_sweep + records_per_iter_max * sweeps_between_records;
        double spin_acc, tau_acc;
#ifdef VMC_CUDA
        double acc = therm_batch_device(ds, cublas, n_walkers, step, therm_re_sweep);
        tune_step(acc, step);
        record_batch_device(ds, cublas, a, wss[0], n_walkers, step, records_per_iter_max, /*with_O=*/false);
        download_iteration(ds, n_walkers, records_per_iter_max, staging, it);
        bs = it.bs;
        spin_tau_from_counts(it.sp_acc, it.tau_acc, n_walkers, total_sweeps, spin_acc, tau_acc);
#else
        double acc = therm_batch(wb, a, step, therm_re_sweep, &pool, wss);
        tune_step(acc, step);
        record_batch(wb, a, step, records_per_iter_max, &pool, wss, E_pool, O_pool, valid_pool, bs);
        batch_spin_tau_acceptance(wb, total_sweeps, spin_acc, tau_acc);
#endif
        acc_sum += acc;
        spin_acc_sum += spin_acc;
        tau_acc_sum += tau_acc;

        bs_all.E_sum += bs.E_sum;
        bs_all.E2_sum += bs.E2_sum;
        bs_all.l2_sum += bs.l2_sum;
        bs_all.r2_sum += bs.r2_sum;
        bs_all.n_valid += bs.n_valid;
        bs_all.n_invalid += bs.n_invalid;
        for (int w = 0; w < n_walkers; w++) {
            bs_all.Ew_sum[w] += bs.Ew_sum[w];
            bs_all.nw[w] += bs.nw[w];
        }
    }

    r.acceptance = acc_sum / eval_iters;
    r.spin_acceptance = spin_acc_sum / eval_iters;
    r.tau_acceptance = tau_acc_sum / eval_iters;
    r.El_exp = bs_all.E_sum / (double)bs_all.n_valid;
    r.L2 = bs_all.l2_sum / (double)bs_all.n_valid;
    r.var = std::max(0.0, bs_all.E2_sum / (double)bs_all.n_valid - r.El_exp * r.El_exp);
    r.r_rms = std::sqrt(bs_all.r2_sum / (double)bs_all.n_valid);
    r.El_err = batch_error(bs_all);
    r.total_node_hits = (double)bs_all.n_invalid;

    std::size_t samp_all = (std::size_t)eval_iters * n_samples_max;
    std::cout << "=== Frozen-parameter evaluation (" << eval_iters << " iterations, "
              << samp_all << " total samples, " << bs_all.n_valid << " valid) ===\n"
              << "E_exp: " << r.El_exp << " +/- " << r.El_err << ", var: " << r.var
              << ", r_rms: " << r.r_rms << ", L2: " << r.L2
              << ", acceptance: " << r.acceptance
              << ", spin_acceptance: " << r.spin_acceptance
              << ", tau_acceptance: " << r.tau_acceptance
              << ", node_hits: " << r.total_node_hits << "\n";
#ifdef VMC_CUDA
    ds.download_walkers(wb, staging);     // run end
    cublasDestroy(cublas);
#endif
    return r;
}
