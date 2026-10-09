#include "train.h"
#include "checkpoint.h"
#include "constants.h"
#include "envelope.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

// Create a backend for training, either CPU or GPU depending on compile-time flags
std::unique_ptr<Backend> make_cpu_backend(const Ansatz& a, bool training);
#ifdef VMC_CUDA
std::unique_ptr<Backend> make_gpu_backend(const Ansatz& a, bool training);
#endif

std::unique_ptr<Backend> make_backend(const Ansatz& a, bool training) {
#ifdef VMC_CUDA
    return make_gpu_backend(a, training);
#else
    return make_cpu_backend(a, training);
#endif
}

// One row of training log containing all saved information
struct IterLog {
    int step = 0;       
    DescentResult r{};
    SRStepLog sr{};
    long long n_valid = 0, n_invalid = 0;
    double alpha = 0.0, grad_alpha = 0.0, delta_alpha = 0.0;
    double metro_ms = 0.0, local_E_ms = 0.0, o_ms = 0.0, sr_ms = 0.0, gpu_ms = 0.0, ms = 0.0;  // 
    double rms_damp_mean = 0.0, xfer_ms = 0.0;
    long long bytes_up = 0, bytes_dn = 0, n_scalar_dl = 0;
};

// Columns of training.csv 
template <typename F>
static void csv_columns(const IterLog& g, F&& col) {
    col("step", g.step);                    col("E_exp", g.r.El_exp);             col("E_err", g.r.El_err);
    col("var", g.r.var);                    col("acceptance", g.r.acceptance);    col("spin_acceptance", g.r.spin_acceptance);
    col("tau_acceptance", g.r.tau_acceptance); col("r_rms", g.r.r_rms);           col("lambda", g.sr.lambda);
    col("cg_iters", g.sr.cg_iters);         col("cg_residual", g.sr.cg_residual); col("delta_norm", g.sr.delta_norm);
    col("sq_metric_norm", g.sr.sq_metric_norm); col("norm_capped", g.sr.norm_capped); col("node_hits", g.r.total_node_hits);
    col("n_valid", g.n_valid);              col("n_invalid", g.n_invalid);        col("alpha", g.alpha);
    col("grad_alpha", g.grad_alpha);        col("delta_alpha", g.delta_alpha);    col("metro_ms", g.metro_ms);
    col("local_E_ms", g.local_E_ms);        col("o_ms", g.o_ms);                  col("sr_ms", g.sr_ms);
    col("gpu_ms", g.gpu_ms);                col("ms_iter", g.ms);                 col("L2", g.r.L2);
    col("rms_damp_mean", g.rms_damp_mean);  col("xfer_ms", g.xfer_ms);            col("bytes_up", g.bytes_up);
    col("bytes_dn", g.bytes_dn);            col("n_scalar_dl", g.n_scalar_dl);
}

// Make a comma separated header 
static void csv_header(std::ostream& o) {
    const char* sep = "";
    csv_columns(IterLog{}, [&](const char* name, const auto&) { o << sep << name; sep = ","; });
    o << "\n";
}

// Make a comma separated row 
static void csv_row(std::ostream& o, const IterLog& g) {
    const char* sep = "";
    csv_columns(g, [&](const char*, const auto& v) { o << sep << v; sep = ","; });
    o << "\n";
}

// Write final results from BatchStats
static void fill_observables(const BatchStats& bs, DescentResult& r) {
    r.El_exp = bs.E_sum / (double)bs.n_valid;
    r.var = std::max(0.0, bs.E2_sum / (double)bs.n_valid - r.El_exp * r.El_exp);
    r.total_node_hits = (double)bs.n_invalid;
    r.L2 = bs.l2_sum / (double)bs.n_valid;
    r.r_rms = std::sqrt(bs.r2_sum / (double)bs.n_valid);
}

// Training step
DescentResult train(Ansatz& a) {
    // Get parameter count and envelope parameter
    const std::size_t n_params = a.n_params(); 
    const std::size_t alpha_idx = n_params - n_params_env;
    
    // Initialize backend in training mode and thermalizing
    std::unique_ptr<Backend> be = make_backend(a, true);
    const std::size_t B = be->walkers();
    be->initial_thermalization(a);
    
    // Initialize statistics for checkpointing 
    double best_E_ucb = std::numeric_limits<double>::infinity();
    std::vector<double> r_rms_hist(diss_watch_window, 0.0);
    int hist_idx = 0;
    
    // Initialize training parameters and statistics structures
    DescentResult r{};
    std::vector<double> m(n_params, 0.), v(n_params, 0.);                    // ADAM moments (warm-up)
    std::vector<double> env_m(n_params_env, 0.), env_v(n_params_env, 0.);    // Adam moments of the envelope block
    std::vector<double> delta(n_params, 0.0), grad(n_params, 0.0);
    SampleStats s;

    // Initialize csv header
    std::ofstream csv("training.csv");
    csv_header(csv);

    // Training loop
    for (int i = 0; i < N_descent; i++) {
        // Begin timer, record count, and sample size
        const auto t0 = std::chrono::steady_clock::now();
        const int records_now = (i >= grow_at_iter) ? records_per_iter_max : records_per_iter;
        const std::size_t n_samples = B * (std::size_t)records_now;

        // Sample, ;pg the observables and the clipped-energy gradient
        be->sample(a, records_now, false, s);
        r.acceptance = s.acceptance;
        r.spin_acceptance = s.spin_acceptance;
        r.tau_acceptance = s.tau_acceptance;
        fill_observables(s.bs, r);
        const ClipStats clip = clip_stats(s.E_pool, s.valid_pool, n_samples, s.bs.n_valid);
        be->gradient(n_samples, s.bs.n_valid, clip);
        r.El_err = batch_error(s.bs);

        // Checkpoints: the lowest upper energy bound so far, and a periodic one
        const double E_ucb = r.El_exp + r.El_err;
        if (i >= N_gd && E_ucb < best_E_ucb && r.El_exp < diss_threshold) {
            best_E_ucb = E_ucb;
            save_checkpoint("best_checkpoint.txt", a);
        }
        if ((i + 1) % ckpt_every == 0) save_checkpoint("periodic_checkpoint.txt", a);
        if (i >= diss_watch_window) {
            const double r_rms_past = r_rms_hist[hist_idx];
            if (r.El_exp > diss_threshold && r.r_rms > r_rms_past * diss_growth_factor) {
                std::cerr << "train: step " << i << " -- possible dissociation: E_exp="
                          << r.El_exp << " > " << diss_threshold << ", r_rms " << r_rms_past
                          << " -> " << r.r_rms << " over " << diss_watch_window << " steps.\n";
            }
        }
        r_rms_hist[hist_idx] = r.r_rms;
        hist_idx = (hist_idx + 1) % diss_watch_window;

        // Parameter update: ADAM warm-up, then SR (plus the optional Adam step for the envelope block)
        IterLog g;
        if (i < N_gd) {
            be->grad_to_host(0, n_params, grad.data());
            ADAM(grad, m, v, i, a);
        } else {
            const auto t_sr0 = std::chrono::steady_clock::now();
            g.sr = be->sr_step(a, i - N_gd, n_samples, s.bs.n_valid, delta, g.rms_damp_mean, g.n_scalar_dl);
            
            // Optional ADAM training for envelope during SR
            if (env_adam_lr > 0.0) {
                double ge[n_params_env];
                be->grad_to_host(alpha_idx, n_params_env, ge);
                const int t = i - N_gd + 1;
                const double lr_t = env_adam_lr / (1.0 + (double)(t - 1) / env_adam_decay_it);
                for (int q = env_adam_alpha ? 0 : 1; q < n_params_env; q++) {
                    env_m[q] = beta1 * env_m[q] + (1.0 - beta1) * ge[q];
                    env_v[q] = beta2 * env_v[q] + (1.0 - beta2) * ge[q] * ge[q];
                    const double mh = env_m[q] / (1.0 - std::pow(beta1, t)), vh = env_v[q] / (1.0 - std::pow(beta2, t));
                    a.add_to_param(alpha_idx + q, -lr_t * mh / (std::sqrt(vh) + 1e-8));
                }
            }

            g.sr_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sr0).count();
        }
        g.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // Log
        g.step = i;
        g.r = r;
        g.n_valid = s.bs.n_valid;
        g.n_invalid = s.bs.n_invalid;
        g.alpha = a.get_param(alpha_idx);
        be->grad_to_host(alpha_idx, 1, &g.grad_alpha);
        g.delta_alpha = delta[alpha_idx];
        g.metro_ms = s.metro_ms; g.local_E_ms = s.local_E_ms; g.o_ms = s.o_ms; g.gpu_ms = s.gpu_ms; g.xfer_ms = s.xfer_ms;
        g.bytes_up = s.bytes_up; g.bytes_dn = s.bytes_dn;
        csv_row(csv, g);
        csv.flush();
        
        // Print results
        std::cout << (i < N_gd ? "ADAM|" : "SR|") << "Step: " << i << ": E_exp: " << r.El_exp << ", E_err: " << r.El_err
                  << ", var: " << r.var << ", acceptance: " << r.acceptance
                  << ", spin_acceptance: " << r.spin_acceptance << ", tau_acceptance: " << r.tau_acceptance
                  << ", r_rms: " << r.r_rms << ", node_hits: " << r.total_node_hits << ", ms/iter: " << g.ms
                  << ", alpha: " << g.alpha << ", grad_alpha: " << g.grad_alpha << ", L2: " << r.L2 << std::endl;
        if (i >= N_gd) std::cout << "               lam: " << g.sr.lambda << ", cg: " << g.sr.cg_iters
                                 << ", delta_alpha: " << g.delta_alpha << ", metro_ms: " << g.metro_ms
                                 << ", local_E_ms: " << g.local_E_ms << ", sr_ms: " << g.sr_ms << ", gpu_ms: " << g.gpu_ms;
        std::cout << std::endl << std::endl;

        // End iteration on backend
        be->iteration_end(g.ms, i, records_now, n_params, "descent");
    }
    // Save checkpoint, end backend, return results
    save_checkpoint("final_checkpoint.txt", a);
    be->finish(records_per_iter, n_params, "descent run end");
    return r;
}


// Energy of fixed parameters: eval_iters rounds of records_per_iter_max records, pooled per walker for the error bar
DescentResult evaluate_frozen(Ansatz& a) {
    // Initialize backend 
    std::unique_ptr<Backend> be = make_backend(a, false);
    const std::size_t B = be->walkers();
    be->initial_thermalization(a);
    
    // Initialize statistics to 0
    DescentResult r{};
    BatchStats bs_all;
    bs_all.Ew_sum.assign(B, 0.0);
    bs_all.nw.assign(B, 0);
    bs_all.E_sum = bs_all.E2_sum = bs_all.l2_sum = bs_all.r2_sum = 0.0;
    bs_all.n_valid = bs_all.n_invalid = 0;
    double acc_sum = 0.0, spin_acc_sum = 0.0, tau_acc_sum = 0.0;
    SampleStats s;

    // Sample and log iterations
    for (int i = 0; i < eval_iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        be->sample(a, records_per_iter_max, true, s);
        acc_sum += s.acceptance;
        spin_acc_sum += s.spin_acceptance;
        tau_acc_sum += s.tau_acceptance;
        bs_all.E_sum += s.bs.E_sum;
        bs_all.E2_sum += s.bs.E2_sum;
        bs_all.l2_sum += s.bs.l2_sum;
        bs_all.r2_sum += s.bs.r2_sum;
        bs_all.n_valid += s.bs.n_valid;
        bs_all.n_invalid += s.bs.n_invalid;
        for (std::size_t w = 0; w < B; w++) {
            bs_all.Ew_sum[w] += s.bs.Ew_sum[w];
            bs_all.nw[w] += s.bs.nw[w];
        }
        be->iteration_end(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), i, records_per_iter_max, a.n_params(), "frozen_eval");
    }
    be->finish(records_per_iter_max, a.n_params(), "frozen_eval");

    // Average everything
    r.acceptance = acc_sum / eval_iters;
    r.spin_acceptance = spin_acc_sum / eval_iters;
    r.tau_acceptance = tau_acc_sum / eval_iters;
    r.El_exp = bs_all.E_sum / (double)bs_all.n_valid;
    r.L2 = bs_all.l2_sum / (double)bs_all.n_valid;
    r.var = std::max(0.0, bs_all.E2_sum / (double)bs_all.n_valid - r.El_exp * r.El_exp);
    r.r_rms = std::sqrt(bs_all.r2_sum / (double)bs_all.n_valid);
    r.El_err = batch_error(bs_all);
    r.total_node_hits = (double)bs_all.n_invalid;

    // Output results
    const std::size_t samp_all = (std::size_t)eval_iters * B * (std::size_t)records_per_iter_max;
    std::cout << "=== Frozen-parameter evaluation (" << eval_iters << " iterations, "
              << samp_all << " total samples, " << bs_all.n_valid << " valid) ===\n"
              << "E_exp: " << r.El_exp << " +/- " << r.El_err << ", var: " << r.var
              << ", r_rms: " << r.r_rms << ", L2: " << r.L2
              << ", acceptance: " << r.acceptance
              << ", spin_acceptance: " << r.spin_acceptance
              << ", tau_acceptance: " << r.tau_acceptance
              << ", node_hits: " << r.total_node_hits << "\n";
    return r;
}