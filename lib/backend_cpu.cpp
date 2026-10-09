#include "train.h"
#include "wavefunction.h"
#include "constants.h"
#include "util.h"
#include "sr.h"
#include "cg.h"
#include "envelope.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>


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

// SR step descent
SRStepLog SR_step(const std::vector<double>& grad, const std::vector<double>& O_pool, const std::vector<double>& O_exp, Ansatz& a, SROp& sr_op, std::vector<double>& delta, int iter, std::size_t n_params, ThreadPool* pool, const double* d_rms, std::size_t n_samples, const uint8_t* valid_pool, std::size_t n_valid, std::vector<double>& M_inv_diag, std::vector<double>& S_delta) {
    // Exponentially decaying lambda until floor hit, same with learning rate
    double lambda_t = std::max(sr_lambda0 * std::pow(sr_rho, iter), sr_lambda_min);
    double sr_lr = std::max(sr_eta * std::pow(0.999, iter), 0.001);
    
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

// Evaluate gradient with clipped energies
static void clipped_gradient(const std::vector<double>& E_pool, const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, std::size_t P, long long n_valid, const ClipStats& clip, const std::vector<double>& O_exp, ThreadPool* pool, std::vector<double>& grad_partials, std::vector<double>& grad) {
    int n_workers = pool -> n_workers();
    if (grad_partials.size() != (std::size_t)n_workers*P) grad_partials.assign((std::size_t)n_workers*P, 0.0);

    // Across threads evaluate <E dlogO_K>
    std::size_t chunk = n_samples / n_workers;
    pool -> run([&](int th) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_workers-1) ? n_samples : start+chunk;
        for (std::size_t k = 0; k < P; k++) grad_partials[(std::size_t)th*P + k] = 0.0;
        for (std::size_t i = start; i < end; i++) {
            if (!valid_pool[i]) continue;
            double e_clip = std::min(std::max(E_pool[i], clip.clip_lo), clip.clip_hi);
            for (std::size_t k = 0; k < P; k++) grad_partials[(std::size_t)th*P + k] += e_clip * O_pool[i*P + k];
        }
    });

    // Reduce across threads and compute final gradient
    grad.assign(P, 0.0);
    for (int th = 0; th < n_workers; th++) {
        for (std::size_t k = 0; k < P; k++) grad[k] += grad_partials[(std::size_t)th*P + k];
    }
    for (std::size_t k = 0; k < P; k++) grad[k] = 2.0 * (grad[k]/(double)n_valid - clip.E_clip_mean * O_exp[k]);
}


// Compute observables here
void compute_obs(const BatchStats& bs, DescentResult& r, std::size_t P, const std::vector<double>& E_pool, const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, ThreadPool* pool, std::vector<double>& O_exp, std::vector<double>& grad) {
    r.El_exp = bs.E_sum / (double)bs.n_valid;
    r.var = std::max(0.0, bs.E2_sum / (double)bs.n_valid - r.El_exp * r.El_exp);
    r.total_node_hits = (double)bs.n_invalid;
    r.L2 = bs.l2_sum / (double)bs.n_valid;
    r.r_rms = std::sqrt(bs.r2_sum / (double)bs.n_valid);

    masked_O_exp(O_pool, valid_pool, n_samples, P, pool, O_exp);
    
    // Clip energies and compute gradient
    const ClipStats clip = clip_stats(E_pool, valid_pool, n_samples, bs.n_valid);
    static std::vector<double> grad_partials;
    clipped_gradient(E_pool, O_pool, valid_pool, n_samples, P, bs.n_valid, clip, O_exp, pool, grad_partials, grad);
}


// The CPU backend: one batch of walkers sampled by a thread pool; the O pool and the SR state live on the host
class CpuBackend : public Backend {
public:
    // Constructor initializes everything
    explicit CpuBackend(const Ansatz& a) : P_(a.n_params()), B_(n_walkers), pool_(n_thread), wss_(n_thread) {
        wb_.init(B_);
        init_batch(wb_, a, &pool_, wss_);
        const std::size_t n_samples_max = (std::size_t)B_ * (std::size_t)records_per_iter_max;
        E_pool_.resize(n_samples_max);
        O_pool_.resize(n_samples_max * P_);
        valid_pool_.resize(n_samples_max);
        grad_.assign(P_, 0.0);
        O_exp_.assign(P_, 0.0);
        M_inv_diag_.assign(P_, 0.0);
        v_rms_.assign(P_, 0.0);
        d_rms_.assign(P_, 0.0);
    }

    // Return walker count
    std::size_t walkers() const override { 
        return (std::size_t)B_; 
    }

    // Do initial thermalization
    void initial_thermalization(const Ansatz& a) override {
        const double acc = therm_init_tuned(wb_, a, step_, &pool_, wss_);
        std::cout << "Initial thermalization: acceptance " << acc
                  << ", tuned step " << step_ << " (from step0=" << step0 << ")\n";
    }

    // Sampler samples as expected
    void sample(const Ansatz& a, int records, bool frozen, SampleStats& s) override {
        const int total_sweeps = therm_re_sweep + records * sweeps_between_records;
        // If not frozen re-evaluate psi with new parameters
        if (!frozen || first_sample_) refresh_logp(wb_, a, &pool_, wss_);
        first_sample_ = false;

        const auto tA = std::chrono::steady_clock::now();
        const double acc = therm_batch(wb_, a, step_, therm_re_sweep, &pool_, wss_);
        const auto tB = std::chrono::steady_clock::now();
        
        if (frozen) tune_step(acc, step_);
        record_batch(wb_, a, step_, records, &pool_, wss_, E_pool_, O_pool_, valid_pool_, bs_);
        const auto tC = std::chrono::steady_clock::now();
        batch_spin_tau_acceptance(wb_, total_sweeps, s.spin_acceptance, s.tau_acceptance);
        if (!frozen) tune_step(acc, step_);
        
        s.acceptance = acc;
        s.bs = bs_;
        s.E_pool = E_pool_;
        s.valid_pool = valid_pool_;
        s.metro_ms = std::chrono::duration<double, std::milli>(tB - tA).count();
        s.local_E_ms = std::chrono::duration<double, std::milli>(tC - tB).count();
    }

    // Compute gradient 
    void gradient(std::size_t n_samples, long long n_valid, const ClipStats& clip) override {
        masked_O_exp(O_pool_, valid_pool_, n_samples, P_, &pool_, O_exp_);
        clipped_gradient(E_pool_, O_pool_, valid_pool_, n_samples, P_, n_valid, clip, O_exp_, &pool_, grad_partials_, grad_);
    }

    // Write gradient to host, as we are in host tho this is trivial
    void grad_to_host(std::size_t first, std::size_t count, double* out) override {
        for (std::size_t k = 0; k < count; k++) out[k] = grad_[first + k];
    }

    // SR step 
    SRStepLog sr_step(Ansatz& a, int iter, std::size_t n_samples, long long n_valid, std::vector<double>& delta, double& rms_damp_mean, long long& n_scalar_dl) override {
        rms_damp_mean = 0.0;
        for (std::size_t k = 0; k < P_; k++) {
            v_rms_[k] = sr_rms_beta * v_rms_[k] + (1.0 - sr_rms_beta) * grad_[k] * grad_[k];
            d_rms_[k] = std::sqrt(v_rms_[k]) + 1e-8;
            rms_damp_mean += sr_rms_eps * d_rms_[k];
        }
        rms_damp_mean /= P_;
        n_scalar_dl = 0;
        return SR_step(grad_, O_pool_, O_exp_, a, sr_op_, delta, iter, P_, &pool_, d_rms_.data(), n_samples, valid_pool_.data(), n_valid, M_inv_diag_, S_delta_);
    }

// Initialize statistics locally
private:
    std::size_t P_;
    int B_;
    ThreadPool pool_;
    std::vector<Workspace> wss_;
    WalkerBatch wb_;
    double step_ = step0;
    bool first_sample_ = true;
    BatchStats bs_;
    std::vector<double> E_pool_, O_pool_;
    std::vector<uint8_t> valid_pool_;
    std::vector<double> grad_, O_exp_, grad_partials_;
    std::vector<double> M_inv_diag_, S_delta_, v_rms_, d_rms_;
    SROp sr_op_;
};

// Caller for CPU backend, point to actual CPU Backend
std::unique_ptr<Backend> make_cpu_backend(const Ansatz& a, bool) {
    return std::make_unique<CpuBackend>(a);
}
