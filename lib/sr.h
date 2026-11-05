#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstddef>

#include "pool.h"
#include "constants.h"

// [sr] settings of the stochastic-reconfiguration step; defaults from constants.h
struct SrSettings {
    double eta = sr_eta;                // Learning rate, decays as eta * lr_decay^iter down to 0.001
    double lr_decay = sr_lr_decay;      // Decay rate of lr
    double lambda0 = sr_lambda0;        // Diagonal shift lambda * S_ii, decays as lambda0 * lambda_decay^iter ...
    double lambda_min = sr_lambda_min;  // ... down to this floor
    double lambda_decay = sr_rho;       // Decay rate of lambda, 0.995 means 0.5 after 138 iterations
    double eps = sr_eps;                // Absolute diagonal shift
    double rms_beta = sr_rms_beta;      // RMS damping: running mean of grad^2 with this decay ...
    double rms_eps = sr_rms_eps;        // ... adds rms_eps * sqrt(mean) to the diagonal; 0 turns it off
    double cg_tol = sr_cg_tol;          // CG stops at this relative residual ...
    int cg_maxit = sr_cg_maxit;         // ... or after this many iterations
    double trust_r2 = sr_trust_r2;      // Trust region: delta S delta is scaled down to this ...
    double delta_max = sr_delta_max;    // ... and |delta| to this

    double lambda(int iter) const { 
        return std::max(lambda0 * std::pow(lambda_decay, iter), lambda_min); 
    }
    double lr(int iter) const { 
        return std::max(eta * std::pow(lr_decay, iter), 0.001); 
    }
};

struct ClipStats { 
    double clip_lo;
    double clip_hi;
    double E_clip_mean;
};

ClipStats clip_stats(const std::vector<double>& E_pool, const std::vector<unsigned char>& valid_pool, std::size_t n_samples, long long n_valid, double clip_mad = ::clip_mad);

struct SROp {
    const std::vector<double>* O_pool = nullptr; 
    const std::vector<double>* O_exp = nullptr; 
    std::size_t Ns = 0, P = 0;
    double lambda_diag = 0.0;
    double eps_abs = 0.0;
    double rms_eps = 0.0;           // diagonal shift rms_eps * d_rms[i]
    const double* d_rms = nullptr;  // length P
    const uint8_t* valid = nullptr; // length Ns; nullptr = all rows of O_pool evaluated
    std::size_t n_valid = 0;        // Only when valid != nullptr
    std::vector<double> S_diag;
    std::vector<double> partials; 
    ThreadPool* pool = nullptr;

    void init(const std::vector<double>& O_pool_, 
                const std::vector<double>& O_exp_, 
                std::size_t Ns_, 
                std::size_t P_, 
                double lamdba_diag_, 
                double eps_abs_,
                ThreadPool* pool_,
                const double* d_rms_,
                const uint8_t* valid,
                std::size_t n_valid_ = 0,
                double rms_eps_ = sr_rms_eps);
                
    void apply(const std::vector<double>& v, 
                std::vector<double>& out, bool raw = false);
};

