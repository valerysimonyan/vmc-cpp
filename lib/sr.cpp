#include "sr.h"
#include "constants.h"

#include <algorithm>
#include <cmath>

// Clip energies
ClipStats clip_stats(const std::vector<double>& E_pool, const std::vector<unsigned char>& valid_pool, std::size_t n_samples, long long n_valid) {
    std::vector<double> E_valid;
    E_valid.reserve(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) {
        if (valid_pool[i]) E_valid.push_back(E_pool[i]);
    }
    std::nth_element(E_valid.begin(), E_valid.begin() + n_valid/2, E_valid.end());
    double E_med = E_valid[n_valid/2];
    double MAD = 0.0;
    for (double e : E_valid) MAD += std::fabs(e - E_med);
    MAD /= (double)n_valid;
    double clip_lo = E_med - clip_mad * MAD;
    double clip_hi = E_med + clip_mad * MAD;

    const std::size_t chunk = n_samples / (std::size_t)n_thread;
    double E_clip_sum = 0.0;
    for (int th = 0; th < n_thread; th++) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_thread - 1) ? n_samples : start + chunk;
        double part = 0.0;
        for (std::size_t i = start; i < end; i++) {
            if (!valid_pool[i]) continue;
            part += std::min(std::max(E_pool[i], clip_lo), clip_hi);
        }
        E_clip_sum += part;
    }
    return {clip_lo, clip_hi, E_clip_sum / (double)n_valid};
}

void SROp::init(const std::vector<double>& O_pool_, const std::vector<double>& O_exp_, std::size_t Ns_, std::size_t P_, double lambda_diag_, double eps_abs_, ThreadPool* pool_, const double* d_rms_, const uint8_t* valid_, std::size_t n_valid_) {
    // Do not copy the vector contain invdividual samples and averages of O_i = dlog psi _ dtheta_i
    O_pool = &O_pool_;
    O_exp = &O_exp_;

    Ns = Ns_;
    P = P_;
    lambda_diag = lambda_diag_;
    eps_abs = eps_abs_;
    pool = pool_;
    d_rms = d_rms_;
    valid = valid_;
    n_valid = (valid == nullptr) ? Ns : n_valid_;

    // Initialize with zeros, for partials only during first initialization
    S_diag.assign(P, 0.0);
    if (partials.size() != (std::size_t)P*n_thread) partials.assign((std::size_t)P * n_thread, 0.0);


    // Split into chunks assigning each a pool, sum over samples samplewise  (O_j - <O_j>)^2 computing the diagonal elements of SR matrix, tell each pool which chunk they work on
    std::size_t chunk = Ns / n_thread;
    pool -> run([&](int th) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_thread -1) ? Ns : start+chunk;
        for (std::size_t j = 0; j < P; j++) partials[(std::size_t)th * P + j] = 0.0;

        for (std::size_t i = start; i < end; i++) {
            if (valid != nullptr && !valid[i]) continue;
            for (std::size_t j = 0; j < P; j++) {
                double diff = (*O_pool)[i*P + j] - (*O_exp)[j];
                partials[(std::size_t)th*P + j] += diff * diff;
            }
        }
    });    

    // Average S_diag across threads and chunks
    for (int th = 0; th < n_thread; th ++) {
        for (std::size_t i = 0; i < P; i++) S_diag[i] += partials[(std::size_t)th * P + i];
    }
    for (std::size_t i = 0; i < P; i++) S_diag[i] /= (double)n_valid;
}

void SROp::apply(const std::vector<double>& v, std::vector<double>& out, bool raw) {
    // Initialize <O_j> v_j
    double Oexp_v = 0.0;
    for (std::size_t i = 0; i < P; i++) Oexp_v += (*O_exp)[i] * v[i];

    // We do the same thread splitting to evaluate (O-<O>)i (O-<O>)_j v_j
    std::size_t chunk = Ns / n_thread;
    pool -> run([&](int th) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_thread-1) ? Ns : start+chunk;
        for (std::size_t j = 0; j < P; j++) partials[(std::size_t)th * P + j] = 0.0;
        
        for (std::size_t i = start; i < end; i++) {
            if (valid != nullptr && !valid[i]) continue;
            double Ov_i = 0.0;
            for (std::size_t j = 0; j < P; j++) Ov_i += (*O_pool)[i*P + j] * v[j];
            double t_i = Ov_i - Oexp_v;

            for (std::size_t j = 0; j < P; j++) partials[(std::size_t)th*P + j] += t_i * (*O_pool)[i*P + j];
        }
    });    

    // Set parameter sized output, average across threads
    out.assign(P, 0.0);
    for (int th = 0; th < n_thread; th ++) {
        for (std::size_t i = 0; i < P; i++) out[i] += partials[(std::size_t)th * P + i];
    }

    // Also add (lambda S_jj + eps) v_j where lambda is regulator that damps contribution to parameter which has high variance more than one with low variance, eps is regulator, add RMS regulator as an option
    for (std::size_t i = 0; i < P; i++) {
        out[i] = out[i] / (double)n_valid;
        if (!raw) {
            out[i] += lambda_diag * S_diag[i] * v[i];
            double damp = eps_abs;
            if constexpr (sr_rms_damp) {
                damp += sr_rms_eps * d_rms[i];
            }
            out[i] += damp * v[i];
        }
    }
}

