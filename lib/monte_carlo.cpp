#include "monte_carlo.h"
#include "constants.h"

#include <iostream>
#include <stdexcept>
#include <string>

// Throws errors when present and informs of all the settings we chose
void validate_config(const Ansatz& a) {
    if (nuc_pot != NucPot::Off) {
        bool spin_ok = (spin_mode == SpinMode::Sampled) || (N_u == N);
        if (!spin_ok || tau_mode != TauMode::Sampled) {
            throw std::runtime_error("nuc_pot != Off requires tau_mode = Sampled and (spin_mode = Sampled or N_u == N).");
        }
    }

    std::size_t P = a.n_params();
    std::size_t Ns_max = (std::size_t)n_walkers * (std::size_t)records_per_iter_max;
    double PN_ratio = (double)P / (double)Ns_max;
    double o_pool_bytes = (double)Ns_max * (double)P * 8.0;
    double o_pool_gb = o_pool_bytes / (1024.0*1024.0*1024.0);

    std::cout << "=== Config ===\n"
              << "N: " << N << " (N_p=" << N_p << " protons, N_n=" << N_n << " neutrons)\n"
              << "S_z sector: N_u=" << N_u << " up, N_d=" << N_d << " down\n"
              << "K: " << K << ", m_feat: " << m_feat << "\n"
              << "envelope: beta_min=" << beta_min << ", alpha_init=" << a.alpha << "\n"
              << "nuc_pot: " << (nuc_pot == NucPot::ModelO ? "ModelO" : "Off") << "\n"
              << "B (n_walkers): " << n_walkers << ", records_per_iter_max: " << records_per_iter_max
              << ", Ns_max (B*records_per_iter_max): " << Ns_max << "\n"
              << "P (n_params): " << P << ", P/Ns_max: " << PN_ratio << "\n"
              << "O_pool: " << o_pool_gb << " GB (Ns_max*P*8 bytes)\n"
              << "==============\n";

    if (o_pool_gb > o_pool_max_gb) {
        throw std::runtime_error("validate_config: projected O_pool size (" + std::to_string(o_pool_gb) +
                                  " GB) exceeds o_pool_max_gb (" + std::to_string(o_pool_max_gb) + " GB).");
    }
}

