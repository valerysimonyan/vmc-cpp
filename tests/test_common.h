#pragma once

// Shared test helper: pin an Ansatz's parameters to a fixed pseudo-random draw.
//
// Tests pin their own parameters instead of relying on the constructor's draw, so a
// test's wavefunction never changes when init_seed or the network layout changes.
// Before this was pinned, test_psi_jet_swap's second-difference check failed roughly
// one run in six on a badly scaled wavefunction. The Jastrow coefficients are drawn
// nonzero so every test exercises the Jastrow paths (BACKLOG D8).
//
// Call this immediately after constructing any Ansatz used by a test.

#include "../lib/hamiltonian.h"

#include <random>

inline void seed_ansatz(Ansatz& a, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (double& p : a.h_net.params)   p = u(rng);
    for (double& p : a.rho_net.params) p = u(rng);
    for (double& p : a.orb_net.params) p = u(rng);
    a.alpha = 0.65;   // constructor's value; not randomized (it sets the envelope)
    for (double& c : a.jc) c = 0.5 * u(rng);   // nonzero Jastrow (trained |c| ~ 0.25); drawn last so the network draws are unchanged
}

// Wrappers for the tests to be able to call the function with std::vector<double> instead of raw pointers
using dvec = std::vector<double>;
inline double psi(const dvec& x, const dvec& s, const dvec& t, const Ansatz& a, Workspace& ws, bool need_inv = false) { 
    return psi(x.data(), s.data(), t.data(), a, ws, need_inv); 
}
inline double log_p(const dvec& x, const dvec& s, const dvec& t, const Ansatz& a, Workspace& ws) { 
    return log_p(x.data(), s.data(), t.data(), a, ws); 
}
inline Jet jpsi(const dvec& x, const dvec& s, const dvec& t, const Ansatz& a, Workspace& ws) { 
    return jpsi(x.data(), s.data(), t.data(), a, ws); 
}
inline void build_st_table(const dvec& x, const Ansatz& a, Workspace& ws) { 
    build_st_table(x.data(), a, ws); 
}
inline double S_from_table(const dvec& s, const dvec& t, const Ansatz& a, Workspace& ws) { 
    return S_from_table(s.data(), t.data(), a, ws); 
}
inline double swap_ratio(const dvec& s, const dvec& t, int i, int j, double s_new_i, double t_new_i, double s_new_j, double t_new_j, double S0, bool use_rank2, const Ansatz& a, Workspace& ws) {
    return swap_ratio(s.data(), t.data(), i, j, s_new_i, t_new_i, s_new_j, t_new_j, S0, use_rank2, a, ws);
}
inline double V_3N(const dvec& x) { 
    return V_3N(x.data()); 
}
inline double V_coulomb(const dvec& x, const dvec& t) { 
    return V_coulomb(x.data(), t.data()); 
}