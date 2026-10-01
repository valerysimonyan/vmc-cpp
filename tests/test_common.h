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

#include "../lib/physics.h"

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
