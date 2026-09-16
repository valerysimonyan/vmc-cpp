#pragma once

// Shared test helper: pin an Ansatz's parameters to a fixed pseudo-random draw.
//
// Network's constructor fills params via gen_uniform_sample, which seeds a
// thread_local mt19937 from std::random_device -- so a freshly constructed
// Ansatz has DIFFERENT parameters on every run. Tests that compare against
// finite differences or fixed tolerances are then non-reproducible: before this
// was pinned, test_psi_jet_swap's second-difference check failed roughly one run
// in six, on a wavefunction that happened to be badly scaled. A committed suite
// has to be green every time or it teaches people to ignore it.
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
}
