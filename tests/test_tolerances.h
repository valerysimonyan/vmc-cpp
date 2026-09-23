#pragma once
// Phase 6.3 tolerance ladder. tol::ff(fp64_bound, fp32_bound) picks the bound
// for the build's fp32_forward flag; the FP64 argument is always the pre-6.3
// bound, unchanged, so the FP64 build is tested exactly as before.
//
// fp32_forward: the network GEMM chains run in float (eps_f = 6.0e-8). Each
// layer rounds at ~sqrt(width) * eps_f ~ 5e-7 (width 64); errors compound with
// depth and are amplified downstream. Measured on the test networks (seeded
// {64}), which set the fp32 bounds at 3-10x headroom:
//   network outputs vs CPU        h 3.5e-6, orb 4.1e-6, xi 7.8e-6, rho 1.1e-4   -> 1e-4 / 1e-3
//   log|psi|                      1.6e-4 abs (= relative on psi)                 -> 2e-3
//   S_from_table                  median 7.9e-7, p99.9 2.3e-4, max 6.0e-4        -> 1e-5 / 3e-3 / 1e-2
//   psi jet vs CPU                v 2.0e-4, g 9.2e-4, l 2.3e-4                   -> 1e-2
//   det jet vs CPU                v 3.2e-3, g 1.4e-2, l 5.6e-3 (ill-conditioned) -> 5e-2
//   E_kin per sample              0.30 MeV abs on |E_kin| up to 5.9e3 MeV        -> 2 MeV
//   E_loc / V_nuc per sample      4e-5 / 3e-5 relative to their terms            -> 1e-3
//   O vs CPU (row-scaled)         ordinary 1e-4, near-node 1.9e-3                -> 1e-2
//   validity mask vs CPU          1-2 of ~500 near-node samples (the FP64 guard
//                                 is 1e-6, the fp32 one 1e-4)                    -> <= 1%
// These are PER-SAMPLE errors; whether they bias the averaged energy is a
// question for the physics runs (BENCH.md 6.3), not for these bounds.
#include "../lib/constants.h"
#include "../lib/gpu/layouts.h"

#include <vector>

namespace tol {
template <typename T>
constexpr T ff(T fp64, T fp32) { return fp32_forward ? fp32 : fp64; }

// fp32_opool: every O entry is rounded once to float (relative 2^-24 = 6.0e-8).
// A bound on O relative to its own scale therefore needs at least ~1e-7.
template <typename T>
constexpr T fo(T fp64, T fp32) { return fp32_opool ? fp32 : fp64; }

// Both flags: the looser of the two applicable bounds.
constexpr double ffo(double fp64, double f_fwd, double f_pool) {
    double b = fp64;
    if (fp32_forward && f_fwd > b) b = f_fwd;
    if (fp32_opool && f_pool > b) b = f_pool;
    return b;
}
}

// O_pool, whatever its storage type, read back widened to double.
template <typename T>
inline std::vector<double> opool_down(const DeviceArray<T>& a, std::size_t n) {
    std::vector<T> h(n); a.down(h.data(), n);
    return std::vector<double>(h.begin(), h.end());
}
// Upload a host pool into the device pool type (float under fp32_opool).
inline void opool_up(DeviceArray<opool_t>& d, const std::vector<double>& h) {
    std::vector<opool_t> c(h.begin(), h.end());
    if (d.n < c.size()) d.alloc(c.size());
    d.up(c.data(), c.size());
}
// Round a host pool to the device pool's precision, so a CPU oracle computed
// on it sees exactly the values the device stores (tests the kernels, not the
// storage precision -- test_precision measures that).
inline void opool_round(std::vector<double>& v) {
    if constexpr (fp32_opool) for (double& x : v) x = (double)(float)x;
}
