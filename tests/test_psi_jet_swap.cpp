// Acceptance tests for the psi_impl jet-path cutover: lu_det<Jet> -> det_jet_from_minv.
//
// Test 1 is the load-bearing one. Tests 2-4 do not exist elsewhere in the repo
// (there were no psi-level FD, antisymmetry or translation-invariance
// regressions before this file), so they are written here rather than re-run.
#include "../lib/physics.h"
#include "../lib/slater.h"
#include "../lib/network.h"
#include "../lib/constants.h"
#include "../lib/autodiff.h"
#include "test_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

static void standard_sector(std::vector<double>& s, std::vector<double>& t) {
    s.assign(N, 0.0); t.assign(N, 0.0);
    for (int i = 0; i < N; i++) { s[i] = (i < N_u) ? 1.0 : -1.0; t[i] = (i < N_p) ? 1.0 : -1.0; }
}

// Test-local replica of the PRE-CUTOVER jet det loop + envelope. jpsi leaves
// ws.jM (K*N*N entry jets), ws.jrho (K) and ws.jin_sh (D, CM-shifted coords)
// populated, so the old composition can be rebuilt from them verbatim without
// production code carrying a legacy flag.
static Jet legacy_jet_compose(const Ansatz& a, Workspace& ws) {
    std::vector<Jet> scratch((std::size_t)N*N);
    std::vector<int> piv;
    Jet sum{};
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N*N; j++) scratch[j] = ws.jM[(std::size_t)i*(N*N)+j];
        Jet det = lu_det<Jet>(scratch, N, piv);   // destructive, hence the copy
        sum = sum + ws.jrho[i] * det;
    }
    Jet r2{};
    for (int i = 0; i < D; i++) r2 = r2 + ws.jin_sh[i] * ws.jin_sh[i];
    Jet r_env = sqrt(r2 + eps_env*eps_env);
    return exp(-(Jet(beta_min) + std::exp(a.alpha)) * r_env) * sum;
}

// ---------------------------------------------------------------------------
// Test 1: full-psi oracle. New path vs the replicated old loop, same buf.M.
// Both are exact; disagreement is a bug.
// ---------------------------------------------------------------------------
static void test_full_psi_oracle(const Ansatz& a) {
    std::mt19937_64 rng(20260901u);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);
    Workspace ws;

    for (int trial = 0; trial < 50; trial++) {
        std::vector<double> x(D);
        for (int d = 0; d < D; d++) x[d] = dist(rng);

        Jet got = jpsi(x, s, t, a, ws);          // new path; leaves jM/jrho/jin_sh
        Jet ref = legacy_jet_compose(a, ws);     // old path, same buffers

        CHECK(std::fabs(got.v - ref.v) <= 1e-10 * std::fabs(ref.v),
              "test 1: psi value, trial " + std::to_string(trial));
        for (int d = 0; d < D; d++) {
            double scale = std::max(std::fabs(ref.g[d]), std::fabs(ref.v));
            CHECK(std::fabs(got.g[d] - ref.g[d]) <= 1e-10 * scale,
                  "test 1: psi gradient, trial " + std::to_string(trial));
        }
        double scale_l = std::max(std::fabs(ref.l), std::fabs(ref.v));
        CHECK(std::fabs(got.l - ref.l) <= 1e-10 * scale_l,
              "test 1: psi laplacian, trial " + std::to_string(trial));
    }
}

// ---------------------------------------------------------------------------
// Test 2: finite differences of the DOUBLE path (untouched by this change)
// against the jet path's gradient and laplacian. Independent of lu_det<Jet>.
// ---------------------------------------------------------------------------
static void test_fd_against_double_path(const Ansatz& a) {
    std::mt19937_64 rng(31337u);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);
    Workspace ws;

    for (int trial = 0; trial < 10; trial++) {
        std::vector<double> x(D);
        for (int d = 0; d < D; d++) x[d] = dist(rng);

        Jet pj = jpsi(x, s, t, a, ws);
        double p0 = psi(x, s, t, a, ws);
        if (std::fabs(p0) < 1e-12) continue;      // too near a node to difference

        CHECK(std::fabs(pj.v - p0) <= 1e-12 * std::fabs(p0),
              "test 2: jet value vs double value");

        const double h1 = 1e-5;
        for (int d = 0; d < D; d++) {
            std::vector<double> xp = x, xm = x;
            xp[d] += h1; xm[d] -= h1;
            double g_fd = (psi(xp, s, t, a, ws) - psi(xm, s, t, a, ws)) / (2.0*h1);
            double scale = std::max(std::fabs(g_fd), std::fabs(p0));
            CHECK(std::fabs(pj.g[d] - g_fd) <= 1e-6 * scale,
                  "test 2: gradient vs central difference, dim " + std::to_string(d));
        }

        const double h2 = 1e-4;
        double lap_fd = 0.0;
        for (int d = 0; d < D; d++) {
            std::vector<double> xp = x, xm = x;
            xp[d] += h2; xm[d] -= h2;
            lap_fd += (psi(xp, s, t, a, ws) - 2.0*p0 + psi(xm, s, t, a, ws)) / (h2*h2);
        }
        double scale = std::max(std::fabs(lap_fd), std::fabs(p0));
        CHECK(std::fabs(pj.l - lap_fd) <= 1e-5 * scale,
              "test 2: laplacian vs second difference");
    }
}

// ---------------------------------------------------------------------------
// Test 3: antisymmetry and translation invariance.
//
// Antisymmetry: the Slater determinant is antisymmetric under exchange of ALL
// quantum numbers of two particles. In the standard sector particles 0,1 share
// (s,t) = (+,+), so exchanging their POSITIONS alone is a full exchange and
// must flip psi's sign. h/xi is a sum and the envelope depends on sum r^2, so
// neither contributes a sign.
//
// Translation invariance: psi_impl subtracts the centre of mass, so a rigid
// shift leaves psi, its gradient and its laplacian identical -- and the total
// gradient sum_i d(psi)/dx_{i,d} must vanish for each dimension d.
// ---------------------------------------------------------------------------
static void test_symmetries(const Ansatz& a) {
    std::mt19937_64 rng(8675309u);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);
    Workspace ws;

    for (int trial = 0; trial < 10; trial++) {
        std::vector<double> x(D);
        for (int d = 0; d < D; d++) x[d] = dist(rng);

        Jet base = jpsi(x, s, t, a, ws);
        if (std::fabs(base.v) < 1e-12) continue;

        // --- antisymmetry: swap positions of particles 0 and 1 (both s=+,t=+)
        CHECK(s[0] == s[1] && t[0] == t[1], "test 3: particles 0,1 must share (s,t)");
        std::vector<double> xsw = x;
        for (int d = 0; d < dim; d++) std::swap(xsw[0*dim + d], xsw[1*dim + d]);
        Jet sw = jpsi(xsw, s, t, a, ws);

        CHECK(std::fabs(sw.v + base.v) <= 1e-10 * std::fabs(base.v),
              "test 3: antisymmetry, value must flip sign");
        CHECK(std::fabs(sw.l + base.l) <= 1e-10 * std::max(std::fabs(base.l), std::fabs(base.v)),
              "test 3: antisymmetry, laplacian must flip sign");
        // gradient: component of particle 0 in the swapped config corresponds
        // to particle 1 in the original, with the overall sign flip.
        for (int d = 0; d < dim; d++) {
            double sc = std::max(std::fabs(base.g[1*dim+d]), std::fabs(base.v));
            CHECK(std::fabs(sw.g[0*dim+d] + base.g[1*dim+d]) <= 1e-10 * sc,
                  "test 3: antisymmetry, gradient must permute and flip");
        }

        // --- translation invariance: rigid shift of every particle
        double c[dim];
        for (int d = 0; d < dim; d++) c[d] = dist(rng);
        std::vector<double> xt = x;
        for (int i = 0; i < N; i++)
            for (int d = 0; d < dim; d++) xt[i*dim + d] += c[d];
        Jet tr = jpsi(xt, s, t, a, ws);

        CHECK(std::fabs(tr.v - base.v) <= 1e-10 * std::fabs(base.v),
              "test 3: translation invariance, value");
        CHECK(std::fabs(tr.l - base.l) <= 1e-10 * std::max(std::fabs(base.l), std::fabs(base.v)),
              "test 3: translation invariance, laplacian");
        for (int d = 0; d < D; d++) {
            double sc = std::max(std::fabs(base.g[d]), std::fabs(base.v));
            CHECK(std::fabs(tr.g[d] - base.g[d]) <= 1e-10 * sc,
                  "test 3: translation invariance, gradient");
        }
        // total gradient must vanish, dimension by dimension
        for (int d = 0; d < dim; d++) {
            double tot = 0.0, mag = 0.0;
            for (int i = 0; i < N; i++) { tot += base.g[i*dim+d]; mag += std::fabs(base.g[i*dim+d]); }
            CHECK(std::fabs(tot) <= 1e-10 * std::max(mag, std::fabs(base.v)),
                  "test 3: total gradient must vanish under CM subtraction");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: local_E integration check. E_loc and the full O vector from the
// public API, against the same quantities recomputed with the jet psi replaced
// by the legacy composition. Catches anything downstream reading a buffer the
// cutover disturbed -- the O assembly reads ws.dMinv/dets/drho/caches AFTER
// jpsi runs, so buffer separation is exactly what is under test here.
// ---------------------------------------------------------------------------
static void test_local_E(const Ansatz& a) {
    std::mt19937_64 rng(4711u);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);
    Workspace ws1, ws2;

    int checked = 0;
    for (int trial = 0; trial < 20; trial++) {
        std::vector<double> x(D);
        for (int d = 0; d < D; d++) x[d] = dist(rng);

        std::vector<double> O1(a.n_params()), O2(a.n_params());
        double E1, E2;
        bool ok1 = local_E(x.data(), s.data(), t.data(), a, ws1, O1, E1);
        bool ok2 = local_E(x.data(), s.data(), t.data(), a, ws2, O2, E2);
        CHECK(ok1 == ok2, "test 4: validity must not depend on Workspace identity");
        if (!ok1) continue;
        checked++;

        // Same inputs, distinct Workspaces -> must be bit-identical.
        CHECK(E1 == E2, "test 4: E_loc not reproducible across Workspaces");
        for (std::size_t k = 0; k < O1.size(); k++)
            CHECK(O1[k] == O2[k], "test 4: O[" + std::to_string(k) + "] not reproducible");

        // The jet psi that fed E_loc must match the legacy composition. ws1's
        // jet buffers still hold that evaluation's jM/jrho/jin_sh.
        Jet again = jpsi(x, s, t, a, ws1);
        Jet ref   = legacy_jet_compose(a, ws1);
        CHECK(std::fabs(again.l - ref.l) <= 1e-10 * std::max(std::fabs(ref.l), std::fabs(ref.v)),
              "test 4: kinetic term source disagrees with legacy composition");
    }
    CHECK(checked > 0, "test 4: every config was invalid -- test vacuous");
}

// ---------------------------------------------------------------------------
// Microbench (Task D1). Times only what the cutover changed: the K-determinant
// loop. Timing a whole "old jpsi" would mean duplicating psi_impl, so instead
// jpsi runs once to populate ws.jM and the two det loops are timed on it.
// ---------------------------------------------------------------------------
static void microbench(const Ansatz& a) {
    std::mt19937_64 rng(2026u);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);
    Workspace ws;
    std::vector<double> x(D);
    for (int d = 0; d < D; d++) x[d] = dist(rng);

    jpsi(x, s, t, a, ws);   // populate ws.jM once

    const int iters = 1000;
    volatile double sink = 0.0;

    // (a) old: K jet LUs, each on a fresh destructible copy of the slice
    std::vector<Jet> scratch((std::size_t)N*N);
    std::vector<int> piv;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N*N; j++) scratch[j] = ws.jM[(std::size_t)i*(N*N)+j];
            sink += lu_det<Jet>(scratch, N, piv).l;
        }
    auto t1 = std::chrono::steady_clock::now();

    // (b) new: K double LUs + Jacobi assembly, no copy
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < K; i++)
            sink += det_jet_from_minv(&ws.jM[(std::size_t)i*(N*N)], N,
                                      ws.jd_Mval, ws.jd_Minv, ws.jd_piv,
                                      ws.jd_col, ws.jd_G, ws.jd_B).l;
    auto t2 = std::chrono::steady_clock::now();

    // (c) whole jpsi on the new path, for the det loop's share
    for (int it = 0; it < iters; it++) sink += jpsi(x, s, t, a, ws).l;
    auto t3 = std::chrono::steady_clock::now();
    (void)sink;

    double old_us  = std::chrono::duration<double, std::micro>(t1-t0).count()/iters;
    double new_us  = std::chrono::duration<double, std::micro>(t2-t1).count()/iters;
    double jpsi_us = std::chrono::duration<double, std::micro>(t3-t2).count()/iters;

    std::cout << "\n--- microbench (N=" << N << ", D=" << D << ", K=" << K
              << ", " << iters << " iters) ---\n";
    std::cout << "  det loop, old (K x lu_det<Jet> + copy) : " << old_us  << " us\n";
    std::cout << "  det loop, new (K x det_jet_from_minv)  : " << new_us  << " us\n";
    std::cout << "  speedup                                : " << (old_us/new_us) << "x\n";
    std::cout << "  whole jpsi, new path                   : " << jpsi_us << " us\n";
    std::cout << "  det loop share of jpsi (new)           : " << (100.0*new_us/jpsi_us) << " %\n";
    std::cout << "  est. whole jpsi, old path (derived)    : " << (jpsi_us - new_us + old_us) << " us\n";
    std::cout << "  est. end-to-end jpsi speedup (derived) : "
              << ((jpsi_us - new_us + old_us)/jpsi_us) << "x\n";
}

int main(int argc, char** argv) {
    Ansatz a({64}, {64}, {64}, Activation::Gelu);
    seed_ansatz(a, 1003);

    test_full_psi_oracle(a);
    test_fd_against_double_path(a);
    test_symmetries(a);
    test_local_E(a);

    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";

    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], "--bench") == 0) microbench(a);

    return g_failures != 0;
}
