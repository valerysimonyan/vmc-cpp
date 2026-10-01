// Independent check of local_E. The kinetic term comes from a finite-difference Laplacian of psi, and the
// exchange terms from psi re-evaluated with the spin/isospin labels actually swapped -- no swap_ratio, no
// rank-2 update, no Jastrow label correction. O = d log|psi| / d theta is checked against finite differences.
// The test shares only psi() and the model-o constants with local_E, so it keeps its meaning after the CPU
// and GPU code share one copy of the Hamiltonian formulas.
#include "../lib/physics.h"
#include "../lib/constants.h"
#include "test_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double psi_at(const std::vector<double>& x, const std::vector<double>& s, const std::vector<double>& t,
                     const Ansatz& a, Workspace& ws) {
    return psi(x.data(), s.data(), t.data(), a, ws);
}

// Regulated Gaussian of model o, written here independently of local_E
static double v_reg(double r2, double R) { return std::exp(-r2 / (R * R)) / (PI * std::sqrt(PI) * R * R * R); }

int main() {
    Ansatz a({16}, {16}, {16}, Activation::Gelu);
    seed_ansatz(a, 3001);
    const std::size_t P = a.n_params();
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(0.0, 1.0);
    Workspace ws;
    double worst_E = 0.0, worst_O = 0.0;
    int checked = 0;

    for (int trial = 0; trial < 8; trial++) {
        std::vector<double> x(D), s(N), t(N);
        for (int d = 0; d < D; d++) x[d] = 1.3 * g(rng);
        for (int i = 0; i < N; i++) { s[i] = (i < N_u) ? 1.0 : -1.0; t[i] = (i < N_p) ? 1.0 : -1.0; }
        std::shuffle(s.begin(), s.end(), rng);
        std::shuffle(t.begin(), t.end(), rng);

        std::vector<double> O;
        double E = 0.0;
        if (!local_E(x.data(), s.data(), t.data(), a, ws, O, E)) continue;
        checked++;

        // kinetic term: central second differences, h = 2e-4 fm
        const double p0 = psi_at(x, s, t, a, ws);
        const double h = 2e-4;
        double lap = 0.0;
        for (int d = 0; d < D; d++) {
            std::vector<double> xp = x, xm = x;
            xp[d] += h; xm[d] -= h;
            lap += (psi_at(xp, s, t, a, ws) - 2.0 * p0 + psi_at(xm, s, t, a, ws)) / (h * h);
        }
        const double e_kin = -hbar2_2m * lap / p0;
        double E_bf = e_kin, mag = std::fabs(e_kin);
        if (nuc_3N)      { E_bf += V_3N(x);          mag += std::fabs(V_3N(x)); }
        if (nuc_coulomb) { E_bf += V_coulomb(x, t);  mag += std::fabs(V_coulomb(x, t)); }

        // exchange terms: ratios psi(swapped labels) / psi
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) {
                if (s[i] == s[j] && t[i] == t[j]) continue;
                double r2 = 0.0;
                for (int d = 0; d < dim; d++) { const double df = x[i*dim + d] - x[j*dim + d]; r2 += df * df; }
                std::vector<double> ss = s, tt = t;
                std::swap(ss[i], ss[j]);
                const double R_s = psi_at(x, ss, t, a, ws) / p0;
                std::swap(tt[i], tt[j]);
                const double R_st = psi_at(x, ss, tt, a, ws) / p0;
                const double R_t = psi_at(x, s, tt, a, ws) / p0;
                const double v01 = v_reg(r2, R01), v10 = v_reg(r2, R10);
                E_bf += (hbarc / 4.0) * (C01 * v01 * (1.0 + R_t - R_s - R_st) + C10 * v10 * (1.0 - R_t + R_s - R_st));
                mag += (hbarc / 4.0) * (std::fabs(C01 * v01) + std::fabs(C10 * v10)) * (1.0 + std::fabs(R_t) + std::fabs(R_s) + std::fabs(R_st));
            }
        const double dE = std::fabs(E - E_bf) / std::max(1.0, mag);
        worst_E = std::max(worst_E, dE);

        // O: 30 random parameters plus alpha and every Jastrow coefficient
        std::uniform_int_distribution<std::size_t> pick(0, P - 1);
        std::vector<std::size_t> ks;
        for (int q = 0; q < 30; q++) ks.push_back(pick(rng));
        for (std::size_t q = P - 1 - n_jas_par; q < P; q++) ks.push_back(q);
        double wO = 0.0;
        for (std::size_t k : ks) {
            const double e = 1e-6;
            a.add_to_param(k, e);       const double lp = std::log(std::fabs(psi_at(x, s, t, a, ws)));
            a.add_to_param(k, -2 * e);  const double lm = std::log(std::fabs(psi_at(x, s, t, a, ws)));
            a.add_to_param(k, e);
            const double fd = (lp - lm) / (2 * e);
            wO = std::max(wO, std::fabs(fd - O[k]) / std::max(1.0, std::fabs(fd)));
        }
        worst_O = std::max(worst_O, wO);
        std::printf("  trial %d: E_L %12.5f  brute force %12.5f  rel-to-terms %.1e   O worst rel %.1e\n", trial, E, E_bf, dE, wO);
    }

    std::printf("checked %d configurations: worst E_L %.2e (bound 2e-6), worst O %.2e (bound 1e-5)\n", checked, worst_E, worst_O);
    const bool ok = checked >= 5 && worst_E <= 2e-6 && worst_O <= 1e-5;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
