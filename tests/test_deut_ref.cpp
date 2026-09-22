// Deuteron reference energy of the CODED Hamiltonian (Phase 6.A).
//
// The VMC deuteron is variational: E_VMC >= E_0 of the Hamiltonian actually
// coded in constants.h / physics.cpp. This oracle computes that E_0 two
// independent ways and requires them to agree:
//   A. Numerov shooting on the radial equation, grid h and box R scanned,
//      Richardson-extrapolated in h;
//   B. diagonalisation in an even-tempered Gaussian basis (analytic matrix
//      elements, generalised eigenproblem) -- no shooting, different errors.
// It also checks that the 1S0 T=1 channel has no bound state, as for real np.
//
// Constants come from constants.h by #include -- nothing is retyped -- so the
// oracle follows the Hamiltonian if the constants are ever retuned. For that
// reason it asserts agreement and convergence, not a hard-coded number; the
// value at the time of writing is E_ref = -2.2403705 MeV (BENCH.md, 6.A).
//
// Reduced mass: local_E applies -hbar2_2m * sum_i lap_i with the single-nucleon
// mass m_n. For a translation-invariant psi(r_1 - r_2) that is exactly
// -(hbar^2 / 2 mu) lap_r with mu = m_n / 2, i.e. hbar^2/(2 mu) = 2 * hbar2_2m.
//
// Potential projection (physics.cpp local_E):
//   V = (hbarc/4) [ C01 v01 (1 + R_t - R_s - R_st) + C10 v10 (1 - R_t + R_s - R_st) ]
// 3S1 T=0: spin triplet exchange-symmetric (R_s = +1), isospin singlet
// antisymmetric (R_t = -1), R_st = R_s R_t = -1. The C01 bracket is
// 1 - 1 - 1 + 1 = 0 and the C10 bracket 1 + 1 + 1 + 1 = 4, so
//   V_3S1(r) = hbarc * C10 * exp(-r^2/R10^2) / (pi^1.5 R10^3).
// 1S0 T=1 (R_s = -1, R_t = +1, R_st = -1): brackets 4 and 0,
//   V_1S0(r) = hbarc * C01 * exp(-r^2/R01^2) / (pi^1.5 R01^3).
#include "../lib/constants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
    g_failures++; } } while (0)

static const double Kr = 2.0 * hbar2_2m;                    // hbar^2 / (2 mu), MeV fm^2

struct Channel { const char* name; double C, R; };
static const Channel ch3S1 = {"3S1 T=0 (deuteron)", C10, R10};
static const Channel ch1S0 = {"1S0 T=1", C01, R01};

static double V(const Channel& c, double r) {
    return hbarc * c.C * std::exp(-r*r / (c.R*c.R)) / (std::pow(PI, 1.5) * c.R*c.R*c.R);
}

// ---------------------------------------------------------------------------
// A. Numerov, outward from r = 0 and inward from the box edge, matched at
// r_m = 3 fm with the Numerov-consistent derivative discontinuity (Giannozzi),
// which keeps the eigenvalue error O(h^4). Outer boundary:
//   Wall: u(R) = 0 -- exposes the box truncation directly;
//   Tail: u(R)/u(R-h) = exp(-kappa h), kappa from the trial E -- the
//         asymptotic log-derivative, box-independent once V(R) ~ 0.
// ---------------------------------------------------------------------------
enum Bc { Wall, Tail };

struct Shot { double f; int nodes; bool um_positive; };

static Shot mismatch(const Channel& c, double E, double h, double Rbox, Bc bc) {
    const int n = (int)std::lround(Rbox / h);
    const int m = (int)std::lround(3.0 / h);
    std::vector<double> f(n + 1);
    for (int i = 0; i <= n; i++) {
        const double r = i * h;
        f[i] = 1.0 + h*h * ((E - V(c, r)) / Kr) / 12.0;
    }
    std::vector<double> uo(m + 2);
    uo[0] = 0.0; uo[1] = h;
    int nodes = 0;
    for (int i = 1; i <= m; i++) {
        uo[i+1] = ((12.0 - 10.0*f[i]) * uo[i] - f[i-1] * uo[i-1]) / f[i+1];
        if (uo[i+1] * uo[i] < 0.0) nodes++;
    }
    std::vector<double> ui(n + 1);
    if (bc == Wall) { ui[n] = 0.0; ui[n-1] = 1e-30; }
    else { ui[n] = std::exp(-std::sqrt(std::fabs(E) / Kr) * h) * 1e-30; ui[n-1] = 1e-30; }
    for (int i = n - 1; i > m - 1; i--) {
        ui[i-1] = ((12.0 - 10.0*f[i]) * ui[i] - f[i+1] * ui[i+1]) / f[i-1];
        if (ui[i-1] * ui[i] < 0.0) nodes++;
    }
    const double scale = uo[m] / ui[m];
    return { (uo[m-1] + ui[m+1] * scale - (14.0 - 12.0*f[m]) * uo[m]) / (h * uo[m]), nodes, uo[m] > 0.0 };
}

// Lowest nodeless eigenvalue, or NaN if none below -1e-4 MeV. A sign change of
// the mismatch is only a root if u(r_m) keeps its sign across the bracket; if
// u(r_m) changes sign, the mismatch (divided by u(r_m)) passed through a pole.
static double numerov_E(const Channel& c, double h, double Rbox, Bc bc) {
    double Elo = -20.0;
    Shot slo = mismatch(c, Elo, h, Rbox, bc);
    double Ehi = NAN;
    for (double E = Elo + 0.02; E < -1e-4; E += 0.02) {
        const Shot s = mismatch(c, E, h, Rbox, bc);
        if (s.nodes == 0 && slo.nodes == 0 && (s.f > 0) != (slo.f > 0) && s.um_positive == slo.um_positive) { Ehi = E; break; }
        Elo = E; slo = s;
    }
    if (std::isnan(Ehi)) return NAN;
    for (int it = 0; it < 200; it++) {
        const double Em = 0.5 * (Elo + Ehi);
        const Shot s = mismatch(c, Em, h, Rbox, bc);
        if ((s.f > 0) == (slo.f > 0)) { Elo = Em; slo = s; } else Ehi = Em;
    }
    return 0.5 * (Elo + Ehi);
}

// ---------------------------------------------------------------------------
// B. Even-tempered Gaussians g_i(r) = exp(-a_i r^2) for R(r) = u/r, analytic
// matrix elements (measure r^2 dr, the 4 pi dropped):
//   S_ij = sqrt(pi)/4 * a^{-3/2},                  a = a_i + a_j
//   T_ij = Kr * 6 a_i a_j / a * S_ij
//   V_ij = V0 * sqrt(pi)/4 * (a + 1/R^2)^{-3/2},   V0 = hbarc C / (pi^1.5 R^3)
// H c = E S c by Cholesky reduction and cyclic Jacobi, in long double because
// a dense even-tempered set is nearly linearly dependent.
// ---------------------------------------------------------------------------
static long double gauss_E(const Channel& c, int nb, double a_min, double a_max) {
    typedef long double ld;
    std::vector<ld> a(nb);
    const ld beta = std::pow((ld)a_max / (ld)a_min, (ld)1 / (nb - 1));
    for (int i = 0; i < nb; i++) a[i] = (ld)a_min * std::pow(beta, (ld)i);
    const ld sp = std::sqrt((ld)PI) / 4, V0 = (ld)hbarc * c.C / (std::pow((ld)PI, (ld)1.5) * c.R*c.R*c.R);
    std::vector<ld> S(nb*nb), H(nb*nb);
    for (int i = 0; i < nb; i++) for (int j = 0; j < nb; j++) {
        const ld s = a[i] + a[j], Sij = sp * std::pow(s, (ld)-1.5);
        S[i*nb+j] = Sij;
        H[i*nb+j] = (ld)Kr * 6 * a[i]*a[j] / s * Sij + V0 * sp * std::pow(s + 1 / ((ld)c.R*c.R), (ld)-1.5);
    }
    std::vector<ld> L(nb*nb, 0);                           // S = L L^T
    for (int j = 0; j < nb; j++) {
        ld d = S[j*nb+j];
        for (int k = 0; k < j; k++) d -= L[j*nb+k]*L[j*nb+k];
        if (d <= 0) return NAN;                             // basis numerically dependent
        L[j*nb+j] = std::sqrt(d);
        for (int i = j+1; i < nb; i++) {
            ld v = S[i*nb+j];
            for (int k = 0; k < j; k++) v -= L[i*nb+k]*L[j*nb+k];
            L[i*nb+j] = v / L[j*nb+j];
        }
    }
    std::vector<ld> Y(nb*nb), A(nb*nb);                    // A = L^-1 H L^-T
    for (int col = 0; col < nb; col++) for (int i = 0; i < nb; i++) {
        ld v = H[i*nb+col];
        for (int k = 0; k < i; k++) v -= L[i*nb+k]*Y[k*nb+col];
        Y[i*nb+col] = v / L[i*nb+i];
    }
    for (int row = 0; row < nb; row++) for (int i = 0; i < nb; i++) {
        ld v = Y[row*nb+i];
        for (int k = 0; k < i; k++) v -= L[i*nb+k]*A[row*nb+k];
        A[row*nb+i] = v / L[i*nb+i];
    }
    for (int i = 0; i < nb; i++) for (int j = i+1; j < nb; j++) { const ld m = (A[i*nb+j]+A[j*nb+i])/2; A[i*nb+j] = A[j*nb+i] = m; }
    for (int sweep = 0; sweep < 100; sweep++) {             // cyclic Jacobi
        ld off = 0;
        for (int i = 0; i < nb; i++) for (int j = i+1; j < nb; j++) off += A[i*nb+j]*A[i*nb+j];
        if (off < 1e-60L) break;
        for (int p = 0; p < nb; p++) for (int q = p+1; q < nb; q++) {
            const ld apq = A[p*nb+q];
            if (std::fabs(apq) < 1e-40L) continue;
            const ld th = (A[q*nb+q] - A[p*nb+p]) / (2*apq);
            const ld t = (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th*th + 1));
            const ld cs = 1 / std::sqrt(t*t + 1), sn = t*cs;
            for (int k = 0; k < nb; k++) { const ld x = A[k*nb+p], y = A[k*nb+q]; A[k*nb+p] = cs*x - sn*y; A[k*nb+q] = sn*x + cs*y; }
            for (int k = 0; k < nb; k++) { const ld x = A[p*nb+k], y = A[q*nb+k]; A[p*nb+k] = cs*x - sn*y; A[q*nb+k] = sn*x + cs*y; }
        }
    }
    ld emin = A[0];
    for (int i = 1; i < nb; i++) emin = std::min(emin, A[i*nb+i]);
    return emin;
}

int main() {
    std::printf("constants.h: hbar^2/(2 mu) = 2*hbar2_2m = %.6f MeV fm^2; C10 = %.3f fm^2, R10 = %.3f fm; V_3S1(0) = %.4f MeV\n",
                Kr, C10, R10, V(ch3S1, 0.0));

    // A. Numerov grid x box table.
    const double hs[] = {0.01, 0.005, 0.0025};
    const double Rs[] = {20.0, 30.0, 60.0, 120.0};
    double E_numerov = NAN, E_wall30 = NAN, E_wall120 = NAN, h_spread = 0.0;
    for (Bc bc : {Wall, Tail}) {
        std::printf("A. Numerov, %s: E(h, R) [MeV]\n", bc == Wall ? "hard wall u(R) = 0" : "asymptotic exp(-kappa r)");
        for (double R : Rs) {
            double e[3];
            for (int k = 0; k < 3; k++) e[k] = numerov_E(ch3S1, hs[k], R, bc);
            const double rich = (16.0*e[2] - e[1]) / 15.0;
            std::printf("   R = %5.0f fm: %.7f  %.7f  %.7f   Richardson %.7f\n", R, e[0], e[1], e[2], rich);
            if (bc == Tail) { h_spread = std::max(h_spread, std::fabs(e[0] - rich)); if (R == 60.0) E_numerov = rich; }
            if (bc == Wall && R == 30.0) E_wall30 = rich;
            if (bc == Wall && R == 120.0) E_wall120 = rich;
        }
    }

    // B. Gaussian basis at several sizes.
    struct Cfg { int nb; double amin, amax; };
    const Cfg cfgs[] = {{20, 1e-3, 1e2}, {30, 1e-3, 1e2}, {40, 1e-3, 3e2}, {50, 5e-4, 5e2}};
    double E_gauss = NAN, g_spread = 0.0;
    std::printf("B. Gaussian basis:\n");
    for (const Cfg& g : cfgs) {
        const double e = (double)gauss_E(ch3S1, g.nb, g.amin, g.amax);
        std::printf("   nb = %2d, a in [%.0e, %.0e] fm^-2: %.7f MeV\n", g.nb, g.amin, g.amax, e);
        if (!std::isnan(E_gauss)) g_spread = std::max(g_spread, std::fabs(e - E_gauss));
        E_gauss = e;
    }

    // 1S0 T=1: no bound state. Gaussian lowest eigenvalue must stay positive
    // (it tends to 0+ as the basis widens -- the continuum threshold), and the
    // Numerov bracket must find nothing.
    const double e1S0 = (double)gauss_E(ch1S0, 50, 1e-5, 1e3);
    const double n1S0 = numerov_E(ch1S0, 0.005, 60.0, Tail);
    std::printf("1S0 T=1: Gaussian lowest eigenvalue %+.7f MeV, Numerov %s\n", e1S0, std::isnan(n1S0) ? "no bound state" : "found a bound state");

    std::printf("\nE_ref (coded Hamiltonian, deuteron) = %.7f MeV   [Numerov %.7f, Gaussian %.7f, |diff| %.1e MeV]\n",
                0.5 * (E_numerov + E_gauss), E_numerov, E_gauss, std::fabs(E_numerov - E_gauss));

    CHECK(std::fabs(E_numerov - E_gauss) < 1e-6, "Numerov and Gaussian-basis energies disagree beyond 1 eV");
    CHECK(h_spread < 1e-6, "Numerov not converged in h at 0.01 fm");
    CHECK(g_spread < 1e-6, "Gaussian basis not converged");
    CHECK(std::fabs(E_wall120 - E_numerov) < 1e-6, "hard-wall result at R = 120 fm disagrees with the asymptotic boundary");
    CHECK(E_wall30 - E_numerov < 1e-4, "a 30 fm box truncates more than 0.1 keV");
    CHECK(e1S0 > 0.0 && std::isnan(n1S0), "1S0 T=1 channel binds -- the coded np singlet should be unbound");

    if (g_failures) { std::cerr << g_failures << " check(s) FAILED\n"; return 1; }
    std::printf("test_deut_ref: all checks passed\n");
    return 0;
}
