#include "../lib/slater.h"
#include "../lib/autodiff.h"
#include "../lib/constants.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Scratch {
    std::vector<double> Mval, Minv, col, G, B;
    std::vector<int>    piv;
};

static Jet call_new(const std::vector<Jet>& M, int n, Scratch& s) {
    return det_jet_from_minv(M, n, s.Mval, s.Minv, s.piv, s.col, s.G, s.B);
}

// lu_det<Jet> eliminates in place, so the oracle always gets a fresh copy.
static Jet call_ref(const std::vector<Jet>& M, int n) {
    std::vector<Jet> copy = M;
    std::vector<int> piv;
    return lu_det<Jet>(copy, n, piv);
}

static void rand_M(std::vector<Jet>& M, int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    M.assign((std::size_t)n*n, Jet());
    for (int i = 0; i < n*n; i++) {
        M[i].v = u(rng);
        for (int a = 0; a < D; a++) M[i].g[a] = u(rng);
        M[i].l = u(rng);
    }
}

// Conditioning is a property of the test INPUT, not of the code under test.
// Uniform random matrices are occasionally near-singular, and at cond ~ 1e8 no
// 1e-10 agreement is meaningful for either method -- so such draws are
// rejected and redrawn rather than the tolerance being loosened.
static double cond_inf(const std::vector<Jet>& M, const std::vector<double>& Minv, int n) {
    double nm = 0.0, ni = 0.0;
    for (int r = 0; r < n; r++) {
        double a = 0.0, b = 0.0;
        for (int c = 0; c < n; c++) {
            a += std::fabs(M[r*n+c].v);
            b += std::fabs(Minv[r*n+c]);
        }
        nm = std::max(nm, a);
        ni = std::max(ni, b);
    }
    return nm * ni;
}

// term1 = det * tr(Minv L). Four lines, computed from Minv and the entry
// Laplacians -- this does NOT re-implement term 2, and term2 = l - term1 then
// follows from the decomposition itself. Used only to size the comparison.
static double term1_of(const std::vector<Jet>& M, const std::vector<double>& Minv, int n, double det_v) {
    double t = 0.0;
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            t += Minv[j*n+k] * M[k*n+j].l;
    return t * det_v;
}

// ---------------------------------------------------------------------------
// Test 1: random jets vs the lu_det<Jet> oracle. Both are exact, so any
// disagreement is a bug.
// ---------------------------------------------------------------------------
static void test_random_vs_oracle() {
    Scratch s;
    std::mt19937 rng(12345u);

    for (int n : {2, 3, 6, 8}) {
        int done = 0, draws = 0;
        while (done < 200) {
            if (++draws > 20000) { CHECK(false, "test 1: too many rejected draws"); break; }

            std::vector<Jet> M;
            rand_M(M, n, rng);

            Jet got = call_new(M, n, s);
            if (got.v == 0.0) continue;                      // singular draw
            if (cond_inf(M, s.Minv, n) > 1e6) continue;      // ill-conditioned draw

            double t1 = term1_of(M, s.Minv, n, got.v);       // before s.Minv is reused
            double t2 = got.l - t1;

            Jet ref = call_ref(M, n);
            done++;

            CHECK(std::fabs(got.v - ref.v) <= 1e-10 * std::fabs(ref.v),
                  "test 1: determinant value");

            // g_a = det * tr(B_a). When tr(B_a) is accidentally near zero a raw
            // ratio measures that cancellation, not the implementation, so the
            // scale floors at |det| -- the natural scale for a derivative of
            // det. Over 200 draws x D directions x 4 sizes an accidental
            // near-cancellation is otherwise close to certain.
            for (int a = 0; a < D; a++) {
                double scale = std::max(std::fabs(ref.g[a]), std::fabs(ref.v));
                CHECK(std::fabs(got.g[a] - ref.g[a]) <= 1e-10 * scale,
                      "test 1: determinant gradient");
            }

            // l = det*(term1 + term2) is a sum of two independently-large
            // terms. Same reasoning as above; for random data max() picks
            // |ref.l| essentially always, so the exponent never moves.
            double scale_l = std::max(std::fabs(ref.l), std::fabs(t1) + std::fabs(t2));
            CHECK(std::fabs(got.l - ref.l) <= 1e-10 * scale_l,
                  "test 1: determinant laplacian");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: column-local rank structure. If column c depends only on directions
// belonging to c, then G_a is nonzero only in column c(a), so B_a is too, so
// B_a[j][i] = 0 unless i = c(a), hence tr(B_a^2) = B[c][c]^2 = (tr B_a)^2 and
// term 2 is EXACTLY zero. This is the regression test for the cross-term
// logic: term 2 is the piece that would be silently wrong if someone assumed
// column-local dependence, and it is nonzero in the real ansatz only because
// psi_impl subtracts the centre of mass.
// ---------------------------------------------------------------------------
static void test_column_local() {
    Scratch s;
    std::mt19937 rng(777u);

    for (int n : {2, 3, 6, 8}) {
        auto col_of = [n](int a) { return (a * n) / D; };

        for (int trial = 0; trial < 50; trial++) {
            std::vector<Jet> M;
            rand_M(M, n, rng);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    for (int a = 0; a < D; a++)
                        if (col_of(a) != c) M[r*n+c].g[a] = 0.0;

            Jet got = call_new(M, n, s);
            if (got.v == 0.0) continue;
            if (cond_inf(M, s.Minv, n) > 1e6) continue;

            // (a) Rebuild B_a independently and verify the structure claim
            //     directly: off-column entries vanish, and the j != i part of
            //     tr(B_a^2) cancels the cross part of (tr B_a)^2 exactly.
            std::vector<double> Ba((std::size_t)n*n);
            for (int a = 0; a < D; a++) {
                int ca = col_of(a);
                double bmax = 0.0;
                for (int j = 0; j < n; j++)
                    for (int c = 0; c < n; c++) {
                        double b = 0.0;
                        for (int r = 0; r < n; r++) b += s.Minv[j*n+r] * M[r*n+c].g[a];
                        Ba[j*n+c] = b;
                        bmax = std::max(bmax, std::fabs(b));
                    }
                for (int j = 0; j < n; j++)
                    for (int c = 0; c < n; c++)
                        if (c != ca)
                            CHECK(std::fabs(Ba[j*n+c]) <= 1e-12 * std::max(1.0, bmax),
                                  "test 2: B_a nonzero outside its own column");

                double trB = 0.0, trB2 = 0.0;
                for (int j = 0; j < n; j++) trB += Ba[j*n+j];
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) trB2 += Ba[j*n+i] * Ba[i*n+j];
                CHECK(std::fabs(trB*trB - trB2) <= 1e-12 * std::max(1.0, trB*trB),
                      "test 2: (tr B)^2 - tr(B^2) must vanish when column-local");
            }

            // (b) The observable consequence at the API boundary: term 2 = 0,
            //     i.e. the whole Laplacian is term 1.
            double t1 = term1_of(M, s.Minv, n, got.v);
            double t2 = got.l - t1;
            CHECK(std::fabs(t2) <= 1e-12 * std::max(1.0, std::fabs(t1)),
                  "test 2: term 2 must vanish under column-local dependence");

            // (c) And the result still matches the oracle.
            Jet ref = call_ref(M, n);
            CHECK(std::fabs(got.v - ref.v) <= 1e-10 * std::fabs(ref.v), "test 2: value");
            for (int a = 0; a < D; a++) {
                double scale = std::max(std::fabs(ref.g[a]), std::fabs(ref.v));
                CHECK(std::fabs(got.g[a] - ref.g[a]) <= 1e-10 * scale, "test 2: gradient");
            }
            double scale_l = std::max(std::fabs(ref.l), std::fabs(t1) + std::fabs(t2));
            CHECK(std::fabs(got.l - ref.l) <= 1e-10 * scale_l, "test 2: laplacian");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: near-singular and exactly singular.
// ---------------------------------------------------------------------------
static void test_singular() {
    Scratch s;
    std::mt19937 rng(4242u);

    // Two nearly-parallel columns. The attainable accuracy here is set by the
    // conditioning, and it is NOT the same for all three outputs: value and
    // gradient are first order in the matrix entries and degrade like
    // cond*eps, but the laplacian is second order (term 2 is quadratic in B)
    // and degrades like cond^2*eps. Measured over this exact draw sequence:
    //
    //   angle   cond      value     gradient   laplacian
    //   1e-2    6.7e4     2.0e-13   4.5e-13    2.3e-10
    //   1e-3    6.7e5     2.0e-12   4.8e-12    3.5e-08
    //   1e-4    6.7e6     2.0e-11   2.5e-11    8.0e-07
    //   1e-5    6.7e7     2.0e-10   6.8e-10    3.2e-04
    //   1e-6    6.7e8     2.0e-09   2.5e-09    2.8e-02
    //
    // Each decade of angle costs the laplacian two decades. So at angle 1e-6
    // there is no 1e-6 of laplacian information left in EITHER implementation;
    // demanding it would only measure which rounding path we happened to take.
    // Two degeneracies are therefore run: 1e-3, where all three hold at 1e-6
    // with ~30x margin, and 1e-6 as specified, where value and gradient still
    // hold at 1e-6 with ~400x margin and the laplacian gets a loose bound that
    // a structurally wrong term 2 (relative error O(1)) would still trip.
    struct NearCase { double angle; double tol_vg; double tol_l; };
    const NearCase near_cases[] = {
        {1e-3, 1e-6, 1e-6},
        {1e-6, 1e-6, 1e-1},
    };

    for (const NearCase& nc : near_cases) {
      std::mt19937 nrng(4242u);   // same matrices for both degeneracies
      for (int n : {3, 6, 8}) {
        for (int trial = 0; trial < 20; trial++) {
            std::vector<Jet> M;
            rand_M(M, n, nrng);
            for (int r = 0; r < n; r++) {
                Jet& a0 = M[r*n + 0];
                Jet& a1 = M[r*n + 1];
                a1.v = a0.v + nc.angle * a1.v;
                for (int a = 0; a < D; a++) a1.g[a] = a0.g[a] + nc.angle * a1.g[a];
                a1.l = a0.l + nc.angle * a1.l;
            }

            Jet got = call_new(M, n, s);
            if (got.v == 0.0) continue;
            double t1 = term1_of(M, s.Minv, n, got.v);
            double t2 = got.l - t1;
            Jet ref = call_ref(M, n);

            CHECK(std::fabs(got.v - ref.v) <= nc.tol_vg * std::fabs(ref.v),
                  "test 3: near-singular value");
            for (int a = 0; a < D; a++) {
                double scale = std::max(std::fabs(ref.g[a]), std::fabs(ref.v));
                CHECK(std::fabs(got.g[a] - ref.g[a]) <= nc.tol_vg * scale,
                      "test 3: near-singular gradient");
            }
            double scale_l = std::max(std::fabs(ref.l), std::fabs(t1) + std::fabs(t2));
            CHECK(std::fabs(got.l - ref.l) <= nc.tol_l * scale_l,
                  "test 3: near-singular laplacian");
        }
      }
    }

    // Exactly singular. This is built as an all-zero COLUMN, not as two
    // identical columns. Duplicate columns do not reliably trip either guard:
    // elimination leaves a final pivot around 1e-17, nowhere near the < 1e-300
    // underflow threshold in lu_det / lu_det_inv, so both functions return a
    // tiny nonzero determinant and the test fails on correct code. (This is the
    // same trap that cost a debugging cycle on test_record.cpp's node-hit mask,
    // where forced-coincident configurations left dets at ~1e-13.) A zero
    // column stays exactly zero under row operations, so the pivot search at
    // that step returns exactly 0.0, deterministically, in both implementations.
    for (int n : {2, 3, 6, 8}) {
        std::vector<Jet> M;
        rand_M(M, n, rng);
        const int cz = n / 2;
        for (int r = 0; r < n; r++) M[r*n + cz] = Jet(0.0);

        Jet got = call_new(M, n, s);
        Jet ref = call_ref(M, n);

        CHECK(ref.v == 0.0, "test 3: oracle must return exactly zero on a zero column");
        CHECK(got.v == 0.0, "test 3: singular value must be exactly zero");
        CHECK(got.l == 0.0, "test 3: singular laplacian must be exactly zero");
        for (int a = 0; a < D; a++)
            CHECK(got.g[a] == 0.0, "test 3: singular gradient must be exactly zero");
    }
}

// ---------------------------------------------------------------------------
// Test 4: finite differences. The only check here that does not trust
// lu_det<Jet>, so it is what would catch an index-order error in the Jacobi
// formula that the oracle happens to share.
//
//   M[r][c](x) = A[r][c] + sum_a C[r][c][a]*x_a + sum_a E[r][c][a]*x_a^2
//
// so exactly:  v = M(x0),  g[a] = C + 2*E*x0_a,  l = sum_a 2*E[r][c][a].
// ---------------------------------------------------------------------------
static const int n_act = 4;   // active variables; the other D - n_act are inert

struct QuadModel {
    int n;
    std::vector<double> A, C, E;   // n*n, n*n*n_act, n*n*n_act
};

static double model_entry(const QuadModel& m, int r, int c, const double* xa) {
    double v = m.A[r*m.n + c];
    for (int a = 0; a < n_act; a++) {
        double x = xa[a];
        v += m.C[(r*m.n + c)*n_act + a] * x + m.E[(r*m.n + c)*n_act + a] * x * x;
    }
    return v;
}

static double model_det(const QuadModel& m, const double* xa) {
    std::vector<double> S((std::size_t)m.n*m.n);
    std::vector<int> piv;
    for (int r = 0; r < m.n; r++)
        for (int c = 0; c < m.n; c++)
            S[r*m.n + c] = model_entry(m, r, c, xa);
    return lu_det<double>(S, m.n, piv);
}

static void model_jets(const QuadModel& m, const double* xa, std::vector<Jet>& M) {
    M.assign((std::size_t)m.n*m.n, Jet());
    for (int r = 0; r < m.n; r++)
        for (int c = 0; c < m.n; c++) {
            Jet& e = M[r*m.n + c];
            e.v = model_entry(m, r, c, xa);
            e.l = 0.0;
            for (int a = 0; a < n_act; a++) {
                double Cv = m.C[(r*m.n + c)*n_act + a];
                double Ev = m.E[(r*m.n + c)*n_act + a];
                e.g[a] = Cv + 2.0 * Ev * xa[a];
                e.l   += 2.0 * Ev;
            }
        }
}

static void test_finite_difference() {
    Scratch s;
    std::mt19937 rng(99991u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    for (int n : {3, 6}) {
        QuadModel m;
        m.n = n;
        m.A.resize((std::size_t)n*n);
        m.C.resize((std::size_t)n*n*n_act);
        m.E.resize((std::size_t)n*n*n_act);
        // Diagonally dominant + weak x-dependence keeps det O(1) and the
        // matrix well conditioned across the FD stencil.
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) {
                m.A[r*n+c] = 0.3 * u(rng) + (r == c ? 2.0 : 0.0);
                for (int a = 0; a < n_act; a++) {
                    m.C[(r*n+c)*n_act + a] = 0.1 * u(rng);
                    m.E[(r*n+c)*n_act + a] = 0.1 * u(rng);
                }
            }

        double x0[n_act];
        for (int a = 0; a < n_act; a++) x0[a] = 0.2 * u(rng);

        std::vector<Jet> M;
        model_jets(m, x0, M);
        Jet got = call_new(M, n, s);
        CHECK(got.v != 0.0, "test 4: model determinant should not be singular");

        double d0 = model_det(m, x0);
        CHECK(std::fabs(got.v - d0) <= 1e-12 * std::max(1.0, std::fabs(d0)),
              "test 4: value vs direct double determinant");

        // Central differences. h1 = 1e-5 balances h^2 truncation (1e-10)
        // against eps/h roundoff (1e-11).
        const double h1 = 1e-5;
        for (int a = 0; a < n_act; a++) {
            double xp[n_act], xm[n_act];
            for (int b = 0; b < n_act; b++) { xp[b] = x0[b]; xm[b] = x0[b]; }
            xp[a] += h1; xm[a] -= h1;
            double g_fd = (model_det(m, xp) - model_det(m, xm)) / (2.0 * h1);
            double scale = std::max(std::fabs(g_fd), std::fabs(d0));
            CHECK(std::fabs(got.g[a] - g_fd) <= 1e-6 * scale,
                  "test 4: gradient vs central difference");
        }

        // Directions with no x-dependence must come back exactly zero.
        for (int a = n_act; a < D; a++)
            CHECK(got.g[a] == 0.0, "test 4: inert direction must have zero gradient");

        // Second differences. h2 = 1e-4 balances h^2 truncation (1e-8) against
        // eps/h^2 roundoff (1e-8).
        const double h2 = 1e-4;
        double lap_fd = 0.0;
        for (int a = 0; a < n_act; a++) {
            double xp[n_act], xm[n_act];
            for (int b = 0; b < n_act; b++) { xp[b] = x0[b]; xm[b] = x0[b]; }
            xp[a] += h2; xm[a] -= h2;
            lap_fd += (model_det(m, xp) - 2.0 * d0 + model_det(m, xm)) / (h2 * h2);
        }
        double scale = std::max(std::fabs(lap_fd), std::fabs(d0));
        CHECK(std::fabs(got.l - lap_fd) <= 1e-6 * scale,
              "test 4: laplacian vs second difference");
    }
}

int main() {
    test_random_vs_oracle();
    test_column_local();
    test_singular();
    test_finite_difference();

    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
