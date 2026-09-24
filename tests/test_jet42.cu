// Phase 4.2: det jets, psi jet composition, E_kin / L^2 / V_3N, validity.
//
// E2 is the load-bearing test: pj.l is the only quantity that exercises term2 of
// the determinant jet, the activation f'', and the envelope composition all at
// once. If g passes and l fails, suspect those three in that order.
#include "../lib/gpu/arena.h"
#include "../tests/test_tolerances.h"
#include "../lib/gpu/jet_kernels.h"
#include "../lib/gpu/detjet_kernels.h"
#include "../lib/gpu/compose_kernels.h"
#include "../lib/gpu/jet_eval.h"
#include "../lib/gpu/eval.h"
#include "../lib/physics.h"
#include "../lib/slater.h"
#include "../tests/test_common.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
    g_failures++; } } while (0)
static double rel(double got, double want) {
    return std::fabs(got - want) / std::max(1.0, std::fabs(want));
}

// cfg_kind: 0 = ordinary, 1 = two particles forced coincident (near-singular
// Slater columns), 2 = coordinates blown up so the envelope underflows to 0.
static void rand_cfg(int B, std::vector<double>& hx, std::vector<double>& hs,
                     std::vector<double>& ht, std::mt19937_64& rng,
                     const std::vector<int>* kinds = nullptr) {
    std::uniform_real_distribution<double> d(-x_init_range, x_init_range);
    hx.assign((std::size_t)B*D, 0.0); hs.assign((std::size_t)B*N, 0.0); ht.assign((std::size_t)B*N, 0.0);
    for (auto& v : hx) v = d(rng);
    for (int w = 0; w < B; w++) {
        for (int i = 0; i < N; i++) {
            hs[(std::size_t)w*N+i] = (i < N_u) ? 1.0 : -1.0;
            ht[(std::size_t)w*N+i] = (i < N_p) ? 1.0 : -1.0;
        }
        if (!kinds) continue;
        if ((*kinds)[w] == 1) {
            for (int d2 = 0; d2 < dim; d2++) hx[(std::size_t)w*D + 1*dim + d2] = hx[(std::size_t)w*D + 0*dim + d2];
        } else if ((*kinds)[w] == 2) {
            for (int i = 0; i < D; i++) hx[(std::size_t)w*D + i] *= 1e4;
        }
    }
}

struct Dev {
    DeviceState ds;
    explicit Dev(const Ansatz& a) : ds(a, false) {
        ds.grow_phase3(a, false); ds.grow_phase33(false);
        ds.grow_phase4(false);    ds.grow_phase42(false);
    }
    void push(const std::vector<double>& hx, const std::vector<double>& hs,
              const std::vector<double>& ht, const Ansatz& a) {
        PinnedArray st; ds.upload_params(a, st);
        std::vector<real> tx(hx.begin(),hx.end()), tsp(hs.begin(),hs.end()), tt(ht.begin(),ht.end());
        ds.x.up(tx.data(),tx.size()); ds.s.up(tsp.data(),tsp.size()); ds.t.up(tt.data(),tt.size());
    }
};

// --- E1: det jet oracle -----------------------------------------------------
static void test_det_jet(const Ansatz& a, cublasHandle_t h) {
    const int B = 256;
    std::mt19937_64 rng(4242);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    Dev dv(a); dv.push(hx, hs, ht, a);
    eval_jet_batch(dv.ds, h, B);

    std::vector<real> Jd(dv.ds.jet_det.n); dv.ds.jet_det.down(Jd.data(), Jd.size());
    std::vector<real> De(dv.ds.dets.n);     dv.ds.dets.down(De.data(), De.size());
    const std::size_t ks = jet_block_stride((std::size_t)B, (std::size_t)K);

    Workspace ws;
    double wv = 0, wg = 0, wl = 0;
    int n_zero = 0;
    std::size_t n_vbit = 0;
    for (int w = 0; w < B; w++) {
        std::vector<double> x(hx.begin()+(std::size_t)w*D, hx.begin()+(std::size_t)(w+1)*D);
        std::vector<double> s(hs.begin()+(std::size_t)w*N, hs.begin()+(std::size_t)(w+1)*N);
        std::vector<double> t(ht.begin()+(std::size_t)w*N, ht.begin()+(std::size_t)(w+1)*N);
        jpsi(x, s, t, a, ws);                      // leaves the jet Slater matrices in ws.jM
        for (int jd = 0; jd < K; jd++) {
            Jet ref = det_jet_from_minv(&ws.jM[(std::size_t)jd*N*N], N, ws.jd_Mval, ws.jd_Minv,
                                       ws.jd_piv, ws.jd_col, ws.jd_G, ws.jd_B);
            if (ref.v == 0.0) n_zero++;
            const std::size_t off = (std::size_t)w*K + jd;
            if (Jd[off] != De[off]) n_vbit++;
            wv = std::max(wv, rel((double)Jd[off], ref.v));
            wl = std::max(wl, rel((double)Jd[(jet_C-1)*ks + off], ref.l));
            for (int A = 0; A < D; A++)
                wg = std::max(wg, rel((double)Jd[(std::size_t)(1+A)*ks + off], ref.g[A]));
        }
    }
    std::printf("  det jet vs det_jet_from_minv: v %.2e  g %.2e  l %.2e   (%d of %d singular)\n",
                wv, wg, wl, n_zero, B*K);
    std::printf("  det jet value block vs ds.dets: %zu of %d differ bitwise\n", n_vbit, B*K);
    // The value block is ds.dets copied verbatim, so bitwise equality is the real
    // assertion; the 1e-12-ish figure above it is entirely cuBLAS-getrf vs the
    // host's lu_det, which Phase 3.2 measured at max 3.9e-12 and bounded at 1e-10.
    // Asserting 1e-12 here would just be re-testing Phase 3's LU, and failing.
    CHECK(n_vbit == 0, "det jet value block is not ds.dets verbatim");
    CHECK(wv <= tol::ff(1e-10, 5e-2), "det jet value disagrees with CPU det_jet_from_minv beyond the known LU gap");
    CHECK(wg <= tol::ff(1e-10, 5e-2), "det jet gradient disagrees with CPU det_jet_from_minv");
    CHECK(wl <= tol::ff(1e-10, 5e-2), "det jet laplacian disagrees with CPU det_jet_from_minv");
}

// --- E2: full psi jet oracle ------------------------------------------------
static void test_psi_jet(const Ansatz& a, cublasHandle_t h) {
    const int B = 256;
    std::mt19937_64 rng(777);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    Dev dv(a); dv.push(hx, hs, ht, a);
    eval_jet_batch(dv.ds, h, B);

    std::vector<real> Jp(dv.ds.jet_psi.n); dv.ds.jet_psi.down(Jp.data(), Jp.size());
    std::vector<real> Sj(dv.ds.S_jet_v.n); dv.ds.S_jet_v.down(Sj.data(), Sj.size());
    const std::size_t ps = (std::size_t)B;

    Workspace ws;
    double wv = 0, wg = 0, wl = 0;
    for (int w = 0; w < B; w++) {
        std::vector<double> x(hx.begin()+(std::size_t)w*D, hx.begin()+(std::size_t)(w+1)*D);
        std::vector<double> s(hs.begin()+(std::size_t)w*N, hs.begin()+(std::size_t)(w+1)*N);
        std::vector<double> t(ht.begin()+(std::size_t)w*N, ht.begin()+(std::size_t)(w+1)*N);
        Jet pj = jpsi(x, s, t, a, ws);
        wv = std::max(wv, rel((double)Jp[w], pj.v));
        wl = std::max(wl, rel((double)Jp[(jet_C-1)*ps + w], pj.l));
        for (int A = 0; A < D; A++)
            wg = std::max(wg, rel((double)Jp[(std::size_t)(1+A)*ps + w], pj.g[A]));
    }
    std::printf("  psi jet vs CPU jpsi: v %.2e  g %.2e  l %.2e\n", wv, wg, wl);
    CHECK(wv <= tol::ff(1e-10, 1e-2), "psi jet value disagrees with CPU jpsi");
    CHECK(wg <= tol::ff(1e-10, 1e-2), "psi jet gradient disagrees with CPU jpsi");
    CHECK(wl <= tol::ff(1e-10, 1e-2), "psi jet laplacian disagrees with CPU jpsi -- suspect term2, then f''");
}

// --- E3: E_kin, L^2, V_3N ---------------------------------------------------
static void test_energy_terms(const Ansatz& a, cublasHandle_t h) {
    const int B = 256;
    std::mt19937_64 rng(31415);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    Dev dv(a); dv.push(hx, hs, ht, a);
    eval_jet_batch(dv.ds, h, B);

    std::vector<real> Ek(dv.ds.E_kin.n), L2(dv.ds.l2_out.n), V3(dv.ds.v3n_out.n);
    dv.ds.E_kin.down(Ek.data(), Ek.size());
    dv.ds.l2_out.down(L2.data(), L2.size());
    dv.ds.v3n_out.down(V3.data(), V3.size());

    Workspace ws;
    double wk = 0, wl2 = 0, wv3 = 0, kscale = 0, l2scale = 0;
    for (int w = 0; w < B; w++) {
        std::vector<double> x(hx.begin()+(std::size_t)w*D, hx.begin()+(std::size_t)(w+1)*D);
        std::vector<double> s(hs.begin()+(std::size_t)w*N, hs.begin()+(std::size_t)(w+1)*N);
        std::vector<double> t(ht.begin()+(std::size_t)w*N, ht.begin()+(std::size_t)(w+1)*N);

        psi(x, s, t, a, ws, false);                 // fills ws.x_sh, as local_E relies on
        Jet pj = jpsi(x, s, t, a, ws);

        const double ek  = -hbar2_2m * (pj.l / pj.v);
        const double l2v = l2_local(ws.x_sh.data(), pj.g.data(), pj.v);
        const double v3  = V_3N(x);

        kscale  = std::max(kscale,  std::fabs(ek));
        l2scale = std::max(l2scale, std::fabs(l2v));
        wk  = std::max(wk,  std::fabs((double)Ek[w] - ek));
        wl2 = std::max(wl2, rel((double)L2[w], l2v));
        wv3 = std::max(wv3, std::fabs((double)V3[w] - v3));
    }
    std::printf("  E_kin abs %.2e MeV (|E_kin| up to %.3g)   V_3N abs %.2e MeV\n", wk, kscale, wv3);
    std::printf("  L2 rel %.2e (|L2| up to %.3g)\n", wl2, l2scale);
    // E_kin and V_3N are energies in MeV, so an absolute bound is the meaningful
    // one. L^2 is NOT -- it is dimensionless and runs to ~4e4 here, so the same
    // absolute bound would be a 1e-13 relative demand on the largest entries and
    // would fail on rounding alone. It gets a relative bound instead.
    CHECK(wk  <= tol::ff(1e-9, 2.0),  "E_kin disagrees with -hbar2_2m*pj.l/pj.v");   // MeV, per sample
    CHECK(wv3 <= 1e-9,  "V_3N disagrees with the CPU triples loop");
    CHECK(wl2 <= tol::ff(1e-10, 1e-3), "L^2 disagrees with l2_local");
    // Scale guards: without these, both bounds pass trivially if the quantity is 0.
    CHECK(kscale > 1.0,   "E_kin is suspiciously small -- the bound above would be vacuous");
    CHECK(l2scale > 1.0,  "L^2 is suspiciously small -- the bound above would be vacuous");
}

// --- E4: validity parity ----------------------------------------------------
// FINDING this test exists to pin down: on deliberately singular configurations
// the two LU implementations do not agree about WHICH matrices are singular.
// cublasDgetrfBatched reports info>0 (an exactly-zero pivot) for ~27 of 31
// determinants per coincident-particle walker; the host's lu_det lands on exactly
// 0.0 for only ~18-21 of them and leaves the rest at ~1e-12. Device S is then 0
// while host S is O(1e-9), so the two paths disagree about the node itself and no
// comparison of their validity masks is meaningful for those walkers.
//
// The test therefore partitions: where the double paths AGREE on the node
// predicate, parity must be exact. Where they disagree, it only requires the
// disagreement to be confined to the configurations constructed to be singular,
// which is what proves this is the constructed degeneracy and not a general drift.
static void test_validity(const Ansatz& a, cublasHandle_t h) {
    const int B = 384;
    std::mt19937_64 rng(2718);
    std::vector<int> kinds(B);
    for (int w = 0; w < B; w++) kinds[w] = w % 3;     // ordinary / coincident / blown up
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng, &kinds);

    Dev dv(a); dv.push(hx, hs, ht, a);
    eval_jet_batch(dv.ds, h, B);

    std::vector<uint8_t> vmask(dv.ds.valid_jet.n);
    dv.ds.valid_jet.down(vmask.data(), vmask.size());
    std::vector<real> Pd(dv.ds.psi_dbl.n); dv.ds.psi_dbl.down(Pd.data(), Pd.size());
    std::vector<real> Sd(dv.ds.S.n);       dv.ds.S.down(Sd.data(), B);

    Workspace ws;
    int mism = 0, n_valid = 0, n_S = 0, n_psi = 0, n_ek = 0;
    int n_split = 0, n_split_outside = 0, n_split_devstricter = 0;
    double wpd = 0;
    for (int w = 0; w < B; w++) {
        std::vector<double> x(hx.begin()+(std::size_t)w*D, hx.begin()+(std::size_t)(w+1)*D);
        std::vector<double> s(hs.begin()+(std::size_t)w*N, hs.begin()+(std::size_t)(w+1)*N);
        std::vector<double> t(ht.begin()+(std::size_t)w*N, ht.begin()+(std::size_t)(w+1)*N);

        // local_E's chain, verbatim, up to the point 4.2 reproduces.
        double dpsi = psi(x, s, t, a, ws, true);
        double S = 0.0;
        for (int i = 0; i < K; i++) S += ws.drho[i] * ws.dets[i];

        bool want;
        if (!std::isfinite(S) || std::fabs(S) < 1e-290)      { want = false; n_S++; }
        else {
            Jet pj = jpsi(x, s, t, a, ws);
            bool psi_mismatch = std::fabs(pj.v - dpsi) > 1e-6 * std::max(1.0, std::fabs(dpsi));
            if (!std::isfinite(pj.v) || std::fabs(pj.v) < 1e-290 || psi_mismatch) { want = false; n_psi++; }
            else {
                double E = -hbar2_2m * (pj.l / pj.v);
                if (!std::isfinite(E)) { want = false; n_ek++; }
                else { want = true; n_valid++; }
            }
        }

        // Do the two double paths even agree that this walker is on a node?
        const bool cpu_node = !std::isfinite(S)              || std::fabs(S)              < 1e-290;
        const bool dev_node = !std::isfinite((double)Sd[w])  || std::fabs((double)Sd[w])  < 1e-290;
        if (cpu_node != dev_node) {
            n_split++;
            if (kinds[w] != 1) n_split_outside++;
            if (dev_node && !cpu_node) n_split_devstricter++;
            continue;                       // not a 4.2 question; see the note above
        }

        wpd = std::max(wpd, rel((double)Pd[w], dpsi));
        if ((vmask[w] != 0) != want) mism++;
    }
    std::printf("  validity: %d of %d disagree where the double paths agree on the node\n", mism, B - n_split);
    std::printf("    CPU outcomes: %d valid, %d S-guard, %d psi-guard, %d E-guard\n",
                n_valid, n_S, n_psi, n_ek);
    std::printf("    LU singularity split: %d walkers (%d outside the coincident set, %d device-stricter)\n",
                n_split, n_split_outside, n_split_devstricter);
    std::printf("  psi_double vs CPU psi(): %.2e\n", wpd);
    CHECK(mism == 0, "device validity mask disagrees with local_E's early-return chain");
    CHECK(n_split_outside == 0, "the LU singularity split reaches walkers that were not constructed singular");
    CHECK(wpd <= tol::ff(1e-10, 1e-3), "device psi_double disagrees with CPU psi()");
    CHECK(n_valid > 0 && n_psi > 0, "validity test saw only one outcome -- it proves nothing");
}

// Guard logic in isolation: the S and mismatch branches are not reachable from
// any coordinate configuration I can construct (|S| stays O(1) and the jet and
// double paths agree to 1e-13), so they are driven by poking the buffers the
// kernel reads. Without this, two of the four branches are dead code in testing.
static void test_validity_guards(const Ansatz& a, cublasHandle_t h) {
    const int B = 8;
    std::mt19937_64 rng(99);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    Dev dv(a); dv.push(hx, hs, ht, a);
    eval_jet_batch(dv.ds, h, B);

    std::vector<uint8_t> base(dv.ds.valid_jet.n);
    dv.ds.valid_jet.down(base.data(), base.size());
    for (int w = 0; w < B; w++) CHECK(base[w] == 1, "baseline config should be valid");

    std::vector<real> Sh(B); dv.ds.S.down(Sh.data(), B);
    std::vector<real> Jp(dv.ds.jet_psi.n); dv.ds.jet_psi.down(Jp.data(), Jp.size());
    std::vector<real> Ek(dv.ds.E_kin.n);   dv.ds.E_kin.down(Ek.data(), Ek.size());

    std::vector<real> S2(Sh), Jp2(Jp), Ek2(Ek);
    S2[0] = (real)NAN;                    // guard 1, non-finite S
    S2[1] = (real)1e-300;                 // guard 1, |S| below 1e-290
    Jp2[2] = (real)NAN;                   // guard 2, non-finite pj.v
    Jp2[3] = (real)0;                     // guard 2, |pj.v| below 1e-290
    Jp2[4] = Jp[4] * (real)1.5;           // guard 2, mismatch against psi_double
    Ek2[5] = (real)INFINITY;              // guard 3, non-finite E_kin
    dv.ds.S.up(S2.data(), B);
    dv.ds.jet_psi.up(Jp2.data(), Jp2.size());
    dv.ds.E_kin.up(Ek2.data(), Ek2.size());

    validity_jet(dv.ds.jet_psi.d, dv.ds.S.d, dv.ds.x_sh.d, dv.ds.s.d, dv.ds.t.d, dv.ds.params.d, dv.ds.P,
                 dv.ds.E_kin.d, dv.ds.psi_dbl.d, dv.ds.valid_jet.d, B, 0, B);

    std::vector<uint8_t> got(dv.ds.valid_jet.n);
    dv.ds.valid_jet.down(got.data(), got.size());
    const uint8_t want[8] = {0,0,0,0,0,0,1,1};
    int bad = 0;
    for (int w = 0; w < B; w++) if (got[w] != want[w]) {
        bad++;
        std::printf("    guard w=%d: got %d want %d\n", w, (int)got[w], (int)want[w]);
    }
    std::printf("  validity guards (NaN S, tiny S, NaN psi, zero psi, mismatch, NaN E_kin): %d wrong\n", bad);
    CHECK(bad == 0, "a validity guard branch does not fire");
}

// --- E5: chunk invariance ---------------------------------------------------
static void test_chunk_invariance(const Ansatz& a, cublasHandle_t h) {
    const int B = 240;
    std::mt19937_64 rng(13);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    Dev dv(a); dv.push(hx, hs, ht, a);

    eval_jet_batch(dv.ds, h, B);                   // one chunk (jet_chunk == 0)
    std::vector<real> Jp1(dv.ds.jet_psi.n), Jd1(dv.ds.jet_det.n), Ek1(dv.ds.E_kin.n);
    dv.ds.jet_psi.down(Jp1.data(), Jp1.size());
    dv.ds.jet_det.down(Jd1.data(), Jd1.size());
    dv.ds.E_kin.down(Ek1.data(), Ek1.size());

    dv.ds.jet_psi.zero(); dv.ds.jet_det.zero(); dv.ds.E_kin.zero();

    eval_jet_prepare(dv.ds, h, B);                 // same double path, chunked jets
    for (int off = 0; off < B; off += 37) {
        const int Bc = std::min(37, B - off);
        eval_jet_chunk(dv.ds, h, Bc, off, B);
    }
    std::vector<real> Jp2(dv.ds.jet_psi.n), Jd2(dv.ds.jet_det.n), Ek2(dv.ds.E_kin.n);
    dv.ds.jet_psi.down(Jp2.data(), Jp2.size());
    dv.ds.jet_det.down(Jd2.data(), Jd2.size());
    dv.ds.E_kin.down(Ek2.data(), Ek2.size());

    std::size_t np = 0, nd = 0, ne = 0;
    for (int c = 0; c < jet_C; c++) for (int w = 0; w < B; w++) {
        if (Jp1[(std::size_t)c*B + w] != Jp2[(std::size_t)c*B + w]) np++;
        for (int k = 0; k < K; k++) {
            const std::size_t o = (std::size_t)c*jet_block_stride((std::size_t)B,(std::size_t)K) + (std::size_t)w*K + k;
            if (Jd1[o] != Jd2[o]) nd++;
        }
    }
    for (int w = 0; w < B; w++) if (Ek1[w] != Ek2[w]) ne++;
    std::printf("  chunk invariance (B=240 whole vs 37-walker chunks): psi %zu, det %zu, E_kin %zu differ\n",
                np, nd, ne);
    // Bitwise: chunking changes only GEMM row counts and jet block strides, never
    // an accumulation order, so anything but exact equality is a layout bug.
    // Not under fp32_forward: cuBLAS SGEMM picks its kernel -- and so its
    // reduction order -- by matrix size, so the jet psi / E_kin move at float
    // rounding level with the chunk. The determinants (FP64 path) stay exact,
    // and run-to-run determinism at a fixed chunk is unaffected.
    CHECK(fp32_forward ? nd == 0 : (np == 0 && nd == 0 && ne == 0), "chunking changes the result");
}

int main() {
    gpu_select_device(true);
    Ansatz a({64},{64},{64}, Activation::Gelu);
    seed_ansatz(a, 2024);
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    try {
        test_det_jet(a, h);
        test_psi_jet(a, h);
        test_energy_terms(a, h);
        test_validity(a, h);
        test_validity_guards(a, h);
        test_chunk_invariance(a, h);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
