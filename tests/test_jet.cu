// Phase 4.1 Task C: the jet pass against the CPU jet path.
//
// The load-bearing test is test_jet_oracle: it is the only thing that proves the
// ANALYTIC shifted-input seeds are equivalent to the CPU's composition, which
// seeds raw-coordinate jets and subtracts the CM through jet arithmetic. Every
// other test here checks a piece in isolation.
#include "../lib/gpu/arena.h"
#include "../lib/gpu/jet_kernels.h"
#include "../lib/gpu/net_forward.h"
#include "../lib/physics.h"
#include "../lib/network.h"
#include "../tests/test_common.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
    g_failures++; } } while (0)
static double rel(double got, double want) {
    return std::fabs(got - want) / std::max(1.0, std::fabs(want));
}

static void rand_cfg(int B, std::vector<double>& hx, std::vector<double>& hs,
                     std::vector<double>& ht, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> d(-x_init_range, x_init_range);
    hx.assign((std::size_t)B*D, 0.0); hs.assign((std::size_t)B*N, 0.0); ht.assign((std::size_t)B*N, 0.0);
    for (auto& v : hx) v = d(rng);
    for (int w = 0; w < B; w++) for (int i = 0; i < N; i++) {
        hs[(std::size_t)w*N+i] = (i < N_u) ? 1.0 : -1.0;
        ht[(std::size_t)w*N+i] = (i < N_p) ? 1.0 : -1.0; }
}

// --- C1: seed correctness ---------------------------------------------------
static void test_seeds(const Ansatz& a) {
    const int B = 32;
    std::mt19937_64 rng(11);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    DeviceState ds(a,false); ds.grow_phase3(a,false); ds.grow_phase33(false); ds.grow_phase4(false);
    std::vector<real> tx(hx.begin(),hx.end()), tsp(hs.begin(),hs.end()), tt(ht.begin(),ht.end());
    ds.x.up(tx.data(),tx.size()); ds.s.up(tsp.data(),tsp.size()); ds.t.up(tt.data(),tt.size());
    build_jet_feat(ds.x.d, ds.s.d, ds.t.d, ds.jet_feat.d, B);

    const int Wf = dim + 2;
    // Blocks are packed for THIS call's rows, not the buffer capacity -- that
    // tight packing is what makes the single GEMM over C*rows valid.
    const std::size_t rows = (std::size_t)B * N;
    const std::size_t stride = jet_block_stride(rows, (std::size_t)Wf);
    std::vector<real> J(ds.jet_feat.n);
    ds.jet_feat.down(J.data(), J.size());

    double wv = 0, wg = 0, wl = 0;
    for (int w = 0; w < B; w++) {
        double R[dim] = {};
        for (int d = 0; d < dim; d++) {
            double acc = 0; for (int i = 0; i < N; i++) acc += hx[(std::size_t)w*D+i*dim+d];
            R[d] = acc / N;
        }
        for (int p = 0; p < N; p++) {
            const std::size_t r = (std::size_t)w*N + p;
            for (int d = 0; d < dim; d++)
                wv = std::max(wv, rel((double)J[r*Wf+d], hx[(std::size_t)w*D+p*dim+d]-R[d]));
            wv = std::max(wv, rel((double)J[r*Wf+dim],   hs[(std::size_t)w*N+p]));
            wv = std::max(wv, rel((double)J[r*Wf+dim+1], ht[(std::size_t)w*N+p]));

            // gradient blocks: delta_{d d'} (delta_{pj} - 1/N), zero on s and t
            for (int A = 0; A < D; A++) {
                const int j = A/dim, dp = A%dim;
                const real* g = &J[(std::size_t)(1+A)*stride + r*Wf];
                for (int d = 0; d < dim; d++) {
                    double want = (d==dp) ? ((p==j?1.0:0.0) - 1.0/N) : 0.0;
                    wg = std::max(wg, std::fabs((double)g[d] - want));
                }
                wg = std::max(wg, std::fabs((double)g[dim]));
                wg = std::max(wg, std::fabs((double)g[dim+1]));
            }
            // laplacian block identically zero
            const real* l = &J[(std::size_t)(jet_C-1)*stride + r*Wf];
            for (int c = 0; c < Wf; c++) wl = std::max(wl, std::fabs((double)l[c]));
        }
    }
    std::printf("  jet seeds: value %.2e  gradient %.2e  laplacian %.2e\n", wv, wg, wl);
    CHECK(wv <= 1e-15, "jet seed values disagree with the host CM shift");
    CHECK(wg == 0.0,   "jet seed gradients are not exactly (delta - 1/N)");
    CHECK(wl == 0.0,   "jet seed laplacian block is not exactly zero");
}

// --- C3: activation parity + act_hess vs central differences ---------------
static void test_jet_activation(const Ansatz& a) {
    // act_hess against central differences of act_grad
    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<double> uz(-6,6), ug(-1,1);
    for (Activation act : {Activation::Gelu, Activation::Tanh}) {
        double worst = 0; const double h = 1e-5;
        for (int i = 0; i < 20000; i++) {
            double z = uz(rng);
            double fd = (act_grad(act, z+h) - act_grad(act, z-h))/(2*h);
            worst = std::max(worst, rel(act_hess(act, z), fd));
        }
        std::printf("  act_hess %-4s vs central diff: %.2e\n",
                    act==Activation::Gelu?"Gelu":"Tanh", worst);
        CHECK(worst <= 1e-7, "act_hess disagrees with central differences");
    }

    // device jet_bias_act on one wide row vs CPU apply_activation<Jet>
    const int rows = 1, width = 4096;
    for (Activation act : {Activation::Gelu, Activation::Tanh}) {
        std::vector<Jet> in(width);
        for (auto& j : in) { j.v = uz(rng); j.l = ug(rng)*5.0;
                             for (int A=0;A<D;A++) j.g[A]=ug(rng); }
        std::vector<double> bias(width, 0.0);    // bias folded into v already

        const std::size_t stride = jet_block_stride((std::size_t)rows, (std::size_t)width);
        std::vector<real> J((std::size_t)jet_C*stride, (real)0);
        for (int n = 0; n < width; n++) {
            J[0*stride + n] = (real)in[n].v;
            for (int A=0;A<D;A++) J[(std::size_t)(1+A)*stride + n] = (real)in[n].g[A];
            J[(std::size_t)(jet_C-1)*stride + n] = (real)in[n].l;
        }
        DeviceArray<real> dJ, dB;
        dJ.alloc(J.size()); dJ.up(J.data(), J.size());
        dB.alloc(bias.size()); { std::vector<real> tb(bias.begin(),bias.end()); dB.up(tb.data(), tb.size()); }
        jet_bias_act(dJ.d, dB.d, rows, width, width, act, /*is_output=*/false);
        dJ.down(J.data(), J.size());

        double wv=0, wg=0, wl=0;
        for (int n = 0; n < width; n++) {
            Jet ref = apply_activation(act, in[n]);
            wv = std::max(wv, rel((double)J[0*stride+n], ref.v));
            wl = std::max(wl, rel((double)J[(std::size_t)(jet_C-1)*stride+n], ref.l));
            for (int A=0;A<D;A++)
                wg = std::max(wg, rel((double)J[(std::size_t)(1+A)*stride+n], ref.g[A]));
        }
        std::printf("  jet_bias_act %-4s vs apply_activation<Jet>: v %.2e  g %.2e  l %.2e\n",
                    act==Activation::Gelu?"Gelu":"Tanh", wv, wg, wl);
        // Analytic f/f'/f'' vs the CPU's COMPOSED operators: mathematically
        // identical, numerically not. Measured 2.8e-14 on Gelu's laplacian and
        // exactly 0 for Tanh (autodiff.h's tanh already uses this form).
        CHECK(wv <= 1e-12 && wg <= 1e-12 && wl <= 1e-12, "jet activation parity");
    }
}

// --- C4: full h / xi / rho / orb jet oracle ---------------------------------
static void test_jet_oracle(const Ansatz& a, cublasHandle_t handle) {
    const int B = 256;
    std::mt19937_64 rng(90210);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);

    DeviceState ds(a,false); ds.grow_phase3(a,false); ds.grow_phase33(false); ds.grow_phase4(false);
    PinnedArray st; ds.upload_params(a, st);
    std::vector<real> tx(hx.begin(),hx.end()), tsp(hs.begin(),hs.end()), tt(ht.begin(),ht.end());
    ds.x.up(tx.data(),tx.size()); ds.s.up(tsp.data(),tsp.size()); ds.t.up(tt.data(),tt.size());

    const int hidden = std::max(1, std::max(std::max(a.h_net.layers[0].output_size,
                                                     a.rho_net.layers[0].output_size),
                                            a.orb_net.layers[0].output_size));
    build_jet_feat(ds.x.d, ds.s.d, ds.t.d, ds.jet_feat.d, B);
    jet_net_forward(handle, ds.h_net_d,   ds.params.d, ds.jet_feat.d, dim+2, B*N,
                    ds.jet_a.d, ds.jet_b.d, hidden, ds.jet_h.d, m_feat);
    jet_xi_reduce(ds.jet_h.d, ds.jet_xi.d, B);
    jet_net_forward(handle, ds.rho_net_d, ds.params.d, ds.jet_xi.d, m_feat, B,
                    ds.jet_a.d, ds.jet_b.d, hidden, ds.jet_rho.d, K);
    jet_net_forward(handle, ds.orb_net_d, ds.params.d, ds.jet_feat.d, dim+2, B*N,
                    ds.jet_a.d, ds.jet_b.d, hidden, ds.jet_orb.d, K*N);

    std::vector<real> Jh(ds.jet_h.n), Jxi(ds.jet_xi.n), Jrho(ds.jet_rho.n), Jorb(ds.jet_orb.n);
    ds.jet_h.down(Jh.data(),Jh.size());   ds.jet_xi.down(Jxi.data(),Jxi.size());
    ds.jet_rho.down(Jrho.data(),Jrho.size()); ds.jet_orb.down(Jorb.data(),Jorb.size());

    const std::size_t rows = (std::size_t)B * N, wrows = (std::size_t)B;
    const std::size_t sxi = jet_block_stride(wrows, (std::size_t)m_feat);
    const std::size_t srh = jet_block_stride(wrows, (std::size_t)K);
    const std::size_t sob = jet_block_stride(rows,  (std::size_t)(K*N));

    Workspace ws;
    std::vector<Jet> single(dim+2), ba, bb;
    double wxi=0, wrho=0, worb=0;
    auto cmp = [&](double& acc, const real* base, std::size_t stride, std::size_t off, const Jet& ref){
        acc = std::max(acc, rel((double)base[0*stride+off], ref.v));
        acc = std::max(acc, rel((double)base[(std::size_t)(jet_C-1)*stride+off], ref.l));
        for (int A=0;A<D;A++)
            acc = std::max(acc, rel((double)base[(std::size_t)(1+A)*stride+off], ref.g[A]));
    };

    for (int w = 0; w < B; w++) {
        // CPU jet path: jpsi leaves jxi, jrho and jin_sh populated.
        std::vector<double> x(hx.begin()+(std::size_t)w*D, hx.begin()+(std::size_t)(w+1)*D);
        std::vector<double> s(hs.begin()+(std::size_t)w*N, hs.begin()+(std::size_t)(w+1)*N);
        std::vector<double> t(ht.begin()+(std::size_t)w*N, ht.begin()+(std::size_t)(w+1)*N);
        jpsi(x, s, t, a, ws);

        for (int f = 0; f < m_feat; f++)
            cmp(wxi, Jxi.data(), sxi, (std::size_t)w*m_feat + f, ws.jxi[f]);
        for (int k = 0; k < K; k++)
            cmp(wrho, Jrho.data(), srh, (std::size_t)w*K + k, ws.jrho[k]);

        // orb jets per particle, from the CPU's own shifted-coordinate jets
        for (int p = 0; p < N; p++) {
            for (int d = 0; d < dim; d++) single[d] = ws.jin_sh[p*dim + d];
            single[dim] = Jet(s[p]); single[dim+1] = Jet(t[p]);
            const std::vector<Jet>& ref = *a.orb_net.forward_opt<Jet>(single, a.orb_net.params, ba, bb);
            for (int j = 0; j < K*N; j++)
                cmp(worb, Jorb.data(), sob, ((std::size_t)w*N+p)*(K*N) + j, ref[j]);
        }
    }
    std::printf("  jet oracle vs CPU jpsi: xi %.2e  rho %.2e  orb %.2e\n", wxi, wrho, worb);
    CHECK(wxi  <= 1e-11, "jet xi disagrees with CPU ws.jxi");
    CHECK(wrho <= 1e-11, "jet rho disagrees with CPU ws.jrho");
    CHECK(worb <= 1e-11, "jet orb disagrees with CPU forward_opt<Jet>");
}

int main() {
    gpu_select_device(true);
    Ansatz a({64},{64},{64}, Activation::Gelu);
    seed_ansatz(a, 2024);
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    try {
        test_seeds(a);
        test_jet_activation(a);
        test_jet_oracle(a, h);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
