#include "../lib/walkers.h"
#include "../lib/physics.h"
#include "../lib/constants.h"
#include "../lib/network.h"
#include "../lib/pool.h"
#include "../lib/sr.h"
#include "test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

// Frozen pre-refactor reference (verbatim pre-Task-A local_E body), kept only
// so Test 1 can check the refactor didn't change any math. Never shipped.
static double legacy_local_E(const std::vector<double>& x, const std::vector<double>& s, const std::vector<double>& t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out) {
    std::size_t n_params = a.n_params();
    if (O_out.size() != n_params) O_out.resize(n_params);

    double dpsi = psi(x, s, t, a, ws, true);
    (void)dpsi;

    double S = 0.0;
    for (int i = 0; i < K; i++) S += ws.drho[i] * ws.dets[i];
    if (!std::isfinite(S) || std::fabs(S) < 1e-290) { std::fill(O_out.begin(), O_out.end(), 0.0); return 0.0; }

    Jet pj = jpsi(x, s, t, a, ws);
    bool mismatch = std::fabs(pj.v - dpsi) > 1e-6 * std::max(1.0, std::fabs(dpsi));
    if (!std::isfinite(pj.v) || std::fabs(pj.v) < 1e-290 || mismatch) { std::fill(O_out.begin(), O_out.end(), 0.0); return 0.0; }

    ws.l2_val = l2_local(ws.x_sh.data(), pj.g.data(), pj.v);
    double E_loc = -hbar2_2m * (pj.l/pj.v);
    if (!std::isfinite(E_loc)) { std::fill(O_out.begin(), O_out.end(), 0.0); return 0.0; }

    if (nuc_3N) {
        E_loc += V_3N(x);
        if (!std::isfinite(E_loc)) { std::fill(O_out.begin(), O_out.end(), 0.0); return 0.0; }
    }
    if (nuc_coulomb) {
        E_loc += V_coulomb(x, t);
        if (!std::isfinite(E_loc)) { std::fill(O_out.begin(), O_out.end(), 0.0); return 0.0; }
    }
    if (nuc_pot != NucPot::Off) {
        build_st_table(x, a, ws);
        double S0 = S_from_table(s, t, a, ws);
        bool use_rank2 = rank2_well_conditioned(ws);
        if (ws.s_swap.size() != (std::size_t)N) ws.s_swap.resize(N);
        if (ws.t_swap.size() != (std::size_t)N) ws.t_swap.resize(N);
        for (int i = 0; i < N; i++) { ws.s_swap[i] = s[i]; ws.t_swap[i] = t[i]; }
        double V_nuc = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                bool same_s = (s[i] == s[j]);
                bool same_t = (t[i] == t[j]);
                if (same_s && same_t) continue;
                double r2 = 0.0;
                for (int d = 0; d < dim; d++) { double diff = x[i*dim+d]-x[j*dim+d]; r2 += diff*diff; }
                double v01 = std::exp(-r2/(R01*R01)) / (std::pow(3.14159265358979323846,1.5)*R01*R01*R01);
                double v10 = std::exp(-r2/(R10*R10)) / (std::pow(3.14159265358979323846,1.5)*R10*R10*R10);
                double R_s, R_t, R_st;
                if (same_s) { R_s=1.0; R_t=swap_ratio(s,t,i,j,s[i],t[j],s[j],t[i],S0,use_rank2,a,ws); R_st=R_s*R_t; }
                else if (same_t) { R_t=1.0; R_s=swap_ratio(s,t,i,j,s[j],t[i],s[i],t[j],S0,use_rank2,a,ws); R_st=R_s*R_t; }
                else {
                    R_t=swap_ratio(s,t,i,j,s[i],t[j],s[j],t[i],S0,use_rank2,a,ws);
                    R_s=swap_ratio(s,t,i,j,s[j],t[i],s[i],t[j],S0,use_rank2,a,ws);
                    R_st=swap_ratio(s,t,i,j,s[j],t[j],s[i],t[i],S0,use_rank2,a,ws);
                }
                V_nuc += (hbarc/4.0)*(C01*v01*(1.0+R_t-R_s-R_st) + C10*v10*(1.0-R_t+R_s-R_st));
            }
        }
        E_loc += V_nuc;
        if (!std::isfinite(E_loc)) { std::fill(O_out.begin(), O_out.end(), 0.0); return 0.0; }
    }

    std::size_t n_h = a.h_net.params.size(), n_rho = a.rho_net.params.size(), n_orb = a.orb_net.params.size();
    if (ws.seed_rho.size() != (std::size_t)K) ws.seed_rho.resize(K);
    for (int i = 0; i < K; i++) ws.seed_rho[i] = ws.dets[i];
    a.rho_net.backprop(ws.rho_cache, ws.seed_rho, ws.dtheta_rho, ws.delta_a, ws.delta_b, &ws.dpsi_dxi);
    for (std::size_t p = 0; p < n_rho; p++) O_out[n_h+p] = ws.dtheta_rho[p]/S;
    ws.dtheta_h.assign(n_h, 0.0);
    for (int i = 0; i < N; i++) a.h_net.backprop_acc(ws.h_caches[i], ws.dpsi_dxi, ws.dtheta_h, ws.delta_a, ws.delta_b);
    for (std::size_t p = 0; p < n_h; p++) O_out[p] = ws.dtheta_h[p]/S;
    if (ws.seed_orb.size() != (std::size_t)(K*N)) ws.seed_orb.resize(K*N);
    ws.dtheta_orb.assign(n_orb, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) for (int k = 0; k < N; k++) ws.seed_orb[j*N+k] = ws.drho[j]*ws.dets[j]*ws.dMinv[j*(N*N)+i*N+k];
        a.orb_net.backprop_acc(ws.orb_caches[i], ws.seed_orb, ws.dtheta_orb, ws.delta_a, ws.delta_b);
    }
    for (std::size_t p = 0; p < n_orb; p++) O_out[n_h+n_rho+p] = ws.dtheta_orb[p]/S;
    double r2 = 0.0;
    for (int i = 0; i < D; i++) { double c = ws.x_sh[i]; r2 += c*c; }
    O_out[n_h+n_rho+n_orb] = -std::exp(a.alpha) * std::sqrt(r2 + eps_env*eps_env);
    return E_loc;
}

static void standard_sector(std::vector<double>& s, std::vector<double>& t) {
    s.assign(N, 0.0); t.assign(N, 0.0);
    for (int i = 0; i < N; i++) { s[i] = (i < N_u) ? 1.0 : -1.0; t[i] = (i < N_p) ? 1.0 : -1.0; }
}

static void test_agreement(const Ansatz& a) {
    std::mt19937_64 rng(1234567);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);
    Workspace ws1, ws2;

    for (int trial = 0; trial < 20; trial++) {
        std::vector<double> x(D);
        for (int d = 0; d < D; d++) x[d] = dist(rng);

        std::vector<double> O_legacy(a.n_params()), O_new(a.n_params());
        double E_legacy = legacy_local_E(x, s, t, a, ws1, O_legacy);
        double E_new;
        bool ok = local_E(x.data(), s.data(), t.data(), a, ws2, O_new, E_new);

        CHECK(ok, "agreement: trial " + std::to_string(trial) + " unexpectedly invalid");
        if (!ok) continue;
        CHECK(std::fabs(E_new - E_legacy) < 1e-14, "agreement: E mismatch trial " + std::to_string(trial));
        for (std::size_t k = 0; k < O_legacy.size(); k++) {
            CHECK(std::fabs(O_new[k] - O_legacy[k]) < 1e-14, "agreement: O[" + std::to_string(k) + "] mismatch trial " + std::to_string(trial));
        }
    }
}

static void test_mask_correctness(const Ansatz& a) {
    std::size_t P = a.n_params();
    const int n_samples = 5;
    std::vector<double> E_pool(n_samples, 0.0);
    std::vector<double> O_pool((std::size_t)n_samples * P, 0.0);
    std::vector<uint8_t> valid_pool(n_samples, 0);
    Workspace ws;

    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> s, t; standard_sector(s, t);

    // Samples 0,1,2: real, valid local_E evaluations (retry guards against
    // the rare node/non-finite draw, though at this network scale it's
    // essentially never triggered).
    for (int i = 0; i < 3; i++) {
        std::vector<double> x(D);
        std::vector<double> O(P);
        double E;
        bool ok = false;
        while (!ok) {
            for (int d = 0; d < D; d++) x[d] = dist(rng);
            ok = local_E(x.data(), s.data(), t.data(), a, ws, O, E);
            if (ok) for (std::size_t k = 0; k < P && ok; k++) ok = std::isfinite(O[k]);
        }
        E_pool[i] = E;
        for (std::size_t k = 0; k < P; k++) O_pool[i*P+k] = O[k];
        valid_pool[i] = 1;
    }

    // Samples 3,4: deliberately marked invalid and poisoned with sentinel
    // values -- matches record_batch's contract ("valid_pool[idx]=0 and
    // NOTHING else is required to be meaningful at that slot"). If masking
    // were broken these obviously-wrong values would leak into the average.
    valid_pool[3] = 0;
    valid_pool[4] = 0;
    E_pool[3] = 999999.0;
    E_pool[4] = -999999.0;
    for (std::size_t k = 0; k < P; k++) { O_pool[3*P+k] = 12345.0; O_pool[4*P+k] = -54321.0; }

    double manual_mean = (E_pool[0] + E_pool[1] + E_pool[2]) / 3.0;
    double masked_sum = 0.0; int masked_n = 0;
    for (int i = 0; i < n_samples; i++) if (valid_pool[i]) { masked_sum += E_pool[i]; masked_n++; }
    CHECK(masked_n == 3, "mask correctness: expected exactly 3 valid samples");
    CHECK(std::fabs(masked_sum/masked_n - manual_mean) < 1e-14, "mask correctness: masked E mean wrong");

    ThreadPool pool(1);
    std::vector<double> O_exp;
    masked_O_exp(O_pool, valid_pool, n_samples, P, &pool, O_exp);
    for (std::size_t k = 0; k < P; k++) {
        double expect = (O_pool[0*P+k] + O_pool[1*P+k] + O_pool[2*P+k]) / 3.0;
        CHECK(std::fabs(O_exp[k] - expect) < 1e-12, "mask correctness: masked_O_exp[" + std::to_string(k) + "] wrong");
    }
}



static void test_srop_mask() {
    const std::size_t Ns = 40, P = 6;
    std::mt19937_64 rng(555);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> O_pool(Ns * P), O_exp(P), d_rms(P), v(P);
    std::vector<uint8_t> mask(Ns);
    for (auto& x : O_pool) x = dist(rng);
    for (auto& x : O_exp) x = dist(rng);
    for (auto& x : d_rms) x = std::fabs(dist(rng)) + 0.1;
    for (auto& x : v) x = dist(rng);
    std::size_t n_valid = 0;
    for (std::size_t i = 0; i < Ns; i++) { mask[i] = (rng() % 2 == 0) ? 1 : 0; n_valid += mask[i]; }
    if (n_valid == 0) { mask[0] = 1; n_valid = 1; }

    double lambda = 2.5, eps_abs = 1e-3;
    ThreadPool pool(n_thread);
    SROp op;
    op.init(O_pool, O_exp, Ns, P, lambda, eps_abs, &pool, d_rms.data(), mask.data(), n_valid);
    std::vector<double> out;
    op.apply(v, out, false);

    // Dense serial reference over the valid subset only.
    std::vector<double> ref_S(P, 0.0);
    for (std::size_t i = 0; i < Ns; i++) {
        if (!mask[i]) continue;
        for (std::size_t j = 0; j < P; j++) { double d = O_pool[i*P+j]-O_exp[j]; ref_S[j] += d*d; }
    }
    for (std::size_t j = 0; j < P; j++) ref_S[j] /= (double)n_valid;

    for (std::size_t j = 0; j < P; j++) CHECK(std::fabs(op.S_diag[j]-ref_S[j]) < 1e-12, "SROp mask: S_diag[" + std::to_string(j) + "] mismatch");

    double Oexp_v = 0.0;
    for (std::size_t j = 0; j < P; j++) Oexp_v += O_exp[j]*v[j];
    std::vector<double> ref_out(P, 0.0);
    for (std::size_t i = 0; i < Ns; i++) {
        if (!mask[i]) continue;
        double Ov_i = 0.0;
        for (std::size_t j = 0; j < P; j++) Ov_i += O_pool[i*P+j]*v[j];
        double t_i = Ov_i - Oexp_v;
        for (std::size_t j = 0; j < P; j++) ref_out[j] += t_i * O_pool[i*P+j];
    }
    for (std::size_t j = 0; j < P; j++) {
        ref_out[j] /= (double)n_valid;
        ref_out[j] += lambda * ref_S[j] * v[j];
        double damp = eps_abs;
        if constexpr (sr_rms_damp) damp += sr_rms_eps * d_rms[j];
        ref_out[j] += damp * v[j];
    }
    for (std::size_t j = 0; j < P; j++) CHECK(std::fabs(out[j]-ref_out[j]) < 1e-12, "SROp mask: apply()[" + std::to_string(j) + "] mismatch");
}

static void test_batch_error_sanity() {
    const int B = 500, trials = 200;
    const double mu = 1.0, sigma = 2.0;
    std::mt19937_64 rng(2026);
    std::normal_distribution<double> dist(mu, sigma);

    double err_avg = 0.0;
    for (int trial = 0; trial < trials; trial++) {
        BatchStats bs;
        bs.Ew_sum.resize(B);
        bs.nw.assign(B, 1);
        for (int w = 0; w < B; w++) bs.Ew_sum[w] = dist(rng);
        err_avg += batch_error(bs);
    }
    err_avg /= trials;

    double expected = sigma / std::sqrt((double)B);
    CHECK(std::fabs(err_avg - expected) < 0.2 * expected, "batch_error sanity: avg=" + std::to_string(err_avg) + " expected~" + std::to_string(expected));
}

static void test_determinism_through_record(const Ansatz& a) {
    auto run = [&](int n_workers) {
        WalkerBatch wb; wb.init(84);
        ThreadPool pool(n_workers);
        std::vector<Workspace> wss(n_workers);
        init_batch(wb, a, &pool, wss);
        therm_batch(wb, a, step0, 10, &pool, wss);

        std::size_t P = a.n_params();
        int records = 3;
        std::vector<double> E_pool((std::size_t)records*wb.B), O_pool((std::size_t)records*wb.B*P);
        std::vector<uint8_t> valid_pool((std::size_t)records*wb.B);
        BatchStats bs;
        record_batch(wb, a, step0, records, &pool, wss, E_pool, O_pool, valid_pool, bs);
        return std::make_tuple(wb, E_pool, O_pool, valid_pool);
    };

    auto [wb1, E1, O1, V1] = run(1);
    auto [wb7, E7, O7, V7] = run(7);

    CHECK(wb1.x == wb7.x && wb1.s == wb7.s && wb1.t == wb7.t && wb1.logp == wb7.logp, "determinism: walker state diverged across worker counts");
    CHECK(E1 == E7, "determinism: E_pool diverged across worker counts");
    CHECK(O1 == O7, "determinism: O_pool diverged across worker counts");
    CHECK(V1 == V7, "determinism: valid_pool diverged across worker counts");
}

int main() {
    Ansatz a({8}, {8}, {8}, Activation::Gelu);
    seed_ansatz(a, 1002);

    test_agreement(a);
    test_mask_correctness(a);
    test_srop_mask();
    test_batch_error_sanity();
    test_determinism_through_record(a);

    if (g_failures == 0) { std::cout << "ALL TESTS PASSED\n"; return 0; }
    std::cout << g_failures << " CHECK(S) FAILED\n";
    return 1;
}