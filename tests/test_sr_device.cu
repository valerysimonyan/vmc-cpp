// Phase 5.2: SR on device -- masked statistics, gradient, S*v, CG, full SR step.
#include "../lib/gpu/arena.h"
#include "../tests/test_tolerances.h"
#include "../lib/gpu/eval.h"
#include "../lib/gpu/gpu_sampler.h"
#include "../lib/gpu/local_e.h"
#include "../lib/gpu/backprop.h"
#include "../lib/gpu/exchange_kernels.h"
#include "../lib/gpu/sr_device.h"
#include "../lib/descent.h"
#include "../lib/sr.h"
#include "../lib/cg.h"
#include "../lib/util.h"
#include "../lib/walkers.h"
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
static double worst_rel(const std::vector<double>& got, const std::vector<double>& want) {
    double w = 0;
    for (std::size_t i = 0; i < want.size(); i++)
        w = std::max(w, std::fabs(got[i] - want[i]) / std::max(1e-300, std::fabs(want[i])));
    return w;
}
static std::vector<double> down(const DeviceArray<double>& d, std::size_t n) { std::vector<double> h(n); d.down(h.data(), n); return h; }
static void up(DeviceArray<double>& d, const std::vector<double>& h) { if (d.n < h.size()) d.alloc(h.size()); d.up(h.data(), h.size()); }

// Host-built masked pool: ~12% invalid rows holding FINITE garbage (not zeros),
// so a missing mask shows up as a wrong answer rather than hiding behind zeros.
struct Pool { std::size_t Ns, P; std::vector<double> O, E; std::vector<unsigned char> v; long long nv; };
static Pool make_pool(std::size_t Ns, std::size_t P, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Pool p{Ns, P, std::vector<double>(Ns*P), std::vector<double>(Ns), std::vector<unsigned char>(Ns), 0};
    std::vector<double> scale(P); for (auto& s : scale) s = std::exp(2.0 * g(rng));        // parameters on very different scales
    for (std::size_t i = 0; i < Ns; i++) {
        p.v[i] = (u(rng) > 0.12) ? 1 : 0;
        p.nv += p.v[i];
        p.E[i] = -30.0 + 8.0 * g(rng) + ((u(rng) < 0.01) ? 400.0 * g(rng) : 0.0);          // a heavy tail for the clip to bite
        for (std::size_t j = 0; j < P; j++) p.O[i*P + j] = p.v[i] ? (0.3 + g(rng)) * scale[j] : 7.0 + g(rng);
    }
    opool_round(p.O);        // the CPU oracle sees exactly what the device pool stores (6.3)
    return p;
}

// The gradient is a covariance, 2*(<E O> - <E><O>): a small difference of two
// large terms, and each term a sum over samples of mixed sign. Its rounding error
// scales with those summands, not with the gradient itself -- measured 1.8e-10
// relative on the synthetic pool against 2e-15 of this scale. Same reasoning as
// the V_nuc and O-assembly oracles.
static std::vector<double> grad_scale(const std::vector<double>& O, const std::vector<double>& E, const std::vector<unsigned char>& v,
                                      std::size_t Ns, std::size_t P, long long nv, const ClipStats& cs, const std::vector<double>& Oexp) {
    std::vector<double> acc(P, 0.0);
    for (std::size_t i = 0; i < Ns; i++) {
        if (!v[i]) continue;
        const double ec = std::fabs(std::min(std::max(E[i], cs.clip_lo), cs.clip_hi));
        for (std::size_t k = 0; k < P; k++) acc[k] += ec * std::fabs(O[i*P + k]);
    }
    for (std::size_t k = 0; k < P; k++) acc[k] = std::max(1e-300, 2.0 * (acc[k] / (double)nv + std::fabs(cs.E_clip_mean * Oexp[k])));
    return acc;
}

// --- D1: O_exp, S_diag, gradient -------------------------------------------------
static void test_statistics(cublasHandle_t h, ThreadPool& pool) {
    Pool pl = make_pool(4000, 3000, 11);
    DeviceArray<opool_t> O; DeviceArray<double> E, m, Oexp, Sd, Ec, gr; DeviceArray<unsigned char> V;
    opool_up(O, pl.O); up(E, pl.E); V.alloc(pl.Ns); V.up(pl.v.data(), pl.Ns);
    m.alloc(pl.Ns); Oexp.alloc(pl.P); Sd.alloc(pl.P); Ec.alloc(pl.Ns); gr.alloc(pl.P);

    build_mask(V.d, m.d, pl.Ns);
    O_exp_device(h, O.d, m.d, pl.Ns, pl.P, pl.nv, Oexp.d);
    S_diag_device(O.d, V.d, Oexp.d, pl.Ns, pl.P, pl.nv, Sd.d);
    const ClipStats cs = clip_stats_host(pl.E, pl.v, pl.Ns, pl.nv);
    grad_device(h, O.d, E.d, V.d, Oexp.d, pl.Ns, pl.P, pl.nv, cs, Ec.d, gr.d);

    // CPU references: the functions descent() calls today
    std::vector<double> Oexp_c, grad_c;
    BatchStats bs; bs.n_valid = pl.nv; bs.n_invalid = (long long)pl.Ns - pl.nv; bs.E_sum = bs.E2_sum = bs.l2_sum = bs.r2_sum = 1.0;
    DescentResult r{};
    compute_obs(bs, r, pl.P, pl.E, pl.O, pl.v, pl.Ns, &pool, Oexp_c, grad_c);
    SROp op; std::vector<double> d_rms(pl.P, 1.0);
    op.init(pl.O, Oexp_c, pl.Ns, pl.P, 0.5, sr_eps, &pool, d_rms.data(), pl.v.data(), pl.nv);

    const std::vector<double> gd = down(gr, pl.P);
    const std::vector<double> gscale = grad_scale(pl.O, pl.E, pl.v, pl.Ns, pl.P, pl.nv, cs, Oexp_c);
    double wG_scaled = 0; for (std::size_t k = 0; k < pl.P; k++) wG_scaled = std::max(wG_scaled, std::fabs(gd[k] - grad_c[k]) / gscale[k]);
    const double wO = worst_rel(down(Oexp, pl.P), Oexp_c), wS = worst_rel(down(Sd, pl.P), op.S_diag), wG = worst_rel(gd, grad_c);
    std::printf("  statistics (Ns=%zu, P=%zu, %lld valid): O_exp rel %.2e   S_diag rel %.2e   grad rel %.2e (%.2e of summand scale)\n",
                pl.Ns, pl.P, pl.nv, wO, wS, wG, wG_scaled);
    CHECK(wO <= 1e-12, "device O_exp disagrees with masked_O_exp");
    CHECK(wS <= 1e-12, "device S_diag disagrees with SROp::init");
    CHECK(wG_scaled <= 1e-12, "device gradient disagrees with compute_obs");

    // RMS update against descent()'s loop
    std::vector<double> g = grad_c, v_c(pl.P, 0.25), dr_c(pl.P);
    double mean_c = 0.0;
    for (std::size_t k = 0; k < pl.P; k++) {
        v_c[k] = sr_rms_beta * v_c[k] + (1.0 - sr_rms_beta) * g[k] * g[k];
        dr_c[k] = std::sqrt(v_c[k]) + 1e-8;
        mean_c += sr_rms_eps * dr_c[k];
    }
    mean_c /= pl.P;
    DeviceArray<double> vr, dr; up(vr, std::vector<double>(pl.P, 0.25)); dr.alloc(pl.P);
    const double mean_d = rms_update_device(h, gr.d, vr.d, dr.d, pl.P);
    const std::vector<double> vr_h = down(vr, pl.P), dr_h = down(dr, pl.P);
    const double wv = worst_rel(vr_h, v_c), wd = worst_rel(dr_h, dr_c), wm = std::fabs(mean_d - mean_c) / mean_c;
    std::printf("  rms update: v_rms rel %.2e   d_rms rel %.2e   rms_damp_mean rel %.2e\n", wv, wd, wm);
    CHECK(wv <= 1e-12 && wd <= 1e-12 && wm <= 1e-12, "device RMS update disagrees with descent()'s loop");
}

// --- D2: S * v ---------------------------------------------------------------------
static void test_apply(cublasHandle_t h, ThreadPool& pool) {
    Pool pl = make_pool(3000, 2500, 12);
    std::mt19937_64 rng(5); std::normal_distribution<double> g(0, 1); std::uniform_real_distribution<double> u(0.1, 2.0);
    std::vector<double> v(pl.P), d_rms(pl.P); for (auto& x : v) x = g(rng); for (auto& x : d_rms) x = u(rng);

    std::vector<double> Oexp_c;
    masked_O_exp(pl.O, pl.v, pl.Ns, pl.P, &pool, Oexp_c);
    SROp op; op.init(pl.O, Oexp_c, pl.Ns, pl.P, 0.7, sr_eps, &pool, d_rms.data(), pl.v.data(), pl.nv);
    std::vector<double> out_raw, out_damp;
    op.apply(v, out_raw, true); op.apply(v, out_damp, false);

    DeviceArray<opool_t> O; DeviceArray<double> m, Oexp, Sd, dr, t, dv, dout; DeviceArray<unsigned char> V;
    opool_up(O, pl.O); V.alloc(pl.Ns); V.up(pl.v.data(), pl.Ns); m.alloc(pl.Ns); t.alloc(pl.Ns);
    up(Oexp, Oexp_c); up(Sd, op.S_diag); up(dr, d_rms); up(dv, v); dout.alloc(pl.P);
    build_mask(V.d, m.d, pl.Ns);
    SROpDevice od; od.h = h; od.O_pool = O.d; od.O_exp = Oexp.d; od.m = m.d; od.S_diag = Sd.d; od.d_rms = dr.d; od.t = t.d;
    od.Ns = pl.Ns; od.P = pl.P; od.n_valid = pl.nv; od.lambda_diag = 0.7; od.eps_abs = sr_eps;
    od.apply(dv.d, dout.d, true);  const double wr = worst_rel(down(dout, pl.P), out_raw);
    od.apply(dv.d, dout.d, false); const double wd = worst_rel(down(dout, pl.P), out_damp);
    std::printf("  S*v apply: raw rel %.2e   damped rel %.2e\n", wr, wd);
    // fp32_opool: same float-rounded values on both sides, but the device sums
    // with the custom mixed-precision kernels (32 slabs, then a tree) instead of
    // cuBLAS -- a different order, magnified on entries with sign cancellation
    // by this plain-relative metric. Measured 1.1e-11 raw / 6.6e-11 damped.
    CHECK(wr <= tol::fo(1e-11, 5e-10), "device raw apply disagrees with SROp::apply");
    CHECK(wd <= tol::fo(1e-11, 5e-10), "device damped apply disagrees with SROp::apply");
}

// --- D3a: CG on a synthetic SPD system ---------------------------------------------------
static void test_cg_synthetic(cublasHandle_t h) {
    const std::size_t n = 400;
    std::mt19937_64 rng(9); std::normal_distribution<double> g(0, 1);
    std::vector<double> A(n*n), S(n*n, 0.0), b(n), Minv(n);
    for (auto& x : A) x = g(rng);
    for (std::size_t i = 0; i < n; i++) for (std::size_t j = 0; j < n; j++) {
        double acc = 0; for (std::size_t k = 0; k < n; k++) acc += A[i*n+k] * A[j*n+k];
        S[i*n+j] = acc / n + (i == j ? 1e-2 * (1 + i % 7) : 0.0);
    }
    for (auto& x : b) x = g(rng);
    for (std::size_t i = 0; i < n; i++) Minv[i] = 1.0 / S[i*n+i];

    auto host_mv = [&](const std::vector<double>& v, std::vector<double>& out) {
        out.assign(n, 0.0);
        for (std::size_t i = 0; i < n; i++) { double acc = 0; for (std::size_t j = 0; j < n; j++) acc += S[i*n+j] * v[j]; out[i] = acc; }
    };
    std::vector<double> x_c(n, 0.0);
    const CGResult rc = cg_solve(host_mv, b, x_c, Minv, 1e-10, 1000);

    DeviceArray<double> dS, db, dx, dM, r, z, p, Ap;
    up(dS, S); up(db, b); up(dx, std::vector<double>(n, 0.0)); up(dM, Minv);
    for (auto* x : {&r, &z, &p, &Ap}) x->alloc(n);
    const double one = 1.0, zero = 0.0;
    DeviceMatVec dev_mv = [&](const double* v, double* out) {   // S symmetric: row/column-major orientation is moot
        cublasDgemv(h, CUBLAS_OP_N, (int)n, (int)n, &one, dS.d, (int)n, v, 1, &zero, out, 1);
    };
    long long ndl = 0;
    const CGResult rd = cg_solve_device(h, dev_mv, db.d, dx.d, dM.d, n, 1e-10, 1000, r.d, z.d, p.d, Ap.d, &ndl);
    const std::vector<double> x_d = down(dx, n);
    double num = 0, den = 0; for (std::size_t i = 0; i < n; i++) { num += (x_d[i]-x_c[i])*(x_d[i]-x_c[i]); den += x_c[i]*x_c[i]; }
    const double rel_sol = std::sqrt(num / den);
    std::printf("  CG synthetic SPD (n=%zu, tol 1e-10): iters host %d device %d;  solution rel %.2e;  %lld scalar downloads\n",
                n, rc.iters, rd.iters, rel_sol, ndl);
    CHECK(rc.converged && rd.converged, "synthetic CG did not converge");
    CHECK(rc.iters == rd.iters, "device CG iteration count differs from cg_solve");
    CHECK(rel_sol <= 1e-8, "device CG solution disagrees with cg_solve");
    CHECK(ndl <= 3LL * rd.iters + 4, "scalar downloads exceed the 3-per-iteration contract");
}

// --- D3b + D4: a REAL O_pool, full SR step, determinism --------------------------------
static void test_sr_real(cublasHandle_t h, ThreadPool& pool, std::vector<Workspace>& wss) {
    const int B = 1024, records = 2;
    Ansatz a({64},{64},{64}, Activation::Gelu);
    seed_ansatz(a, 2024);
    const std::size_t P = a.n_params(), Ns = (std::size_t)B * records;

    DeviceState ds(a, false);
    ds.grow_phase3(a, false); ds.grow_phase33(false); ds.grow_phase4(false);
    ds.grow_phase42(false);   ds.grow_phase43(false); ds.grow_phase5(a, false); ds.grow_phase52(false);
    PinnedArray st; ds.upload_params(a, st);
    WalkerBatch wb; wb.init(B); init_batch(wb, a, &pool, wss);
    upload_and_reset(ds, wb, st);
    eval_logp_batch(ds, h, B);
    therm_batch_device(ds, h, B, 0.6, 30);
    for (int rr = 0; rr < records; rr++) {
        therm_batch_device(ds, h, B, 0.6, sweeps_between_records);
        eval_local_E_device(ds, h, a, wss[0], B, 0, /*stash_for_O=*/true);
        pool_write_row(ds.E_loc.d, ds.valid_loc.d, ds.E_pool.d, ds.valid_pool.d, rr, B);
        assemble_O_batch(ds, h, rr, B);
    }
    // Host copies for the CPU reference -- the test's download, not production's.
    std::vector<double> O_h(Ns * P), E_h(Ns); std::vector<unsigned char> V_h(Ns);
    O_h = opool_down(ds.O_pool, O_h.size()); ds.E_pool.down(E_h.data(), Ns); ds.valid_pool.down(V_h.data(), Ns);
    long long nv = 0; for (auto x : V_h) nv += x;

    // ---- CPU: compute_obs, RMS, SR_step ----
    BatchStats bs; bs.n_valid = nv; bs.n_invalid = (long long)Ns - nv; bs.E_sum = bs.E2_sum = bs.l2_sum = bs.r2_sum = 1.0;
    DescentResult r{}; std::vector<double> Oexp_c, grad_c;
    compute_obs(bs, r, P, E_h, O_h, V_h, Ns, &pool, Oexp_c, grad_c);
    std::vector<double> v_rms(P, 0.0), d_rms(P);
    for (std::size_t k = 0; k < P; k++) { v_rms[k] = sr_rms_beta * v_rms[k] + (1.0 - sr_rms_beta) * grad_c[k] * grad_c[k]; d_rms[k] = std::sqrt(v_rms[k]) + 1e-8; }
    const int iter = 1500;
    Ansatz a_cpu = a; SROp sr_op; std::vector<double> delta_c(P, 0.0), Minv(P), S_delta;
    const SRStepLog lc = SR_step(grad_c, O_h, Oexp_c, a_cpu, sr_op, delta_c, iter, P, &pool, d_rms.data(), Ns, V_h.data(), nv, Minv, S_delta);

    // ---- device ----
    auto device_step = [&](Ansatz& a_dev, std::vector<double>& delta_d, long long& ndl) {
        ds.v_rms_d.zero(); ds.delta_d.zero();
        build_mask(ds.valid_pool.d, ds.mask_d.d, Ns);
        O_exp_device(h, ds.O_pool.d, ds.mask_d.d, Ns, P, nv, ds.O_exp_d.d);
        const ClipStats cs = clip_stats_host(E_h, V_h, Ns, nv);
        grad_device(h, ds.O_pool.d, ds.E_pool.d, ds.valid_pool.d, ds.O_exp_d.d, Ns, P, nv, cs, ds.E_clip_d.d, ds.grad_d.d);
        rms_update_device(h, ds.grad_d.d, ds.v_rms_d.d, ds.d_rms_d.d, P);
        return SR_step_device(ds, h, a_dev, iter, Ns, nv, delta_d, &ndl);
    };
    Ansatz a_dev1 = a, a_dev2 = a; std::vector<double> delta_d1, delta_d2; long long ndl1 = 0, ndl2 = 0;
    const SRStepLog ld = device_step(a_dev1, delta_d1, ndl1);
    const std::vector<double> gd_real = down(ds.grad_d, P);
    const std::vector<double> gsc_real = grad_scale(O_h, E_h, V_h, Ns, P, nv, clip_stats_host(E_h, V_h, Ns, nv), Oexp_c);
    double wg = 0; for (std::size_t k = 0; k < P; k++) wg = std::max(wg, std::fabs(gd_real[k] - grad_c[k]) / gsc_real[k]);
    const SRStepLog ld2 = device_step(a_dev2, delta_d2, ndl2);

    double num = 0, den = 0; for (std::size_t j = 0; j < P; j++) { num += (delta_d1[j]-delta_c[j])*(delta_d1[j]-delta_c[j]); den += delta_c[j]*delta_c[j]; }
    const double rel_delta = std::sqrt(num / den);
    std::size_t nbit = 0; for (std::size_t j = 0; j < P; j++) if (delta_d1[j] != delta_d2[j]) nbit++;
    std::printf("  real O_pool (B=%d x %d records, P=%zu, %lld valid), iter %d:\n", B, records, P, nv, iter);
    std::printf("    grad %.2e of summand scale;  CG iters host %d device %d;  q host %.6e device %.6e;  ||delta|| host %.6e device %.6e\n",
                wg, lc.cg_iters, ld.cg_iters, lc.sq_metric_norm, ld.sq_metric_norm, lc.delta_norm, ld.delta_norm);
    std::printf("    delta rel (2-norm) %.2e;  scalar downloads %lld;  determinism: %zu of %zu delta entries differ\n",
                rel_delta, ndl1, nbit, P);
    CHECK(wg <= 1e-12, "device gradient disagrees with compute_obs on a real pool");
    CHECK(nbit == 0 && ld.cg_iters == ld2.cg_iters, "two identical device SR solves are not bit-identical");

    // Why the prompt's delta rel 1e-6 cannot hold at production settings, and
    // what is checked instead. SR_step stops CG at sr_cg_tol = 1e-3 relative
    // residual; host and device reach it on different iterations (77 vs 75 here)
    // because rounding in S*v (2e-12, D2) is amplified along the Krylov sequence
    // of an ill-conditioned S. Two solutions each good to residual 1e-3 need not
    // agree to better than ~1e-3, and measured 4.4e-4. So, on this same real
    // operator, with the device's own S_diag / M_inv / d_rms mirrored on the host:
    //   (i)  at production tolerance, the DEVICE solution must also satisfy the
    //        tolerance under the HOST operator -- it is a valid CG answer to the
    //        host's system, not merely close to the host's answer;
    //   (ii) at tight tolerance, both CGs must reach the same solution to 1e-6.
    const double lambda_t = std::max(sr_lambda0 * std::pow(sr_rho, iter), sr_lambda_min);
    SROp hop; hop.init(O_h, Oexp_c, Ns, P, lambda_t, sr_eps, &pool, d_rms.data(), V_h.data(), nv);
    std::vector<double> Minv_h(P);
    for (std::size_t j = 0; j < P; j++) {
        double dd = sr_eps; if constexpr (sr_rms_damp) dd += sr_rms_eps * d_rms[j];
        Minv_h[j] = 1.0 / (hop.S_diag[j] * (1.0 + lambda_t) + dd);
    }
    auto host_mv = [&](const std::vector<double>& vv, std::vector<double>& out) { hop.apply(vv, out); };

    SROpDevice dop; dop.h = h; dop.O_pool = ds.O_pool.d; dop.O_exp = ds.O_exp_d.d; dop.m = ds.mask_d.d; dop.S_diag = ds.S_diag_d.d;
    dop.d_rms = ds.d_rms_d.d; dop.t = ds.t_ns_d.d; dop.Ns = Ns; dop.P = P; dop.n_valid = nv; dop.lambda_diag = lambda_t; dop.eps_abs = sr_eps;
    DeviceMatVec dev_mv = [&](const double* vv, double* out) { dop.apply(vv, out, false); };

    auto rel_diff = [&](const std::vector<double>& x, const std::vector<double>& y) {
        double nn = 0, dd = 0; for (std::size_t j = 0; j < P; j++) { nn += (x[j]-y[j])*(x[j]-y[j]); dd += y[j]*y[j]; } return std::sqrt(nn / dd);
    };
    auto run_both = [&](double tol, int maxit, std::vector<double>& xh, std::vector<double>& xd, CGResult& rh, CGResult& rd) {
        xh.assign(P, 0.0); rh = cg_solve(host_mv, grad_c, xh, Minv_h, tol, maxit);
        ds.delta_d.zero();
        rd = cg_solve_device(h, dev_mv, ds.grad_d.d, ds.delta_d.d, ds.M_inv_d.d, P, tol, maxit, ds.cg_r.d, ds.cg_z.d, ds.cg_p.d, ds.cg_Ap.d);
        xd = down(ds.delta_d, P);
    };
    std::vector<double> xh, xd; CGResult rh, rd;

    run_both(sr_cg_tol, sr_cg_maxit, xh, xd, rh, rd);
    std::vector<double> Sx; host_mv(xd, Sx);
    double rn = 0, bn = 0; for (std::size_t j = 0; j < P; j++) { rn += (grad_c[j]-Sx[j])*(grad_c[j]-Sx[j]); bn += grad_c[j]*grad_c[j]; }
    const double dev_resid_on_host = std::sqrt(rn / bn);
    std::printf("    CG at production tol %.0e: iters host %d device %d, solutions differ %.2e;  device solution's residual under the HOST operator %.2e\n",
                sr_cg_tol, rh.iters, rd.iters, rel_diff(xd, xh), dev_resid_on_host);
    CHECK(dev_resid_on_host < sr_cg_tol, "device CG solution does not satisfy the host operator's tolerance");

    run_both(1e-10, 5000, xh, xd, rh, rd);
    const double tight = rel_diff(xd, xh);
    std::printf("    CG at tol 1e-10: iters host %d device %d (converged %d/%d), solutions differ %.2e\n",
                rh.iters, rd.iters, (int)rh.converged, (int)rd.converged, tight);
    CHECK(rh.converged && rd.converged, "tight-tolerance CG did not converge");
    CHECK(tight <= 1e-6, "host and device CG do not reach the same solution at tight tolerance");

    // The full SR_step sequence at production settings: bounded by the CG tolerance, not 1e-6.
    CHECK(rel_delta <= 10.0 * sr_cg_tol, "device SR delta differs from SR_step by more than the CG tolerance allows");
    CHECK(ndl1 <= 4LL * sr_cg_maxit + 16, "scalar downloads exceed the contract bound");
    CHECK(lc.norm_capped == ld.norm_capped, "norm-cap decision differs");
}

int main() {
    gpu_select_device(true);
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    ThreadPool pool(n_thread);
    std::vector<Workspace> wss(n_thread);
    std::printf("  sr_rms_damp = %s\n", sr_rms_damp ? "true" : "false");
    try {
        test_statistics(h, pool);
        test_apply(h, pool);
        test_cg_synthetic(h);
        test_sr_real(h, pool, wss);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
