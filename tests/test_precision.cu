// Phase 6.3: precision experiments -- measurements that set the fp32 settings.
//
// 1. The validity guard's own statistic on realistic (thermalised) walkers:
//      g = |psi_jet - psi_double| / max(1, |psi_double|)
//    psi_mismatch_tol is set to 10x its p99.9 (floor 1e-6) under fp32_forward;
//    this prints the distribution and checks that the chosen tolerance leaves
//    the guard firing on essentially no ordinary sample (< 1e-3 of them).
// 2. Determinism within the build: the same walkers twice -> identical bits.
// 3. fp32_opool only -- what storing O in float costs, isolated from everything
//    else. With B <= opool_stage_rows the whole record is one chunk, so after
//    assemble_O_batch the FP64 staging buffer still holds backprop's rows: the
//    exact FP64 O the pool would have held (row / S, the division o_finalize
//    performs). Against it:
//      a. round trip: every float entry within 1e-7 relative (2^-24 = 6e-8);
//      b. the SR step: host solves on the FP64 and on the float pool, both at
//         CG tol 1e-10, at the lambda floor -- storage precision alone.
//
// Wavefunction: the Li6 checkpoint named by VMC_PRECISION_CKPT if set (the
// production-relevant case; the file is not in the repository), otherwise a
// seeded random network.
#include "../lib/gpu/arena.h"
#include "../lib/gpu/eval.h"
#include "../lib/gpu/local_e.h"
#include "../lib/gpu/gpu_sampler.h"
#include "../lib/gpu/backprop.h"
#include "../lib/descent.h"
#include "../lib/sr.h"
#include "../lib/cg.h"
#include "../lib/checkpoint.h"
#include "../lib/physics.h"
#include "../lib/walkers.h"
#include "../lib/pool.h"
#include "../tests/test_common.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
    g_failures++; } } while (0)

template <typename T>
static std::vector<T> dl(const DeviceArray<T>& a, std::size_t n) { std::vector<T> h(n); a.down(h.data(), n); return h; }

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (std::size_t)(q * (double)(v.size() - 1) + 0.5))];
}

// Hidden widths of one network from a checkpoint header line "name L n0 n1 ... nL".
static std::vector<int> ckpt_hidden(const char* path, const char* name) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) throw std::runtime_error(std::string("cannot open checkpoint ") + path);
    char line[4096];
    std::vector<int> hidden;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, name, std::strlen(name)) != 0 || line[std::strlen(name)] != ' ') continue;
        std::vector<int> v; char* p = line + std::strlen(name);
        for (char* tok = std::strtok(p, " \n"); tok; tok = std::strtok(nullptr, " \n")) v.push_back(std::atoi(tok));
        for (std::size_t i = 2; i + 1 < v.size(); i++) hidden.push_back(v[i]);   // v[0] = L, v[1] = input, v[L+1] = output
        break;
    }
    std::fclose(f);
    return hidden;
}

// Section 3. A template on the pool type so the FP64 build never instantiates it.
template <typename PT>
static void storage_checks(DeviceState& ds, cublasHandle_t h, const Ansatz& a, Workspace& ws, int B) {
    static_assert(opool_stage_rows >= 512, "the storage check needs one record in a single staging chunk");
    ds.grow_phase5(a, false); ds.grow_phase52(false);
    eval_local_E_device(ds, h, a, ws, B, 0, /*stash_for_O=*/true);
    assemble_O_batch(ds, h, 0, B);
    const std::size_t P = ds.P, Ns = (std::size_t)B;
    auto stage = dl(ds.O_stage, Ns * P);
    std::vector<PT> pool_f(Ns * P); ds.O_pool.down(pool_f.data(), pool_f.size());
    auto Sv = dl(ds.S, Ns); auto E = dl(ds.E_loc, Ns); auto vv = dl(ds.valid_loc, Ns);
    std::vector<double> O64(Ns * P), O32(Ns * P);
    double worst_rt = 0.0; std::size_t n_rt = 0;
    for (std::size_t i = 0; i < Ns; i++) for (std::size_t k = 0; k < P; k++) {
        const std::size_t q = i * P + k;
        O32[q] = (double)pool_f[q];
        if (!vv[i]) { O64[q] = 0.0; continue; }
        // alpha column: computed in o_finalize, not staged; take the stored value.
        O64[q] = (k + 1 < P) ? stage[q] / Sv[i] : O32[q];
        if (k + 1 < P && O64[q] != 0.0) { worst_rt = std::max(worst_rt, std::fabs(O32[q] - O64[q]) / std::fabs(O64[q])); n_rt++; }
    }
    std::printf("  fp32_opool round trip: %zu entries, worst relative %.2e (2^-24 = 5.96e-08)\n", n_rt, worst_rt);
    CHECK(n_rt > 0 && worst_rt <= 1e-7, "float O_pool entries differ from the FP64 rows beyond one rounding");

    // SR step at the lambda floor on each pool, host solver, tight CG.
    std::vector<uint8_t> vp(vv.begin(), vv.end());
    long long nv = 0; for (auto x : vp) nv += x;
    std::vector<double> Ed(E.begin(), E.end());
    BatchStats bs; bs.n_valid = nv; bs.n_invalid = (long long)Ns - nv; bs.E_sum = bs.E2_sum = bs.l2_sum = bs.r2_sum = 1.0;
    ThreadPool tp(n_thread);          // SROp / compute_obs split work into n_thread chunks
    auto solve = [&](const std::vector<double>& O, std::vector<double>& delta) {
        DescentResult r{};
        std::vector<double> Oexp, grad;
        compute_obs(bs, r, P, Ed, O, vp, Ns, &tp, Oexp, grad);
        std::vector<double> d_rms(P, 1.0);
        SROp op; op.init(O, Oexp, Ns, P, sr_lambda_min, sr_eps, &tp, d_rms.data(), vp.data(), nv);
        std::vector<double> Minv(P);
        for (std::size_t j = 0; j < P; j++) Minv[j] = 1.0 / (op.S_diag[j] * (1.0 + sr_lambda_min) + sr_eps + (sr_rms_damp ? sr_rms_eps : 0.0));
        delta.assign(P, 0.0);
        auto mv = [&](const std::vector<double>& x, std::vector<double>& y) { op.apply(x, y); };
        return cg_solve(mv, grad, delta, Minv, 1e-10, 20000);
    };
    std::vector<double> d64, d32;
    const CGResult c64 = solve(O64, d64), c32 = solve(O32, d32);
    double num = 0.0, den = 0.0;
    for (std::size_t j = 0; j < P; j++) { num += (d32[j] - d64[j]) * (d32[j] - d64[j]); den += d64[j] * d64[j]; }
    const double rel = std::sqrt(num / den);
    std::printf("  fp32_opool SR step (lambda = %.1f, CG tol 1e-10, iters %d / %d): ||delta_f32 - delta_f64|| / ||delta_f64|| = %.2e  (CG tol in production: %.0e)\n",
                sr_lambda_min, c64.iters, c32.iters, rel, sr_cg_tol);
    CHECK(c64.converged && c32.converged, "tight CG did not converge");
    CHECK(rel <= 1e-3, "storing O in float moves the SR step beyond the production CG tolerance");
}

int main() {
    gpu_select_device(true);
    const char* ckpt = std::getenv("VMC_PRECISION_CKPT");
    const bool use_ckpt = ckpt && *ckpt;
    // Seeded mode: hidden widths from VMC_PRECISION_HIDDEN ("64,64,64"), default {64}.
    std::vector<int> seeded{64};
    if (const char* hw = std::getenv("VMC_PRECISION_HIDDEN")) {
        seeded.clear();
        for (const char* q = hw; *q; ) { seeded.push_back(std::atoi(q)); while (*q && *q != ',') q++; if (*q) q++; }
    }
    Ansatz a(use_ckpt ? ckpt_hidden(ckpt, "h_net")   : seeded,
             use_ckpt ? ckpt_hidden(ckpt, "rho_net") : seeded,
             use_ckpt ? ckpt_hidden(ckpt, "orb_net") : seeded, Activation::Gelu);
    if (use_ckpt) { load_checkpoint(ckpt, a); std::printf("wavefunction: checkpoint %s (P = %zu)\n", ckpt, a.n_params()); }
    else { seed_ansatz(a, 2024); std::printf("wavefunction: seeded random network, %zu hidden layers of %d (P = %zu)\n", seeded.size(), seeded[0], a.n_params()); }
    std::printf("build: fp32_forward=%d fp32_opool=%d psi_mismatch_tol=%.1e\n", (int)fp32_forward, (int)fp32_opool, psi_mismatch_tol);

    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    try {
        const int B = 512, rounds = 16;
        ThreadPool pool(4);
        std::vector<Workspace> wss(4);
        WalkerBatch wb; wb.init(B);
        init_batch(wb, a, &pool, wss);
        DeviceState ds(a, false);
        ds.grow_phase3(a, false); ds.grow_phase33(false); ds.grow_phase4(false); ds.grow_phase42(false); ds.grow_phase43(false);
        PinnedArray st;
        ds.upload_params(a, st);
        upload_and_reset(ds, wb, st);
        eval_logp_batch(ds, h, B);
        therm_batch_device(ds, h, B, 1.0, 100);

        std::vector<double> g, rel;
        std::size_t n_eval = 0, n_invalid = 0, n_flag_tol = 0;
        for (int r = 0; r < rounds; r++) {
            therm_batch_device(ds, h, B, 1.0, 5);
            eval_local_E_device(ds, h, a, wss[0], B);
            auto v = dl(ds.jet_psi, (std::size_t)B);          // value block of the jet psi
            auto pd = dl(ds.psi_dbl, (std::size_t)B);
            auto S = dl(ds.S, (std::size_t)B);
            auto valid = dl(ds.valid_loc, (std::size_t)B);
            for (int w = 0; w < B; w++) {
                n_eval++;
                if (!valid[w]) n_invalid++;
                if (!std::isfinite(S[w]) || std::fabs(S[w]) < 1e-290 || !std::isfinite(v[w]) || std::fabs(v[w]) < 1e-290) continue;
                const double d = std::fabs(v[w] - pd[w]);
                g.push_back(d / std::max(1.0, std::fabs(pd[w])));
                rel.push_back(d / std::fabs(pd[w]));
                if (d > psi_mismatch_tol * std::max(1.0, std::fabs(pd[w]))) n_flag_tol++;
            }
        }
        const double p999 = pct(g, 0.999);
        std::printf("  guard statistic g = |psi_jet - psi_double| / max(1,|psi|), %zu samples:\n", g.size());
        std::printf("    p50 %.2e  p99 %.2e  p99.9 %.2e  max %.2e    (plain relative: p50 %.2e  p99.9 %.2e  max %.2e)\n",
                    pct(g, 0.5), pct(g, 0.99), p999, pct(g, 1.0), pct(rel, 0.5), pct(rel, 0.999), pct(rel, 1.0));
        std::printf("    10 x p99.9 = %.2e  (floor 1e-6 -> suggested tol %.1e);  current tol %.1e flags %zu of %zu;  n_invalid %zu of %zu evaluated\n",
                    10.0 * p999, std::max(1e-6, 10.0 * p999), psi_mismatch_tol, n_flag_tol, g.size(), n_invalid, n_eval);
        CHECK(!g.empty(), "no finite samples");
        CHECK((double)n_flag_tol <= 1e-3 * (double)g.size(), "psi_mismatch_tol makes the guard fire on ordinary samples");

        // Determinism: the same walker state evaluated twice.
        eval_local_E_device(ds, h, a, wss[0], B);
        auto E1 = dl(ds.E_loc, (std::size_t)B);
        eval_local_E_device(ds, h, a, wss[0], B);
        auto E2 = dl(ds.E_loc, (std::size_t)B);
        const bool same = std::memcmp(E1.data(), E2.data(), E1.size() * sizeof(real)) == 0;
        std::printf("  determinism: local_E twice on the same walkers: %s\n", same ? "bit-identical" : "DIFFERS");
        CHECK(same, "local_E is not bit-reproducible within this build");

        if constexpr (fp32_opool) storage_checks<opool_t>(ds, h, a, wss[0], B);
        else std::printf("  (fp32_opool off: storage-precision checks not applicable)\n");
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures) { std::cerr << g_failures << " check(s) FAILED\n"; return 1; }
    std::printf("test_precision: all checks passed\n");
    return 0;
}
