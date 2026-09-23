// Phase 4.3: 2-body exchange on device, full local_E parity, hybrid-v2 determinism.
#include "../lib/gpu/arena.h"
#include "../tests/test_tolerances.h"
#include "../lib/gpu/eval.h"
#include "../lib/gpu/jet_eval.h"
#include "../lib/gpu/record_device.h"
#include "../lib/gpu/exchange_kernels.h"
#include "../lib/gpu/local_e.h"
#include "../lib/gpu/gpu_sampler.h"
#include "../lib/physics.h"
#include "../lib/walkers.h"
#include "../lib/pool.h"
#include "../tests/test_common.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#include <stdexcept>

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
    g_failures++; } } while (0)

// Random coordinates, and spin / isospin labels shuffled independently per walker
// (counts preserved), so every walker mixes same_s, same_t and fully-mixed pairs.
static void rand_cfg(int B, std::vector<double>& hx, std::vector<double>& hs, std::vector<double>& ht,
                     std::mt19937_64& rng, bool coincident_every_third = false) {
    std::uniform_real_distribution<double> d(-x_init_range, x_init_range);
    hx.assign((std::size_t)B*D, 0.0); hs.assign((std::size_t)B*N, 0.0); ht.assign((std::size_t)B*N, 0.0);
    for (auto& v : hx) v = d(rng);
    for (int w = 0; w < B; w++) {
        std::vector<double> sw(N), tw(N);
        for (int i = 0; i < N; i++) { sw[i] = (i < N_u) ? 1.0 : -1.0; tw[i] = (i < N_p) ? 1.0 : -1.0; }
        std::shuffle(sw.begin(), sw.end(), rng);
        std::shuffle(tw.begin(), tw.end(), rng);
        for (int i = 0; i < N; i++) { hs[(std::size_t)w*N+i] = sw[i]; ht[(std::size_t)w*N+i] = tw[i]; }
        if (coincident_every_third && w % 3 == 1)
            for (int q = 0; q < dim; q++) hx[(std::size_t)w*D + 1*dim + q] = hx[(std::size_t)w*D + q];
    }
}

struct Dev {
    DeviceState ds;
    explicit Dev(const Ansatz& a) : ds(a, false) {
        ds.grow_phase3(a, false); ds.grow_phase33(false); ds.grow_phase4(false);
        ds.grow_phase42(false);   ds.grow_phase43(false);
    }
    void push(const std::vector<double>& hx, const std::vector<double>& hs, const std::vector<double>& ht, const Ansatz& a) {
        PinnedArray st; ds.upload_params(a, st);
        std::vector<real> tx(hx.begin(),hx.end()), ts(hs.begin(),hs.end()), tt(ht.begin(),ht.end());
        ds.x.up(tx.data(),tx.size()); ds.s.up(ts.data(),ts.size()); ds.t.up(tt.data(),tt.size());
    }
};

// local_E's exchange loop, reproduced with use_rank2 overridable. Returns V_nuc
// and, through the out-params, how many ratios it evaluated, whether the natural
// rank-2 gate passed, and `vmag` -- the sum of the MAGNITUDES of every product
// that enters V_nuc. Everything else is swap_ratio, called as local_E calls it.
//
// Why vmag: V_nuc is a signed sum over pairs that can nearly cancel, so its
// rounding error scales with its summands, not with its value. Neither an
// absolute bound (fails at |V_nuc| ~ 1e4 MeV) nor one relative to |V_nuc| (fails
// where it cancels -- measured 2.8e-11 that way) is right. Same trap, and same
// fix, as Phase 1.1's gsum-vs-gmag selfcheck.
static double cpu_vnuc(const std::vector<double>& x, const std::vector<double>& s, const std::vector<double>& t,
                       const Ansatz& a, Workspace& ws, bool force_lu, int& n_eval, bool& gate, double& vmag) {
    psi(x, s, t, a, ws, true);
    build_st_table(x, a, ws);
    double S0 = S_from_table(s, t, a, ws);
    gate = rank2_well_conditioned(ws);
    bool use_rank2 = force_lu ? false : gate;
    ws.s_swap.assign(s.begin(), s.end());
    ws.t_swap.assign(t.begin(), t.end());
    n_eval = 0;
    vmag = 0.0;
    double V_nuc = 0.0;
    auto v_gauss_reg = [](double r2, double R) {
        return std::exp(-r2 / (R*R)) / (std::pow(3.14159265358979323846, 1.5) * R*R*R);
    };
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            bool same_s = (s[i] == s[j]), same_t = (t[i] == t[j]);
            if (same_s && same_t) continue;
            double r2 = 0.0;
            for (int d = 0; d < dim; d++) { double diff = x[i*dim+d] - x[j*dim+d]; r2 += diff*diff; }
            double v01 = v_gauss_reg(r2, R01), v10 = v_gauss_reg(r2, R10);
            double R_s, R_t, R_st;
            if (same_s) {
                R_s = 1.0; R_t = swap_ratio(s, t, i, j, s[i], t[j], s[j], t[i], S0, use_rank2, a, ws); R_st = R_s*R_t; n_eval += 1;
            } else if (same_t) {
                R_t = 1.0; R_s = swap_ratio(s, t, i, j, s[j], t[i], s[i], t[j], S0, use_rank2, a, ws); R_st = R_s*R_t; n_eval += 1;
            } else {
                R_t  = swap_ratio(s, t, i, j, s[i], t[j], s[j], t[i], S0, use_rank2, a, ws);
                R_s  = swap_ratio(s, t, i, j, s[j], t[i], s[i], t[j], S0, use_rank2, a, ws);
                R_st = swap_ratio(s, t, i, j, s[j], t[j], s[i], t[i], S0, use_rank2, a, ws);
                n_eval += 3;
            }
            V_nuc += (hbarc/4.0) * (C01*v01*(1.0 + R_t - R_s - R_st) + C10*v10*(1.0 - R_t + R_s - R_st));
            const double rsum = 1.0 + std::fabs(R_t) + std::fabs(R_s) + std::fabs(R_st);
            vmag += (hbarc/4.0) * (std::fabs(C01*v01) + std::fabs(C10*v10)) * rsum;
        }
    }
    return V_nuc;
}

static std::vector<double> slice(const std::vector<double>& v, int w, int n) {
    return std::vector<double>(v.begin() + (std::size_t)w*n, v.begin() + (std::size_t)(w+1)*n);
}

// --- F1: exchange oracle ------------------------------------------------------
static void test_exchange(const Ansatz& a, cublasHandle_t h) {
    const int B = 256;
    std::mt19937_64 rng(4343);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);
    Dev dv(a); dv.push(hx, hs, ht, a);
    Workspace ws_dev;
    const int nf0 = eval_local_E_device(dv.ds, h, a, ws_dev, B);

    std::vector<real> Vn(dv.ds.V_nuc.n); dv.ds.V_nuc.down(Vn.data(), (std::size_t)B);
    std::vector<uint8_t> act(dv.ds.ex_active.n); dv.ds.ex_active.down(act.data(), (std::size_t)B*ex_types*ex_npairs);
    std::vector<uint8_t> ok(dv.ds.rank2_ok.n); dv.ds.rank2_ok.down(ok.data(), (std::size_t)B);

    Workspace ws;
    double worst = 0, worst_rel = 0, vscale = 0;
    long long n_eval_cpu = 0, n_active_dev = 0;
    int n_same_s = 0, n_same_t = 0, n_mixed = 0, gate_mism = 0;
    std::vector<double> ref(B), mag(B);
    for (int w = 0; w < B; w++) {
        auto x = slice(hx, w, D), s = slice(hs, w, N), t = slice(ht, w, N);
        int ne; bool gate; double vmag;
        ref[w] = cpu_vnuc(x, s, t, a, ws, false, ne, gate, vmag);
        mag[w] = vmag;
        n_eval_cpu += ne;
        if ((ok[w] != 0) != gate) gate_mism++;
        vscale = std::max(vscale, std::fabs(ref[w]));
        worst = std::max(worst, std::fabs((double)Vn[w] - ref[w]));
        worst_rel = std::max(worst_rel, std::fabs((double)Vn[w] - ref[w]) / std::max(1.0, mag[w]));
        for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
            bool ss = s[i]==s[j], st = t[i]==t[j];
            if (ss && st) continue;
            if (ss) n_same_s++; else if (st) n_same_t++; else n_mixed++;
        }
    }
    for (std::size_t q = 0; q < (std::size_t)B*ex_types*ex_npairs; q++) n_active_dev += act[q];
    std::printf("  exchange (rank-2 path): V_nuc abs %.2e / rel-to-summands %.2e (|V_nuc| up to %.3g MeV)  fallback walkers %d  gate disagreements %d\n",
                worst, worst_rel, vscale, nf0, gate_mism);
    std::printf("    pairs: %d same_s, %d same_t, %d mixed;  evaluated ratios CPU %lld  device active slots %lld\n",
                n_same_s, n_same_t, n_mixed, n_eval_cpu, n_active_dev);
    // Relative to the summand magnitude, not the prompt's absolute 1e-9: see cpu_vnuc.
    CHECK(worst_rel <= tol::ff(1e-11, 1e-3), "device V_nuc disagrees with local_E's exchange loop");
    CHECK(n_active_dev == n_eval_cpu, "active slots are not exactly the ratios local_E evaluates");
    CHECK(gate_mism == 0, "device rank-2 gate disagrees with rank2_well_conditioned");
    // Coverage, for the branches this (N, N_u, N_p) sector can produce at all: a
    // deuteron's single pair differs in both labels, so it has no same_s or same_t.
    const bool can_same_s = (N_u >= 2) || (N_d >= 2);
    const bool can_same_t = (N_p >= 2) || (N_n >= 2);
    const bool can_mixed  = (N_u >= 1 && N_d >= 1 && N_p >= 1 && N_n >= 1);
    CHECK((!can_same_s || n_same_s > 0) && (!can_same_t || n_same_t > 0) && (!can_mixed || n_mixed > 0),
          "configs did not cover every pair branch this sector allows");
    CHECK(vscale > 1.0, "V_nuc suspiciously small -- the bound would be vacuous");

    // Forced fallback: route every other walker through the host LU path by
    // clearing its gate, then rerun only the stages downstream of the gate. The
    // table buffers are untouched since eval_local_E_device, so this is exactly
    // the production fallback on well-conditioned walkers, compared against
    // local_E's loop with use_rank2 forced false.
    std::vector<uint8_t> forced(ok);
    int n_forced = 0;
    for (int w = 0; w < B; w += 2) { forced[w] = 0; n_forced++; }
    dv.ds.rank2_ok.up(forced.data(), (std::size_t)B);
    // Re-run the S' kernel under the poked gate first, as production orders it:
    // it zeroes S_swap for every gated-out walker. Without this the forced walkers
    // keep their rank-2 S' and the value check below passes even with the
    // fallback deleted (a negative control caught exactly that). B fits one slot
    // chunk, so rho_swap still holds this batch's rows.
    if (B > ex_walkers) throw std::runtime_error("test_exchange: B must fit one exchange chunk");
    ex_S_swap(dv.ds.rho_swap.d, dv.ds.dets_psi.d, dv.ds.Minv_batch.d, dv.ds.orb_out.d, dv.ds.s.d, dv.ds.t.d,
              dv.ds.pair_ij.d, dv.ds.ex_active.d, dv.ds.rank2_ok.d, dv.ds.S_swap.d, B, 0);
    const int nf = ex_fallback_host(dv.ds, a, ws_dev, B);
    ex_assemble(dv.ds.x.d, dv.ds.s.d, dv.ds.t.d, dv.ds.pair_ij.d, dv.ds.S_swap.d, dv.ds.S0.d, dv.ds.E_kin.d, dv.ds.v3n_out.d,
                dv.ds.V_coul.d, dv.ds.valid_jet.d, dv.ds.V_nuc.d, dv.ds.E_loc.d, dv.ds.valid_loc.d, B);
    dv.ds.V_nuc.down(Vn.data(), (std::size_t)B);
    double worst_fb = 0, worst_kept = 0;  // relative to summand magnitude
    for (int w = 0; w < B; w++) {
        auto x = slice(hx, w, D), s = slice(hs, w, N), t = slice(ht, w, N);
        int ne; bool gate; double vmag;
        if (w % 2 == 0) {
            double r = cpu_vnuc(x, s, t, a, ws, true, ne, gate, vmag);
            worst_fb = std::max(worst_fb, std::fabs((double)Vn[w] - r) / std::max(1.0, vmag));
        } else {
            worst_kept = std::max(worst_kept, std::fabs((double)Vn[w] - ref[w]) / std::max(1.0, mag[w]));
        }
    }
    std::printf("  exchange (forced host fallback): %d walkers flagged, %d took it;  V_nuc rel-to-summands %.2e  (untouched walkers %.2e)\n",
                n_forced, nf, worst_fb, worst_kept);
    CHECK(nf == n_forced, "fallback did not run for exactly the flagged walkers");
    CHECK(worst_fb <= tol::ff(1e-11, 1e-3), "fallback V_nuc disagrees with local_E's LU branch");
    CHECK(worst_kept <= tol::ff(1e-11, 1e-3), "fallback disturbed walkers it should not have touched");

    // Natural gate: exactly coincident particles zero SOME determinants (4.2
    // found cuBLAS and lu_det disagree on how many), which trips the per-walker
    // criterion on both sides. Values are not compared here -- on those walkers
    // the two LU paths disagree about S itself (Phase 4.2 E4) -- only that the
    // gate genuinely fires in production code without a poke.
    std::vector<double> cx, cs, ct; rand_cfg(B, cx, cs, ct, rng, true);
    dv.push(cx, cs, ct, a);
    const int nf_nat = eval_local_E_device(dv.ds, h, a, ws_dev, B);
    int cpu_gate_fail = 0;
    for (int w = 1; w < B; w += 3) {
        psi(slice(cx, w, D), slice(cs, w, N), slice(ct, w, N), a, ws, true);
        if (!rank2_well_conditioned(ws)) cpu_gate_fail++;
    }
    std::printf("  exchange (natural gate, %d coincident walkers): device %d fallback, CPU gate fails %d\n",
                (B + 1) / 3, nf_nat, cpu_gate_fail);
    // At N = 2 two coincident particles make EVERY determinant vanish together, so
    // max_det is itself ~0 and the relative criterion cannot trip. It needs N >= 3,
    // where the coincidence zeroes only some of the K determinants.
    if (N >= 3) CHECK(nf_nat > 0, "the rank-2 gate never fired on deliberately singular walkers");
}

// --- F2: full local_E parity ----------------------------------------------------
// E_loc on UNTHERMALISED random configs with a random ansatz reaches |E| ~ 7e3 MeV,
// and at that scale the Phase 3.2 LU gap (cuBLAS getrf vs lu_det, ~1e-13 relative
// on pj.v) alone is ~1e-9 MeV of E_kin -- measured 1.3e-9 at |E_kin| = 7259. An
// absolute 1e-9 on the SUM is therefore a bound on Phase 3's LU, not on this code.
// So the test decomposes: each term against its own CPU function, and every
// bound scaled by the magnitude of what was summed (see cpu_vnuc for why the
// value itself is the wrong scale).
// V_nuc is the same story: measured 1.2e-8 MeV absolute on a walker whose
// V_nuc is -1.1e4 MeV with exchange ratios |R| up to ~1e3 -- 1.1e-12 relative.
// The per-term split is what localises a regression; the total alone could not.
static void test_local_E_parity(const Ansatz& a, cublasHandle_t h) {
    const int B = 512;
    std::mt19937_64 rng(5151);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);
    Dev dv(a); dv.push(hx, hs, ht, a);
    Workspace ws_dev;

    // Sampler-state contract: recording now runs BETWEEN sweeps, so it must not
    // touch ds.logp. Seed it with a sentinel pattern and require it bitwise intact.
    std::vector<real> lp0(B);
    for (int w = 0; w < B; w++) lp0[w] = (real)(-1000.0 - 0.001 * w);
    dv.ds.logp.up(lp0.data(), (std::size_t)B);

    eval_local_E_device(dv.ds, h, a, ws_dev, B);

    std::vector<real> lp1(B); dv.ds.logp.down(lp1.data(), (std::size_t)B);
    int logp_touched = 0;
    for (int w = 0; w < B; w++) if (lp1[w] != lp0[w]) logp_touched++;
    std::printf("  sampler logp untouched by eval_local_E_device: %d of %d changed\n", logp_touched, B);
    CHECK(logp_touched == 0, "eval_local_E_device overwrote the sampler's cached logp");

    std::vector<real> E(B), Ek(B), V3(B), Vc(B), Vn(B);
    dv.ds.E_loc.down(E.data(), (std::size_t)B);   dv.ds.E_kin.down(Ek.data(), (std::size_t)B);
    dv.ds.v3n_out.down(V3.data(), (std::size_t)B); dv.ds.V_coul.down(Vc.data(), (std::size_t)B);
    dv.ds.V_nuc.down(Vn.data(), (std::size_t)B);
    std::vector<uint8_t> v(dv.ds.valid_loc.n); dv.ds.valid_loc.down(v.data(), (std::size_t)B);

    Workspace ws;
    std::vector<double> O;
    double wE = 0, wErel = 0, escale = 0, wk_rel = 0, wv3 = 0, wc = 0, wn = 0, wn_rel = 0, wn_at = 0;
    int mask_mism = 0, n_valid = 0;
    for (int w = 0; w < B; w++) {
        auto x = slice(hx, w, D), s = slice(hs, w, N), t = slice(ht, w, N);
        double Ecpu = 0.0;
        bool okc = local_E(x.data(), s.data(), t.data(), a, ws, O, Ecpu);
        if ((v[w] != 0) != okc) mask_mism++;
        if (!(okc && v[w])) continue;
        n_valid++;

        psi(x, s, t, a, ws, false);
        Jet pj = jpsi(x, s, t, a, ws);
        const double ek = -hbar2_2m * (pj.l / pj.v);
        int ne; bool gate; double vmag;
        const double vn = cpu_vnuc(x, s, t, a, ws, false, ne, gate, vmag);

        const double emag = std::fabs(ek) + std::fabs(V_3N(x)) + std::fabs(V_coulomb(x, t)) + vmag;
        escale = std::max(escale, std::fabs(Ecpu));
        wE     = std::max(wE,     std::fabs((double)E[w] - Ecpu));
        wErel  = std::max(wErel,  std::fabs((double)E[w] - Ecpu) / std::max(1.0, emag));
        wk_rel = std::max(wk_rel, std::fabs((double)Ek[w] - ek) / std::max(1.0, std::fabs(ek)));
        wv3    = std::max(wv3,    std::fabs((double)V3[w] - V_3N(x)));
        wc     = std::max(wc,     std::fabs((double)Vc[w] - V_coulomb(x, t)));
        if (std::fabs((double)Vn[w] - vn) > wn) { wn = std::fabs((double)Vn[w] - vn); wn_at = vn; }
        wn_rel = std::max(wn_rel, std::fabs((double)Vn[w] - vn) / std::max(1.0, vmag));
    }
    std::printf("  local_E parity (%d valid, |E| up to %.3g MeV): E_loc abs %.2e / rel-to-terms %.2e   mask disagreements %d of %d\n",
                n_valid, escale, wE, wErel, mask_mism, B);
    std::printf("    per term: E_kin rel %.2e   V_3N abs %.2e   V_coul abs %.2e   V_nuc abs %.2e / rel-to-summands %.2e\n", wk_rel, wv3, wc, wn, wn_rel);
    std::printf("    worst-abs V_nuc walker has V_nuc = %.4g MeV\n", wn_at);
    CHECK(mask_mism <= tol::ff(0, B / 100), "device validity mask disagrees with CPU local_E");
    CHECK(wErel <= tol::ff(1e-11, 1e-3),  "device E_loc disagrees with CPU local_E");
    CHECK(wk_rel <= tol::ff(1e-11, 1e-2), "E_kin term disagrees");
    CHECK(wv3 <= 1e-9,     "V_3N term disagrees");
    CHECK(wc <= 1e-12,     "V_coulomb term disagrees");
    CHECK(wn_rel <= tol::ff(1e-11, 1e-3), "V_nuc term disagrees");
    CHECK(n_valid > B / 2, "too few valid walkers for the parity check to mean anything");
}

// --- F1b: host O pass is local_E's O, bit for bit ---------------------------------
static void test_assemble_O(const Ansatz& a) {
    const int B = 64;
    std::mt19937_64 rng(6161);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng);
    Workspace ws1, ws2;
    std::vector<double> O1, O2;
    std::size_t ndiff = 0; int n = 0;
    for (int w = 0; w < B; w++) {
        double E;
        if (!local_E(&hx[(std::size_t)w*D], &hs[(std::size_t)w*N], &ht[(std::size_t)w*N], a, ws1, O1, E)) continue;
        assemble_O(&hx[(std::size_t)w*D], &hs[(std::size_t)w*N], &ht[(std::size_t)w*N], a, ws2, O2);
        for (std::size_t k = 0; k < O1.size(); k++) if (O1[k] != O2[k]) ndiff++;
        n++;
    }
    std::printf("  assemble_O vs local_E's O: %zu entries differ over %d walkers x %zu params\n", ndiff, n, O1.size());
    CHECK(ndiff == 0, "assemble_O is not bitwise local_E's O");
    CHECK(n > 0, "no valid walkers to compare O on");
}

// --- F4: determinism, and the device record path's per-walker statistics ---------
// Two full device record passes from an identical start must agree bitwise in
// everything download_iteration returns. The per-walker sums are accumulated on
// device; E_pool's rows are downloaded independently, so the host can rebuild
// Ew / E2w / nw from them in the same add order and require BITWISE equality.
static void test_determinism(const Ansatz& a, cublasHandle_t h) {
    const int B = 256, records = 2;
    ThreadPool pool(4);
    std::vector<Workspace> wss(4);
    WalkerBatch wb0; wb0.init(B);
    init_batch(wb0, a, &pool, wss);

    Dev dv(a);
    dv.ds.grow_phase5(a, false); dv.ds.grow_phase52(false); dv.ds.grow_phase53(false);
    PinnedArray st; dv.ds.upload_params(a, st);
    const std::size_t P = a.n_params(), Ns = (std::size_t)B * records;

    struct Run { IterStatsHost it; std::vector<double> O; long long up = 0, dn = 0; };
    auto run_once = [&](Run& R) {
        WalkerBatch wb = wb0;
        upload_and_reset(dv.ds, wb, st);
        eval_logp_batch(dv.ds, h, B);
        therm_batch_device(dv.ds, h, B, 0.5, 2);
        record_batch_device(dv.ds, h, a, wss[0], B, 0.5, records, /*with_O=*/true);
        xfer_stats().reset();
        download_iteration(dv.ds, B, records, st, R.it);
        R.up = xfer_stats().bytes_up; R.dn = xfer_stats().bytes_dn;
        R.O = opool_down(dv.ds.O_pool, Ns * P);
    };
    Run r1, r2; run_once(r1); run_once(r2);

    std::size_t dE = 0, dv_ = 0, dO = 0;
    for (std::size_t q = 0; q < Ns; q++) {
        if (r1.it.valid_pool[q] != r2.it.valid_pool[q]) dv_++;
        if (r1.it.valid_pool[q] && r1.it.E_pool[q] != r2.it.E_pool[q]) dE++;
        if (r1.it.valid_pool[q]) for (std::size_t k = 0; k < P; k++) if (r1.O[q*P+k] != r2.O[q*P+k]) dO++;
    }
    const BatchStats& b1 = r1.it.bs; const BatchStats& b2 = r2.it.bs;
    const bool stats_eq = b1.E_sum == b2.E_sum && b1.E2_sum == b2.E2_sum && b1.l2_sum == b2.l2_sum && b1.r2_sum == b2.r2_sum
                       && b1.n_valid == b2.n_valid && b1.Ew_sum == b2.Ew_sum && r1.it.acc == r2.it.acc && r1.it.sp_acc == r2.it.sp_acc;
    std::printf("  determinism (2 device record passes, B=%d x %d records): E_pool %zu, valid_pool %zu, O_pool %zu differ; stats %s  (%lld valid)\n",
                B, records, dE, dv_, dO, stats_eq ? "identical" : "DIFFER", b1.n_valid);
    CHECK(dE == 0 && dv_ == 0 && dO == 0 && stats_eq, "device record path is not bit-reproducible");
    CHECK(b1.n_valid > 0, "no valid samples recorded");

    // Per-walker sums rebuilt from the downloaded E_pool rows, record order.
    std::size_t bad = 0;
    for (int w = 0; w < B; w++) {
        double Ew = 0.0, E2w = 0.0; int nw = 0;
        for (int r = 0; r < records; r++) {
            const std::size_t q = (std::size_t)r * B + w;
            if (!r1.it.valid_pool[q]) continue;
            const double E = r1.it.E_pool[q];
            Ew += E; E2w += E * E; nw++;
        }
        if (Ew != b1.Ew_sum[w] || E2w != b1.E2w_sum[w] || nw != b1.nw[w]) bad++;
    }
    const long long expect_dn = (long long)(Ns * (sizeof(double) + 1) + (std::size_t)B * (4*sizeof(double) + sizeof(int) + 3*sizeof(long long)));
    std::printf("  device per-walker Ew/E2w/nw vs rebuilt from downloaded E_pool: %zu of %d walkers differ;  download_iteration bytes up %lld down %lld (expect 0 / %lld)\n",
                bad, B, r1.up, r1.dn, expect_dn);
    CHECK(bad == 0, "device per-walker accumulation is not record_one_walker's");
    CHECK(r1.up == 0 && r1.dn == expect_dn, "download_iteration transfers more than its contract");
}


int main() {
    gpu_select_device(true);
    Ansatz a({64},{64},{64}, Activation::Gelu);
    seed_ansatz(a, 2024);
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    try {
        test_exchange(a, h);
        test_local_E_parity(a, h);
        test_assemble_O(a);
        test_determinism(a, h);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
