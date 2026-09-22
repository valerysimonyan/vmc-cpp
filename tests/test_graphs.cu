// Phase 6.1: elementwise fusions and CUDA-graph sweeps.
//
// Every check here is BITWISE. A fusion performs the same operations in the
// same order as the kernels it replaces, and a graph replays the same kernels
// with the same arguments, so any difference at all is a bug.
#include "../lib/gpu/arena.h"
#include "../lib/gpu/eval.h"
#include "../lib/gpu/net_kernels.h"
#include "../lib/gpu/det_kernels.h"
#include "../lib/gpu/gpu_sampler.h"
#include "../lib/gpu/sampler_kernels.h"
#include "../lib/physics.h"
#include "../lib/walkers.h"
#include "../lib/pool.h"
#include "../tests/test_common.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
    g_failures++; } } while (0)

template <typename T>
static std::vector<T> dl(const DeviceArray<T>& a, std::size_t n) { std::vector<T> h(n); a.down(h.data(), n); return h; }

// Count elements whose bit patterns differ (so -0/+0 and NaN payloads count too).
template <typename T>
static std::size_t bitdiff(const std::vector<T>& a, const std::vector<T>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); i++) if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) n++;
    return n;
}

struct Dev {
    DeviceState ds;
    explicit Dev(const Ansatz& a) : ds(a, false) {
        ds.grow_phase3(a, false); ds.grow_phase33(false); ds.grow_phase4(false);
        ds.grow_phase42(false);   ds.grow_phase43(false);
    }
};

// Random walkers. In every fourth walker particle 1 is a full copy of particle
// 0 -- position AND spin AND isospin, since (s,t) are network inputs -- so two
// columns of every Slater matrix coincide. (Spin counts are not conserved in
// those walkers; the fused kernels are pure functions of their inputs, so that
// does not matter here.)
static void load_random(DeviceState& ds, const Ansatz& a, int B, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-x_init_range, x_init_range);
    std::vector<real> hx((std::size_t)B*D), hs((std::size_t)B*N), ht((std::size_t)B*N);
    for (auto& v : hx) v = (real)d(rng);
    for (int w = 0; w < B; w++) {
        for (int i = 0; i < N; i++) { hs[(std::size_t)w*N+i] = (i < N_u) ? 1 : -1; ht[(std::size_t)w*N+i] = (i < N_p) ? 1 : -1; }
        std::shuffle(hs.begin() + (std::size_t)w*N, hs.begin() + (std::size_t)(w+1)*N, rng);
        std::shuffle(ht.begin() + (std::size_t)w*N, ht.begin() + (std::size_t)(w+1)*N, rng);
        if (N >= 2 && w % 4 == 1) {
            for (int q = 0; q < dim; q++) hx[(std::size_t)w*D + dim + q] = hx[(std::size_t)w*D + q];
            hs[(std::size_t)w*N + 1] = hs[(std::size_t)w*N];
            ht[(std::size_t)w*N + 1] = ht[(std::size_t)w*N];
        }
    }
    PinnedArray st; ds.upload_params(a, st);
    ds.x.up(hx.data(), hx.size()); ds.s.up(hs.data(), hs.size()); ds.t.up(ht.data(), ht.size());
}

// --- Task A: each fusion against the kernels it replaced ---------------------
static void test_fusions(const Ansatz& a, cublasHandle_t h) {
    const int B = 512;
    Dev dv(a);
    DeviceState& ds = dv.ds;
    load_random(ds, a, B, 7);
    const std::size_t nx = (std::size_t)B*D, nf = (std::size_t)B*N*(dim+2), nfc = nf*4;

    // 1. shift_to_com + build_feat
    shift_to_com(ds.x.d, ds.x_sh.d, B);
    build_feat(ds.x_sh.d, ds.s.d, ds.t.d, ds.feat_in.d, B);
    auto xsh0 = dl(ds.x_sh, nx); auto f0 = dl(ds.feat_in, nf);
    ds.x_sh.zero(); ds.feat_in.zero();
    shift_build_feat(ds.x.d, ds.s.d, ds.t.d, ds.x_sh.d, ds.feat_in.d, B);
    auto xsh1 = dl(ds.x_sh, nx); auto f1 = dl(ds.feat_in, nf);
    const std::size_t d1 = bitdiff(xsh0, xsh1) + bitdiff(f0, f1);
    std::printf("  shift_build_feat vs shift_to_com+build_feat: %zu of %zu values differ\n", d1, nx + nf);
    CHECK(d1 == 0, "fused shift+feat is not bitwise the unfused pair");

    // 1b. the combo variant used by the (s,t) table
    feat_combo_unfused(ds, B);
    auto xshc0 = dl(ds.x_sh, nx); auto fc0 = dl(ds.feat_in, nfc);
    ds.x_sh.zero(); ds.feat_in.zero();
    build_st_table_batch(ds, h, B);                  // fused feat stage, then the nets (which only read feat_in)
    auto xshc1 = dl(ds.x_sh, nx); auto fc1 = dl(ds.feat_in, nfc);
    const std::size_t d1b = bitdiff(xshc0, xshc1) + bitdiff(fc0, fc1);
    std::printf("  shift_feat_combo vs shift_to_com+build_feat_combo: %zu of %zu values differ\n", d1b, nx + nfc);
    CHECK(d1b == 0, "fused combo feat is not bitwise the unfused pair");

    // 2. The eval tail: S_combine + envelope_logp vs combine_envelope, on
    // identical dets. Round 1 is the natural batch; round 2 marks every matrix
    // of every fourth walker singular through lu_info -- exactly what getrf
    // reports for an exact zero pivot -- so the S == 0 / logp == -inf branch is
    // compared too. (Coincident particles do not reach it: elimination leaves
    // rounding residue, so their pivots are tiny rather than zero.)
    eval_logp_batch(ds, h, B);                       // leaves orb_out, rho_out, x_sh
    assemble_M(ds.orb_out.d, ds.M_batch.d, B);
    lu_factor(h, B*K, ds.lu_ptrs.d, ds.lu_piv.d, ds.lu_info.d);
    auto info = dl(ds.lu_info, (std::size_t)B*K);
    for (int round = 0; round < 2; round++) {
        if (round == 1) {
            for (int w = 1; w < B; w += 4) for (int k = 0; k < K; k++) info[(std::size_t)w*K + k] = 1;
            ds.lu_info.up(info.data(), info.size());
        }
        dets_from_lu(ds.M_batch.d, ds.lu_piv.d, ds.lu_info.d, ds.dets.d, B*K);
        S_combine(ds.rho_out.d, ds.dets.d, ds.S.d, B);
        envelope_logp(ds.x_sh.d, ds.S.d, ds.params.d, ds.P, ds.logp.d, B);
        auto S0 = dl(ds.S, B); auto lp0 = dl(ds.logp, B);
        ds.S.zero(); ds.logp.zero();
        combine_envelope(ds.rho_out.d, ds.dets.d, ds.S.d, ds.x_sh.d, ds.params.d, ds.P, ds.logp.d, B);
        auto S1 = dl(ds.S, B); auto lp1 = dl(ds.logp, B);
        std::size_t n_ninf = 0;
        for (real v : lp0) if (v == -INFINITY) n_ninf++;
        const std::size_t d2 = bitdiff(S0, S1) + bitdiff(lp0, lp1);
        std::printf("  combine_envelope (%s) vs S_combine+envelope_logp: %zu of %d values differ  (%zu -inf logp)\n",
                    round ? "forced singular" : "natural", d2, 2*B, n_ninf);
        CHECK(d2 == 0, "fused combine+envelope is not bitwise the two kernels");
        if (round == 1) CHECK(n_ninf == (std::size_t)B/4, "forced-singular walkers did not reach the -inf branch");
    }
}

// --- Task B/C1: 50 sweeps, graphs on vs off ----------------------------------
struct Snap {
    std::vector<real> x, s, t, logp;
    std::vector<unsigned long long> ctr;
    std::vector<long long> acc, sp, tau;
};

static Snap snap(DeviceState& ds, int B) {
    Snap z;
    z.x = dl(ds.x, (std::size_t)B*D); z.s = dl(ds.s, (std::size_t)B*N); z.t = dl(ds.t, (std::size_t)B*N);
    z.logp = dl(ds.logp, B); z.ctr = dl(ds.rng_ctr, B);
    z.acc = dl(ds.acc, B); z.sp = dl(ds.sp_acc, B); z.tau = dl(ds.tau_acc, B);
    return z;
}

static void test_graph_sweeps(const Ansatz& a, cublasHandle_t h) {
    const int B = 512, B_small = 300;
    ThreadPool pool(4);
    std::vector<Workspace> wss(4);
    WalkerBatch wb0; wb0.init(B);
    init_batch(wb0, a, &pool, wss);

    Dev dv(a);
    DeviceState& ds = dv.ds;
    PinnedArray st; ds.upload_params(a, st);

    // 25 sweeps at one step, 20 at another (forces a re-capture + in-place
    // update), then 5 at a smaller B (forces re-instantiation: grid sizes change).
    auto run = [&](bool graphs) {
        ds.graphs.enabled = graphs;
        WalkerBatch wb = wb0;
        upload_and_reset(ds, wb, st);
        eval_logp_batch(ds, h, B);
        for (int k = 0; k < 50; k++) {
            const int    Bk = (k < 45) ? B : B_small;
            const double step = (k < 25) ? 0.5 : 0.65;
            sweep_device(ds, h, Bk, step);
            recenter_device(ds, Bk);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return snap(ds, B);
    };

    const Snap off = run(false);
    const long long cap0 = ds.graphs.n_captures;
    const Snap on  = run(true);
    const long long caps = ds.graphs.n_captures - cap0;

    const std::size_t d = bitdiff(off.x, on.x) + bitdiff(off.s, on.s) + bitdiff(off.t, on.t) + bitdiff(off.logp, on.logp)
                        + bitdiff(off.ctr, on.ctr) + bitdiff(off.acc, on.acc) + bitdiff(off.sp, on.sp) + bitdiff(off.tau, on.tau);
    long long acc = 0, sp = 0, tau = 0;
    for (int w = 0; w < B; w++) { acc += on.acc[w]; sp += on.sp[w]; tau += on.tau[w]; }
    std::printf("  50 sweeps graphs-on vs graphs-off (B=%d, step change at 25, B=%d from 45): %zu values differ in x/s/t/logp/rng_ctr/counters\n",
                B, B_small, d);
    std::printf("    graph path: %lld captures, %lld in-place updates; nodes per stage: coord %zu, spin %zu, tau %zu; accepted coord %lld spin %lld tau %lld\n",
                caps, ds.graphs.n_updates, ds.graphs.coord_nodes, ds.graphs.spin_nodes, ds.graphs.tau_nodes, acc, sp, tau);
    CHECK(d == 0, "graph sweeps are not bitwise the eager sweeps");
    CHECK(caps >= 3 && ds.graphs.coord_nodes > 0, "the graph path did not actually run (no captures)");
    CHECK(ds.graphs.n_updates >= 1, "the step change did not go through cudaGraphExecUpdate");
    CHECK(acc > 0 && acc < (long long)50 * B * draws, "acceptance degenerate -- the comparison would be vacuous");
    if (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0) CHECK(sp > 0, "no spin swaps accepted");

    // Replays must draw fresh random numbers: every walker's Philox counter
    // (zeroed by upload_and_reset) has to have advanced. Equality with the eager
    // counters is already part of the bitdiff above.
    std::size_t moved = 0;
    for (int w = 0; w < B; w++) if (on.ctr[w] != 0ull) moved++;
    CHECK(moved == (std::size_t)B, "some walkers' Philox counters never advanced under replay");
}

int main() {
    gpu_select_device(true);
    Ansatz a({64},{64},{64}, Activation::Gelu);
    seed_ansatz(a, 2024);
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    try {
        test_fusions(a, h);
        test_graph_sweeps(a, h);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures) { std::cerr << g_failures << " check(s) FAILED\n"; return 1; }
    std::printf("test_graphs: all checks passed\n");
    return 0;
}
