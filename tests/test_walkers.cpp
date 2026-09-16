#include "../lib/walkers.h"
#include "../lib/physics.h"
#include "../lib/constants.h"
#include "../lib/network.h"
#include "../lib/pool.h"
#include "test_common.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

static bool state_equal(const WalkerBatch& a, const WalkerBatch& b) {
    return a.x == b.x && a.s == b.s && a.t == b.t && a.logp == b.logp;
}

static WalkerBatch run_scenario(const Ansatz& a, int B, int n_workers, int n_sweeps, double step) {
    WalkerBatch wb;
    wb.init(B);
    ThreadPool pool(n_workers);
    std::vector<Workspace> wss(n_workers);
    init_batch(wb, a, &pool, wss);
    therm_batch(wb, a, step, n_sweeps, &pool, wss);
    return wb;
}

static void test_determinism(const Ansatz& a) {
    WalkerBatch w1 = run_scenario(a, 84, 1, 20, step0);
    WalkerBatch w7 = run_scenario(a, 84, 7, 20, step0);
    CHECK(state_equal(w1, w7), "determinism: n_thread=1 vs n_thread=7 diverged");
}

static void test_chunking(const Ansatz& a) {
    WalkerBatch w2 = run_scenario(a, 84, 2, 20, step0);
    WalkerBatch w3 = run_scenario(a, 84, 3, 20, step0);
    CHECK(state_equal(w2, w3), "chunking: n_thread=2 vs n_thread=3 diverged");
}

static void test_sector_conservation(const Ansatz& a) {
    WalkerBatch wb = run_scenario(a, 84, 7, 200, step0);
    for (int w = 0; w < wb.B; w++) {
        int n_up = 0, n_p = 0;
        for (int i = 0; i < N; i++) {
            if (wb.s[(std::size_t)w*N + i] > 0) n_up++;
            if (wb.t[(std::size_t)w*N + i] > 0) n_p++;
        }
        CHECK(n_up == N_u, "sector conservation: walker " + std::to_string(w) + " wrong N_u");
        CHECK(n_p == N_p, "sector conservation: walker " + std::to_string(w) + " wrong N_p");
    }
}

static void test_logp_cache(const Ansatz& a) {
    WalkerBatch wb = run_scenario(a, 84, 7, 20, step0);
    Workspace ws;
    for (int w = 0; w < 50 && w < wb.B; w++) {
        double fresh = log_p(&wb.x[(std::size_t)w*D], &wb.s[(std::size_t)w*N], &wb.t[(std::size_t)w*N], a, ws);
        CHECK(std::fabs(fresh - wb.logp[w]) < 1e-12, "logp cache stale for walker " + std::to_string(w));
    }
}

static void test_stream_independence(const Ansatz& a) {
    WalkerBatch wb;
    wb.init(4);
    ThreadPool pool(1);
    std::vector<Workspace> wss(1);
    init_batch(wb, a, &pool, wss);

    // Force walker 1 to match walker 0 exactly, so the only difference left is the rng stream.
    for (int d = 0; d < D; d++) wb.x[D + d] = wb.x[d];
    for (int i = 0; i < N; i++) { wb.s[N + i] = wb.s[i]; wb.t[N + i] = wb.t[i]; }
    refresh_logp(wb, a, &pool, wss);

    Workspace ws;
    sweep_one(wb, 0, a, step0, ws);
    sweep_one(wb, 1, a, step0, ws);

    bool identical = true;
    for (int d = 0; d < D; d++) if (wb.x[d] != wb.x[D + d]) { identical = false; break; }
    CHECK(!identical, "stream independence: walkers 0 and 1 evolved identically");
}

// ---------------------------------------------------------------------------
// Task D1: full pipeline determinism. init + therm + record must produce
// bit-identical pools and statistics for any worker count. Per-walker RNG
// streams make the sampling worker-independent; the fixed-order walker sweep in
// record_batch makes the REDUCTIONS worker-independent. Both are required --
// the second is what this case exists to pin down, since a per-worker partial
// sum would still pass every statistical check while changing the last bits.
// ---------------------------------------------------------------------------
struct RunOut {
    std::vector<double> E_pool, O_pool;
    std::vector<uint8_t> valid_pool;
    BatchStats bs;
};

static void run_full(const Ansatz& a, int B, int n_workers, int records, RunOut& out) {
    std::size_t P  = a.n_params();
    std::size_t ns = (std::size_t)records * (std::size_t)B;

    WalkerBatch wb;
    wb.init(B);
    ThreadPool pool(n_workers);
    std::vector<Workspace> wss(n_workers);

    init_batch(wb, a, &pool, wss);
    therm_batch(wb, a, step0, 10, &pool, wss);

    out.E_pool.assign(ns, 0.0);
    out.O_pool.assign(ns * P, 0.0);
    out.valid_pool.assign(ns, 0);
    record_batch(wb, a, step0, records, &pool, wss,
                 out.E_pool, out.O_pool, out.valid_pool, out.bs);
}

static void check_run_equal(const RunOut& a, const RunOut& b, const std::string& tag) {
    CHECK(a.E_pool     == b.E_pool,     tag + ": E_pool differs");
    CHECK(a.O_pool     == b.O_pool,     tag + ": O_pool differs");
    CHECK(a.valid_pool == b.valid_pool, tag + ": valid_pool differs");

    CHECK(a.bs.E_sum    == b.bs.E_sum,    tag + ": BatchStats.E_sum differs");
    CHECK(a.bs.E2_sum   == b.bs.E2_sum,   tag + ": BatchStats.E2_sum differs");
    CHECK(a.bs.l2_sum   == b.bs.l2_sum,   tag + ": BatchStats.l2_sum differs");
    CHECK(a.bs.r2_sum   == b.bs.r2_sum,   tag + ": BatchStats.r2_sum differs");
    CHECK(a.bs.n_valid  == b.bs.n_valid,  tag + ": BatchStats.n_valid differs");
    CHECK(a.bs.n_invalid== b.bs.n_invalid,tag + ": BatchStats.n_invalid differs");

    CHECK(a.bs.Ew_sum  == b.bs.Ew_sum,  tag + ": BatchStats.Ew_sum differs");
    CHECK(a.bs.E2w_sum == b.bs.E2w_sum, tag + ": BatchStats.E2w_sum differs");
    CHECK(a.bs.l2w_sum == b.bs.l2w_sum, tag + ": BatchStats.l2w_sum differs");
    CHECK(a.bs.r2w_sum == b.bs.r2w_sum, tag + ": BatchStats.r2w_sum differs");
    CHECK(a.bs.nw      == b.bs.nw,      tag + ": BatchStats.nw differs");
}

static void test_record_determinism(const Ansatz& a) {
    const int B = 84, records = 2;
    RunOut r1, r3, r7;
    run_full(a, B, 1, records, r1);
    run_full(a, B, 3, records, r3);
    run_full(a, B, 7, records, r7);

    check_run_equal(r1, r3, "record determinism n_thread 1 vs 3");
    check_run_equal(r1, r7, "record determinism n_thread 1 vs 7");

    // A run in which nothing was valid would satisfy every equality above
    // trivially, so make sure the comparison had something to compare.
    CHECK(r1.bs.n_valid > 0, "record determinism: no valid samples -- test vacuous");
    CHECK(r1.bs.n_valid + r1.bs.n_invalid == (long long)records * B,
          "record determinism: valid + invalid must cover every slot");
}

// ---------------------------------------------------------------------------
// Task D2: init spread. init_batch must actually write positions. The old code
// drew from usym() and discarded the result, leaving x at the zeros from
// WalkerBatch::init, so every walker started at the origin.
//
// Caveat on coverage: at N=6 in the standard sector, particles 0 and 1 share
// (s,t) = (+,+), so at the origin they give identical orbital columns, the
// determinant is exactly zero, log_p is -inf, and init_batch's off-node retry
// randomizes the positions anyway -- masking the bug. It is unmasked in sectors
// where every (s,t) is distinct, e.g. the deuteron (N=2, N_u=1, N_p=1): no node
// hit, no retry, and every walker sits at exactly x=0. So this case is a
// regression guard that bites at N=2/N=3, which is where the next runs are.
// ---------------------------------------------------------------------------
static void test_init_spread(const Ansatz& a) {
    const int B = 84;
    WalkerBatch wb;
    wb.init(B);
    ThreadPool pool(3);
    std::vector<Workspace> wss(3);
    init_batch(wb, a, &pool, wss);   // deliberately no thermalization

    bool all_same = true;
    for (int w = 1; w < B && all_same; w++)
        for (int d = 0; d < D; d++)
            if (wb.x[(std::size_t)w*D + d] != wb.x[d]) { all_same = false; break; }
    CHECK(!all_same, "init spread: all walkers have identical positions");

    for (int w = 0; w < B; w++) {
        const double* xw = &wb.x[(std::size_t)w*D];
        double Rcm[dim] = {};
        for (int p = 0; p < N; p++)
            for (int d = 0; d < dim; d++) Rcm[d] += xw[p*dim + d];
        for (int d = 0; d < dim; d++) Rcm[d] /= N;

        double r2 = 0.0;
        for (int p = 0; p < N; p++)
            for (int d = 0; d < dim; d++) {
                double diff = xw[p*dim + d] - Rcm[d];
                r2 += diff * diff;
            }
        CHECK(std::sqrt(r2 / N) > 0.0,
              "init spread: walker " + std::to_string(w) + " has zero r_rms");
    }
}


int main() {
    Ansatz a({8}, {8}, {8}, Activation::Gelu);
    seed_ansatz(a, 1001);

    test_determinism(a);
    test_chunking(a);
    test_sector_conservation(a);
    test_logp_cache(a);
    test_stream_independence(a);
    test_stream_independence(a);
    test_record_determinism(a);
    test_init_spread(a);
    
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cout << g_failures << " CHECK(S) FAILED\n";
    return 1;
}