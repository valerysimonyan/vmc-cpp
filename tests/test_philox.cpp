// Philox4x32-10 correctness, CPU-only. Builds with no CUDA toolkit.
//
// Test 1 is THE ANCHOR. Everything else here checks that the generator behaves
// like a good RNG; only the known-answer vectors establish that it IS Philox
// rather than something Philox-shaped. Confirmed by negative control: building
// with 9 rounds instead of 10 fails 12 KAT checks while passing every
// statistical test in this file.
#include "../lib/gpu/philox.h"
#include "../lib/rng_common.h"
#include "../lib/constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

// --- 1. Known-answer vectors ----------------------------------------------
// The published Random123 test vectors for philox4x32-10. Independently
// confirmed against curand_Philox4x32_10 from CUDA 12.8's own
// curand_philox4x32_x.h: 0 mismatches over 200000 random (ctr, key) pairs.
static void test_kat() {
    struct KAT {
        std::uint32_t c[4], k[2], want[4];
        const char* name;
    };
    const KAT v[] = {
        {{0,0,0,0}, {0,0},
         {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}, "ctr=0 key=0"},
        {{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu}, {0xffffffffu,0xffffffffu},
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}, "all ones"},
        {{0x243f6a88u,0x85a308d3u,0x13198a2eu,0x03707344u}, {0xa4093822u,0x299f31d0u},
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}, "digits of pi"},
    };
    for (const KAT& t : v) {
        Philox4 c; for (int i = 0; i < 4; i++) c.v[i] = t.c[i];
        Philox4 got = philox4x32_10_ctr(c, t.k[0], t.k[1]);
        for (int i = 0; i < 4; i++)
            CHECK(got.v[i] == t.want[i],
                  std::string("KAT ") + t.name + ": lane " + std::to_string(i) + " wrong");
    }

    // The 64-bit convenience form must agree with the raw form on lanes 2-3 = 0.
    Philox4 c0{{0,0,0,0}};
    Philox4 raw = philox4x32_10_ctr(c0, 0, 0);
    Philox4 wrapped = philox4x32_10(0ull, 0ull);
    for (int i = 0; i < 4; i++)
        CHECK(raw.v[i] == wrapped.v[i], "64-bit wrapper disagrees with raw form");
}

// --- 2. u01 range and moments ---------------------------------------------
static void test_u01() {
    PhiloxStream s = stream_for(12345ull, 0, 0ull);
    const long long n = 10000000;
    double sum = 0.0, sum2 = 0.0, lo = 2.0, hi = -1.0;
    for (long long i = 0; i < n; i++) {
        double u = s.u01();
        if (u < lo) lo = u;
        if (u > hi) hi = u;
        sum += u; sum2 += u * u;
    }
    CHECK(lo >= 0.0, "u01 returned a negative value");
    CHECK(hi <  1.0, "u01 returned 1.0 -- the half-open range is what the "
                     "Metropolis acceptance test relies on");

    const double mean = sum / n, var = sum2 / n - mean * mean;
    // Uniform[0,1): mean 1/2, variance 1/12. Standard error of the mean is
    // sqrt(1/12/n); 4 sigma is a ~6e-5 window at n = 1e7.
    const double se = std::sqrt((1.0/12.0) / (double)n);
    CHECK(std::fabs(mean - 0.5) < 4.0 * se,
          "u01 mean " + std::to_string(mean) + " is more than 4 sigma from 0.5");
    CHECK(std::fabs(var - 1.0/12.0) < 1e-4,
          "u01 variance " + std::to_string(var) + " is far from 1/12");
}

// --- 3. reproducibility and stream independence ---------------------------
static void test_streams() {
    // Same (seed, w) must give the same sequence -- this is the property that
    // makes a run reproducible regardless of how walkers are scheduled.
    PhiloxStream a = stream_for(999ull, 7, 0ull);
    PhiloxStream b = stream_for(999ull, 7, 0ull);
    for (int i = 0; i < 1000; i++)
        CHECK(a.u01() == b.u01(), "same (seed, walker) gave different draws");

    // Different walkers must not. splitmix64 on the walker index is what
    // decorrelates adjacent indices; without it, keys 0 and 1 would differ in
    // one bit and Philox would still separate them, but relying on that is
    // exactly the assumption worth testing.
    const int n = 10000;
    std::vector<double> u(n), v(n);
    PhiloxStream s0 = stream_for(999ull, 0, 0ull);
    PhiloxStream s1 = stream_for(999ull, 1, 0ull);
    for (int i = 0; i < n; i++) { u[i] = s0.u01(); v[i] = s1.u01(); }

    int identical = 0;
    for (int i = 0; i < n; i++) if (u[i] == v[i]) identical++;
    CHECK(identical == 0, "adjacent walkers produced identical draws");

    double mu = 0, mv = 0;
    for (int i = 0; i < n; i++) { mu += u[i]; mv += v[i]; }
    mu /= n; mv /= n;
    double cov = 0, su = 0, sv = 0;
    for (int i = 0; i < n; i++) {
        cov += (u[i]-mu)*(v[i]-mv); su += (u[i]-mu)*(u[i]-mu); sv += (v[i]-mv)*(v[i]-mv);
    }
    const double corr = cov / std::sqrt(su * sv);
    // Statistical smoke only. For independent streams the sample correlation is
    // ~N(0, 1/sqrt(n)) = 0.01 at n = 1e4, so 0.05 is ~5 sigma: loose enough not
    // to flake, tight enough to catch two streams that are actually related.
    CHECK(std::fabs(corr) < 0.05,
          "adjacent walker streams correlate at " + std::to_string(corr));
}

// --- 4. counter persistence ------------------------------------------------
static void test_counter() {
    // Two runs of 100 draws must equal one run of 200. This is what lets a
    // kernel load the counter, advance it locally, and write it back once --
    // if it failed, every kernel boundary would perturb the stream.
    std::vector<double> split, whole;
    PhiloxStream a = stream_for(4242ull, 3, 0ull);
    for (int i = 0; i < 100; i++) split.push_back(a.u01());
    PhiloxStream b = stream_for(4242ull, 3, a.ctr);   // resume from the counter alone
    for (int i = 0; i < 100; i++) split.push_back(b.u01());

    PhiloxStream c = stream_for(4242ull, 3, 0ull);
    for (int i = 0; i < 200; i++) whole.push_back(c.u01());

    CHECK(split == whole, "100 + 100 draws differ from 200 -- the counter does "
                          "not fully determine stream position");
    CHECK(a.ctr == 100 && b.ctr == 200 && c.ctr == 200,
          "counter did not advance one tick per u01");
}

// --- 5. uint_below and usym -----------------------------------------------
static void test_derived() {
    PhiloxStream s = stream_for(77ull, 0, 0ull);
    const int n = D;                       // the value the Metropolis sweep uses
    std::vector<long long> hist(n, 0);
    const long long draws = 2000000;
    for (long long i = 0; i < draws; i++) {
        int k = s.uint_below(n);
        CHECK(k >= 0 && k < n, "uint_below out of range");
        if (k >= 0 && k < n) hist[k]++;
    }
    // Multiply-shift has relative bias <= n/2^32 ~ 4e-9 for n = 18, far below
    // the ~7e-4 sampling noise at 2e6 draws, so this is really a uniformity
    // check on the generator rather than on the bias bound.
    const double expect = (double)draws / n;
    for (int k = 0; k < n; k++) {
        const double dev = std::fabs(hist[k] - expect) / std::sqrt(expect);
        CHECK(dev < 5.0, "uint_below bucket " + std::to_string(k)
                         + " deviates by " + std::to_string(dev) + " sigma");
    }

    PhiloxStream t = stream_for(78ull, 0, 0ull);
    double lo = 1e9, hi = -1e9;
    for (int i = 0; i < 100000; i++) { double x = t.usym(2.5); lo = std::min(lo,x); hi = std::max(hi,x); }
    CHECK(lo >= -2.5 && hi < 2.5, "usym escaped [-a, a)");
    CHECK(lo < -2.4 && hi > 2.4, "usym did not cover its range");
}

int main() {
    test_kat();
    test_u01();
    test_streams();
    test_counter();
    test_derived();
    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
