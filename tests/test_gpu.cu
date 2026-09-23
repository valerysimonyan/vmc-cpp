// GPU infrastructure tests. Built only when VMC_CUDA=ON.
//
// No physics kernels yet: this proves the plumbing. If any of these fail, no
// later kernel result can be trusted, because every one of them depends on the
// allocation, transfer and error paths exercised here.
#include "../lib/gpu/arena.h"
#include "../tests/test_tolerances.h"
#include "../lib/precision.h"
#include "../lib/gpu/gpu_util.h"
#include "../lib/gpu/smoke.h"
#include "../lib/physics.h"
#include "../lib/walkers.h"
#include "../lib/pool.h"
#include "../tests/test_common.h"
#include "../lib/gpu/philox.h"
#include "../lib/gpu/philox_kernels.h"
#include "../lib/gpu/layouts.h"
#include "../lib/gpu/net_kernels.h"
#include "../lib/gpu/net_forward.h"
#include "../lib/gpu/det_kernels.h"
#include "../lib/gpu/eval.h"
#include "../lib/gpu/sampler_kernels.h"
#include "../lib/gpu/gpu_sampler.h"
#include "../lib/slater.h"
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <type_traits>
#include <string>
#include <vector>

// With real = double every round trip is exact. With real = float a host double
// is narrowed on upload, so the round trip is lossy BY DESIGN and an exact
// comparison would fail on correct code. Compare exactly in FP64 mode and to
// float epsilon otherwise -- anything worse than that is a transfer bug, not
// rounding.
static constexpr bool real_is_double = std::is_same<real, double>::value;

static bool real_roundtrip_ok(double got, double want) {
    if (real_is_double) return got == want;
    return std::fabs(got - want) <= 1e-6 * std::max(1.0, std::fabs(want));
}

static bool vec_roundtrip_ok(const std::vector<double>& got, const std::vector<double>& want) {
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); i++)
        if (!real_roundtrip_ok(got[i], want[i])) return false;
    return true;
}

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

// --- 0. smoke: launch + sync + error path ---------------------------------
static void test_smoke() {
    const std::size_t n = 1000;
    std::vector<double> x(n), y(n), out(n, 0.0);
    std::mt19937_64 rng(1); std::uniform_real_distribution<double> u(-1, 1);
    for (std::size_t i = 0; i < n; i++) { x[i] = u(rng); y[i] = u(rng); }

    gpu_smoke_saxpy(2.5, x.data(), y.data(), out.data(), n);

    // Exact, not a tolerance -- and that is only valid because the build passes
    // -fmad=false. With nvcc's default contraction the device fuses a*x+y into
    // one instruction that rounds ONCE where this host expression rounds twice,
    // and roughly 19% of elements differ in the last bit. See VMC_CUDA_STRICT_FP.
    for (std::size_t i = 0; i < n; i++)
        CHECK(out[i] == 2.5 * x[i] + y[i], "smoke: saxpy mismatch at " + std::to_string(i));
}

// --- 1. DeviceArray round trip --------------------------------------------
template <typename T>
static void roundtrip(std::size_t n, const std::string& tag) {
    std::mt19937_64 rng(1234 + n);
    std::vector<T> h(n), back(n);
    for (std::size_t i = 0; i < n; i++) h[i] = (T)(rng() % 251);

    DeviceArray<T> d;
    d.alloc(n);
    d.up(h.data(), n);
    d.down(back.data(), n);
    CHECK(h == back, tag + ": round trip differs at n=" + std::to_string(n));

    if (n > 0) {
        d.zero();
        d.down(back.data(), n);
        bool all_zero = true;
        for (std::size_t i = 0; i < n; i++) if (back[i] != T(0)) all_zero = false;
        CHECK(all_zero, tag + ": zero() left nonzero bytes at n=" + std::to_string(n));
    }
}

static void test_device_array() {
    // n = 0 must not allocate or crash; odd and prime sizes catch any hidden
    // assumption that lengths are multiples of a warp or a block.
    const std::size_t sizes[] = {0, 1, 7, 33, 1023, 4096};
    for (std::size_t n : sizes) {
        roundtrip<double>(n, "DeviceArray<double>");
        roundtrip<uint8_t>(n, "DeviceArray<uint8_t>");
    }

    DeviceArray<double> empty;
    CHECK(empty.d == nullptr && empty.n == 0, "default DeviceArray must be null");
    empty.alloc(0);
    CHECK(empty.d == nullptr && empty.n == 0, "alloc(0) must stay null");

    // Move must transfer ownership and leave the source null -- otherwise the
    // pointer is freed twice.
    DeviceArray<double> a; a.alloc(16);
    double* raw = a.d;
    DeviceArray<double> b(std::move(a));
    CHECK(b.d == raw && b.n == 16, "move ctor must transfer the pointer");
    CHECK(a.d == nullptr && a.n == 0, "moved-from DeviceArray must be null");
}

// --- 2. DeviceState sizing -------------------------------------------------
static void test_device_state(const Ansatz& a) {
    DeviceState ds(a, /*verbose=*/true);

    const std::size_t P  = a.n_params();
    const std::size_t B  = (std::size_t)n_walkers;
    const std::size_t Ns = (std::size_t)n_walkers * (std::size_t)records_per_iter_max;

    CHECK(ds.B == B && ds.P == P && ds.Ns_max == Ns, "DeviceState: sizes wrong");

    // Hand-computed expectation, independent of total_bytes()'s own arithmetic.
    // Note which members follow `real` and which are pinned to double.
    const std::size_t expect =
          B * (std::size_t)D * sizeof(real)               // x
        + B * (std::size_t)N * sizeof(real) * 2           // s, t
        + B * sizeof(real)                                // logp
        + B * sizeof(uint8_t)                             // valid
        + B * sizeof(unsigned long long)                  // rng_ctr
        + P * sizeof(real)                                // params
        + (fp32_forward ? P * sizeof(float) : 0)          // params_f, the float mirror (6.3)
        + Ns * sizeof(double)                             // E_pool
        + Ns * P * sizeof(opool_t)                        // O_pool (float under fp32_opool)
        + Ns * sizeof(uint8_t);                           // valid_pool
    CHECK(ds.total_bytes() == expect,
          "DeviceState: reported bytes " + std::to_string(ds.total_bytes())
          + " != hand-computed " + std::to_string(expect));
}

// --- 3. walker round trip --------------------------------------------------
static void test_walker_roundtrip(const Ansatz& a) {
    WalkerBatch wb;
    wb.init(n_walkers);
    ThreadPool pool(4);
    std::vector<Workspace> wss(4);
    init_batch(wb, a, &pool, wss);
    therm_batch(wb, a, step0, 5, &pool, wss);

    // valid[] is not written by therm_batch; set a pattern so the round trip
    // has something to preserve.
    for (int w = 0; w < wb.B; w++) wb.valid[w] = (uint8_t)(w % 2);

    const std::vector<double>  x0 = wb.x,  s0 = wb.s, t0 = wb.t, lp0 = wb.logp;
    const std::vector<uint8_t> v0 = wb.valid;

    DeviceState ds(a, /*verbose=*/false);
    PinnedArray staging;
    ds.upload_walkers(wb, staging);

    // Scribble over the host copy so a no-op download cannot pass.
    std::fill(wb.x.begin(), wb.x.end(), -7.0);
    std::fill(wb.s.begin(), wb.s.end(), -7.0);
    std::fill(wb.t.begin(), wb.t.end(), -7.0);
    std::fill(wb.logp.begin(), wb.logp.end(), -7.0);
    std::fill(wb.valid.begin(), wb.valid.end(), 0xAB);

    ds.download_walkers(wb, staging);

    CHECK(vec_roundtrip_ok(wb.x,    x0),  "walker round trip: x differs");
    CHECK(vec_roundtrip_ok(wb.s,    s0),  "walker round trip: s differs");
    CHECK(vec_roundtrip_ok(wb.t,    t0),  "walker round trip: t differs");
    CHECK(vec_roundtrip_ok(wb.logp, lp0), "walker round trip: logp differs");
    CHECK(wb.valid == v0,                 "walker round trip: valid differs");

    // s and t are exactly +/-1 in every mode: they are quantum numbers, and a
    // narrowing that perturbed them would be a real bug even in FP32.
    CHECK(wb.s == s0, "walker round trip: spins must survive EXACTLY in any precision");
    CHECK(wb.t == t0, "walker round trip: isospins must survive EXACTLY in any precision");
}

// --- 4. flat parameter layout ---------------------------------------------
static void test_params_flat(const Ansatz& a) {
    const std::size_t P = a.n_params();

    CHECK(check_param_layout(a) == (std::size_t)-1,
          "copy_params_flat disagrees with get_param");

    DeviceState ds(a, /*verbose=*/false);
    PinnedArray staging;
    ds.upload_params(a, staging);

    std::vector<real> raw(P, (real)0);
    ds.params.down(raw.data(), P);
    std::vector<double> back(P, 0.0);
    convert_copy(back.data(), raw.data(), P);

    for (std::size_t k = 0; k < P; k++)
        CHECK(real_roundtrip_ok(back[k], a.get_param(k)),
              "params round trip differs at index " + std::to_string(k));

    // Independently re-derive the boundaries, so a layout change that moved
    // alpha or swapped two networks would be caught even if get_param moved with it.
    const std::size_t n_h = a.h_net.params.size(), n_rho = a.rho_net.params.size(),
                      n_orb = a.orb_net.params.size();
    CHECK(n_h + n_rho + n_orb + 1 == P, "network sizes do not sum to n_params");
    for (std::size_t i = 0; i < n_h; i++)
        CHECK(real_roundtrip_ok(back[i], a.h_net.params[i]), "flat layout: h_net block wrong");
    for (std::size_t i = 0; i < n_rho; i++)
        CHECK(real_roundtrip_ok(back[n_h + i], a.rho_net.params[i]), "flat layout: rho_net block wrong");
    for (std::size_t i = 0; i < n_orb; i++)
        CHECK(real_roundtrip_ok(back[n_h + n_rho + i], a.orb_net.params[i]), "flat layout: orb_net block wrong");
    CHECK(real_roundtrip_ok(back[P-1], a.alpha), "flat layout: alpha must be the trailing scalar");
}

// --- 5. Philox: device output must equal the host stream, bit for bit -------
//
// This is the test the whole __host__ __device__ arrangement exists for. The
// CPU walker batch guarantees walker w's stream depends only on (seed, w, draw
// index); this shows the device honours the identical guarantee with the
// identical bits, so a Phase 3 GPU sampler can be diffed against the CPU
// reference rather than only compared statistically.
static void test_philox_device_matches_host() {
    const int B = 1000, draws = 1000;              // 1e6 doubles
    const std::size_t n = (std::size_t)B * draws;

    DeviceArray<double> dout;            dout.alloc(n);
    DeviceArray<unsigned long long> dctr; dctr.alloc(B); dctr.zero();

    philox_fill_u01(dout, B, draws, dctr);

    std::vector<double> got(n);
    dout.down(got.data(), n);

    // The same computation on the host, through the same PhiloxStream code.
    std::size_t bad = 0;
    for (int w = 0; w < B; w++) {
        PhiloxStream st = stream_for(rng_seed, w, 0ull);
        for (int i = 0; i < draws; i++) {
            const double want = st.u01();
            if (got[(std::size_t)w * draws + i] != want) bad++;
        }
    }
    CHECK(bad == 0, "philox CPU/GPU bit-identity: " + std::to_string(bad)
                    + " of " + std::to_string(n) + " draws differ");

    // Counters must have advanced by exactly one tick per draw.
    std::vector<unsigned long long> ctr(B);
    dctr.down(ctr.data(), B);
    bool ok = true;
    for (int w = 0; w < B; w++) if (ctr[w] != (unsigned long long)draws) ok = false;
    CHECK(ok, "philox_fill_u01 did not advance rng_ctr by draws_per_walker");
}

// --- 6. Philox: counter persistence across kernel launches -----------------
//
// Two fills of 100 must equal one fill of 200. This is what licenses the
// kernel-side pattern of loading the counter once, advancing locally, and
// writing back once -- if a kernel boundary perturbed the stream, every phase
// transition would silently change the sampling.
static void test_philox_counter_persistence() {
    const int B = 64, half = 100;

    DeviceArray<double> a, b, whole;
    a.alloc((std::size_t)B*half); b.alloc((std::size_t)B*half);
    whole.alloc((std::size_t)B*2*half);
    DeviceArray<unsigned long long> c1, c2;
    c1.alloc(B); c1.zero();
    c2.alloc(B); c2.zero();

    philox_fill_u01(a, B, half, c1);       // ctr 0   -> 100
    philox_fill_u01(b, B, half, c1);       // ctr 100 -> 200, same counter array
    philox_fill_u01(whole, B, 2*half, c2); // ctr 0   -> 200

    std::vector<double> ha((std::size_t)B*half), hb((std::size_t)B*half),
                        hw((std::size_t)B*2*half);
    a.down(ha.data(), ha.size());
    b.down(hb.data(), hb.size());
    whole.down(hw.data(), hw.size());

    std::size_t bad = 0;
    for (int w = 0; w < B; w++)
        for (int i = 0; i < half; i++) {
            if (ha[(std::size_t)w*half + i] != hw[(std::size_t)w*2*half + i])        bad++;
            if (hb[(std::size_t)w*half + i] != hw[(std::size_t)w*2*half + half + i]) bad++;
        }
    CHECK(bad == 0, "philox counter persistence: 100+100 differs from 200 in "
                    + std::to_string(bad) + " places");
}

// ===========================================================================
// Phase 3: batched network forward passes
// ===========================================================================
//
// TOLERANCES. The device is NOT bit-identical to the CPU here, and cannot be:
//
//   * the CPU seeds each accumulator with the bias and adds products onto it,
//     while the GEMM produces products first and bias_act adds the bias after;
//   * cuBLAS reduces the k dimension in its own order, not the CPU's
//     sequential one;
//   * device tanh differs from glibc's by an ulp or so.
//
// None of that is avoidable without giving up cuBLAS, so these tests use
// relative tolerances rather than ==. xi_reduce is the exception: it sums over
// p sequentially in the same order as psi_impl, so it IS bit-comparable, and
// is checked at 1e-13 to leave room only for its input differing.

static double rel_diff(double got, double want) {
    return std::fabs(got - want) / std::max(1.0, std::fabs(want));
}

// --- 7. gemm_rowmajor against a naive host triple loop ---------------------
static void test_gemm_rowmajor(cublasHandle_t handle) {
    std::mt19937_64 rng(20260903);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int shapes[][3] = {{1,1,1},{7,3,5},{256,64,64},{1024,61,31},{4096,64,64},{4096,5,186}};

    double worst = 0.0;
    for (const auto& sh : shapes) {
        const int rows = sh[0], in_w = sh[1], out_w = sh[2];
        std::vector<double> In((std::size_t)rows*in_w), W((std::size_t)out_w*in_w),
                            got((std::size_t)rows*out_w), ref((std::size_t)rows*out_w, 0.0);
        for (auto& v : In) v = u(rng);
        for (auto& v : W)  v = u(rng);

        // Out[r][o] = sum_i In[r][i] * W[o][i]   -- W is [out][in], see layouts.h
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < out_w; o++) {
                double acc = 0.0;
                for (int i = 0; i < in_w; i++)
                    acc += In[(std::size_t)r*in_w + i] * W[(std::size_t)o*in_w + i];
                ref[(std::size_t)r*out_w + o] = acc;
            }

        DeviceArray<real> dI, dW, dO;
        dI.alloc(In.size()); dW.alloc(W.size()); dO.alloc(got.size());
        std::vector<real> tI(In.begin(), In.end()), tW(W.begin(), W.end());
        dI.up(tI.data(), tI.size()); dW.up(tW.data(), tW.size());

        gemm_rowmajor(handle, rows, in_w, out_w, dI.d, dW.d, dO.d);
        cuda_sync_check("test_gemm_rowmajor");

        std::vector<real> tO(got.size());
        dO.down(tO.data(), tO.size());
        for (std::size_t k = 0; k < got.size(); k++)
            worst = std::max(worst, rel_diff((double)tO[k], ref[k]));
    }
    const double tol = real_is_double ? 1e-13 : 1e-5;
    CHECK(worst <= tol, "gemm_rowmajor: worst relative difference "
                        + std::to_string(worst) + " exceeds tolerance");
    std::printf("  gemm_rowmajor worst rel diff: %.3e\n", worst);
}

// --- 8. bias_act against the CPU apply_activation --------------------------
static void test_activation_parity() {
    const int rows = 1000, width = 1000;      // 1e6 values
    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<double> u(-6.0, 6.0);   // Gelu's interesting range

    std::vector<double> z((std::size_t)rows*width), bias(width);
    for (auto& v : z) v = u(rng);
    for (auto& v : bias) v = u(rng);

    for (Activation act : {Activation::Gelu, Activation::Tanh}) {
        std::vector<real> tz(z.begin(), z.end()), tb(bias.begin(), bias.end());
        DeviceArray<real> dz, db;
        dz.alloc(tz.size()); dz.up(tz.data(), tz.size());
        db.alloc(tb.size()); db.up(tb.data(), tb.size());

        bias_act(dz.d, db.d, rows, width, act, /*is_output=*/false);
        std::vector<real> got(tz.size());
        dz.down(got.data(), got.size());

        double worst = 0.0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < width; c++) {
                const std::size_t k = (std::size_t)r*width + c;
                const double want = apply_activation(act, z[k] + bias[c]);
                worst = std::max(worst, rel_diff((double)got[k], want));
            }
        const double tol = real_is_double ? 1e-13 : 1e-5;
        CHECK(worst <= tol, std::string("bias_act parity (")
                            + (act == Activation::Gelu ? "Gelu" : "Tanh")
                            + "): worst rel diff " + std::to_string(worst));
        std::printf("  bias_act %-4s worst rel diff: %.3e\n",
                    act == Activation::Gelu ? "Gelu" : "Tanh", worst);
    }
}

// --- 9. full per-net oracle against psi_impl's double path ------------------
static void test_net_oracle(const Ansatz& a, cublasHandle_t handle) {
    const int B = 512;
    std::mt19937_64 rng(90210);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);

    std::vector<double> hx((std::size_t)B*D), hs((std::size_t)B*N), ht((std::size_t)B*N);
    for (auto& v : hx) v = dist(rng);
    for (int w = 0; w < B; w++)
        for (int i = 0; i < N; i++) {
            hs[(std::size_t)w*N+i] = (i < N_u) ? 1.0 : -1.0;
            ht[(std::size_t)w*N+i] = (i < N_p) ? 1.0 : -1.0;
        }

    DeviceState ds(a, /*verbose=*/false);
    ds.grow_phase3(a, /*verbose=*/true);
    PinnedArray staging;
    ds.upload_params(a, staging);

    std::vector<real> tx(hx.begin(), hx.end()), tsp(hs.begin(), hs.end()), tt(ht.begin(), ht.end());
    ds.x.up(tx.data(), tx.size());
    ds.s.up(tsp.data(), tsp.size());
    ds.t.up(tt.data(), tt.size());

    shift_to_com(ds.x.d, ds.x_sh.d, B);
    build_feat(ds.x_sh.d, ds.s.d, ds.t.d, ds.feat_in.d, B);
    net_forward(handle, ds.h_net_d, ds.params.d, ds.feat_in.d, B*N,
                ds.act_a.d, ds.act_b.d, ds.h_out.d);
    xi_reduce(ds.h_out.d, ds.xi.d, B);
    net_forward(handle, ds.rho_net_d, ds.params.d, ds.xi.d, B,
                ds.act_a.d, ds.act_b.d, ds.rho_out.d);
    net_forward(handle, ds.orb_net_d, ds.params.d, ds.feat_in.d, B*N,
                ds.act_a.d, ds.act_b.d, ds.orb_out.d);

    std::vector<real> g_xsh((std::size_t)B*D), g_h((std::size_t)B*N*m_feat),
                      g_xi((std::size_t)B*m_feat), g_rho((std::size_t)B*K),
                      g_orb((std::size_t)B*N*K*N);
    ds.x_sh.down(g_xsh.data(), g_xsh.size());
    ds.h_out.down(g_h.data(), g_h.size());
    ds.xi.down(g_xi.data(), g_xi.size());
    ds.rho_out.down(g_rho.data(), g_rho.size());
    ds.orb_out.down(g_orb.data(), g_orb.size());

    const double tol = tol::ff(real_is_double ? 1e-12 : 1e-4, 1e-3);
    double w_xsh = 0, w_h = 0, w_xi = 0, w_rho = 0, w_orb = 0;

    std::vector<double> ba, bb, single(dim + 2), cpu_xi(m_feat);
    for (int w = 0; w < B; w++) {
        double Rcm[dim] = {};
        for (int d = 0; d < dim; d++) {
            double R = 0.0;
            for (int i = 0; i < N; i++) R += hx[(std::size_t)w*D + i*dim + d];
            Rcm[d] = R / N;
        }
        for (int p = 0; p < N; p++)
            for (int d = 0; d < dim; d++) {
                const double want = hx[(std::size_t)w*D + p*dim + d] - Rcm[d];
                w_xsh = std::max(w_xsh, rel_diff((double)g_xsh[(std::size_t)w*D + p*dim + d], want));
            }

        std::fill(cpu_xi.begin(), cpu_xi.end(), 0.0);
        for (int p = 0; p < N; p++) {
            for (int d = 0; d < dim; d++) single[d] = hx[(std::size_t)w*D + p*dim + d] - Rcm[d];
            single[dim]     = hs[(std::size_t)w*N + p];
            single[dim + 1] = ht[(std::size_t)w*N + p];

            const std::vector<double>& href = *a.h_net.forward_opt<double>(single, a.h_net.params, ba, bb);
            for (int f = 0; f < m_feat; f++) {
                w_h = std::max(w_h, rel_diff((double)g_h[(std::size_t)(w*N+p)*m_feat + f], href[f]));
                cpu_xi[f] += href[f];
            }
            const std::vector<double>& oref = *a.orb_net.forward_opt<double>(single, a.orb_net.params, ba, bb);
            for (int j = 0; j < K*N; j++)
                w_orb = std::max(w_orb, rel_diff((double)g_orb[(std::size_t)(w*N+p)*(K*N) + j], oref[j]));
        }
        for (int f = 0; f < m_feat; f++)
            w_xi = std::max(w_xi, rel_diff((double)g_xi[(std::size_t)w*m_feat + f], cpu_xi[f]));

        // rho is fed the DEVICE xi, so compare against the CPU rho of the CPU
        // xi: any disagreement here is rho's GEMM, not accumulated xi error.
        const std::vector<double>& rref = *a.rho_net.forward_opt<double>(cpu_xi, a.rho_net.params, ba, bb);
        for (int k = 0; k < K; k++)
            w_rho = std::max(w_rho, rel_diff((double)g_rho[(std::size_t)w*K + k], rref[k]));
    }

    std::printf("  oracle worst rel diff:  x_sh %.2e  h %.2e  xi %.2e  rho %.2e  orb %.2e\n",
                w_xsh, w_h, w_xi, w_rho, w_orb);
    CHECK(w_xsh <= (real_is_double ? 1e-15 : 1e-6), "shift_to_com disagrees with CPU");
    CHECK(w_h   <= tol, "h_net forward disagrees with CPU forward_opt<double>");
    CHECK(w_xi  <= tol::ff(real_is_double ? 1e-13 : 1e-5, 1e-4), "xi_reduce disagrees with CPU accumulation");
    CHECK(w_rho <= tol, "rho_net forward disagrees with CPU forward_opt<double>");
    CHECK(w_orb <= tol, "orb_net forward disagrees with CPU forward_opt<double>");
}

// --- 10. determinism -------------------------------------------------------
// Two identical invocations must give bitwise-identical output. This is a claim
// about THIS machine and THIS cuBLAS configuration only: cuBLAS is deterministic
// run-to-run for a fixed architecture, library version and problem size, but
// nothing guarantees the same bits on a different card or after a library
// upgrade. That is all Phase 3 needs -- the cross-machine guarantee lives in
// the Philox streams, not here.
static void test_net_determinism(const Ansatz& a, cublasHandle_t handle) {
    const int B = 256;
    std::mt19937_64 rng(4711);
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    std::vector<double> hx((std::size_t)B*D), hs((std::size_t)B*N), ht((std::size_t)B*N);
    for (auto& v : hx) v = dist(rng);
    for (int w = 0; w < B; w++)
        for (int i = 0; i < N; i++) {
            hs[(std::size_t)w*N+i] = (i < N_u) ? 1.0 : -1.0;
            ht[(std::size_t)w*N+i] = (i < N_p) ? 1.0 : -1.0;
        }

    DeviceState ds(a, false);
    ds.grow_phase3(a, false);
    PinnedArray staging;
    ds.upload_params(a, staging);
    std::vector<real> tx(hx.begin(), hx.end()), tsp(hs.begin(), hs.end()), tt(ht.begin(), ht.end());
    ds.x.up(tx.data(), tx.size()); ds.s.up(tsp.data(), tsp.size()); ds.t.up(tt.data(), tt.size());

    std::vector<real> first, second;
    for (int pass = 0; pass < 2; pass++) {
        shift_to_com(ds.x.d, ds.x_sh.d, B);
        build_feat(ds.x_sh.d, ds.s.d, ds.t.d, ds.feat_in.d, B);
        net_forward(handle, ds.h_net_d, ds.params.d, ds.feat_in.d, B*N,
                    ds.act_a.d, ds.act_b.d, ds.h_out.d);
        xi_reduce(ds.h_out.d, ds.xi.d, B);
        net_forward(handle, ds.rho_net_d, ds.params.d, ds.xi.d, B,
                    ds.act_a.d, ds.act_b.d, ds.rho_out.d);
        net_forward(handle, ds.orb_net_d, ds.params.d, ds.feat_in.d, B*N,
                    ds.act_a.d, ds.act_b.d, ds.orb_out.d);
        std::vector<real> snap((std::size_t)B*N*K*N);
        ds.orb_out.down(snap.data(), snap.size());
        if (pass == 0) first = std::move(snap); else second = std::move(snap);
    }
    CHECK(first == second, "two identical forward passes gave different bits");
}

// ===========================================================================
// Phase 3.2: determinants, S, log|psi|, and the (s,t) table
// ===========================================================================

// Fill a walker batch with random positions in the standard sector, optionally
// collapsing particle 1 onto particle 0 to drive determinants toward zero.
static void make_configs(int B, std::vector<double>& hx, std::vector<double>& hs,
                         std::vector<double>& ht, std::mt19937_64& rng,
                         int near_node_every = 0) {
    std::uniform_real_distribution<double> dist(-x_init_range, x_init_range);
    hx.assign((std::size_t)B*D, 0.0); hs.assign((std::size_t)B*N, 0.0); ht.assign((std::size_t)B*N, 0.0);
    for (auto& v : hx) v = dist(rng);
    for (int w = 0; w < B; w++)
        for (int i = 0; i < N; i++) {
            hs[(std::size_t)w*N+i] = (i < N_u) ? 1.0 : -1.0;
            ht[(std::size_t)w*N+i] = (i < N_p) ? 1.0 : -1.0;
        }
    // Particles 0 and 1 share (s,t) in the standard sector, so putting them at
    // the same point makes two columns of every Slater matrix identical.
    if (near_node_every > 0)
        for (int w = 0; w < B; w += near_node_every)
            for (int d = 0; d < dim; d++)
                hx[(std::size_t)w*D + 1*dim + d] = hx[(std::size_t)w*D + 0*dim + d];
}

static void upload_configs(DeviceState& ds, const std::vector<double>& hx,
                           const std::vector<double>& hs, const std::vector<double>& ht) {
    std::vector<real> tx(hx.begin(),hx.end()), ts(hs.begin(),hs.end()), tt(ht.begin(),ht.end());
    ds.x.up(tx.data(), tx.size()); ds.s.up(ts.data(), ts.size()); ds.t.up(tt.data(), tt.size());
}

// --- 11. full psi oracle ---------------------------------------------------
//
// Compared as an ABSOLUTE difference on log|psi|, not a relative one. That is
// the correct measure, not a concession: an absolute error in a log IS a
// relative error in psi, whereas a relative tolerance on the log would blow up
// wherever |psi| passes through 1 and log|psi| passes through 0.
//
// Ordinary configurations only. Near-node configurations are NOT an accuracy
// test and are handled in test_logpsi_nodes below -- see the note there.
static void test_logpsi_oracle(const Ansatz& a, cublasHandle_t handle) {
    const int B = 512;
    std::mt19937_64 rng(777);
    std::vector<double> hx, hs, ht;
    make_configs(B, hx, hs, ht, rng);

    DeviceState ds(a, false); ds.grow_phase3(a, false);
    PinnedArray st; ds.upload_params(a, st);
    upload_configs(ds, hx, hs, ht);

    eval_logp_batch(ds, handle, B);
    std::vector<real> g_logp(B);
    ds.logp.down(g_logp.data(), B);

    Workspace ws;
    double worst = 0.0;
    int n_finite = 0;
    for (int w = 0; w < B; w++) {
        const double want = log_p(&hx[(std::size_t)w*D], &hs[(std::size_t)w*N],
                                  &ht[(std::size_t)w*N], a, ws);
        const double got = (double)g_logp[w];
        CHECK(!std::isnan(got), "log|psi| is NaN at walker " + std::to_string(w));
        if (std::isfinite(got) && std::isfinite(want)) {
            worst = std::max(worst, std::fabs(got - want));
            n_finite++;
        }
    }
    std::printf("  log|psi| oracle (%d configs): worst abs diff %.3e\n", n_finite, worst);
    CHECK(n_finite >= B - 2, "too many non-finite log|psi| on ordinary configs");
    CHECK(worst <= tol::ff(real_is_double ? 1e-11 : 1e-3, 2e-3),
          "log|psi| worst absolute difference " + std::to_string(worst));
}

// --- 11b. the node path ----------------------------------------------------
//
// Near-node configurations are a ROBUSTNESS test, not an accuracy one, and the
// distinction is worth stating because the obvious version of this test is
// wrong. Collapsing two same-(s,t) particles onto each other makes two columns
// of every Slater matrix identical, so every det is mathematically zero -- but
// LU with partial pivoting leaves a last pivot around 1e-17 rather than exactly
// 0, so lu_det's 1e-300 guard does not fire. (The same trap cost a debugging
// cycle in test_detjet.) What actually happens is that S = sum_k rho_k det_k
// collapses to ~1e-11 through roughly twenty digits of cancellation between
// terms of order 1e5..1e9. At that point CPU and GPU disagree by factors of
// tens and NEITHER is meaningful -- measured: |S| 4.0e-10 vs 1.2e-11.
//
// So this checks what can honestly be checked: nothing is NaN, node
// configurations are driven to overwhelmingly improbable log|psi|, and the
// exact -INFINITY branch works when S really is zero -- which is tested by
// injecting S = 0 directly rather than hoping a physical configuration lands
// there.
static void test_logpsi_nodes(const Ansatz& a, cublasHandle_t handle) {
    const int B = 512, every = 37;
    std::mt19937_64 rng(778);
    std::vector<double> hx, hs, ht;
    make_configs(B, hx, hs, ht, rng, every);

    DeviceState ds(a, false); ds.grow_phase3(a, false);
    PinnedArray st; ds.upload_params(a, st);
    upload_configs(ds, hx, hs, ht);
    eval_logp_batch(ds, handle, B);

    std::vector<real> g_logp(B), g_S(B);
    ds.logp.down(g_logp.data(), B);
    ds.S.down(g_S.data(), B);

    std::vector<double> normal;
    for (int w = 0; w < B; w++) {
        CHECK(!std::isnan((double)g_logp[w]), "node test: NaN log|psi| at walker " + std::to_string(w));
        CHECK(!std::isnan((double)g_S[w]),    "node test: NaN S at walker " + std::to_string(w));
        if (w % every != 0 && std::isfinite((double)g_logp[w])) normal.push_back((double)g_logp[w]);
    }
    std::sort(normal.begin(), normal.end());
    const double typical = normal[normal.size()/2];

    int suppressed = 0, total_nodes = 0;
    for (int w = 0; w < B; w += every) {
        total_nodes++;
        const double lp = (double)g_logp[w];
        // -inf, or log|psi| at least e^10 below typical: either way the sampler
        // will essentially never accept it, which is the property that matters.
        if (!std::isfinite(lp) || lp < typical - 10.0) suppressed++;
    }
    std::printf("  node configs: %d of %d driven below typical log|psi| (%.2f)\n",
                suppressed, total_nodes, typical);
    CHECK(suppressed == total_nodes,
          "a forced-node configuration was not suppressed: " + std::to_string(suppressed)
          + " of " + std::to_string(total_nodes));

    // The exact -INFINITY branch, tested directly: inject S = 0.
    DeviceArray<real> zeroS;
    zeroS.alloc(B); zeroS.zero();
    envelope_logp(ds.x_sh.d, zeroS.d, ds.params.d, ds.P, ds.logp.d, B);
    std::vector<real> g2(B); ds.logp.down(g2.data(), B);
    int not_inf = 0;
    for (int w = 0; w < B; w++)
        if (!(std::isinf((double)g2[w]) && (double)g2[w] < 0)) not_inf++;
    CHECK(not_inf == 0, "S = 0 did not give -INFINITY for " + std::to_string(not_inf) + " walkers");
}

// --- 12. determinants in isolation -----------------------------------------
static void test_dets_isolated(const Ansatz& a, cublasHandle_t handle) {
    const int B = 256;
    std::mt19937_64 rng(31415);
    std::vector<double> hx, hs, ht;
    make_configs(B, hx, hs, ht, rng);

    DeviceState ds(a, false); ds.grow_phase3(a, false);
    PinnedArray st; ds.upload_params(a, st);
    upload_configs(ds, hx, hs, ht);

    // Run the chain up to assembly, then snapshot M BEFORE getrf overwrites it
    // in place with the LU factors -- otherwise the CPU oracle would be handed
    // the factorisation instead of the matrix.
    eval_logp_batch(ds, handle, B);          // warms everything
    shift_to_com(ds.x.d, ds.x_sh.d, B);
    build_feat(ds.x_sh.d, ds.s.d, ds.t.d, ds.feat_in.d, B);
    net_forward(handle, ds.orb_net_d, ds.params.d, ds.feat_in.d, B*N,
                ds.act_a.d, ds.act_b.d, ds.orb_out.d);
    assemble_M(ds.orb_out.d, ds.M_batch.d, B);

    std::vector<real> M_before((std::size_t)B*K*N*N);
    ds.M_batch.down(M_before.data(), M_before.size());

    batched_det(handle, B*K, ds.M_batch.d, ds.lu_ptrs.d, ds.lu_piv.d, ds.lu_info.d, ds.dets.d);
    std::vector<real> g_dets((std::size_t)B*K);
    ds.dets.down(g_dets.data(), g_dets.size());

    std::vector<double> Mi((std::size_t)N*N);
    std::vector<int> piv;
    std::vector<double> rd;
    rd.reserve((std::size_t)B*K);
    for (std::size_t m = 0; m < (std::size_t)B*K; m++) {
        for (int e = 0; e < N*N; e++) Mi[e] = (double)M_before[m*N*N + e];
        const double want = lu_det<double>(Mi, N, piv);   // destroys Mi, hence the copy
        rd.push_back(rel_diff((double)g_dets[m], want));
    }
    std::sort(rd.begin(), rd.end());
    const double med = rd[rd.size()/2];
    const double p999 = rd[(std::size_t)(rd.size()*0.999)];
    const double mx = rd.back();
    std::printf("  batched_det vs lu_det over %zu matrices: median %.2e  p99.9 %.2e  max %.2e\n",
                rd.size(), med, p999, mx);

    // A determinant is a PRODUCT of N pivots, and cuBLAS's partial pivoting
    // does not choose the same pivots as lu_det, so the two are different
    // computations whose relative errors grow with the matrix conditioning.
    // Measured here: median 4.6e-16, p99.9 5.5e-13, max 3.9e-12 -- a
    // conditioning tail, not disagreement. Asserting a single tight bound on
    // the MAX would be asserting that no matrix in 7936 is ill-conditioned,
    // which is a claim about the ansatz, not about this kernel.
    //
    // The bounds below still catch every realistic bug: a wrong index mapping
    // gives O(1) relative error and a pivot-parity bug gives ~2, either of
    // which moves the MEDIAN, not just the tail.
    if (real_is_double) {
        CHECK(med  <= 1e-14, "batched_det: median rel diff " + std::to_string(med));
        CHECK(p999 <= 1e-12, "batched_det: p99.9 rel diff " + std::to_string(p999));
        CHECK(mx   <= 1e-10, "batched_det: max rel diff " + std::to_string(mx));
    } else {
        CHECK(mx <= 1e-4, "batched_det: max rel diff " + std::to_string(mx));
    }
}

// --- 12b. Slater assembly orientation --------------------------------------
//
// det(M^T) = det(M), so a transposed assembly yields an identical wavefunction
// and every determinant, S and log|psi| test passes regardless -- confirmed by
// deliberately swapping k and i, which left the whole suite green. The
// orientation only bites in Phase 4, where M^-1 enters the O assembly and
// (M^T)^-1 = (M^-1)^T is a different matrix.
//
// So it is checked directly against the CPU's own buffer: psi() fills ws.dM
// with exactly buf.M[j*(N*N) + k*N + i], which is the quantity M_batch is
// supposed to reproduce. Element-by-element, no determinant in between.
static void test_assembly_orientation(const Ansatz& a, cublasHandle_t handle) {
    const int B = 64;
    std::mt19937_64 rng(13131);
    std::vector<double> hx, hs, ht;
    make_configs(B, hx, hs, ht, rng);

    DeviceState ds(a, false); ds.grow_phase3(a, false);
    PinnedArray st; ds.upload_params(a, st);
    upload_configs(ds, hx, hs, ht);

    shift_to_com(ds.x.d, ds.x_sh.d, B);
    build_feat(ds.x_sh.d, ds.s.d, ds.t.d, ds.feat_in.d, B);
    net_forward(handle, ds.orb_net_d, ds.params.d, ds.feat_in.d, B*N,
                ds.act_a.d, ds.act_b.d, ds.orb_out.d);
    assemble_M(ds.orb_out.d, ds.M_batch.d, B);

    std::vector<real> gM((std::size_t)B*K*N*N);
    ds.M_batch.down(gM.data(), gM.size());

    Workspace ws;
    double worst = 0.0;
    std::size_t asym = 0;
    for (int w = 0; w < B; w++) {
        psi(&hx[(std::size_t)w*D], &hs[(std::size_t)w*N], &ht[(std::size_t)w*N], a, ws);
        for (std::size_t e = 0; e < (std::size_t)K*N*N; e++)
            worst = std::max(worst, rel_diff((double)gM[(std::size_t)w*K*N*N + e], ws.dM[e]));
        // Guard against the comparison being vacuous: if the matrices happened
        // to be symmetric, a transpose would be undetectable here too.
        for (int j = 0; j < K; j++)
            for (int k = 0; k < N; k++)
                for (int i = k+1; i < N; i++)
                    if (ws.dM[(std::size_t)j*N*N + k*N + i] != ws.dM[(std::size_t)j*N*N + i*N + k]) asym++;
    }
    std::printf("  Slater assembly vs CPU ws.dM: worst rel diff %.3e (%zu asymmetric pairs)\n",
                worst, asym);
    CHECK(asym > 0, "Slater matrices are symmetric -- orientation test is vacuous");
    CHECK(worst <= tol::ff(real_is_double ? 1e-13 : 1e-5, 1e-4),
          "assemble_M disagrees with the CPU's buf.M: " + std::to_string(worst));
}

// --- 13. pivot parity ------------------------------------------------------
//
// THE point of this test. cublasDgetrfBatched follows LAPACK: ipiv is 1-BASED,
// so "no swap at step i" is ipiv[i] == i+1. Testing against i instead counts a
// swap at every step and flips the sign of every determinant with an even
// number of real swaps -- |det| stays perfect, so nothing else here would
// notice. Permutation matrices have det = +-1 with the sign fixed by the
// permutation's parity, which pins the convention exactly. Verified: flipping
// the convention fails 236 of these 512.
static void test_pivot_parity(cublasHandle_t handle) {
    std::mt19937_64 rng(2718);
    const int n_mats = 512;
    std::vector<real> M((std::size_t)n_mats*N*N, (real)0);
    std::vector<double> want(n_mats);

    for (int m = 0; m < n_mats; m++) {
        int perm[N];
        for (int i = 0; i < N; i++) perm[i] = i;
        for (int i = N-1; i > 0; i--) std::swap(perm[i], perm[(int)(rng() % (unsigned)(i+1))]);
        // parity by cycle decomposition
        bool seen[N] = {};
        int transpositions = 0;
        for (int i = 0; i < N; i++) {
            if (seen[i]) continue;
            int len = 0, j = i;
            while (!seen[j]) { seen[j] = true; j = perm[j]; len++; }
            transpositions += len - 1;
        }
        want[m] = (transpositions & 1) ? -1.0 : 1.0;
        for (int i = 0; i < N; i++) M[(std::size_t)m*N*N + i*N + perm[i]] = (real)1;
    }

    DeviceArray<real> dM, ddet;
    DeviceArray<int> dpiv, dinfo;
    DeviceArray<double*> dptr;
    dM.alloc(M.size()); dM.up(M.data(), M.size());
    ddet.alloc(n_mats); dpiv.alloc((std::size_t)n_mats*N); dinfo.alloc(n_mats);
    dptr.alloc(n_mats);
    {
        std::vector<double*> h(n_mats);
        for (int m = 0; m < n_mats; m++) h[m] = (double*)(dM.d + (std::size_t)m*N*N);
        dptr.up(h.data(), h.size());
    }

    batched_det(handle, n_mats, dM.d, dptr.d, dpiv.d, dinfo.d, ddet.d);
    std::vector<real> got(n_mats);
    ddet.down(got.data(), n_mats);

    int bad = 0, neg = 0;
    for (int m = 0; m < n_mats; m++) {
        if ((double)got[m] != want[m]) bad++;
        if (want[m] < 0) neg++;
    }
    CHECK(bad == 0, "pivot parity: " + std::to_string(bad) + " of "
                    + std::to_string(n_mats) + " permutation determinants wrong");
    CHECK(neg > 0 && neg < n_mats, "permutation sample lacks both parities -- test weak");

    // A genuinely singular matrix must come back exactly zero, matching lu_det.
    // Built as a matrix with one nonzero COLUMN rather than two equal columns:
    // equal columns leave a last pivot around 1e-17, not exactly 0, and would
    // not trip the guard -- the same trap as in test_detjet.
    std::vector<real> Z((std::size_t)N*N, (real)0);
    for (int i = 0; i < N; i++) Z[i*N + 0] = (real)1;
    DeviceArray<real> zM, zdet; DeviceArray<int> zpiv, zinfo; DeviceArray<double*> zptr;
    zM.alloc(Z.size()); zM.up(Z.data(), Z.size());
    zdet.alloc(1); zpiv.alloc(N); zinfo.alloc(1); zptr.alloc(1);
    { double* h = (double*)zM.d; zptr.up(&h, 1); }
    batched_det(handle, 1, zM.d, zptr.d, zpiv.d, zinfo.d, zdet.d);
    real zg; zdet.down(&zg, 1);
    CHECK((double)zg == 0.0, "singular matrix did not give exactly 0.0");
}

// --- 14. table oracle ------------------------------------------------------
static void test_table_oracle(const Ansatz& a, cublasHandle_t handle) {
    const int B = 512, relabelings = 5;
    std::mt19937_64 rng(90909);
    std::vector<double> hx, hs, ht;
    make_configs(B, hx, hs, ht, rng);

    DeviceState ds(a, false); ds.grow_phase3(a, false);
    PinnedArray st; ds.upload_params(a, st);
    upload_configs(ds, hx, hs, ht);

    build_st_table_batch(ds, handle, B);

    DeviceArray<real> dS; dS.alloc(B);
    Workspace ws;
    std::vector<double> rd;

    for (int rel = 0; rel < relabelings; rel++) {
        // Relabelings that PRESERVE the sector (shuffle within), so the
        // configurations stay physical rather than drifting to another N_u/N_p.
        std::vector<double> rs = hs, rt = ht;
        for (int w = 0; w < B; w++) {
            for (int i = N-1; i > 0; i--) {
                int j = (int)(rng() % (unsigned)(i+1));
                std::swap(rs[(std::size_t)w*N+i], rs[(std::size_t)w*N+j]);
                std::swap(rt[(std::size_t)w*N+i], rt[(std::size_t)w*N+j]);
            }
        }
        std::vector<real> trs(rs.begin(),rs.end()), trt(rt.begin(),rt.end());
        DeviceArray<real> drs, drt;
        drs.alloc(trs.size()); drs.up(trs.data(), trs.size());
        drt.alloc(trt.size()); drt.up(trt.data(), trt.size());

        S_from_table_batch(ds, handle, B, drs.d, drt.d, dS.d);
        std::vector<real> gS(B); dS.down(gS.data(), B);

        for (int w = 0; w < B; w++) {
            build_st_table(&hx[(std::size_t)w*D], a, ws);
            const double want = S_from_table(&rs[(std::size_t)w*N], &rt[(std::size_t)w*N], a, ws);
            rd.push_back(rel_diff((double)gS[w], want));
        }
    }
    std::sort(rd.begin(), rd.end());
    const double med = rd[rd.size()/2], p999 = rd[(std::size_t)(rd.size()*0.999)], mx = rd.back();
    std::printf("  S_from_table oracle (%d configs x %d relabelings): median %.2e  p99.9 %.2e  max %.2e\n",
                B, relabelings, med, p999, mx);
    // S is a sum of K determinants, so it inherits their conditioning tail --
    // same reasoning as test_dets_isolated. A wrong combo selection or a
    // transposed assembly would move the median, not the tail.
    if (fp32_forward) {                // 6.3 ladder (test_tolerances.h)
        CHECK(med  <= 1e-5, "S_from_table_batch: median rel diff " + std::to_string(med));
        CHECK(p999 <= 3e-3, "S_from_table_batch: p99.9 rel diff " + std::to_string(p999));
        CHECK(mx   <= 1e-2, "S_from_table_batch: max rel diff " + std::to_string(mx));
    } else if (real_is_double) {
        CHECK(med  <= 1e-14, "S_from_table_batch: median rel diff " + std::to_string(med));
        CHECK(p999 <= 1e-11, "S_from_table_batch: p99.9 rel diff " + std::to_string(p999));
        CHECK(mx   <= 1e-9,  "S_from_table_batch: max rel diff " + std::to_string(mx));
    } else {
        CHECK(mx <= 1e-4, "S_from_table_batch: max rel diff " + std::to_string(mx));
    }
}

// ===========================================================================
// Phase 3.3: the device sampler
// ===========================================================================

static void seed_walkers(WalkerBatch& wb, const Ansatz& a, int B,
                         ThreadPool& pool, std::vector<Workspace>& wss) {
    wb.init(B);
    init_batch(wb, a, &pool, wss);
}

// --- 15. draw-order determinism --------------------------------------------
//
// Two identical device runs must agree bit for bit after 50 sweeps. This is
// the property the whole Philox design exists for: walker w's stream depends
// only on (seed, w, counter), never on block scheduling, so the same seed gives
// the same trajectory regardless of how the GPU happens to order the warps.
static void test_sampler_determinism(const Ansatz& a, cublasHandle_t handle) {
    const int B = 512, sweeps = 50;
    ThreadPool pool(4);
    std::vector<Workspace> wss(4);
    WalkerBatch wb0;
    seed_walkers(wb0, a, B, pool, wss);

    std::vector<real> x[2], sp[2], tt[2], lp[2];
    std::vector<long long> ctr[2];

    for (int run = 0; run < 2; run++) {
        WalkerBatch wb = wb0;                       // identical starting state
        DeviceState ds(a, false); ds.grow_phase3(a, false); ds.grow_phase33(false);
        PinnedArray st; ds.upload_params(a, st);
        upload_and_reset(ds, wb, st);
        // logp must be recomputed on the device: upload_walkers brings the CPU's
        // value across, but the device's log|psi| differs from it in the last
        // bits (different summation order), and the Metropolis test compares
        // logp against logp_prop -- both must come from the same evaluator.
        eval_logp_batch(ds, handle, B);

        therm_batch_device(ds, handle, B, step0, sweeps);

        x[run].resize((std::size_t)B*D);   ds.x.down(x[run].data(), x[run].size());
        sp[run].resize((std::size_t)B*N);  ds.s.down(sp[run].data(), sp[run].size());
        tt[run].resize((std::size_t)B*N);  ds.t.down(tt[run].data(), tt[run].size());
        lp[run].resize(B);                 ds.logp.down(lp[run].data(), B);
        std::vector<unsigned long long> c(B); ds.rng_ctr.down(c.data(), B);
        ctr[run].assign(c.begin(), c.end());
    }

    CHECK(x[0]  == x[1],  "sampler determinism: x differs between identical runs");
    CHECK(sp[0] == sp[1], "sampler determinism: s differs between identical runs");
    CHECK(tt[0] == tt[1], "sampler determinism: t differs between identical runs");
    CHECK(lp[0] == lp[1], "sampler determinism: logp differs between identical runs");
    CHECK(ctr[0] == ctr[1], "sampler determinism: rng counters differ");

    // Guard against passing vacuously: the sampler must actually have moved.
    bool moved = false;
    for (std::size_t i = 0; i < x[0].size(); i++)
        if ((double)x[0][i] != wb0.x[i]) { moved = true; break; }
    CHECK(moved, "sampler determinism: walkers never moved -- test vacuous");

    // Counter advance is DATA-DEPENDENT (the accept draw is skipped when
    // logp_old is -inf), so counters need not be equal ACROSS walkers -- only
    // across runs. Check they advanced at all.
    bool advanced = true;
    for (int w = 0; w < B; w++) if (ctr[0][w] == 0) advanced = false;
    CHECK(advanced, "sampler determinism: some walker consumed no draws");
}

// --- 16. CPU/GPU trajectory replay -----------------------------------------
//
// The strongest sampler test available here. Rather than writing a Philox shim
// into the CPU sampler (which would mean maintaining a second sampler inside
// the reference one), the device logs every (idx, displacement, accepted) tuple
// and the host replays them through its own commit logic.
//
// The counter is logged alongside each tuple because its advance is
// data-dependent -- see the contract note in sampler_kernels.cu.
//
// LIMITATION, stated plainly: this replays the DEVICE's own accept decisions,
// so it cannot detect a divergence in the accept RULE itself -- in particular
// it would not notice if the device consumed the acceptance draw
// unconditionally instead of skipping it when logp_old is -inf. Only a genuine
// CPU-vs-GPU bit comparison closes that, and the cheap version is to seed
// walkers onto nodes and assert the counter advanced by 2 rather than 3.
static void test_sampler_replay(const Ansatz& a, cublasHandle_t handle) {
    const int B = 64, sweeps = 5;
    ThreadPool pool(2);
    std::vector<Workspace> wss(2);
    WalkerBatch wb;
    seed_walkers(wb, a, B, pool, wss);

    DeviceState ds(a, false); ds.grow_phase3(a, false); ds.grow_phase33(false);
    PinnedArray st; ds.upload_params(a, st);
    upload_and_reset(ds, wb, st);
    eval_logp_batch(ds, handle, B);

    std::vector<double> hx((std::size_t)B*D);
    { std::vector<real> t((std::size_t)B*D); ds.x.down(t.data(), t.size());
      for (std::size_t i = 0; i < t.size(); i++) hx[i] = (double)t[i]; }

    for (int sweep = 0; sweep < sweeps; sweep++) {
        for (int j = 0; j < draws; j++) {
            propose_coord(ds, B, step0);
            std::vector<int> idx(B);   ds.prop_idx.down(idx.data(), B);
            std::vector<real> xp((std::size_t)B*D); ds.x_prop.down(xp.data(), xp.size());

            // The proposal must differ from the current state in EXACTLY the
            // one logged coordinate. A commit-path bug that perturbed the wrong
            // entry, or more than one, shows up here and nowhere else.
            for (int w = 0; w < B; w++)
                for (int d = 0; d < D; d++) {
                    const double got = (double)xp[(std::size_t)w*D + d];
                    if (d == idx[w]) continue;
                    CHECK(got == hx[(std::size_t)w*D + d],
                          "replay: proposal changed coordinate " + std::to_string(d)
                          + " but claimed index " + std::to_string(idx[w]));
                }

            eval_logp_batch_prop(ds, handle, B, ds.x_prop.d, ds.S_prop.d, ds.logp_prop.d);
            std::vector<long long> before(B); ds.acc.down(before.data(), B);
            accept_coord(ds, B);
            std::vector<long long> after(B);  ds.acc.down(after.data(), B);

            for (int w = 0; w < B; w++)
                if (after[w] != before[w])
                    hx[(std::size_t)w*D + idx[w]] = (double)xp[(std::size_t)w*D + idx[w]];
        }
        recenter_device(ds, B);
        for (int w = 0; w < B; w++) {
            double R[dim] = {};
            for (int d = 0; d < dim; d++) {
                double acc = 0.0;
                for (int i = 0; i < N; i++) acc += hx[(std::size_t)w*D + i*dim + d];
                R[d] = acc / N;
            }
            for (int i = 0; i < N; i++)
                for (int d = 0; d < dim; d++) hx[(std::size_t)w*D + i*dim + d] -= R[d];
        }
    }

    std::vector<real> gx((std::size_t)B*D); ds.x.down(gx.data(), gx.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < gx.size(); i++) if ((double)gx[i] != hx[i]) bad++;
    CHECK(bad == 0, "replay: host reconstruction differs from device x in "
                    + std::to_string(bad) + " of " + std::to_string(gx.size()) + " entries");
}

// --- 17. logp cache integrity ----------------------------------------------
//
// After many sweeps the cached logp must still equal a fresh evaluation of the
// committed state. Drift here means x and logp have got out of step, which
// corrupts every subsequent Metropolis test while leaving the acceptance rate
// looking entirely reasonable -- verified by negative control: removing the
// logp commit gives drift 3.296 and an acceptance rate of 0.460, which is a
// perfectly healthy-looking number.
static void test_logp_cache_integrity(const Ansatz& a, cublasHandle_t handle) {
    const int B = 256, sweeps = 100;
    ThreadPool pool(4);
    std::vector<Workspace> wss(4);
    WalkerBatch wb;
    seed_walkers(wb, a, B, pool, wss);

    DeviceState ds(a, false); ds.grow_phase3(a, false); ds.grow_phase33(false);
    PinnedArray st; ds.upload_params(a, st);
    upload_and_reset(ds, wb, st);
    eval_logp_batch(ds, handle, B);
    therm_batch_device(ds, handle, B, step0, sweeps);

    ds.download_walkers(wb, st);
    std::vector<real> g_logp(B); ds.logp.down(g_logp.data(), B);

    Workspace ws;
    double worst = 0.0;
    int checked = 0;
    for (int w = 0; w < B; w++) {
        const double fresh = log_p(&wb.x[(std::size_t)w*D], &wb.s[(std::size_t)w*N],
                                   &wb.t[(std::size_t)w*N], a, ws);
        const double cached = (double)g_logp[w];
        if (!std::isfinite(fresh) || !std::isfinite(cached)) continue;
        worst = std::max(worst, std::fabs(fresh - cached));
        checked++;
    }
    std::printf("  logp cache after %d sweeps: worst abs drift %.3e over %d walkers\n",
                sweeps, worst, checked);
    CHECK(checked > B/2, "logp cache: too many non-finite walkers to judge");
    // 1e-9 absolute on a log, i.e. 1e-9 relative on |psi|. The device and CPU
    // evaluators differ in the last bits anyway (GEMM vs sequential sums), so
    // this bounds DRIFT, not agreement -- a commit bug shows as O(1).
    // fp32_forward: the table and full-evaluation paths run differently shaped
    // float GEMMs, so a cached logp and a recomputed one differ at ~1e-6.
    CHECK(worst <= tol::ff(1e-9, 1e-4), "logp cache drifted by " + std::to_string(worst));
}

// --- 18. acceptance sanity --------------------------------------------------
static void test_acceptance_rates(const Ansatz& a, cublasHandle_t handle) {
    const int B = 512, sweeps = 20;
    ThreadPool pool(4);
    std::vector<Workspace> wss(4);
    WalkerBatch wb;
    seed_walkers(wb, a, B, pool, wss);

    DeviceState ds(a, false); ds.grow_phase3(a, false); ds.grow_phase33(false);
    PinnedArray st; ds.upload_params(a, st);
    upload_and_reset(ds, wb, st);
    eval_logp_batch(ds, handle, B);

    const double acc = therm_batch_device(ds, handle, B, step0, sweeps);
    long long a_c, a_s, a_t;
    download_acceptance(ds, B, a_c, a_s, a_t);
    const double sp_rate  = (double)a_s / ((double)B*sweeps*spin_draws);
    const double tau_rate = (double)a_t / ((double)B*sweeps*tau_draws);
    std::printf("  acceptance over %d sweeps: coord %.4f  spin %.4f  tau %.4f\n",
                sweeps, acc, sp_rate, tau_rate);

    // Only a sanity band. A sampler that accepted everything or nothing would
    // still be deterministic and still pass the replay test, so this is the one
    // check that the Metropolis rule is doing anything at all.
    CHECK(acc > 0.01 && acc < 0.99, "coordinate acceptance " + std::to_string(acc)
                                    + " is degenerate");
    CHECK(sp_rate  > 0.0 && sp_rate  < 1.0, "spin acceptance degenerate");
    CHECK(tau_rate > 0.0 && tau_rate < 1.0, "tau acceptance degenerate");
}

int main() {
    try {
        gpu_select_device(/*verbose=*/true);
        std::printf("device: %s\n  sampling precision: real = %s\n", gpu_device_name(), real_name);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: no usable CUDA device -- " << e.what() << "\n";
        return 1;
    }

    Ansatz a({64}, {64}, {64}, Activation::Gelu);
    seed_ansatz(a, 2024);

    try {
        test_smoke();
        test_device_array();
        test_device_state(a);
        test_walker_roundtrip(a);
        test_params_flat(a);
        test_philox_device_matches_host();
        test_philox_counter_persistence();
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
            
        test_gemm_rowmajor(handle);
        test_activation_parity();
        test_net_oracle(a, handle);
        test_net_determinism(a, handle);
        test_pivot_parity(handle);
        test_assembly_orientation(a, handle);
        test_dets_isolated(a, handle);
        test_logpsi_oracle(a, handle);
        test_logpsi_nodes(a, handle);
        test_table_oracle(a, handle);
        test_sampler_determinism(a, handle);
        test_sampler_replay(a, handle);
        test_logp_cache_integrity(a, handle);
        test_acceptance_rates(a, handle);
        
        cublasDestroy(handle);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n";
        g_failures++;
    }

    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
