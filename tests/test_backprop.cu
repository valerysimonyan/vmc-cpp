// Phase 5: device O assembly (value-level backprop) against CPU local_E.
//
// D3 is the load-bearing test: it compares every one of P entries per sample
// against the host's O. D1 and D2 pin the two pieces most likely to be subtly
// wrong in isolation -- the activation derivative and the cuBLAS orientation
// plus O_pool placement of the weight gradients.
#include "../lib/gpu/arena.h"
#include "../lib/gpu/backprop.h"
#include "../lib/gpu/local_e.h"
#include "../lib/physics.h"
#include "../lib/network.h"
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

struct Dev {
    DeviceState ds;
    explicit Dev(const Ansatz& a) : ds(a, false) {
        ds.grow_phase3(a, false); ds.grow_phase33(false); ds.grow_phase4(false);
        ds.grow_phase42(false);   ds.grow_phase43(false); ds.grow_phase5(a, false);
    }
    void push(const std::vector<double>& hx, const std::vector<double>& hs, const std::vector<double>& ht, const Ansatz& a) {
        PinnedArray st; ds.upload_params(a, st);
        std::vector<real> tx(hx.begin(),hx.end()), ts(hs.begin(),hs.end()), tt(ht.begin(),ht.end());
        ds.x.up(tx.data(),tx.size()); ds.s.up(ts.data(),ts.size()); ds.t.up(tt.data(),tt.size());
    }
};

// Shuffled labels as in test_local_e; every third walker gets particle 1 placed
// 1e-3 fm from particle 0 -- close enough to push determinants toward a node,
// far enough that the sample stays valid and its O must still match.
static void rand_cfg(int B, std::vector<double>& hx, std::vector<double>& hs, std::vector<double>& ht,
                     std::mt19937_64& rng, bool near_node) {
    std::uniform_real_distribution<double> d(-x_init_range, x_init_range);
    hx.assign((std::size_t)B*D, 0.0); hs.assign((std::size_t)B*N, 0.0); ht.assign((std::size_t)B*N, 0.0);
    for (auto& v : hx) v = d(rng);
    for (int w = 0; w < B; w++) {
        std::vector<double> sw(N), tw(N);
        for (int i = 0; i < N; i++) { sw[i] = (i < N_u) ? 1.0 : -1.0; tw[i] = (i < N_p) ? 1.0 : -1.0; }
        std::shuffle(sw.begin(), sw.end(), rng); std::shuffle(tw.begin(), tw.end(), rng);
        for (int i = 0; i < N; i++) { hs[(std::size_t)w*N+i] = sw[i]; ht[(std::size_t)w*N+i] = tw[i]; }
        if (near_node && w % 3 == 1)
            for (int q = 0; q < dim; q++) hx[(std::size_t)w*D + 1*dim + q] = hx[(std::size_t)w*D + q] + 1e-3;
    }
}

// --- D1: act_grad ------------------------------------------------------------
static void test_act_grad() {
    const std::size_t n = 1000000;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-8.0, 8.0);
    std::vector<double> z(n); for (auto& v : z) v = u(rng);
    z[0] = 0.0; z[1] = -0.0; z[2] = 1e-300; z[3] = -30.0; z[4] = 30.0;
    DeviceArray<real> dz, dout; dz.alloc(n); dout.alloc(n);
    std::vector<real> tz(z.begin(), z.end()); dz.up(tz.data(), n);
    for (Activation act : {Activation::Gelu, Activation::Tanh}) {
        act_grad_eval(dz.d, dout.d, n, act);
        std::vector<real> got(n); dout.down(got.data(), n);
        double worst = 0; std::size_t nbit = 0;
        for (std::size_t i = 0; i < n; i++) {
            const double want = act_grad(act, z[i]);
            worst = std::max(worst, rel((double)got[i], want));
            if ((double)got[i] != want) nbit++;
        }
        std::printf("  act_grad %-4s vs CPU, 1e6 points: worst rel %.2e  (%zu not bitwise)\n",
                    act == Activation::Gelu ? "Gelu" : "Tanh", worst, nbit);
        CHECK(worst <= 1e-13, "device act_grad disagrees with network.h act_grad");
    }
}

// --- D2: strided-batched dW and db, including O_pool placement -----------------
static void test_dW_strided(cublasHandle_t h) {
    const int R = 5, in_w = 7, out_w = 3, Bc = 11;
    const std::size_t P = 257, w_off = 123, b_off = w_off + (std::size_t)in_w * out_w;
    std::mt19937_64 rng(2);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> a((std::size_t)Bc*R*in_w), dl((std::size_t)Bc*R*out_w);
    for (auto& v : a) v = u(rng);
    for (auto& v : dl) v = u(rng);

    // P-wide rows full of a sentinel, plus one extra row either side
    const double SENT = 12345.678;
    std::vector<double> O((std::size_t)(Bc + 2) * P, SENT);
    DeviceArray<real> da, dd; DeviceArray<double> dO;
    da.alloc(a.size()); dd.alloc(dl.size()); dO.alloc(O.size());
    { std::vector<real> t(a.begin(), a.end()); da.up(t.data(), t.size()); }
    { std::vector<real> t(dl.begin(), dl.end()); dd.up(t.data(), t.size()); }
    dO.up(O.data(), O.size());

    double* first = dO.d + P;                                 // row 0 of the batch is the 2nd row
    dW_strided(h, da.d, dd.d, R, in_w, out_w, Bc, first + w_off, (long long)P);
    db_rows(dd.d, R, out_w, Bc, first, b_off, P);
    dO.down(O.data(), O.size());

    double worst = 0; std::size_t touched = 0;
    for (std::size_t row = 0; row < (std::size_t)Bc + 2; row++) {
        for (std::size_t k = 0; k < P; k++) {
            const double got = O[row*P + k];
            const bool in_batch = row >= 1 && row <= (std::size_t)Bc;
            const bool in_W = k >= w_off && k < w_off + (std::size_t)in_w*out_w;
            const bool in_b = k >= b_off && k < b_off + (std::size_t)out_w;
            if (in_batch && (in_W || in_b)) {
                const std::size_t w = row - 1;
                double want = 0.0;
                if (in_W) {
                    const int q = (int)(k - w_off), i = q / in_w, j = q % in_w;   // row-major out_w x in_w
                    for (int r = 0; r < R; r++) want += dl[(w*R + r)*out_w + i] * a[(w*R + r)*in_w + j];
                } else {
                    const int i = (int)(k - b_off);
                    for (int r = 0; r < R; r++) want += dl[(w*R + r)*out_w + i];
                }
                worst = std::max(worst, rel(got, want));
            } else if (got != SENT) touched++;
        }
    }
    std::printf("  dW/db strided oracle: worst rel %.2e;  %zu entries outside the target blocks modified\n", worst, touched);
    CHECK(worst <= 1e-13, "strided-batched dW or db disagrees with the host outer-product loop");
    CHECK(touched == 0, "dW/db wrote outside its O_pool blocks");
}

// --- D3 / D4 / D5 --------------------------------------------------------------
static void run_device_O(Dev& dv, cublasHandle_t h, const Ansatz& a, int B, int chunk, std::vector<double>& O, std::vector<uint8_t>& valid) {
    Workspace ws;
    eval_local_E_device(dv.ds, h, a, ws, B, 0, /*stash_for_O=*/true);
    assemble_O_batch(dv.ds, h, 0, B, 0, chunk);
    O.resize((std::size_t)B * dv.ds.P);
    dv.ds.O_pool.down(O.data(), O.size());
    valid.resize(B);
    dv.ds.valid_loc.down(valid.data(), (std::size_t)B);
}

// Network::backprop_impl with every factor replaced by its absolute value: the
// sum of |terms| that each dtheta entry is built from. This is the scale a
// correct implementation's rounding error is proportional to -- measured, the
// device lands at 1e-16 .. 7e-16 of it -- and it is what makes near-node samples
// testable: there the particle sums behind orb's weight gradients cancel by
// 6e5 .. 7e6, so an entrywise relative bound would demand sub-epsilon accuracy.
static void mag_backprop(const Network& net, const ForwardCache& c, const std::vector<double>& seed_mag,
                         std::vector<double>& dtheta_mag, std::vector<double>* dinput_mag) {
    std::vector<double> cur(seed_mag), nxt;
    for (int l = (int)net.layers.size() - 1; l >= 0; l--) {
        const auto& L = net.layers[l];
        nxt.assign(L.input_size, 0.0);
        for (int i = 0; i < L.output_size; i++) {
            dtheta_mag[L.bias_offset + i] += cur[i];
            for (int j = 0; j < L.input_size; j++) {
                dtheta_mag[L.weight_offset + L.input_size*i + j] += cur[i] * std::fabs(c.a[net.a_offset[l] + j]);
                nxt[j] += std::fabs(net.params[L.weight_offset + L.input_size*i + j]) * cur[i];
            }
        }
        if (l > 0) for (int j = 0; j < L.input_size; j++) nxt[j] *= std::fabs(act_grad(net.activation, c.z[net.z_offset[l-1] + j]));
        cur.swap(nxt);
    }
    if (dinput_mag) *dinput_mag = cur;
}

// Copy one row of a device activation stash into the host ForwardCache layout
// (a[a_offset[l] + j] = layer l's input, z[z_offset[l] + i] = its pre-activation).
struct HostStash { std::vector<std::vector<real>> a_in, z; };
static HostStash download_stash(const NetCache& c) {
    HostStash h; h.a_in.resize(c.a_in.size()); h.z.resize(c.z.size());
    for (std::size_t l = 0; l < c.a_in.size(); l++) {
        h.a_in[l].resize(c.a_in[l].n); c.a_in[l].down(h.a_in[l].data(), h.a_in[l].size());
        h.z[l].resize(c.z[l].n);       c.z[l].down(h.z[l].data(), h.z[l].size());
    }
    return h;
}
static void stash_row_to_cache(const Network& net, const HostStash& hs, std::size_t row, ForwardCache& fc) {
    fc.a.assign(net.a_tot_size, 0.0); fc.z.assign(net.z_tot_size, 0.0);
    for (std::size_t l = 0; l < net.layers.size(); l++) {
        const int in_w = net.layers[l].input_size, out_w = net.layers[l].output_size;
        for (int j = 0; j < in_w; j++)  fc.a[net.a_offset[l] + j] = (double)hs.a_in[l][row*in_w + j];
        for (int i = 0; i < out_w; i++) fc.z[net.z_offset[l] + i] = (double)hs.z[l][row*out_w + i];
    }
}

// fill_O (physics.cpp) run on EVERYTHING the device backprop reads: its dets,
// Minv, S, rho, x_sh and its stashed activations, copied into ws's caches. So the
// only remaining difference is the backprop arithmetic itself. Same statements,
// same order as fill_O. `mag` receives each entry's summand magnitude, divided
// by |S| exactly as the entry is.
//
// Sharing the caches matters: with the host's own forward pass instead, rho's
// gradients differed by up to 7e-12 -- exactly the difference between the two
// forward passes' xi (the device's centre-of-mass subtraction and sums round
// differently), not anything in the backprop.
static void cpu_O_with_device_inputs(const Ansatz& a, Workspace& ws, const real* rho, const real* dets, const real* Minv, double S,
                                     std::vector<double>& O, std::vector<double>& mag) {
    const std::size_t n_h = a.h_net.params.size(), n_rho = a.rho_net.params.size(), n_orb = a.orb_net.params.size();
    const std::size_t P = a.n_params();
    O.assign(P, 0.0); mag.assign(P, 0.0);
    std::vector<double> m_h(n_h, 0.0), m_rho(n_rho, 0.0), m_orb(n_orb, 0.0), dmag, smag;

    ws.seed_rho.resize(K); smag.resize(K);
    for (int i = 0; i < K; i++) { ws.seed_rho[i] = (double)dets[i]; smag[i] = std::fabs((double)dets[i]); }
    a.rho_net.backprop(ws.rho_cache, ws.seed_rho, ws.dtheta_rho, ws.delta_a, ws.delta_b, &ws.dpsi_dxi);
    mag_backprop(a.rho_net, ws.rho_cache, smag, m_rho, &dmag);
    for (std::size_t p = 0; p < n_rho; p++) O[n_h + p] = ws.dtheta_rho[p] / S;

    ws.dtheta_h.assign(n_h, 0.0);
    for (int i = 0; i < N; i++) {
        a.h_net.backprop_acc(ws.h_caches[i], ws.dpsi_dxi, ws.dtheta_h, ws.delta_a, ws.delta_b);
        mag_backprop(a.h_net, ws.h_caches[i], dmag, m_h, nullptr);
    }
    for (std::size_t p = 0; p < n_h; p++) O[p] = ws.dtheta_h[p] / S;

    ws.seed_orb.resize((std::size_t)K * N); smag.resize((std::size_t)K * N);
    ws.dtheta_orb.assign(n_orb, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++)
            for (int k = 0; k < N; k++) {
                ws.seed_orb[j*N + k] = (double)rho[j] * (double)dets[j] * (double)Minv[(std::size_t)j*(N*N) + i*N + k];
                smag[j*N + k] = std::fabs(ws.seed_orb[j*N + k]);
            }
        a.orb_net.backprop_acc(ws.orb_caches[i], ws.seed_orb, ws.dtheta_orb, ws.delta_a, ws.delta_b);
        mag_backprop(a.orb_net, ws.orb_caches[i], smag, m_orb, nullptr);
    }
    for (std::size_t p = 0; p < n_orb; p++) O[n_h + n_rho + p] = ws.dtheta_orb[p] / S;

    double r2 = 0.0;
    for (int i = 0; i < D; i++) { double c = ws.x_sh[i]; r2 += c * c; }
    O[P - 1] = -std::exp(a.alpha) * std::sqrt(r2 + eps_env * eps_env);

    const double aS = std::fabs(S);
    for (std::size_t p = 0; p < n_h; p++)   mag[p] = m_h[p] / aS;
    for (std::size_t p = 0; p < n_rho; p++) mag[n_h + p] = m_rho[p] / aS;
    for (std::size_t p = 0; p < n_orb; p++) mag[n_h + n_rho + p] = m_orb[p] / aS;
    mag[P - 1] = std::fabs(O[P - 1]);
}

// D3, in two parts.
//
// (a) SAME INPUTS -- the backprop test. Host fill_O fed every input the device
//     backprop reads, stashed activations included, so any index, orientation,
//     seed, sign or layer error in the device backprop shows up here at full
//     size, and nothing upstream can. Bound: |err| <= 1e-12 * max(|O|, summand
//     magnitude) -- see mag_backprop. Measured worst 7e-16 of that scale, so the
//     bound has ~3 orders of margin and is still ~1e11 below a real bug.
//
// (b) END TO END -- device O vs CPU local_E's O, as the prompt specifies. On
//     near-node samples this inherits the Phase 3.2 LU gap: device Minv and dets
//     differ from lu_det_inv's by up to 5.5e-10 relative there (3.2e-11 on
//     ordinary samples), and small O entries are sums of large cancelling terms,
//     so entrywise rel 1e-10 fails on ~1e3 of 5.8e6 entries -- all small ones,
//     all traceable to that input gap (part (a) is what proves it). The bound is
//     therefore |err| <= 1e-10 * max(|want|, row max|O|): still ~1e7 below any
//     real bug, which moves entries by a fraction of their own size.
static void test_full_O(const Ansatz& a, cublasHandle_t h) {
    const int B = 256;
    std::mt19937_64 rng(3);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng, true);
    Dev dv(a); dv.push(hx, hs, ht, a);
    const std::size_t P = dv.ds.P;
    std::vector<double> Od; std::vector<uint8_t> vd;
    run_device_O(dv, h, a, B, 0, Od, vd);
    const HostStash st_h = download_stash(dv.ds.cache_h), st_rho = download_stash(dv.ds.cache_rho), st_orb = download_stash(dv.ds.cache_orb);
    std::vector<real> Xs((std::size_t)B*D); dv.ds.x_sh.down(Xs.data(), Xs.size());
    std::vector<real> De((std::size_t)B*K), Mi((std::size_t)B*K*N*N), Sd(B), Rh((std::size_t)B*K);
    dv.ds.dets_psi.down(De.data(), De.size()); dv.ds.Minv_batch.down(Mi.data(), Mi.size()); dv.ds.S.down(Sd.data(), B);
    dv.ds.cache_rho.z.back().down(Rh.data(), Rh.size());

    const std::size_t n_h = a.h_net.params.size(), n_rho = a.rho_net.params.size();
    Workspace ws; std::vector<double> Oc, Os, Omag;
    int n_cmp = 0, n_near = 0, mask_mism = 0;
    std::size_t fail_same = 0, fail_e2e = 0, fail_e2e_strict = 0, bad_invalid = 0;
    std::size_t fs_h = 0, fs_rho = 0, fs_orb = 0, fs_alpha = 0;
    double worst_same = 0, worst_e2e_row_near = 0, worst_e2e_row_far = 0, worst_entry_near = 0;
    for (int w = 0; w < B; w++) {
        const double* xw = &hx[(std::size_t)w*D]; const double* sw = &hs[(std::size_t)w*N]; const double* tw = &ht[(std::size_t)w*N];
        double E;
        const bool ok = local_E(xw, sw, tw, a, ws, Oc, E);
        if (ok != (vd[w] != 0)) { mask_mism++; continue; }
        if (!ok) {
            for (std::size_t k = 0; k < P; k++) if (Od[(std::size_t)w*P + k] != 0.0) { bad_invalid++; break; }
            continue;
        }
        n_cmp++; const bool near = (w % 3 == 1); if (near) n_near++;

        // shared-input ws: device activations, device x_sh
        ws.h_caches.resize(N); ws.orb_caches.resize(N);
        stash_row_to_cache(a.rho_net, st_rho, (std::size_t)w, ws.rho_cache);
        for (int i = 0; i < N; i++) {
            stash_row_to_cache(a.h_net,   st_h,   (std::size_t)w*N + i, ws.h_caches[i]);
            stash_row_to_cache(a.orb_net, st_orb, (std::size_t)w*N + i, ws.orb_caches[i]);
        }
        ws.drho.resize(K); for (int k = 0; k < K; k++) ws.drho[k] = (double)Rh[(std::size_t)w*K + k];
        ws.x_sh.resize(D); for (int q = 0; q < D; q++) ws.x_sh[q] = (double)Xs[(std::size_t)w*D + q];
        cpu_O_with_device_inputs(a, ws, &Rh[(std::size_t)w*K], &De[(std::size_t)w*K], &Mi[(std::size_t)w*K*N*N], (double)Sd[w], Os, Omag);

        double rowmax = 0; for (std::size_t k = 0; k + 1 < P; k++) rowmax = std::max(rowmax, std::fabs(Oc[k]));
        for (std::size_t k = 0; k < P; k++) {
            const double got = Od[(std::size_t)w*P + k];
            // (a)
            const double ds_ = std::fabs(got - Os[k]);
            const double scale = std::max(std::fabs(Os[k]), Omag[k]);
            if (ds_ > 0) worst_same = std::max(worst_same, ds_ / std::max(1e-300, scale));
            if (near && std::fabs(Os[k]) > 1e-14) worst_entry_near = std::max(worst_entry_near, ds_ / std::fabs(Os[k]));
            if (!(ds_ <= 1e-12 * scale)) {
                fail_same++;
                if (k == P - 1) fs_alpha++; else if (k < n_h) fs_h++; else if (k < n_h + n_rho) fs_rho++; else fs_orb++;
            }
            // (b)
            const double de = std::fabs(got - Oc[k]);
            if (!(de <= 1e-10 * std::fabs(Oc[k]) || de <= 1e-14)) fail_e2e_strict++;
            if (!(de <= 1e-10 * std::max(std::fabs(Oc[k]), rowmax))) fail_e2e++;
            if (k + 1 < P) {
                double& wr = near ? worst_e2e_row_near : worst_e2e_row_far;
                wr = std::max(wr, de / std::max(1e-300, rowmax));
            }
        }
    }
    std::printf("  full O oracle, %d samples x %zu params (%d near-node), mask disagreements %d, non-zero invalid rows %zu\n",
                n_cmp, P, n_near, mask_mism, bad_invalid);
    std::printf("    (a) identical inputs (stash, dets, Minv, S, rho): worst |err|/summand scale %.2e;  outside 1e-12 of it: %zu  (h %zu, rho %zu, orb %zu, alpha %zu)\n",
                worst_same, fail_same, fs_h, fs_rho, fs_orb, fs_alpha);
    std::printf("        (entrywise |err|/|O| on near-node samples reaches %.2e -- the cancellation the scale accounts for)\n", worst_entry_near);
    std::printf("    (b) vs CPU local_E end to end:   worst |err|/row max  near-node %.2e  ordinary %.2e;  outside row-scaled 1e-10: %zu  (entrywise-strict would flag %zu)\n",
                worst_e2e_row_near, worst_e2e_row_far, fail_e2e, fail_e2e_strict);
    CHECK(fail_same == 0, "device backprop disagrees with host fill_O given the same dets / Minv / S");
    CHECK(fail_e2e == 0, "device O disagrees with CPU local_E's O beyond the LU-gap bound");
    CHECK(mask_mism == 0, "device and CPU validity differ on the O test configs");
    CHECK(bad_invalid == 0, "an invalid sample's O_pool row is not zeroed");
    CHECK(n_cmp > B / 2 && n_near > 0, "too few valid (or near-node) samples compared");
}

static void test_determinism_and_chunking(const Ansatz& a, cublasHandle_t h) {
    const int B = 240;
    std::mt19937_64 rng(4);
    std::vector<double> hx, hs, ht; rand_cfg(B, hx, hs, ht, rng, true);
    Dev dv(a); dv.push(hx, hs, ht, a);
    std::vector<double> O1, O2, O3; std::vector<uint8_t> v1, v2, v3;
    run_device_O(dv, h, a, B, 0, O1, v1);

    // D4: backprop again from the SAME stash -- no re-evaluation in between.
    assemble_O_batch(dv.ds, h, 0, B);
    O2.resize(O1.size()); dv.ds.O_pool.down(O2.data(), O2.size());

    // D5: same stash, backprop in 37-walker chunks.
    dv.ds.O_pool.zero();
    assemble_O_batch(dv.ds, h, 0, B, 0, 37);
    O3.resize(O1.size()); dv.ds.O_pool.down(O3.data(), O3.size());

    std::size_t d12 = 0, d13 = 0;
    for (std::size_t q = 0; q < O1.size(); q++) { if (O1[q] != O2[q]) d12++; if (O1[q] != O3[q]) d13++; }
    std::printf("  determinism: assemble twice -> %zu of %zu O_pool entries differ\n", d12, O1.size());
    std::printf("  chunking: whole batch vs 37-walker chunks -> %zu differ\n", d13);
    CHECK(d12 == 0, "assemble_O_batch is not bit-reproducible");
    CHECK(d13 == 0, "chunked assemble_O_batch differs from whole-batch");
}

int main() {
    gpu_select_device(true);
    Ansatz a({64},{64},{64}, Activation::Gelu);
    seed_ansatz(a, 2024);
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::cerr << "cublasCreate failed\n"; return 1; }
    try {
        test_act_grad();
        test_dW_strided(h);
        test_full_O(a, h);
        test_determinism_and_chunking(a, h);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: uncaught exception -- " << e.what() << "\n"; g_failures++;
    }
    cublasDestroy(h);
    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
