// forward_opt's accumulation was rewritten from
//     sum = sum + w * cur[j];            (two 160-byte temporaries per iteration)
// to
//     fma_into(sum, w, cur[j]);          (in place, accumulator stays in registers)
//
// The claim is that this is BIT-IDENTICAL, not merely close: same operands, same
// order over j, and IEEE multiplication is commutative. That claim is what makes
// the speedup free, so it needs a test that can actually falsify it.
//
// Nothing else in the suite can. test_psi_jet_swap compares jpsi against the
// legacy determinant composition and test_record compares local_E against a
// frozen copy -- but forward_opt sits on BOTH sides of each, so a bug in the
// accumulation shifts both equally and slips through. This file keeps a
// test-local replica of the ORIGINAL loop as the oracle, the same pattern as
// lu_det<Jet> in test_detjet and legacy_local_E in test_record.
#include "../lib/physics.h"
#include "test_common.h"
#include "../lib/network.h"
#include "../lib/autodiff.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << (msg) << " (" << #cond << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        g_failures++; \
    } \
} while (0)

// Verbatim pre-change forward_opt. Never shipped; exists only to be compared against.
template<typename T>
static std::vector<T>* legacy_forward_opt(const Network& net, const std::vector<T>& input,
                                          const std::vector<double>& param_values,
                                          std::vector<T>& buf_a, std::vector<T>& buf_b) {
    if (buf_a.size() != net.max_width) buf_a.resize(net.max_width);
    if (buf_b.size() != net.max_width) buf_b.resize(net.max_width);
    for (std::size_t i = 0; i < input.size(); i++) buf_a[i] = input[i];

    std::vector<T>* cur = &buf_a;
    std::vector<T>* nxt = &buf_b;
    for (std::size_t l = 0; l < net.layers.size(); l++) {
        int in_size  = net.layers[l].input_size;
        int out_size = net.layers[l].output_size;
        bool is_out  = (l+1 == net.layers.size());
        for (int i = 0; i < out_size; i++) {
            T sum = T(param_values[net.layers[l].bias_offset+i]);
            for (int j = 0; j < in_size; j++) {
                sum = sum + param_values[net.layers[l].weight_offset+in_size*i+j] * (*cur)[j];
            }
            (*nxt)[i] = is_out ? sum : apply_activation(net.activation, sum);
        }
        std::swap(cur, nxt);
    }
    return cur;
}

static void rand_jets(std::vector<Jet>& v, int n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    v.assign(n, Jet());
    for (auto& j : v) {
        j.v = u(rng);
        for (int a = 0; a < D; a++) j.g[a] = u(rng);
        j.l = u(rng);
    }
}

static void compare_jet(const Network& net, int in_width, const std::string& tag) {
    std::mt19937_64 rng(4242);
    std::vector<Jet> a1, b1, a2, b2, in;
    for (int trial = 0; trial < 20; trial++) {
        rand_jets(in, in_width, rng);
        std::vector<Jet> ref = *legacy_forward_opt<Jet>(net, in, net.params, a1, b1);
        std::vector<Jet> got = *net.forward_opt<Jet>(in, net.params, a2, b2);
        CHECK(got.size() == ref.size(), tag + ": output size differs");
        if (got.size() != ref.size()) return;
        for (std::size_t i = 0; i < ref.size(); i++) {
            CHECK(got[i].v == ref[i].v, tag + ": jet value differs at neuron " + std::to_string(i));
            CHECK(got[i].l == ref[i].l, tag + ": jet laplacian differs at neuron " + std::to_string(i));
            for (int a = 0; a < D; a++)
                CHECK(got[i].g[a] == ref[i].g[a],
                      tag + ": jet gradient differs at neuron " + std::to_string(i));
        }
    }
}

static void compare_double(const Network& net, int in_width, const std::string& tag) {
    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> a1, b1, a2, b2, in;
    for (int trial = 0; trial < 20; trial++) {
        in.assign(in_width, 0.0);
        for (auto& v : in) v = u(rng);
        std::vector<double> ref = *legacy_forward_opt<double>(net, in, net.params, a1, b1);
        std::vector<double> got = *net.forward_opt<double>(in, net.params, a2, b2);
        CHECK(got.size() == ref.size(), tag + ": output size differs");
        if (got.size() != ref.size()) return;
        for (std::size_t i = 0; i < ref.size(); i++)
            CHECK(got[i] == ref[i], tag + ": double output differs at neuron " + std::to_string(i));
    }
}

int main() {
    Ansatz a({64}, {64}, {64}, Activation::Gelu);
    seed_ansatz(a, 2024);

    // Jet path: what jpsi uses, and where the speedup came from.
    compare_jet(a.h_net,   dim + 2, "h_net<Jet>");
    compare_jet(a.rho_net, m_feat,  "rho_net<Jet>");
    compare_jet(a.orb_net, dim + 2, "orb_net<Jet>");

    // Double path: fma_into's other overload, used by psi() when need_inv is false.
    compare_double(a.h_net,   dim + 2, "h_net<double>");
    compare_double(a.rho_net, m_feat,  "rho_net<double>");
    compare_double(a.orb_net, dim + 2, "orb_net<double>");

    if (g_failures == 0) std::cout << "All tests passed\n";
    else                 std::cout << g_failures << " failure(s)\n";
    return g_failures != 0;
}
