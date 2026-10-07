#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <array>

#include "network.h"
#include "constants.h"
#include "autodiff.h"
#include "envelope.h"



// Return index of orbital matrix for orbital j, particle i, determinant k
VMC_HD inline std::size_t orb_idx(std::size_t i, int j) { 
    return i*N + j; 
}
VMC_HD inline std::size_t slater_idx(std::size_t k, int i, int j) { 
    return k*(N*N) + (std::size_t)i*N + j; 
}

// Return index of spin, isospin table
template <typename L>
VMC_HD inline int st_combo(L s, L t) {
    return (s > L(0) ? 0 : 1) + 2 * (t > L(0) ? 0 : 1);
}
// Look at value of first and 2nd bits of c, return spin and isospin, inverse of above
template <typename L>
VMC_HD inline L st_spin(int c) { 
    return (c & 1) ? L(-1) : L(1); 
}
template <typename L>
VMC_HD inline L st_iso(int c)  { 
    return (c & 2) ? L(-1) : L(1); 
}
// Row of the (s,t) table for particle row p (p = i on the CPU, w*N + i on the GPU) and combination c
VMC_HD inline std::size_t st_row(std::size_t p, int c) {
    return p*4 + c;
}

// Single particle input for particle i
template<typename T, typename L>
VMC_HD inline void particle_input(const T* x_i, L s_i, L t_i, T* in) {
    for (int d = 0; d < dim; d++) in[d] = x_i[d];
    in[dim] = T(s_i);
    in[dim+1] = T(t_i);
}

// Compose FermiSets sum
template <typename T>
VMC_HD inline T S_sum(const T* rho, const T* dets) {
    T S = T(0);
    for (int k = 0; k < K; k++) S += rho[k] * dets[k];
    return S;
}

// Evaluate log|Ψ| = log|envelope| + log|S|
template <typename T, typename L>
VMC_HD inline T log_psi(T alpha, const T* jc, const T* x_sh, const L* s, const L* t, T S) {
    using std::isfinite; using std::log; using std::fabs;
    if (!(S != T(0)) || !isfinite(S)) return T(-INFINITY);
    return envelope::log_env_J(alpha, jc, x_sh, s, t) + log(fabs(S));
}

struct Ansatz {
    Network h_net;    // dim -> m_feat
    Network rho_net;  // m_feat -> K
    Network orb_net;  // dim -> K*N
    double alpha;     // Envelope exponent
    std::array<double, n_jas_par> jc{};   // Jastrow coefficients, after alpha in the flat layout

    // Initialize networks
    Ansatz(std::vector<int> h_hidden, std::vector<int> rho_hidden, std::vector<int> orb_hidden, Activation act)
        : h_net(dim+1+1, h_hidden, m_feat, act),
          rho_net(m_feat, rho_hidden, K, act),
          orb_net(dim+1+1, orb_hidden, K * N, act),
          alpha(alpha_init) {}

    // Return total parameter count 
    std::size_t n_params() const {
        return h_net.params.size() + rho_net.params.size() + orb_net.params.size() + 1 + n_jas_par;
    }

    // Extract specific parameter
    double get_param(std::size_t k) const {
        std::size_t nh = h_net.params.size();
        std::size_t nr = rho_net.params.size();
        std::size_t no = orb_net.params.size();
        if (k < nh) return h_net.params[k];
        k -= nh;
        if (k < nr) return rho_net.params[k];
        k -= nr;
        if (k < no) return orb_net.params[k];
        k -= no;
        if (k == 0) return alpha;
        return jc[k - 1];    
    }

    // Copy parameters
    void copy_params_flat(double* dst) const {
        std::size_t off = 0;
        std::memcpy(dst + off, h_net.params.data(), h_net.params.size() * sizeof(double));
        off += h_net.params.size();
        std::memcpy(dst + off, rho_net.params.data(), rho_net.params.size() * sizeof(double));
        off += rho_net.params.size();
        std::memcpy(dst + off, orb_net.params.data(), orb_net.params.size() * sizeof(double));
        off += orb_net.params.size();
        dst[off] = alpha;
        for (int m = 0; m < n_jas_par; m++) dst[off + 1 + m] = jc[m];
    }

    // Add to specific parameter
    void add_to_param(std::size_t k, double delta) {
        std::size_t nh = h_net.params.size();
        std::size_t nr = rho_net.params.size();
        std::size_t no = orb_net.params.size();
        if (k < nh) { h_net.params[k] += delta; return; }
        k -= nh;
        if (k < nr) { rho_net.params[k] += delta; return; }
        k -= nr;
        if (k < no) { orb_net.params[k] += delta; return; }
        k -= no;
        if (k == 0) { alpha += delta; return; }
        jc[k - 1] += delta;
    }
};

struct Workspace {
    std::vector<Jet> jin;             // Full jet encoded degrees of freedom
    std::vector<Jet> jin_sh;          // Same as above but in COM coordinates
    std::vector<Jet> jsingle;         // Jet encoded single particle input
    std::vector<Jet> jbuf_a, jbuf_b;  // Shared network ping-pong (h_net / rho_net / orb_net)
    std::vector<Jet> jxi;             // Accumulataed embedding (xi = sum h(r_i)), size m_feat
    std::vector<Jet> jrho;            // Output of rho_net, size K
    std::vector<Jet> jM;              // K orbital matrices, flat, size K*N*N
    std::vector<Jet> jM_scratch;      // destructible copy of one K slice, size N*N
    
    // Double version, scratch, meant for metropolis
    std::vector<double> x_sh;
    std::vector<double> dsingle;
    std::vector<double> dbuf_a, dbuf_b;
    std::vector<double> dxi;
    std::vector<double> drho;
    std::vector<double> dM;
    std::vector<double> dM_scratch;
    std::vector<double> dets;
    std::vector<int>    piv;

    // Parameter-derivative scratch
    std::vector<double> dMinv;              // K_det*N*N, every determinant's inverse
    std::vector<double> dMinv_k;            // N*N, lu_det_inv's own output before the copy into dMinv
    std::vector<double> col_scratch;   
    std::vector<double> jd_Mval, jd_Minv, jd_G, jd_B;  // N*N each
    std::vector<double> jd_col;                        // N
    std::vector<int>    jd_piv;                        // N

    ForwardCache rho_cache;
    std::vector<ForwardCache> h_caches;     // Size N
    std::vector<ForwardCache> orb_caches;   // Size N
    std::vector<double> fc_out;             // Throwaway forward_cached() output, value unused
    std::vector<double> dpsi_dxi;           // Size m_feat, rho_net's input gradient
    std::vector<double> seed_rho;           // Size K
    std::vector<double> seed_orb;           // Size K*N, rebuilt per particle
    std::vector<double> dtheta_h, dtheta_rho, dtheta_orb;
    std::vector<double> delta_a, delta_b;   // Shared backprop ping-pong

    int n_node_hits = 0;

    // For spin and isospin
    std::vector<int> up_list, dn_list;              
    std::vector<int> p_list, n_list;              
    std::vector<double> s_swap; // Scratch for Heisenberg spin interaction
    std::vector<double> t_swap; // Scratch for Heisenberg isospin interaction

    // (s,t) particle combos at shifted position
    std::vector<double> tab_h; 
    std::vector<double> tab_orb;
    bool table_valid = false; 
    
    // More scratch for table
    std::vector<double> tab_xi;   // m_feat
    std::vector<double> tab_rho;  // K
    std::vector<double> tab_M;    // N*N
    std::vector<int> tab_piv;     // N

    // Rank 2 exchange ratio scratch
    std::vector<double> sm_xi;
    std::vector<double> sm_rho;
    std::vector<double> sm_dci, sm_dcj;

    // Persample L^2
    double l2_val = 0.0;

    Workspace() {
        x_sh.resize(D);
        dsingle.resize(dim + 2);
        dxi.resize(m_feat);
        drho.resize(K);
        dM.resize((std::size_t)K*N*N);
        dM_scratch.resize((std::size_t)N*N);
        dets.resize(K);
        dMinv.resize((std::size_t)K*N*N);
        h_caches.resize(N);
        orb_caches.resize(N);
        seed_rho.resize(K);
        seed_orb.resize((std::size_t)K*N);
        s_swap.resize(N);
        t_swap.resize(N);
        tab_h.resize((std::size_t)N*4*m_feat);
        tab_orb.resize((std::size_t)N*4*K*N);
        tab_xi.resize(m_feat);
        tab_rho.resize(K);
        tab_M.resize((std::size_t)N*N);
        sm_xi.resize(m_feat);
        sm_rho.resize(K);
        sm_dci.resize((std::size_t)K*N);
        sm_dcj.resize((std::size_t)K*N);
    }

    // Larger so only allocate when needed
    void ensure_jet_buffers() {
        if (!jM.empty()) return;
        jin.resize(D);
        jin_sh.resize(D);
        jsingle.resize(dim + 2);
        jxi.resize(m_feat);
        jrho.resize(K);
        jM.resize((std::size_t)K*N*N);
    }
};


bool rank2_well_conditioned(const Workspace& ws);

double psi(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, bool need_inv = false);

double log_p(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws);

Jet jpsi(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws);

void build_st_table(const double* x, const Ansatz& a, Workspace& ws);

double S_from_table(const double* s, const double* t, const Ansatz& a, Workspace& ws);

double swap_ratio(const double* s, const double* t, int i, int j, double s_new_i, double t_new_i, double s_new_j, double t_new_j, double S0, bool use_rank2, const Ansatz& a, Workspace& ws);

void fill_O(const Ansatz& a, Workspace& ws, double S, std::vector<double>& O_out, const double* s, const double* t);

void assemble_O(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out);
