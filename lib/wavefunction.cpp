#include "network.h"
#include "constants.h"
#include "wavefunction.h"
#include "autodiff.h"
#include "slater.h"
#include "envelope.h"

#include <cmath>
#include <limits>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <cassert>

// Pile of scratch for psi_impl<T> 
template<typename T>
struct EvalBuffers {
    std::vector<T>& single;
    std::vector<T>& xi;
    std::vector<T>& rho;
    std::vector<T>& M;
    std::vector<T>& M_scratch;
    std::vector<T>& buf_a;
    std::vector<T>& buf_b;
};

// This picks certain set of buffers based on type
template<typename T>
static EvalBuffers<T> get_buffers(Workspace& ws) {
    if constexpr (std::is_same_v<T, Jet>) {
        return { ws.jsingle, ws.jxi, ws.jrho, ws.jM, ws.jM_scratch, ws.jbuf_a, ws.jbuf_b };
    } else {
        return { ws.dsingle, ws.dxi, ws.drho, ws.dM, ws.dM_scratch, ws.dbuf_a, ws.dbuf_b };
    }
}

// Shift to center of mass coordinates 
template <typename T>
static void shift_to_com(const T* x, T* x_sh) {
    for (int d = 0; d < dim; d++) {
        T R_cm_d = 0.0;
        for (int i = 0; i < N; i++) R_cm_d = R_cm_d + x[i*dim + d] / N;
        for (int i = 0; i < N; i++) x_sh[i*dim+d]= x[i*dim + d] - R_cm_d;
    }
}

// One network forward pass: cached (value path, for backprop) or plain
template <typename T>
static const std::vector<T>* fwd(const Network& net, const std::vector<T>& in, ForwardCache* cache, Workspace& ws, EvalBuffers<T>& buf) {
    if constexpr (std::is_same_v<T, double>) {
        if (cache) { 
            net.forward_cached(in, *cache, ws.fc_out); 
            return &ws.fc_out; 
        }
    }
    return net.forward_opt<T>(in, net.params, buf.buf_a, buf.buf_b);
}

// Given jet or double pass in input coordinates
template <typename T>
static const T* com_coordinates(const double* x, Workspace& ws) {
    if constexpr (std::is_same_v<T, Jet>) {
        ws.ensure_jet_buffers();
        for (int i = 0; i < D; i++) ws.jin[i] = Jet::input(x[i], i);
        shift_to_com<Jet>(ws.jin.data(), ws.jin_sh.data());
        return ws.jin_sh.data();
    } else {
        shift_to_com<double>(x, ws.x_sh.data());
        return ws.x_sh.data();
    }
}

// Evalute xi = sum_i h(r_i) and rho(xi), store in buffers
template <typename T>
static void eval_rho(const T* x_sh, const double* s, const double* t, const Ansatz& a, Workspace& ws, EvalBuffers<T>& buf, bool cache) {
    // As xi accumulates initialize to zero 
    std::fill(buf.xi.begin(), buf.xi.end(), T(0.0));

    // Compute xi = sum_i h(r_i)
    for (int i = 0; i < N; i++) {
        particle_input(x_sh + i*dim, s[i], t[i], buf.single.data());
        const std::vector<T>* h_out = fwd(a.h_net, buf.single, cache ? &ws.h_caches[i] : nullptr, ws, buf);
        for (int f = 0; f < m_feat; f++) buf.xi[f] = buf.xi[f] + (*h_out)[f];
    }

    // Compute rho(xi)
    const std::vector<T>* rho_out = fwd(a.rho_net, buf.xi, cache ? &ws.rho_cache : nullptr, ws, buf);
    for (int k = 0; k < K; k++) buf.rho[k] = (*rho_out)[k];
}

// Evaluate orbital matrices phi_k(r_i) and store in buffer
template <typename T>
static void eval_slater(const T* x_sh, const double* s, const double* t, const Ansatz& a, Workspace& ws, EvalBuffers<T>& buf, bool cache) {
    for (int i = 0; i < N; i++) {
        particle_input(x_sh + i*dim, s[i], t[i], buf.single.data());
        const std::vector<T>* orb_out = fwd(a.orb_net, buf.single, cache ? &ws.orb_caches[i] : nullptr, ws, buf);
        for (int k = 0; k < K; k++) {
            for (int row = 0; row < N; row++) buf.M[slater_idx(k, row, i)] = (*orb_out)[orb_idx(k, row)];
        }
    }
}

// Evaluate determinants of orbital matrices, then combine with rho to get S = sum_k rho_k det phi_k
template <typename T>
static T eval_S(Workspace& ws, EvalBuffers<T>& buf, bool need_inv) {
    if constexpr (std::is_same_v<T, Jet>) {
        T S{};
        for (int k = 0; k < K; k++) {
            const T det = det_jet_from_minv(&buf.M[slater_idx(k, 0, 0)], N, ws.jd_Mval, ws.jd_Minv, ws.jd_piv, ws.jd_col, ws.jd_G, ws.jd_B);
            S = S + buf.rho[k] * det;
        }
        return S;
    } else {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N*N; j++) buf.M_scratch[j] = buf.M[slater_idx(k, 0, 0) + j];
            if (need_inv) {
                ws.dets[k] = lu_det_inv(buf.M_scratch, N, ws.dMinv_k, ws.piv, ws.col_scratch);
                for (int j = 0; j < N*N; j++) ws.dMinv[slater_idx(k, 0, 0) + j] = ws.dMinv_k[j];
            } else {
                ws.dets[k] = lu_det<double>(buf.M_scratch, N, ws.piv);
            }
        }
        return S_sum(buf.rho.data(), ws.dets.data());
    }
}

// psi = envelope * exp(J) * sum_k rho_k det M_k
template<typename T>
static T psi_impl(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, bool need_inv) {
    using std::exp;
    constexpr bool is_jet = std::is_same_v<T, Jet>;
    const bool cache = !is_jet && need_inv;   
    EvalBuffers<T> buf = get_buffers<T>(ws);

    const T* x_sh = com_coordinates<T>(x, ws);
    eval_rho(x_sh, s, t, a, ws, buf, cache);
    eval_slater(x_sh, s, t, a, ws, buf, cache);
    const T S = eval_S(ws, buf, need_inv);

    // Combine with envelope and jastrow, depending on type
    if constexpr (!is_jet) {
        return exp(log_env_J(a.alpha, a.jc.data(), x_sh, s, t)) * S;
    } else {
        const Jet r2 = env_r2(x_sh);
        double xs[D];
        for (int i = 0; i < D; i++) xs[i] = x_sh[i].v;
        Jet Jj;
        Jj.v = jastrow<double, double>(xs, s, t, a.jc.data(), Jj.g.data(), &Jj.l);
        return exp(log_factor(a.alpha, env_radius(r2)) + Jj) * S;
    }
}

// Double evaluation of psi 
double psi (const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, bool need_inv) {
    return psi_impl<double>(x, s, t, a, ws, need_inv);
}

// Store log psi for regular use, if on node return 0
double log_p (const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws) {
    double p = psi(x, s, t, a, ws);
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    return std::log(std::abs(p));
}

// Jet evaluation of psi
Jet jpsi (const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws) {
    return psi_impl<Jet>(x, s, t, a, ws, false);
}

// Preompute h_net and orb_net for all values of spin and isospin
void build_st_table(const double* x, const Ansatz& a, Workspace& ws) {
    shift_to_com(x, ws.x_sh.data());

    for (int i = 0; i < N; i++) {
        for (int c = 0; c < 4; c++) {
            particle_input(ws.x_sh.data() + i*dim, st_spin<double>(c), st_iso<double>(c), ws.dsingle.data());

            std::vector<double>* h_out = a.h_net.forward_opt<double>(ws.dsingle, a.h_net.params, ws.dbuf_a, ws.dbuf_b);
            double* dst_h = &ws.tab_h[st_row(i, c)*m_feat];
            for (int f = 0; f < m_feat; f++) dst_h[f] = (*h_out)[f];
            
            std::vector<double>* orb_out = a.orb_net.forward_opt<double>(ws.dsingle, a.orb_net.params, ws.dbuf_a, ws.dbuf_b);
            double* dst_orb = &ws.tab_orb[st_row(i, c)*(K*N)];
            for (int j = 0; j < K*N; j++) dst_orb[j] = (*orb_out)[j];
        }
    }
    ws.table_valid = true;
}

// Given a value of s and t draw from precomputed table instead of computing from scratch
double S_from_table(const double* s, const double* t, const Ansatz& a, Workspace& ws) {
    assert(ws.table_valid);

    std::fill(ws.tab_xi.begin(), ws.tab_xi.end(), 0.0);
    for (int i = 0; i < N; i++) {
        int c = st_combo(s[i],t[i]);
        const double* h_i = &ws.tab_h[st_row(i, c)*m_feat];
        for (int f = 0; f < m_feat; f++) ws.tab_xi[f] += h_i[f];
    }

    std::vector<double>* rho_out = a.rho_net.forward_opt<double> (ws.tab_xi, a.rho_net.params, ws.dbuf_a, ws.dbuf_b);
    for (int i = 0; i < K; i++) ws.tab_rho[i] = (*rho_out)[i];

    double S = 0.0;
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) {
            int c = st_combo(s[i], t[i]);
            const double* orb_i = &ws.tab_orb[st_row(i, c)*(K*N)];
            for (int k = 0; k < N; k++) ws.tab_M[slater_idx(0, k, i)] = orb_i[orb_idx(j, k)];
        }
        double det = lu_det<double>(ws.tab_M, N, ws.tab_piv);
        S += ws.tab_rho[j] * det;
    }
    return S;
}

// Find the largest determinant by magnitude, if the largest is still within machine precision 0 return false, matrix is singular
bool rank2_well_conditioned(const Workspace& ws) {
    double max_det = 0.0;
    for (int k = 0; k < K; k++) max_det = std::max(max_det, std::fabs(ws.dets[k]));
    for (int k = 0; k < K; k++) {
        if (std::fabs(ws.dets[k]) < 1e-12 * max_det) return false; 
    }
    
    return true;
}

// Compute updated psi ratio table from swapped s, and t
double swap_ratio(const double* s, const double* t, int i, int j, double s_new_i, double t_new_i, double s_new_j, double t_new_j, double S0, bool use_rank2, const Ansatz& a, Workspace& ws) {
    if (use_rank2) {
        int c_old_i = st_combo(s[i], t[i]);
        int c_old_j = st_combo(s[j], t[j]);
        int c_new_i = st_combo(s_new_i, t_new_i);
        int c_new_j = st_combo(s_new_j, t_new_j);

        // Evaluate updated xi, as xi is just sum of h's just subtract the old modified h and add the new one
        const double* h_old_i = &ws.tab_h[st_row(i, c_old_i)  * m_feat];
        const double* h_new_i = &ws.tab_h[st_row(i, c_new_i) * m_feat];
        const double* h_old_j = &ws.tab_h[st_row(j, c_old_j) * m_feat];
        const double* h_new_j = &ws.tab_h[st_row(j, c_new_j) * m_feat];
        for (int f = 0; f < m_feat; f++) ws.sm_xi[f] = ws.dxi[f] + (h_new_i[f]-h_old_i[f]) + (h_new_j[f]-h_old_j[f]);

        std::vector<double>* rho_out = a.rho_net.forward_opt<double>(ws.sm_xi, a.rho_net.params, ws.dbuf_a, ws.dbuf_b);
        for (int k = 0; k < K; k++) ws.sm_rho[k] = (*rho_out)[k];

        // Record column change
        const double* orb_old_i = &ws.tab_orb[st_row(i, c_old_i) * (K*N)];
        const double* orb_new_i = &ws.tab_orb[st_row(i, c_new_i) * (K*N)];
        const double* orb_old_j = &ws.tab_orb[st_row(j, c_old_j) * (K*N)];
        const double* orb_new_j = &ws.tab_orb[st_row(j, c_new_j) * (K*N)];
        for (int kn = 0; kn < K*N; kn++) {
            ws.sm_dci[kn] = orb_new_i[kn]-orb_old_i[kn];
            ws.sm_dcj[kn] = orb_new_j[kn]-orb_old_j[kn];
        }   

        double S_new = 0.0;
        for(int k = 0; k < K; k++) {
            double ratio2 = det_ratio_rank2(&ws.dMinv[slater_idx(k, 0, 0)], N, &ws.sm_dci[orb_idx(k, 0)], i, &ws.sm_dcj[orb_idx(k, 0)], j);
            S_new += ws.sm_rho[k] * ws.dets[k] * ratio2;
        }
        return S_new/S0;
    }

    // Fallback: Old way of computing update
    double si = ws.s_swap[i], ti = ws.t_swap[i], sj = ws.s_swap[j], tj = ws.t_swap[j];
    ws.s_swap[i] = s_new_i; ws.t_swap[i] = t_new_i;
    ws.s_swap[j] = s_new_j; ws.t_swap[j] = t_new_j;
    double R = S_from_table(ws.s_swap.data(), ws.t_swap.data(), a, ws) / S0;
    ws.s_swap[i] = si; ws.t_swap[i] = ti;
    ws.s_swap[j] = sj; ws.t_swap[j] = tj;
    return R;
}

void fill_O(const Ansatz& a, Workspace& ws, double S, std::vector<double>& O_out, const double* s, const double* t) {
    std::size_t n_h = a.h_net.params.size();
    std::size_t n_rho = a.rho_net.params.size();
    std::size_t n_orb = a.orb_net.params.size();

    for (int i = 0; i < K; i++) ws.seed_rho[i] = ws.dets[i];
    a.rho_net.backprop(ws.rho_cache, ws.seed_rho, ws.dtheta_rho, ws.delta_a, ws.delta_b, &ws.dpsi_dxi);
    for (std::size_t p = 0; p < n_rho; p++) O_out[n_h + p] = ws.dtheta_rho[p]/S;

    ws.dtheta_h.assign(n_h, 0.0);
    for (int i = 0; i < N; i++) a.h_net.backprop_acc(ws.h_caches[i], ws.dpsi_dxi, ws.dtheta_h, ws.delta_a, ws.delta_b);
    for (std::size_t p = 0; p < n_h; p++) O_out[p] = ws.dtheta_h[p]/S;

    ws.dtheta_orb.assign(n_orb, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                ws.seed_orb[orb_idx(j, k)] = ws.drho[j] * ws.dets[j] * ws.dMinv[slater_idx(j, i, k)];
            }
        }
        a.orb_net.backprop_acc(ws.orb_caches[i], ws.seed_orb, ws.dtheta_orb, ws.delta_a, ws.delta_b);
    }
    for (std::size_t p = 0; p < n_orb; p++) O_out[n_h + n_rho + p] = ws.dtheta_orb[p]/S;

    const double r_env = env_radius(env_r2(ws.x_sh.data()));
    O_out[n_h+n_rho+n_orb] = O_alpha(a.alpha, r_env);

    double feat[n_jas_par + 1];
    jastrow_O<double, double>(ws.x_sh.data(), s, t, feat);
    for (int m = 0; m < n_jas_par; m++) O_out[n_h+n_rho+n_orb+1+m] = feat[m];
}

void assemble_O(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out) {
    std::size_t n_params = a.n_params();
    if (O_out.size() != n_params) O_out.resize(n_params);

    psi(x, s, t, a, ws, true);
    const double S = S_sum(ws.drho.data(), ws.dets.data());

    fill_O(a, ws, S, O_out, s, t);
}
