#include "network.h"
#include "constants.h"
#include "physics.h"
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
static void shift_to_com(const double* x, Workspace& ws) {
    if (ws.x_sh.size() != (std::size_t)D) ws.x_sh.resize(D);
    for (int d = 0; d < dim; d++) {
        double R_cm_d = 0.0;
        for (int i = 0; i < N; i++) R_cm_d += x[i*dim + d] / N;
        for (int i = 0; i < N; i++) ws.x_sh[i*dim+d]= x[i*dim + d] - R_cm_d;
    }
}

template<typename T>
static T psi_impl(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, bool need_inv) {
    constexpr bool is_jet = std::is_same_v<T,Jet>;
    using std::exp;

    EvalBuffers<T> buf = get_buffers<T>(ws);

    // Depending on type represent position and spins differently
    if constexpr (is_jet) {
        if (ws.jin.size() != (std::size_t)D) ws.jin.resize(D);
        for (int i = 0; i < D; i++) ws.jin[i] = Jet::input(x[i],i);
    }

    auto coord_raw =[&](int i) -> T {
        if constexpr (is_jet) return ws.jin[i];
        else return x[i];
    };

    // Subtract COM coords dimension by dimmnsion
    if constexpr (is_jet) {
        if (ws.jin_sh.size() != (std::size_t)D) ws.jin_sh.resize(D);
        for (int d = 0; d < dim; d++) {
            T R_cm_d{};
            for (int i = 0; i < N; i++) R_cm_d = R_cm_d + coord_raw(i*dim + d) / N;
            for (int i = 0; i < N; i++) ws.jin_sh[i*dim + d] = coord_raw(i*dim + d) - R_cm_d;
        }
    } else {
        shift_to_com(x, ws);
    }

    auto coord = [&](int i) -> T {
        if constexpr (is_jet) return ws.jin_sh[i];
        else return ws.x_sh[i];
    };

    auto spin = [&](int i) -> T {
        if constexpr (is_jet) return T(s[i]);
        else return s[i];
    };
    
    auto iso = [&](int i) -> T {
        if constexpr (is_jet) return T(t[i]);
        else return t[i];
    };


    // Forward pass also dependent on data type, depending on if we are taking a derivative in which case we need inverse, either do a regular forward pass or cached
    auto h_forward = [&](int i) -> std::vector<T>* {
        if constexpr (!is_jet) {
            if (need_inv) {
                a.h_net.forward_cached(buf.single, ws.h_caches[i], ws.fc_out);
                return &ws.fc_out;
            }
        }
        return a.h_net.forward_opt<T>(buf.single, a.h_net.params, buf.buf_a, buf.buf_b);
    };
    auto rho_forward = [&]() -> std::vector<T>* {
        if constexpr (!is_jet) {
            if (need_inv) {
                a.rho_net.forward_cached(buf.xi, ws.rho_cache, ws.fc_out);
                return &ws.fc_out;
            }
        }
        return a.rho_net.forward_opt<T>(buf.xi, a.rho_net.params, buf.buf_a, buf.buf_b);
    };
    auto orb_forward = [&](int i) -> std::vector<T>* {
        if constexpr (!is_jet) {
            if (need_inv) {
                a.orb_net.forward_cached(buf.single, ws.orb_caches[i], ws.fc_out);
                return &ws.fc_out;
            }
        }
        return a.orb_net.forward_opt<T>(buf.single, a.orb_net.params, buf.buf_a, buf.buf_b);
    };

    // Prepare buffers for use, then create xi = sum h(r_i)
    if (buf.single.size() != (std::size_t)(dim+1+1)) buf.single.resize(dim+1+1);
    if (buf.xi.size() != (std::size_t)m_feat) buf.xi.resize(m_feat);
    else std::fill(buf.xi.begin(), buf.xi.end(), T(0.0));
    if constexpr (!is_jet) {
        if (need_inv && ws.h_caches.size() != (std::size_t)N) ws.h_caches.resize(N);
    }    
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dim; d++) buf.single[d] = coord(i*dim + d);
        buf.single[dim] = spin(i);
        buf.single[dim+1] = iso(i);

        std::vector<T>* h_out = h_forward(i);
        for (int f = 0; f < m_feat; f++) buf.xi[f] = buf.xi[f] + (*h_out)[f];
    }
    
    // Create rho(xi)
    if (buf.rho.size() != (std::size_t)K) buf.rho.resize(K);
    std::vector<T>* rho_out = rho_forward();
    for (int i = 0; i < K; i++) buf.rho[i] = (*rho_out)[i];

    // Create orbital matrices (phi_k(ri))_j
    if constexpr (!is_jet) {
        if (need_inv && ws.orb_caches.size() != (std::size_t)N) ws.orb_caches.resize(N);
    }
    if (buf.M.size() != (std::size_t)K*N*N) buf.M.resize(K*N*N);
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dim; d++) buf.single[d] = coord(i*dim + d);
        buf.single[dim] = spin(i);
        buf.single[dim+1] = iso(i);
        std::vector<T>* orb_out = orb_forward(i);
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                buf.M[j*(N*N)+k*N+i] = (*orb_out)[j*N+k];
            }
        }
    }
   
    // Compute determinant of orbital matrices det phi_k, then compute sum rho_k det phi_k
    T sum{};
    if constexpr (!is_jet) {
        if (ws.dets.size() != (std::size_t)K) ws.dets.resize(K);
        if (buf.M_scratch.size() != (std::size_t)N*N) buf.M_scratch.resize(N*N);
        if (need_inv && ws.dMinv.size() != (std::size_t)(K*N*N)) ws.dMinv.resize(K*N*N);
    }
    for (int i = 0; i < K; i++) {
        T det{};
        if constexpr (is_jet) {
            det = det_jet_from_minv(&buf.M[i*(N*N)], N, ws.jd_Mval, ws.jd_Minv, ws.jd_piv, ws.jd_col, ws.jd_G, ws.jd_B);
        } else {
            for (int j = 0; j < N*N; j++) buf.M_scratch[j] = buf.M[i*N*N + j];
            if (need_inv) {
                det = lu_det_inv(buf.M_scratch, N, ws.dMinv_k, ws.piv, ws.col_scratch);
                for (int j = 0; j < N*N; j++) ws.dMinv[i*N*N + j] = ws.dMinv_k[j];
            } else {
                det = lu_det<double>(buf.M_scratch, N, ws.piv);
            }
        }
        if constexpr (!is_jet) ws.dets[i] = det;
        sum = sum + buf.rho[i] * det;
    }
    
    // Envelope
    T r2{};
    for (int i = 0; i < D; i++) r2 = r2 + coord(i) * coord(i);
    double xs[D];
    for (int i = 0; i < D; i++) { 
        if constexpr (is_jet) xs[i] = coord(i).v; 
        else xs[i] = coord(i); 
    }
    if constexpr (is_jet) {
        Jet Jj;
        Jj.v = envelope::jastrow<double, double>(xs, s, t, a.jc.data(), Jj.g.data(), &Jj.l);
        return exp(envelope::log_factor(a.alpha, envelope::radius(r2)) + Jj) * sum;
    } else {
        const double Jv = envelope::jastrow<double, double>(xs, s, t, a.jc.data(), nullptr, nullptr);
        return exp(envelope::log_factor(a.alpha, envelope::radius(r2)) + Jv) * sum;
    }
}



// Double evaluation of psi 
double psi (const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, bool need_inv) {
    return psi_impl<double>(x, s, t, a, ws, need_inv);
}
double psi (const std::vector<double>& x, const std::vector<double>& s, const std::vector<double>& t, const Ansatz& a, Workspace& ws, bool need_inv) {
    return psi(x.data(), s.data(), t.data(), a, ws, need_inv);
}

// Store log psi for regular use, if on node return 0
double log_p (const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws) {
    double p = psi(x, s, t, a, ws);
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    return std::log(std::abs(p));
}
double log_p (const std::vector<double>& x, const std::vector<double>& s, const std::vector<double>& t, const Ansatz& a, Workspace& ws) {
    return log_p(x.data(),s.data(),t.data(), a, ws);
}

// Jet evaluation of psi
Jet jpsi (const std::vector<double>& x, const std::vector<double>& s, const std::vector<double>& t, const Ansatz& a, Workspace& ws) {
     return psi_impl<Jet>(x.data(), s.data(), t.data(), a, ws, false);
}

// Model O 2-body interaction
static double v_gauss_reg(double r2, double R) {
    return std::exp(-r2 / (R*R)) / (std::pow(3.14159265358979323846, 1.5) * R*R*R);
}

// Preompute h_net and orb_net for all values of spin and isospin
void build_st_table(const double* x, const Ansatz& a, Workspace& ws) {
    shift_to_com(x, ws);

    if(ws.tab_h.size() != (std::size_t)N*4*m_feat) ws.tab_h.resize((std::size_t)N*4*m_feat);
    if(ws.tab_orb.size() != (std::size_t)N*4*K*N) ws.tab_orb.resize((std::size_t)N*4*K*N  );
    if(ws.dsingle.size() != (std::size_t)(dim+2)) ws.dsingle.resize((std::size_t)(dim+2));

    static const double s_of[4] = {1.0, -1.0, 1.0, -1.0};
    static const double t_of[4] = {1.0, 1.0, -1.0, -1.0};

    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dim; d++) ws.dsingle[d] = ws.x_sh[i*dim+d];
        for (int c = 0; c < 4; c++) {
            ws.dsingle[dim] = s_of[c];
            ws.dsingle[dim+1] = t_of[c];

            std::vector<double>* h_out = a.h_net.forward_opt<double>(ws.dsingle, a.h_net.params, ws.dbuf_a, ws.dbuf_b);
            double* dst_h = &ws.tab_h[((std::size_t)i*4 + c)*m_feat];
            for (int f = 0; f < m_feat; f++) dst_h[f] = (*h_out)[f];

            
            std::vector<double>* orb_out = a.orb_net.forward_opt<double>(ws.dsingle, a.orb_net.params, ws.dbuf_a, ws.dbuf_b);
            double* dst_orb = &ws.tab_orb[((std::size_t)i*4 + c)*(K*N)];
            for (int j = 0; j < K*N; j++) dst_orb[j] = (*orb_out)[j];
        }
    }
    ws.table_valid = true;
}
void build_st_table(const std::vector<double>& x, const Ansatz& a, Workspace& ws) {
    build_st_table(x.data(), a, ws);
}

// Given a value of s and t draw from precomputed table instead of computing from scratch
double S_from_table(const double* s, const double* t, const Ansatz& a, Workspace& ws) {
    assert(ws.table_valid);

    if (ws.tab_xi.size() != (std::size_t)m_feat) ws.tab_xi.resize(m_feat);
    std::fill(ws.tab_xi.begin(), ws.tab_xi.end(), 0.0);
    for (int i = 0; i < N; i++) {
        int c = st_combo(s[i],t[i]);
        const double* h_i = &ws.tab_h[((std::size_t)i*4 + c)*m_feat];
        for (int f = 0; f < m_feat; f++) ws.tab_xi[f] += h_i[f];
    }

    if (ws.tab_rho.size() != (std::size_t)K) ws.tab_rho.resize(K);
    std::vector<double>* rho_out = a.rho_net.forward_opt<double> (ws.tab_xi, a.rho_net.params, ws.dbuf_a, ws.dbuf_b);
    for (int i = 0; i < K; i++) ws.tab_rho[i] = (*rho_out)[i];

    if (ws.tab_M.size() != (std::size_t)N*N) ws.tab_M.resize((std::size_t)N*N); 

    double S = 0.0;
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) {
            int c = st_combo(s[i], t[i]);
            const double* orb_i = &ws.tab_orb[((std::size_t)i*4 +c)*(K*N)];
            for (int k = 0; k < N; k++) ws.tab_M[k*N + i] = orb_i[j*N + k];
        }
        double det = lu_det<double>(ws.tab_M, N, ws.tab_piv);
        S += ws.tab_rho[j] * det;
    }
    return S;
}
double S_from_table(const std::vector<double>& s, const std::vector<double>& t, const Ansatz& a, Workspace& ws) {
    return S_from_table(s.data(), t.data(), a, ws);
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
double swap_ratio(const std::vector<double>& s, const std::vector<double>& t, int i, int j, double s_new_i, double t_new_i, double s_new_j, double t_new_j, double S0, bool use_rank2, const Ansatz& a, Workspace& ws) {
    if (use_rank2) {
        int c_old_i = st_combo(s[i], t[i]);
        int c_old_j = st_combo(s[j], t[j]);
        int c_new_i = st_combo(s_new_i, t_new_i);
        int c_new_j = st_combo(s_new_j, t_new_j);

        // Evaluate updated xi, as xi is just sum of h's just subtract the old modified h and add the new one
        if (ws.sm_xi.size() != (std::size_t)m_feat) ws.sm_xi.resize(m_feat);
        const double* h_old_i = &ws.tab_h[((std::size_t)i*4 + c_old_i) * m_feat];
        const double* h_new_i = &ws.tab_h[((std::size_t)i*4 + c_new_i) * m_feat];
        const double* h_old_j = &ws.tab_h[((std::size_t)j*4 + c_old_j) * m_feat];
        const double* h_new_j = &ws.tab_h[((std::size_t)j*4 + c_new_j) * m_feat];
        for (int f = 0; f < m_feat; f++) ws.sm_xi[f] = ws.dxi[f] + (h_new_i[f]-h_old_i[f]) + (h_new_j[f]-h_old_j[f]);

        if (ws.sm_rho.size() != (std::size_t)K) ws.sm_rho.resize(K);
        std::vector<double>* rho_out = a.rho_net.forward_opt<double>(ws.sm_xi, a.rho_net.params, ws.dbuf_a, ws.dbuf_b);
        for (int k = 0; k < K; k++) ws.sm_rho[k] = (*rho_out)[k];

        // Record column change
        if (ws.sm_dci.size() != (std::size_t)K*N) ws.sm_dci.resize((std::size_t)K*N);
        if (ws.sm_dcj.size() != (std::size_t)K*N) ws.sm_dcj.resize((std::size_t)K*N);
        const double* orb_old_i = &ws.tab_orb[((std::size_t)i*4 + c_old_i) * (K*N)];
        const double* orb_new_i = &ws.tab_orb[((std::size_t)i*4 + c_new_i) * (K*N)];
        const double* orb_old_j = &ws.tab_orb[((std::size_t)j*4 + c_old_j) * (K*N)];
        const double* orb_new_j = &ws.tab_orb[((std::size_t)j*4 + c_new_j) * (K*N)];
        for (int kn = 0; kn < K*N; kn++) {
            ws.sm_dci[kn] = orb_new_i[kn]-orb_old_i[kn];
            ws.sm_dcj[kn] = orb_new_j[kn]-orb_old_j[kn];
        }   

        double S_new = 0.0;
        for(int k = 0; k < K; k++) {
            double ratio2 = det_ratio_rank2(&ws.dMinv[(std::size_t)k*N*N], N, &ws.sm_dci[(std::size_t)k*N], i, &ws.sm_dcj[(std::size_t)k*N], j);
            S_new += ws.sm_rho[k] * ws.dets[k] * ratio2;
        }
        return S_new/S0;
    }

    // Fallback: Old way of computing update
    double si = ws.s_swap[i], ti = ws.t_swap[i], sj = ws.s_swap[j], tj = ws.t_swap[j];
    ws.s_swap[i] = s_new_i; ws.t_swap[i] = t_new_i;
    ws.s_swap[j] = s_new_j; ws.t_swap[j] = t_new_j;
    double R = S_from_table(ws.s_swap, ws.t_swap, a, ws) / S0;
    ws.s_swap[i] = si; ws.t_swap[i] = ti;
    ws.s_swap[j] = sj; ws.t_swap[j] = tj;
    return R;
}

// Model O 3-body interaction
double V_3N(const std::vector<double>& x) {
    if (N < 3) return 0;

    double V = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N ; j++) {
            for (int k = j+1; k < N; k++) {
                double rij2 = 0.0, rjk2 = 0.0, rki2 = 0.0;
                for (int d = 0; d < dim; d++) {
                    double dij = x[i*dim + d] - x[j*dim + d];
                    double djk = x[j*dim + d] - x[k*dim + d];
                    double dki = x[k*dim + d] - x[i*dim + d];
                    
                    rij2 += dij * dij;
                    rjk2 += djk * djk;
                    rki2 += dki * dki;
                }
                V += std::exp(-(rki2+rij2)/(R3*R3)); // i center
                V += std::exp(-(rij2+rjk2)/(R3*R3)); // j center
                V += std::exp(-(rjk2+rki2)/(R3*R3)); // k center
            }
        }
    }
    return V3_0 * V;
}

// Add regulator where beyond certain distance just do a small x expansion
static double coulomb_shape(double r) {
    double x = b_coul * r;
    if (x < 1e-3) return b_coul * (5.0/16.0 - x*x/96.0);
    double F = 1.0 - (1.0 + 11.0*x/16.0 + 3.0*x*x/16.0 + x*x*x/48.0) * std::exp(-x);
    return F / r;
}

// Coulomb interaction between protons, account for finite size effects
double V_coulomb(const std::vector<double>& x, const std::vector<double>& t) {
    double V = 0.0;
    for (int i = 0; i < N; i++) {
        if (t[i] < 0.0) continue;
        for (int j = i+1; j < N; j++) {
            if (t[j] < 0.0) continue;
            double r2 = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = x[i*dim + d] - x[j*dim + d];
                r2 += diff * diff;
            }
            V += coulomb_shape(std::sqrt(r2));
        }
    }
    return alpha_em * hbarc * V;
}

// Act on Psi with \int (L Psi)*(L Psi)
double l2_local(const double* x_shifted, const double* grad, double psi_val) {
    static_assert(dim == 3, "l2_local's cross product assumes dim == 3");
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;

    for (int i = 0; i < N; i++) {
        double rx = x_shifted[i*dim + 0], ry = x_shifted[i*dim + 1], rz = x_shifted[i*dim + 2];
        double gx = grad[i*dim+0], gy = grad[i*dim+1], gz = grad[i*dim+2];
        Lx += ry*gz - rz*gy;
        Ly += rz*gx - rx*gz;
        Lz += rx*gy - ry*gx;
    }

    return (Lx*Lx + Ly*Ly + Lz*Lz) / (psi_val*psi_val);

}

static void fill_O(const Ansatz& a, Workspace& ws, double S, std::vector<double>& O_out, const double* s, const double* t) {
    std::size_t n_h = a.h_net.params.size();
    std::size_t n_rho = a.rho_net.params.size();
    std::size_t n_orb = a.orb_net.params.size();

    if (ws.seed_rho.size() != (std::size_t)K) ws.seed_rho.resize(K);
    for (int i = 0; i < K; i++) ws.seed_rho[i] = ws.dets[i];
    a.rho_net.backprop(ws.rho_cache, ws.seed_rho, ws.dtheta_rho, ws.delta_a, ws.delta_b, &ws.dpsi_dxi);
    for (std::size_t p = 0; p < n_rho; p++) O_out[n_h + p] = ws.dtheta_rho[p]/S;

    ws.dtheta_h.assign(n_h, 0.0);
    for (int i = 0; i < N; i++) a.h_net.backprop_acc(ws.h_caches[i], ws.dpsi_dxi, ws.dtheta_h, ws.delta_a, ws.delta_b);
    for (std::size_t p = 0; p < n_h; p++) O_out[p] = ws.dtheta_h[p]/S;

    if (ws.seed_orb.size() != (std::size_t)(K*N)) ws.seed_orb.resize(K*N);
    ws.dtheta_orb.assign(n_orb, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                ws.seed_orb[j*N+k] = ws.drho[j] * ws.dets[j] * ws.dMinv[j*(N*N) + i*N + k];
            }
        }
        a.orb_net.backprop_acc(ws.orb_caches[i], ws.seed_orb, ws.dtheta_orb, ws.delta_a, ws.delta_b);
    }
    for (std::size_t p = 0; p < n_orb; p++) O_out[n_h + n_rho + p] = ws.dtheta_orb[p]/S;

    const double r_env = envelope::radius(envelope::r2(ws.x_sh.data()));
    O_out[n_h+n_rho+n_orb] = envelope::O_alpha(a.alpha, r_env);

    double feat[n_jas_par + 1];
    envelope::jastrow_O<double, double>(ws.x_sh.data(), s, t, feat);
    for (int m = 0; m < n_jas_par; m++) O_out[n_h+n_rho+n_orb+1+m] = feat[m];


}

void assemble_O(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out) {
    std::size_t n_params = a.n_params();
    if (O_out.size() != n_params) O_out.resize(n_params);

    psi(x, s, t, a, ws, true);

    double S = 0.0;
    for (int i = 0; i < K; i++) {
        S += ws.drho[i] * ws.dets[i];
    }
    fill_O(a, ws, S, O_out, s, t);
}

// Log local energy and dlogpsi_d theta_i simultaneously as more computationally efficient
bool local_E(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out, double& E_out) {
    std::size_t n_params = a.n_params();
    if(O_out.size() != n_params) O_out.resize(n_params);

    double dpsi = psi(x, s, t, a, ws, true);
    (void)dpsi;

    double S = 0.0;
    for (int i = 0; i < K; i++) {
        S += ws.drho[i] * ws.dets[i];
    }

    // Return 0 if we hit a node, as parameters diverge we set them to zero
    if (!std::isfinite(S) || std::fabs(S) < 1e-290) {
        ws.n_node_hits++;
        return false;
    }

    std::vector<double> x_vec(x, x+D);
    std::vector<double> s_vec(s, s+N);
    std::vector<double> t_vec(t, t+N);

    // Declare Jet psi, throw error if it is different from double psi
    Jet pj = jpsi(x_vec, s_vec, t_vec, a, ws);
    bool psi_mismatch = std::fabs(pj.v - dpsi) > 1e-6 * std::max(1.0, std::fabs(dpsi));
    if (!std::isfinite(pj.v) || std::fabs(pj.v) < 1e-290 || psi_mismatch) {
        ws.n_node_hits++;
        return false;
    }

    // Record L^2
    ws.l2_val = l2_local(ws.x_sh.data(), pj.g.data(), pj.v);

    // Now we build E_loc
    double E_loc = -hbar2_2m * (pj.l/pj.v);
    if (!std::isfinite(E_loc)) {
        ws.n_node_hits++;
        return false;
    }

    if (nuc_3N) {
        E_loc += V_3N(x_vec);
        if (!std::isfinite(E_loc)) {
            ws.n_node_hits++;
            return false;
        }
    }

    if (nuc_coulomb) {
        E_loc += V_coulomb(x_vec, t_vec);
        if (!std::isfinite(E_loc)) {
            ws.n_node_hits++;
            return false;
        }
    }

    if (nuc_pot != NucPot::Off) {
        build_st_table(x, a, ws);
        double S0 = S_from_table(s, t, a, ws);
        assert(std::fabs(S0 - S) < 1e-10 * std::max(1.0, std::fabs(S)));

        bool use_rank2 = rank2_well_conditioned(ws);

        if (ws.s_swap.size() != (std::size_t)N) ws.s_swap.resize(N);
        if (ws.t_swap.size() != (std::size_t)N) ws.t_swap.resize(N);
        for (int i = 0; i < N; i++) {
            ws.s_swap[i] = s[i];
            ws.t_swap[i] = t[i];
        }
        double V_nuc = 0.0;

        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                bool same_s = (s[i] == s[j]);
                bool same_t = (t[i] == t[j]);
                if (same_s && same_t) continue; // Either spin or isospin are antisymmetric

                double r2 = 0.0; 
                for (int d = 0; d < dim; d++) {
                    double diff = x[i*dim + d] - x[j*dim + d];
                    r2 += diff * diff;
                }

                double v01 = v_gauss_reg(r2, R01);
                double v10 = v_gauss_reg(r2, R10);

                double R_s, R_t, R_st;
                if (same_s) {
                    R_s = 1.0;
                    R_t = swap_ratio(s_vec, t_vec, i, j, s[i], t[j], s[j], t[i], S0, use_rank2, a, ws);
                    R_st = R_s * R_t;
                } else if (same_t) {
                    R_t = 1.0;
                    R_s = swap_ratio(s_vec, t_vec, i, j, s[j], t[i], s[i], t[j], S0, use_rank2, a, ws);
                    R_st = R_s * R_t;
                } else {
                    R_t = swap_ratio(s_vec, t_vec, i, j, s[i], t[j], s[j], t[i], S0, use_rank2, a, ws);
                    R_s = swap_ratio(s_vec, t_vec, i, j, s[j], t[i], s[i], t[j], S0, use_rank2, a, ws);
                    R_st = swap_ratio(s_vec, t_vec, i, j, s[j], t[j], s[i], t[i], S0, use_rank2, a, ws);
                }
                // Channel-dependent Jastrow: exchange ratios pick up exp(dJ)
                if (n_jas_cls > 1) {   
                    double s1[N], t1[N];
                    for (int q = 0; q < N; q++) { 
                        s1[q] = s[q]; 
                        t1[q] = t[q]; 
                    }
                    s1[i] = s[j]; s1[j] = s[i];
                    const double dS = envelope::jastrow_dlabel<double, double>(x, s, t, s1, t, a.jc.data());
                    t1[i] = t[j]; t1[j] = t[i];
                    const double dST = envelope::jastrow_dlabel<double, double>(x, s, t, s1, t1, a.jc.data());
                    const double dT = envelope::jastrow_dlabel<double, double>(x, s, t, s, t1, a.jc.data());
                    R_s *= std::exp(dS); 
                    R_t *= std::exp(dT); 
                    R_st *= std::exp(dST);
                }
                V_nuc += (hbarc/4.0) * (C01*v01*(1.0 + R_t - R_s - R_st) + C10*v10*(1.0 - R_t + R_s - R_st));
            }
        }
        E_loc += V_nuc;

        if (!std::isfinite(E_loc)) {
            ws.n_node_hits++;
            return false;
        }
    }
    fill_O(a, ws, S, O_out, s, t);

    E_out = E_loc;
    return true;
}

