#include "local_energy.h"
#include "constants.h"
#include "envelope.h"

#include <cassert>
#include <cmath>
#include <vector>

// Model O 2-body interaction
static double v_gauss_reg(double r2, double R) {
    return std::exp(-r2 / (R*R)) / (std::pow(3.14159265358979323846, 1.5) * R*R*R);
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
    if (!std::isfinite(S) || std::fabs(S) < psi_floor) {
        ws.n_node_hits++;
        return false;
    }

    std::vector<double> x_vec(x, x+D);
    std::vector<double> s_vec(s, s+N);
    std::vector<double> t_vec(t, t+N);

    // Declare Jet psi, throw error if it is different from double psi
    Jet pj = jpsi(x_vec, s_vec, t_vec, a, ws);
    bool psi_mismatch = std::fabs(pj.v - dpsi) > 1e-6 * std::max(1.0, std::fabs(dpsi));
    if (!std::isfinite(pj.v) || std::fabs(pj.v) < psi_floor || psi_mismatch) {
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

