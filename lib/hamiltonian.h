#pragma once

#include "wavefunction.h"
#include "envelope.h"
#include "constants.h"
#include "rng_common.h"

#include <cmath>
#include <cstddef>
#include <vector>

// All but local_E are called on device and host so we put in header to avoid duplicate definitions
inline constexpr double pi_1p5 = 5.568327996831708;  // pi^(3/2)

// Exchange types for pairwaise potential
inline constexpr int EX_S  = 0;  // Swap s_i <-> s_j
inline constexpr int EX_T  = 1;  // Swap t_i <-> t_j
inline constexpr int EX_ST = 2;  // Swap both       

// Kinetic energy operator, T = -hbar^2/(2m) * (∇^2 Ψ) / Ψ
template<typename T>
VMC_HD inline T kinetic(T lap, T psi) {
    return -T(hbar2_2m) * lap / psi;
}

// Return the regulated Gaussian potential, v(r) = exp(-r^2/R^2) / (pi^(3/2) R^3) for two particle interactions
template <typename T>
VMC_HD inline T v_reg(T r2, double R) {
    using std::exp;
    return exp(-r2 / T(R*R)) / (T(pi_1p5) * T(R)*T(R)*T(R));
}

// Pairwise potential term, V = (sum_{i<j} v(r_ij) Ψ) / Ψ
template <typename T>
VMC_HD inline T pair_term(T r2, T R_s, T R_t, T R_st) {
    const T v01 = v_reg(r2, R01);  // S=0 T=1 channel
    const T v10 = v_reg(r2, R10);  // S=1 T=0 channel
    return T(hbarc/4.0) * (T(C01)*v01*(T(1.0) + R_t - R_s - R_st) + T(C10)*v10*(T(1.0) - R_t + R_s - R_st));
}

// Depending on exchange type, determine if exchange is needed
template <typename L>
VMC_HD inline bool ex_slot_active(L si, L ti, L sj, L tj, int type) {
    const bool same_s = (si == sj), same_t = (ti == tj);
    if (same_s && same_t) return false;
    if (same_s) return type == EX_T;
    if (same_t) return type == EX_S;
    return true;
}

// Depending on exchange type, return the new labels after exchange
template <typename L>
VMC_HD inline void ex_swapped_labels(L si, L ti, L sj, L tj, int type, L& sni, L& tni, L& snj, L& tnj) {
    if (type == EX_T) { 
        sni = si; 
        tni = tj; 
        snj = sj; 
        tnj = ti; 
    }
    else if (type == EX_S) { 
        sni = sj; 
        tni = ti; 
        snj = si; 
        tnj = tj; 
    }
    else { 
        sni = sj; 
        tni = tj; 
        snj = si; 
        tnj = ti; 
    }
}

// Jastrow factor exchange
template <typename T, typename L>
VMC_HD inline void jastrow_exchange(const T* x, const L* s, const L* t, int i, int j, const T* jc, T& R_s, T& R_t, T& R_st) {
    // If only one class of Jastrow factor or no Jastrow basis functions may skip as no change to u
    using std::exp;
    if (n_jas_cls <= 1) return;
    
    // Compute the change in Jastrow factor for the spin and isospin exchanges
    L s1[N]; 
    L t1[N];
    for (int q = 0; q < N; q++) { 
        s1[q] = s[q]; 
        t1[q] = t[q]; 
    }

    // Spin exchange
    s1[i] = s[j]; 
    s1[j] = s[i];                                    
    const T dS = envelope::jastrow_dlabel<T, L>(x, s, t, s1, t, jc);

    // Isospin exchange
    t1[i] = t[j]; 
    t1[j] = t[i];                                               
    const T dT = envelope::jastrow_dlabel<T, L>(x, s, t, s, t1, jc);          

    // Both spin and isospin exchange
    const T dST = envelope::jastrow_dlabel<T, L>(x, s, t, s1, t1, jc);

    // Evalute the ratio of the Jastrow factors for the exchanges
    R_s *= exp(dS); 
    R_t *= exp(dT); 
    R_st *= exp(dST);
}

// 2N potential, V = (sum_{i<j} v(r_ij) Ψ) / Ψ, with spin/isospin exchange ratios
template <typename T, typename L, typename F>
VMC_HD inline T V_2N(const T* x, const L* s, const L* t, const T* jc, F ratio) {
    T V = T(0);
    int p = -1;
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            p++;
            const bool same_s = (s[i] == s[j]);
            const bool same_t = (t[i] == t[j]);
            if (same_s && same_t) continue;   // no exchange term between equal spins and equal isospins

            T r2 = T(0);
            for (int d = 0; d < dim; d++) {
                const T diff = x[i*dim + d] - x[j*dim + d];
                r2 += diff * diff;
            }

            T R_s, R_t, R_st;
            if (same_s) {
                R_s = T(1.0);
                R_t = ratio(p, i, j, EX_T);
                R_st = R_s * R_t;
            } else if (same_t) {
                R_t = T(1.0);
                R_s = ratio(p, i, j, EX_S);
                R_st = R_s * R_t;
            } else {
                R_t = ratio(p, i, j, EX_T);
                R_s = ratio(p, i, j, EX_S);
                R_st = ratio(p, i, j, EX_ST);
            }
            jastrow_exchange(x, s, t, i, j, jc, R_s, R_t, R_st);  // spin/isospin-dependent Jastrow
            V += pair_term(r2, R_s, R_t, R_st);
        }
    }
    return V;
}

// Return three partile potential V_3N = (sum_{i<j<k} V_3N(r_ij, r_jk, r_ki) Ψ) / Ψ, also Gaussian regulated where V_3N(r_ij, r_jk, r_ki) = exp(-(r_ij^2 + r_ki^2)/R^2) + exp(-(r_ij^2 + r_jk^2)/R^2) + exp(-(r_jk^2 + r_ki^2)/R^2)
template <typename T>
VMC_HD inline T V_3N(const T* x) {
    using std::exp;
    if (N < 3) return T(0);
    T V = T(0);

    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                T rij2 = T(0), rjk2 = T(0), rki2 = T(0);
                for (int d = 0; d < dim; d++) {
                    const T dij = x[i*dim + d] - x[j*dim + d];
                    const T djk = x[j*dim + d] - x[k*dim + d];
                    const T dki = x[k*dim + d] - x[i*dim + d];
                    rij2 += dij * dij;
                    rjk2 += djk * djk;
                    rki2 += dki * dki;
                }
                V += exp(-(rki2+rij2)/T(R3*R3));  // i centre
                V += exp(-(rij2+rjk2)/T(R3*R3));  // j centre
                V += exp(-(rjk2+rki2)/T(R3*R3));  // k centre
            }
        }
    }
    return T(V3_0) * V;
}

// Return the shape of the Coulomb potential, v(r) = (1 - (1 + b r + b^2 r^2 / 3) exp(-b r)) / r 
template <typename T>
VMC_HD inline T coulomb_shape(T r) {
    using std::exp;
    const T x = T(b_coul) * r;

    // For small r, use Taylor expansion to avoid numerical issues
    if (x < T(1e-3)) return T(b_coul) * (T(5.0/16.0) - x*x/T(96.0));

    const T F = T(1.0) - (T(1.0) + T(11.0)*x/T(16.0) + T(3.0)*x*x/T(16.0) + x*x*x/T(48.0)) * exp(-x);
    return F / r;
}

// If neutron skip, otherwise evaluate Coulomb interaction between them
template <typename T, typename L>
VMC_HD inline T V_coulomb(const T* x, const L* t) {
    using std::sqrt;
    T V = T(0);
    for (int i = 0; i < N; i++) {
        if (t[i] < L(0)) continue;
        for (int j = i+1; j < N; j++) {
            if (t[j] < L(0)) continue;
            T r2 = T(0);
            for (int d = 0; d < dim; d++) {
                const T diff = x[i*dim + d] - x[j*dim + d];
                r2 += diff * diff;
            }
            V += coulomb_shape(sqrt(r2));
        }
    }
    return T(alpha_em) * T(hbarc) * V;
}

// Return the local angular momentum operator L^2 = -(sum_i r_i x ∇_i)^2 Ψ / Ψ = ((sum_i r_i x ∇_i)Ψ / Ψ)^2 
template <typename T>
VMC_HD inline T l2_local(const T* x_sh, const T* g, std::size_t g_stride, T psi) {
    static_assert(dim == 3, "l2_local's cross product assumes dim == 3");
    T Lx = T(0), Ly = T(0), Lz = T(0);
    for (int i = 0; i < N; i++) {
        const T rx = x_sh[i*dim + 0];
        const T ry = x_sh[i*dim + 1];
        const T rz = x_sh[i*dim + 2];
        const T gx = g[(std::size_t)(i*dim + 0) * g_stride];
        const T gy = g[(std::size_t)(i*dim + 1) * g_stride];
        const T gz = g[(std::size_t)(i*dim + 2) * g_stride];
        
        Lx += ry*gz - rz*gy;
        Ly += rz*gx - rx*gz;
        Lz += rx*gy - ry*gx;
    }
    return (Lx*Lx + Ly*Ly + Lz*Lz) / (psi*psi);
}

bool local_E(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out, double& E_out);
