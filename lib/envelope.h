#pragma once

#include "constants.h"
#include "rng_common.h"

#include <cmath>
#include <type_traits>

namespace envelope {

// if T is number set to true, if true set type to T otherwise a double
template <typename T>
using scalar_t = std::conditional_t<std::is_arithmetic<T>::value, T, double>;

// Take cusp regulator and square it
template <typename T>
VMC_HD inline scalar_t<T> eps2() { 
    return scalar_t<T>(eps_env * eps_env); 
}

// Return position squared as template typename T 
template <typename T>
VMC_HD inline T r2(const T* x) {
    T s = T(0);
    for (int i = 0; i < D; i++) s += x[i] * x[i];
    return s;
}

// r_env = sqrt(r^2 + eps_env^2).
template <typename T>
VMC_HD inline T radius(const T& r2) {
    using std::sqrt;
    return sqrt(r2 + eps2<T>());
}

// Envelope floor
template <typename A>
VMC_HD inline A rate(A alpha) {
    using std::exp;
    return A(beta_min) + exp(alpha);
}

// Log of the envelope
template <typename T, typename A>
VMC_HD inline T log_factor(A alpha, const T& r_env) {
    using std::exp;
    return -(T(beta_min) + exp(alpha)) * r_env;
}

// Derivative with respect to envelope parameter of log|Ψ|
template <typename A>
VMC_HD inline A O_alpha(A alpha, A r_env) {
    using std::exp;
    return -exp(alpha) * r_env;
}

inline constexpr int n_params_env = 1 + n_jas_par;
static_assert(n_jas >= 0 && n_jas <= 8, "jas_basis defines 8 pair basis functions");
static_assert(n_j3 >= 0 && n_j3 <= 4, "jastrow3 defines 4 three-body widths");

// m is which basis function, s is |r_ij|^2, and we return g, the value of the Jastrow factor, with d1 and d2 being the first and 2nd derivative w.r.t. s
template <typename T>
VMC_HD inline T jas_basis(int m, T s, T& d1, T& d2) {
    using std::exp; using std::sqrt;

    // Bethe Peierls term for m = 0 g = R / sq(r2 + R2)
    if (m == 0) {
        const T q = s + T(jas_R * jas_R);
        const T g = T(jas_R) / sqrt(q);
        d1 = T(-0.5) * g / q;
        d1 = T(0.75) * g / (q * q);
        return g;
    }
    // Rest of the Jastrow factors are just Gaussians, for now with what seem to be random widths
    const T f = (m == 1) ? T(0.5) : (m == 2) ? T(0.8) : (m == 3) ? T(1.3) : (m == 4) ? T(2.0) : (m == 5) ? T(0.3) : (m == 6) ? T(3.0) : T(4.5);
    const T b2 = f * f * T(jas_R * jas_R);
    const T g = exp(-s / b2);
    d1 = -g / b2;
    d2 = g / (b2 * b2);
    return g;
}

// Take a pair of particles, take spin and isospin,s return index based on alignment of spin and isospin values
template <typename L>
VMC_HD inline int jas_cls(L si, L sj, L ti, L tj) {
    if (n_jas_cls == 1) return 0;
    return ((si * sj < L(0)) ? 2 : 0) + ((ti * tj < L(0)) ? 1 : 0);
}

// Evaluate cyclic |r_ij|^2 for three body Jastrow factor, compute grad if grad is not nullptr
template <typename T>
VMC_HD inline T jastrow3(const T* x, const T* c3, T* grad, T* lap, T* feat) {
    T J = T(0);
    // Iterate through all cyclic pairs
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            for (int k = j + 1; k < N; k++) {
                const int tri[3] = {i, j, k};
                for (int cyc = 0; cyc < 3; cyc++) {
                    const int pa = tri[cyc], pb = tri[(cyc + 1) % 3], pc = tri[(cyc + 2) % 3];
                    T d1[dim], d2[dim], A = T(0), B = T(0);
                    // Compute r_ij ^2m and r_jk ^2
                    for (int q = 0; q < dim; q++) { 
                        d1[q] = x[pa*dim + q] - x[pb*dim + q]; 
                        d2[q] = x[pb*dim + q] - x[pc*dim + q]; 
                        A += d1[q]*d1[q]; 
                        B += d2[q]*d2[q]; 
                    }
                    // Compute rik ^2
                    T dd = T(0);
                    for (int q = 0; q < dim; q++) { 
                        const T e = d1[q] - d2[q]; 
                        dd += e * e; 
                    }
                    // Based on value of m we compute and add to Jastrow factor c3[m] * exp (-(r_ij^2 + r_jk^2)/ (fct^2 * R3 ^2))
                    for (int m = 0; m < n_j3; m++) {
                        using std::exp;
                        const T fct = (m == 0) ? T(1.0) : (m == 1) ? T(1.6) : (m == 2) ? T(0.7) : T(2.5);
                        const T b2 = fct * fct * T(R3 * R3);
                        const T f = exp(-(A + B) / b2);
                        if (feat) feat[m] += f;
                        if (!c3) continue;
                        const T cf = c3[m] * f;
                        J += cf;
                        // Compute gradient and Laplacian if needed
                        if (grad) {
                            for (int q = 0; q < dim; q++) {
                                grad[pa*dim + q] += T(-2) * d1[q] / b2 * cf;
                                grad[pb*dim + q] += T(2) * (d1[q] - d2[q]) / b2 * cf;
                                grad[pc*dim + q] += T(2) * d2[q] / b2 * cf;
                            }
                            *lap += cf * (T(-8 * dim) / b2 + T(4) * (A + dd + B) / (b2 * b2));
                        }
                    }
                }
            }
        }
    }
    return J; 
}

// Compute total J 
template <typename T, typename L>
VMC_HD inline T jastrow(const T* x, const L* s, const L* t, const T* c, T* grad, T* lap) {
    // Initialize value gradient and Laplacian
    T J = T(0), Lp = T(0);
    if (grad) for (int a = 0; a < D; a++) grad[a] = T(0);

    // Iterate through particle pairs
    for (int i = 0; i < N && n_jas > 0; i++) {
        for (int j = i + 1; j < N; j++) {
            // Pick coefficient for pair
            const T* cc = c + (n_jas_cls == 1 ? 0 : n_jas * jas_cls(s[i], s[j], t[i], t[j]));
            T d[dim]; T r2 = T(0);
            // Evaluate r_ij^2
            for (int k = 0; k < dim; k++) { 
                d[k] = x[i*dim + k] - x[j*dim + k]; 
                r2 += d[k] * d[k]; 
            }
            // Evaluate u and its derivatives
            T u = T(0), u1 = T(0), u2 = T(0);
            for (int m = 0; m < n_jas; m++) {
                T g1, g2;
                const T g = jas_basis<T>(m, r2, g1, g2);
                u += cc[m] * g; u1 += cc[m] * g1; u2 += cc[m] * g2;
            }
            J += u;
            // Compute gradient if needed
            if (grad) {
                for (int k = 0; k < dim; k++) {
                    grad[i*dim + k] += T(2) * u1 * d[k];
                    grad[j*dim + k] -= T(2) * u1 * d[k];
                }
                Lp += T(2) * (T(2 * dim) * u1 + T(4) * r2 * u2);
            }
        }
    }
    // Add three particle cusps
    if (n_j3 > 0 && N >= 3) J += jastrow3<T>(x, c + n_jas_pair, grad, grad ? &Lp : nullptr, nullptr);
    if (lap) *lap = Lp;
    return J;
}

// J(labels_new) - J(labels_old) at fixed positions: the Jastrow factor of a spin/isospin
template <typename T, typename L>
VMC_HD inline T jastrow_dlabel(const T* x, const L* s0, const L* t0, const L* s1, const L* t1, const T* c) {
    // If only one class of Jastrow factor or no Jastrow basis functions may skip as no change to u
    if (n_jas_cls == 1 || n_jas == 0) return T(0);
    T dJ = T(0);
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            // If new positions and pairs have same index have same coefficient 
            const int c0 = jas_cls(s0[i], s0[j], t0[i], t0[j]), c1 = jas_cls(s1[i], s1[j], t1[i], t1[j]);
            if (c0 == c1) continue;
            // Evaluate |r_ij|^2
            T r2 = T(0);
            for (int k = 0; k < dim; k++) { 
                const T dk = x[i*dim + k] - x[j*dim + k]; 
                r2 += dk * dk; 
            }
            // Update to new coefficient
            for (int m = 0; m < n_jas; m++) { 
                T g1, g2; 
                const T g = jas_basis<T>(m, r2, g1, g2); 
                dJ += (c[n_jas*c1 + m] - c[n_jas*c0 + m]) * g; 
            }
        }
    }
    return dJ;
}


// dJ/dc[cls][m] = sum over pairs of that class of g_m(r_ij^2): the Jastrow columns of O, derivative on c's 
template <typename T, typename L>
VMC_HD inline void jastrow_O(const T* x, const L* s, const L* t, T* feat) {
    // Initialize parameter gradient array
    for (int m = 0; m < n_jas_par; m++) feat[m] = T(0);
    // Go over pairs, evaluate |r_ij|^2, then compute dlog|Ψ|
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            const int off = (n_jas_cls == 1) ? 0 : n_jas * jas_cls(s[i], s[j], t[i], t[j]);
            T r2 = T(0);
            for (int k = 0; k < dim; k++) { 
                const T dk = x[i*dim + k] - x[j*dim + k]; 
                r2 += dk * dk; 
            }
            for (int m = 0; m < n_jas; m++) { 
                T g1, g2; 
                feat[off + m] += jas_basis<T>(m, r2, g1, g2); 
            }
        }
    }
    if (n_j3 > 0 && N >= 3) jastrow3<T>(x, (const T*)nullptr, nullptr, nullptr, feat + n_jas_pair);
}



}