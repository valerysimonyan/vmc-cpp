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


}