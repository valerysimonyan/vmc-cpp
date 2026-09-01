#pragma once

#include "constants.h"

#include <cmath>
#include <array>

// Describe a Jet object in terms of its value, gradient, and Laplacian
struct Jet {
    double v;
    std::array<double, D> g;
    double l;

    // Intialize Jet  
    Jet(double c = 0.0) : v(c), g{}, l(0.0) {}
    
    // Acts on existing jet object by setting its value and the value of the gradient
    static Jet input(double xi, int i) {
        Jet out(xi);
        out.g[i] = 1.0;
        return out;
    }
};

// Define addition operation between Jets and Jets and between Jets and constants
inline Jet operator+ (const Jet& a, const Jet& b) {
    Jet out;
    out.v = a.v + b.v;
    for (int i = 0; i < D; i++) {
        out.g[i] = a.g[i] + b.g[i];
    }
    out.l = a.l + b.l;
    return out;
}

inline Jet operator+ (const Jet& a, double c) {
    Jet out = a;
    out.v += c;
    return out;
}

inline Jet operator+ (double c, const Jet& a) {
    return a + c;
}

// Define multiplication operation between Jets and Jets and between Jets and constants
inline Jet operator* (const Jet& a, const Jet& b) {
    Jet out;
    out.v = a.v * b.v;
    double dot = 0.0;
    for (int i = 0; i < D; i++) {
        out.g[i] = a.g[i] * b.v + a.v * b.g[i];
        dot += a.g[i] * b.g[i];
    }
    out.l = a.l * b.v + a.v * b.l + 2.0 * dot; 
    return out;
}

inline Jet operator* (const Jet& a, double c) {
    Jet out = a;
    out.v = a.v * c;
    for (int i = 0; i < D; i++) {
        out.g[i] = a.g[i] * c;
    }
    out.l = a.l * c;
    return out;
}

inline Jet operator* (double c, const Jet& a) {
    return a*c;
}

// Define negative operation between Jets and Jets and between Jets and constants
inline Jet operator- (const Jet& a) {
    Jet out;
    out.v = -a.v;
    for (int i = 0; i < D; i++) {
        out.g[i] = -a.g[i];
    }
    out.l = -a.l;
    return out;
}

inline Jet operator- (const Jet& a, const Jet& b) {
    Jet out;
    out.v = a.v - b.v;
    for (int i = 0; i < D; i++) {
        out.g[i] = a.g[i] - b.g[i];
    }
    out.l = a.l - b.l;
    return out;
}

inline Jet operator- (const Jet& a, double c) {
    Jet out = a;
    out.v -= c;
    return out;
}

inline Jet operator- (double c, const Jet& a) {
    return c + (-a);
}

// Define tanh operation on a Jet
inline Jet tanh(const Jet& a) {
    double t = std::tanh(a.v);
    double tp = 1.0 - t * t;
    double tpp = -2.0 * t * tp;

    double dot = 0.0;
    for (int i = 0; i < D; i++) {
        dot += a.g[i] * a.g[i];
    }

    Jet out;
    out.v = t;
    for (int i = 0; i < D; i++) {
        out.g[i] = tp * a.g[i];
    }
    out.l = tp * a.l + tpp * dot;
    return out;
}

// Define inverse operation
inline Jet inv(const Jet& a) {
    double iv = 1.0 / a.v;

    double dot = 0.0;
    for (int i = 0; i < D; i++) {
        dot += a.g[i] * a.g[i];
    }

    Jet out; 
    out.v = iv;
    for (int i = 0; i < D; i++) {
        out.g[i] = -a.g[i] * iv * iv;
    }
    out.l = -a.l * iv * iv + 2.0 * dot * iv * iv * iv;
    return out;
}

// Define division operation between Jets and Jets and between Jets and constants
inline Jet operator/ (const Jet& a, const Jet& b) {
    return a * inv(b);
}

inline Jet operator/ (const Jet& a, double c) {
    return a * (1.0 / c);
}

inline Jet operator/ (double c, const Jet& a) {
    return c * inv(a);
}

// Define exponentiation on a Jet
inline Jet exp(const Jet& a) {
    double e = std::exp(a.v);

    double dot = 0.0;
    for (int i = 0; i < D; i++) {
        dot += a.g[i] * a.g[i];
    }

    Jet out; 
    out.v = e;
    for (int i = 0; i < D; i++) {
        out.g[i] = e * a.g[i];
    }
    out.l = e * (a.l + dot);
    return out;
}

// Define square root on a Jet
inline Jet sqrt(const Jet& a) {
    double sv = std::sqrt(a.v);
    double fp = 0.5 / sv;
    double fpp = -0.25 / (sv*sv*sv);

    double dot = 0.0;
    for (int i = 0; i < D; i++) {
        dot += a.g[i] * a.g[i];
    }

    Jet out;
    out.v = sv;
    for (int i = 0; i < D; i++) {
        out.g[i] = fp * a.g[i];
    }
    out.l = fp * a.l + fpp * dot;
    return out;
}