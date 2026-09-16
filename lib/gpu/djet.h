#pragma once

#include "layouts.h"

#include <cmath> 

struct DJet {
    real v; 
    real g[D];
    real l;
};

// Initialize to zero
VMC_HD inline void djet_zero(DJet& o) {
    o.v = (real)0;
    for (int i = 0; i < D; i++) o.g[i] = (real)0;
    o.l = (real)0;
}

// Set to constant
VMC_HD inline void djet_const(DJet& o, real c) {
    o.v = c;
    for (int i = 0; i < D; i++) o.g[i] = (real)0;
    o.l = (real)0;
}

// Addition operator between Jets
VMC_HD inline void djet_add(DJet& o, const DJet& a, const DJet& b) {
    o.v = a.v + b.v;
    for (int i = 0; i < D; i++) o.g[i] = a.g[i] + b.g[i];
    o.l = a.l + b.l;
}

// Multiplication operator between Jets
VMC_HD inline void djet_mul(DJet& o, const DJet& a, const DJet& b) {
    const real av = a.v, bv = b.v;
    real dot = (real)0;
    real og[D];
    for (int i = 0; i < D; i++) {
        og[i] = a.g[i] * bv + av * b.g[i];
        dot += a.g[i] * b.g[i];
    }
    o.v = av * bv;
    for (int i = 0; i < D; i++) o.g[i] = og[i];
    o.l = a.l * bv + av * b.l + (real)2 * dot;
}

// Jet * constant
VMC_HD inline void djet_scale(DJet& o, const DJet& a, real c) {
    o.v = a.v * c;
    for (int i = 0; i < D; i++) o.g[i] = a.g[i] * c;
    o.l = a.l * c;
}

// Square root operation on a Jet
VMC_HD inline void djet_sqrt(DJet& o, const DJet& a) {
    const real sv = sqrt(a.v);
    const real fp = (real)0.5 / sv;
    const real fpp = (real)-0.25 / (sv*sv*sv);
    real dot = (real)0;
    for (int i = 0; i < D; i++) dot += a.g[i] * a.g[i];
    o.v = sv;
    for (int i = 0; i < D; i++) o.g[i] = fp * a.g[i];
    o.l = fp * a.l + fpp * dot;
}

// Exponential of a Jet
VMC_HD inline void djet_exp(DJet& o, const DJet& a) {
    const real e = exp(a.v);
    real dot = (real)0;
    for (int i = 0; i < D; i++) dot += a.g[i] * a.g[i];
    o.v = e;
    for (int i = 0; i < D; i++) o.g[i] = e * a.g[i];
    o.l = e * (a.l + dot);
}