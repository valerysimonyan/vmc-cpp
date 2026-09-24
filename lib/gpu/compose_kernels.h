#pragma once

#include "layouts.h"

void psi_jet_compose(const real* J_rho, const real* J_det, const real* x_sh, const real* s, const real* t, const real* params, std::size_t P, real* J_psi, real* S_jet_v, int Bc, int w_off, int B_tot, cudaStream_t stream = 0);

void kinetic_l2(const real* J_psi, const real* x_sh, real* E_kin, real* l2, int Bc, int w_off, int B_tot, cudaStream_t stream = 0);

void v3n_batch(const real* x, real* v3n, int Bc, int w_off, cudaStream_t stream = 0);

void validity_jet(const real* J_psi, const real* S, const real* x_sh, const real* s, const real* t, const real* params, std::size_t P, const real* E_kin, real* psi_dbl, unsigned char* valid, int Bc, int w_off, int B_tot, cudaStream_t stream = 0);
