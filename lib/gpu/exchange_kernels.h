#pragma once

#include "layouts.h"
#include "../hamiltonian.h"

#include <vector>

struct DeviceState;

inline constexpr int ex_npairs = N * (N - 1) / 2;
inline constexpr int ex_types  = 3;

inline constexpr int ex_walkers = (int)(rows_max_phase3 / (std::size_t)(ex_types * ex_npairs));

// Assign number for given S and T
VMC_HD inline int ex_combo(real s, real t) {
    return (s > (real)0 ? 0 : 1) + 2 * (t > (real)0 ? 0 : 1);
}

std::vector<int> ex_pair_table();

void ex_plan(const real* s, const real* t, const int* pair_ij, unsigned char* active, int B, cudaStream_t stream = 0);

void ex_rank2_gate(const real* dets_psi, unsigned char* rank2_ok, int B, cudaStream_t stream = 0);

void ex_xi_swap(const real* xi_psi, const real* tab_h, const real* s, const real* t, const int* pair_ij, real* xi_swap, int Bc, int w_off, cudaStream_t stream = 0);

void ex_S_swap(const real* rho_swap, const real* dets_psi, const real* Minv_batch, const real* tab_orb, const real* s, const real* t, const int* pair_ij, const unsigned char* active, const unsigned char* rank2_ok, real* S_swap, int Bc, int w_off, cudaStream_t stream = 0);

int ex_fallback_host(DeviceState& ds, const Ansatz& a, Workspace& ws, int B);

void coulomb_batch(const real* x, const real* t, real* V_coul, int B, cudaStream_t stream = 0);

void ex_assemble(const real* x, const real* s, const real* t, const real* S_swap, const real* S0, const real* E_kin, const real* v3n, const real* V_coul, const unsigned char* valid_jet, real* V_nuc, real* E_loc, unsigned char* valid_loc, int B, const real* params, std::size_t P, cudaStream_t stream = 0);

void pool_write_row(const real* E_loc, const unsigned char* valid_loc, double* E_pool, unsigned char* valid_pool, int r, int B, cudaStream_t stream = 0);
