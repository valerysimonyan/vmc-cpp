#pragma once

#include "layouts.h"
#include "../physics.h"

#include <vector>

struct DeviceState;

inline constexpr int ex_npairs = N * (N - 1) / 2;
inline constexpr int ex_types  = 3;
inline constexpr int EX_S  = 0;   // swap s_i <-> s_j        (R_s)
inline constexpr int EX_T  = 1;   // swap t_i <-> t_j        (R_t)
inline constexpr int EX_ST = 2;   // swap both               (R_st)

inline constexpr int ex_walkers = (int)(rows_max_phase3 / (std::size_t)(ex_types * ex_npairs));

// Check if swap happens
VMC_HD inline bool ex_slot_active(real si, real ti, real sj, real tj, int type) {
    const bool same_s = (si == sj), same_t = (ti == tj);
    if (same_s && same_t) return false;
    if (same_s) return type == EX_T;
    if (same_t) return type == EX_S;
    return true;
}

// Relabel indices after swapping labels
VMC_HD inline void ex_swapped_labels(real si, real ti, real sj, real tj, int type, real& sni, real& tni, real& snj, real& tnj) {
    if (type == EX_T)      { sni = si; tni = tj; snj = sj; tnj = ti; }
    else if (type == EX_S) { sni = sj; tni = ti; snj = si; tnj = tj; }
    else                   { sni = sj; tni = tj; snj = si; tnj = ti; }
}

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
void ex_assemble(const real* x, const real* s, const real* t, const int* pair_ij, const real* S_swap, const real* S0, const real* E_kin, const real* v3n, const real* V_coul, const unsigned char* valid_jet, real* V_nuc, real* E_loc, unsigned char* valid_loc, int B, const real* params, std::size_t P, cudaStream_t stream = 0);
void pool_write_row(const real* E_loc, const unsigned char* valid_loc, double* E_pool, unsigned char* valid_pool, int r, int B, cudaStream_t stream = 0);
