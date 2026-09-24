#pragma once

#include "layouts.h"

struct DeviceState; 

void assemble_M(const real* orb_out, real* M_batch, int B, cudaStream_t stream = 0);

void assemble_M_combo(const real* tab_orb, const real* s, const real* t, real* M_batch, int B, cudaStream_t stream = 0);

void batched_det(cublasHandle_t handle, int n_mats, real* M_batch, double** lu_ptrs, int* lu_piv, int* lu_info, real* dets, cudaStream_t stream = 0);

void S_combine(const real* rho_out, const real* dets, real* S, int B, cudaStream_t stream = 0);

void envelope_logp(const real* x_sh, const real* s, const real* t, const real* S, const real* params, std::size_t P, real* logp, int B, cudaStream_t stream = 0);

void lu_factor(cublasHandle_t handle, int n_mats, double** lu_ptrs, int* lu_piv, int* lu_info, cudaStream_t stream = 0);

void dets_from_lu(const real* M_batch, const int* lu_piv, const int* lu_info, real* dets, int n_mats, cudaStream_t stream = 0);

void combine_envelope(const real* rho_out, const real* dets, real* S, const real* x_sh, const real* s, const real* t, const real* params, std::size_t P, real* logp, int B, cudaStream_t stream = 0);
