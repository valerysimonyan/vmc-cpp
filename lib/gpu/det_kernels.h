#pragma once

#include "layouts.h"

struct DeviceState; 

void assemble_M(const real* orb_out, real* M_batch, int B, cudaStream_t stream = 0);

void assemble_M_combo(const real* tab_orb, const real* s, const real* t, real* M_batch, int B, cudaStream_t stream = 0);

void batched_det(cublasHandle_t handle, int n_mats, real* M_batch, double** lu_ptrs, int* lu_piv, int* lu_info, real* dets, cudaStream_t stream = 0);

void S_combine(const real* rho_out, const real* dets, real* S, int B, cudaStream_t stream = 0);

void envelope_logp(const real* x_sh, const real* S, const real* params, std::size_t P, real* logp, int B, cudaStream_t stream = 0);