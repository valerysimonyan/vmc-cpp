#pragma once

#include "layouts.h"

// Flattened index for orbital matrices
VMC_HD inline std::size_t orb_jet_index(int c, int w, int jd, int k, int i, std::size_t rows_tot) {
    const std::size_t stride = jet_block_stride(rows_tot, (std::size_t)(K * N));
    return (std::size_t)c * stride + (std::size_t)(w * N + i) * (K * N) + (std::size_t)(jd * N + k);
}

void batched_inverse(cublasHandle_t handle, int n_mats, const real* M_batch_lu, double** lu_ptrs, double** inv_ptrs, const int* lu_piv, int* inv_info, real* Minv_batch, cudaStream_t stream = 0);

void det_jet_assemble(const real* J_orb, const real* Minv_batch, const real* dets, real* J_det, int Bc, int w_off, int B_tot, cudaStream_t stream = 0);
