#pragma once

#include "gpu_util.h"
#include "../constants.h"
#include "../precision.h"
#include "../rng_common.h"  

#include <cublas_v2.h>

inline constexpr int rows_per_combo = 4;
inline constexpr std::size_t rows_max_phase3 =  (std::size_t)n_walkers * (std::size_t)N * (std::size_t)rows_per_combo;

inline constexpr int jet_C = D + 2;

// Walkers per jet chunk: jet_chunk, or the whole batch when jet_chunk is 0.
inline constexpr int jet_walkers = (jet_chunk > 0) ? jet_chunk : n_walkers;
inline constexpr std::size_t jet_rows = (std::size_t)jet_walkers * (std::size_t)N;

VMC_HD inline constexpr std::size_t jet_block_stride(std::size_t rows, std::size_t width_max) {
    return rows * width_max;
}

// Bind a stream to the handle
inline void blas_bind(cublasHandle_t handle, cudaStream_t stream) {
    cudaStream_t cur = nullptr;
    cublasGetStream(handle, &cur);
    if (cur != stream) cublasSetStream(handle, stream);
}

// GPU accelerated pass through network layer, after all it is just matrix multiplication
inline void gemm_rowmajor(cublasHandle_t handle, int rows, int in_w, int out_w, const real* In, const real* W, real* Out, cudaStream_t stream = 0) {
    blas_bind(handle, stream);    
    const real alpha = (real)1.0, beta = (real)0.0;
    cublasStatus_t st; 
    if constexpr (std::is_same<real, double>::value) {
        st = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_w, rows, in_w, (const double*)&alpha, (const double*)W, in_w, (const double*)In, in_w, (const double*)&beta,  (double*)Out, out_w);
    } else {
        st = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_w, rows, in_w, (const float*)&alpha, (const float*)W, in_w, (const float*)In, in_w, (const float*)&beta,  (float*)Out, out_w);
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("gemm_rowmajor: cublas GEMM failed with status " + std::to_string((int)st));
    }
}
