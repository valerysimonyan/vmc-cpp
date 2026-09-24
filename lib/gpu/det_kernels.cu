#include "det_kernels.h"
#include "../physics.h"
#include "../envelope.h"

#include <cmath>

// Assemble single particle Slater matrix
__global__ void assemble_M_kernel(const real* __restrict__ orb_out, real* __restrict__ M_batch, int B) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = (std::size_t)B * K * N * N;
    if (idx >= total) return;

    const int i = (int)(idx % N);
    const int k = (int)((idx / N) % N);
    const int j = (int)(idx / (N*N) % K);
    const int w = (int)(idx / ((std::size_t)K*N*N));

    M_batch[idx] = orb_out[(std::size_t)(w*N + i) * (K*N) + j*N + k];
}

void assemble_M(const real* orb_out, real* M_batch, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const std::size_t total = (std::size_t)B * K * N * N;
    const int threads = 256;
    assemble_M_kernel<<<(unsigned)((total+threads-1)/threads), threads, 0, stream>>>(orb_out, M_batch, B);
    cuda_sync_check("assemble_M");
}

// Assemble a single particle Slater matrix with the full table 
__global__ void assemble_M_combo_kernel(const real* __restrict__ tab_orb, const real* __restrict__ s, const real* __restrict__ t, real* __restrict__ M_batch, int B) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = (std::size_t)B * K * N * N;
    if (idx >= total) return;

    const int i = (int)( idx % N );
    const int k = (int)((idx / N) % N);
    const int j = (int)((idx / (N*N)) % K);
    const int w = (int)( idx / ((std::size_t)K*N*N));

    // st_combo from physics.h, inlined: (s>0?0:1) + 2*(t>0?0:1)
    const real sv = s[(std::size_t)w*N + i], tv = t[(std::size_t)w*N + i];
    const int  c  = (sv > (real)0 ? 0 : 1) + 2 * (tv > (real)0 ? 0 : 1);

    M_batch[idx] = tab_orb[((std::size_t)(w*N + i)*4 + c) * (K*N) + j*N + k];    
}

void assemble_M_combo(const real* tab_orb, const real* s, const real* t, real* M_batch, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const std::size_t total = (std::size_t)B * K * N * N;
    const int threads = 256;
    assemble_M_combo_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(tab_orb, s, t, M_batch, B);
    cuda_sync_check("assemble_M_combo");
}

// Determinant assembly, 
__global__ void det_from_lu_kernel(const real* __restrict__ M_batch, const int* __restrict__ ipiv, const int* __restrict__ info, real* __restrict__ dets, int n_mats) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_mats) return;
    
    if(info[m] > 0) {
        dets[m] = (real)0;
        return;
    }

    const real* A = M_batch + (std::size_t)m * N * N;
    const int*  p = ipiv + (std::size_t)m * N;

    real det = (real)1;
    int swaps = 0;
    for (int i = 0; i < N; i++) {
        det *= A[i*N + i];
        if (p[i] != i + 1) swaps++;
    }
    if (swaps & 1) det = -det;

    if (!isfinite(det) || fabs(det) < (real)1e-300) det = (real)0;

    dets[m] = det;
}

// Get determinant from LU decomposition
void dets_from_lu(const real* M_batch, const int* lu_piv, const int* lu_info, real* dets, int n_mats, cudaStream_t stream) {
    if (n_mats <= 0) return;
    const int threads = 256;
    det_from_lu_kernel<<<(n_mats + threads - 1)/threads, threads, 0, stream>>>(M_batch, lu_piv, lu_info, dets, n_mats);
    cuda_sync_check("det_from_lu");
}

void batched_det(cublasHandle_t handle, int n_mats, real* M_batch, double** lu_ptrs, int* lu_piv, int* lu_info, real* dets, cudaStream_t stream) {
    if (n_mats <= 0) return;
    static_assert(std::is_same<real, double>::value, "batched_det uses cublasDgetrfBatched; the FP32 path needs cublasSgetrfBatched and its own oracle tolerances");

    // lu_ptrs stores pointers to all NxN matrices containing L and U, P is stored in lu_piv, and lu_info contains information on singulatirties
    lu_factor(handle, n_mats, lu_ptrs, lu_piv, lu_info, stream);

    // Evaluate determinant, return 0 if singular matrix
    dets_from_lu(M_batch, lu_piv, lu_info, dets, n_mats, stream);
}

// Evlauate ansatz
__global__ void S_combine_kernel(const real* __restrict__ rho, const real* __restrict__ dets, real* __restrict__ S, int B) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    real acc = (real)0;
    for (int k = 0; k < K; k++) acc += rho[(std::size_t)w*K + k] * dets[(std::size_t)w*K + k];
    S[w] = acc;
}

void S_combine(const real* rho_out, const real* dets, real* S, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    S_combine_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(rho_out, dets, S, B);
    cuda_sync_check("S_combine");
}

// Also GPU_ize envelope evaluation
__global__ void envelope_logp_kernel(const real* __restrict__ x_sh, const real* __restrict__ S, const real* __restrict__ alpha_d, real* __restrict__ logp, int B) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    const real alpha = *alpha_d;

    const real r_env = envelope::radius(envelope::r2(x_sh + (std::size_t)w * D));

    const real Sv = S[w];
    if (!(Sv != (real)0) || !isfinite(Sv)) {   // catches 0, -0 and NaN
        logp[w] = -INFINITY;
        return;
    }
    logp[w] = envelope::log_factor(alpha, r_env) + log(fabs(Sv));
}

void envelope_logp(const real* x_sh, const real* S, const real* params, std::size_t P, real* logp, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    envelope_logp_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(x_sh, S, params + (P - 1), logp, B);
    cuda_sync_check("envelope_logp");
}

__global__ void combine_envelope_kernel(const real* __restrict__ rho, const real* __restrict__ dets, real* __restrict__ S, const real* __restrict__ x_sh, const real* __restrict__ alpha_d, real* __restrict__ logp, int B) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;

    real acc = (real)0;
    for (int k = 0; k < K; k++) acc += rho[(std::size_t)w*K + k] * dets[(std::size_t)w*K + k];
    S[w] = acc;

    const real alpha = *alpha_d;
    const real r_env = envelope::radius(envelope::r2(x_sh + (std::size_t)w * D));

    const real Sv = acc;
    if (!(Sv != (real)0) || !isfinite(Sv)) {
        logp[w] = -INFINITY;
        return;
    }
    logp[w] = envelope::log_factor(alpha, r_env) + log(fabs(Sv));
}

void combine_envelope(const real* rho_out, const real* dets, real* S, const real* x_sh, const real* params, std::size_t P, real* logp, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    combine_envelope_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(rho_out, dets, S, x_sh, params + (P - 1), logp, B);
    cuda_sync_check("combine_envelope");
}

void lu_factor(cublasHandle_t handle, int n_mats, double** lu_ptrs, int* lu_piv, int* lu_info, cudaStream_t stream) {
    if (n_mats <= 0) return;
    blas_bind(handle, stream);
    cublasStatus_t st = cublasDgetrfBatched(handle, N, lu_ptrs, N, lu_piv, lu_info, n_mats);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("lu_factor: cublasDgetrfBatched failed, status " + std::to_string((int)st));
}