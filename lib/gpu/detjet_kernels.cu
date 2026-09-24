#include "detjet_kernels.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

void batched_inverse(cublasHandle_t handle, int n_mats, const real* M_batch_lu, double** lu_ptrs, double** inv_ptrs, const int* lu_piv, int* inv_info, real* Minv_batch, cudaStream_t stream) {
    if (n_mats <= 0) return;
    static_assert(std::is_same<real, double>::value, "batched_inverse uses cublasDgetriBatched; the FP32 path needs cublasSgetriBatched");
    (void)M_batch_lu; (void)Minv_batch;

    cublasSetStream(handle, stream);
    cublasStatus_t st = cublasDgetriBatched(handle, N, (const double* const*)lu_ptrs, N, (int*)lu_piv, inv_ptrs, N, inv_info, n_mats);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("batched_inverse: cublasDgetriBatched failed, status " + std::to_string((int)st));
    cuda_sync_check("batched_inverse");
}

// Evaluate Jet for orbital matrices
__global__ void det_jet_kernel(const real* __restrict__ J_orb, const real* __restrict__ Minv_batch, const real* __restrict__ dets, real* __restrict__ J_det, int Bc, int w_off, int B_tot) {
    const int m = blockIdx.x;                 
    const int w = m / K, jd = m % K;
    if (w >= Bc) return;

    const std::size_t rows_tot = (std::size_t)Bc * N;
    const std::size_t out_stride = jet_block_stride((std::size_t)B_tot, (std::size_t)K);
    const std::size_t out_off = (std::size_t)(w_off + w) * K + jd;

    const std::size_t gm = (std::size_t)(w_off + w) * K + jd;
    const real det_v = dets[gm];

    if (det_v == (real)0) {
        for (int c = threadIdx.x; c < jet_C; c += blockDim.x) J_det[(std::size_t)c*out_stride + out_off] = (real)0;
        return;
    }

    __shared__ real sMinv[N*N];
    __shared__ real sgsum[D+1];  // [0,D) = gsum[a]; [D] = term1
    __shared__ real sT2[D];
    __shared__ real sB[D][N*N];
#ifdef DETJET_SELFCHECK
    __shared__ real sgmag[D];
#endif

    for (int e = threadIdx.x; e < N*N; e += blockDim.x) sMinv[e] = Minv_batch[gm * (N*N) + e];
    __syncthreads();

    const int a = (int)threadIdx.x;
    if (a <= D) {
        const int comp = (a < D) ? (1 + a) : (jet_C - 1);  // gradient block, or the Laplacian block
        real acc = (real)0;
#ifdef DETJET_SELFCHECK
        real mag = (real)0;
#endif
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                const real mv = sMinv[j*N + k];
                const real e  = J_orb[orb_jet_index(comp, w, jd, k, j, rows_tot)];
                acc += mv * e;
#ifdef DETJET_SELFCHECK
                mag += fabs(mv * e);
#endif
            }
        }
        sgsum[a] = acc;
#ifdef DETJET_SELFCHECK
        if (a < D) sgmag[a] = mag;
#endif
    }
    __syncthreads();

    if (a < D) {
        real* B = sB[a];
        for (int j = 0; j < N; j++) {
            for (int c = 0; c < N; c++) {
                real acc = (real)0;
                for (int r = 0; r < N; r++)
                    acc += sMinv[j*N + r] * J_orb[orb_jet_index(1 + a, w, jd, r, c, rows_tot)];
                B[j*N + c] = acc;
            }
        }
        const real trB = sgsum[a];
#ifdef DETJET_SELFCHECK
        real trB_direct = (real)0;
        for (int j = 0; j < N; j++) trB_direct += B[j*N + j];
        if (fabs(trB_direct - trB) > (real)1e-11 * fmax((real)1, sgmag[a])) {
            printf("det_jet_kernel SELFCHECK: w=%d jd=%d a=%d trB=%.17g direct=%.17g gmag=%.17g\n", w_off + w, jd, a, (double)trB, (double)trB_direct, (double)sgmag[a]);
        }
#endif
        real trB2 = (real)0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) trB2 += B[j*N + i] * B[i*N + j];
        sT2[a] = trB * trB - trB2;
    }
    __syncthreads();

    if (a < D) J_det[(std::size_t)(1 + a)*out_stride + out_off] = det_v * sgsum[a];
    if (a == 0) {
        J_det[out_off] = det_v;
        real term2 = (real)0;                      
        for (int q = 0; q < D; q++) term2 += sT2[q];
        J_det[(std::size_t)(jet_C - 1)*out_stride + out_off] = det_v * (sgsum[D] + term2);
    }
}

void det_jet_assemble(const real* J_orb, const real* Minv_batch, const real* dets, real* J_det, int Bc, int w_off, int B_tot, cudaStream_t stream) {
    if (Bc <= 0) return;
    static_assert((std::size_t)(D*N*N + N*N + 2*D + 1) * sizeof(real) <= 48 * 1024, "det_jet_kernel: static shared memory exceeds 48 KiB at this N");
    const int threads = ((D + 1 + 31) / 32) * 32;
    det_jet_kernel<<<(unsigned)((std::size_t)Bc * K), threads, 0, stream>>>(J_orb, Minv_batch, dets, J_det, Bc, w_off, B_tot);
    cuda_sync_check("det_jet_assemble");
}