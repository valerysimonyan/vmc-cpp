#include "backprop.h"
#include "arena.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Activations w/ derivatives
__device__ __forceinline__ real dev_act_grad(Activation act, real z) {
    if (act == Activation::Tanh) {
        real t = tanh(z);
        return (real)1.0 - t * t;
    }
    real s = (real)0.7978845608;
    real k = (real)0.044715;
    real g = s * (z + k * z * z * z);
    real gp = s * ((real)1.0 + (real)3.0 * k * z * z);
    real t = tanh(g);
    return (real)0.5 * ((real)1.0 + t) + (real)0.5 * z * ((real)1.0 - t * t) * gp;
}

// Batch activation gradients
__global__ void act_grad_eval_kernel(const real* __restrict__ z, real* __restrict__ out, std::size_t n, Activation act) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = dev_act_grad(act, z[i]);
}

void act_grad_eval(const real* z, real* out, std::size_t n, Activation act, cudaStream_t stream) {
    if (n == 0) return;
    const int threads = 256;
    act_grad_eval_kernel<<<(unsigned)((n + threads - 1)/threads), threads, 0, stream>>>(z, out, n, act);
    cuda_sync_check("act_grad_eval");
}

__global__ void act_grad_mul_kernel(real* __restrict__ delta, const real* __restrict__ z, std::size_t n, Activation act) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] *= dev_act_grad(act, z[i]);
}
void act_grad_mul(real* delta, const real* z, std::size_t n, Activation act, cudaStream_t stream) {
    if (n == 0) return;
    const int threads = 256;
    act_grad_mul_kernel<<<(unsigned)((n + threads - 1)/threads), threads, 0, stream>>>(delta, z, n, act);
    cuda_sync_check("act_grad_mul");
}

// Evaluate gradients w.r.t. weights
void dW_strided(cublasHandle_t handle, const real* a, const real* delta, int R, int in_w, int out_w, int Bc, double* C_first, long long strideC, cudaStream_t stream) {
    if (Bc <= 0) return;
    static_assert(std::is_same<real, double>::value, "dW_strided writes into the double O_pool with cublasDgemmStridedBatched");
    cublasSetStream(handle, stream);
    const double one = 1.0, zero = 0.0;
    cublasStatus_t st = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        in_w, out_w, R, &one,
        a,     in_w,  (long long)R * in_w,
        delta, out_w, (long long)R * out_w,
        &zero,
        C_first, in_w, strideC,
        Bc);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("dW_strided: cublasDgemmStridedBatched failed, status " + std::to_string((int)st));
    cuda_sync_check("dW_strided");
}

// Evaluate gradients w.r.t. biases
__global__ void db_rows_kernel(const real* __restrict__ delta, int R, int out_w, int Bc, double* __restrict__ O_first, std::size_t b_off, std::size_t P) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (std::size_t)Bc * out_w) return;
    const std::size_t w = idx / out_w;
    const int i = (int)(idx % out_w);
    double acc = 0.0;
    for (int r = 0; r < R; r++) acc += delta[(w*R + r)*out_w + i];
    O_first[w*P + b_off + i] = acc;
}

void db_rows(const real* delta, int R, int out_w, int Bc, double* O_first, std::size_t b_off, std::size_t P, cudaStream_t stream) {
    if (Bc <= 0) return;
    const std::size_t total = (std::size_t)Bc * out_w;
    const int threads = 256;
    db_rows_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(delta, R, out_w, Bc, O_first, b_off, P);
    cuda_sync_check("db_rows");
}

// Fill transpose of params
__global__ void stage_wt_kernel(const real* __restrict__ W, real* __restrict__ WT, int in_w, int out_w) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (std::size_t)in_w * out_w) return;
    const int j = (int)(idx / out_w), i = (int)(idx % out_w);
    WT[idx] = W[(std::size_t)i * in_w + j];
}

// Ping pong for backprop
__global__ void delta_prop_kernel(const real* __restrict__ cur, const real* __restrict__ WT, real* __restrict__ nxt, int rows, int in_w, int out_w) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (std::size_t)rows * in_w) return;
    const std::size_t r = idx / in_w;
    const int j = (int)(idx % in_w);
    const real* wt = WT + (std::size_t)j * out_w;
    const real* c = cur + r * out_w;
    real acc = (real)0;
    for (int i = 0; i < out_w; i++) acc += wt[i] * c[i];
    nxt[idx] = acc;
}

static void delta_prop(const real* cur, const real* W, real* WT, real* nxt, int rows, int in_w, int out_w, cudaStream_t stream) {
    const int threads = 256;
    const std::size_t nwt = (std::size_t)in_w * out_w;
    stage_wt_kernel<<<(unsigned)((nwt + threads - 1)/threads), threads, 0, stream>>>(W, WT, in_w, out_w);
    const std::size_t total = (std::size_t)rows * in_w;
    delta_prop_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(cur, WT, nxt, rows, in_w, out_w);
    cuda_sync_check("delta_prop");
}

// Actual backprop
void backprop_net(cublasHandle_t handle, const DeviceNet& dn, const NetCache& cache, const real* params, real* cur, real* nxt, real* WT, int R, int Bc, int w_off, double* O_first, std::size_t P, real* dinput, cudaStream_t stream) {
    if (Bc <= 0) return;
    const int rows = Bc * R;
    const std::size_t row0 = (std::size_t)w_off * R;              
    const int n_layers = (int)dn.layers.size();

    for (int l = n_layers - 1; l >= 0; l--) {
        const DeviceLayer& L = dn.layers[l];

        db_rows(cur, R, L.out_w, Bc, O_first, L.b_off, P, stream);
        dW_strided(handle, cache.a_in[l].d + row0 * L.in_w, cur, R, L.in_w, L.out_w, Bc, O_first + L.w_off, (long long)P, stream);


        if (l == 0 && !dinput) break;

        real* dst = (l == 0) ? dinput : nxt;
        delta_prop(cur, params + L.w_off, WT, dst, rows, L.in_w, L.out_w, stream);
        if (l > 0) {
            act_grad_mul(dst, cache.z[l - 1].d + row0 * L.in_w, (std::size_t)rows * L.in_w, dn.act, stream);
            std::swap(cur, nxt);
        }
    }
}

// Build input for backprop
__global__ void h_seed_kernel(const real* __restrict__ dpsi_dxi, real* __restrict__ seed, int Bc) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (std::size_t)Bc * N * m_feat) return;
    const std::size_t row = idx / m_feat;
    const int f = (int)(idx % m_feat);
    seed[idx] = dpsi_dxi[(row / N) * m_feat + f];
}

__global__ void orb_seed_kernel(const real* __restrict__ rho, const real* __restrict__ dets, const real* __restrict__ Minv, real* __restrict__ seed, int Bc) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t width = (std::size_t)K * N;
    if (idx >= (std::size_t)Bc * N * width) return;
    const std::size_t row = idx / width;
    const std::size_t w = row / N;
    const int i = (int)(row % N);
    const int col = (int)(idx % width);
    const int j = col / N, k = col % N;
    seed[idx] = rho[w*K + j] * dets[w*K + j] * Minv[(w*K + j)*(N*N) + (std::size_t)i*N + k];
}

// Fill O's
__global__ void o_finalize_kernel(double* __restrict__ O_first, const unsigned char* __restrict__ valid, const real* __restrict__ S, const real* __restrict__ x_sh, real alpha, std::size_t P, int Bc, int w_off) {
    const int wl = blockIdx.x;
    if (wl >= Bc) return;
    const std::size_t w = (std::size_t)w_off + wl;
    double* row = O_first + (std::size_t)wl * P;

    if (!valid[w]) {
        for (std::size_t k = threadIdx.x; k < P; k += blockDim.x) row[k] = 0.0;
        return;
    }
    const real Sw = S[w];
    for (std::size_t k = threadIdx.x; k < P - 1; k += blockDim.x) row[k] = row[k] / Sw;

    if (threadIdx.x == 0) {
        const real* xw = x_sh + w * D;
        real r2 = (real)0;
        for (int i = 0; i < D; i++) { const real c = xw[i]; r2 += c * c; }
        const real r_env = sqrt(r2 + (real)(eps_env * eps_env));
        row[P - 1] = -exp(alpha) * r_env;
    }
}

// Evaluate full Os 
void assemble_O_batch(DeviceState& ds, cublasHandle_t handle, int r, int B, cudaStream_t stream, int chunk) {
    if (B <= 0) return;
    if (ds.cache_rho.rows_cap == 0) throw std::runtime_error("assemble_O_batch: grow_phase5 has not run");
    if ((std::size_t)(r + 1) * (std::size_t)B > ds.Ns_max) throw std::runtime_error("assemble_O_batch: record row beyond O_pool");
    const int C = (chunk > 0) ? chunk : B;
    const std::size_t P = ds.P;

    real alpha;
    CUDA_CHECK(cudaMemcpyAsync(&alpha, ds.params.d + (P - 1), sizeof(real), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // rho's output layer is linear, so its stashed pre-activation IS rho.
    const real* rho = ds.cache_rho.z.back().d;
    const int threads = 256;

    for (int w_off = 0; w_off < B; w_off += C) {
        const int Bc = std::min(C, B - w_off);
        double* O_first = ds.O_pool.d + ((std::size_t)r * B + (std::size_t)w_off) * P;

        CUDA_CHECK(cudaMemcpy(ds.bp_a.d, ds.dets_psi.d + (std::size_t)w_off * K, (std::size_t)Bc * K * sizeof(real), cudaMemcpyDeviceToDevice));
        backprop_net(handle, ds.rho_net_d, ds.cache_rho, ds.params.d, ds.bp_a.d, ds.bp_b.d, ds.bp_wt.d, 1, Bc, w_off, O_first, P, ds.dpsi_dxi.d, stream);

        {
            const std::size_t total = (std::size_t)Bc * N * m_feat;
            h_seed_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(ds.dpsi_dxi.d, ds.bp_a.d, Bc);
            cuda_sync_check("h_seed");
        }
        backprop_net(handle, ds.h_net_d, ds.cache_h, ds.params.d, ds.bp_a.d, ds.bp_b.d, ds.bp_wt.d, N, Bc, w_off, O_first, P, nullptr, stream);

        // orb: rho_j * det_j * Minv_j[i][k], R = N.
        {
            const std::size_t total = (std::size_t)Bc * N * K * N;
            orb_seed_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(
                rho + (std::size_t)w_off * K, ds.dets_psi.d + (std::size_t)w_off * K,
                ds.Minv_batch.d + (std::size_t)w_off * K * N * N, ds.bp_a.d, Bc);
            cuda_sync_check("orb_seed");
        }
        backprop_net(handle, ds.orb_net_d, ds.cache_orb, ds.params.d, ds.bp_a.d, ds.bp_b.d, ds.bp_wt.d, N, Bc, w_off, O_first, P, nullptr, stream);

        o_finalize_kernel<<<(unsigned)Bc, 128, 0, stream>>>(O_first, ds.valid_loc.d, ds.S.d, ds.x_sh.d, alpha, P, Bc, w_off);
        cuda_sync_check("o_finalize");
    }
}
