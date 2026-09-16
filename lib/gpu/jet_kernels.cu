#include "jet_kernels.h"

#include <cmath>
#include <stdexcept>

__global__ void build_jet_feat_kernel(const real* __restrict__ x, const real* __restrict__ s, const real* __restrict__ t, real* __restrict__ J, int Bc) {
    const int Wf = dim + 2;
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t rows = (std::size_t)Bc * N; 
    if (idx >= rows) return;

    const int w = (int)(idx / N), p = (int)(idx % N);
    const std::size_t stride = jet_block_stride(rows, (std::size_t)Wf);

    // Inputs, takes COM subtracted coordinates
    real* v = J + 0*stride + idx*Wf;
    for (int d = 0; d < dim; d++) {
        real R = (real)0;
        for (int i = 0; i < N; i++) R += x[(std::size_t)w*D + i*dim + d];
        R /= (real)N;
        v[d] = x[(std::size_t)w*D + p*dim + d] - R;
    }
    v[dim] = s[(std::size_t)w*N + p];
    v[dim + 1] = t[(std::size_t)w*N + p];

    // Gradient evaluation, get 1-1/N as derivative aside from accumulated derivative
    const real inv_N = (real)1 / (real)N;
    for (int a = 0; a < D; a++) {
        const int j  = a / dim;          
        const int dp = a % dim;
        real* g = J + (std::size_t)(1 + a)*stride + idx*Wf;
        for (int d = 0; d < dim; d++) {
            g[d] = (d == dp) ? ((p == j ? (real)1 : (real)0) - inv_N) : (real)0;
        }
        g[dim] = (real)0;      // s is a constant
        g[dim + 1] = (real)0;  // t is a constant
    }

    // Raw Laplacian is zero without accumulated quantity
    real* l = J + (std::size_t)(jet_C - 1)*stride + idx*Wf;
    for (int c = 0; c < Wf; c++) l[c] = (real)0;
}

void build_jet_feat(const real* x, const real* s, const real* t, real* J_feat, int Bc, cudaStream_t stream) {
    if (Bc <= 0) return;
    const std::size_t rows = (std::size_t)Bc * N;
    const int threads = 256;
    build_jet_feat_kernel<<<(unsigned)((rows + threads - 1)/threads), threads, 0, stream>>>(x, s, t, J_feat, Bc);
    cuda_sync_check("build_jet_feat");
}

// Evaluate derivative
__device__ __forceinline__ void dev_act_derivs(Activation act, real z, real& f, real& fp, real& fpp) {
    if (act == Activation::Tanh) {
        real t = tanh(z);
        f = t; fp = (real)1 - t*t; fpp = (real)-2 * t * ((real)1 - t*t);
        return;
    }
    // Gelu, transcribed from act_grad/act_hess in network.h.
    const real s = (real)0.7978845608, k = (real)0.044715;
    real u    = s * (z + k*z*z*z);
    real up   = s * ((real)1 + (real)3*k*z*z);
    real upp  = s * (real)6 * k * z;
    real t    = tanh(u);
    real sech2 = (real)1 - t*t;
    real tp   = sech2 * up;
    real tpp  = (real)-2*t*sech2*up*up + sech2*upp;
    f   = (real)0.5 * z * ((real)1 + t);
    fp  = (real)0.5 * ((real)1 + t) + (real)0.5 * z * tp;
    fpp = tp + (real)0.5 * z * tpp;
}

// Take derivative
__global__ void jet_bias_act_kernel(real* __restrict__ J, const real* __restrict__ bias, int rows, int width, std::size_t stride, Activation act, bool is_output) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = (std::size_t)rows * width;
    if (idx >= total) return;

    const int col = (int)(idx % (std::size_t)width);
    real* v = J + idx;                          
    const real z = *v + bias[col];

    if (is_output) { 
        *v = z; 
        return; 
    }     

    real f, fp, fpp;
    dev_act_derivs(act, z, f, fp, fpp);

    real dot = (real)0;
    for (int a = 1; a <= D; a++) {
        real* ga = J + (std::size_t)a*stride + idx;
        const real g = *ga;
        dot += g * g;
        *ga = fp * g;
    }
    real* l = J + (std::size_t)(jet_C - 1)*stride + idx;
    *l = fp * (*l) + fpp * dot;
    *v = f;
}

void jet_bias_act(real* J, const real* bias, int rows, int width, int width_max, Activation act, bool is_output, cudaStream_t stream) {
    if (rows <= 0 || width <= 0) return;
    const std::size_t total = (std::size_t)rows * width;
    const int threads = 256;
    jet_bias_act_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(J, bias, rows, width, jet_block_stride((std::size_t)rows, (std::size_t)width_max), act, is_output);
    cuda_sync_check("jet_bias_act");
}


// Forward pass through network 
void jet_net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* J_in, int in_width_max, int rows, real* J_a, real* J_b, int pp_width_max, real* J_out, int out_width_max, cudaStream_t stream) {
    if (rows <= 0 || dn.layers.empty()) return;

    const real* cur = J_in;
    int cur_wmax = in_width_max;
    real* nxt = J_a;

    for (std::size_t l = 0; l < dn.layers.size(); l++) {
        const DeviceLayer& L = dn.layers[l];
        const bool last = (l + 1 == dn.layers.size());
        real* dst = last ? J_out : nxt;
        const int dst_wmax = last ? out_width_max : pp_width_max;

        if (cur_wmax != L.in_w || dst_wmax != L.out_w) {
            throw std::runtime_error(
                "jet_net_forward: buffer width_max must equal the layer width for the "
                "single-GEMM component stacking to be valid (in " + std::to_string(cur_wmax)
                + " vs " + std::to_string(L.in_w) + ", out " + std::to_string(dst_wmax)
                + " vs " + std::to_string(L.out_w) + ")");
        }

        // Money shot Wx + B
        gemm_rowmajor(handle, jet_C * rows, L.in_w, L.out_w, cur, params + L.w_off, dst, stream);
        jet_bias_act(dst, params + L.b_off, rows, L.out_w, dst_wmax, dn.act, L.is_output, stream);

        cur = dst; cur_wmax = dst_wmax;
        nxt = (nxt == J_a) ? J_b : J_a;
    }
}

// Evaluate Jet for xi = sum h(r_i)
__global__ void jet_xi_reduce_kernel(const real* __restrict__ J_h, real* __restrict__ J_xi, int Bc) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t out_rows = (std::size_t)Bc;
    const std::size_t total = out_rows * (std::size_t)m_feat;
    if (idx >= total) return;

    const int w = (int)(idx / m_feat), f = (int)(idx % m_feat);
    const std::size_t h_rows = (std::size_t)Bc * N;
    const std::size_t h_stride = jet_block_stride(h_rows, (std::size_t)m_feat);
    const std::size_t x_stride = jet_block_stride(out_rows, (std::size_t)m_feat);

    for (int c = 0; c < jet_C; c++) {
        real acc = (real)0;
        for (int p = 0; p < N; p++) acc += J_h[(std::size_t)c*h_stride + (std::size_t)(w*N + p)*m_feat + f];
        J_xi[(std::size_t)c*x_stride + idx] = acc;
    }
}

void jet_xi_reduce(const real* J_h, real* J_xi, int Bc, cudaStream_t stream) {
    if (Bc <= 0) return;
    const std::size_t total = (std::size_t)Bc * m_feat;
    const int threads = 256;
    jet_xi_reduce_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(J_h, J_xi, Bc);
    cuda_sync_check("jet_xi_reduce");
}