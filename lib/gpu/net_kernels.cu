#include "net_kernels.h"

#include <cmath>

// __global__ tells compiler to hit gpu with compilation coming from cpu device, __restrict__ tells only this pointer points to this data
__global__ void shift_to_com_kernel(const real* __restrict__ x, real* __restrict__ x_sh, int B) {
    // Assign thread to each walker
    int w = blockIdx.x;
    if (w >= B) return;
    const real* xw = x + (std::size_t)w * D;
    real* ow = x_sh + (std::size_t)w * D;
    
    // Assign thread for each dimension, then do COM subtraction across each dimension per thread
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        real R = (real)0;
        for (int i = 0; i < N; i++) R += xw[i*dim + d];
        R /= (real)N;
        for (int i = 0; i < N; i++) ow[i*dim + d] = xw[i*dim + d] - R;
    }
}

// Launch B thread blocks for each walker, each block has 32 thread, if issue appears point out error happens at shift_to_com
void shift_to_com(const real* x, real* x_sh, int B, cudaStream_t stream) {
    if (B <= 0) return;
    shift_to_com_kernel<<<B, 32, 0, stream>>>(x, x_sh, B);
    cuda_sync_check("shift_to_com");
}

// GPU accelerated construction of input layer
__global__ void build_feat_kernel(const real* __restrict__ x_sh, const real* __restrict__ s, const real* __restrict__ t, real* __restrict__ feat_in, int B) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= B*N) return;
    int w = r/N, p = r % N; 
    real* f = feat_in + (std::size_t)r * (dim + 2);
    for (int d = 0; d < dim; d++) f[d] = x_sh[(std::size_t)w*D + p*dim + d];
    f[dim] = s[(std::size_t)w*N + p];
    f[dim + 1] = t[(std::size_t)w*N + p];
}

void build_feat(const real* x_sh, const real* s, const real* t, real* feat_in, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int total = B*N, threads = 256;
    build_feat_kernel<<<(total + threads - 1)/threads, threads, 0, stream>>>(x_sh, s, t, feat_in, B);
    cuda_sync_check("build_feat");
}

// Launch activation function, if tanh return tanh, else do GELU
__device__ __forceinline__ real dev_apply_activation(Activation act, real x) {
    if (act == Activation::Tanh) return tanh(x);
    
    real x3 = x*x*x;
    real inner = x + (real)0.044715 * x3;
    real scaled = (real)0.7978845608 * inner;
    real th = tanh(scaled);
    return (real)0.5 * x * ((real)1.0 + th);
}

// Parallelize application of bias and activation to layer
__global__ void bias_act_kernel(real* __restrict__ z, const real* __restrict__ bias, int rows, int width, Activation act, bool is_output) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = (std::size_t)rows * width;
    if (idx >= total) return;
    int col = (int)(idx % (std::size_t)width);
    real v = z[idx] + bias[col];
    z[idx] = is_output ? v : dev_apply_activation(act, v);
}

void bias_act(real* z, const real* bias, int rows, int width, Activation act, bool is_output, cudaStream_t stream) {
    if (rows <= 0 || width <= 0) return;
    const std::size_t total = (std::size_t)rows * width;
    const int threads = 256;
    const std::size_t blocks = (total + threads - 1) / threads;
    bias_act_kernel<<<(unsigned)blocks, threads, 0, stream>>>(z, bias, rows, width, act, is_output);
    cuda_sync_check("bias_act");
}

// Evaluate xi =  sum_p h_out
__global__ void xi_reduce_kernel(const real* __restrict__ h_out, real* __restrict__ xi, int B) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = (std::size_t)B * m_feat;
    if (idx >= total) return;
    int w = (int)(idx / (std::size_t)m_feat);
    int f = (int)(idx % (std::size_t)m_feat);

    real acc = (real)0;
    for (int p = 0; p < N; p++) acc += h_out[(std::size_t)(w*N + p)*m_feat + f];
    xi[idx] = acc;
}

void xi_reduce(const real* h_out, real* xi, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const std::size_t total = (std::size_t)B * m_feat;
    const int threads = 256;
    const std::size_t blocks = (total + threads - 1) / threads;
    xi_reduce_kernel<<<(unsigned)blocks, threads, 0, stream>>>(h_out, xi, B);
    cuda_sync_check("xi_reduce");
}