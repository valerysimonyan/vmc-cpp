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

// Combine shift to COM coordinates and building the input layer for single particle networks for memroy benefits
__global__ void shift_build_feat_kernel(const real* __restrict__ x, const real* __restrict__ s, const real* __restrict__ t, real* __restrict__ x_sh, real* __restrict__ feat_in, int B) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= B*N) return;
    const int w = r / N, p = r % N;
    const real* xw = x + (std::size_t)w * D;
    real* ow = x_sh + (std::size_t)w * D;
    real* f = feat_in + (std::size_t)r * (dim + 2);
    for (int d = 0; d < dim; d++) {
        real R = (real)0;
        for (int i = 0; i < N; i++) R += xw[i*dim + d];
        R /= (real)N;
        const real v = xw[p*dim + d] - R;
        ow[p*dim + d] = v;
        f[d] = v;
    }
    f[dim] = s[(std::size_t)w*N + p];
    f[dim + 1] = t[(std::size_t)w*N + p];
}

void shift_build_feat(const real* x, const real* s, const real* t, real* x_sh, real* feat_in, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int total = B*N, threads = 256;
    shift_build_feat_kernel<<<(total + threads - 1)/threads, threads, 0, stream>>>(x, s, t, x_sh, feat_in, B);
    cuda_sync_check("shift_build_feat");
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

// Apply Bias
__global__ void bias_act_stash_kernel(real* __restrict__ z, real* __restrict__ z_keep, const real* __restrict__ bias, int rows, int width, Activation act, bool is_output) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = (std::size_t)rows * width;
    if (idx >= total) return;
    int col = (int)(idx % (std::size_t)width);
    real v = z[idx] + bias[col];
    z_keep[idx] = v;
    z[idx] = is_output ? v : dev_apply_activation(act, v);
}

void bias_act_stash(real* z, real* z_keep, const real* bias, int rows, int width, Activation act, bool is_output, cudaStream_t stream) {
    if (rows <= 0 || width <= 0) return;
    const std::size_t total = (std::size_t)rows * width;
    const int threads = 256;
    const std::size_t blocks = (total + threads - 1) / threads;
    bias_act_stash_kernel<<<(unsigned)blocks, threads, 0, stream>>>(z, z_keep, bias, rows, width, act, is_output);
    cuda_sync_check("bias_act_stash");
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

// Double to float
__global__ void cast_d2f_kernel(const double* __restrict__ in, float* __restrict__ out, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

void cast_to_float(const double* in, float* out, std::size_t n, cudaStream_t stream) {
    if (n == 0) return;
    cast_d2f_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(in, out, n);
    cuda_sync_check("cast_to_float");
}

// Float to Double
__global__ void cast_f2d_kernel(const float* __restrict__ in, double* __restrict__ out, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (double)in[i];
}

void cast_to_double(const float* in, double* out, std::size_t n, cudaStream_t stream) {
    if (n == 0) return;
    cast_f2d_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(in, out, n);
    cuda_sync_check("cast_to_double");
}

// The same activations as dev_apply_activation, evaluated in float.
__device__ __forceinline__ float dev_apply_activation_f(Activation act, float x) {
    if (act == Activation::Tanh) return tanhf(x);
    const float x3 = x*x*x;
    const float inner = x + 0.044715f * x3;
    const float th = tanhf(0.7978845608f * inner);
    return 0.5f * x * (1.0f + th);
}

// Hidden layer: z <- act(z + b) in float
__global__ void bias_act_f_kernel(float* __restrict__ z, double* __restrict__ z_keep, const float* __restrict__ bias, int rows, int width, Activation act) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (std::size_t)rows * width) return;
    const float v = z[idx] + bias[(int)(idx % (std::size_t)width)];
    if (z_keep) z_keep[idx] = (double)v;
    z[idx] = dev_apply_activation_f(act, v);
}
void bias_act_f(float* z, double* z_keep, const float* bias, int rows, int width, Activation act, cudaStream_t stream) {
    if (rows <= 0 || width <= 0) return;
    const std::size_t total = (std::size_t)rows * width;
    bias_act_f_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(z, z_keep, bias, rows, width, act);
    cuda_sync_check("bias_act_f");
}

// Output layer: out = (double)(z + b). 
__global__ void bias_out_f_kernel(const float* __restrict__ z, const float* __restrict__ bias, double* __restrict__ out, double* __restrict__ z_keep, int rows, int width) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (std::size_t)rows * width) return;
    const double v = (double)(z[idx] + bias[(int)(idx % (std::size_t)width)]);
    out[idx] = v;
    if (z_keep) z_keep[idx] = v;
}
void bias_out_f(const float* z, const float* bias, double* out, double* z_keep, int rows, int width, cudaStream_t stream) {
    if (rows <= 0 || width <= 0) return;
    const std::size_t total = (std::size_t)rows * width;
    bias_out_f_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(z, bias, out, z_keep, rows, width);
    cuda_sync_check("bias_out_f");
}