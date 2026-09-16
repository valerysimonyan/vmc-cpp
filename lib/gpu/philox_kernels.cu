#include "philox_kernels.h"
#include "philox.h"
#include "../constants.h"

// Populate batches of B walkers with random numbers
__global__ void philox_fill_u01_kernel(double* __restrict__ out, int B, int draws, unsigned long long* __restrict__ rng_ctr, unsigned long long seed) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;

    // ONE global read of the counter, then everything local. The key is derived rather than stored -- see stream_for.
    PhiloxStream st = stream_for(seed, w, rng_ctr[w]);

    double* o = out + (std::size_t)w * draws;
    for (int i = 0; i < draws; i++) o[i] = st.u01();

    // ONE global write at the end. The counter is the only RNG state that crosses a kernel boundary.
    rng_ctr[w] = st.ctr;
}

// Checks safety before passing
void philox_fill_u01(DeviceArray<double>& out, int B, int draws_per_walker, DeviceArray<unsigned long long>& rng_ctr, cudaStream_t stream) {
    if (B <= 0 || draws_per_walker <= 0) return;
    const std::size_t need = (std::size_t)B * (std::size_t)draws_per_walker;
    if (out.n < need) throw std::runtime_error("philox_fill_u01: output array too small");
    if (rng_ctr.n < (std::size_t)B) throw std::runtime_error("philox_fill_u01: rng_ctr shorter than B");

    const int threads = 256;
    const int blocks  = (B + threads - 1) / threads;
    philox_fill_u01_kernel<<<blocks, threads, 0, stream>>>(out.d, B, draws_per_walker, rng_ctr.d, rng_seed);
    cuda_sync_check("philox_fill_u01");
}