#include "smoke.h"
#include "gpu_util.h"

#include <string> 

// Runs a test first by ensuring memory allocation is working properly
__global__ void saxpy_check(double alpha, const double* __restrict__ x,
                            const double* __restrict__ y, double* __restrict__ out,
                            std::size_t n) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = alpha * x[i] + y[i];
}

// Runs smoke test
void gpu_smoke_saxpy(double alpha, const double* hx, const double* hy,
                     double* hout, std::size_t n) {
    if (n == 0) return;

    DeviceArray<double> dx, dy, dout;
    dx.alloc(n);   dx.up(hx, n);
    dy.alloc(n);   dy.up(hy, n);
    dout.alloc(n); dout.zero();

    const int threads = 256;
    const std::size_t blocks = (n + threads - 1) / threads;
    saxpy_check<<<(unsigned)blocks, threads>>>(alpha, dx.d, dy.d, dout.d, n);
    cuda_sync_check("saxpy_check");

    dout.down(hout, n);
    // dx/dy/dout free themselves here, including on any throw above.
}

// Prints device name in human language
const char* gpu_device_name() {
    static std::string name;
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("no CUDA device visible");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    name = std::string(prop.name) + " (sm_" + std::to_string(prop.major)
         + std::to_string(prop.minor) + ", "
         + std::to_string(prop.totalGlobalMem / (1024ull*1024*1024)) + " GiB)";
    return name.c_str();
}