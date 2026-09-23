#pragma once

#include "layouts.h"
#include "net_kernels.h"
#include "../network.h"
#include "../physics.h"

#include <vector>

struct DeviceLayer {
    int in_w, out_w;
    std::size_t w_off, b_off;
    bool is_output;
};

struct DeviceNet {
    std::vector<DeviceLayer> layers;
    std::size_t base_off = 0;
    Activation act = Activation::Gelu;
    int max_width = 0; 
    int hidden_width = 0;

    const float* params_f = nullptr;
    const real*  params_src = nullptr;
    float* in_f = nullptr;   std::size_t in_f_cap = 0;
    float* out_f = nullptr;  std::size_t out_f_cap = 0;
    float* jin_f = nullptr;  std::size_t jin_f_cap = 0;
    float* jout_f = nullptr; std::size_t jout_f_cap = 0;

    void build(const Network& net, std::size_t base);
};

struct NetCache {
    std::vector<DeviceArray<real>> a_in, z;
    std::size_t rows_cap = 0;
    void alloc(const DeviceNet& dn, std::size_t rows);
    std::size_t bytes() const;
};

void net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* in, int rows, real* a, real* b, real* out, cudaStream_t stream = 0, NetCache* cache = nullptr);
