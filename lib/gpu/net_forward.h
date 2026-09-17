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

    void build(const Network& net, std::size_t base);
};

struct NetCache {
    std::vector<DeviceArray<real>> a_in, z;
    std::size_t rows_cap = 0;
    void alloc(const DeviceNet& dn, std::size_t rows);
    std::size_t bytes() const;
};

void net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* in, int rows, real* a, real* b, real* out, cudaStream_t stream = 0, NetCache* cache = nullptr);
