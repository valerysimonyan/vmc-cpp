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

void net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* in, int rows, real* a, real* b, real* out, cudaStream_t stream = 0);
