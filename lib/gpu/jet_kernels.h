#pragma once

#include "layouts.h"
#include "net_forward.h"
#include "../network.h"

void build_jet_feat(const real* x, const real* s, const real* t, real* J_feat, int Bc, cudaStream_t stream = 0);

void jet_bias_act(real* J, const real* bias, int rows, int width, int width_max, Activation act, bool is_output, cudaStream_t stream = 0);

void jet_net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* J_in, int in_width_max, int rows, real* J_a, real* J_b, int pp_width_max, real* J_out, int out_width_max, cudaStream_t stream = 0);

void jet_xi_reduce(const real* J_h, real* J_xi, int Bc, cudaStream_t stream = 0);
