#pragma once

#include "layouts.h"
#include "../network.h"

void shift_to_com(const real* x, real* x_sh, int B, cudaStream_t stream = 0);

void build_feat(const real* x_sh, const real* s, const real* t, real* feat_in, int B, cudaStream_t stream = 0);

void bias_act(real* x, const real* bias, int rows, int width, Activation act, bool is_output, cudaStream_t stream = 0);

void xi_reduce(const real* h_out, real* xi, int B, cudaStream_t stream = 0);
