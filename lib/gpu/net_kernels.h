#pragma once

#include "layouts.h"
#include "../network.h"

void shift_to_com(const real* x, real* x_sh, int B, cudaStream_t stream = 0);

void build_feat(const real* x_sh, const real* s, const real* t, real* feat_in, int B, cudaStream_t stream = 0);

void shift_build_feat(const real* x, const real* s, const real* t, real* x_sh, real* feat_in, int B, cudaStream_t stream = 0);

void bias_act(real* x, const real* bias, int rows, int width, Activation act, bool is_output, cudaStream_t stream = 0);

void bias_act_stash(real* x, real* z_keep, const real* bias, int rows, int width, Activation act, bool is_output, cudaStream_t stream = 0);

void xi_reduce(const real* h_out, real* xi, int B, cudaStream_t stream = 0);

void cast_to_float(const double* in, float* out, std::size_t n, cudaStream_t stream = 0);

void cast_to_double(const float* in, double* out, std::size_t n, cudaStream_t stream = 0);

void bias_act_f(float* z, double* z_keep, const float* bias, int rows, int width, Activation act, cudaStream_t stream = 0);

void bias_out_f(const float* z, const float* bias, double* out, double* z_keep, int rows, int width, cudaStream_t stream = 0);
