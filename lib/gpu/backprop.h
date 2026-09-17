#pragma once

#include "layouts.h"
#include "net_forward.h"

struct DeviceState;

void act_grad_eval(const real* z, real* out, std::size_t n, Activation act, cudaStream_t stream = 0);

void act_grad_mul(real* delta, const real* z, std::size_t n, Activation act, cudaStream_t stream = 0);

void dW_strided(cublasHandle_t handle, const real* a, const real* delta, int R, int in_w, int out_w, int Bc, double* C_first, long long strideC, cudaStream_t stream = 0);

void db_rows(const real* delta, int R, int out_w, int Bc, double* O_first, std::size_t b_off, std::size_t P, cudaStream_t stream = 0);

void backprop_net(cublasHandle_t handle, const DeviceNet& dn, const NetCache& cache, const real* params, real* cur, real* nxt, real* WT, int R, int Bc, int w_off, double* O_first, std::size_t P, real* dinput, cudaStream_t stream = 0);

void assemble_O_batch(DeviceState& ds, cublasHandle_t handle, int r, int B, cudaStream_t stream = 0, int chunk = 0);
