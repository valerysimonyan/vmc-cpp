#pragma once

#include "arena.h"

void eval_jet_prepare(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream = 0, bool stash = false);

void eval_jet_chunk(DeviceState& ds, cublasHandle_t handle, int Bc, int w_off, int B_tot, cudaStream_t stream = 0);

void eval_jet_batch(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream = 0);
