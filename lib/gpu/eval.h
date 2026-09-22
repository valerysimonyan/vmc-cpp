#pragma once

#include "arena.h"

void eval_logp_batch(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream = 0);

void eval_logp_batch_prop(DeviceState& ds, cublasHandle_t handle, int B, const real* x_prop, real* S_prop, real* logp_prop, cudaStream_t stream = 0, bool stash = false);

void build_st_table_batch(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream = 0);

void S_from_table_batch(DeviceState& ds, cublasHandle_t handle, int B, const real* s, const real* t, real* S_out,  cudaStream_t stream = 0);

void feat_combo_unfused(DeviceState& ds, int B, cudaStream_t stream = 0);
