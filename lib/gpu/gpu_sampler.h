#pragma once

#include "arena.h"
#include "../walkers.h"
#include "../pool.h"

void sweep_device(DeviceState& ds, cublasHandle_t handle, int B, double step, cudaStream_t stream = 0);

double therm_batch_device(DeviceState& ds, cublasHandle_t handle, int B, double step, int n_sweeps, cudaStream_t stream = 0);

void upload_and_reset(DeviceState& ds, const WalkerBatch& wb, PinnedArray& staging);
