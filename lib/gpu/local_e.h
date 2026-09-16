#pragma once

#include "arena.h"

int eval_local_E_device(DeviceState& ds, cublasHandle_t handle, const Ansatz& a, Workspace& ws, int B, cudaStream_t stream = 0);
