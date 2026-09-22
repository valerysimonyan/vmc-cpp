#pragma once

#include "arena.h"

void sweep_device_graph(DeviceState& ds, cublasHandle_t handle, int B, double step);
