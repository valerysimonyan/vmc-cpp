#pragma once

#include "gpu_util.h"

#include <cstddef>

void philox_fill_u01(DeviceArray<double>& out, int B, int draws_per_walker, DeviceArray<unsigned long long>& rng_ctr, cudaStream_t stream = 0);
