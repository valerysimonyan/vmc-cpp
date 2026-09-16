#pragma once

#include <cstddef>

void gpu_smoke_saxpy(double alpha, const double* hx, const double* hy, double* hout, std::size_t n);

const char* gpu_device_name();  