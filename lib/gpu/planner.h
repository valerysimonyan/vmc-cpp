#pragma once

#include <cstddef>
#include <vector>

struct Ansatz;

struct GpuPlan{
    int dev = 0;          // CUDA device index
    std::size_t B = 0;    // Walkers on this device
    double need_gib = 0;  // Needed device memory for B walkers
    double free_gib = 0;  // Available memory reported at time of planning
    unsigned util = 0;    // GPU utilisation (%) at time of planning
};

std::size_t arena_bytes_for(const Ansatz& a, std::size_t B);

std::vector<GpuPlan> plan_gpus(const Ansatz& a);