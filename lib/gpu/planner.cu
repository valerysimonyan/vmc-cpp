#include "planner.h"
#include "arena.h"
#include "gpu_util.h"
#include "../wavefunction.h"

#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>


// Get total needed byte count for ansatz and amount of walkers
std::size_t arena_bytes_for(const Ansatz& a, std::size_t B, int records_max) {
    g_dry_alloc = true;
    std::size_t total = 0;
    try {
        DeviceState ds(a, false, B, records_max);
        ds.allocate(a, true, false);
        total = ds.allocated_bytes();
    } catch (...) { 
        g_dry_alloc = false; throw; 
    }
    g_dry_alloc = false;
    return total;
}

// Function take a look at how busy GPU dev is right now as a percent
static unsigned gpu_percent_util(int dev) {
    char bus[32] = {0}; // Allocate 32 characters as zero
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) return 0;  // Try to get physical address on motherboard for device, if nothing found return zero, otherwise write to bus PCId of device
    nvmlDevice_t h;
    if (nvmlDeviceGetHandleByPciBusId_v2(bus, &h) != NVML_SUCCESS) return 0;  // Now given PCId get handle of device
    unsigned mx = 0;  // Store an array to assign a maximum utilization
    // Sample five times to account for variation in GPU, wait 200 ms between each, store maximum utilization extracted, u is filled with gpu utilization rate which we find the maximum of 
    for (int k = 0; k < 5; k++) {
        nvmlUtilization_t u{};
        if (nvmlDeviceGetUtilizationRates(h, &u) == NVML_SUCCESS) mx = std::max(mx, u.gpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return mx;
}

std::vector<GpuPlan> plan_gpus(const Ansatz& a, int records_max, const GpuSettings& gs) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;                    // A GiB of bytes
    const double overhead = gs.overhead_gb, margin = gs.margin_gb;  // Overhead for non arena memory, margin is in case of incorrect estimate or anything else we don't accidentally run out of memory
    const unsigned max_util = (unsigned)gs.max_util;                // Maximum utilization (def is 30%) of a GPU before using another one
    const std::size_t cap = (std::size_t)gs.max_walkers;            // Optional cap on number of walkers per GPU, by default put 0 as in no cap
    const std::size_t B_min = (std::size_t)gs.min_walkers;          // If GPU has less that a threshold amount of walkers allocated by the end its not worth using, it takes more time to bus the information back and forth
    const std::size_t step = 32;                                    // Round walker counts to multiples of 32, use 32 as GPU launches in lockstep 32 parallel threads called a warp

    // Start NVML checking GPU telemetry, then count stores how many available GPUs
    const bool nvml_ok = nvmlInit_v2() == NVML_SUCCESS;
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    // Store plan of how memory will be allocated
    std::vector<GpuPlan> plan;
    std::printf("GPU planner: overhead %.2f GiB + margin %.2f GiB per device, skip above %u%% utilisation\n", overhead, margin, max_util);
    // Iterate through devices available
    for (int d = 0; d < count; d++) {
        // Initialize device
        GpuPlan g;
        g.dev = d;  
        // Set device of focus to device d, all following commands apply to it. Assign amount of free and total memory to f and t respectively, get free GiB and what is utilization for device
        CUDA_CHECK(cudaSetDevice(d));
        std::size_t f = 0, t = 0;
        CUDA_CHECK(cudaMemGetInfo(&f, &t));   
        g.free_gib = (double)f / GiB;
        g.util = nvml_ok ? gpu_percent_util(d) : 0;
        // Device budget is free memory minues overhead and margin. If surpass max_util report and skip device
        const double budget = (double)f - (overhead + margin) * GiB;
        if (g.util > max_util) {
            std::printf("  GPU%d: %.2f GiB free, %u%% busy -> skipped (busy)\n", d, g.free_gib, g.util);
            continue;
        }
        // Look at number of walkers that we can maximally fit for the ansatz while staying withing budget and also hi is beneath a certain absolute limit (4,194,304 walkers denoted by 1u << 22)
        std::size_t lo = 0, hi = step;
        while ((double)arena_bytes_for(a, hi, records_max) <= budget && hi < (1u << 22)) { 
            lo = hi; hi *= 2; 
        }
        // We now have the range of walkers we can fit bounded by a single of power 2. We can now narrow it down to a single warp. Do a binary search , if mid is within budget make it the lower bound. Else the higher bound. Walker allocation always a multiple of step in the end
        while (hi - lo > step) {
            const std::size_t mid = ((lo + hi) / 2 / step) * step;
            if ((double)arena_bytes_for(a, mid, records_max) <= budget) lo = mid; 
            else hi = mid;
        }
        // If there is a cap and the lower bound is above it, set the lo to cap
        if (cap && lo > cap) lo = cap;
        // If lo is beneath the minimum walker count skip the device
        if (lo < B_min) {
            std::printf("  GPU%d: %.2f GiB free -> skipped (fits only %zu walkers)\n", d, g.free_gib, lo);
            continue;
        }
        // We assign lo amount of walkers to g.B and the amount of data we need to g.need_gb. Print amount per device and add to the plan
        g.B = lo;
        g.need_gib = (double)arena_bytes_for(a, lo, records_max) / GiB;
        std::printf("  GPU%d: %.2f GiB free, %u%% busy -> %zu walkers (arena %.2f GiB)\n", d, g.free_gib, g.util, g.B, g.need_gib);
        plan.push_back(g);
    }

    // Now we do shutdown of telemetry reading (if succesfully launched), if empty throw error meaning not enough space. Set maximum amount of device allocations to first two (which in our case is all of them)
    if (nvml_ok) nvmlShutdown();
    if (plan.empty()) throw std::runtime_error("GPU planner: no device can hold min_walkers walkers");
    if (plan.size() > 2) plan.resize(2);
    // Now set active GPU to first device approved by plan
    CUDA_CHECK(cudaSetDevice(plan[0].dev));
    return plan;
}