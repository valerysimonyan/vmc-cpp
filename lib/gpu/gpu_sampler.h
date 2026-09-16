#pragma once

#include "arena.h"
#include "../walkers.h"
#include "../pool.h"

struct HybridTimes {
    double sweep_ms = 0.0;      // device sweeps + recenter
    double localE_ms = 0.0;     // device jet + exchange + energy sum + pool row
    double download_ms = 0.0;   // walkers + per-walker E_loc / l2 / valid
    double o_ms = 0.0;          // host O pass (need_inv double pass + backprops)
    long long n_fallback = 0;   // walkers routed to the host rank-2 fallback
};

double therm_batch_device(DeviceState& ds, cublasHandle_t handle, int B, double step, int n_sweeps, cudaStream_t stream = 0);

void record_batch_hybrid(DeviceState& ds, cublasHandle_t handle, WalkerBatch& wb, const Ansatz& a, double step, int records, ThreadPool* pool, std::vector<Workspace>& wss, PinnedArray& staging, std::vector<double>& E_pool, std::vector<double>& O_pool, std::vector<uint8_t>& valid_pool, BatchStats& bs, HybridTimes* times = nullptr, cudaStream_t stream = 0);

void upload_and_reset(DeviceState& ds, const WalkerBatch& wb, PinnedArray& staging);
