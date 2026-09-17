#pragma once

#include "arena.h"
#include "../walkers.h"

#include <vector>

struct RecordTimes {
    double sweep_ms = 0.0;    // record-round sweeps + recenter
    double localE_ms = 0.0;   // device local_E (jets + exchange + energy) + pool row + walker stats
    double o_ms = 0.0;        // assemble_O_batch
    long long n_fallback = 0; // walkers routed to the host rank-2 fallback
};

void record_batch_device(DeviceState& ds, cublasHandle_t handle, const Ansatz& a, Workspace& ws, int B, double step, int records, bool with_O, RecordTimes* times = nullptr, cudaStream_t stream = 0);

struct IterStatsHost {
    std::vector<double> E_pool;
    std::vector<unsigned char> valid_pool;
    BatchStats bs;
    long long acc = 0, sp_acc = 0, tau_acc = 0;
};

void download_iteration(DeviceState& ds, int B, int records, PinnedArray& staging, IterStatsHost& out);
