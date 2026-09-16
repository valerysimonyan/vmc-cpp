#pragma once

#include "arena.h"

void propose_coord(DeviceState& ds, int B, double step, cudaStream_t stream = 0);

void accept_coord(DeviceState& ds, int B, cudaStream_t stream = 0);

void recenter_device(DeviceState& ds, int B, cudaStream_t stream = 0);

void propose_discrete(DeviceState& ds, int B, bool is_spin, cudaStream_t stream = 0);
void accept_discrete(DeviceState& ds, int B, bool is_spin, cudaStream_t stream = 0);

void download_acceptance(DeviceState& ds, int B, long long& acc, long long& sp_acc, long long& tau_acc);


