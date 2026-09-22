#pragma once

#include <cstdint> 
#include <random> 
#include <vector> 

#include "constants.h"
#include "physics.h"
#include "pool.h"
#include "rng_common.h"

struct WalkerBatch {
    int B = 0;
    std::vector<double> x;     // B * D
    std::vector<double> s, t;  // B * N
    std::vector<double> logp;  // B

    std::vector<uint8_t> valid;                   // Can store number in binary from 0-255
    std::vector<std::mt19937> rng;                // B, one rng stream per walker
    std::vector<long long> acc, sp_acc, tau_acc;  // B, counters 
    
    void init(int B_);
};

double u01(WalkerBatch& wb, int w);

double usym(WalkerBatch& wb, int w, double a);

int uint_below(WalkerBatch& wb, int w, int n);


void init_batch(WalkerBatch& wb, const Ansatz& a, ThreadPool* pool, std::vector<Workspace>& wss);

void recenter_batch(WalkerBatch& wb, ThreadPool* pool);

void refresh_logp(WalkerBatch& wb, const Ansatz& a, ThreadPool* pool, std::vector<Workspace>& wss);

void sweep_one(WalkerBatch& wb, int w, const Ansatz& a, double step, Workspace& ws);

double therm_batch(WalkerBatch& wb, const Ansatz& a, double step, int n_sweeps, ThreadPool* pool, std::vector<Workspace>& wss);

struct BatchStats {
    double E_sum = 0.0, E2_sum, l2_sum, r2_sum = 0.0; // Sum over all samples
    long long n_valid = 0, n_invalid = 0; 

    std::vector<double> Ew_sum;   // B, valid sample E sums per walker
    std::vector<double> E2w_sum;  // B, valid sample E2 sums per walker
    std::vector<double> l2w_sum;  // B, valid sample l2 sums per walker
    std::vector<double> r2w_sum;  // B, valid sample r2 sums per walker
    std::vector<int> nw;          // B, valid counts per walker
};

void chunk_range_pub(int B, int n_workers, int th, int& w0, int& w1);

double walker_r2_pub(const double* xw);

void record_one_walker(WalkerBatch& wb, const Ansatz& a, Workspace& ws, std::vector<double>& O, int w, int r, std::vector<double>& E_pool, std::vector<double>& O_pool, std::vector<uint8_t>& valid_pool, BatchStats& bs);

void record_batch(WalkerBatch& wb, const Ansatz& a, double step, int records, ThreadPool* pool, std::vector<Workspace>& wss, std::vector<double>& E_pool, std::vector<double>& O_pool, std::vector<uint8_t>& valid_pool, BatchStats& bs);

double batch_error(const BatchStats& bs);

void masked_O_exp(const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, std::size_t P, ThreadPool* pool, std::vector<double>& O_exp); 