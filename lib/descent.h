#pragma once

#include <vector>

#include "physics.h"
#include "sr.h"
#include "pool.h"
#include "checkpoint.h"
#include "walkers.h"

struct DescentResult {
    double El_exp;
    double El_err;
    double var;
    double acceptance;
    double spin_acceptance;
    double tau_acceptance;
    double L2;
    double r_rms;
    double total_node_hits;
};

struct SRStepLog {
    double lambda = 0.0;
    int cg_iters = 0;
    double cg_residual = 0.0;
    double delta_norm = 0.0;
    double sq_metric_norm = 0.0;
    bool norm_capped = false;
};

void ADAM(const std::vector<double>& grad, std::vector<double>& m, std::vector<double>& v, int i, Ansatz& a);

SRStepLog SR_step(const std::vector<double>& grad, const std::vector<double>& O_pool, const std::vector<double>& O_exp, Ansatz& a, SROp& sr_op, std::vector<double>& delta, int iter, std::size_t n_params, ThreadPool* pool, const double* d_rms, std::size_t n_samples, const uint8_t* valid_pool, std::size_t n_valid, std::vector<double>& M_inv_diag, std::vector<double>& S_delta);

void compute_obs(const BatchStats& bs, DescentResult& r, std::size_t P, const std::vector<double>& E_pool, const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, ThreadPool* pool, std::vector<double>& O_exp, std::vector<double>& grad);

DescentResult descent(Ansatz& a);

DescentResult evaluate_frozen(Ansatz& a);
