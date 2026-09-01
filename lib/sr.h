#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

#include "pool.h"

struct SROp {
    const std::vector<double>* O_pool = nullptr; 
    const std::vector<double>* O_exp = nullptr; 
    std::size_t Ns = 0, P = 0;
    double lambda_diag = 0.0;
    double eps_abs = 0.0;
    const double* d_rms = nullptr;  // length P, only read when sr_rms_damp
    const uint8_t* valid = nullptr; // length Ns; nullptr = all rows of O_pool evaluated
    std::size_t n_valid = 0;        // Only when valid != nullptr
    std::vector<double> S_diag;
    std::vector<double> partials; 
    ThreadPool* pool = nullptr;

    void init(const std::vector<double>& O_pool_, 
                const std::vector<double>& O_exp_, 
                std::size_t Ns_, 
                std::size_t P_, 
                double lamdba_diag_, 
                double eps_abs_,
                ThreadPool* pool_,
                const double* d_rms_,
                const uint8_t* valid,
                std::size_t n_valid_ = 0);

    void apply(const std::vector<double>& v, 
                std::vector<double>& out, bool raw = false);
};

