#pragma once

#include <limits>
#include <string> 

#include "wavefunction.h"

// Training state stored with the parameters, so a resumed run continues where this one stopped
struct CheckpointState {
    int iter = 0;                                                 // next iteration to run
    double lambda = 0.0;                                          // SR lambda of that iteration
    double best_E_ucb = std::numeric_limits<double>::infinity();  // lowest E + E_err of best_checkpoint.txt so far
};

void save_checkpoint(const std::string& path, const Ansatz& a, const CheckpointState& st);

CheckpointState load_checkpoint(const std::string& path, Ansatz& a);

void load_transfer(const std::string& path, Ansatz& a);