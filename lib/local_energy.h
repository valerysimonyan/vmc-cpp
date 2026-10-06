#pragma once

#include "wavefunction.h"

#include <vector>

double V_3N(const std::vector<double>& x);

double V_coulomb(const std::vector<double>& x, const std::vector<double>& t);

double l2_local(const double* x_shifted, const double* grad, double psi_val);

bool local_E(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out, double& E_out);
