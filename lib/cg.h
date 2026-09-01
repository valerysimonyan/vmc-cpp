#pragma once

#include <functional>
#include <vector> 

struct CGResult {
  int iters; 
  double rel_residual;
  bool converged;  
};

CGResult cg_solve(const std::function<void(const std::vector<double>& v, std::vector<double>& Av)>& matvec, 
                    const std::vector<double>& b, 
                    std::vector<double>& x, 
                    const std::vector<double>& M_inv_diag, 
                    double rel_tol, 
                    int max_iters);