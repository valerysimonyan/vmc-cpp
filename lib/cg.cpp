#include "cg.h"
#include "util.h"

#include <cmath>

// Solve A v = b for v unknown, v0 = x which we then modify to find v
CGResult cg_solve(const std::function<void(const std::vector<double>& v, std::vector<double>& Av)>& matvec, const std::vector<double>& b, std::vector<double>& x, const std::vector<double>& M_inv_diag, double rel_tol, int max_iters){
    std::size_t n = b.size();
    double b_norm = norm(b);

    // If b is zero have trivial v
    if (b_norm == 0.0) return {0, 0.0, true};

    // Given some x and A which we do not have stored, compute A x
    std::vector<double> Ax(n);
    matvec(x, Ax);

    // Calculate residual r = b - A x
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; i++) {
        r[i] = b[i] - Ax[i];
    }
    
    // Define z = M^-1 r = M^-1 (b - A x), where M^-1 is approximation of just A's diagonal
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; i++) {
       z[i] = M_inv_diag[i] * r[i];
    }

    // Define p = z, compute r . z = r M^-1 r
    std::vector<double> p = z;
    double rz = dot(r,z);

    // We also compute |b - A x| / |b|, when x is close to v within tolerance we may declare success
    double rel_residual = norm(r) / b_norm;
    if (rel_residual < rel_tol) {
        return {0, rel_residual, true};
    }

    // Now we iterate
    std::vector<double> Ap(n);
    for (int i = 0; i < max_iters; i++) {
        // Store A p where for i = 0, A M^-1 r
        matvec(p, Ap);

        // Compute p A p = r M^-1 A M^-1 r which for symmetric positive A must always be greater than zero, if theis is untrue or our value is divergent something went wrong and we didn't converge
        double pAp = dot(p, Ap);
        if(!(pAp > 0.0) || !std::isfinite(pAp)) return {i, norm(r) / b_norm, false};

        // Calculate alpha = (r . z) / (p A p) = (r M^-1 r) / (r M^-1 A M^-1 r), update x_new = x + alpha p and r_new = b - A x_new = b - A x - alpha A p, take step down along gradient
        double alpha = rz / pAp;
        for (std::size_t j = 0; j < n; j++) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Ap[j];
        }

        // Check if we have converged
        rel_residual = norm(r) / b_norm;
        if (rel_residual < rel_tol) return {i+1, rel_residual, true};

        // If not, update z to M^-1 r_new
        for (std::size_t j = 0; j < n; j++) {
            z[j] = M_inv_diag[j] * r[j];
        }

        // Update r . z and beta which is (r . z)_new / (r . z)_old, update p_new = z_new + beta p _old 
        double rz_new = dot(r, z);
        double beta = rz_new/rz;
        for (std::size_t j = 0; j < n; j++) {
            p[j] = z[j] + beta * p[j];
        }

        rz = rz_new;
    }
    
    return {max_iters, rel_residual, false};
}