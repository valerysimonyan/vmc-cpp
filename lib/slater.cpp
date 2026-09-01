#include "slater.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>


// Evaluate inverse and derterminant simultaneously for d(det M)_dM(i,j) = Minv(i,j) det M
double lu_det_inv(std::vector<double>& M, int n, std::vector<double>& Minv, std::vector<int>& piv_scratch, std::vector<double>& col_scratch) {
    if ((int)piv_scratch.size() != n) piv_scratch.resize(n);
    if ((int)col_scratch.size() != n) col_scratch.resize(n);
    if ((int)Minv.size() != n*n) Minv.resize(n*n);

    int sign = 1.;

    for (int i = 0; i < n; i++) {
        // Pick a column, then go to diagonal elements, go down in rows below the diagonal and find the largest value, store it 
        int piv_row = i;
        double piv_val = std::fabs(value_of(M[i*n+i]));
        for (int j = i+1; j < n; j++) {
            double v = std::fabs(value_of(M[j*n+i]));
            if (v > piv_val) {
                piv_val = v;
                piv_row = j;
            }
        }
        
        // If pivot zero determinant zero
        if (piv_val < 1e-300) {
            std::fill(Minv.begin(), Minv.end(), 0.0);
            return 0.0;
        }
            
        // Store the row with the largest element in the column, if the largest in the column wasn't along the diagonal swap the rows so the diagonal has the largest value, every such swap flips sign
        piv_scratch[i] = piv_row;
        if(piv_row != i) {
            for (int j = 0; j < n; j++) {
                std::swap(M[i*n+j],M[piv_row*n+j]);
            }
            sign = -sign;
        }

        // For every row below the new diagonal you subtract M(j,k) -= M(j,i) * M(i,k) / M(i,i) at the end setting every row below the diagonal to zero
        double pivot = M[i*n+i];
        for (int j = i+1; j < n; j ++) {
            double factor = M[j*n + i] / pivot;
            M[j*n+i] = factor; 
            for (int k = i +1; k < n; k++) {
                M[j*n+k] = M[j*n+k] - factor * M[i*n+k];
            }
        }
    }

    // Evaluate determinant, in this proccess M has been put into triangular form, M x_i = e_i -> P M x_i = P e_i -> L U x_i = P e_i -> L y_i = P e_i where diag(L) = (1,..,1)
    double det = M[0];
    for (int i = 1; i < n; i++) {
        det = det * M[i*n+i];
    }
    if (sign < 0) det = -det;

    // Evaluate inverse by solving for inverse column by column
    for (int i = 0; i < n; i++) {
        // Set element of col_scratch to 1 if along the right column, otherwise zero
        for (int j = 0; j < n; j++) {
            col_scratch[j] = (i == j) ? 1.0 : 0.0;
        } 
        // If row got swapped, swap where unit vector is to proper index, get P e_i
        for (int j = 0; j < n; j++) {
            if (piv_scratch[j] != j) std::swap(col_scratch[j], col_scratch[piv_scratch[j]]);
        } 
        
        // Solve y_j = (Pe_i)_j - sum_k<j (L)_jk y_k
        for (int j = 0; j < n; j++) {
            double sum = col_scratch[j];
            for (int k = 0; k < j; k++) {
                sum -= M[j*n+k] * col_scratch[k];
            }
            col_scratch[j] = sum; 
        }

        // Solve y_j = U_jk x_k 
        for (int j = n-1; j >= 0; j--) {
            double sum = col_scratch[j];
            for (int k = j+1; k < n; k++) {
                sum -= M[j*n+k] * col_scratch[k];
            }
            col_scratch[j] = sum / M[j*n+j];
        }

        for (int j = 0; j < n; j++) {
            Minv[j*n+i] = col_scratch[j];
        }
    }
    return det;
}

// Get determinant, derivative, and Laplacian of determinant from inverse using doubles and the fact that d(det M)/dM(i,j) = Minv(j,i) det M
Jet det_jet_from_minv(const Jet* Mjet, int n, std::vector<double>& Mval, std::vector<double>& Minv, std::vector<int>& piv, std::vector<double>& col_scratch, std::vector<double>& G, std::vector<double>& B) {
    const int nn = n*n;
    if ((int)Mval.size() != nn) Mval.resize(nn);
    if ((int)G.size() != nn) G.resize(nn);
    if ((int)B.size() != nn) B.resize(nn);

    for (int i = 0; i < nn; i++) Mval[i] = Mjet[i].v;
    double det_v = lu_det_inv(Mval, n, Minv, piv, col_scratch);
    if (det_v == 0.0) return Jet(0.0);

    std::array<double, D> gsum{};
#ifdef DETJET_SELFCHECK
    std::array<double, D> gmag{};
#endif
    double term1 = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            const double mv = Minv[j*n + k];
            const Jet& e = Mjet[k*n + j];
            for (int a = 0; a < D; a++) {
                gsum[a] += mv * e.g[a];
#ifdef DETJET_SELFCHECK
                gmag[a] += std::fabs(mv * e.g[a]);                
#endif        
            }
            term1 += mv * e.l;
        }
    }

    double term2 = 0.0;
    for (int a = 0; a < D; a++) {
        for (int i = 0; i < nn; i++) G[i] = Mjet[i].g[a];

        std::fill(B.begin(), B.end(), 0.0);
        for (int j = 0; j < n; j++) {
            for (int r = 0; r < n; r++) {
                const double mv = Minv[j*n + r];
                for (int c = 0; c < n; c++) B[j*n + c] += mv * G[r*n + c];
            }
        }
        
        double trB = 0.0;
        for (int j = 0; j < n; j++) trB += B[j*n + j];
#ifdef DETJET_SELFCHECK
        assert(std::fabs(trB - gsum[a]) <= 1e-11 * std::max(1.0, gmag[a]));
#endif
        double trB2 = 0.0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) trB2 += B[j*n + i] * B[i*n + j];
        }
        term2 += trB * trB - trB2;
    } 

    Jet out;
    out.v = det_v;
    for (int a = 0; a < D; a++) out.g[a] = det_v * gsum[a];
    out.l = det_v * (term1 + term2);
    return out;
}

// If only two columbns of M change (i and j) the determinant ratio is simply det detM'/detM = det(1 + V^T Minv U) where RHS matrix is 2x2
// V = [e_i, e_j] (Nx2) and U = [dci, dcj] (Nx2) where M' = M + dci * e_i^T + dcj * e_j^T
double det_ratio_rank2(const double* Minv, int n, const double* dci, int i, const double* dcj, int j) {
    double a00 = 0.0, a01 = 0.0, a10 = 0.0, a11 = 0.0;

    for (int m = 0; m < n; m++) {
        a00 += Minv[i*n + m] * dci[m];
        a01 += Minv[i*n + m] * dcj[m];
        a10 += Minv[j*n + m] * dci[m];
        a11 += Minv[j*n + m] * dcj[m];
    }

    return (1.0 + a00) * (1.0 + a11) - a01 * a10;
}