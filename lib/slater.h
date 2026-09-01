#pragma once

#include "autodiff.h"

#include <vector>
#include <cmath>

inline double value_of(double x) { return x; }
inline double value_of(const Jet& x) { return x.v; }

// Take determinant of Jet or double by decomposing into traingular matrix who's determinant is just the product of the diagonals, M is flatteneed nxn matrix where M(i,j) = M[i*n+j]
template<typename T>
T lu_det(std::vector<T>& M, int n, std::vector<int>& piv_scratch){
    // Scratch row is there to store information during use then get thrown away
    if ((int)piv_scratch.size() != n) piv_scratch.resize(n);

    // Track sign of determinant independently
    int sign = 1;

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
        if (piv_val < 1e-300) return T(0.0);
            
        // Store the row with the largest element in the column, if the largest in the column wasn't along the diagonal swap the rows so the diagonal has the largest value, every such swap flips sign
        piv_scratch[i] = piv_row;
        if(piv_row != i) {
            for (int j = 0; j < n; j++) {
                std::swap(M[i*n+j],M[piv_row*n+j]);
            }
            sign = -sign;
        }

        // For every row below the new diagonal you subtract M(j,k) -= M(j,i) * M(i,k) / M(i,i) at the end setting every row below the diagonal to zero
        T pivot = M[i*n+i];
        for (int j = i+1; j < n; j ++) {
            T factor = M[j*n + i] / pivot;
            for (int k = i +1; k < n; k++) {
                M[j*n+k] = M[j*n+k] - factor * M[i*n+k];
            }
        }
    }

    // With the matrix now triangular the determinant is just the product of the diagonals 
    T det = M[0];
    for (int i = 1; i < n; i++) {
        det = det * M[i*n+i];
    }
    return sign < 0 ? -det : det;
}

double lu_det_inv(std::vector<double>& M, int n, std::vector<double>& Minv, std::vector<int>& piv_scratch, std::vector<double>& col_scratch);

Jet det_jet_from_minv(const Jet* Mjet, int n, std::vector<double>& Mval, std::vector<double>& Minv, std::vector<int>& piv, std::vector<double>& col_scratch, std::vector<double>& G, std::vector<double>& B);
inline Jet det_jet_from_minv(const std::vector<Jet>& Mjet, int n, std::vector<double>& Mval, std::vector<double>& Minv, std::vector<int>& piv, std::vector<double>& col_scratch, std::vector<double>& G, std::vector<double>& B) {
    return det_jet_from_minv(Mjet.data(), n, Mval, Minv, piv, col_scratch, G, B);
}

double det_ratio_rank2(const double* Minv, int n, const double* dci, int i, const double* dcj, int j);