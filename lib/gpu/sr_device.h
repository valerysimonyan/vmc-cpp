#pragma once

#include "layouts.h"
#include "../cg.h"
#include "../descent.h"

#include <functional>
#include <vector>

struct DeviceState;


void build_mask(const unsigned char* valid, double* m, std::size_t Ns, cudaStream_t stream = 0);
void O_exp_device(cublasHandle_t h, const double* O_pool, const double* m, std::size_t Ns, std::size_t P, long long n_valid, double* O_exp, cudaStream_t stream = 0);
void S_diag_device(const double* O_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, double* S_diag, cudaStream_t stream = 0);

double rms_update_device(cublasHandle_t h, const double* grad, double* v_rms, double* d_rms, std::size_t P, cudaStream_t stream = 0);


struct ClipStats { double clip_lo, clip_hi, E_clip_mean; };

ClipStats clip_stats_host(const std::vector<double>& E_pool, const std::vector<unsigned char>& valid_pool, std::size_t n_samples, long long n_valid);
void grad_device(cublasHandle_t h, const double* O_pool, const double* E_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, const ClipStats& cs, double* E_clip, double* grad, cudaStream_t stream = 0);


struct SROpDevice {
    cublasHandle_t h = nullptr;
    const double* O_pool = nullptr, *O_exp = nullptr, *m = nullptr, *S_diag = nullptr, *d_rms = nullptr;
    double* t = nullptr;                       
    std::size_t Ns = 0, P = 0;
    long long n_valid = 0;
    double lambda_diag = 0.0, eps_abs = 0.0;
    void apply(const double* v, double* out, bool raw = false);   
};


void M_inv_device(const double* S_diag, const double* d_rms, double lambda_t, double* M_inv, std::size_t P, cudaStream_t stream = 0);

using DeviceMatVec = std::function<void(const double* v, double* out)>;

CGResult cg_solve_device(cublasHandle_t h, const DeviceMatVec& matvec, const double* b, double* x, const double* M_inv_diag, std::size_t n, double rel_tol, int max_iters, double* r, double* z, double* p, double* Ap, long long* n_scalar_downloads = nullptr, cudaStream_t stream = 0);

SRStepLog SR_step_device(DeviceState& ds, cublasHandle_t h, Ansatz& a, int iter, std::size_t n_samples, long long n_valid, std::vector<double>& delta_host, long long* n_scalar_downloads = nullptr, cudaStream_t stream = 0);
