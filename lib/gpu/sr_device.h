#pragma once

#include "layouts.h"
#include "../cg.h"
#include "../sr.h"
#include "../train.h"

#include <functional>
#include <vector>

struct DeviceState;


void build_mask(const unsigned char* valid, double* m, std::size_t Ns, cudaStream_t stream = 0);
void O_exp_device(cublasHandle_t h, const opool_t* O_pool, const double* m, std::size_t Ns, std::size_t P, long long n_valid, double* O_exp, cudaStream_t stream = 0);
void S_diag_device(const opool_t* O_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, double* S_diag, cudaStream_t stream = 0);


double rms_update_device(cublasHandle_t h, const double* grad, double* v_rms, double* d_rms, std::size_t P, cudaStream_t stream = 0);

void grad_device(cublasHandle_t h, const opool_t* O_pool, const double* E_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, const ClipStats& cs, double* E_clip, double* grad, cudaStream_t stream = 0);

void M_inv_device(const double* S_diag, const double* d_rms, double lambda_t, double* M_inv, std::size_t P, cudaStream_t stream = 0);

using DeviceMatVec = std::function<void(const double* v, double* out)>;

CGResult cg_solve_device(cublasHandle_t h, const DeviceMatVec& matvec, const double* b, double* x, const double* M_inv_diag, std::size_t n, double rel_tol, int max_iters, double* r, double* z, double* p, double* Ap, long long* n_scalar_downloads = nullptr, cudaStream_t stream = 0);



struct ReplicaRef { 
    DeviceState* ds = nullptr;   // Initialize DeviceState as null
    cublasHandle_t h = nullptr;  // Handle of device
    int dev = 0;                 // Device number
    std::size_t Ns = 0;          // Device sample rows
};

// <O> over every replica's valid rows; the result is copied to every replica
void O_exp_replicas(const std::vector<ReplicaRef>& R, std::size_t P, long long n_valid);

// Energy gradient with clipped local energies, summed into replica 0
void grad_replicas(const std::vector<ReplicaRef>& R, std::size_t P, long long n_valid, const ClipStats& cs);

// out = (S + regularisation) v over every replica's rows; v and out live on replica 0
void sr_apply(const std::vector<ReplicaRef>& R, std::size_t P, long long n_valid, double lambda_diag, const double* v, double* out, bool raw, long long* n_scalar_downloads = nullptr);

// One SR step (CG solve, trust and norm caps, parameter update); returns the step's log
SRStepLog SR_step_device(const std::vector<ReplicaRef>& R, Ansatz& a, int iter, long long n_valid, std::vector<double>& delta_host, long long* n_scalar_downloads = nullptr);
