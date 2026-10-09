#include "sr_device.h"
#include "arena.h"
#include "prof.h"
#include "../constants.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check_blas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int)st));
}
static double ddot(cublasHandle_t h, std::size_t n, const double* a, const double* b, long long* n_dl) {
    VMC_PROF_HOST("scalars_dn");   // host pointer mode: this call synchronises
    double r = 0.0;
    check_blas(cublasDdot(h, (int)n, a, 1, b, 1, &r), "cublasDdot");   // host pointer mode: blocks, one scalar down
    if (n_dl) (*n_dl)++;
    return r;
}
static const int T256 = 256;
static unsigned blocks_for(std::size_t n) { return (unsigned)((n + T256 - 1) / T256); }

static constexpr int ot_slabs = 32;

// Sum_i O_ij x_i
__global__ void ot_x_partial_kernel(const float* __restrict__ O, const double* __restrict__ x, double* __restrict__ part, std::size_t Ns, std::size_t P) {
    const std::size_t j = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P) return;
    const std::size_t s = blockIdx.y, per = (Ns + ot_slabs - 1) / ot_slabs;
    const std::size_t i0 = s * per, i1 = (i0 + per < Ns) ? i0 + per : Ns;
    double acc = 0.0;
    for (std::size_t i = i0; i < i1; i++) acc += (double)O[i*P + j] * x[i];
    part[s*P + j] = acc;
}

__global__ void ot_x_reduce_kernel(const double* __restrict__ part, double* __restrict__ y, std::size_t P) {
    const std::size_t j = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P) return;
    double acc = 0.0;
    for (int s = 0; s < ot_slabs; s++) acc += part[(std::size_t)s*P + j];
    y[j] = acc;
}

// Sum_j O_ij v_j
__global__ void o_v_kernel(const float* __restrict__ O, const double* __restrict__ v, double* __restrict__ t, std::size_t P) {
    __shared__ double red[T256];
    const std::size_t i = blockIdx.x;
    const float* row = O + i * P;
    double acc = 0.0;
    for (std::size_t j = threadIdx.x; j < P; j += T256) acc += (double)row[j] * v[j];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int w = T256 / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) t[i] = red[0];
}

static DeviceArray<double>& ot_scratch() { 
    static DeviceArray<double> a[8]; // Allocate up to 8 devices 
    int d = 0; 
    CUDA_CHECK(cudaGetDevice(&d)); // Calll current device ID
    return a[d]; // Return d'th active DeviceArray 
}   

// y = O^T x 
template <typename T>
static void gemv_OT(cublasHandle_t h, const T* O, const double* x, double* y, std::size_t Ns, std::size_t P, cudaStream_t stream, const char* what) {
    if constexpr (std::is_same<T, float>::value) {
        (void)h; (void)what;
        DeviceArray<double>& part = ot_scratch();
        if (part.n < (std::size_t)ot_slabs * P) part.alloc((std::size_t)ot_slabs * P);
        ot_x_partial_kernel<<<dim3(blocks_for(P), ot_slabs), T256, 0, stream>>>(O, x, part.d, Ns, P);
        ot_x_reduce_kernel<<<blocks_for(P), T256, 0, stream>>>(part.d, y, P);
        cuda_sync_check("gemv_OT (fp32_opool)");
    } else {
        cublasSetStream(h, stream);
        const double one = 1.0, zero = 0.0;
        check_blas(cublasDgemv(h, CUBLAS_OP_N, (int)P, (int)Ns, &one, O, (int)P, x, 1, &zero, y, 1), what);
    }
}
// t = O v.
template <typename T>
static void gemv_O(cublasHandle_t h, const T* O, const double* v, double* t, std::size_t Ns, std::size_t P, cudaStream_t stream, const char* what) {
    if constexpr (std::is_same<T, float>::value) {
        (void)h; (void)what;
        o_v_kernel<<<(unsigned)Ns, T256, 0, stream>>>(O, v, t, P);
        cuda_sync_check("gemv_O (fp32_opool)");
    } else {
        cublasSetStream(h, stream);
        const double one = 1.0, zero = 0.0;
        check_blas(cublasDgemv(h, CUBLAS_OP_T, (int)P, (int)Ns, &one, O, (int)P, v, 1, &zero, t, 1), what);
    }
}


// Check sample validity, ignore if invalid
__global__ void mask_kernel(const unsigned char* __restrict__ valid, double* __restrict__ m, std::size_t Ns) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ns) return;
    m[i] = valid[i] ? 1.0 : 0.0;
}
void build_mask(const unsigned char* valid, double* m, std::size_t Ns, cudaStream_t stream) {
    if (Ns == 0) return;
    mask_kernel<<<blocks_for(Ns), T256, 0, stream>>>(valid, m, Ns);
    cuda_sync_check("build_mask");
}

// Mask statistics, if invalid drop
__global__ void div_kernel(double* __restrict__ x, double d, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = x[i] / d;
}

void O_exp_device(cublasHandle_t h, const opool_t* O_pool, const double* m, std::size_t Ns, std::size_t P, long long n_valid, double* O_exp, cudaStream_t stream) {
    gemv_OT(h, O_pool, m, O_exp, Ns, P, stream, "O_exp_device");
    if (n_valid == 0) { CUDA_CHECK(cudaMemset(O_exp, 0, P * sizeof(double))); return; }
    div_kernel<<<blocks_for(P), T256, 0, stream>>>(O_exp, (double)n_valid, P);
    cuda_sync_check("O_exp_device");
}

// Evaluate diagonal part of SR matrix
__global__ void S_diag_kernel(const opool_t* __restrict__ O_pool, const unsigned char* __restrict__ valid, const double* __restrict__ O_exp, std::size_t Ns, std::size_t P, double n_valid, double* __restrict__ S_diag) {
    std::size_t j = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P) return;
    const double oe = O_exp[j];
    double acc = 0.0;
    for (std::size_t i = 0; i < Ns; i++) {
        if (!valid[i]) continue;
        const double diff = (double)O_pool[i*P + j] - oe;
        acc += diff * diff;
    }
    S_diag[j] = acc / n_valid;
}

void S_diag_device(const opool_t* O_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, double* S_diag, cudaStream_t stream) {
    if (P == 0) return;
    S_diag_kernel<<<blocks_for(P), T256, 0, stream>>>(O_pool, valid, O_exp, Ns, P, (double)n_valid, S_diag);
    cuda_sync_check("S_diag_device");
}

// Evalutate regulators from Argonne paper for SR gradient descent
__global__ void rms_kernel(const double* __restrict__ g, double* __restrict__ v, double* __restrict__ d, std::size_t P) {
    std::size_t k = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= P) return;
    v[k] = sr_rms_beta * v[k] + (1.0 - sr_rms_beta) * g[k] * g[k];
    d[k] = sqrt(v[k]) + 1e-8;
}

double rms_update_device(cublasHandle_t h, const double* grad, double* v_rms, double* d_rms, std::size_t P, cudaStream_t stream) {
    rms_kernel<<<blocks_for(P), T256, 0, stream>>>(grad, v_rms, d_rms, P);
    cuda_sync_check("rms_update_device");
    cublasSetStream(h, stream);
    double asum = 0.0;
    check_blas(cublasDasum(h, (int)P, d_rms, 1, &asum), "rms_update_device");
    return sr_rms_eps * asum / (double)P;
}


__global__ void clip_kernel(const double* __restrict__ E, const unsigned char* __restrict__ valid, double lo, double hi, double* __restrict__ E_clip, std::size_t Ns) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ns) return;
    if (!valid[i]) { E_clip[i] = 0.0; return; }
    const double mx = (E[i] < lo) ? lo : E[i];
    E_clip[i] = (hi < mx) ? hi : mx;
}

// Evaluate gradient
__global__ void grad_final_kernel(double* __restrict__ g, const double* __restrict__ O_exp, double nv, double E_clip_mean, std::size_t P) {
    std::size_t k = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= P) return;
    g[k] = 2.0 * (g[k] / nv - E_clip_mean * O_exp[k]);
}

void grad_device(cublasHandle_t h, const opool_t* O_pool, const double* E_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, const ClipStats& cs, double* E_clip, double* grad, cudaStream_t stream) {
    clip_kernel<<<blocks_for(Ns), T256, 0, stream>>>(E_pool, valid, cs.clip_lo, cs.clip_hi, E_clip, Ns);
    cuda_sync_check("clip_kernel");
    gemv_OT(h, O_pool, E_clip, grad, Ns, P, stream, "grad_device");
    grad_final_kernel<<<blocks_for(P), T256, 0, stream>>>(grad, O_exp, (double)n_valid, cs.E_clip_mean, P);
    cuda_sync_check("grad_device");
}

// t_i = (Ov_i - Oexp_v) * m_i 
__global__ void center_kernel(double* __restrict__ t, const double* __restrict__ m, double c, std::size_t Ns) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ns) return;
    t[i] = (t[i] - c) * m[i];
}

// Apply SR regulators
__global__ void apply_tail_kernel(double* __restrict__ out, const double* __restrict__ v, const double* __restrict__ S_diag, const double* __restrict__ d_rms, double nv, double lambda_diag, double eps_abs, bool raw, std::size_t P) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    out[i] = out[i] / nv;
    if (!raw) {
        out[i] += lambda_diag * S_diag[i] * v[i];
        double damp = eps_abs;
        if constexpr (sr_rms_damp) {
            damp += sr_rms_eps * d_rms[i];
        }
        out[i] += damp * v[i];
    }
}

// Full conjugate gradient evaluation
__global__ void sub_kernel(const double* __restrict__ b, const double* __restrict__ Ax, double* __restrict__ r, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r[i] = b[i] - Ax[i];
}
__global__ void jacobi_kernel(const double* __restrict__ Minv, const double* __restrict__ r, double* __restrict__ z, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = Minv[i] * r[i];
}

__global__ void xr_update_kernel(double* __restrict__ x, double* __restrict__ r, const double* __restrict__ p, const double* __restrict__ Ap, double alpha, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
}

__global__ void p_update_kernel(double* __restrict__ p, const double* __restrict__ z, double beta, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = z[i] + beta * p[i];
}

CGResult cg_solve_device(cublasHandle_t h, const DeviceMatVec& matvec, const double* b, double* x, const double* M_inv_diag, std::size_t n, double rel_tol, int max_iters, double* r, double* z, double* p, double* Ap, long long* n_dl, cudaStream_t stream) {
    cublasSetStream(h, stream);
    const unsigned nb = blocks_for(n);
    double b_norm = std::sqrt(ddot(h, n, b, b, n_dl));
    if (b_norm == 0.0) return {0, 0.0, true};

    matvec(x, Ap);                                                    
    sub_kernel<<<nb, T256, 0, stream>>>(b, Ap, r, n);
    { VMC_PROF("precond", stream); jacobi_kernel<<<nb, T256, 0, stream>>>(M_inv_diag, r, z, n); cuda_sync_check("cg init"); }
    CUDA_CHECK(cudaMemcpy(p, z, n * sizeof(double), cudaMemcpyDeviceToDevice));
    double rz = ddot(h, n, r, z, n_dl);

    double rel_residual = std::sqrt(ddot(h, n, r, r, n_dl)) / b_norm;
    if (rel_residual < rel_tol) return {0, rel_residual, true};

    for (int i = 0; i < max_iters; i++) {
        matvec(p, Ap);
        double pAp = ddot(h, n, p, Ap, n_dl);
        if (!(pAp > 0.0) || !std::isfinite(pAp)) return {i, std::sqrt(ddot(h, n, r, r, n_dl)) / b_norm, false};

        double alpha = rz / pAp;
        xr_update_kernel<<<nb, T256, 0, stream>>>(x, r, p, Ap, alpha, n);
        cuda_sync_check("cg xr_update");

        rel_residual = std::sqrt(ddot(h, n, r, r, n_dl)) / b_norm;
        if (rel_residual < rel_tol) return {i + 1, rel_residual, true};

        { VMC_PROF("precond", stream); jacobi_kernel<<<nb, T256, 0, stream>>>(M_inv_diag, r, z, n); cuda_sync_check("cg jacobi"); }
        double rz_new = ddot(h, n, r, z, n_dl);
        double beta = rz_new / rz;
        p_update_kernel<<<nb, T256, 0, stream>>>(p, z, beta, n);
        cuda_sync_check("cg p_update");
        rz = rz_new;
    }
    return {max_iters, rel_residual, false};
}

// M_inv_diag[j] = 1.0 / (S_diag[j] * (1.0 + lambda_t) + diag_damp)
__global__ void minv_kernel(const double* __restrict__ S_diag, const double* __restrict__ d_rms, double lambda_t, double* __restrict__ Minv, std::size_t P) {
    std::size_t j = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P) return;
    double diag_damp = sr_eps;
    if constexpr (sr_rms_damp) {
        diag_damp += sr_rms_eps * d_rms[j];
    }
    Minv[j] = 1.0 / (S_diag[j] * (1.0 + lambda_t) + diag_damp);
}

void M_inv_device(const double* S_diag, const double* d_rms, double lambda_t, double* M_inv, std::size_t P, cudaStream_t stream) {
    minv_kernel<<<blocks_for(P), T256, 0, stream>>>(S_diag, d_rms, lambda_t, M_inv, P);
    cuda_sync_check("M_inv_device");
}

// Assigns scratch space
static double* peer_buf(int slot, std::size_t P) {   
    static DeviceArray<double> b[8][3];  // Make an array of DeviceArrays, 3 slots for each device, up to 8 devices
    int d = 0; 
    CUDA_CHECK(cudaGetDevice(&d));  // Access device
    DeviceArray<double>& a = b[d][slot];  // Record slot
    if (a.n < P) a.alloc(P);  // Resize to size P  
    return a.d;
}

// Parallelized addition of two vectors
__global__ void add_kernel(double* __restrict__ y, const double* __restrict__ x, std::size_t n) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

// Add GPU 0 and GPU 1 partial vectors (assigned in plan 0 and 1, not as named in computer)
template <typename Get>
static void sum_to_replica0(const std::vector<ReplicaRef>& R, std::size_t P, Get vec) {
    CUDA_CHECK(cudaSetDevice(R[0].dev));  // Select GPU 0 
    double* y0 = vec(R[0]);  // get GPU 0's partial array 
    double* tmp = peer_buf(0, P);  // Allocate temporary scratch space on GPU 0 
    // Iterate over the other GPUs
    for (std::size_t k = 1; k < R.size(); k++) {
        CUDA_CHECK(cudaMemcpyPeer(tmp, R[0].dev, vec(R[k]), R[k].dev, P * sizeof(double)));  // Move k'th GPU vector into tmp on GPU 0
        add_kernel<<<blocks_for(P), T256>>>(y0, tmp, P); // Add the two in parallel 
        cuda_sync_check("sum_to_replica0"); 
    }
}

// Multi GPU calculation of O_exp_replicas
void O_exp_replicas(const std::vector<ReplicaRef>& R, std::size_t P, long long n_valid) {
    // Launch jobs across GPUs from the greatest to the smallest, write to O_exp_d
    for (std::size_t k = R.size(); k-- > 0;) {   
        CUDA_CHECK(cudaSetDevice(R[k].dev));
        build_mask(R[k].ds->valid_pool.d, R[k].ds->mask_d.d, R[k].Ns);
        O_exp_device(R[k].h, R[k].ds->O_pool.d, R[k].ds->mask_d.d, R[k].Ns, P, n_valid, R[k].ds->O_exp_d.d); 
    }
    // Move to GPU 0 and add the arrays
    sum_to_replica0(R, P, [](const ReplicaRef& r) { 
        return r.ds->O_exp_d.d; 
    });
    // Move calculation of O_exp_d to all other GPUs, they now all have a copy
    for (std::size_t k = 1; k < R.size(); k++) {
        CUDA_CHECK(cudaMemcpyPeer(R[k].ds->O_exp_d.d, R[k].dev, R[0].ds->O_exp_d.d, R[0].dev, P * sizeof(double)));
    }
}

// Multi GPU calculation of gradient
void grad_replicas(const std::vector<ReplicaRef>& R, std::size_t P, long long n_valid, const ClipStats& cs) {
    // Launch jobs across GPUs from the greatest to the smallest, write to grad_d
    for (std::size_t k = R.size(); k-- > 0;) {
        const ReplicaRef& r = R[k];
        CUDA_CHECK(cudaSetDevice(r.dev));
        clip_kernel<<<blocks_for(r.Ns), T256>>>(r.ds->E_pool.d, r.ds->valid_pool.d, cs.clip_lo, cs.clip_hi, r.ds->E_clip_d.d, r.Ns);
        cuda_sync_check("grad_replicas clip");
        gemv_OT(r.h, r.ds->O_pool.d, r.ds->E_clip_d.d, r.ds->grad_d.d, r.Ns, P, 0, "grad_replicas"); 
    }
    // Sum all the samples in GPU 0
    sum_to_replica0(R, P, [](const ReplicaRef& r) { 
        return r.ds->grad_d.d;
     });
    grad_final_kernel<<<blocks_for(P), T256>>>(R[0].ds->grad_d.d, R[0].ds->O_exp_d.d, (double)n_valid, cs.E_clip_mean, P);
    cuda_sync_check("grad_replicas final");
}

// out = S v over every replica's rows, v resides in GPU 0 (0 as in plan)
void sr_apply(const std::vector<ReplicaRef>& R, std::size_t P, long long n_valid, double lambda_diag, const double* v, double* out, bool raw, long long* dl) {
    CUDA_CHECK(cudaSetDevice(R[0].dev));
    const double c = ddot(R[0].h, P, R[0].ds->O_exp_d.d, v, dl);   // O_exp . v (O_exp identical everywhere)
    // Iterate through GPUs 
    for (std::size_t k = R.size(); k-- > 0;) {
        const ReplicaRef& r = R[k];
        CUDA_CHECK(cudaSetDevice(r.dev));
        const double* vk = v;
        double* outk = out;
        // If not GPU 0 write to scratch space slot 1 the original vector, and slot 2 
        if (k > 0) {
            double* v1 = peer_buf(1, P);
            CUDA_CHECK(cudaMemcpyPeer(v1, r.dev, v, R[0].dev, P * sizeof(double)));
            vk = v1;
            outk = peer_buf(2, P);
        }
        // Compute (O - O_exp). v
        gemv_O(r.h, r.ds->O_pool.d, vk, r.ds->t_ns_d.d, r.Ns, P, 0, "sr_apply Ov");
        center_kernel<<<blocks_for(r.Ns), T256>>>(r.ds->t_ns_d.d, r.ds->mask_d.d, c, r.Ns);
        cuda_sync_check("sr_apply center");
        gemv_OT(r.h, r.ds->O_pool.d, r.ds->t_ns_d.d, outk, r.Ns, P, 0, "sr_apply OTt");
    }
    // Add Results from different devices
    CUDA_CHECK(cudaSetDevice(R[0].dev));
    for (std::size_t k = 1; k < R.size(); k++) {
        CUDA_CHECK(cudaSetDevice(R[k].dev));
        double* outk = peer_buf(2, P);
        CUDA_CHECK(cudaSetDevice(R[0].dev));
        double* tmp = peer_buf(0, P);
        CUDA_CHECK(cudaMemcpyPeer(tmp, R[0].dev, outk, R[k].dev, P * sizeof(double)));
        add_kernel<<<blocks_for(P), T256>>>(out, tmp, P);
        cuda_sync_check("sr_apply sum");
    }
    // Apply regularization
    apply_tail_kernel<<<blocks_for(P), T256>>>(out, v, R[0].ds->S_diag_d.d, R[0].ds->d_rms_d.d, (double)n_valid, lambda_diag, sr_eps, raw, P);
    cuda_sync_check("sr_apply tail");
}

// Do an SR step over multiple GPUs
SRStepLog SR_step_device(const std::vector<ReplicaRef>& R, Ansatz& a, int iter, long long n_valid, std::vector<double>& delta_host, long long* n_dl) {
    // Get 0th GPUs state and handle, dl counts device to host downloads
    DeviceState& ds = *R[0].ds;
    const std::size_t P = ds.P;
    cublasHandle_t h = R[0].h;
    long long dl = 0;

    // Regularization parameter lamda (+ lambda S_ii) and learning rate
    double lambda_t = std::max(sr_lambda0 * std::pow(sr_rho, iter), sr_lambda_min);
    double sr_lr = std::max(sr_eta * std::pow(0.999, iter), 0.001);

    // Sum diagonal terms of SR matrix across devices
    for (std::size_t k = R.size(); k-- > 0;) {
        CUDA_CHECK(cudaSetDevice(R[k].dev));
        S_diag_device(R[k].ds->O_pool.d, R[k].ds->valid_pool.d, R[k].ds->O_exp_d.d, R[k].Ns, P, n_valid, R[k].ds->S_diag_d.d);
    }
    sum_to_replica0(R, P, [](const ReplicaRef& r) { 
        return r.ds->S_diag_d.d; 
    });
    // Evaulate 1/S_ii(1+lambda)
    M_inv_device(ds.S_diag_d.d, ds.d_rms_d.d, lambda_t, ds.M_inv_d.d, P);

    // Use conjugate gradient to find the step 
    auto matvec = [&](const double* v, double* out) { 
        sr_apply(R, P, n_valid, lambda_t, v, out, false, &dl); 
    };
    CGResult cg = cg_solve_device(h, matvec, ds.grad_d.d, ds.delta_d.d, ds.M_inv_d.d, P, sr_cg_tol, sr_cg_maxit, ds.cg_r.d, ds.cg_z.d, ds.cg_p.d, ds.cg_Ap.d, &dl, 0);

    // Add limiting cap on step 
    double q = 0.0, delta_norm = 0.0;
    bool norm_capped = false;
    sr_apply(R, P, n_valid, lambda_t, ds.delta_d.d, ds.S_delta_d.d, true, &dl);
    q = ddot(h, P, ds.delta_d.d, ds.S_delta_d.d, &dl);
    if (q > sr_trust_r2) {
        const double scale = std::sqrt(sr_trust_r2 / q);
        check_blas(cublasDscal(h, (int)P, &scale, ds.delta_d.d, 1), "SR_step_device trust scale");
    }
    double delta_norm_raw = std::sqrt(ddot(h, P, ds.delta_d.d, ds.delta_d.d, &dl));
    norm_capped = delta_norm_raw > sr_delta_max;
    if (norm_capped) {
        const double scale = sr_delta_max / delta_norm_raw;
        check_blas(cublasDscal(h, (int)P, &scale, ds.delta_d.d, 1), "SR_step_device norm cap");
        std::cerr << "SR_step: norm cap triggered -- raw ||delta||=" << delta_norm_raw << " > sr_delta_max=" << sr_delta_max << ", rescaled.\n";
    }
    delta_norm = std::sqrt(ddot(h, P, ds.delta_d.d, ds.delta_d.d, &dl));    
    // Apply update on CPU
    delta_host.resize(P);
    ds.delta_d.down(delta_host.data(), P);
    for (std::size_t j = 0; j < P; j++) a.add_to_param(j, -sr_lr * delta_host[j]);
    if (n_dl) *n_dl = dl;
    return {lambda_t, cg.iters, cg.rel_residual, delta_norm, q, norm_capped};
}