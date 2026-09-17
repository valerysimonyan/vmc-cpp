#include "sr_device.h"
#include "arena.h"
#include "../constants.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check_blas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int)st));
}
static double ddot(cublasHandle_t h, std::size_t n, const double* a, const double* b, long long* n_dl) {
    double r = 0.0;
    check_blas(cublasDdot(h, (int)n, a, 1, b, 1, &r), "cublasDdot");   // host pointer mode: blocks, one scalar down
    if (n_dl) (*n_dl)++;
    return r;
}
static const int T256 = 256;
static unsigned blocks_for(std::size_t n) { return (unsigned)((n + T256 - 1) / T256); }

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

void O_exp_device(cublasHandle_t h, const double* O_pool, const double* m, std::size_t Ns, std::size_t P, long long n_valid, double* O_exp, cudaStream_t stream) {
    cublasSetStream(h, stream);
    const double one = 1.0, zero = 0.0;
    check_blas(cublasDgemv(h, CUBLAS_OP_N, (int)P, (int)Ns, &one, O_pool, (int)P, m, 1, &zero, O_exp, 1), "O_exp_device");
    if (n_valid == 0) { CUDA_CHECK(cudaMemset(O_exp, 0, P * sizeof(double))); return; }
    div_kernel<<<blocks_for(P), T256, 0, stream>>>(O_exp, (double)n_valid, P);
    cuda_sync_check("O_exp_device");
}

// Evaluate diagonal part of SR matrix
__global__ void S_diag_kernel(const double* __restrict__ O_pool, const unsigned char* __restrict__ valid, const double* __restrict__ O_exp, std::size_t Ns, std::size_t P, double n_valid, double* __restrict__ S_diag) {
    std::size_t j = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P) return;
    const double oe = O_exp[j];
    double acc = 0.0;
    for (std::size_t i = 0; i < Ns; i++) {
        if (!valid[i]) continue;
        const double diff = O_pool[i*P + j] - oe;
        acc += diff * diff;
    }
    S_diag[j] = acc / n_valid;
}

void S_diag_device(const double* O_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, double* S_diag, cudaStream_t stream) {
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

// Clip energies
ClipStats clip_stats_host(const std::vector<double>& E_pool, const std::vector<unsigned char>& valid_pool, std::size_t n_samples, long long n_valid) {
    std::vector<double> E_valid;
    E_valid.reserve(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) {
        if (valid_pool[i]) E_valid.push_back(E_pool[i]);
    }
    std::nth_element(E_valid.begin(), E_valid.begin() + n_valid/2, E_valid.end());
    double E_med = E_valid[n_valid/2];
    double MAD = 0.0;
    for (double e : E_valid) MAD += std::fabs(e - E_med);
    MAD /= (double)n_valid;
    double clip_lo = E_med - clip_mad * MAD;
    double clip_hi = E_med + clip_mad * MAD;

    const std::size_t chunk = n_samples / (std::size_t)n_thread;
    double E_clip_sum = 0.0;
    for (int th = 0; th < n_thread; th++) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_thread - 1) ? n_samples : start + chunk;
        double part = 0.0;
        for (std::size_t i = start; i < end; i++) {
            if (!valid_pool[i]) continue;
            part += std::min(std::max(E_pool[i], clip_lo), clip_hi);
        }
        E_clip_sum += part;
    }
    return {clip_lo, clip_hi, E_clip_sum / (double)n_valid};
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

void grad_device(cublasHandle_t h, const double* O_pool, const double* E_pool, const unsigned char* valid, const double* O_exp, std::size_t Ns, std::size_t P, long long n_valid, const ClipStats& cs, double* E_clip, double* grad, cudaStream_t stream) {
    clip_kernel<<<blocks_for(Ns), T256, 0, stream>>>(E_pool, valid, cs.clip_lo, cs.clip_hi, E_clip, Ns);
    cuda_sync_check("clip_kernel");
    cublasSetStream(h, stream);
    const double one = 1.0, zero = 0.0;
    check_blas(cublasDgemv(h, CUBLAS_OP_N, (int)P, (int)Ns, &one, O_pool, (int)P, E_clip, 1, &zero, grad, 1), "grad_device");
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

// Full SR apply
void SROpDevice::apply(const double* v, double* out, bool raw) {
    cublasSetStream(h, 0);
    const double one = 1.0, zero = 0.0;
    double Oexp_v = 0.0;
    check_blas(cublasDdot(h, (int)P, O_exp, 1, v, 1, &Oexp_v), "SROpDevice::apply dot");
    check_blas(cublasDgemv(h, CUBLAS_OP_T, (int)P, (int)Ns, &one, O_pool, (int)P, v, 1, &zero, t, 1), "SROpDevice::apply Ov");
    center_kernel<<<blocks_for(Ns), T256, 0, 0>>>(t, m, Oexp_v, Ns);
    cuda_sync_check("center_kernel");
    check_blas(cublasDgemv(h, CUBLAS_OP_N, (int)P, (int)Ns, &one, O_pool, (int)P, t, 1, &zero, out, 1), "SROpDevice::apply OTt");
    apply_tail_kernel<<<blocks_for(P), T256, 0, 0>>>(out, v, S_diag, d_rms, (double)n_valid, lambda_diag, eps_abs, raw, P);
    cuda_sync_check("apply_tail_kernel");
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
    jacobi_kernel<<<nb, T256, 0, stream>>>(M_inv_diag, r, z, n);
    cuda_sync_check("cg init");
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

        jacobi_kernel<<<nb, T256, 0, stream>>>(M_inv_diag, r, z, n);
        cuda_sync_check("cg jacobi");
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

// Take an SR step
SRStepLog SR_step_device(DeviceState& ds, cublasHandle_t h, Ansatz& a, int iter, std::size_t n_samples, long long n_valid, std::vector<double>& delta_host, long long* n_dl, cudaStream_t stream) {
    const std::size_t P = ds.P;
    if (ds.O_exp_d.n < P) throw std::runtime_error("SR_step_device: grow_phase52 has not run");
    long long dl = 0;

    double lambda_t = std::max(sr_lambda0 * std::pow(sr_rho, iter), sr_lambda_min);
    double sr_lr = std::max(sr_eta * std::pow(0.999, iter), 0.001);

    // sr_op.init: S_diag (O_exp, mask and d_rms are already current)
    S_diag_device(ds.O_pool.d, ds.valid_pool.d, ds.O_exp_d.d, n_samples, P, n_valid, ds.S_diag_d.d, stream);
    M_inv_device(ds.S_diag_d.d, ds.d_rms_d.d, lambda_t, ds.M_inv_d.d, P, stream);

    SROpDevice op;
    op.h = h; op.O_pool = ds.O_pool.d; op.O_exp = ds.O_exp_d.d; op.m = ds.mask_d.d; op.S_diag = ds.S_diag_d.d;
    op.d_rms = ds.d_rms_d.d; op.t = ds.t_ns_d.d; op.Ns = n_samples; op.P = P; op.n_valid = n_valid;
    op.lambda_diag = lambda_t; op.eps_abs = sr_eps;
    auto matvec = [&op, &dl](const double* v, double* out) { op.apply(v, out, false); dl++; };   // 1 scalar (Oexp.v) per apply

    CGResult cg = cg_solve_device(h, matvec, ds.grad_d.d, ds.delta_d.d, ds.M_inv_d.d, P, sr_cg_tol, sr_cg_maxit,
                                  ds.cg_r.d, ds.cg_z.d, ds.cg_p.d, ds.cg_Ap.d, &dl, stream);

    op.apply(ds.delta_d.d, ds.S_delta_d.d, true); dl++;
    double q = ddot(h, P, ds.delta_d.d, ds.S_delta_d.d, &dl);
    if (q > sr_trust_r2) {
        const double scale = std::sqrt(sr_trust_r2 / q);
        check_blas(cublasDscal(h, (int)P, &scale, ds.delta_d.d, 1), "SR_step_device trust scale");
    }
    double delta_norm_raw = std::sqrt(ddot(h, P, ds.delta_d.d, ds.delta_d.d, &dl));
    bool norm_capped = delta_norm_raw > sr_delta_max;
    if (norm_capped) {
        const double scale = sr_delta_max / delta_norm_raw;
        check_blas(cublasDscal(h, (int)P, &scale, ds.delta_d.d, 1), "SR_step_device norm cap");
        std::cerr << "SR_step: norm cap triggered -- raw ||delta||=" << delta_norm_raw
                  << " > sr_delta_max=" << sr_delta_max << ", rescaled.\n";
    }

    const double delta_norm = std::sqrt(ddot(h, P, ds.delta_d.d, ds.delta_d.d, &dl));

    // The one per-iteration vector download. The warm start stays on device.
    delta_host.resize(P);
    ds.delta_d.down(delta_host.data(), P);
    for (std::size_t j = 0; j < P; j++) a.add_to_param(j, -sr_lr * delta_host[j]);

    if (n_dl) *n_dl = dl;
    return {lambda_t, cg.iters, cg.rel_residual, delta_norm, q, norm_capped};
}
