#include "compose_kernels.h"
#include "djet.h"

#include <cmath>

__device__ __forceinline__ void seed_coord_jet(DJet& o, const real* x_sh_w, int i) {
    const int p = i / dim, d = i % dim;
    const real inv_N = (real)1 / (real)N;
    o.v = x_sh_w[i];
    for (int aa = 0; aa < D; aa++) {
        const int j = aa / dim, dp = aa % dim;
        o.g[aa] = (dp == d) ? ((p == j ? (real)1 : (real)0) - inv_N) : (real)0;
    }
    o.l = (real)0;
}

// Compose Jet Psi
__global__ void psi_jet_compose_kernel(const real* __restrict__ J_rho, const real* __restrict__ J_det, const real* __restrict__ x_sh, real alpha, real* __restrict__ J_psi, real* __restrict__ S_jet_v, int Bc, int w_off, int B_tot) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Bc) return;

    const std::size_t rho_stride = jet_block_stride((std::size_t)Bc, (std::size_t)K);
    const std::size_t k_stride = jet_block_stride((std::size_t)B_tot, (std::size_t)K);
    const std::size_t p_stride = (std::size_t)B_tot;
    const std::size_t gw = (std::size_t)(w_off + w);

    DJet S, rk, dk, prod;
    djet_zero(S);
    for (int k = 0; k < K; k++) {
        const std::size_t roff = (std::size_t)w * K + k;
        const std::size_t doff = gw * K + k;
        rk.v = J_rho[roff];
        dk.v = J_det[doff];
        for (int a = 0; a < D; a++) {
            rk.g[a] = J_rho[(std::size_t)(1 + a)*rho_stride + roff];
            dk.g[a] = J_det[(std::size_t)(1 + a)*k_stride + doff];
        }
        rk.l = J_rho[(std::size_t)(jet_C - 1)*rho_stride + roff];
        dk.l = J_det[(std::size_t)(jet_C - 1)*k_stride + doff];
        djet_mul(prod, rk, dk);
        djet_add(S, S, prod);
    }
    if (S_jet_v) S_jet_v[gw] = S.v;

    const real* xw = x_sh + gw * D;
    DJet r2, c_i, sq;
    djet_zero(r2);
    for (int i = 0; i < D; i++) {
        seed_coord_jet(c_i, xw, i);
        djet_mul(sq, c_i, c_i);
        djet_add(r2, r2, sq);
    }
    r2.v += (real)(eps_env * eps_env);     

    DJet r_env; djet_sqrt(r_env, r2);

    DJet beta; djet_const(beta, (real)beta_min + exp(alpha));
    DJet negb; djet_scale(negb, beta, (real)-1);
    DJet arg;  djet_mul(arg, negb, r_env);
    DJet env;  djet_exp(env, arg);

    DJet pj; djet_mul(pj, env, S);

    const std::size_t off = gw;
    J_psi[off] = pj.v;
    for (int a = 0; a < D; a++) J_psi[(std::size_t)(1 + a)*p_stride + off] = pj.g[a];
    J_psi[(std::size_t)(jet_C - 1)*p_stride + off] = pj.l;
}

void psi_jet_compose(const real* J_rho, const real* J_det, const real* x_sh, const real* params, std::size_t P, real* J_psi, real* S_jet_v, int Bc, int w_off, int B_tot, cudaStream_t stream) {
    if (Bc <= 0) return;
    real alpha_h;
    CUDA_CHECK(cudaMemcpyAsync(&alpha_h, params + (P - 1), sizeof(real), cudaMemcpyDeviceToHost, stream));
    xfer_note_dn(sizeof(real));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int threads = 128;
    psi_jet_compose_kernel<<<(Bc + threads - 1)/threads, threads, 0, stream>>>(J_rho, J_det, x_sh, alpha_h, J_psi, S_jet_v, Bc, w_off, B_tot);
    cuda_sync_check("psi_jet_compose");
}

// Compose Angular Momentum square and Laplacian
__global__ void kinetic_l2_kernel(const real* __restrict__ J_psi, const real* __restrict__ x_sh, real* __restrict__ E_kin, real* __restrict__ l2, int Bc, int w_off, int B_tot) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Bc) return;

    const std::size_t p_stride = (std::size_t)B_tot;
    const std::size_t gw = (std::size_t)(w_off + w);
    const real v = J_psi[gw];
    const real l = J_psi[(std::size_t)(jet_C - 1)*p_stride + gw];

    E_kin[gw] = -(real)hbar2_2m * (l / v);

    // l2_local (physics.cpp): L = sum_i r_i x grad_i, then |L|^2 / psi^2.
    const real* xw = x_sh + gw * D;
    real Lx = (real)0, Ly = (real)0, Lz = (real)0;
    for (int i = 0; i < N; i++) {
        const real rx = xw[i*dim + 0], ry = xw[i*dim + 1], rz = xw[i*dim + 2];
        const real gx = J_psi[(std::size_t)(1 + i*dim + 0)*p_stride + gw];
        const real gy = J_psi[(std::size_t)(1 + i*dim + 1)*p_stride + gw];
        const real gz = J_psi[(std::size_t)(1 + i*dim + 2)*p_stride + gw];
        Lx += ry*gz - rz*gy;
        Ly += rz*gx - rx*gz;
        Lz += rx*gy - ry*gx;
    }
    l2[gw] = (Lx*Lx + Ly*Ly + Lz*Lz) / (v*v);
}

void kinetic_l2(const real* J_psi, const real* x_sh, real* E_kin, real* l2, int Bc, int w_off, int B_tot, cudaStream_t stream) {
    if (Bc <= 0) return;
    static_assert(dim == 3, "kinetic_l2's cross product assumes dim == 3, as l2_local does");
    const int threads = 128;
    kinetic_l2_kernel<<<(Bc + threads - 1)/threads, threads, 0, stream>>>(J_psi, x_sh, E_kin, l2, Bc, w_off, B_tot);
    cuda_sync_check("kinetic_l2");
}

// Compose 3 Particle interaction
__global__ void v3n_kernel(const real* __restrict__ x, real* __restrict__ v3n, int Bc, int w_off) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Bc) return;

    const std::size_t gw = (std::size_t)(w_off + w);
    if (N < 3) { v3n[gw] = (real)0; return; }

    const real* xw = x + gw * D;
    real V = (real)0;
    for (int i = 0; i < N; i++) {   
        for (int j = i+1; j < N; j++) {
            for (int k = j+1; k < N; k++) {
                real rij2 = (real)0, rjk2 = (real)0, rki2 = (real)0;
                for (int d = 0; d < dim; d++) {
                    const real dij = xw[i*dim + d] - xw[j*dim + d];
                    const real djk = xw[j*dim + d] - xw[k*dim + d];
                    const real dki = xw[k*dim + d] - xw[i*dim + d];
                    rij2 += dij * dij;
                    rjk2 += djk * djk;
                    rki2 += dki * dki;
                }
                V += exp(-(rki2+rij2)/(real)(R3*R3));
                V += exp(-(rij2+rjk2)/(real)(R3*R3));
                V += exp(-(rjk2+rki2)/(real)(R3*R3));
            }
        }
    }
    v3n[gw] = (real)V3_0 * V;
}

void v3n_batch(const real* x, real* v3n, int Bc, int w_off, cudaStream_t stream) {
    if (Bc <= 0) return;
    const int threads = 128;
    v3n_kernel<<<(Bc + threads - 1)/threads, threads, 0, stream>>>(x, v3n, Bc, w_off);
    cuda_sync_check("v3n_batch");
}

// Assemble wave function and check if it works
__global__ void validity_jet_kernel(const real* __restrict__ J_psi, const real* __restrict__ S, const real* __restrict__ x_sh, real alpha, const real* __restrict__ E_kin, real* __restrict__ psi_dbl, unsigned char* __restrict__ valid, int Bc, int w_off, int B_tot) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Bc) return;

    const std::size_t gw = (std::size_t)(w_off + w);
    const real Sv = S[gw];

    // psi_double = psi_impl<double>'s last line: exp(-(beta_min+exp(alpha))*r_env) * S
    const real* xw = x_sh + gw * D;
    real r2 = (real)0;
    for (int i = 0; i < D; i++) r2 += xw[i] * xw[i];
    const real r_env = sqrt(r2 + (real)(eps_env * eps_env));
    const real pd = exp(-((real)beta_min + exp(alpha)) * r_env) * Sv;
    psi_dbl[gw] = pd;

    if (!isfinite(Sv) || fabs(Sv) < (real)1e-290) { valid[gw] = 0; return; }

    const real v = J_psi[gw];
    const bool mismatch = fabs(v - pd) > (real)psi_mismatch_tol * fmax((real)1, fabs(pd)); // Set precision tolerance depending on precision chosen
    if (!isfinite(v) || fabs(v) < (real)1e-290 || mismatch) { valid[gw] = 0; return; }

    if (!isfinite(E_kin[gw])) { valid[gw] = 0; return; }

    valid[gw] = 1;
}

void validity_jet(const real* J_psi, const real* S, const real* x_sh, const real* params, std::size_t P, const real* E_kin, real* psi_dbl, unsigned char* valid, int Bc, int w_off, int B_tot, cudaStream_t stream) {
    if (Bc <= 0) return;
    real alpha_h;
    CUDA_CHECK(cudaMemcpyAsync(&alpha_h, params + (P - 1), sizeof(real), cudaMemcpyDeviceToHost, stream));
    xfer_note_dn(sizeof(real));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int threads = 128;
    validity_jet_kernel<<<(Bc + threads - 1)/threads, threads, 0, stream>>>(J_psi, S, x_sh, alpha_h, E_kin, psi_dbl, valid, Bc, w_off, B_tot);
    cuda_sync_check("validity_jet");
}
