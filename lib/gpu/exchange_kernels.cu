#include "exchange_kernels.h"
#include "arena.h"

#include <cmath>
#include <stdexcept>

// Store table of pairs with j > i
std::vector<int> ex_pair_table() {
    std::vector<int> pij(2 * (std::size_t)ex_npairs);
    int p = 0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) { pij[2*p] = i; pij[2*p + 1] = j; p++; }
    return pij;
}

// Assign thread and record if spin, isopsin are same, if same assign 0 else 1
__global__ void ex_plan_kernel(const real* __restrict__ s, const real* __restrict__ t, const int* __restrict__ pair_ij, unsigned char* __restrict__ active, int B) {
    std::size_t slot = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= (std::size_t)B * ex_npairs * ex_types) return;
    const int type = (int)(slot % ex_types);
    const int p = (int)((slot / ex_types) % ex_npairs);
    const std::size_t w = slot / ((std::size_t)ex_types * ex_npairs);
    const int i = pair_ij[2*p], j = pair_ij[2*p + 1];
    active[slot] = ex_slot_active(s[w*N + i], t[w*N + i], s[w*N + j], t[w*N + j], type) ? 1 : 0;
}

void ex_plan(const real* s, const real* t, const int* pair_ij, unsigned char* active, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const std::size_t total = (std::size_t)B * ex_npairs * ex_types;
    const int threads = 256;
    ex_plan_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(s, t, pair_ij, active, B);
    cuda_sync_check("ex_plan");
}

// Guard against singular orbital matrices for Minv
__global__ void ex_rank2_gate_kernel(const real* __restrict__ dets_psi, unsigned char* __restrict__ rank2_ok, int B) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    const real* d = dets_psi + (std::size_t)w * K;
    real max_det = (real)0;
    for (int k = 0; k < K; k++) { const real f = fabs(d[k]); max_det = (max_det < f) ? f : max_det; }
    unsigned char ok = 1;
    for (int k = 0; k < K; k++) if (fabs(d[k]) < (real)1e-12 * max_det) { ok = 0; break; }
    rank2_ok[w] = ok;
}

void ex_rank2_gate(const real* dets_psi, unsigned char* rank2_ok, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 128;
    ex_rank2_gate_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(dets_psi, rank2_ok, B);
    cuda_sync_check("ex_rank2_gate");
}

// Update xi = sum h after swap
__global__ void ex_xi_swap_kernel(const real* __restrict__ xi_psi, const real* __restrict__ tab_h, const real* __restrict__ s, const real* __restrict__ t, const int* __restrict__ pair_ij, real* __restrict__ xi_swap, int Bc, int w_off) {
    const std::size_t ls = blockIdx.x;                     
    if (ls >= (std::size_t)Bc * ex_npairs * ex_types) return;
    const int type = (int)(ls % ex_types);
    const int p = (int)((ls / ex_types) % ex_npairs);
    const std::size_t w = (std::size_t)w_off + ls / ((std::size_t)ex_types * ex_npairs);
    const int i = pair_ij[2*p], j = pair_ij[2*p + 1];

    const real si = s[w*N + i], ti = t[w*N + i], sj = s[w*N + j], tj = t[w*N + j];
    real sni, tni, snj, tnj;
    ex_swapped_labels(si, ti, sj, tj, type, sni, tni, snj, tnj);
    const std::size_t oi = ((w*N + i)*4 + ex_combo(si, ti))  * m_feat;
    const std::size_t ni = ((w*N + i)*4 + ex_combo(sni, tni)) * m_feat;
    const std::size_t oj = ((w*N + j)*4 + ex_combo(sj, tj))  * m_feat;
    const std::size_t nj = ((w*N + j)*4 + ex_combo(snj, tnj)) * m_feat;

    for (int f = threadIdx.x; f < m_feat; f += blockDim.x)
        xi_swap[ls*m_feat + f] = xi_psi[w*m_feat + f] + (tab_h[ni + f] - tab_h[oi + f]) + (tab_h[nj + f] - tab_h[oj + f]);
}

void ex_xi_swap(const real* xi_psi, const real* tab_h, const real* s, const real* t, const int* pair_ij, real* xi_swap, int Bc, int w_off, cudaStream_t stream) {
    if (Bc <= 0) return;
    const std::size_t slots = (std::size_t)Bc * ex_npairs * ex_types;
    ex_xi_swap_kernel<<<(unsigned)slots, 64, 0, stream>>>(xi_psi, tab_h, s, t, pair_ij, xi_swap, Bc, w_off);
    cuda_sync_check("ex_xi_swap");
}

// Update slater determinant after swap
__global__ void ex_S_swap_kernel(const real* __restrict__ rho_swap, const real* __restrict__ dets_psi, const real* __restrict__ Minv_batch, const real* __restrict__ tab_orb, const real* __restrict__ s, const real* __restrict__ t, const int* __restrict__ pair_ij, const unsigned char* __restrict__ active, const unsigned char* __restrict__ rank2_ok, real* __restrict__ S_swap, int Bc, int w_off) {
    const std::size_t ls = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ls >= (std::size_t)Bc * ex_npairs * ex_types) return;
    const std::size_t per_w = (std::size_t)ex_types * ex_npairs;
    const std::size_t w  = (std::size_t)w_off + ls / per_w;
    const std::size_t gs = w * per_w + ls % per_w;                
    
    if (!active[gs] || !rank2_ok[w]) { S_swap[gs] = (real)0; return; }

    const int type = (int)(ls % ex_types);
    const int p = (int)((ls / ex_types) % ex_npairs);
    const int i = pair_ij[2*p], j = pair_ij[2*p + 1];
    const real si = s[w*N + i], ti = t[w*N + i], sj = s[w*N + j], tj = t[w*N + j];
    real sni, tni, snj, tnj;
    ex_swapped_labels(si, ti, sj, tj, type, sni, tni, snj, tnj);
    const real* orb_oi = tab_orb + ((w*N + i)*4 + ex_combo(si, ti))  * (K*N);
    const real* orb_ni = tab_orb + ((w*N + i)*4 + ex_combo(sni, tni)) * (K*N);
    const real* orb_oj = tab_orb + ((w*N + j)*4 + ex_combo(sj, tj))  * (K*N);
    const real* orb_nj = tab_orb + ((w*N + j)*4 + ex_combo(snj, tnj)) * (K*N);

    real S_new = (real)0;
    for (int k = 0; k < K; k++) {
        const real* Mi = Minv_batch + (w*K + k)*(N*N) + (std::size_t)i*N;
        const real* Mj = Minv_batch + (w*K + k)*(N*N) + (std::size_t)j*N;
        real a00 = (real)0, a01 = (real)0, a10 = (real)0, a11 = (real)0;
        for (int m = 0; m < N; m++) {
            const real dci = orb_ni[k*N + m] - orb_oi[k*N + m];
            const real dcj = orb_nj[k*N + m] - orb_oj[k*N + m];
            a00 += Mi[m] * dci;
            a01 += Mi[m] * dcj;
            a10 += Mj[m] * dci;
            a11 += Mj[m] * dcj;
        }
        const real ratio2 = ((real)1 + a00) * ((real)1 + a11) - a01 * a10;
        S_new += rho_swap[ls*K + k] * dets_psi[w*K + k] * ratio2;
    }
    S_swap[gs] = S_new;
}

void ex_S_swap(const real* rho_swap, const real* dets_psi, const real* Minv_batch, const real* tab_orb, const real* s, const real* t, const int* pair_ij, const unsigned char* active, const unsigned char* rank2_ok, real* S_swap, int Bc, int w_off, cudaStream_t stream) {
    if (Bc <= 0) return;
    const std::size_t slots = (std::size_t)Bc * ex_npairs * ex_types;
    const int threads = 128;
    ex_S_swap_kernel<<<(unsigned)((slots + threads - 1)/threads), threads, 0, stream>>>(rho_swap, dets_psi, Minv_batch, tab_orb, s, t, pair_ij, active, rank2_ok, S_swap, Bc, w_off);
    cuda_sync_check("ex_S_swap");
}

static void copy_down_range(void* dst, const void* dev, std::size_t off_bytes, std::size_t n_bytes) {
    CUDA_CHECK(cudaMemcpy(dst, (const char*)dev + off_bytes, n_bytes, cudaMemcpyDeviceToHost));
    xfer_note_dn(n_bytes);
}

static void copy_up_range(void* dev, const void* src, std::size_t off_bytes, std::size_t n_bytes) {
    CUDA_CHECK(cudaMemcpy((char*)dev + off_bytes, src, n_bytes, cudaMemcpyHostToDevice));
    xfer_note_up(n_bytes);
}

// If rank of walker matrix is deficient go to CPU, have slower method that doesn't evaluate inverse and so is more stable as for small eigenvalues inverse computaiton is unstable
int ex_fallback_host(DeviceState& ds, const Ansatz& a, Workspace& ws, int B) {
    if (B <= 0) return 0;
    std::vector<unsigned char> ok((std::size_t)B);
    ds.rank2_ok.down(ok.data(), ok.size());

    int n_fallback = 0;
    const std::size_t per_w = (std::size_t)ex_types * ex_npairs;
    std::vector<real> hs(N), ht(N), rh((std::size_t)N*4*m_feat), ro((std::size_t)N*4*K*N), Sw(per_w);
    ws.tab_h.resize((std::size_t)N*4*m_feat);
    ws.tab_orb.resize((std::size_t)N*4*K*N);
    std::vector<double> s_sw(N), t_sw(N);

    for (int w = 0; w < B; w++) {
        if (ok[w]) continue;
        n_fallback++;

        copy_down_range(hs.data(), ds.s.d, (std::size_t)w*N*sizeof(real), (std::size_t)N*sizeof(real));
        copy_down_range(ht.data(), ds.t.d, (std::size_t)w*N*sizeof(real), (std::size_t)N*sizeof(real));
        copy_down_range(rh.data(), ds.h_out.d,   (std::size_t)w*N*4*m_feat*sizeof(real), rh.size()*sizeof(real));
        copy_down_range(ro.data(), ds.orb_out.d, (std::size_t)w*N*4*K*N*sizeof(real),    ro.size()*sizeof(real));
        for (std::size_t q = 0; q < rh.size(); q++) ws.tab_h[q]   = (double)rh[q];
        for (std::size_t q = 0; q < ro.size(); q++) ws.tab_orb[q] = (double)ro[q];
        ws.table_valid = true;

        for (int p = 0, pi = 0; pi < N; pi++) {
            for (int pj = pi + 1; pj < N; pj++, p++) {
                for (int type = 0; type < ex_types; type++) {
                    const std::size_t ls = (std::size_t)p*ex_types + type;
                    if (!ex_slot_active(hs[pi], ht[pi], hs[pj], ht[pj], type)) { Sw[ls] = (real)0; continue; }
                    real sni, tni, snj, tnj;
                    ex_swapped_labels(hs[pi], ht[pi], hs[pj], ht[pj], type, sni, tni, snj, tnj);
                    for (int q = 0; q < N; q++) { s_sw[q] = (double)hs[q]; t_sw[q] = (double)ht[q]; }
                    s_sw[pi] = (double)sni; t_sw[pi] = (double)tni;
                    s_sw[pj] = (double)snj; t_sw[pj] = (double)tnj;
                    Sw[ls] = (real)S_from_table(s_sw, t_sw, a, ws);
                }
            }
        }
        copy_up_range(ds.S_swap.d, Sw.data(), (std::size_t)w*per_w*sizeof(real), per_w*sizeof(real));
    }
    ws.table_valid = false;
    return n_fallback;
}

__device__ __forceinline__ real dev_coulomb_shape(real r) {
    const real x = (real)b_coul * r;
    if (x < (real)1e-3) return (real)b_coul * ((real)(5.0/16.0) - x*x/(real)96.0);
    const real F = (real)1.0 - ((real)1.0 + (real)11.0*x/(real)16.0 + (real)3.0*x*x/(real)16.0 + x*x*x/(real)48.0) * exp(-x);
    return F / r;
}

// Evaluate proton proton repulsion term
__global__ void coulomb_kernel(const real* __restrict__ x, const real* __restrict__ t, real* __restrict__ V_coul, int B) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    const real* xw = x + (std::size_t)w * D;
    const real* tw = t + (std::size_t)w * N;
    real V = (real)0;
    for (int i = 0; i < N; i++) {
        if (tw[i] < (real)0) continue;
        for (int j = i+1; j < N; j++) {
            if (tw[j] < (real)0) continue;
            real r2 = (real)0;
            for (int d = 0; d < dim; d++) {
                const real diff = xw[i*dim + d] - xw[j*dim + d];
                r2 += diff * diff;
            }
            V += dev_coulomb_shape(sqrt(r2));
        }
    }
    V_coul[w] = (real)alpha_em * (real)hbarc * V;
}

void coulomb_batch(const real* x, const real* t, real* V_coul, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 128;
    coulomb_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(x, t, V_coul, B);
    cuda_sync_check("coulomb_batch");
}

// Combine everything into E_loc
__global__ void ex_assemble_kernel(const real* __restrict__ x, const real* __restrict__ s, const real* __restrict__ t, const int* __restrict__ pair_ij, const real* __restrict__ S_swap, const real* __restrict__ S0, const real* __restrict__ E_kin, const real* __restrict__ v3n, const real* __restrict__ V_coul, const unsigned char* __restrict__ valid_jet, real* __restrict__ V_nuc_out, real* __restrict__ E_loc, unsigned char* __restrict__ valid_loc, real pi15, int B) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    const std::size_t per_w = (std::size_t)ex_types * ex_npairs;
    const real* xw = x + (std::size_t)w * D;
    const real* sw = s + (std::size_t)w * N;
    const real* tw = t + (std::size_t)w * N;
    const real S0w = S0[w];

    real E = E_kin[w];
    if (nuc_3N)      E += v3n[w];
    if (nuc_coulomb) E += V_coul[w];

    real V_nuc = (real)0;
    if (nuc_pot != NucPot::Off) {
        for (int p = 0; p < ex_npairs; p++) {
            const int i = pair_ij[2*p], j = pair_ij[2*p + 1];
            const bool same_s = (sw[i] == sw[j]);
            const bool same_t = (tw[i] == tw[j]);
            if (same_s && same_t) continue;

            real r2 = (real)0;
            for (int d = 0; d < dim; d++) {
                const real diff = xw[i*dim + d] - xw[j*dim + d];
                r2 += diff * diff;
            }
            const real v01 = exp(-r2 / (real)(R01*R01)) / (pi15 * (real)R01*(real)R01*(real)R01);
            const real v10 = exp(-r2 / (real)(R10*R10)) / (pi15 * (real)R10*(real)R10*(real)R10);

            const std::size_t base = (std::size_t)w * per_w + (std::size_t)p * ex_types;
            real R_s, R_t, R_st;
            if (same_s) {
                R_s = (real)1.0;
                R_t = S_swap[base + EX_T] / S0w;
                R_st = R_s * R_t;
            } else if (same_t) {
                R_t = (real)1.0;
                R_s = S_swap[base + EX_S] / S0w;
                R_st = R_s * R_t;
            } else {
                R_t  = S_swap[base + EX_T]  / S0w;
                R_s  = S_swap[base + EX_S]  / S0w;
                R_st = S_swap[base + EX_ST] / S0w;
            }
            V_nuc += (real)(hbarc/4.0) * ((real)C01*v01*((real)1.0 + R_t - R_s - R_st) + (real)C10*v10*((real)1.0 - R_t + R_s - R_st));
        }
        E += V_nuc;
    }

    V_nuc_out[w] = V_nuc;
    E_loc[w] = E;
    valid_loc[w] = (valid_jet[w] && isfinite(E)) ? 1 : 0;
}

void ex_assemble(const real* x, const real* s, const real* t, const int* pair_ij, const real* S_swap, const real* S0, const real* E_kin, const real* v3n, const real* V_coul, const unsigned char* valid_jet, real* V_nuc, real* E_loc, unsigned char* valid_loc, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const real pi15 = (real)std::pow(3.14159265358979323846, 1.5);
    const int threads = 128;
    ex_assemble_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(x, s, t, pair_ij, S_swap, S0, E_kin, v3n, V_coul, valid_jet, V_nuc, E_loc, valid_loc, pi15, B);
    cuda_sync_check("ex_assemble");
}

// Record walker energies into pool
__global__ void pool_write_kernel(const real* __restrict__ E_loc, const unsigned char* __restrict__ valid_loc, double* __restrict__ E_pool, unsigned char* __restrict__ valid_pool, int r, int B) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    const std::size_t idx = (std::size_t)r * B + w;
    valid_pool[idx] = valid_loc[w];
    if (valid_loc[w]) E_pool[idx] = (double)E_loc[w];
}

void pool_write_row(const real* E_loc, const unsigned char* valid_loc, double* E_pool, unsigned char* valid_pool, int r, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 128;
    pool_write_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(E_loc, valid_loc, E_pool, valid_pool, r, B);
    cuda_sync_check("pool_write_row");
}
