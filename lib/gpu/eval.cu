#include "eval.h"
#include "prof.h"
#include "det_kernels.h"
#include "net_kernels.h"
#include "net_forward.h"

// Build table of all s,t combos
__global__ void build_feat_combo_kernel(const real* __restrict__ x_sh, real* __restrict__ feat_in, int B) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = (std::size_t)B * N * 4;
    if (idx >= total) return;

    const int c = (int)(idx % 4);
    const int r = (int)(idx / 4);     
    const int w = r / N, p = r % N;
    
    const real s_of[4] = {(real)1, (real)-1, (real)1, (real)-1};
    const real t_of[4] = {(real)1, (real)1, (real)-1, (real)-1};
    
    real* f = feat_in + idx * (dim + 2);
    for (int d = 0; d < dim; d++) f[d] = x_sh[(std::size_t)w*D + p*dim + d];
    f[dim] = s_of[c];
    f[dim + 1] = t_of[c];
}

// Build features 
__global__ void shift_feat_combo_kernel(const real* __restrict__ x, real* __restrict__ x_sh, real* __restrict__ feat_in, int B) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = (std::size_t)B * N * 4;
    if (idx >= total) return;

    const int c = (int)(idx % 4);
    const int r = (int)(idx / 4);
    const int w = r / N, p = r % N;

    const real s_of[4] = {(real)1, (real)-1, (real)1, (real)-1};
    const real t_of[4] = {(real)1, (real)1, (real)-1, (real)-1};

    const real* xw = x + (std::size_t)w * D;
    real* f = feat_in + idx * (dim + 2);
    for (int d = 0; d < dim; d++) {
        real R = (real)0;
        for (int i = 0; i < N; i++) R += xw[i*dim + d];
        R /= (real)N;
        const real v = xw[p*dim + d] - R;
        f[d] = v;
        if (c == 0) x_sh[(std::size_t)w*D + p*dim + d] = v;
    }
    f[dim] = s_of[c];
    f[dim + 1] = t_of[c];
}

// Evaluate xi from combos
__global__ void xi_reduce_combo_kernel(const real* __restrict__ tab_h, const real* __restrict__ s, const real* __restrict__ t, real* __restrict__ xi, int B) {
    std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total = (std::size_t)B * m_feat;

    if (idx >= total) return;
    const int w = (int)(idx / m_feat), f = (int)(idx % m_feat);

    real acc = (real)0;
    for (int p = 0; p < N; p++) {
        const real sv = s[(std::size_t)w*N + p], tv = t[(std::size_t)w*N + p];
        const int  c  = (sv > (real)0 ? 0 : 1) + 2 * (tv > (real)0 ? 0 : 1);
        acc += tab_h[((std::size_t)(w*N + p)*4 + c) * m_feat + f];
    }
    xi[idx] = acc;    
}

// Evaluate Psi through the chain, assign timers for every step
static void eval_chain(DeviceState& ds, cublasHandle_t handle, int B, const real* x_src, real* S_dst, real* logp_dst, cudaStream_t stream, bool stash = false) {
    {
        VMC_PROF("net_fwd", stream);
        shift_build_feat(x_src, ds.s.d, ds.t.d, ds.x_sh.d, ds.feat_in.d, B, stream);

        net_forward(handle, ds.h_net_d, ds.params.d, ds.feat_in.d, B*N, ds.act_a.d, ds.act_b.d, ds.h_out.d, stream, stash ? &ds.cache_h : nullptr);
        xi_reduce(ds.h_out.d, ds.xi.d, B, stream);
        net_forward(handle, ds.rho_net_d, ds.params.d, ds.xi.d, B, ds.act_a.d, ds.act_b.d, ds.rho_out.d, stream, stash ? &ds.cache_rho : nullptr);
        net_forward(handle, ds.orb_net_d, ds.params.d, ds.feat_in.d, B*N, ds.act_a.d, ds.act_b.d, ds.orb_out.d, stream, stash ? &ds.cache_orb : nullptr);
    }
    { VMC_PROF("assemble", stream); assemble_M(ds.orb_out.d, ds.M_batch.d, B, stream); }
    { VMC_PROF("lu",       stream); batched_det(handle, B*K, ds.M_batch.d, ds.lu_ptrs.d, ds.lu_piv.d, ds.lu_info.d, ds.dets.d, stream); }
    { VMC_PROF("combine_envelope", stream); combine_envelope(ds.rho_out.d, ds.dets.d, S_dst, ds.x_sh.d, ds.s.d, ds.t.d, ds.params.d, ds.P, logp_dst, B, stream); }
}

void eval_logp_batch_prop(DeviceState& ds, cublasHandle_t handle, int B, const real* x_prop, real* S_prop, real* logp_prop, cudaStream_t stream, bool stash) {
    eval_chain(ds, handle, B, x_prop, S_prop, logp_prop, stream, stash);
}

void eval_logp_batch(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream) {
    eval_chain(ds, handle, B, ds.x.d, ds.S.d, ds.logp.d, stream);
}


// Build st table
void build_st_table_batch(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream) {
    if (B <= 0) return;
    {
        VMC_PROF("feat_combo", stream);
        const std::size_t total = (std::size_t)B * N * 4;
        const int threads = 256;
        shift_feat_combo_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(ds.x.d, ds.x_sh.d, ds.feat_in.d, B);
        cuda_sync_check("shift_feat_combo");
    }
    {
        VMC_PROF("net_fwd", stream);
        const int rows = B * N * 4;
        net_forward(handle, ds.h_net_d,   ds.params.d, ds.feat_in.d, rows, ds.act_a.d, ds.act_b.d, ds.h_out.d, stream);
        net_forward(handle, ds.orb_net_d, ds.params.d, ds.feat_in.d, rows, ds.act_a.d, ds.act_b.d, ds.orb_out.d, stream);
    }
}


// Batched pull from st table
void S_from_table_batch(DeviceState& ds, cublasHandle_t handle, int B, const real* s, const real* t, real* S_out, cudaStream_t stream) {
    if (B <= 0) return;
    {
        VMC_PROF("xi_combo", stream);
        const std::size_t total = (std::size_t)B * m_feat;
        const int threads = 256;
        xi_reduce_combo_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(ds.h_out.d, s, t, ds.xi.d, B);
        cuda_sync_check("xi_reduce_combo");
    }
    { VMC_PROF("net_fwd",     stream); net_forward(handle, ds.rho_net_d, ds.params.d, ds.xi.d, B, ds.act_a.d, ds.act_b.d, ds.rho_out.d, stream); }
    { VMC_PROF("assemble",    stream); assemble_M_combo(ds.orb_out.d, s, t, ds.M_batch.d, B, stream); }
    { VMC_PROF("lu",          stream); batched_det(handle, B*K, ds.M_batch.d, ds.lu_ptrs.d, ds.lu_piv.d, ds.lu_info.d, ds.dets.d, stream); }
    { VMC_PROF("det_combine", stream); S_combine(ds.rho_out.d, ds.dets.d, S_out, B, stream); }
}

// Build for test comparison of fused vs unfused
void feat_combo_unfused(DeviceState& ds, int B, cudaStream_t stream) {
    if (B <= 0) return;
    shift_to_com(ds.x.d, ds.x_sh.d, B, stream);
    const std::size_t total = (std::size_t)B * N * 4;
    const int threads = 256;
    build_feat_combo_kernel<<<(unsigned)((total + threads - 1)/threads), threads, 0, stream>>>(ds.x_sh.d, ds.feat_in.d, B);
    cuda_sync_check("build_feat_combo");
}