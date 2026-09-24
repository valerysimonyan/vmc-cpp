#include "sampler_kernels.h"
#include "philox.h"
#include "../envelope.h"

#include <cmath>
#include <vector>

// Coordinate proposal mechanism
__global__ void propose_coord_kernel(const real* __restrict__ x, real* __restrict__ x_prop, int* __restrict__ prop_idx, unsigned long long* __restrict__ rng_ctr, unsigned long long seed, int B, double step) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;

    // Draw a random coordinate and step size
    PhiloxStream st = stream_for(seed, w, rng_ctr[w]);
    const int idx = st.uint_below(D);
    const real disp = (real)st.usym(step);
    rng_ctr[w] = st.ctr;

    // Propose a new coordinate, log the coordinate with propose update
    const real* xw = x + (std::size_t)w * D;
    real* pw = x_prop + (std::size_t)w * D;
    for (int d = 0; d < D; d++) pw[d] = xw[d];
    pw[idx] += disp;

    prop_idx[w] = idx;
}

void propose_coord(DeviceState& ds, int B, double step, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    propose_coord_kernel<<<(B+threads-1)/threads, threads, 0, stream>>>(ds.x.d, ds.x_prop.d, ds.prop_idx.d, ds.rng_ctr.d, rng_seed, B, step);
    cuda_sync_check("propose_coord");
}

// Accept reject step
__device__ __forceinline__ bool dev_metro_accept(PhiloxStream& st, double logp_old, double logp_new) {
    if (logp_old == -INFINITY) return isfinite(logp_new);
    return st.u01() < exp(2.0 * (logp_new - logp_old));
}

// Overwrite old coordinate and logp dev_metro_accept is true
__global__ void accept_coord_kernel(real* __restrict__ x, const real* __restrict__ x_prop, real* __restrict__ logp, const real* __restrict__ logp_prop, long long* __restrict__ acc, unsigned long long* __restrict__ rng_ctr, unsigned long long seed, int B) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;

    PhiloxStream st = stream_for(seed, w, rng_ctr[w]);
    const double lo = (double)logp[w], ln = (double)logp_prop[w];
    const bool ok = dev_metro_accept(st, lo, ln);
    rng_ctr[w] = st.ctr;

    if (ok) {
        real* xw = x + (std::size_t)w * D;
        const real* pw = x_prop + (std::size_t)w * D;
        for (int d = 0; d < D; d++) xw[d] = pw[d];
        logp[w] = logp_prop[w];
        acc[w]++;
    }
}

void accept_coord(DeviceState& ds, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    accept_coord_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(ds.x.d, ds.x_prop.d, ds.logp.d, ds.logp_prop.d, ds.acc.d, ds.rng_ctr.d, rng_seed, B);
    cuda_sync_check("accept_coord");
}

// After coordinate update recenter coordinate
__global__ void recenter_kernel(real* __restrict__ x, int B) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;
    real* xw = x + (std::size_t)w * D;
    for (int d = 0; d < dim; d++) {
        real R = (real)0;
        for (int i = 0; i < N; i++) R += xw[i*dim + d];
        R /= (real)N;
        for (int i = 0; i < N; i++) xw[i*dim + d] -= R;
    }
}

void recenter_device(DeviceState& ds, int B, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    recenter_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(ds.x.d, B);
    cuda_sync_check("recenter_device");
}

// Propose Tz and Sz swap
__global__ void propose_discrete_kernel(const real* __restrict__ cur, real* __restrict__ prop, int* __restrict__ pick_a, int* __restrict__ pick_b, unsigned long long* __restrict__ rng_ctr, unsigned long long seed, int B, int n_a, int n_b) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;

    const real* cw = cur + (std::size_t)w * N;
    real* pw = prop + (std::size_t)w * N;
    for (int i = 0; i < N; i++) pw[i] = cw[i];

    int list_a[N], list_b[N];
    int na = 0, nb = 0;
    for (int i = 0; i < N; i++) {
        if (cw[i] > (real)0) list_a[na++] = i;
        else list_b[nb++] = i;
    }

    PhiloxStream st = stream_for(seed, w, rng_ctr[w]);
    const int ia = list_a[st.uint_below(n_a)];   
    const int ib = list_b[st.uint_below(n_b)];   
    rng_ctr[w] = st.ctr;

    real tmp = pw[ia]; pw[ia] = pw[ib]; pw[ib] = tmp;
    pick_a[w] = ia;
    pick_b[w] = ib;
}

void propose_discrete(DeviceState& ds, int B, bool is_spin, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    propose_discrete_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(is_spin ? ds.s.d : ds.t.d, is_spin ? ds.s_prop.d : ds.t_prop.d, ds.pick_a.d, ds.pick_b.d, ds.rng_ctr.d, rng_seed, B, is_spin ? N_u : N_p, is_spin ? N_d : N_n);
    cuda_sync_check("propose_discrete");
}

// After swap check do accept-reject, find new logp from swap
__device__ __forceinline__ double dev_logp_from_S_ratio(double logp_cur, double S_cur, double S_new) {
    if (!isfinite(logp_cur)) return log(fabs(S_new));
    return logp_cur + log(fabs(S_new)) - log(fabs(S_cur));
}

__global__ void accept_discrete_kernel(real* __restrict__ cur, const real* __restrict__ prop, const real* __restrict__ other, bool is_spin, const real* __restrict__ x, const real* __restrict__ jc, real* __restrict__ S_cur, const real* __restrict__ S_prop, real* __restrict__ logp, long long* __restrict__ counter, unsigned long long* __restrict__ rng_ctr, unsigned long long seed, int B) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= B) return;

    const double lo = (double)logp[w];
    double ln = dev_logp_from_S_ratio(lo, (double)S_cur[w], (double)S_prop[w]);
    if (n_jas_cls > 1 && isfinite(lo)) {  
        const real* cw = cur + (std::size_t)w * N; const real* pw = prop + (std::size_t)w * N; const real* ow = other + (std::size_t)w * N;
        const real* xw = x + (std::size_t)w * D;
        ln += (double)(is_spin ? envelope::jastrow_dlabel<real, real>(xw, cw, ow, pw, ow, jc)
                               : envelope::jastrow_dlabel<real, real>(xw, ow, cw, ow, pw, jc));
    }


    PhiloxStream st = stream_for(seed, w, rng_ctr[w]);
    const bool ok = dev_metro_accept(st, lo, ln);  
    rng_ctr[w] = st.ctr;

    if (ok) {
        real* cw = cur + (std::size_t)w * N;
        const real* pw = prop + (std::size_t)w * N;
        for (int i = 0; i < N; i++) cw[i] = pw[i];
        S_cur[w] = S_prop[w];
        logp[w] = (real)ln;
        counter[w]++;
    }
}

void accept_discrete(DeviceState& ds, int B, bool is_spin, cudaStream_t stream) {
    if (B <= 0) return;
    const int threads = 256;
    accept_discrete_kernel<<<(B + threads - 1)/threads, threads, 0, stream>>>(is_spin ? ds.s.d : ds.t.d, is_spin ? ds.s_prop.d : ds.t_prop.d, is_spin ? ds.t.d : ds.s.d, is_spin, ds.x.d, ds.params.d + (ds.P - envelope::n_params_env) + 1, ds.S_cur.d, ds.S_prop.d, ds.logp.d, is_spin ? ds.sp_acc.d : ds.tau_acc.d, ds.rng_ctr.d, rng_seed, B);
    cuda_sync_check("accept_discrete");
}

// Download acceptance counters
void download_acceptance(DeviceState& ds, int B, long long& acc, long long& sp_acc, long long& tau_acc) {
    std::vector<long long> h(B);
    acc = sp_acc = tau_acc = 0;
    ds.acc.down(h.data(), B);      
    for (int w = 0; w < B; w++) acc += h[w];
    ds.sp_acc.down(h.data(), B);   
    for (int w = 0; w < B; w++) sp_acc += h[w];
    ds.tau_acc.down(h.data(), B);  
    for (int w = 0; w < B; w++) tau_acc += h[w];
}