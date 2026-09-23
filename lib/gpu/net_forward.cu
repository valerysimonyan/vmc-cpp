#include "net_forward.h"

#include <stdexcept>

// Store network architecture without parameters
void DeviceNet::build(const Network& net, std::size_t base) {
    layers.clear();
    base_off = base;
    act = net.activation; 
    max_width = 0; 
    const std::size_t n = net.layers.size();
    for (std::size_t l = 0; l < n; l++) {
        DeviceLayer d;
        d.in_w = net.layers[l].input_size;
        d.out_w = net.layers[l].output_size;
        d.w_off = base + (std::size_t)net.layers[l].weight_offset;
        d.b_off = base + (std::size_t)net.layers[l].bias_offset;
        d.is_output = (l+1 == n);
        layers.push_back(d);
        if (d.in_w  > max_width) max_width = d.in_w;
        if (d.out_w > max_width) max_width = d.out_w;
        if (!d.is_output && d.out_w > hidden_width) hidden_width = d.out_w;
    }
}

// Allocate memory for it
void NetCache::alloc(const DeviceNet& dn, std::size_t rows) {
    a_in.clear(); z.clear();
    a_in.resize(dn.layers.size()); z.resize(dn.layers.size());
    for (std::size_t l = 0; l < dn.layers.size(); l++) {
        a_in[l].alloc(rows * (std::size_t)dn.layers[l].in_w);
        z[l].alloc(rows * (std::size_t)dn.layers[l].out_w);
    }
    rows_cap = rows;
}

// Compute bytes
std::size_t NetCache::bytes() const {
    std::size_t b = 0;
    for (const auto& x : a_in) b += x.bytes();
    for (const auto& x : z) b += x.bytes();
    return b;
}

// Float forward pass
static void net_forward_f32(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* in, int rows, real* a, real* b, real* out, cudaStream_t stream, NetCache* cache) {
    if (!dn.params_f || params != dn.params_src)
        throw std::runtime_error("net_forward (fp32_forward): the float parameter mirror is missing or mirrors a different "
                                 "parameter array -- run DeviceState::upload_params for these parameters");
    const int in0 = dn.layers.front().in_w, outL = dn.layers.back().out_w;
    if ((std::size_t)rows * in0 > dn.in_f_cap || (std::size_t)rows * outL > dn.out_f_cap)
        throw std::runtime_error("net_forward (fp32_forward): float scratch smaller than this call");

    cast_to_float(in, dn.in_f, (std::size_t)rows * in0, stream);
    const float* cur = dn.in_f;
    float* fa = reinterpret_cast<float*>(a);        // the FP64 ping-pong holds twice the floats
    float* fb = reinterpret_cast<float*>(b);
    float* nxt = fa;
    for (std::size_t l = 0; l < dn.layers.size(); l++) {
        const DeviceLayer& L = dn.layers[l];
        const bool last = (l + 1 == dn.layers.size());
        if (cache) cast_to_double(cur, cache->a_in[l].d, (std::size_t)rows * L.in_w, stream);
        float* dst = last ? dn.out_f : nxt;
        gemm_rm<float>(handle, rows, L.in_w, L.out_w, cur, dn.params_f + L.w_off, dst, stream);
        if (last) {
            bias_out_f(dst, dn.params_f + L.b_off, out, cache ? cache->z[l].d : nullptr, rows, L.out_w, stream);
        } else {
            bias_act_f(dst, cache ? cache->z[l].d : nullptr, dn.params_f + L.b_off, rows, L.out_w, dn.act, stream);
            cur = dst;
            nxt = (nxt == fa) ? fb : fa;
        }
    }
}

// Forward pass
void net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* in, int rows, real* a, real* b, real* out, cudaStream_t stream, NetCache* cache) {
    if (rows <= 0 || dn.layers.empty()) return;
    if (cache && ((std::size_t)rows > cache->rows_cap || cache->z.size() != dn.layers.size()))
        throw std::runtime_error("net_forward: activation stash is smaller than this call (grow_phase5 not run, or rows over capacity)");
    if constexpr (fp32_forward) { net_forward_f32(handle, dn, params, in, rows, a, b, out, stream, cache); return; }
    const real* cur = in;
    real* nxt = a;

    for (std::size_t l = 0; l < dn.layers.size(); l++) {
        const DeviceLayer& L = dn.layers[l];
        real* dst = (l + 1 == dn.layers.size()) ? out : nxt;

        if (cache) CUDA_CHECK(cudaMemcpy(cache->a_in[l].d, cur, (std::size_t)rows * L.in_w * sizeof(real), cudaMemcpyDeviceToDevice));
        gemm_rowmajor(handle, rows, L.in_w, L.out_w, cur, params + L.w_off, dst, stream);
        if (cache) bias_act_stash(dst, cache->z[l].d, params + L.b_off, rows, L.out_w, dn.act, L.is_output, stream);
        else       bias_act(dst, params + L.b_off, rows, L.out_w, dn.act, L.is_output, stream);

        cur = dst;
        nxt = (nxt == a) ? b : a;
    }
}
