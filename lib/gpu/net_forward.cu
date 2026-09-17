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

// Forward pass
void net_forward(cublasHandle_t handle, const DeviceNet& dn, const real* params, const real* in, int rows, real* a, real* b, real* out, cudaStream_t stream, NetCache* cache) {
    if (rows <= 0 || dn.layers.empty()) return;
    if (cache && ((std::size_t)rows > cache->rows_cap || cache->z.size() != dn.layers.size()))
        throw std::runtime_error("net_forward: activation stash is smaller than this call (grow_phase5 not run, or rows over capacity)");
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
