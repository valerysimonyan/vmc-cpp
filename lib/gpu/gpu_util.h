#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include<atomic>

#include "../constants.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t vmc_err_ = (call);                                          \
        if (vmc_err_ != cudaSuccess) {                                          \
            std::ostringstream vmc_oss_;                                        \
            vmc_oss_ << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                     << "\n  call: " << #call                                   \
                     << "\n  what: " << cudaGetErrorString(vmc_err_);           \
            throw std::runtime_error(vmc_oss_.str());                           \
        }                                                                       \
    } while (0)

// Option for time tracking
inline std::atomic<long long>& launch_counter() { static std::atomic<long long> c{0}; return c; }

// Check if kernel fails to launch or there is an error while CPU is waiting for GPUs to finish
inline void cuda_sync_check(const char* where) {
    if constexpr (prof_enabled) launch_counter().fetch_add(1, std::memory_order_relaxed);
    cudaError_t launch = cudaGetLastError();
        if (launch != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA launch failed in " << where << ": " << cudaGetErrorString(launch);
        throw std::runtime_error(oss.str());
    }
#ifdef VMC_CUDA_SYNCCHECK
    cudaError_t exec = cudaDeviceSynchronize();
    if (exec != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA execution failed in " << where << ": " << cudaGetErrorString(exec);
        throw std::runtime_error(oss.str());
    }
#endif
}


// If no device available throw error. May either manually pick device or will pick one with most available memory. verbose decides if device properties get printed or not
inline int gpu_select_device(bool verbose = true) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("gpu_select_device: no CUDA device visible");

    int chosen = -1;
    if (const char* env = std::getenv("VMC_CUDA_DEVICE")) {
        chosen = std::atoi(env);
        if (chosen < 0 || chosen >= count) {
            std::ostringstream oss;
            oss << "VMC_CUDA_DEVICE=" << env << " but only " << count << " device(s) visible";
            throw std::runtime_error(oss.str());
        }
    } else {
        std::size_t best_free = 0;
        for (int i = 0; i < count; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            std::size_t f = 0, t = 0;
            CUDA_CHECK(cudaMemGetInfo(&f, &t));
            if (f > best_free) { best_free = f; chosen = i; }
        }
    }

    CUDA_CHECK(cudaSetDevice(chosen));
    if (verbose) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, chosen));
        std::size_t f = 0, t = 0;
        CUDA_CHECK(cudaMemGetInfo(&f, &t));
        std::printf("CUDA device %d: %s (sm_%d%d), %.2f of %.2f GiB free\n",
                    chosen, prop.name, prop.major, prop.minor,
                    (double)f / (1024.0*1024*1024), (double)t / (1024.0*1024*1024));
    }
    return chosen;
}

struct XferStats {
    std::atomic<long long> bytes_up{0}, bytes_dn{0}, n_up{0}, n_dn{0};
    void reset() { bytes_up = 0; bytes_dn = 0; n_up = 0; n_dn = 0; }
};
inline XferStats& xfer_stats() { static XferStats s; return s; }
inline void xfer_note_up(std::size_t b) { xfer_stats().bytes_up += (long long)b; xfer_stats().n_up++; }
inline void xfer_note_dn(std::size_t b) { xfer_stats().bytes_dn += (long long)b; xfer_stats().n_dn++; }

// This functions as a CUDA vector wrapper
template <typename T>
struct DeviceArray {
    // Initialize empty state 
    T* d = nullptr;
    std::size_t n = 0;

    DeviceArray() = default;

    // Descrtructor: If pointer has allocated memory free it
    ~DeviceArray() { if (d) cudaFree(d); }

    // Do not allow two DeviceArrays to point to same memroy
    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // Allows for data movement to another array and freeing the originial memory
    DeviceArray(DeviceArray&& o) noexcept : d(o.d), n(o.n) { o.d = nullptr; o.n = 0; }
    DeviceArray& operator=(DeviceArray&& o) noexcept {
        if (this != &o) {
            if (d) cudaFree(d);
            d = o.d; n = o.n;
            o.d = nullptr; o.n = 0;
        }
        return *this;
    }

    // Memory allocation
    void alloc(std::size_t n_) {
        if (d) { cudaFree(d); d = nullptr; n = 0; }
        if (n_ == 0) return;                 // cudaMalloc(0) is unspecified; a
                                             // zero-length array stays null
        CUDA_CHECK(cudaMalloc(&d, n_ * sizeof(T)));
        n = n_;
    }

    // Move from CPU to GPU
    void up(const T* h, std::size_t n_) {
        if (n_ == 0) return;
        if (n_ > n) throw std::runtime_error("DeviceArray::up: source larger than allocation");
        CUDA_CHECK(cudaMemcpy(d, h, n_ * sizeof(T), cudaMemcpyHostToDevice));
        xfer_note_up(n_ * sizeof(T));
    }

    // Move from GPU to CPU
    void down(T* h, std::size_t n_) const {
        if (n_ == 0) return;
        if (n_ > n) throw std::runtime_error("DeviceArray::down: request larger than allocation");
        CUDA_CHECK(cudaMemcpy(h, d, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        xfer_note_dn(n_ * sizeof(T));
    }

    // Clear GPU memory
    void zero() { if (d && n) CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T))); }

    std::size_t bytes() const { return n * sizeof(T); }
};

// Infrastructure for pushing memory from CPU to GPU
struct PinnedArray {
    // Initialize
    void*       h  = nullptr;
    std::size_t nb = 0;

    PinnedArray() = default;
    ~PinnedArray() { if (h) cudaFreeHost(h); }   // no throw, same reason as above

    PinnedArray(const PinnedArray&)            = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;

    PinnedArray(PinnedArray&& o) noexcept : h(o.h), nb(o.nb) { o.h = nullptr; o.nb = 0; }
    PinnedArray& operator=(PinnedArray&& o) noexcept {
        if (this != &o) {
            if (h) cudaFreeHost(h);
            h = o.h; nb = o.nb;
            o.h = nullptr; o.nb = 0;
        }
        return *this;
    }

    void alloc(std::size_t bytes) {
        if (h) { cudaFreeHost(h); h = nullptr; nb = 0; }
        if (bytes == 0) return;
        CUDA_CHECK(cudaHostAlloc(&h, bytes, cudaHostAllocDefault));
        nb = bytes;
    }

    // Grow to at least `bytes`, keeping any existing larger allocation.
    void ensure(std::size_t bytes) { if (bytes > nb) alloc(bytes); }

    template <typename T> T*       as()       { return static_cast<T*>(h); }
    template <typename T> const T* as() const { return static_cast<const T*>(h); }

    std::size_t bytes() const { return nb; }
};

