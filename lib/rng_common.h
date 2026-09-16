#pragma once

#include <cstdint> 

// Option to compile either for GPU or CPU
#if defined(__CUDACC__)
    #define VMC_HD __host__ __device__
#else
    #define VMC_HD
#endif

// Take walker seed, return scrambled result
VMC_HD inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;                   // Add odd constant from golden ratio
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;  // Do an xor on a bit and a bit 30 to the right, multiply by random 64 bit constant
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;  // Do an xor on a bit and a bit 27 to the right, multiply by random 64 bit constant
    return x ^ (x >> 31);                         // Do another xor on all bits and bit 31 to the right
}
