#include "../rng_common.h"

#include <cstdint>

struct Philox4 {
    std::uint32_t v[4]; 
};

// Philox functions serve tp completely scramble numbers of pRNG, VMC_HD has macro definiing it as __host__ __device__ or nothing for CPU depending on how we compile
VMC_HD inline std::uint32_t philox_mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t* hi) {
    std::uint64_t product = (std::uint64_t)a * (std::uint64_t)b;
    *hi = (std::uint32_t)(product >> 32);
    return (std::uint32_t)product;
}
VMC_HD inline Philox4 philox4x32_round(Philox4 c, std::uint32_t k0, std::uint32_t k1) {
    std::uint32_t hi0, hi1;
    std::uint32_t lo0 = philox_mulhilo32(0xD2511F53u, c.v[0], &hi0);
    std::uint32_t lo1 = philox_mulhilo32(0xCD9E8D57u, c.v[2], &hi1);
    Philox4 out;
    out.v[0] = hi1 ^ c.v[1] ^ k0;
    out.v[1] = lo1;
    out.v[2] = hi0 ^ c.v[3] ^ k1;
    out.v[3] = lo0;
    return out;
}
VMC_HD inline Philox4 philox4x32_10_ctr(Philox4 c, std::uint32_t k0, std::uint32_t k1) {
    c = philox4x32_round(c, k0, k1);
    for (int r = 1; r < 10; r++) {
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
        c = philox4x32_round(c, k0, k1);
    }
    return c;
}
VMC_HD inline Philox4 philox4x32_10(std::uint64_t counter, std::uint64_t key) {
    Philox4 c;
    c.v[0] = (std::uint32_t)(counter & 0xFFFFFFFFu);
    c.v[1] = (std::uint32_t)(counter >> 32);
    c.v[2] = 0u;
    c.v[3] = 0u;
    return philox4x32_10_ctr(c, (std::uint32_t)(key & 0xFFFFFFFFu),
                                (std::uint32_t)(key >> 32));
}

// Generates 64-bit random number between [0,1)
VMC_HD inline double u01_from(std::uint32_t hi, std::uint32_t lo) {
    return ((double)(hi >> 5) * 67108864.0 + (double)(lo >> 6)) * (1.0 / 9007199254740992.0);
}

// Using philox random number generator define uniform random number generators for a keu and counter
struct PhiloxStream {
    std::uint64_t key;
    std::uint64_t ctr;

    VMC_HD double u01() {
        Philox4 r = philox4x32_10(ctr++, key);
        return u01_from(r.v[0], r.v[1]);
    }

    VMC_HD double usym(double a) { return (2.0 * u01() - 1.0) * a; }

    VMC_HD int uint_below(int n) {
        Philox4 r = philox4x32_10(ctr++, key);
        return (int)(((std::uint64_t)r.v[0] * (std::uint64_t)(std::uint32_t)n) >> 32);
    }
};

// Generate RNG for a stream and seed
VMC_HD inline PhiloxStream stream_for(unsigned long long seed, int w, unsigned long long ctr) {
    PhiloxStream s;
    s.key = seed ^ splitmix64((unsigned long long)w);
    s.ctr = ctr;
    return s;
}
