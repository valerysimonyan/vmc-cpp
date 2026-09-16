#pragma once

#include <cstddef>

// Choose precision 
#ifdef VMC_REAL32
using real = float;
inline constexpr const char* real_name = "float";
#else
using real = double;
inline constexpr const char* real_name = "double";
#endif

// Map elements of Src data type to Dst data type
template <typename Dst, typename Src> 
inline void convert_copy(Dst* dst, const Src* src, std::size_t n) {
    for (std::size_t i = 0; i < n; i++) dst[i] = static_cast<Dst>(src[i]);
}