#pragma once

#include "constants.h"

inline thread_local unsigned long long vmc_seed_salt = 0ULL; // Initialize to zero 

// Return unique seed basedon thread
inline unsigned long long vmc_seed() { 
    return rng_seed ^ vmc_seed_salt; 
}