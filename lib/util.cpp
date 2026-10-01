#include "util.h"
#include "constants.h"

#include <random>
#include <cmath>

double gen_uniform_sample(double x_min, double x_max) {
    thread_local std::mt19937 rng(init_seed);
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return x_min + (x_max - x_min) * dist(rng);
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

double norm(const std::vector<double>& a) {
    return std::sqrt(dot(a, a));
}
