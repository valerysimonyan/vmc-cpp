#include <random>
#include "util.h"

#include <cmath>

double gen_uniform_sample(double x_min, double x_max) {
    thread_local std::mt19937 rng(std::random_device{}());
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

double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

double mean(const std::vector<double>& data) {
    double s = sum(data);
    return s / data.size();
}

// Take in data set, and a function of said observable, jacknife it for error
double jackknife_error(const std::vector<double>& data, const std::function<double(const std::vector<double>&)>& observable) {
    std::size_t N = data.size();
    std::vector<double> loo(N - 1);
    std::vector<double> theta_i(N);

    // Omit bin and compute average
    for (std::size_t i = 0; i < N; i++) {
        std::size_t idx = 0;
        for (std::size_t j = 0; j < N; j++) {
            if (j == i) continue;
            loo[idx++] = data[j];
        }
        theta_i[i] = observable(loo);
    }

    // Compute error of subsets
    double theta_bar = 0.0;
    for (std::size_t i = 0; i < N; i++) theta_bar += theta_i[i];
    theta_bar /= N;

    double var = 0.0;
    for (std::size_t i = 0; i < N; i++) {
        double d = theta_i[i] - theta_bar;
        var += d * d;
    }
    var *= (double)(N - 1) / (double)N;

    return std::sqrt(var);
}