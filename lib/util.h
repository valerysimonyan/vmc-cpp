#pragma once

#include <vector>
#include <functional>

double gen_uniform_sample(double x_min, double x_max);

double dot(const std::vector<double>& a, const std::vector<double>& b);

double norm(const std::vector<double>& a);

double sum(const std::vector<double>& v);

double mean(const std::vector<double>& data);

double jackknife_error(const std::vector<double>& data, const std::function<double(const std::vector<double>&)>& observable);