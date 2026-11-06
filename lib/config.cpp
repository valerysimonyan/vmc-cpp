#include "config.h"
#include "constants.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <ios>

// Remove blank space from a string
static std::string trim(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\r");
    if (a == std::string::npos) return "";
    const std::size_t b = s.find_last_not_of(" \t\r");
    return s.substr(a, b - a + 1);
}

// Map string to integer
static int parse_int(const std::string& v, const std::string& where) {
    std::size_t used = 0;
    int x = 0;
    try { x = std::stoi(v, &used); } catch (...) { used = 0; }
    if (used == 0 || used != v.size()) throw std::runtime_error(where + ": '" + v + "' is not an integer");
    return x;
}

// Map string to double
static double parse_double(const std::string& v, const std::string& where) {
    std::size_t used = 0;
    double x = 0.0;
    try { x = std::stod(v, &used); } catch (...) { used = 0; }
    if (used == 0 || used != v.size()) throw std::runtime_error(where + ": '" + v + "' is not a number");
    return x;
}

// Map true/false (or 1/0) to a bool
static bool parse_bool(const std::string& v, const std::string& where) {
    if (v == "true" || v == "1") return true;
    if (v == "false" || v == "0") return false;
    throw std::runtime_error(where + ": '" + v + "' is not true or false");
}

// Map a comma seperated string into a vector of integers, each at least `min`
static std::vector<int> parse_int_list(const std::string& v, const std::string& where, int min) {
    std::vector<int> w;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) w.push_back(parse_int(trim(item), where));
    if (w.empty()) throw std::runtime_error(where + ": no values given");
    for (int x : w) if (x < min) throw std::runtime_error(where + ": values must be >= " + std::to_string(min));
    return w;
}

// Assign each value in string to certain variable
RunConfig load_run_config(const char* path) {
    RunConfig c;
    if (!path) return c;
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("run file: cannot open ") + path);

    // Every accepted key: (section, key) -> how to store its value
    using Setter = std::function<void(const std::string& value, const std::string& where)>;
    const std::map<std::string, Setter> keys = {
        {"build.N",   [&](const std::string& v, const std::string& w) { if (parse_int(v, w) != N)   throw std::runtime_error(w + ": run file is for N = " + v + ", this build has N = " + std::to_string(N)); }},
        {"build.N_p", [&](const std::string& v, const std::string& w) { if (parse_int(v, w) != N_p) throw std::runtime_error(w + ": run file is for N_p = " + v + ", this build has N_p = " + std::to_string(N_p)); }},
        {"build.N_u", [&](const std::string& v, const std::string& w) { if (parse_int(v, w) != N_u) throw std::runtime_error(w + ": run file is for N_u = " + v + ", this build has N_u = " + std::to_string(N_u)); }},

        {"network.arch", [&](const std::string& v, const std::string& w) { c.h_hidden = c.rho_hidden = c.orb_hidden = parse_int_list(v, w, 1); }},
        {"network.h",    [&](const std::string& v, const std::string& w) { c.h_hidden = parse_int_list(v, w, 1); }},
        {"network.rho",  [&](const std::string& v, const std::string& w) { c.rho_hidden = parse_int_list(v, w, 1); }},
        {"network.orb",  [&](const std::string& v, const std::string& w) { c.orb_hidden = parse_int_list(v, w, 1); }},

        {"training.gd_iterations",  [&](const std::string& v, const std::string& w) { c.gd_iterations = parse_int(v, w); }},
        {"training.sr_iterations",  [&](const std::string& v, const std::string& w) { c.sr_iterations = parse_int(v, w); }},
        {"training.walkers",        [&](const std::string& v, const std::string& w) { c.walkers = parse_int(v, w); }},
        {"training.records",        [&](const std::string& v, const std::string& w) { c.records = parse_int(v, w); }},
        {"training.grow_factor",    [&](const std::string& v, const std::string& w) { c.grow_factor = parse_int(v, w); }},
        {"training.grow_at",        [&](const std::string& v, const std::string& w) { c.grow_at = parse_double(v, w); }},
        {"training.sweeps_between", [&](const std::string& v, const std::string& w) { c.sweeps_between = parse_int(v, w); }},
        {"training.therm_sweeps",   [&](const std::string& v, const std::string& w) { c.therm_sweeps = parse_int(v, w); }},
        {"training.therm_init",     [&](const std::string& v, const std::string& w) { c.therm_init = parse_int(v, w); }},
        {"training.therm_blocks",   [&](const std::string& v, const std::string& w) { c.therm_blocks = parse_int(v, w); }},
        {"training.step",           [&](const std::string& v, const std::string& w) { c.step = parse_double(v, w); }},
        {"training.ckpt_every",     [&](const std::string& v, const std::string& w) { c.ckpt_every = parse_int(v, w); }},
        {"training.eval_iters",     [&](const std::string& v, const std::string& w) { c.eval_iters = parse_int(v, w); }},
        {"training.diss_threshold", [&](const std::string& v, const std::string& w) { c.diss_threshold = parse_double(v, w); }},
        {"training.diss_window",    [&](const std::string& v, const std::string& w) { c.diss_window = parse_int(v, w); }},
        {"training.diss_growth",    [&](const std::string& v, const std::string& w) { c.diss_growth = parse_double(v, w); }},
        {"training.clip_mad",       [&](const std::string& v, const std::string& w) { c.clip_mad = parse_double(v, w); }},
        {"training.resume",         [&](const std::string& v, const std::string&)   { c.resume = v; }},
        {"training.transfer",       [&](const std::string& v, const std::string&)   { c.transfer = v; }},
        
        {"sr.eta",          [&](const std::string& v, const std::string& w) { c.sr.eta = parse_double(v, w); }},
        {"sr.lr_decay",     [&](const std::string& v, const std::string& w) { c.sr.lr_decay = parse_double(v, w); }},
        {"sr.lambda0",      [&](const std::string& v, const std::string& w) { c.sr.lambda0 = parse_double(v, w); }},
        {"sr.lambda_min",   [&](const std::string& v, const std::string& w) { c.sr.lambda_min = parse_double(v, w); }},
        {"sr.lambda_decay", [&](const std::string& v, const std::string& w) { c.sr.lambda_decay = parse_double(v, w); }},
        {"sr.eps",          [&](const std::string& v, const std::string& w) { c.sr.eps = parse_double(v, w); }},
        {"sr.rms_beta",     [&](const std::string& v, const std::string& w) { c.sr.rms_beta = parse_double(v, w); }},
        {"sr.rms_eps",      [&](const std::string& v, const std::string& w) { c.sr.rms_eps = parse_double(v, w); }},
        {"sr.cg_tol",       [&](const std::string& v, const std::string& w) { c.sr.cg_tol = parse_double(v, w); }},
        {"sr.cg_maxit",     [&](const std::string& v, const std::string& w) { c.sr.cg_maxit = parse_int(v, w); }},
        {"sr.trust_r2",     [&](const std::string& v, const std::string& w) { c.sr.trust_r2 = parse_double(v, w); }},
        {"sr.delta_max",    [&](const std::string& v, const std::string& w) { c.sr.delta_max = parse_double(v, w); }},

        {"adam.lr",           [&](const std::string& v, const std::string& w) { c.adam.lr = parse_double(v, w); }},
        {"adam.beta1",        [&](const std::string& v, const std::string& w) { c.adam.beta1 = parse_double(v, w); }},
        {"adam.beta2",        [&](const std::string& v, const std::string& w) { c.adam.beta2 = parse_double(v, w); }},
        {"adam.eps",          [&](const std::string& v, const std::string& w) { c.adam.eps = parse_double(v, w); }},
        {"adam.env_lr",       [&](const std::string& v, const std::string& w) { c.adam.env_lr = parse_double(v, w); }},
        {"adam.env_decay_it", [&](const std::string& v, const std::string& w) { c.adam.env_decay_it = parse_double(v, w); }},
        {"adam.env_alpha",    [&](const std::string& v, const std::string& w) { c.adam.env_alpha = parse_bool(v, w); }},

        {"gpu.gpus",        [&](const std::string& v, const std::string& w) {
            c.gpu.gpus = v;
            c.gpu.devices.clear();
            if (v == "best" || v == "auto" || v == "auto_one") return;
            if (!std::isdigit((unsigned char)v[0])) throw std::runtime_error(w + ": gpus = '" + v + "' is not best, auto, auto_one or a list of GPU numbers");
            c.gpu.devices = parse_int_list(v, w, 0);
            for (std::size_t i = 0; i < c.gpu.devices.size(); i++)
                for (std::size_t j = 0; j < i; j++)
                    if (c.gpu.devices[i] == c.gpu.devices[j]) throw std::runtime_error(w + ": GPU " + std::to_string(c.gpu.devices[i]) + " listed twice");
        }},
        {"gpu.same_seed",   [&](const std::string& v, const std::string& w) { c.gpu.same_seed = parse_bool(v, w); }},
        {"gpu.plan_only",   [&](const std::string& v, const std::string& w) { c.gpu.plan_only = parse_bool(v, w); }},
        {"gpu.overhead_gb", [&](const std::string& v, const std::string& w) { c.gpu.overhead_gb = parse_double(v, w); }},
        {"gpu.margin_gb",   [&](const std::string& v, const std::string& w) { c.gpu.margin_gb = parse_double(v, w); }},
        {"gpu.max_util",    [&](const std::string& v, const std::string& w) { c.gpu.max_util = parse_int(v, w); }},
        {"gpu.max_walkers", [&](const std::string& v, const std::string& w) { c.gpu.max_walkers = parse_int(v, w); }},
        {"gpu.min_walkers", [&](const std::string& v, const std::string& w) { c.gpu.min_walkers = parse_int(v, w); }},
        {"gpu.jet_chunk",   [&](const std::string& v, const std::string& w) { c.gpu.jet_chunk = parse_int(v, w); }},

    };

    std::string line, section;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        const std::string where = std::string(path) + ":" + std::to_string(lineno);
        if (line.front() == '[') {
            if (line.back() != ']') throw std::runtime_error(where + ": malformed section header '" + line + "'");
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) throw std::runtime_error(where + ": expected 'key = value'");
        const std::string key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
        const auto it = keys.find(section + "." + key);
        if (it == keys.end()) throw std::runtime_error(where + ": unknown setting '" + key + "' in section [" + section + "]");
        it->second(value, where);
    }

    if (c.gd_iterations < 0 || c.sr_iterations < 0) throw std::runtime_error("run file: iteration counts must be >= 0");
    if (c.walkers <= 0 || c.records <= 0 || c.grow_factor <= 0) throw std::runtime_error("run file: walkers, records and grow_factor must be positive");
    if (c.ckpt_every <= 0 || c.eval_iters <= 0 || c.therm_blocks <= 0 || c.diss_window <= 0) throw std::runtime_error("run file: ckpt_every, eval_iters, therm_blocks and diss_window must be positive");
    if (c.gpu.jet_chunk <= 0) throw std::runtime_error("run file: [gpu] jet_chunk must be positive");
    if (c.sr.cg_maxit <= 0 || c.adam.env_decay_it <= 0) throw std::runtime_error("run file: sr.cg_maxit and adam.env_decay_it must be positive");
    if (c.gpu.overhead_gb < 0 || c.gpu.margin_gb < 0 || c.gpu.max_walkers < 0 || c.gpu.min_walkers < 0) throw std::runtime_error("run file: [gpu] memory and walker limits must be >= 0");

    return c;
}

// Write widths as a comma seperated string
static std::string widths(const std::vector<int>& w) {
    std::string s;
    for (std::size_t k = 0; k < w.size(); k++) s += (k ? "," : "") + std::to_string(w[k]);
    return s;
}

// Write run parameters to string 
void write_run_config(std::ostream& o, const RunConfig& c) {
    const std::streamsize old_precision = o.precision(15);   // enough digits that every value reads back as written

    o << "[build]\n"
      << "N = " << N << "\n" << "N_p = " << N_p << "\n" << "N_u = " << N_u << "\n\n"
      << "[network]\n"
      << "h = " << widths(c.h_hidden) << "\n" << "rho = " << widths(c.rho_hidden) << "\n" << "orb = " << widths(c.orb_hidden) << "\n\n"
      << "[training]\n"
      << "gd_iterations = " << c.gd_iterations << "\n"
      << "sr_iterations = " << c.sr_iterations << "\n"
      << "walkers = " << c.walkers << "\n"
      << "records = " << c.records << "\n"
      << "grow_factor = " << c.grow_factor << "\n"
      << "grow_at = " << c.grow_at << "\n"
      << "sweeps_between = " << c.sweeps_between << "\n"
      << "therm_sweeps = " << c.therm_sweeps << "\n"
      << "therm_init = " << c.therm_init << "\n"
      << "therm_blocks = " << c.therm_blocks << "\n"
      << "step = " << c.step << "\n"
      << "ckpt_every = " << c.ckpt_every << "\n"
      << "eval_iters = " << c.eval_iters << "\n"
      << "diss_threshold = " << c.diss_threshold << "\n"
      << "diss_window = " << c.diss_window << "\n"
      << "diss_growth = " << c.diss_growth << "\n"
      << "clip_mad = " << c.clip_mad << "\n"
      << "resume = " << c.resume << "\n"
      << "transfer = " << c.transfer << "\n\n"
      << "[sr]\n"
      << "eta = " << c.sr.eta << "\n"
      << "lr_decay = " << c.sr.lr_decay << "\n"
      << "lambda0 = " << c.sr.lambda0 << "\n"
      << "lambda_min = " << c.sr.lambda_min << "\n"
      << "lambda_decay = " << c.sr.lambda_decay << "\n"
      << "eps = " << c.sr.eps << "\n"
      << "rms_beta = " << c.sr.rms_beta << "\n"
      << "rms_eps = " << c.sr.rms_eps << "\n"
      << "cg_tol = " << c.sr.cg_tol << "\n"
      << "cg_maxit = " << c.sr.cg_maxit << "\n"
      << "trust_r2 = " << c.sr.trust_r2 << "\n"
      << "delta_max = " << c.sr.delta_max << "\n\n"
      << "[adam]\n"
      << "lr = " << c.adam.lr << "\n"
      << "beta1 = " << c.adam.beta1 << "\n"
      << "beta2 = " << c.adam.beta2 << "\n"
      << "eps = " << c.adam.eps << "\n"
      << "env_lr = " << c.adam.env_lr << "\n"
      << "env_decay_it = " << c.adam.env_decay_it << "\n"
      << "env_alpha = " << (c.adam.env_alpha ? "true" : "false") << "\n\n"
      << "[gpu]\n"
      << "gpus = " << c.gpu.gpus << "\n"
      << "same_seed = " << (c.gpu.same_seed ? "true" : "false") << "\n"
      << "plan_only = " << (c.gpu.plan_only ? "true" : "false") << "\n"
      << "overhead_gb = " << c.gpu.overhead_gb << "\n"
      << "margin_gb = " << c.gpu.margin_gb << "\n"
      << "max_util = " << c.gpu.max_util << "\n"
      << "max_walkers = " << c.gpu.max_walkers << "\n"
      << "min_walkers = " << c.gpu.min_walkers << "\n"
      << "jet_chunk = " << c.gpu.jet_chunk << "\n";
    o.precision(old_precision);
}

// Throws errors when present and informs of all the settings we chose
void validate_config(const Ansatz& a, const RunConfig& c) {
    if (nuc_pot != NucPot::Off) {
        bool spin_ok = (spin_mode == SpinMode::Sampled) || (N_u == N);
        if (!spin_ok || tau_mode != TauMode::Sampled) {
            throw std::runtime_error("nuc_pot != Off requires tau_mode = Sampled and (spin_mode = Sampled or N_u == N).");
        }
    }

#ifndef VMC_CUDA
    if (c.walkers % n_thread != 0) throw std::runtime_error("validate_config: walkers must be a multiple of n_thread on the CPU");
#endif

    std::size_t P = a.n_params();
    std::size_t Ns_max = (std::size_t)c.walkers * (std::size_t)c.records_max();
    double PN_ratio = (double)P / (double)Ns_max;
    double o_pool_bytes = (double)Ns_max * (double)P * 8.0;
    double o_pool_gb = o_pool_bytes / (1024.0*1024.0*1024.0);

    std::cout << "=== Config ===\n"
              << "N: " << N << " (N_p=" << N_p << " protons, N_n=" << N_n << " neutrons)\n"
              << "S_z sector: N_u=" << N_u << " up, N_d=" << N_d << " down\n"
              << "K: " << K << ", m_feat: " << m_feat << "\n"
              << "envelope: beta_min=" << beta_min << ", alpha_init=" << a.alpha << "\n"
              << "nuc_pot: " << (nuc_pot == NucPot::ModelO ? "ModelO" : "Off") << "\n"
              << "B (walkers): " << c.walkers << ", records_max: " << c.records_max()
              << ", Ns_max (B*records_per_iter_max): " << Ns_max << "\n"
              << "P (n_params): " << P << ", P/Ns_max: " << PN_ratio << "\n"
              << "O_pool: " << o_pool_gb << " GB (Ns_max*P*8 bytes)\n"
              << "==============\n";

    if (o_pool_gb > o_pool_max_gb) {
        throw std::runtime_error("validate_config: projected O_pool size (" + std::to_string(o_pool_gb) +
                                  " GB) exceeds o_pool_max_gb (" + std::to_string(o_pool_max_gb) + " GB).");
    }
}

