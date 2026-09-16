#include "walkers.h"

#include <algorithm>
#include <cmath>
#include <limits>



void WalkerBatch::init(int B_) {
    B = B_;
    x.assign((std::size_t)B * D, 0.0);
    s.assign((std::size_t)B * N, 0.0);
    t.assign((std::size_t)B * N, 0.0);
    
    logp.assign(B, 0.0);
    valid.assign(B, 0);
    acc.assign(B, 0);
    sp_acc.assign(B, 0);
    tau_acc.assign(B, 0);
    
    // Populate rng with random seeds depending on index
    rng.clear();
    rng.reserve(B);
    for (int w = 0; w < B; w++) rng.emplace_back(rng_seed ^ splitmix64((unsigned long long)w));
}

// Draw from uniform dist 0, 1
double u01(WalkerBatch& wb, int w) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(wb.rng[w]);
}

// Draw from uniform distribution between -a and a
double usym(WalkerBatch& wb, int w, double a) {
    std::uniform_real_distribution<double> dist(-a, a);
    return dist(wb.rng[w]);
}

// Draw from uniform integer range between 0 and n-1
int uint_below(WalkerBatch& wb, int w, int n) {
    std::uniform_int_distribution<int> dist(0, n-1);
    return dist(wb.rng[w]);
}

// Divide [0, B) into n_workers, the last worker takes the remainder
static void chunk_range(int B, int n_workers, int th, int& w0, int& w1) {
    int chunk = B / n_workers; 
    w0 = th * chunk;
    w1 = (th == n_workers - 1) ? B : w0 + chunk;
}

// Go to center of mass coordinates
static void recenter_walker(double* xw) {
    for (int d = 0; d < dim; d++) {
        double R_cm_d = 0.0;
        for (int i = 0; i < N; i++) R_cm_d += xw[i*dim + d];
        R_cm_d /= N;
        for (int i = 0; i < N; i++) xw[i*dim + d] -= R_cm_d;
    }
}

// Initialize batch of all walkers, populate positions, spins, and isospins
void init_batch(WalkerBatch& wb, const Ansatz& a, ThreadPool* pool, std::vector<Workspace>& wss) {
    int B = wb.B; 
    int n_workers = (int)wss.size();
    pool -> run([&](int th) {
        int w0, w1;
        chunk_range(B, n_workers, th, w0, w1);
        Workspace& ws = wss[th];
        for (int w = w0; w < w1; w++) {
            double* xw = &wb.x[(std::size_t)w * D];
            double* sw = &wb.s[(std::size_t)w * N];
            double* tw = &wb.t[(std::size_t)w * N];
            for (int p = 0; p < N; p++) {
                sw[p] = (p < N_u) ? 1.0 : - 1.0;
                tw[p] = (p < N_p) ? 1.0 : - 1.0;
                for (int d = 0; d < dim; d++) usym(wb, w, x_init_range);
            }
            
            // Keep repopulating until we hit a non-zero probability distribution
            while (!std::isfinite(log_p(xw, sw, tw, a, ws))) {
                for (int d = 0; d < D; d++) xw[d] = usym(wb, w, x_init_range);
            }
        }
    });
    refresh_logp(wb, a, pool, wss);
}

// Recenter walker batch by batch
void recenter_batch(WalkerBatch& wb, ThreadPool* pool){
    int B = wb.B;
    int n_workers = pool -> n_workers();
    pool -> run([&](int th) {
        int w0, w1;
        chunk_range(B, n_workers, th, w0, w1);
        for (int w = w0; w < w1; w++) recenter_walker(&wb.x[(std::size_t)w * D]);
    });
}

// Evaluate log|Ψ| for each walker 
void refresh_logp(WalkerBatch& wb, const Ansatz& a, ThreadPool* pool, std::vector<Workspace>& wss) {
    int B = wb.B;
    int n_workers = (int)wss.size();
    pool -> run([&](int th) {
        int w0, w1;
        chunk_range(B, n_workers, th, w0, w1);
        Workspace& ws = wss[th];
        for (int w = w0; w < w1; w++) {
            const double* xw = &wb.x[(std::size_t)w * D];
            const double* sw = &wb.s[(std::size_t)w * N];
            const double* tw = &wb.t[(std::size_t)w * N];

            wb.logp[w] = log_p(xw, sw, tw, a, ws);
        }
    });
}

// Accept-reject step
static bool wb_metro_accept(WalkerBatch& wb, int w, double logp_old, double logp_new) {
    if (logp_old == -std::numeric_limits<double>::infinity()) return std::isfinite(logp_new);
    return u01(wb, w) < std::exp(2.0 * (logp_new - logp_old));
}

// Update log|Ψ| when doing s, t swaps
static double wb_logp_from_S_ratio(double logp_cur, double S_cur, double S_new) {
    if (!std::isfinite(logp_cur)) return std::log(std::fabs(S_new));
    return logp_cur + std::log(std::fabs(S_new)) - std::log(std::fabs(S_cur));
}

// Per COORDINATE proposal, walker w consumes, in this order:
//     1. uint_below(D)      the coordinate index
//     2. usym(step)         the displacement
//     3. u01()              the acceptance draw -- CONDITIONAL, see below
//
// Per DISCRETE (spin or isospin) proposal:
//     1. uint_below(n_a)    index into the up/proton list
//     2. uint_below(n_b)    index into the down/neutron list
//     3. u01()              the acceptance draw -- CONDITIONAL, see below
// Do one metropolis sweep
void sweep_one(WalkerBatch& wb, int w, const Ansatz& a, double step, Workspace& ws) {
    double* xw = &wb.x[(std::size_t)w * D];
    double* sw = &wb.s[(std::size_t)w * N];
    double* tw = &wb.t[(std::size_t)w * N];
    double logp = wb.logp[w];

    // Metropolis over position
    for (int j = 0; j < draws; j++) {
        int idx = uint_below(wb, w, D);
        double old = xw[idx];
        xw[idx] += usym(wb, w, step);
        double logp_new = log_p(xw, sw, tw, a, ws);
        if (wb_metro_accept(wb, w, logp, logp_new)) {
            logp = logp_new;
            wb.acc[w]++;
        } else {
            xw[idx] = old;
        }        
    }

    // Check if spin and isospin are fixed or not
    bool do_spin = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0);
    bool do_tau  = (tau_mode == TauMode::Sampled && N_p > 0 && N_n > 0);

    // Compute for all possible spin and isospin swaps, propose swaps (fixed Sz and Tz), compare log|Ψ| and do accept-reject
    if (do_spin || do_tau) {
        build_st_table(xw, a, ws);
        double S_cur = S_from_table(sw, tw, a, ws);

        if (do_spin) {
            for (int j = 0; j < spin_draws; j++) {
                ws.up_list.clear();
                ws.dn_list.clear();
                for (int i = 0; i < N; i++) {
                    if (sw[i] > 0) ws.up_list.push_back(i);
                    else ws.dn_list.push_back(i);
                }
                int iu = ws.up_list[uint_below(wb, w, N_u)];
                int id = ws.dn_list[uint_below(wb, w, N_d)];
                std::swap(sw[iu], sw[id]);
                double S_new = S_from_table(sw, tw, a, ws);
                double logp_new = wb_logp_from_S_ratio(logp, S_cur, S_new);
                if (wb_metro_accept(wb, w, logp, logp_new)) {
                    logp = logp_new;
                    S_cur = S_new;
                    wb.sp_acc[w]++;
                } else {
                    std::swap(sw[iu], sw[id]);
                }
            }
        }
        if (do_tau) {
            for (int j = 0; j < tau_draws; j++) {
                ws.p_list.clear();
                ws.n_list.clear();
                for (int i = 0; i < N; i++) {
                    if (tw[i] > 0) ws.p_list.push_back(i);
                    else ws.n_list.push_back(i);
                }
                int ip = ws.p_list[uint_below(wb, w, N_p)];
                int in = ws.n_list[uint_below(wb, w, N_n)];
                std::swap(tw[ip], tw[in]);
                double S_new = S_from_table(sw, tw, a, ws);
                double logp_new = wb_logp_from_S_ratio(logp, S_cur, S_new);
                if (wb_metro_accept(wb, w, logp, logp_new)) {
                    logp = logp_new;
                    S_cur = S_new;
                    wb.tau_acc[w]++;
                } else {
                    std::swap(tw[ip], tw[in]);
                }
            }
        }
    }
    wb.logp[w] = logp;    
}

// Thermalize all the bathces
double therm_batch(WalkerBatch& wb, const Ansatz& a, double step, int n_sweeps, ThreadPool* pool, std::vector<Workspace>& wss) {
    int B = wb.B;
    int n_workers = (int)wss.size();

    std::fill(wb.acc.begin(), wb.acc.end(), 0LL);
    std::fill(wb.sp_acc.begin(), wb.sp_acc.end(), 0LL);
    std::fill(wb.tau_acc.begin(), wb.tau_acc.end(), 0LL);

    // Do full sweeps, recenter walker after each one
    pool->run([&](int th) {
        int w0, w1;
        chunk_range(B, n_workers, th, w0, w1);
        Workspace& ws = wss[th];
        for (int w = w0; w < w1; w++) {
            for (int sweep = 0; sweep < n_sweeps; sweep++) {
                sweep_one(wb, w, a, step, ws);
                recenter_walker(&wb.x[(std::size_t)w * D]);
            }
        }
    });

    // Batch acceptance rates
    long long total_acc = 0;
    for (int w = 0; w < B; w++) total_acc += wb.acc[w];
    return (double)total_acc / ((double)B * (double)n_sweeps * (double)draws);
}

static double walker_r2(const double* xw) {
    double R_cm[dim] = {};
    for (int p = 0; p < N; p++) for (int d = 0; d < dim; d++) R_cm[d] += xw[p*dim + d];
    for (int d = 0; d < dim; d++) R_cm[d] /= N;

    double r2 = 0.0;
    for (int p = 0; p < N; p++) {
        for (int d = 0; d < dim; d++) {
            double diff = xw[p*dim + d] - R_cm[d];
            r2 += diff * diff;
        }
    }
    return r2 / N;
}

void chunk_range_pub(int B, int n_workers, int th, int& w0, int& w1) {
    chunk_range(B, n_workers, th, w0, w1);
}

double walker_r2_pub(const double* xw) {
    return walker_r2(xw);
}

// Take measurement from one walker
void record_one_walker(WalkerBatch& wb, const Ansatz& a, Workspace& ws, std::vector<double>& O, int w, int r, std::vector<double>& E_pool, std::vector<double>& O_pool, std::vector<uint8_t>& valid_pool, BatchStats& bs) {
    const std::size_t P = a.n_params();
    const int B = wb.B;

    double* xw = &wb.x[(std::size_t)w*D];
    double* sw = &wb.s[(std::size_t)w*N];
    double* tw = &wb.t[(std::size_t)w*N];

    std::size_t idx = (std::size_t)r*B + w;
    double E_loc;
    bool ok = local_E(xw, sw, tw, a, ws, O, E_loc);
    valid_pool[idx] = ok ? 1 : 0;

    if (ok) {
        E_pool[idx] = E_loc;
        for (std::size_t k = 0; k < P; k++) O_pool[idx*P + k] = O[k];

        bs.Ew_sum[w]  += E_loc;
        bs.E2w_sum[w] += E_loc * E_loc;
        bs.l2w_sum[w] += ws.l2_val;
        bs.r2w_sum[w] += walker_r2(xw);
        bs.nw[w]++;
    }
}

// Take samples across walkers, sum net result
void record_batch(WalkerBatch& wb, const Ansatz& a, double step, int records, ThreadPool* pool, std::vector<Workspace>& wss, std::vector<double>& E_pool, std::vector<double>& O_pool, std::vector<uint8_t>& valid_pool, BatchStats& bs) {
    int B = wb.B;
    int n_workers = (int)wss.size();
    std::size_t P = a.n_params();

    bs.Ew_sum.assign(B, 0.0);
    bs.E2w_sum.assign(B, 0.0);
    bs.l2w_sum.assign(B, 0.0);
    bs.r2w_sum.assign(B, 0.0);
    bs.nw.assign(B, 0);

    pool->run([&](int th) {
        int w0, w1;
        chunk_range(B, n_workers, th, w0, w1);
        Workspace& ws = wss[th];
        std::vector<double> O(P);

        // Sum across samples done by each worker, skip between steps
        for (int r = 0; r < records; r++) {
            for (int w = w0; w < w1; w++) {
                for (int sweep = 0; sweep < sweeps_between_records; sweep++) {
                    sweep_one(wb, w, a, step, ws);
                    recenter_walker(&wb.x[(std::size_t)w * D]);
                }
                record_one_walker(wb, a, ws, O, w, r, E_pool, O_pool, valid_pool, bs);
            }   
        }
    });

    // Sum over walkers
    bs.E_sum = bs.E2_sum = bs.l2_sum = bs.r2_sum = 0.0;
    bs.n_valid = bs.n_invalid = 0;
    for (int w = 0; w < B; w++) {
        bs.E_sum += bs.Ew_sum[w];
        bs.E2_sum += bs.E2w_sum[w];
        bs.l2_sum += bs.l2w_sum[w];
        bs.r2_sum += bs.r2w_sum[w];
        bs.n_valid += bs.nw[w];
    }    
    bs.n_invalid = (long long)records * (long long)B - bs.n_valid;
}

// Evaluate average in mean across batches
double batch_error(const BatchStats& bs) {
    std::vector<double> means;
    means.reserve(bs.nw.size());
    for (std::size_t w = 0; w < bs.nw.size(); w++) {
        if (bs.nw[w] > 0) means.push_back(bs.Ew_sum[w]/bs.nw[w]);
    }
    std::size_t n_active = means.size();
    if (n_active < 2) return std::numeric_limits<double>::infinity();

    double mean_of_means = 0.0;
    for (double m : means) mean_of_means += m;
    mean_of_means /= n_active;

    double var = 0.0;
    for (double m : means) {
        double d = m - mean_of_means;
        var += d*d;
    }
    var /= (double)(n_active-1);

    return std::sqrt(var / n_active);
}

void masked_O_exp(const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, std::size_t P, ThreadPool* pool, std::vector<double>& O_exp) {
    int n_workers = pool -> n_workers();
    std::vector<double> partials((std::size_t)n_workers*P, 0.0);
    std::vector<long long> n_valid_part(n_workers, 0);        

    std::size_t chunk = n_samples / n_workers;
    pool -> run([&](int th) {
        std::size_t start = (std::size_t)th * chunk;
        std::size_t end = (th == n_workers - 1) ? n_samples : start + chunk;
        for (std::size_t k = 0; k < P; k++) partials[(std::size_t)th * P + k] = 0.0;
        long long nv = 0;
        for (std::size_t i = start; i < end; i++) {
            if (!valid_pool[i]) continue;
            nv++;
            for (std::size_t k = 0; k < P; k++) partials[(std::size_t)th*P + k] += O_pool[i*P + k];
        }
        n_valid_part[th] = nv;
    });

    long long n_valid_total = 0;
    for (int th = 0; th < n_workers; th++) n_valid_total += n_valid_part[th];
    O_exp.assign(P, 0.0);
    if (n_valid_total == 0) return;
    for (int th = 0; th < n_workers; th++) {
        for (std::size_t k = 0; k < P; k++) O_exp[k] += partials[(std::size_t)th * P + k];
    }
    for (std::size_t k = 0; k < P; k++) O_exp[k] /= (double)n_valid_total;
}