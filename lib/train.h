#pragma once

#include "wavefunction.h"
#include "walkers.h"
#include "sr.h"
#include "pool.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Structure holding all results per descent step 
struct DescentResult {
    double El_exp;           // Expectation value of local energy
    double El_err;           // Standard error of the mean of local energy
    double var;              // Variance of local energy
    double acceptance;       // Position update acceptance rate
    double spin_acceptance;  // Spin update acceptance rate
    double tau_acceptance;   // Isospin update acceptance rate
    double L2;               // Angular momentum squared expectation value
    double r_rms;            // Root mean square radius
    double total_node_hits;  // Total number of node hits across all walkers
};

// Structure holding all results per SR step
struct SRStepLog {
    double lambda = 0.0;          // SR regularization parameter ( + lambda S_ii term)
    int cg_iters = 0;             // How many CG iterations were used
    double cg_residual = 0.0;     // What the final CG residual was
    double delta_norm = 0.0;      // Norm of the SR step delta
    double sq_metric_norm = 0.0;  // Norm of the square root metric (S^1/2) times the SR step delta
    bool norm_capped = false;     // Whether the SR step was capped by the trust region
};

// Stats from one round of sampling merging over all CPU and GPU walkers
struct SampleStats {
    BatchStats bs;                          // Stats from the batch of walkers
    std::vector<double> E_pool;             // Pool of all valid local energies
    std::vector<unsigned char> valid_pool;  // Pool of all valid flags (1=valid, 0=invalid)

    double acceptance = 0.0, spin_acceptance = 0.0, tau_acceptance = 0.0;              // Acceptance rates
    double metro_ms = 0.0, local_E_ms = 0.0, o_ms = 0.0, gpu_ms = 0.0, xfer_ms = 0.0;  // Times
    long long bytes_up = 0, bytes_dn = 0;                                              // Bytes transferred to/from GPU
};

// Initialize class Backend for training, derived classes are for CPU and GPU
class Backend {
public:
    virtual ~Backend() = default;

    virtual std::size_t walkers() const = 0;
    
    virtual void initial_thermalization(const Ansatz& a) = 0;
    
    virtual void sample(const Ansatz& a, int records, bool frozen, SampleStats& s) = 0;

    virtual void gradient(std::size_t n_samples, long long n_valid, const ClipStats& clip) = 0;
    
    virtual void grad_to_host(std::size_t first, std::size_t count, double* out) = 0;
    
    virtual SRStepLog sr_step(Ansatz& a, int iter, std::size_t n_samples, long long n_valid, std::vector<double>& delta, double& rms_damp_mean, long long& n_scalar_dl) = 0;
    
    virtual void iteration_end(double ms, int i, int records, std::size_t P, const char* tag) { 
        (void)ms; 
        (void)i; 
        (void)records; 
        (void)P; 
        (void)tag; 
    }
    
    virtual void finish(int records, std::size_t P, const char* tag) { 
        (void)records; 
        (void)P; 
        (void)tag; 
    }
};

std::unique_ptr<Backend> make_backend(const Ansatz& a, bool training);

DescentResult train(Ansatz& a);
DescentResult evaluate_frozen(Ansatz& a);

void ADAM(const std::vector<double>& grad, std::vector<double>& m, std::vector<double>& v, int i, Ansatz& a);
SRStepLog SR_step(const std::vector<double>& grad, const std::vector<double>& O_pool, const std::vector<double>& O_exp, Ansatz& a, SROp& sr_op, std::vector<double>& delta, int iter, std::size_t n_params, ThreadPool* pool, const double* d_rms, std::size_t n_samples, const uint8_t* valid_pool, std::size_t n_valid, std::vector<double>& M_inv_diag, std::vector<double>& S_delta);
void compute_obs(const BatchStats& bs, DescentResult& r, std::size_t P, const std::vector<double>& E_pool, const std::vector<double>& O_pool, const std::vector<uint8_t>& valid_pool, std::size_t n_samples, ThreadPool* pool, std::vector<double>& O_exp, std::vector<double>& grad);
