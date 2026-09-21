#pragma once

// Thread count
inline constexpr int n_thread = 20;

// Degrees of freedom
inline constexpr int N = 6;
inline constexpr int dim = 3;
inline constexpr int D = N*dim;

// Spin Related Parametrs, set spins to be sampled or fixed, set determinant to be block diagonal or full, and the number of spin ups and downs
enum class SpinMode {Fixed, Sampled};
inline constexpr SpinMode spin_mode = SpinMode::Sampled;
inline constexpr int N_u = 4;
inline constexpr int N_d = N - N_u;
static_assert(N_u <= N && N_u >=  0, "N_u must be between 0 and N");

enum class TauMode {Fixed, Sampled};
inline constexpr TauMode tau_mode = TauMode::Sampled;
inline constexpr int N_p = 3;
inline constexpr int N_n = N - N_p;
static_assert(N_p <= N && N_p >=  0, "N_p must be between 0 and N");

//--- Model O Potential ---//
// Physical constants
inline constexpr double m_n = 938.91875;    // Nucleon mass (MeV)
inline constexpr double hbarc = 197.32697;  // hbar c in MeV*fm
inline constexpr double hbar2_2m = hbarc * hbarc / (2.0 * m_n);
inline constexpr double PI = 3.14159265358979323846;

// 2-Body
inline constexpr double C10 = -7.040;  // fm^2, S=1 T=0 channel
inline constexpr double C01 = -5.275;  // fm^2, S=0 T=1 channel
inline constexpr double R10 = 1.546;   // fm
inline constexpr double R01 = 1.830;   // fm

// 3-Body
inline constexpr double fpi        = 92.4;     // MeV, pion decay constant
inline constexpr double Lambda_chi = 1000.0;   // MeV
inline constexpr double R3         = 1.1;      // fm
inline constexpr double cE         = 1.2945;   // R3=1.1 refit (Table II rounds to 1.295)
inline constexpr double V3_0 = cE * (hbarc*hbarc*hbarc*hbarc*hbarc*hbarc) / (fpi*fpi*fpi*fpi * Lambda_chi * (PI * PI * PI) * (R3*R3*R3*R3*R3*R3));

// Finite size Coulomb interaction
inline constexpr bool nuc_coulomb = true;                // Coulomb force switch
inline constexpr double alpha_em = 1.0 / 137.035999084;  // Fine structure constant
inline constexpr double b_coul = 4.27;                   // fm^-1, Finite size proton form factor

// Settings for potential
enum class NucPot { Off, ModelO };
inline constexpr NucPot nuc_pot = NucPot::ModelO;
inline constexpr bool nuc_3N = true;  // 3N force switch
// ----------------- //

// Monte Carlo parameters
inline constexpr int therm_steps_init = 500;    // Thermalization steps upon initialization
inline constexpr int therm_init_blocks = 20;    // Initial thermalization is split into this many blocks, tune acceptance at end of each block
inline constexpr double batch_grow_frac = 0.7;  // of N_descent
inline constexpr int batch_grow_factor = 2;     // Maximum batch growth
inline constexpr int draws = N*dim;             // Number of times we draw a position in a Metropolis update before recording a sample
inline constexpr int spin_draws = N;            // Number of times we draw a spin in a Metropolis update before recording a sample
inline constexpr int tau_draws = N;             // Number of times we draw an isospin in a Metropolis update before recording a sample
inline constexpr double step0 = 1.0;            // Initial step size for the thread
inline constexpr double x_init_range = 2.0;     // Initial range of positions

// Walker configuration
inline constexpr int walker_per_th = 290;           // Number of walkers per thread
inline constexpr int records_per_iter = 2;       // Samples recorded per iteration
inline constexpr int sweeps_between_records = 3;  // Decorrelation sweeps
inline constexpr int therm_re_sweep = 3;          // Thermalization sweeps upon parameter update

inline constexpr int n_walkers = n_thread * walker_per_th;
inline constexpr int records_per_iter_max = records_per_iter * batch_grow_factor;

inline constexpr unsigned long long rng_seed = 20260826ULL;
static_assert(n_walkers % n_thread == 0);

inline constexpr int jet_chunk = 0;
inline constexpr double o_pool_max_gb = 8.0;  // Set cap of maximum amount of data O_pool may hold

// Ansatz parameters
inline constexpr int K = N*(dim+2)+1;  // Determinant count
inline constexpr int m_feat = 2*N*(dim + 2)+1;  // Deepsets feature dimension

// Envelope
inline constexpr double eps_env = 0.7;   // fm -- envelope softening length, avoids the r=0 cusp in exp(-alpha*sqrt(r2+eps_env^2))
inline constexpr double beta_min = 0.1;  // fm^-1 -- envelope

// Total Descent Parameters
inline constexpr int N_descent = 10000;  // Total step count
inline constexpr double clip_mad = 5.0;  // Clipping parameter for gradient
inline constexpr int grow_at_iter = (int)(batch_grow_frac * N_descent);

// ADAM descent parameters
inline constexpr int N_gd = 0;
inline constexpr double lr = 0.075;
inline constexpr double beta1 = .9;
inline constexpr double beta2 = .999;
inline constexpr double eps = 1e-8;

// SR descent parameters
inline constexpr int N_sr = N_descent - N_gd;
inline constexpr double sr_eta        = 0.10;   // Initial learning rate
inline constexpr double sr_lambda0    = 100.0;  // Initial lambda (added weight to diagonal elements)
inline constexpr double sr_lambda_min = 0.4;    // Minimum lambda
inline constexpr double sr_rho        = 0.995;  // Decay rate of lambda 
inline constexpr double sr_eps        = 1e-5;   // Epsilon regulator for SR matrix
inline constexpr double sr_cg_tol     = 1e-3;   // Residual before we declare the inverse solved
inline constexpr int    sr_cg_maxit   = 200;     // Maximum number of iterations before we give up trying to solve the inverse
inline constexpr double sr_trust_r2   = 0.1;    // Bound for CG solution to (delta S delta)
inline constexpr double sr_delta_max  = 75.0;   // Maximum size on SR step

// Further inverse stabilizers
inline constexpr bool sr_rms_damp = true;    // Add RMS damping to stabilize inversion
inline constexpr double sr_rms_beta = 0.99;  // Parameter which decays and stabilizes inversion
inline constexpr double sr_rms_eps = 1e-3;   // Regulator we add the SR matrix

// Checkpoint data read
inline constexpr double diss_threshold = -2.5;     // If energy above this we can worry about breakup of nucleon, roughly deuteron binding energy
inline constexpr int diss_watch_window = 50;       // Look at RMS of average radius over last 50 steps
inline constexpr double diss_growth_factor = 1.2;  // Flag only if r_rms also grew >20% over that window
inline constexpr int eval_iters = 50;              // Frozen-eval sampling iterations (pooled with training's `steps`*`n_thread` per iteration for a tight final E_err)
inline constexpr int ckpt_every = 100;             // Take a checkpoint every N descent iterations

// Settings for time measurement
inline constexpr bool prof_enabled     = true;
inline constexpr int  prof_max_depth   = 4;        // ranges nested deeper than this are not measured
inline constexpr int  prof_report_every = 100;     // iterations between reports
inline constexpr const char* prof_report_file = "BENCH.md";

// Transfer Learning Settings
inline constexpr bool resume = false;             // Load best_checkpoint.txt at startup if present
inline constexpr const char* transfer_from = "";  // Path to a source checkpoint; empty = disabled