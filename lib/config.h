#pragma once

#include "wavefunction.h"
#include "constants.h"
#include "sr.h"

#include <iosfwd>
#include <string>
#include <vector>


// [adam] settings of the ADAM warm-up and of the optional envelope Adam during SR; defaults from constants.h
struct AdamSettings {
    double lr = ::lr;
    double beta1 = ::beta1;
    double beta2 = ::beta2;
    double eps = ::eps;
    double env_lr = env_adam_lr;              // Adam step on the envelope block after each SR step; 0 disables ...
    double env_decay_it = env_adam_decay_it;  // ... with lr_t = env_lr / (1 + t / env_decay_it)
    bool env_alpha = env_adam_alpha;          // false: Adam on the Jastrow only, alpha left to SR
};

struct GpuSettings {
    std::string gpus = "best";  // Best: the GPU with the most free memory, `walkers` walkers; a list (0,1): these GPUs, `walkers` each;
                                // auto: the planner sizes walkers on every usable GPU; auto_one: the single GPU where it fits the most
    std::vector<int> devices;   // The list, when gpus is one
    bool same_seed = false;     // Test mode: every GPU draws the same random numbers
    bool plan_only = false;     // Print the arena size for `walkers` and exit
    double overhead_gb = 0.4;   // Planner: memory per GPU kept back for non-arena use ...
    double margin_gb = 0.5;     // ... plus this safety margin
    int max_util = 30;          // Planner: skip GPUs busier than this (%)
    int max_walkers = 0;        // Planner: cap on walkers per GPU (0: none)
    int min_walkers = 256;      // Planner: skip GPUs that fit fewer
    int jet_chunk = ::jet_chunk;  // Walkers per jet chunk; the jet buffers (the largest after the O pool) hold min(walkers, jet_chunk)

};

// Run-time settings. Every default is the value in constants.h; a run file (INI) overrides any of them:
//   ./main run.ini        sections [build] [network] [training] [sr] [adam], key = value, '#' starts a comment

struct RunConfig {
    // Network hidden layer sizes
    std::vector<int> h_hidden{32}, rho_hidden{32}, orb_hidden{32};

    // [training]
    int gd_iterations = N_gd;                     // ADAM warm-up iterations
    int sr_iterations = N_sr;                     // SR iterations after the warm-up
    int walkers = n_walkers;                      // CPU: all walkers; GPU: per GPU unless the planner sizes it
    int records = records_per_iter;               // samples recorded per walker and iteration
    int grow_factor = batch_grow_factor;          // records grow to records * grow_factor ...
    double grow_at = batch_grow_frac;             // ... after this fraction of all iterations
    int sweeps_between = sweeps_between_records;  // decorrelation sweeps between records
    int therm_sweeps = therm_re_sweep;            // re-thermalisation sweeps after each parameter update
    int therm_init = therm_steps_init;            // initial thermalisation sweeps ...
    int therm_blocks = therm_init_blocks;         // ... in this many blocks, step tuned after each
    double step = step0;                          // initial Metropolis step (fm)
    int ckpt_every = ::ckpt_every;                // periodic checkpoint interval
    int eval_iters = ::eval_iters;                // frozen-evaluation iterations
    double diss_threshold = ::diss_threshold;     // dissociation watch: energy above this ...
    int diss_window = diss_watch_window;          // ... and r_rms grown over this many iterations ...
    double diss_growth = diss_growth_factor;      // ... by more than this factor
    double clip_mad = ::clip_mad;                 // gradient: local energies clipped to median +- clip_mad * MAD
    std::string resume;                           // checkpoint to continue from (empty: none)
    std::string transfer;                         // checkpoint to transfer weights from (empty: none)

    SrSettings sr;      // [sr]
    AdamSettings adam;  // [adam]
    GpuSettings gpu;    // [gpu]

    int iterations() const { 
        return gd_iterations + sr_iterations; 
    }
    
    int records_max() const { 
        return records * grow_factor; 
    }
    
    int grow_at_iter() const { 
        return (int)(grow_at * iterations()); 
    }
};

RunConfig load_run_config(const char* path);

void write_run_config(std::ostream& o, const RunConfig& c);

void validate_config(const Ansatz& a, const RunConfig& c);