#include "eval.h"
#include "prof.h"
#include "det_kernels.h"
#include "exchange_kernels.h"
#include "sampler_kernels.h"
#include "gpu_sampler.h"

#include <algorithm>
#include <chrono>

// Upload walkers and reset acceptance rates
void upload_and_reset(DeviceState& ds, const WalkerBatch& wb, PinnedArray& staging) {
    ds.upload_walkers(wb, staging);
    ds.acc.zero();
    ds.sp_acc.zero();
    ds.tau_acc.zero();
}

void sweep_device(DeviceState& ds, cublasHandle_t handle, int B, double step, cudaStream_t stream) {
    // Propose new coordinates
    for (int j = 0; j < draws; j++) {
        { VMC_PROF("propose",     stream); propose_coord(ds, B, step, stream); }
        { VMC_PROF("eval_double", stream); eval_logp_batch_prop(ds, handle, B, ds.x_prop.d, ds.S_prop.d, ds.logp_prop.d, stream); }
        { VMC_PROF("accept",      stream); accept_coord(ds, B, stream); }
    }

    // Propose discrete steps
    const bool do_spin = (spin_mode == SpinMode::Sampled && N_u > 0 && N_d > 0);
    const bool do_tau  = (tau_mode  == TauMode::Sampled  && N_p > 0 && N_n > 0);
    if (!(do_spin || do_tau)) return;

    {
        VMC_PROF("st_table", stream);
        build_st_table_batch(ds, handle, B, stream);
        S_from_table_batch(ds, handle, B, ds.s.d, ds.t.d, ds.S_cur.d, stream);
    }

    VMC_PROF("discrete_block", stream);
    if (do_spin) {
        for (int j = 0; j < spin_draws; j++) {
            propose_discrete(ds, B, true, stream);
            S_from_table_batch(ds, handle, B, ds.s_prop.d, ds.t.d, ds.S_prop.d, stream);
            accept_discrete(ds, B, true, stream);
        }
    }
    if (do_tau) {
        for (int j = 0; j < tau_draws; j++) {
            propose_discrete(ds, B, false, stream);
            S_from_table_batch(ds, handle, B, ds.s.d, ds.t_prop.d, ds.S_prop.d, stream);
            accept_discrete(ds, B, false, stream);
        }
    }
}


// Thermalization sweeps
double therm_batch_device(DeviceState& ds, cublasHandle_t handle, int B, double step, int n_sweeps, cudaStream_t stream) {
    ds.acc.zero();
    ds.sp_acc.zero();
    ds.tau_acc.zero();

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        sweep_device(ds, handle, B, step, stream);
        recenter_device(ds, B, stream);
    }

    long long acc = 0, sp = 0, tau = 0;
    download_acceptance(ds, B, acc, sp, tau);
    if (n_sweeps <= 0) return 0.0;
    return (double)acc / ((double)B * (double)n_sweeps * (double)draws);
}
