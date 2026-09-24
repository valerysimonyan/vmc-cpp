#include "local_e.h"
#include "jet_eval.h"
#include "eval.h"
#include "prof.h"
#include "exchange_kernels.h"
#include "net_forward.h"

#include <algorithm>

// Compose energy computation
int eval_local_E_device(DeviceState& ds, cublasHandle_t handle, const Ansatz& a, Workspace& ws, int B, cudaStream_t stream, bool stash_for_O) {
    if (B <= 0) return 0;

    {
        VMC_PROF("eval_cached", stream);
        eval_jet_prepare(ds, handle, B, stream, stash_for_O);

        CUDA_CHECK(cudaMemcpy(ds.dets_psi.d, ds.dets.d, (std::size_t)K * B * sizeof(real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(ds.xi_psi.d,   ds.xi.d,   (std::size_t)B * m_feat * sizeof(real), cudaMemcpyDeviceToDevice));
    }

    {
        VMC_PROF("jet_pass", stream);
        for (int w_off = 0; w_off < B; w_off += jet_walkers) {
            const int Bc = std::min((int)jet_walkers, B - w_off);
            eval_jet_chunk(ds, handle, Bc, w_off, B, stream);
        }
    }

    int n_fallback = 0;
    if (nuc_pot != NucPot::Off) {
        VMC_PROF("exchange", stream);
        // S0 = S_from_table(s,t), local_E's denominator for every ratio. This
        // overwrites xi, rho_out, M_batch and dets -- hence the snapshots.
        {
            VMC_PROF("st_table", stream);
            build_st_table_batch(ds, handle, B, stream);
            S_from_table_batch(ds, handle, B, ds.s.d, ds.t.d, ds.S0.d, stream);
        }
        {
            VMC_PROF("gate_plan", stream);
            ex_rank2_gate(ds.dets_psi.d, ds.rank2_ok.d, B, stream);
            ex_plan(ds.s.d, ds.t.d, ds.pair_ij.d, ds.ex_active.d, B, stream);
        }

        const int per_w = ex_types * ex_npairs;
        for (int w_off = 0; w_off < B; w_off += ex_walkers) {
            const int Bc = std::min((int)ex_walkers, B - w_off);
            {
                VMC_PROF("rho_slots", stream);
                ex_xi_swap(ds.xi_psi.d, ds.h_out.d, ds.s.d, ds.t.d, ds.pair_ij.d, ds.xi_swap.d, Bc, w_off, stream);
                net_forward(handle, ds.rho_net_d, ds.params.d, ds.xi_swap.d, Bc * per_w, ds.act_a.d, ds.act_b.d, ds.rho_swap.d, stream);
            }
            { VMC_PROF("rank2", stream); ex_S_swap(ds.rho_swap.d, ds.dets_psi.d, ds.Minv_batch.d, ds.orb_out.d, ds.s.d, ds.t.d, ds.pair_ij.d, ds.ex_active.d, ds.rank2_ok.d, ds.S_swap.d, Bc, w_off, stream); }
        }
        { VMC_PROF_HOST("fallback"); n_fallback = ex_fallback_host(ds, a, ws, B); }
    }

    VMC_PROF("assemble", stream);
    coulomb_batch(ds.x.d, ds.t.d, ds.V_coul.d, B, stream);
    ex_assemble(ds.x.d, ds.s.d, ds.t.d, ds.pair_ij.d, ds.S_swap.d, ds.S0.d, ds.E_kin.d, ds.v3n_out.d, ds.V_coul.d, ds.valid_jet.d, ds.V_nuc.d, ds.E_loc.d, ds.valid_loc.d, B, ds.params.d, ds.P, stream);
    return n_fallback;
}
