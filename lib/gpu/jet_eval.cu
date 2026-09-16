#include "jet_eval.h"
#include "jet_kernels.h"
#include "det_kernels.h"
#include "detjet_kernels.h"
#include "compose_kernels.h"
#include "eval.h"

#include <algorithm>


// Evlaute double log p
void eval_jet_prepare(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream) {
    if (B <= 0) return;

    eval_logp_batch_prop(ds, handle, B, ds.x.d, ds.S.d, ds.logp_prop.d, stream);
    batched_inverse(handle, B*K, ds.M_batch.d, ds.lu_ptrs.d, ds.inv_ptrs.d, ds.lu_piv.d, ds.inv_info.d, ds.Minv_batch.d, stream);
}

void eval_jet_chunk(DeviceState& ds, cublasHandle_t handle, int Bc, int w_off, int B_tot, cudaStream_t stream) {
    if (Bc <= 0) return;

    const int hidden = std::max(1, std::max(std::max(ds.h_net_d.hidden_width, ds.rho_net_d.hidden_width), ds.orb_net_d.hidden_width));

    // Jet forward passes
    build_jet_feat(ds.x.d + (std::size_t)w_off * D, ds.s.d + (std::size_t)w_off * N, ds.t.d + (std::size_t)w_off * N, ds.jet_feat.d, Bc, stream);
    jet_net_forward(handle, ds.h_net_d, ds.params.d, ds.jet_feat.d, dim+2, Bc*N, ds.jet_a.d, ds.jet_b.d, hidden, ds.jet_h.d, m_feat, stream);
    jet_xi_reduce(ds.jet_h.d, ds.jet_xi.d, Bc, stream);
    jet_net_forward(handle, ds.rho_net_d, ds.params.d, ds.jet_xi.d, m_feat, Bc, ds.jet_a.d, ds.jet_b.d, hidden, ds.jet_rho.d, K, stream);
    jet_net_forward(handle, ds.orb_net_d, ds.params.d, ds.jet_feat.d, dim+2, Bc*N, ds.jet_a.d, ds.jet_b.d, hidden, ds.jet_orb.d, K*N, stream);

    // Energy assembly
    det_jet_assemble(ds.jet_orb.d, ds.Minv_batch.d, ds.dets.d, ds.jet_det.d, Bc, w_off, B_tot, stream);
    psi_jet_compose(ds.jet_rho.d, ds.jet_det.d, ds.x_sh.d, ds.params.d, ds.P, ds.jet_psi.d, ds.S_jet_v.d, Bc, w_off, B_tot, stream);
    kinetic_l2(ds.jet_psi.d, ds.x_sh.d, ds.E_kin.d, ds.l2_out.d, Bc, w_off, B_tot, stream);
    v3n_batch(ds.x.d, ds.v3n_out.d, Bc, w_off, stream);
    validity_jet(ds.jet_psi.d, ds.S.d, ds.x_sh.d, ds.params.d, ds.P, ds.E_kin.d, ds.psi_dbl.d, ds.valid_jet.d, Bc, w_off, B_tot, stream);
}

// Assign above function computation to chunks
void eval_jet_batch(DeviceState& ds, cublasHandle_t handle, int B, cudaStream_t stream) {
    if (B <= 0) return;
    eval_jet_prepare(ds, handle, B, stream);
    for (int w_off = 0; w_off < B; w_off += jet_walkers) {
        const int Bc = std::min((int)jet_walkers, B - w_off);
        eval_jet_chunk(ds, handle, Bc, w_off, B, stream);
    }
}