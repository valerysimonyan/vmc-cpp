#include "hamiltonian.h"
#include "wavefunction.h"
#include "constants.h"
#include "envelope.h"

#include <cassert>
#include <cmath>
#include <vector>

// Log local energy and dlogpsi_d theta_i simultaneously as more computationally efficient
bool local_E(const double* x, const double* s, const double* t, const Ansatz& a, Workspace& ws, std::vector<double>& O_out, double& E_out) {
    std::size_t n_params = a.n_params();
    if(O_out.size() != n_params) O_out.resize(n_params);

    double dpsi = psi(x, s, t, a, ws, true);
    (void)dpsi;
    
    const double S = S_sum(ws.drho.data(), ws.dets.data());
    // Return 0 if we hit a node, as parameters diverge we set them to zero
    if (!std::isfinite(S) || std::fabs(S) < psi_floor) {
        ws.n_node_hits++;
        return false;
    }

    // Declare Jet psi, throw error if it is different from double psi
    Jet pj = jpsi(x, s, t, a, ws);
    bool psi_mismatch = std::fabs(pj.v - dpsi) > 1e-6 * std::max(1.0, std::fabs(dpsi));
    if (!std::isfinite(pj.v) || std::fabs(pj.v) < psi_floor || psi_mismatch) {
        ws.n_node_hits++;
        return false;
    }

    // Record L^2
    ws.l2_val = l2_local(ws.x_sh.data(), pj.g.data(), 1, pj.v);

    // Now we build E_loc
    double E_loc = kinetic(pj.l, pj.v);
    if (!std::isfinite(E_loc)) {
        ws.n_node_hits++;
        return false;
    }

    if (nuc_3N) {
        E_loc += V_3N(x);
        if (!std::isfinite(E_loc)) {
            ws.n_node_hits++;
            return false;
        }
    }

    if (nuc_coulomb) {
        E_loc += V_coulomb(x, t);
        if (!std::isfinite(E_loc)) {
            ws.n_node_hits++;
            return false;
        }
    }

    if (nuc_pot != NucPot::Off) {
        build_st_table(x, a, ws);
        double S0 = S_from_table(s, t, a, ws);
        assert(std::fabs(S0 - S) < 1e-10 * std::max(1.0, std::fabs(S)));

        bool use_rank2 = rank2_well_conditioned(ws);
        for (int i = 0; i < N; i++) {
            ws.s_swap[i] = s[i];
            ws.t_swap[i] = t[i];
        }

        // Exchange ratio from the (s,t) table: rank-2 determinant update
        auto ratio = [&](int, int i, int j, int type) {
            double sni, tni, snj, tnj;
            ex_swapped_labels(s[i], t[i], s[j], t[j], type, sni, tni, snj, tnj);
            return swap_ratio(s, t, i, j, sni, tni, snj, tnj, S0, use_rank2, a, ws);
        };
        E_loc += V_2N(x, s, t, a.jc.data(), ratio);

        if (!std::isfinite(E_loc)) {
            ws.n_node_hits++;
            return false;
        }
    }
    fill_O(a, ws, S, O_out, s, t);

    E_out = E_loc;
    return true;
}

