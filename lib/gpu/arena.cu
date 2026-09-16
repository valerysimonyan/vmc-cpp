#include "arena.h"
#include "exchange_kernels.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <vector>

// Write params into a flat layour then check if it matches
std::size_t check_param_layout(const Ansatz& a) {
    const std::size_t P = a.n_params();
    std::vector<double> flat(P);
    a.copy_params_flat(flat.data());
    for (std::size_t k = 0; k < P; k++) {
        if (flat[k] != a.get_param(k)) return k;
    }
    return (std::size_t)-1;
}

std::size_t DeviceState::total_bytes() const {
    return x.bytes() + s.bytes() + t.bytes() + logp.bytes() + valid.bytes()
            + rng_ctr.bytes() + params.bytes()
            + E_pool.bytes() + O_pool.bytes() + valid_pool.bytes();
}

DeviceState::DeviceState(const Ansatz& a, bool verbose) {
    B = (std::size_t)n_walkers;
    P = a.n_params();
    Ns_max = (std::size_t) n_walkers * (std::size_t)records_per_iter_max;

    // Check for issues with flattening parameters
    const std::size_t bad = check_param_layout(a);
    if (bad != (std::size_t)-1) {
        std::ostringstream oss;
        oss << "DeviceState: flat parameter layout disagrees with Ansatz::get_param at index "
            << bad << ".\n  copy_params_flat and get_param/add_to_param must both be "
            << "[h | rho | orb | alpha]; see the layout note in arena.h.";
        throw std::runtime_error(oss.str());
    }

    // Check if we're over the O_pool data limit
    const double o_pool_gb = (double)Ns_max * (double)P * sizeof(double)
                             / (1024.0 * 1024.0 * 1024.0);
    if (o_pool_gb > o_pool_max_gb) {
        std::ostringstream oss;
        oss << "DeviceState: O_pool would need " << o_pool_gb << " GiB, over the "
            << o_pool_max_gb << " GiB cap (o_pool_max_gb in constants.h).\n"
            << "  Ns_max = " << Ns_max << " (n_walkers " << n_walkers
            << " x records_per_iter_max " << records_per_iter_max << "), P = " << P
            << ".\n  Reduce walker_per_th, records_per_iter, or the network widths.";
        throw std::runtime_error(oss.str());
    }

    // How many bytes I want
    const std::size_t want =
          B*(std::size_t)D*sizeof(real) + 2*B*(std::size_t)N*sizeof(real)
        + B*sizeof(real) + B*sizeof(uint8_t) + B*sizeof(unsigned long long)
        + P*sizeof(real) + Ns_max*sizeof(double) + Ns_max*P*sizeof(double)
        + Ns_max*sizeof(uint8_t);

    // Try allocating thememory, if it fails print issue and throw an error
    try {
        x.alloc(B * (std::size_t)D);
        s.alloc(B * (std::size_t)N);
        t.alloc(B * (std::size_t)N);
        logp.alloc(B);
        valid.alloc(B);
        rng_ctr.alloc(B);
        params.alloc(P);
        E_pool.alloc(Ns_max);
        O_pool.alloc(Ns_max * P);
        valid_pool.alloc(Ns_max);
    } catch (const std::exception& e) {
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::ostringstream oss;
        int dev = -1; cudaGetDevice(&dev);
        oss << "DeviceState: allocation failed.\n  " << e.what()
            << "\n  wanted " << (double)want / (1024.0*1024*1024) << " GiB on device " << dev
            << ", which reports " << (double)free_b / (1024.0*1024*1024) << " GiB free of "
            << (double)total_b / (1024.0*1024*1024) << " GiB."
            << "\n  On a shared machine another job may hold the card; set VMC_CUDA_DEVICE"
            << " or check nvidia-smi.";
        throw std::runtime_error(oss.str());
    }

    // Print information
    if (verbose) {
        std::printf("DeviceState: B=%zu  P=%zu  Ns_max=%zu  (real = %s)\n",
                    B, P, Ns_max, real_name);
        std::printf("  walkers (x,s,t,logp,valid,rng) %8.3f MiB\n",
                    (double)(x.bytes()+s.bytes()+t.bytes()+logp.bytes()
                             +valid.bytes()+rng_ctr.bytes()) / (1024.0*1024));
        std::printf("  params                         %8.3f MiB\n",
                    (double)params.bytes() / (1024.0*1024));
        std::printf("  E_pool + valid_pool            %8.3f MiB\n",
                    (double)(E_pool.bytes()+valid_pool.bytes()) / (1024.0*1024));
        std::printf("  O_pool                         %8.3f GiB\n",
                    (double)O_pool.bytes() / (1024.0*1024*1024));
        std::printf("  TOTAL                          %8.3f GiB  (cap %.1f GiB)\n",
                    (double)total_bytes() / (1024.0*1024*1024), o_pool_max_gb);        
    }
}

std::size_t DeviceState::phase43_bytes() const {
    return dets_psi.bytes() + xi_psi.bytes() + S0.bytes() + rank2_ok.bytes() + pair_ij.bytes()
         + ex_active.bytes() + xi_swap.bytes() + rho_swap.bytes() + S_swap.bytes()
         + V_coul.bytes() + V_nuc.bytes() + E_loc.bytes() + valid_loc.bytes();
}

void DeviceState::grow_phase43(bool verbose) {
    const std::size_t per_w = (std::size_t)ex_types * ex_npairs;
    const std::size_t slots_chunk = (std::size_t)ex_walkers * per_w;
    if (ex_walkers < 1 || slots_chunk > rows_max_phase3)
        throw std::runtime_error("DeviceState::grow_phase43: ex_walkers chunk does not fit the Phase 3 ping-pong");

    dets_psi.alloc((std::size_t)K * B);
    xi_psi.alloc(B * (std::size_t)m_feat);
    S0.alloc(B);
    rank2_ok.alloc(B);
    {
        const std::vector<int> pij = ex_pair_table();
        pair_ij.alloc(pij.size());
        pair_ij.up(pij.data(), pij.size());
    }
    ex_active.alloc(B * per_w);
    xi_swap.alloc(slots_chunk * (std::size_t)m_feat);
    rho_swap.alloc(slots_chunk * (std::size_t)K);
    S_swap.alloc(B * per_w);
    V_coul.alloc(B);
    V_nuc.alloc(B);
    E_loc.alloc(B);
    valid_loc.alloc(B);

    const double grand_gb = (double)(total_bytes() + phase3_bytes() + phase33_bytes() + phase4_bytes()
                                     + phase42_bytes() + phase43_bytes()) / (1024.0*1024.0*1024.0);
    if (grand_gb > o_pool_max_gb) {
        std::ostringstream oss;
        oss << "DeviceState::grow_phase43: total would be " << grand_gb << " GiB, over the "
            << o_pool_max_gb << " GiB cap (o_pool_max_gb in constants.h).";
        throw std::runtime_error(oss.str());
    }

    if (verbose) {
        auto mib = [](std::size_t b){ return (double)b/(1024.0*1024.0); };
        std::printf("DeviceState::grow_phase43: npairs=%d  slot chunk=%d walkers (%zu rows)\n",
                    ex_npairs, ex_walkers, slots_chunk);
        std::printf("  dets_psi + xi_psi + S0          %8.3f MiB\n", mib(dets_psi.bytes()+xi_psi.bytes()+S0.bytes()));
        std::printf("  xi_swap + rho_swap (chunked)    %8.3f MiB\n", mib(xi_swap.bytes()+rho_swap.bytes()));
        std::printf("  S_swap + ex_active (full batch) %8.3f MiB\n", mib(S_swap.bytes()+ex_active.bytes()));
        std::printf("  PHASE 4.3 ADDED                 %8.3f MiB\n", mib(phase43_bytes()));
        std::printf("  GRAND TOTAL                     %8.3f GiB  (cap %.1f GiB)\n", grand_gb, o_pool_max_gb);
    }
}


std::size_t DeviceState::phase3_bytes() const {
    return x_sh.bytes() + feat_in.bytes() + h_out.bytes() + xi.bytes()
         + rho_out.bytes() + orb_out.bytes() + act_a.bytes() + act_b.bytes()
         + lu_ptrs.bytes() + lu_info.bytes() + lu_piv.bytes()
         + M_batch.bytes() + dets.bytes() + S.bytes();
}

// Build networks
void DeviceState::grow_phase3(const Ansatz& a, bool verbose) {
    const std::size_t n_h = a.h_net.params.size();
    const std::size_t n_rho = a.rho_net.params.size();
    h_net_d.build(a.h_net, 0);
    rho_net_d.build(a.rho_net, n_h);
    orb_net_d.build(a.orb_net, n_h + n_rho);

    const int max_width = std::max(std::max(h_net_d.max_width, rho_net_d.max_width), orb_net_d.max_width);
    const int hidden_width = std::max(1, std::max(std::max(h_net_d.hidden_width, rho_net_d.hidden_width), orb_net_d.hidden_width));

    const std::size_t rows = rows_max_phase3;

    // Allocate memory
    x_sh.alloc(B * (std::size_t)D);
    feat_in.alloc(rows * (std::size_t)(dim + 2));
    h_out.alloc(rows * (std::size_t)m_feat);
    xi.alloc(B * (std::size_t)m_feat);
    rho_out.alloc(B * (std::size_t)K);
    orb_out.alloc(rows * (std::size_t)(K * N));    
    act_a.alloc(rows * (std::size_t)hidden_width);
    act_b.alloc(rows * (std::size_t)hidden_width);
    lu_ptrs.alloc((std::size_t)K * B);
    lu_info.alloc((std::size_t)K * B);    
    lu_piv.alloc((std::size_t)N * K * B);
    M_batch.alloc((std::size_t)K * B * N * N);
    dets.alloc((std::size_t)K * B);
    S.alloc(B);

    // Upload pointers, variables declared in here are destroyed upon closing
    {
        std::vector<double*> h_ptrs((std::size_t)K * B);
        for (std::size_t m = 0; m < h_ptrs.size(); m++) h_ptrs[m] = (double*)(M_batch.d + m * (std::size_t)N * N);
        lu_ptrs.up(h_ptrs.data(), h_ptrs.size());
    }

    if (verbose) {
        std::printf("DeviceState::grow_phase3: rows_max=%zu  hidden_width=%d\n", rows, hidden_width);
        std::printf("  x_sh + feat_in + xi + rho_out  %8.3f MiB\n",
                    (double)(x_sh.bytes()+feat_in.bytes()+xi.bytes()+rho_out.bytes())/(1024.0*1024));
        std::printf("  h_out                          %8.3f MiB\n", (double)h_out.bytes()/(1024.0*1024));
        std::printf("  orb_out                        %8.3f MiB\n", (double)orb_out.bytes()/(1024.0*1024));
        std::printf("  activation ping-pong (x2)      %8.3f MiB\n",
                    (double)(act_a.bytes()+act_b.bytes())/(1024.0*1024));
        std::printf("  M_batch + dets + S             %8.3f MiB\n",
                    (double)(M_batch.bytes()+dets.bytes()+S.bytes())/(1024.0*1024));
        std::printf("  batched-LU ptr/piv/info        %8.3f MiB\n",
                    (double)(lu_ptrs.bytes()+lu_info.bytes()+lu_piv.bytes())/(1024.0*1024));
        std::printf("  PHASE 3 ADDED                  %8.3f GiB\n",
                    (double)phase3_bytes()/(1024.0*1024*1024));
        std::printf("  TOTAL                          %8.3f GiB  (cap %.1f GiB)\n",
                    (double)(total_bytes()+phase3_bytes())/(1024.0*1024*1024), o_pool_max_gb);
    }
}


// Upload params to GPU
void DeviceState::upload_params(const Ansatz& a, PinnedArray& staging) {
    staging.ensure(P * (sizeof(double) + sizeof(real)));
    double* hd = staging.as<double>();
    a.copy_params_flat(hd);
    real* hr = reinterpret_cast<real*>(hd + P);
    convert_copy(hr, hd, P);
    params.up(hr, P);
}

// Upload walkers to GPU
void DeviceState::upload_walkers(const WalkerBatch& wb, PinnedArray& staging) {
    if ((std::size_t)wb.B > B) throw std::runtime_error("upload_walkers: WalkerBatch::B exceeds DeviceState::B");
    const std::size_t nb = (std::size_t)wb.B;

    staging.ensure(nb * (std::size_t)D * sizeof(real));
    real*    hr = staging.as<real>();
    uint8_t* hb = staging.as<uint8_t>();

    convert_copy(hr, wb.x.data(),    nb * (std::size_t)D); x.up(hr, nb * (std::size_t)D);
    convert_copy(hr, wb.s.data(),    nb * (std::size_t)N); s.up(hr, nb * (std::size_t)N);
    convert_copy(hr, wb.t.data(),    nb * (std::size_t)N); t.up(hr, nb * (std::size_t)N);
    convert_copy(hr, wb.logp.data(), nb);                  logp.up(hr, nb);
    std::copy(wb.valid.begin(), wb.valid.end(), hb);       valid.up(hb, nb);

    rng_ctr.zero();
}

// Download walkers from GPU
void DeviceState::download_walkers(WalkerBatch& wb, PinnedArray& staging) {
    if ((std::size_t)wb.B > B) throw std::runtime_error("download_walkers: WalkerBatch::B exceeds DeviceState::B");
    const std::size_t nb = (std::size_t)wb.B;

    staging.ensure(nb * (std::size_t)D * sizeof(real));
    real*    hr = staging.as<real>();
    uint8_t* hb = staging.as<uint8_t>();

    x.down(hr, nb * (std::size_t)D);   convert_copy(wb.x.data(),    hr, nb * (std::size_t)D);
    s.down(hr, nb * (std::size_t)N);   convert_copy(wb.s.data(),    hr, nb * (std::size_t)N);
    t.down(hr, nb * (std::size_t)N);   convert_copy(wb.t.data(),    hr, nb * (std::size_t)N);
    logp.down(hr, nb);                 convert_copy(wb.logp.data(), hr, nb);
    valid.down(hb, nb);                std::copy(hb, hb + nb, wb.valid.begin());
}

std::size_t DeviceState::phase33_bytes() const {
    return x_prop.bytes() + logp_prop.bytes() + S_prop.bytes() + S_cur.bytes()
         + prop_idx.bytes() + s_prop.bytes() + t_prop.bytes()
         + pick_a.bytes() + pick_b.bytes()
         + acc.bytes() + sp_acc.bytes() + tau_acc.bytes();
}

void DeviceState::grow_phase33(bool verbose) {
    x_prop.alloc(B * (std::size_t)D);
    logp_prop.alloc(B);
    S_prop.alloc(B);
    S_cur.alloc(B);
    prop_idx.alloc(B);
    s_prop.alloc(B * (std::size_t)N);
    t_prop.alloc(B * (std::size_t)N);
    pick_a.alloc(B);
    pick_b.alloc(B);
    acc.alloc(B);      acc.zero();
    sp_acc.alloc(B);   sp_acc.zero();
    tau_acc.alloc(B);  tau_acc.zero();

    if (verbose) {
        std::printf("DeviceState::grow_phase33: sampler state %8.3f MiB  (TOTAL %.3f GiB)\n",
                    (double)phase33_bytes()/(1024.0*1024),
                    (double)(total_bytes()+phase3_bytes()+phase33_bytes())/(1024.0*1024*1024));
    }
}

// Jet network size
std::size_t DeviceState::phase4_bytes() const {
    return jet_feat.bytes() + jet_h.bytes() + jet_xi.bytes() + jet_rho.bytes()
         + jet_orb.bytes() + jet_a.bytes() + jet_b.bytes();
}

// 
void DeviceState::grow_phase4(bool verbose) {
    const std::size_t C = (std::size_t)jet_C;
    const std::size_t rows = jet_rows; 
    const std::size_t W = (std::size_t)jet_walkers;

    // Take the largest hidden width
    const int hidden = std::max(1, std::max(std::max(h_net_d.hidden_width, rho_net_d.hidden_width), orb_net_d.hidden_width));

    // Measure memory
    const std::size_t n_feat = C * rows * (std::size_t)(dim + 2);
    const std::size_t n_h = C * rows * (std::size_t)m_feat;
    const std::size_t n_xi = C * W * (std::size_t)m_feat;
    const std::size_t n_rho = C * W * (std::size_t)K;
    const std::size_t n_orb = C * rows * (std::size_t)(K * N);
    const std::size_t n_pp = C * rows * (std::size_t)hidden;
    const std::size_t want = (n_feat + n_h + n_xi + n_rho + n_orb + 2*n_pp) * sizeof(real);

    const double grand_gb = (double)(total_bytes() + phase3_bytes() + phase33_bytes() + want) / (1024.0*1024.0*1024.0);

    // Throw error if memory exceeds limit on allocation
    if (grand_gb > o_pool_max_gb) {
        std::ostringstream oss;
        oss << "DeviceState::grow_phase4: total would be " << grand_gb << " GiB, over the "
            << o_pool_max_gb << " GiB cap (o_pool_max_gb in constants.h).\n"
            << "  Phase 4 alone wants " << (double)want/(1024.0*1024*1024) << " GiB at jet_chunk="
            << jet_chunk << " (" << jet_walkers << " walkers).\n"
            << "  REMEDY: set jet_chunk in constants.h to a fraction of n_walkers ("
            << n_walkers << "). The jet pipeline is walker-independent, so chunking is\n"
            << "  exactly equivalent -- only GEMM row counts change. jet_chunk="
            << (n_walkers/2) << " roughly halves the figure above.";
        throw std::runtime_error(oss.str());
    }

    // Allocate memory
    jet_feat.alloc(n_feat);
    jet_h.alloc(n_h);
    jet_xi.alloc(n_xi);
    jet_rho.alloc(n_rho);
    jet_orb.alloc(n_orb);
    jet_a.alloc(n_pp);
    jet_b.alloc(n_pp);

    if (verbose) {
        auto mib = [](std::size_t b){ return (double)b/(1024.0*1024.0); };
        std::printf("DeviceState::grow_phase4: C=%d  jet_walkers=%d  rows=%zu  hidden=%d\n",
                    jet_C, jet_walkers, rows, hidden);
        std::printf("  jet_feat                       %8.3f MiB\n", mib(jet_feat.bytes()));
        std::printf("  jet_h                          %8.3f MiB\n", mib(jet_h.bytes()));
        std::printf("  jet_xi + jet_rho               %8.3f MiB\n",
                    mib(jet_xi.bytes()+jet_rho.bytes()));
        std::printf("  jet_orb                        %8.3f MiB\n", mib(jet_orb.bytes()));
        std::printf("  jet ping-pong (x2)             %8.3f MiB\n",
                    mib(jet_a.bytes()+jet_b.bytes()));
        std::printf("  PHASE 4 ADDED                  %8.3f GiB\n",
                    (double)phase4_bytes()/(1024.0*1024*1024));
        std::printf("  GRAND TOTAL                    %8.3f GiB  (cap %.1f GiB)\n",
                    grand_gb, o_pool_max_gb);
    }
}

std::size_t DeviceState::phase42_bytes() const {
    return Minv_batch.bytes() + inv_ptrs.bytes() + inv_info.bytes()
         + jet_det.bytes() + jet_psi.bytes() + S_jet_v.bytes() + psi_dbl.bytes()
         + E_kin.bytes() + l2_out.bytes() + v3n_out.bytes() + valid_jet.bytes();
}

// Memory allocation
void DeviceState::grow_phase42(bool verbose) {
    const std::size_t C = (std::size_t)jet_C;
    const std::size_t W = B;

    Minv_batch.alloc((std::size_t)K * B * N * N);
    inv_ptrs.alloc((std::size_t)K * B);
    inv_info.alloc((std::size_t)K * B);
    {
        std::vector<double*> h_ptrs((std::size_t)K * B);
        for (std::size_t m = 0; m < h_ptrs.size(); m++) h_ptrs[m] = (double*)(Minv_batch.d + m * (std::size_t)N * N);
        inv_ptrs.up(h_ptrs.data(), h_ptrs.size());
    }

    jet_det.alloc(C * W * (std::size_t)K);
    jet_psi.alloc(C * W);
    S_jet_v.alloc(W);
    psi_dbl.alloc(W);
    E_kin.alloc(W);
    l2_out.alloc(W);
    v3n_out.alloc(W);
    valid_jet.alloc(W);

    const double grand_gb = (double)(total_bytes() + phase3_bytes() + phase33_bytes()
                                     + phase4_bytes() + phase42_bytes()) / (1024.0*1024.0*1024.0);
    if (grand_gb > o_pool_max_gb) {
        std::ostringstream oss;
        oss << "DeviceState::grow_phase42: total would be " << grand_gb << " GiB, over the "
            << o_pool_max_gb << " GiB cap (o_pool_max_gb in constants.h).";
        throw std::runtime_error(oss.str());
    }

    if (verbose) {
        auto mib = [](std::size_t b){ return (double)b/(1024.0*1024.0); };
        std::printf("DeviceState::grow_phase42:\n");
        std::printf("  Minv_batch (RESIDENT for Ph5) %8.3f MiB\n", mib(Minv_batch.bytes()));
        std::printf("  getri ptr/info                %8.3f MiB\n",
                    mib(inv_ptrs.bytes()+inv_info.bytes()));
        std::printf("  jet_det                       %8.3f MiB\n", mib(jet_det.bytes()));
        std::printf("  jet_psi + per-walker outputs  %8.3f MiB\n",
                    mib(jet_psi.bytes()+S_jet_v.bytes()+psi_dbl.bytes()
                        +E_kin.bytes()+l2_out.bytes()+v3n_out.bytes()+valid_jet.bytes()));
        std::printf("  PHASE 4.2 ADDED               %8.3f MiB\n", mib(phase42_bytes()));
        std::printf("  GRAND TOTAL                   %8.3f GiB  (cap %.1f GiB)\n",
                    grand_gb, o_pool_max_gb);
    }
}