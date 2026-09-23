#pragma once

#include "../constants.h"
#include "../physics.h"
#include "../walkers.h"
#include "../precision.h"
#include "gpu_util.h"
#include "layouts.h"
#include "net_forward.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Graph object, instead of launching on device seperately every time tell it at once to launch
struct SweepGraphs {
    bool            enabled = use_cuda_graphs;   
    cudaStream_t    stream  = nullptr;           
    DeviceArray<unsigned char> blas_ws;          
    cudaGraphExec_t coord = nullptr, spin = nullptr, tau = nullptr;
    int             coord_B = -1, disc_B = -1;
    double          coord_step = 0.0;
    std::size_t     coord_nodes = 0, spin_nodes = 0, tau_nodes = 0;
    long long       n_captures = 0, n_updates = 0;

    SweepGraphs() = default;
    SweepGraphs(const SweepGraphs&) = delete;
    SweepGraphs& operator=(const SweepGraphs&) = delete;
    ~SweepGraphs();                              
};

struct DeviceState {
    SweepGraphs graphs;

    DeviceArray<real> x;         // B*N    
    DeviceArray<real> s;         // B*N
    DeviceArray<real> t;         // B*N
    DeviceArray<real> logp;      // B
    DeviceArray<uint8_t> valid;  // B
    
    DeviceArray<unsigned long long> rng_ctr;  // B
    
    DeviceArray<real> params;  //P, flat w/ [h_net.params, rho_net.params, orb_net.paramas, alpha]

    DeviceArray<double> E_pool;       // Ns_max
    DeviceArray<opool_t> O_pool;
    DeviceArray<double>  O_stage;     // fp32_opool: opool_stage_rows * P FP64 rows before the cast

    // Float version of params
    DeviceArray<float> params_f;                 // P
    DeviceArray<float> fwd_in_f, fwd_out_f;      // value chain: cast-down input, output layer
    DeviceArray<float> jet_in_f, jet_out_f;      // jet chain: same

    DeviceArray<uint8_t> valid_pool;  // Ns_max
    
    std::size_t B = 0, P = 0, Ns_max = 0;

    DeviceArray<real> x_sh;          // B*D
    DeviceArray<real> feat_in;       // rows_max x (dim+2)
    DeviceArray<real> h_out;         // rows_max x m_feat
    DeviceArray<real> xi;            // B x m_feat
    DeviceArray<real> rho_out;       // B x K
    DeviceArray<real> orb_out;       // rows_max x (K*N)
    DeviceArray<real> act_a, act_b;  // ping-pong, rows_max x max_width

    DeviceArray<double*> lu_ptrs;  // K*B
    DeviceArray<int> lu_info;      // K*B

    DeviceArray<int> lu_piv;    // N*K*B  (getrfBatched pivots, 1-BASED)
    DeviceArray<real> M_batch;  // K*B*N*N, fixed stride
    DeviceArray<real> dets;     // K*B
    DeviceArray<real> S;        // B, the Slater sum sum_k rho_k det_k    

    DeviceArray<real> x_prop;     // B*D, one perturbed coordinate
    DeviceArray<real> logp_prop;  // B
    DeviceArray<real> S_prop;     // B
    DeviceArray<real> S_cur;      // B, current S across the discrete block
    DeviceArray<int> prop_idx;    // B, which coordinate was proposed
    DeviceArray<real> s_prop;     // B*N
    DeviceArray<real> t_prop;     // B*N
    DeviceArray<int> pick_a;      // B, up/proton index chosen
    DeviceArray<int> pick_b;      // B, down/neutron index chosen
    DeviceArray<long long> acc, sp_acc, tau_acc;  // B each

    DeviceArray<real> jet_feat;     // C * jet_rows * (dim+2)
    DeviceArray<real> jet_h;        // C * jet_rows * m_feat
    DeviceArray<real> jet_xi;       // C * jet_walkers * m_feat
    DeviceArray<real> jet_rho;      // C * jet_walkers * K
    DeviceArray<real> jet_orb;      // C * jet_rows * (K*N)   <- the big one
    DeviceArray<real> jet_a, jet_b; // ping-pong, C * jet_rows * hidden_width
    
    DeviceArray<real> Minv_batch;    // K*B*N*N, fixed stride, row-major inverses
    DeviceArray<double*> inv_ptrs;   // K*B, getriBatched output pointers
    DeviceArray<int> inv_info;       // K*B

    DeviceArray<real> jet_det;       // C * B * K
    DeviceArray<real> jet_psi;       // C * B
    DeviceArray<real> S_jet_v;       // B, S_jet.v (oracle + 4.3)
    DeviceArray<real> psi_dbl;       // B, env*S from the double path
    DeviceArray<real> E_kin;         // B
    DeviceArray<real> l2_out;        // B
    DeviceArray<real> v3n_out;       // B
    DeviceArray<uint8_t> valid_jet;  // B

    DeviceArray<real> dets_psi;          // K*B
    DeviceArray<real> xi_psi;            // B*m_feat
    DeviceArray<real> S0;                // B, S_from_table(s,t), local_E's ratio denominator
    DeviceArray<uint8_t> rank2_ok;       // B, rank2_well_conditioned per walker
    DeviceArray<int> pair_ij;            // 2*ex_npairs
    DeviceArray<uint8_t> ex_active;      // B*3*npairs, dense slots
    DeviceArray<real> xi_swap;           // ex_walkers*3*npairs * m_feat  (chunked)
    DeviceArray<real> rho_swap;          // ex_walkers*3*npairs * K       (chunked)
    DeviceArray<real> S_swap;            // B*3*npairs, S' per slot (full batch)
    DeviceArray<real> V_coul;            // B
    DeviceArray<real> V_nuc;             // B
    DeviceArray<real> E_loc;             // B
    DeviceArray<uint8_t> valid_loc;      // B, valid_jet && isfinite(E_loc)

    NetCache cache_h, cache_rho, cache_orb;
    DeviceArray<real> bp_a, bp_b;     // backprop delta ping-pong, B*N x widest layer
    DeviceArray<real> dpsi_dxi;       // B x m_feat, rho's input gradient = the h seed
    DeviceArray<real> bp_wt;          // largest in_w*out_w, one layer's transposed weights
    
    DeviceArray<double> mask_d, E_clip_d, t_ns_d;                   // Ns_max
    DeviceArray<double> O_exp_d, S_diag_d, grad_d, v_rms_d, d_rms_d, M_inv_d;
    DeviceArray<double> cg_r, cg_z, cg_p, cg_Ap, delta_d, S_delta_d;  // P

    DeviceArray<double> Ew_d, E2w_d, l2w_d, r2w_d;   // B
    DeviceArray<int> nw_d;                           // B
    DeviceArray<unsigned char> pack_d;               // Ns_max*9 + B*60 bytes

    void grow_phase53(bool verbose = true);
    std::size_t phase53_bytes() const;

    void grow_phase52(bool verbose = true);
    std::size_t phase52_bytes() const;

    void grow_phase5(const Ansatz& a, bool verbose = true);
    std::size_t phase5_bytes() const;

    void grow_phase43(bool verbose = true);
    std::size_t phase43_bytes() const;

    void grow_phase42(bool verbose = true);
    std::size_t phase42_bytes() const;
    
    void grow_phase33(bool verbose = true);
    std::size_t phase33_bytes() const;
    
    void grow_phase4(bool verbose = true);
    std::size_t phase4_bytes() const;

    DeviceNet h_net_d, rho_net_d, orb_net_d;

    explicit DeviceState(const Ansatz& a, bool verbose = true);

    std::size_t total_bytes() const;

    std::size_t phase3_bytes() const;
    void grow_phase3(const Ansatz& a, bool verbose = true);

    void upload_params(const Ansatz& a, PinnedArray& staging);

    void upload_walkers(const WalkerBatch& wb, PinnedArray& staging);
    void download_walkers( WalkerBatch& wb, PinnedArray& staging);
};

std::size_t check_param_layout(const Ansatz& a);