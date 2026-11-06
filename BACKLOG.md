# Backlog: deferred work and known inefficiencies

Started 2026-09-24, reviewed 2026-11-05 at the end of the clean-up. "scratch tree X" = /tmp/claude-1006/.../scratchpad/X
(not in the repo). Closed items move to the table at the end, with the commit that closed them.

## Deferred (decided to do later)

| # | Item | Status / where | Notes |
|---|---|---|---|
| D2 | Sample-size curriculum | proposed 2026-09-24 | small B + faster lambda decay early, full B later; fix I10 (lambda reset on resume) first, it is most of the win |
| D3 | Jastrow basis width scan | tabled 2026-09-24 | widths are hand-chosen, ~even-tempered 0.3*R*1.57^k |
| D4 | Trainable Jastrow widths | shelved 2026-09-24; scratch tree `tw` | unbounded + Adam ran away (widths x3-10); bounded (tanh, e^+-1) + SR only: no clear gain on Li6 {32} (-30.645+-0.013 var 203 vs fixed -30.625+-0.007 var 62); Jastrow-only fine-tune on converged networks (2026-09-24, scratch tree `jonly`, `train_nets=false`): -0.02 to -0.04 MeV (Li6 -30.850, Li7 -37.031, Be7 -35.543) but variance up for Li6 (31->69) and Li7 (124->154) |
| D5 | Network width/depth scan | partly done | Li6: {32} -30.625, {32,32,32} -30.657, {64,64,64} -30.778 (final checkpoints, same recipe); Li7/Be7: {64} ~0.3-0.4 MeV below {32} at equal iterations |
| D6 | Complex wavefunction | only if needed | cannot lower E for model o (real H); may help L>0 nuclei via lighter E_loc tails. Test first: Li7 E_loc histogram, power-law tail? |
| D7 | Heavier nuclei (C11, C12, O16) | deferred | need FP64 (fp32 forward: ~7% det errors at N=12); cold start dissolves (C12 r_rms 9 fm); warm start now possible via transfer (scratch tree `xfer`); det_jet_kernel shared memory caps N<=13, O16 needs dynamic smem |
| D8 | Tests with a nonzero Jastrow | open | test_psi_jet_swap and test_record's legacy exchange loop lack exp(dJ); test_local_e's own exchange oracle too; psi_double tolerance should be normalised by e^J; test_sr_device tolerance with width columns; test_jet42 bounds at N=12 |
| D9 | Port scratch features to the repo | partly done | seeded init is in (`init_seed`); the run file's `[network]` and `transfer` replace VMC_ARCH and VMC_TRANSFER; still open: fixed `m_feat` (N_feat) and partial rho_net transfer (scratch tree `xfer`), needed for transfer between nuclei (I12) |
| D10 | Value-only Jastrow basis | only if profiled | `jas_basis<Deriv>` split; jastrow_dlabel could loop only over pairs touching the swapped particles |
| D11 | Li7/Be7 envelope softening and transfer chain | in progress | {64} pipelines running; their checkpoints are the transfer sources for heavier nuclei |
| D12 | Pair-aware ansatz (message-passing backflow or Pfaffian pair orbitals) | proposed 2026-09-24 | reference paper (arXiv 2607.09223) reaches ⟨L²⟩(Li7)=2.0012 with 11.0k/13.7k params (MLPs 2x16, tanh), fixed across A up to 58Ni, via pf[phi(x_i,x_j)] + one MPNN layer; our DeepSets of single-particle orbitals needs width instead ({32} short for Li7/Be7) and grows with N (K*N outputs, m_feat) |
| D13 | Rotational invariance by rotating each configuration to a canonical frame before the network | rough idea 2026-09-24 | model o has no tensor/LS force, so spatial rotation alone is a symmetry; pure invariance only fits L=0 (He4, Li6), L>0 (Li7, Be7, 17O) needs Wigner-D equivariance; principal-axis frames jump where eigenvalues are degenerate or eigenvector signs flip, making psi non-smooth (kinetic-energy spikes); smooth alternative: invariant pair features (overlaps D12); check first how much energy L² contamination actually costs |
| D14 | Grow-transfer (wider or deeper network from a trained checkpoint) | ready 2026-09-24, scratch tree `grow` | `VMC_TRANSFER_GROW=wide` (exact: duplicated units, random outgoing split) or `deep` (inserted 64-wide layer GELU(200(a+0.2)), exact to 1e-9); verified CPU psi to 4e-12/1e-9 and GPU start energies; widening raises memory (Li7 {96} at 5800 walkers exceeds the 12 GiB O_pool cap); to port to the repo with D9 |
| D15 | Does the FP64 refine stage beat simply training longer in fp32? | open 2026-09-25 | C11 {64}: -61.46 -> -62.33, C12 {64}: -81.39 -> -81.75 after 500 FP64 its, but the fp32 runs were still improving at a similar rate; controlled test: from one checkpoint run 500 fp32 vs 500 FP64 its, FP64-evaluate both; fp32 energy estimates are biased low (+0.08 MeV for {64}, ~+0.25 for a deep-grown {64,64} with its x200 inserted layer) |

## Inefficiencies / problems found

| # | Problem | Impact | Possible fix |
|---|---|---|---|
| I1 | SR's single trust region across ~20-50k params freezes alpha and the Jastrow (delta_alpha ~1e-4/iter) | envelope never adapts | separate step for the envelope block (done: env_adam_*); or block-wise trust region |
| I2 | best_checkpoint = lowest E + E_err iterate | picks lucky early iterates (Li7 baseline: -28.29 vs final -34.24) | evaluate the final/periodic checkpoint (pipeline does); or select on a smoothed window |
| I6 | SR/CG dominates at the lambda floor (CG 80-120 iters) | 40-70% of iteration time late in training | better preconditioner, higher lambda floor, or fewer CG iters with warm start; CG warm start from the uncapped previous solution done (Phase 6 item 3: -5 to -9% CG iterations at low lambda), preconditioner still open |
| I8 | fp32_forward inadequate for N >= 12 | 7% det errors at N=12 | FP64 for heavier nuclei, or mixed precision in the orbital GEMM only |
| I9 | det_jet_kernel limited to D+1 <= 32 (repo, fixed in handover); static smem limits N <= 13 | blocks O16 | dynamic shared memory (scratch `big`: N <= 16 in double after dropping the SELFCHECK-only sgmag buffer; N >= 17 needs sB[D][N*N] tiled or moved to global memory) |
| I12 | load_transfer assumed m_feat and K independent of N | transfer between nuclei impossible | fixed m_feat + partial rho_net (done in scratch `xfer`) |
| I14 | Two processes on one GPU slow each other ~3x (not 2x) | many small kernels + host syncs | fewer syncs (I5); CUDA graphs cover part; the alpha syncs (I5) are gone, not re-measured since |
| I15 | Jastrow basis computes derivatives when unused | negligible now | see D10 |
| I16 | Transfer-learning comment/assumption in load_transfer was wrong | misleading | fixed in scratch `xfer` |
| I17 | SR drifts uphill at the lambda floor when P/Ns is large (C12: 73k params, 8000 samples, floor 0.4 -> E rose from -75.6 to -73.8 over ~500 its while r_rms grew) | wasted hours, worse start for later stages | raise the lambda floor (2.0 used for C11/C12), more walkers, or stop on a rising smoothed energy and fall back to the best checkpoint automatically |
| I18 | Device memory grows ~15% per added nucleon at fixed walkers (C13 15.5 GiB, C14 18.2 GiB at 4000 walkers, fp32) | O16 needs fewer walkers (3200) on a 24 GB card, raising P/Ns | shrink phase-3/4 buffers, chunk more, or multi-GPU (D1) |
| I19 | CG reads every dot product back to the host | ~340 scalar reads per Li6 iteration (~15 us each), ~0.4% of an iteration; less for heavier nuclei | cuBLAS device pointer mode, alpha/beta in a small kernel, a convergence check every few iterations |
| I20 | `o_finalize_kernel` computes `jastrow_O` on thread 0 of each block | ~1.7% of a Li6 run (nsys, 2026-11) | spread the Jastrow features over the block's threads |
| I21 | The size mode (r2) decorrelates slowly: tau ~ 20-25 sweeps with whole-particle moves, ~10 with single-coordinate moves | records 3 sweeps apart are correlated in r2 (rho ~ 0.7); no sweep count fixes it cheaply | an occasional global scaling move (x -> lambda x with the Jacobian), or a larger step only for the centre-of-mass-free breathing mode |
| I22 | Master parameters live on the host; each iteration downloads delta and uploads the parameters to every GPU | ~180 KB each way per Li6 iteration, ~0.003%; the two 3090s here cannot do peer-to-peer | only worth it on one large GPU together with I19 |

## Closed

| # | Item | Resolution |
|---|---|---|
| D1 | Multi-GPU walker sharding (final phase prompt) | Phase 4 (5253130): one training loop with a GPU backend that runs 1..G replicas; the planner sizes walkers per GPU. Bit-identity across G was not pursued; golden `gpu2` and `mg1` cover it |
| I3 | O_pool sized for the grown batch from the start | Phase 6 item 6 (c612d01): O pool allocated for the current record count and grown at batch growth; Li6 4.4 -> 3.3 GiB before growth, frozen_eval allocates none |
| I4 | Full-batch jet buffers (jet_chunk = 0) | Phase 6 item 8 (c612d01): jet buffers hold min(walkers, `[gpu] jet_chunk`); 640 walkers 2.41 -> 0.43 GiB |
| I5 | alpha downloaded with a blocking sync 3x per chunk (compose, validity, o_finalize) | Phase 6 item 1 (c612d01): compose, validity and o_finalize read alpha through a device pointer |
| I7 | Single-coordinate Metropolis: D full-batch evaluations per sweep | Phase 6 item 4 (c612d01): whole-particle moves; Li6 training -28% wall time at equal energy |
| I10 | lambda restarts at sr_lambda0 = 100 on every resume | Phase 5 (eaa2d91): checkpoint v2 stores iter, lambda and best E_ucb; resume continues the schedule |
| I11 | Unseeded network init (std::random_device) | `init_seed` in constants.h seeds the network initialisation |
| I13 | main aborts at the end when no best_checkpoint exists | Phase 4 (5253130): main evaluates the final parameters; no best checkpoint is needed |
