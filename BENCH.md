# VMC Validation Bench

## Config per row

| # | N | (N_up,N_down) | det_mode | spin_mode | int_kind | g_int | r_soft | spin_int_on | J_ss / J_range | omega2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | (1,1) | Full  | Fixed   | None    | -   | -   | false | -        | 1.0  |
| 2 | 2 | (1,1) | Block | Fixed   | None    | -   | -   | false | -        | 1.0  |
| 3 | 4 | (2,2) | Full  | Fixed   | None    | -   | -   | false | -        | 1.0  |
| 4 | 4 | (2,2) | Block | Sampled | None    | -   | -   | false | -        | 1.0  |
| 5 | 2 | (1,1) | either | either | Coulomb | 1.0 | 0.0 | false | -        | 0.25 |
| 6 | 2 | (1,1) | either | Sampled | None    | -   | -   | true  | 0.1 / 0.0 | 1.0  |
| 7a| 4 | (2,2) | Full  | Fixed   | Coulomb | 1.0 | 0.0 | false | -        | 1.0  |
| 7b| 4 | (2,2) | Block | Fixed   | Coulomb | 1.0 | 0.0 | false | -        | 1.0  |

## Results

| config | E_final | E_err | var | iters to converge | ms/iter | spin_acceptance | node_hits |
|---|---|---|---|---|---|---|---|
| 1: N=2 Full Fixed, no int    | (target 3.0) | | | | | n/a | |
| 2: N=2 Block Fixed, no int   | (target 3.0) | | | | | n/a | |
| 3: N=4 Full Fixed, no int    | (target 8.0) | | | | | n/a | |
| 4: N=4 Block Sampled, no int | (target 8.0) | | | | | | |
| 5: N=2 harmonium (Coulomb)   | (target 2.0) | | | | | | |
| 6: N=2 J=0.1 constant sigma.sigma | (target 2.7) | | | | | | |
| 7a: N=4 Coulomb, Full        | (no exact value) | | | | | n/a | |
| 7b: N=4 Coulomb, Block       | (no exact value) | | | | | n/a | |

## Phase 3.2: determinants, S, log|psi|, and the (s,t) table

### Correctness (tests_gpu)

| check | measured | bound |
|---|---|---|
| Slater assembly vs CPU ws.dM (elementwise) | 6.2e-15 | 1e-13 |
| batched_det vs lu_det, 7936 matrices | median 4.6e-16, p99.9 5.5e-13, max 3.9e-12 | 1e-14 / 1e-12 / 1e-10 |
| log\|psi\| vs CPU log_p, 512 configs | worst abs 2.1e-13 | 1e-11 |
| S_from_table vs CPU, 512 x 5 relabelings | median 1.5e-15, p99.9 5.6e-13, max 1.1e-12 | 1e-14 / 1e-11 / 1e-9 |
| forced-node configs suppressed | 14 of 14 | all |
| S = 0 injected -> -INFINITY | 512 of 512 | all |

Determinant and S bounds are stated as median/p99.9/max rather than a single
tight number: a determinant is a PRODUCT of N pivots and cuBLAS does not choose
the same pivots as lu_det, so the tail reflects matrix conditioning, not
disagreement. Every realistic bug (index mapping, pivot parity) moves the
MEDIAN, which is bounded at 1e-14.

Negative controls, both run: flipping the pivot convention to `ipiv[i] != i`
fails 236 of 512 permutation determinants; transposing the assembly is caught
at 21.27 vs 6.2e-15.

### Timing -- CONTENDED, see caveat

Both RTX 3090s were at 100% utilisation with other users' jobs when these were
taken. Measured directly: the 3.1 forward stages, which took **2.71 ms on an
idle card**, took **23.7 ms** here -- a **8.7x contention factor**.

| stage | contended | note |
|---|---|---|
| forward stages only | 23.7 ms | 2.71 ms when idle |
| eval_logpsi_batch (full) | 24.3 - 32.8 ms | 4 repeats |
| of which batched LU + det | 2.07 - 2.69 ms | stable across all runs |
| build_st_table_batch | 21.0 - 27.4 ms | 4x rows |

**The 8% LU share these numbers imply is an artefact.** The LU stage held at
~2.1-2.7 ms regardless of contention while the GEMMs inflated 8.7x, which is
what one expects if the LU is launch-latency-bound on 6x6 matrices and the
GEMMs are SM-bound. Projecting the forwards back to their idle 2.71 ms puts the
full chain near 5.5-6 ms with the LU at roughly **45%** -- which would make the
warp-per-matrix kernel a Phase 6 priority rather than a footnote.

That is a projection, not a measurement. RE-RUN `bench_eval` ON AN IDLE CARD
before acting on it.

**Follow-up (Phase 4.2, idle card):** the full double chain
(`eval_logp_batch`, B = 5800) measured **4.51 - 4.63 ms**, below the 5.5-6 ms
projection. The LU share was not timed separately in that run, so the 45% figure
is still unconfirmed.

## Earlier measurements (Phases 1.5 - 3.1)

- **Phase 1.5, `fma_into`:** jpsi 2.65x faster. `sizeof(Jet) == 160` (20
  contiguous doubles), so the win came from eliminating temporaries, not layout.
- **Hardware reality, RTX 3090:** FP64 throughput 0.295 - 0.47 TFLOP/s vs the CPU's
  1.062; FP32 is 29 TFLOP/s (62x). GA102 has 2 FP64 units per SM against 128 FP32.
- **Phase 3.1 forward stages (h, rho, orb GEMMs):** 2.71 ms on an idle card.

## Phase 3.3: device sampler

### Correctness

| check | measured | bound |
|---|---|---|
| draw-order determinism, 50 sweeps, x/s/t/logp/counters | bitwise equal | exact |
| replay: proposal touches exactly the logged coordinate | 0 violations | exact |
| replay: host reconstruction vs device x | 0 of 9216 differ | exact |
| logp cache drift after 100 sweeps | 9.77e-15 | 1e-9 |
| acceptance coord / spin / tau | 0.812 / 0.385 / 0.380 | non-degenerate |

Negative control: removing the `logp[w] = logp_prop[w]` commit drives the cache
drift to 3.296 while coordinate acceptance still looks healthy at 0.460.

### Timing

- **Per-kernel `cudaDeviceSynchronize` was 90% of sampler runtime:** 1003 ->
  103 ms/sweep once `cuda_sync_check` was moved behind `VMC_CUDA_SYNCCHECK`
  (~400 kernels/sweep at ~2.25 ms each on a time-sliced card).
- After the fix, contended: GPU sampler 246.5 ms/sweep vs CPU 229 ms/sweep.
- Hybrid v1 descent iteration (training.csv, card load unknown): ms_iter
  2964 - 3344, metro 311, record window 2043 - 2428 (sweeps + CPU local_E),
  sr 534 - 570, gpu_ms ~883.

## Phase 4.1: jet forwards

| check | measured | bound |
|---|---|---|
| jet seeds (value / gradient / laplacian) | 0 / 0 / 0 | exact |
| act_hess vs central differences, Gelu / Tanh | 1.58e-10 / 7.75e-11 | 1e-7 |
| jet_bias_act vs apply_activation<Jet>, Gelu (v / g / l) | 2.22e-16 / 6.66e-16 / 3.22e-14 | 1e-12 |
| jet_bias_act, Tanh (v / g / l) | 2.22e-16 / 3.33e-16 / 4.91e-15 | 1e-12 |
| jet oracle vs CPU jpsi (xi / rho / orb) | 1.42e-14 / 8.26e-12 / 7.99e-15 | 1e-11 |

Negative control: dropping the -1/N from the analytic CM-shift seeds moves the
oracle to 4.90 / 274 / 1.00.

Memory: Phase 4 added 3.015 GiB with a jet-valued `jet_M` (988 MiB); 2.050 GiB
after Phase 4.2 removed it.

## Phase 4.2: det jets, psi jet, E_kin / L2 / V_3N (idle card)

### Correctness

| check | measured | bound |
|---|---|---|
| det jet value block vs ds.dets | 0 of 7936 differ | bitwise |
| det jet vs det_jet_from_minv (v / g / l) | 5.54e-12 / 7.36e-11 / 5.98e-11 | 1e-10 |
| psi jet vs CPU jpsi (v / g / l) | 4.58e-13 / 5.23e-12 / 3.41e-13 | 1e-10 |
| E_kin (abs, \|E_kin\| to 5.9e3 MeV) | 5.42e-10 MeV | 1e-9 |
| V_3N (abs) | 4.44e-16 MeV | 1e-9 |
| L2 (relative, \|L2\| to 3.75e4) | 2.58e-13 | 1e-10 |
| validity where the double paths agree on the node | 0 of 383 | exact |
| validity guard branches, driven directly | 0 wrong | exact |
| chunk invariance, whole vs 37-walker chunks | 0 differ | bitwise |

The det-jet value residual is entirely the cuBLAS getrf vs `lu_det` gap. On
deliberately singular walkers cuBLAS reports ~27 of 31 exact-zero pivots where
`lu_det` finds ~18-21, so the device rejects some near-node walkers the CPU keeps.

Negative controls: drop term2 (det l 3.99e3, psi l 8.49); transposed Minv (det g
4.36e4); drop 2*dot in the jet product (psi l 1.65e2 only); jet_det stride from Bc
(chunk invariance only, 141,360 entries).

### Timing, B = 5800, idle RTX 3090

| stage | ms |
|---|---|
| double chain (eval_logp_batch) | 4.51 |
| batched_inverse (getri) | 0.61 |
| jet forwards (4.1) | 54.93 |
| det_jet_assemble | 13.19 |
| compose + kin/l2/v3n/valid | 0.62 |
| **total eval_jet_batch** | **73.86** |
| CPU same work, 20 threads (ideal scaling) | 584.3 (7.91x) |
| CPU full local_E incl. V_nuc, 20 threads (ideal) | 718.2 |

V_nuc was 18.6% of CPU local_E. Phase 4.2 added 80 MiB (grand total 6.528 GiB).

## Phase 4.3: 2-body exchange, full device local_E

### Correctness

| check | measured | bound |
|---|---|---|
| V_nuc vs local_E's exchange loop, rel to summand magnitude | 1.63e-12 | 1e-11 |
| active slots vs ratios the CPU evaluates | 5576 = 5576 | exact |
| forced host fallback, 128 walkers | 4.95e-14 | 1e-11 |
| natural rank-2 gate, 85 coincident walkers | 18 device / 18 CPU | fires |
| E_loc vs CPU local_E, rel to summed terms (512 valid) | 1.43e-13 | 1e-11 |
| per term: E_kin rel / V_3N abs / V_coul abs / V_nuc rel | 5.77e-12 / 1.42e-14 / 8.88e-16 / 1.77e-13 | |
| validity mask vs CPU local_E | 0 of 512 | exact |
| sampler logp untouched by eval_local_E_device | 0 of 512 changed | exact |
| assemble_O vs local_E's O | 0 of 1.46M entries | bitwise |
| two device record passes | 0 differ | bitwise |

Also green rebuilt at N=2 (deuteron) and N=3 (triton). The table pass reproduces
the psi-pass dets bitwise on this card (0 of 179,800), so the dets/xi snapshot is
numerically redundant here and kept only for cuBLAS algorithm independence.

Negative controls: R_s/R_t labels swapped (V_nuc 0.60); same_s branch reads the
wrong slot (0.35); slot activity rule wrong (0.35, counts still equal); fallback
disabled (0.45, 0 of 128 took it); Coulomb dropped (E_loc 1.7e-2); sampler logp
overwritten (512 of 512); snapshot removed (not detectable on this card).

### Timing -- CONTENDED (GPUs at 100% from other users), 2 iterations

| B = 5800, 2 records | ms/iter |
|---|---|
| CPU record_batch, 20 threads | 7507 |
| hybrid v2 | 3108 |
| of which therm / sweeps / device local_E / download / host O | 818 / 1639 / 461 / 0.7 / 189 |

Statistical parity over those 2 iterations: E, E^2, L^2, r^2 and invalid fraction
all within |z| < 1.5. The 30-iteration run was stopped before completing.

## Phase 5.1: device O assembly (value-level backprop)

### Correctness

| check | measured | bound |
|---|---|---|
| act_grad vs CPU, 1e6 points, Gelu / Tanh | 1.33e-15 / 4.44e-16 | 1e-13 |
| strided-batched dW/db oracle | 3.05e-16, 0 entries outside blocks | 1e-13 |
| O with identical inputs, rel to summand magnitude | 4.74e-15 | 1e-12 |
| O end to end vs CPU local_E, rel to row max (near-node / ordinary) | 1.26e-11 / 4.76e-13 | 1e-10 |
| determinism, assemble twice | 0 of 5.47M | bitwise |
| chunk invariance, whole vs 37-walker chunks | 0 of 5.47M | bitwise |

cuBLAS broke chunk invariance for orb's 64x186 delta propagation (algorithm
changes with row count, up to 1.5e-9); the delta-propagation kernel fixed it.

Negative controls: Minv transposed in the orb seed (6.2e4); W instead of W^T in
delta propagation (1.9); h seed to particle 0 only (0.25); no /S (4.5e8); psi pass
without the stash (end-to-end 1.0, identical-input check passes); act_grad skipped.

### Timing -- CONTENDED (GPU 0 at 100%)

| B = 5800, P = 22807 | ms |
|---|---|
| assemble_O_batch (device) | 46.6 |
| host O pass, 20 threads | 123.2 (2.6x) |

Phase 5.1 added 247 MiB (grand total 6.87 GiB).

## Phase 5.2: SR on device

### Correctness

| check | measured | bound |
|---|---|---|
| O_exp / S_diag vs host | 6.00e-16 / 6.28e-15 | 1e-12 |
| gradient, rel to summand magnitude (synthetic / real pool) | 2.90e-16 / 5.06e-16 | 1e-12 |
| RMS update (v_rms / d_rms / mean) | 2.08e-13 / 1.04e-13 / 9.12e-16 | 1e-12 |
| S*v apply, raw / damped | 1.59e-12 / 1.98e-12 | 1e-11 |
| CG, synthetic SPD n=400 tol 1e-10 | iters 127 = 127, solution 1.01e-11 | 1e-8 |
| real pool, production tol: device solution's residual under host operator | 8.56e-4 | < 1e-3 |
| real pool, tol 1e-10: host vs device solutions | 2.36e-10 | 1e-6 |
| two device SR solves | 0 of 22807 delta entries | bitwise |
| sr_rms_damp = false build | all green | |

At production tolerance host and device CG stop at 77 vs 75 iterations and agree
to 4.4e-4: the gap is where CG stops, not arithmetic (tol-1e-10 row).

Negative controls: centering ignores the mask (8.9e4, synthetic pool only); S_diag
does not skip invalid (5.9e6, synthetic only); lambda*S_diag dropped (91, CG 77 ->
200); CG without beta (no convergence); grad sign flip (0.59); O^T y with OP_T (3.5e5).

### Timing -- idle card, real O_pool from best_checkpoint.txt, iter 1500

| Ns | path | O_exp + grad | SR solve | CG iters | ms / CG iter | scalar downloads |
|---|---|---|---|---|---|---|
| 11600 | CPU, 20 threads | 60 | 3185 | 106 | 30.0 | -- |
| 11600 | device | 5 | **546** | 110 | 5.0 | 446 (~8.6 ms) |
| | speedup | 12.6x | **5.8x** | | | |
| 23200 | CPU, 20 threads | 114 | 6701 | 112 | 59.8 | -- |
| 23200 | device | 10 | **1054** | 107 | 9.9 | 434 (~8.3 ms) |
| | speedup | 11.8x | **6.4x** | | | |

Scalar downloads are ~1.6% of the device solve (0.019 ms per dot).

## Phase 5.3: fully device-resident descent (idle RTX 3090, Li6, P = 22807)

### Per-iteration decomposition, B = 5800 (ms), 300-iteration window from best_checkpoint.txt

| stage | CPU 20 thr, rec=2 | GPU rec=2 | CPU rec=4 | GPU rec=4 |
|---|---|---|---|---|
| therm sweeps (3) | 1968 | 317 | 1910 | 318 |
| record sweeps (3/record) | -- | 629 | -- | 1258 |
| local_E | 5230 (incl. O) | 195 | 10096 | 390 |
| O assembly | -- | 38 | -- | 76 |
| SR (CG iters) | 548 (15) | 90 (15) | 1292 (18) | 214 (18) |
| transfers | -- | 0.1 | -- | 0.1 |
| host residue | -- | ~5 | -- | ~6 |
| **total** | **7840** | **1274** | **13448** | **2262** |

Wall time for the 300 iterations: GPU 9 min 38 s, CPU 58 min.
Walker-sweeps/s: GPU ~55,000. Samples/s: GPU 9,105 vs CPU 1,480 (6.2x).
Earlier rows: Phase 0 CPU (Aug 28, run5) ~8.8 s/iter; Phase 3.3 hybrid v1
~3.0 - 3.3 s/iter (card load unknown). Sweeps are 74% of GPU wall time.

### Transfer contract, measured (records = 2)

bytes_up 182,456 = P*8 exactly. bytes_dn 787,032 = download_iteration 452,400 +
therm counters 139,200 + delta 182,456 + rank-2 gate flags 11,600 + alpha
scalars 1,368 (171 calls) + grad[alpha] 8. At records = 4, bytes_dn 903,960.

### Scaling (n_thread 16, records 2, early SR, 100-sweep initial therm)

| B | ms/iter | ms/sweep | walker-sweeps/s | samples/s | E_err | E_err*sqrt(B) | GPU mem |
|---|---|---|---|---|---|---|---|
| 3584 | 794 | 66.5 | 53,935 | 9,028 | 0.464 | 27.8 | 2.9 GiB |
| 8192 | 1830 | 151.2 | 54,184 | 8,952 | 0.309 | 28.0 | 6.6 GiB |
| 16384 | 3450 | 285.7 | 57,353 | 9,497 | 0.218 | 27.9 | 13.2 GiB |
| 16384, jet_chunk 4096 | 3469 | 287.1 | 57,071 | 9,447 | 0.218 | 27.9 | 8.8 GiB |

The card saturates by B = 3584: throughput is flat, so B trades wall time for
error bars one-for-one. Working envelope on 24 GB: B = 16384 with
o_pool_max_gb >= 14, or >= 9 with jet_chunk 4096 (<1% cost).

### Protocol

- Both configure modes green: CUDA 12/12, CPU 6/6.
- Determinism: descent run twice, all 25 non-timing csv columns bit-identical;
  jet_chunk 0 vs 1000, all physics columns identical (bytes_dn differs by the
  per-chunk alpha downloads).
- Li6, CPU vs GPU from the same checkpoint, 300 iterations: windowed E agrees
  within |z| <= 1.7 across 6 windows, variances comparable, identical tuned step.

| window | CPU E | GPU E | z |
|---|---|---|---|
| 0-50 | -22.1748 +/- 0.0650 | -22.0674 +/- 0.0738 | +1.09 |
| 50-100 | -23.2709 +/- 0.0605 | -23.2818 +/- 0.0687 | -0.12 |
| 100-150 | -24.0280 +/- 0.0630 | -24.0275 +/- 0.0608 | +0.01 |
| 150-210 | -24.4170 +/- 0.0445 | -24.5318 +/- 0.0514 | -1.69 |
| 210-260 | -24.9561 +/- 0.0538 | -24.9625 +/- 0.0397 | -0.10 |
| 260-300 | -25.3086 +/- 0.0489 | -25.3426 +/- 0.0376 | -0.55 |

### Physics

| system | steps | GPU frozen eval | CPU frozen eval (same checkpoint) | z | coded-H E_0 | experiment |
|---|---|---|---|---|---|---|
| deuteron | 2000 | -2.21402 +/- 0.00157 | -2.21587 +/- 0.00159 | 0.83 | -2.24037 | -2.2246 |
| deuteron | 5000 | -2.23036 +/- 0.00110 | -2.23087 +/- 0.00114 | 0.32 | -2.24037 | -2.2246 |
| triton | 2000 | -8.35675 +/- 0.00425 | -8.36276 +/- 0.00403 | 1.03 | not computed | -8.482 |
| triton | 5000 | -8.3905 +/- 0.0035 | -- | -- | not computed | -8.482 |

(Corrected in Phase 6.A. This table originally compared against -2.2245, which is
the experimental binding energy, not the ground state of the coded Hamiltonian.)

- Deuteron at 5000 steps is 9.9 +/- 1.1 keV ABOVE the coded Hamiltonian's ground
  state -2.2403705 MeV (Phase 6.A oracle), in both builds. The variational
  principle holds; the deuteron is not yet converged at the keV level.
- Triton at 5000 steps is 92 keV above the experimental -8.482. The coded
  Hamiltonian's triton energy is unknown (no 3-body oracle), so that gap cannot
  be split between unconverged training and the model missing experiment.
- A full-length Li6 production run was not done (300-iteration window only).

### Status: what stays on the host per iteration, and why

The host still owns the Ansatz parameters -- the checkpoint and schedule machinery
live there, so params go up once per iteration (P doubles). It computes the
median/MAD clip bounds from the downloaded E_pool, because an nth_element on
~10^4 doubles is trivial next to a device selection kernel. It runs CG's control
flow, because each convergence and pAp decision is a scalar branch (~3 cuBLAS
scalars per CG iteration, ~1.6% of the SR time). And it advances the
lambda/eta/RMS schedules and applies delta. Everything that touches O_pool, the
walkers or the activations stays on the device. The remaining time is 74% sweeps;
the first Phase 6 lead is the per-call alpha download, which synchronises the
stream on every proposal evaluation.
## Phase 6.0 -- instrumentation, and the measurements that pick Phase 6's work

`lib/gpu/prof.h` / `prof.cu`. Named ranges, nestable, each backed by a cudaEvent
pair on the profiled stream (device time) and a steady_clock pair (host wall
time), resolved once per iteration after a single device sync. Ranges are keyed
by their '/'-joined path, so the same helper measured under two parents gives two
rows, and rows are INCLUSIVE -- a parent contains its children.

Launches are counted in `cuda_sync_check`, not with a macro at each `<<<>>>`:
56 of the 57 launch sites already called it (the 57th, `walker_stats_kernel`,
gained the call here), so one line yields complete coverage and no launch site
had to be restructured. NVTX ranges are emitted under `-DVMC_NVTX`; the header
is never required to build.

`prof_enabled = false` in constants.h removes it all: `nm` on the CUDA objects
shows zero references to GpuProf, and both configurations build.

### Profiler overhead, and that it perturbs nothing

Li6 production config, 10 iterations, identical seed, prof off vs on:

| build | ms/iter | trajectory | bytes up/dn per iteration |
|---|---:|---|---|
| prof_enabled = false | 1251.88 | -- | 182456 / 787032 |
| prof_enabled = true  | 1259.07 | bit-identical to the above | 182456 / 787032 |

**+0.57%**, inside the 2% budget. The CSV columns agree bit-for-bit between the
two builds, and the transfer contract is untouched: events move no bytes.

### Probe 1: Li6 production config, 50 iterations

B = 5800, records = 2, resume from best_checkpoint.txt, batch_grow_factor = 1 so
the record count is fixed across the window. Iteration 0 is discarded (cuBLAS
workspace allocation and arena first-touch), 49 profiled.

card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=22807 jet_chunk=0 real=fp64
iterations profiled: 49, mean 1255.40 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.026 | 0.00 | 1.3 | 0.0 | 1.00 |
| net_fwd | gpu | 3.860 | 0.062 | 0.31 | 189.1 | 9.0 | 1.00 |
| assemble | gpu | 0.124 | 0.004 | 0.01 | 6.1 | 1.0 | 1.00 |
| lu | gpu | 0.581 | 0.007 | 0.05 | 28.5 | 1.0 | 1.00 |
| det_combine | gpu | 0.010 | 0.004 | 0.00 | 0.5 | 1.0 | 1.00 |
| envelope | gpu | 0.024 | 4.510 | 0.00 | 1.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 314.502 | 314.513 | 25.05 | 15410.6 | 1137.0 | 1.00 |
| therm_sweeps/propose | gpu | 0.499 | 0.226 | 0.04 | 24.5 | 54.0 | 54.00 |
| therm_sweeps/eval_double | gpu | 235.440 | 286.339 | 18.75 | 11536.6 | 702.0 | 54.00 |
| therm_sweeps/eval_double/net_fwd | gpu | 196.702 | 2.972 | 15.67 | 9638.4 | 486.0 | 54.00 |
| therm_sweeps/eval_double/assemble | gpu | 6.662 | 0.226 | 0.53 | 326.4 | 54.0 | 54.00 |
| therm_sweeps/eval_double/lu | gpu | 29.762 | 0.372 | 2.37 | 1458.3 | 54.0 | 54.00 |
| therm_sweeps/eval_double/det_combine | gpu | 0.502 | 0.225 | 0.04 | 24.6 | 54.0 | 54.00 |
| therm_sweeps/eval_double/envelope | gpu | 1.265 | 281.877 | 0.10 | 62.0 | 54.0 | 54.00 |
| therm_sweeps/accept | gpu | 0.498 | 0.227 | 0.04 | 24.4 | 54.0 | 54.00 |
| therm_sweeps/st_table | gpu | 41.927 | 0.277 | 3.34 | 2054.4 | 36.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.115 | 0.020 | 0.01 | 5.6 | 6.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 39.635 | 0.150 | 3.16 | 1942.1 | 18.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.093 | 0.013 | 0.01 | 4.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.375 | 0.013 | 0.03 | 18.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.641 | 0.021 | 0.13 | 80.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.028 | 0.013 | 0.00 | 1.4 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 35.778 | 1.838 | 2.85 | 1753.1 | 288.0 | 3.00 |
| therm_sweeps/discrete_block/xi_combo | gpu | 1.039 | 0.153 | 0.08 | 50.9 | 36.0 | 36.00 |
| therm_sweeps/discrete_block/net_fwd | gpu | 9.411 | 0.616 | 0.75 | 461.2 | 72.0 | 36.00 |
| therm_sweeps/discrete_block/assemble | gpu | 4.496 | 0.149 | 0.36 | 220.3 | 36.0 | 36.00 |
| therm_sweeps/discrete_block/lu | gpu | 19.689 | 0.239 | 1.57 | 964.8 | 36.0 | 36.00 |
| therm_sweeps/discrete_block/det_combine | gpu | 0.333 | 0.149 | 0.03 | 16.3 | 36.0 | 36.00 |
| record_sweeps | gpu | 625.042 | 625.047 | 49.79 | 30627.1 | 2274.0 | 2.00 |
| record_sweeps/propose | gpu | 0.995 | 0.450 | 0.08 | 48.8 | 108.0 | 108.00 |
| record_sweeps/eval_double | gpu | 467.028 | 568.796 | 37.20 | 22884.4 | 1404.0 | 108.00 |
| record_sweeps/eval_double/net_fwd | gpu | 390.025 | 5.934 | 31.07 | 19111.2 | 972.0 | 108.00 |
| record_sweeps/eval_double/assemble | gpu | 13.317 | 0.450 | 1.06 | 652.5 | 108.0 | 108.00 |
| record_sweeps/eval_double/lu | gpu | 59.067 | 0.743 | 4.71 | 2894.3 | 108.0 | 108.00 |
| record_sweeps/eval_double/det_combine | gpu | 0.999 | 0.450 | 0.08 | 49.0 | 108.0 | 108.00 |
| record_sweeps/eval_double/envelope | gpu | 2.527 | 559.876 | 0.20 | 123.8 | 108.0 | 108.00 |
| record_sweeps/accept | gpu | 0.992 | 0.454 | 0.08 | 48.6 | 108.0 | 108.00 |
| record_sweeps/st_table | gpu | 83.863 | 0.553 | 6.68 | 4109.3 | 72.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.230 | 0.040 | 0.02 | 11.2 | 12.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 79.278 | 0.299 | 6.31 | 3884.6 | 36.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.186 | 0.026 | 0.01 | 9.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.751 | 0.025 | 0.06 | 36.8 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.282 | 0.041 | 0.26 | 160.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.056 | 0.025 | 0.00 | 2.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 71.539 | 3.679 | 5.70 | 3505.4 | 576.0 | 6.00 |
| record_sweeps/discrete_block/xi_combo | gpu | 2.077 | 0.308 | 0.17 | 101.8 | 72.0 | 72.00 |
| record_sweeps/discrete_block/net_fwd | gpu | 18.805 | 1.234 | 1.50 | 921.4 | 144.0 | 72.00 |
| record_sweeps/discrete_block/assemble | gpu | 8.993 | 0.298 | 0.72 | 440.7 | 72.0 | 72.00 |
| record_sweeps/discrete_block/lu | gpu | 39.381 | 0.478 | 3.14 | 1929.7 | 72.0 | 72.00 |
| record_sweeps/discrete_block/det_combine | gpu | 0.667 | 0.296 | 0.05 | 32.7 | 72.0 | 72.00 |
| record | gpu | 230.584 | 230.586 | 18.37 | 11298.6 | 150.0 | 2.00 |
| record/eval_cached | gpu | 10.253 | 9.069 | 0.82 | 502.4 | 28.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 7.623 | 0.167 | 0.61 | 373.5 | 18.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.247 | 0.009 | 0.02 | 12.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.095 | 0.014 | 0.09 | 53.6 | 2.0 | 2.00 |
| record/eval_cached/det_combine | gpu | 0.019 | 0.008 | 0.00 | 0.9 | 2.0 | 2.00 |
| record/eval_cached/envelope | gpu | 0.048 | 8.811 | 0.00 | 2.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.164 | 0.010 | 0.09 | 57.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 131.264 | 132.442 | 10.46 | 6431.9 | 26.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 104.636 | 0.108 | 8.33 | 5127.2 | 16.0 | 2.00 |
| record/jet_pass/detjet | gpu | 25.375 | 0.009 | 2.02 | 1243.4 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.240 | 132.307 | 0.10 | 60.7 | 8.0 | 2.00 |
| record/exchange | gpu | 51.002 | 51.008 | 4.06 | 2499.1 | 44.0 | 2.00 |
| record/exchange/st_table | gpu | 27.969 | 0.189 | 2.23 | 1370.5 | 24.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.085 | 0.014 | 0.01 | 4.2 | 4.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 26.431 | 0.103 | 2.11 | 1295.1 | 12.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 3.0 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.250 | 0.008 | 0.02 | 12.3 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.095 | 0.014 | 0.09 | 53.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 0.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.029 | 0.013 | 0.00 | 1.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 18.573 | 0.081 | 1.48 | 910.1 | 12.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.375 | 0.017 | 0.35 | 214.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 50.682 | 4.04 | 2483.4 | 0.0 | 2.00 |
| record/assemble | gpu | 0.269 | 0.016 | 0.02 | 13.2 | 4.0 | 2.00 |
| record/stats | gpu | 0.025 | 0.013 | 0.00 | 1.2 | 4.0 | 2.00 |
| record/o_assemble | gpu | 37.739 | 37.739 | 3.01 | 1849.2 | 44.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.015 | 0.00 | 0.7 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.324 | 0.031 | 0.03 | 15.9 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 17.132 | 0.119 | 1.36 | 839.5 | 24.0 | 12.00 |
| record/o_assemble/delta_prop | gpu | 10.897 | 0.067 | 0.87 | 533.9 | 14.0 | 8.00 |
| record/o_assemble/o_finalize | gpu | 9.313 | 0.008 | 0.74 | 456.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.062 | 0.00 | 3.0 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.033 | 0.00 | 1.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 7.908 | 0.022 | 0.63 | 387.5 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.118 | 0.01 | 5.8 | 0.0 | 1.00 |
| sr/grad | gpu | 2.393 | 0.012 | 0.19 | 117.3 | 2.0 | 1.00 |
| sr/cg | gpu | 64.745 | 70.243 | 5.16 | 3172.5 | 63.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 69.540 | 5.54 | 3407.5 | 0.0 | 39.35 |
| sr/cg/matvec | gpu | 63.852 | 0.516 | 5.09 | 3128.8 | 26.9 | 13.45 |
| sr/cg/precond | gpu | 0.071 | 0.056 | 0.01 | 3.5 | 12.4 | 12.45 |
| sr/trust | gpu | 4.810 | 4.810 | 0.38 | 235.7 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 4.747 | 0.038 | 0.38 | 232.6 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 4.765 | 0.38 | 233.5 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.026 | 0.00 | 1.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.007 | 0.00 | 0.3 | 0.0 | 1.00 |

Reading it:

| bucket | ms/iter | % |
|---|---:|---:|
| sweeps (therm 314.6 + record 626.7) | 941.3 | 75.0 |
| local_E (eval_cached + jet_pass + exchange + assemble) | 193.0 | 15.4 |
| O assembly | 37.8 | 3.0 |
| SR (o_stats + grad + cg + trust) | 79.9 | 6.4 |
| transfers (all five sites) | 0.14 | 0.011 |

- The three network forwards are **69.8%** of the iteration (876.4 ms: 589 ms in
  the coordinate sweeps' `eval_double`, 119 ms in the (s,t) table, 28 ms in the
  discrete block, 105 ms in the jet pass, 26 ms in the exchange table, 19 ms in
  the exchange rho slots).
- Determinants -- batched LU everywhere, plus getri and the jet determinant --
  are **14.5%** (181.6 ms).
- `propose` + `accept` together are **0.24%** (3.0 ms). The Metropolis machinery
  costs nothing; the wavefunction evaluation it triggers costs everything.
- Every range's children sum to within 0.5% of the parent, so there is no
  unattributed time hiding between kernels at this batch size.
- Achieved FP64 throughput in the forwards: one `eval_chain` is 1.213 GFLOP of
  GEMM, measured at 3.62 ms -> **335 GFLOP/s, 60% of this card's 0.556 TF FP64
  peak**. The forwards are not latency-bound; they are at the FP64 roof.

### Probe 2: batch-size sweep, 10 iterations each

Records fixed at 2, n_thread = 16 so B is exact. Memory is the arena's own
GRAND TOTAL at the end of growth.

| B | jet_chunk | ms/iter | walker-sweeps/s | samples/s | E_err | E_err*sqrt(B) | device memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3584 | 0 | 805.8 | 40030 | 8895 | 0.4653 | 27.85 | 3.03 GiB |
| 5800 | 0 | 1255.4 | 41580 | 9240 | 0.3723 | 28.35 | 4.90 GiB |
| 8192 | 0 | 1837.2 | 40130 | 8918 | 0.3119 | 28.23 | 6.92 GiB |
| 16384 | 4096 | 3453.2 | 42702 | 9489 | 0.2177 | 27.87 | 9.50 GiB |

The saturation shape is: **flat**. Throughput varies by 6% over a 4.6x batch
range, time per iteration is linear in B, and `E_err * sqrt(B)` is constant to
2%. The card is fully occupied at B = 3584; larger batches buy statistics at
exactly proportional cost and nothing else. B = 16384 needs `jet_chunk = 4096`
and `o_pool_max_gb` raised above 8, as in Phase 5.3.

### Probe 3: deuteron, 50 iterations -- the small-system regime

N = 2, B = 5800, K = 11, m_feat = 21, P = 5687. This is the launch-latency
worst case if there is one.

card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=2 K=11 m_feat=21 P=5687 jet_chunk=0 real=fp64
iterations profiled: 49, mean 128.37 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.012 | 0.01 | 0.6 | 0.0 | 1.00 |
| net_fwd | gpu | 0.874 | 0.061 | 0.68 | 42.8 | 9.0 | 1.00 |
| assemble | gpu | 0.005 | 0.004 | 0.00 | 0.2 | 1.0 | 1.00 |
| lu | gpu | 0.013 | 0.007 | 0.01 | 0.6 | 1.0 | 1.00 |
| det_combine | gpu | 0.006 | 0.004 | 0.00 | 0.3 | 1.0 | 1.00 |
| envelope | gpu | 0.022 | 0.836 | 0.02 | 1.1 | 1.0 | 1.00 |
| therm_sweeps | gpu | 27.392 | 27.398 | 21.34 | 1342.2 | 405.0 | 1.00 |
| therm_sweeps/propose | gpu | 0.089 | 0.076 | 0.07 | 4.4 | 18.0 | 18.00 |
| therm_sweeps/eval_double | gpu | 16.727 | 23.045 | 13.03 | 819.6 | 234.0 | 18.00 |
| therm_sweeps/eval_double/net_fwd | gpu | 15.736 | 0.996 | 12.26 | 771.0 | 162.0 | 18.00 |
| therm_sweeps/eval_double/assemble | gpu | 0.093 | 0.073 | 0.07 | 4.5 | 18.0 | 18.00 |
| therm_sweeps/eval_double/lu | gpu | 0.231 | 0.121 | 0.18 | 11.3 | 18.0 | 18.00 |
| therm_sweeps/eval_double/det_combine | gpu | 0.107 | 0.073 | 0.08 | 5.3 | 18.0 | 18.00 |
| therm_sweeps/eval_double/envelope | gpu | 0.380 | 21.571 | 0.30 | 18.6 | 18.0 | 18.00 |
| therm_sweeps/accept | gpu | 0.123 | 0.074 | 0.10 | 6.0 | 18.0 | 18.00 |
| therm_sweeps/st_table | gpu | 7.001 | 0.275 | 5.45 | 343.1 | 36.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.048 | 0.020 | 0.04 | 2.3 | 6.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 6.806 | 0.151 | 5.30 | 333.5 | 18.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.025 | 0.012 | 0.02 | 1.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.026 | 0.012 | 0.02 | 1.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 0.038 | 0.020 | 0.03 | 1.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.019 | 0.012 | 0.01 | 0.9 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 3.288 | 0.611 | 2.56 | 161.1 | 96.0 | 3.00 |
| therm_sweeps/discrete_block/xi_combo | gpu | 0.098 | 0.050 | 0.08 | 4.8 | 12.0 | 12.00 |
| therm_sweeps/discrete_block/net_fwd | gpu | 2.598 | 0.208 | 2.02 | 127.3 | 24.0 | 12.00 |
| therm_sweeps/discrete_block/assemble | gpu | 0.106 | 0.048 | 0.08 | 5.2 | 12.0 | 12.00 |
| therm_sweeps/discrete_block/lu | gpu | 0.155 | 0.078 | 0.12 | 7.6 | 12.0 | 12.00 |
| therm_sweeps/discrete_block/det_combine | gpu | 0.071 | 0.048 | 0.06 | 3.5 | 12.0 | 12.00 |
| record_sweeps | gpu | 54.354 | 54.354 | 42.34 | 2663.4 | 810.0 | 2.00 |
| record_sweeps/propose | gpu | 0.180 | 0.150 | 0.14 | 8.8 | 36.0 | 36.00 |
| record_sweeps/eval_double | gpu | 33.282 | 45.801 | 25.93 | 1630.8 | 468.0 | 36.00 |
| record_sweeps/eval_double/net_fwd | gpu | 31.300 | 1.991 | 24.38 | 1533.7 | 324.0 | 36.00 |
| record_sweeps/eval_double/assemble | gpu | 0.183 | 0.147 | 0.14 | 9.0 | 36.0 | 36.00 |
| record_sweeps/eval_double/lu | gpu | 0.458 | 0.242 | 0.36 | 22.4 | 36.0 | 36.00 |
| record_sweeps/eval_double/det_combine | gpu | 0.215 | 0.146 | 0.17 | 10.5 | 36.0 | 36.00 |
| record_sweeps/eval_double/envelope | gpu | 0.769 | 42.850 | 0.60 | 37.7 | 36.0 | 36.00 |
| record_sweeps/accept | gpu | 0.246 | 0.148 | 0.19 | 12.0 | 36.0 | 36.00 |
| record_sweeps/st_table | gpu | 13.905 | 0.549 | 10.83 | 681.3 | 72.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.095 | 0.040 | 0.07 | 4.7 | 12.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 13.515 | 0.303 | 10.53 | 662.2 | 36.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.050 | 0.025 | 0.04 | 2.4 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.053 | 0.024 | 0.04 | 2.6 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 0.078 | 0.040 | 0.06 | 3.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.036 | 0.025 | 0.03 | 1.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 6.505 | 1.223 | 5.07 | 318.8 | 192.0 | 6.00 |
| record_sweeps/discrete_block/xi_combo | gpu | 0.195 | 0.100 | 0.15 | 9.6 | 24.0 | 24.00 |
| record_sweeps/discrete_block/net_fwd | gpu | 5.132 | 0.416 | 4.00 | 251.5 | 48.0 | 24.00 |
| record_sweeps/discrete_block/assemble | gpu | 0.211 | 0.097 | 0.16 | 10.4 | 24.0 | 24.00 |
| record_sweeps/discrete_block/lu | gpu | 0.308 | 0.157 | 0.24 | 15.1 | 24.0 | 24.00 |
| record_sweeps/discrete_block/det_combine | gpu | 0.142 | 0.097 | 0.11 | 7.0 | 24.0 | 24.00 |
| record | gpu | 26.547 | 26.547 | 20.68 | 1300.8 | 142.0 | 2.00 |
| record/eval_cached | gpu | 1.968 | 1.940 | 1.53 | 96.5 | 28.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 1.814 | 0.156 | 1.41 | 88.9 | 18.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.010 | 0.008 | 0.01 | 0.5 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 0.026 | 0.014 | 0.02 | 1.3 | 2.0 | 2.00 |
| record/eval_cached/det_combine | gpu | 0.012 | 0.008 | 0.01 | 0.6 | 2.0 | 2.00 |
| record/eval_cached/envelope | gpu | 0.043 | 1.703 | 0.03 | 2.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 0.024 | 0.009 | 0.02 | 1.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 7.439 | 7.461 | 5.80 | 364.5 | 26.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 6.581 | 0.107 | 5.13 | 322.5 | 16.0 | 2.00 |
| record/jet_pass/detjet | gpu | 0.674 | 0.009 | 0.53 | 33.0 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 0.170 | 7.329 | 0.13 | 8.4 | 8.0 | 2.00 |
| record/exchange | gpu | 5.843 | 5.847 | 4.55 | 286.3 | 36.0 | 2.00 |
| record/exchange/st_table | gpu | 4.629 | 0.186 | 3.61 | 226.8 | 24.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.028 | 0.013 | 0.02 | 1.4 | 4.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 4.502 | 0.103 | 3.51 | 220.6 | 12.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.017 | 0.008 | 0.01 | 0.8 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.017 | 0.008 | 0.01 | 0.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 0.026 | 0.014 | 0.02 | 1.3 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.012 | 0.008 | 0.01 | 0.6 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.017 | 0.013 | 0.01 | 0.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 1.084 | 0.040 | 0.84 | 53.1 | 6.0 | 2.00 |
| record/exchange/rank2 | gpu | 0.069 | 0.009 | 0.05 | 3.4 | 2.0 | 2.00 |
| record/exchange/fallback | host | 0.000 | 5.581 | 4.35 | 273.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.027 | 0.014 | 0.02 | 1.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.016 | 0.014 | 0.01 | 0.8 | 4.0 | 2.00 |
| record/o_assemble | gpu | 11.222 | 11.222 | 8.74 | 549.9 | 44.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.014 | 0.01 | 0.7 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.042 | 0.027 | 0.03 | 2.1 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 6.858 | 0.119 | 5.34 | 336.0 | 24.0 | 12.00 |
| record/o_assemble/delta_prop | gpu | 1.524 | 0.066 | 1.19 | 74.7 | 14.0 | 8.00 |
| record/o_assemble/o_finalize | gpu | 2.727 | 0.009 | 2.12 | 133.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.061 | 0.05 | 3.0 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.032 | 0.03 | 1.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 3.419 | 0.034 | 2.66 | 167.5 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.116 | 0.09 | 5.7 | 0.0 | 1.00 |
| sr/grad | gpu | 0.626 | 0.023 | 0.49 | 30.7 | 2.0 | 1.00 |
| sr/cg | gpu | 13.614 | 16.382 | 10.61 | 667.1 | 48.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 15.763 | 12.28 | 772.4 | 0.0 | 30.35 |
| sr/cg/matvec | gpu | 13.071 | 0.478 | 10.18 | 640.5 | 20.9 | 10.45 |
| sr/cg/precond | gpu | 0.048 | 0.040 | 0.04 | 2.3 | 9.4 | 9.45 |
| sr/trust | gpu | 1.304 | 1.304 | 1.02 | 63.9 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 1.257 | 0.052 | 0.98 | 61.6 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 1.245 | 0.97 | 61.0 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.011 | 0.01 | 0.6 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.006 | 0.00 | 0.3 | 0.0 | 1.00 |

- 128.4 ms/iter, ~10x faster than Li6 for a system with 1/9 the network output
  width. The shape is the same, not different: forwards **69.2%**, determinants
  **1.6%**, propose+accept **0.5%**.
- Leaf ranges still tile the iteration (97.8% of wall), so there is no large
  pool of pure launch latency to reclaim -- see the alpha experiment below for
  the direct measurement.
- What does change is efficiency: one deuteron `eval_chain` is 102.5 MFLOP in
  0.869 ms = **118 GFLOP/s, 21% of FP64 peak**, against 60% for Li6. Small
  systems waste the card on GEMM shape, not on launches.

### The alpha download: measured, and smaller than Phase 5.3 assumed

Phase 5.3 flagged the per-call alpha download in `envelope_logp` (and the two in
the jet composition) as "the first Phase 6 lead": it synchronises the stream
~170 times per iteration. The profiler shows exactly that stall --
`eval_double/envelope` has 0.77 ms of device time against **42.9 ms of host
wall** in the record sweeps -- but a host stall is only a cost if the device
runs dry behind it.

A/B, scratchpad only: the kernels read `params[P-1]` from device memory instead,
deleting every alpha download and its sync (numerically neutral -- the deuteron
trajectory is bit-identical, checkpoint-free, over all 50 iterations).

| config | with alpha sync | alpha read on device | gain |
|---|---:|---:|---:|
| deuteron, B=5800 | 128.4 ms/iter | 125.7 ms/iter | 2.1% |
| Li6, B=5800 | 1255.4 ms/iter | 1255.4 ms/iter | 0.0% |

The host blocks, but the queue behind it is deep enough that the device never
starves. **The alpha sync is worth 2% on the smallest system and nothing on the
production one.** It is a tidy 5-line cleanup, not a Phase 6 lead.

### SR at mature lambda

The 50-iteration probes start at lambda = 100 (iteration 0 of the schedule), where
CG converges in ~12 iterations and SR is 6.4% of the iteration. Production spends
most of its life near lambda_min = 0.4. Re-running 10 iterations with
`sr_lambda0 = 0.4`:

| lambda | cg iters | sr ms/iter | ms/iter | SR share |
|---:|---:|---:|---:|---:|
| 100 (schedule start) | 12.5 | 79.1 | 1255 | 6.3% |
| 0.4 (schedule floor) | 100.0 | 499.6 | 1685 | 29.7% |

CG's cost is entirely `matvec` (two Ns x P GEMVs per apply) plus ~3 blocking
cuBLAS scalars per iteration. At the schedule floor SR is the second-largest
block in the run, after the forwards.


### Task D: the Phase 6 decision table

Every trigger below is the one written into the Phase 6.0 prompt, evaluated
against the probes above. "sweep-overhead share" = (propose + accept + parent-
minus-children gaps at sweep level + the same inside eval_double) / sweep time.

| candidate | prompted trigger | measured | verdict |
|---|---|---|---|
| **6.1** sweep latency / CUDA graphs | sweep-overhead share > 20%, or deuteron iteration dominated by launches | Li6 **0.60%**, deuteron **1.93%** of sweep time. The largest identified host stall (the alpha sync, ~170 blocking downloads per iteration) is worth **2.1% on the deuteron and 0.0% on Li6** when removed outright | **not triggered** |
| **6.2** custom batched LU | lu share of eval_double + detjet > 15% of iteration | strict reading **9.1%** (88.8 ms lu in eval_double + 25.4 ms detjet); widest reading, every LU plus getri plus detjet, **14.5%**; deuteron 1.6% | **not triggered** (14.5 vs 15). Revisit at larger K: LU work scales as K*B while the forwards scale as K*B too, so the ratio moves only with N |
| **6.3** FP32 forward | net_fwd + jet_net > 30% AND card FP64:FP32 worse than 1:2 | **69.8%** (Li6), **69.2%** (deuteron); RTX 3090 is **1:64**; the forwards already run at **60% of FP64 peak**, so the ceiling is real silicon, not software | **TRIGGERED -- the only candidate with a large ceiling** |
| **6.4** multi-GPU | B sweep saturated AND statistics-limited physics goals | saturated: 40.0k -> 42.7k walker-sweeps/s across B = 3584..16384 (6% spread over 4.6x batch); statistics-limited: E_err*sqrt(B) constant at 27.9-28.4, and Li6 sits at E_err ~ 0.37 MeV per iteration | **triggered**, but it multiplies an FP64-bound iteration. Sequence it after 6.3 |
| **minSR dual** (not prompted yet) | P/Ns > 1/3 at planned configs | **P/Ns = 22807 / 11600 = 1.97**, six times the threshold. SR is 6.3% of the iteration at lambda = 100 but **29.7% at lambda = 0.4**, with CG at 100 iterations | log only; the dual is worth planning for the schedule floor, not the start |

Order implied by the measurements: **6.3, then 6.4**, with 6.2 held as a
second-order item and 6.1 closed as measured-and-rejected. The one cheap piece
of 6.1 worth keeping is deleting the alpha downloads, which is a cleanup (it
removes ~170 stream syncs and 3 of the 5 remaining transfer sites) rather than a
performance change.

### Carried forward, unchanged by this phase

- Deuteron at 5000 steps sits ~10 keV above the coded-H ground state -2.24037
  (resolved in Phase 6.A: the old -2.2245 "target" was the experimental value).
- Triton is 92 keV above the experimental -8.482; no coded-H oracle yet.
- A full-length Li6 production run has still not been done.

## Phase 6.1 -- elementwise fusion and CUDA graphs for the sweep

### What was built

**Fusions** (each bitwise identical to the kernels it replaces, tested pre/post):

| fusion | status |
|---|---|
| `shift_to_com` + `build_feat` -> `shift_build_feat` | done |
| `shift_to_com` + `build_feat_combo` -> `shift_feat_combo` ((s,t) table) | done |
| `S_combine` + `envelope_logp` -> `combine_envelope` | done: Li6 3.50 -> 1.85 ms/iter, deuteron 0.96 -> 0.42 |
| `det_from_lu` into the same tail | **measured and rejected**: its parallelism is one thread per matrix (B*K = 180k on Li6); thread-per-walker fused took 110 us/call and warp-per-walker 140 us/call against 59 us separate |
| propose + x_prop row copy | already one kernel since Phase 3.3 |
| accept + acceptance counter | already one kernel since Phase 3.3 |
| jet features + analytic seeds | already one kernel (`build_jet_feat_kernel` computes the COM inline) |

Still split, by necessity: `getrf` (cuBLAS), the `assemble_M` scatter into the per-matrix layout that getrf's pointer array needs, and bias+activation after each GEMM (a cuBLAS GEMM has no epilogue hook).

**Graphs.** One captured graph per coordinate draw (20 kernel nodes: propose, the evaluation chain, accept), one per spin round and one per isospin round (11 nodes each), each replayed `draws` / `spin_draws` / `tau_draws` times per sweep. They are re-captured and updated in place (`cudaGraphExecUpdate`) only when B or the proposal step changes, which is at most once per iteration. The (s,t) table build runs once per sweep and stays eager. `use_cuda_graphs` in constants.h selects the path; the eager path is kept as the debug path and must reproduce the graphs bit for bit.

The prompt's premise that a draw contains no syncs was **false**: `envelope_logp` downloaded alpha and called `cudaStreamSynchronize` on every evaluation, so capture fails as the code stood. It now reads alpha on the device (bitwise neutral; the same A/B as 6.0), which also deletes 57 blocking 8-byte downloads per deuteron iteration.

Capture requirements, each checked: cuBLAS gets its stream and a 32 MiB user workspace before capture and is never re-bound inside it (`blas_bind` only calls `cublasSetStream` on a real change); capture uses `cudaStreamCaptureModeGlobal` on a dedicated blocking stream, so every op on legacy stream 0 is implicitly ordered around the graph launches; and the profiler and the launch counter pause while capturing. Replay is correct because every kernel reads and advances its walker's Philox counter in device memory (the Phase 2 counter-based RNG): identical graphs draw fresh random numbers.

### Correctness

- `test_graphs`, 50 sweeps graphs-on vs graphs-off, B = 512, with a step change at sweep 25 (re-capture + in-place update) and B = 300 from sweep 45 (re-instantiation): **0 differing values** in x, s, t, logp, rng_ctr and all three acceptance counters. Negative control -- the graph not re-captured when the step changes: **13,654 values differ**.
- Every fusion: 0 differing values against the unfused kernels, including forced-singular walkers (S = 0, logp = -inf).
- Suite: 13/13 with `use_cuda_graphs = true`, 13/13 with `false`, CPU 6/6.
- Physics: deuteron 200 iterations and Li6 50 iterations, before-6.1 vs fusions-only vs fusions+graphs, two repetitions each: **bit-identical CSV trajectories**. The only changed field is `bytes_dn` (-456 bytes/iteration on the deuteron, the deleted alpha downloads).

### Timing: before / fusions only / fusions + graphs

Same configs as the 6.0 probes (resume, therm_steps_init = 100, batch_grow_factor = 1); deuteron 200 iterations, Li6 50; iteration 0 discarded. ms/iter is the mean over two interleaved repetitions (SD within a run 1.3-1.7 ms deuteron, 3.3-4.7 ms Li6).

| | before | fusions only | fusions + graphs | graphs gain |
|---|---:|---:|---:|---:|
| **deuteron** ms/iter | 129.83 | 129.12 | **124.87** | **-3.8%** |
| sweep time (rep 2) | 82.56 | 81.40 | 77.12 | -6.6% |
| launches/iter, whole iteration | 1370 | 1245 | 345 | -75% |
| launches/iter, sweeps only | 1215 | 1098 | 198 | -84% |
| **Li6** ms/iter | 1266.11 | 1262.16 | **1251.45** | **-1.2%** |
| sweep time (rep 2) | 948.66 | 943.98 | 933.91 | -1.6% |
| launches/iter, whole iteration | 3574 | 3233 | 533 | -85% |

With graphs on, "launches" counts host-issued launches, one per graph replay.

Honest reading:

- **Graphs earn their keep where the prompt predicted, and only there.** The deuteron's sweeps lose 5.4 ms/iter; Li6 gains 1.2%, inside what its GEMMs dominate.
- **The 6.0 overhead metric underestimated launch cost.** 6.0 put the reclaimable sweep overhead at 1.93% of deuteron sweep time; graphs recovered 6.6%. The 6.0 metric could only see gaps *between* profiled ranges; gaps between the kernels *inside* a leaf range (one net_fwd call is 8 counted launches plus 6 cuBLAS GEMMs) were counted as that range's GPU time. The graph run is the direct measurement.
- **The fusions are worth ~0.3-0.5%**, mostly from `combine_envelope` and the deleted alpha syncs. They matter more as a precondition for graphs than on their own.
- The iteration is still ~70% FP64 network forwards on both systems. 6.1 moves the deuteron from 129.8 to 124.9 ms; 6.3 is what can move either system substantially.

### Two bugs found in the committed 6.0 tree

Both came from applying 6.0 edits where the replaced lines stayed in place:

1. `download_iteration` ran the old staging loop AND the new one, copying every segment twice; the second pass starts at `off = total` and writes past the end of `pack_d`. Production survives by accident (its pack buffer is sized for `records_per_iter_max`, twice the records used before `grow_at_iter`) but would crash at iteration 7000; every 6.1 probe (batch_grow_factor = 1) crashed on the first iteration.
2. `assemble_O_batch` called `cuda_sync_check("o_finalize")` twice, double-counting one launch.

Both are fixed in the 6.1 code. The "before" column above was measured with them fixed, so the comparison isolates 6.1.

## Phase 6.2 -- custom batched LU / det / inverse

`lib/gpu/lu_batched_small.cu`: the host `lu_det_inv` transcribed operation for operation (strict `>` pivot search so ties go to the lowest row, division by the pivot, the 1e-300 pivot guard zeroing det and Minv, the column-by-column inverse). `batched_lu_det_inv` is now the single entry point for the double evaluation, the (s,t) table and the jet pass's Minv, which comes from the same factorisation as the determinants (the separate `getri` step is gone). `use_custom_lu` (default **true**, set by the measurement below) selects it; cuBLAS getrf/getri stays compiled as the oracle and as the fallback for N > 16.

### Correctness (`test_lu`)

| check | result |
|---|---|
| vs host `lu_det_inv` on identical input, 100,002 matrices, n = 2, 3, 6, 8, 12, 16 (half random; half duplicate rows, duplicate columns, zero column, 1e-301 column, pivot ties) | **det and Minv bit-identical for every matrix**; singular flags agree with the host's verdict everywhere; input never modified |
| production path (`eval_jet_prepare`, B = 1024): 31,744 real Slater matrices vs `lu_det_inv` | **bit-identical** det and Minv |
| vs cuBLAS getrf/getri on the same batches (generic matrices) | values agree to <= 0.43 x kappa * n * eps; kappa <= 100: max rel det 5.3e-15; raw max rel det up to 9.4e-12 on kappa ~ 1e6 matrices; bitwise agreement only occasional |
| negative control: pivot tie rule `>=` (last maximum wins) | fails at every n (bit identity broken on the tie cases) |
| pipeline twice from identical state (3 sweeps + local_E) | 0 differing values |
| suite | 14/14 with `use_custom_lu = true`, 14/14 with `false`, CPU 6/6 |

Why cuBLAS is compared by value and against kappa, not bits: it reads the row-major storage as column-major and so factorises the **transpose**, a different pivot sequence from the host's. Two backward-stable LUs with different pivot orders disagree on det by up to ~ kappa n eps. Since the custom kernel is bit-identical to the host, it -- not cuBLAS -- is the production and determinism reference from 6.2 on.

### Design: the prompt's warp-per-matrix kernel lost; thread-per-matrix won

| kernel (Li6, 180k 6x6 matrices per call) | (s,t)-table LU per call |
|---|---:|
| cuBLAS getrf + det_from_lu | 0.55 ms |
| warp per matrix, shared memory (as prompted) | 2.10 ms (3.8x slower; 11x slower at n = 2) |
| **thread per matrix, registers** | **0.21 ms (2.7x faster)** |

The prompt's premise that cuBLAS is slow at N <= 6 did not hold (~3 ns per 6x6 matrix). A 6x6 elimination has too little parallelism for 32 lanes; one thread running the host loop with the matrix unrolled in registers (runtime-indexed swaps rewritten as predicated moves, so nothing spills) is the fast layout. The warp kernel remains for 9 <= n <= 16, where registers run out; it is tested but unused in production.

### Timing: `use_custom_lu` off (cuBLAS) vs on

Same probe configs as 6.0/6.1 (graphs on); two interleaved repetitions each, SD within a run 1.5 ms deuteron / 3.3 ms Li6.

| | cuBLAS | custom | change |
|---|---:|---:|---:|
| **Li6** ms/iter | 1245.40 | **1147.93** | **-7.8%** |
| Li6 sweep time (therm + record) | 928.2 | 832.8 | -10.3% |
| Li6 (s,t)-table LU, 6 calls/iter | 3.30 | 1.24 | 2.7x |
| Li6 det + Minv for the jet pass, 2 calls/iter | 2.27 | 1.31 | 1.7x |
| **deuteron** ms/iter | 123.56 | 123.46 | wash (2x2 matrices) |

Most of the Li6 saving sits inside the coordinate-draw graphs (54 LU calls per sweep), which the profiler times as one range. 6.0 measured all determinant work at 14.5% of the Li6 iteration and placed 6.2 just under its 15% trigger; the realised gain, 7.8% end to end, is about half of that share.

### Physics

- Bit-reproducible with the flag on: rep 1 vs rep 2, 0 differing CSV fields (deuteron 200 iterations, Li6 50).
- Flag on vs off follow the same trajectory in the printed CSV until last-bit determinant differences grow chaotically (visible from iteration 171 deuteron, 43 Li6). Mean E over the second half: deuteron -1.9552 (custom) vs -1.9553 (cuBLAS); Li6 -22.532 vs -22.521 (se ~ 0.06).

## Phase 6.A -- the deuteron reference, audited

The frozen-eval deuteron (-2.2304 +/- 0.0011 GPU, -2.2309 +/- 0.0011 CPU) sat 5-6 sigma *below* the table's target -2.2245. VMC is variational, so either that target was not the ground state of the coded Hamiltonian, or both builds shared an estimator bias. `tests/test_deut_ref.cpp` settles it: the ground state of the Hamiltonian as coded, with the constants `#include`d from `constants.h`, by two independent methods.

- Relative motion with hbar^2/(2 mu) = 2 * hbar2_2m = 41.471036 MeV fm^2 (local_E's single-nucleon Laplacian acting on a translation-invariant psi).
- 3S1 T=0 potential from local_E's projection: R_s = +1, R_t = -1, R_st = -1 zero the C01 bracket and make the C10 bracket 4, so V(r) = hbarc C10 exp(-r^2/R10^2) / (pi^1.5 R10^3).

**A. Numerov** (Giannozzi matching at 3 fm, O(h^4)), E in MeV:

| box R | h = 0.01 | h = 0.005 | h = 0.0025 | Richardson |
|---:|---:|---:|---:|---:|
| 20 fm, hard wall | -2.2389922 | -2.2389924 | -2.2389924 | -2.2389924 |
| 30 fm, hard wall | -2.2403572 | -2.2403573 | -2.2403573 | -2.2403573 |
| 60 fm, hard wall | -2.2403703 | -2.2403705 | -2.2403705 | -2.2403705 |
| 120 fm, hard wall | -2.2403703 | -2.2403705 | -2.2403705 | -2.2403705 |
| 20-120 fm, asymptotic exp(-kappa r) | -2.2403703 | -2.2403705 | -2.2403705 | -2.2403705 |

**B. Gaussian basis** (20 to 50 even-tempered Gaussians, analytic matrix elements, generalised eigenproblem): -2.2403705 MeV at every size.

**E_ref = -2.2403705 MeV** for the coded Hamiltonian; the two methods agree to 1e-9 MeV. The grid is converged at h = 0.01 fm; the box costs 1.4 keV at 20 fm, 13 eV at 30 fm and nothing measurable beyond -- neither was the cause. The 1S0 T=1 channel is unbound (Gaussian lowest eigenvalue -> 0+ as the basis widens).

### Verdict

**The old target was not an oracle.** -2.2245 was never computed from our constants -- no solver for it existed in the repository -- and it is the experimental deuteron binding energy (2.2246 MeV). The paper fits C10 and R10 to three triplet observables (scattering length, effective range, binding energy) with two parameters, so the coded model need not reproduce the experimental binding, and it does not: it binds 15.9 keV more. Rounding of the printed constants cannot account for that: the last digit of C10 moves E_d by +/-0.9 keV and of R10 by +/-5.7 keV; reproducing -2.2245 would take C10 = -7.0314 fm^2 against the coded -7.040.

**The VMC is not converged at the keV level.** Against the correct reference the frozen eval is **9.9 +/- 1.1 keV above** the ground state (0.45%), as the variational principle requires, and consistent with training iterates that were still falling at 5000 steps. There is no evidence of a shared estimator bias, so the suspects listed for that case (invalid-sample exclusion, error-bar underestimation, a mass-factor mismatch) were not pursued. CPU and GPU agree with each other throughout.

For context only, and not our criterion: the paper does not state model "o"'s own deuteron energy in its text; the experimental value is 2.2246 MeV.

### Triton

The triton "target" -8.482 is the **experimental** 3H binding energy (8.4818 MeV), to which the paper fits c_E for its R3; `constants.h` uses a refit c_E = 1.2945. It did not come from a box or grid calculation, so the Numerov box/grid question does not apply, and it is not the ground state of the coded Hamiltonian. The error bar it deserves as a reference is unknown until the coded Hamiltonian's triton is computed -- given a 16 keV model-vs-experiment offset already in the deuteron, tens to ~100 keV would be unsurprising. What is certain is only E_0(coded) <= -8.3905 +/- 0.0035 (the VMC). A real oracle needs a three-body solver (e.g. correlated Gaussians with the stochastic variational method); not done.

### prof 2026-09-22 13:05:36 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 99, mean 2521.19 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.063 | 0.00 | 6.2 | 0.0 | 1.00 |
| net_fwd | gpu | 8.559 | 0.114 | 0.34 | 847.3 | 14.0 | 1.00 |
| assemble | gpu | 0.148 | 0.005 | 0.01 | 14.7 | 1.0 | 1.00 |
| lu | gpu | 0.632 | 0.008 | 0.03 | 62.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 1.8 | 1.0 | 1.00 |
| therm_sweeps | gpu | 613.438 | 622.679 | 24.33 | 60730.4 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 474.183 | 0.125 | 18.81 | 46944.1 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 90.624 | 0.418 | 3.59 | 8971.7 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.089 | 0.014 | 0.00 | 8.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.299 | 0.295 | 3.50 | 8741.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.093 | 0.013 | 0.00 | 9.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.374 | 0.013 | 0.01 | 37.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.697 | 0.021 | 0.07 | 168.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 3.0 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.524 | 0.084 | 1.92 | 4803.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1235.972 | 1235.983 | 49.02 | 122361.2 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 954.279 | 0.247 | 37.85 | 94473.6 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.233 | 0.850 | 7.27 | 18140.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.202 | 0.029 | 0.01 | 20.0 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.440 | 0.602 | 7.08 | 17665.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.234 | 0.026 | 0.01 | 23.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.749 | 0.026 | 0.03 | 74.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.468 | 0.044 | 0.14 | 343.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.026 | 0.00 | 6.0 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.169 | 0.166 | 3.89 | 9718.7 | 72.0 | 6.00 |
| record | gpu | 469.826 | 469.802 | 18.64 | 46512.7 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.720 | 0.445 | 0.78 | 1952.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.915 | 0.348 | 0.67 | 1674.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.446 | 0.010 | 0.02 | 44.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.099 | 0.017 | 0.04 | 108.8 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 3.4 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.171 | 0.012 | 0.05 | 115.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 247.151 | 266.462 | 9.80 | 24468.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 219.080 | 0.209 | 8.69 | 21689.0 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.500 | 0.010 | 1.05 | 2623.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.558 | 266.225 | 0.06 | 154.2 | 8.0 | 2.00 |
| record/exchange | gpu | 111.174 | 111.323 | 4.41 | 11006.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.802 | 0.296 | 2.41 | 6019.4 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.010 | 0.00 | 8.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.092 | 0.209 | 2.34 | 5850.1 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 6.1 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.322 | 0.009 | 0.01 | 31.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.200 | 0.015 | 0.05 | 118.8 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 1.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.029 | 0.015 | 0.00 | 2.9 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.526 | 0.146 | 1.81 | 4507.1 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.755 | 0.018 | 0.19 | 470.7 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.820 | 4.40 | 10971.1 | 0.0 | 2.00 |
| record/assemble | gpu | 0.458 | 0.019 | 0.02 | 45.4 | 4.0 | 2.00 |
| record/stats | gpu | 0.025 | 0.015 | 0.00 | 2.5 | 4.0 | 2.00 |
| record/o_assemble | gpu | 91.060 | 91.048 | 3.61 | 9015.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.219 | 0.01 | 21.7 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.530 | 0.035 | 0.02 | 52.5 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.467 | 0.251 | 1.29 | 3214.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.488 | 0.178 | 1.01 | 2523.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.245 | 0.009 | 1.28 | 3192.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.159 | 0.01 | 15.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.041 | 0.00 | 4.1 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.565 | 0.030 | 0.58 | 1442.0 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 12.2 | 0.0 | 1.00 |
| sr/grad | gpu | 5.901 | 0.014 | 0.23 | 584.2 | 2.0 | 1.00 |
| sr/cg | gpu | 160.455 | 169.103 | 6.36 | 15885.0 | 67.3 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 167.250 | 6.63 | 16557.7 | 0.0 | 41.79 |
| sr/cg/matvec | gpu | 156.774 | 1.636 | 6.22 | 15520.6 | 28.5 | 14.26 |
| sr/cg/precond | gpu | 0.888 | 0.061 | 0.04 | 87.9 | 13.3 | 13.26 |
| sr/trust | gpu | 11.136 | 11.136 | 0.44 | 1102.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.974 | 0.112 | 0.44 | 1086.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.014 | 0.44 | 1090.4 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 4.8 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 0.8 | 0.0 | 1.00 |

### prof 2026-09-22 13:09:44 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 199, mean 2499.54 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.059 | 0.00 | 11.8 | 0.0 | 1.00 |
| net_fwd | gpu | 8.465 | 0.112 | 0.34 | 1684.6 | 14.0 | 1.00 |
| assemble | gpu | 0.136 | 0.005 | 0.01 | 27.0 | 1.0 | 1.00 |
| lu | gpu | 0.618 | 0.008 | 0.02 | 122.9 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 3.6 | 1.0 | 1.00 |
| therm_sweeps | gpu | 607.151 | 616.269 | 24.29 | 120823.1 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 469.093 | 0.123 | 18.77 | 93349.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 89.854 | 0.414 | 3.59 | 17881.0 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.090 | 0.013 | 0.00 | 18.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 87.540 | 0.293 | 3.50 | 17420.4 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.093 | 0.013 | 0.00 | 18.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.375 | 0.013 | 0.01 | 74.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.686 | 0.021 | 0.07 | 335.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 6.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.096 | 0.083 | 1.92 | 9571.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1219.708 | 1219.728 | 48.80 | 242722.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 941.881 | 0.242 | 37.68 | 187434.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 180.814 | 0.841 | 7.23 | 35982.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.192 | 0.028 | 0.01 | 38.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 176.093 | 0.596 | 7.05 | 35042.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.210 | 0.026 | 0.01 | 41.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.749 | 0.026 | 0.03 | 149.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.430 | 0.043 | 0.14 | 682.6 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 12.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 96.769 | 0.164 | 3.87 | 19257.1 | 72.0 | 6.00 |
| record | gpu | 462.967 | 462.958 | 18.52 | 92130.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.463 | 0.438 | 0.78 | 3873.1 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.728 | 0.342 | 0.67 | 3329.0 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.375 | 0.010 | 0.02 | 74.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.099 | 0.017 | 0.04 | 218.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 6.9 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.170 | 0.011 | 0.05 | 232.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 243.824 | 262.875 | 9.75 | 48521.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.152 | 0.208 | 8.65 | 43014.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.201 | 0.010 | 1.05 | 5214.0 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.458 | 262.639 | 0.06 | 290.2 | 8.0 | 2.00 |
| record/exchange | gpu | 109.785 | 109.886 | 4.39 | 21847.2 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.127 | 0.292 | 2.41 | 11965.3 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.080 | 0.010 | 0.00 | 16.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.468 | 0.207 | 2.34 | 11635.2 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 12.3 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.286 | 0.009 | 0.01 | 56.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.185 | 0.015 | 0.05 | 235.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 3.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.029 | 0.015 | 0.00 | 5.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 44.923 | 0.145 | 1.80 | 8939.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.645 | 0.018 | 0.19 | 924.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.389 | 4.38 | 21768.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.389 | 0.018 | 0.02 | 77.4 | 4.0 | 2.00 |
| record/stats | gpu | 0.025 | 0.015 | 0.00 | 5.1 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.331 | 89.305 | 3.57 | 17776.8 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.132 | 0.01 | 26.4 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.460 | 0.034 | 0.02 | 91.6 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.017 | 0.247 | 1.28 | 6371.3 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.081 | 0.176 | 1.00 | 4991.0 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.516 | 0.009 | 1.26 | 6271.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.124 | 0.00 | 24.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.041 | 0.00 | 8.1 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.319 | 0.029 | 0.57 | 2849.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 24.3 | 0.0 | 1.00 |
| sr/grad | gpu | 5.793 | 0.014 | 0.23 | 1152.9 | 2.0 | 1.00 |
| sr/cg | gpu | 168.314 | 176.827 | 6.73 | 33494.5 | 72.4 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 175.247 | 7.01 | 34874.2 | 0.0 | 44.81 |
| sr/cg/matvec | gpu | 165.369 | 1.352 | 6.62 | 32908.5 | 30.5 | 15.27 |
| sr/cg/precond | gpu | 0.647 | 0.065 | 0.03 | 128.7 | 14.3 | 14.27 |
| sr/trust | gpu | 10.980 | 10.980 | 0.44 | 2185.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.842 | 0.089 | 0.43 | 2157.6 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 10.883 | 0.44 | 2165.7 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 9.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 1.6 | 0.0 | 1.00 |

### prof 2026-09-22 13:16:06 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 299, mean 2942.37 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.064 | 0.00 | 19.1 | 0.0 | 1.00 |
| net_fwd | gpu | 9.638 | 0.114 | 0.33 | 2881.9 | 14.0 | 1.00 |
| assemble | gpu | 0.132 | 0.005 | 0.00 | 39.4 | 1.0 | 1.00 |
| lu | gpu | 0.919 | 0.008 | 0.03 | 274.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 5.3 | 1.0 | 1.00 |
| therm_sweeps | gpu | 698.679 | 709.267 | 23.75 | 208905.1 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 539.737 | 0.123 | 18.34 | 161381.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 103.152 | 0.414 | 3.51 | 30842.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 27.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 100.617 | 0.293 | 3.42 | 30084.4 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.132 | 0.013 | 0.00 | 39.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.433 | 0.013 | 0.01 | 129.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.807 | 0.021 | 0.06 | 540.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 9.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 55.680 | 0.083 | 1.89 | 16648.3 | 36.0 | 3.00 |
| record_sweeps | gpu | 1399.390 | 1399.386 | 47.56 | 418417.7 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1079.898 | 0.243 | 36.70 | 322889.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 208.297 | 0.846 | 7.08 | 62280.7 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.213 | 0.029 | 0.01 | 63.7 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 201.990 | 0.601 | 6.86 | 60395.1 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.218 | 0.026 | 0.01 | 65.0 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.850 | 0.026 | 0.03 | 254.0 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.886 | 0.044 | 0.17 | 1461.0 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 18.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 110.322 | 0.165 | 3.75 | 32986.3 | 72.0 | 6.00 |
| record | gpu | 532.132 | 532.240 | 18.09 | 159107.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 22.125 | 0.453 | 0.75 | 6615.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 19.424 | 0.355 | 0.66 | 5807.7 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.340 | 0.010 | 0.01 | 101.8 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.099 | 0.018 | 0.04 | 328.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 10.4 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.171 | 0.012 | 0.04 | 350.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 280.973 | 302.689 | 9.55 | 84010.8 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 248.403 | 0.210 | 8.44 | 74272.5 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 29.913 | 0.010 | 1.02 | 8944.1 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.643 | 302.451 | 0.09 | 790.2 | 8.0 | 2.00 |
| record/exchange | gpu | 125.092 | 125.788 | 4.25 | 37402.4 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 68.529 | 0.294 | 2.33 | 20490.2 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.010 | 0.00 | 24.1 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 66.698 | 0.208 | 2.27 | 19942.8 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 18.5 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.486 | 0.009 | 0.02 | 145.2 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.157 | 0.015 | 0.04 | 345.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 5.6 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.029 | 0.015 | 0.00 | 8.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 51.289 | 0.146 | 1.74 | 15335.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.183 | 0.018 | 0.18 | 1549.8 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 125.287 | 4.26 | 37460.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.910 | 0.019 | 0.03 | 272.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 7.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 102.331 | 102.335 | 3.48 | 30597.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.659 | 0.02 | 197.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 1.018 | 0.035 | 0.03 | 304.4 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 35.941 | 0.248 | 1.22 | 10746.4 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 28.574 | 0.176 | 0.97 | 8543.5 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 36.042 | 0.010 | 1.22 | 10776.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.324 | 0.01 | 96.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.042 | 0.00 | 12.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 16.103 | 0.030 | 0.55 | 4814.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 36.9 | 0.0 | 1.00 |
| sr/grad | gpu | 6.455 | 0.014 | 0.22 | 1929.9 | 2.0 | 1.00 |
| sr/cg | gpu | 263.784 | 273.398 | 8.97 | 78871.3 | 85.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 264.385 | 8.99 | 79051.0 | 0.0 | 52.75 |
| sr/cg/matvec | gpu | 239.660 | 8.741 | 8.15 | 71658.4 | 35.8 | 17.92 |
| sr/cg/precond | gpu | 7.576 | 0.077 | 0.26 | 2265.4 | 16.9 | 16.92 |
| sr/trust | gpu | 13.845 | 13.845 | 0.47 | 4139.6 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 13.076 | 0.410 | 0.44 | 3909.8 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 13.426 | 0.46 | 4014.4 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 14.6 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.010 | 0.00 | 3.1 | 0.0 | 1.00 |

### prof 2026-09-22 13:24:42 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 399, mean 3496.29 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.064 | 0.00 | 25.7 | 0.0 | 1.00 |
| net_fwd | gpu | 11.210 | 0.115 | 0.32 | 4472.8 | 14.0 | 1.00 |
| assemble | gpu | 0.137 | 0.005 | 0.00 | 54.8 | 1.0 | 1.00 |
| lu | gpu | 1.183 | 0.008 | 0.03 | 472.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 7.1 | 1.0 | 1.00 |
| therm_sweeps | gpu | 810.084 | 822.511 | 23.17 | 323223.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 625.552 | 0.123 | 17.89 | 249595.2 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 119.949 | 0.414 | 3.43 | 47859.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.098 | 0.013 | 0.00 | 39.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 116.592 | 0.293 | 3.33 | 46520.2 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.157 | 0.013 | 0.00 | 62.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.625 | 0.013 | 0.02 | 249.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 2.406 | 0.021 | 0.07 | 960.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 12.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 64.461 | 0.083 | 1.84 | 25720.1 | 36.0 | 3.00 |
| record_sweeps | gpu | 1620.575 | 1620.623 | 46.35 | 646609.3 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1250.304 | 0.243 | 35.76 | 498871.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 241.521 | 0.849 | 6.91 | 96367.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.224 | 0.029 | 0.01 | 89.4 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 233.678 | 0.602 | 6.68 | 93237.4 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.217 | 0.026 | 0.01 | 86.8 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.915 | 0.026 | 0.03 | 365.0 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 6.347 | 0.044 | 0.18 | 2532.5 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 24.3 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 127.184 | 0.165 | 3.64 | 50746.2 | 72.0 | 6.00 |
| record | gpu | 618.279 | 618.308 | 17.68 | 246693.2 | 236.0 | 2.00 |
| record/eval_cached | gpu | 25.110 | 0.455 | 0.72 | 10019.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 22.424 | 0.357 | 0.64 | 8947.0 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.323 | 0.010 | 0.01 | 128.8 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.101 | 0.018 | 0.03 | 439.3 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 13.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.174 | 0.012 | 0.03 | 468.3 | 2.0 | 2.00 |
| record/jet_pass | gpu | 324.907 | 349.599 | 9.29 | 129638.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 286.357 | 0.210 | 8.19 | 114256.4 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 34.515 | 0.010 | 0.99 | 13771.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 4.022 | 349.361 | 0.12 | 1604.8 | 8.0 | 2.00 |
| record/exchange | gpu | 144.850 | 146.245 | 4.14 | 57795.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 79.173 | 0.295 | 2.26 | 31590.0 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.010 | 0.00 | 32.3 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 77.009 | 0.209 | 2.20 | 30726.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.010 | 0.00 | 24.7 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.832 | 0.009 | 0.02 | 331.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.144 | 0.015 | 0.03 | 456.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 7.5 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.029 | 0.015 | 0.00 | 11.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 59.608 | 0.146 | 1.70 | 23783.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.979 | 0.018 | 0.17 | 2385.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 145.743 | 4.17 | 58151.5 | 0.0 | 2.00 |
| record/assemble | gpu | 1.625 | 0.019 | 0.05 | 648.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 10.3 | 4.0 | 2.00 |
| record/o_assemble | gpu | 120.364 | 120.318 | 3.44 | 48025.2 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 1.380 | 0.04 | 550.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 1.823 | 0.035 | 0.05 | 727.4 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 41.288 | 0.249 | 1.18 | 16473.8 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 33.247 | 0.176 | 0.95 | 13265.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 42.481 | 0.009 | 1.22 | 16949.7 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.709 | 0.02 | 283.0 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.042 | 0.00 | 16.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 19.183 | 0.030 | 0.55 | 7654.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 49.1 | 0.0 | 1.00 |
| sr/grad | gpu | 7.694 | 0.014 | 0.22 | 3070.1 | 2.0 | 1.00 |
| sr/cg | gpu | 389.220 | 400.769 | 11.13 | 155298.6 | 98.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 380.951 | 10.90 | 151999.4 | 0.0 | 60.71 |
| sr/cg/matvec | gpu | 334.797 | 19.503 | 9.58 | 133583.9 | 41.1 | 20.57 |
| sr/cg/precond | gpu | 17.312 | 0.090 | 0.50 | 6907.3 | 19.6 | 19.57 |
| sr/trust | gpu | 17.181 | 17.181 | 0.49 | 6855.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 15.676 | 0.753 | 0.45 | 6254.8 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 16.420 | 0.47 | 6551.5 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 19.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.013 | 0.00 | 5.1 | 0.0 | 1.00 |

### prof 2026-09-22 13:29:05 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 499, mean 3323.08 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.063 | 0.00 | 31.7 | 0.0 | 1.00 |
| net_fwd | gpu | 10.633 | 0.115 | 0.32 | 5305.8 | 14.0 | 1.00 |
| assemble | gpu | 0.135 | 0.005 | 0.00 | 67.1 | 1.0 | 1.00 |
| lu | gpu | 1.063 | 0.008 | 0.03 | 530.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 8.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 767.446 | 779.172 | 23.09 | 382955.6 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 592.724 | 0.123 | 17.84 | 295769.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 113.578 | 0.415 | 3.42 | 56675.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.097 | 0.013 | 0.00 | 48.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 110.436 | 0.294 | 3.32 | 55107.4 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.144 | 0.013 | 0.00 | 71.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.575 | 0.013 | 0.02 | 286.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 2.256 | 0.021 | 0.07 | 1125.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 15.2 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 61.025 | 0.083 | 1.84 | 30451.5 | 36.0 | 3.00 |
| record_sweeps | gpu | 1534.084 | 1534.124 | 46.16 | 765508.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1183.748 | 0.243 | 35.62 | 590690.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 228.393 | 0.847 | 6.87 | 113968.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.216 | 0.029 | 0.01 | 107.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 221.206 | 0.601 | 6.66 | 110381.8 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.211 | 0.026 | 0.01 | 105.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.882 | 0.026 | 0.03 | 440.0 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 5.738 | 0.044 | 0.17 | 2863.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 30.4 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 120.671 | 0.165 | 3.63 | 60214.6 | 72.0 | 6.00 |
| record | gpu | 584.413 | 584.437 | 17.59 | 291621.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 23.893 | 0.449 | 0.72 | 11922.7 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 21.221 | 0.352 | 0.64 | 10589.1 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.308 | 0.010 | 0.01 | 153.5 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.101 | 0.018 | 0.03 | 549.5 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 17.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.174 | 0.012 | 0.04 | 585.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 307.284 | 330.756 | 9.25 | 153334.7 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 271.077 | 0.210 | 8.16 | 135267.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 32.728 | 0.010 | 0.98 | 16331.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 3.465 | 330.519 | 0.10 | 1729.1 | 8.0 | 2.00 |
| record/exchange | gpu | 137.238 | 138.354 | 4.13 | 68481.7 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 75.068 | 0.294 | 2.26 | 37459.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.010 | 0.00 | 40.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 73.029 | 0.208 | 2.20 | 36441.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 30.9 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.715 | 0.009 | 0.02 | 356.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.136 | 0.015 | 0.03 | 566.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 9.3 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.015 | 0.00 | 14.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 56.413 | 0.146 | 1.70 | 28150.1 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.666 | 0.018 | 0.17 | 2827.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 137.853 | 4.15 | 68788.5 | 0.0 | 2.00 |
| record/assemble | gpu | 1.354 | 0.019 | 0.04 | 675.7 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 12.9 | 4.0 | 2.00 |
| record/o_assemble | gpu | 113.495 | 113.458 | 3.42 | 56633.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 1.107 | 0.03 | 552.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 1.523 | 0.035 | 0.05 | 759.9 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 39.259 | 0.248 | 1.18 | 19590.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 31.460 | 0.176 | 0.95 | 15698.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 40.010 | 0.009 | 1.20 | 19965.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.580 | 0.02 | 289.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.041 | 0.00 | 20.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 18.143 | 0.029 | 0.55 | 9053.5 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 61.5 | 0.0 | 1.00 |
| sr/grad | gpu | 7.276 | 0.014 | 0.22 | 3630.8 | 2.0 | 1.00 |
| sr/cg | gpu | 382.662 | 393.572 | 11.52 | 190948.6 | 111.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 377.330 | 11.35 | 188287.7 | 0.0 | 68.51 |
| sr/cg/matvec | gpu | 338.632 | 15.888 | 10.19 | 168977.3 | 46.3 | 23.17 |
| sr/cg/precond | gpu | 13.881 | 0.102 | 0.42 | 6926.7 | 22.2 | 22.17 |
| sr/trust | gpu | 15.866 | 15.866 | 0.48 | 7917.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 14.649 | 0.611 | 0.44 | 7309.7 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 15.247 | 0.46 | 7608.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 24.4 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.012 | 0.00 | 5.9 | 0.0 | 1.00 |

### prof 2026-09-22 13:33:35 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 599, mean 3219.00 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.061 | 0.00 | 36.8 | 0.0 | 1.00 |
| net_fwd | gpu | 10.248 | 0.114 | 0.32 | 6138.7 | 14.0 | 1.00 |
| assemble | gpu | 0.133 | 0.005 | 0.00 | 79.5 | 1.0 | 1.00 |
| lu | gpu | 0.983 | 0.008 | 0.03 | 588.5 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 10.6 | 1.0 | 1.00 |
| therm_sweeps | gpu | 739.012 | 750.271 | 22.96 | 442668.4 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 570.834 | 0.123 | 17.73 | 341929.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 109.330 | 0.415 | 3.40 | 65488.7 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.096 | 0.013 | 0.00 | 57.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 106.331 | 0.293 | 3.30 | 63692.2 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.135 | 0.013 | 0.00 | 81.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.541 | 0.013 | 0.02 | 324.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 2.155 | 0.021 | 0.07 | 1291.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 18.2 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 58.732 | 0.083 | 1.82 | 35180.4 | 36.0 | 3.00 |
| record_sweeps | gpu | 1476.405 | 1476.440 | 45.87 | 884366.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1139.364 | 0.242 | 35.39 | 682479.0 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 219.638 | 0.844 | 6.82 | 131563.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.211 | 0.029 | 0.01 | 126.1 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 212.888 | 0.599 | 6.61 | 127520.1 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.207 | 0.026 | 0.01 | 123.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.860 | 0.026 | 0.03 | 515.0 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 5.332 | 0.043 | 0.17 | 3194.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 36.6 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 116.326 | 0.165 | 3.61 | 69679.0 | 72.0 | 6.00 |
| record | gpu | 561.839 | 561.860 | 17.45 | 336541.3 | 236.0 | 2.00 |
| record/eval_cached | gpu | 23.081 | 0.442 | 0.72 | 13825.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 20.418 | 0.346 | 0.63 | 12230.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.297 | 0.010 | 0.01 | 178.2 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.101 | 0.017 | 0.03 | 659.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 20.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 703.6 | 2.0 | 2.00 |
| record/jet_pass | gpu | 295.536 | 318.197 | 9.18 | 177026.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 260.890 | 0.209 | 8.10 | 156273.4 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 31.538 | 0.010 | 0.98 | 18891.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 3.094 | 317.961 | 0.10 | 1853.2 | 8.0 | 2.00 |
| record/exchange | gpu | 132.162 | 133.093 | 4.11 | 79165.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 72.331 | 0.292 | 2.25 | 43326.2 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 48.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 70.375 | 0.207 | 2.19 | 42154.5 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 37.1 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.638 | 0.009 | 0.02 | 381.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.130 | 0.015 | 0.04 | 676.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 11.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.015 | 0.00 | 17.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 54.283 | 0.146 | 1.69 | 32515.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.458 | 0.018 | 0.17 | 3269.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 132.594 | 4.12 | 79423.8 | 0.0 | 2.00 |
| record/assemble | gpu | 1.173 | 0.018 | 0.04 | 702.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 15.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 108.920 | 108.889 | 3.38 | 65243.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.925 | 0.03 | 553.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 1.323 | 0.034 | 0.04 | 792.4 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 37.907 | 0.246 | 1.18 | 22706.4 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 30.267 | 0.176 | 0.94 | 18130.0 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 38.368 | 0.009 | 1.19 | 22982.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.494 | 0.02 | 296.1 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.040 | 0.00 | 24.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 17.448 | 0.029 | 0.54 | 10451.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 73.6 | 0.0 | 1.00 |
| sr/grad | gpu | 6.996 | 0.014 | 0.22 | 4190.8 | 2.0 | 1.00 |
| sr/cg | gpu | 389.739 | 400.223 | 12.11 | 233453.5 | 125.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 386.306 | 12.00 | 231397.3 | 0.0 | 76.93 |
| sr/cg/matvec | gpu | 352.550 | 13.523 | 10.95 | 211177.7 | 52.0 | 25.98 |
| sr/cg/precond | gpu | 11.602 | 0.114 | 0.36 | 6949.8 | 25.0 | 24.98 |
| sr/trust | gpu | 14.991 | 14.991 | 0.47 | 8979.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 13.965 | 0.516 | 0.43 | 8365.0 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 14.466 | 0.45 | 8665.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 29.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.011 | 0.00 | 6.6 | 0.0 | 1.00 |

### prof 2026-09-22 13:38:14 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 699, mean 3157.95 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.061 | 0.00 | 42.8 | 0.0 | 1.00 |
| net_fwd | gpu | 9.978 | 0.114 | 0.32 | 6974.6 | 14.0 | 1.00 |
| assemble | gpu | 0.131 | 0.005 | 0.00 | 91.9 | 1.0 | 1.00 |
| lu | gpu | 0.925 | 0.008 | 0.03 | 646.9 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 12.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 718.739 | 729.668 | 22.76 | 502398.2 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 555.235 | 0.123 | 17.58 | 388109.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 106.294 | 0.414 | 3.37 | 74299.8 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.095 | 0.013 | 0.00 | 66.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 103.397 | 0.293 | 3.27 | 72274.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.129 | 0.013 | 0.00 | 90.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.518 | 0.013 | 0.02 | 361.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 2.084 | 0.021 | 0.07 | 1456.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 21.3 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 57.094 | 0.083 | 1.81 | 39908.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1435.212 | 1435.244 | 45.45 | 1003213.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1107.668 | 0.242 | 35.08 | 774260.2 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 213.383 | 0.843 | 6.76 | 149154.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.207 | 0.028 | 0.01 | 144.6 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 206.945 | 0.598 | 6.55 | 144654.8 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.204 | 0.026 | 0.01 | 142.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.844 | 0.026 | 0.03 | 589.9 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 5.043 | 0.043 | 0.16 | 3524.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 42.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 113.223 | 0.164 | 3.59 | 79142.7 | 72.0 | 6.00 |
| record | gpu | 545.725 | 545.744 | 17.28 | 381461.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 22.501 | 0.440 | 0.71 | 15728.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 19.845 | 0.345 | 0.63 | 13871.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.290 | 0.010 | 0.01 | 202.9 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.03 | 770.0 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 24.2 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 821.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 287.150 | 309.230 | 9.09 | 200718.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 253.619 | 0.208 | 8.03 | 177279.8 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 30.689 | 0.010 | 0.97 | 21451.4 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.829 | 308.994 | 0.09 | 1977.4 | 8.0 | 2.00 |
| record/exchange | gpu | 128.537 | 129.335 | 4.07 | 89847.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 70.375 | 0.292 | 2.23 | 49192.3 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 56.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 68.479 | 0.207 | 2.17 | 47866.6 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 43.3 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.582 | 0.009 | 0.02 | 406.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.126 | 0.015 | 0.04 | 787.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 13.1 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.015 | 0.00 | 20.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 52.762 | 0.145 | 1.67 | 36880.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.309 | 0.018 | 0.17 | 3711.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 128.837 | 4.08 | 90057.1 | 0.0 | 2.00 |
| record/assemble | gpu | 1.044 | 0.018 | 0.03 | 730.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 18.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 105.655 | 105.629 | 3.35 | 73852.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.795 | 0.03 | 555.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 1.180 | 0.034 | 0.04 | 824.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 36.942 | 0.246 | 1.17 | 25822.3 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 29.416 | 0.176 | 0.93 | 20562.1 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 37.196 | 0.009 | 1.18 | 26000.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.433 | 0.01 | 302.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.040 | 0.00 | 28.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 16.953 | 0.028 | 0.54 | 11850.0 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 85.9 | 0.0 | 1.00 |
| sr/grad | gpu | 6.796 | 0.014 | 0.22 | 4750.7 | 2.0 | 1.00 |
| sr/cg | gpu | 408.016 | 418.196 | 12.92 | 285203.3 | 142.1 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 405.866 | 12.85 | 283700.5 | 0.0 | 86.68 |
| sr/cg/matvec | gpu | 375.616 | 11.889 | 11.89 | 262555.3 | 58.5 | 29.23 |
| sr/cg/precond | gpu | 9.983 | 0.128 | 0.32 | 6978.0 | 28.2 | 28.23 |
| sr/trust | gpu | 14.363 | 14.363 | 0.45 | 10039.7 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 13.474 | 0.448 | 0.43 | 9418.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 13.906 | 0.44 | 9720.4 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 34.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.011 | 0.00 | 7.4 | 0.0 | 1.00 |

### prof 2026-09-22 13:43:03 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 799, mean 3124.75 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.061 | 0.00 | 49.1 | 0.0 | 1.00 |
| net_fwd | gpu | 9.779 | 0.115 | 0.31 | 7813.3 | 14.0 | 1.00 |
| assemble | gpu | 0.130 | 0.005 | 0.00 | 104.2 | 1.0 | 1.00 |
| lu | gpu | 0.883 | 0.008 | 0.03 | 705.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 14.3 | 1.0 | 1.00 |
| therm_sweeps | gpu | 703.550 | 714.236 | 22.52 | 562136.2 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 543.549 | 0.122 | 17.39 | 434295.7 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 104.019 | 0.413 | 3.33 | 83111.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.095 | 0.013 | 0.00 | 75.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 101.199 | 0.292 | 3.24 | 80857.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.125 | 0.013 | 0.00 | 99.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.500 | 0.013 | 0.02 | 399.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 2.030 | 0.021 | 0.06 | 1622.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 24.3 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 55.867 | 0.083 | 1.79 | 44637.9 | 36.0 | 3.00 |
| record_sweeps | gpu | 1404.335 | 1404.364 | 44.94 | 1122063.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1083.913 | 0.241 | 34.69 | 866046.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 208.694 | 0.841 | 6.68 | 166746.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.204 | 0.028 | 0.01 | 162.9 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 202.491 | 0.596 | 6.48 | 161790.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.202 | 0.026 | 0.01 | 161.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.832 | 0.026 | 0.03 | 664.9 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.826 | 0.043 | 0.15 | 3855.6 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 48.8 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 110.894 | 0.164 | 3.55 | 88604.4 | 72.0 | 6.00 |
| record | gpu | 533.648 | 533.665 | 17.08 | 426384.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 22.068 | 0.438 | 0.71 | 17632.1 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 19.416 | 0.343 | 0.62 | 15513.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.285 | 0.010 | 0.01 | 227.6 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.04 | 880.5 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.010 | 0.00 | 27.7 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 938.8 | 2.0 | 2.00 |
| record/jet_pass | gpu | 280.865 | 302.510 | 8.99 | 224411.1 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 248.169 | 0.208 | 7.94 | 198287.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 30.052 | 0.010 | 0.96 | 24011.7 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.630 | 302.275 | 0.08 | 2101.6 | 8.0 | 2.00 |
| record/exchange | gpu | 125.820 | 126.519 | 4.03 | 100530.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 68.909 | 0.291 | 2.21 | 55058.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 64.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 67.057 | 0.206 | 2.15 | 53578.8 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 49.5 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.541 | 0.009 | 0.02 | 431.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.123 | 0.015 | 0.04 | 897.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 14.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.015 | 0.00 | 23.6 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 51.623 | 0.145 | 1.65 | 41246.5 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.198 | 0.018 | 0.17 | 4153.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 126.023 | 4.03 | 100692.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.948 | 0.018 | 0.03 | 757.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 20.5 | 4.0 | 2.00 |
| record/o_assemble | gpu | 103.207 | 103.184 | 3.30 | 82462.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.697 | 0.02 | 557.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 1.073 | 0.034 | 0.03 | 857.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 36.218 | 0.245 | 1.16 | 28938.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 28.778 | 0.175 | 0.92 | 22993.7 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 36.318 | 0.009 | 1.16 | 29017.7 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.387 | 0.01 | 309.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.040 | 0.00 | 32.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 16.581 | 0.028 | 0.53 | 13248.6 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 97.7 | 0.0 | 1.00 |
| sr/grad | gpu | 6.648 | 0.014 | 0.21 | 5311.5 | 2.0 | 1.00 |
| sr/cg | gpu | 434.272 | 444.225 | 13.90 | 346983.2 | 160.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 433.020 | 13.86 | 345983.3 | 0.0 | 97.52 |
| sr/cg/matvec | gpu | 405.374 | 10.712 | 12.97 | 323893.6 | 65.7 | 32.84 |
| sr/cg/precond | gpu | 8.775 | 0.143 | 0.28 | 7011.6 | 31.8 | 31.84 |
| sr/trust | gpu | 13.894 | 13.894 | 0.44 | 11101.6 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 13.108 | 0.398 | 0.42 | 10473.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 13.488 | 0.43 | 10777.0 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 38.9 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.010 | 0.00 | 8.1 | 0.0 | 1.00 |

### prof 2026-09-22 13:48:06 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 899, mean 3113.48 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.061 | 0.00 | 54.8 | 0.0 | 1.00 |
| net_fwd | gpu | 9.642 | 0.115 | 0.31 | 8668.3 | 14.0 | 1.00 |
| assemble | gpu | 0.130 | 0.005 | 0.00 | 116.6 | 1.0 | 1.00 |
| lu | gpu | 0.851 | 0.008 | 0.03 | 765.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 16.1 | 1.0 | 1.00 |
| therm_sweeps | gpu | 691.819 | 702.335 | 22.22 | 621944.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 534.540 | 0.122 | 17.17 | 480551.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 102.253 | 0.413 | 3.28 | 91925.0 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.095 | 0.013 | 0.00 | 85.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 99.491 | 0.292 | 3.20 | 89442.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.121 | 0.013 | 0.00 | 108.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.486 | 0.013 | 0.02 | 436.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.989 | 0.021 | 0.06 | 1787.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 27.4 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 54.913 | 0.083 | 1.76 | 49366.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1380.347 | 1380.373 | 44.33 | 1240932.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1065.459 | 0.241 | 34.22 | 957847.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 205.050 | 0.840 | 6.59 | 184340.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.202 | 0.028 | 0.01 | 181.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 199.029 | 0.596 | 6.39 | 178927.3 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.200 | 0.026 | 0.01 | 179.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.823 | 0.026 | 0.03 | 739.8 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.657 | 0.043 | 0.15 | 4186.4 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 54.9 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 109.085 | 0.164 | 3.50 | 98067.4 | 72.0 | 6.00 |
| record | gpu | 524.258 | 524.274 | 16.84 | 471308.0 | 236.0 | 2.00 |
| record/eval_cached | gpu | 21.730 | 0.436 | 0.70 | 19535.2 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 19.082 | 0.341 | 0.61 | 17155.1 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.281 | 0.010 | 0.01 | 252.3 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.04 | 990.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 31.2 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 1056.5 | 2.0 | 2.00 |
| record/jet_pass | gpu | 275.978 | 297.286 | 8.86 | 248104.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 243.932 | 0.208 | 7.83 | 219294.8 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 29.557 | 0.010 | 0.95 | 26572.1 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.476 | 297.052 | 0.08 | 2225.7 | 8.0 | 2.00 |
| record/exchange | gpu | 123.709 | 124.331 | 3.97 | 111214.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 67.770 | 0.290 | 2.18 | 60925.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 72.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 65.953 | 0.205 | 2.12 | 59291.8 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 55.7 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.508 | 0.009 | 0.02 | 456.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.121 | 0.015 | 0.04 | 1007.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 16.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.015 | 0.00 | 26.6 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 50.737 | 0.145 | 1.63 | 45612.3 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.112 | 0.018 | 0.16 | 4595.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 123.835 | 3.98 | 111328.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.873 | 0.018 | 0.03 | 784.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 23.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 101.304 | 101.284 | 3.25 | 91072.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.622 | 0.02 | 558.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.990 | 0.034 | 0.03 | 889.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 35.656 | 0.245 | 1.15 | 32055.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 28.282 | 0.175 | 0.91 | 25425.6 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 35.634 | 0.009 | 1.14 | 32035.4 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.351 | 0.01 | 315.7 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.040 | 0.00 | 36.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 16.292 | 0.028 | 0.52 | 14646.6 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 110.1 | 0.0 | 1.00 |
| sr/grad | gpu | 6.530 | 0.014 | 0.21 | 5870.8 | 2.0 | 1.00 |
| sr/cg | gpu | 469.106 | 478.882 | 15.07 | 421726.5 | 181.0 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 468.469 | 15.05 | 421153.8 | 0.0 | 110.02 |
| sr/cg/matvec | gpu | 442.822 | 9.859 | 14.22 | 398097.0 | 74.0 | 37.01 |
| sr/cg/precond | gpu | 7.845 | 0.162 | 0.25 | 7053.0 | 36.0 | 36.01 |
| sr/trust | gpu | 13.530 | 13.530 | 0.43 | 12163.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 12.824 | 0.358 | 0.41 | 11528.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 13.163 | 0.42 | 11833.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 43.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.010 | 0.00 | 8.9 | 0.0 | 1.00 |

### prof 2026-09-22 13:53:22 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 999, mean 3118.55 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.060 | 0.00 | 60.2 | 0.0 | 1.00 |
| net_fwd | gpu | 9.558 | 0.114 | 0.31 | 9548.0 | 14.0 | 1.00 |
| assemble | gpu | 0.129 | 0.005 | 0.00 | 129.0 | 1.0 | 1.00 |
| lu | gpu | 0.827 | 0.008 | 0.03 | 826.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 18.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 682.515 | 692.923 | 21.89 | 681832.3 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 527.411 | 0.122 | 16.91 | 526883.1 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 100.842 | 0.413 | 3.23 | 100741.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.094 | 0.013 | 0.00 | 94.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 98.129 | 0.292 | 3.15 | 98030.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.118 | 0.013 | 0.00 | 118.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.475 | 0.013 | 0.02 | 474.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.955 | 0.021 | 0.06 | 1953.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 30.5 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 54.150 | 0.083 | 1.74 | 54095.9 | 36.0 | 3.00 |
| record_sweeps | gpu | 1361.184 | 1361.208 | 43.65 | 1359822.7 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1050.717 | 0.241 | 33.69 | 1049666.2 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 202.141 | 0.839 | 6.48 | 201939.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.200 | 0.028 | 0.01 | 199.6 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 196.266 | 0.595 | 6.29 | 196069.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.198 | 0.026 | 0.01 | 198.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.816 | 0.026 | 0.03 | 814.8 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.522 | 0.043 | 0.14 | 4517.2 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 60.9 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 107.637 | 0.164 | 3.45 | 107529.7 | 72.0 | 6.00 |
| record | gpu | 516.751 | 516.766 | 16.57 | 516234.6 | 236.0 | 2.00 |
| record/eval_cached | gpu | 21.460 | 0.433 | 0.69 | 21438.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 18.816 | 0.339 | 0.60 | 18796.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.277 | 0.009 | 0.01 | 277.0 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.04 | 1101.0 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 34.6 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 1174.1 | 2.0 | 2.00 |
| record/jet_pass | gpu | 272.073 | 293.111 | 8.72 | 271800.7 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 240.546 | 0.207 | 7.71 | 240305.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 29.161 | 0.009 | 0.94 | 29132.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.352 | 292.877 | 0.08 | 2349.9 | 8.0 | 2.00 |
| record/exchange | gpu | 122.022 | 122.582 | 3.91 | 121899.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 66.861 | 0.290 | 2.14 | 66793.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 80.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 65.071 | 0.205 | 2.09 | 65005.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 61.9 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.482 | 0.009 | 0.02 | 481.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.119 | 0.015 | 0.04 | 1117.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 18.6 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 29.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 50.029 | 0.145 | 1.60 | 49978.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 5.043 | 0.018 | 0.16 | 5037.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 122.087 | 3.91 | 121964.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.812 | 0.018 | 0.03 | 811.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 25.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 99.781 | 99.763 | 3.20 | 99681.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.561 | 0.02 | 560.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.923 | 0.034 | 0.03 | 922.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 35.206 | 0.244 | 1.13 | 35171.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 27.886 | 0.175 | 0.89 | 27858.0 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 35.087 | 0.009 | 1.13 | 35052.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.323 | 0.01 | 322.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.040 | 0.00 | 39.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 16.061 | 0.028 | 0.52 | 16045.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 122.4 | 0.0 | 1.00 |
| sr/grad | gpu | 6.438 | 0.013 | 0.21 | 6431.3 | 2.0 | 1.00 |
| sr/cg | gpu | 510.926 | 520.561 | 16.38 | 510415.5 | 204.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 510.703 | 16.38 | 510191.9 | 0.0 | 123.95 |
| sr/cg/matvec | gpu | 486.630 | 9.234 | 15.60 | 486143.3 | 83.3 | 41.65 |
| sr/cg/precond | gpu | 7.110 | 0.183 | 0.23 | 7102.6 | 40.6 | 40.65 |
| sr/trust | gpu | 13.242 | 13.242 | 0.42 | 13228.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 12.599 | 0.327 | 0.40 | 12586.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.907 | 0.41 | 12893.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 48.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.010 | 0.00 | 9.6 | 0.0 | 1.00 |

### prof 2026-09-22 13:58:56 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1099, mean 3138.28 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.060 | 0.00 | 66.0 | 0.0 | 1.00 |
| net_fwd | gpu | 9.460 | 0.114 | 0.30 | 10396.4 | 14.0 | 1.00 |
| assemble | gpu | 0.129 | 0.005 | 0.00 | 141.7 | 1.0 | 1.00 |
| lu | gpu | 0.806 | 0.008 | 0.03 | 885.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 19.8 | 1.0 | 1.00 |
| therm_sweeps | gpu | 674.833 | 685.121 | 21.50 | 741641.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 521.507 | 0.122 | 16.62 | 573135.9 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 99.690 | 0.413 | 3.18 | 109559.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.094 | 0.013 | 0.00 | 103.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 97.015 | 0.292 | 3.09 | 106620.0 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.116 | 0.013 | 0.00 | 127.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.466 | 0.013 | 0.01 | 511.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.928 | 0.021 | 0.06 | 2118.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 33.5 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 53.525 | 0.083 | 1.71 | 58823.9 | 36.0 | 3.00 |
| record_sweeps | gpu | 1345.544 | 1345.567 | 42.88 | 1478752.9 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1038.690 | 0.241 | 33.10 | 1141520.6 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 199.767 | 0.839 | 6.37 | 219543.7 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.198 | 0.028 | 0.01 | 217.9 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 194.010 | 0.595 | 6.18 | 213217.5 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.197 | 0.026 | 0.01 | 216.8 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.810 | 0.025 | 0.03 | 889.7 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.411 | 0.043 | 0.14 | 4848.0 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 67.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 106.452 | 0.164 | 3.39 | 116990.7 | 72.0 | 6.00 |
| record | gpu | 510.614 | 510.628 | 16.27 | 561164.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 21.239 | 0.432 | 0.68 | 23342.2 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 18.598 | 0.337 | 0.59 | 20439.1 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.274 | 0.009 | 0.01 | 301.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.04 | 1211.2 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 38.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 1291.7 | 2.0 | 2.00 |
| record/jet_pass | gpu | 268.879 | 289.698 | 8.57 | 295498.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 237.777 | 0.207 | 7.58 | 261316.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 28.838 | 0.009 | 0.92 | 31692.4 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.251 | 289.463 | 0.07 | 2474.1 | 8.0 | 2.00 |
| record/exchange | gpu | 120.644 | 121.153 | 3.84 | 132587.2 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 66.119 | 0.289 | 2.11 | 72664.3 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 88.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 64.352 | 0.205 | 2.05 | 70722.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 68.1 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.461 | 0.009 | 0.01 | 506.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.117 | 0.015 | 0.04 | 1228.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 20.5 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 32.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 49.449 | 0.145 | 1.58 | 54344.9 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.986 | 0.018 | 0.16 | 5479.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 120.659 | 3.84 | 132604.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.763 | 0.018 | 0.02 | 838.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 28.1 | 4.0 | 2.00 |
| record/o_assemble | gpu | 98.535 | 98.518 | 3.14 | 108289.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.511 | 0.02 | 562.1 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.869 | 0.034 | 0.03 | 954.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 34.838 | 0.244 | 1.11 | 38287.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 27.562 | 0.175 | 0.88 | 30290.7 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 34.638 | 0.009 | 1.10 | 38067.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.299 | 0.01 | 328.7 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.039 | 0.00 | 43.3 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.873 | 0.028 | 0.51 | 17444.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.123 | 0.00 | 134.7 | 0.0 | 1.00 |
| sr/grad | gpu | 6.362 | 0.013 | 0.20 | 6991.3 | 2.0 | 1.00 |
| sr/cg | gpu | 560.772 | 570.291 | 17.87 | 616288.2 | 230.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 560.801 | 17.87 | 616320.3 | 0.0 | 139.75 |
| sr/cg/matvec | gpu | 537.986 | 8.786 | 17.14 | 591247.0 | 93.8 | 46.92 |
| sr/cg/precond | gpu | 6.517 | 0.206 | 0.21 | 7162.2 | 45.9 | 45.92 |
| sr/trust | gpu | 13.004 | 13.004 | 0.41 | 14291.6 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 12.414 | 0.301 | 0.40 | 13642.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.695 | 0.40 | 13951.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 53.4 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 10.4 | 0.0 | 1.00 |

### prof 2026-09-22 14:04:38 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1199, mean 3162.05 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.060 | 0.00 | 71.7 | 0.0 | 1.00 |
| net_fwd | gpu | 9.377 | 0.114 | 0.30 | 11242.4 | 14.0 | 1.00 |
| assemble | gpu | 0.128 | 0.005 | 0.00 | 154.0 | 1.0 | 1.00 |
| lu | gpu | 0.788 | 0.008 | 0.02 | 944.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 21.6 | 1.0 | 1.00 |
| therm_sweeps | gpu | 668.444 | 678.630 | 21.14 | 801463.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 516.595 | 0.122 | 16.34 | 619397.0 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 98.734 | 0.413 | 3.12 | 118381.6 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.094 | 0.013 | 0.00 | 112.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 96.091 | 0.292 | 3.04 | 115213.5 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.114 | 0.013 | 0.00 | 136.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.458 | 0.013 | 0.01 | 549.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.905 | 0.021 | 0.06 | 2284.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 36.6 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 53.005 | 0.083 | 1.68 | 63552.6 | 36.0 | 3.00 |
| record_sweeps | gpu | 1332.548 | 1332.570 | 42.14 | 1597725.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1028.698 | 0.241 | 32.53 | 1233408.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 197.796 | 0.838 | 6.26 | 237157.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.197 | 0.028 | 0.01 | 236.3 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 192.139 | 0.594 | 6.08 | 230374.4 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.196 | 0.026 | 0.01 | 235.4 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.805 | 0.025 | 0.03 | 964.6 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.319 | 0.043 | 0.14 | 5178.7 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 73.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 105.464 | 0.164 | 3.34 | 126451.5 | 72.0 | 6.00 |
| record | gpu | 505.503 | 505.516 | 15.99 | 606098.0 | 236.0 | 2.00 |
| record/eval_cached | gpu | 21.057 | 0.430 | 0.67 | 25246.8 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 18.417 | 0.336 | 0.58 | 22082.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.272 | 0.009 | 0.01 | 326.4 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.016 | 0.03 | 1321.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 41.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.175 | 0.011 | 0.04 | 1409.4 | 2.0 | 2.00 |
| record/jet_pass | gpu | 266.217 | 286.853 | 8.42 | 319194.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 235.469 | 0.207 | 7.45 | 282327.8 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 28.568 | 0.009 | 0.90 | 34252.6 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.167 | 286.620 | 0.07 | 2598.2 | 8.0 | 2.00 |
| record/exchange | gpu | 119.498 | 119.966 | 3.78 | 143278.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 65.503 | 0.289 | 2.07 | 78537.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 96.6 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 63.755 | 0.205 | 2.02 | 76441.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 74.3 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.444 | 0.009 | 0.01 | 531.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.116 | 0.015 | 0.04 | 1338.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 22.4 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 35.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 48.967 | 0.145 | 1.55 | 58711.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.939 | 0.018 | 0.16 | 5921.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 119.472 | 3.78 | 143246.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.722 | 0.018 | 0.02 | 866.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 30.7 | 4.0 | 2.00 |
| record/o_assemble | gpu | 97.496 | 97.481 | 3.08 | 116897.8 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.470 | 0.01 | 563.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.823 | 0.034 | 0.03 | 987.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 34.532 | 0.244 | 1.09 | 41403.6 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 27.293 | 0.175 | 0.86 | 32724.6 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 34.263 | 0.009 | 1.08 | 41081.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.280 | 0.01 | 335.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.039 | 0.00 | 47.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.716 | 0.027 | 0.50 | 18842.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 146.7 | 0.0 | 1.00 |
| sr/grad | gpu | 6.299 | 0.013 | 0.20 | 7552.7 | 2.0 | 1.00 |
| sr/cg | gpu | 609.590 | 619.013 | 19.28 | 730898.0 | 255.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 609.794 | 19.28 | 731143.1 | 0.0 | 154.96 |
| sr/cg/matvec | gpu | 588.013 | 8.440 | 18.60 | 705027.3 | 104.0 | 51.99 |
| sr/cg/precond | gpu | 6.027 | 0.229 | 0.19 | 7226.3 | 51.0 | 50.99 |
| sr/trust | gpu | 12.806 | 12.806 | 0.40 | 15354.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 12.259 | 0.279 | 0.39 | 14698.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.518 | 0.40 | 15009.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 58.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 11.1 | 0.0 | 1.00 |

### prof 2026-09-22 14:10:22 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1299, mean 3182.94 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.060 | 0.00 | 78.5 | 0.0 | 1.00 |
| net_fwd | gpu | 9.306 | 0.115 | 0.29 | 12089.1 | 14.0 | 1.00 |
| assemble | gpu | 0.128 | 0.005 | 0.00 | 166.4 | 1.0 | 1.00 |
| lu | gpu | 0.772 | 0.008 | 0.02 | 1003.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 23.4 | 1.0 | 1.00 |
| therm_sweeps | gpu | 663.052 | 673.151 | 20.83 | 861304.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 512.450 | 0.123 | 16.10 | 665673.1 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 97.927 | 0.415 | 3.08 | 127207.0 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.094 | 0.013 | 0.00 | 121.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 95.312 | 0.293 | 2.99 | 123810.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.112 | 0.013 | 0.00 | 146.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.452 | 0.013 | 0.01 | 586.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.886 | 0.021 | 0.06 | 2449.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 39.6 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 52.564 | 0.083 | 1.65 | 68280.6 | 36.0 | 3.00 |
| record_sweeps | gpu | 1321.584 | 1321.605 | 41.52 | 1716737.9 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1020.268 | 0.241 | 32.05 | 1325328.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 196.134 | 0.841 | 6.16 | 254777.6 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.196 | 0.028 | 0.01 | 254.7 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 190.560 | 0.597 | 5.99 | 247537.8 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.196 | 0.026 | 0.01 | 254.0 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.800 | 0.026 | 0.03 | 1039.6 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.241 | 0.043 | 0.13 | 5509.5 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 79.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 104.629 | 0.164 | 3.29 | 135912.6 | 72.0 | 6.00 |
| record | gpu | 501.184 | 501.196 | 15.75 | 651037.6 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.904 | 0.433 | 0.66 | 27154.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 18.266 | 0.338 | 0.57 | 23727.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.270 | 0.009 | 0.01 | 351.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.03 | 1431.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 45.0 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.176 | 0.011 | 0.04 | 1527.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 263.967 | 284.447 | 8.29 | 342893.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 233.518 | 0.208 | 7.34 | 303340.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 28.339 | 0.009 | 0.89 | 36812.7 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.096 | 284.212 | 0.07 | 2722.8 | 8.0 | 2.00 |
| record/exchange | gpu | 118.530 | 118.962 | 3.72 | 153970.8 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 64.983 | 0.290 | 2.04 | 84413.3 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 104.7 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 63.251 | 0.206 | 1.99 | 82163.2 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 80.5 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.429 | 0.009 | 0.01 | 556.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.115 | 0.015 | 0.04 | 1448.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 24.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 38.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 48.558 | 0.145 | 1.53 | 63077.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.899 | 0.018 | 0.15 | 6363.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 118.466 | 3.72 | 153887.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.688 | 0.018 | 0.02 | 893.4 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 33.2 | 4.0 | 2.00 |
| record/o_assemble | gpu | 96.617 | 96.603 | 3.04 | 125505.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.435 | 0.01 | 565.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.785 | 0.034 | 0.02 | 1019.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 34.273 | 0.245 | 1.08 | 44520.3 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 27.066 | 0.176 | 0.85 | 35158.6 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 33.945 | 0.009 | 1.07 | 44094.0 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.263 | 0.01 | 342.0 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.040 | 0.00 | 51.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.584 | 0.028 | 0.49 | 20243.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 159.1 | 0.0 | 1.00 |
| sr/grad | gpu | 6.246 | 0.013 | 0.20 | 8112.9 | 2.0 | 1.00 |
| sr/cg | gpu | 651.621 | 660.963 | 20.47 | 846455.2 | 277.7 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 651.951 | 20.48 | 846884.6 | 0.0 | 168.03 |
| sr/cg/matvec | gpu | 631.053 | 8.162 | 19.83 | 819737.9 | 112.7 | 56.34 |
| sr/cg/precond | gpu | 5.614 | 0.250 | 0.18 | 7292.6 | 55.3 | 55.34 |
| sr/trust | gpu | 12.639 | 12.638 | 0.40 | 16417.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 12.128 | 0.261 | 0.38 | 15754.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.369 | 0.39 | 16066.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 63.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 12.0 | 0.0 | 1.00 |

### prof 2026-09-22 14:16:06 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1399, mean 3201.37 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.060 | 0.00 | 84.0 | 0.0 | 1.00 |
| net_fwd | gpu | 9.246 | 0.115 | 0.29 | 12935.8 | 14.0 | 1.00 |
| assemble | gpu | 0.128 | 0.005 | 0.00 | 178.8 | 1.0 | 1.00 |
| lu | gpu | 0.759 | 0.008 | 0.02 | 1062.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 25.3 | 1.0 | 1.00 |
| therm_sweeps | gpu | 658.440 | 668.466 | 20.57 | 921157.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 508.907 | 0.123 | 15.90 | 711961.3 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 97.237 | 0.415 | 3.04 | 136034.0 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.094 | 0.013 | 0.00 | 131.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 94.645 | 0.293 | 2.96 | 132408.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.111 | 0.013 | 0.00 | 155.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.446 | 0.013 | 0.01 | 624.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.869 | 0.021 | 0.06 | 2615.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.030 | 0.013 | 0.00 | 42.7 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 52.186 | 0.083 | 1.63 | 73008.4 | 36.0 | 3.00 |
| record_sweeps | gpu | 1312.202 | 1312.222 | 40.99 | 1835770.6 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1013.058 | 0.241 | 31.64 | 1417268.2 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 194.711 | 0.841 | 6.08 | 272400.8 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.195 | 0.028 | 0.01 | 273.0 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 189.210 | 0.597 | 5.91 | 264704.4 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.195 | 0.026 | 0.01 | 272.6 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.797 | 0.026 | 0.02 | 1114.5 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.175 | 0.043 | 0.13 | 5840.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 85.3 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 103.911 | 0.164 | 3.25 | 145371.9 | 72.0 | 6.00 |
| record | gpu | 497.479 | 497.491 | 15.54 | 695973.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.772 | 0.431 | 0.65 | 29059.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 18.136 | 0.337 | 0.57 | 25371.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.269 | 0.009 | 0.01 | 375.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.017 | 0.03 | 1541.9 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 48.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.176 | 0.011 | 0.04 | 1644.7 | 2.0 | 2.00 |
| record/jet_pass | gpu | 262.037 | 282.385 | 8.19 | 366589.7 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 231.845 | 0.208 | 7.24 | 324351.5 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 28.143 | 0.009 | 0.88 | 39372.7 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 2.035 | 282.150 | 0.06 | 2846.9 | 8.0 | 2.00 |
| record/exchange | gpu | 117.702 | 118.103 | 3.68 | 164664.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 64.539 | 0.290 | 2.02 | 90289.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 112.8 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 62.820 | 0.205 | 1.96 | 87885.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 86.7 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.416 | 0.009 | 0.01 | 581.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.114 | 0.015 | 0.03 | 1558.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 26.1 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 41.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 48.208 | 0.145 | 1.51 | 67443.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.865 | 0.018 | 0.15 | 6805.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 117.608 | 3.67 | 164532.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.658 | 0.018 | 0.02 | 920.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 35.8 | 4.0 | 2.00 |
| record/o_assemble | gpu | 95.863 | 95.850 | 2.99 | 134112.6 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.405 | 0.01 | 567.1 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.752 | 0.034 | 0.02 | 1052.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 34.050 | 0.245 | 1.06 | 47636.6 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.872 | 0.176 | 0.84 | 37593.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 33.671 | 0.009 | 1.05 | 47105.8 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.249 | 0.01 | 348.5 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.039 | 0.00 | 55.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.470 | 0.028 | 0.48 | 21642.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 171.2 | 0.0 | 1.00 |
| sr/grad | gpu | 6.200 | 0.013 | 0.19 | 8673.3 | 2.0 | 1.00 |
| sr/cg | gpu | 688.148 | 697.422 | 21.50 | 962719.6 | 296.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 688.599 | 21.51 | 963349.4 | 0.0 | 179.38 |
| sr/cg/matvec | gpu | 668.448 | 7.917 | 20.88 | 935158.1 | 120.3 | 60.13 |
| sr/cg/precond | gpu | 5.260 | 0.267 | 0.16 | 7358.1 | 59.1 | 59.13 |
| sr/trust | gpu | 12.494 | 12.494 | 0.39 | 17479.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 12.015 | 0.246 | 0.38 | 16809.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.240 | 0.38 | 17123.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 68.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 12.7 | 0.0 | 1.00 |

### prof 2026-09-22 14:21:50 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1499, mean 3217.54 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.060 | 0.00 | 89.3 | 0.0 | 1.00 |
| net_fwd | gpu | 9.195 | 0.115 | 0.29 | 13783.5 | 14.0 | 1.00 |
| assemble | gpu | 0.128 | 0.005 | 0.00 | 191.1 | 1.0 | 1.00 |
| lu | gpu | 0.748 | 0.008 | 0.02 | 1121.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 27.1 | 1.0 | 1.00 |
| therm_sweeps | gpu | 654.455 | 664.419 | 20.34 | 981028.6 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 505.847 | 0.123 | 15.72 | 758264.0 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 96.640 | 0.415 | 3.00 | 144863.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.094 | 0.013 | 0.00 | 140.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 94.069 | 0.293 | 2.92 | 141009.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.110 | 0.013 | 0.00 | 164.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.441 | 0.013 | 0.01 | 661.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.855 | 0.021 | 0.06 | 2781.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 45.7 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 51.859 | 0.083 | 1.61 | 77736.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1304.088 | 1304.107 | 40.53 | 1954827.3 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1006.822 | 0.241 | 31.29 | 1509226.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 193.481 | 0.840 | 6.01 | 290028.7 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.194 | 0.028 | 0.01 | 291.4 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 188.042 | 0.596 | 5.84 | 281875.5 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.194 | 0.026 | 0.01 | 291.2 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.794 | 0.026 | 0.02 | 1189.5 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.117 | 0.043 | 0.13 | 6171.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 91.4 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 103.290 | 0.164 | 3.21 | 154832.4 | 72.0 | 6.00 |
| record | gpu | 494.269 | 494.280 | 15.36 | 740908.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.657 | 0.429 | 0.64 | 30965.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 18.023 | 0.335 | 0.56 | 27016.1 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.267 | 0.009 | 0.01 | 400.4 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.016 | 0.03 | 1652.1 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 51.9 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.176 | 0.011 | 0.04 | 1762.3 | 2.0 | 2.00 |
| record/jet_pass | gpu | 260.364 | 280.598 | 8.09 | 390285.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 230.395 | 0.207 | 7.16 | 345361.5 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.974 | 0.009 | 0.87 | 41932.8 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.982 | 280.364 | 0.06 | 2970.9 | 8.0 | 2.00 |
| record/exchange | gpu | 116.984 | 117.360 | 3.64 | 175359.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 64.155 | 0.290 | 1.99 | 96167.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 120.8 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 62.448 | 0.205 | 1.94 | 93609.6 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 92.9 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.405 | 0.009 | 0.01 | 606.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.113 | 0.015 | 0.03 | 1669.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 28.0 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 44.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 47.905 | 0.145 | 1.49 | 71809.8 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.835 | 0.018 | 0.15 | 7247.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 116.864 | 3.63 | 175179.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.632 | 0.018 | 0.02 | 947.7 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 38.3 | 4.0 | 2.00 |
| record/o_assemble | gpu | 95.209 | 95.197 | 2.96 | 142718.6 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.379 | 0.01 | 568.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.724 | 0.034 | 0.02 | 1084.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.858 | 0.244 | 1.05 | 50752.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.703 | 0.176 | 0.83 | 40028.1 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 33.434 | 0.009 | 1.04 | 50117.0 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.237 | 0.01 | 354.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.039 | 0.00 | 58.5 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.371 | 0.027 | 0.48 | 23040.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 183.2 | 0.0 | 1.00 |
| sr/grad | gpu | 6.160 | 0.013 | 0.19 | 9233.2 | 2.0 | 1.00 |
| sr/cg | gpu | 719.961 | 729.175 | 22.38 | 1079221.7 | 313.1 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 720.516 | 22.39 | 1080053.8 | 0.0 | 189.25 |
| sr/cg/matvec | gpu | 701.011 | 7.704 | 21.79 | 1050815.6 | 126.8 | 63.42 |
| sr/cg/precond | gpu | 4.952 | 0.281 | 0.15 | 7423.5 | 62.4 | 62.42 |
| sr/trust | gpu | 12.369 | 12.369 | 0.38 | 18541.2 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.917 | 0.232 | 0.37 | 17864.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.128 | 0.38 | 18180.4 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 72.8 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 13.5 | 0.0 | 1.00 |

### prof 2026-09-22 14:27:34 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1599, mean 3231.12 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.059 | 0.00 | 94.4 | 0.0 | 1.00 |
| net_fwd | gpu | 9.150 | 0.114 | 0.28 | 14631.6 | 14.0 | 1.00 |
| assemble | gpu | 0.127 | 0.005 | 0.00 | 203.5 | 1.0 | 1.00 |
| lu | gpu | 0.738 | 0.008 | 0.02 | 1180.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 28.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 650.974 | 660.883 | 20.15 | 1040907.8 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 503.173 | 0.122 | 15.57 | 804573.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 96.119 | 0.415 | 2.97 | 153694.1 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 149.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 93.566 | 0.293 | 2.90 | 149611.4 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.109 | 0.013 | 0.00 | 173.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.437 | 0.013 | 0.01 | 699.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.843 | 0.021 | 0.06 | 2946.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 48.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 51.573 | 0.083 | 1.60 | 82465.4 | 36.0 | 3.00 |
| record_sweeps | gpu | 1297.003 | 1297.021 | 40.14 | 2073907.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 1001.379 | 0.241 | 30.99 | 1601204.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 192.408 | 0.839 | 5.95 | 307660.6 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.194 | 0.028 | 0.01 | 309.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 187.024 | 0.595 | 5.79 | 299050.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.194 | 0.026 | 0.01 | 309.8 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.791 | 0.026 | 0.02 | 1264.5 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.066 | 0.043 | 0.13 | 6501.9 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 97.6 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 102.747 | 0.164 | 3.18 | 164292.1 | 72.0 | 6.00 |
| record | gpu | 491.461 | 491.473 | 15.21 | 785846.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.558 | 0.428 | 0.64 | 32871.6 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.924 | 0.334 | 0.55 | 28660.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.266 | 0.009 | 0.01 | 425.2 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.016 | 0.03 | 1762.6 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 55.4 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.176 | 0.011 | 0.04 | 1879.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 258.900 | 279.036 | 8.01 | 413981.4 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 229.126 | 0.207 | 7.09 | 366372.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.826 | 0.009 | 0.86 | 44493.0 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.936 | 278.802 | 0.06 | 3094.9 | 8.0 | 2.00 |
| record/exchange | gpu | 116.358 | 116.710 | 3.60 | 186056.2 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 63.820 | 0.289 | 1.98 | 102047.4 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 128.8 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 62.123 | 0.205 | 1.92 | 99335.2 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 99.1 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.395 | 0.009 | 0.01 | 631.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.113 | 0.015 | 0.03 | 1779.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 29.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 47.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 47.640 | 0.145 | 1.47 | 76176.1 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.809 | 0.018 | 0.15 | 7689.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 116.215 | 3.60 | 185828.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.610 | 0.018 | 0.02 | 974.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 40.9 | 4.0 | 2.00 |
| record/o_assemble | gpu | 94.637 | 94.625 | 2.93 | 151324.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.357 | 0.01 | 570.4 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.699 | 0.034 | 0.02 | 1117.2 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.689 | 0.244 | 1.04 | 53868.9 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.556 | 0.176 | 0.82 | 42463.2 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 33.226 | 0.009 | 1.03 | 53127.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.226 | 0.01 | 361.4 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.039 | 0.00 | 62.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.284 | 0.027 | 0.47 | 24439.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 195.4 | 0.0 | 1.00 |
| sr/grad | gpu | 6.125 | 0.013 | 0.19 | 9793.5 | 2.0 | 1.00 |
| sr/cg | gpu | 747.229 | 756.390 | 23.13 | 1194818.8 | 327.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 747.878 | 23.15 | 1195856.1 | 0.0 | 197.74 |
| sr/cg/matvec | gpu | 728.940 | 7.516 | 22.56 | 1165574.9 | 132.5 | 66.25 |
| sr/cg/precond | gpu | 4.683 | 0.294 | 0.14 | 7488.5 | 65.2 | 65.25 |
| sr/trust | gpu | 12.260 | 12.260 | 0.38 | 19603.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.832 | 0.220 | 0.37 | 18919.7 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 12.031 | 0.37 | 19237.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.049 | 0.00 | 77.6 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 14.2 | 0.0 | 1.00 |

### prof 2026-09-22 14:33:19 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1699, mean 3244.43 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.058 | 0.00 | 99.3 | 0.0 | 1.00 |
| net_fwd | gpu | 9.112 | 0.114 | 0.28 | 15481.4 | 14.0 | 1.00 |
| assemble | gpu | 0.127 | 0.005 | 0.00 | 215.9 | 1.0 | 1.00 |
| lu | gpu | 0.729 | 0.008 | 0.02 | 1239.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 30.7 | 1.0 | 1.00 |
| therm_sweeps | gpu | 647.995 | 657.856 | 19.97 | 1100942.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 500.888 | 0.122 | 15.44 | 851009.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 95.674 | 0.414 | 2.95 | 162549.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 158.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 93.136 | 0.293 | 2.87 | 158237.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.108 | 0.013 | 0.00 | 183.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.434 | 0.013 | 0.01 | 736.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.832 | 0.021 | 0.06 | 3112.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 51.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 51.324 | 0.083 | 1.58 | 87198.9 | 36.0 | 3.00 |
| record_sweeps | gpu | 1290.899 | 1290.918 | 39.79 | 2193237.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 996.696 | 0.241 | 30.72 | 1693386.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 191.486 | 0.839 | 5.90 | 325334.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.193 | 0.028 | 0.01 | 328.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 186.149 | 0.595 | 5.74 | 316267.1 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.193 | 0.026 | 0.01 | 328.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.788 | 0.026 | 0.02 | 1339.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 4.022 | 0.043 | 0.12 | 6833.4 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 103.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 102.270 | 0.164 | 3.15 | 173756.6 | 72.0 | 6.00 |
| record | gpu | 489.047 | 489.058 | 15.07 | 830890.8 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.472 | 0.426 | 0.63 | 34781.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.839 | 0.333 | 0.55 | 30308.4 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.265 | 0.009 | 0.01 | 449.9 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.102 | 0.016 | 0.03 | 1873.1 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 58.9 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.176 | 0.011 | 0.04 | 1997.8 | 2.0 | 2.00 |
| record/jet_pass | gpu | 257.643 | 277.694 | 7.94 | 437735.3 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 228.037 | 0.207 | 7.03 | 387434.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.698 | 0.009 | 0.85 | 47059.1 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.895 | 277.460 | 0.06 | 3218.8 | 8.0 | 2.00 |
| record/exchange | gpu | 115.821 | 116.152 | 3.57 | 196779.2 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 63.532 | 0.289 | 1.96 | 107941.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 136.9 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 61.845 | 0.205 | 1.91 | 105074.6 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 105.3 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.387 | 0.009 | 0.01 | 656.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.112 | 0.015 | 0.03 | 1889.8 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 31.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 50.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 47.413 | 0.145 | 1.46 | 80553.9 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.787 | 0.018 | 0.15 | 8132.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 115.658 | 3.56 | 196503.4 | 0.0 | 2.00 |
| record/assemble | gpu | 0.590 | 0.018 | 0.02 | 1002.0 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 43.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 94.143 | 94.132 | 2.90 | 159948.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.337 | 0.01 | 572.0 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.677 | 0.033 | 0.02 | 1149.7 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.545 | 0.243 | 1.03 | 56993.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.430 | 0.176 | 0.81 | 44904.9 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 33.044 | 0.009 | 1.02 | 56141.9 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.216 | 0.01 | 367.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 65.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.208 | 0.027 | 0.47 | 25838.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 207.3 | 0.0 | 1.00 |
| sr/grad | gpu | 6.094 | 0.013 | 0.19 | 10353.5 | 2.0 | 1.00 |
| sr/cg | gpu | 772.301 | 781.416 | 23.80 | 1312139.2 | 340.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 773.029 | 23.83 | 1313377.1 | 0.0 | 205.50 |
| sr/cg/matvec | gpu | 754.589 | 7.353 | 23.26 | 1282046.2 | 137.7 | 68.83 |
| sr/cg/precond | gpu | 4.446 | 0.305 | 0.14 | 7554.2 | 67.8 | 67.83 |
| sr/trust | gpu | 12.164 | 12.164 | 0.37 | 20666.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.757 | 0.210 | 0.36 | 19975.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.945 | 0.37 | 20295.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 82.4 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 14.9 | 0.0 | 1.00 |

### prof 2026-09-22 14:39:05 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1799, mean 3256.13 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.058 | 0.00 | 104.5 | 0.0 | 1.00 |
| net_fwd | gpu | 9.079 | 0.114 | 0.28 | 16334.0 | 14.0 | 1.00 |
| assemble | gpu | 0.127 | 0.005 | 0.00 | 228.2 | 1.0 | 1.00 |
| lu | gpu | 0.722 | 0.008 | 0.02 | 1298.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 32.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 645.468 | 655.290 | 19.82 | 1161197.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 498.957 | 0.122 | 15.32 | 897624.1 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 95.296 | 0.414 | 2.93 | 171437.6 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 167.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 92.772 | 0.293 | 2.85 | 166896.4 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.107 | 0.013 | 0.00 | 192.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.430 | 0.013 | 0.01 | 774.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.823 | 0.021 | 0.06 | 3278.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 54.9 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 51.107 | 0.083 | 1.57 | 91940.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1285.665 | 1285.683 | 39.48 | 2312911.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 992.689 | 0.241 | 30.49 | 1785848.0 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 190.699 | 0.838 | 5.86 | 343067.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.193 | 0.028 | 0.01 | 346.5 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 185.404 | 0.594 | 5.69 | 333541.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.193 | 0.026 | 0.01 | 346.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.786 | 0.026 | 0.02 | 1414.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.983 | 0.043 | 0.12 | 7165.9 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 109.8 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 101.849 | 0.164 | 3.13 | 183225.6 | 72.0 | 6.00 |
| record | gpu | 486.984 | 486.995 | 14.96 | 876084.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.399 | 0.425 | 0.63 | 36697.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.766 | 0.331 | 0.55 | 31961.3 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.264 | 0.009 | 0.01 | 474.6 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.103 | 0.016 | 0.03 | 1983.8 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 62.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.176 | 0.011 | 0.04 | 2116.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 256.570 | 276.549 | 7.88 | 461570.1 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 227.110 | 0.206 | 6.97 | 408571.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.589 | 0.009 | 0.85 | 49632.2 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.858 | 276.316 | 0.06 | 3343.0 | 8.0 | 2.00 |
| record/exchange | gpu | 115.364 | 115.677 | 3.54 | 207539.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 63.287 | 0.289 | 1.94 | 113854.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 145.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 61.608 | 0.204 | 1.89 | 110832.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 111.5 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.379 | 0.009 | 0.01 | 681.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.112 | 0.015 | 0.03 | 2000.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 33.6 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 53.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 47.220 | 0.145 | 1.45 | 84948.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.767 | 0.018 | 0.15 | 8576.0 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 115.184 | 3.54 | 207215.6 | 0.0 | 2.00 |
| record/assemble | gpu | 0.572 | 0.018 | 0.02 | 1029.4 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 46.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 93.718 | 93.708 | 2.88 | 168598.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.319 | 0.01 | 573.6 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.657 | 0.033 | 0.02 | 1182.1 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.423 | 0.243 | 1.03 | 60128.8 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.324 | 0.175 | 0.81 | 47356.1 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.886 | 0.009 | 1.01 | 59161.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.208 | 0.01 | 374.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 68.8 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.142 | 0.027 | 0.47 | 27240.1 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 219.6 | 0.0 | 1.00 |
| sr/grad | gpu | 6.066 | 0.013 | 0.19 | 10913.1 | 2.0 | 1.00 |
| sr/cg | gpu | 794.059 | 803.135 | 24.39 | 1428512.5 | 351.4 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 794.862 | 24.41 | 1429957.0 | 0.0 | 212.26 |
| sr/cg/matvec | gpu | 776.863 | 7.206 | 23.86 | 1397576.4 | 142.2 | 71.09 |
| sr/cg/precond | gpu | 4.235 | 0.314 | 0.13 | 7619.5 | 70.1 | 70.09 |
| sr/trust | gpu | 12.078 | 12.078 | 0.37 | 21728.8 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.690 | 0.201 | 0.36 | 21031.1 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.869 | 0.36 | 21352.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 87.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 15.7 | 0.0 | 1.00 |

### prof 2026-09-22 14:44:51 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1899, mean 3267.14 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.058 | 0.00 | 109.3 | 0.0 | 1.00 |
| net_fwd | gpu | 9.050 | 0.114 | 0.28 | 17185.0 | 14.0 | 1.00 |
| assemble | gpu | 0.127 | 0.005 | 0.00 | 240.6 | 1.0 | 1.00 |
| lu | gpu | 0.715 | 0.008 | 0.02 | 1357.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 34.4 | 1.0 | 1.00 |
| therm_sweeps | gpu | 643.207 | 652.992 | 19.69 | 1221449.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 497.227 | 0.122 | 15.22 | 944234.7 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 94.959 | 0.414 | 2.91 | 180326.6 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 176.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 92.446 | 0.293 | 2.83 | 175555.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.106 | 0.013 | 0.00 | 201.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.427 | 0.013 | 0.01 | 811.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.814 | 0.021 | 0.06 | 3445.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 58.0 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.913 | 0.083 | 1.56 | 96683.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1280.982 | 1281.000 | 39.21 | 2432584.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 989.104 | 0.240 | 30.27 | 1878309.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 189.994 | 0.837 | 5.82 | 360799.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.192 | 0.028 | 0.01 | 364.9 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 184.737 | 0.594 | 5.65 | 350815.3 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.192 | 0.026 | 0.01 | 365.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.784 | 0.025 | 0.02 | 1489.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.949 | 0.043 | 0.12 | 7498.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 116.0 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 101.472 | 0.164 | 3.11 | 192695.4 | 72.0 | 6.00 |
| record | gpu | 485.139 | 485.149 | 14.85 | 921279.6 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.333 | 0.423 | 0.62 | 38612.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.701 | 0.330 | 0.54 | 33614.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.263 | 0.009 | 0.01 | 499.2 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.103 | 0.016 | 0.03 | 2094.6 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 65.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.177 | 0.011 | 0.04 | 2234.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 255.612 | 275.526 | 7.82 | 485406.7 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 226.282 | 0.206 | 6.93 | 429709.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.491 | 0.009 | 0.84 | 52205.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.826 | 275.293 | 0.06 | 3467.1 | 8.0 | 2.00 |
| record/exchange | gpu | 114.955 | 115.252 | 3.52 | 218299.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 63.069 | 0.288 | 1.93 | 119767.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 153.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 61.396 | 0.204 | 1.88 | 116591.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 117.7 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.372 | 0.009 | 0.01 | 706.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.112 | 0.015 | 0.03 | 2111.5 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 35.5 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 56.2 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 47.047 | 0.145 | 1.44 | 89342.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.750 | 0.018 | 0.15 | 9019.7 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 114.759 | 3.51 | 217928.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.556 | 0.018 | 0.02 | 1056.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 48.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 93.338 | 93.328 | 2.86 | 177248.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.303 | 0.01 | 575.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.640 | 0.033 | 0.02 | 1214.6 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.315 | 0.243 | 1.02 | 63264.3 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.228 | 0.175 | 0.80 | 49807.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.744 | 0.009 | 1.00 | 62180.7 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.200 | 0.01 | 380.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 72.1 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.083 | 0.027 | 0.46 | 28642.3 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 231.7 | 0.0 | 1.00 |
| sr/grad | gpu | 6.042 | 0.013 | 0.18 | 11474.6 | 2.0 | 1.00 |
| sr/cg | gpu | 814.075 | 823.115 | 24.92 | 1545927.8 | 361.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 814.943 | 24.94 | 1547576.5 | 0.0 | 218.46 |
| sr/cg/matvec | gpu | 797.337 | 7.077 | 24.40 | 1514143.3 | 146.3 | 73.15 |
| sr/cg/precond | gpu | 4.047 | 0.323 | 0.12 | 7684.9 | 72.2 | 72.15 |
| sr/trust | gpu | 12.002 | 12.002 | 0.37 | 22791.9 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.631 | 0.192 | 0.36 | 22087.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.801 | 0.36 | 22410.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 92.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 16.4 | 0.0 | 1.00 |

### prof 2026-09-22 14:50:39 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 1999, mean 3277.37 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.058 | 0.00 | 115.1 | 0.0 | 1.00 |
| net_fwd | gpu | 9.023 | 0.114 | 0.28 | 18037.1 | 14.0 | 1.00 |
| assemble | gpu | 0.127 | 0.005 | 0.00 | 253.0 | 1.0 | 1.00 |
| lu | gpu | 0.709 | 0.008 | 0.02 | 1417.1 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 36.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 641.176 | 650.928 | 19.56 | 1281710.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 495.674 | 0.122 | 15.12 | 990853.2 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 94.656 | 0.414 | 2.89 | 189217.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 186.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 92.154 | 0.293 | 2.81 | 184216.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.106 | 0.013 | 0.00 | 211.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.425 | 0.013 | 0.01 | 849.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.807 | 0.021 | 0.06 | 3612.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 61.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.738 | 0.083 | 1.55 | 101424.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1276.775 | 1276.793 | 38.96 | 2552273.6 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 985.884 | 0.240 | 30.08 | 1970782.8 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 189.362 | 0.837 | 5.78 | 378534.8 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.192 | 0.028 | 0.01 | 383.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 184.138 | 0.594 | 5.62 | 368092.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.192 | 0.026 | 0.01 | 384.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.783 | 0.025 | 0.02 | 1564.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.917 | 0.043 | 0.12 | 7830.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 122.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 101.133 | 0.164 | 3.09 | 202164.9 | 72.0 | 6.00 |
| record | gpu | 483.481 | 483.491 | 14.75 | 966478.3 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.275 | 0.423 | 0.62 | 40528.8 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.643 | 0.330 | 0.54 | 35267.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.262 | 0.009 | 0.01 | 523.9 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.103 | 0.016 | 0.03 | 2205.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 69.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.177 | 0.011 | 0.04 | 2352.4 | 2.0 | 2.00 |
| record/jet_pass | gpu | 254.750 | 274.606 | 7.77 | 509244.4 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 225.537 | 0.206 | 6.88 | 450847.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.403 | 0.009 | 0.84 | 54778.8 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.797 | 274.373 | 0.05 | 3591.3 | 8.0 | 2.00 |
| record/exchange | gpu | 114.587 | 114.870 | 3.50 | 229059.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.872 | 0.288 | 1.92 | 125681.2 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 161.1 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 61.206 | 0.204 | 1.87 | 122350.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 123.9 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.366 | 0.009 | 0.01 | 731.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.112 | 0.015 | 0.03 | 2222.3 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 37.4 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 59.2 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.892 | 0.145 | 1.43 | 93736.8 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.734 | 0.018 | 0.14 | 9463.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 114.377 | 3.49 | 228640.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.542 | 0.018 | 0.02 | 1083.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 51.1 | 4.0 | 2.00 |
| record/o_assemble | gpu | 92.996 | 92.987 | 2.84 | 185899.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.289 | 0.01 | 576.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.624 | 0.033 | 0.02 | 1247.1 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.217 | 0.243 | 1.01 | 66399.9 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.142 | 0.175 | 0.80 | 52258.8 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.617 | 0.009 | 1.00 | 65200.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.194 | 0.01 | 387.1 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 75.9 | 0.0 | 1.00 |
| sr/o_stats | gpu | 15.030 | 0.027 | 0.46 | 30044.8 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 244.0 | 0.0 | 1.00 |
| sr/grad | gpu | 6.021 | 0.013 | 0.18 | 12035.0 | 2.0 | 1.00 |
| sr/cg | gpu | 832.388 | 841.397 | 25.40 | 1663944.4 | 371.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 833.312 | 25.43 | 1665790.5 | 0.0 | 224.12 |
| sr/cg/matvec | gpu | 816.061 | 6.962 | 24.90 | 1631305.7 | 150.1 | 75.04 |
| sr/cg/precond | gpu | 3.877 | 0.332 | 0.12 | 7751.1 | 74.0 | 74.04 |
| sr/trust | gpu | 11.933 | 11.933 | 0.36 | 23854.2 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.577 | 0.185 | 0.35 | 23142.7 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.740 | 0.36 | 23468.0 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 96.8 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 17.2 | 0.0 | 1.00 |

### prof 2026-09-22 14:56:25 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2099, mean 3286.06 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 120.2 | 0.0 | 1.00 |
| net_fwd | gpu | 8.999 | 0.114 | 0.27 | 18889.3 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 265.3 | 1.0 | 1.00 |
| lu | gpu | 0.703 | 0.008 | 0.02 | 1476.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 38.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 639.339 | 649.062 | 19.46 | 1341973.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 494.270 | 0.122 | 15.04 | 1037472.7 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 94.382 | 0.414 | 2.87 | 198107.0 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 195.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 91.890 | 0.293 | 2.80 | 192876.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.105 | 0.013 | 0.00 | 220.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.422 | 0.013 | 0.01 | 886.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.800 | 0.021 | 0.05 | 3778.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 64.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.580 | 0.083 | 1.54 | 106168.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1272.968 | 1272.985 | 38.74 | 2671959.6 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 982.969 | 0.240 | 29.91 | 2063252.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 188.790 | 0.836 | 5.75 | 396271.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.191 | 0.028 | 0.01 | 401.6 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 183.597 | 0.593 | 5.59 | 385369.8 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.192 | 0.026 | 0.01 | 402.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.781 | 0.025 | 0.02 | 1639.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.889 | 0.043 | 0.12 | 8163.2 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 128.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 100.827 | 0.164 | 3.07 | 211635.2 | 72.0 | 6.00 |
| record | gpu | 481.980 | 481.989 | 14.67 | 1011675.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.221 | 0.422 | 0.62 | 42444.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.590 | 0.329 | 0.54 | 36920.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.261 | 0.009 | 0.01 | 548.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.103 | 0.016 | 0.03 | 2316.2 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 72.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.177 | 0.011 | 0.04 | 2470.6 | 2.0 | 2.00 |
| record/jet_pass | gpu | 253.970 | 273.774 | 7.73 | 533083.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 224.863 | 0.206 | 6.84 | 471988.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.324 | 0.009 | 0.83 | 57352.1 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.770 | 273.541 | 0.05 | 3715.4 | 8.0 | 2.00 |
| record/exchange | gpu | 114.254 | 114.524 | 3.48 | 239819.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.694 | 0.288 | 1.91 | 131594.9 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 169.2 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 61.034 | 0.204 | 1.86 | 128109.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 130.1 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.361 | 0.009 | 0.01 | 756.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.112 | 0.015 | 0.03 | 2333.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 39.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 62.2 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.751 | 0.145 | 1.42 | 98130.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.720 | 0.018 | 0.14 | 9907.2 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 114.032 | 3.47 | 239352.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.529 | 0.018 | 0.02 | 1111.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.015 | 0.00 | 53.7 | 4.0 | 2.00 |
| record/o_assemble | gpu | 92.687 | 92.678 | 2.82 | 194549.2 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.276 | 0.01 | 578.4 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.610 | 0.033 | 0.02 | 1279.5 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.128 | 0.242 | 1.01 | 69535.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 26.065 | 0.175 | 0.79 | 54710.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.501 | 0.009 | 0.99 | 68219.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.187 | 0.01 | 393.5 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 79.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.982 | 0.026 | 0.46 | 31447.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 256.2 | 0.0 | 1.00 |
| sr/grad | gpu | 6.001 | 0.013 | 0.18 | 12595.5 | 2.0 | 1.00 |
| sr/cg | gpu | 848.394 | 857.375 | 25.82 | 1780779.6 | 379.5 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 849.372 | 25.85 | 1782831.7 | 0.0 | 229.09 |
| sr/cg/matvec | gpu | 832.442 | 6.856 | 25.33 | 1747295.7 | 153.4 | 76.70 |
| sr/cg/precond | gpu | 3.724 | 0.339 | 0.11 | 7816.4 | 75.7 | 75.70 |
| sr/trust | gpu | 11.871 | 11.871 | 0.36 | 24916.9 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.529 | 0.178 | 0.35 | 24198.6 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.684 | 0.36 | 24525.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 101.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.009 | 0.00 | 17.9 | 0.0 | 1.00 |

### prof 2026-09-22 15:02:12 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2199, mean 3294.70 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 125.4 | 0.0 | 1.00 |
| net_fwd | gpu | 8.977 | 0.114 | 0.27 | 19740.6 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 277.7 | 1.0 | 1.00 |
| lu | gpu | 0.698 | 0.008 | 0.02 | 1535.5 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 39.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 637.672 | 647.367 | 19.35 | 1402239.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 492.995 | 0.122 | 14.96 | 1084096.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 94.133 | 0.414 | 2.86 | 206997.9 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 204.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 91.650 | 0.293 | 2.78 | 201537.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.104 | 0.013 | 0.00 | 229.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.420 | 0.013 | 0.01 | 924.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.794 | 0.021 | 0.05 | 3945.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 67.2 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.436 | 0.083 | 1.53 | 110909.6 | 36.0 | 3.00 |
| record_sweeps | gpu | 1269.514 | 1269.531 | 38.53 | 2791662.3 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 980.325 | 0.240 | 29.75 | 2155735.7 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 188.272 | 0.836 | 5.71 | 414009.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.191 | 0.028 | 0.01 | 420.0 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 183.106 | 0.593 | 5.56 | 402649.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.192 | 0.026 | 0.01 | 421.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.780 | 0.025 | 0.02 | 1714.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.863 | 0.043 | 0.12 | 8495.7 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 134.3 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 100.548 | 0.164 | 3.05 | 221105.8 | 72.0 | 6.00 |
| record | gpu | 480.616 | 480.625 | 14.59 | 1056874.0 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.173 | 0.421 | 0.61 | 44360.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.542 | 0.328 | 0.53 | 38574.1 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.261 | 0.009 | 0.01 | 573.3 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.104 | 0.016 | 0.03 | 2427.0 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 76.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.177 | 0.011 | 0.04 | 2588.8 | 2.0 | 2.00 |
| record/jet_pass | gpu | 253.261 | 273.017 | 7.69 | 556921.8 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 224.251 | 0.206 | 6.81 | 493128.0 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.251 | 0.009 | 0.83 | 59925.4 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.746 | 272.785 | 0.05 | 3839.4 | 8.0 | 2.00 |
| record/exchange | gpu | 113.952 | 114.210 | 3.46 | 250580.8 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.533 | 0.288 | 1.90 | 137509.4 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 177.3 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.877 | 0.204 | 1.85 | 133869.5 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 136.3 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.356 | 0.009 | 0.01 | 781.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 2443.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 41.1 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 65.2 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.623 | 0.145 | 1.42 | 102525.0 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.707 | 0.018 | 0.14 | 10350.9 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 113.718 | 3.45 | 250065.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.518 | 0.017 | 0.02 | 1138.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 56.2 | 4.0 | 2.00 |
| record/o_assemble | gpu | 92.405 | 92.397 | 2.80 | 203199.1 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.264 | 0.01 | 580.1 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.597 | 0.033 | 0.02 | 1312.0 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 33.047 | 0.242 | 1.00 | 72670.6 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.994 | 0.175 | 0.79 | 57161.9 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.396 | 0.009 | 0.98 | 71237.9 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.182 | 0.01 | 399.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 82.9 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.938 | 0.026 | 0.45 | 32849.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 268.6 | 0.0 | 1.00 |
| sr/grad | gpu | 5.983 | 0.013 | 0.18 | 13155.8 | 2.0 | 1.00 |
| sr/cg | gpu | 863.681 | 872.636 | 26.21 | 1899234.5 | 387.4 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 864.702 | 26.25 | 1901480.1 | 0.0 | 233.81 |
| sr/cg/matvec | gpu | 848.063 | 6.763 | 25.74 | 1864890.7 | 156.5 | 78.27 |
| sr/cg/precond | gpu | 3.585 | 0.346 | 0.11 | 7883.2 | 77.3 | 77.27 |
| sr/trust | gpu | 11.814 | 11.814 | 0.36 | 25978.6 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.484 | 0.172 | 0.35 | 25253.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.634 | 0.35 | 25582.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 106.4 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 18.7 | 0.0 | 1.00 |

### prof 2026-09-22 15:08:00 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2299, mean 3302.63 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 131.3 | 0.0 | 1.00 |
| net_fwd | gpu | 8.957 | 0.114 | 0.27 | 20593.2 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 290.1 | 1.0 | 1.00 |
| lu | gpu | 0.694 | 0.008 | 0.02 | 1594.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 41.7 | 1.0 | 1.00 |
| therm_sweeps | gpu | 636.150 | 645.821 | 19.26 | 1462508.2 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 491.832 | 0.122 | 14.89 | 1130721.9 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 93.906 | 0.414 | 2.84 | 215888.9 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 213.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 91.431 | 0.293 | 2.77 | 210199.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.104 | 0.013 | 0.00 | 239.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.418 | 0.013 | 0.01 | 961.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.788 | 0.021 | 0.05 | 4111.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 70.3 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.305 | 0.083 | 1.52 | 115650.9 | 36.0 | 3.00 |
| record_sweeps | gpu | 1266.362 | 1266.378 | 38.34 | 2911365.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 977.913 | 0.240 | 29.61 | 2248220.9 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 187.798 | 0.836 | 5.69 | 431747.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.191 | 0.028 | 0.01 | 438.4 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 182.657 | 0.593 | 5.53 | 419929.0 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.191 | 0.026 | 0.01 | 439.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.778 | 0.025 | 0.02 | 1789.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.840 | 0.043 | 0.12 | 8828.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 140.5 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 100.294 | 0.164 | 3.04 | 230575.9 | 72.0 | 6.00 |
| record | gpu | 479.370 | 479.379 | 14.51 | 1102072.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.129 | 0.420 | 0.61 | 46277.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.498 | 0.328 | 0.53 | 40227.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.260 | 0.009 | 0.01 | 598.0 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.104 | 0.016 | 0.03 | 2537.8 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 79.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.177 | 0.011 | 0.04 | 2707.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 252.614 | 272.326 | 7.65 | 580759.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 223.691 | 0.206 | 6.77 | 514266.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.185 | 0.009 | 0.82 | 62498.8 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.724 | 272.093 | 0.05 | 3963.7 | 8.0 | 2.00 |
| record/exchange | gpu | 113.677 | 113.923 | 3.44 | 261342.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.385 | 0.287 | 1.89 | 143424.2 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 185.3 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.735 | 0.204 | 1.84 | 139629.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 142.5 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.351 | 0.009 | 0.01 | 806.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 2554.7 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 43.0 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 68.1 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.507 | 0.145 | 1.41 | 106919.2 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.695 | 0.018 | 0.14 | 10794.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 113.432 | 3.43 | 260779.2 | 0.0 | 2.00 |
| record/assemble | gpu | 0.507 | 0.017 | 0.02 | 1165.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 58.8 | 4.0 | 2.00 |
| record/o_assemble | gpu | 92.148 | 92.140 | 2.79 | 211848.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.253 | 0.01 | 581.7 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.585 | 0.033 | 0.02 | 1344.5 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.974 | 0.242 | 1.00 | 75806.4 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.930 | 0.175 | 0.79 | 59613.5 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.299 | 0.009 | 0.98 | 74256.4 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.177 | 0.01 | 406.4 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 86.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.898 | 0.026 | 0.45 | 34251.6 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 280.8 | 0.0 | 1.00 |
| sr/grad | gpu | 5.966 | 0.013 | 0.18 | 13716.7 | 2.0 | 1.00 |
| sr/cg | gpu | 877.666 | 886.597 | 26.57 | 2017754.8 | 394.5 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 878.727 | 26.61 | 2020194.0 | 0.0 | 238.13 |
| sr/cg/matvec | gpu | 862.354 | 6.678 | 26.11 | 1982552.3 | 159.4 | 79.71 |
| sr/cg/precond | gpu | 3.458 | 0.352 | 0.10 | 7949.9 | 78.7 | 78.71 |
| sr/trust | gpu | 11.762 | 11.762 | 0.36 | 27041.7 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.444 | 0.166 | 0.35 | 26309.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.588 | 0.35 | 26640.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 111.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 19.4 | 0.0 | 1.00 |

### prof 2026-09-22 15:13:47 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2399, mean 3309.61 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 136.5 | 0.0 | 1.00 |
| net_fwd | gpu | 8.939 | 0.114 | 0.27 | 21445.8 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 302.5 | 1.0 | 1.00 |
| lu | gpu | 0.689 | 0.008 | 0.02 | 1654.0 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 43.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 634.757 | 644.405 | 19.18 | 1522781.0 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 490.768 | 0.122 | 14.83 | 1177351.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 93.698 | 0.414 | 2.83 | 224780.7 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 223.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 91.230 | 0.293 | 2.76 | 218861.5 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.104 | 0.013 | 0.00 | 248.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.416 | 0.013 | 0.01 | 999.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.783 | 0.021 | 0.05 | 4278.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 73.4 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.184 | 0.083 | 1.52 | 120392.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1263.474 | 1263.490 | 38.18 | 3031074.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 975.703 | 0.240 | 29.48 | 2340710.8 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 187.364 | 0.835 | 5.66 | 449485.8 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.190 | 0.028 | 0.01 | 456.7 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 182.246 | 0.593 | 5.51 | 437209.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.191 | 0.026 | 0.01 | 458.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.777 | 0.025 | 0.02 | 1864.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.818 | 0.043 | 0.12 | 9160.6 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 146.6 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 100.061 | 0.164 | 3.02 | 240045.9 | 72.0 | 6.00 |
| record | gpu | 478.228 | 478.237 | 14.45 | 1147269.8 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.089 | 0.420 | 0.61 | 48192.9 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.458 | 0.327 | 0.53 | 41881.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.260 | 0.009 | 0.01 | 622.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.104 | 0.016 | 0.03 | 2648.5 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 83.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.011 | 0.04 | 2825.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 252.020 | 271.692 | 7.61 | 604596.9 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 223.179 | 0.206 | 6.74 | 535405.7 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.125 | 0.009 | 0.82 | 65071.8 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.704 | 271.460 | 0.05 | 4087.7 | 8.0 | 2.00 |
| record/exchange | gpu | 113.424 | 113.661 | 3.43 | 272103.7 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.251 | 0.287 | 1.88 | 149339.0 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 193.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.604 | 0.204 | 1.83 | 145389.8 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 148.7 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.347 | 0.009 | 0.01 | 831.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 2665.5 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 44.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 71.1 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.400 | 0.145 | 1.40 | 111313.1 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.685 | 0.018 | 0.14 | 11238.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 113.169 | 3.42 | 271492.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.497 | 0.017 | 0.02 | 1192.8 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 61.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 91.913 | 91.905 | 2.78 | 220498.1 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.243 | 0.01 | 583.3 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.574 | 0.033 | 0.02 | 1376.9 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.906 | 0.242 | 0.99 | 78941.9 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.871 | 0.175 | 0.78 | 62065.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.211 | 0.009 | 0.97 | 77274.4 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.172 | 0.01 | 412.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 90.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.862 | 0.026 | 0.45 | 35653.3 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 293.2 | 0.0 | 1.00 |
| sr/grad | gpu | 5.952 | 0.013 | 0.18 | 14277.8 | 2.0 | 1.00 |
| sr/cg | gpu | 890.200 | 899.109 | 26.90 | 2135590.0 | 401.0 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 891.298 | 26.93 | 2138223.3 | 0.0 | 242.01 |
| sr/cg/matvec | gpu | 875.169 | 6.600 | 26.44 | 2099531.2 | 162.0 | 81.00 |
| sr/cg/precond | gpu | 3.342 | 0.358 | 0.10 | 8016.7 | 80.0 | 80.00 |
| sr/trust | gpu | 11.715 | 11.715 | 0.35 | 28105.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.407 | 0.161 | 0.34 | 27366.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.546 | 0.35 | 27698.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 116.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 20.1 | 0.0 | 1.00 |

### prof 2026-09-22 15:19:35 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2499, mean 3316.63 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 141.7 | 0.0 | 1.00 |
| net_fwd | gpu | 8.923 | 0.113 | 0.27 | 22299.7 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 314.8 | 1.0 | 1.00 |
| lu | gpu | 0.686 | 0.008 | 0.02 | 1713.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 45.4 | 1.0 | 1.00 |
| therm_sweeps | gpu | 633.478 | 643.106 | 19.10 | 1583060.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 489.790 | 0.122 | 14.77 | 1223986.2 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 93.507 | 0.414 | 2.82 | 233673.4 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 232.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 91.046 | 0.293 | 2.75 | 227524.5 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.103 | 0.013 | 0.00 | 257.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.415 | 0.013 | 0.01 | 1036.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.779 | 0.021 | 0.05 | 4444.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 76.5 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 50.074 | 0.083 | 1.51 | 125134.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1260.820 | 1260.836 | 38.02 | 3150788.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 973.672 | 0.240 | 29.36 | 2433206.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 186.965 | 0.835 | 5.64 | 467225.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.190 | 0.028 | 0.01 | 475.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 181.869 | 0.593 | 5.48 | 454490.4 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.191 | 0.026 | 0.01 | 477.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.776 | 0.025 | 0.02 | 1939.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.799 | 0.043 | 0.11 | 9493.0 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 152.8 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 99.846 | 0.164 | 3.01 | 249514.6 | 72.0 | 6.00 |
| record | gpu | 477.178 | 477.187 | 14.39 | 1192467.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.052 | 0.419 | 0.60 | 50109.1 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.421 | 0.327 | 0.53 | 43534.7 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.259 | 0.009 | 0.01 | 647.4 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.104 | 0.016 | 0.03 | 2759.3 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 86.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.011 | 0.04 | 2943.4 | 2.0 | 2.00 |
| record/jet_pass | gpu | 251.474 | 271.109 | 7.58 | 628433.9 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 222.707 | 0.206 | 6.71 | 556544.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.069 | 0.009 | 0.82 | 67644.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.685 | 270.877 | 0.05 | 4211.7 | 8.0 | 2.00 |
| record/exchange | gpu | 113.192 | 113.419 | 3.41 | 282865.8 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.127 | 0.287 | 1.87 | 155254.2 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 201.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.484 | 0.204 | 1.82 | 151150.3 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 154.9 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.343 | 0.009 | 0.01 | 856.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 2776.3 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 46.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 74.1 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.301 | 0.145 | 1.40 | 115707.2 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.675 | 0.018 | 0.14 | 11682.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 112.928 | 3.40 | 282206.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.488 | 0.017 | 0.01 | 1220.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 63.9 | 4.0 | 2.00 |
| record/o_assemble | gpu | 91.696 | 91.688 | 2.76 | 229147.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.234 | 0.01 | 584.9 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.564 | 0.033 | 0.02 | 1409.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.844 | 0.242 | 0.99 | 82077.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.817 | 0.175 | 0.78 | 64517.1 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.130 | 0.009 | 0.97 | 80292.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.168 | 0.01 | 419.3 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 93.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.828 | 0.026 | 0.45 | 37056.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 305.7 | 0.0 | 1.00 |
| sr/grad | gpu | 5.938 | 0.013 | 0.18 | 14839.7 | 2.0 | 1.00 |
| sr/cg | gpu | 902.327 | 911.216 | 27.21 | 2254915.6 | 407.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 903.455 | 27.24 | 2257734.4 | 0.0 | 245.74 |
| sr/cg/matvec | gpu | 887.551 | 6.531 | 26.76 | 2217989.0 | 164.5 | 82.25 |
| sr/cg/precond | gpu | 3.235 | 0.363 | 0.10 | 8084.5 | 81.2 | 81.25 |
| sr/trust | gpu | 11.672 | 11.672 | 0.35 | 29168.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.374 | 0.157 | 0.34 | 28422.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.507 | 0.35 | 28756.7 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 120.8 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 20.9 | 0.0 | 1.00 |

### prof 2026-09-22 15:25:22 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2599, mean 3322.53 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 146.7 | 0.0 | 1.00 |
| net_fwd | gpu | 8.909 | 0.113 | 0.27 | 23153.6 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 327.2 | 1.0 | 1.00 |
| lu | gpu | 0.682 | 0.008 | 0.02 | 1772.7 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 47.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 632.299 | 641.909 | 19.03 | 1643344.0 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 488.890 | 0.122 | 14.71 | 1270624.9 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 93.331 | 0.414 | 2.81 | 242566.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 241.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.876 | 0.293 | 2.74 | 236187.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.103 | 0.013 | 0.00 | 266.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.413 | 0.013 | 0.01 | 1074.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.774 | 0.021 | 0.05 | 4611.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 79.5 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.971 | 0.083 | 1.50 | 129875.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1258.373 | 1258.389 | 37.87 | 3270511.7 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 971.800 | 0.240 | 29.25 | 2525706.9 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 186.598 | 0.835 | 5.62 | 484967.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.190 | 0.028 | 0.01 | 493.5 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 181.521 | 0.593 | 5.46 | 471773.8 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.191 | 0.026 | 0.01 | 495.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.775 | 0.025 | 0.02 | 2014.4 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.780 | 0.043 | 0.11 | 9825.4 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 158.9 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 99.648 | 0.164 | 3.00 | 258985.0 | 72.0 | 6.00 |
| record | gpu | 476.209 | 476.217 | 14.33 | 1237666.2 | 236.0 | 2.00 |
| record/eval_cached | gpu | 20.017 | 0.418 | 0.60 | 52025.2 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.387 | 0.326 | 0.52 | 45188.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.259 | 0.009 | 0.01 | 672.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.104 | 0.016 | 0.03 | 2870.1 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 90.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.010 | 0.04 | 3061.6 | 2.0 | 2.00 |
| record/jet_pass | gpu | 250.970 | 270.571 | 7.55 | 652270.9 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 222.271 | 0.205 | 6.69 | 577682.6 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 27.017 | 0.009 | 0.81 | 70218.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.668 | 270.339 | 0.05 | 4335.8 | 8.0 | 2.00 |
| record/exchange | gpu | 112.977 | 113.197 | 3.40 | 293628.4 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 62.012 | 0.287 | 1.87 | 161170.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 209.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.374 | 0.203 | 1.82 | 156911.6 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 161.1 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.339 | 0.009 | 0.01 | 881.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 2887.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 48.6 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 77.0 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.211 | 0.145 | 1.39 | 120101.3 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.666 | 0.018 | 0.14 | 12126.1 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 112.705 | 3.39 | 292920.4 | 0.0 | 2.00 |
| record/assemble | gpu | 0.480 | 0.017 | 0.01 | 1247.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 66.5 | 4.0 | 2.00 |
| record/o_assemble | gpu | 91.496 | 91.488 | 2.75 | 237797.2 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.226 | 0.01 | 586.6 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.555 | 0.033 | 0.02 | 1441.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.787 | 0.242 | 0.99 | 85212.9 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.767 | 0.175 | 0.78 | 66969.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 32.055 | 0.009 | 0.96 | 83310.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.164 | 0.00 | 425.7 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 97.1 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.797 | 0.026 | 0.45 | 38457.7 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 318.2 | 0.0 | 1.00 |
| sr/grad | gpu | 5.926 | 0.013 | 0.18 | 15400.8 | 2.0 | 1.00 |
| sr/cg | gpu | 912.928 | 921.798 | 27.48 | 2372699.6 | 412.7 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 914.087 | 27.51 | 2375712.4 | 0.0 | 249.02 |
| sr/cg/matvec | gpu | 898.390 | 6.465 | 27.04 | 2334916.1 | 166.7 | 83.34 |
| sr/cg/precond | gpu | 3.136 | 0.368 | 0.09 | 8151.3 | 82.3 | 82.34 |
| sr/trust | gpu | 11.632 | 11.632 | 0.35 | 30231.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.342 | 0.152 | 0.34 | 29478.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.471 | 0.35 | 29814.4 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 125.6 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 21.6 | 0.0 | 1.00 |

### prof 2026-09-22 15:31:10 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2699, mean 3328.32 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 153.1 | 0.0 | 1.00 |
| net_fwd | gpu | 8.895 | 0.114 | 0.27 | 24007.8 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 339.6 | 1.0 | 1.00 |
| lu | gpu | 0.679 | 0.008 | 0.02 | 1832.0 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 49.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 631.210 | 640.803 | 18.96 | 1703634.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 488.059 | 0.122 | 14.66 | 1317270.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 93.168 | 0.414 | 2.80 | 251459.8 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 250.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.719 | 0.293 | 2.73 | 244851.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.102 | 0.013 | 0.00 | 276.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.412 | 0.013 | 0.01 | 1111.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.770 | 0.021 | 0.05 | 4777.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 82.6 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.877 | 0.083 | 1.50 | 134616.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1256.110 | 1256.126 | 37.74 | 3390241.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 970.068 | 0.240 | 29.15 | 2618212.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 186.258 | 0.835 | 5.60 | 502710.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.190 | 0.028 | 0.01 | 511.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 181.200 | 0.593 | 5.44 | 489058.1 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.191 | 0.026 | 0.01 | 514.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.774 | 0.025 | 0.02 | 2089.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.764 | 0.043 | 0.11 | 10157.9 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 165.0 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 99.465 | 0.164 | 2.99 | 268455.2 | 72.0 | 6.00 |
| record | gpu | 475.313 | 475.321 | 14.28 | 1282868.6 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.986 | 0.419 | 0.60 | 53942.9 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.356 | 0.327 | 0.52 | 46843.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.258 | 0.009 | 0.01 | 696.8 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.104 | 0.016 | 0.03 | 2980.9 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 93.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.011 | 0.04 | 3179.7 | 2.0 | 2.00 |
| record/jet_pass | gpu | 250.503 | 270.073 | 7.53 | 676108.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 221.868 | 0.206 | 6.67 | 598821.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.970 | 0.009 | 0.81 | 72791.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.653 | 269.840 | 0.05 | 4460.3 | 8.0 | 2.00 |
| record/exchange | gpu | 112.780 | 112.991 | 3.39 | 304392.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.907 | 0.287 | 1.86 | 167086.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 217.7 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.272 | 0.203 | 1.81 | 162672.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 167.3 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.336 | 0.009 | 0.01 | 906.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 2997.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 50.5 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 80.0 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.127 | 0.145 | 1.39 | 124496.2 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.657 | 0.018 | 0.14 | 12569.8 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 112.499 | 3.38 | 303635.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.472 | 0.017 | 0.01 | 1274.7 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 69.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 91.311 | 91.304 | 2.74 | 246447.4 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.218 | 0.01 | 588.3 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.546 | 0.033 | 0.02 | 1474.4 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.734 | 0.242 | 0.98 | 88348.8 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.721 | 0.175 | 0.77 | 69421.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.985 | 0.009 | 0.96 | 86328.4 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.160 | 0.00 | 432.3 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 101.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.768 | 0.026 | 0.44 | 39860.0 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 330.3 | 0.0 | 1.00 |
| sr/grad | gpu | 5.914 | 0.013 | 0.18 | 15961.3 | 2.0 | 1.00 |
| sr/cg | gpu | 923.062 | 931.915 | 27.73 | 2491344.7 | 417.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 924.249 | 27.77 | 2494547.6 | 0.0 | 252.14 |
| sr/cg/matvec | gpu | 908.744 | 6.404 | 27.30 | 2452699.0 | 168.8 | 84.38 |
| sr/cg/precond | gpu | 3.045 | 0.373 | 0.09 | 8218.1 | 83.4 | 83.38 |
| sr/trust | gpu | 11.594 | 11.594 | 0.35 | 31293.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.313 | 0.148 | 0.34 | 30533.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.438 | 0.34 | 30871.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 130.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 22.4 | 0.0 | 1.00 |

### prof 2026-09-22 15:36:58 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2799, mean 3333.49 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 158.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.882 | 0.114 | 0.27 | 24862.0 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 351.9 | 1.0 | 1.00 |
| lu | gpu | 0.676 | 0.008 | 0.02 | 1891.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 50.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 630.198 | 639.776 | 18.91 | 1763925.3 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 487.286 | 0.122 | 14.62 | 1363914.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 93.017 | 0.414 | 2.79 | 260353.7 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 259.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.574 | 0.293 | 2.72 | 253515.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.102 | 0.013 | 0.00 | 285.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.411 | 0.013 | 0.01 | 1149.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.766 | 0.021 | 0.05 | 4944.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 85.7 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.789 | 0.083 | 1.49 | 139359.1 | 36.0 | 3.00 |
| record_sweeps | gpu | 1254.008 | 1254.023 | 37.62 | 3509968.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 968.459 | 0.240 | 29.05 | 2710716.6 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 185.943 | 0.835 | 5.58 | 520453.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.189 | 0.028 | 0.01 | 530.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 180.901 | 0.593 | 5.43 | 506342.7 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 532.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.773 | 0.025 | 0.02 | 2164.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.748 | 0.043 | 0.11 | 10490.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 171.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 99.294 | 0.164 | 2.98 | 277924.9 | 72.0 | 6.00 |
| record | gpu | 474.479 | 474.487 | 14.23 | 1328066.7 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.957 | 0.418 | 0.60 | 55859.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.327 | 0.326 | 0.52 | 48497.0 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.258 | 0.009 | 0.01 | 721.6 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3091.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 97.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.010 | 0.04 | 3297.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 250.069 | 269.610 | 7.50 | 699944.3 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 221.493 | 0.205 | 6.64 | 619958.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.926 | 0.009 | 0.81 | 75364.7 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.638 | 269.378 | 0.05 | 4584.3 | 8.0 | 2.00 |
| record/exchange | gpu | 112.596 | 112.800 | 3.38 | 315155.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.809 | 0.287 | 1.85 | 173002.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 225.7 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.177 | 0.203 | 1.81 | 168434.8 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 173.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.333 | 0.009 | 0.01 | 931.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 3108.8 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 52.4 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 83.0 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 46.049 | 0.145 | 1.38 | 128890.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.649 | 0.018 | 0.14 | 13013.5 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 112.308 | 3.37 | 314351.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.465 | 0.017 | 0.01 | 1301.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 71.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 91.139 | 91.132 | 2.73 | 255096.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.211 | 0.01 | 589.9 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.538 | 0.033 | 0.02 | 1506.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.685 | 0.242 | 0.98 | 91484.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.678 | 0.175 | 0.77 | 71873.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.921 | 0.009 | 0.96 | 89346.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.157 | 0.00 | 438.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 104.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.742 | 0.026 | 0.44 | 41262.0 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 342.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.903 | 0.013 | 0.18 | 16523.2 | 2.0 | 1.00 |
| sr/cg | gpu | 932.280 | 941.117 | 27.97 | 2609452.0 | 422.7 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 933.495 | 28.00 | 2612853.2 | 0.0 | 254.99 |
| sr/cg/matvec | gpu | 918.168 | 6.346 | 27.54 | 2569951.6 | 170.7 | 85.33 |
| sr/cg/precond | gpu | 2.960 | 0.377 | 0.09 | 8284.1 | 84.3 | 84.33 |
| sr/trust | gpu | 11.560 | 11.559 | 0.35 | 32355.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.286 | 0.144 | 0.34 | 31588.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.407 | 0.34 | 31928.1 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 135.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 23.2 | 0.0 | 1.00 |

### prof 2026-09-22 15:42:45 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2899, mean 3338.46 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 164.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.871 | 0.114 | 0.27 | 25716.2 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 364.3 | 1.0 | 1.00 |
| lu | gpu | 0.673 | 0.008 | 0.02 | 1950.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 52.7 | 1.0 | 1.00 |
| therm_sweeps | gpu | 629.257 | 638.820 | 18.85 | 1824216.8 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 486.568 | 0.122 | 14.57 | 1410559.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.876 | 0.414 | 2.78 | 269247.4 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 269.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.438 | 0.293 | 2.71 | 262179.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.102 | 0.013 | 0.00 | 294.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.409 | 0.013 | 0.01 | 1186.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.763 | 0.021 | 0.05 | 5110.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 88.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.707 | 0.083 | 1.49 | 144101.3 | 36.0 | 3.00 |
| record_sweeps | gpu | 1252.052 | 1252.067 | 37.50 | 3629699.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 966.963 | 0.240 | 28.96 | 2803225.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 185.649 | 0.835 | 5.56 | 538197.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.189 | 0.028 | 0.01 | 548.5 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 180.624 | 0.593 | 5.41 | 523628.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 551.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.772 | 0.025 | 0.02 | 2239.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.733 | 0.043 | 0.11 | 10822.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 177.3 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 99.135 | 0.164 | 2.97 | 287393.0 | 72.0 | 6.00 |
| record | gpu | 473.704 | 473.712 | 14.19 | 1373267.2 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.930 | 0.418 | 0.60 | 57776.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.300 | 0.326 | 0.52 | 50151.4 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.257 | 0.009 | 0.01 | 746.2 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3202.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 100.8 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.011 | 0.04 | 3416.1 | 2.0 | 2.00 |
| record/jet_pass | gpu | 249.666 | 269.179 | 7.48 | 723781.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 221.144 | 0.205 | 6.62 | 641096.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.884 | 0.009 | 0.81 | 77937.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.624 | 268.947 | 0.05 | 4708.6 | 8.0 | 2.00 |
| record/exchange | gpu | 112.425 | 112.622 | 3.37 | 325919.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.717 | 0.287 | 1.85 | 178918.9 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 233.8 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.088 | 0.203 | 1.80 | 174196.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 179.7 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.330 | 0.009 | 0.01 | 956.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.111 | 0.015 | 0.03 | 3219.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 54.3 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 86.0 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.976 | 0.145 | 1.38 | 133285.2 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.642 | 0.018 | 0.14 | 13457.2 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 112.130 | 3.36 | 325065.7 | 0.0 | 2.00 |
| record/assemble | gpu | 0.459 | 0.017 | 0.01 | 1329.2 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 74.2 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.978 | 90.972 | 2.73 | 263746.2 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.204 | 0.01 | 591.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.531 | 0.033 | 0.02 | 1539.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.639 | 0.242 | 0.98 | 94620.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.638 | 0.175 | 0.77 | 74325.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.861 | 0.009 | 0.95 | 92363.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.154 | 0.00 | 445.3 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 108.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.717 | 0.026 | 0.44 | 42663.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 354.7 | 0.0 | 1.00 |
| sr/grad | gpu | 5.893 | 0.013 | 0.18 | 17083.6 | 2.0 | 1.00 |
| sr/cg | gpu | 941.007 | 949.830 | 28.19 | 2727980.1 | 427.1 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 942.245 | 28.22 | 2731567.9 | 0.0 | 257.69 |
| sr/cg/matvec | gpu | 927.084 | 6.295 | 27.77 | 2687615.9 | 172.5 | 86.23 |
| sr/cg/precond | gpu | 2.881 | 0.381 | 0.09 | 8351.4 | 85.2 | 85.23 |
| sr/trust | gpu | 11.528 | 11.528 | 0.35 | 33418.7 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.261 | 0.141 | 0.34 | 32644.9 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.378 | 0.34 | 32986.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 140.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 23.9 | 0.0 | 1.00 |

### prof 2026-09-22 15:48:33 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 2999, mean 3342.99 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 170.9 | 0.0 | 1.00 |
| net_fwd | gpu | 8.860 | 0.114 | 0.27 | 26570.5 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 376.7 | 1.0 | 1.00 |
| lu | gpu | 0.670 | 0.008 | 0.02 | 2010.1 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 54.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 628.379 | 637.928 | 18.80 | 1884507.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 485.897 | 0.122 | 14.53 | 1457205.0 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.745 | 0.414 | 2.77 | 278141.4 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 278.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.312 | 0.293 | 2.70 | 270844.2 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.101 | 0.013 | 0.00 | 304.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.408 | 0.013 | 0.01 | 1224.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.760 | 0.021 | 0.05 | 5277.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 91.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.631 | 0.083 | 1.48 | 148842.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1250.228 | 1250.242 | 37.40 | 3749432.6 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 965.567 | 0.240 | 28.88 | 2895734.2 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 185.376 | 0.835 | 5.55 | 555941.6 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.189 | 0.028 | 0.01 | 566.9 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 180.365 | 0.593 | 5.40 | 540914.3 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 570.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.772 | 0.025 | 0.02 | 2314.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.720 | 0.043 | 0.11 | 11155.2 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 183.5 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.987 | 0.164 | 2.96 | 296862.2 | 72.0 | 6.00 |
| record | gpu | 472.980 | 472.988 | 14.15 | 1418466.3 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.905 | 0.419 | 0.60 | 59693.8 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.274 | 0.327 | 0.52 | 51806.1 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.257 | 0.009 | 0.01 | 771.0 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3313.2 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 104.3 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.178 | 0.011 | 0.04 | 3534.3 | 2.0 | 2.00 |
| record/jet_pass | gpu | 249.289 | 268.776 | 7.46 | 747616.7 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 220.818 | 0.206 | 6.61 | 662233.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.846 | 0.009 | 0.80 | 80511.2 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.611 | 268.544 | 0.05 | 4832.9 | 8.0 | 2.00 |
| record/exchange | gpu | 112.265 | 112.456 | 3.36 | 336682.7 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.632 | 0.287 | 1.84 | 184835.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 241.9 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 60.006 | 0.203 | 1.79 | 179958.3 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 185.9 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.327 | 0.009 | 0.01 | 981.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3330.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 56.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 89.0 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.908 | 0.145 | 1.37 | 137679.5 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.635 | 0.018 | 0.14 | 13900.9 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.964 | 3.35 | 335780.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.452 | 0.017 | 0.01 | 1356.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 76.7 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.829 | 90.823 | 2.72 | 272395.8 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.198 | 0.01 | 593.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.524 | 0.033 | 0.02 | 1571.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.596 | 0.242 | 0.98 | 97756.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.601 | 0.175 | 0.77 | 76777.5 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.804 | 0.009 | 0.95 | 95381.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.151 | 0.00 | 452.0 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 112.5 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.694 | 0.026 | 0.44 | 44066.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 367.1 | 0.0 | 1.00 |
| sr/grad | gpu | 5.883 | 0.013 | 0.18 | 17644.3 | 2.0 | 1.00 |
| sr/cg | gpu | 949.041 | 957.850 | 28.39 | 2846174.9 | 431.3 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 950.300 | 28.43 | 2849951.0 | 0.0 | 260.17 |
| sr/cg/matvec | gpu | 935.295 | 6.246 | 27.98 | 2804950.3 | 174.1 | 87.06 |
| sr/cg/precond | gpu | 2.807 | 0.385 | 0.08 | 8418.5 | 86.1 | 86.06 |
| sr/trust | gpu | 11.497 | 11.497 | 0.34 | 34480.9 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.237 | 0.138 | 0.34 | 33700.1 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.352 | 0.34 | 34043.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 145.1 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 24.7 | 0.0 | 1.00 |

### prof 2026-09-22 15:54:21 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3099, mean 3347.37 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 177.0 | 0.0 | 1.00 |
| net_fwd | gpu | 8.850 | 0.114 | 0.26 | 27424.8 | 14.0 | 1.00 |
| assemble | gpu | 0.126 | 0.005 | 0.00 | 389.0 | 1.0 | 1.00 |
| lu | gpu | 0.668 | 0.008 | 0.02 | 2069.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 56.4 | 1.0 | 1.00 |
| therm_sweeps | gpu | 627.558 | 637.094 | 18.75 | 1944801.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 485.270 | 0.122 | 14.50 | 1503851.9 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.622 | 0.414 | 2.77 | 287036.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 287.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.193 | 0.293 | 2.69 | 279509.5 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.101 | 0.013 | 0.00 | 313.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.407 | 0.013 | 0.01 | 1261.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.757 | 0.021 | 0.05 | 5443.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 94.9 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.559 | 0.083 | 1.48 | 153583.6 | 36.0 | 3.00 |
| record_sweeps | gpu | 1248.523 | 1248.538 | 37.30 | 3869174.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 964.263 | 0.240 | 28.81 | 2988251.7 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 185.120 | 0.836 | 5.53 | 573687.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.189 | 0.028 | 0.01 | 585.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 180.123 | 0.593 | 5.38 | 558201.3 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 588.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.771 | 0.025 | 0.02 | 2389.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.707 | 0.043 | 0.11 | 11487.7 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 189.6 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.848 | 0.164 | 2.95 | 306330.1 | 72.0 | 6.00 |
| record | gpu | 472.303 | 472.311 | 14.11 | 1463667.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.881 | 0.419 | 0.59 | 61611.6 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.251 | 0.327 | 0.52 | 53461.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.257 | 0.009 | 0.01 | 795.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3424.0 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 107.7 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.011 | 0.04 | 3652.5 | 2.0 | 2.00 |
| record/jet_pass | gpu | 248.936 | 268.399 | 7.44 | 771452.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 220.513 | 0.206 | 6.59 | 683369.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.810 | 0.009 | 0.80 | 83084.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.600 | 268.167 | 0.05 | 4957.4 | 8.0 | 2.00 |
| record/exchange | gpu | 112.116 | 112.300 | 3.35 | 347446.7 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.553 | 0.287 | 1.84 | 190752.5 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 250.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.929 | 0.204 | 1.79 | 185720.6 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 192.2 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.325 | 0.009 | 0.01 | 1006.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3441.2 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 58.0 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 91.9 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.845 | 0.145 | 1.37 | 142074.0 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.629 | 0.018 | 0.14 | 14344.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.809 | 3.34 | 346495.2 | 0.0 | 2.00 |
| record/assemble | gpu | 0.447 | 0.017 | 0.01 | 1383.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 79.3 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.689 | 90.683 | 2.71 | 281046.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.192 | 0.01 | 594.9 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.518 | 0.033 | 0.02 | 1604.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.556 | 0.242 | 0.97 | 100892.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.566 | 0.175 | 0.76 | 79229.6 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.752 | 0.009 | 0.95 | 98398.9 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.148 | 0.00 | 458.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 116.5 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.672 | 0.026 | 0.44 | 45469.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 379.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.874 | 0.013 | 0.18 | 18204.8 | 2.0 | 1.00 |
| sr/cg | gpu | 956.707 | 965.503 | 28.58 | 2964835.6 | 435.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 957.986 | 28.62 | 2968799.6 | 0.0 | 262.53 |
| sr/cg/matvec | gpu | 943.126 | 6.201 | 28.18 | 2922746.2 | 175.7 | 87.84 |
| sr/cg/precond | gpu | 2.738 | 0.389 | 0.08 | 8485.7 | 86.8 | 86.84 |
| sr/trust | gpu | 11.469 | 11.469 | 0.34 | 35543.6 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.215 | 0.135 | 0.34 | 34755.9 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.326 | 0.34 | 35100.7 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 150.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 25.5 | 0.0 | 1.00 |

### prof 2026-09-22 16:00:09 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3199, mean 3351.64 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 182.8 | 0.0 | 1.00 |
| net_fwd | gpu | 8.840 | 0.114 | 0.26 | 28278.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 401.4 | 1.0 | 1.00 |
| lu | gpu | 0.665 | 0.008 | 0.02 | 2128.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 58.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 626.788 | 636.312 | 18.70 | 2005095.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 484.682 | 0.122 | 14.46 | 1550497.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.508 | 0.414 | 2.76 | 295931.6 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 296.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 90.083 | 0.293 | 2.69 | 288175.1 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.101 | 0.013 | 0.00 | 322.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.406 | 0.013 | 0.01 | 1299.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.754 | 0.021 | 0.05 | 5610.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 98.0 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.492 | 0.083 | 1.48 | 158326.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1246.926 | 1246.940 | 37.20 | 3988915.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 963.041 | 0.240 | 28.73 | 3080768.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 184.881 | 0.836 | 5.52 | 591433.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.189 | 0.028 | 0.01 | 603.5 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 179.896 | 0.593 | 5.37 | 575488.9 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 607.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.770 | 0.025 | 0.02 | 2464.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.695 | 0.043 | 0.11 | 11820.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 195.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.718 | 0.164 | 2.95 | 315797.9 | 72.0 | 6.00 |
| record | gpu | 471.669 | 471.677 | 14.07 | 1508868.8 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.859 | 0.419 | 0.59 | 63529.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.229 | 0.327 | 0.51 | 55116.3 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.256 | 0.009 | 0.01 | 820.4 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3534.8 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 111.2 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.011 | 0.04 | 3770.7 | 2.0 | 2.00 |
| record/jet_pass | gpu | 248.605 | 268.046 | 7.42 | 795288.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 220.227 | 0.206 | 6.57 | 704506.8 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.776 | 0.009 | 0.80 | 85657.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.589 | 267.814 | 0.05 | 5081.8 | 8.0 | 2.00 |
| record/exchange | gpu | 111.976 | 112.155 | 3.34 | 358211.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.479 | 0.287 | 1.83 | 196670.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 258.1 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.857 | 0.204 | 1.79 | 191483.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 198.4 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.323 | 0.009 | 0.01 | 1031.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3552.0 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 59.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 94.9 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.786 | 0.145 | 1.37 | 146468.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.623 | 0.018 | 0.14 | 14788.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.663 | 3.33 | 357210.7 | 0.0 | 2.00 |
| record/assemble | gpu | 0.441 | 0.017 | 0.01 | 1411.2 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 81.8 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.558 | 90.552 | 2.70 | 289695.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.186 | 0.01 | 596.6 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.512 | 0.033 | 0.02 | 1636.9 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.519 | 0.242 | 0.97 | 104028.0 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.534 | 0.175 | 0.76 | 81681.9 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.703 | 0.009 | 0.95 | 101416.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.145 | 0.00 | 465.1 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 120.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.652 | 0.026 | 0.44 | 46871.8 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 391.6 | 0.0 | 1.00 |
| sr/grad | gpu | 5.866 | 0.013 | 0.18 | 18765.1 | 2.0 | 1.00 |
| sr/cg | gpu | 964.044 | 972.827 | 28.76 | 3083975.7 | 439.0 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 965.340 | 28.80 | 3088121.7 | 0.0 | 264.79 |
| sr/cg/matvec | gpu | 950.615 | 6.160 | 28.36 | 3041017.3 | 177.2 | 88.60 |
| sr/cg/precond | gpu | 2.674 | 0.392 | 0.08 | 8553.6 | 87.6 | 87.60 |
| sr/trust | gpu | 11.443 | 11.443 | 0.34 | 36606.8 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.195 | 0.132 | 0.33 | 35812.1 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.303 | 0.34 | 36158.7 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 154.9 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 26.3 | 0.0 | 1.00 |

### prof 2026-09-22 16:05:57 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3299, mean 3355.45 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 188.0 | 0.0 | 1.00 |
| net_fwd | gpu | 8.831 | 0.114 | 0.26 | 29133.3 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 413.8 | 1.0 | 1.00 |
| lu | gpu | 0.663 | 0.008 | 0.02 | 2188.1 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 60.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 626.067 | 635.580 | 18.66 | 2065395.5 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 484.132 | 0.122 | 14.43 | 1597150.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.400 | 0.414 | 2.75 | 304826.6 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 305.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.979 | 0.293 | 2.68 | 296840.5 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.101 | 0.013 | 0.00 | 332.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.405 | 0.013 | 0.01 | 1336.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.751 | 0.021 | 0.05 | 5776.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 101.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.429 | 0.083 | 1.47 | 163067.4 | 36.0 | 3.00 |
| record_sweeps | gpu | 1245.425 | 1245.440 | 37.12 | 4108658.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 961.893 | 0.240 | 28.67 | 3173286.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 184.656 | 0.836 | 5.50 | 609179.7 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.189 | 0.028 | 0.01 | 621.9 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 179.684 | 0.593 | 5.35 | 592777.1 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 625.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.770 | 0.025 | 0.02 | 2539.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.684 | 0.043 | 0.11 | 12152.5 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 201.9 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.595 | 0.164 | 2.94 | 325265.7 | 72.0 | 6.00 |
| record | gpu | 471.073 | 471.081 | 14.04 | 1554069.1 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.838 | 0.419 | 0.59 | 65446.2 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.208 | 0.327 | 0.51 | 56770.4 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.256 | 0.009 | 0.01 | 845.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3645.6 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 114.7 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.011 | 0.04 | 3888.8 | 2.0 | 2.00 |
| record/jet_pass | gpu | 248.295 | 267.715 | 7.40 | 819125.8 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 219.959 | 0.206 | 6.56 | 725645.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.745 | 0.009 | 0.80 | 88231.2 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.578 | 267.483 | 0.05 | 5205.9 | 8.0 | 2.00 |
| record/exchange | gpu | 111.845 | 112.018 | 3.33 | 368975.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.409 | 0.287 | 1.83 | 202587.2 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 266.2 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.790 | 0.203 | 1.78 | 197245.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 204.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.320 | 0.009 | 0.01 | 1056.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3662.7 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 61.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 97.9 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.730 | 0.145 | 1.36 | 150862.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.617 | 0.018 | 0.14 | 15232.0 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.527 | 3.32 | 367926.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.436 | 0.017 | 0.01 | 1438.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 84.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.435 | 90.429 | 2.70 | 298344.9 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.181 | 0.01 | 598.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.506 | 0.033 | 0.02 | 1669.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.484 | 0.242 | 0.97 | 107163.8 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.503 | 0.175 | 0.76 | 84134.1 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.656 | 0.009 | 0.94 | 104433.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.143 | 0.00 | 471.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.038 | 0.00 | 123.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.633 | 0.026 | 0.44 | 48274.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 403.6 | 0.0 | 1.00 |
| sr/grad | gpu | 5.858 | 0.013 | 0.17 | 19325.9 | 2.0 | 1.00 |
| sr/cg | gpu | 970.744 | 979.516 | 28.93 | 3202485.9 | 442.4 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 972.060 | 28.97 | 3206825.2 | 0.0 | 266.86 |
| sr/cg/matvec | gpu | 957.462 | 6.119 | 28.53 | 3158667.5 | 178.6 | 89.29 |
| sr/cg/precond | gpu | 2.613 | 0.395 | 0.08 | 8620.1 | 88.3 | 88.29 |
| sr/trust | gpu | 11.419 | 11.418 | 0.34 | 37669.8 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.176 | 0.129 | 0.33 | 36868.2 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.281 | 0.34 | 37216.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 159.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 27.0 | 0.0 | 1.00 |

### prof 2026-09-22 16:11:44 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3399, mean 3358.84 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 192.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.823 | 0.114 | 0.26 | 29987.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 426.1 | 1.0 | 1.00 |
| lu | gpu | 0.661 | 0.008 | 0.02 | 2247.5 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 61.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 625.389 | 634.891 | 18.62 | 2125696.3 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 483.614 | 0.122 | 14.40 | 1643804.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.298 | 0.414 | 2.75 | 313721.4 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 315.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.881 | 0.293 | 2.68 | 305505.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.100 | 0.013 | 0.00 | 341.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.404 | 0.013 | 0.01 | 1374.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.748 | 0.021 | 0.05 | 5943.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 104.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.370 | 0.083 | 1.47 | 167809.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1244.013 | 1244.027 | 37.04 | 4228399.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 960.813 | 0.240 | 28.61 | 3265802.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 184.444 | 0.835 | 5.49 | 626925.6 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 640.3 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 179.484 | 0.593 | 5.34 | 610064.5 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 644.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.769 | 0.025 | 0.02 | 2614.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.673 | 0.043 | 0.11 | 12484.9 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 208.0 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.480 | 0.164 | 2.93 | 334734.5 | 72.0 | 6.00 |
| record | gpu | 470.512 | 470.519 | 14.01 | 1599268.7 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.818 | 0.418 | 0.59 | 67362.6 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.189 | 0.326 | 0.51 | 58424.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.256 | 0.009 | 0.01 | 869.8 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3756.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 118.2 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.04 | 4007.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 248.003 | 267.404 | 7.38 | 842962.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 219.707 | 0.206 | 6.54 | 746783.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.715 | 0.009 | 0.80 | 90804.2 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.568 | 267.172 | 0.05 | 5330.0 | 8.0 | 2.00 |
| record/exchange | gpu | 111.721 | 111.890 | 3.33 | 379739.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.343 | 0.287 | 1.83 | 208504.5 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 274.3 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.726 | 0.203 | 1.78 | 203008.5 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 210.8 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.318 | 0.009 | 0.01 | 1081.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3773.5 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 63.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 100.9 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.677 | 0.145 | 1.36 | 155256.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.612 | 0.018 | 0.14 | 15675.7 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.398 | 3.32 | 378642.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.431 | 0.017 | 0.01 | 1465.7 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 87.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.319 | 90.313 | 2.69 | 306994.4 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.176 | 0.01 | 599.9 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.501 | 0.033 | 0.01 | 1701.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.450 | 0.242 | 0.97 | 110299.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.474 | 0.175 | 0.76 | 86586.7 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.613 | 0.009 | 0.94 | 107451.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.141 | 0.00 | 477.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 127.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.615 | 0.026 | 0.44 | 49676.3 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 415.8 | 0.0 | 1.00 |
| sr/grad | gpu | 5.851 | 0.013 | 0.17 | 19886.0 | 2.0 | 1.00 |
| sr/cg | gpu | 976.849 | 985.610 | 29.08 | 3320310.3 | 445.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 978.185 | 29.12 | 3324849.5 | 0.0 | 268.75 |
| sr/cg/matvec | gpu | 963.707 | 6.079 | 28.69 | 3275638.6 | 179.8 | 89.92 |
| sr/cg/precond | gpu | 2.555 | 0.398 | 0.08 | 8686.1 | 88.9 | 88.92 |
| sr/trust | gpu | 11.395 | 11.395 | 0.34 | 38732.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.157 | 0.127 | 0.33 | 37924.0 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.260 | 0.34 | 38274.2 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 164.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 27.7 | 0.0 | 1.00 |

### prof 2026-09-22 16:17:32 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3499, mean 3362.23 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 197.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.815 | 0.114 | 0.26 | 30842.3 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 438.5 | 1.0 | 1.00 |
| lu | gpu | 0.659 | 0.008 | 0.02 | 2306.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 63.7 | 1.0 | 1.00 |
| therm_sweeps | gpu | 624.748 | 634.240 | 18.58 | 2185993.0 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 483.125 | 0.122 | 14.37 | 1690453.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.203 | 0.414 | 2.74 | 322617.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 324.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.789 | 0.293 | 2.67 | 314171.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.100 | 0.013 | 0.00 | 350.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.403 | 0.013 | 0.01 | 1411.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.746 | 0.021 | 0.05 | 6109.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 107.2 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.314 | 0.083 | 1.47 | 172551.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1242.683 | 1242.697 | 36.96 | 4348148.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 959.796 | 0.240 | 28.55 | 3358326.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 184.244 | 0.835 | 5.48 | 644671.3 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 658.6 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 179.295 | 0.593 | 5.33 | 627351.7 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.190 | 0.026 | 0.01 | 663.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.769 | 0.025 | 0.02 | 2689.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.663 | 0.043 | 0.11 | 12817.4 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 214.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.372 | 0.164 | 2.93 | 344204.6 | 72.0 | 6.00 |
| record | gpu | 469.983 | 469.990 | 13.98 | 1644469.3 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.800 | 0.418 | 0.59 | 69279.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.170 | 0.326 | 0.51 | 60078.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.256 | 0.009 | 0.01 | 894.5 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3867.1 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 121.7 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.04 | 4125.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 247.728 | 267.110 | 7.37 | 866800.3 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 219.469 | 0.205 | 6.53 | 767922.6 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.687 | 0.009 | 0.79 | 93377.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.559 | 266.878 | 0.05 | 5454.2 | 8.0 | 2.00 |
| record/exchange | gpu | 111.604 | 111.768 | 3.32 | 390502.9 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.281 | 0.287 | 1.82 | 214422.0 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 282.3 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.666 | 0.203 | 1.77 | 208771.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 217.0 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.316 | 0.009 | 0.01 | 1106.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3884.3 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 65.5 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 103.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.627 | 0.145 | 1.36 | 159650.5 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.607 | 0.018 | 0.14 | 16119.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.277 | 3.31 | 389358.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.427 | 0.017 | 0.01 | 1493.0 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 89.5 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.210 | 90.204 | 2.68 | 315643.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.172 | 0.01 | 601.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.496 | 0.033 | 0.01 | 1734.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.419 | 0.242 | 0.96 | 113434.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.447 | 0.175 | 0.76 | 89039.0 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.572 | 0.009 | 0.94 | 110468.7 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.138 | 0.00 | 484.4 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 130.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.598 | 0.026 | 0.43 | 51079.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 428.1 | 0.0 | 1.00 |
| sr/grad | gpu | 5.844 | 0.013 | 0.17 | 20447.2 | 2.0 | 1.00 |
| sr/cg | gpu | 982.796 | 991.547 | 29.23 | 3438802.1 | 448.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 984.149 | 29.27 | 3443537.0 | 0.0 | 270.58 |
| sr/cg/matvec | gpu | 969.783 | 6.043 | 28.84 | 3393271.7 | 181.1 | 90.53 |
| sr/cg/precond | gpu | 2.501 | 0.400 | 0.07 | 8752.6 | 89.5 | 89.53 |
| sr/trust | gpu | 11.373 | 11.373 | 0.34 | 39795.6 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.140 | 0.124 | 0.33 | 38980.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.241 | 0.33 | 39332.3 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 169.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 28.5 | 0.0 | 1.00 |

### prof 2026-09-22 16:23:20 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3599, mean 3365.65 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 203.2 | 0.0 | 1.00 |
| net_fwd | gpu | 8.807 | 0.114 | 0.26 | 31696.7 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 450.9 | 1.0 | 1.00 |
| lu | gpu | 0.657 | 0.008 | 0.02 | 2366.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 65.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 624.143 | 633.626 | 18.54 | 2246289.1 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 482.663 | 0.122 | 14.34 | 1737102.7 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.112 | 0.414 | 2.74 | 331512.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 333.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.702 | 0.293 | 2.67 | 322837.2 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.100 | 0.013 | 0.00 | 360.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.403 | 0.013 | 0.01 | 1449.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.744 | 0.021 | 0.05 | 6276.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 110.3 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.262 | 0.083 | 1.46 | 177292.3 | 36.0 | 3.00 |
| record_sweeps | gpu | 1241.428 | 1241.442 | 36.89 | 4467898.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 958.836 | 0.240 | 28.49 | 3450851.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 184.056 | 0.835 | 5.47 | 662417.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 676.9 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 179.116 | 0.593 | 5.32 | 644639.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 681.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.768 | 0.025 | 0.02 | 2764.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.654 | 0.043 | 0.11 | 13149.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 220.3 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.270 | 0.164 | 2.92 | 353673.3 | 72.0 | 6.00 |
| record | gpu | 469.483 | 469.491 | 13.95 | 1689670.7 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.782 | 0.418 | 0.59 | 71196.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.153 | 0.326 | 0.51 | 61732.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.255 | 0.009 | 0.01 | 919.2 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 3977.9 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 125.2 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.04 | 4243.4 | 2.0 | 2.00 |
| record/jet_pass | gpu | 247.468 | 266.833 | 7.35 | 890637.8 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 219.245 | 0.205 | 6.51 | 789061.5 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.660 | 0.009 | 0.79 | 95950.4 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.550 | 266.601 | 0.05 | 5578.4 | 8.0 | 2.00 |
| record/exchange | gpu | 111.494 | 111.654 | 3.31 | 401267.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.222 | 0.287 | 1.82 | 220339.4 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 290.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.609 | 0.203 | 1.77 | 214534.2 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 223.2 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.314 | 0.009 | 0.01 | 1131.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 3995.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 67.4 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 106.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.581 | 0.145 | 1.35 | 164044.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.602 | 0.018 | 0.14 | 16563.2 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.163 | 3.30 | 400074.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.422 | 0.017 | 0.01 | 1520.2 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 92.1 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.106 | 90.101 | 2.68 | 324293.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.168 | 0.00 | 603.1 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.491 | 0.033 | 0.01 | 1766.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.390 | 0.242 | 0.96 | 116570.3 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.421 | 0.175 | 0.76 | 91491.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.533 | 0.009 | 0.94 | 113486.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.136 | 0.00 | 490.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 134.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.582 | 0.026 | 0.43 | 52481.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 440.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.837 | 0.013 | 0.17 | 21007.2 | 2.0 | 1.00 |
| sr/cg | gpu | 988.634 | 997.376 | 29.37 | 3558094.7 | 451.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 990.002 | 29.41 | 3563016.5 | 0.0 | 272.38 |
| sr/cg/matvec | gpu | 975.743 | 6.010 | 28.99 | 3511697.7 | 182.3 | 91.13 |
| sr/cg/precond | gpu | 2.451 | 0.403 | 0.07 | 8820.0 | 90.1 | 90.13 |
| sr/trust | gpu | 11.353 | 11.353 | 0.34 | 40859.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.124 | 0.122 | 0.33 | 40036.9 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.223 | 0.33 | 40390.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 174.1 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 29.2 | 0.0 | 1.00 |

### prof 2026-09-22 16:29:07 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3699, mean 3368.34 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 209.3 | 0.0 | 1.00 |
| net_fwd | gpu | 8.800 | 0.114 | 0.26 | 32551.4 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 463.3 | 1.0 | 1.00 |
| lu | gpu | 0.656 | 0.008 | 0.02 | 2425.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 67.4 | 1.0 | 1.00 |
| therm_sweeps | gpu | 623.571 | 633.046 | 18.51 | 2306590.1 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 482.226 | 0.122 | 14.32 | 1783755.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 92.027 | 0.414 | 2.73 | 340408.0 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 342.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.620 | 0.293 | 2.66 | 331503.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.100 | 0.013 | 0.00 | 369.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.402 | 0.013 | 0.01 | 1486.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.742 | 0.021 | 0.05 | 6442.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 113.4 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.212 | 0.083 | 1.46 | 182034.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1240.241 | 1240.255 | 36.82 | 4587651.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 957.928 | 0.240 | 28.44 | 3543375.7 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.878 | 0.835 | 5.46 | 680164.3 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 695.3 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.948 | 0.593 | 5.31 | 661927.5 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 700.6 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.768 | 0.025 | 0.02 | 2839.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.645 | 0.043 | 0.11 | 13482.2 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 226.4 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.173 | 0.164 | 2.91 | 363143.4 | 72.0 | 6.00 |
| record | gpu | 469.012 | 469.019 | 13.92 | 1734873.6 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.766 | 0.418 | 0.59 | 73113.7 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.136 | 0.326 | 0.51 | 63387.3 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.255 | 0.009 | 0.01 | 943.9 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 4088.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 128.7 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.04 | 4361.6 | 2.0 | 2.00 |
| record/jet_pass | gpu | 247.223 | 266.571 | 7.34 | 914476.9 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 219.032 | 0.205 | 6.50 | 810201.0 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.635 | 0.009 | 0.79 | 98524.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.542 | 266.339 | 0.05 | 5702.7 | 8.0 | 2.00 |
| record/exchange | gpu | 111.390 | 111.546 | 3.31 | 412031.2 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.167 | 0.287 | 1.82 | 226257.0 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 298.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.556 | 0.203 | 1.77 | 220297.0 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 229.4 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.313 | 0.009 | 0.01 | 1156.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4105.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 69.3 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 109.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.536 | 0.145 | 1.35 | 168438.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.598 | 0.018 | 0.14 | 17006.9 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 111.054 | 3.30 | 410789.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.418 | 0.017 | 0.01 | 1547.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 94.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 90.009 | 90.003 | 2.67 | 332942.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.163 | 0.00 | 604.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.486 | 0.033 | 0.01 | 1799.2 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.362 | 0.242 | 0.96 | 119705.7 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.397 | 0.175 | 0.75 | 93943.7 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.496 | 0.009 | 0.94 | 116503.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.134 | 0.00 | 497.5 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 138.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.567 | 0.026 | 0.43 | 53883.6 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 452.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.831 | 0.013 | 0.17 | 21568.6 | 2.0 | 1.00 |
| sr/cg | gpu | 993.602 | 1002.335 | 29.50 | 3675333.8 | 454.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 994.986 | 29.54 | 3680452.4 | 0.0 | 273.92 |
| sr/cg/matvec | gpu | 980.828 | 5.977 | 29.12 | 3628083.6 | 183.3 | 91.64 |
| sr/cg/precond | gpu | 2.402 | 0.405 | 0.07 | 8886.4 | 90.6 | 90.64 |
| sr/trust | gpu | 11.334 | 11.333 | 0.34 | 41922.7 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.109 | 0.120 | 0.33 | 41093.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.205 | 0.33 | 41448.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 179.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 30.0 | 0.0 | 1.00 |

### prof 2026-09-22 16:34:55 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3799, mean 3371.39 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 214.2 | 0.0 | 1.00 |
| net_fwd | gpu | 8.793 | 0.114 | 0.26 | 33405.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 475.6 | 1.0 | 1.00 |
| lu | gpu | 0.654 | 0.008 | 0.02 | 2485.0 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 69.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 623.031 | 632.497 | 18.48 | 2366893.6 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 481.814 | 0.122 | 14.29 | 1830411.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.946 | 0.414 | 2.73 | 349303.6 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 351.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.542 | 0.293 | 2.66 | 340169.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.100 | 0.013 | 0.00 | 378.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.401 | 0.013 | 0.01 | 1524.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.740 | 0.021 | 0.05 | 6609.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 116.4 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.164 | 0.083 | 1.46 | 186775.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1239.116 | 1239.130 | 36.75 | 4707401.7 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 957.068 | 0.240 | 28.39 | 3635900.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.709 | 0.835 | 5.45 | 697911.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 713.7 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.788 | 0.593 | 5.30 | 679215.8 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 719.2 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.767 | 0.025 | 0.02 | 2914.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.636 | 0.043 | 0.11 | 13814.7 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 232.5 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 98.082 | 0.164 | 2.91 | 372612.3 | 72.0 | 6.00 |
| record | gpu | 468.564 | 468.572 | 13.90 | 1780074.8 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.750 | 0.417 | 0.59 | 75030.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.121 | 0.325 | 0.51 | 65041.3 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.255 | 0.009 | 0.01 | 968.6 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 4199.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 132.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.03 | 4479.8 | 2.0 | 2.00 |
| record/jet_pass | gpu | 246.990 | 266.323 | 7.33 | 938315.1 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 218.831 | 0.205 | 6.49 | 831340.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.612 | 0.009 | 0.79 | 101097.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.534 | 266.091 | 0.05 | 5826.9 | 8.0 | 2.00 |
| record/exchange | gpu | 111.291 | 111.443 | 3.30 | 422795.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.115 | 0.287 | 1.81 | 232174.7 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 306.6 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.505 | 0.203 | 1.77 | 226060.1 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 235.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.311 | 0.009 | 0.01 | 1181.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4216.7 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 71.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 112.8 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.494 | 0.145 | 1.35 | 172832.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.593 | 0.018 | 0.14 | 17450.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.952 | 3.29 | 421506.3 | 0.0 | 2.00 |
| record/assemble | gpu | 0.415 | 0.017 | 0.01 | 1574.8 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 97.2 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.916 | 89.911 | 2.67 | 341591.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.160 | 0.00 | 606.4 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.482 | 0.033 | 0.01 | 1831.7 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.335 | 0.242 | 0.96 | 122841.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.374 | 0.175 | 0.75 | 96396.0 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.461 | 0.009 | 0.93 | 119520.4 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.133 | 0.00 | 503.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 141.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.553 | 0.026 | 0.43 | 55285.8 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 464.7 | 0.0 | 1.00 |
| sr/grad | gpu | 5.825 | 0.013 | 0.17 | 22128.4 | 2.0 | 1.00 |
| sr/cg | gpu | 998.814 | 1007.538 | 29.63 | 3794492.7 | 456.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1000.211 | 29.67 | 3799802.4 | 0.0 | 275.52 |
| sr/cg/matvec | gpu | 986.149 | 5.947 | 29.25 | 3746378.5 | 184.3 | 92.17 |
| sr/cg/precond | gpu | 2.357 | 0.408 | 0.07 | 8953.4 | 91.2 | 91.17 |
| sr/trust | gpu | 11.315 | 11.315 | 0.34 | 42985.9 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.095 | 0.118 | 0.33 | 42149.7 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.189 | 0.33 | 42507.0 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 183.8 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 30.7 | 0.0 | 1.00 |

### prof 2026-09-22 16:40:42 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3899, mean 3373.82 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 220.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.787 | 0.114 | 0.26 | 34260.3 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 488.0 | 1.0 | 1.00 |
| lu | gpu | 0.653 | 0.008 | 0.02 | 2544.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 71.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 622.519 | 631.977 | 18.45 | 2427200.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 481.423 | 0.122 | 14.27 | 1877068.9 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.870 | 0.414 | 2.72 | 358200.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 360.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.468 | 0.293 | 2.65 | 348836.4 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 387.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.401 | 0.013 | 0.01 | 1561.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.738 | 0.021 | 0.05 | 6775.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 119.5 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.120 | 0.083 | 1.46 | 191518.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1238.050 | 1238.064 | 36.70 | 4827156.9 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 956.252 | 0.240 | 28.34 | 3728427.0 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.549 | 0.836 | 5.44 | 715659.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 732.1 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.637 | 0.593 | 5.29 | 696505.3 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 737.8 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.767 | 0.025 | 0.02 | 2989.3 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.628 | 0.043 | 0.11 | 14147.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 238.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.995 | 0.164 | 2.90 | 382081.9 | 72.0 | 6.00 |
| record | gpu | 468.141 | 468.149 | 13.88 | 1825283.2 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.736 | 0.418 | 0.58 | 76949.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.106 | 0.326 | 0.51 | 66697.5 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.255 | 0.009 | 0.01 | 993.3 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.105 | 0.016 | 0.03 | 4310.2 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 135.6 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.03 | 4598.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 246.770 | 266.087 | 7.31 | 962155.3 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 218.641 | 0.206 | 6.48 | 852480.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.589 | 0.009 | 0.79 | 103671.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.527 | 265.855 | 0.05 | 5952.0 | 8.0 | 2.00 |
| record/exchange | gpu | 111.198 | 111.346 | 3.30 | 433560.6 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.065 | 0.287 | 1.81 | 238092.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 314.7 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.457 | 0.204 | 1.76 | 231823.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 241.8 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.310 | 0.009 | 0.01 | 1206.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4327.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 73.1 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 115.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.454 | 0.145 | 1.35 | 177226.5 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.589 | 0.018 | 0.14 | 17894.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.854 | 3.29 | 432220.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.411 | 0.017 | 0.01 | 1602.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 99.8 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.829 | 89.824 | 2.66 | 350241.6 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.156 | 0.00 | 608.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.478 | 0.033 | 0.01 | 1864.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.310 | 0.242 | 0.96 | 125977.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.352 | 0.175 | 0.75 | 98848.8 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.428 | 0.009 | 0.93 | 122537.8 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.131 | 0.00 | 510.7 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 145.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.539 | 0.026 | 0.43 | 56688.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 477.3 | 0.0 | 1.00 |
| sr/grad | gpu | 5.819 | 0.013 | 0.17 | 22688.9 | 2.0 | 1.00 |
| sr/cg | gpu | 1003.289 | 1012.005 | 29.74 | 3911824.0 | 459.2 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1004.697 | 29.78 | 3917311.8 | 0.0 | 276.91 |
| sr/cg/matvec | gpu | 990.727 | 5.920 | 29.37 | 3862845.6 | 185.3 | 92.64 |
| sr/cg/precond | gpu | 2.314 | 0.410 | 0.07 | 9020.9 | 91.6 | 91.64 |
| sr/trust | gpu | 11.297 | 11.297 | 0.33 | 44048.5 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.081 | 0.116 | 0.33 | 43205.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.173 | 0.33 | 43564.3 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 188.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 31.6 | 0.0 | 1.00 |

### prof 2026-09-22 16:46:30 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 3999, mean 3376.55 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 226.3 | 0.0 | 1.00 |
| net_fwd | gpu | 8.781 | 0.114 | 0.26 | 35115.0 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 500.4 | 1.0 | 1.00 |
| lu | gpu | 0.651 | 0.008 | 0.02 | 2603.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 72.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 622.031 | 631.481 | 18.42 | 2487502.2 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 481.051 | 0.122 | 14.25 | 1923721.7 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.797 | 0.414 | 2.72 | 367096.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 370.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.398 | 0.293 | 2.65 | 357502.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 397.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.400 | 0.013 | 0.01 | 1599.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.736 | 0.021 | 0.05 | 6942.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 122.6 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.077 | 0.083 | 1.45 | 196260.1 | 36.0 | 3.00 |
| record_sweeps | gpu | 1237.038 | 1237.052 | 36.64 | 4946915.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 955.478 | 0.240 | 28.30 | 3820956.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.398 | 0.836 | 5.43 | 733407.3 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 750.4 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.493 | 0.593 | 5.29 | 713795.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 756.4 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.766 | 0.025 | 0.02 | 3064.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.621 | 0.043 | 0.11 | 14479.6 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 244.8 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.912 | 0.164 | 2.90 | 391551.9 | 72.0 | 6.00 |
| record | gpu | 467.739 | 467.746 | 13.85 | 1870486.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.722 | 0.418 | 0.58 | 78866.8 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.092 | 0.326 | 0.51 | 68352.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.255 | 0.009 | 0.01 | 1018.0 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 4421.0 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 139.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.03 | 4716.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 246.560 | 265.863 | 7.30 | 985992.8 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 218.459 | 0.206 | 6.47 | 873619.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.568 | 0.009 | 0.79 | 106244.6 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.519 | 265.631 | 0.04 | 6076.1 | 8.0 | 2.00 |
| record/exchange | gpu | 111.109 | 111.254 | 3.29 | 444326.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 61.018 | 0.287 | 1.81 | 244011.5 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 322.8 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.412 | 0.204 | 1.76 | 237587.5 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 248.0 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.308 | 0.009 | 0.01 | 1231.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4438.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 74.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 118.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.417 | 0.145 | 1.35 | 181620.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.586 | 0.018 | 0.14 | 18338.1 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.762 | 3.28 | 442937.4 | 0.0 | 2.00 |
| record/assemble | gpu | 0.407 | 0.017 | 0.01 | 1629.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 102.3 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.745 | 89.740 | 2.66 | 358891.3 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.152 | 0.00 | 609.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.474 | 0.033 | 0.01 | 1896.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.286 | 0.242 | 0.96 | 129113.0 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.332 | 0.175 | 0.75 | 101301.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.397 | 0.009 | 0.93 | 125555.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.129 | 0.00 | 517.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 149.5 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.526 | 0.026 | 0.43 | 58091.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 489.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.814 | 0.013 | 0.17 | 23249.5 | 2.0 | 1.00 |
| sr/cg | gpu | 1007.970 | 1016.679 | 29.85 | 4030873.8 | 461.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1009.391 | 29.89 | 4036553.5 | 0.0 | 278.35 |
| sr/cg/matvec | gpu | 995.507 | 5.892 | 29.48 | 3981032.2 | 186.2 | 93.12 |
| sr/cg/precond | gpu | 2.272 | 0.412 | 0.07 | 9087.6 | 92.1 | 92.12 |
| sr/trust | gpu | 11.281 | 11.281 | 0.33 | 45111.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.068 | 0.114 | 0.33 | 44261.2 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.158 | 0.33 | 44621.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 193.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 32.3 | 0.0 | 1.00 |

### prof 2026-09-22 16:52:18 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4099, mean 3378.97 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 232.4 | 0.0 | 1.00 |
| net_fwd | gpu | 8.775 | 0.114 | 0.26 | 35969.2 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 512.7 | 1.0 | 1.00 |
| lu | gpu | 0.650 | 0.008 | 0.02 | 2663.0 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 74.7 | 1.0 | 1.00 |
| therm_sweeps | gpu | 621.567 | 631.010 | 18.40 | 2547803.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 480.696 | 0.122 | 14.23 | 1970373.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.728 | 0.414 | 2.71 | 375993.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 379.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.332 | 0.293 | 2.64 | 366170.1 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 406.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.399 | 0.013 | 0.01 | 1636.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.734 | 0.021 | 0.05 | 7108.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 125.7 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 49.037 | 0.083 | 1.45 | 201002.3 | 36.0 | 3.00 |
| record_sweeps | gpu | 1236.076 | 1236.090 | 36.58 | 5066676.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 954.742 | 0.240 | 28.26 | 3913488.7 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.254 | 0.835 | 5.42 | 751156.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.188 | 0.028 | 0.01 | 768.7 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.357 | 0.593 | 5.28 | 731085.9 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 775.0 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.766 | 0.025 | 0.02 | 3139.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.614 | 0.043 | 0.11 | 14812.0 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 251.0 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.834 | 0.164 | 2.90 | 401021.4 | 72.0 | 6.00 |
| record | gpu | 467.356 | 467.363 | 13.83 | 1915690.5 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.708 | 0.418 | 0.58 | 80784.2 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.079 | 0.326 | 0.51 | 70007.0 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.254 | 0.009 | 0.01 | 1042.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 4531.8 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 142.6 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.03 | 4834.4 | 2.0 | 2.00 |
| record/jet_pass | gpu | 246.360 | 265.650 | 7.29 | 1009830.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 218.287 | 0.206 | 6.46 | 894758.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.547 | 0.009 | 0.79 | 108817.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.513 | 265.418 | 0.04 | 6200.3 | 8.0 | 2.00 |
| record/exchange | gpu | 111.025 | 111.166 | 3.29 | 455090.9 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.973 | 0.287 | 1.80 | 249929.7 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 330.9 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.368 | 0.204 | 1.76 | 243350.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 254.2 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.307 | 0.009 | 0.01 | 1256.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4549.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 76.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 121.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.381 | 0.145 | 1.34 | 186014.8 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.582 | 0.018 | 0.14 | 18781.8 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.674 | 3.28 | 453654.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.404 | 0.017 | 0.01 | 1656.8 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 104.9 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.666 | 89.662 | 2.65 | 367542.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.149 | 0.00 | 611.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.471 | 0.033 | 0.01 | 1929.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.264 | 0.242 | 0.95 | 132249.0 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.312 | 0.175 | 0.75 | 103753.8 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.367 | 0.009 | 0.93 | 128573.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.128 | 0.00 | 523.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 153.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.514 | 0.026 | 0.43 | 59493.9 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 501.7 | 0.0 | 1.00 |
| sr/grad | gpu | 5.809 | 0.013 | 0.17 | 23809.5 | 2.0 | 1.00 |
| sr/cg | gpu | 1012.240 | 1020.941 | 29.96 | 4149171.9 | 463.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1013.673 | 30.00 | 4155047.3 | 0.0 | 279.67 |
| sr/cg/matvec | gpu | 999.871 | 5.865 | 29.59 | 4098472.3 | 187.1 | 93.56 |
| sr/cg/precond | gpu | 2.233 | 0.414 | 0.07 | 9154.1 | 92.6 | 92.56 |
| sr/trust | gpu | 11.265 | 11.265 | 0.33 | 46174.8 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.056 | 0.112 | 0.33 | 45317.8 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.144 | 0.33 | 45680.3 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 198.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 33.1 | 0.0 | 1.00 |

### prof 2026-09-22 16:58:06 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4199, mean 3381.43 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 238.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.769 | 0.114 | 0.26 | 36822.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 525.1 | 1.0 | 1.00 |
| lu | gpu | 0.648 | 0.008 | 0.02 | 2722.3 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 76.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 621.125 | 630.561 | 18.37 | 2608104.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 480.359 | 0.122 | 14.21 | 2017026.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.662 | 0.414 | 2.71 | 384889.7 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 388.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.268 | 0.293 | 2.64 | 374837.0 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 415.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.399 | 0.013 | 0.01 | 1674.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.733 | 0.021 | 0.05 | 7275.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 128.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.998 | 0.083 | 1.45 | 205743.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1235.160 | 1235.174 | 36.53 | 5186438.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 954.042 | 0.240 | 28.21 | 4006022.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 183.116 | 0.836 | 5.42 | 768905.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 787.0 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.227 | 0.593 | 5.27 | 748376.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 793.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.765 | 0.025 | 0.02 | 3214.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.607 | 0.043 | 0.11 | 15144.4 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 257.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.759 | 0.164 | 2.89 | 410490.6 | 72.0 | 6.00 |
| record | gpu | 466.991 | 466.998 | 13.81 | 1960894.3 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.696 | 0.418 | 0.58 | 82701.6 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.066 | 0.326 | 0.50 | 71661.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.254 | 0.009 | 0.01 | 1067.4 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 4642.6 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 146.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.179 | 0.010 | 0.03 | 4952.6 | 2.0 | 2.00 |
| record/jet_pass | gpu | 246.170 | 265.447 | 7.28 | 1033669.2 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 218.123 | 0.206 | 6.45 | 915897.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.528 | 0.009 | 0.78 | 111391.1 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.506 | 265.215 | 0.04 | 6324.7 | 8.0 | 2.00 |
| record/exchange | gpu | 110.945 | 111.082 | 3.28 | 465856.0 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.931 | 0.287 | 1.80 | 255847.9 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 339.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.327 | 0.203 | 1.75 | 249114.5 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 260.4 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.305 | 0.009 | 0.01 | 1281.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4660.0 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 78.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 124.7 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.346 | 0.145 | 1.34 | 190409.2 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.579 | 0.018 | 0.14 | 19225.5 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.591 | 3.27 | 464370.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.401 | 0.017 | 0.01 | 1684.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 107.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.591 | 89.586 | 2.65 | 376191.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.146 | 0.00 | 613.1 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.467 | 0.033 | 0.01 | 1961.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.242 | 0.242 | 0.95 | 135384.8 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.293 | 0.175 | 0.75 | 106206.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.338 | 0.009 | 0.93 | 131590.3 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.126 | 0.00 | 530.4 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 157.3 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.503 | 0.026 | 0.43 | 60897.0 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 513.8 | 0.0 | 1.00 |
| sr/grad | gpu | 5.804 | 0.013 | 0.17 | 24369.9 | 2.0 | 1.00 |
| sr/cg | gpu | 1016.462 | 1025.156 | 30.06 | 4268123.3 | 465.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1017.907 | 30.10 | 4274190.6 | 0.0 | 280.97 |
| sr/cg/matvec | gpu | 1004.182 | 5.841 | 29.70 | 4216561.6 | 188.0 | 93.99 |
| sr/cg/precond | gpu | 2.196 | 0.416 | 0.06 | 9220.9 | 93.0 | 92.99 |
| sr/trust | gpu | 11.250 | 11.250 | 0.33 | 47237.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.044 | 0.111 | 0.33 | 46373.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.131 | 0.33 | 46737.7 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 203.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 33.9 | 0.0 | 1.00 |

### prof 2026-09-22 17:03:55 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4299, mean 3383.89 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 244.0 | 0.0 | 1.00 |
| net_fwd | gpu | 8.764 | 0.114 | 0.26 | 37677.3 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 537.5 | 1.0 | 1.00 |
| lu | gpu | 0.647 | 0.008 | 0.02 | 2781.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 78.3 | 1.0 | 1.00 |
| therm_sweeps | gpu | 620.704 | 630.134 | 18.34 | 2668408.1 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 480.037 | 0.122 | 14.19 | 2063680.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.600 | 0.414 | 2.71 | 393787.1 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 397.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.208 | 0.293 | 2.64 | 383504.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 425.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.398 | 0.013 | 0.01 | 1711.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.731 | 0.021 | 0.05 | 7441.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 131.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.961 | 0.083 | 1.45 | 210484.9 | 36.0 | 3.00 |
| record_sweeps | gpu | 1234.286 | 1234.300 | 36.48 | 5306196.1 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 953.373 | 0.240 | 28.17 | 4098551.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.985 | 0.835 | 5.41 | 786654.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 805.4 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 178.104 | 0.593 | 5.26 | 765667.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 812.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.765 | 0.025 | 0.02 | 3289.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.600 | 0.043 | 0.11 | 15476.8 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 263.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.688 | 0.164 | 2.89 | 419959.5 | 72.0 | 6.00 |
| record | gpu | 466.642 | 466.650 | 13.79 | 2006095.7 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.683 | 0.418 | 0.58 | 84618.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.054 | 0.326 | 0.50 | 73315.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.254 | 0.009 | 0.01 | 1092.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 4753.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 149.6 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5070.7 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.989 | 265.254 | 7.27 | 1057506.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.966 | 0.206 | 6.44 | 937036.1 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.509 | 0.009 | 0.78 | 113964.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.500 | 265.022 | 0.04 | 6448.7 | 8.0 | 2.00 |
| record/exchange | gpu | 110.868 | 111.003 | 3.28 | 476621.6 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.890 | 0.287 | 1.80 | 261766.7 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 347.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.288 | 0.203 | 1.75 | 254878.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 266.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.304 | 0.009 | 0.01 | 1306.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4771.1 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 80.6 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 127.6 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.314 | 0.145 | 1.34 | 194803.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.575 | 0.018 | 0.14 | 19669.2 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.511 | 3.27 | 475088.7 | 0.0 | 2.00 |
| record/assemble | gpu | 0.398 | 0.017 | 0.01 | 1711.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 110.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.519 | 89.514 | 2.65 | 384840.6 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.143 | 0.00 | 614.7 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.464 | 0.033 | 0.01 | 1994.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.221 | 0.242 | 0.95 | 138520.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.275 | 0.175 | 0.75 | 108658.8 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.311 | 0.009 | 0.93 | 134607.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.125 | 0.00 | 536.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 160.9 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.492 | 0.026 | 0.43 | 62299.0 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 525.9 | 0.0 | 1.00 |
| sr/grad | gpu | 5.799 | 0.013 | 0.17 | 24930.5 | 2.0 | 1.00 |
| sr/cg | gpu | 1020.605 | 1029.293 | 30.16 | 4387582.7 | 468.1 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1022.062 | 30.20 | 4393843.0 | 0.0 | 282.24 |
| sr/cg/matvec | gpu | 1008.411 | 5.817 | 29.80 | 4335157.7 | 188.8 | 94.41 |
| sr/cg/precond | gpu | 2.160 | 0.418 | 0.06 | 9287.6 | 93.4 | 93.41 |
| sr/trust | gpu | 11.235 | 11.235 | 0.33 | 48299.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.032 | 0.109 | 0.33 | 47428.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.118 | 0.33 | 47794.6 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 207.9 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 34.6 | 0.0 | 1.00 |

### prof 2026-09-22 17:09:43 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4399, mean 3385.98 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 249.0 | 0.0 | 1.00 |
| net_fwd | gpu | 8.759 | 0.114 | 0.26 | 38531.5 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 549.8 | 1.0 | 1.00 |
| lu | gpu | 0.646 | 0.008 | 0.02 | 2840.9 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 80.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 620.303 | 629.726 | 18.32 | 2728712.0 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 479.731 | 0.122 | 14.17 | 2110336.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.540 | 0.414 | 2.70 | 402683.1 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.093 | 0.013 | 0.00 | 406.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.150 | 0.293 | 2.63 | 392171.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 434.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.398 | 0.013 | 0.01 | 1749.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.729 | 0.021 | 0.05 | 7607.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 134.9 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.926 | 0.083 | 1.44 | 215226.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1233.452 | 1233.465 | 36.43 | 5425954.6 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 952.735 | 0.240 | 28.14 | 4191080.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.860 | 0.835 | 5.40 | 804403.3 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 823.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.985 | 0.592 | 5.26 | 782957.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 830.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.765 | 0.025 | 0.02 | 3364.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.594 | 0.043 | 0.11 | 15809.2 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 269.4 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.620 | 0.164 | 2.88 | 429430.0 | 72.0 | 6.00 |
| record | gpu | 466.310 | 466.317 | 13.77 | 2051298.2 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.672 | 0.417 | 0.58 | 86535.2 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.043 | 0.326 | 0.50 | 74970.0 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.254 | 0.009 | 0.01 | 1116.8 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 4864.2 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 153.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5188.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.816 | 265.070 | 7.26 | 1081344.3 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.817 | 0.205 | 6.43 | 958175.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.492 | 0.009 | 0.78 | 116537.4 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.494 | 264.838 | 0.04 | 6572.8 | 8.0 | 2.00 |
| record/exchange | gpu | 110.795 | 110.927 | 3.27 | 487386.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.851 | 0.287 | 1.80 | 267684.9 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 355.1 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.250 | 0.203 | 1.75 | 260641.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 272.8 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.303 | 0.009 | 0.01 | 1331.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4881.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 82.4 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 130.6 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.282 | 0.145 | 1.34 | 199197.3 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.572 | 0.018 | 0.14 | 20113.2 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.436 | 3.26 | 485805.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.395 | 0.017 | 0.01 | 1738.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 112.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.450 | 89.446 | 2.64 | 393490.4 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.140 | 0.00 | 616.3 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.461 | 0.033 | 0.01 | 2026.8 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.202 | 0.242 | 0.95 | 141655.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.258 | 0.175 | 0.75 | 111111.3 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.286 | 0.009 | 0.92 | 137625.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.123 | 0.00 | 543.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 164.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.481 | 0.026 | 0.43 | 63701.5 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 538.0 | 0.0 | 1.00 |
| sr/grad | gpu | 5.795 | 0.013 | 0.17 | 25490.4 | 2.0 | 1.00 |
| sr/cg | gpu | 1024.311 | 1032.992 | 30.25 | 4505942.1 | 470.0 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1025.779 | 30.29 | 4512402.2 | 0.0 | 283.38 |
| sr/cg/matvec | gpu | 1012.199 | 5.793 | 29.89 | 4452662.5 | 189.6 | 94.79 |
| sr/cg/precond | gpu | 2.126 | 0.419 | 0.06 | 9353.7 | 93.8 | 93.79 |
| sr/trust | gpu | 11.221 | 11.221 | 0.33 | 49362.0 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.022 | 0.108 | 0.33 | 48484.2 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.105 | 0.33 | 48852.1 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 212.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 35.3 | 0.0 | 1.00 |

### prof 2026-09-22 17:15:31 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4499, mean 3388.10 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 255.2 | 0.0 | 1.00 |
| net_fwd | gpu | 8.754 | 0.114 | 0.26 | 39385.9 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 562.2 | 1.0 | 1.00 |
| lu | gpu | 0.645 | 0.008 | 0.02 | 2900.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 82.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 619.918 | 629.335 | 18.30 | 2789012.8 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 479.437 | 0.122 | 14.15 | 2156988.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.483 | 0.414 | 2.70 | 411580.1 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 416.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.095 | 0.293 | 2.63 | 400838.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 443.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.397 | 0.013 | 0.01 | 1786.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.728 | 0.021 | 0.05 | 7774.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 138.0 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.893 | 0.083 | 1.44 | 219968.1 | 36.0 | 3.00 |
| record_sweeps | gpu | 1232.654 | 1232.668 | 36.38 | 5545712.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 952.125 | 0.240 | 28.10 | 4283611.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.741 | 0.835 | 5.39 | 822152.4 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 842.1 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.872 | 0.592 | 5.25 | 800248.4 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 849.4 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.764 | 0.025 | 0.02 | 3439.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.588 | 0.043 | 0.11 | 16141.6 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 275.5 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.555 | 0.164 | 2.88 | 438897.7 | 72.0 | 6.00 |
| record | gpu | 465.993 | 466.000 | 13.75 | 2096500.6 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.660 | 0.417 | 0.58 | 88452.6 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.031 | 0.326 | 0.50 | 76624.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.254 | 0.009 | 0.01 | 1141.6 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 4974.9 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 156.6 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5307.1 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.650 | 264.893 | 7.25 | 1105180.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.674 | 0.205 | 6.42 | 979313.6 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.475 | 0.009 | 0.78 | 119110.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.489 | 264.661 | 0.04 | 6697.0 | 8.0 | 2.00 |
| record/exchange | gpu | 110.725 | 110.854 | 3.27 | 498151.4 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.814 | 0.287 | 1.79 | 273603.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 363.2 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.214 | 0.203 | 1.75 | 266405.4 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 279.0 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.302 | 0.009 | 0.01 | 1356.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 4992.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 84.3 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 133.6 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.253 | 0.145 | 1.34 | 203591.5 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.569 | 0.018 | 0.13 | 20556.9 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.363 | 3.26 | 496522.6 | 0.0 | 2.00 |
| record/assemble | gpu | 0.392 | 0.017 | 0.01 | 1765.8 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 115.1 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.384 | 89.380 | 2.64 | 402140.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.137 | 0.00 | 618.0 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.458 | 0.033 | 0.01 | 2059.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.183 | 0.242 | 0.95 | 144791.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.242 | 0.175 | 0.75 | 113564.2 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.261 | 0.009 | 0.92 | 140642.9 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.122 | 0.00 | 549.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 168.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.471 | 0.026 | 0.43 | 65103.7 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 550.1 | 0.0 | 1.00 |
| sr/grad | gpu | 5.790 | 0.013 | 0.17 | 26051.3 | 2.0 | 1.00 |
| sr/cg | gpu | 1027.954 | 1036.629 | 30.34 | 4624766.0 | 471.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1029.433 | 30.38 | 4631419.8 | 0.0 | 284.51 |
| sr/cg/matvec | gpu | 1015.920 | 5.771 | 29.98 | 4570625.8 | 190.3 | 95.17 |
| sr/cg/precond | gpu | 2.094 | 0.421 | 0.06 | 9420.4 | 94.2 | 94.17 |
| sr/trust | gpu | 11.208 | 11.208 | 0.33 | 50424.8 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.011 | 0.106 | 0.33 | 49540.1 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.094 | 0.33 | 49909.7 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 217.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 36.1 | 0.0 | 1.00 |

### prof 2026-09-22 17:21:18 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4599, mean 3389.84 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 260.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.750 | 0.114 | 0.26 | 40240.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 574.6 | 1.0 | 1.00 |
| lu | gpu | 0.644 | 0.008 | 0.02 | 2959.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 83.9 | 1.0 | 1.00 |
| therm_sweeps | gpu | 619.552 | 628.963 | 18.28 | 2849318.0 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 479.158 | 0.122 | 14.14 | 2203645.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.428 | 0.414 | 2.70 | 420476.4 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 425.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 89.042 | 0.293 | 2.63 | 409505.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.099 | 0.013 | 0.00 | 453.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.397 | 0.013 | 0.01 | 1824.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.727 | 0.021 | 0.05 | 7940.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 141.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.861 | 0.083 | 1.44 | 224710.0 | 36.0 | 3.00 |
| record_sweeps | gpu | 1231.893 | 1231.906 | 36.34 | 5665474.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 951.542 | 0.240 | 28.07 | 4376143.8 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.627 | 0.834 | 5.39 | 839901.3 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 860.5 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.764 | 0.592 | 5.24 | 817538.9 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 867.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.764 | 0.025 | 0.02 | 3514.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.582 | 0.043 | 0.11 | 16474.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 281.6 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.492 | 0.164 | 2.88 | 448367.7 | 72.0 | 6.00 |
| record | gpu | 465.689 | 465.696 | 13.74 | 2141701.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.650 | 0.417 | 0.58 | 90369.5 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.021 | 0.325 | 0.50 | 78278.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.254 | 0.009 | 0.01 | 1166.3 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5085.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 160.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5425.3 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.492 | 264.724 | 7.24 | 1129017.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.537 | 0.205 | 6.42 | 1000451.7 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.459 | 0.009 | 0.78 | 121683.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.483 | 264.492 | 0.04 | 6821.0 | 8.0 | 2.00 |
| record/exchange | gpu | 110.658 | 110.785 | 3.26 | 508916.7 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.779 | 0.287 | 1.79 | 279521.3 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 371.2 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.180 | 0.203 | 1.75 | 272169.0 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 285.2 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.300 | 0.009 | 0.01 | 1381.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 5103.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 86.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 136.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.224 | 0.145 | 1.33 | 207985.8 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.566 | 0.018 | 0.13 | 21000.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.294 | 3.25 | 507240.6 | 0.0 | 2.00 |
| record/assemble | gpu | 0.390 | 0.017 | 0.01 | 1793.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 117.7 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.322 | 89.317 | 2.63 | 410789.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.135 | 0.00 | 619.6 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.455 | 0.033 | 0.01 | 2091.7 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.165 | 0.241 | 0.95 | 147927.0 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.227 | 0.175 | 0.74 | 116016.7 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.237 | 0.009 | 0.92 | 143659.9 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.121 | 0.00 | 556.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 171.8 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.461 | 0.026 | 0.43 | 66506.4 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 562.2 | 0.0 | 1.00 |
| sr/grad | gpu | 5.786 | 0.013 | 0.17 | 26612.1 | 2.0 | 1.00 |
| sr/cg | gpu | 1031.162 | 1039.831 | 30.42 | 4742312.1 | 473.5 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1032.653 | 30.46 | 4749170.8 | 0.0 | 285.50 |
| sr/cg/matvec | gpu | 1019.205 | 5.749 | 30.07 | 4687322.5 | 191.0 | 95.50 |
| sr/cg/precond | gpu | 2.063 | 0.422 | 0.06 | 9485.8 | 94.5 | 94.50 |
| sr/trust | gpu | 11.196 | 11.196 | 0.33 | 51489.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 11.002 | 0.105 | 0.32 | 50597.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.083 | 0.33 | 50969.0 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 222.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 36.8 | 0.0 | 1.00 |

### prof 2026-09-22 17:27:06 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4699, mean 3391.91 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 266.7 | 0.0 | 1.00 |
| net_fwd | gpu | 8.746 | 0.114 | 0.26 | 41095.4 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 587.0 | 1.0 | 1.00 |
| lu | gpu | 0.642 | 0.008 | 0.02 | 3018.9 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 85.7 | 1.0 | 1.00 |
| therm_sweeps | gpu | 619.202 | 628.608 | 18.26 | 2909628.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 478.891 | 0.122 | 14.12 | 2250307.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.375 | 0.414 | 2.69 | 429373.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 434.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.992 | 0.293 | 2.62 | 418172.5 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 462.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.396 | 0.013 | 0.01 | 1861.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.725 | 0.021 | 0.05 | 8107.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 144.1 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.830 | 0.083 | 1.44 | 229451.6 | 36.0 | 3.00 |
| record_sweeps | gpu | 1231.164 | 1231.177 | 36.30 | 5785237.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 950.985 | 0.240 | 28.04 | 4468677.0 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.518 | 0.834 | 5.38 | 857650.8 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 878.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.661 | 0.592 | 5.24 | 834830.0 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 886.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.764 | 0.025 | 0.02 | 3589.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.577 | 0.043 | 0.11 | 16806.5 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 287.8 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.433 | 0.164 | 2.87 | 457837.5 | 72.0 | 6.00 |
| record | gpu | 465.398 | 465.405 | 13.72 | 2186904.8 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.640 | 0.417 | 0.58 | 92287.1 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.011 | 0.325 | 0.50 | 79933.8 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1191.0 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5196.5 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 163.6 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5543.5 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.340 | 264.562 | 7.23 | 1152854.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.406 | 0.205 | 6.41 | 1021590.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.443 | 0.009 | 0.78 | 124256.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.478 | 264.330 | 0.04 | 6945.4 | 8.0 | 2.00 |
| record/exchange | gpu | 110.594 | 110.718 | 3.26 | 519682.4 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.745 | 0.287 | 1.79 | 285439.7 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 379.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.147 | 0.203 | 1.74 | 277932.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 291.4 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.299 | 0.009 | 0.01 | 1406.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 5214.2 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 88.1 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 139.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.197 | 0.145 | 1.33 | 212380.0 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.564 | 0.018 | 0.13 | 21444.5 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.227 | 3.25 | 517957.5 | 0.0 | 2.00 |
| record/assemble | gpu | 0.387 | 0.017 | 0.01 | 1820.4 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 120.3 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.261 | 89.257 | 2.63 | 419439.1 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.132 | 0.00 | 621.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.452 | 0.033 | 0.01 | 2124.2 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.148 | 0.241 | 0.95 | 151062.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.212 | 0.175 | 0.74 | 118469.6 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.214 | 0.009 | 0.92 | 146676.8 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.120 | 0.00 | 562.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 175.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.452 | 0.026 | 0.43 | 67909.3 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 574.5 | 0.0 | 1.00 |
| sr/grad | gpu | 5.783 | 0.013 | 0.17 | 27172.4 | 2.0 | 1.00 |
| sr/cg | gpu | 1034.641 | 1043.305 | 30.50 | 4861779.9 | 475.3 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1036.141 | 30.55 | 4868826.3 | 0.0 | 286.57 |
| sr/cg/matvec | gpu | 1022.754 | 5.729 | 30.15 | 4805922.8 | 191.7 | 95.86 |
| sr/cg/precond | gpu | 2.033 | 0.424 | 0.06 | 9553.0 | 94.9 | 94.86 |
| sr/trust | gpu | 11.183 | 11.183 | 0.33 | 52551.2 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.992 | 0.104 | 0.32 | 51652.7 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.072 | 0.33 | 52025.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 227.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 37.6 | 0.0 | 1.00 |

### prof 2026-09-22 17:32:54 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4799, mean 3393.67 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 271.6 | 0.0 | 1.00 |
| net_fwd | gpu | 8.741 | 0.114 | 0.26 | 41946.4 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 599.3 | 1.0 | 1.00 |
| lu | gpu | 0.641 | 0.008 | 0.02 | 3078.0 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 87.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 618.863 | 628.263 | 18.24 | 2969923.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 478.632 | 0.122 | 14.10 | 2296953.4 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.325 | 0.414 | 2.69 | 438270.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 443.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.944 | 0.293 | 2.62 | 426840.0 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 471.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.396 | 0.013 | 0.01 | 1899.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.724 | 0.021 | 0.05 | 8273.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 147.2 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.800 | 0.083 | 1.44 | 234193.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1230.465 | 1230.479 | 36.26 | 5905003.9 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 950.451 | 0.240 | 28.01 | 4561214.7 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.413 | 0.834 | 5.38 | 875401.0 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 897.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.562 | 0.592 | 5.23 | 852121.7 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 905.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.764 | 0.025 | 0.02 | 3664.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.571 | 0.043 | 0.11 | 17138.9 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 293.9 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.376 | 0.164 | 2.87 | 467306.0 | 72.0 | 6.00 |
| record | gpu | 465.119 | 465.126 | 13.71 | 2232106.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.630 | 0.417 | 0.58 | 94203.9 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 17.001 | 0.325 | 0.50 | 81588.0 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1215.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5307.3 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 167.1 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5661.7 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.195 | 264.407 | 7.23 | 1176690.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.280 | 0.205 | 6.40 | 1042727.4 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.428 | 0.009 | 0.78 | 126830.3 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.473 | 264.175 | 0.04 | 7069.5 | 8.0 | 2.00 |
| record/exchange | gpu | 110.533 | 110.655 | 3.26 | 530448.4 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.712 | 0.287 | 1.79 | 291358.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 387.4 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.116 | 0.203 | 1.74 | 283697.0 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 297.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.298 | 0.009 | 0.01 | 1431.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 5325.0 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 89.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 142.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.171 | 0.144 | 1.33 | 216774.6 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.561 | 0.018 | 0.13 | 21888.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.164 | 3.25 | 528675.6 | 0.0 | 2.00 |
| record/assemble | gpu | 0.385 | 0.017 | 0.01 | 1847.6 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 122.8 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.204 | 89.200 | 2.63 | 428088.7 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.130 | 0.00 | 622.9 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.449 | 0.033 | 0.01 | 2156.7 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.131 | 0.241 | 0.95 | 154198.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.197 | 0.175 | 0.74 | 120922.2 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.193 | 0.009 | 0.92 | 149694.1 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.119 | 0.00 | 569.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 179.0 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.443 | 0.026 | 0.43 | 69311.6 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 586.8 | 0.0 | 1.00 |
| sr/grad | gpu | 5.779 | 0.013 | 0.17 | 27732.7 | 2.0 | 1.00 |
| sr/cg | gpu | 1037.749 | 1046.407 | 30.58 | 4980157.1 | 476.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1039.257 | 30.62 | 4987396.7 | 0.0 | 287.53 |
| sr/cg/matvec | gpu | 1025.930 | 5.710 | 30.23 | 4923439.5 | 192.4 | 96.18 |
| sr/cg/precond | gpu | 2.005 | 0.425 | 0.06 | 9619.9 | 95.2 | 95.18 |
| sr/trust | gpu | 11.172 | 11.172 | 0.33 | 53613.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.983 | 0.102 | 0.32 | 52707.9 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.061 | 0.33 | 53082.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 231.9 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 38.4 | 0.0 | 1.00 |

### prof 2026-09-22 17:38:42 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4899, mean 3395.38 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.057 | 0.00 | 277.1 | 0.0 | 1.00 |
| net_fwd | gpu | 8.736 | 0.114 | 0.26 | 42798.3 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 611.7 | 1.0 | 1.00 |
| lu | gpu | 0.640 | 0.008 | 0.02 | 3137.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 89.3 | 1.0 | 1.00 |
| therm_sweeps | gpu | 618.539 | 627.933 | 18.22 | 3030220.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 478.384 | 0.122 | 14.09 | 2343601.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.277 | 0.414 | 2.69 | 447166.9 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 452.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.897 | 0.293 | 2.62 | 435507.0 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 480.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.395 | 0.013 | 0.01 | 1936.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.723 | 0.021 | 0.05 | 8440.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 150.3 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.772 | 0.083 | 1.44 | 238935.2 | 36.0 | 3.00 |
| record_sweeps | gpu | 1229.796 | 1229.809 | 36.22 | 6024769.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 949.939 | 0.240 | 27.98 | 4653752.7 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.313 | 0.834 | 5.37 | 893150.3 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 915.5 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.467 | 0.592 | 5.23 | 869412.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.189 | 0.026 | 0.01 | 923.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.763 | 0.025 | 0.02 | 3739.2 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.566 | 0.043 | 0.11 | 17471.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 300.1 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.321 | 0.164 | 2.87 | 476774.3 | 72.0 | 6.00 |
| record | gpu | 464.852 | 464.859 | 13.69 | 2277309.0 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.621 | 0.416 | 0.58 | 96121.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.992 | 0.325 | 0.50 | 83242.4 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1240.4 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5418.1 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 170.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5779.9 | 2.0 | 2.00 |
| record/jet_pass | gpu | 245.056 | 264.259 | 7.22 | 1200527.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.160 | 0.205 | 6.40 | 1063865.7 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.414 | 0.009 | 0.78 | 129403.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.468 | 264.027 | 0.04 | 7193.6 | 8.0 | 2.00 |
| record/exchange | gpu | 110.474 | 110.593 | 3.25 | 541214.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.681 | 0.287 | 1.79 | 297277.4 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 395.5 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.086 | 0.203 | 1.74 | 289461.2 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 303.8 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.297 | 0.009 | 0.01 | 1456.9 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 5435.8 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 91.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 145.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.146 | 0.144 | 1.33 | 221169.0 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.558 | 0.018 | 0.13 | 22332.0 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.103 | 3.24 | 539393.2 | 0.0 | 2.00 |
| record/assemble | gpu | 0.383 | 0.017 | 0.01 | 1874.9 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 125.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.148 | 89.144 | 2.63 | 436738.4 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.127 | 0.00 | 624.5 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.447 | 0.033 | 0.01 | 2189.2 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.116 | 0.241 | 0.95 | 157333.9 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.184 | 0.175 | 0.74 | 123374.8 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.172 | 0.009 | 0.92 | 152711.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.117 | 0.00 | 575.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 182.5 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.434 | 0.026 | 0.43 | 70714.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 599.1 | 0.0 | 1.00 |
| sr/grad | gpu | 5.775 | 0.013 | 0.17 | 28293.7 | 2.0 | 1.00 |
| sr/cg | gpu | 1040.751 | 1049.404 | 30.65 | 5098638.8 | 478.4 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1042.268 | 30.70 | 5106070.0 | 0.0 | 288.46 |
| sr/cg/matvec | gpu | 1028.998 | 5.692 | 30.31 | 5041060.8 | 193.0 | 96.49 |
| sr/cg/precond | gpu | 1.977 | 0.427 | 0.06 | 9686.7 | 95.5 | 95.49 |
| sr/trust | gpu | 11.161 | 11.161 | 0.33 | 54676.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.974 | 0.101 | 0.32 | 53764.0 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.051 | 0.33 | 54140.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 236.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 39.1 | 0.0 | 1.00 |

### prof 2026-09-22 17:44:30 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 4999, mean 3397.08 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 282.1 | 0.0 | 1.00 |
| net_fwd | gpu | 8.732 | 0.114 | 0.26 | 43649.7 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 624.1 | 1.0 | 1.00 |
| lu | gpu | 0.639 | 0.008 | 0.02 | 3196.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 91.2 | 1.0 | 1.00 |
| therm_sweeps | gpu | 618.227 | 627.616 | 18.20 | 3090517.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 478.145 | 0.122 | 14.08 | 2390249.3 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.231 | 0.414 | 2.69 | 456064.1 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 462.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.853 | 0.293 | 2.62 | 444174.3 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 490.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.395 | 0.013 | 0.01 | 1974.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.722 | 0.021 | 0.05 | 8607.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 153.3 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.745 | 0.083 | 1.43 | 243677.3 | 36.0 | 3.00 |
| record_sweeps | gpu | 1229.153 | 1229.166 | 36.18 | 6144538.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 949.448 | 0.240 | 27.95 | 4746291.0 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.216 | 0.834 | 5.36 | 910900.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 933.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.376 | 0.592 | 5.22 | 886704.1 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 942.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.763 | 0.025 | 0.02 | 3814.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.561 | 0.043 | 0.10 | 17803.7 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 306.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.268 | 0.164 | 2.86 | 486244.1 | 72.0 | 6.00 |
| record | gpu | 464.595 | 464.602 | 13.68 | 2322509.2 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.612 | 0.416 | 0.58 | 98038.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.983 | 0.324 | 0.50 | 84896.7 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1265.1 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5528.8 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 174.0 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 5898.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.922 | 264.116 | 7.21 | 1224363.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 217.044 | 0.205 | 6.39 | 1085003.0 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.401 | 0.009 | 0.78 | 131976.8 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.464 | 263.884 | 0.04 | 7317.6 | 8.0 | 2.00 |
| record/exchange | gpu | 110.418 | 110.535 | 3.25 | 551979.5 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.651 | 0.287 | 1.79 | 303195.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 403.6 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.057 | 0.203 | 1.74 | 295224.9 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 310.0 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.296 | 0.009 | 0.01 | 1481.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 5546.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 93.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 148.5 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.122 | 0.144 | 1.33 | 225563.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.556 | 0.018 | 0.13 | 22775.7 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 110.044 | 3.24 | 550111.0 | 0.0 | 2.00 |
| record/assemble | gpu | 0.380 | 0.017 | 0.01 | 1902.1 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 127.9 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.095 | 89.091 | 2.62 | 445387.2 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.125 | 0.00 | 626.1 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.444 | 0.033 | 0.01 | 2221.6 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.100 | 0.241 | 0.94 | 160469.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.171 | 0.175 | 0.74 | 125827.4 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.152 | 0.009 | 0.92 | 155728.5 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.116 | 0.00 | 582.0 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 185.9 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.426 | 0.026 | 0.42 | 72115.7 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 611.1 | 0.0 | 1.00 |
| sr/grad | gpu | 5.772 | 0.013 | 0.17 | 28854.3 | 2.0 | 1.00 |
| sr/cg | gpu | 1043.685 | 1052.333 | 30.72 | 5217383.4 | 479.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1045.211 | 30.77 | 5225010.2 | 0.0 | 289.37 |
| sr/cg/matvec | gpu | 1031.996 | 5.674 | 30.38 | 5158946.6 | 193.6 | 96.79 |
| sr/cg/precond | gpu | 1.951 | 0.428 | 0.06 | 9753.0 | 95.8 | 95.79 |
| sr/trust | gpu | 11.150 | 11.150 | 0.33 | 55739.9 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.966 | 0.100 | 0.32 | 54820.6 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.042 | 0.33 | 55199.3 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 241.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 39.8 | 0.0 | 1.00 |

### prof 2026-09-22 17:50:17 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5099, mean 3398.59 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 286.8 | 0.0 | 1.00 |
| net_fwd | gpu | 8.727 | 0.113 | 0.26 | 44500.6 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 636.4 | 1.0 | 1.00 |
| lu | gpu | 0.638 | 0.008 | 0.02 | 3255.5 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 93.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 617.928 | 627.311 | 18.18 | 3150812.4 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 477.916 | 0.122 | 14.06 | 2436895.2 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.187 | 0.414 | 2.68 | 464961.3 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 471.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.810 | 0.293 | 2.61 | 452842.0 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 499.5 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.395 | 0.013 | 0.01 | 2011.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.721 | 0.021 | 0.05 | 8773.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 156.4 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.719 | 0.083 | 1.43 | 248418.4 | 36.0 | 3.00 |
| record_sweeps | gpu | 1228.536 | 1228.548 | 36.15 | 6264303.5 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 948.976 | 0.240 | 27.92 | 4838827.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.124 | 0.834 | 5.36 | 928650.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 952.2 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.289 | 0.592 | 5.22 | 903995.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 960.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.763 | 0.025 | 0.02 | 3889.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.557 | 0.043 | 0.10 | 18136.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 312.4 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.218 | 0.164 | 2.86 | 495712.9 | 72.0 | 6.00 |
| record | gpu | 464.348 | 464.355 | 13.66 | 2367710.3 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.603 | 0.416 | 0.58 | 99954.8 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.974 | 0.324 | 0.50 | 86550.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1289.8 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5639.6 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 177.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6016.2 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.793 | 263.979 | 7.20 | 1248199.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.933 | 0.205 | 6.38 | 1106140.8 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.388 | 0.009 | 0.78 | 134549.9 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.459 | 263.748 | 0.04 | 7441.5 | 8.0 | 2.00 |
| record/exchange | gpu | 110.364 | 110.478 | 3.25 | 562744.8 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.623 | 0.286 | 1.78 | 309114.4 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 411.7 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.029 | 0.203 | 1.74 | 300988.8 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 316.2 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.296 | 0.009 | 0.01 | 1506.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.110 | 0.015 | 0.03 | 5657.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 95.5 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 151.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.099 | 0.144 | 1.33 | 229957.7 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.554 | 0.018 | 0.13 | 23219.4 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.988 | 3.24 | 560828.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.378 | 0.017 | 0.01 | 1929.3 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 130.5 | 4.0 | 2.00 |
| record/o_assemble | gpu | 89.044 | 89.040 | 2.62 | 454037.0 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.123 | 0.00 | 627.7 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.442 | 0.033 | 0.01 | 2254.1 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.086 | 0.241 | 0.94 | 163605.5 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.158 | 0.175 | 0.74 | 128280.0 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.133 | 0.009 | 0.92 | 158745.5 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.115 | 0.00 | 588.4 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 189.2 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.418 | 0.026 | 0.42 | 73517.8 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 623.1 | 0.0 | 1.00 |
| sr/grad | gpu | 5.769 | 0.013 | 0.17 | 29415.0 | 2.0 | 1.00 |
| sr/cg | gpu | 1046.396 | 1055.039 | 30.79 | 5335572.5 | 481.3 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1047.931 | 30.83 | 5343399.2 | 0.0 | 290.21 |
| sr/cg/matvec | gpu | 1034.768 | 5.656 | 30.45 | 5276281.6 | 194.1 | 97.07 |
| sr/cg/precond | gpu | 1.926 | 0.429 | 0.06 | 9818.9 | 96.1 | 96.07 |
| sr/trust | gpu | 11.140 | 11.140 | 0.33 | 56802.4 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.958 | 0.099 | 0.32 | 55876.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.033 | 0.32 | 56256.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 246.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 40.6 | 0.0 | 1.00 |

### prof 2026-09-22 17:56:06 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5199, mean 3400.22 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 291.5 | 0.0 | 1.00 |
| net_fwd | gpu | 8.723 | 0.113 | 0.26 | 45351.6 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 648.8 | 1.0 | 1.00 |
| lu | gpu | 0.638 | 0.008 | 0.02 | 3314.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 94.8 | 1.0 | 1.00 |
| therm_sweeps | gpu | 617.639 | 627.018 | 18.16 | 3211105.4 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 477.696 | 0.122 | 14.05 | 2483539.2 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.144 | 0.414 | 2.68 | 473858.1 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 480.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.769 | 0.293 | 2.61 | 461509.2 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 508.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.394 | 0.013 | 0.01 | 2049.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.720 | 0.021 | 0.05 | 8939.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 159.5 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.694 | 0.083 | 1.43 | 253160.6 | 36.0 | 3.00 |
| record_sweeps | gpu | 1227.942 | 1227.954 | 36.11 | 6384068.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 948.522 | 0.240 | 27.90 | 4931364.8 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 182.035 | 0.833 | 5.35 | 946400.6 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 970.6 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.205 | 0.591 | 5.21 | 921287.7 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 979.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.762 | 0.025 | 0.02 | 3964.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.552 | 0.043 | 0.10 | 18468.5 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 318.5 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.169 | 0.164 | 2.86 | 505180.3 | 72.0 | 6.00 |
| record | gpu | 464.111 | 464.117 | 13.65 | 2412910.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.594 | 0.415 | 0.58 | 101871.7 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.966 | 0.324 | 0.50 | 88205.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1314.5 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5750.4 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 181.0 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6134.4 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.669 | 263.847 | 7.20 | 1272035.5 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.826 | 0.205 | 6.38 | 1127278.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.375 | 0.009 | 0.78 | 137123.0 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.455 | 263.616 | 0.04 | 7565.6 | 8.0 | 2.00 |
| record/exchange | gpu | 110.312 | 110.424 | 3.24 | 573510.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.595 | 0.286 | 1.78 | 315033.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 419.7 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 59.002 | 0.203 | 1.74 | 306753.0 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 322.4 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.295 | 0.009 | 0.01 | 1531.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 5768.2 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 97.4 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 154.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.076 | 0.144 | 1.33 | 234352.0 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.551 | 0.018 | 0.13 | 23663.1 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.934 | 3.23 | 571546.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.376 | 0.017 | 0.01 | 1956.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 133.0 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.995 | 88.991 | 2.62 | 462686.3 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.121 | 0.00 | 629.3 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.440 | 0.033 | 0.01 | 2286.5 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.072 | 0.241 | 0.94 | 166741.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.146 | 0.175 | 0.74 | 130732.6 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.114 | 0.009 | 0.92 | 161762.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.114 | 0.00 | 594.8 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 192.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.410 | 0.026 | 0.42 | 74919.5 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 635.2 | 0.0 | 1.00 |
| sr/grad | gpu | 5.766 | 0.013 | 0.17 | 29975.0 | 2.0 | 1.00 |
| sr/cg | gpu | 1049.168 | 1057.807 | 30.86 | 5454624.7 | 482.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1050.711 | 30.90 | 5462646.0 | 0.0 | 291.06 |
| sr/cg/matvec | gpu | 1037.598 | 5.640 | 30.52 | 5394473.1 | 194.7 | 97.35 |
| sr/cg/precond | gpu | 1.901 | 0.430 | 0.06 | 9885.4 | 96.4 | 96.35 |
| sr/trust | gpu | 11.130 | 11.130 | 0.33 | 57864.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.950 | 0.098 | 0.32 | 56931.2 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.024 | 0.32 | 57313.5 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 251.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 41.3 | 0.0 | 1.00 |

### prof 2026-09-22 18:01:53 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5299, mean 3401.69 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 297.4 | 0.0 | 1.00 |
| net_fwd | gpu | 8.719 | 0.113 | 0.26 | 46203.3 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 661.2 | 1.0 | 1.00 |
| lu | gpu | 0.637 | 0.008 | 0.02 | 3373.8 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 96.6 | 1.0 | 1.00 |
| therm_sweeps | gpu | 617.363 | 626.737 | 18.15 | 3271404.9 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 477.484 | 0.122 | 14.04 | 2530189.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.103 | 0.414 | 2.68 | 482755.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 489.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.729 | 0.293 | 2.61 | 470177.0 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 518.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.394 | 0.013 | 0.01 | 2086.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.719 | 0.021 | 0.05 | 9106.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 162.6 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.670 | 0.083 | 1.43 | 257901.7 | 36.0 | 3.00 |
| record_sweeps | gpu | 1227.371 | 1227.384 | 36.08 | 6503838.9 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 948.085 | 0.240 | 27.87 | 5023904.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 181.950 | 0.833 | 5.35 | 964150.9 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 989.0 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.124 | 0.591 | 5.21 | 938579.5 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 998.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.762 | 0.025 | 0.02 | 4039.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.548 | 0.043 | 0.10 | 18800.9 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 324.7 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.122 | 0.164 | 2.86 | 514650.4 | 72.0 | 6.00 |
| record | gpu | 463.882 | 463.889 | 13.64 | 2458113.0 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.586 | 0.415 | 0.58 | 103788.7 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.958 | 0.324 | 0.50 | 89859.6 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1339.2 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5861.1 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 184.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6252.6 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.550 | 263.721 | 7.19 | 1295872.1 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.723 | 0.205 | 6.37 | 1148415.9 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.363 | 0.009 | 0.77 | 139696.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.451 | 263.489 | 0.04 | 7689.7 | 8.0 | 2.00 |
| record/exchange | gpu | 110.261 | 110.372 | 3.24 | 584275.4 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.568 | 0.286 | 1.78 | 320951.5 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 427.8 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.977 | 0.203 | 1.73 | 312516.7 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 328.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.294 | 0.009 | 0.01 | 1556.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 5879.0 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 99.3 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 157.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.055 | 0.144 | 1.32 | 238746.2 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.549 | 0.018 | 0.13 | 24106.8 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.882 | 3.23 | 582263.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.374 | 0.017 | 0.01 | 1983.8 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 135.6 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.948 | 88.944 | 2.61 | 471336.5 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.119 | 0.00 | 630.9 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.438 | 0.033 | 0.01 | 2319.0 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.058 | 0.241 | 0.94 | 169876.6 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.134 | 0.175 | 0.74 | 133185.2 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.097 | 0.009 | 0.91 | 164780.6 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.113 | 0.00 | 601.2 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 196.3 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.403 | 0.026 | 0.42 | 76321.1 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 647.5 | 0.0 | 1.00 |
| sr/grad | gpu | 5.763 | 0.013 | 0.17 | 30535.9 | 2.0 | 1.00 |
| sr/cg | gpu | 1051.744 | 1060.378 | 30.92 | 5573189.1 | 484.1 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1053.294 | 30.96 | 5581403.0 | 0.0 | 291.86 |
| sr/cg/matvec | gpu | 1040.230 | 5.625 | 30.58 | 5512177.2 | 195.2 | 97.62 |
| sr/cg/precond | gpu | 1.878 | 0.431 | 0.06 | 9952.2 | 96.6 | 96.62 |
| sr/trust | gpu | 11.120 | 11.120 | 0.33 | 58926.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.943 | 0.097 | 0.32 | 57986.4 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.015 | 0.32 | 58370.5 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 255.8 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 42.0 | 0.0 | 1.00 |

### prof 2026-09-22 18:07:42 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5399, mean 3403.19 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 302.3 | 0.0 | 1.00 |
| net_fwd | gpu | 8.716 | 0.113 | 0.26 | 47057.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 673.5 | 1.0 | 1.00 |
| lu | gpu | 0.636 | 0.008 | 0.02 | 3433.1 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 98.5 | 1.0 | 1.00 |
| therm_sweeps | gpu | 617.099 | 626.469 | 18.13 | 3331716.2 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 477.283 | 0.122 | 14.02 | 2576851.5 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.064 | 0.414 | 2.68 | 491652.9 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 498.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.691 | 0.293 | 2.61 | 478844.8 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 527.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.393 | 0.013 | 0.01 | 2124.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.718 | 0.021 | 0.05 | 9272.9 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 165.7 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.647 | 0.083 | 1.43 | 262643.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1226.822 | 1226.834 | 36.05 | 6623611.4 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 947.665 | 0.240 | 27.85 | 5116445.1 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 181.867 | 0.833 | 5.34 | 981902.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 1007.3 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 177.046 | 0.591 | 5.20 | 955872.3 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 1016.7 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.762 | 0.025 | 0.02 | 4114.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.544 | 0.043 | 0.10 | 19133.3 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 330.8 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.077 | 0.164 | 2.85 | 524120.5 | 72.0 | 6.00 |
| record | gpu | 463.663 | 463.670 | 13.62 | 2503316.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.579 | 0.415 | 0.58 | 105705.8 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.950 | 0.323 | 0.50 | 91513.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1363.9 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 5971.9 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 188.0 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6370.8 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.436 | 263.599 | 7.18 | 1319710.0 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.624 | 0.205 | 6.37 | 1169555.3 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.351 | 0.009 | 0.77 | 142269.6 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.447 | 263.367 | 0.04 | 7813.8 | 8.0 | 2.00 |
| record/exchange | gpu | 110.213 | 110.322 | 3.24 | 595041.2 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.543 | 0.286 | 1.78 | 326870.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 435.9 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.952 | 0.203 | 1.73 | 318281.1 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 334.8 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.293 | 0.009 | 0.01 | 1581.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 5989.8 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 101.2 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 160.4 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.034 | 0.144 | 1.32 | 243140.4 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.547 | 0.018 | 0.13 | 24550.6 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.832 | 3.23 | 592981.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.372 | 0.017 | 0.01 | 2011.0 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 138.2 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.903 | 88.899 | 2.61 | 479986.6 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.117 | 0.00 | 632.6 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.436 | 0.033 | 0.01 | 2351.5 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.045 | 0.241 | 0.94 | 173012.1 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.123 | 0.175 | 0.74 | 135637.9 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.080 | 0.009 | 0.91 | 167798.7 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.113 | 0.00 | 607.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 199.7 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.396 | 0.026 | 0.42 | 77723.7 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 659.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.760 | 0.013 | 0.17 | 31097.3 | 2.0 | 1.00 |
| sr/cg | gpu | 1054.302 | 1062.932 | 30.98 | 5692178.3 | 485.4 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1055.859 | 31.03 | 5700585.4 | 0.0 | 292.64 |
| sr/cg/matvec | gpu | 1042.842 | 5.610 | 30.64 | 5630303.5 | 195.8 | 97.88 |
| sr/cg/precond | gpu | 1.856 | 0.432 | 0.05 | 10019.1 | 96.9 | 96.88 |
| sr/trust | gpu | 11.111 | 11.111 | 0.33 | 59988.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.936 | 0.096 | 0.32 | 59041.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 11.007 | 0.32 | 59427.4 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 260.5 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 42.8 | 0.0 | 1.00 |

### prof 2026-09-22 18:13:29 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5499, mean 3404.53 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 307.4 | 0.0 | 1.00 |
| net_fwd | gpu | 8.713 | 0.113 | 0.26 | 47910.9 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 685.9 | 1.0 | 1.00 |
| lu | gpu | 0.635 | 0.008 | 0.02 | 3492.4 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 100.3 | 1.0 | 1.00 |
| therm_sweeps | gpu | 616.844 | 626.210 | 18.12 | 3392024.6 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 477.088 | 0.122 | 14.01 | 2623509.3 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 91.026 | 0.414 | 2.67 | 500550.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 508.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.655 | 0.293 | 2.60 | 487512.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 536.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.393 | 0.013 | 0.01 | 2161.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.717 | 0.021 | 0.05 | 9439.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 168.7 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.625 | 0.083 | 1.43 | 267386.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1226.293 | 1226.305 | 36.02 | 6743384.8 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 947.261 | 0.240 | 27.82 | 5208987.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 181.788 | 0.833 | 5.34 | 999653.8 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.187 | 0.028 | 0.01 | 1025.7 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 176.971 | 0.591 | 5.20 | 973165.6 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 1035.3 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.762 | 0.025 | 0.02 | 4189.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.540 | 0.043 | 0.10 | 19465.7 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 336.9 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 97.034 | 0.164 | 2.85 | 533589.7 | 72.0 | 6.00 |
| record | gpu | 463.452 | 463.458 | 13.61 | 2548520.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.571 | 0.414 | 0.57 | 107622.7 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.943 | 0.323 | 0.50 | 93168.2 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.253 | 0.009 | 0.01 | 1388.6 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 6082.7 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 191.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6489.0 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.326 | 263.481 | 7.18 | 1343548.3 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.529 | 0.205 | 6.36 | 1190695.0 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.340 | 0.009 | 0.77 | 144842.7 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.444 | 263.250 | 0.04 | 7938.0 | 8.0 | 2.00 |
| record/exchange | gpu | 110.167 | 110.273 | 3.24 | 605806.6 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.518 | 0.286 | 1.78 | 332789.6 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 443.9 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.928 | 0.203 | 1.73 | 324045.5 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 341.0 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.292 | 0.009 | 0.01 | 1606.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 6100.6 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 103.0 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 163.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 45.014 | 0.144 | 1.32 | 247534.3 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.545 | 0.018 | 0.13 | 24994.3 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.783 | 3.22 | 603699.4 | 0.0 | 2.00 |
| record/assemble | gpu | 0.371 | 0.017 | 0.01 | 2038.2 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 140.7 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.859 | 88.856 | 2.61 | 488636.3 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.115 | 0.00 | 634.2 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.434 | 0.033 | 0.01 | 2383.9 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.033 | 0.241 | 0.94 | 176147.6 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.112 | 0.175 | 0.74 | 138090.5 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.063 | 0.009 | 0.91 | 170816.2 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.112 | 0.00 | 614.1 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 203.1 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.389 | 0.026 | 0.42 | 79126.1 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 671.4 | 0.0 | 1.00 |
| sr/grad | gpu | 5.757 | 0.013 | 0.17 | 31658.6 | 2.0 | 1.00 |
| sr/cg | gpu | 1056.660 | 1065.286 | 31.04 | 5810572.7 | 486.6 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1058.224 | 31.08 | 5819174.1 | 0.0 | 293.37 |
| sr/cg/matvec | gpu | 1045.252 | 5.595 | 30.70 | 5747838.4 | 196.2 | 98.12 |
| sr/cg/precond | gpu | 1.834 | 0.433 | 0.05 | 10086.1 | 97.1 | 97.12 |
| sr/trust | gpu | 11.102 | 11.102 | 0.33 | 61050.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.929 | 0.095 | 0.32 | 60096.5 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 10.999 | 0.32 | 60484.1 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 265.3 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 43.5 | 0.0 | 1.00 |

### prof 2026-09-22 18:19:17 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5599, mean 3405.83 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 313.4 | 0.0 | 1.00 |
| net_fwd | gpu | 8.709 | 0.113 | 0.26 | 48763.2 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 698.3 | 1.0 | 1.00 |
| lu | gpu | 0.634 | 0.008 | 0.02 | 3551.6 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 102.1 | 1.0 | 1.00 |
| therm_sweeps | gpu | 616.597 | 625.959 | 18.10 | 3452327.2 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 476.900 | 0.122 | 14.00 | 2670160.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 90.989 | 0.414 | 2.67 | 509448.9 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 517.2 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.620 | 0.293 | 2.60 | 496181.6 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.098 | 0.013 | 0.00 | 546.0 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.393 | 0.013 | 0.01 | 2199.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.716 | 0.021 | 0.05 | 9605.8 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 171.8 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.603 | 0.083 | 1.43 | 272128.8 | 36.0 | 3.00 |
| record_sweeps | gpu | 1225.783 | 1225.795 | 35.99 | 6863156.6 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 946.870 | 0.240 | 27.80 | 5301527.5 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 181.712 | 0.833 | 5.34 | 1017404.9 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.186 | 0.028 | 0.01 | 1044.0 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 176.899 | 0.591 | 5.19 | 990458.2 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 1053.9 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.762 | 0.025 | 0.02 | 4264.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.536 | 0.043 | 0.10 | 19798.1 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 343.0 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 96.992 | 0.164 | 2.85 | 543060.0 | 72.0 | 6.00 |
| record | gpu | 463.248 | 463.254 | 13.60 | 2593724.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.564 | 0.414 | 0.57 | 109540.0 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.936 | 0.323 | 0.50 | 94822.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.252 | 0.009 | 0.01 | 1413.3 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 6193.5 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 195.0 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6607.1 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.220 | 263.368 | 7.17 | 1367387.9 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.438 | 0.205 | 6.35 | 1211835.8 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.329 | 0.009 | 0.77 | 147416.0 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.440 | 263.137 | 0.04 | 8062.2 | 8.0 | 2.00 |
| record/exchange | gpu | 110.122 | 110.227 | 3.23 | 616572.1 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.495 | 0.286 | 1.78 | 338708.8 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 452.0 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.905 | 0.203 | 1.73 | 329810.0 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 347.2 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.291 | 0.009 | 0.01 | 1631.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 6211.4 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 104.9 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 166.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 44.995 | 0.144 | 1.32 | 251927.9 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.543 | 0.018 | 0.13 | 25438.1 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.737 | 3.22 | 614416.8 | 0.0 | 2.00 |
| record/assemble | gpu | 0.369 | 0.017 | 0.01 | 2065.5 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 143.3 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.817 | 88.813 | 2.61 | 497285.2 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.114 | 0.00 | 635.8 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.432 | 0.033 | 0.01 | 2416.4 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.021 | 0.241 | 0.94 | 179283.0 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.101 | 0.175 | 0.74 | 140543.2 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.047 | 0.009 | 0.91 | 173832.9 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.111 | 0.00 | 620.6 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 206.9 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.383 | 0.026 | 0.42 | 80528.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 683.7 | 0.0 | 1.00 |
| sr/grad | gpu | 5.754 | 0.013 | 0.17 | 32218.8 | 2.0 | 1.00 |
| sr/cg | gpu | 1058.948 | 1067.570 | 31.09 | 5929048.6 | 487.8 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1060.519 | 31.14 | 5937845.1 | 0.0 | 294.08 |
| sr/cg/matvec | gpu | 1047.590 | 5.581 | 30.76 | 5865456.0 | 196.7 | 98.36 |
| sr/cg/precond | gpu | 1.813 | 0.434 | 0.05 | 10152.6 | 97.4 | 97.36 |
| sr/trust | gpu | 11.094 | 11.093 | 0.33 | 62113.0 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.922 | 0.094 | 0.32 | 61152.3 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 10.992 | 0.32 | 61541.8 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 270.2 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 44.3 | 0.0 | 1.00 |

### prof 2026-09-22 18:25:05 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5699, mean 3407.03 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 318.5 | 0.0 | 1.00 |
| net_fwd | gpu | 8.706 | 0.113 | 0.26 | 49616.8 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 710.6 | 1.0 | 1.00 |
| lu | gpu | 0.634 | 0.008 | 0.02 | 3610.9 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 104.0 | 1.0 | 1.00 |
| therm_sweeps | gpu | 616.361 | 625.719 | 18.09 | 3512638.7 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 476.719 | 0.122 | 13.99 | 2716822.8 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 90.954 | 0.414 | 2.67 | 518346.5 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 526.4 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.586 | 0.293 | 2.60 | 504849.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.097 | 0.013 | 0.00 | 555.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.392 | 0.013 | 0.01 | 2236.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.715 | 0.021 | 0.05 | 9772.3 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 174.9 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.582 | 0.083 | 1.43 | 276870.5 | 36.0 | 3.00 |
| record_sweeps | gpu | 1225.290 | 1225.303 | 35.96 | 6982929.3 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 946.494 | 0.239 | 27.78 | 5394068.3 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 181.638 | 0.833 | 5.33 | 1035156.2 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.186 | 0.028 | 0.01 | 1062.4 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 176.829 | 0.591 | 5.19 | 1007751.0 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 1072.5 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.761 | 0.025 | 0.02 | 4339.1 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.532 | 0.043 | 0.10 | 20130.6 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 349.2 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 96.952 | 0.163 | 2.85 | 552530.3 | 72.0 | 6.00 |
| record | gpu | 463.051 | 463.058 | 13.59 | 2638928.9 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.557 | 0.414 | 0.57 | 111457.3 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.929 | 0.323 | 0.50 | 96477.5 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.252 | 0.009 | 0.01 | 1438.0 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 6304.2 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 198.5 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6725.3 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.118 | 263.259 | 7.17 | 1391226.8 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.350 | 0.205 | 6.35 | 1232976.0 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.318 | 0.009 | 0.77 | 149989.1 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.436 | 263.028 | 0.04 | 8186.4 | 8.0 | 2.00 |
| record/exchange | gpu | 110.079 | 110.182 | 3.23 | 627337.8 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.472 | 0.286 | 1.77 | 344628.0 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 460.1 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.883 | 0.203 | 1.73 | 335574.6 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 353.4 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.291 | 0.009 | 0.01 | 1656.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 6322.2 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 106.8 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 169.3 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 44.977 | 0.144 | 1.32 | 256321.8 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.541 | 0.018 | 0.13 | 25881.8 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.692 | 3.22 | 625134.9 | 0.0 | 2.00 |
| record/assemble | gpu | 0.367 | 0.017 | 0.01 | 2092.8 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 145.9 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.776 | 88.772 | 2.61 | 505934.1 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.112 | 0.00 | 637.4 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.430 | 0.033 | 0.01 | 2448.9 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 32.009 | 0.241 | 0.94 | 182418.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.091 | 0.175 | 0.74 | 142996.2 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.032 | 0.009 | 0.91 | 176849.8 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.110 | 0.00 | 626.9 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 210.4 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.376 | 0.026 | 0.42 | 81931.1 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 695.9 | 0.0 | 1.00 |
| sr/grad | gpu | 5.752 | 0.013 | 0.17 | 32779.3 | 2.0 | 1.00 |
| sr/cg | gpu | 1061.098 | 1069.717 | 31.14 | 6047199.3 | 488.9 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1062.677 | 31.19 | 6056196.5 | 0.0 | 294.75 |
| sr/cg/matvec | gpu | 1049.790 | 5.567 | 30.81 | 5982752.1 | 197.2 | 98.58 |
| sr/cg/precond | gpu | 1.793 | 0.435 | 0.05 | 10218.5 | 97.6 | 97.58 |
| sr/trust | gpu | 11.085 | 11.085 | 0.33 | 63176.1 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.916 | 0.093 | 0.32 | 62208.6 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 10.984 | 0.32 | 62599.9 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 275.0 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 45.0 | 0.0 | 1.00 |

### prof 2026-09-22 18:30:53 | rev 930c75d | descent
card: NVIDIA GeForce RTX 3090, sm_86, 82 SM, 1.70 GHz, FP64 peak ~0.556 TF (est: 2 FP64/SM), FP64:FP32 = 1:64
config: B=5800 records=2 sweeps/iter=9 N=6 K=31 m_feat=61 P=47767 jet_chunk=0 real=fp64
iterations profiled: 5799, mean 3408.32 ms/iter
rows are INCLUSIVE (a parent contains its children). gpu_ms is cudaEvent time on the
profiled stream; host_ms is the wall time the host spent inside the range. host_ms much
larger than gpu_ms means the host is not keeping the device fed (launch latency, or a
blocking copy); host_ms much smaller means the range only enqueued work.

| range | kind | gpu_ms/iter | host_ms/iter | %iter | total_ms | launches/iter | calls/iter |
|---|---|---:|---:|---:|---:|---:|---:|
| transfers/params_up | host | 0.000 | 0.056 | 0.00 | 323.2 | 0.0 | 1.00 |
| net_fwd | gpu | 8.703 | 0.113 | 0.26 | 50470.5 | 14.0 | 1.00 |
| assemble | gpu | 0.125 | 0.005 | 0.00 | 723.0 | 1.0 | 1.00 |
| lu | gpu | 0.633 | 0.008 | 0.02 | 3670.2 | 1.0 | 1.00 |
| combine_envelope | gpu | 0.018 | 0.005 | 0.00 | 105.8 | 1.0 | 1.00 |
| therm_sweeps | gpu | 616.132 | 625.486 | 18.08 | 3572948.3 | 144.0 | 1.00 |
| therm_sweeps/coord_draws | gpu | 476.545 | 0.122 | 13.98 | 2763482.6 | 54.0 | 3.00 |
| therm_sweeps/st_table | gpu | 90.920 | 0.414 | 2.67 | 527244.2 | 51.0 | 3.00 |
| therm_sweeps/st_table/feat_combo | gpu | 0.092 | 0.013 | 0.00 | 535.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/net_fwd | gpu | 88.553 | 0.293 | 2.60 | 513517.7 | 36.0 | 6.00 |
| therm_sweeps/st_table/xi_combo | gpu | 0.097 | 0.013 | 0.00 | 564.6 | 3.0 | 3.00 |
| therm_sweeps/st_table/assemble | gpu | 0.392 | 0.013 | 0.01 | 2274.1 | 3.0 | 3.00 |
| therm_sweeps/st_table/lu | gpu | 1.714 | 0.021 | 0.05 | 9938.7 | 3.0 | 3.00 |
| therm_sweeps/st_table/det_combine | gpu | 0.031 | 0.013 | 0.00 | 177.9 | 3.0 | 3.00 |
| therm_sweeps/discrete_block | gpu | 48.562 | 0.083 | 1.42 | 281612.5 | 36.0 | 3.00 |
| record_sweeps | gpu | 1224.815 | 1224.827 | 35.94 | 7102701.0 | 288.0 | 2.00 |
| record_sweeps/coord_draws | gpu | 946.130 | 0.239 | 27.76 | 5486609.4 | 108.0 | 6.00 |
| record_sweeps/st_table | gpu | 181.567 | 0.833 | 5.33 | 1052907.1 | 102.0 | 6.00 |
| record_sweeps/st_table/feat_combo | gpu | 0.186 | 0.028 | 0.01 | 1080.8 | 6.0 | 6.00 |
| record_sweeps/st_table/net_fwd | gpu | 176.762 | 0.591 | 5.19 | 1025043.5 | 72.0 | 12.00 |
| record_sweeps/st_table/xi_combo | gpu | 0.188 | 0.026 | 0.01 | 1091.1 | 6.0 | 6.00 |
| record_sweeps/st_table/assemble | gpu | 0.761 | 0.025 | 0.02 | 4414.0 | 6.0 | 6.00 |
| record_sweeps/st_table/lu | gpu | 3.529 | 0.043 | 0.10 | 20463.0 | 6.0 | 6.00 |
| record_sweeps/st_table/det_combine | gpu | 0.061 | 0.025 | 0.00 | 355.3 | 6.0 | 6.00 |
| record_sweeps/discrete_block | gpu | 96.913 | 0.163 | 2.84 | 562000.1 | 72.0 | 6.00 |
| record | gpu | 462.861 | 462.867 | 13.58 | 2684130.4 | 236.0 | 2.00 |
| record/eval_cached | gpu | 19.551 | 0.414 | 0.57 | 113374.4 | 36.0 | 2.00 |
| record/eval_cached/net_fwd | gpu | 16.922 | 0.322 | 0.50 | 98131.9 | 28.0 | 2.00 |
| record/eval_cached/assemble | gpu | 0.252 | 0.009 | 0.01 | 1462.7 | 2.0 | 2.00 |
| record/eval_cached/lu | gpu | 1.106 | 0.016 | 0.03 | 6415.0 | 2.0 | 2.00 |
| record/eval_cached/combine_envelope | gpu | 0.035 | 0.009 | 0.00 | 202.0 | 2.0 | 2.00 |
| record/eval_cached/getri | gpu | 1.180 | 0.010 | 0.03 | 6843.5 | 2.0 | 2.00 |
| record/jet_pass | gpu | 244.019 | 263.154 | 7.16 | 1415064.6 | 38.0 | 2.00 |
| record/jet_pass/jet_net | gpu | 216.264 | 0.205 | 6.35 | 1254115.2 | 28.0 | 2.00 |
| record/jet_pass/detjet | gpu | 26.308 | 0.009 | 0.77 | 152562.5 | 2.0 | 2.00 |
| record/jet_pass/compose | gpu | 1.433 | 262.923 | 0.04 | 8310.4 | 8.0 | 2.00 |
| record/exchange | gpu | 110.037 | 110.138 | 3.23 | 638103.3 | 62.0 | 2.00 |
| record/exchange/st_table | gpu | 60.450 | 0.286 | 1.77 | 350547.1 | 34.0 | 2.00 |
| record/exchange/st_table/feat_combo | gpu | 0.081 | 0.009 | 0.00 | 468.2 | 2.0 | 2.00 |
| record/exchange/st_table/net_fwd | gpu | 58.862 | 0.203 | 1.73 | 341339.0 | 24.0 | 4.00 |
| record/exchange/st_table/xi_combo | gpu | 0.062 | 0.009 | 0.00 | 359.6 | 2.0 | 2.00 |
| record/exchange/st_table/assemble | gpu | 0.290 | 0.009 | 0.01 | 1681.8 | 2.0 | 2.00 |
| record/exchange/st_table/lu | gpu | 1.109 | 0.015 | 0.03 | 6432.9 | 2.0 | 2.00 |
| record/exchange/st_table/det_combine | gpu | 0.019 | 0.009 | 0.00 | 108.7 | 2.0 | 2.00 |
| record/exchange/gate_plan | gpu | 0.030 | 0.014 | 0.00 | 172.2 | 4.0 | 2.00 |
| record/exchange/rho_slots | gpu | 44.959 | 0.144 | 1.32 | 260715.8 | 20.0 | 4.00 |
| record/exchange/rank2 | gpu | 4.540 | 0.018 | 0.13 | 26325.5 | 4.0 | 4.00 |
| record/exchange/fallback | host | 0.000 | 109.649 | 3.22 | 635853.1 | 0.0 | 2.00 |
| record/assemble | gpu | 0.366 | 0.017 | 0.01 | 2120.0 | 4.0 | 2.00 |
| record/stats | gpu | 0.026 | 0.014 | 0.00 | 148.4 | 4.0 | 2.00 |
| record/o_assemble | gpu | 88.736 | 88.733 | 2.60 | 514582.3 | 92.0 | 2.00 |
| transfers/alpha_dn | host | 0.000 | 0.110 | 0.00 | 639.0 | 0.0 | 2.00 |
| record/o_assemble/seeds | gpu | 0.428 | 0.033 | 0.01 | 2481.3 | 4.0 | 6.00 |
| record/o_assemble/dW_gemms | gpu | 31.997 | 0.241 | 0.94 | 185553.2 | 48.0 | 24.00 |
| record/o_assemble/delta_prop | gpu | 25.082 | 0.175 | 0.74 | 145448.8 | 38.0 | 20.00 |
| record/o_assemble/o_finalize | gpu | 31.017 | 0.009 | 0.91 | 179866.4 | 2.0 | 2.00 |
| transfers/download_iter | host | 0.000 | 0.109 | 0.00 | 633.3 | 0.0 | 1.00 |
| host/reduce_iter | host | 0.000 | 0.037 | 0.00 | 213.6 | 0.0 | 1.00 |
| sr/o_stats | gpu | 14.370 | 0.026 | 0.42 | 83333.2 | 4.0 | 2.00 |
| host/clip_stats | host | 0.000 | 0.122 | 0.00 | 707.7 | 0.0 | 1.00 |
| sr/grad | gpu | 5.749 | 0.013 | 0.17 | 33339.6 | 2.0 | 1.00 |
| sr/cg | gpu | 1063.300 | 1071.915 | 31.20 | 6166079.4 | 490.0 | 1.00 |
| sr/cg/scalars_dn | host | 0.000 | 1064.886 | 31.24 | 6175273.3 | 0.0 | 295.43 |
| sr/cg/matvec | gpu | 1052.039 | 5.553 | 30.87 | 6100772.7 | 197.6 | 98.81 |
| sr/cg/precond | gpu | 1.774 | 0.436 | 0.05 | 10284.8 | 97.8 | 97.81 |
| sr/trust | gpu | 11.077 | 11.077 | 0.33 | 64238.3 | 2.0 | 1.00 |
| sr/trust/matvec | gpu | 10.909 | 0.092 | 0.32 | 63263.9 | 2.0 | 1.00 |
| sr/trust/scalars_dn | host | 0.000 | 10.977 | 0.32 | 63657.1 | 0.0 | 3.00 |
| transfers/delta_dn | host | 0.000 | 0.048 | 0.00 | 279.7 | 0.0 | 1.00 |
| transfers/grad_alpha_dn | host | 0.000 | 0.008 | 0.00 | 45.7 | 0.0 | 1.00 |
