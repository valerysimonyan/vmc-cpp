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

| system | steps | GPU frozen eval | CPU frozen eval (same checkpoint) | z | target |
|---|---|---|---|---|---|
| deuteron | 2000 | -2.21402 +/- 0.00157 | -2.21587 +/- 0.00159 | 0.83 | -2.2245 |
| deuteron | 5000 | -2.23036 +/- 0.00110 | -2.23087 +/- 0.00114 | 0.32 | -2.2245 |
| triton | 2000 | -8.35675 +/- 0.00425 | -8.36276 +/- 0.00403 | 1.03 | -8.482 |
| triton | 5000 | -8.3905 +/- 0.0035 | -- | -- | -8.482 |

- Deuteron at 5000 steps is 5-6 sigma BELOW -2.2245 in both builds (training
  iterates still falling, variance 0.58). VMC is variational, so either this
  model-o Hamiltonian binds the deuteron at about -2.231 MeV or the CPU local
  energy has an error shared by both builds. Open question; not a porting error.
- Triton at 5000 steps is 92 keV above -8.482 and still converging.
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

- Deuteron at 5000 steps sits 5-6 sigma below -2.2245 in both builds. Open.
- Triton is 92 keV above -8.482 and still converging.
- A full-length Li6 production run has still not been done.
