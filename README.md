Neural-network variational Monte Carlo for light nuclei (pionless EFT, model "o").
The wavefunction is envelope x Jastrow x sum of K Slater determinants of
DeepSets orbitals; training is stochastic reconfiguration (SR) on a CPU or on
one or more CUDA GPUs.

Contents: [Build and test](#build-and-test) · [Running](#running) ·
[Regression check](#regression-check-golden) · [Training monitor](#training-monitor) ·
[Run status](#run-status-text) · [Code map](#code-map)

## Build and test

Requires g++ with C++17 and CMake >= 3.18. Ubuntu 20.04 ships CMake 3.16, which
is too old for `CMAKE_CUDA_ARCHITECTURES`; `pip3 install --user cmake` gets a
current one.

    make            # CPU only, builds ./build/main
    make test       # CPU suite via ctest (8 tests, ~15 s)

Or CMake directly:

    cmake -B build && cmake --build build -j
    ctest --test-dir build --output-on-failure

The nucleus is fixed at compile time: `N`, `N_u` and `N_p` (particles, spin-up,
protons) at the top of `lib/constants.h`. Everything else is a run-time setting
(see [Running](#running)).

### CUDA (optional)

    make gpu        # or: cmake -B build-cuda -DVMC_CUDA=ON && cmake --build build-cuda -j
    make gpu-test   # CPU and GPU suites (16 tests, ~1 min)

The CPU configuration must build and pass with **no CUDA toolkit installed** --
that is the CI invariant, and it is what `cmake -B build` verifies. Nothing
outside an `if(VMC_CUDA)` block in CMakeLists.txt may reference CUDA.

Options:

  - `-DCMAKE_CUDA_ARCHITECTURES=86`   default is `70;80;86;90`; prudence's
    RTX 3090s are sm_86.
  - `-DVMC_CUDA_STRICT_FP=OFF`        re-enables nvcc's FMA contraction. ON by
    default, passing `-fmad=false`, because contraction rounds once where the
    host rounds twice and device results then do not match the CPU reference
    bit-for-bit.
  - `-DVMC_REAL32=ON`                 sampling precision becomes float (62x
    faster on a 3090). NOT validated: read the "STILL UNVALIDATED" list at the
    top of lib/precision.h first.

Which GPUs a run uses is a run-file setting, `[gpu] gpus` (see below). The
default picks the GPU with the most free memory, which matters on a shared
machine.

### Profiling a run

`lib/gpu/prof.h` times named ranges of the device iteration -- the sweeps, the
double evaluation and its stages, the jet pass, the exchange ratios, the O
assembly, SR/CG, and every host/device transfer. Each range carries a cudaEvent
pair on the profiled stream (device time), a wall-clock pair (host time), and a
count of the kernel launches issued inside it.

It is on by default. `prof_enabled = false` in `lib/constants.h` compiles it out
entirely -- with it off the CUDA objects contain no reference to the profiler at
all. Measured cost with it on: **+0.57%** per iteration, and the run's trajectory
and transfer byte counts are bit-identical either way.

Reports print to stdout and append to `BENCH.md` in the run's working directory:
one every `prof_report_every` iterations (default 100) and one at the end of
training (`train()` in `lib/train.cpp`) and of the frozen evaluation. Each report
carries the date, the git revision the binary was built from, the card (name, SM
count, estimated FP64 peak and FP64:FP32 ratio) and the run's configuration, so a
table can always be traced back to the tree and the hardware that produced it.
Rows are **inclusive**: a parent range contains its children.

Build with `-DVMC_NVTX` (e.g. `-DCMAKE_CUDA_FLAGS=-DVMC_NVTX`) to emit the same
range names as NVTX ranges, so an `nsys` trace lines up with the tables. The
NVTX header is optional: without the define nothing includes it.

The repository's `BENCH.md` collects the measurements behind the design
decisions: the clean-up's Phase 6 results first, then the earlier phases in the
order they were done, with the raw profiler reports in an appendix.

# Running

    ./main [run.ini]                        # train, then evaluate the final parameters
    ./frozen_eval checkpoint.txt [run.ini]  # evaluate a checkpoint, nothing else

Run both from the directory that should hold the run's files. The run file is
optional: any setting it leaves out keeps its default, which is the value in
`lib/constants.h`. A minimal run file:

```ini
[training]
sr_iterations = 3000
walkers = 4000

[network]
arch = 64          # all three networks one hidden layer of 64; "64,64" for two

[gpu]
gpus = auto        # every usable GPU, walkers sized from free memory
```

Syntax: `[section]` headers, `key = value` lines, `#` starts a comment, a
section may appear more than once. An unknown section or key is an error, so a
typo cannot silently fall back to a default. Booleans are `true`/`false` or
`1`/`0`; lists are comma-separated. `main` prints the resolved settings and
writes them to `run_config.ini`, so every run directory records how it was made
(15 significant digits; reusable as a run file).

## Settings

`[build]` -- checks only. `N`, `N_p`, `N_u` must equal the compiled values;
a run file written for another nucleus stops with an error instead of running
the wrong one.

`[network]` -- hidden layer widths, comma-separated, default `32` for each.

| key | meaning |
|---|---|
| `arch` | sets `h`, `rho` and `orb` at once |
| `h`, `rho`, `orb` | one network each: the per-particle embedding, the DeepSets determinant weights rho_k, the orbitals |

`[training]`

| key | default | meaning |
|---|---|---|
| `gd_iterations` | 0 | Adam warm-up iterations before SR |
| `sr_iterations` | 2000 | SR iterations |
| `walkers` | 5800 | walkers (CPU: all of them; GPU: per GPU, unless `gpus = auto`) |
| `records` | 2 | samples recorded per walker per iteration ... |
| `grow_factor` | 2 | ... multiplied by this ... |
| `grow_at` | 0.7 | ... after this fraction of all iterations |
| `sweeps_between` | 3 | Metropolis sweeps between two records |
| `therm_sweeps` | 0 | extra sweeps at the start of each iteration (the first record already waits `sweeps_between`) |
| `therm_init` | 500 | initial thermalisation sweeps ... |
| `therm_blocks` | 20 | ... in this many blocks, the step tuned after each |
| `step` | 1.0 | initial Metropolis step (fm); tuned to 30-40% acceptance |
| `ckpt_every` | 100 | `periodic_checkpoint.txt` interval |
| `eval_iters` | 50 | frozen-evaluation iterations at the end (and in `frozen_eval`) |
| `diss_threshold` | -2.5 | dissociation watch: warn when E is above this ... ; `best_checkpoint.txt` only takes iterations below it |
| `diss_window` | 50 | ... and r_rms grew over this many iterations ... |
| `diss_growth` | 1.2 | ... by more than this factor |
| `clip_mad` | 5 | gradient: local energies clipped to median +- clip_mad x MAD |
| `resume` | (empty) | checkpoint to continue from |
| `transfer` | (empty) | checkpoint to start from (weights, alpha, Jastrow); a new run |

`[sr]` -- the SR step. Learning rate `eta * lr_decay^i` (floor 0.001), diagonal
shift `lambda0 * lambda_decay^i` (floor `lambda_min`), `i` the SR iteration.

| key | default | meaning |
|---|---|---|
| `eta`, `lr_decay` | 0.1, 0.999 | learning rate and its decay |
| `lambda0`, `lambda_decay`, `lambda_min` | 100, 0.995, 0.4 | relative diagonal shift lambda x S_ii |
| `eps` | 1e-5 | absolute diagonal shift |
| `rms_beta`, `rms_eps` | 0.99, 1e-3 | RMS damping: adds rms_eps x sqrt(running mean of grad^2); `rms_eps = 0` turns it off |
| `cg_tol`, `cg_maxit` | 1e-3, 200 | CG stops at this relative residual or iteration count |
| `trust_r2` | 0.1 | step scaled so that delta S delta <= trust_r2 ... |
| `delta_max` | 75 | ... and \|delta\| <= delta_max |

`[adam]` -- the warm-up, and an optional Adam step on the envelope block after each SR step.

| key | default | meaning |
|---|---|---|
| `lr`, `beta1`, `beta2`, `eps` | 0.075, 0.9, 0.999, 1e-8 | Adam (warm-up and envelope) |
| `env_lr` | 0 | envelope Adam learning rate; 0 disables it ... |
| `env_decay_it` | 1000 | ... decaying as env_lr / (1 + t / env_decay_it) |
| `env_alpha` | false | include alpha (false: the Jastrow only; alpha stays with SR) |

`[gpu]` -- CUDA builds only.

| key | default | meaning |
|---|---|---|
| `gpus` | `best` | `best`: the GPU with the most free memory, `walkers` walkers. `0,1`: these GPUs, `walkers` each. `auto`: every usable GPU, walkers sized by the planner. `auto_one`: the single GPU where the planner fits the most |
| `overhead_gb`, `margin_gb` | 0.4, 0.5 | planner: memory per GPU kept free |
| `max_util` | 30 | planner: skip GPUs busier than this (%) |
| `max_walkers`, `min_walkers` | 0, 256 | planner: cap walkers per GPU (0: none); skip GPUs that fit fewer |
| `jet_chunk` | 5800 | walkers per local-energy chunk; the jet buffers hold min(walkers, jet_chunk), lower it when memory is short |
| `plan_only` | false | print the memory a run with `walkers` would need, then exit |
| `same_seed` | false | test mode: every GPU draws the same random numbers |

GPU numbers are as CUDA sees them, so they follow `CUDA_VISIBLE_DEVICES`. The
planner rounds walker counts to multiples of 32 and budgets for the largest
batch (after `grow_at`), so the batch growth cannot run out of memory unless
another job takes the card in between; then the run keeps the smaller batch and
says so.

## What a run writes

| file | content |
|---|---|
| `run_config.ini` | the resolved settings |
| `training.csv` | one row per iteration (below) |
| `periodic_checkpoint.txt` | every `ckpt_every` iterations |
| `best_checkpoint.txt` | lowest E + E_err among SR iterations with E below `diss_threshold` (it can be a lucky early iteration) |
| `final_checkpoint.txt` | the parameters after the last iteration; `main` evaluates these |
| `BENCH.md` | profiler reports (CUDA builds with `prof_enabled`) |

At the end `main` prints `Frozen-eval E: E +/- err, var: ..., r_rms: ...`, a
fixed-parameter evaluation of the final parameters over `eval_iters`
iterations. `frozen_eval` prints one `FROZEN` line with the same quantities; it
reads the network widths from the checkpoint, so it needs no `[network]`.

`training.csv` columns: `step`; `E_exp`, `E_err`, `var` (energy, its error and
the local-energy variance); `acceptance`, `spin_acceptance`, `tau_acceptance`
(position moves over all the iteration's sweeps, spin swaps, isospin swaps);
`r_rms`, `L2` (rms radius, <L^2>); `lambda`, `cg_iters`, `cg_residual`,
`delta_norm`, `sq_metric_norm` (delta S delta before scaling), `norm_capped`;
`node_hits`, `n_valid`, `n_invalid` (samples dropped at nodes or by the
validity guard); `alpha`, `grad_alpha`, `delta_alpha` (envelope exponent, its
gradient and applied step); `rms_damp_mean`; timings `metro_ms`, `local_E_ms`,
`o_ms`, `sr_ms`, `gpu_ms`, `ms_iter`, `xfer_ms`; transfer counts `bytes_up`,
`bytes_dn`, `n_scalar_dl`.

## Resume and transfer

`resume = periodic_checkpoint.txt` continues a run: the checkpoint stores the
iteration, lambda and the best E + E_err, so the schedules carry on where they
stopped, and `training.csv` keeps its rows before that iteration.
`sr_iterations` must be larger than the checkpoint's iteration. A resumed run is
not bit-identical to an uninterrupted one: the walkers are re-thermalised and
the Adam moments and RMS-damping averages start again.

`transfer = other_run/final_checkpoint.txt` starts a new run (iteration 0) from
another run's weights, alpha and Jastrow. Orbital layers whose shapes differ are
re-initialised. Transfer between different nuclei does not work yet (`m_feat`
depends on N; BACKLOG I12).

# Regression check (golden)

    python3 tools/golden.py check            # after any change that should not move the numbers
    python3 tools/golden.py record --force   # after an intended numerical change
    python3 tools/golden.py check --only cpu gpu1

It builds the CPU and CUDA configurations from the working tree in
`build-golden/`, runs five short setups (10 SR iterations, 640 walkers) from run
files, and compares each against `tests/golden/<setup>.txt` **bit for bit**: the
physics columns of `training.csv` (timings excluded), the sha256 of
`final_checkpoint.txt`, and the `FROZEN` line of a `frozen_eval` of it.

| setup | build | what it covers |
|---|---|---|
| `cpu` | CPU | the CPU backend |
| `gpu1` | CUDA | one GPU |
| `mg1` | CUDA | `gpus = auto` on one GPU, 1280 walkers split into two jet chunks |
| `gpu2` | CUDA | two GPUs, 640 walkers each; skipped with a warning when a GPU is busy |
| `envadam` | CUDA | `env_lr = 0.01` (envelope Adam) |

A change that only reorganises code must leave all five IDENTICAL. When numbers
are meant to change, the FROZEN energies should stay within their error bars of
the old references (the runs are short, so 1-2 sigma differences are normal),
then record.

# Training monitor

`plot_training.py` is a live dashboard for watching a VMC training run. It reads
`training.csv` (written by `train()` in `lib/train.cpp`) and refreshes
automatically as the C++ program appends new rows — it never touches or depends
on any C++ source itself.

## Setup (one-time)

From the project root:

```
python3 -m venv venv
source venv/bin/activate
pip install dash plotly pandas
```

`source venv/bin/activate` needs to be run again in any new terminal — the
`venv/` folder persists on disk, but the activation only applies to the shell
session it was run in. You'll know it's active when your prompt shows `(venv)`
at the start. To leave it later, run `deactivate`.

## Running it

1. Start (or already have running) the C++ training program, e.g. `./main`,
   from the project root — this is what writes `training.csv`.
2. In a separate terminal, with the venv active:
   ```
   python3 plot_training.py
   ```
3. Open **http://127.0.0.1:8050** in a browser.

The page refreshes itself once a second. If `training.csv` doesn't exist yet,
it shows "waiting for training.csv ..." until the C++ program creates it.

## What you'll see

- Stat cards: current step, phase (`ADAM` or `SR`), energy ± error, variance,
  acceptance rate — plus lambda/CG iteration stats once training reaches the
  SR phase.
- **Energy** plot, shaded with the `±E_err` band.
- **Variance** plot (log scale, since it should shrink by orders of magnitude
  as training converges).
- **Acceptance rate** plot, with dotted reference lines at 0.30/0.40 (the band
  the step-size tuning, `tune_step`, targets).

Each plot marks the ADAM→SR transition with a dashed vertical line, detected
from the data itself (`lambda > 0` marks an SR row) rather than a hardcoded
step number, so it stays correct whatever `gd_iterations` is.

To stop the dashboard, `Ctrl+C` in the terminal running `plot_training.py`.

# Run status (text)

`run_status.py` prints a one-line summary per training run: a plain-text
complement to `plot_training.py` for ssh sessions, and for comparing several
runs side by side. Standard library only, so it needs no venv. Like the
dashboard, it only reads files.

A run is any directory with a `training.csv` (or one whose `run/`
subdirectory holds it). Pass run directories, or a directory of them:

```
python3 run_status.py                      # the run in the current directory
python3 run_status.py runs/                # every run under runs/
python3 run_status.py runs/a runs/b --jastrow
python3 run_status.py runs/ --watch 60     # redraw every 60 s, Ctrl+C to stop
python3 run_status.py --help
```

Columns, averaged over the last `--window` iterations (default 100):

- `stat`: `RUN` if a live `main` process has the run directory as its working
  directory, else `done`. Start `./main` from the run directory for this to work.
- `nuc`, `it/total`: the nucleus and the total iteration count, read from the
  run's `run_config.ini` (runs made before run files existed: the nearest
  `lib/constants.h` at or above the run directory).
- `E`, `sd`: mean energy and its iteration-to-iteration scatter.
- `var`: median local-energy variance. `r_rms` and `L2` are means.
- `alpha`, `lam`, `ms/it`: the latest envelope exponent and SR lambda, and the
  median time per iteration.
- `E_ref`, `gap`: the model "o" reference energy (deuteron: the coded-H oracle;
  others: arXiv:2607.09223 Fig. 5) and `E - E_ref`.
- `frozen eval`: `run` is `main`'s own evaluation of the final parameters,
  found in any `*.log` next to `training.csv` (redirect `./main`'s output to a
  file to keep it). `feval` is any `frozen_eval` output saved under `feval/*.log`.

`--jastrow` adds the envelope settings from `constants.h` and the `alpha` and
`jastrow` lines of `periodic_checkpoint.txt`.

# Code map

## One training iteration

`main.cpp` reads the run file (`lib/config.cpp`), builds the `Ansatz`
(`lib/wavefunction.h`), and calls `train()` (`lib/train.cpp`). `train()` owns the
loop and talks to a `Backend` (`lib/train.h`): `CpuBackend` in
`lib/backend_cpu.cpp` or `GpuBackend` in `lib/gpu/backend_gpu.cu`. One iteration:

| step | `train()` calls | CPU | GPU |
|---|---|---|---|
| 1. room for this batch | `prepare_records` | (nothing) | grow the O pool at batch growth (`DeviceState::ensure_O_pool`, `arena.cu`) |
| 2. sample | `sample` | `therm_batch`, `record_batch` (`walkers.cpp`): `sweep_one` moves each walker | `upload_params`, `therm_batch_device` and `record_batch_device` (`gpu_sampler.cu`, `record_device.cu`); one sweep = `sweep_device` / `sweep_device_graph` (`sampler_kernels.cu`, `sweep_graph.cu`, `eval.cu`) |
| 2a. per record: local energy | (inside `sample`) | `local_E` (`hamiltonian.cpp`) on `psi` / `jpsi` (`wavefunction.cpp`, `slater.cpp`) | `eval_local_E_device` (`local_e.cu`): jets in chunks (`jet_eval.cu` -> `jet_kernels.cu`, `detjet_kernels.cu`, `compose_kernels.cu`), 2-body exchange (`exchange_kernels.cu`) |
| 2b. per record: O = dlog psi / dtheta | (inside `sample`) | `assemble_O` (`wavefunction.cpp`) | `assemble_O_batch` (`backprop.cu`) |
| 3. clip and gradient | `clip_stats` (`sr.cpp`), `gradient` | `clipped_gradient` | `build_mask`, `O_exp_replicas`, `grad_replicas` (`sr_device.cu`) |
| 4. update | `ADAM` (warm-up) or `sr_step` | `SR_step` (`backend_cpu.cpp`): `SROp` (`sr.cpp`) + `cg_solve` (`cg.cpp`) | `SR_step_device` (`sr_device.cu`): `cg_solve_device` with `sr_apply` summing S·v over GPUs |
| 5. envelope Adam | in `train()` | | |
| 6. log | `csv_row`, `save_checkpoint` (`checkpoint.cpp`), `iteration_end` | | profiler (`prof.cu`) |

The parameters' master copy is the host `Ansatz`; the GPU backend uploads it to
every replica each iteration and downloads the SR step from GPU 0.

## lib/

| file | contents |
|---|---|
| `constants.h` | the nucleus (N, N_u, N_p), Hamiltonian and envelope constants, compile-time switches, and the defaults of every run-file setting |
| `config.h/.cpp` | `RunConfig`, `AdamSettings`, `GpuSettings`; the run-file parser and `run_config.ini` writer; `validate_config` |
| `train.h/.cpp` | the `Backend` interface, `train()` (the loop, checkpoints, `training.csv`), `evaluate_frozen()` |
| `backend_cpu.cpp` | `CpuBackend`, `ADAM`, `SR_step`, the clipped gradient, step tuning |
| `wavefunction.h/.cpp` | `Ansatz` (networks, alpha, Jastrow coefficients), `Workspace`; `psi`, `log_p`, `jpsi`, the (s,t) table for spin/isospin swaps, rank-2 swap ratios, `assemble_O` |
| `network.h` | MLP `Network` with forward caches and the activations' derivatives |
| `autodiff.h` | `Jet`: value, D first derivatives and the Laplacian, carried through the networks |
| `slater.h/.cpp` | LU determinants and inverses, `det_jet_from_minv` / `det_jet_given_inv`, the rank-2 determinant ratio |
| `envelope.h` | envelope and the 2-body/3-body Jastrow (shared host/device code) |
| `hamiltonian.h/.cpp` | `local_E`; kinetic term, V_2N with spin/isospin exchange, V_3N, Coulomb, L^2 (shared host/device code) |
| `walkers.h/.cpp` | `WalkerBatch`, CPU Metropolis sweeps, thermalisation, recording, `masked_O_exp` |
| `sr.h/.cpp` | `SrSettings`, `clip_stats`, `SROp` (S·v without forming S) |
| `cg.h/.cpp` | preconditioned conjugate gradient |
| `checkpoint.h/.cpp` | checkpoint v2 (networks, alpha, Jastrow, iteration state), resume and transfer |
| `pool.h/.cpp` | `ThreadPool` |
| `precision.h` | `real` (double, or float with `VMC_REAL32`) and conversion helpers |
| `seed.h`, `rng_common.h`, `util.h/.cpp` | seeds and per-GPU salt, RNG helpers, small vector utilities |

## lib/gpu/

| file | contents |
|---|---|
| `backend_gpu.cu` | `GpuBackend`: replicas (one per GPU), sampling, gradient, SR step, O pool growth, profiler hooks |
| `planner.h/.cu` | walkers per GPU from free memory (`plan_gpus`), arena size of a configuration (`arena_bytes_for`) |
| `arena.h/.cu` | `DeviceState`: every device buffer of one replica, sized once (`allocate`), parameter upload |
| `layouts.h` | index helpers for the device buffers, row-major GEMM wrappers |
| `gpu_util.h` | `DeviceArray`, `PinnedArray`, `CUDA_CHECK`, device selection, transfer counters |
| `philox.h`, `philox_kernels.*` | counter-based RNG, one stream per walker (reproducible across launch shapes) |
| `gpu_sampler.*`, `sampler_kernels.*` | a sweep: whole-particle moves, spin/isospin swaps, recentring, acceptance counters |
| `sweep_graph.*` | the same sweep captured as a CUDA graph |
| `eval.*` | log psi for the batch and for proposals; the (s,t) table |
| `net_forward.*`, `net_kernels.*` | MLP forward on the device (features, bias + activation, stashing for backprop) |
| `det_kernels.*` | Slater matrices, batched LU, determinants, envelope combination |
| `jet_eval.*`, `jet_kernels.*` | the jet pass in chunks of `jet_chunk` walkers |
| `detjet_kernels.*`, `djet.h` | determinant jets from batched inverses |
| `compose_kernels.*` | psi jet = envelope x Jastrow x S; kinetic energy and L^2; V_3N; the validity guard |
| `exchange_kernels.*` | V_2N's spin/isospin exchange through rank-2 determinant ratios; Coulomb; pool rows |
| `local_e.*` | `eval_local_E_device`: the local energy of every walker |
| `record_device.*` | one iteration's records: sweeps, local energy, O; the single download per iteration |
| `backprop.*` | O = dlog psi / dtheta by value-level backprop, written into the O pool |
| `sr_device.*` | masked statistics, gradient, S·v across replicas, device CG, `SR_step_device` |
| `prof.*` | the range profiler and its reports |
| `smoke.*` | a minimal kernel (`test_gpu` uses it to check the toolchain) |

## tests/ and tools/

| file | checks |
|---|---|
| `test_walkers.cpp`, `test_philox.cpp` | CPU sampler (determinism across thread counts, sector conservation, logp cache), Philox streams |
| `test_detjet.cpp`, `test_psi_jet_swap.cpp`, `test_fma_into.cpp` | determinant jets, the jet path of psi, forward accumulation |
| `test_record.cpp` | `local_E` against a frozen copy of its earlier body, record bookkeeping |
| `test_eloc_bruteforce.cpp` | `local_E` against a finite-difference Laplacian and exchange terms from re-evaluated psi; O against finite differences |
| `test_deut_ref.cpp` | the deuteron reference energy of the coded Hamiltonian |
| `test_gpu.cu` | arena, RNG, sampler determinism and CPU replay, logp cache |
| `test_jet.cu`, `test_jet42.cu` | the device jet pass, determinant jets, psi jet, kinetic energy, validity |
| `test_local_e.cu` | device local energy against the CPU, exchange, determinism |
| `test_backprop.cu` | device O against the CPU |
| `test_sr_device.cu` | device statistics, gradient, S·v, CG and SR step against the CPU |
| `test_graphs.cu` | fused kernels and graph sweeps against the plain path |
| `test_precision.cu` | the fp32 measurements behind `test_tolerances.h` |
| `tools/frozen_eval.cpp` | evaluate a checkpoint |
| `tools/golden.py` | the bit-exact regression check above |
