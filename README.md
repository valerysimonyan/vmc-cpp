## Build and test

Requires g++ with C++17 and CMake >= 3.18. Ubuntu 20.04 ships CMake 3.16, which
is too old for `CMAKE_CUDA_ARCHITECTURES`; `pip3 install --user cmake` gets a
current one.

    make            # CPU only, builds ./build/main
    make test       # CPU suite via ctest (~5 s)

Or CMake directly:

    cmake -B build && cmake --build build -j
    ctest --test-dir build --output-on-failure

### CUDA (optional)

    make gpu        # or: cmake -B build-cuda -DVMC_CUDA=ON && cmake --build build-cuda -j
    make gpu-test

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
  - `VMC_CUDA_DEVICE=1`               environment variable pinning a GPU.
    Without it the most-free device is chosen, which matters on a shared machine.

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

Reports print to stdout and append to `BENCH.md`: one every
`prof_report_every` iterations (default 100) and one at the end of `descent()`
and of the frozen evaluation. Each report carries the date, the git revision the
binary was built from, the card (name, SM count, estimated FP64 peak and
FP64:FP32 ratio) and the run's configuration, so a table can always be traced
back to the tree and the hardware that produced it. Rows are **inclusive**: a
parent range contains its children.

Build with `-DVMC_NVTX` (e.g. `-DCMAKE_CUDA_FLAGS=-DVMC_NVTX`) to emit the same
range names as NVTX ranges, so an `nsys` trace lines up with the tables. The
NVTX header is optional: without the define nothing includes it.

The decomposition tables, the batch-size sweep and the Phase 6 decision table
derived from them are in `BENCH.md`.

# Training monitor

`plot_training.py` is a live dashboard for watching a VMC training run. It reads
`training.csv` (written by `descent()` in `lib/monte_carlo.cpp`) and refreshes
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
- **Acceptance rate** plot, with dotted reference lines at 0.4/0.6 (the band
  the step-size adaptation in `parallel_run` targets).

Each plot marks the ADAM→SR transition with a dashed vertical line, detected
from the data itself (`lambda > 0` marks an SR row) rather than a hardcoded
step number, so it stays correct even if `N_gd` in `constants.h` changes.

To stop the dashboard, `Ctrl+C` in the terminal running `plot_training.py`.
