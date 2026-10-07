'''
runs for all setups and save the results in tests/golden/*.txt, 
python3 tools/golden.py record 

runs for all setups and compare the results to the saved references,
python3 tools/golden.py check
'''

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, "build-golden")
REF = os.path.join(REPO, "tests", "golden")

SHORT = {"N_descent": "10", 
         "walker_per_th": "32",
         "therm_steps_init": "5",
         "ckpt_every": "10", 
         "eval_iters": "5", 
         "diss_threshold": "1e9"
        }

BUILDS = {
    "cpu":     (False, {}),
    "cuda":    (True,  {}),
    "envadam": (True,  {"env_adam_lr": "0.01"}),
}

SETUPS = {
    "cpu":     ("cpu",     0, {}),
    "gpu1":    ("cuda",    1, {}),
    "mg1":     ("cuda",    1, {"VMC_GPUS": "auto", "VMC_MAX_WALKERS_PER_GPU": "1280", "VMC_GPU_MAX_UTIL": "100"}),
    "gpu2":    ("cuda",    2, {"VMC_NGPU": "2"}),
    "envadam": ("envadam", 1, {}),
}

SKIP_COLS = re.compile(r"(_ms$|^ms_|^bytes_|^n_scalar_dl$)")   
BUSY_UTIL, BUSY_MIB = 30, 2048                            

def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"golden: command failed: {' '.join(cmd)}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    return r.stdout

def build(name):
    cuda, extra = BUILDS[name]
    src = os.path.join(WORK, name)
    shutil.rmtree(src, ignore_errors=True)
    keep = ("lib", "tests", "tools", "main.cpp", "CMakeLists.txt")
    shutil.copytree(REPO, src, ignore=lambda d, names: [n for n in names if d == REPO and n not in keep])
    path = os.path.join(src, "lib", "constants.h")
    text = open(path).read()
    for key, val in {**SHORT, **extra}.items():
        text, n = re.subn(r"(inline constexpr [\w: ]+?\b%s\s*=\s*)[^;]+;" % key,
                          lambda m: m.group(1) + val + ";", text)
        if n != 1:
            sys.exit(f"golden: constant '{key}' matched {n} times in constants.h")
    open(path, "w").write(text)
    b = os.path.join(src, "b")
    print(f"  building {name} ...", flush=True)
    sh(["cmake", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release", f"-DVMC_CUDA={'ON' if cuda else 'OFF'}"])
    sh(["cmake", "--build", b, "-j16", "--target", "main", "frozen_eval"])
    return b

def gpu_state():
    out = sh(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    return {int(i): (int(u), int(m)) for i, u, m in (l.split(",") for l in out.strip().splitlines())}

def run(name, bdir, gpus):
    _, ngpu, env = SETUPS[name]
    d = os.path.join(WORK, "runs", name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "feval"))
    e = {k: v for k, v in os.environ.items() if not k.startswith("VMC_")}  
    e.update(env)
    if ngpu:
        e["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus[:ngpu])
    print(f"  running {name} ...", flush=True)
    with open(os.path.join(d, "run.log"), "w") as log:
        subprocess.run([os.path.join(bdir, "main")], cwd=d, env=e, stdout=log, stderr=subprocess.STDOUT)
    if not os.path.exists(os.path.join(d, "final_checkpoint.txt")):
        sys.exit(f"golden: {name} produced no final_checkpoint.txt, see {d}/run.log")
    shutil.copy(os.path.join(d, "final_checkpoint.txt"), os.path.join(d, "feval", "final_checkpoint.txt"))
    with open(os.path.join(d, "feval", "frozen.log"), "w") as log:
        subprocess.run([os.path.join(bdir, "frozen_eval"), "final_checkpoint.txt"], cwd=os.path.join(d, "feval"),
                       env=e, stdout=log, stderr=subprocess.STDOUT)
    return d

def fingerprint(d, cols=None):
    rows = list(csv.reader(open(os.path.join(d, "training.csv"))))
    head = rows[0]
    if cols is None:
        cols = [c for c in head if not SKIP_COLS.search(c)]
    missing = [c for c in cols if c not in head]
    idx = [head.index(c) for c in cols if c in head]
    lines = [",".join(cols)] + [",".join(r[i] for i in idx) for r in rows[1:]]
    if missing:
        lines.append("MISSING COLUMNS " + ",".join(missing))
    with open(os.path.join(d, "final_checkpoint.txt"), "rb") as f:
        lines.append("final_checkpoint sha256 " + hashlib.sha256(f.read()).hexdigest())
    frozen = [l for l in open(os.path.join(d, "feval", "frozen.log")) if l.startswith("FROZEN")]
    lines.append(frozen[0].strip() if frozen else "FROZEN line missing")
    return lines

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--only", nargs="+", choices=list(SETUPS), help="run only these setups")
    ap.add_argument("--force", action="store_true", help="record: overwrite existing references")
    args = ap.parse_args()

    names = args.only or list(SETUPS)
    if args.mode == "record" and not args.force and any(os.path.exists(os.path.join(REF, n + ".txt")) for n in names):
        sys.exit("golden: references exist; use --force to overwrite them")

    gpus = []
    if any(SETUPS[n][1] for n in names):
        state = gpu_state()
        gpus = sorted(state, key=lambda g: state[g])
        busy = [g for g, (u, m) in state.items() if u > BUSY_UTIL or m > BUSY_MIB]
        if "gpu2" in names and (len(state) < 2 or busy):
            print(f"WARNING: skipping gpu2 (GPUs in use: {busy or 'fewer than 2 GPUs'}); run it later with --only gpu2")
            names = [n for n in names if n != "gpu2"]

    bdirs = {b: build(b) for b in dict.fromkeys(SETUPS[n][0] for n in names)}
    os.makedirs(REF, exist_ok=True)
    failed = []
    for n in names:
        ref = os.path.join(REF, n + ".txt")
        if args.mode == "check" and not os.path.exists(ref):
            print(f"  {n:8s} NO REFERENCE: record it with  python3 tools/golden.py record --only {n}")
            failed.append(n)
            continue

        d = run(n, bdirs[SETUPS[n][0]], gpus)
        if args.mode == "record":
            open(ref, "w").write("\n".join(fingerprint(d)) + "\n")
            print(f"  {n:8s} recorded")
            continue
        want = open(ref).read().splitlines()
        got = fingerprint(d, cols=want[0].split(","))
        if got == want:
            print(f"  {n:8s} IDENTICAL")
        else:
            failed.append(n)
            bad = next(i for i in range(max(len(got), len(want))) if i >= len(got) or i >= len(want) or got[i] != want[i])
            print(f"  {n:8s} DIFFERS at line {bad + 1}:\n      want: {want[bad] if bad < len(want) else '(none)'}"
                  f"\n      got:  {got[bad] if bad < len(got) else '(none)'}")
    if args.mode == "check":
        print("golden: all identical" if not failed else f"golden: {len(failed)} setup(s) differ: {', '.join(failed)}")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()