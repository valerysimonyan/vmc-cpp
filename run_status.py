#!/usr/bin/env python3
import csv, os, re, sys, time, statistics as st

USAGE = "usage: python3 run_status.py [DIR ...] [--window W] [--jastrow] [--watch SECONDS]"

REF = {(2, 1): ("H2", -2.2404), (3, 1): ("H3", -8.48), (3, 2): ("He3", -7.81), (4, 2): ("He4", -28.20),
       (6, 3): ("Li6", -30.95), (7, 3): ("Li7", -37.72), (7, 4): ("Be7", -36.24),
       (12, 6): ("C12", -86.20), (16, 8): ("O16", -126.40)}

CONST_KEYS = ["N", "N_p", "N_descent", "n_jas", "n_jas_cls", "n_j3", "env_adam_lr", "fp32_forward"]


def csv_dir(d):
    for c in (d, os.path.join(d, "run")):
        if os.path.isfile(os.path.join(c, "training.csv")):
            return os.path.abspath(c)
    return None


def find_runs(paths):
    runs = []
    for p in paths:
        if csv_dir(p):
            runs.append(p)
        elif os.path.isdir(p):
            runs += [os.path.join(p, s) for s in sorted(os.listdir(p)) if csv_dir(os.path.join(p, s))]
    return runs


def consts(run):
    d = os.path.abspath(run)
    for _ in range(3):
        path = os.path.join(d, "lib", "constants.h")
        if os.path.isfile(path):
            s = open(path).read()
            out = {}
            for key in CONST_KEYS:
                m = re.search(r"inline constexpr [\w ]+?\b%s\s*=\s*([^;]+);" % key, s)
                if m:
                    out[key] = m.group(1).strip()
            return out
        d = os.path.dirname(d)
    return {}


def alive_dirs():
    out = set()
    if not os.path.isdir("/proc"):
        return out
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.path.basename(os.readlink(f"/proc/{pid}/exe")) == "main":
                out.add(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            pass
    return out


def frozen(cdir):
    res = []
    for d, label, pat in ((cdir, "run", r"Frozen-eval E: ([^,\s]+) \+/- ([^,\s]+), var: ([^,\s]+)"),
                          (os.path.join(cdir, "feval"), "feval", r"FROZEN \w+ E (\S+) err (\S+) var (\S+)")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".log"):
                continue
            for line in open(os.path.join(d, fn), errors="replace"):
                m = re.search(pat, line)
                if m:
                    res.append((label, float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return res


def ckpt_env(cdir):
    alpha, jc = None, []
    path = os.path.join(cdir, "periodic_checkpoint.txt")
    if os.path.isfile(path):
        for line in open(path):
            if line.startswith("alpha "):
                alpha = float(line.split()[1])
            elif line.startswith("jastrow "):
                jc = [float(v) for v in line.split()[2:]]
    return alpha, jc


def fmt(x, f):
    return f % x if x is not None else "-"


def summarize(runs, window, show_jas):
    alive = alive_dirs()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    hdr = ("%-20s %-4s %-4s %11s | %8s %5s %9s %6s %6s %6s %6s %7s | %7s %6s | %s"
           % ("run", "stat", "nuc", "it/total", "E", "sd", "var", "r_rms", "L2", "alpha", "lam", "ms/it", "E_ref", "gap", "frozen eval"))
    print(hdr)
    print("-" * len(hdr))
    for run in runs:
        cdir = csv_dir(run)
        name = os.path.basename(os.path.abspath(run))
        rows = list(csv.DictReader(open(os.path.join(cdir, "training.csv"))))
        c = consts(run)
        N, Np = int(c.get("N", 0) or 0), int(c.get("N_p", 0) or 0)
        nuc, Eref = REF.get((N, Np), ("N=%d" % N if N else "?", None))
        stat = "RUN" if cdir in alive else "done"
        total = c.get("N_descent", "?")
        if len(rows) < 2:
            print("%-20s %-4s %-4s %5d/%-5s | (thermalizing)" % (name, stat, nuc, len(rows), total))
            continue
        w = rows[-window:]
        col = lambda k: [float(r[k]) for r in w if r.get(k) not in (None, "")]
        E = col("E_exp")
        Em = st.mean(E)
        last = rows[-1]
        L2 = st.mean(col("L2")) if "L2" in last else None
        lam = float(last["lambda"]) if last.get("lambda") else None
        ms = st.median(col("ms_iter")) if "ms_iter" in last else None
        gap = Em - Eref if Eref is not None else None
        fz = "; ".join("%s %.3f±%.3f v%.0f" % t for t in frozen(cdir))
        print("%-20s %-4s %-4s %5d/%-5s | %8.3f %5.2f %9.1f %6.2f %6s %6.3f %6s %7s | %7s %6s | %s"
              % (name, stat, nuc, len(rows), total, Em, st.pstdev(E), st.median(col("var")), st.mean(col("r_rms")),
                 fmt(L2, "%.3f"), float(last["alpha"]), fmt(lam, "%.1f"), fmt(ms, "%.0f"),
                 fmt(Eref, "%.2f"), fmt(gap, "%+.2f"), fz))
        if show_jas:
            a, jc = ckpt_env(cdir)
            cfg = " ".join("%s=%s" % (k, c[k]) for k in CONST_KEYS[3:] if k in c)
            print("      %s | ckpt alpha %s | jastrow %s" % (cfg or "-", fmt(a, "%.3f"), " ".join("%.3f" % v for v in jc) or "-"))


def main():
    args = sys.argv[1:]
    window, show_jas, watch, paths = 100, False, None, []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-h", "--help"):
            print(USAGE); return
        if a == "--window": window = int(args[i + 1]); i += 2; continue
        if a == "--watch": watch = float(args[i + 1]); i += 2; continue
        if a == "--jastrow": show_jas = True
        else: paths.append(a)
        i += 1
    runs = find_runs(paths or ["."])
    if not runs:
        sys.exit("run_status: no training.csv found under " + " ".join(paths or ["."]))
    if watch is None:
        summarize(runs, window, show_jas)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            summarize(find_runs(paths or ["."]), window, show_jas)
            time.sleep(watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
