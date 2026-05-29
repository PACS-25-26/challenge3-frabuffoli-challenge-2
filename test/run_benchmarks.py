#!/usr/bin/env python3
"""
run_benchmarks.py — end-to-end benchmark driver for the parallel Laplace solver.

For each of the 5 benchmarks it:
  1. generates the sub-cases (each a directory with BC.json + setup.json),
  2. runs every sub-case via mpirun and parses results/run.log,
  3. saves a per-benchmark plot (PNG) and a per-benchmark log (.log),
and finally writes an aggregated RESULT.md collecting all results and figures.

Designed for a single CINECA node (up to 48 cores). It uses only the bundled,
self-contained binary `laplace-solver` built by the top-level Makefile; no
system numerical libraries are required at run time (only the MPI runtime).

Usage (from the test/ directory, on a compute node or via the SBATCH script):
    python3 run_benchmarks.py [--quick] [--only N] [--max-cores 48]

Options:
    --quick        shrink every case (small n, few iters) for a fast smoke test
    --only N       run only benchmark N (1..5)
    --max-cores C  cap the largest MPI rank count (default 48)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no X server on compute nodes
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent
BINARY = REPO_DIR / "laplace-solver"

F_CANON = "8*pi^2*sin(2*pi*x)*sin(2*pi*y)"
EXACT = "sin(2*pi*x)*sin(2*pi*y)"
BC_DIRICHLET = {
    "top":    {"type": "dirichlet", "expr": "0.0"},
    "bottom": {"type": "dirichlet", "expr": "0.0"},
    "left":   {"type": "dirichlet", "expr": "0.0"},
    "right":  {"type": "dirichlet", "expr": "0.0"},
}

# ── Globals set from CLI ──────────────────────────────────────────────────────
QUICK = False
MAX_CORES = 48


# ── Case execution ────────────────────────────────────────────────────────────

def write_case(case_dir: Path, setup: dict, bc: dict = None):
    """Create a case directory with setup.json and BC.json."""
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "setup.json").write_text(json.dumps(setup, indent=4))
    (case_dir / "BC.json").write_text(json.dumps(bc or BC_DIRICHLET, indent=4))


def run_case(case_dir: Path) -> dict:
    """Run a single case with mpirun, parse results/run.log, return a dict."""
    setup = json.loads((case_dir / "setup.json").read_text())
    np_procs = int(setup.get("n_procs", 1))

    env = dict(os.environ)
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMP_NUM_THREADS"] = str(setup.get("omp_threads", 1))

    (case_dir / "results").mkdir(exist_ok=True)

    # Note: no --oversubscribe flag — that's OpenMPI-only and breaks Intel MPI.
    # On CINECA nodes (48 cores) we never oversubscribe anyway.
    cmd = ["mpirun", "-np", str(np_procs), str(BINARY)]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=case_dir, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True)
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"    ERROR: mpirun exited with status {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"    stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return parse_log(case_dir / "results" / "run.log", wall)


def parse_log(log_path: Path, wall: float) -> dict:
    """Extract metrics from a run.log. Lines are prefixed with [hh:mm:ss]."""
    text = log_path.read_text()

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    return {
        "n":      grab(r"Grid size\s*:\s*(\d+)", int),
        "omp":    grab(r"OpenMP threads\s*:\s*(\d+)", int),
        "iters":  grab(r"Convergence after\s+(\d+)", int),
        "time":   grab(r"Solve time\s*:\s*([0-9.]+)", float),
        "L2":     grab(r"L2=([0-9.eE+-]+)", float),
        "wall":   wall,
    }


# ── Plot / log helpers ────────────────────────────────────────────────────────

def save_log(bench_dir: Path, header: str, rows: list, columns: list):
    """Write a plain-text results.log: header comment + aligned table."""
    log = bench_dir / "results.log"
    widths = [max(len(str(r.get(c, ""))) for r in rows + [{c: c}]) for c in columns]
    lines = [f"# {header}", "# " + "  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    for r in rows:
        lines.append("  " + "  ".join(str(r.get(c, "")).ljust(w)
                                       for c, w in zip(columns, widths)))
    log.write_text("\n".join(lines) + "\n")
    return log


# ── Benchmark 1: strong scaling ───────────────────────────────────────────────

def bench_strong(bench_dir: Path):
    n = 64 if QUICK else 512
    iters = 100 if QUICK else 3000
    procs = [p for p in [1, 2, 4, 8, 16, 24, 48] if p <= MAX_CORES]
    rows = []
    for p in procs:
        cd = bench_dir / f"np{p}"
        write_case(cd, {
            "n": n, "f": F_CANON, "solver": "jacobi", "exact": EXACT,
            "tol": 1e-30, "max_iter": iters, "omp_threads": 1, "n_procs": p,
        })
        print(f"  [strong] np={p} ...", flush=True)
        r = run_case(cd)
        r["np"] = p
        rows.append(r)

    t1 = rows[0]["time"]
    for r in rows:
        r["speedup"] = t1 / r["time"] if r["time"] else 0.0
        r["eff"] = r["speedup"] / r["np"]

    save_log(bench_dir, f"Strong scaling: n={n}, {iters} iters fixed",
             rows, ["np", "time", "speedup", "eff", "iters"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    p = [r["np"] for r in rows]
    s = [r["speedup"] for r in rows]
    ax1.plot(p, p, "k--", label="ideal", lw=1)
    ax1.plot(p, s, "o-", label="measured")
    ax1.set_xlabel("MPI processes"); ax1.set_ylabel("speedup")
    ax1.set_title(f"Strong scaling speedup (n={n})"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(p, [r["eff"] for r in rows], "s-", color="tab:red")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("MPI processes"); ax2.set_ylabel("efficiency")
    ax2.set_title("Parallel efficiency"); ax2.set_ylim(0, 1.1); ax2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(bench_dir / "plot.png", dpi=120); plt.close(fig)
    return rows, n, iters


# ── Benchmark 2: weak scaling ─────────────────────────────────────────────────

def bench_weak(bench_dir: Path):
    iters = 100 if QUICK else 2000
    rows_per_rank = 32 if QUICK else 64
    procs = [p for p in [1, 2, 4, 8, 16, 24, 48] if p <= MAX_CORES]
    rows = []
    for p in procs:
        n = rows_per_rank * p + 2
        cd = bench_dir / f"np{p}_n{n}"
        write_case(cd, {
            "n": n, "f": F_CANON, "solver": "jacobi", "exact": EXACT,
            "tol": 1e-30, "max_iter": iters, "omp_threads": 1, "n_procs": p,
        })
        print(f"  [weak] np={p} n={n} ...", flush=True)
        r = run_case(cd)
        r["np"] = p
        r["tpi_ms"] = (r["time"] / r["iters"] * 1e3) if r["iters"] else 0.0
        rows.append(r)

    save_log(bench_dir, f"Weak scaling: ~{rows_per_rank} rows/rank, {iters} iters",
             rows, ["np", "n", "time", "tpi_ms", "iters"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    p = [r["np"] for r in rows]
    ax.plot(p, [r["tpi_ms"] for r in rows], "o-")
    ax.axhline(rows[0]["tpi_ms"], color="k", ls="--", lw=1, label="ideal (flat)")
    ax.set_xlabel("MPI processes"); ax.set_ylabel("time per iteration (ms)")
    ax.set_title(f"Weak scaling (~{rows_per_rank} rows/rank)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(bench_dir / "plot.png", dpi=120); plt.close(fig)
    return rows, rows_per_rank, iters


# ── Benchmark 3: hybrid MPI x OpenMP ──────────────────────────────────────────

def bench_hybrid(bench_dir: Path):
    n = 64 if QUICK else 512
    iters = 100 if QUICK else 3000
    total = min(MAX_CORES, 8 if QUICK else 48)
    # splits where mpi*omp = total
    splits = [(p, total // p) for p in [1, 2, 4, 8, 12, 24, 48]
              if p <= total and total % p == 0 and (total // p) <= total]
    rows = []
    for mpi, omp in splits:
        cd = bench_dir / f"mpi{mpi}_omp{omp}"
        write_case(cd, {
            "n": n, "f": F_CANON, "solver": "jacobi", "exact": EXACT,
            "tol": 1e-30, "max_iter": iters, "omp_threads": omp, "n_procs": mpi,
        })
        print(f"  [hybrid] mpi={mpi} omp={omp} ...", flush=True)
        r = run_case(cd)
        r["mpi"] = mpi; r["omp"] = omp
        rows.append(r)

    save_log(bench_dir, f"Hybrid: total={total} workers, n={n}, {iters} iters",
             rows, ["mpi", "omp", "time", "iters"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{r['mpi']}x{r['omp']}" for r in rows]
    ax.bar(labels, [r["time"] for r in rows], color="tab:blue")
    ax.set_xlabel("MPI x OpenMP"); ax.set_ylabel("solve time (s)")
    ax.set_title(f"Hybrid split (total={total} workers, n={n})")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(bench_dir / "plot.png", dpi=120); plt.close(fig)
    return rows, n, total


# ── Benchmark 4: Jacobi vs Schwarz ────────────────────────────────────────────

def bench_jvs(bench_dir: Path):
    ns = [16, 32, 64] if QUICK else [16, 32, 64, 128]
    np_procs = min(4, MAX_CORES)
    rows = []
    for solver in ("jacobi", "schwarz"):
        for n in ns:
            cd = bench_dir / f"{solver}_n{n}"
            write_case(cd, {
                "n": n, "f": F_CANON, "solver": solver, "exact": EXACT,
                "tol": 1e-6, "max_iter": 500000, "omp_threads": 1, "n_procs": np_procs,
            })
            print(f"  [jvs] {solver} n={n} ...", flush=True)
            r = run_case(cd)
            r["solver"] = solver; r["n"] = n
            rows.append(r)

    save_log(bench_dir, f"Jacobi vs Schwarz: tol=1e-6, NP={np_procs}",
             rows, ["solver", "n", "iters", "time", "L2"])

    jac = [r for r in rows if r["solver"] == "jacobi"]
    sch = [r for r in rows if r["solver"] == "schwarz"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog([r["n"] for r in jac], [r["iters"] for r in jac], "o-", label="Jacobi")
    ax1.loglog([r["n"] for r in sch], [r["iters"] for r in sch], "s-", label="Schwarz")
    ax1.set_xlabel("n"); ax1.set_ylabel("iterations to converge")
    ax1.set_title("Iterations vs n"); ax1.legend(); ax1.grid(alpha=.3, which="both")
    ax2.loglog([r["n"] for r in jac], [r["time"] for r in jac], "o-", label="Jacobi")
    ax2.loglog([r["n"] for r in sch], [r["time"] for r in sch], "s-", label="Schwarz")
    ax2.set_xlabel("n"); ax2.set_ylabel("solve time (s)")
    ax2.set_title("Time vs n"); ax2.legend(); ax2.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(bench_dir / "plot.png", dpi=120); plt.close(fig)
    return rows, np_procs


# ── Benchmark 5: accuracy L2 vs h ─────────────────────────────────────────────

def bench_accuracy(bench_dir: Path):
    ks = [4, 5, 6] if QUICK else [4, 5, 6, 7, 8]
    np_procs = min(4, MAX_CORES)
    rows = []
    for k in ks:
        n = 2 ** k
        cd = bench_dir / f"n{n}"
        write_case(cd, {
            "n": n, "f": F_CANON, "solver": "jacobi", "exact": EXACT,
            "tol": 1e-9, "max_iter": 2000000, "omp_threads": 2, "n_procs": np_procs,
        })
        print(f"  [accuracy] n={n} ...", flush=True)
        r = run_case(cd)
        r["n"] = n; r["h"] = 1.0 / (n - 1)
        rows.append(r)

    # observed order between consecutive refinements
    for i, r in enumerate(rows):
        if i == 0:
            r["order"] = None
        else:
            p = rows[i - 1]
            r["order"] = np.log(p["L2"] / r["L2"]) / np.log(p["h"] / r["h"])

    save_log(bench_dir, "Accuracy: L2 vs h, n=2^k",
             rows, ["n", "h", "L2", "order", "iters"])

    h = np.array([r["h"] for r in rows])
    l2 = np.array([r["L2"] for r in rows])
    slope = np.polyfit(np.log(h), np.log(l2), 1)[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(h, l2, "o-", label=f"measured (slope={slope:.2f})")
    ax.loglog(h, l2[0] * (h / h[0]) ** 2, "k--", lw=1, label="order 2 ref")
    ax.set_xlabel("h"); ax.set_ylabel("L2 error")
    ax.set_title("Accuracy: L2 error vs h")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(bench_dir / "plot.png", dpi=120); plt.close(fig)
    return rows, slope


# ── RESULT.md aggregation ─────────────────────────────────────────────────────

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, spec=".4g"):
    return "NA" if x is None else format(x, spec)


def write_result_md(results: dict):
    md = ["# Benchmark Results", "",
          "Generated by `run_benchmarks.py`. Hardware: see `hw_LOCAL.info`.", ""]

    if "strong" in results:
        rows, n, iters = results["strong"]
        md += [f"## 1. Strong scaling (n={n}, {iters} iterations fixed)", "",
               "![strong](1_strong_scaling_LOCAL/plot.png)", "",
               md_table(["NP", "Time (s)", "Speedup", "Efficiency"],
                        [[r["np"], fmt(r["time"]), fmt(r["speedup"]), fmt(r["eff"])]
                         for r in rows]),
               "", "Efficiency drops as ranks grow because the per-iteration ghost "
               "exchange and the `MPI_Allreduce` become a larger fraction of the work "
               "(fixed problem size).", ""]

    if "weak" in results:
        rows, rpr, iters = results["weak"]
        md += [f"## 2. Weak scaling (~{rpr} rows/rank, {iters} iterations)", "",
               "![weak](2_weak_scaling_LOCAL/plot.png)", "",
               md_table(["NP", "n", "Time (s)", "Time/iter (ms)"],
                        [[r["np"], r["n"], fmt(r["time"]), fmt(r["tpi_ms"])]
                         for r in rows]),
               "", "Ideal weak scaling keeps time-per-iteration flat; the rise reflects "
               "communication overhead and the log(p) cost of the all-reduce.", ""]

    if "hybrid" in results:
        rows, n, total = results["hybrid"]
        md += [f"## 3. Hybrid MPI x OpenMP (total={total} workers, n={n})", "",
               "![hybrid](3_hybrid_LOCAL/plot.png)", "",
               md_table(["MPI", "OMP", "Time (s)"],
                        [[r["mpi"], r["omp"], fmt(r["time"])] for r in rows]),
               "", "The optimum balances fewer MPI ranks (less communication) against "
               "OpenMP's memory-bandwidth limit — Jacobi is memory-bound, so pure-OpenMP "
               "rarely scales to the full node.", ""]

    if "jvs" in results:
        rows, npp = results["jvs"]
        jac = [r for r in rows if r["solver"] == "jacobi"]
        sch = [r for r in rows if r["solver"] == "schwarz"]
        tbl = []
        for j, s in zip(jac, sch):
            tbl.append([j["n"], j["iters"], fmt(j["time"]), s["iters"], fmt(s["time"])])
        md += [f"## 4. Jacobi vs Schwarz (tol=1e-6, NP={npp})", "",
               "![jvs](4_jacobi_vs_schwarz_LOCAL/plot.png)", "",
               md_table(["n", "Jacobi iters", "Jacobi time", "Schwarz iters", "Schwarz time"], tbl),
               "", "Jacobi's iteration count grows steeply with n; Schwarz (block-Jacobi "
               "with a direct local solve) needs far fewer outer iterations, at a higher "
               "cost per iteration.", ""]

    if "accuracy" in results:
        rows, slope = results["accuracy"]
        md += [f"## 5. Accuracy: L2 vs h (fitted order = {slope:.2f})", "",
               "![accuracy](5_accuracy_L2_LOCAL/plot.png)", "",
               md_table(["n", "h", "L2 error", "Observed order"],
                        [[r["n"], fmt(r["h"]), fmt(r["L2"], ".4e"), fmt(r["order"], ".3f")]
                         for r in rows]),
               "", "A 5-point Laplacian is formally 2nd order. The fitted slope is the "
               "measured convergence rate; deviations from 2 indicate residual iterative "
               "error or the influence of the chosen discrete L2 norm.", ""]

    (TEST_DIR / "RESULT_LOCAL.md").write_text("\n".join(md))
    print(f"\nWrote {TEST_DIR / 'RESULT_LOCAL.md'}")


# ── Hardware info ─────────────────────────────────────────────────────────────

def collect_hw():
    out = [f"# environment, generated {time.ctime()}", ""]
    for label, cmd in [("uname", ["uname", "-a"]),
                       ("lscpu", ["lscpu"]),
                       ("mpirun", ["mpirun", "--version"])]:
        out.append(f"== {label} ==")
        try:
            out.append(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
        except Exception as e:
            out.append(f"(unavailable: {e})")
        out.append("")
    (TEST_DIR / "hw_LOCAL.info").write_text("\n".join(out))


# ── Main ──────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    1: ("1_strong_scaling_LOCAL", bench_strong, "strong"),
    2: ("2_weak_scaling_LOCAL", bench_weak, "weak"),
    3: ("3_hybrid_LOCAL", bench_hybrid, "hybrid"),
    4: ("4_jacobi_vs_schwarz_LOCAL", bench_jvs, "jvs"),
    5: ("5_accuracy_L2_LOCAL", bench_accuracy, "accuracy"),
}


def main():
    global QUICK, MAX_CORES
    ap = argparse.ArgumentParser(description="Run the Laplace solver benchmark suite.")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (tiny cases)")
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4, 5], help="run only one benchmark")
    ap.add_argument("--max-cores", type=int, default=48, help="cap MPI ranks (default 48)")
    args = ap.parse_args()
    QUICK = args.quick
    MAX_CORES = args.max_cores

    if not BINARY.exists():
        sys.exit(f"ERROR: binary not found at {BINARY}\n"
                 f"Build it first:  (cd {REPO_DIR} && module load <mpi> && make)")

    collect_hw()
    selected = [args.only] if args.only else [1, 2, 3, 4, 5]
    results = {}
    for k in selected:
        name, fn, key = BENCHMARKS[k]
        print(f"\n=== Benchmark {k}: {name} ===", flush=True)
        bench_dir = TEST_DIR / name
        bench_dir.mkdir(exist_ok=True)
        results[key] = fn(bench_dir)

    write_result_md(results)
    print("\nAll done.")


if __name__ == "__main__":
    main()
