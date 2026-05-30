# Benchmark suite (Python-driven)

A single Python script generates, runs, plots and aggregates all 5 benchmarks.
Used for benchmark on CINECA Galileo100, can be used for local benchmarking 
(generates "_LOCAL" files and directories).

## Prerequisites

- the solver binary built at the repo root (`make` — fully self-contained)
- `python3` with `matplotlib` and `numpy`
- an MPI runtime (`mpirun`)

## Run everything

From this `test/` directory, on a compute node (or via the SBATCH script):

```bash
python3 run_benchmarks.py                 # full suite, up to 48 MPI ranks
python3 run_benchmarks.py --quick         # tiny cases, fast smoke test
python3 run_benchmarks.py --only 1        # just benchmark 1
python3 run_benchmarks.py --max-cores 24  # cap the largest rank count
```

## What it produces

```
test/
├── run_benchmarks.py
├── hw.info                         # node hardware (auto-collected)
├── RESULT.md                       # aggregated report with tables + figures
├── 1_strong_scaling/
│   ├── np1/ np2/ ... np48/         # generated sub-cases (BC.json, setup.json, results/)
│   ├── results.log                 # per-benchmark data table
│   └── plot.png                    # speedup + efficiency
├── 2_weak_scaling/   ... plot.png + results.log
├── 3_hybrid/         ... plot.png + results.log
├── 4_jacobi_vs_schwarz/ ... plot.png + results.log
└── 5_accuracy_L2/    ... plot.png + results.log
```

## The 5 benchmarks

| # | Dir | Measures | Plot |
|---|-----|----------|------|
| 1 | `1_strong_scaling`    | fixed n, NP∈{1..48}, fixed iters | speedup + efficiency |
| 2 | `2_weak_scaling`      | ~64 rows/rank, n grows with NP   | time per iteration |
| 3 | `3_hybrid`            | 48 workers, varying MPI×OpenMP   | bar chart of times |
| 4 | `4_jacobi_vs_schwarz` | iters & time to tol, both solvers| log-log iters & time vs n |
| 5 | `5_accuracy_L2`       | L2 error vs h, n=2^k             | log-log with order-2 ref |

Scaling benchmarks (1–3) use a fixed iteration count so every configuration does
identical work and wall times are directly comparable. Benchmarks 4–5 run to a
real tolerance, where iteration count / accuracy is the quantity of interest.

## On CINECA

Build and run via SLURM (one node, 48 cores):

```bash
sbatch ../submit_benchmark.sh
```

Edit the `#SBATCH --account/--partition` lines and the `module load` names in
that script to match your system (`module avail` to list available modules).
