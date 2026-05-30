# Parallel Laplace Solver — MPI + OpenMP - Completed

A hybrid MPI/OpenMP solver for the 2D Laplace / Poisson equation on the unit
square, with two iterative methods (Jacobi and one-level Schwarz) and full
support for Dirichlet, Neumann and Robin boundary conditions.

The code is **fully self-contained**: every third-party dependency is bundled
under `external/` (Eigen, muParser, nlohmann/json). The only system requirement
is an MPI compiler with OpenMP support — no system numerical libraries are
needed, no administrator permissions are needed to install anything. This makes
the project trivial to build on an HPC cluster: load an MPI module and run
`make`.

## Problem statement

We solve
$$
-\Delta u(x,y) = f(x,y) \quad \text{on } \Omega = (0,1)\times(0,1),
$$
with boundary conditions configurable per side (Dirichlet, Neumann or Robin).
The domain is discretised on a uniform $n\times n$ grid with spacing $h = 1/(n-1)$;
the second-order 5-point finite-difference stencil
$$
\frac{-u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1} + 4u_{i,j}}{h^2} = f_{i,j}
$$
yields a linear system that is solved iteratively. The canonical test case
adopted throughout the suite is
$$
f(x,y) = 8\pi^2 \sin(2\pi x)\sin(2\pi y), \qquad
u_{\text{exact}}(x,y) = \sin(2\pi x)\sin(2\pi y),
$$
with homogeneous Dirichlet BCs; it admits a closed-form solution, so the
discretisation error can be measured directly.

## Parallelisation strategy

The global grid is decomposed **by rows** among MPI ranks: each rank owns a
contiguous horizontal band, with one ghost row above and one below to hold
neighbours' data. Within a rank, the Jacobi sweep loop is parallelised with
OpenMP via `#pragma omp parallel for reduction`.

```
              y=1  ┌─────────────────────┐
                   │     rank size-1     │   ← owns top boundary
                   ├─────────────────────┤
                   │       ...           │
                   ├─────────────────────┤
                   │       rank 1        │
                   ├─────────────────────┤
                   │       rank 0        │   ← owns bottom boundary
              y=0  └─────────────────────┘
                   x=0                 x=1
```

The vertical sides (`left`, `right`) are owned entirely within each rank, so
their BCs need no communication. The horizontal sides (`top`, `bottom`) belong
only to rank 0 / rank size−1 respectively; ghost-row exchange uses
`MPI_Sendrecv` with direction tags to prevent deadlock.

Convergence is checked globally each iteration via `MPI_Allreduce` on the
local squared increment.

## Solvers

Two iterative strategies, selectable in `setup.json`:

- **`jacobi`** — classic point-wise Jacobi. Cheap per iteration, parallel-friendly,
  but slow to converge (number of iterations grows as $\mathcal{O}(n^2)$).
- **`schwarz`** — one-level additive Schwarz (block-Jacobi). Each rank assembles
  the local 5-point stencil into a sparse matrix, factorises it once with
  Eigen `SparseLU`, and at every outer iteration solves the local sub-problem
  with the ghost rows acting as Dirichlet data. Far fewer outer iterations
  than Jacobi, at the cost of a more expensive iteration.

The implementation follows the Strategy pattern: `Solver` is an abstract base,
concrete solvers (`JacobiSolver`, `SchwarzSolver`) are produced by the factory
`Solver::create(...)` from a string read from `setup.json`.

## Repository layout

```
.
├── README.md                  # this file
├── Makefile                   # self-contained build
├── submit_benchmark.sh        # SLURM script for one full CINECA node (48 cores)
├── include/                   # public headers (.hpp)
│   ├── BoundaryCondition.hpp
│   ├── Grid.hpp
│   ├── JacobiSolver.hpp
│   ├── SchwarzSolver.hpp
│   ├── Solver.hpp
│   ├── Logger.hpp
│   ├── Utils.hpp
│   └── VTKExporter.hpp
├── src/                       # implementations (.cpp) + main.cpp
├── external/                  # bundled dependencies (no system libs needed)
│   ├── eigen3/                #   Eigen 3.4 (header-only)
│   ├── muparser/              #   muParser 2.3 (compiled from source)
│   └── json.hpp               #   nlohmann/json single header
└── test/
    ├── debugging/             # tutorial-style minimal case for sanity checks
    ├── run_benchmarks.py      # Python driver for the full benchmark suite
    ├── README.md              # how to run the benchmarks
    ├── RESULT.md              # aggregated benchmark report (generated)
    └── 1_strong_scaling/  2_weak_scaling/  3_hybrid/
        4_jacobi_vs_schwarz/  5_accuracy_L2/
```

## Build

The project compiles into a single executable, `./laplace-solver`, with
no system numerical libraries linked (verifiable with `ldd`).

### On any machine

```bash
make CXX=mpicxx
```

That's it. The `CXX=mpicxx` is important: a plain `make` may pick up the
system `g++`, which has no MPI headers. If your environment exports
`CXX=g++`, the explicit override is required.

### On a cluster (e.g. CINECA)

No special privileges. From your home directory:

```bash
module load openmpi          # or intelmpi, mpich, ...
make CXX=mpicxx
```

The bundled muParser is compiled together with the project sources; Eigen
is header-only. The build produces a fully static-looking binary in the sense
that it does not depend on any external numerical library at run time.

## Usage

Each "case" is a directory containing two JSON files and is executed inside
that directory. This OpenFOAM-style convention keeps configurations
self-contained.

### `setup.json`

```json
{
    "n":           128,
    "f":           "8*pi^2*sin(2*pi*x)*sin(2*pi*y)",
    "solver":      "jacobi",
    "exact":       "sin(2*pi*x)*sin(2*pi*y)",
    "tol":         1e-6,
    "max_iter":    100000,
    "omp_threads": 2
}
```

| Field         | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `n`           | grid points per side (boundary included)                                |
| `f`           | forcing term, any expression in `x`, `y` (uses muParser; `pi`, `e` predefined) |
| `solver`      | `"jacobi"` or `"schwarz"`                                               |
| `exact`       | optional; when present, the L2 error is computed against this expression |
| `tol`         | convergence tolerance on the global L2 increment                        |
| `max_iter`    | iteration cap                                                           |
| `omp_threads` | OpenMP threads per MPI rank                                             |

### `BC.json`

```json
{
    "top":    { "type": "dirichlet", "expr": "0.0" },
    "bottom": { "type": "dirichlet", "expr": "0.0" },
    "left":   { "type": "dirichlet", "expr": "0.0" },
    "right":  { "type": "dirichlet", "expr": "0.0" }
}
```

Each side accepts:
- `"dirichlet"`: `u = expr`
- `"neumann"`: $\partial u/\partial n = $`expr`, with the **outward** normal
- `"robin"`: $\partial u/\partial n + \alpha \, u = $`expr`, requires an extra
  `"alpha": <value>` key

The expression can depend on `x` (for top/bottom) or `y` (for left/right).

### Running a case

```bash
cd test/debugging
mpirun -np 4 ../../laplace-solver
```

or 

```bash
cd test/debugging
./run.sh
```

Outputs are written to `results/`:

```
results/
├── run.log           # timestamped log: config, convergence, timings, L2 error
├── solution.vtk      # numerical solution (legacy VTK, openable in ParaView)
└── exact.vtk         # exact solution, when `exact` is provided
```

## Benchmarks

Five benchmarks cover the analyses required by the challenge text. They are
driven by a single Python script that generates cases, runs them, parses logs,
plots results and produces an aggregated `RESULT.md`.

```bash
cd test
python3 run_benchmarks.py                  # full suite (default cap: 48 ranks)
python3 run_benchmarks.py --quick          # fast smoke test
python3 run_benchmarks.py --only 5         # one benchmark
python3 run_benchmarks.py --max-cores 8    # for local runs on a laptop
```

The five benchmarks are:

| # | Directory             | Measures                                              |
|---|-----------------------|-------------------------------------------------------|
| 1 | `1_strong_scaling`    | fixed n, NP ∈ {1…48}, fixed iterations — speedup & efficiency |
| 2 | `2_weak_scaling`      | ~64 rows/rank, n grows with NP — time per iteration   |
| 3 | `3_hybrid`            | 48 workers, varying MPI×OpenMP split                  |
| 4 | `4_jacobi_vs_schwarz` | iterations & time to convergence for both solvers     |
| 5 | `5_accuracy_L2`       | L2 error vs h, n = 2^k — measured convergence order   |

Scaling benchmarks (1–3) use a **fixed iteration count** so every
configuration does identical work and wall times are directly comparable;
benchmarks 4–5 run to a real tolerance, where iteration count and accuracy
are the quantities of interest. Each benchmark directory contains the
generated sub-cases plus a `plot.png` and a `results.log`. The aggregated
report is `test/RESULT.md`.

Details — including dependencies, options and how to run on CINECA via
SLURM — are in [`test/README.md`](test/README.md).

### Running on CINECA (one full node, 48 cores)

```bash
sbatch submit_benchmark.sh
```

Edit the `#SBATCH --account` / `--partition` lines and the `module load` names
to match your specific system (`module avail` to list available modules).
The script will build the solver and run the complete benchmark suite.

## Implementation notes

A few design choices worth flagging:

- **Row-major storage.** `Grid::U_` uses an explicit row-major Eigen matrix
  (`Eigen::Matrix<double, ..., Eigen::RowMajor>`) so that `U_.row(i).data()`
  is contiguous and can be passed directly to MPI without copies. Eigen's
  default column-major layout would silently send garbage across ghost rows.

- **Outward normal conventions.** Neumann and Robin BCs use the *outward*
  normal: $\partial u/\partial n = -\partial u/\partial x$ on the left wall
  ($x = 0$), $-\partial u/\partial y$ on the bottom ($y = 0$), and the
  corresponding $+$ signs on right and top. The discretisation in
  `Grid::applyBoundaryConditions` reflects this for all four sides.

- **MPI tag discipline.** `Grid::exchangeGhostRows` uses two direction-encoded
  tags (`TAG_UP`, `TAG_DOWN`); each `MPI_Sendrecv` uses the same tag on the
  sender and receiver of the same message, which is necessary to avoid
  deadlock and tag mismatch.

- **VTK boundary export.** When boundary conditions are non-homogeneous, the
  physical boundary rows live in ghost slots (`U(0,*)` on rank 0, `U(lr+1,*)`
  on rank size−1) and are not part of the `MPI_Gatherv` of interior rows.
  `VTKExporter` collects them with a dedicated `MPI_Send`/`MPI_Recv` pair so
  ParaView sees the correct field everywhere.

- **Expression parsing.** Forcing terms and boundary expressions are arbitrary
  formulas in `x`, `y` and the constants `pi`, `e`, parsed at runtime by
  muParser. This keeps the binary fully data-driven — no recompilation is
  needed to change the problem.

## Dependencies (all bundled in `external/`)

| Library         | Version | Role                              | Bundled as     |
|-----------------|---------|-----------------------------------|----------------|
| Eigen           | 3.4.0   | dense + sparse linear algebra     | headers only   |
| muParser        | 2.3.4   | runtime expression evaluation     | sources, compiled with project |
| nlohmann/json   | 3.11+   | JSON parsing for setup/BC files   | single header  |
| MPI runtime     | any     | process-level parallelism         | system (via `module load`) |
| OpenMP          | any     | thread-level parallelism          | provided by `mpicxx` |

Python is used only for the benchmark driver and requires `numpy` and
`matplotlib`.
