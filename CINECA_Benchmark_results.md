# Performance Analysis and Numerical Verification

This document presents a critical analysis of the benchmark results obtained using the hybrid MPI + OpenMP parallel solver for the 2D Laplace equation. The tests were executed on a dedicated cluster node matching the environment described in `hw.info`.

---

## Hardware & Environment Reference
The benchmarks were conducted on a dual-socket computing node under the following environment:
* **CPU:** 2x Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz (Total 48 physical cores, Hyper-Threading disabled)
* **Architecture:** 2 NUMA nodes (24 cores per socket) with 35.75 MB L3 cache per socket
* **MPI Runtime:** Intel(R) MPI Library for Linux* OS (Version 2021.5)

---

## 1. Strong Scaling Analysis (n = 512, 3000 Fixed Iterations)

Strong scaling evaluates parallel efficiency by keeping the global problem size fixed ($512 \times 512$) while increasing the number of MPI processes ($NP$).

### Observed Trends
* **Speedup Limitation:** Execution time drops sharply from 1 to 2 processes (**1.96x speedup**) but rapidly flattens beyond 16 processes, yielding a peak speedup of only **4.38x** at $NP = 48$.
* **Efficiency Drop:** Parallel efficiency drops from **98.3%** ($NP = 2$) down to **9.1%** ($NP = 48$).

### Architectural Insights & Bottlenecks
1. **Diminishing Local Workload:** At $NP = 48$, each process owns only about 10 rows of the global mesh. The computational work per process per iteration becomes exceptionally small.
2. **Communication-to-Computation Ratio:** As the local domain shrinks, the time spent exchanging ghost cell rows between adjacent processes and executing collective barriers (`MPI_Allreduce` for convergence checks) begins to dominate the entire execution path.
3. **Amdahl's Law:** The non-parallelizable synchronization components of the loop dictate the performance upper bound, leading to severe resource starvation at higher process counts.

---

## 2. Weak Scaling Analysis (~64 rows/rank, 2000 Iterations)

Weak scaling assesses the code's capability to handle larger data sets by scaling both the problem size ($n$) and the computational resources ($NP$) proportionally, keeping the local work per process roughly constant.

### Observed Trends
* **Time Inefficiency:** In an ideal weak-scaling scenario, the time per iteration should remain flat. Instead, we observe a steady rise from **3.12 ms/iter** ($NP = 1$, $n = 66$) to **37.52 ms/iter** ($NP = 48$, $n = 3074$).

### Architectural Insights & Bottlenecks
1. **Message Volume Scaling:** Because the problem domain expands as $n$ grows, the boundary rows scannable for ghost cell communication grow from 66 elements to 3074 elements. The network/bus transmission overhead scales linearly with $n$.
2. **Collective Network Overhead:** The `MPI_Allreduce` function operates on a tree-like topology. Its latency scales logarithmically $\mathcal{O}(\log P)$ with the process count, adding an unavoidable software delay at $NP = 48$.

---

## 3. Hybrid Parallelism (MPI x OpenMP, 48 Total Workers, n = 512)

This benchmark examines the behavior of shared-memory threading (OpenMP) paired with distributed-memory tasking (MPI), maintaining a total allocation of 48 computing units.

### Observed Trends
* **Optimal Layout:** The fastest runtimes are achieved using pure MPI or near-pure MPI setups: `24 MPI x 2 OMP` (**18.09 s**) and `48 MPI x 1 OMP` (**18.04 s**).
* **Worst Layout:** The pure OpenMP layout `1 MPI x 48 OMP` yields the poorest performance (**77.74 s**).

### Architectural Insights & Bottlenecks
1. **Memory-Bound Bottleneck:** The 5-point Jacobi stencil possesses a very low operational intensity (few floating-point operations per byte of memory moved). Jacobi is heavily memory-bandwidth bound.
2. **NUMA Effects:** A single MPI process spanning 48 OpenMP threads must access memory regions across both physical sockets (NUMA nodes 0 and 1). Threads scheduled on Socket 1 accessing data allocated on Socket 0 suffer major cross-socket latency penalties (UPI interconnect congestion).
3. **Cache Coherency & Contention:** 48 threads executing on shared memory compete fiercely for the memory controller lanes. Breaking down the problem into individual MPI ranks localizes the rows within the specific NUMA node domain, leveraging independent memory controllers and maximizing effective bandwidth.

---

## 4. Algorithmic Comparison: Jacobi vs Schwarz (tol = 1e-6, NP = 4)

This test compares the standard point-wise Jacobi method against a domain decomposition alternative: Block-Jacobi (One-Level Schwarz) using a direct solver for local subdomains.

### Observed Trends
* **Iteration Count:** Jacobi's iteration requirements explode with grid density (from 135 iterations at $n=16$ up to 7221 iterations at $n=128$). In contrast, Schwarz scales exceptionally well (from 40 to 287 iterations).
* **Execution Time:** For a $128 \times 128$ grid, Schwarz accelerates execution by a factor of over 11x (**1.44 s** vs **15.93 s**).

### Algorithmic Insights
While a single Schwarz iteration is computationally more intensive (owing to the local linear matrix factorization and forward/backward substitutions), it drastically limits the overall iteration footprint. Point-wise Jacobi only propagates information across one grid spacing ($h$) per iteration. The Schwarz method allows information to propagate immediately across the entire local subdomain block, destroying the strict mathematical convergence barrier of standard Jacobi.

---

## 5. Numerical Accuracy Verification: $L_2$ Error vs $h$

The spatial convergence rate was verified against the exact analytical solution $u(x, y) = \sin(2\pi x)\sin(2\pi y)$ using the discrete $L_2$ norm.

### Observed Trends
* **Fitted Convergence Order:** **1.51**

### Numerical Insights
The standard 5-point finite difference stencil for the Laplace operator possesses a theoretical truncation error of $\mathcal{O}(h^2)$ (second-order accuracy). The observed reduction to $1.51$ stems from two primary experimental factors:
1. **Iterative Error Interference:** Stopping the solver at a fixed algebraic tolerance of $10^{-6}$ means the residual iterative error is still larger than the underlying geometric discretization error on fine grids ($n=256$). This prematurely cuts off the second-order asymptotic trend.
2. **Discrete Boundary Scaling:** The boundary inclusion and normalization properties of the chosen discrete $L_2$ norm formula can introduce minor shifts in slope estimation across small grid ranges.

---

## Conclusions
The parallel implementation demonstrates proper structural behavior. For memory-bound workloads on modern dual-socket Xeon architectures, utilizing pure MPI processes (or limiting OpenMP to 2 threads per rank) minimizes memory controller starvation and avoids cross-NUMA socket delays. Additionally, upgrading the math layer to an overlapping/block domain method (Schwarz) proves mandatory to combat the poor convergence scaling of standard point-wise relaxation on refined grids.
