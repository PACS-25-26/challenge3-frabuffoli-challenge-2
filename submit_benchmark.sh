#!/bin/bash
#SBATCH --job-name=laplace_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --account=IscrC_EXTA-PMF
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
# =============================================================================
# submit_benchmark.sh — run the full Laplace benchmark suite on one node.
#
# Submit with:
#     sbatch submit_benchmark.sh
#
# What it does:
#   1. loads the MPI + Python modules,
#   2. (re)builds the self-contained binary from external/ sources,
#   3. runs test/run_benchmarks.py, which generates cases, runs them,
#      saves per-benchmark plots/logs and the aggregated RESULT.md.
#
# All dependencies (Eigen, muParser, nlohmann/json) are bundled under external/
# and compiled from source, so no system numerical libraries are required.
# Only an MPI module and Python with matplotlib/numpy are needed.
# =============================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "Nodes: $SLURM_NNODES  Tasks/node: $SLURM_NTASKS_PER_NODE"

# ── Modules ───────────────────────────────────────────────────────────────────
# Adjust these names to your CINECA system (run 'module avail' to check).
module purge
module load openmpi            # provides mpicxx + mpirun
module load python             # provides python3 (+ matplotlib/numpy if available)
# If matplotlib/numpy are missing, uncomment to install into your user space:
# pip install --user matplotlib numpy

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve the repository root from the location of this script.
REPO_DIR="$(cd "$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")" && pwd)"
# If you submit from the repo root, REPO_DIR is just the submit dir:
if [ -f "$SLURM_SUBMIT_DIR/Makefile" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_DIR"
echo "Repository: $REPO_DIR"

# ── Build (self-contained) ────────────────────────────────────────────────────
echo "Building solver ..."
make clean
make CXX=mpicxx
echo "Build done."

# ── Run the benchmark suite ───────────────────────────────────────────────────
# The Python driver launches mpirun internally for each case, sizing MPI ranks
# from each case's setup.json. We cap at 48 (one full node).
cd "$REPO_DIR/test"
echo "Starting benchmark suite ..."
python3 run_benchmarks.py --max-cores 48

echo ""
echo "Benchmark complete. Outputs:"
echo "  test/RESULT.md                      (aggregated report)"
echo "  test/<benchmark>/plot.png           (per-benchmark figure)"
echo "  test/<benchmark>/results.log        (per-benchmark data)"
echo "  test/hw.info                        (node hardware)"
echo "Finished — $(date)"
