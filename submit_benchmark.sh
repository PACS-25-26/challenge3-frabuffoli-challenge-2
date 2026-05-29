#!/bin/bash
#SBATCH --job-name=laplace_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=g100_usr_prod
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --account=IscrC_EXTA-PMF
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
# =============================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "Nodes: $SLURM_NNODES  Tasks/node: $SLURM_NTASKS_PER_NODE"

# ── Modules ───────────────────────────────────────────────────────────────────
# Adjust these names to your CINECA system (run 'module avail' to check).
module purge
module load autoload
module load gcc/10.2.0
module load openmpi/4.1.6--gcc--12.2.0
module load cineca-hpyc/2023.10

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

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=vader,self

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
