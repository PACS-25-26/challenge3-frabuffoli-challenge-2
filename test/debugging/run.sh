#!/usr/bin/env bash
# =============================================================================
# run.sh – Case launcher for test/debugging
#
# Run from this directory:
#   ./run.sh          # uses n_procs from setup.json
#   ./run.sh -n 2     # override MPI process count
# =============================================================================

set -euo pipefail

# ── Locate the binary: always two levels up from this script ──────────────────
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_DIR/laplace-solver"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${CYAN}[run]${NC}  $*"; }
ok()   { echo -e "${GREEN}[run]${NC}  $*"; }
warn() { echo -e "${YELLOW}[run]${NC}  $*"; }
err()  { echo -e "${RED}[run] ERROR:${NC}  $*" >&2; exit 1; }

# ── Parse arguments ────────────────────────────────────────────────────────────
NP=""
while getopts "n:h" opt; do
    case $opt in
        n) NP="$OPTARG" ;;
        h) grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'; exit 0 ;;
        *) err "Unknown option. Use -h for help." ;;
    esac
done

# ── Checks ─────────────────────────────────────────────────────────────────────
[ -x "$BINARY" ] || err "Binary not found: $BINARY\n       Run 'make' inside $REPO_DIR first."
[ -f "setup.json" ] || err "setup.json not found in $(pwd)"
[ -f "BC.json"    ] || err "BC.json not found in $(pwd)"

# ── Number of processes ────────────────────────────────────────────────────────
if [ -z "$NP" ]; then
    if command -v python3 &>/dev/null; then
        NP=$(python3 -c "import json; print(json.load(open('setup.json')).get('n_procs', 4))")
    else
        warn "python3 not found – defaulting to 4 MPI processes."
        NP=4
    fi
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
mkdir -p results

info "Repo              : $REPO_DIR"
info "Binary            : $BINARY"
info "Case directory    : $(pwd)"
info "MPI processes     : $NP"
echo "────────────────────────────────────────────────────────────────"

mpirun --oversubscribe -np "$NP" "$BINARY"

echo "────────────────────────────────────────────────────────────────"
ok "Done. Results in: $(pwd)/results/"
ls -lh results/ 2>/dev/null || true