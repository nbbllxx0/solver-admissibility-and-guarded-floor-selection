#!/usr/bin/env bash
# Million-element scale check with a restarted Krylov method.
#
# At 10^6 elements the free-DOF count is ~3.1e6, so the flexible Krylov basis of
# the reported configuration (V and Z, 300 FP64 vectors each) needs ~14.9 GB on
# top of the multigrid hierarchy. That does not fit alongside the hierarchy on a
# 32 GiB card, so this block keeps the 300-iteration budget but restarts every
# 100 iterations (basis ~5.0 GB). Restarting is a change of solver
# configuration and is reported as such; the acceptance guard is unchanged.
#
# Usage:
#   PYTHON=/path/to/python bash experiments/phase5/run_scale_1m_restarted.sh [RESTART]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"
RESTART="${1:-100}"

export CUPY_ACCELERATORS="${CUPY_ACCELERATORS-}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$ROOT/.cupy_cache}"

cd "$ROOT" || exit 1

run_scale () {
  local artifact="$1" preset="$2"
  echo "--- $(date +%H:%M:%S) $artifact ($preset), restart=$RESTART"
  "$PY" experiments/phase5/run_gmg_floor_detector_density_field.py \
    --preset "$preset" \
    --density-paths "experiments/paper2/runs/${artifact}/rho_final.npy" \
    --baseline-rho-min 1e-12 --raised-rho-mins 1e-3,1e-2 \
    --restart "$RESTART" --maxiter 300 \
    --out-dir "experiments/phase5/results/scale_${artifact}_restart${RESTART}" 2>&1 | tail -8
}

echo "=== $(date +%H:%M:%S) million-element scale check, restart=$RESTART"
run_scale C1M_MF cantilever_gpu_xxlarge
run_scale B1M_MF bridge_gpu_1M
run_scale Brk1M_MF bracket_gpu_1M
echo "=== $(date +%H:%M:%S) million-element block done"
